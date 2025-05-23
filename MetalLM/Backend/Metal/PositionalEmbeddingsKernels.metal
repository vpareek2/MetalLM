// MetalLM/Backend/Metal/PositionalEmbeddingKernels.metal

#include <metal_stdlib>
#include <metal_math> // Required for pow, log, cos, sin, max, min, floor, ceil

using namespace metal;

//------------------------------------------------------------------------------
// RoPE Kernel Arguments
//------------------------------------------------------------------------------

// Corrected: Apply alignment attribute to the struct type declaration
//struct [[metal::aligned(4)]] MetalRopeArgs {
                                           // using metal::aligned is often more explicit in MSL
struct MetalRopeArgs { // <-- CORRECT placement for struct alignment


    // Dimensions (Natural alignment is 4 bytes)
    uint n_dims;

    // RoPE Parameters (Natural alignment is 4 bytes)
    float freq_base;
    float freq_scale;
    int   pos_offset;

    // YaRN Parameters (Natural alignment is 4 bytes)
    int   n_ctx_orig;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    // Flag (Natural alignment is 1 byte)
    bool has_freq_factors;

    // Add explicit padding to ensure total size is 40 bytes, matching Swift's MemoryLayout expectation.
    // Size before padding: 9 * 4 bytes (uint/float/int) + 1 byte (bool) = 37 bytes.
    // The compiler might add padding automatically based on the struct alignment,
    // but explicit padding makes the size definite.
    uint8_t _padding[3];

};


//------------------------------------------------------------------------------
// YaRN Helper Functions (Directly Ported from ggml-metal.metal)
//------------------------------------------------------------------------------

// NOTE: M_PI_F might not be defined by default in Metal stdlib. Use M_PI_F_METAL.
constant float M_PI_F_METAL = 3.14159265358979323846f;

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    // Ensure high > low to avoid division by zero or negative values
    const float diff = high - low;
    const float denominator = max(0.001f, diff); // Prevent division by zero/small number
    // Use float division
    const float y = (float(i0) / 2.0f - low) / denominator;
    // Clamp the result between 0 and 1
    return 1.0f - min(1.0f, max(0.0f, y));
}

// Calculates the correction factor based on original context length etc.
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    // Ensure inputs are valid to prevent log(0) or division by zero
    if (n_rot <= 0.0f || base <= 0.0f || n_ctx_orig <= 0) {
        return 0.0f; // Or some other safe default/error indication
    }
    float log_base = log(base);
    if (abs(log_base) < 1e-6f) { // Avoid division by zero if base is 1
        return 0.0f;
    }
    float term = float(n_ctx_orig) / (n_rot * 2.0f * M_PI_F_METAL);
    if (term <= 0.0f) { // Avoid log of non-positive number
         return 0.0f;
    }
    return float(n_dims) * log(term) / (2.0f * log_base);
}

// Calculates the start and end dimensions for YaRN correction
static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow,
    thread float corr_dims[2] // Output parameter
) {
    // Calculate factors using the helper
    float factor_fast = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base);
    float factor_slow = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base);

    // Clamp results
    corr_dims[0] = max(0.0f, floor(factor_fast));
    // Ensure end dim is not less than start dim and within bounds
    corr_dims[1] = min(float(n_dims) - 2.0f, max(corr_dims[0], ceil(factor_slow))); // Use n_dims-2 since we work in pairs
}

// Main YaRN calculation function - calculates scaled cos_theta and sin_theta
static void calculate_rope_yarn(
    float theta_extrap, // Base theta before factors/scaling
    float freq_scale,   // Linear scaling factor
    float corr_dims[2], // Pre-calculated correction dims [start_dim, end_dim]
    int i0,             // Dimension index (0 to n_dims-1, must be even)
    float ext_factor,   // YaRN extension factor
    float mscale,       // Base magnitude scale (args.attn_factor)
    thread float & cos_theta, // Output cosine
    thread float & sin_theta  // Output sine
) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp; // Default to interpolated theta

    // Apply YaRN extrapolation/interpolation blend if ext_factor is non-zero
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;

        // Adjust magnitude scale based on interpolation
        // Avoid log(0) or log of negative if freq_scale is <= 0
        if (freq_scale > 0.0f) {
            mscale *= (1.0f + 0.1f * log(1.0f / freq_scale));
        }
    }

    // Calculate final cos/sin using the (potentially YaRN-adjusted) theta
    // and the (potentially YaRN-adjusted) magnitude scale
    cos_theta = cos(theta) * mscale;
    sin_theta = sin(theta) * mscale;
}

//------------------------------------------------------------------------------
// RoPE Kernel (Float16, In-Place, Normal Mode)
//------------------------------------------------------------------------------
// In PositionalEmbeddingKernels.metal

// ... (Keep includes, MetalRopeArgs, Helper Functions) ...

//------------------------------------------------------------------------------
// RoPE Kernel (Float16, In-Place, Normal Mode)
//------------------------------------------------------------------------------
kernel void kernel_rope_f16_inplace(
    constant MetalRopeArgs   & args  [[buffer(0)]],
    device   half            * data  [[buffer(1)]],
    device const float       * freqs [[buffer(2)]],
    uint3 tid  [[thread_position_in_grid]],
    uint3 gdim [[threads_per_grid]]
) {

    // Map thread ID vector components to dimensions
    uint dim_pair_idx = tid.x; // Index for the pair of dimensions (0 to n_dims/2 - 1)
    uint head_idx     = tid.y; // Head index (0 to num_heads - 1)
    uint seq_idx      = tid.z; // Sequence index (0 to seq_len - 1)

    // Calculate the actual dimension index (always even)
    uint i0 = dim_pair_idx * 2;

    // Bounds check: Only process dimensions that need rotation
    if (i0 >= args.n_dims) {
        return;
    }

    // Calculate sequence position including the offset
    int current_pos = args.pos_offset + int(seq_idx);

    // --- Calculate Theta ---
    float theta_base = float(current_pos);
    float inv_ndims = -1.0f / float(args.n_dims);
    float theta_extrap = theta_base * pow(args.freq_base, inv_ndims * float(i0));

    // --- Get Frequency Factor ---
    uint ic = dim_pair_idx;
    float freq_factor = args.has_freq_factors ? freqs[ic] : 1.0f;
    float theta_scaled = theta_extrap / max(0.0001f, freq_factor);

    // --- Calculate Sin/Cos using YaRN logic ---
    float cos_theta;
    float sin_theta;
    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);
    calculate_rope_yarn(theta_scaled, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, cos_theta, sin_theta);

    // --- Apply Rotation ---
    // Assuming layout [SequenceLength, NumHeads, HeadDim]
    // Get dimensions from grid (passed by host)
    uint num_heads = gdim.y;
    uint head_dim = gdim.x * 2; // width of grid is n_dims/2

    // Calculate offset using the indices derived from tid components
    uint base_offset = seq_idx * (num_heads * head_dim)
                     + head_idx * head_dim
                     + i0;

    device half * element_pair_ptr = data + base_offset;

    // Bounds check before accessing memory (more robust)
    // Calculate the total number of elements in the buffer
    // uint total_elements = gdim.x * 2 * gdim.y * gdim.z; // Or get buffer length / sizeof(half)
    // if (base_offset + 1 >= total_elements) { // Ensure accessing pair is safe
    //     // Handle error or return - this depends on how buffer size vs grid size is managed
    //     return;
    // }

    half x0 = element_pair_ptr[0];
    half x1 = element_pair_ptr[1];

    element_pair_ptr[0] = x0 * half(cos_theta) - x1 * half(sin_theta);
    element_pair_ptr[1] = x0 * half(sin_theta) + x1 * half(cos_theta);
}
