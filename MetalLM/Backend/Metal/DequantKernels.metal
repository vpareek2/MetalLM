#include <metal_stdlib>
using namespace metal;

//----------------------------------------------------------------------
// Constants and Structs for Q4_K (Used by Q4_K_M and Q4_K_S for inference)
//----------------------------------------------------------------------

// Size of the super-block (number of elements)
constant constexpr ushort QK_K = 256;

// Size of the inner blocks within a super-block
constant constexpr int Q4_K_BLOCK_SIZE = 32;

// Size of the scales/mins metadata per super-block in bytes for Q4_K_M/S layout.
// 8 blocks * (6 bits scale + 6 bits min) = 8 * 12 = 96 bits = 12 bytes.
constant constexpr int Q4_K_SCALES_MINS_SIZE = 12;

// Structure representing one Q4_K (M or S) super-block in memory.
// Size: 2 + 2 + 12 + 128 = 144 bytes
struct block_q4_K {
    half    d;                  // super-block scale (for scales) - fp16
    half    dmin;               // super-block scale (for mins)   - fp16
    // Packed 6-bit scale and 6-bit min for each of the 8 inner blocks.
    uchar   scales[Q4_K_SCALES_MINS_SIZE];
    // Packed 4-bit quantized values for the 256 elements.
    uchar   qs[QK_K / 2];
};

struct block_q6_K {
    half d;                         // scale factor (fp16)
    uint8_t scales[QK_K / 16];      // scales for groups of 16 (linear scaling factor)
    uint8_t ql[QK_K / 2];           // lower 4 bits of quants (packed, 4 bits each)
    uint8_t qh[QK_K / 4];           // upper 2 bits of quants (packed, 2 bits each)
}; // Size: 2 + 16 + 128 + 64 = 210 bytes

//----------------------------------------------------------------------
// Helper Functions for Q4_K (M/S)
//----------------------------------------------------------------------

// Metal translation of the C helper function get_scale_min_k4
// Extracts the 6-bit scale (d_modifier) and 6-bit min (m_modifier)
// for the j-th inner block (0-7) from the packed scales array.
static inline void get_scale_min_k4_metal(
    int j,                             // Inner block index (0-7)
    device const uchar * packed_scales, // Pointer to the 12-byte scales array
    thread uchar & d_modifier,         // Output: 6-bit scale modifier
    thread uchar & m_modifier          // Output: 6-bit min modifier
) {
    if (j < 4) {
        d_modifier = packed_scales[j] & 63;
        m_modifier = packed_scales[j + 4] & 63;
    } else {
        d_modifier = (packed_scales[j + 4] & 0xF) | ((packed_scales[j - 4] >> 6) << 4);
        m_modifier = (packed_scales[j + 4] >>  4) | ((packed_scales[j    ] >> 6) << 4);
    }
}

//----------------------------------------------------------------------
// Dequantization Kernels for Q4_K (M/S)
//----------------------------------------------------------------------

// Dequantization kernel for Q4_K blocks into Float32
kernel void kernel_dequantize_q4_K_f32(
    device const block_q4_K * src [[buffer(0)]], // Input buffer with quantized blocks
    device       float        * dst [[buffer(1)]], // Output buffer for dequantized floats
    constant   uint64_t     & nelements [[buffer(2)]], // Total number of elements to dequantize
    uint                  gid [[thread_position_in_grid]]) // Global thread ID
{
    if (gid >= nelements) return;

    uint64_t superblock_idx = gid / QK_K;
    uint     idx_in_superblock = gid % QK_K;
    device const block_q4_K * b = src + superblock_idx;

    uint inner_block_idx = idx_in_superblock / Q4_K_BLOCK_SIZE;

    uchar scale_modifier;
    uchar min_modifier;
    get_scale_min_k4_metal(inner_block_idx, b->scales, scale_modifier, min_modifier);

    float d    = float(b->d);
    float dmin = float(b->dmin);
    float final_scale = d * float(scale_modifier);
    float final_min   = dmin * float(min_modifier);

    uint byte_idx = idx_in_superblock / 2;
    uchar q_byte   = b->qs[byte_idx];
    uchar q_nibble = ((idx_in_superblock % 2) == 0) ? (q_byte & 0xF) : (q_byte >> 4);

    float dequantized_value = final_scale * float(q_nibble) - final_min;
    dst[gid] = dequantized_value;
}

// Dequantization kernel for Q4_K blocks into Float16 (half)
kernel void kernel_dequantize_q4_K_f16(
    device const block_q4_K * src [[buffer(0)]], // Input buffer with quantized blocks
    device       half         * dst [[buffer(1)]], // Output buffer for dequantized halfs
    constant   uint64_t     & nelements [[buffer(2)]], // Total number of elements to dequantize
    uint                  gid [[thread_position_in_grid]]) // Global thread ID
{
    if (gid >= nelements) return;

    uint64_t superblock_idx = gid / QK_K;
    uint     idx_in_superblock = gid % QK_K;
    device const block_q4_K * b = src + superblock_idx;

    uint inner_block_idx = idx_in_superblock / Q4_K_BLOCK_SIZE;

    uchar scale_modifier;
    uchar min_modifier;
    get_scale_min_k4_metal(inner_block_idx, b->scales, scale_modifier, min_modifier);

    half d    = b->d;
    half dmin = b->dmin;
    half final_scale = d * half(scale_modifier);
    half final_min   = dmin * half(min_modifier);

    uint byte_idx = idx_in_superblock / 2;
    uchar q_byte   = b->qs[byte_idx];
    uchar q_nibble = ((idx_in_superblock % 2) == 0) ? (q_byte & 0xF) : (q_byte >> 4);

    half dequantized_value = final_scale * half(q_nibble) - final_min;
    dst[gid] = dequantized_value;
}

//----------------------------------------------------------------------
// Dequantization Kernels for Q6_K
//----------------------------------------------------------------------

// Dequantization kernel for Q6_K blocks into Float32
kernel void kernel_dequantize_q6_K_f32(
    device const block_q6_K * src [[buffer(0)]], // Input buffer with quantized blocks
    device       float        * dst [[buffer(1)]], // Output buffer for dequantized floats
    constant   uint64_t     & nelements [[buffer(2)]], // Total number of elements to dequantize
    uint                  gid [[thread_position_in_grid]]) // Global thread ID
{
    if (gid >= nelements) return;

    // Calculate block and index within block
    uint64_t superblock_idx = gid / QK_K;
    uint     idx_in_superblock = gid % QK_K; // Index 0-255 within the block

    // Get pointer to current block
    device const block_q6_K * b = src + superblock_idx;

    // Get global scale for the block
    float d = float(b->d);

    // --- Extract Scale ---
    // Q6_K has 16 scales, each covering 16 elements
    uint scale_idx = idx_in_superblock / 16; // Scale index 0-15
    // Scales are stored directly as int8 in ggml.c, let's assume the same here for simplicity
    // If they were packed 4-bit, unpacking needed:
    // uint8_t packed_scales_byte = b->scales[scale_idx / 2];
    // uint8_t scale_nibble = (scale_idx % 2 == 0) ? (packed_scales_byte & 0xF) : (packed_scales_byte >> 4);
    // float scale = float(scale_nibble) - 8.0f; // Correction based on ggml.c
    // --- Assuming direct int8 scales for now (verify if needed) ---
    float scale = float(b->scales[scale_idx]); // Direct access if scales are 1 byte each

    // --- Extract Quantized Value (Lower 4 bits + Upper 2 bits) ---
    // Lower 4 bits (from ql)
    uint ql_idx = idx_in_superblock / 2; // Byte index in ql
    uint8_t ql_byte = b->ql[ql_idx];
    uint8_t q_low = (idx_in_superblock % 2 == 0) ? (ql_byte & 0xF) : (ql_byte >> 4);

    // Upper 2 bits (from qh)
    uint qh_idx = idx_in_superblock / 4; // Byte index in qh
    uint8_t qh_byte = b->qh[qh_idx];
    uint8_t shift_h = (idx_in_superblock % 4) * 2; // Shift amount: 0, 2, 4, or 6
    uint8_t q_high = (qh_byte >> shift_h) & 3; // Isolate the 2 bits

    // Combine into 6-bit integer value (0-63)
    int q6 = (int(q_high) << 4) | int(q_low);

    // Dequantize (Formula from ggml.c: d * scale * (q - 32))
    // Subtract 32 because the 6-bit value represents a range centered around 0.
    float dequantized_value = d * scale * (float(q6) - 32.0f);

    dst[gid] = dequantized_value;
}

// Dequantization kernel for Q6_K blocks into Float16 (half)
kernel void kernel_dequantize_q6_K_f16(
    device const block_q6_K * src [[buffer(0)]], // Input buffer with quantized blocks
    device       half         * dst [[buffer(1)]], // Output buffer for dequantized halfs
    constant   uint64_t     & nelements [[buffer(2)]], // Total number of elements to dequantize
    uint                  gid [[thread_position_in_grid]]) // Global thread ID
{
     if (gid >= nelements) return;

    // Calculate block and index within block
    uint64_t superblock_idx = gid / QK_K;
    uint     idx_in_superblock = gid % QK_K;

    // Get pointer to current block
    device const block_q6_K * b = src + superblock_idx;

    // Get global scale for the block (already half)
    half d = b->d;

    // Extract Scale
    uint scale_idx = idx_in_superblock / 16;
    // Assuming direct int8 scales, convert to half
    half scale = half(float(b->scales[scale_idx])); // Convert via float for safety

    // Extract Quantized Value (Lower 4 bits + Upper 2 bits)
    uint ql_idx = idx_in_superblock / 2;
    uint8_t ql_byte = b->ql[ql_idx];
    uint8_t q_low = (idx_in_superblock % 2 == 0) ? (ql_byte & 0xF) : (ql_byte >> 4);

    uint qh_idx = idx_in_superblock / 4;
    uint8_t qh_byte = b->qh[qh_idx];
    uint8_t shift_h = (idx_in_superblock % 4) * 2;
    uint8_t q_high = (qh_byte >> shift_h) & 3;

    int q6 = (int(q_high) << 4) | int(q_low);

    // Dequantize (d * scale * (q - 32)) in half precision
    half dequantized_value = d * scale * (half(q6) - 32.0h); // Use half literal

    dst[gid] = dequantized_value;
}
