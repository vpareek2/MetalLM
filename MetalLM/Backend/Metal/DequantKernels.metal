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

// Add Q6_K struct and dequant kernels here later if needed
// Add Q4_K_S struct and dequant kernels here later if needed (but currently not necessary as it shares Q4_K_M logic/struct)
