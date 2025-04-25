// ADD SUPPORT FOR Q4_K_S, Needed for inferencing llama

#include <metal_stdlib>
using namespace metal;

//----------------------------------------------------------------------
// Constants and Structs for Q4_K
//----------------------------------------------------------------------

// Size of the super-block (number of elements)
constant constexpr ushort QK_K = 256;

// Size of the inner blocks within a super-block
constant constexpr int Q4_K_BLOCK_SIZE = 32;

// Number of inner blocks in a super-block (QK_K / Q4_K_BLOCK_SIZE)
//constant constexpr int Q4_K_NUM_INNER_BLOCKS = 8;

// Size of the scales/mins metadata per super-block in bytes.
// 8 blocks * (6 bits scale + 6 bits min) = 8 * 12 = 96 bits = 12 bytes.
constant constexpr int Q4_K_SCALES_MINS_SIZE = 12;

// Structure representing one Q4_K super-block in memory.
struct block_q4_K {
    half    d;                  // super-block scale (for scales) - fp16
    half    dmin;               // super-block scale (for mins)   - fp16
    // Packed 6-bit scale and 6-bit min for each of the 8 inner blocks.
    uchar   scales[Q4_K_SCALES_MINS_SIZE];
    // Packed 4-bit quantized values for the 256 elements.
    uchar   qs[QK_K / 2];
}; // Total size = 4 + 12 + 128 = 144 bytes.

//----------------------------------------------------------------------
// Helper Functions
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
    // Logic directly ported from the provided C function
    if (j < 4) {
        // First 4 blocks: scale is lower 6 bits of scales[j], min is lower 6 bits of scales[j+4]
        d_modifier = packed_scales[j] & 63;
        m_modifier = packed_scales[j + 4] & 63;
    } else {
        // Last 4 blocks: scale uses lower 4 bits of scales[j+4] and upper 2 bits of scales[j-4]
        //                min uses upper 4 bits of scales[j+4] and upper 2 bits of scales[j] (j-0 == j)
        d_modifier = (packed_scales[j + 4] & 0xF) | ((packed_scales[j - 4] >> 6) << 4);
        m_modifier = (packed_scales[j + 4] >>  4) | ((packed_scales[j    ] >> 6) << 4);
    }
}

//----------------------------------------------------------------------
// Dequantization Kernels
//----------------------------------------------------------------------

// Dequantization kernel for Q4_K blocks into Float32
kernel void kernel_dequantize_q4_K_f32(
    device const block_q4_K * src [[buffer(0)]], // Input buffer with quantized blocks
    device       float        * dst [[buffer(1)]], // Output buffer for dequantized floats
    constant   uint64_t     & nelements [[buffer(2)]], // Total number of elements to dequantize
    uint                  gid [[thread_position_in_grid]]) // Global thread ID
{
    // Check if the thread is out of bounds
    if (gid >= nelements) {
        return;
    }

    // Calculate the super-block index and the index within the super-block
    uint64_t superblock_idx = gid / QK_K;
    uint     idx_in_superblock = gid % QK_K;

    // Access the current super-block
    device const block_q4_K * b = src + superblock_idx;

    // Determine the inner block index (0-7)
    // Each inner block has 32 elements
    uint inner_block_idx = idx_in_superblock / Q4_K_BLOCK_SIZE;

    // Get the 6-bit scale and min modifiers for this inner block using the helper
    uchar scale_modifier;
    uchar min_modifier;
    get_scale_min_k4_metal(inner_block_idx, b->scales, scale_modifier, min_modifier);

    // Get the super-block scales and convert to float
    float d    = float(b->d);    // Super-block scale for scales
    float dmin = float(b->dmin); // Super-block scale for mins

    // Calculate the final scale and min for this inner block
    float final_scale = d * float(scale_modifier);
    float final_min   = dmin * float(min_modifier);

    // Find the byte containing the 4-bit weight within the super-block's qs array
    uint byte_idx = idx_in_superblock / 2; // Each byte holds two 4-bit quants

    // Extract the 4-bit weight (lower nibble for even index within superblock, upper for odd)
    uchar q_byte   = b->qs[byte_idx];
    uchar q_nibble = ((idx_in_superblock % 2) == 0) ? (q_byte & 0xF) : (q_byte >> 4); // Value 0-15

    // Dequantize the value using the formula from dequantize_row_q4_K: scale * q - min
    float dequantized_value = final_scale * float(q_nibble) - final_min;

    // Write the dequantized value to the destination buffer
    dst[gid] = dequantized_value;
}

// Dequantization kernel for Q4_K blocks into Float16 (half)
kernel void kernel_dequantize_q4_K_f16(
    device const block_q4_K * src [[buffer(0)]], // Input buffer with quantized blocks
    device       half         * dst [[buffer(1)]], // Output buffer for dequantized halfs
    constant   uint64_t     & nelements [[buffer(2)]], // Total number of elements to dequantize
    uint                  gid [[thread_position_in_grid]]) // Global thread ID
{
    // Check if the thread is out of bounds
    if (gid >= nelements) {
        return;
    }

    // Calculate the super-block index and the index within the super-block
    uint64_t superblock_idx = gid / QK_K;
    uint     idx_in_superblock = gid % QK_K;

    // Access the current super-block
    device const block_q4_K * b = src + superblock_idx;

    // Determine the inner block index (0-7)
    uint inner_block_idx = idx_in_superblock / Q4_K_BLOCK_SIZE;

    // Get the 6-bit scale and min modifiers for this inner block using the helper
    uchar scale_modifier;
    uchar min_modifier;
    get_scale_min_k4_metal(inner_block_idx, b->scales, scale_modifier, min_modifier);

    // Get the super-block scales (already half)
    half d    = b->d;
    half dmin = b->dmin;

    // Calculate the final scale and min for this inner block
    // Perform calculations in half precision
    half final_scale = d * half(scale_modifier);
    half final_min   = dmin * half(min_modifier);

    // Find the byte containing the 4-bit weight within the super-block's qs array
    uint byte_idx = idx_in_superblock / 2;

    // Extract the 4-bit weight
    uchar q_byte   = b->qs[byte_idx];
    uchar q_nibble = ((idx_in_superblock % 2) == 0) ? (q_byte & 0xF) : (q_byte >> 4);

    // Dequantize the value using the formula: scale * q - min
    half dequantized_value = final_scale * half(q_nibble) - final_min;

    // Write the dequantized value to the destination buffer
    dst[gid] = dequantized_value;
}
