#include <metal_stdlib>
#include <metal_math>

using namespace metal;

// Arguments for the GQA KV head repeating kernel
struct RepeatKVHeadsArgs {
    uint num_kv_heads;     // Number of unique KV heads (e.g., 8 for Llama3 8B)
    uint num_query_groups; // Ratio: n_head / n_kv_head (e.g., 32 / 8 = 4)
    uint head_dim;         // Dimension of each head (e.g., 128)
    uint seq_len;          // Current sequence length being processed (e.g., pos + 1)
};

// Kernel to repeat KV heads for Grouped Query Attention
// Takes K or V tensor with n_kv_heads and outputs a tensor with n_heads (n_kv_heads * num_query_groups)
kernel void kernel_repeat_kv_heads_f16(
    device const half * src            [[buffer(0)]], // Input K or V (Layout assumed: [seq_len, n_kv_head, head_dim])
    device       half * dst            [[buffer(1)]], // Output K or V (Layout assumed: [seq_len, n_head, head_dim])
    constant RepeatKVHeadsArgs & args  [[buffer(2)]], // Kernel arguments
    uint                gid            [[thread_position_in_grid]] // 1D Grid Dim: dst_elements = seq_len * n_head * head_dim
) {
    // Calculate total elements in the DESTINATION buffer
    uint n_head = args.num_kv_heads * args.num_query_groups;
    uint total_dst_elements = args.seq_len * n_head * args.head_dim;

    // Bounds check for the destination buffer
    if (gid >= total_dst_elements) {
        return;
    }

    // --- Deconstruct gid to find target indices [s, h, d] in the destination ---
    // This logic maps the linear grid ID to the 3D structure of the destination tensor.
    // Order of dimensions in memory assumed: seq_len is slowest, head_dim is fastest.
    // Example: element at [s, h, d] is at offset s*(n_head*head_dim) + h*(head_dim) + d

    uint d = gid % args.head_dim; // Dimension within the head (fastest changing)
    uint temp = gid / args.head_dim;
    uint h = temp % n_head;       // Target head index (0 to n_head-1) (medium changing)
    uint s = temp / n_head;       // Target sequence index (0 to seq_len-1) (slowest changing)

    // --- Find the corresponding source head index ---
    // Multiple destination heads 'h' map to the same source head 'src_h'.
    uint src_h = h / args.num_query_groups; // Integer division gives the source head index (0 to n_kv_heads-1)

    // --- Calculate source index ---
    // Source layout assumed: [s, src_h, d]
    // Offset calculation based on source dimensions.
    uint src_idx = s * (args.num_kv_heads * args.head_dim)  // Offset for sequence position 's'
                 + src_h * args.head_dim                   // Offset for source head 'src_h'
                 + d;                                      // Offset for dimension 'd'

    // --- Calculate destination index ---
    // The destination index is simply the grid ID, as the grid iterates linearly through the destination buffer.
    uint dst_idx = gid;

    // --- Copy the value ---
    // Read from the calculated source index and write to the calculated destination index.
    dst[dst_idx] = src[src_idx];
}
