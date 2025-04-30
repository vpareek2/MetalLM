#include <metal_stdlib>
using namespace metal;

// Kernel to apply SiLU (Sigmoid Linear Unit) element-wise: f(x) = x * sigmoid(x)
// Uses float for intermediate calculation for better precision with exp().
kernel void kernel_silu_f16(
    device const half   * src [[buffer(0)]],      // Input buffer (f16)
    device       half   * dst [[buffer(1)]],      // Output buffer (f16)
    constant   uint64_t & elementCount [[buffer(2)]], // Number of elements
    uint                gid [[thread_position_in_grid]]) // Global thread ID
{
    // Bounds check
    if (gid >= elementCount) {
        return;
    }

    // Read input
    half x_h = src[gid];

    // Calculate SiLU in float for precision with exp/sigmoid
    float x_f = float(x_h);
    float sigmoid_x = 1.0f / (1.0f + exp(-x_f)); // sigmoid(x) = 1 / (1 + exp(-x))
    float silu_x = x_f * sigmoid_x;             // x * sigmoid(x)

    // Write output
    dst[gid] = half(silu_x);
}


