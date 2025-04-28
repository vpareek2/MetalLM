#include <metal_stdlib>
using namespace metal;

// Kernel to convert Half (f16) to Float (f32)
kernel void kernel_convert_f16_f32(
    device const half   * src [[buffer(0)]],      // Input buffer (f16)
    device       float  * dst [[buffer(1)]],      // Output buffer (f32)
    constant   uint64_t & elementCount [[buffer(2)]], // Number of elements
    uint                gid [[thread_position_in_grid]]) // Global thread ID
{
    // Bounds check
    if (gid >= elementCount) {
        return;
    }

    // Perform the conversion
    dst[gid] = float(src[gid]);
}

// Add F32 -> F16 kernel here if needed later
