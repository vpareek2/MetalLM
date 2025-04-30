#include <metal_stdlib>
using namespace metal;

kernel void kernel_mul_f16(
    device const half   * a   [[buffer(0)]], // Input A
    device const half   * b   [[buffer(1)]], // Input B
    device       half   * c   [[buffer(2)]], // Output C = A * B
    constant   uint64_t & ne  [[buffer(3)]], // Number of elements
    uint                gid [[thread_position_in_grid]])
{
    if (gid >= ne) return;
    c[gid] = a[gid] * b[gid];
}

kernel void kernel_add_f16(
    device const half   * a   [[buffer(0)]], // Input A
    device const half   * b   [[buffer(1)]], // Input B
    device       half   * c   [[buffer(2)]], // Output C = A + B
    constant   uint64_t & ne  [[buffer(3)]], // Number of elements
    uint                gid [[thread_position_in_grid]])
{
    if (gid >= ne) return;
    c[gid] = a[gid] + b[gid];
}
