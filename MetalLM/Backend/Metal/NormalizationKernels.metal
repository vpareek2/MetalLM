#include <metal_stdlib>
using namespace metal;
struct MetalRMSNormArgs {
    float    eps;
    uint64_t ne00;
};
// RMSNorm kernel for Float16 data
// Applies normalization and scales by a weight vector (gamma)
kernel void kernel_rms_norm_f16(
    device const half   * src [[buffer(0)]],
    device const half   * weight [[buffer(1)]],
    device       half   * dst [[buffer(2)]],
    constant MetalRMSNormArgs & args [[buffer(3)]],
    threadgroup float * shmem_f32 [[threadgroup(0)]],
    uint   tgpig[[threadgroup_position_in_grid]], // Row index (or batch/sequence index)
    ushort tpitg[[thread_position_in_threadgroup]], // Thread index within group
    ushort sgitg[[simdgroup_index_in_threadgroup]], // SIMD group index
    ushort tiisg[[thread_index_in_simdgroup]],      // Thread index within SIMD group
    ushort   ntg[[threads_per_threadgroup]])      // Total threads per group
{
    // Calculate pointers for the current row
    device const half * x = src + tgpig * args.ne00;
    device       half * y = dst + tgpig * args.ne00;

    // --- Calculate sum of squares using F32 for better precision ---
    if (sgitg == 0 && ntg > 32) {
        shmem_f32[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_sq_f32 = 0.0f;

    // Parallel sum of squares
    for (uint i = tpitg; i < args.ne00; i += ntg) {
        float val_f32 = float(x[i]);
        sum_sq_f32 += val_f32 * val_f32;
    }
    sum_sq_f32 = simd_sum(sum_sq_f32);

    // Reduce sums across SIMD groups if necessary
    if (ntg > 32) {
        if (tiisg == 0) {
            shmem_f32[sgitg] = sum_sq_f32;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tpitg == 0) {
             sum_sq_f32 = 0.0f;
             for (uint i = 0; i < ntg / 32; ++i) {
                 sum_sq_f32 += shmem_f32[i];
             }
             shmem_f32[0] = sum_sq_f32;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_sq_f32 = shmem_f32[0];
    }
    // --- End sum of squares ---

    // Calculate RMS and scale factor (using F32)
    float mean_sq = sum_sq_f32 / float(args.ne00);
    // Add safety floor and ceiling
    float safe_mean_sq = max(mean_sq, 1e-10f);
    float rms = sqrt(safe_mean_sq + args.eps);
    float scale = min(1.0f / rms, 1e6f); // Cap scale factor to avoid extreme values

    // Add debug output for troubleshooting
    if (tpitg == 0 && tgpig == 0) {
//        print("RMSNorm: sum_sq=%f, mean_sq=%f, safe_value=%f, rms=%f, scale=%f, eps=%f\n",
//               sum_sq_f32, mean_sq, safe_mean_sq, rms, scale, args.eps);
    }

    // Apply normalization and weight scaling
    for (uint i = tpitg; i < args.ne00; i += ntg) {
        half w = weight[i];
        // Safety check on weights
        if (isnan(float(w)) || isinf(float(w))) {
            w = 1.0h; // Use safe default if weight is corrupt
        }
        y[i] = half( (float(x[i]) * scale) * float(w) );
    }
}
