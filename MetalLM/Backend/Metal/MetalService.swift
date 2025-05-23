import Foundation
import Metal
import MetalPerformanceShaders

// Argument struct for RMSNorm kernel
// (Can also be defined globally or in a shared header/file)
struct MetalRMSNormArgs {
    let eps: Float
    let ne00: UInt64
}
// Define the error enum here or keep it separate
enum MetalServiceError: Error {
    case kernelNotFound(String)
    case pipelineCreationFailed(String, Error)
    // Add other potential errors
}

// Define the Swift struct matching Metal
struct MetalRopeArgs {
    // Match fields in the Metal struct exactly
    var n_dims: UInt32 = 0
    var freq_base: Float = 0
    var freq_scale: Float = 1.0
    var pos_offset: Int32 = 0
    // YaRN Parameters
    var n_ctx_orig: Int32 = 0
    var ext_factor: Float = 0.0
    var attn_factor: Float = 1.0  // Keep default 1.0 for now
    var beta_fast: Float = 0.0
    var beta_slow: Float = 0.0
    // Flag
    var has_freq_factors: Bool = false
    // Strides removed for now, assuming kernel calculates offsets
}

/// Mirrors the RepeatKVHeadsArgs struct in AttentionKernels.metal
struct RepeatKVHeadsArgs {
    let num_kv_heads: UInt32  // Use fixed-size types matching Metal's uint
    let num_query_groups: UInt32
    let head_dim: UInt32
    let seq_len: UInt32  // Sequence length being processed

    // Ensure layout matches Metal struct if needed, though for simple types it's usually fine.
    // Swift automatically handles padding/alignment for these basic types to match Metal.
}

// Service class to manage Metal device, command queue, and compute pipelines
class MetalService {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    // Pipeline states for different kernels
    let dequantizeQ4KMF32PipelineState: MTLComputePipelineState  // Handles Q4_K_M AND Q4_K_S
    let dequantizeQ4KMF16PipelineState: MTLComputePipelineState  // Handles Q4_K_M AND Q4_K_S
    let dequantizeQ6KF32PipelineState: MTLComputePipelineState  // <-- ADD
    let dequantizeQ6KF16PipelineState: MTLComputePipelineState  // <-- ADD
    let convertF16toF32PipelineState: MTLComputePipelineState
    let rmsNormF16PipelineState: MTLComputePipelineState
    let ropeF16PipelineState: MTLComputePipelineState?  // Optional if kernel might fail loading
    let siluF16PipelineState: MTLComputePipelineState
    let mulF16PipelineState: MTLComputePipelineState
    let addF16PipelineState: MTLComputePipelineState
    let repeatKVHeadsPipelineState: MTLComputePipelineState

    static let shared = MetalService()

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Error: Metal is not supported on this device.")
            return nil
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            print("Error: Could not create Metal command queue.")
            return nil
        }
        self.commandQueue = commandQueue

        guard let library = device.makeDefaultLibrary() else {
            print("Error: Could not load default Metal library.")
            return nil
        }
        print("Metal library loaded successfully.")

        // --- Simplified Pipeline State Initialization ---
        do {
            // Helper to load function and create pipeline state
            func makePipeline(functionName: String) throws -> MTLComputePipelineState {
                guard let function = library.makeFunction(name: functionName) else {
                    throw MetalServiceError.kernelNotFound(functionName)
                }
                do {
                    return try device.makeComputePipelineState(function: function)
                } catch {
                    throw MetalServiceError.pipelineCreationFailed(functionName, error)
                }
            }

            // Load all required pipelines
            self.dequantizeQ4KMF32PipelineState = try makePipeline(
                functionName: "kernel_dequantize_q4_K_f32")
            print("Pipeline state created for kernel_dequantize_q4_K_f32")

            self.dequantizeQ4KMF16PipelineState = try makePipeline(
                functionName: "kernel_dequantize_q4_K_f16")
            print("Pipeline state created for kernel_dequantize_q4_K_f16")

            // *** ADD Q6_K Pipelines ***
            self.dequantizeQ6KF32PipelineState = try makePipeline(
                functionName: "kernel_dequantize_q6_K_f32")
            print("Pipeline state created for kernel_dequantize_q6_K_f32")

            self.dequantizeQ6KF16PipelineState = try makePipeline(
                functionName: "kernel_dequantize_q6_K_f16")
            print("Pipeline state created for kernel_dequantize_q6_K_f16")
            // *** END ADD ***

            self.convertF16toF32PipelineState = try makePipeline(
                functionName: "kernel_convert_f16_f32")
            print("Pipeline state created for kernel_convert_f16_f32")

            self.rmsNormF16PipelineState = try makePipeline(functionName: "kernel_rms_norm_f16")
            print("Pipeline state created for kernel_rms_norm_f16")

            // Load RoPE pipeline state
            if let fn = library.makeFunction(name: "kernel_rope_f16_inplace") {
                self.ropeF16PipelineState = try device.makeComputePipelineState(function: fn)
                print("Pipeline state created for kernel_rope_f16_inplace")
            } else {
                print("Error: kernel_rope_f16_inplace function not found.")
                self.ropeF16PipelineState = nil  // Or handle error more strictly
            }

            // --- ADD SiLU Pipeline Loading ---
            self.siluF16PipelineState = try makePipeline(functionName: "kernel_silu_f16")
            print("Pipeline state created for kernel_silu_f16")
            // --- END ADD ---

            self.mulF16PipelineState = try makePipeline(functionName: "kernel_mul_f16")
            print("Pipeline state created for kernel_mul_f16")

            self.addF16PipelineState = try makePipeline(functionName: "kernel_add_f16")
            print("Pipeline state created for kernel_add_f16")

            self.repeatKVHeadsPipelineState = try makePipeline(
                functionName: "kernel_repeat_kv_heads_f16")
            print("Pipeline state created for kernel_repeat_kv_heads_f16")

        } catch let error as MetalServiceError {
            // Catch specific errors from makePipeline helper
            print("MetalService initialization failed: \(error)")
            return nil
        } catch {
            // Catch unexpected errors during pipeline creation
            print("MetalService initialization failed with unexpected error: \(error)")
            return nil
        }
        // --- End Simplified Init ---

        print("MetalService initialized successfully for device: \(device.name)")
    }

    // MARK: - Dequantization Functions
    // (Keep dequantizeQ4KM_to_f32 and dequantizeQ4KM_to_f16 as they handle both S and M)

    /// Dequantizes a Q4_K_M buffer into a new Float32 buffer using Metal.
    /// Handles both Q4_K_M and Q4_K_S types as they share the kernel.
    func dequantizeQ4KM_to_f32(quantizedBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer? {
        guard elementCount > 0 else {
            print("Warning: Attempting Q4_K -> F32 dequantization with zero elements.")
            return device.makeBuffer(length: 0, options: .storageModeShared)
        }

        // Basic size validation (using 144 bytes per block for Q4_K_M/S)
        let expectedInputSize = ((elementCount + 255) / 256) * 144
        guard quantizedBuffer.length >= expectedInputSize else {
            print(
                "Error: Input Q4_K buffer length (\(quantizedBuffer.length)) is less than expected (\(expectedInputSize)) for \(elementCount) elements."
            )
            return nil
        }

        // Calculate output buffer size
        let outputBufferSize = elementCount * MemoryLayout<Float>.size
        guard
            let outputBuffer = device.makeBuffer(
                length: outputBufferSize, options: .storageModeShared)
        else {
            print("Error: Failed to create output buffer for Q4_K -> F32 dequantization.")
            return nil
        }
        outputBuffer.label = (quantizedBuffer.label ?? "unknown") + "_q4k_f32"

        // Create buffer for element count argument
        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg,
                length: MemoryLayout<UInt64>.size,
                options: .storageModeShared)
        else {
            print("Error: Failed to create nelements buffer for Q4_K -> F32 dequantization.")
            return nil
        }

        // Encode and dispatch kernel
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Error: Failed to create command buffer or encoder for Q4_K -> F32.")
            return nil
        }
        encoder.label = "Q4K -> F32 Encoder"

        encoder.setComputePipelineState(dequantizeQ4KMF32PipelineState)  // Use the M pipeline state for both M & S
        encoder.setBuffer(quantizedBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)  // dst
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 2)  // nelements

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(
            dequantizeQ4KMF32PipelineState.maxTotalThreadsPerThreadgroup, 256)  // 256 is often reasonable
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()  // Wait for simplicity

        if let error = commandBuffer.error {
            print("Error during Q4_K -> f32 dequantization kernel execution: \(error)")
            return nil
        }
        print("Successfully executed Q4_K -> F32 dequantization kernel.")
        return outputBuffer
    }

    /// Dequantizes a Q4_K_M buffer into a new Float16 buffer using Metal.
    /// Handles both Q4_K_M and Q4_K_S types as they share the kernel.
    func dequantizeQ4KM_to_f16(quantizedBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer? {
        guard elementCount > 0 else {
            print("Warning: Attempting Q4_K -> F16 dequantization with zero elements.")
            return device.makeBuffer(length: 0, options: .storageModeShared)
        }

        // Basic size validation (using 144 bytes per block for Q4_K_M/S)
        let expectedInputSize = ((elementCount + 255) / 256) * 144
        guard quantizedBuffer.length >= expectedInputSize else {
            print(
                "Error: Input Q4_K buffer length (\(quantizedBuffer.length)) is less than expected (\(expectedInputSize)) for \(elementCount) elements."
            )
            return nil
        }

        // Calculate output buffer size
        let outputBufferSize = elementCount * MemoryLayout<Float16>.size  // Use Float16 size
        guard
            let outputBuffer = device.makeBuffer(
                length: outputBufferSize, options: .storageModeShared)
        else {
            print("Error: Failed to create output buffer for Q4_K -> F16 dequantization.")
            return nil
        }
        outputBuffer.label = (quantizedBuffer.label ?? "unknown") + "_q4k_f16"

        // Create buffer for element count argument
        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg,
                length: MemoryLayout<UInt64>.size,
                options: .storageModeShared)
        else {
            print("Error: Failed to create nelements buffer for Q4_K -> F16 dequantization.")
            return nil
        }

        // Encode and dispatch kernel
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Error: Failed to create command buffer or encoder for Q4_K -> F16.")
            return nil
        }
        encoder.label = "Q4K -> F16 Encoder"

        // Use the F16 pipeline state
        encoder.setComputePipelineState(dequantizeQ4KMF16PipelineState)  // Use the M pipeline state for both M & S
        encoder.setBuffer(quantizedBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)  // dst (now half type)
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 2)  // nelements

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(
            dequantizeQ4KMF16PipelineState.maxTotalThreadsPerThreadgroup, 256)
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()  // Wait for simplicity

        if let error = commandBuffer.error {
            print("Error during Q4_K -> f16 dequantization kernel execution: \(error)")
            return nil
        }
        print("Successfully executed Q4_K -> F16 dequantization kernel.")
        return outputBuffer
    }

    // *** ADD Q6_K Functions ***

    /// Dequantizes a Q6_K buffer into a new Float32 buffer using Metal.
    func dequantizeQ6K_to_f32(quantizedBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer? {
        guard elementCount > 0 else {
            print("Warning: Attempting Q6_K -> F32 dequantization with zero elements.")
            return device.makeBuffer(length: 0, options: .storageModeShared)
        }

        // Validate input size (using 210 bytes per block for Q6_K)
        let blocks = (elementCount + 255) / 256
        let expectedInputSize = blocks * 210
        guard quantizedBuffer.length >= expectedInputSize else {
            print(
                "Error: Input Q6_K buffer length (\(quantizedBuffer.length)) is less than expected (\(expectedInputSize)) for \(elementCount) elements."
            )
            return nil
        }

        // Calculate output buffer size
        let outputBufferSize = elementCount * MemoryLayout<Float>.size
        guard
            let outputBuffer = device.makeBuffer(
                length: outputBufferSize, options: .storageModeShared)
        else {
            print("Error: Failed to create output buffer for Q6_K -> F32 dequantization.")
            return nil
        }
        outputBuffer.label = (quantizedBuffer.label ?? "unknown") + "_q6k_f32"

        // Element count buffer
        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg, length: MemoryLayout<UInt64>.size, options: .storageModeShared
            )
        else {
            print("Error: Failed to create nelements buffer for Q6_K -> F32 dequantization.")
            return nil
        }

        // Encode and dispatch kernel
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Error: Failed to create command buffer or encoder for Q6_K -> F32.")
            return nil
        }
        encoder.label = "Q6K -> F32 Encoder"

        encoder.setComputePipelineState(dequantizeQ6KF32PipelineState)  // Use Q6_K pipeline
        encoder.setBuffer(quantizedBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)  // dst
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 2)  // nelements

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(dequantizeQ6KF32PipelineState.maxTotalThreadsPerThreadgroup, 256)
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()  // Wait for simplicity

        if let error = commandBuffer.error {
            print("Error during Q6_K -> f32 dequantization kernel execution: \(error)")
            return nil
        }
        print("Successfully executed Q6_K -> F32 dequantization kernel.")
        return outputBuffer
    }

    /// Dequantizes a Q6_K buffer into a new Float16 buffer using Metal.
    func dequantizeQ6K_to_f16(quantizedBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer? {
        guard elementCount > 0 else {
            print("Warning: Attempting Q6_K -> F16 dequantization with zero elements.")
            return device.makeBuffer(length: 0, options: .storageModeShared)
        }

        // Validate input size (using 210 bytes per block for Q6_K)
        let blocks = (elementCount + 255) / 256
        let expectedInputSize = blocks * 210
        guard quantizedBuffer.length >= expectedInputSize else {
            print(
                "Error: Input Q6_K buffer length (\(quantizedBuffer.length)) is less than expected (\(expectedInputSize)) for \(elementCount) elements."
            )
            return nil
        }

        // Calculate output buffer size (Float16)
        let outputBufferSize = elementCount * MemoryLayout<Float16>.size
        guard
            let outputBuffer = device.makeBuffer(
                length: outputBufferSize, options: .storageModeShared)
        else {
            print("Error: Failed to create output buffer for Q6_K -> F16 dequantization.")
            return nil
        }
        outputBuffer.label = (quantizedBuffer.label ?? "unknown") + "_q6k_f16"

        // Element count buffer
        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg, length: MemoryLayout<UInt64>.size, options: .storageModeShared
            )
        else {
            print("Error: Failed to create nelements buffer for Q6_K -> F16 dequantization.")
            return nil
        }

        // Encode and dispatch kernel
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Error: Failed to create command buffer or encoder for Q6_K -> F16.")
            return nil
        }
        encoder.label = "Q6K -> F16 Encoder"

        encoder.setComputePipelineState(dequantizeQ6KF16PipelineState)  // Use Q6_K F16 pipeline
        encoder.setBuffer(quantizedBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)  // dst (half)
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 2)  // nelements

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(dequantizeQ6KF16PipelineState.maxTotalThreadsPerThreadgroup, 256)
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()  // Wait for simplicity

        if let error = commandBuffer.error {
            print("Error during Q6_K -> f16 dequantization kernel execution: \(error)")
            return nil
        }
        print("Successfully executed Q6_K -> F16 dequantization kernel.")
        return outputBuffer
    }

    // *** END ADD Q6_K ***

    // MARK: - Conversion Functions
    // (Keep convertF16toF32)

    /// Converts an F16 buffer into a new F32 buffer using Metal.
    func convertF16toF32(inputBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer? {
        guard elementCount > 0 else {
            print("Warning: Attempting F16 -> F32 conversion with zero elements.")
            return device.makeBuffer(length: 0, options: .storageModeShared)
        }

        let expectedInputSize = elementCount * MemoryLayout<Float16>.size
        guard inputBuffer.length >= expectedInputSize else {
            print(
                "Error: Input F16 buffer length (\(inputBuffer.length)) is less than expected (\(expectedInputSize)) for \(elementCount) elements."
            )
            return nil
        }

        let outputBufferSize = elementCount * MemoryLayout<Float>.size
        guard
            let outputBuffer = device.makeBuffer(
                length: outputBufferSize, options: .storageModeShared)
        else {
            print("Error: Failed to create output buffer for F16->F32 conversion.")
            return nil
        }
        outputBuffer.label = (inputBuffer.label ?? "unknown") + "_f16_to_f32"

        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg, length: MemoryLayout<UInt64>.size, options: .storageModeShared
            )
        else {
            print("Error: Failed to create nelements buffer for F16->F32 conversion.")
            return nil
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Error: Failed to create command buffer or encoder for F16->F32 conversion.")
            return nil
        }
        encoder.label = "F16 -> F32 Encoder"

        encoder.setComputePipelineState(convertF16toF32PipelineState)  // Use F16->F32 pipeline
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)  // src (f16)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)  // dst (f32)
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 2)  // elementCount

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(convertF16toF32PipelineState.maxTotalThreadsPerThreadgroup, 1024)  // Can be higher for simple conversions
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()  // Wait for simplicity

        if let error = commandBuffer.error {
            print("Error during F16->F32 conversion kernel execution: \(error)")
            return nil
        }
        print("Successfully executed F16->F32 conversion kernel.")
        return outputBuffer
    }

    // Inside MetalService class

    // MARK: - Inference Op Functions

    /// Encodes the RMS Normalization operation onto a command buffer.
    /// Normalizes each row of the input buffer and scales by the weight buffer.
    ///
    /// - Parameters:
    ///   - commandBuffer: The command buffer to encode the operation onto. <--- ADDED
    ///   - inputBuffer: The buffer containing input data (Float16).
    ///   - weightBuffer: The buffer containing the scaling weights (gamma) (Float16).
    ///   - outputBuffer: The buffer to write the normalized output (Float16).
    ///   - rowCount: The number of rows (or batch size) to process.
    ///   - elementCountPerRow: The number of elements in each row (embedding dimension).
    ///   - eps: Epsilon value for numerical stability.
    ///   - label: Optional label for the command encoder.
    /// - Returns: True if encoding was successful, false otherwise.
    func encodeRMSNormF16(  // <-- Renamed to encode...
        commandBuffer: MTLCommandBuffer,  // <-- ADDED parameter
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        rowCount: Int,
        elementCountPerRow: Int,
        eps: Float = 1e-5,
        label: String? = nil
    ) -> Bool {  // <-- Return Bool indicating encoding success
        guard rowCount > 0, elementCountPerRow > 0 else {
            print("Warning [RMSNorm Enc]: rowCount or elementCountPerRow is zero.")
            return true  // Nothing to encode
        }

        // --- Buffer Size Validation ---
        let expectedInputSize = rowCount * elementCountPerRow * MemoryLayout<Float16>.stride
        let expectedWeightSize = elementCountPerRow * MemoryLayout<Float16>.stride  // Weights are per element in a row
        guard inputBuffer.length >= expectedInputSize else {
            print(
                "Error [RMSNorm Enc]: Input buffer too small. Needs \(expectedInputSize), has \(inputBuffer.length)."
            )
            return false
        }
        guard weightBuffer.length >= expectedWeightSize else {
            print(
                "Error [RMSNorm Enc]: Weight buffer too small. Needs \(expectedWeightSize), has \(weightBuffer.length)."
            )
            return false
        }
        guard outputBuffer.length >= expectedInputSize else {  // Output has same size as input
            print(
                "Error [RMSNorm Enc]: Output buffer too small. Needs \(expectedInputSize), has \(outputBuffer.length)."
            )
            return false
        }

        // --- Prepare Arguments Buffer ---
        // Use the Swift struct here when creating the buffer
        var args = MetalRMSNormArgs(eps: eps, ne00: UInt64(elementCountPerRow))  // ne00 is element count per row
        guard
            let argsBuffer = device.makeBuffer(
                bytes: &args, length: MemoryLayout<MetalRMSNormArgs>.size,
                options: .storageModeShared  // Shared is fine for small, read-only args
            )
        else {
            print("Error [RMSNorm Enc]: Failed to create args buffer.")
            return false
        }
        argsBuffer.label = label.map { "\($0)_Args" } ?? "RMSNorm_Args"

        // --- Create Compute Encoder ---
        // Use the *provided* commandBuffer
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Error [RMSNorm Enc]: Failed to create compute command encoder.")
            return false
        }
        encoder.label = label ?? "RMSNorm F16 Encoder"

        // --- Set Pipeline and Buffers ---
        encoder.setComputePipelineState(rmsNormF16PipelineState)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)  // weight (gamma)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)  // dst
        encoder.setBuffer(argsBuffer, offset: 0, index: 3)  // args

        // --- Calculate Threadgroup Size and Memory ---
        let maxThreads = rmsNormF16PipelineState.maxTotalThreadsPerThreadgroup
        // Ensure threadGroupWidth calculation doesn't rely on potentially undefined behavior if elementCountPerRow is 0 or 1
        var threadGroupWidth = min(maxThreads, 512)  // Default reasonable size
        if elementCountPerRow > 0 {  // Check before using flsl
            if elementCountPerRow < threadGroupWidth {
                // Find nearest power of 2 >= elementCountPerRow, capped by maxThreads
                let highestBitIndex = flsl(elementCountPerRow - 1)  // Find index of highest set bit (0-based)
                // Use 1 << highestBitIndex only if highestBitIndex is valid (elementCountPerRow > 0)
                if highestBitIndex >= 0 {
                    threadGroupWidth = 1 << highestBitIndex  // Power of 2 <= elementCountPerRow
                    if threadGroupWidth < elementCountPerRow {  // If not exact power of 2, go to next power of 2
                        threadGroupWidth <<= 1
                    }
                } else {  // elementCountPerRow was 1
                    threadGroupWidth = 1
                }
                threadGroupWidth = max(32, threadGroupWidth)  // Ensure at least SIMD group size (or reasonable minimum)
                threadGroupWidth = min(threadGroupWidth, maxThreads)  // Cap at max allowed
            }
        } else {
            threadGroupWidth = min(32, maxThreads)  // Use a minimum if elementCountPerRow is 0 (though caught earlier)
        }
        threadGroupWidth = max(32, threadGroupWidth)  // Final check for minimum sensible size

        // Threadgroup memory for reduction sum
        let numSimdGroups = (threadGroupWidth + 31) / 32  // Ceiling division

        // Calculate required size IF reducing (numSimdGroups > 1)
        // Ensure the size is a multiple of 16 if > 0
        var requiredMemoryForReduction = 0
        if numSimdGroups > 1 {
            let calculatedSize = numSimdGroups * MemoryLayout<Float>.size
            // Ensure size is padded up to the next multiple of 16 if needed
            requiredMemoryForReduction = (calculatedSize + 15) & ~15  // Pad to multiple of 16 bytes
        }

        // --- FIX: Allocate a minimum of 16 bytes, or the padded required size ---
        // Use the calculated padded size if > 0, otherwise use the minimum alignment (16 bytes).
        let threadGroupMemoryLength =
            (requiredMemoryForReduction > 0) ? requiredMemoryForReduction : 16

        // Always set threadgroup memory length with the potentially adjusted, aligned, non-zero value
        encoder.setThreadgroupMemoryLength(threadGroupMemoryLength, index: 0)
        print(
            "      Setting threadgroup memory length to \(threadGroupMemoryLength) for index 0 (numSimdGroups=\(numSimdGroups)). Required(padded): \(requiredMemoryForReduction)"
        )
        // --- END FIX ---

        // --- Dispatch ---
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)
        // Grid size is based on the number of rows we need to process independently
        let gridSize = MTLSize(width: rowCount, height: 1, depth: 1)

        print(
            "Encoding RMSNormF16 (\(label ?? "No Label")): Grid=\(gridSize.width)x\(gridSize.height)x\(gridSize.depth), Group=\(threadGroupSize.width)x\(threadGroupSize.height)x\(threadGroupSize.depth)"
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)

        // --- End Encoding ---
        encoder.endEncoding()

        // DO NOT commit or wait here - that's the caller's responsibility
        // if let error = commandBuffer.error { ... } // Error checking happens after commit

        print("Successfully encoded RMSNorm F16 kernel (\(label ?? "No Label")).")
        return true  // Indicate encoding success
    }

    // MARK: - RoPE Functions

    /// Applies Rotary Position Embeddings (RoPE) to a buffer in-place.
    /// Assumes buffer layout allows kernel to calculate offsets based on grid/thread IDs.
    // In MetalService.swift

    func applyRoPE(
        commandBuffer: MTLCommandBuffer,
        buffer: MTLBuffer,
        ropeFrequencies: MTLBuffer?,
        config: LlamaConfig,
        posOffset: Int,
        sequenceLength: Int,
        numHeads: Int,
        headDim: Int
    ) -> Bool {

        guard let pipeline = ropeF16PipelineState else {
            print("Error [RoPE]: Pipeline state not available.")
            return false
        }

        // --- Validate Dimensions ---
        // ... (validation remains the same) ...
        guard sequenceLength > 0, numHeads > 0, headDim > 0 else { /* ... */ return false }
        let ropeDims = config.ropeDimensionCount
        guard ropeDims > 0, ropeDims <= headDim, ropeDims % 2 == 0 else { /* ... */ return false }

        // --- 1. Setup Arguments ---
        var args = MetalRopeArgs()
        // ... (populate args as before) ...
        args.n_dims = UInt32(ropeDims)
        args.freq_base = config.ropeFreqBase
        args.freq_scale =
            (config.ropeScalingType == .linear || config.ropeScalingType == .yarn)
            ? config.ropeScalingFactor : 1.0
        if args.freq_scale == 0.0 { args.freq_scale = 1.0 }
        args.pos_offset = Int32(posOffset)
        args.n_ctx_orig = Int32(
            config.ropeScalingOrigContextLength > 0
                ? config.ropeScalingOrigContextLength : config.sequenceLength)
        args.ext_factor = (config.ropeScalingType == .yarn) ? config.ropeScalingFactor : 0.0
        args.attn_factor = 1.0
        args.beta_fast = config.ropeScalingBetaFast
        args.beta_slow = config.ropeScalingBetaSlow
        args.has_freq_factors = (ropeFrequencies != nil)

        // --- Use STRIDE ---
        let swiftStructStride = MemoryLayout<MetalRopeArgs>.stride  // Use stride (should be 40)

        print(
            "DEBUG: Swift MetalRopeArgs size = \(MemoryLayout<MetalRopeArgs>.size), stride = \(swiftStructStride)"
        )
        print("DEBUG: Creating args buffer with size = \(swiftStructStride)")  // Should print 40

        // Create args buffer using the Swift struct's stride
        guard
            let argsBuffer = device.makeBuffer(
                bytes: &args,
                length: swiftStructStride,  // <-- USE STRIDE (should be 40)
                options: .storageModeShared)
        else {
            print("Error [RoPE]: Failed to create args buffer with size \(swiftStructStride).")
            return false
        }
        // --- END FIX ---

        // --- 2. Encode ---
        // ... (rest of encoding and dispatch remains the same) ...
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        encoder.label = "RoPE Kernel Encoder"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(argsBuffer, offset: 0, index: 0)  // Pass the 40-byte buffer
        encoder.setBuffer(buffer, offset: 0, index: 1)
        encoder.setBuffer(ropeFrequencies, offset: 0, index: 2)

        // --- 3. Dispatch ---
        // ... (dispatch logic remains the same) ...
        let gridWidth = ropeDims / 2
        let gridHeight = numHeads
        let gridDepth = sequenceLength
        guard gridWidth > 0, gridHeight > 0, gridDepth > 0 else {
            print(
                "Error [RoPE]: Calculated grid dimensions are invalid [\(gridWidth), \(gridHeight), \(gridDepth)]"
            )
            encoder.endEncoding()
            return false
        }
        let tpgW = min(pipeline.maxTotalThreadsPerThreadgroup, 32)
        let threadsPerGroup = MTLSize(width: tpgW, height: 1, depth: 1)
        let numGroupsWidth = (gridWidth + tpgW - 1) / tpgW
        let numThreadgroups = MTLSize(width: numGroupsWidth, height: gridHeight, depth: gridDepth)
        print(
            "[RoPE Dispatch] Grid=\(gridWidth)x\(gridHeight)x\(gridDepth), Groups=\(numThreadgroups.width)x\(numThreadgroups.height)x\(numThreadgroups.depth), GroupSize=\(threadsPerGroup.width)x\(threadsPerGroup.height)x\(threadsPerGroup.depth)"
        )
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        return true
    }

    /// Encodes a matrix multiplication operation (C = alpha * op(A) * op(B) + beta * C) onto the command buffer using MPS.
    /// Handles Float16 precision.
    ///
    /// - Parameters:
    ///   - commandBuffer: The command buffer to encode onto.
    ///   - inputA: Buffer containing the left matrix (A).
    ///   - inputB: Buffer containing the right matrix (B).
    ///   - outputC: Buffer where the result matrix (C) will be written.
    ///   - rowsA: Actual number of rows in the inputA buffer layout.
    ///   - colsA: Actual number of columns in the inputA buffer layout.
    ///   - rowsB: Actual number of rows in the inputB buffer layout.
    ///   - colsB: Actual number of columns in the inputB buffer layout.
    ///   - transposeA: Whether to treat A as transposed for the multiplication.
    ///   - transposeB: Whether to treat B as transposed for the multiplication.
    ///   - alpha: Scaling factor for the product. Defaults to 1.0.
    ///   - beta: Scaling factor for the initial value of C. Defaults to 0.0.
    ///   - label: Optional label for the MPS kernel encoding.
    /// - Returns: True if encoding was successful, false otherwise.
    func encodeMPSMatrixMultiply(
        commandBuffer: MTLCommandBuffer,
        inputA: MTLBuffer,
        inputB: MTLBuffer,
        outputC: MTLBuffer,
        // Dimensions of matrices AS STORED IN BUFFERS
        rowsA: Int,
        colsA: Int,
        rowsB: Int,
        colsB: Int,
        // Transpose flags for the MATH operation
        transposeA: Bool = false,
        transposeB: Bool = false,
        alpha: Double = 1.0,
        beta: Double = 0.0,
        label: String? = nil
    ) -> Bool {

        // --- Basic Validation ---
        let bytesPerElement = MemoryLayout<Float16>.stride
        guard rowsA > 0, colsA > 0, rowsB > 0, colsB > 0 else {
            print("Error [MPS MatMul]: Buffer dimensions rowsA/colsA/rowsB/colsB must be positive.")
            return false
        }

        // --- Determine Mathematical Dimensions M, N, K based on transpose flags ---
        // C[M, N] = op(A)[M, K] * op(B)[K, N]
        let M = transposeA ? colsA : rowsA
        let K_A = transposeA ? rowsA : colsA  // K derived from A
        let K_B = transposeB ? colsB : rowsB  // K derived from B
        let N = transposeB ? rowsB : colsB

        // Validate K matches
        guard K_A == K_B else {
            print(
                "Error [MPS MatMul]: Inner dimensions do not match after transpose. K_A(\(K_A)) != K_B(\(K_B))."
            )
            print(
                "  rowsA=\(rowsA), colsA=\(colsA), rowsB=\(rowsB), colsB=\(colsB), tA=\(transposeA), tB=\(transposeB)"
            )
            return false
        }
        let K = K_A  // The common inner dimension

        // --- Calculate Row Bytes based on ACTUAL columns in buffers ---
        let rowBytesA_Int = colsA * bytesPerElement
        let rowBytesB_Int = colsB * bytesPerElement

        guard rowBytesA_Int > 0, rowBytesB_Int > 0 else {
            print("Error [MPS MatMul]: Calculated rowBytes must be positive.")
            return false
        }

        // --- Buffer Size Checks based on ACTUAL layout ---
        let expectedSizeA = rowsA * rowBytesA_Int
        let expectedSizeB = rowsB * rowBytesB_Int

        guard inputA.length >= expectedSizeA else { /* ... error ... */ return false }
        guard inputB.length >= expectedSizeB else { /* ... error ... */ return false }

        // --- Create Descriptors based on ACTUAL layout ---
        let descA = MPSMatrixDescriptor(
            rows: rowsA, columns: colsA, rowBytes: rowBytesA_Int, dataType: .float16)
        let descB = MPSMatrixDescriptor(
            rows: rowsB, columns: colsB, rowBytes: rowBytesB_Int, dataType: .float16)

        // --- FIX: Determine Output Type based on outputC buffer ---
        // Determine expected element size based on buffer length and dimensions
        let outputElementSize = (M > 0 && N > 0) ? (outputC.length / (M * N)) : 0
        let outputDataType: MPSDataType
        if outputElementSize == MemoryLayout<Float>.stride {
            outputDataType = .float32
            print("  DescC: Using MPSDataType.float32 (Inferred Size: \(outputElementSize))")
        } else if outputElementSize == MemoryLayout<Float16>.stride {
            outputDataType = .float16
             print("  DescC: Using MPSDataType.float16 (Inferred Size: \(outputElementSize))")
        } else {
            // Fallback or error - default to F16? Or throw? Let's default and warn.
            print("  DescC WARNING: Could not infer element size (\(outputElementSize)) for output buffer \(outputC.label ?? "C"). Defaulting to float16.")
            outputDataType = .float16
        }
        // --- END FIX ---

        // Ensure rowBytesC_Int is correct for the INTENDED output type
        let rowBytesC_Int = N * outputElementSize // Use inferred/actual element size

        guard rowBytesC_Int > 0 else {
            print("Error [MPS MatMul]: Calculated rowBytes must be positive.")
            return false
        }

        let expectedSizeC = M * rowBytesC_Int  // Output C always has M rows
        guard outputC.length >= expectedSizeC else { /* ... error ... */ return false }

        let descC = MPSMatrixDescriptor(
            rows: M, columns: N, rowBytes: rowBytesC_Int, dataType: outputDataType) // Use determined type

        // --- Create MPSMatrix Objects ---
        let matrixA = MPSMatrix(buffer: inputA, descriptor: descA)
        let matrixB = MPSMatrix(buffer: inputB, descriptor: descB)
        let matrixC = MPSMatrix(buffer: outputC, descriptor: descC)

        // --- Create MPSMatrixMultiplication Kernel ---
        // Use the MATHEMATICAL dimensions M, N, K derived earlier
        let matMulKernel = MPSMatrixMultiplication(
            device: self.device,
            transposeLeft: transposeA,
            transposeRight: transposeB,
            resultRows: M,
            resultColumns: N,
            interiorColumns: K,
            alpha: alpha,
            beta: beta
        )
        matMulKernel.label = label ?? "MPSMatrixMultiplication"

        // --- Encode the Kernel ---
        matMulKernel.encode(
            commandBuffer: commandBuffer,
            leftMatrix: matrixA,
            rightMatrix: matrixB,
            resultMatrix: matrixC
        )

        print(
            "Successfully encoded \(matMulKernel.label ?? "MPS MatMul") M=\(M), N=\(N), K=\(K), tA=\(transposeA), tB=\(transposeB) (bufA:[\(rowsA)x\(colsA)], bufB:[\(rowsB)x\(colsB)])"
        )
        return true
    }

    /// Encodes a Softmax operation on the rows of a matrix using MPSMatrixSoftMax.
    ///
    /// - Parameters:
    ///   - commandBuffer: The command buffer to encode onto.
    ///   - inputMatrixBuffer: Buffer containing the input matrix (e.g., attention scores).
    ///   - outputMatrixBuffer: Buffer where the softmax result will be written.
    ///   - rows: Number of rows in the matrix (e.g., number of heads or batch size).
    ///   - columns: Number of columns in the matrix (e.g., sequence length). Softmax is applied across this dimension.
    ///   - label: Optional label for the MPS kernel encoding.
    /// - Returns: True if encoding was successful, false otherwise.
    func encodeMPSSoftMax(
        commandBuffer: MTLCommandBuffer,
        inputMatrixBuffer: MTLBuffer,
        outputMatrixBuffer: MTLBuffer,
        rows: Int,
        columns: Int,
        label: String? = nil
    ) -> Bool {

        guard rows > 0, columns > 0 else {
            print(
                "Error [MPS SoftMax]: Rows and columns must be positive. Got rows=\(rows), columns=\(columns)"
            )
            return false
        }

        let bytesPerElement = MemoryLayout<Float16>.stride
        let rowBytes = columns * bytesPerElement  // Each row has 'columns' elements
        guard rowBytes > 0 else {  // Ensure columns > 0 resulted in positive rowBytes
            print("Error [MPS SoftMax]: Calculated rowBytes is not positive.")
            return false
        }

        // --- Buffer Size Checks ---
        let expectedSize = rows * rowBytes
        guard inputMatrixBuffer.length >= expectedSize else {
            print(
                "Error [MPS SoftMax]: Input buffer too small. Needs \(expectedSize) (rows=\(rows), cols=\(columns)), has \(inputMatrixBuffer.length)."
            )
            return false
        }
        guard outputMatrixBuffer.length >= expectedSize else {
            print(
                "Error [MPS SoftMax]: Output buffer too small. Needs \(expectedSize) (rows=\(rows), cols=\(columns)), has \(outputMatrixBuffer.length)."
            )
            return false
        }

        // --- Create Descriptors & Matrices ---
        // MPSMatrixSoftMax operates row-wise, so descriptor matches input/output shape.
        // Initializers are non-failable according to compiler.
        let desc = MPSMatrixDescriptor(
            rows: rows, columns: columns, rowBytes: rowBytes, dataType: .float16)

        // Initialize directly, relying on size checks above.
        let inputMatrix = MPSMatrix(buffer: inputMatrixBuffer, descriptor: desc)
        let outputMatrix = MPSMatrix(buffer: outputMatrixBuffer, descriptor: desc)

        // --- Create and Encode Kernel ---
        // MPSMatrixSoftMax initializer is non-failable according to previous findings
        let softmaxKernel = MPSMatrixSoftMax(device: self.device)
        softmaxKernel.label = label ?? "MPSMatrixSoftMax"

        softmaxKernel.encode(
            commandBuffer: commandBuffer,
            inputMatrix: inputMatrix,
            resultMatrix: outputMatrix
        )

        print(
            "Successfully encoded \(softmaxKernel.label ?? "MPS SoftMax") for \(rows)x\(columns) matrix."
        )
        return true
    }

    /// Encodes the SiLU activation function onto a command buffer.
    /// Operates element-wise: output = input * sigmoid(input)
    ///
    /// - Parameters:
    ///   - inputBuffer: The buffer containing input data (Float16).
    ///   - outputBuffer: The buffer to write the output data (Float16).
    ///   - elementCount: The number of elements in the buffers.
    ///   - commandBuffer: The command buffer to encode the operation onto.
    /// - Returns: True if encoding was successful, false otherwise.
    func applySILU(
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        elementCount: Int,
        commandBuffer: MTLCommandBuffer  // Takes command buffer as input
    ) -> Bool {
        guard elementCount > 0 else {
            print("Warning [SiLU]: Attempting to apply SiLU with zero elements.")
            // Technically okay, just encode nothing.
            return true
        }

        // Basic Size Check (Optional but good practice)
        let bufferSize = elementCount * MemoryLayout<Float16>.stride
        guard inputBuffer.length >= bufferSize, outputBuffer.length >= bufferSize else {
            print(
                "Error [SiLU]: Buffer size mismatch. Need \(bufferSize), Input=\(inputBuffer.length), Output=\(outputBuffer.length)"
            )
            return false
        }

        // Create buffer for element count argument
        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg,
                length: MemoryLayout<UInt64>.size,
                options: .storageModeShared  // Or .storageModePrivate if only GPU access needed after creation
            )
        else {
            print("Error [SiLU]: Failed to create nelements buffer.")
            return false
        }
        nelementsBuffer.label = "SiLU_nelements"

        // Encode kernel
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Error [SiLU]: Failed to create compute command encoder.")
            return false
        }
        encoder.label = "SiLU Kernel Encoder"

        encoder.setComputePipelineState(siluF16PipelineState)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)  // dst
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 2)  // elementCount

        // Calculate grid and threadgroup sizes
        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(siluF16PipelineState.maxTotalThreadsPerThreadgroup, 1024)  // Simple kernel can use larger groups
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        print("Successfully encoded SiLU kernel for \(elementCount) elements.")
        return true
    }

    /// Encodes an element-wise multiplication operation (C = A * B) onto a command buffer.
    func applyElementWiseMul(
        inputBufferA: MTLBuffer,
        inputBufferB: MTLBuffer,
        outputBufferC: MTLBuffer,
        elementCount: Int,
        commandBuffer: MTLCommandBuffer
    ) -> Bool {
        guard elementCount > 0 else { return true }

        let bufferSize = elementCount * MemoryLayout<Float16>.stride
        guard inputBufferA.length >= bufferSize,
            inputBufferB.length >= bufferSize,
            outputBufferC.length >= bufferSize
        else {
            print("Error [Mul]: Buffer size mismatch.")
            // Add details if needed
            return false
        }

        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg, length: MemoryLayout<UInt64>.size, options: .storageModeShared
            )
        else {
            print("Error [Mul]: Failed to create nelements buffer.")
            return false
        }
        nelementsBuffer.label = "Mul_nelements"

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Error [Mul]: Failed to create compute command encoder.")
            return false
        }
        encoder.label = "ElementWise Mul Kernel Encoder"

        encoder.setComputePipelineState(mulF16PipelineState)
        encoder.setBuffer(inputBufferA, offset: 0, index: 0)  // a
        encoder.setBuffer(inputBufferB, offset: 0, index: 1)  // b
        encoder.setBuffer(outputBufferC, offset: 0, index: 2)  // c
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 3)  // ne

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(mulF16PipelineState.maxTotalThreadsPerThreadgroup, 1024)
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        print("Successfully encoded ElementWise Mul kernel for \(elementCount) elements.")
        return true
    }

    func applyElementWiseAdd(
        inputBufferA: MTLBuffer,
        inputBufferB: MTLBuffer,
        outputBufferC: MTLBuffer,
        elementCount: Int,
        commandBuffer: MTLCommandBuffer
    ) -> Bool {
        guard elementCount > 0 else { return true }

        let bufferSize = elementCount * MemoryLayout<Float16>.stride
        guard inputBufferA.length >= bufferSize,
            inputBufferB.length >= bufferSize,
            outputBufferC.length >= bufferSize
        else {
            print("Error [Add]: Buffer size mismatch.")
            return false
        }

        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg, length: MemoryLayout<UInt64>.size, options: .storageModeShared
            )
        else {
            print("Error [Add]: Failed to create nelements buffer.")
            return false
        }
        nelementsBuffer.label = "Add_nelements"

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Error [Add]: Failed to create compute command encoder.")
            return false
        }
        encoder.label = "ElementWise Add Kernel Encoder"

        encoder.setComputePipelineState(addF16PipelineState)
        encoder.setBuffer(inputBufferA, offset: 0, index: 0)  // a
        encoder.setBuffer(inputBufferB, offset: 0, index: 1)  // b
        encoder.setBuffer(outputBufferC, offset: 0, index: 2)  // c
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 3)  // ne

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(addF16PipelineState.maxTotalThreadsPerThreadgroup, 1024)
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        print("Successfully encoded ElementWise Add kernel for \(elementCount) elements.")
        return true
    }

    /// Encodes the kernel to repeat KV heads for Grouped Query Attention.
    /// Takes K or V with n_kv_heads and expands it to n_heads.
    ///
    /// - Parameters:
    ///   - sourceBuffer: Input K or V buffer (layout: [seq_len, n_kv_head, head_dim]).
    ///   - destinationBuffer: Output K or V buffer (layout: [seq_len, n_head, head_dim]).
    ///   - numKVHeads: Number of key/value heads in the source.
    ///   - numQueryGroups: Ratio n_head / n_kv_head.
    ///   - headDim: Dimension of each head.
    ///   - seqLen: Current sequence length being processed (number of rows in the source/dest seq dim).
    ///   - commandBuffer: The command buffer to encode onto.
    /// - Returns: True if encoding was successful, false otherwise.
    func applyRepeatKVHeads(
        sourceBuffer: MTLBuffer,
        destinationBuffer: MTLBuffer,
        numKVHeads: Int,
        numQueryGroups: Int,
        headDim: Int,
        seqLen: Int,
        commandBuffer: MTLCommandBuffer
    ) -> Bool {
        guard numKVHeads > 0, numQueryGroups > 0, headDim > 0, seqLen > 0 else {
            print("Error [RepeatKV]: Invalid dimensions provided.")
            return false
        }

        let nHead = numKVHeads * numQueryGroups

        // --- Validate Buffer Sizes ---
        let bytesPerElement = MemoryLayout<Float16>.stride
        let expectedSourceSize = seqLen * numKVHeads * headDim * bytesPerElement
        let expectedDestSize = seqLen * nHead * headDim * bytesPerElement

        guard sourceBuffer.length >= expectedSourceSize else {
            print(
                "Error [RepeatKV]: Source buffer too small. Needs \(expectedSourceSize), has \(sourceBuffer.length)."
            )
            return false
        }
        guard destinationBuffer.length >= expectedDestSize else {
            print(
                "Error [RepeatKV]: Destination buffer too small. Needs \(expectedDestSize), has \(destinationBuffer.length)."
            )
            return false
        }

        // --- Prepare Arguments ---
        var args = RepeatKVHeadsArgs(
            num_kv_heads: UInt32(numKVHeads),
            num_query_groups: UInt32(numQueryGroups),
            head_dim: UInt32(headDim),
            seq_len: UInt32(seqLen)
        )

        guard
            let argsBuffer = device.makeBuffer(
                bytes: &args,
                length: MemoryLayout<RepeatKVHeadsArgs>.size,  // Use size of Swift struct
                options: .storageModeShared
            )
        else {
            print("Error [RepeatKV]: Failed to create args buffer.")
            return false
        }
        argsBuffer.label = "RepeatKVHeads_Args"

        // --- Encode Kernel ---
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Error [RepeatKV]: Failed to create compute command encoder.")
            return false
        }
        encoder.label = "RepeatKVHeads Kernel Encoder"

        encoder.setComputePipelineState(repeatKVHeadsPipelineState)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(destinationBuffer, offset: 0, index: 1)  // dst
        encoder.setBuffer(argsBuffer, offset: 0, index: 2)  // args

        // --- Dispatch Threads ---
        // Grid size matches the total number of elements in the *destination* buffer
        let totalDestElements = expectedDestSize / bytesPerElement
        guard totalDestElements > 0 else {  // Should be covered by dimension checks, but good practice
            print("Error [RepeatKV]: Calculated zero destination elements.")
            encoder.endEncoding()  // Need to end encoding before returning false
            return false
        }

        let gridSize = MTLSize(width: totalDestElements, height: 1, depth: 1)
        // Choose thread group size (can be reasonably large for simple copy)
        let threadGroupWidth = min(repeatKVHeadsPipelineState.maxTotalThreadsPerThreadgroup, 1024)
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        print(
            "Successfully encoded RepeatKVHeads kernel for DestSize: \(seqLen)x\(nHead)x\(headDim)."
        )
        return true
    }
}
