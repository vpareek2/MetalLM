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

    // MARK: - Inference Op Functions
    func rmsNormF16(
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        rowCount: Int,
        elementCountPerRow: Int,
        eps: Float = 1e-5  // Corrected float literal
    ) -> Bool {
        guard rowCount > 0, elementCountPerRow > 0 else { return true }

        let expectedInputSize = rowCount * elementCountPerRow * MemoryLayout<Float16>.size
        let expectedWeightSize = elementCountPerRow * MemoryLayout<Float16>.size
        guard inputBuffer.length >= expectedInputSize,
            weightBuffer.length >= expectedWeightSize,
            outputBuffer.length >= expectedInputSize
        else {
            print("Error [RMSNormF16]: Buffer size mismatch.")
            // ... (print details) ...
            return false
        }

        // Use the Swift struct here when creating the buffer
        var args = MetalRMSNormArgs(eps: eps, ne00: UInt64(elementCountPerRow))
        guard
            let argsBuffer = device.makeBuffer(
                bytes: &args, length: MemoryLayout<MetalRMSNormArgs>.size,
                options: .storageModeShared)
        else {
            print("Error [RMSNormF16]: Failed to create args buffer.")
            return false
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Error [RMSNormF16]: Failed to create command buffer/encoder.")
            return false
        }
        encoder.label = "RMSNorm F16 Encoder"

        encoder.setComputePipelineState(rmsNormF16PipelineState)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)  // weight (gamma)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)  // dst
        encoder.setBuffer(argsBuffer, offset: 0, index: 3)  // args

        let maxThreads = rmsNormF16PipelineState.maxTotalThreadsPerThreadgroup
        var threadGroupWidth = min(maxThreads, 512)
        if elementCountPerRow < threadGroupWidth {
            threadGroupWidth = max(32, 1 << (flsl(elementCountPerRow - 1)))
            threadGroupWidth = min(threadGroupWidth, maxThreads)
        }
        threadGroupWidth = max(32, threadGroupWidth)

        let threadGroupMemoryLength = (threadGroupWidth > 32) ? 32 * MemoryLayout<Float>.size : 0
        if threadGroupMemoryLength > 0 {
            encoder.setThreadgroupMemoryLength(threadGroupMemoryLength, index: 0)
        }

        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)
        let gridSize = MTLSize(width: rowCount, height: 1, depth: 1)

        print(
            "Dispatching RMSNormF16: Grid=\(gridSize.width)x\(gridSize.height)x\(gridSize.depth), Group=\(threadGroupSize.width)x\(threadGroupSize.height)x\(threadGroupSize.depth)"
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            print("Error [RMSNormF16]: Kernel execution failed: \(error)")
            return false
        }
        print("Successfully executed RMSNorm F16 kernel.")  // Added success log
        return true
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

    // Inside the MetalService class:

    /// Encodes a matrix multiplication operation (C = alpha * op(A) * op(B) + beta * C) onto the command buffer using MPS.
    /// Handles Float16 precision.
    ///
    /// - Parameters:
    ///   - commandBuffer: The command buffer to encode onto.
    ///   - inputA: Buffer containing the left matrix (A).
    ///   - inputB: Buffer containing the right matrix (B) - often the weights.
    ///   - outputC: Buffer where the result matrix (C) will be written.
    ///   - M: Number of rows in matrix A (and C).
    ///   - N: Number of columns in matrix B (and C).
    ///   - K: Number of columns in matrix A / rows in matrix B (inner dimension).
    ///   - transposeA: Whether to transpose matrix A before multiplication.
    ///   - transposeB: Whether to transpose matrix B before multiplication. **Crucial for weight layout.**
    ///   - alpha: Scaling factor for the product (A*B). Defaults to 1.0.
    ///   - beta: Scaling factor for the initial value of C. Defaults to 0.0 (overwrite).
    ///   - label: Optional label for the MPS kernel encoding for debugging.
    /// - Returns: True if encoding was successful, false otherwise (basic check).
    func encodeMPSMatrixMultiply(
        commandBuffer: MTLCommandBuffer,
        inputA: MTLBuffer,
        inputB: MTLBuffer,
        outputC: MTLBuffer,
        M: Int,  // Use Int consistently
        N: Int,
        K: Int,
        transposeA: Bool = false,
        transposeB: Bool = false,  // Often TRUE if weights are loaded row-major
        alpha: Double = 1.0,
        beta: Double = 0.0,
        label: String? = nil
    ) -> Bool {

        // --- Basic Validation ---
        guard M > 0, N > 0, K > 0 else {
            print(
                "Error [MPS MatMul]: Dimensions M, N, K must be positive. Got M=\(M), N=\(N), K=\(K)"
            )
            return false
        }

        // --- Calculate Row Bytes (as Int) ---
        let bytesPerElement = MemoryLayout<Float16>.stride
        let rowBytesA_Int = K * bytesPerElement  // A is MxK, so K columns
        let rowBytesB_Int = N * bytesPerElement  // B is KxN, so N columns
        let rowBytesC_Int = N * bytesPerElement  // C is MxN, so N columns

        guard rowBytesA_Int > 0, rowBytesB_Int > 0, rowBytesC_Int > 0 else {
            print(
                "Error [MPS MatMul]: Calculated rowBytes must be positive. K=\(K), N=\(N)"
            )
            return false
        }

        // --- *** CRITICAL: Explicit Buffer Size Checks *** ---
        // Size is rows * rowBytes needed.
        let rowsA_desc = M
        let rowsB_desc = K  // KxN descriptor
        let rowsC_desc = M

        let expectedSizeA = rowsA_desc * rowBytesA_Int
        let expectedSizeB = rowsB_desc * rowBytesB_Int
        let expectedSizeC = rowsC_desc * rowBytesC_Int

        guard inputA.length >= expectedSizeA else {
            print(
                "Error [MPS MatMul]: Buffer A too small. Needs \(expectedSizeA) (rows=\(rowsA_desc), rowBytes=\(rowBytesA_Int)), has \(inputA.length)."
            )
            return false
        }
        guard inputB.length >= expectedSizeB else {
            print(
                "Error [MPS MatMul]: Buffer B too small. Needs \(expectedSizeB) (rows=\(rowsB_desc), rowBytes=\(rowBytesB_Int)), has \(inputB.length)."
            )
            return false
        }
        guard outputC.length >= expectedSizeC else {
            print(
                "Error [MPS MatMul]: Buffer C too small. Needs \(expectedSizeC) (rows=\(rowsC_desc), rowBytes=\(rowBytesC_Int)), has \(outputC.length)."
            )
            return false
        }
        // --- End Buffer Size Checks ---

        // --- Create MPSMatrix Descriptors ---
        let descA = MPSMatrixDescriptor(
            rows: M, columns: K, rowBytes: rowBytesA_Int, dataType: .float16)
        let descB = MPSMatrixDescriptor(
            rows: K, columns: N, rowBytes: rowBytesB_Int, dataType: .float16)  // KxN descriptor
        let descC = MPSMatrixDescriptor(
            rows: M, columns: N, rowBytes: rowBytesC_Int, dataType: .float16)

        // --- Create MPSMatrix Objects ---
        let matrixA = MPSMatrix(buffer: inputA, descriptor: descA)
        let matrixB = MPSMatrix(buffer: inputB, descriptor: descB)
        let matrixC = MPSMatrix(buffer: outputC, descriptor: descC)

        // --- Create MPSMatrixMultiplication Kernel ---
        // Kernel dimensions are based on the logical operation C[M,N] = op(A)[M,K] * op(B)[K,N]
        let matMulKernel = MPSMatrixMultiplication(
            device: self.device,
            transposeLeft: transposeA,
            transposeRight: transposeB,
            resultRows: M,  // M rows in result C
            resultColumns: N,  // N columns in result C
            interiorColumns: K,  // K is the shared dimension
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
            "Successfully encoded \(matMulKernel.label ?? "MPS MatMul") M=\(M), N=\(N), K=\(K), tA=\(transposeA), tB=\(transposeB)"
        )
        return true
    }
}
