import Foundation
import Metal

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

    // Helper function from rmsNormF16 (keep private)
    private func flsl(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        return Int.bitWidth - n.leadingZeroBitCount
    }
}
