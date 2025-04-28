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
    func dequantizeQ4KM_to_f32(quantizedBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer?
    { /* ... as before ... */  }
    func dequantizeQ4KM_to_f16(quantizedBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer?
    { /* ... as before ... */  }

    // MARK: - Conversion Functions
    // (Keep convertF16toF32)
    func convertF16toF32(inputBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer?
    { /* ... as before ... */  }

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
