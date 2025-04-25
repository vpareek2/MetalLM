import Foundation
import Metal

// Service class to manage Metal device, command queue, and compute pipelines
class MetalService {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    // Pipeline states for different kernels
    let dequantizeQ4KMF32PipelineState: MTLComputePipelineState
    let dequantizeQ4KMF16PipelineState: MTLComputePipelineState
    let dequantizeQ4KSF32PipelineState: MTLComputePipelineState  // Keep this from Q4KS step

    // *** NEW: Pipeline state for F16 -> F32 conversion ***
    let convertF16toF32PipelineState: MTLComputePipelineState

    // Singleton pattern for easy access (optional, but convenient)
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

        // Load the default Metal library (contains kernels from .metal files)
        guard let library = device.makeDefaultLibrary() else {
            print("Error: Could not load default Metal library. Ensure .metal files are in target.")
            return nil
        }

        // Create pipeline state for Q4_K -> Float32 kernel
        guard let q4kmF32Function = library.makeFunction(name: "kernel_dequantize_q4_K_f32") else {
            print("Error: Could not find kernel function 'kernel_dequantize_q4_K_f32'.")
            return nil
        }
        do {
            self.dequantizeQ4KMF32PipelineState = try device.makeComputePipelineState(
                function: q4kmF32Function)
        } catch {
            print("Error: Could not create pipeline state for kernel_dequantize_q4_K_f32: \(error)")
            return nil
        }

        // Create pipeline state for Q4_K -> Float16 kernel
        guard let q4kmF16Function = library.makeFunction(name: "kernel_dequantize_q4_K_f16") else {
            print("Error: Could not find kernel function 'kernel_dequantize_q4_K_f16'.")
            return nil
        }
        do {
            self.dequantizeQ4KMF16PipelineState = try device.makeComputePipelineState(
                function: q4kmF16Function)
        } catch {
            print("Error: Could not create pipeline state for kernel_dequantize_q4_K_f16: \(error)")
            return nil
        }

        // Create pipeline state for Q4_K_S -> Float32 kernel
        guard let q4ksF32Function = library.makeFunction(name: "kernel_dequantize_q4_K_S_f32")
        else {
            print("Error: Could not find kernel function 'kernel_dequantize_q4_K_S_f32'.")
            return nil
        }
        do {
            self.dequantizeQ4KSF32PipelineState = try device.makeComputePipelineState(
                function: q4ksF32Function)
        } catch {
            print(
                "Error: Could not create pipeline state for kernel_dequantize_q4_K_S_f32: \(error)")
            return nil
        }

        // *** NEW: Create pipeline state for F16 -> F32 kernel ***
        guard let convertF16F32Function = library.makeFunction(name: "kernel_convert_f16_f32")
        else {
            print("Error: Could not find kernel function 'kernel_convert_f16_f32'.")
            return nil
        }
        do {
            self.convertF16toF32PipelineState = try device.makeComputePipelineState(
                function: convertF16F32Function)
            print("Successfully created pipeline state for F16->F32 conversion.")
        } catch {
            print("Error: Could not create pipeline state for kernel_convert_f16_f32: \(error)")
            return nil
        }

        print("MetalService initialized successfully for device: \(device.name)")
    }

    // MARK: - Dequantization Functions

    /// Dequantizes a Q4_K_M buffer into a new Float32 buffer using Metal.
    /// - Parameters:
    ///   - quantizedBuffer: The MTLBuffer containing the Q4_K_M quantized data.
    ///   - elementCount: The total number of elements represented by the quantized data.
    /// - Returns: A new MTLBuffer containing the dequantized Float32 data, or nil on failure.
    func dequantizeQ4KM_to_f32(quantizedBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer? {
        guard elementCount > 0 else { return nil }

        // Calculate output buffer size
        let outputBufferSize = elementCount * MemoryLayout<Float>.size
        guard
            let outputBuffer = device.makeBuffer(
                length: outputBufferSize, options: .storageModeShared)
        else {
            print("Error: Failed to create output buffer for f32 dequantization.")
            return nil
        }

        // Create buffer for element count argument
        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg,
                length: MemoryLayout<UInt64>.size,
                options: .storageModeShared)
        else {
            print("Error: Failed to create nelements buffer.")
            return nil
        }

        // Encode and dispatch kernel
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Error: Failed to create command buffer or encoder.")
            return nil
        }

        encoder.setComputePipelineState(dequantizeQ4KMF32PipelineState)
        encoder.setBuffer(quantizedBuffer, offset: 0, index: 0)  // src
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)  // dst
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 2)  // nelements

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(
            dequantizeQ4KMF32PipelineState.maxTotalThreadsPerThreadgroup, 256)
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        // For simplicity now, wait synchronously. Use completion handlers for async later.
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            print("Error during Q4_K -> f32 dequantization kernel execution: \(error)")
            return nil
        }

        return outputBuffer
    }

    /// Dequantizes a Q4_K_M buffer into a new Float16 buffer using Metal.
    /// - Parameters:
    ///   - quantizedBuffer: The MTLBuffer containing the Q4_K_M quantized data.
    ///   - elementCount: The total number of elements represented by the quantized data.
    /// - Returns: A new MTLBuffer containing the dequantized Float16 data, or nil on failure.
    func dequantizeQ4KM_to_f16(quantizedBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer? {
        guard elementCount > 0 else { return nil }

        // Calculate output buffer size
        let outputBufferSize = elementCount * MemoryLayout<Float16>.size  // Use Float16 size
        guard
            let outputBuffer = device.makeBuffer(
                length: outputBufferSize, options: .storageModeShared)
        else {
            print("Error: Failed to create output buffer for f16 dequantization.")
            return nil
        }

        // Create buffer for element count argument
        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg,
                length: MemoryLayout<UInt64>.size,
                options: .storageModeShared)
        else {
            print("Error: Failed to create nelements buffer.")
            return nil
        }

        // Encode and dispatch kernel
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Error: Failed to create command buffer or encoder.")
            return nil
        }

        // Use the F16 pipeline state
        encoder.setComputePipelineState(dequantizeQ4KMF16PipelineState)
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
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            print("Error during Q4_K -> f16 dequantization kernel execution: \(error)")
            return nil
        }

        return outputBuffer
    }

    // *** NEW: F16 -> F32 Conversion Function ***
    func convertF16toF32(inputBuffer: MTLBuffer, elementCount: Int) -> MTLBuffer? {
        guard elementCount > 0 else {
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
        else { return nil }
        outputBuffer.label = (inputBuffer.label ?? "unknown") + "_f16_to_f32"

        var nelementsArg = UInt64(elementCount)
        guard
            let nelementsBuffer = device.makeBuffer(
                bytes: &nelementsArg, length: MemoryLayout<UInt64>.size, options: .storageModeShared
            )
        else { return nil }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }
        encoder.label = "F16 -> F32 Encoder"

        encoder.setComputePipelineState(convertF16toF32PipelineState)  // Use F16->F32 pipeline
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)  // src (f16)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)  // dst (f32)
        encoder.setBuffer(nelementsBuffer, offset: 0, index: 2)  // elementCount

        let gridSize = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadGroupWidth = min(convertF16toF32PipelineState.maxTotalThreadsPerThreadgroup, 1024)
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            print("Error during F16->F32 conversion kernel execution: \(error)")
            return nil
        }
        print("Successfully executed F16->F32 conversion kernel.")
        return outputBuffer
    }
}
