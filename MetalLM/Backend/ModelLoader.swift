// Class responsible for loading a GGUF model and managing Metal interactions for tensors
// TODO: Make this class thread-safe if loadLlamaModel is called concurrently,
//       specifically protect access to the caches (quantizedBufferCache, dequantizedBufferCache).
//       Consider using an actor or DispatchQueue/NSLock for cache access.
import Accelerate
import Foundation
import Metal  // Required for MTLBuffer

// Error types specific to ModelLoader
enum ModelLoaderError: Error {
    case metalServiceUnavailable
    case modelNotLoaded
    case tensorNotFound(String)
    case failedToGetTensorData(String)
    case failedToCreateMetalBuffer(String)
    case dequantizationFailed(String, Error?)
    case unsupportedTensorType(String, GGUFDataType)
    case configCreationFailed(Error)
    case tensorNameCreationFailed(layer: Int, type: String)
}

// ... (Keep Error enum) ...

class ModelLoader {

    // Instance of the Metal service
    private let metalService: MetalService

    // Holds the parsed GGUF file data
    private(set) var ggufFile: GGUFFile?

    // --- Thread-Safe Cache Access ---
    // Use a serial queue to synchronize access to the dictionaries below
    private let cacheQueue = DispatchQueue(label: "com.MetalLM.ModelLoader.cacheQueue")

    // Caches - Access MUST be synchronized via cacheQueue
    private var _quantizedBufferCache: [String: MTLBuffer] = [:]
    private var _dequantizedBufferCache: [Int: MTLBuffer] = [:]
    // --- End Thread-Safe Cache Access ---

    /// Initializes the ModelLoader.
    init(metalService: MetalService) {
        self.metalService = metalService
    }

    /// Loads only the GGUF file structure and metadata.
    func loadMetadata(url: URL) throws {
        print("Attempting to load GGUF metadata from: \(url.path)")
        // Clear caches synchronously when loading new metadata
        self.clearCaches()
        self.ggufFile = try GGUFFile(url: url)
        print("Metadata for '\(url.lastPathComponent)' loaded successfully.")
    }

    /// Unloads the current model and clears caches.
    func unloadModel() {
        self.ggufFile = nil
        self.clearCaches()  // Use the synchronized clear
        print("Model unloaded and caches cleared.")
    }

    /// Clears the internal buffer caches (Thread-Safe).
    func clearCaches() {
        cacheQueue.sync {  // Use sync wait for clearing
            _quantizedBufferCache.removeAll()
            _dequantizedBufferCache.removeAll()
        }
        print("Buffer caches cleared.")
    }

    /// Retrieves a specific tensor descriptor by name.
    func getTensorDescriptor(name tensorName: String) -> GGUFTensorDescriptor? {
        guard let file = ggufFile else { return nil }
        // Reading ggufFile.tensors should be safe if ggufFile is populated once
        return file.tensors.first { $0.name == tensorName }
    }

    // MARK: - Tensor Loading and Processing -

    /// Gets the raw (potentially quantized) tensor data as an MTLBuffer.
    /// Handles F64 -> F32 CPU conversion. Caches the result. (Thread-Safe Cache Access)
    private func getQuantizedTensorBuffer(tensorName: String) throws -> MTLBuffer {

        // --- Cache Read (Synchronized) ---
        if let cachedBuffer = cacheQueue.sync(execute: { _quantizedBufferCache[tensorName] }) {
            return cachedBuffer
        }
        // --- End Cache Read ---

        // Get Descriptor and Data (Outside sync block)
        guard let ggufFile = self.ggufFile else { throw ModelLoaderError.modelNotLoaded }
        guard let tensor = getTensorDescriptor(name: tensorName) else {
            throw ModelLoaderError.tensorNotFound(tensorName)
        }
        let rawData = ggufFile.getTensorData(for: tensor)

        // Validation
        guard !rawData.isEmpty || tensor.elementCount == 0 else {
            print(
                "Error: Tensor '\(tensorName)' has \(tensor.elementCount) elements but getTensorData returned empty data. Expected byteSize: \(tensor.byteSize)."
            )
            throw ModelLoaderError.failedToGetTensorData(
                tensorName + " (empty data for non-empty tensor)")
        }

        var bufferData: Data
        var bufferLabel = tensorName
        let options: MTLResourceOptions = .storageModeShared

        // F64 -> F32 Conversion (Outside sync block)
        if tensor.type == .f64 {
            // ... (F64 conversion code remains the same) ...
            print("--- CPU Conversion: Converting F64 tensor '\(tensorName)' to F32 ---")
            let elementCount = Int(tensor.elementCount)
            if elementCount > 0 {
                // ... F64 conversion logic ...
                let expectedF64Size = elementCount * MemoryLayout<Double>.size
                let expectedF32Size = elementCount * MemoryLayout<Float>.size
                guard rawData.count == expectedF64Size else {
                    throw ModelLoaderError.failedToGetTensorData(
                        "Size mismatch for F64 tensor \(tensorName)")
                }
                var floatData = Data(count: expectedF32Size)
                let conversionSuccess = floatData.withUnsafeMutableBytes { mutableF32RawBufferPtr in
                    rawData.withUnsafeBytes { rawF64BufferPtr in
                        guard let f64BaseAddress = rawF64BufferPtr.baseAddress,
                            let f32BaseAddress = mutableF32RawBufferPtr.baseAddress
                        else { return false }
                        let sourceBufferPtr = UnsafeBufferPointer(
                            start: f64BaseAddress.assumingMemoryBound(to: Double.self),
                            count: elementCount)
                        var destinationBufferPtr = UnsafeMutableBufferPointer(
                            start: f32BaseAddress.assumingMemoryBound(to: Float.self),
                            count: elementCount)
                        #if canImport(Accelerate)
                            vDSP.convertElements(of: sourceBufferPtr, to: &destinationBufferPtr)
                        #else
                            for i in 0..<elementCount {
                                destinationBufferPtr[i] = Float(sourceBufferPtr[i])
                            }
                        #endif
                        return true
                    }
                }
                guard conversionSuccess else {
                    throw ModelLoaderError.dequantizationFailed(tensorName, nil)
                }
                bufferData = floatData
                bufferLabel = "\(tensorName)_f64_to_f32"
                print(
                    "--- CPU Conversion: Finished converting \(elementCount) F64 elements for \(tensorName) ---"
                )

            } else {
                bufferData = Data()
                bufferLabel = "\(tensorName)_f64_to_f32_empty"
                print("--- CPU Conversion: F64 tensor '\(tensorName)' has 0 elements. ---")
            }
        } else {
            if tensor.elementCount == 0 && !rawData.isEmpty {
                print(
                    "Warning: Tensor '\(tensorName)' has 0 elements but rawData is not empty (\(rawData.count) bytes). Using empty data."
                )
                bufferData = Data()
            } else {
                bufferData = rawData
            }
        }

        // Create Metal Buffer (Outside sync block)
        let bufferLength = bufferData.count
        guard
            let buffer = metalService.device.makeBuffer(
                length: max(bufferLength, 1), options: options)
        else {
            throw ModelLoaderError.failedToCreateMetalBuffer(tensorName)
        }
        if bufferLength > 0 {
            try bufferData.withUnsafeBytes { rawBufferPointer in
                if let baseAddress = rawBufferPointer.baseAddress {
                    buffer.contents().copyMemory(from: baseAddress, byteCount: bufferLength)
                } else {
                    throw ModelLoaderError.failedToCreateMetalBuffer(
                        tensorName + " (data pointer issue)")
                }
            }
        }
        buffer.label = bufferLabel

        // --- Cache Write (Synchronized) ---
        cacheQueue.sync {
            _quantizedBufferCache[tensorName] = buffer
        }
        // --- End Cache Write ---

        return buffer
    }

    /// Dequantizes or converts a tensor to the desired output type using Metal kernels. (Thread-Safe Cache Access)
    private func dequantizeTensor(tensorName: String, outputType: GGUFDataType) throws -> MTLBuffer
    {
        guard let tensor = getTensorDescriptor(name: tensorName) else {
            throw ModelLoaderError.tensorNotFound(tensorName)
        }

        var hasher = Hasher()
        hasher.combine(tensorName)
        hasher.combine(outputType)
        let cacheKey = hasher.finalize()

        // --- Cache Read (Synchronized) ---
        if let cachedBuffer = cacheQueue.sync(execute: { _dequantizedBufferCache[cacheKey] }) {
            return cachedBuffer
        }
        // --- End Cache Read ---

        // Get Source Buffer (Calls getQuantizedTensorBuffer which handles its own cache sync)
        let sourceBuffer = try getQuantizedTensorBuffer(tensorName: tensorName)
        let elementCount = Int(tensor.elementCount)
        let originalType = tensor.type

        // --- Handle Trivial Cases ---
        if originalType == .f64 {
            if outputType == .f32 {
                // Cache Write (Synchronized)
                cacheQueue.sync { _dequantizedBufferCache[cacheKey] = sourceBuffer }
                return sourceBuffer
            } else {
                throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)
            }
        }
        if originalType == outputType {
            // Cache Write (Synchronized)
            cacheQueue.sync { _dequantizedBufferCache[cacheKey] = sourceBuffer }
            return sourceBuffer
        }
        guard elementCount > 0 else {
            print(
                "Warning: Tensor '\(tensorName)' has zero elements. Creating empty buffer for type \(outputType)."
            )
            guard
                let emptyBuffer = metalService.device.makeBuffer(
                    length: 1, options: .storageModeShared)
            else {
                throw ModelLoaderError.failedToCreateMetalBuffer("empty buffer for \(tensorName)")
            }
            emptyBuffer.label = "\(tensorName)_processed_\(outputType)_empty"
            // Cache Write (Synchronized)
            cacheQueue.sync { _dequantizedBufferCache[cacheKey] = emptyBuffer }
            return emptyBuffer
        }

        // --- Perform Dequantization / Conversion via MetalService ---
        var processedBuffer: MTLBuffer?
        print("Processing tensor '\(tensorName)' from \(originalType) to \(outputType)...")

        switch (originalType, outputType) {
        // Q4_K Dequantization (Type 12)
        case (.q4_K, .f32):
            processedBuffer = metalService.dequantizeQ4KM_to_f32(
                quantizedBuffer: sourceBuffer, elementCount: elementCount)  // Keep using combined kernel func
        case (.q4_K, .f16):
            processedBuffer = metalService.dequantizeQ4KM_to_f16(
                quantizedBuffer: sourceBuffer, elementCount: elementCount)  // Keep using combined kernel func

        // Q6_K Dequantization (Type 14)
        case (.q6_K, .f32):
            processedBuffer = metalService.dequantizeQ6K_to_f32(
                quantizedBuffer: sourceBuffer, elementCount: elementCount)  // Call new func
        case (.q6_K, .f16):
            processedBuffer = metalService.dequantizeQ6K_to_f16(
                quantizedBuffer: sourceBuffer, elementCount: elementCount)  // Call new func

        // F16 Conversion (Type 1)
        case (.f16, .f32):
            processedBuffer = metalService.convertF16toF32(
                inputBuffer: sourceBuffer, elementCount: elementCount)

        // F32 Conversion (Type 0)
        case (.f32, .f16):
            print(
                "Error: F32 to F16 quantization kernel not implemented yet for tensor '\(tensorName)'."
            )
            throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)

        // F64 Handling (Type 28) - Should have been handled by trivial case check earlier
        // but adding case here for completeness, though it should ideally not be reached.
        case (.f64, .f32):
            print(
                "Internal Warning: F64 case reached in switch; should have been handled earlier for tensor \(tensorName)."
            )
            processedBuffer = sourceBuffer  // It's already F32
        case (.f64, .f16):  // Should definitely not be reached
            print(
                "Internal Error: Attempting F64 to F16 conversion in switch for tensor \(tensorName)."
            )
            throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)

        // Add other cases here (Q2_K, Q3_K, Q5_K, Q8_K, IQ types) when kernels are available

        default:
            print(
                "Error: Unsupported original type or conversion requested for tensor '\(tensorName)': from \(originalType) to \(outputType)."
            )
            throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)
        }

        // --- Validation and Caching ---
        guard let finalBuffer = processedBuffer else {
            throw ModelLoaderError.dequantizationFailed(tensorName, nil)
        }

        let expectedSize: Int
        switch outputType {
        case .f16: expectedSize = elementCount * MemoryLayout<Float16>.size
        case .f32: expectedSize = elementCount * MemoryLayout<Float>.size
        default: expectedSize = -1
        }
        if expectedSize > 0 && finalBuffer.length < expectedSize {
            print(
                "Error: Processed buffer for '\(tensorName)' (\(outputType)) has incorrect size. Expected >= \(expectedSize), Got \(finalBuffer.length)."
            )
            throw ModelLoaderError.dequantizationFailed(tensorName, nil)
        }

        finalBuffer.label = "\(tensorName)_processed_\(outputType)"

        // --- Cache Write (Synchronized) ---
        cacheQueue.sync {
            _dequantizedBufferCache[cacheKey] = finalBuffer
        }
        // --- End Cache Write ---

        print("Successfully processed and cached: \(tensorName) -> \(outputType)")
        return finalBuffer
    }

    // MARK: - Full Model Loading -

    // In ModelLoader.swift

    // MARK: - Full Model Loading -

    func loadLlamaModel(
        url: URL,
        computePrecision: GGUFDataType = .f16,
        normWeightType: GGUFDataType = .f32,
        embeddingType: GGUFDataType = .f32  // Keep this as F32
    ) async throws -> LlamaModel {

        try loadMetadata(url: url)
        guard let file = ggufFile else { throw ModelLoaderError.modelNotLoaded }
        let config = try LlamaConfig(metadata: file.metadata)

        print("--- Verifying Tensor Names (Available in GGUF) ---")
        // Create a lookup dictionary for faster access to original types
        let tensorTypeLookup: [String: GGUFDataType] = Dictionary(
            uniqueKeysWithValues: file.tensors.map { ($0.name, $0.type) })
        file.tensors.forEach {
            print("  - \($0.name) (Type: \($0.type), Elements: \($0.elementCount))")
        }
        print("-------------------------------------------------")

        // --- MODIFIED getBuffer Helper ---
        // Now checks original type before deciding final target type
        let getBuffer: @Sendable (String, GGUFDataType) async throws -> MTLBuffer = {
            name, requestedTypeIfNonF64 in
            print(
                "Requesting tensor: \(name) (Preferred type if not F64: \(requestedTypeIfNonF64))")

            // Find the original type from the GGUF file info
            guard let originalType = tensorTypeLookup[name] else {
                print("Error: Tensor '\(name)' not found in GGUF tensor list during lookup.")
                throw ModelLoaderError.tensorNotFound(name + " (lookup failed)")
            }
            print("  Original type is: \(originalType)")

            // Determine the actual target type for dequantization/processing
            let targetType: GGUFDataType
            if originalType == .f64 {
                targetType = .f32  // If original is F64, ALWAYS target F32 (due to CPU conversion)
                print("  Original type is F64, forcing target type to F32.")
            } else {
                targetType = requestedTypeIfNonF64  // Otherwise, use the requested type
            }

            // Use Task to wrap the synchronous call
            return try await Task { [self] in  // Explicitly capture self
                try self.dequantizeTensor(tensorName: name, outputType: targetType)
            }.value
        }
        // --- END MODIFIED getBuffer Helper ---

        let tensorName: @Sendable (String, Int?) throws -> String = { pattern, index in
            if pattern.contains("%d") {
                guard let index = index else {
                    throw ModelLoaderError.tensorNameCreationFailed(layer: -1, type: pattern)
                }
                return String(format: pattern, index)
            } else {
                return pattern
            }
        }

        print("Loading non-block tensors...")
        // Pass the desired precision (embeddingType, normWeightType) as the second arg to getBuffer
        let tokenEmbeddingsBuffer = try await getBuffer(
            try tensorName("token_embd.weight", nil), embeddingType)
        let finalNormWeightBuffer = try await getBuffer(
            try tensorName("output_norm.weight", nil), normWeightType)
        let outputWeightBuffer = try await getBuffer(
            try tensorName("output.weight", nil), embeddingType)  // Use embeddingType for output? Or computePrecision? Check Llama arch. Often shares type with embeddings.

        var ropeFreqsBuffer: MTLBuffer? = nil
        let ropeFreqsTensorName = "rope_freqs.weight"  // Standard name
        if let ropeTensorDesc = self.getTensorDescriptor(name: ropeFreqsTensorName) {
            print("Found optional RoPE frequencies tensor: \(ropeFreqsTensorName)")
            // Load it as F32, as the kernel expects float factors
            ropeFreqsBuffer = try await getBuffer(ropeFreqsTensorName, .f32)
        } else {
            print("Optional RoPE frequencies tensor '\(ropeFreqsTensorName)' not found.")
        }

        print("Non-block tensors loaded.")

        print("Loading \(config.numLayers) transformer blocks...")
        var blocks: [LlamaTransformerBlock] = []
        blocks.reserveCapacity(config.numLayers)

        for i in 0..<config.numLayers {
            print("  Loading Block \(i)...")
            // Pass the desired precision (computePrecision, normWeightType) as the second arg to getBuffer
            async let attnNormWeight = getBuffer(
                try tensorName("blk.%d.attn_norm.weight", i), normWeightType)
            async let ffnNormWeight = getBuffer(
                try tensorName("blk.%d.ffn_norm.weight", i), normWeightType)
            async let qWeight = getBuffer(
                try tensorName("blk.%d.attn_q.weight", i), computePrecision)
            async let kWeight = getBuffer(
                try tensorName("blk.%d.attn_k.weight", i), computePrecision)
            async let vWeight = getBuffer(
                try tensorName("blk.%d.attn_v.weight", i), computePrecision)
            async let oWeight = getBuffer(
                try tensorName("blk.%d.attn_output.weight", i), computePrecision)
            async let gateWeight = getBuffer(
                try tensorName("blk.%d.ffn_gate.weight", i), computePrecision)
            async let upWeight = getBuffer(
                try tensorName("blk.%d.ffn_up.weight", i), computePrecision)
            async let downWeight = getBuffer(
                try tensorName("blk.%d.ffn_down.weight", i), computePrecision)

            // Await results and assemble the block (remains the same)
            let attention = try await LlamaAttention(
                qWeight: qWeight, kWeight: kWeight, vWeight: vWeight, oWeight: oWeight)
            let mlp = try await LlamaMLP(
                gateWeight: gateWeight, upWeight: upWeight, downWeight: downWeight)
            let block = try await LlamaTransformerBlock(
                attentionNormWeight: attnNormWeight,
                ffnNormWeight: ffnNormWeight,
                attention: attention,
                mlp: mlp
            )
            blocks.append(block)
            print("  Block \(i) loaded.")
        }
        print("All transformer blocks loaded.")

        print("Assembling final LlamaModel...")
        let llamaModel = LlamaModel(
            config: config,
            tokenEmbeddings: tokenEmbeddingsBuffer,
            blocks: blocks,
            finalNormWeight: finalNormWeightBuffer,
            outputWeight: outputWeightBuffer,
            ropeFrequencies: ropeFreqsBuffer  // Pass the optional buffer
        )
        print("LlamaModel assembly complete.")
        return llamaModel
    }

}  // End of ModelLoader class
