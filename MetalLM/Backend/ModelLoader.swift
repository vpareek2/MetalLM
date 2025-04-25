import Foundation
import Metal // Required for MTLBuffer
import Accelerate

// Error types specific to ModelLoader
enum ModelLoaderError: Error {
    case metalServiceUnavailable
    case modelNotLoaded
    case tensorNotFound(String)
    case failedToGetTensorData(String)
    case failedToCreateMetalBuffer(String)
    case dequantizationFailed(String, Error?)
    case unsupportedTensorType(String, GGUFDataType)
}

// Class responsible for loading a GGUF model and managing Metal interactions for tensors
class ModelLoader {

    // Instance of the Metal service (initialized externally and passed in)
    private let metalService: MetalService

    // Holds the parsed GGUF file data (nil until a model is loaded)
    private(set) var ggufFile: GGUFFile?

    // Cache for already created quantized tensor buffers (optional optimization)
    // Key: Tensor Name, Value: MTLBuffer
    private var quantizedBufferCache: [String: MTLBuffer] = [:]

    // Cache for already dequantized tensor buffers (optional optimization)
    // Key: Tensor Name + Output Type Hash, Value: MTLBuffer
    private var dequantizedBufferCache: [Int: MTLBuffer] = [:]

    /// Initializes the ModelLoader with a required MetalService instance.
    init(metalService: MetalService) {
        self.metalService = metalService
    }

    /// Loads a GGUF model from the specified URL.
    /// Parses the file structure using GGUFFile.
    /// Clears any previously cached buffers.
    /// - Parameter url: The file URL of the GGUF model.
    /// - Throws: Errors from GGUFFile initialization (e.g., file not found, parsing errors).
    func loadModel(url: URL) throws {
        print("Attempting to load GGUF model from: \(url.path)")
        // Clear caches when loading a new model
        clearCaches()

        // Attempt to initialize GGUFFile
        self.ggufFile = try GGUFFile(url: url)

        print("Model '\(url.lastPathComponent)' loaded successfully.")
    }

    /// Unloads the current model and clears caches.
    func unloadModel() {
        self.ggufFile = nil
        clearCaches()
        print("Model unloaded.")
    }

    /// Clears the internal buffer caches.
    func clearCaches() {
        quantizedBufferCache.removeAll()
        dequantizedBufferCache.removeAll()
        print("Buffer caches cleared.")
    }

    /// Retrieves a specific tensor descriptor by name.
    /// - Parameter tensorName: The name of the tensor.
    /// - Returns: The GGUFTensorDescriptor or nil if not found.
    func getTensorDescriptor(name tensorName: String) -> GGUFTensorDescriptor? {
        guard ggufFile != nil else { return nil }
        return ggufFile!.tensors.first { $0.name == tensorName }
    }

    /// Gets the raw quantized tensor data as an MTLBuffer.
    /// Creates the buffer from the GGUF file's data slice.
    /// Uses a cache to avoid recreating buffers.
    func getQuantizedTensorBuffer(tensorName: String) throws -> MTLBuffer {
        if let cachedBuffer = quantizedBufferCache[tensorName] {
             print("Cache hit for processed tensor buffer: \(tensorName)")
             return cachedBuffer
        }
        print("Cache miss for processed tensor buffer: \(tensorName)")

        // Keep guard let, as ggufFile is used below
        guard let ggufFile = self.ggufFile else {
            throw ModelLoaderError.modelNotLoaded
        }
        guard let tensor = getTensorDescriptor(name: tensorName) else {
            throw ModelLoaderError.tensorNotFound(tensorName)
        }

        // Get raw data from the GGUF file
        let rawData = ggufFile.getTensorData(for: tensor) // Uses ggufFile

        guard !rawData.isEmpty || tensor.elementCount == 0 else {
             throw ModelLoaderError.failedToGetTensorData(tensorName)
        }

        var bufferData: Data
        var bufferLabel = tensorName
        let options: MTLResourceOptions = .storageModeShared

        if tensor.type == .f64 {
            print("--- CPU Conversion: Converting F64 tensor '\(tensorName)' to F32 ---")
            let elementCount = Int(tensor.elementCount)
            let expectedF64Size = elementCount * MemoryLayout<Double>.size
            let expectedF32Size = elementCount * MemoryLayout<Float>.size

            guard rawData.count == expectedF64Size else {
                print("Error: F64 raw data size mismatch for \(tensorName). Expected \(expectedF64Size), got \(rawData.count)")
                throw ModelLoaderError.failedToGetTensorData("Size mismatch for \(tensorName)")
            }

            // --- DIRECT CONVERSION TO DATA ---
            // 1. Allocate target Data buffer for F32
            var floatData = Data(count: expectedF32Size) // Allocate directly

            // 2. Perform conversion directly into the mutable bytes of floatData
            let conversionSuccess = floatData.withUnsafeMutableBytes { mutableF32RawBufferPtr in // Get raw mutable buffer pointer
                rawData.withUnsafeBytes { rawF64BufferPtr in // Get raw immutable buffer pointer
                    // Get base addresses safely
                    guard let f64BaseAddress = rawF64BufferPtr.baseAddress,
                          let f32BaseAddress = mutableF32RawBufferPtr.baseAddress else {
                        print("Error: Could not get base addresses for conversion.")
                        return false // Indicate failure
                    }

                    // Create the necessary buffer pointers explicitly
                    let sourceBufferPtr = UnsafeBufferPointer(start: f64BaseAddress.assumingMemoryBound(to: Double.self),
                                                              count: elementCount)
                    // Create a MUTABLE buffer pointer for the destination
                    var destinationBufferPtr = UnsafeMutableBufferPointer(start: f32BaseAddress.assumingMemoryBound(to: Float.self),
                                                                          count: elementCount)

                    #if canImport(Accelerate)
                        print("--- CPU Conversion: Using Accelerate/vDSP directly into Data buffer ---")
                        // Pass the immutable source and the mutable destination
                        vDSP.convertElements(of: sourceBufferPtr, to: &destinationBufferPtr) // Pass destination as inout
                    #else
                        print("--- CPU Conversion: Using manual loop directly into Data buffer ---")
                        // Use the buffer pointers for the loop
                        for i in 0..<elementCount {
                            destinationBufferPtr[i] = Float(sourceBufferPtr[i])
                            // Add NaN/Inf check here if needed
                        }
                    #endif
                    return true // Indicate success
                }
            }

            guard conversionSuccess else {
                // Handle the error if base addresses were nil
                 throw ModelLoaderError.dequantizationFailed(tensorName, nil) // Or a more specific error
            }
            // --- END DIRECT CONVERSION ---

            bufferData = floatData // Use the directly converted data
            bufferLabel = "\(tensorName)_f64_to_f32"
            print("--- CPU Conversion: Finished converting \(elementCount) elements for \(tensorName) ---")

        } else {
            bufferData = rawData
        }

        let bufferLength = bufferData.count
        guard let buffer = metalService.device.makeBuffer(length: bufferLength, options: options) else {
             throw ModelLoaderError.failedToCreateMetalBuffer(tensorName)
        }
        if bufferLength > 0 {
            bufferData.withUnsafeBytes { rawBufferPointer in
                 buffer.contents().copyMemory(from: rawBufferPointer.baseAddress!, byteCount: bufferLength)
            }
        }

        buffer.label = bufferLabel
        quantizedBufferCache[tensorName] = buffer
        print("--- Debug [getQuantizedTensorBuffer]: Returning buffer for \(tensorName). Label: \(buffer.label ?? "nil"), Length: \(buffer.length), Type Originally: \(tensor.type)")
        return buffer
    }

    /// Dequantizes a specific tensor to the desired output type using Metal kernels.
    /// Handles caching of dequantized results.
    func dequantizeTensor(tensorName: String, outputType: GGUFDataType = .f32) throws -> MTLBuffer {
        guard self.ggufFile != nil else { // Just check if it exists
            throw ModelLoaderError.modelNotLoaded
        }
        guard let tensor = getTensorDescriptor(name: tensorName) else {
            throw ModelLoaderError.tensorNotFound(tensorName)
        }

        var hasher = Hasher()
        hasher.combine(tensorName)
        hasher.combine(outputType)
        let cacheKey = hasher.finalize()

        if let cachedBuffer = dequantizedBufferCache[cacheKey] {
             print("Cache hit for dequantized tensor: \(tensorName) -> \(outputType)")
             return cachedBuffer
        }
        print("Cache miss for dequantized tensor: \(tensorName) -> \(outputType)")

        // Get the buffer (which might already be F32 if original was F64)
        let sourceBuffer = try getQuantizedTensorBuffer(tensorName: tensorName)
        let elementCount = Int(tensor.elementCount)

        // If the tensor was originally F64, sourceBuffer is already F32
        if tensor.type == .f64 {
            if outputType == .f32 {
                print("Info: Returning pre-converted F32 buffer for originally F64 tensor '\(tensorName)'.")
                return sourceBuffer
            } else {
                print("Error: Cannot produce \(outputType) output from originally F64 tensor '\(tensorName)' (only F32 supported via CPU conversion).")
                throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)
            }
        }

        guard elementCount > 0 else {
             print("Warning: Tensor '\(tensorName)' has zero elements.")
             guard let emptyBuffer = metalService.device.makeBuffer(length: 0, options: .storageModeShared) else {
                  throw ModelLoaderError.failedToCreateMetalBuffer("empty buffer for \(tensorName)")
             }
             emptyBuffer.label = "\(tensorName)_dequantized_\(outputType)_empty"
             dequantizedBufferCache[cacheKey] = emptyBuffer
             return emptyBuffer
        }

        var dequantizedBuffer: MTLBuffer?

        // --- Switch now only handles types needing GPU dequant/conversion ---
        switch tensor.type {
        // *** Combined Q4_K_S and Q4_K_M Case ***
        case .q4_K_S, .q4_K_M: // Handles both types 14 and 15
            print("Dispatching Q4_K dequantization for \(tensorName) (Type: \(tensor.type.rawValue)) -> \(outputType)...")
            // Use the existing Q4_K_M MetalService functions
            if outputType == .f32 {
                 dequantizedBuffer = metalService.dequantizeQ4KM_to_f32(quantizedBuffer: sourceBuffer, elementCount: elementCount)
                 if dequantizedBuffer == nil { throw ModelLoaderError.dequantizationFailed(tensorName, nil) }
            } else if outputType == .f16 {
                 dequantizedBuffer = metalService.dequantizeQ4KM_to_f16(quantizedBuffer: sourceBuffer, elementCount: elementCount)
                 if dequantizedBuffer == nil { throw ModelLoaderError.dequantizationFailed(tensorName, nil) }
            } else {
                 print("Error: Cannot dequantize Q4_K tensor '\(tensorName)' to \(outputType).")
                 throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)
            }
        // *** End Combined Case ***

        case .f16:
            if outputType == .f16 {
                 print("Info: Returning original F16 buffer for tensor '\(tensorName)'.")
                 dequantizedBuffer = sourceBuffer
            } else if outputType == .f32 {
                 print("Error: F16 to F32 dequantization not implemented yet.")
                 throw ModelLoaderError.unsupportedTensorType(tensorName, .f16)
            } else {
                 print("Error: Cannot convert F16 tensor '\(tensorName)' to \(outputType).")
                 throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)
            }

        case .f32:
             if outputType == .f32 {
                 print("Info: Returning original F32 buffer for tensor '\(tensorName)'.")
                 dequantizedBuffer = sourceBuffer
             } else if outputType == .f16 {
                  print("Error: F32 to F16 quantization not implemented yet.")
                  throw ModelLoaderError.unsupportedTensorType(tensorName, .f32)
             } else {
                  print("Error: Cannot convert F32 tensor '\(tensorName)' to \(outputType).")
                  throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)
             }

        case .q6_K: // Added previously for parsing
             print("Error: Dequantization for source tensor type \(tensor.type) (Q6_K) is not yet implemented.")
             throw ModelLoaderError.unsupportedTensorType(tensorName, tensor.type)
        // NOTE: No .f64 case needed here (handled by CPU conversion earlier)
        default: // Catches any other types not handled above
            print("Error: Dequantization for source tensor type \(tensor.type) is not supported.")
            throw ModelLoaderError.unsupportedTensorType(tensorName, tensor.type)
        }

        guard let finalBuffer = dequantizedBuffer else {
             // This should ideally not happen if all paths assign or throw
             print("Internal Error: Dequantized buffer is unexpectedly nil after switch for \(tensorName).")
             throw ModelLoaderError.dequantizationFailed(tensorName, nil)
        }

        // Cache the result (using the same cache key logic)
        dequantizedBufferCache[cacheKey] = finalBuffer
        // Update label if conversion happened, otherwise it keeps original label
        if finalBuffer !== sourceBuffer { // Check if it's a new buffer
             finalBuffer.label = "\(tensorName)_converted_\(outputType)"
        }
        print("Successfully processed and cached: \(tensorName) -> \(outputType)")

        return finalBuffer
    }
}
