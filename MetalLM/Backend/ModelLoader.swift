import Foundation
import Metal // Required for MTLBuffer

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
            return cachedBuffer
        }

        guard ggufFile != nil else {
            throw ModelLoaderError.modelNotLoaded
        }
        guard let tensor = getTensorDescriptor(name: tensorName) else {
            throw ModelLoaderError.tensorNotFound(tensorName)
        }

        let rawData = ggufFile!.getTensorData(for: tensor)
        guard !rawData.isEmpty else {
            if tensor.elementCount == 0 {
                 print("Warning: Tensor '\(tensorName)' has 0 elements, returning empty data.")
                 guard let buffer = metalService.device.makeBuffer(length: 0, options: .storageModeShared) else {
                     throw ModelLoaderError.failedToCreateMetalBuffer("empty buffer for \(tensorName)")
                 }
                 buffer.label = "\(tensorName)_empty"
                 quantizedBufferCache[tensorName] = buffer
                 return buffer
            } else {
                throw ModelLoaderError.failedToGetTensorData(tensorName)
            }
        }

        guard let buffer = metalService.device.makeBuffer(bytes: rawData.withUnsafeBytes { $0.baseAddress! },
                                                       length: rawData.count,
                                                       options: .storageModeShared) else {
            throw ModelLoaderError.failedToCreateMetalBuffer(tensorName)
        }
        buffer.label = tensorName
        quantizedBufferCache[tensorName] = buffer
        return buffer
    }

    /// Dequantizes a specific tensor to the desired output type using Metal kernels.
    /// Handles caching of dequantized results.
    func dequantizeTensor(tensorName: String, outputType: GGUFDataType = .f32) throws -> MTLBuffer {
        guard ggufFile != nil else {
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

        let quantizedBuffer = try getQuantizedTensorBuffer(tensorName: tensorName)
        let elementCount = Int(tensor.elementCount)

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

        // Dispatch the appropriate Metal kernel based on the *source* tensor type
        switch tensor.type {
        case .q4_K_M:
            print("Dispatching Q4_K_M dequantization for \(tensorName) -> \(outputType)...")
            do {
                if outputType == .f16 {
                    dequantizedBuffer = metalService.dequantizeQ4KM_to_f16(quantizedBuffer: quantizedBuffer, elementCount: elementCount)
                } else { // Default to f32
                    dequantizedBuffer = metalService.dequantizeQ4KM_to_f32(quantizedBuffer: quantizedBuffer, elementCount: elementCount)
                }
                if dequantizedBuffer == nil {
                     throw ModelLoaderError.dequantizationFailed(tensorName, nil)
                }
            } catch {
                 throw ModelLoaderError.dequantizationFailed(tensorName, error)
            }

        case .f16:
            if outputType == .f16 {
                 print("Info: Returning original F16 buffer for tensor '\(tensorName)'.")
                 dequantizedBuffer = quantizedBuffer
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
                 dequantizedBuffer = quantizedBuffer
             } else if outputType == .f16 {
                  print("Error: F32 to F16 quantization not implemented yet.")
                  throw ModelLoaderError.unsupportedTensorType(tensorName, .f32)
             } else {
                  print("Error: Cannot convert F32 tensor '\(tensorName)' to \(outputType).")
                  throw ModelLoaderError.unsupportedTensorType(tensorName, outputType)
             }
        
        // Added default case to handle any other GGUFDataType cases
        // This makes the switch exhaustive as required by the compiler.
        default:
            print("Error: Dequantization for source tensor type \(tensor.type) is not supported.")
            throw ModelLoaderError.unsupportedTensorType(tensorName, tensor.type)
        }

        guard let finalBuffer = dequantizedBuffer else {
             print("Internal Error: Dequantized buffer is unexpectedly nil after switch for \(tensorName).")
             throw ModelLoaderError.dequantizationFailed(tensorName, nil)
        }

        dequantizedBufferCache[cacheKey] = finalBuffer
        finalBuffer.label = "\(tensorName)_dequantized_\(outputType)"
        print("Successfully dequantized and cached: \(tensorName) -> \(outputType)")

        return finalBuffer
    }
}
