import Foundation

// GGUF data types
// NOTE: Ensure these values match the GGUF spec version you are targeting.
enum GGUFDataType: UInt32 {
    case f32 = 0
    case f16 = 1
    case f64 = 12
    case q4_K_S = 14 // <<< ADD Q4_K_S
    case q4_K_M = 15
    case q6_K = 18
    // Add others from the list if needed (e.g., Q8_0 = 7, Q5_K_M = 17 etc.)

    var blockInfo: (blockSize: UInt64, blockBytes: UInt64)? {
        switch self {
        case .f32:    return (1, 4)
        case .f16:    return (1, 2)
        case .f64:    return (1, 8)
        case .q4_K_S: return (256, 146)
        case .q4_K_M: return (256, 144)
        case .q6_K:   return (256, 210)
        // Add block info for other types if you add them to the enum
        }
    }
}

// GGUF value types for metadata
enum GGUFValue {
    case uint32(UInt32)
    case float32(Float)
    case bool(Bool)
    case uint64(UInt64)
    case float64(Double)
    case string(String)
    case array([GGUFValue])

    // Helper properties
    var uint64: UInt64? {
        switch self {
        case .uint32(let val): return UInt64(val)
        case .uint64(let val): return val
        default: return nil
        }
    }
    var string: String? {
        guard case .string(let str) = self else { return nil }
        return str
    }
    var uint32: UInt32? {
        switch self {
        case .uint32(let val): return val
        case .uint64(let val): return UInt32(exactly: val)
        default: return nil
        }
    }
}

// GGUF header structure
struct GGUFHeader {
    let magic: String
    let version: UInt32
    let tensorCount: UInt64
    let metaKeyCount: UInt64  // Value read from file
}

// Tensor descriptor - clarify meaning of offset
struct GGUFTensorDescriptor {
    let name: String
    let dims: [UInt64]
    let type: GGUFDataType
    let offset: UInt64  // *** Offset RELATIVE to start of tensor data block ***

    var elementCount: UInt64 {
        dims.isEmpty ? 0 : dims.reduce(1, *)
    }

    // Corrected byteSize calculation
    var byteSize: UInt64 {
        guard elementCount > 0 else { return 0 }  // No size if no elements

        // Get block info directly from the tensor's type
        guard let info = type.blockInfo else {
            print(
                "Warning: Unknown blockInfo for type \(type). Cannot calculate byteSize for tensor '\(name)'. Returning 0."
            )
            return 0  // Return 0 if block info isn't defined for the type
        }
        let (blockSize, blockBytes) = info

        // Ensure block size is valid to prevent division by zero
        guard blockSize > 0 else {
            print(
                "Warning: Block size is 0 for type \(type). Cannot calculate byteSize for tensor '\(name)'. Returning 0."
            )
            return 0
        }

        // Calculate number of blocks needed (ceiling division)
        let numBlocks = (elementCount + blockSize - 1) / blockSize

        // Calculate total size
        let totalSize = numBlocks * blockBytes
        return totalSize  // This is definitely UInt64
    }
}

// Memory access error types
enum GGUFError: Error {
    case invalidMagic(String)
    case invalidOffset(Int)
    case invalidSize(Int, expected: Int)
    case missingMetadata(String)
    case unsupportedType(UInt32)
    case invalidString(Int)
    case dataOutOfBounds(offset: Int, size: Int, total: Int)
    case fileNotFound
    case fileReadError(Error)
    // Add other specific errors
}
// Main GGUF file parser class
class GGUFFile {
    let url: URL
    let data: Data  // Memory-mapped data
    let header: GGUFHeader
    let metadata: [String: GGUFValue]
    let tensors: [GGUFTensorDescriptor]
    let alignment: UInt64
    let dataSectionStartOffset: UInt64 // NEW: Store the start offset of the actual tensor data

    // MARK: - Initialization & Parsing
    init(url: URL) throws {
        self.url = url
        do {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw GGUFError.fileNotFound
            }
            // Keep memory mapping unless proven problematic
            self.data = try Data(contentsOf: url, options: .mappedIfSafe)
            print("--- Info: Loaded GGUF file using memory mapping.")
        } catch {
            throw GGUFError.fileReadError(error)
        }

        let reader = DataReader(data: data)

        // 1. Parse header
        self.header = try reader.readHeader()
        // --- Keep concise header log ---
        print("--- GGUF Header: Version=\(self.header.version), Tensors=\(self.header.tensorCount), KVs=\(self.header.metaKeyCount), OffsetAfterHeader=\(reader.currentOffset)")

        // 2. Parse metadata KV pairs
        let metaKeyCount = header.metaKeyCount
        self.metadata = try reader.readMetadata(count: metaKeyCount)
        let offsetAfterMetadata = reader.currentOffset
        print("--- GGUF Checkpoint: Offset after Metadata KVs = \(offsetAfterMetadata)")

        // Get alignment value
        self.alignment = metadata["general.alignment"]?.uint64 ?? 32 // Default 32 if key missing
        print("--- GGUF Info: Using alignment = \(self.alignment)")

        // 3. Parse tensor info table *sequentially*
        self.tensors = try reader.readTensorDescriptors(count: header.tensorCount)
        let offsetAfterTensorInfo = reader.currentOffset
        print("--- GGUF Checkpoint: Offset after Tensor Info Table = \(offsetAfterTensorInfo)")

        // *** ADD TEMPORARY DEBUG: Print first few tensor names and types ***
        print("--- Debug: First 15 Tensor Names & Types ---")
        for i in 0..<min(15, self.tensors.count) {
            let t = self.tensors[i]
            print("  [\(i)] Name: \(t.name), Type: \(t.type) (\(t.type.rawValue))")
        }
        // *** END TEMPORARY DEBUG ***

        // 4. Align the offset *AFTER* the tensor info table to find the start of the data section
        print("--- GGUF Checkpoint: Offset before Data Section Alignment = \(offsetAfterTensorInfo)")
        reader.alignTo(alignment: Int(alignment)) // Apply padding based on current offset
        self.dataSectionStartOffset = UInt64(reader.currentOffset) // Store the aligned offset
        print("--- GGUF Info: Calculated Tensor Data Section start offset = \(self.dataSectionStartOffset)")

        // 5. Validate metadata (Keep - simple check)
        try validateMetadata()

        // 7. *** REVISED Final Sanity Check ***
        if let lastTensor = self.tensors.last {
            let lastTensorEnd = self.dataSectionStartOffset + lastTensor.offset + lastTensor.byteSize
            let fileSize = UInt64(self.data.count)
            print("--- GGUF Info: Last tensor '\(lastTensor.name)' ends at offset \(lastTensorEnd) (relative offset \(lastTensor.offset), size \(lastTensor.byteSize))")
            print("--- GGUF Info: Total file size = \(fileSize)")
            if lastTensorEnd > fileSize {
                // This would be a genuine error
                print("--- ERROR: Calculated end of last tensor (\(lastTensorEnd)) exceeds file size (\(fileSize)). File is likely truncated or corrupt.")
                // Consider throwing an error here if strictness is needed
                // throw GGUFError.dataOutOfBounds(offset: Int(lastTensorEnd), size: 0, total: Int(fileSize))
            } else if lastTensorEnd < fileSize {
                 print("--- GGUF Info: File contains \(fileSize - lastTensorEnd) bytes after the last tensor's data (padding or extra data).")
            }
        } else if header.tensorCount > 0 {
             print("--- WARNING: No tensors found in array despite header reporting tensor count > 0.")
        }

        print("--- GGUF Success: File loaded and parsed successfully.")
    }

    // MARK: - Validation
    private func validateMetadata() throws {
        guard metadata["general.architecture"] != nil else {
            throw GGUFError.missingMetadata("general.architecture")
        }
        // Add more checks as needed...
    }

    // MARK: - Tensor Data Access
    func getTensorData(for tensor: GGUFTensorDescriptor) -> Data {
        // Calculate absolute offset using the data section start + tensor's relative offset
        let absoluteOffset = Int(self.dataSectionStartOffset + tensor.offset)
        let byteSize = Int(tensor.byteSize) // byteSize is the *unpadded* size needed

        guard byteSize >= 0 else { // Should not happen with UInt64 byteSize, but safety check
            print("Error: Tensor '\(tensor.name)' has negative byte size \(byteSize).")
            return Data()
        }
        if byteSize == 0 {
            if tensor.elementCount == 0 { return Data() }
            print("Warning: Tensor '\(tensor.name)' has calculated byte size 0 but non-zero elements.")
            return Data()
        }

        let endOffset = absoluteOffset + byteSize

        guard absoluteOffset >= 0, // Ensure start isn't negative (unlikely with UInt64)
              absoluteOffset >= Int(self.dataSectionStartOffset), // Ensure it's within the data section
              endOffset <= data.count // Ensure reading doesn't go past end of file
        else {
            print("Error: Tensor '\(tensor.name)' data range [\(absoluteOffset)..<\(endOffset)] is out of bounds. DataStart: \(self.dataSectionStartOffset), RelativeOffset: \(tensor.offset), ByteSize: \(byteSize), FileSize: \(data.count)")
            return Data() // Return empty Data on error
        }
        // Read the exact byteSize required by the tensor
        return data.subdata(in: absoluteOffset..<endOffset)
    }

    // MARK: - Information & Debugging
    func printContents() {
        print("=== GGUF File Contents ===")
        print("Header:")
        print("  Magic: \(header.magic)")
        print("  Version: \(header.version)")
        print("  Tensor Count: \(header.tensorCount)")
        print("  Metadata Key Count: \(header.metaKeyCount)")  // Removed override note
        print("  Alignment: \(alignment)")

        print("\nMetadata:")
        // Print all metadata items read
        for (key, value) in metadata.sorted(by: { $0.key < $1.key }) {
            printValue(key: key, value: value, indent: 2)
        }

        print("\nTensors:")
        for tensor in tensors {
            print("  Name: \(tensor.name)")
            print("    Dims: \(tensor.dims) (elements: \(tensor.elementCount))")
            // Make GGUFDataType printable
            print("    Type: \(String(describing: tensor.type)) (\(tensor.type.rawValue))")
            print("    Offset: \(tensor.offset)")
            print("    Size (bytes): \(tensor.byteSize)")  // Uses corrected property
        }
    }

    private func printValue(key: String, value: GGUFValue, indent: Int) {
        let indentStr = String(repeating: " ", count: indent)
        switch value {
        case .uint32(let val): print("\(indentStr)\(key): \(val)")
        case .float32(let val): print("\(indentStr)\(key): \(val)")
        case .bool(let val): print("\(indentStr)\(key): \(val)")
        case .uint64(let val): print("\(indentStr)\(key): \(val)")
        case .float64(let val): print("\(indentStr)\(key): \(val)")
        case .string(let val):
            let displayVal = val.count > 100 ? String(val.prefix(100)) + "..." : val
            print("\(indentStr)\(key): \"\(displayVal)\"")
        case .array(let arr):
            print("\(indentStr)\(key): Array[\(arr.count)]")
        }
    }
}

// MARK: - Safe Data Reader
class DataReader {
    private let data: Data
    private var offset: Int

    var currentOffset: Int { offset }

    init(data: Data) {
        self.data = data
        self.offset = 0
    }

    // MARK: - Basic Reading Operations
    func read<T>() throws -> T {
        let size = MemoryLayout<T>.size
        let readOffset = offset
        guard readOffset + size <= data.count else {
            print(
                "!!! DataReader.read<\(T.self)> OutOfBounds Error: Trying to read \(size) bytes at offset \(readOffset) but data count is \(data.count)"
            )
            throw GGUFError.dataOutOfBounds(offset: readOffset, size: size, total: data.count)
        }
        return data.withUnsafeBytes { ptr -> T in
            let boundPtr = ptr.baseAddress!.advanced(by: readOffset).assumingMemoryBound(to: T.self)
            let value = boundPtr.pointee
            offset += size
            return value
        }
    }

    func readString() throws -> String {
        let callStartOffset = offset
        // Read length (8 bytes)
        let length: UInt64 = try read()
        guard length < 1_000_000_000 else {
             throw GGUFError.invalidSize(Int(clamping: length), expected: 1_000_000_000)
        }
        let intLength = Int(length)

        // Read string data (intLength bytes)
        guard offset + intLength <= data.count else {
            throw GGUFError.dataOutOfBounds(offset: offset, size: intLength, total: data.count)
        }
        let stringData = try self.readBytes(count: intLength) // Reads bytes and advances offset
        let finalOffset = offset

        // Decode
        guard let string = String(data: stringData, encoding: .utf8) else {
            throw GGUFError.invalidString(offset - intLength)
        }

        // --- CRITICAL CHECK ONLY ---
        let expectedAdvance = 8 + intLength
        let actualAdvance = finalOffset - callStartOffset
        if actualAdvance != expectedAdvance {
             print("!!!!!! CRITICAL readString Offset Mismatch! At start offset \(callStartOffset), expected advance \(expectedAdvance), got \(actualAdvance). String length was \(length).")
        }
        // --- End Critical Check ---

        return string
    }

    func readBytes(count: Int) throws -> Data {
        guard offset + count <= data.count else {
            print(
                "!!! DataReader.readBytes OutOfBounds Error: Trying to read \(count) bytes at offset \(offset) but data count is \(data.count)"
            )
            throw GGUFError.dataOutOfBounds(offset: offset, size: count, total: data.count)
        }
        let subData = data.subdata(in: offset..<(offset + count))
        offset += count
        return subData
    }

    func peekBytes(count: Int) throws -> Data {
        guard offset + count <= data.count else {
            print(
                "!!! DataReader.peekBytes OutOfBounds Error: Trying to peek \(count) bytes at offset \(offset) but data count is \(data.count)"
            )
            throw GGUFError.dataOutOfBounds(offset: offset, size: count, total: data.count)
        }
        // Read without advancing offset
        let subData = data.subdata(in: offset..<(offset + count))
        return subData
    }

    func alignTo(alignment: Int) {
        guard alignment > 1 else { return }
        let remainder = offset % alignment
        if remainder != 0 {
            offset += (alignment - remainder)
        }
    }

    // MARK: - Structure Parsing

    func readHeader() throws -> GGUFHeader {
        let expectedHeaderSize = 4 + 4 + 8 + 8
        guard offset + expectedHeaderSize <= data.count else {
            print(
                "!!! DataReader.readHeader Error: Insufficient data for header. Need \(expectedHeaderSize) bytes at offset \(offset), have \(data.count - offset)."
            )
            throw GGUFError.dataOutOfBounds(
                offset: offset, size: expectedHeaderSize, total: data.count)
        }

        // 1. Read Magic (4 bytes)
        let magicData = try readBytes(count: 4)
        guard let magic = String(data: magicData, encoding: .utf8), magic == "GGUF" else {
            throw GGUFError.invalidMagic(
                String(data: magicData, encoding: .utf8) ?? "Invalid Bytes")
        }
        // Offset is now 4

        // 2. Read Version (4 bytes)
        let version: UInt32 = try read()
        // Offset is now 8

        // 3. Read Tensor Count (8 bytes)
        let tensorCount: UInt64 = try read()
        // Offset is now 16

        // 4. Read Metadata Key Count (8 bytes)
        let metaKeyCount: UInt64 = try read()  // Actual read that advances offset
        // Offset is now 24

        // Return the parsed header
        return GGUFHeader(
            magic: magic,
            version: version,
            tensorCount: tensorCount,
            metaKeyCount: metaKeyCount
        )
    }

    func readMetadata(count: UInt64) throws -> [String: GGUFValue] {
        var metadata: [String: GGUFValue] = [:]
        guard count <= 100_000 else {
            print("Warning: Metadata count \(count) seems excessively large.")
            return [:]
        }

        for i in 0..<count {
            let key = try readString()
            let valueType: UInt32 = try read()

            // Only pass printDebug=true if you want detail on last few items or specific keys
            let printDebug = (i >= count - 5) // Debug last 5 KVs
            let value = try readValue(type: valueType, keyHint: key, printDebug: printDebug)
            metadata[key] = value
        }

        return metadata
    }

    // Inside DataReader.swift -> readTensorDescriptors
    func readTensorDescriptors(count: UInt64) throws -> [GGUFTensorDescriptor] {
        var tensors: [GGUFTensorDescriptor] = []
        guard count <= 1_000_000 else { /* ... */ return [] }
        print("--- Debug: Reading \(count) tensor descriptors...")
        for i in 0..<count {
            let name = try readString()

            let nDims: UInt32 = try read()

            var dims: [UInt64] = []
            guard nDims <= 16 else { throw GGUFError.invalidSize(Int(nDims), expected: 16) }
            for _ in 0..<nDims {
                dims.append(try read())
            }

            // *** FOCUSED DEBUG FOR TYPE READ ***
            let typeReadOffset = offset
            var typeBytesString = "N/A"
            do {
                let peekedBytes = try self.peekBytes(count: 4) // Peek 4 bytes for UInt32 type
                typeBytesString = peekedBytes.map { String(format: "%02X", $0) }.joined(separator: " ")
                print("------ Tensor #\(i+1) '\(name)': PEEKing for type at offset \(typeReadOffset). Bytes: \(typeBytesString)")
            } catch {
                print("------ Tensor #\(i+1) '\(name)': FAILED to peek for type at offset \(typeReadOffset): \(error)")
            }

            let typeRaw: UInt32 = try read() // Read the type
            print("------ Tensor #\(i+1) '\(name)': READ typeRaw = \(typeRaw) from offset \(typeReadOffset). Offset now \(offset).")
            // *** END FOCUSED DEBUG ***

            guard let type = GGUFDataType(rawValue: typeRaw) else {
                print("!!!!!! ERROR: Unsupported tensor type raw value \(typeRaw) read for tensor '\(name)' at offset \(typeReadOffset). Bytes were: \(typeBytesString)")
                throw GGUFError.unsupportedType(typeRaw)
            }

            let dataOffset: UInt64 = try read() // Relative offset

            // Optional: Full debug log per tensor if needed
            // print("""
            // ------ Tensor #\(i+1) Parsed:
            //        Name: '\(name)'
            //        NDims: \(nDims)
            //        Dims: \(dims)
            //        Type: \(type) (\(typeRaw))
            //        RelOffset: \(dataOffset)
            // """)

            tensors.append(
                GGUFTensorDescriptor(name: name, dims: dims, type: type, offset: dataOffset)
            )
        }
        print("--- Debug: Finished reading tensor descriptors.")
        return tensors
    }

    private func readValue(type: UInt32, keyHint: String, printDebug: Bool) throws -> GGUFValue {
        let valueResult: GGUFValue

        switch type {
        case 0:
            let val: UInt8 = try read()
            valueResult = .uint32(UInt32(val))
        case 1:
            let val: Int8 = try read()
            valueResult = .uint32(UInt32(bitPattern: Int32(val)))
        case 2:
            let val: UInt16 = try read()
            valueResult = .uint32(UInt32(val))
        case 3:
            let val: Int16 = try read()
            valueResult = .uint32(UInt32(bitPattern: Int32(val)))
        case 4: valueResult = .uint32(try read())
        case 5:
            let val: Int32 = try read()
            valueResult = .uint32(UInt32(bitPattern: val))
        case 6: valueResult = .float32(try read())
        case 7:
            let val: UInt8 = try read()
            guard val == 0 || val == 1 else { throw GGUFError.invalidSize(Int(val), expected: 1) }
            valueResult = .bool(val == 1)
        case 8:  // STRING
            valueResult = .string(try readString())
        case 9:  // ARRAY
            let arrType: UInt32 = try read()
            let arrLength: UInt64 = try read()
            let arrayMetaOffset = offset // Offset *before* reading any elements

            if printDebug {
                print("--- Debug [Array for key '\(keyHint)']: Type=\(arrType), Length=\(arrLength). Offset before elements: \(arrayMetaOffset)")
            }

            var arr: [GGUFValue] = []
            guard arrLength <= 5_000_000 else {
                print("Error: Array length \(arrLength) for key '\(keyHint)' seems excessively large.")
                throw GGUFError.invalidSize(Int(clamping: arrLength), expected: 5_000_000)
            }
            arr.reserveCapacity(Int(clamping: arrLength))  // Good practice

            for i in 0..<arrLength {
                if printDebug && (i < 2 || i >= arrLength - 2) { // First/last 2
                    print("--- Debug [Array '\(keyHint)' Elem \(i+1)/\(arrLength)]: Reading at offset \(offset)")
                }

                switch arrType {
                case 4: arr.append(.uint32(try read()))
                case 5:
                    let int32Val: Int32 = try read()
                    arr.append(.uint32(UInt32(bitPattern: int32Val)))
                case 6: arr.append(.float32(try read()))
                case 8: arr.append(.string(try readString()))
                default:
                    print("Error: Unsupported array element type \(arrType) at offset \(offset) for key '\(keyHint)'")
                    throw GGUFError.unsupportedType(arrType)
                }
            }
            let arrayEndOffset = offset // Offset *after* reading all elements

            if printDebug {
                print("--- Debug [Array for key '\(keyHint)']: Finished reading array elements. Offset now: \(arrayEndOffset). Total bytes read for elements: \(arrayEndOffset - arrayMetaOffset)")
            }
            valueResult = .array(arr)
        case 10: valueResult = .uint64(try read())
        case 11:
            let val: Int64 = try read()
            valueResult = .uint64(UInt64(bitPattern: val))
        case 12: valueResult = .float64(try read())
        default:
            print("Error: Unsupported metadata value type \(type) at offset \(offset) for key '\(keyHint)'")
            throw GGUFError.unsupportedType(type)
        }

        return valueResult
    }
}
