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
            // Keep memory mapping for efficiency unless proven problematic
            self.data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw GGUFError.fileReadError(error)
        }

        let reader = DataReader(data: data)

        // 1. Parse header
        self.header = try reader.readHeader()
        print(
            "--- Debug: Read Header -> Version: \(self.header.version), TensorCount: \(self.header.tensorCount), MetaKeyCount (from header): \(self.header.metaKeyCount). Offset now: \(reader.currentOffset)"
        )

        // 2. Parse metadata KV pairs (using the correct count from header)
        let metaKeyCount = header.metaKeyCount
        print("--- Debug: Reading \(metaKeyCount) metadata key-value pairs...")
        self.metadata = try reader.readMetadata(count: metaKeyCount)
        let offsetAfterMetadata = reader.currentOffset
        print("--- Debug: Offset after reading metadata: \(offsetAfterMetadata)")

        // Get alignment value (needed before reading tensors if default isn't used, but applied later)
        self.alignment = metadata["general.alignment"]?.uint64 ?? 32 // Default 32 if key missing

        // 3. Parse tensor info table *sequentially*
        print("--- Debug: Reading \(header.tensorCount) tensor descriptors starting at offset \(offsetAfterMetadata)...")
        self.tensors = try reader.readTensorDescriptors(count: header.tensorCount) // Reads all tensor infos
        let offsetAfterTensorInfo = reader.currentOffset
        print("--- Debug: Offset after reading tensor info table: \(offsetAfterTensorInfo)")

        // 4. Align the offset *AFTER* the tensor info table to find the start of the data section
        print("--- Debug: Offset before data section alignment padding (alignment=\(alignment)): \(offsetAfterTensorInfo)")
        reader.alignTo(alignment: Int(alignment)) // Apply padding based on current offset
        self.dataSectionStartOffset = UInt64(reader.currentOffset) // Store the aligned offset
        print("--- Debug: Calculated tensor data section start offset: \(self.dataSectionStartOffset)")

        // 5. Validate metadata (optional)
        try validateMetadata()
        print("GGUF file loaded successfully. Header Version: \(self.header.version), Tensors: \(self.header.tensorCount)")

        // 6. (Optional) Validate total data size matches calculated size (like llama.cpp does)
        var calculatedDataSizeStrict: UInt64 = 0 // Tracks strictly calculated offset
        var calculatedDataSizeWarn: UInt64 = 0 // Used for warnings, resets on mismatch

        for tensor in self.tensors {
            let tensorSize = tensor.byteSize
            let paddedTensorSize = (tensorSize + self.alignment - 1) & ~(self.alignment - 1)

            // Check against the warning tracker
            if tensor.offset != calculatedDataSizeWarn {
                print("--- WARNING: Tensor '\(tensor.name)' has relative offset \(tensor.offset), but expected offset \(calculatedDataSizeWarn) based on previous tensors (warning calc).")
                calculatedDataSizeWarn = tensor.offset // Reset warning tracker to file's offset
            }
            calculatedDataSizeWarn += paddedTensorSize // Advance warning tracker by padded size

            // Check against the strict tracker (no reset)
            if tensor.offset != calculatedDataSizeStrict {
                 print("--- INFO: Tensor '\(tensor.name)' relative offset \(tensor.offset) differs from strictly calculated offset \(calculatedDataSizeStrict).")
            }
            calculatedDataSizeStrict += paddedTensorSize // Advance strict tracker by padded size
        }
        let endOfFileOffset = UInt64(self.data.count)
        let expectedEndOfDataStrict = self.dataSectionStartOffset + calculatedDataSizeStrict
        print("--- Debug: Calculated total tensor data size (padded, strict): \(calculatedDataSizeStrict)")
        print("--- Debug: Expected end of data offset (strict): \(expectedEndOfDataStrict)")
        print("--- Debug: Actual end of file offset: \(endOfFileOffset)")
        if expectedEndOfDataStrict > endOfFileOffset {
            print("--- WARNING: Calculated end of tensor data (\(expectedEndOfDataStrict)) exceeds file size (\(endOfFileOffset)). File might be truncated or calculation error.")
        } else if expectedEndOfDataStrict < endOfFileOffset {
            print("--- Debug: File contains \(endOfFileOffset - expectedEndOfDataStrict) extra bytes after calculated tensor data.")
        }
    }

    // MARK: - Validation
    private func validateMetadata() throws {
        guard metadata["general.architecture"] != nil else {
            throw GGUFError.missingMetadata("general.architecture")
        }
        // Add more checks as needed...
    }

    // MARK: - Tensor Data Access *** UPDATED ***
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

        // *** ADD SPECIFIC LOGGING FOR rope_freqs.weight ***
        if tensor.name == "rope_freqs.weight" {
            print("--- Debug [getTensorData for rope_freqs.weight]:")
            print("    Relative Offset: \(tensor.offset)")
            print("    Byte Size: \(byteSize)")
            print("    Data Section Start: \(self.dataSectionStartOffset)")
            print("    Calculated Absolute Offset: \(absoluteOffset)")
            print("    Calculated End Offset: \(endOffset)")
            print("    File Size: \(data.count)")

            // Peek at the bytes where the data should start
            if absoluteOffset >= 0 && absoluteOffset + min(byteSize, 16) <= data.count { // Peek up to 16 bytes
                 let peekCount = min(byteSize, 16)
                 let peekRange = absoluteOffset..<(absoluteOffset + peekCount)
                 let peekedData = data.subdata(in: peekRange)
                 let hexString = peekedData.map { String(format: "%02X", $0) }.joined(separator: " ")
                 print("    Bytes Peeked at Absolute Offset \(absoluteOffset): \(hexString)")
            } else {
                 print("    Cannot peek bytes: Calculated range [\(absoluteOffset)..<\(endOffset)] is out of bounds or invalid.")
            }
        }
        // *** END SPECIFIC LOGGING ***

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
             print("!!! DataReader.readString Error: Read excessively large string length (\(length)) at offset \(callStartOffset). Possible data corruption or parsing error.")
             throw GGUFError.invalidSize(Int(clamping: length), expected: 1_000_000_000)
        }
        let intLength = Int(length)

        // Read string data (intLength bytes)
        guard offset + intLength <= data.count else {
            print("!!! DataReader.readString OutOfBounds Error: Trying to read \(intLength) string bytes at offset \(offset) but data count is \(data.count). Length read at offset \(callStartOffset) was \(length).")
            throw GGUFError.dataOutOfBounds(offset: offset, size: intLength, total: data.count)
        }
        let stringData = try self.readBytes(count: intLength) // Reads bytes and advances offset
        let finalOffset = offset

        // Decode
        guard let string = String(data: stringData, encoding: .utf8) else {
             print("!!! DataReader.readString Error: Failed to decode UTF8 string at offset \(offset - intLength) (length \(intLength)).")
            throw GGUFError.invalidString(offset - intLength)
        }

        // --- CRITICAL CHECK ONLY ---
        let expectedAdvance = 8 + intLength
        let actualAdvance = finalOffset - callStartOffset
        if actualAdvance != expectedAdvance {
             print("!!!!!! CRITICAL readString Offset Mismatch! At start offset \(callStartOffset), expected advance \(expectedAdvance), got \(actualAdvance). String length was \(length).")
             // Consider throwing an error here to stop immediately
             // throw GGUFError.invalidOffset(finalOffset)
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

    // Updated readHeader with more debugging
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
        let metaKeyCountOffset = offset
        // --- BEGIN DEBUG BLOCK FOR METAKEYCOUNT ---
        let metaKeyCountBytes = data.subdata(in: metaKeyCountOffset..<(metaKeyCountOffset + 8))
        let metaKeyCountRead: UInt64 = metaKeyCountBytes.withUnsafeBytes {
            $0.load(as: UInt64.self)
        }
        print(
            "--- Debug: Reading MetaKeyCount at offset \(metaKeyCountOffset). Bytes: \(metaKeyCountBytes.map { String(format: "%02X", $0) }.joined(separator: " ")). Value read: \(metaKeyCountRead)"
        )
        // --- END DEBUG BLOCK ---
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
        print("--- Debug: Reading \(count) metadata key-value pairs...")
        for i in 0..<count {
            let keyReadOffset = offset

            // *** MODIFICATION 4: Add Peek Debug for item #36 (index 35) ***
            if i == 35 {  // Check right before attempting to read the 36th item's key
                print(
                    "--- Debug [Meta \(i+1)/\(count)]: PEEKING before reading key at offset \(keyReadOffset)"
                )
                do {
                    let peekBytes = try self.peekBytes(count: 16)
                    let hexString = peekBytes.map { String(format: "%02X", $0) }.joined(
                        separator: " ")
                    print(
                        "--- Debug [Meta \(i+1)/\(count)]: Bytes peeked at \(keyReadOffset): \(hexString)"
                    )
                } catch {
                    print("--- Debug [Meta \(i+1)/\(count)]: Failed to peek bytes: \(error)")
                }
            }

            let key = try readString()

            // Decide if we still need per-item logs (maybe just for last few?)
            let printDebug = (i >= count - 5) // Only log last 5 items

            if printDebug {
                print("--- Debug [Meta \(i+1)/\(count)]: Reading Key '\(key)' (len=\(key.utf8.count)). Offset now: \(offset)")
            }

            let valueType: UInt32 = try read()
            if printDebug {
                print("--- Debug [Meta \(i+1)/\(count)]: Reading Value Type \(valueType) for key '\(key)'. Offset now: \(offset)")
            }

            // Pass printDebug flag down so readValue can decide whether to log array details
            let value = try readValue(type: valueType, keyHint: key, printDebug: printDebug)

            if printDebug {
                var valueDesc = ""
                switch value {
                case .string(let s): valueDesc = "String(len:\(s.utf8.count))"
                case .array(let a): valueDesc = "Array(len:\(a.count))" // Array log now happens inside readValue
                default: valueDesc = "\(value)"
                }
                // Don't print array desc here if logged inside readValue
                if case .array(_) = value {} else {
                    print("--- Debug [Meta \(i+1)/\(count)]: Read Value (\(valueDesc)) for key '\(key)'. Offset AFTER value read: \(offset)")
                }
            }
            metadata[key] = value
        }
        print("--- Debug: Finished reading metadata.")
        return metadata
    }

    func readTensorDescriptors(count: UInt64) throws -> [GGUFTensorDescriptor] {
        var tensors: [GGUFTensorDescriptor] = []
        guard count <= 1_000_000 else {
            print("Warning: Tensor count \(count) seems excessively large.")
            return []
        }
        print("--- Debug: Reading \(count) tensor descriptors...")
        for i in 0..<count {
            print("--- Debug: Reading tensor descriptor #\(i+1)")

            // --- Direct Read Attempt for First Tensor Name Length ---
            let nameLengthOffset = offset
            print("------ Debug: Attempting to read UInt64 tensor name length at offset \(nameLengthOffset)")
            var nameLengthBytesString = "N/A"
            do {
                let peekedBytes = try self.peekBytes(count: 8)
                nameLengthBytesString = peekedBytes.map { String(format: "%02X", $0) }.joined(separator: " ")
                print("------ Debug: Bytes peeked for length at \(nameLengthOffset): \(nameLengthBytesString)")

                // Directly call read<UInt64>()
                let nameLength: UInt64 = try self.read() // Uses the generic read<T>
                print("------ Debug: Successfully read nameLength = \(nameLength) using read<T>(). Offset now: \(offset)")

                // Now read the actual string bytes using the length
                guard nameLength < 1000 else { // Smaller limit for tensor names
                     throw GGUFError.invalidSize(Int(clamping: nameLength), expected: 1000)
                }
                let intNameLength = Int(nameLength)
                let nameData = try self.readBytes(count: intNameLength)
                guard let name = String(data: nameData, encoding: .utf8) else {
                     throw GGUFError.invalidString(offset - intNameLength)
                }
                 print("------ Debug: Successfully read tensor name = '\(name)'")

                 // --- Continue reading the rest of the descriptor ---
                 let nDims: UInt32 = try read()
                 var dims: [UInt64] = []
                 guard nDims <= 16 else { throw GGUFError.invalidSize(Int(nDims), expected: 16) }
                 for _ in 0..<nDims { dims.append(try read()) }
                 let typeRaw: UInt32 = try read()
                 guard let type = GGUFDataType(rawValue: typeRaw) else { throw GGUFError.unsupportedType(typeRaw) }
                 let dataOffset: UInt64 = try read()
                 tensors.append( GGUFTensorDescriptor(name: name, dims: dims, type: type, offset: dataOffset) )


            } catch {
                 print("!!!!!! ERROR during direct read/processing of tensor #\(i+1) name/descriptor at offset \(nameLengthOffset): \(error)")
                 print("!!!!!! Bytes peeked for length were: \(nameLengthBytesString)")
                 throw error // Re-throw the error
            }
            // --- End Direct Read Attempt ---

            // Original code (commented out for now):
            // let name = try readString()
            // let nDims: UInt32 = try read()
            // ... rest ...
            // tensors.append(...)
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
            // No extra logging here, readString handles its own check
            valueResult = .string(try readString())
        case 9:  // ARRAY
            let arrType: UInt32 = try read()
            let arrLength: UInt64 = try read()
            let arrayMetaOffset = offset // Offset *before* reading any elements

            // Log start of array reading only if printDebug is true for the parent metadata item
            if printDebug {
                print("--- Debug [Array for key '\(keyHint)']: Type=\(arrType), Length=\(arrLength). Offset before elements: \(arrayMetaOffset)")
            }

            var arr: [GGUFValue] = []
            guard arrLength <= 5_000_000 else {
                print(
                    "Error: Array length \(arrLength) for key '\(keyHint)' seems excessively large."
                )
                throw GGUFError.invalidSize(Int(clamping: arrLength), expected: 5_000_000)
            }
            arr.reserveCapacity(Int(clamping: arrLength))  // Good practice

            for i in 0..<arrLength {
                // Minimal inner loop logging (optional)
                let elementStartOffset = offset
                if printDebug && (i < 2 || i >= arrLength - 2) { // First/last 2
                    print("--- Debug [Array '\(keyHint)' Elem \(i+1)/\(arrLength)]: Reading at offset \(elementStartOffset)")
                }

                switch arrType {
                case 4: arr.append(.uint32(try read()))
                case 5:
                    let int32Val: Int32 = try read()
                    arr.append(.uint32(UInt32(bitPattern: int32Val)))
                case 6: arr.append(.float32(try read()))
                case 8: arr.append(.string(try readString())) // Minimal logging inside
                default:
                    print(
                        "Error: Unsupported array element type \(arrType) at offset \(offset) for key '\(keyHint)'"
                    )
                    throw GGUFError.unsupportedType(arrType)
                }
            }
            let arrayEndOffset = offset // Offset *after* reading all elements

            // Log end of array reading only if printDebug is true
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
            print(
                "Error: Unsupported metadata value type \(type) at offset \(offset) for key '\(keyHint)'"
            )
            throw GGUFError.unsupportedType(type)
        }
        // Optional: Log after value read IF it wasn't an array (arrays logged their own end)
        if printDebug && type != 9 {
            print("--- Debug [Value Read for key '\(keyHint)']: Type=\(type). Offset AFTER value read: \(offset)")
        }
        return valueResult
    }
}
