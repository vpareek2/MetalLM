import SwiftUI
import UniformTypeIdentifiers // For UTType

struct ContentView: View {
    @StateObject private var modelLoaderWrapper = ModelLoaderWrapper()

    @State private var isLoading: Bool = false
    @State private var statusMessage: String = "Select a GGUF file to load."
    @State private var dequantizedValues: [Float]? = nil
    @State private var selectedFileURL: URL? = nil

    // Define tensors to test
    let testCases: [(label: String, tensorName: String, expectedType: String)] = [
        ("F32", "rope_freqs.weight", "f32"),
        ("F64->F32", "token_embd.weight", "f64"),
        ("Q4_K_S", "blk.0.ffn_down.weight", "q4_K_S") // Type 14
        // Add more here later, e.g., find a Q4_K_M (15) or Q6_K (18) tensor if they exist
        // ("Q4_K_M", "some_tensor_name", "q4_K_M") // Placeholder
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("MetaLLM Dequantization Test")
                .font(.title)
                .padding(.bottom, 5)

            // --- File Selection ---
            HStack {
                Button {
                    selectGGUFFile()
                } label: {
                    Label("Select GGUF File...", systemImage: "folder.badge.plus")
                }
                Spacer() // Push button left
            }

            if let url = selectedFileURL {
                Text("Selected: \(url.lastPathComponent)")
                    .font(.caption).lineLimit(1).truncationMode(.middle)
            } else {
                Text("No file selected.").font(.caption)
            }
            Divider()

            // --- Test Buttons ---
            Text("Run Tests:").font(.headline)
            ScrollView(.horizontal, showsIndicators: false) { // Horizontal scroll for buttons
                 HStack {
                     ForEach(testCases, id: \.tensorName) { testCase in
                         Button("Test \(testCase.label) (\(testCase.tensorName.split(separator: ".").last ?? ""))") {
                             Task {
                                 await loadAndDequantizeSpecific(
                                     tensorName: testCase.tensorName,
                                     expectedType: testCase.expectedType
                                 )
                             }
                         }
                         .disabled(selectedFileURL == nil || isLoading)
                         .help("Test dequantization for \(testCase.tensorName) (Expected type: \(testCase.expectedType))")
                     }
                 }
                 .padding(.vertical, 5)
            }


            Divider()

            // --- Status and Results ---
            if isLoading {
                ProgressView("Processing...")
                    .padding(.vertical)
            } else {
                 Text(statusMessage)
                    .foregroundColor(statusMessage.starts(with: "Error") ? .red : (statusMessage.starts(with: "Success") ? .green : .secondary))
                    .font(.footnote)
                    .lineLimit(4)
                    .padding(.vertical)
            }

            if let values = dequantizedValues {
                Text("Dequantized Values (First ~10):")
                    .font(.headline)
                ScrollView(.vertical) {
                     VStack(alignment: .leading) {
                         ForEach(0..<min(values.count, 10), id: \.self) { index in
                            Text(String(format: "[%d]: %.6f", index, values[index]))
                                .font(.system(.body, design: .monospaced))
                        }
                         if values.count > 10 {
                             Text("...")
                         }
                     }
                     .padding(.leading, 5)
                }
                .frame(maxHeight: 150)
                .border(Color.gray.opacity(0.5)) // Add border for clarity
            }

            Spacer()
        }
        .padding()
        .frame(minWidth: 500, minHeight: 450) // Slightly wider
    }

    // Function to handle file selection only (remains synchronous)
    private func selectGGUFFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]

        if panel.runModal() == .OK {
            if let url = panel.url {
                selectedFileURL = url // Store the selected URL
                statusMessage = "File selected. Ready to run tests."
                dequantizedValues = nil // Clear previous results
            } else {
                statusMessage = "Error: Could not get URL for selected file."
                selectedFileURL = nil
            }
        } else {
            // User cancelled
            statusMessage = "File selection cancelled."
        }
    }


    // Function to handle loading and dequantizing - NOW ASYNC
    @MainActor // Ensure UI updates happen on main thread after await
    private func loadAndDequantize() async { // Mark function as async
        guard let urlToLoad = selectedFileURL else {
            statusMessage = "Error: No file selected to load."
            return
        }

        // Update UI state before starting background work
        isLoading = true
        statusMessage = "Loading GGUF metadata..."
        dequantizedValues = nil

        var finalMessage: String
        var finalValues: [Float]? = nil

        // Perform potentially long-running tasks in a non-main actor context
        // Use Task.detached or Task { await Task.yield() } if needed, but often
        // the awaits below handle switching away from main thread implicitly.
        // For clarity, we can use Task.detached or a Task running on a global executor.
        // However, since ModelLoader methods might switch threads internally (e.g., Metal),
        // we just need to ensure calls TO the @MainActor wrapper are done correctly.

        // We'll call the wrapper's methods which are @MainActor isolated.
        // Swift concurrency handles the context switching.
        do {
            // Step 1: Load the model metadata (calls @MainActor func)
            try await modelLoaderWrapper.loadModel(url: urlToLoad)
            // UI updates are safe here because we are effectively back on MainActor after await
            finalMessage = "Model loaded. Dequantizing first tensor..."
            statusMessage = finalMessage // Update status message

            // Step 2: Dequantize the first tensor (calls @MainActor func)
            finalValues = try await modelLoaderWrapper.dequantizeFirstTensor()
            finalMessage = modelLoaderWrapper.currentStatus // Read status (safe on MainActor)

        } catch let error as ModelLoaderError {
             // Handle specific ModelLoader errors
             switch error {
             case .modelNotLoaded: finalMessage = "Error: Model wasn't loaded before dequantization."
             case .tensorNotFound(let name): finalMessage = "Error: Tensor '\(name)' not found."
             case .failedToGetTensorData(let name): finalMessage = "Error: Couldn't get data for tensor '\(name)'."
             case .failedToCreateMetalBuffer(let name): finalMessage = "Error: Couldn't create Metal buffer for '\(name)'."
             case .dequantizationFailed(let name, let underlyingError):
                  finalMessage = "Error: Dequantization failed for '\(name)'."
                  if let underlyingError = underlyingError {
                      finalMessage += " (\(underlyingError.localizedDescription))"
                  }
             case .unsupportedTensorType(let name, let type): finalMessage = "Error: Unsupported tensor type '\(type)' for tensor '\(name)'."
             // Removed default case as ModelLoaderError is frozen or non-frozen enum handling is exhaustive
             case .metalServiceUnavailable: finalMessage = "Error: Metal service is unavailable."
             }
             print("Caught ModelLoaderError: \(finalMessage)") // Log details
        } catch let error as GGUFError {
             // Handle specific GGUF parsing errors from loadModel
             finalMessage = "Error parsing GGUF file: \(error)" // Customize based on GGUFError cases
             print("Caught GGUFError: \(finalMessage)")
        } catch {
             // Handle any other unexpected errors
             finalMessage = "An unexpected error occurred: \(error.localizedDescription)"
             print("Caught unexpected error: \(finalMessage)")
        }

        // Final UI updates (already on MainActor)
        statusMessage = finalMessage
        dequantizedValues = finalValues
        isLoading = false
    }

    // Updated async function
    @MainActor
    private func loadAndDequantizeSpecific(tensorName: String, expectedType: String) async {
        guard let urlToLoad = selectedFileURL else {
            statusMessage = "Error: No file selected to load."
            return
        }

        isLoading = true
        statusMessage = "Loading GGUF metadata for \(urlToLoad.lastPathComponent)..."
        dequantizedValues = nil // Clear previous results

        var finalMessage: String
        var finalValues: [Float]? = nil

        do {
            // Step 1: Load model metadata (clears cache internally)
            try await modelLoaderWrapper.loadModel(url: urlToLoad)
            statusMessage = "Model loaded. Attempting to dequantize '\(tensorName)' (Expected: \(expectedType))..."

            // Step 2: Dequantize the SPECIFIC tensor
            finalValues = try await modelLoaderWrapper.dequantizeTensorByName(name: tensorName)
            // Use status from wrapper if it's detailed, otherwise craft success message
            // finalMessage = modelLoaderWrapper.currentStatus
            finalMessage = "Success: Dequantized '\(tensorName)'. First 10 values displayed."


        } catch let error as ModelLoaderError {
            switch error {
            case .tensorNotFound(let name): finalMessage = "Error: Tensor '\(name)' not found."
            case .unsupportedTensorType(let name, let type): finalMessage = "Error: Dequantization for type '\(type)' (Tensor: '\(name)') is not supported."
            case .dequantizationFailed(let name, let underlying): finalMessage = "Error: Dequantization failed for '\(name)'" + (underlying == nil ? "." : " (\(underlying!))")
            default: finalMessage = "ModelLoaderError: \(error)" // Catch-all for other loader errors
            }
             print("Caught ModelLoaderError: \(finalMessage)")
        } catch let error as GGUFError {
             finalMessage = "GGUFError loading/parsing file: \(error)"
             print("Caught GGUFError: \(finalMessage)")
        } catch {
             finalMessage = "An unexpected error occurred: \(error)"
             print("Caught unexpected error: \(finalMessage)")
        }

        // Final UI updates
        statusMessage = finalMessage
        dequantizedValues = finalValues
        isLoading = false
    }
}

// Wrapper class to hold onto our services and model data
@MainActor // Ensures changes/accesses happen on main thread
class ModelLoaderWrapper: ObservableObject {

    private var metalService: MetalService?
    private var modelLoader: ModelLoader?

    @Published var currentStatus: String = "Not loaded"

    init() {
        self.metalService = MetalService.shared
        if self.metalService == nil {
             currentStatus = "Error: Metal initialization failed!"
        } else {
             currentStatus = "Metal Service Ready."
        }
    }

    /// Loads the model using ModelLoader instance. Now async.
    func loadModel(url: URL) async throws { // Mark as async
        guard let metal = self.metalService else {
             currentStatus = "Error: Metal Service not available."
             throw ModelLoaderError.metalServiceUnavailable
        }

        print("Loading model from: \(url.path)")
        if self.modelLoader == nil {
             self.modelLoader = ModelLoader(metalService: metal)
        } else {
             // If modelLoader exists, ensure caches are cleared for a potentially new file
             self.modelLoader?.clearCaches()
        }

        // Perform file loading potentially off the main thread if GGUFFile init is heavy
        // For now, assume GGUFFile init is acceptable on main actor or handles its own async
        try self.modelLoader?.loadModel(url: url) // This call is now safe

        currentStatus = "Model loaded: \(url.lastPathComponent)"
        print("GGUF metadata loaded successfully.")
    }

    /// Dequantizes the first tensor. Now async.
    func dequantizeFirstTensor() async throws -> [Float]? { // Mark as async
        guard let loader = self.modelLoader else {
            currentStatus = "Error: No model loaded."
            throw ModelLoaderError.modelNotLoaded
        }
        guard let firstTensorDesc = loader.ggufFile?.tensors.first else {
             currentStatus = "Error: Model contains no tensors."
             throw ModelLoaderError.tensorNotFound("N/A - No tensors found")
        }
        let firstTensorName = firstTensorDesc.name

        print("Attempting to dequantize tensor: \(firstTensorName)")
        currentStatus = "Dequantizing \(firstTensorName)..."

        // Perform the potentially long-running dequantization off the main thread
        // ModelLoader.dequantizeTensor itself might use background threads (e.g., Metal completion)
        // but the initial call needs to be managed.
        // We can make ModelLoader.dequantizeTensor async or wrap the call here.
        // Let's assume ModelLoader.dequantizeTensor can be called and might block
        // or use its own async pattern. For safety, wrap in Task.

        // *** Option 1: Make ModelLoader.dequantizeTensor async (Preferred) ***
        // If ModelLoader.dequantizeTensor is marked async:
        // let dequantizedBuffer = try await loader.dequantizeTensor(tensorName: firstTensorName, outputType: .f32)

        // *** Option 2: Wrap synchronous ModelLoader.dequantizeTensor in Task ***
        let dequantizedBuffer = try await Task { // Run potentially blocking work in background Task
             try loader.dequantizeTensor(tensorName: firstTensorName, outputType: .f32)
        }.value // Get the result back


        print("Tensor dequantized successfully. Reading back values...")
        currentStatus = "Reading back values for \(firstTensorName)..."

        let elementCount = Int(firstTensorDesc.elementCount)
        let valuesToRead = min(elementCount, 10)

        guard valuesToRead > 0 else {
             currentStatus = "Tensor '\(firstTensorName)' dequantized (0 elements)."
             return []
        }
        guard dequantizedBuffer.length >= valuesToRead * MemoryLayout<Float>.size else {
             currentStatus = "Error: Dequantized buffer size (\(dequantizedBuffer.length)) is too small for \(valuesToRead) floats."
             throw ModelLoaderError.dequantizationFailed(firstTensorName, nil)
        }

        // Reading buffer contents might be okay on main actor if quick,
        // but could also be done in the background task if needed.
        let pointer = dequantizedBuffer.contents().bindMemory(to: Float.self, capacity: valuesToRead)
        let values = Array(UnsafeBufferPointer(start: pointer, count: valuesToRead))

        currentStatus = "Successfully dequantized '\(firstTensorName)'."
        return values
    }

    /// Dequantizes a tensor by the specified name. Now async.
    func dequantizeTensorByName(name tensorName: String) async throws -> [Float]? { // Renamed and added parameter
        guard let loader = self.modelLoader else {
            currentStatus = "Error: No model loaded."
            throw ModelLoaderError.modelNotLoaded
        }
        // Find the tensor descriptor (optional, but good for getting element count)
        guard let tensorDesc = loader.getTensorDescriptor(name: tensorName) else {
            currentStatus = "Error: Tensor '\(tensorName)' not found in loaded model."
            throw ModelLoaderError.tensorNotFound(tensorName)
        }

        print("Attempting to dequantize tensor: \(tensorName) (Type: \(tensorDesc.type))")
        currentStatus = "Dequantizing \(tensorName)..."

        // Dequantize the specified tensor to F32
        let dequantizedBuffer = try await Task { // Run potentially blocking work in background Task
            try loader.dequantizeTensor(tensorName: tensorName, outputType: .f32)
        }.value

        print("Tensor dequantized successfully. Reading back values...")
        currentStatus = "Reading back values for \(tensorName)..."

        let elementCount = Int(tensorDesc.elementCount) // Use descriptor for count
        let valuesToRead = min(elementCount, 10) // Read first 10 again

        guard valuesToRead > 0 else {
            currentStatus = "Tensor '\(tensorName)' dequantized (0 elements)."
            return []
        }
        let expectedBufferSize = valuesToRead * MemoryLayout<Float>.size
        guard dequantizedBuffer.length >= expectedBufferSize else {
             currentStatus = "Error: Dequantized buffer size (\(dequantizedBuffer.length)) is too small for \(valuesToRead) floats (expected at least \(expectedBufferSize))."
             // It's possible elementCount was wrong, or dequantization failed silently.
             // Let's try reading what we can, up to buffer length.
             let actualFloatsReadable = dequantizedBuffer.length / MemoryLayout<Float>.size
             if actualFloatsReadable == 0 {
                 print("Buffer length \(dequantizedBuffer.length) is less than one float.")
                 throw ModelLoaderError.dequantizationFailed(tensorName, nil)
             }
             print("Warning: Buffer size mismatch. Reading only \(actualFloatsReadable) floats.")
             let pointer = dequantizedBuffer.contents().bindMemory(to: Float.self, capacity: actualFloatsReadable)
             let values = Array(UnsafeBufferPointer(start: pointer, count: actualFloatsReadable))
             currentStatus = "Successfully dequantized '\(tensorName)' (Warning: Buffer size mismatch)."
             return values
         }

        // Reading buffer contents
        let pointer = dequantizedBuffer.contents().bindMemory(to: Float.self, capacity: valuesToRead)
        let values = Array(UnsafeBufferPointer(start: pointer, count: valuesToRead))

        currentStatus = "Successfully dequantized '\(tensorName)'."
        return values
    }
}

// Ensure other supporting files (GGUFFile, ModelLoader, MetalService) are correct

#Preview {
    ContentView()
}
