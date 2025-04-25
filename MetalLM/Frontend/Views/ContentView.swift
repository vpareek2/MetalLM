import SwiftUI
import UniformTypeIdentifiers // For UTType

struct ContentView: View {
    // Use @StateObject for the wrapper class managing our services
    @StateObject private var modelLoaderWrapper = ModelLoaderWrapper()

    // State for UI feedback
    @State private var isLoading: Bool = false
    @State private var statusMessage: String = "Select a GGUF file to load."
    @State private var dequantizedValues: [Float]? = nil // Store results for display
    @State private var selectedFileURL: URL? = nil // Keep track of the selected file

    var body: some View {
        VStack(alignment: .leading, spacing: 15) { // Adjusted spacing
            Text("MetaLLM Dequantization Test")
                .font(.title)
                .padding(.bottom, 5)

            // File Selection Button
            Button {
                selectGGUFFile()
            } label: {
                Label("Select GGUF File...", systemImage: "folder.badge.plus")
            }
            .padding(.bottom, 5)

            // Display selected file path
            if let url = selectedFileURL {
                Text("Selected: \(url.lastPathComponent)")
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
            } else {
                Text("No file selected.")
                    .font(.caption)
            }

            // Load and Dequantize Button (only enabled if a file is selected)
            // Use Task to run async code from button action
            Button("Load & Dequantize First Tensor") {
                // Create a Task to run the async function
                Task {
                    await loadAndDequantize()
                }
            }
            .disabled(selectedFileURL == nil || isLoading) // Disable if no file or loading
            .padding(.top, 5)


            Divider() // Visually separate controls from status/output

            // Display status or results
            if isLoading {
                ProgressView("Processing...")
                    .padding(.vertical)
            } else {
                 Text(statusMessage)
                    .foregroundColor(statusMessage.starts(with: "Error") ? .red : .secondary) // Use secondary for normal status
                    .font(.footnote)
                    .lineLimit(3) // Allow more lines for errors
                    .padding(.vertical)
            }


            if let values = dequantizedValues {
                Text("Dequantized Values (First ~10):")
                    .font(.headline)
                // Use a ScrollView in case tensor names/values are long
                ScrollView(.vertical) {
                     VStack(alignment: .leading) {
                         ForEach(0..<min(values.count, 10), id: \.self) { index in
                            Text(String(format: "[%d]: %.6f", index, values[index]))
                                .font(.system(.body, design: .monospaced)) // Monospaced for numbers
                        }
                     }
                     .padding(.leading, 5) // Indent values slightly
                }
                .frame(maxHeight: 150) // Limit height of value list
            }

            Spacer() // Pushes content to the top
        }
        .padding()
        .frame(minWidth: 450, minHeight: 400) // Give the window some size
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
                statusMessage = "File selected. Ready to load."
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
}

// Ensure other supporting files (GGUFFile, ModelLoader, MetalService) are correct

#Preview {
    ContentView()
}
