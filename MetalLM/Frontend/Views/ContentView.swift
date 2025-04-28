import SwiftUI
import UniformTypeIdentifiers  // For UTType

struct ContentView: View {
    // Use the updated wrapper
    @StateObject private var modelLoaderWrapper = ModelLoaderWrapper()

    @State private var isLoading: Bool = false  // Keep for UI feedback
    @State private var selectedFileURL: URL? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("MetaLLM Model Loader")
                .font(.title)
                .padding(.bottom, 5)

            // --- File Selection ---
            HStack {
                Button {
                    selectGGUFFile()
                } label: {
                    Label("Select GGUF File...", systemImage: "folder.badge.plus")
                }
                // Optional: Add button to clear/unload
                Button("Clear") {
                    selectedFileURL = nil
                    modelLoaderWrapper.clearLoadedModel()
                }
                .disabled(selectedFileURL == nil && !modelLoaderWrapper.isMetadataLoaded)

                Spacer()  // Push buttons left
            }

            if let url = selectedFileURL {
                Text("Selected: \(url.lastPathComponent)")
                    .font(.caption).lineLimit(1).truncationMode(.middle)
            } else {
                Text("No file selected.").font(.caption)
            }
            Divider()

            // --- Load Button ---
            Button {
                Task {  // Run the async loading function
                    await loadFullModel()
                }
            } label: {
                Label("Load Full Model", systemImage: "memorychip")
            }
            // Enable only if a file is selected AND metadata is loaded
            .disabled(selectedFileURL == nil || !modelLoaderWrapper.isMetadataLoaded || isLoading)
            .padding(.vertical)

            Divider()

            // --- Status and Model Info ---
            if isLoading {
                ProgressView("Loading Model...")  // More specific message
                    .padding(.vertical)
            }

            // Display status from the wrapper
            Text(modelLoaderWrapper.currentStatus)
                .font(.footnote)
                .lineLimit(nil)  // Allow multiple lines
                .fixedSize(horizontal: false, vertical: true)  // Prevent truncation
                .padding(.vertical)
                .foregroundColor(
                    modelLoaderWrapper.currentStatus.lowercased().contains("error")
                        ? .red
                        : (modelLoaderWrapper.currentStatus.lowercased().contains("success")
                            ? .green : .secondary))

            // Optionally display loaded config details
            if let config = modelLoaderWrapper.loadedModelConfig {
                Divider()
                Text("Loaded Model Config:")
                    .font(.headline)
                // Use Text concatenation or formatting for multi-line display
                VStack(alignment: .leading, spacing: 2) {
                    Text(
                        String(
                            format: "Layers: %d, Embed: %d, Hidden: %d", config.numLayers,
                            config.embeddingDim, config.hiddenDim))
                    Text(
                        String(
                            format: "Heads: %d, KV Heads: %d, Head Dim: %d", config.numHeads,
                            config.numKeyValueHeads, config.headDim))
                    Text(
                        String(
                            format: "Vocab: %d, Seq Len: %d", config.vocabSize,
                            config.sequenceLength))
                    Text(
                        String(
                            format: "RoPE Dim: %d, RoPE Freq Base: %.1f", config.ropeDimensionCount,
                            config.ropeFreqBase))
                }
                .font(.system(.caption, design: .monospaced))
            }

            Spacer()  // Pushes content up
        }
        .padding()
        .frame(minWidth: 500, minHeight: 350)  // Adjusted height
    }

    // Function to handle file selection
    // Calls the wrapper to load metadata
    private func selectGGUFFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]

        if panel.runModal() == .OK {
            if let url = panel.url {
                selectedFileURL = url
                // Trigger metadata loading in the wrapper asynchronously
                Task {
                    isLoading = true  // Show loading indicator for metadata load
                    await modelLoaderWrapper.loadMetadata(url: url)
                    isLoading = false  // Hide indicator after metadata load attempt
                }
            } else {
                // Handle error getting URL (rare)
                Task { @MainActor in  // Ensure UI update is on main thread
                    selectedFileURL = nil
                    modelLoaderWrapper.currentStatus = "Error: Could not get URL for selected file."
                }
            }
        } else {
            // User cancelled - update status gently
            Task { @MainActor in
                // Only update status if no file was previously selected
                if selectedFileURL == nil {
                    modelLoaderWrapper.currentStatus = "File selection cancelled."
                }
            }
        }
    }

    // Function to trigger full model loading
    @MainActor
    private func loadFullModel() async {
        guard let urlToLoad = selectedFileURL else {
            // This should be prevented by button disable state, but safety check
            modelLoaderWrapper.currentStatus = "Error: No file selected to load."
            return
        }

        isLoading = true  // Show loading indicator

        // Call the wrapper's assembly function
        await modelLoaderWrapper.assembleFullModel(url: urlToLoad)

        isLoading = false  // Hide loading indicator
    }
}

// Wrapper class to hold onto our services and model data
@MainActor  // Ensures changes/accesses happen on main thread
class ModelLoaderWrapper: ObservableObject {

    private var metalService: MetalService?
    private var modelLoader: ModelLoader?

    // State properties
    @Published var currentStatus: String = "Not loaded"
    @Published var isMetadataLoaded: Bool = false  // Track if metadata is ready
    @Published var loadedModelConfig: LlamaConfig? = nil  // Store config after loading

    // Store the fully loaded model (optional for now)
    private var llamaModel: LlamaModel?

    init() {
        self.metalService = MetalService.shared
        if self.metalService == nil {
            currentStatus = "Error: Metal initialization failed!"
            isMetadataLoaded = false
        } else {
            // Initialize ModelLoader here
            self.modelLoader = ModelLoader(metalService: metalService!)
            currentStatus = "Metal Service Ready. Select a GGUF file."
            isMetadataLoaded = false
        }
    }

    /// Loads *only* the metadata from the GGUF file.
    func loadMetadata(url: URL) async {
        guard let loader = self.modelLoader else {
            currentStatus = "Error: ModelLoader not initialized."
            isMetadataLoaded = false
            return
        }

        print("Wrapper: Attempting to load metadata from \(url.path)")
        currentStatus = "Loading metadata..."
        isMetadataLoaded = false
        loadedModelConfig = nil
        llamaModel = nil  // Clear any previously loaded model

        do {
            // Use Task to ensure loadMetadata runs off the main thread if it becomes heavy
            try await Task { try loader.loadMetadata(url: url) }.value
            currentStatus =
                "Metadata loaded for \(url.lastPathComponent). Ready to load full model."
            isMetadataLoaded = true
            // Optionally, extract and publish config immediately after metadata load
            // if LlamaConfig init doesn't throw often and is fast
            if let metadata = loader.ggufFile?.metadata {
                do {
                    self.loadedModelConfig = try LlamaConfig(metadata: metadata)
                    currentStatus += "\nConfig parsed."
                } catch {
                    currentStatus =
                        "Metadata loaded, but failed to parse config: \(error.localizedDescription)"
                    isMetadataLoaded = false  // Treat as not fully ready if config fails
                }
            }

        } catch let error as GGUFError {
            currentStatus = "Error loading metadata (GGUF): \(error)"
            isMetadataLoaded = false
        } catch {
            currentStatus = "Error loading metadata: \(error.localizedDescription)"
            isMetadataLoaded = false
        }
        print("Wrapper: Metadata loading complete. Status: \(currentStatus)")
    }

    /// Loads the full model structure and tensors. Requires metadata to be loaded first.
    func assembleFullModel(url: URL) async {
        guard let loader = self.modelLoader else {
            currentStatus = "Error: ModelLoader not initialized."
            return
        }
        guard isMetadataLoaded else {
            currentStatus = "Error: Metadata must be loaded before assembling the full model."
            return
        }
        // Ensure the URL passed matches the one metadata was loaded from (optional sanity check)
        guard url == loader.ggufFile?.url else {
            currentStatus =
                "Error: Attempting to load full model from a different URL (\(url.lastPathComponent)) than the loaded metadata (\(loader.ggufFile?.url.lastPathComponent ?? "None")). Please re-select the file."
            isMetadataLoaded = false  // Require re-selection
            return
        }

        print("Wrapper: Attempting to assemble full model from \(url.path)")
        currentStatus = "Loading full model tensors..."
        llamaModel = nil  // Clear previous model instance

        do {
            // Call the main loading function in ModelLoader
            let loadedModel = try await loader.loadLlamaModel(
                url: url,
                computePrecision: .f16,  // Keep compute as f16
                normWeightType: .f32,  // Keep norms as f32
                embeddingType: .f32  // Request F32 for embeddings
            )

            // Store the loaded model
            self.llamaModel = loadedModel
            self.loadedModelConfig = loadedModel.config  // Ensure config is updated

            currentStatus = "Success: Full model '\(url.lastPathComponent)' loaded and assembled!"  // This message might change if Q6_K fails
            print("Wrapper: Full model assembly complete.")

        } catch let error as ModelLoaderError {
            currentStatus = "Error assembling model: \(error)"  // More specific error from ModelLoader
            print("Caught ModelLoaderError during assembly: \(error)")
        } catch let error as ConfigError {
            currentStatus = "Error reading config during assembly: \(error)"
            print("Caught ConfigError during assembly: \(error)")
        } catch {
            currentStatus = "Unexpected error assembling model: \(error.localizedDescription)"
            print("Caught unexpected error during assembly: \(error)")
        }
    }

    // Optional: Function to clear the loaded model
    func clearLoadedModel() {
        llamaModel = nil
        loadedModelConfig = nil
        isMetadataLoaded = false
        currentStatus = "Model unloaded. Select a GGUF file."
        modelLoader?.unloadModel()  // Also clear caches in ModelLoader
        print("Wrapper: Cleared loaded model.")
    }
}

#Preview {
    ContentView()
}
