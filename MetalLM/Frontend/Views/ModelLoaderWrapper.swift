// MetalLM/Frontend/ViewModels/ModelLoaderWrapper.swift

import Foundation  // For URL, etc.
import Metal  // Needed because LlamaModel holds MTLBuffers
import SwiftUI  // For ObservableObject, @Published, @MainActor

// Wrapper class to hold onto our services and model data
@MainActor  // Ensures changes/accesses happen on main thread
class ModelLoaderWrapper: ObservableObject {

    private var metalService: MetalService?
    private var modelLoader: ModelLoader?

    // State properties
    @Published var currentStatus: String = "Not loaded"
    @Published var isMetadataLoaded: Bool = false  // Track if metadata is ready
    @Published var loadedModelConfig: LlamaConfig? = nil  // Store config after loading

    // Store the fully loaded model (make accessible via getter)
    private var llamaModel: LlamaModel?
    private var llamaRunner: LlamaRunner?

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
        // Ensure the URL passed matches the one metadata was loaded from
        guard url == loader.ggufFile?.url else {
            currentStatus =
                "Error: Attempting to load full model from a different URL (\(url.lastPathComponent)) than the loaded metadata (\(loader.ggufFile?.url.lastPathComponent ?? "None")). Please re-select the file."
            isMetadataLoaded = false  // Require re-selection
            return
        }

        print("Wrapper: Attempting to assemble full model from \(url.path)")
        currentStatus = "Loading full model tensors..."
        // Keep previous model until new one is successfully loaded? Or clear immediately?
        // Clearing immediately prevents using stale model if load fails.
        llamaModel = nil

        do {
            // Call the main loading function in ModelLoader
            let loadedModel = try await loader.loadLlamaModel(
                url: url,
                computePrecision: .f16,
                normWeightType: .f32,
                embeddingType: .f32  // Keep consistent with previous fix
            )

            // Store the loaded model
            self.llamaModel = loadedModel
            self.loadedModelConfig = loadedModel.config  // Ensure config is updated

            // --- Instantiate the Runner ---
            guard let service = self.metalService else {
                currentStatus = "Error: MetalService unavailable for Runner creation."
                print("Error: MetalService unavailable when creating LlamaRunner.")
                self.llamaModel = nil
                return
            }
            do {
                // --- CHOOSE ONE RUNNER TO UNCOMMENT ---
                // Option 1: Use the original runner
                self.llamaRunner = try LlamaRunner(model: loadedModel, metalService: service)
                currentStatus =
                    "Success: Full model '\(url.lastPathComponent)' loaded and Runner initialized!"
                print("Wrapper: Full model assembly and Runner initialization complete.")

                // Option 2: Use the debug runner (if still debugging forward pass)
                // self.llamaRunner = try DebugLlamaRunner(model: loadedModel, metalService: service)
                // currentStatus =
                //    "Success: Full model '\(url.lastPathComponent)' loaded and DEBUG Runner initialized!"
                // print("Wrapper: Full model assembly and DEBUG Runner initialization complete.")
                // --------------------------------------

            } catch let error as LlamaRunnerError { // Now reachable
                currentStatus = "Model loaded, but Runner init failed (KV Cache?): \(error)"
                print("LlamaRunner initialization failed: \(error)")
                self.llamaModel = nil
            } catch let error as DebugRunnerError { // Now reachable (only if using DebugLlamaRunner)
                currentStatus = "Model loaded, but DEBUG Runner init failed: \(error)"
                print("DebugLlamaRunner initialization failed: \(error)")
                self.llamaModel = nil
            } catch { // Now reachable
                currentStatus = "Model loaded, but Runner init failed unexpectedly: \(error)"
                print("LlamaRunner initialization failed with unexpected error: \(error)")
                self.llamaModel = nil
            }
            // --- End Runner Instantiation ---

        } catch let error as ModelLoaderError {
            currentStatus = "Error assembling model: \(error)"
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
        llamaRunner = nil
        llamaModel = nil
        loadedModelConfig = nil
        isMetadataLoaded = false
        currentStatus = "Model unloaded. Select a GGUF file."
        modelLoader?.unloadModel()  // Also clear caches in ModelLoader
        print("Wrapper: Cleared loaded model.")
    }
}

// Helper functions to access private properties safely
// Helper functions to access private properties safely
@MainActor
extension ModelLoaderWrapper {
    func getMetalService() -> MetalService? {
        // Ensure metalService is accessible if needed outside
        // If it's only needed internally, this getter might not be necessary
        return self.metalService
    }

    func getLoadedModel() -> LlamaModel? {
        // Provide read-only access to the loaded model
        return self.llamaModel
    }

    // V-- CHANGE THE RETURN TYPE HERE --V
    func getLlamaRunner() -> LlamaRunner? {
        return self.llamaRunner
    }
}
