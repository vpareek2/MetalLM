import Foundation
import Metal
import SwiftUI

@MainActor
class ModelLoaderWrapper: ObservableObject {
    private var metalService: MetalService?
    private var modelLoader: ModelLoader?
    private var tokenizer: Tokenizer?

    @Published var currentStatus: String = "Not loaded"
    @Published var isMetadataLoaded: Bool = false
    @Published var loadedModelConfig: LlamaConfig? = nil

    private var llamaModel: LlamaModel?
    private var llamaRunner: LlamaRunner?

    init() {
        self.metalService = MetalService.shared
        if self.metalService == nil {
            currentStatus = "Error: Metal initialization failed!"
            isMetadataLoaded = false
        } else {
            self.modelLoader = ModelLoader(metalService: metalService!)
            currentStatus = "Metal Service Ready. Select a GGUF file."
            isMetadataLoaded = false
        }
    }

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
        llamaModel = nil
        llamaRunner = nil
        tokenizer = nil

        do {
            try await Task { try loader.loadMetadata(url: url) }.value
            currentStatus = "Metadata loaded for \(url.lastPathComponent). Ready to load full model."
            isMetadataLoaded = true
            if let metadata = loader.ggufFile?.metadata {
                do {
                    self.loadedModelConfig = try LlamaConfig(metadata: metadata)
                    currentStatus += "\nConfig parsed."
                } catch {
                    currentStatus = "Metadata loaded, but failed to parse config: \(error.localizedDescription)"
                    isMetadataLoaded = false
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

    func loadTokenizer() async {
        guard let loader = self.modelLoader, let ggufFile = loader.ggufFile else {
            currentStatus = "Error: ModelLoader or GGUF file not initialized. Load metadata first."
            return
        }
        print("Wrapper: Attempting to load tokenizer...")
        currentStatus = "Loading tokenizer..."

        do {
            self.tokenizer = try Tokenizer(ggufFile: ggufFile)
            currentStatus = "Tokenizer loaded successfully."
        } catch {
            currentStatus = "Error loading tokenizer: \(error.localizedDescription)"
            self.tokenizer = nil
        }
        print("Wrapper: Tokenizer loading complete. Status: \(currentStatus)")
    }

    func assembleFullModel(url: URL) async {
        guard let loader = self.modelLoader else {
            currentStatus = "Error: ModelLoader not initialized."
            return
        }
        guard let config = self.loadedModelConfig else {
            currentStatus = "Error: Model config not available. Load metadata first."
            isMetadataLoaded = false
            return
        }
        guard isMetadataLoaded else {
            currentStatus = "Error: Metadata must be loaded before assembling the full model."
            return
        }
        guard url == loader.ggufFile?.url else {
            currentStatus = "Error: Attempting to load full model from a different URL (\(url.lastPathComponent)) than the loaded metadata (\(loader.ggufFile?.url.lastPathComponent ?? "None")). Please re-select the file."
            isMetadataLoaded = false
            return
        }

        print("Wrapper: Attempting to assemble full model from \(url.path)")
        currentStatus = "Loading full model tensors..."
        llamaModel = nil
        llamaRunner = nil
        tokenizer = nil

        let getBuffer: @Sendable (String, GGUFDataType) async throws -> MTLBuffer = { name, targetTypeIfNonF64 in
            print("Requesting tensor: \(name) (Preferred type if not F64: \(targetTypeIfNonF64))")
            return try await Task { [weak self] in
                guard let self = self, let loader = await self.modelLoader else {
                    throw ModelLoaderError.modelNotLoaded
                }
                guard let tensorDesc = loader.getTensorDescriptor(name: name) else {
                    print("Error: Tensor '\(name)' not found in GGUF tensor list during lookup.")
                    throw ModelLoaderError.tensorNotFound(name + " (lookup failed)")
                }
                let originalType = tensorDesc.type
                print("  Original type is: \(originalType)")

                let finalTargetType: GGUFDataType
                if originalType == .f64 {
                    finalTargetType = .f32
                    print("  Original type is F64, forcing target type to F32.")
                } else {
                    finalTargetType = targetTypeIfNonF64
                }
                print("  Final target type for processing: \(finalTargetType)")
                return try loader.dequantizeTensor(tensorName: name, outputType: finalTargetType)
            }.value
        }

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

        let computePrecision: GGUFDataType = .f16
        let normWeightType: GGUFDataType = .f32
        let embeddingType: GGUFDataType = .f16

        do {
            async let tokenEmbeddingsBuffer = getBuffer(try tensorName("token_embd.weight", nil), embeddingType)
            async let finalNormWeightBuffer = getBuffer(try tensorName("output_norm.weight", nil), normWeightType)
            async let outputWeightBuffer = getBuffer(try tensorName("output.weight", nil), embeddingType)

            var ropeFreqsBuffer: MTLBuffer? = nil
            let ropeFreqsTensorName = "rope_freqs.weight"
            do {
                ropeFreqsBuffer = try await getBuffer(ropeFreqsTensorName, .f32)
                print("Found and loaded optional RoPE frequencies tensor: \(ropeFreqsTensorName)")
            } catch ModelLoaderError.tensorNotFound {
                print("Optional RoPE frequencies tensor '\(ropeFreqsTensorName)' not found (this is often expected).")
                ropeFreqsBuffer = nil
            } catch {
                print("Error loading optional RoPE frequencies tensor '\(ropeFreqsTensorName)': \(error)")
                throw error
            }

            print("Loading \(config.numLayers) transformer blocks...")
            var blocks: [LlamaTransformerBlock] = []
            try await withThrowingTaskGroup(of: (Int, LlamaTransformerBlock).self) { group in
                for i in 0..<config.numLayers {
                    group.addTask {
                        print("  Loading Block \(i)...")
                        async let attnNormWeight = getBuffer(try tensorName("blk.%d.attn_norm.weight", i), normWeightType)
                        async let ffnNormWeight = getBuffer(try tensorName("blk.%d.ffn_norm.weight", i), normWeightType)
                        async let qWeight = getBuffer(try tensorName("blk.%d.attn_q.weight", i), computePrecision)
                        async let kWeight = getBuffer(try tensorName("blk.%d.attn_k.weight", i), computePrecision)
                        async let vWeight = getBuffer(try tensorName("blk.%d.attn_v.weight", i), computePrecision)
                        async let oWeight = getBuffer(try tensorName("blk.%d.attn_output.weight", i), computePrecision)
                        async let gateWeight = getBuffer(try tensorName("blk.%d.ffn_gate.weight", i), computePrecision)
                        async let upWeight = getBuffer(try tensorName("blk.%d.ffn_up.weight", i), computePrecision)
                        async let downWeight = getBuffer(try tensorName("blk.%d.ffn_down.weight", i), computePrecision)

                        let attention = try await LlamaAttention(qWeight: qWeight, kWeight: kWeight, vWeight: vWeight, oWeight: oWeight)
                        let mlp = try await LlamaMLP(gateWeight: gateWeight, upWeight: upWeight, downWeight: downWeight)
                        let block = try await LlamaTransformerBlock(attentionNormWeight: attnNormWeight, ffnNormWeight: ffnNormWeight, attention: attention, mlp: mlp)
                        print("  Block \(i) loaded.")
                        return (i, block)
                    }
                }
                var tempBlocks = [LlamaTransformerBlock?](repeating: nil, count: config.numLayers)
                for try await (index, block) in group {
                    tempBlocks[index] = block
                }
                blocks = tempBlocks.compactMap { $0 }
                guard blocks.count == config.numLayers else {
                    throw ModelLoaderError.tensorNameCreationFailed(layer: -1, type: "Failed to load all blocks")
                }
            }
            print("All transformer blocks loaded.")

            let awaitedTokenEmbeddings = try await tokenEmbeddingsBuffer
            let awaitedFinalNorm = try await finalNormWeightBuffer
            let awaitedOutputWeight = try await outputWeightBuffer

            print(">>> DEBUG WRAPPER: Re-validating awaitedOutputWeight before LlamaModel creation...")
            let outputWeightElementCount = awaitedOutputWeight.length / MemoryLayout<Float>.stride
            var isValidBeforeModel = true
            var firstProblem = -1

            if awaitedOutputWeight.storageMode != .private {
                let ptr = awaitedOutputWeight.contents().bindMemory(to: Float.self, capacity: outputWeightElementCount)
                let checkCount = min(outputWeightElementCount, 10000)
                let bp = UnsafeBufferPointer(start: ptr, count: checkCount)

                for i in 0..<bp.count {
                    if bp[i].isNaN || bp[i].isInfinite {
                        print("!!! DEBUG WRAPPER: NaN/Inf found in awaitedOutputWeight BEFORE LlamaModel creation at index \(i) with value \(bp[i]) !!!")
                        isValidBeforeModel = false
                        firstProblem = i
                        break
                    }
                }
            } else {
                print(">>> DEBUG WRAPPER: Skipping CPU validation for private awaitedOutputWeight.")
            }

            guard isValidBeforeModel else {
                currentStatus = "Error: output.weight buffer corrupted immediately after loading! Problem at index \(firstProblem)."
                self.llamaModel = nil
                self.loadedModelConfig = nil
                return
            }
            print(">>> DEBUG WRAPPER: awaitedOutputWeight validation PASSED before LlamaModel creation.")

            print("Assembling final LlamaModel...")
            let finalLlamaModel = LlamaModel(
                config: config,
                tokenEmbeddings: awaitedTokenEmbeddings,
                blocks: blocks,
                finalNormWeight: awaitedFinalNorm,
                outputWeight: awaitedOutputWeight,
                ropeFrequencies: ropeFreqsBuffer
            )
            self.llamaModel = finalLlamaModel
            self.loadedModelConfig = finalLlamaModel.config
            print("LlamaModel assembly complete.")

            guard let ggufFile = loader.ggufFile else {
                throw ModelLoaderError.modelNotLoaded
            }
            self.tokenizer = try Tokenizer(ggufFile: ggufFile)
            print("Tokenizer initialized during model assembly.")

            guard let service = self.metalService else {
                currentStatus = "Error: MetalService unavailable for Runner creation."
                self.llamaModel = nil
                return
            }
            do {
                guard let modelToRun = self.llamaModel else {
                    currentStatus = "Error: LlamaModel became nil before runner creation."
                    return
                }
                self.llamaRunner = try LlamaRunner(model: modelToRun, metalService: service)
                currentStatus =
                    "Success: Full model '\(url.lastPathComponent)' loaded, tokenizer initialized, and Runner ready!"
                print("Wrapper: Full model assembly, tokenizer, and Runner initialization complete.")
            } catch let error as LlamaRunnerError {
                currentStatus = "Model loaded, but Runner init failed (KV Cache?): \(error)"
                print("LlamaRunner initialization failed: \(error)")
                self.llamaModel = nil
            } catch {
                currentStatus = "Model loaded, but Runner init failed unexpectedly: \(error)"
                print("LlamaRunner initialization failed with unexpected error: \(error)")
                self.llamaModel = nil
            }
        } catch let error as ModelLoaderError {
            currentStatus = "Error assembling model: \(error)"
            print("Caught ModelLoaderError during assembly: \(error)")
            self.llamaModel = nil
            self.llamaRunner = nil
            self.loadedModelConfig = nil
            self.isMetadataLoaded = false
            self.tokenizer = nil
        } catch let error as ConfigError {
            currentStatus = "Error reading config during assembly: \(error)"
            print("Caught ConfigError during assembly: \(error)")
            self.llamaModel = nil
            self.llamaRunner = nil
            self.loadedModelConfig = nil
            self.isMetadataLoaded = false
            self.tokenizer = nil
        } catch {
            currentStatus = "Unexpected error assembling model: \(error.localizedDescription)"
            print("Caught unexpected error during assembly: \(error)")
            self.llamaModel = nil
            self.llamaRunner = nil
            self.loadedModelConfig = nil
            self.isMetadataLoaded = false
            self.tokenizer = nil
        }
    }

    func clearLoadedModel() {
        llamaRunner = nil
        llamaModel = nil
        tokenizer = nil
        loadedModelConfig = nil
        isMetadataLoaded = false
        currentStatus = "Model unloaded. Select a GGUF file."
        modelLoader?.unloadModel()
        print("Wrapper: Cleared loaded model and tokenizer.")
    }

    func getMetalService() -> MetalService? {
        return self.metalService
    }

    func getLoadedModel() -> LlamaModel? {
        return self.llamaModel
    }

    func getLlamaRunner() -> LlamaRunner? {
        return self.llamaRunner
    }

    func getTokenizer() -> Tokenizer? {
        return self.tokenizer
    }
}
