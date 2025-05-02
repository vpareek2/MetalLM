import Metal
import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct ContentView: View {
    @StateObject private var modelLoaderWrapper = ModelLoaderWrapper()
    @State private var isLoading: Bool = false
    @State private var selectedFileURL: URL?
    @State private var fileBookmark: Data?

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("MetaLLM Model Loader")
                .font(.title)
                .padding(.bottom, 5)

            HStack {
                Button {
                    selectGGUFFile()
                } label: {
                    Label("Select GGUF File...", systemImage: "folder.badge.plus")
                }
                Button("Clear") {
                    selectedFileURL = nil
                    fileBookmark = nil
                    modelLoaderWrapper.clearLoadedModel()
                }
                .disabled(selectedFileURL == nil && !modelLoaderWrapper.isMetadataLoaded)
                Spacer()
            }

            if let url = selectedFileURL {
                Text("Selected: \(url.lastPathComponent)")
                    .font(.caption).lineLimit(1).truncationMode(.middle)
            } else {
                Text("No file selected.").font(.caption)
            }
            Divider()

            Button {
                testForwardPassAndSample()
            } label: {
                Label("Run Decoding Test", systemImage: "forward.fill")
            }
            .padding(.bottom)
            .disabled(modelLoaderWrapper.getLlamaRunner() == nil || modelLoaderWrapper.getTokenizer() == nil)

            Divider()

            if isLoading {
                ProgressView(
                    modelLoaderWrapper.currentStatus.contains("Loading full model")
                        ? "Loading Model..." : "Processing..."
                )
                .padding(.vertical)
            }

            Text(modelLoaderWrapper.currentStatus)
                .font(.footnote)
                .lineLimit(nil)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.vertical)
                .foregroundColor(
                    modelLoaderWrapper.currentStatus.lowercased().contains("error")
                        ? .red
                        : (modelLoaderWrapper.currentStatus.lowercased().contains("success")
                            ? .green : .secondary)
                )

            if let config = modelLoaderWrapper.loadedModelConfig {
                Divider()
                Text("Loaded Model Config:")
                    .font(.headline)
                VStack(alignment: .leading, spacing: 2) {
                    Text(String(format: "Layers: %d, Embed: %d, Hidden: %d", config.numLayers, config.embeddingDim, config.hiddenDim))
                    Text(String(format: "Heads: %d, KV Heads: %d, Head Dim: %d", config.numHeads, config.numKeyValueHeads, config.headDim))
                    Text(String(format: "Vocab: %d, Seq Len: %d", config.vocabSize, config.sequenceLength))
                    Text(String(format: "RoPE Dim: %d, RoPE Freq Base: %.1f", config.ropeDimensionCount, config.ropeFreqBase))
                }
                .font(.system(.caption, design: .monospaced))
            }
            Spacer()
        }
        .padding()
        .frame(minWidth: 500, minHeight: 350)
    }

    private func selectGGUFFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]

        if panel.runModal() == .OK {
            if let url = panel.url {
                guard url.startAccessingSecurityScopedResource() else {
                    modelLoaderWrapper.currentStatus = "Error: Failed to access security-scoped resource."
                    return
                }
                if let bookmarkData = try? url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil) {
                    self.fileBookmark = bookmarkData
                }
                selectedFileURL = url
                Task {
                    isLoading = true
                    await modelLoaderWrapper.loadMetadata(url: url)
                    await modelLoaderWrapper.assembleFullModel(url: url)
                    isLoading = false
                    url.stopAccessingSecurityScopedResource()
                }
            } else {
                selectedFileURL = nil
                modelLoaderWrapper.currentStatus = "Error: Could not get URL for selected file."
            }
        } else {
            if selectedFileURL == nil {
                modelLoaderWrapper.currentStatus = "File selection cancelled."
            }
        }
    }

    private func argmax(logits: [Float]) -> Int? {
        guard !logits.isEmpty else {
            print("Warning: Argmax called with empty logits array.")
            return nil
        }
        var maxVal: Float = -Float.infinity
        var maxIndex: Int = -1
        for (index, value) in logits.enumerated() {
            if value.isNaN {
                print("!!! Argmax Error: Found NaN at index \(index) !!!")
                return nil
            }
            if value > maxVal {
                maxVal = value
                maxIndex = index
            }
        }
        if maxIndex == -1 {
            print("Warning: Argmax did not find a valid maximum value.")
            return nil
        }
        return maxIndex
    }

    private func validateEmbeddingRowCPU(
        embeddingBuffer: MTLBuffer,
        tokenID: Int,
        embeddingDim: Int,
        label: String
    ) -> Bool {
        print("--- Validating Embedding Row on CPU: \(label) (Token ID: \(tokenID))...")
        let elementSize = MemoryLayout<Float16>.stride
        let rowSize = embeddingDim * elementSize
        let sourceOffset = tokenID * rowSize

        guard embeddingDim > 0 else {
            print("!!! Validation FAILED: embeddingDim is zero.")
            return false
        }
        guard embeddingBuffer.length >= sourceOffset + rowSize else {
            print("!!! Validation FAILED: Buffer too small (length: \(embeddingBuffer.length), needed: \(sourceOffset + rowSize)).")
            return false
        }
        guard embeddingBuffer.storageMode == .shared || embeddingBuffer.storageMode == .managed else {
            print("!!! Validation FAILED: Buffer is private, cannot access on CPU.")
            return false
        }

        let pointer = embeddingBuffer.contents().advanced(by: sourceOffset).bindMemory(to: Float16.self, capacity: embeddingDim)
        let bufferPointer = UnsafeBufferPointer(start: pointer, count: embeddingDim)

        var isValid = true
        var nanCount = 0
        var infCount = 0
        var firstNanIndex = -1
        var firstInfIndex = -1
        for i in 0..<embeddingDim {
            let value = bufferPointer[i]
            if value.isNaN {
                if firstNanIndex == -1 { firstNanIndex = i }
                nanCount += 1
                isValid = false
            }
            if value.isInfinite {
                if firstInfIndex == -1 { firstInfIndex = i }
                infCount += 1
                isValid = false
            }
        }

        if !isValid {
            print("!!! CPU Validation FAILED for embedding row '\(label)' (Token ID: \(tokenID)): \(nanCount) NaNs (first @ \(firstNanIndex)), \(infCount) Infs (first @ \(firstInfIndex)).")
        } else {
            print("--- CPU Validation PASSED for embedding row '\(label)' (Token ID: \(tokenID)).")
        }
        return isValid
    }

    private func testForwardPassAndSample() {
        self.modelLoaderWrapper.currentStatus = "Running Decoding Test..."
        print("--- Running Decoding Test ---")

        guard let runner = modelLoaderWrapper.getLlamaRunner(),
              let loadedModel = modelLoaderWrapper.getLoadedModel(),
              let metalService = modelLoaderWrapper.getMetalService(),
              let tokenizer = modelLoaderWrapper.getTokenizer() else {
            let msg = "Decoding Test Error: Runner, Service, Model, or Tokenizer not initialized. Load model first."
            print(msg)
            self.modelLoaderWrapper.currentStatus = msg
            return
        }

        runner.resetState()
        guard runner.currentPosition == 0 else {
            let msg = "Decoding Test Error: Runner position not reset."
            print(msg)
            self.modelLoaderWrapper.currentStatus = msg
            return
        }

        let prompt = "hello"
        let inputTokens = tokenizer.tokenize(prompt)
        print("Input Prompt: \(prompt)")
        print("Input Tokens: \(inputTokens)")

        let vocabSize = runner.config.vocabSize
        for tokenID in inputTokens {
            guard tokenID >= 0 && tokenID < vocabSize else {
                let msg = "Decoding Test Error: Invalid token ID \(tokenID) (vocabSize=\(vocabSize))."
                print(msg)
                self.modelLoaderWrapper.currentStatus = msg
                return
            }
        }

        let embeddingDim = loadedModel.config.embeddingDim
        let isEmbeddingRowValid = validateEmbeddingRowCPU(
            embeddingBuffer: loadedModel.tokenEmbeddings,
            tokenID: inputTokens.last ?? 1,
            embeddingDim: embeddingDim,
            label: "token_embd.weight"
        )
        guard isEmbeddingRowValid else {
            let msg = "Decoding Test Error: Source embedding data contains NaN/Inf."
            print(msg)
            self.modelLoaderWrapper.currentStatus = msg
            return
        }
        print("Source embedding row validated successfully.")

        var tokenSequence = inputTokens
        var currentTokenID = inputTokens.last ?? 1
        let maxOutputTokens = 5
        var generatedTokens = 0

        print("Processing input tokens...")
        for tokenID in inputTokens {
            print("  Processing token \(tokenID)...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let logitsBuffer = runner.forward(tokenID: tokenID)
            let endTime = CFAbsoluteTimeGetCurrent()
            print("  Forward pass for token \(tokenID) completed in \(String(format: "%.3f", endTime - startTime)) seconds.")
            guard logitsBuffer != nil else {
                let msg = "Decoding Test Error: Forward pass failed for token \(tokenID)."
                print(msg)
                self.modelLoaderWrapper.currentStatus = msg
                return
            }
        }

        print("Generating additional tokens...")
        while generatedTokens < maxOutputTokens {
            print("  Generating token \(generatedTokens + 1)...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let logitsBuffer = runner.forward(tokenID: currentTokenID)
            let endTime = CFAbsoluteTimeGetCurrent()
            print("  Forward pass completed in \(String(format: "%.3f", endTime - startTime)) seconds.")

            guard let returnedLogits = logitsBuffer else {
                let msg = "Decoding Test Error: Forward pass failed for token \(currentTokenID)."
                print(msg)
                self.modelLoaderWrapper.currentStatus = msg
                return
            }

            let logitsCount = vocabSize
            let logitsSizeBytes = logitsCount * MemoryLayout<Float>.stride
            guard returnedLogits.length >= logitsSizeBytes else {
                let msg = "Decoding Test Error: Logits buffer size mismatch."
                print(msg)
                self.modelLoaderWrapper.currentStatus = msg
                return
            }

            var cpuLogits = [Float](repeating: 0, count: logitsCount)
            if returnedLogits.storageMode != .private {
                if returnedLogits.storageMode == .managed {
                    guard let syncCommandBuffer = metalService.commandQueue.makeCommandBuffer(),
                          let blitEncoder = syncCommandBuffer.makeBlitCommandEncoder() else {
                        let msg = "Decoding Test Error: Failed to create sync command buffer."
                        print(msg)
                        self.modelLoaderWrapper.currentStatus = msg
                        return
                    }
                    blitEncoder.label = "ContentView Logits Sync Blit"
                    blitEncoder.synchronize(resource: returnedLogits)
                    blitEncoder.endEncoding()
                    syncCommandBuffer.commit()
                    syncCommandBuffer.waitUntilCompleted()
                    if let error = syncCommandBuffer.error {
                        let msg = "Decoding Test Error: Logits sync failed: \(error)"
                        print(msg)
                        self.modelLoaderWrapper.currentStatus = msg
                        return
                    }
                }

                let logitsPtr = returnedLogits.contents().bindMemory(to: Float.self, capacity: logitsCount)
                let sourceBufferPtr = UnsafeBufferPointer(start: logitsPtr, count: logitsCount)
                _ = cpuLogits.withUnsafeMutableBufferPointer { $0.initialize(from: sourceBufferPtr) }
                print("  Successfully copied logits to CPU (\(logitsCount) elements).")

                if let nextTokenID = argmax(logits: cpuLogits) {
                    print("  Sampled Next Token ID: \(nextTokenID)")
                    tokenSequence.append(nextTokenID)
                    currentTokenID = nextTokenID
                    generatedTokens += 1
                } else {
                    let msg = "Decoding Test Error: Argmax sampling failed."
                    print(msg)
                    self.modelLoaderWrapper.currentStatus = msg
                    return
                }
            } else {
                let msg = "Decoding Test Error: Cannot copy private logits buffer to CPU."
                print(msg)
                self.modelLoaderWrapper.currentStatus = msg
                return
            }
        }

        let outputText = tokenizer.decode(tokenSequence)
        let testResultMessage = "Decoding Test: PASSED\nInput: \(prompt)\nOutput: \(outputText)"
        print(testResultMessage)
        print("--- Decoding Test Complete ---")
        self.modelLoaderWrapper.currentStatus = testResultMessage
    }
}

#Preview {
    ContentView()
}
