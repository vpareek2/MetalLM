import Metal  // Needed for MTLBuffer in test function
import MetalKit  // Potentially useful for test data generation
import SwiftUI
import UniformTypeIdentifiers  // For UTType

struct ContentView: View {
    // Use the updated wrapper
    @StateObject private var modelLoaderWrapper = ModelLoaderWrapper()  // Error occurs here if class isn't defined yet

    // ... (Rest of ContentView struct code as provided in the previous correct version) ...
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
                Button("Clear") {
                    selectedFileURL = nil
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

            // --- Load Button ---
            Button {
                Task {
                    await loadFullModelAndTestRope()  // Modified action
                }
            } label: {
                Label("Load Full Model & Test RoPE", systemImage: "memorychip")  // Updated label
            }
            .disabled(selectedFileURL == nil || !modelLoaderWrapper.isMetadataLoaded || isLoading)
            .padding(.vertical)

            Divider()

            // --- Status and Model Info ---
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
            Spacer()
        }
        .padding()
        .frame(minWidth: 500, minHeight: 350)
    }

    // File selection function remains the same
    private func selectGGUFFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]

        if panel.runModal() == .OK {
            if let url = panel.url {
                selectedFileURL = url
                Task {
                    isLoading = true
                    await modelLoaderWrapper.loadMetadata(url: url)
                    isLoading = false
                }
            } else {
                Task { @MainActor in
                    selectedFileURL = nil
                    modelLoaderWrapper.currentStatus = "Error: Could not get URL for selected file."
                }
            }
        } else {
            Task { @MainActor in
                if selectedFileURL == nil {
                    modelLoaderWrapper.currentStatus = "File selection cancelled."
                }
            }
        }
    }

    // Modified function to load model AND run the RoPE test
    @MainActor
    private func loadFullModelAndTestRope() async {
        guard let urlToLoad = selectedFileURL else {
            modelLoaderWrapper.currentStatus = "Error: No file selected to load."
            return
        }
        guard modelLoaderWrapper.isMetadataLoaded, modelLoaderWrapper.loadedModelConfig != nil
        else {
            modelLoaderWrapper.currentStatus =
                "Error: Metadata or Config not loaded successfully before loading full model."
            return
        }

        isLoading = true

        // Step 1: Assemble the full model
        await modelLoaderWrapper.assembleFullModel(url: urlToLoad)

        // Step 2: Check if model loading succeeded and run RoPE test
        if !modelLoaderWrapper.currentStatus.lowercased().contains("error"),
            let service = modelLoaderWrapper.getMetalService(),
            let loadedModel = modelLoaderWrapper.getLoadedModel()
        {
            modelLoaderWrapper.currentStatus += "\nRunning RoPE kernel sanity check..."
            testRopeKernel(metalService: service, model: loadedModel)
        } else {
            print("Skipping RoPE test due to model loading failure or missing components.")
            // Optionally update status message here too
            // modelLoaderWrapper.currentStatus += "\nSkipping RoPE test."
        }

        isLoading = false
    }

    // --- TEMPORARY RoPE TEST FUNCTION ---
    private func testRopeKernel(metalService: MetalService, model: LlamaModel) {
        // ... (Implementation of testRopeKernel as provided previously) ...
        print("--- Running RoPE Sanity Check ---")
        let config = model.config  // Get config from loaded model

        let testSeqLen = 4
        let testNumHeads = 2
        let testHeadDim = config.headDim
        guard config.ropeDimensionCount <= testHeadDim else {
            print(
                "RoPE Test Error: ropeDimensionCount (\(config.ropeDimensionCount)) > headDim (\(testHeadDim))"
            )
            return
        }
        let elementCount = testSeqLen * testNumHeads * testHeadDim
        let bufferSize = elementCount * MemoryLayout<Float16>.size

        guard bufferSize > 0 else {
            print("RoPE Test Error: Invalid dimensions leading to zero buffer size.")
            return
        }

        var inputData = [Float16](repeating: 0, count: elementCount)
        for i in 0..<elementCount { inputData[i] = Float16(i % 10) }
        let originalData = inputData

        guard
            let testDataBuffer = metalService.device.makeBuffer(
                bytes: &inputData, length: bufferSize, options: .storageModeShared)
        else {
            print("RoPE Test Error: Failed to create test data buffer.")
            return
        }
        testDataBuffer.label = "RoPE Test Input/Output"

        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
            print("RoPE Test Error: Failed to create command buffer.")
            return
        }
        commandBuffer.label = "RoPE Test Command Buffer"

        let success = metalService.applyRoPE(
            commandBuffer: commandBuffer, buffer: testDataBuffer,
            ropeFrequencies: model.ropeFrequencies,
            config: config, posOffset: 0, sequenceLength: testSeqLen, numHeads: testNumHeads,
            headDim: testHeadDim
        )

        guard success else {
            print("RoPE Test FAILED: applyRoPE function returned false.")
            return
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            print("RoPE Test FAILED: Command buffer execution failed: \(error)")
            if let nsError = error as NSError? { print("  User Info: \(nsError.userInfo)") }
            return
        }

        let resultPtr = testDataBuffer.contents().bindMemory(
            to: Float16.self, capacity: elementCount)
        let resultData = Array(UnsafeBufferPointer(start: resultPtr, count: elementCount))

        var changedRotatedCount = 0
        var changedNonRotatedCount = 0
        var nanInfCount = 0
        let ropeDims = config.ropeDimensionCount

        for s in 0..<testSeqLen {
            for h in 0..<testNumHeads {
                for d in 0..<testHeadDim {
                    let index = s * (testNumHeads * testHeadDim) + h * testHeadDim + d
                    let originalValue = originalData[index]
                    let resultValue = resultData[index]
                    if !resultValue.isFinite { nanInfCount += 1 }
                    if resultValue != originalValue {
                        if d < ropeDims {
                            changedRotatedCount += 1
                        } else {
                            changedNonRotatedCount += 1
                        }
                    }
                }
            }
        }

        print("RoPE Test Results:")
        print("  - Total Elements: \(elementCount)")
        print("  - Rotated Dimensions per Head: \(ropeDims)")
        print("  - Total Expected Rotated Elements: \(testSeqLen * testNumHeads * ropeDims)")
        print("  - Values Changed within Rotated Dimensions: \(changedRotatedCount)")
        print("  - Values Changed outside Rotated Dimensions: \(changedNonRotatedCount)")
        print("  - NaN/Infinity count: \(nanInfCount)")

        if nanInfCount > 0 {
            print("  - Sanity Check: FAILED (NaN/Inf detected)")
        } else if changedRotatedCount == 0 {
            print("  - Sanity Check: FAILED (No values changed within rotated dimensions)")
        } else if changedNonRotatedCount > 0 {
            print("  - Sanity Check: FAILED (Values outside rotated dimensions were changed!)")
        } else {
            print("  - Sanity Check: PASSED")
        }

        let printCount = min(elementCount, 32)
        print(
            "  - Original First \(printCount): \(originalData.prefix(printCount).map { String(format: "%.2f", Float($0)) })"
        )
        print(
            "  - Result First \(printCount):   \(resultData.prefix(printCount).map { String(format: "%.2f", Float($0)) })"
        )
        print("--- RoPE Sanity Check Complete ---")
    }
}

#Preview {
    ContentView()
}
