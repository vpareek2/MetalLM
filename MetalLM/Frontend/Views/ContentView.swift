import Metal  // Needed for MTLBuffer in test function
import MetalKit  // Potentially useful for test data generation
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

            // --- Load/Test Button ---
            // Modified Button to run MatMul test
            Button {
                // No need for async Task for this simple CPU-bound + short GPU test setup
                // Ensure MetalService is available before running the test
                if let service = modelLoaderWrapper.getMetalService() {
                    testMPSMatMul(metalService: service)  // Just run the MatMul test directly
                } else {
                    modelLoaderWrapper.currentStatus =
                        "Error: Metal Service not available for test."
                }
            } label: {
                Label("Run MPS MatMul Test", systemImage: "function")  // Updated label
            }
            // .disabled(selectedFileURL == nil || !modelLoaderWrapper.isMetadataLoaded || isLoading) // Enable if needed based on dependencies
            .padding(.vertical)

            // --- Optional: Keep the original Load & RoPE Test button ---
            /*
            Button {
                Task {
                    await loadFullModelAndTestRope()
                }
            } label: {
                Label("Load Full Model & Test RoPE", systemImage: "memorychip")
            }
            .disabled(selectedFileURL == nil || !modelLoaderWrapper.isMetadataLoaded || isLoading)
            .padding(.vertical)
            */

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
                .lineLimit(nil)  // Allow multiple lines
                .fixedSize(horizontal: false, vertical: true)  // Prevent truncation
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
                    // Load metadata when file is selected
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

    // Original function to load model AND run the RoPE test (Keep for reference or later use)
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
        modelLoaderWrapper.currentStatus = "Starting full model load..."  // Update status

        // Step 1: Assemble the full model
        await modelLoaderWrapper.assembleFullModel(url: urlToLoad)

        // Step 2: Check if model loading succeeded and run RoPE test
        if !modelLoaderWrapper.currentStatus.lowercased().contains("error"),
            let service = modelLoaderWrapper.getMetalService(),
            let loadedModel = modelLoaderWrapper.getLoadedModel()
        {
            modelLoaderWrapper.currentStatus += "\nRunning RoPE kernel sanity check..."
            testRopeKernel(metalService: service, model: loadedModel)  // Call RoPE test
        } else {
            print("Skipping RoPE test due to model loading failure or missing components.")
            modelLoaderWrapper.currentStatus += "\nSkipping RoPE test (load failed?)."
        }

        isLoading = false
    }

    // --- TEMPORARY RoPE TEST FUNCTION --- (Keep as is for now)
    private func testRopeKernel(metalService: MetalService, model: LlamaModel) {
        print("--- Running RoPE Sanity Check ---")
        // --- COMMENTED OUT FOR NOW TO FOCUS ON MATMUL ---
        /*
        let config = model.config

        let testSeqLen = 4
        let testNumHeads = 2
        let testHeadDim = config.headDim
        guard config.ropeDimensionCount <= testHeadDim else {
            print("RoPE Test Error: ropeDimensionCount (\(config.ropeDimensionCount)) > headDim (\(testHeadDim))")
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

        guard let testDataBuffer = metalService.device.makeBuffer(bytes: &inputData, length: bufferSize, options: .storageModeShared) else {
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

        let resultPtr = testDataBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
        let resultData = Array(UnsafeBufferPointer(start: resultPtr, count: elementCount))

        // ... (rest of RoPE verification logic) ...

        print("--- RoPE Sanity Check Complete ---")
        */
        print("--- RoPE Sanity Check SKIPPED (Commented out in ContentView) ---")
    }

    // --- ADDED MPS MatMul TEST FUNCTION ---
    // Example Test Function (Place appropriately, e.g., in ContentView for quick testing)
    @MainActor  // Ensure UI updates happen on main thread if needed
    private func testMPSMatMul(metalService: MetalService) {  // Pass service as argument
        // Update status on main thread
        self.modelLoaderWrapper.currentStatus = "Running MPS MatMul Test..."
        print("--- Running MPS MatMul Test ---")

        let device = metalService.device

        // --- Define Test Matrices (A: 2x3, B: 3x4) ---
        let M = 2
        let N = 4
        let K = 3

        let A_data: [Float16] = [
            1, 2, 3,  // Row 0
            4, 5, 6,
        ]  // Row 1 (Total 6 elements)

        // Let's store B row-major, so we'll likely need transposeB = true
        let B_data_rowMajor: [Float16] = [
            1, 0, 1, 0,  // Row 0
            0, 1, 0, 1,  // Row 1
            1, 1, 0, 0,
        ]  // Row 2 (Total 12 elements)

        let expected_C_data: [Float16] = [
            4, 5, 1, 2,  // Expected Row 0: (1*1+2*0+3*1), (1*0+2*1+3*1), (1*1+2*0+3*0), (1*0+2*1+3*0)
            10, 11, 4, 5,
        ]  // Expected Row 1: (4*1+5*0+6*1), (4*0+5*1+6*1), (4*1+5*0+6*0), (4*0+5*1+6*0)

        // --- Create Metal Buffers ---
        let sizeA = M * K * MemoryLayout<Float16>.stride
        let sizeB = K * N * MemoryLayout<Float16>.stride
        let sizeC = M * N * MemoryLayout<Float16>.stride

        // Use optional binding for safety
        guard
            let bufferA = device.makeBuffer(
                bytes: A_data, length: sizeA, options: .storageModeShared),
            let bufferB = device.makeBuffer(
                bytes: B_data_rowMajor, length: sizeB, options: .storageModeShared),
            let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared)
        else {
            print("MPS Test Error: Failed to create buffers.")
            self.modelLoaderWrapper.currentStatus = "MPS Test Error: Failed to create buffers."
            return
        }
        bufferA.label = "Test_MatrixA"
        bufferB.label = "Test_MatrixB_RowMajor"
        bufferC.label = "Test_MatrixC_Output"

        // --- Encode and Execute ---
        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
            print("MPS Test Error: Failed to create command buffer.")
            self.modelLoaderWrapper.currentStatus =
                "MPS Test Error: Failed to create command buffer."
            return
        }
        commandBuffer.label = "MPS MatMul Test CB"

        // Call the helper function - CRITICAL: Set transposeB = true because B_data is row-major
        // Inside testMPSMatMul in ContentView.swift

        // Call the helper function - Set transposeB = false
        let success = metalService.encodeMPSMatrixMultiply(
            commandBuffer: commandBuffer,
            inputA: bufferA,
            inputB: bufferB, // Still contains B_data_rowMajor (3x4)
            outputC: bufferC,
            M: M, N: N, K: K,
            transposeA: false,
            transposeB: false, // <--- CHANGE THIS TO FALSE
            label: "TestMatMul_2x3_3x4_tB_false" // Update label for clarity
        )

        guard success else {
            print("MPS Test Error: encodeMPSMatrixMultiply returned false.")
            self.modelLoaderWrapper.currentStatus =
                "MPS Test Error: Encoding failed (check console logs)."
            // Note: The helper function itself doesn't throw, check its print statements for clues.
            return
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()  // Wait for GPU execution

        // --- Verify Results ---
        var testResultMessage = ""  // Build result message
        if let error = commandBuffer.error {
            print("MPS Test Error: Command buffer execution failed: \(error)")
            testResultMessage =
                "MPS Test FAILED: Command buffer execution failed: \(error.localizedDescription)"
        } else {
            // Copy result back to CPU
            var result_C_data = [Float16](repeating: 0, count: M * N)
            let resultPtr = bufferC.contents().bindMemory(to: Float16.self, capacity: M * N)
            let sourceBuffer = UnsafeBufferPointer(start: resultPtr, count: M * N)  // Create a buffer pointer to the source data

            // Use the recommended UnsafeMutableBufferPointer.initialize(from:)
            result_C_data.withUnsafeMutableBufferPointer { destinationPointer in
                _ = destinationPointer.initialize(from: sourceBuffer)  // <-- CORRECTED LINE
            }

            // Compare (allowing for small floating point differences)
            let tolerance: Float16 = 0.01
            var mismatch = false
            for i in 0..<(M * N) {
                if abs(result_C_data[i] - expected_C_data[i]) > tolerance {
                    mismatch = true
                    print(
                        "Mismatch at index \(i): Got \(result_C_data[i]), Expected \(expected_C_data[i])"
                    )
                    break
                }
            }

            print(
                "MPS Test Result (Floats): \(result_C_data.map { String(format: "%.2f", Float($0)) })"
            )
            print(
                "MPS Test Expected:      \(expected_C_data.map { String(format: "%.2f", Float($0)) })"
            )

            if mismatch {
                testResultMessage =
                    "MPS Test FAILED: Results do not match expected values (check console)."
                print(testResultMessage)
            } else {
                testResultMessage = "MPS Test PASSED!"
                print(testResultMessage)
            }
        }

        print("--- MPS MatMul Test Complete ---")
        // Update status on main thread
        self.modelLoaderWrapper.currentStatus = testResultMessage
    }
}

#Preview {
    ContentView()
}
