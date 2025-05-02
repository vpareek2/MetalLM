//import Metal  // Needed for MTLBuffer in test function
//import MetalKit  // Potentially useful for test data generation
//import SwiftUI
//import UniformTypeIdentifiers  // For UTType
//
//struct ContentView: View {
//    // Use the updated wrapper
//    @StateObject private var modelLoaderWrapper = ModelLoaderWrapper()
//    
//    @State private var isLoading: Bool = false  // Keep for UI feedback
//    @State private var selectedFileURL: URL? = nil
//    
//    var body: some View {
//        VStack(alignment: .leading, spacing: 15) {
//            Text("MetaLLM Model Loader")
//                .font(.title)
//                .padding(.bottom, 5)
//            
//            // --- File Selection ---
//            HStack {
//                Button {
//                    selectGGUFFile()
//                } label: {
//                    Label("Select GGUF File...", systemImage: "folder.badge.plus")
//                }
//                Button("Clear") {
//                    selectedFileURL = nil
//                    modelLoaderWrapper.clearLoadedModel()
//                }
//                .disabled(selectedFileURL == nil && !modelLoaderWrapper.isMetadataLoaded)
//                Spacer()
//            }
//            
//            if let url = selectedFileURL {
//                Text("Selected: \(url.lastPathComponent)")
//                    .font(.caption).lineLimit(1).truncationMode(.middle)
//            } else {
//                Text("No file selected.").font(.caption)
//            }
//            Divider()
//            
//            // --- ADD Mul Test Button ---
//            Button {
//                if let service = modelLoaderWrapper.getMetalService() {
//                    testElementWiseMul(metalService: service)
//                } else {
//                    modelLoaderWrapper.currentStatus =
//                    "Error: Metal Service not available for test."
//                }
//            } label: {
//                Label("Run ElemWise Mul Test", systemImage: "multiply.square")
//            }
//            .padding(.bottom)
//            
//            Button {
//                if let service = modelLoaderWrapper.getMetalService() {
//                    testElementWiseAdd(metalService: service)
//                } else {
//                    modelLoaderWrapper.currentStatus =
//                    "Error: Metal Service not available for test."
//                }
//            } label: {
//                Label("Run ElemWise Add Test", systemImage: "plus.square")
//            }
//            .padding(.bottom)
//            
//            Button {
//                if let service = modelLoaderWrapper.getMetalService() {
//                    testRepeatKVHeads(metalService: service)
//                } else {
//                    modelLoaderWrapper.currentStatus =
//                    "Error: Metal Service not available for test."
//                }
//            } label: {
//                Label("Run GQA Repeat Test", systemImage: "rectangle.3.group")  // Icon suggestion
//            }
//            .padding(.bottom)
//            
//            // --- ADD SiLU Test Button ---
//            Button {
//                if let service = modelLoaderWrapper.getMetalService() {
//                    testSILU(metalService: service)  // Run the SiLU test
//                } else {
//                    modelLoaderWrapper.currentStatus =
//                    "Error: Metal Service not available for test."
//                }
//            } label: {
//                Label("Run SiLU Kernel Test", systemImage: "function")
//            }
//            .padding(.bottom)  // Add some spacing if needed
//            
//            // --- Load/Test Button for MatMul (Keep or comment out) ---
//            Button {
//                if let service = modelLoaderWrapper.getMetalService() {
//                    testMPSMatMul(metalService: service)
//                } else {
//                    modelLoaderWrapper.currentStatus =
//                    "Error: Metal Service not available for test."
//                }
//            } label: {
//                Label("Run MPS MatMul Test", systemImage: "function")
//            }
//            .padding(.vertical)
//            
//            Button {
//                if let service = modelLoaderWrapper.getMetalService() {
//                    testMPSSoftMax(metalService: service)
//                } else {
//                    modelLoaderWrapper.currentStatus =
//                    "Error: Metal Service not available for test."
//                }
//            } label: {
//                Label("Run MPS SoftMax Test", systemImage: "chart.bar")  // Icon suggestion
//            }
//            .padding(.bottom)
//            
//            Button {
//                if let service = modelLoaderWrapper.getMetalService() {
//                    testEncodeRMSNorm(metalService: service)  // Call the new test
//                } else {
//                    modelLoaderWrapper.currentStatus =
//                    "Error: Metal Service not available for test."
//                }
//            } label: {
//                Label("Run RMSNorm Encode Test", systemImage: "divide.square")  // Icon suggestion
//            }
//            .padding(.bottom)
//            
//            // --- ADD Forward Pass Test Button ---
//            Button {
//                // Ensure model is loaded before running forward pass
//                guard modelLoaderWrapper.getLlamaRunner() != nil else {
//                    modelLoaderWrapper.currentStatus =
//                    "Please load a model before running the forward pass test."
//                    return
//                }
//                testForwardPassNoAttention()  // Call the test
//            } label: {
//                Label("Run Forward Pass (No Attn)", systemImage: "forward.frame")
//            }
//            .padding(.bottom)
//            // Disable button if runner isn't ready
//            .disabled(modelLoaderWrapper.getLlamaRunner() == nil || isLoading)
//            
//            // Inside ContentView body
//            
//            // --- Update Forward Pass Test Button ---
//            Button {
//                guard modelLoaderWrapper.getLlamaRunner() != nil else { /* ... */ return }
//                testForwardPassAndSample()  // Call the UPDATED test function
//            } label: {
//                Label("Run Forward Pass + Sample", systemImage: "forward.fill")  // Updated label
//            }
//            .padding(.bottom)
//            .disabled(modelLoaderWrapper.getLlamaRunner() == nil || isLoading)
//            Button {
//                Task {
//                    await loadFullModelAndTestRope()
//                }
//            } label: {
//                Label("Load Full Model & Test RoPE", systemImage: "memorychip")
//            }
//            .disabled(selectedFileURL == nil || !modelLoaderWrapper.isMetadataLoaded || isLoading)
//            .padding(.vertical)
//            
//            Divider()
//            
//            // --- Status and Model Info ---
//            if isLoading {
//                ProgressView(
//                    modelLoaderWrapper.currentStatus.contains("Loading full model")
//                    ? "Loading Model..." : "Processing..."
//                )
//                .padding(.vertical)
//            }
//            
//            Text(modelLoaderWrapper.currentStatus)
//                .font(.footnote)
//                .lineLimit(nil)  // Allow multiple lines
//                .fixedSize(horizontal: false, vertical: true)  // Prevent truncation
//                .padding(.vertical)
//                .foregroundColor(
//                    modelLoaderWrapper.currentStatus.lowercased().contains("error")
//                    ? .red
//                    : (modelLoaderWrapper.currentStatus.lowercased().contains("success")
//                       ? .green : .secondary)
//                )
//            
//            if let config = modelLoaderWrapper.loadedModelConfig {
//                Divider()
//                Text("Loaded Model Config:")
//                    .font(.headline)
//                VStack(alignment: .leading, spacing: 2) {
//                    Text(
//                        String(
//                            format: "Layers: %d, Embed: %d, Hidden: %d", config.numLayers,
//                            config.embeddingDim, config.hiddenDim))
//                    Text(
//                        String(
//                            format: "Heads: %d, KV Heads: %d, Head Dim: %d", config.numHeads,
//                            config.numKeyValueHeads, config.headDim))
//                    Text(
//                        String(
//                            format: "Vocab: %d, Seq Len: %d", config.vocabSize,
//                            config.sequenceLength))
//                    Text(
//                        String(
//                            format: "RoPE Dim: %d, RoPE Freq Base: %.1f", config.ropeDimensionCount,
//                            config.ropeFreqBase))
//                }
//                .font(.system(.caption, design: .monospaced))
//            }
//            Spacer()
//        }
//        .padding()
//        .frame(minWidth: 500, minHeight: 350)
//    }
//    // Add this function inside ContentView or somewhere accessible
//    
//    // Inside ContentView.swift
//    
//    @MainActor
//    func validateEmbeddingRowCPU(
//        embeddingBuffer: MTLBuffer,
//        tokenID: Int,
//        embeddingDim: Int,
//        label: String
//    ) -> Bool {
//        print("--- Validating Embedding Row on CPU: \(label) (Token ID: \(tokenID))...")
//        
//        // Use Float16 size and type
//        let elementSize = MemoryLayout<Float16>.stride // 2 bytes
//        let rowSize = embeddingDim * elementSize
//        let sourceOffset = tokenID * rowSize
//        
//        // Basic Checks
//        guard embeddingDim > 0 else {
//            print("!!! Validation FAILED: embeddingDim is zero.")
//            return false
//        }
//        guard embeddingBuffer.length >= sourceOffset + rowSize else {
//            print("!!! Validation FAILED: Buffer too small (length: \(embeddingBuffer.length), needed: \(sourceOffset + rowSize)).")
//            return false
//        }
//        guard embeddingBuffer.storageMode == .shared || embeddingBuffer.storageMode == .managed else {
//            print("!!! Validation FAILED: Buffer is private, cannot access on CPU.")
//            return false
//        }
//        
//        // Read Data as Float16
//        let pointer = embeddingBuffer.contents().advanced(by: sourceOffset).bindMemory(to: Float16.self, capacity: embeddingDim)
//        let bufferPointer = UnsafeBufferPointer(start: pointer, count: embeddingDim)
//        
//        // Check for NaN/Inf
//        var isValid = true
//        var nanCount = 0
//        var infCount = 0
//        var firstNanIndex = -1
//        var firstInfIndex = -1
//        for i in 0..<embeddingDim {
//            let value = bufferPointer[i] // Reads Float16
//            if value.isNaN {
//                if firstNanIndex == -1 { firstNanIndex = i }
//                nanCount += 1
//                isValid = false
//            }
//            if value.isInfinite {
//                if firstInfIndex == -1 { firstInfIndex = i }
//                infCount += 1
//                isValid = false
//            }
//        }
//        
//        if !isValid {
//            print("!!! CPU Validation FAILED for embedding row '\(label)' (Token ID: \(tokenID)): \(nanCount) NaNs (first @ \(firstNanIndex)), \(infCount) Infs (first @ \(firstInfIndex)).")
//        } else {
//            print("--- CPU Validation PASSED for embedding row '\(label)' (Token ID: \(tokenID)).")
//        }
//        return isValid
//    }
//    
//    private func argmax(logits: [Float]) -> Int? { // <-- New function for Float
//        guard !logits.isEmpty else {
//            print("Warning: Argmax called with empty logits array.")
//            return nil
//        }
//        
//        var maxVal: Float = -Float.infinity
//        var maxIndex: Int = -1
//        
//        for (index, value) in logits.enumerated() {
//            // --- FIX: Check for NaN in Float argmax ---
//            if value.isNaN {
//                print("!!! Argmax Error: Found NaN at index \(index) !!!")
//                // Decide how to handle NaN: return nil, return a default, or continue?
//                // Let's return nil for safety.
//                return nil
//            }
//            // --- END FIX ---
//            if value > maxVal {
//                maxVal = value
//                maxIndex = index
//            }
//        }
//        
//        // Add a check for maxVal remaining -infinity if all inputs were -infinity or NaN
//        if maxIndex == -1 {
//            print("Warning: Argmax did not find a valid maximum value (all inputs might be -inf or NaN).")
//            return nil
//        }
//        
//        return maxIndex
//    }
//    
//    // File selection function remains the same
//    private func selectGGUFFile() {
//        let panel = NSOpenPanel()
//        panel.canChooseFiles = true
//        panel.canChooseDirectories = false
//        panel.allowsMultipleSelection = false
//        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]
//        
//        if panel.runModal() == .OK {
//            if let url = panel.url {
//                selectedFileURL = url
//                Task {
//                    isLoading = true
//                    // Load metadata when file is selected
//                    await modelLoaderWrapper.loadMetadata(url: url)
//                    isLoading = false
//                }
//            } else {
//                Task { @MainActor in
//                    selectedFileURL = nil
//                    modelLoaderWrapper.currentStatus = "Error: Could not get URL for selected file."
//                }
//            }
//        } else {
//            Task { @MainActor in
//                if selectedFileURL == nil {
//                    modelLoaderWrapper.currentStatus = "File selection cancelled."
//                }
//            }
//        }
//    }
//    
//    // Original function to load model AND run the RoPE test (Keep for reference or later use)
//    @MainActor
//    private func loadFullModelAndTestRope() async {
//        guard let urlToLoad = selectedFileURL else {
//            modelLoaderWrapper.currentStatus = "Error: No file selected to load."
//            return
//        }
//        guard modelLoaderWrapper.isMetadataLoaded, modelLoaderWrapper.loadedModelConfig != nil
//        else {
//            modelLoaderWrapper.currentStatus =
//            "Error: Metadata or Config not loaded successfully before loading full model."
//            return
//        }
//        
//        isLoading = true
//        modelLoaderWrapper.currentStatus = "Starting full model load..."  // Update status
//        
//        // Step 1: Assemble the full model
//        await modelLoaderWrapper.assembleFullModel(url: urlToLoad)
//        
//        // Step 2: Check if model loading succeeded and run RoPE test
//        if !modelLoaderWrapper.currentStatus.lowercased().contains("error"),
//           let service = modelLoaderWrapper.getMetalService(),
//           let loadedModel = modelLoaderWrapper.getLoadedModel()
//        {
//            modelLoaderWrapper.currentStatus += "\nRunning RoPE kernel sanity check..."
//            testRopeKernel(metalService: service, model: loadedModel)  // Call RoPE test
//        } else {
//            print("Skipping RoPE test due to model loading failure or missing components.")
//            modelLoaderWrapper.currentStatus += "\nSkipping RoPE test (load failed?)."
//        }
//        
//        isLoading = false
//    }
//    
//    // --- TEMPORARY RoPE TEST FUNCTION --- (Keep as is for now)
//    private func testRopeKernel(metalService: MetalService, model: LlamaModel) {
//        print("--- Running RoPE Sanity Check ---")
//        // --- COMMENTED OUT FOR NOW TO FOCUS ON MATMUL ---
//        /*
//         let config = model.config
//         
//         let testSeqLen = 4
//         let testNumHeads = 2
//         let testHeadDim = config.headDim
//         guard config.ropeDimensionCount <= testHeadDim else {
//         print("RoPE Test Error: ropeDimensionCount (\(config.ropeDimensionCount)) > headDim (\(testHeadDim))")
//         return
//         }
//         let elementCount = testSeqLen * testNumHeads * testHeadDim
//         let bufferSize = elementCount * MemoryLayout<Float16>.size
//         
//         guard bufferSize > 0 else {
//         print("RoPE Test Error: Invalid dimensions leading to zero buffer size.")
//         return
//         }
//         
//         var inputData = [Float16](repeating: 0, count: elementCount)
//         for i in 0..<elementCount { inputData[i] = Float16(i % 10) }
//         let originalData = inputData
//         
//         guard let testDataBuffer = metalService.device.makeBuffer(bytes: &inputData, length: bufferSize, options: .storageModeShared) else {
//         print("RoPE Test Error: Failed to create test data buffer.")
//         return
//         }
//         testDataBuffer.label = "RoPE Test Input/Output"
//         
//         guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
//         print("RoPE Test Error: Failed to create command buffer.")
//         return
//         }
//         commandBuffer.label = "RoPE Test Command Buffer"
//         
//         let success = metalService.applyRoPE(
//         commandBuffer: commandBuffer, buffer: testDataBuffer,
//         ropeFrequencies: model.ropeFrequencies,
//         config: config, posOffset: 0, sequenceLength: testSeqLen, numHeads: testNumHeads,
//         headDim: testHeadDim
//         )
//         
//         guard success else {
//         print("RoPE Test FAILED: applyRoPE function returned false.")
//         return
//         }
//         
//         commandBuffer.commit()
//         commandBuffer.waitUntilCompleted()
//         
//         if let error = commandBuffer.error {
//         print("RoPE Test FAILED: Command buffer execution failed: \(error)")
//         if let nsError = error as NSError? { print("  User Info: \(nsError.userInfo)") }
//         return
//         }
//         
//         let resultPtr = testDataBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
//         let resultData = Array(UnsafeBufferPointer(start: resultPtr, count: elementCount))
//         
//         // ... (rest of RoPE verification logic) ...
//         
//         print("--- RoPE Sanity Check Complete ---")
//         */
//        print("--- RoPE Sanity Check SKIPPED (Commented out in ContentView) ---")
//    }
//    
//    // --- ADDED ElementWise Mul TEST FUNCTION ---
//    @MainActor
//    private func testElementWiseMul(metalService: MetalService) {
//        self.modelLoaderWrapper.currentStatus = "Running ElementWise Mul Test..."
//        print("--- Running ElementWise Mul Test ---")
//        
//        let device = metalService.device
//        
//        // --- Define Test Data ---
//        let inputA: [Float16] = [1.0, 2.0, -3.0, 0.5, 10.0]
//        let inputB: [Float16] = [2.0, 0.5, 2.0, -4.0, 0.1]
//        let expectedOutput: [Float16] = [2.0, 1.0, -6.0, -2.0, 1.0]  // A * B
//        let elementCount = inputA.count
//        
//        // --- Create Metal Buffers ---
//        let bufferSize = elementCount * MemoryLayout<Float16>.stride
//        guard
//            let bufferA = device.makeBuffer(
//                bytes: inputA, length: bufferSize, options: .storageModeShared),
//            let bufferB = device.makeBuffer(
//                bytes: inputB, length: bufferSize, options: .storageModeShared),
//            let bufferC = device.makeBuffer(length: bufferSize, options: .storageModeShared)
//        else {
//            print("Mul Test Error: Failed to create buffers.")
//            self.modelLoaderWrapper.currentStatus = "Mul Test Error: Failed to create buffers."
//            return
//        }
//        bufferA.label = "Mul Test Input A"
//        bufferB.label = "Mul Test Input B"
//        bufferC.label = "Mul Test Output C"
//        
//        // --- Encode and Execute ---
//        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
//            print("Mul Test Error: Failed to create command buffer.")
//            self.modelLoaderWrapper.currentStatus =
//            "Mul Test Error: Failed to create command buffer."
//            return
//        }
//        commandBuffer.label = "Mul Test CB"
//        
//        let success = metalService.applyElementWiseMul(
//            inputBufferA: bufferA,
//            inputBufferB: bufferB,
//            outputBufferC: bufferC,
//            elementCount: elementCount,
//            commandBuffer: commandBuffer
//        )
//        
//        guard success else {
//            print("Mul Test Error: applyElementWiseMul returned false.")
//            self.modelLoaderWrapper.currentStatus = "Mul Test Error: Encoding failed."
//            return
//        }
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        // --- Verify Results ---
//        var testResultMessage = ""
//        if let error = commandBuffer.error {
//            print("Mul Test Error: Command buffer execution failed: \(error)")
//            testResultMessage =
//            "Mul Test FAILED: Command buffer execution failed: \(error.localizedDescription)"
//        } else {
//            var resultData = [Float16](repeating: 0, count: elementCount)
//            let resultPtr = bufferC.contents().bindMemory(to: Float16.self, capacity: elementCount)
//            let sourceBuffer = UnsafeBufferPointer(start: resultPtr, count: elementCount)
//            _ = resultData.withUnsafeMutableBufferPointer { $0.initialize(from: sourceBuffer) }
//            
//            let tolerance: Float16 = 0.01
//            var mismatch = false
//            for i in 0..<elementCount {
//                if abs(resultData[i] - expectedOutput[i]) > tolerance {
//                    mismatch = true
//                    print(
//                        "Mismatch at index \(i): Got \(resultData[i]), Expected \(expectedOutput[i])"
//                    )
//                }
//            }
//            
//            let resultStrings = resultData.map { String(format: "%.2f", Float($0)) }
//            let expectedStrings = expectedOutput.map { String(format: "%.2f", Float($0)) }
//            
//            print("Mul Test Result:   \(resultStrings)")
//            print("Mul Test Expected: \(expectedStrings)")
//            
//            if mismatch {
//                testResultMessage = "Mul Test FAILED: Results do not match expected values."
//            } else {
//                testResultMessage = "Mul Test PASSED!"
//            }
//            print(testResultMessage)
//        }
//        print("--- ElementWise Mul Test Complete ---")
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//    
//    @MainActor
//    private func testElementWiseAdd(metalService: MetalService) {
//        self.modelLoaderWrapper.currentStatus = "Running ElementWise Add Test..."
//        print("--- Running ElementWise Add Test ---")
//        
//        let device = metalService.device
//        
//        // --- Define Test Data ---
//        let inputA: [Float16] = [1.0, 2.0, -3.0, 0.5, 10.0, -5.0]
//        let inputB: [Float16] = [2.0, -0.5, 2.0, 4.0, -0.1, 5.0]
//        let expectedOutput: [Float16] = [3.0, 1.5, -1.0, 4.5, 9.9, 0.0]  // A + B
//        let elementCount = inputA.count
//        
//        // --- Create Metal Buffers ---
//        let bufferSize = elementCount * MemoryLayout<Float16>.stride
//        guard
//            let bufferA = device.makeBuffer(
//                bytes: inputA, length: bufferSize, options: .storageModeShared),
//            let bufferB = device.makeBuffer(
//                bytes: inputB, length: bufferSize, options: .storageModeShared),
//            let bufferC = device.makeBuffer(length: bufferSize, options: .storageModeShared)
//        else {
//            print("Add Test Error: Failed to create buffers.")
//            self.modelLoaderWrapper.currentStatus = "Add Test Error: Failed to create buffers."
//            return
//        }
//        bufferA.label = "Add Test Input A"
//        bufferB.label = "Add Test Input B"
//        bufferC.label = "Add Test Output C"
//        
//        // --- Encode and Execute ---
//        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
//            print("Add Test Error: Failed to create command buffer.")
//            self.modelLoaderWrapper.currentStatus =
//            "Add Test Error: Failed to create command buffer."
//            return
//        }
//        commandBuffer.label = "Add Test CB"
//        
//        let success = metalService.applyElementWiseAdd(
//            inputBufferA: bufferA,
//            inputBufferB: bufferB,
//            outputBufferC: bufferC,
//            elementCount: elementCount,
//            commandBuffer: commandBuffer
//        )
//        
//        guard success else {
//            print("Add Test Error: applyElementWiseAdd returned false.")
//            self.modelLoaderWrapper.currentStatus = "Add Test Error: Encoding failed."
//            return
//        }
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        // --- Verify Results ---
//        var testResultMessage = ""
//        if let error = commandBuffer.error {
//            print("Add Test Error: Command buffer execution failed: \(error)")
//            testResultMessage =
//            "Add Test FAILED: Command buffer execution failed: \(error.localizedDescription)"
//        } else {
//            var resultData = [Float16](repeating: 0, count: elementCount)
//            let resultPtr = bufferC.contents().bindMemory(to: Float16.self, capacity: elementCount)
//            let sourceBuffer = UnsafeBufferPointer(start: resultPtr, count: elementCount)
//            // Apply fix for warning
//            _ = resultData.withUnsafeMutableBufferPointer { $0.initialize(from: sourceBuffer) }
//            
//            let tolerance: Float16 = 0.01
//            var mismatch = false
//            for i in 0..<elementCount {
//                if abs(resultData[i] - expectedOutput[i]) > tolerance {
//                    mismatch = true
//                    print(
//                        "Mismatch at index \(i): Got \(resultData[i]), Expected \(expectedOutput[i])"
//                    )
//                }
//            }
//            
//            let resultStrings = resultData.map { String(format: "%.2f", Float($0)) }
//            let expectedStrings = expectedOutput.map { String(format: "%.2f", Float($0)) }
//            
//            print("Add Test Result:   \(resultStrings)")
//            print("Add Test Expected: \(expectedStrings)")
//            
//            if mismatch {
//                testResultMessage = "Add Test FAILED: Results do not match expected values."
//            } else {
//                testResultMessage = "Add Test PASSED!"
//            }
//            print(testResultMessage)
//        }
//        print("--- ElementWise Add Test Complete ---")
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//    
//    @MainActor
//    private func testRepeatKVHeads(metalService: MetalService) {
//        self.modelLoaderWrapper.currentStatus = "Running GQA Repeat KV Heads Test..."
//        print("--- Running GQA Repeat KV Heads Test ---")
//        
//        let device = metalService.device
//        
//        // --- Define Test Parameters ---
//        let seqLen = 2
//        let numKVHeads = 2  // Number of K/V heads in source
//        let headDim = 4  // Dimension of each head
//        let numQueryGroups = 3  // Each KV head is shared by 3 Query heads
//        let nHead = numKVHeads * numQueryGroups  // Total number of heads in destination (2 * 3 = 6)
//        
//        print(
//            "  Params: SeqLen=\(seqLen), KVHeads=\(numKVHeads), HeadDim=\(headDim), Groups=\(numQueryGroups), TotalHeads=\(nHead)"
//        )
//        
//        // --- Create Source Data ---
//        // Layout: [seqLen, numKVHeads, headDim]
//        // Example: s0_kv0_d0..3, s0_kv1_d0..3, s1_kv0_d0..3, s1_kv1_d0..3
//        var sourceData = [Float16]()
//        for s in 0..<seqLen {
//            for kvh in 0..<numKVHeads {
//                for d in 0..<headDim {
//                    // Create unique values like 1000*s + 100*kvh + d
//                    sourceData.append(Float16(1000 * s + 100 * kvh + d))
//                }
//            }
//        }
//        let sourceElementCount = sourceData.count
//        print("  Source Data (\(sourceElementCount) elements): \(sourceData.map { Float($0) })")
//        
//        // --- Calculate Expected Output Data ---
//        // Layout: [seqLen, nHead, headDim]
//        var expectedOutput = [Float16]()
//        for s in 0..<seqLen {
//            for h in 0..<nHead {  // Iterate through destination heads
//                let src_h = h / numQueryGroups  // Find the source head index
//                for d in 0..<headDim {
//                    // Get the corresponding value from the source head
//                    let srcValue = Float16(1000 * s + 100 * src_h + d)
//                    expectedOutput.append(srcValue)
//                }
//            }
//        }
//        let destElementCount = expectedOutput.count
//        print(
//            "  Expected Dest Data (\(destElementCount) elements): \(expectedOutput.map { Float($0) })"
//        )
//        
//        // --- Create Metal Buffers ---
//        let sourceBufferSize = sourceElementCount * MemoryLayout<Float16>.stride
//        let destBufferSize = destElementCount * MemoryLayout<Float16>.stride
//        
//        guard sourceBufferSize > 0, destBufferSize > 0 else {
//            print("GQA Repeat Test Error: Calculated buffer size is zero.")
//            self.modelLoaderWrapper.currentStatus = "GQA Repeat Test Error: Zero buffer size."
//            return
//        }
//        
//        guard
//            let sourceBuffer = device.makeBuffer(
//                bytes: sourceData, length: sourceBufferSize, options: .storageModeShared),
//            let destBuffer = device.makeBuffer(length: destBufferSize, options: .storageModeShared)
//        else {
//            print("GQA Repeat Test Error: Failed to create buffers.")
//            self.modelLoaderWrapper.currentStatus =
//            "GQA Repeat Test Error: Failed to create buffers."
//            return
//        }
//        sourceBuffer.label = "GQA Repeat Test Source"
//        destBuffer.label = "GQA Repeat Test Dest"
//        
//        // --- Encode and Execute ---
//        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
//            print("GQA Repeat Test Error: Failed to create command buffer.")
//            self.modelLoaderWrapper.currentStatus =
//            "GQA Repeat Test Error: Failed to create command buffer."
//            return
//        }
//        commandBuffer.label = "GQA Repeat Test CB"
//        
//        let success = metalService.applyRepeatKVHeads(
//            sourceBuffer: sourceBuffer,
//            destinationBuffer: destBuffer,
//            numKVHeads: numKVHeads,
//            numQueryGroups: numQueryGroups,
//            headDim: headDim,
//            seqLen: seqLen,
//            commandBuffer: commandBuffer
//        )
//        
//        guard success else {
//            print("GQA Repeat Test Error: applyRepeatKVHeads returned false.")
//            self.modelLoaderWrapper.currentStatus = "GQA Repeat Test Error: Encoding failed."
//            return
//        }
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        // --- Verify Results ---
//        var testResultMessage = ""
//        if let error = commandBuffer.error {
//            print("GQA Repeat Test Error: Command buffer execution failed: \(error)")
//            testResultMessage =
//            "GQA Repeat Test FAILED: Command buffer execution failed: \(error.localizedDescription)"
//        } else {
//            var resultData = [Float16](repeating: 0, count: destElementCount)
//            let resultPtr = destBuffer.contents().bindMemory(
//                to: Float16.self, capacity: destElementCount)
//            let sourceBufferPtr = UnsafeBufferPointer(start: resultPtr, count: destElementCount)
//            // Apply fix for warning
//            _ = resultData.withUnsafeMutableBufferPointer { $0.initialize(from: sourceBufferPtr) }
//            
//            let tolerance: Float16 = 0.01  // Exact copy, tolerance should be near zero
//            var mismatch = false
//            for i in 0..<destElementCount {
//                if abs(resultData[i] - expectedOutput[i]) > tolerance {
//                    mismatch = true
//                    print(
//                        "Mismatch at index \(i): Got \(resultData[i]), Expected \(expectedOutput[i])"
//                    )
//                }
//            }
//            
//            let resultStrings = resultData.map { String(format: "%.0f", Float($0)) }  // Use %.0f for integer-like values
//            let expectedStrings = expectedOutput.map { String(format: "%.0f", Float($0)) }
//            
//            // Print smaller chunks if output is large
//            let printLimit = 64
//            print("GQA Repeat Test Result:   \(resultStrings.prefix(printLimit))...")
//            print("GQA Repeat Test Expected: \(expectedStrings.prefix(printLimit))...")
//            
//            if mismatch {
//                testResultMessage = "GQA Repeat Test FAILED: Results do not match expected values."
//            } else {
//                testResultMessage = "GQA Repeat Test PASSED!"
//            }
//            print(testResultMessage)
//        }
//        print("--- GQA Repeat KV Heads Test Complete ---")
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//    
//    // --- ADDED SiLU TEST FUNCTION ---
//    @MainActor
//    private func testSILU(metalService: MetalService) {
//        self.modelLoaderWrapper.currentStatus = "Running SiLU Kernel Test..."
//        print("--- Running SiLU Kernel Test ---")
//        
//        let device = metalService.device
//        
//        // --- Define Test Data ---
//        let inputData: [Float16] = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
//        let elementCount = inputData.count
//        
//        // Calculate expected output using Float for precision
//        let expectedOutput: [Float16] = inputData.map { x_h in
//            let x = Float(x_h)
//            let sigmoid_x = 1.0 / (1.0 + exp(-x))
//            return Float16(x * sigmoid_x)
//        }
//        
//        // --- Create Metal Buffers ---
//        let bufferSize = elementCount * MemoryLayout<Float16>.stride
//        guard
//            let inputBuffer = device.makeBuffer(
//                bytes: inputData, length: bufferSize, options: .storageModeShared),
//            let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
//        else {
//            print("SiLU Test Error: Failed to create buffers.")
//            self.modelLoaderWrapper.currentStatus = "SiLU Test Error: Failed to create buffers."
//            return
//        }
//        inputBuffer.label = "SiLU Test Input"
//        outputBuffer.label = "SiLU Test Output"
//        
//        // --- Encode and Execute ---
//        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
//            print("SiLU Test Error: Failed to create command buffer.")
//            self.modelLoaderWrapper.currentStatus =
//            "SiLU Test Error: Failed to create command buffer."
//            return
//        }
//        commandBuffer.label = "SiLU Test CB"
//        
//        // Call the dispatch function
//        let success = metalService.applySILU(
//            inputBuffer: inputBuffer,
//            outputBuffer: outputBuffer,
//            elementCount: elementCount,
//            commandBuffer: commandBuffer
//        )
//        
//        guard success else {
//            print("SiLU Test Error: applySILU returned false.")
//            self.modelLoaderWrapper.currentStatus =
//            "SiLU Test Error: Encoding failed (check console logs)."
//            return
//        }
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        // --- Verify Results ---
//        var testResultMessage = ""
//        if let error = commandBuffer.error {
//            print("SiLU Test Error: Command buffer execution failed: \(error)")
//            testResultMessage =
//            "SiLU Test FAILED: Command buffer execution failed: \(error.localizedDescription)"
//        } else {
//            // Copy result back to CPU
//            var resultData = [Float16](repeating: 0, count: elementCount)
//            let resultPtr = outputBuffer.contents().bindMemory(
//                to: Float16.self, capacity: elementCount)
//            let sourceBuffer = UnsafeBufferPointer(start: resultPtr, count: elementCount)
//            resultData.withUnsafeMutableBufferPointer { destinationPointer in
//                _ = destinationPointer.initialize(from: sourceBuffer)
//            }
//            
//            // Compare
//            let tolerance: Float16 = 0.01  // Adjust tolerance as needed for f16
//            var mismatch = false
//            for i in 0..<elementCount {
//                if abs(resultData[i] - expectedOutput[i]) > tolerance {
//                    mismatch = true
//                    print(
//                        "Mismatch at index \(i): Got \(resultData[i]), Expected \(expectedOutput[i])"
//                    )
//                    // break // Uncomment to stop at first mismatch
//                }
//            }
//            
//            // Use String(format:) for cleaner float printing
//            let resultStrings = resultData.map { String(format: "%.4f", Float($0)) }
//            let expectedStrings = expectedOutput.map { String(format: "%.4f", Float($0)) }
//            
//            print("SiLU Test Result:   \(resultStrings)")
//            print("SiLU Test Expected: \(expectedStrings)")
//            
//            if mismatch {
//                testResultMessage =
//                "SiLU Test FAILED: Results do not match expected values (check console)."
//                print(testResultMessage)
//            } else {
//                testResultMessage = "SiLU Test PASSED!"
//                print(testResultMessage)
//            }
//        }
//        print("--- SiLU Kernel Test Complete ---")
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//    
//    // --- ADDED MPS MatMul TEST FUNCTION ---
//    // Example Test Function (Place appropriately, e.g., in ContentView for quick testing)
//    @MainActor  // Ensure UI updates happen on main thread if needed
//    private func testMPSMatMul(metalService: MetalService) {  // Pass service as argument
//        // Update status on main thread
//        self.modelLoaderWrapper.currentStatus = "Running MPS MatMul Test..."
//        print("--- Running MPS MatMul Test ---")
//        
//        let device = metalService.device
//        
//        // --- Define Test Matrices (A: 2x3, B: 3x4) ---
//        let M = 2
//        let N = 4
//        let K = 3
//        
//        let A_data: [Float16] = [
//            1, 2, 3,  // Row 0
//            4, 5, 6,
//        ]  // Row 1 (Total 6 elements)
//        
//        // Let's store B row-major, so we'll likely need transposeB = true
//        let B_data_rowMajor: [Float16] = [
//            1, 0, 1, 0,  // Row 0
//            0, 1, 0, 1,  // Row 1
//            1, 1, 0, 0,
//        ]  // Row 2 (Total 12 elements)
//        
//        let expected_C_data: [Float16] = [
//            4, 5, 1, 2,  // Expected Row 0: (1*1+2*0+3*1), (1*0+2*1+3*1), (1*1+2*0+3*0), (1*0+2*1+3*0)
//            10, 11, 4, 5,
//        ]  // Expected Row 1: (4*1+5*0+6*1), (4*0+5*1+6*1), (4*1+5*0+6*0), (4*0+5*1+6*0)
//        
//        // --- Create Metal Buffers ---
//        let sizeA = M * K * MemoryLayout<Float16>.stride
//        let sizeB = K * N * MemoryLayout<Float16>.stride
//        let sizeC = M * N * MemoryLayout<Float16>.stride
//        
//        // Use optional binding for safety
//        guard
//            let bufferA = device.makeBuffer(
//                bytes: A_data, length: sizeA, options: .storageModeShared),
//            let bufferB = device.makeBuffer(
//                bytes: B_data_rowMajor, length: sizeB, options: .storageModeShared),
//            let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared)
//        else {
//            print("MPS Test Error: Failed to create buffers.")
//            self.modelLoaderWrapper.currentStatus = "MPS Test Error: Failed to create buffers."
//            return
//        }
//        bufferA.label = "Test_MatrixA"
//        bufferB.label = "Test_MatrixB_RowMajor"
//        bufferC.label = "Test_MatrixC_Output"
//        
//        // --- Encode and Execute ---
//        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
//            print("MPS Test Error: Failed to create command buffer.")
//            self.modelLoaderWrapper.currentStatus =
//            "MPS Test Error: Failed to create command buffer."
//            return
//        }
//        commandBuffer.label = "MPS MatMul Test CB"
//        
//        // --- FIX: Update the call to use the new signature ---
//        let success = metalService.encodeMPSMatrixMultiply(
//            commandBuffer: commandBuffer,
//            inputA: bufferA,  // Buffer A
//            inputB: bufferB,  // Buffer B (row-major 3x4)
//            outputC: bufferC,  // Buffer C
//            // Actual dimensions of data in buffers:
//            rowsA: M, colsA: K,  // bufferA is M rows, K columns (2x3)
//            rowsB: K, colsB: N,  // bufferB is K rows, N columns (3x4)
//            // Mathematical transpose flags for A * B:
//            transposeA: false,
//            transposeB: false,  // Keep false as determined by previous successful test
//            alpha: 1.0,  // Optional, default
//            beta: 0.0,  // Optional, default
//            label: "TestMatMul_2x3_3x4_tB_false"
//        )
//        // --- END FIX ---
//        
//        guard success else {
//            print("MPS Test Error: encodeMPSMatrixMultiply returned false.")
//            self.modelLoaderWrapper.currentStatus =
//            "MPS Test Error: Encoding failed (check console logs)."
//            // Note: The helper function itself doesn't throw, check its print statements for clues.
//            return
//        }
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()  // Wait for GPU execution
//        
//        // --- Verify Results ---
//        var testResultMessage = ""  // Build result message
//        if let error = commandBuffer.error {
//            print("MPS Test Error: Command buffer execution failed: \(error)")
//            testResultMessage =
//            "MPS Test FAILED: Command buffer execution failed: \(error.localizedDescription)"
//        } else {
//            // Copy result back to CPU
//            var result_C_data = [Float16](repeating: 0, count: M * N)
//            let resultPtr = bufferC.contents().bindMemory(to: Float16.self, capacity: M * N)
//            let sourceBuffer = UnsafeBufferPointer(start: resultPtr, count: M * N)  // Create a buffer pointer to the source data
//            
//            // Use the recommended UnsafeMutableBufferPointer.initialize(from:)
//            result_C_data.withUnsafeMutableBufferPointer { destinationPointer in
//                _ = destinationPointer.initialize(from: sourceBuffer)  // <-- CORRECTED LINE
//            }
//            
//            // Compare (allowing for small floating point differences)
//            let tolerance: Float16 = 0.01
//            var mismatch = false
//            for i in 0..<(M * N) {
//                if abs(result_C_data[i] - expected_C_data[i]) > tolerance {
//                    mismatch = true
//                    print(
//                        "Mismatch at index \(i): Got \(result_C_data[i]), Expected \(expected_C_data[i])"
//                    )
//                    break
//                }
//            }
//            
//            print(
//                "MPS Test Result (Floats): \(result_C_data.map { String(format: "%.2f", Float($0)) })"
//            )
//            print(
//                "MPS Test Expected:      \(expected_C_data.map { String(format: "%.2f", Float($0)) })"
//            )
//            
//            if mismatch {
//                testResultMessage =
//                "MPS Test FAILED: Results do not match expected values (check console)."
//                print(testResultMessage)
//            } else {
//                testResultMessage = "MPS Test PASSED!"
//                print(testResultMessage)
//            }
//        }
//        
//        print("--- MPS MatMul Test Complete ---")
//        // Update status on main thread
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//    
//    // Inside ContentView struct
//    
//    // --- ADDED RMSNorm Encode TEST FUNCTION ---
//    @MainActor
//    private func testEncodeRMSNorm(metalService: MetalService) {
//        self.modelLoaderWrapper.currentStatus = "Running RMSNorm Encode Test..."
//        print("--- Running RMSNorm Encode Test ---")
//        
//        let device = metalService.device
//        
//        // --- Define Test Parameters ---
//        let rowCount = 2
//        let elementCountPerRow = 8  // Keep it small for easy verification
//        let eps: Float = 1e-5
//        
//        // --- Create Test Data ---
//        // Row 0: Some positive/negative values
//        // Row 1: All same value
//        let inputData: [Float16] = [
//            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0,
//            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
//        ]
//        // Simple weights (gamma) = 1.0 for easy verification of normalization
//        let weightData: [Float16] = Array(repeating: 1.0, count: elementCountPerRow)
//        
//        let elementCountTotal = rowCount * elementCountPerRow
//        
//        // --- Calculate Expected Output ---
//        var expectedOutput = [Float16](repeating: 0, count: elementCountTotal)
//        for r in 0..<rowCount {
//            let rowOffset = r * elementCountPerRow
//            var sumSq: Float = 0.0
//            // Calculate sum of squares for the row
//            for c in 0..<elementCountPerRow {
//                let val = Float(inputData[rowOffset + c])
//                sumSq += val * val
//            }
//            // Calculate RMS scale factor
//            let meanSq = sumSq / Float(elementCountPerRow)
//            let rms = sqrt(meanSq + eps)
//            let scale = 1.0 / rms
//            // Apply normalization and weight (which is 1.0)
//            for c in 0..<elementCountPerRow {
//                let normalizedVal = Float(inputData[rowOffset + c]) * scale * Float(weightData[c])
//                expectedOutput[rowOffset + c] = Float16(normalizedVal)
//            }
//        }
//        
//        // --- Create Metal Buffers ---
//        let inputBufferSize = elementCountTotal * MemoryLayout<Float16>.stride
//        let weightBufferSize = elementCountPerRow * MemoryLayout<Float16>.stride
//        let outputBufferSize = elementCountTotal * MemoryLayout<Float16>.stride
//        
//        guard
//            let inputBuffer = device.makeBuffer(
//                bytes: inputData, length: inputBufferSize, options: .storageModeShared),
//            let weightBuffer = device.makeBuffer(
//                bytes: weightData, length: weightBufferSize, options: .storageModeShared),
//            let outputBuffer = device.makeBuffer(
//                length: outputBufferSize, options: .storageModeShared)
//        else {
//            print("RMSNorm Test Error: Failed to create buffers.")
//            self.modelLoaderWrapper.currentStatus = "RMSNorm Test Error: Failed to create buffers."
//            return
//        }
//        inputBuffer.label = "RMSNorm Test Input"
//        weightBuffer.label = "RMSNorm Test Weights"
//        outputBuffer.label = "RMSNorm Test Output"
//        
//        // --- Get Command Buffer ---
//        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
//            print("RMSNorm Test Error: Failed to create command buffer.")
//            self.modelLoaderWrapper.currentStatus =
//            "RMSNorm Test Error: Failed to create command buffer."
//            return
//        }
//        commandBuffer.label = "RMSNorm Test CB"
//        
//        // --- Encode the RMSNorm Kernel ---
//        let success = metalService.encodeRMSNormF16(
//            commandBuffer: commandBuffer,  // Pass the command buffer
//            inputBuffer: inputBuffer,
//            weightBuffer: weightBuffer,
//            outputBuffer: outputBuffer,
//            rowCount: rowCount,
//            elementCountPerRow: elementCountPerRow,
//            eps: eps,
//            label: "TestRMSNormEncode"
//        )
//        
//        guard success else {
//            print("RMSNorm Test Error: encodeRMSNormF16 returned false.")
//            self.modelLoaderWrapper.currentStatus = "RMSNorm Test Error: Encoding failed."
//            // No need to commit if encoding failed
//            return
//        }
//        
//        // --- Commit and Wait --- THIS IS NOW DONE IN THE TEST
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        // --- Verify Results ---
//        var testResultMessage = ""
//        if let error = commandBuffer.error {
//            print("RMSNorm Test Error: Command buffer execution failed: \(error)")
//            testResultMessage =
//            "RMSNorm Test FAILED: Command buffer execution failed: \(error.localizedDescription)"
//        } else {
//            var resultData = [Float16](repeating: 0, count: elementCountTotal)
//            let resultPtr = outputBuffer.contents().bindMemory(
//                to: Float16.self, capacity: elementCountTotal)
//            let sourceBufferPtr = UnsafeBufferPointer(start: resultPtr, count: elementCountTotal)
//            // Apply fix for warning
//            _ = resultData.withUnsafeMutableBufferPointer { $0.initialize(from: sourceBufferPtr) }
//            
//            let tolerance: Float16 = 0.01
//            var mismatch = false
//            print("RMSNorm Test Verification:")
//            for i in 0..<elementCountTotal {
//                if abs(resultData[i] - expectedOutput[i]) > tolerance {
//                    mismatch = true
//                    print(
//                        "  Mismatch at index \(i): Got \(resultData[i]), Expected \(expectedOutput[i])"
//                    )
//                }
//            }
//            
//            let resultStrings = resultData.map { String(format: "%.3f", Float($0)) }
//            let expectedStrings = expectedOutput.map { String(format: "%.3f", Float($0)) }
//            
//            print("RMSNorm Test Result:   \(resultStrings)")
//            print("RMSNorm Test Expected: \(expectedStrings)")
//            
//            if mismatch {
//                testResultMessage = "RMSNorm Test FAILED: Results do not match expected values."
//            } else {
//                testResultMessage = "RMSNorm Test PASSED!"
//            }
//            print(testResultMessage)
//        }
//        print("--- RMSNorm Encode Test Complete ---")
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//    // Inside ContentView struct
//    
//    // --- ADDED MPS SoftMax TEST FUNCTION ---
//    @MainActor
//    private func testMPSSoftMax(metalService: MetalService) {
//        self.modelLoaderWrapper.currentStatus = "Running MPS SoftMax Test..."
//        print("--- Running MPS SoftMax Test ---")
//        
//        let device = metalService.device
//        
//        // --- Define Test Data (e.g., 2 rows, 4 columns) ---
//        let rows = 2
//        let columns = 4
//        let inputData: [Float16] = [
//            1.0, 2.0, 3.0, 4.0,  // Row 0 (logits)
//            -1.0, 0.0, 1.0, 0.5,
//        ]  // Row 1 (logits)
//        let elementCount = rows * columns
//        
//        // Calculate expected output manually (using Float for intermediate precision)
//        var expectedOutput = [Float16](repeating: 0, count: elementCount)
//        for r in 0..<rows {
//            let rowOffset = r * columns
//            let inputRow = (0..<columns).map { Float(inputData[rowOffset + $0]) }
//            
//            // Find max for numerical stability
//            let maxVal = inputRow.max() ?? 0.0
//            // Calculate exp and sum
//            let exps = inputRow.map { exp($0 - maxVal) }
//            let sumExps = exps.reduce(0, +)
//            // Calculate softmax
//            for c in 0..<columns {
//                expectedOutput[rowOffset + c] = Float16(exps[c] / sumExps)
//            }
//        }
//        
//        // --- Create Metal Buffers ---
//        let bufferSize = elementCount * MemoryLayout<Float16>.stride
//        guard
//            let inputBuffer = device.makeBuffer(
//                bytes: inputData, length: bufferSize, options: .storageModeShared),
//            let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
//        else {
//            print("SoftMax Test Error: Failed to create buffers.")
//            self.modelLoaderWrapper.currentStatus = "SoftMax Test Error: Failed to create buffers."
//            return
//        }
//        inputBuffer.label = "SoftMax Test Input"
//        outputBuffer.label = "SoftMax Test Output"
//        
//        // --- Encode and Execute ---
//        guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
//            print("SoftMax Test Error: Failed to create command buffer.")
//            self.modelLoaderWrapper.currentStatus =
//            "SoftMax Test Error: Failed to create command buffer."
//            return
//        }
//        commandBuffer.label = "SoftMax Test CB"
//        
//        let success = metalService.encodeMPSSoftMax(
//            commandBuffer: commandBuffer,
//            inputMatrixBuffer: inputBuffer,
//            outputMatrixBuffer: outputBuffer,
//            rows: rows,
//            columns: columns
//        )
//        
//        guard success else {
//            print("SoftMax Test Error: encodeMPSSoftMax returned false.")
//            self.modelLoaderWrapper.currentStatus = "SoftMax Test Error: Encoding failed."
//            return
//        }
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        // --- Verify Results ---
//        var testResultMessage = ""
//        if let error = commandBuffer.error {
//            print("SoftMax Test Error: Command buffer execution failed: \(error)")
//            testResultMessage =
//            "SoftMax Test FAILED: Command buffer execution failed: \(error.localizedDescription)"
//        } else {
//            var resultData = [Float16](repeating: 0, count: elementCount)
//            let resultPtr = outputBuffer.contents().bindMemory(
//                to: Float16.self, capacity: elementCount)
//            let sourceBufferPtr = UnsafeBufferPointer(start: resultPtr, count: elementCount)
//            // Apply fix for warning
//            _ = resultData.withUnsafeMutableBufferPointer { $0.initialize(from: sourceBufferPtr) }
//            
//            let tolerance: Float16 = 0.01  // Softmax involves exp, might need slightly larger tolerance
//            var mismatch = false
//            var rowSums: [Float] = Array(repeating: 0.0, count: rows)
//            
//            print("SoftMax Test Verification:")
//            for r in 0..<rows {
//                var currentRowSum: Float = 0.0
//                print("  Row \(r):")
//                for c in 0..<columns {
//                    let index = r * columns + c
//                    let resultVal = resultData[index]
//                    let expectedVal = expectedOutput[index]
//                    currentRowSum += Float(resultVal)  // Sum results as Float
//                    
//                    if abs(resultVal - expectedVal) > tolerance {
//                        mismatch = true
//                        print(
//                            "    Mismatch at [\(r),\(c)]: Got \(resultVal), Expected \(expectedVal)"
//                        )
//                    }
//                    // Check if value is valid probability (0 to 1)
//                    if !(resultVal >= 0.0 && resultVal <= 1.0) {
//                        mismatch = true
//                        print("    Invalid probability at [\(r),\(c)]: Got \(resultVal)")
//                    }
//                }
//                rowSums[r] = currentRowSum
//                print("    Row Sum: \(currentRowSum)")
//                // Check if row sum is close to 1.0
//                if abs(currentRowSum - 1.0) > Float(tolerance * Float16(columns)) {  // Allow larger tolerance for sum
//                    mismatch = true
//                    print("    Row Sum deviates significantly from 1.0!")
//                }
//            }
//            
//            let resultStrings = resultData.map { String(format: "%.4f", Float($0)) }
//            let expectedStrings = expectedOutput.map { String(format: "%.4f", Float($0)) }
//            
//            print("SoftMax Test Result:   \(resultStrings)")
//            print("SoftMax Test Expected: \(expectedStrings)")
//            
//            if mismatch {
//                testResultMessage = "SoftMax Test FAILED: Verification failed (check console)."
//            } else {
//                testResultMessage = "SoftMax Test PASSED!"
//            }
//            print(testResultMessage)
//        }
//        print("--- MPS SoftMax Test Complete ---")
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//    
//    // Inside ContentView struct
//    
//    // --- ADDED Forward Pass (No Attention) TEST FUNCTION ---
//    @MainActor
//    private func testForwardPassNoAttention() {
//        self.modelLoaderWrapper.currentStatus = "Running Forward Pass (No Attention) Test..."
//        print("--- Running Forward Pass (No Attention) Test ---")
//        
//        // 1. Get Runner Instance
//        guard let runner = modelLoaderWrapper.getLlamaRunner() else {
//            let msg = "Forward Pass Test Error: LlamaRunner not initialized. Load model first."
//            print(msg)
//            self.modelLoaderWrapper.currentStatus = msg
//            // Optionally trigger model load here if desired for the test button
//            // Task { await loadFullModelAndTestRope() }
//            return
//        }
//        
//        // 2. Reset State
//        runner.resetState()
//        guard runner.currentPosition == 0 else {
//            let msg = "Forward Pass Test Error: Runner state did not reset correctly."
//            print(msg)
//            self.modelLoaderWrapper.currentStatus = msg
//            return
//        }
//        
//        // 3. Define Starting Token (Use a common BOS token ID - check your model's tokenizer.json if unsure)
//        // Common Llama BOS IDs are 1 or 128000. Let's assume 1 for now.
//        let bosTokenID = 1
//        // Alternative for Llama 3: 128000
//        // let bosTokenID = 128000
//        print("  Using BOS Token ID: \(bosTokenID)")
//        
//        // 4. Execute Forward Pass
//        // Use Metal Frame Capture DURING this call if debugging!
//        print("  Calling runner.forward(tokenID: \(bosTokenID))...")
//        let startTime = CFAbsoluteTimeGetCurrent()
//        
//        let logitsBuffer = runner.forward(tokenID: bosTokenID)
//        
//        let endTime = CFAbsoluteTimeGetCurrent()
//        let duration = String(format: "%.3f", endTime - startTime)
//        print("  runner.forward call completed in \(duration) seconds.")
//        
//        // 5. Verify Basic Results
//        var testResultMessage = ""
//        if let returnedLogits = logitsBuffer {
//            print("  Forward pass returned a logits buffer.")
//            // Basic validation on buffer
//            // --- FIX: Expect F16 size to match allocation ---
//            let expectedLogitsSize = runner.config.vocabSize * MemoryLayout<Float16>.stride
//            // --- END FIX ---
//            
//            if returnedLogits.length >= expectedLogitsSize {
//                print(
//                    "  Logits buffer has expected size (or larger): \(returnedLogits.length) bytes (Expected >= \(expectedLogitsSize))."
//                )
//                // Check position increment
//                if runner.currentPosition == 1 {
//                    testResultMessage =
//                    "Forward Pass (No Attention) Test: PASSED (Executed without errors, position updated)."
//                    print("  Runner position correctly updated to \(runner.currentPosition).")
//                } else {
//                    testResultMessage =
//                    "Forward Pass (No Attention) Test: FAILED (Position not updated correctly, is \(runner.currentPosition))."
//                }
//            } else {
//                testResultMessage =
//                "Forward Pass (No Attention) Test: FAILED (Logits buffer size mismatch. Expected >= \(expectedLogitsSize), Got \(returnedLogits.length))."
//            }
//            // TODO: Later, add sampling and check if the *next* token makes sense.
//            // For now, just checking execution success.
//            
//        } else {
//            testResultMessage =
//            "Forward Pass (No Attention) Test: FAILED (Returned nil logits buffer - check console for errors)."
//        }
//        
//        print(testResultMessage)
//        print("--- Forward Pass (No Attention) Test Complete ---")
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//    // Inside ContentView struct
//    
//    // Inside ContentView.swift
//    
//    @MainActor
//    private func testForwardPassAndSample() {
//        self.modelLoaderWrapper.currentStatus = "Running Forward Pass + Sample Test..."
//        print("--- Running Forward Pass + Sample Test ---")
//        
//        guard let runner = modelLoaderWrapper.getLlamaRunner(),
//              let loadedModel = modelLoaderWrapper.getLoadedModel()
//        else {
//            let msg = "Forward Pass Test Error: Runner, Service, or Model not initialized. Load model first."
//            print(msg)
//            self.modelLoaderWrapper.currentStatus = msg
//            return
//        }
//        
//        // Reset State
//        runner.resetState()
//        guard runner.currentPosition == 0 else {
//            let msg = "Forward Pass Test Error: Runner position not reset."
//            print(msg)
//            self.modelLoaderWrapper.currentStatus = msg
//            return
//        }
//        
//        // Define Starting Token
//        let bosTokenID = 1
//        print("  Using BOS Token ID: \(bosTokenID)")
//        
//        // Validate Source Embedding
//        let embeddingDim = loadedModel.config.embeddingDim
//        let isEmbeddingRowValid = validateEmbeddingRowCPU(
//            embeddingBuffer: loadedModel.tokenEmbeddings,
//            tokenID: bosTokenID,
//            embeddingDim: embeddingDim,
//            label: "token_embd.weight"
//        )
//        guard isEmbeddingRowValid else {
//            let msg = "Forward Pass Test Error: Source embedding data for token \(bosTokenID) contains NaN/Inf."
//            print(msg)
//            self.modelLoaderWrapper.currentStatus = msg
//            return
//        }
//        print("  Source embedding row for token \(bosTokenID) validated successfully.")
//        
//        // Execute Forward Pass
//        print("  Calling runner.forward(tokenID: \(bosTokenID))...")
//        let startTime = CFAbsoluteTimeGetCurrent()
//        let logitsBuffer = runner.forward(tokenID: bosTokenID)
//        let endTime = CFAbsoluteTimeGetCurrent()
//        let duration = String(format: "%.3f", endTime - startTime)
//        
//        print("  runner.forward call completed in \(duration) seconds.")
//        
//        // Verify Execution and Sample
//        var testResultMessage = ""
//        if let returnedLogits = logitsBuffer {
//            print("  Forward pass returned a logits buffer.")
//            let vocabSize = runner.config.vocabSize
//            let logitsCount = vocabSize
//            let logitsSizeBytes = logitsCount * MemoryLayout<Float>.stride
//            
//            if returnedLogits.length >= logitsSizeBytes {
//                print("  Logits buffer has expected size (or larger).")
//                
//                // Copy Logits to CPU
//                print("  Copying logits to CPU...")
//                var cpuLogits = [Float](repeating: 0, count: logitsCount)
//                if returnedLogits.storageMode != .private {
//                    if returnedLogits.storageMode == .managed {
//                        if let metalService = modelLoaderWrapper.getMetalService(),
//                           let syncCommandBuffer = metalService.commandQueue.makeCommandBuffer(),
//                           let blitEncoder = syncCommandBuffer.makeBlitCommandEncoder() {
//                            blitEncoder.label = "ContentView Logits Sync Blit"
//                            blitEncoder.synchronize(resource: returnedLogits)
//                            blitEncoder.endEncoding()
//                            syncCommandBuffer.commit()
//                            syncCommandBuffer.waitUntilCompleted()
//                            if let error = syncCommandBuffer.error {
//                                print("   !!! Logits sync command buffer failed in ContentView: \(error) !!!")
//                            } else {
//                                print("   Logits sync complete in ContentView.")
//                            }
//                        } else {
//                            print("   !!! Failed to create command buffer/encoder for logits sync in ContentView !!!")
//                        }
//                    } else {
//                        print("   Logits buffer is not Managed, skipping explicit synchronize in ContentView.")
//                    }
//                    
//                    let logitsPtr = returnedLogits.contents().bindMemory(to: Float.self, capacity: logitsCount)
//                    let sourceBufferPtr = UnsafeBufferPointer(start: logitsPtr, count: logitsCount)
//                    _ = cpuLogits.withUnsafeMutableBufferPointer { $0.initialize(from: sourceBufferPtr) }
//                    print("  Successfully copied logits to CPU (\(logitsCount) elements).")
//                    
//                    // Debug Logs
//                    let printLimit = 20
//                    print("  CPU Logits (First \(printLimit)): \(cpuLogits.prefix(printLimit))")
//                    print("  CPU Logits (Last \(printLimit)): \(cpuLogits.suffix(printLimit))")
//                    let nanCount = cpuLogits.filter { $0.isNaN }.count
//                    let infCount = cpuLogits.filter { $0.isInfinite }.count
//                    if nanCount > 0 || infCount > 0 {
//                        print("  WARNING: Copied CPU Logits contain \(nanCount) NaNs and \(infCount) Infinities!")
//                    }
//                    
//                    // Argmax Sampling
//                    if let nextTokenID = argmax(logits: cpuLogits) {
//                        print("  Sampled Next Token ID (Argmax): \(nextTokenID)")
//                        if nextTokenID >= 0 && nextTokenID < vocabSize {
//                            if runner.currentPosition == 1 {
//                                testResultMessage = "Forward Pass + Sample Test: PASSED (Executed, pos=\(runner.currentPosition), sampled ID: \(nextTokenID))."
//                            } else {
//                                testResultMessage = "Forward Pass + Sample Test: FAILED (Position mismatch)"
//                            }
//                        } else {
//                            testResultMessage = "Forward Pass + Sample Test: FAILED (Invalid sampled ID)"
//                        }
//                    } else {
//                        testResultMessage = "Forward Pass + Sample Test: FAILED (Argmax sampling failed)."
//                        if nanCount > 0 {
//                            testResultMessage += " (Likely due to NaNs in logits)"
//                        }
//                    }
//                } else {
//                    print("  Error: Cannot copy private logits buffer to CPU.")
//                    testResultMessage = "Forward Pass + Sample Test: FAILED (Cannot access private logits buffer)."
//                }
//            } else {
//                testResultMessage = "Forward Pass + Sample Test: FAILED (Logits buffer size mismatch)."
//            }
//        } else {
//            testResultMessage = "Forward Pass + Sample Test: FAILED (No logits buffer returned)."
//        }
//        
//        print(testResultMessage)
//        print("--- Forward Pass + Sample Test Complete ---")
//        self.modelLoaderWrapper.currentStatus = testResultMessage
//    }
//}
//
//#Preview {
//    ContentView()
//}
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
                Task {
                    await modelLoaderWrapper.loadTokenizer()
                }
            } label: {
                Label("Load Tokenizer", systemImage: "textformat")
            }
            .padding(.bottom)
            .disabled(!modelLoaderWrapper.isMetadataLoaded || modelLoaderWrapper.getTokenizer() != nil)

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
        .fileImporter(
            isPresented: $isLoading,
            allowedContentTypes: [.data],
            allowsMultipleSelection: false
        ) { result in
            do {
                let selectedFile = try result.get().first!
                guard selectedFile.startAccessingSecurityScopedResource() else {
                    modelLoaderWrapper.currentStatus = "Error: Failed to access security-scoped resource."
                    isLoading = false
                    return
                }
                if let bookmarkData = try? selectedFile.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil) {
                    self.fileBookmark = bookmarkData
                }
                Task {
                    isLoading = true
                    await modelLoaderWrapper.loadMetadata(url: selectedFile)
                    await modelLoaderWrapper.assembleFullModel(url: selectedFile)
                    isLoading = false
                    selectedFile.stopAccessingSecurityScopedResource()
                }
            } catch {
                modelLoaderWrapper.currentStatus = "Error selecting file: \(error.localizedDescription)"
                isLoading = false
            }
        }
    }

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
                    await modelLoaderWrapper.assembleFullModel(url: url)
                    isLoading = false
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
//
//#Preview {
//    ContentView()
//}
