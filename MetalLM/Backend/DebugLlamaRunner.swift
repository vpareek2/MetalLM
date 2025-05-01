// MetalLM/Backend/DebugLlamaRunner.swift

import Foundation
import Metal

// Error specific to the debug runner
enum DebugRunnerError: Error {
    case nanDetected(bufferName: String, firstIndex: Int)
    case infDetected(bufferName: String, firstIndex: Int)
    case validationFailed(bufferName: String, reason: String)
    case bufferAllocationFailed(String)
    case commandEncoderCreationFailed(String)
    case underlyingError(Error)  // To wrap other errors
}

/// A version of LlamaRunner with aggressive NaN/Inf checks for debugging.
/// NOTE: This runner is SIGNIFICANTLY slower due to synchronous validation checks.
class DebugLlamaRunner {
    private let metalService: MetalService
    private let model: LlamaModel
    public let config: LlamaConfig

    // KV Cache Buffers
    private let kvCacheK: MTLBuffer
    private let kvCacheV: MTLBuffer

    // State for inference
    private(set) var currentPosition: Int = 0
    private let maxSequenceLength: Int

    // MARK: - Initialization & State

    init(model: LlamaModel, metalService: MetalService) throws {
        self.model = model
        self.config = model.config
        self.metalService = metalService
        self.maxSequenceLength = model.config.sequenceLength

        // --- Same KV Cache allocation as LlamaRunner ---
        let kvCacheElementCountPerBuffer =
            config.numLayers * config.sequenceLength * config.numKeyValueHeads * config.headDim
        let kvCacheSizeBytes = kvCacheElementCountPerBuffer * MemoryLayout<Float16>.stride

        guard kvCacheSizeBytes > 0 else {
            // Use the specific error type
            throw DebugRunnerError.bufferAllocationFailed(
                "KV Cache size calculation resulted in zero.")
        }

        let options: MTLResourceOptions = .storageModeShared  // Use Shared for easier validation access
        print(
            "[DebugRunner] Attempting to allocate KV Cache (Shared Storage): \(kvCacheSizeBytes * 2 / (1024*1024)) MB total..."
        )

        guard
            let tempCacheK = metalService.device.makeBuffer(
                length: kvCacheSizeBytes, options: options),
            let tempCacheV = metalService.device.makeBuffer(
                length: kvCacheSizeBytes, options: options)
        else {
            throw DebugRunnerError.bufferAllocationFailed("KV Cache (Shared)")
        }
        print("[DebugRunner] KV Cache allocated successfully (Shared storage).")

        self.kvCacheK = tempCacheK
        self.kvCacheV = tempCacheV
        self.kvCacheK.label = "KV_Cache_K_Debug"
        self.kvCacheV.label = "KV_Cache_V_Debug"
        self.currentPosition = 0
        // --- End KV Cache Allocation ---
    }

    func resetState() {
        currentPosition = 0
        print("[DebugRunner] State reset (position = 0).")
    }

    // MARK: - Validation Helper

    /// Validates buffer contents for NaNs/Infs. Commits and waits for the command buffer.
    /// Throws DebugRunnerError if validation fails.
    private func validateBuffer(
        _ buffer: MTLBuffer,
        name: String,
        expectedElementCount: Int,
        commandBuffer: MTLCommandBuffer  // The command buffer used for the *preceding* operation
    ) throws {
        print("--- Validating Buffer: \(name) (\(expectedElementCount) elements)...")
        guard expectedElementCount > 0 else {
            print("--- Validation skipped for '\(name)': Zero element count.")
            return  // Nothing to validate
        }

        let elementSize = MemoryLayout<Float16>.stride
        let expectedSize = expectedElementCount * elementSize
        guard buffer.length >= expectedSize else {
            throw DebugRunnerError.validationFailed(
                bufferName: name,
                reason: "Buffer length \(buffer.length) is less than expected \(expectedSize)."
            )
        }

        // We MUST ensure the GPU work writing to this buffer is complete before reading.
        // This makes the debug runner slow but guarantees correctness of the check.
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Check for command buffer errors *after* waiting
        if let error = commandBuffer.error {
            print("!!! GPU Error occurred before validating buffer '\(name)': \(error)")
            throw DebugRunnerError.underlyingError(error)
        }

        // Access contents (Buffer MUST be CPU accessible - Shared or Managed)
        // Our KV Cache and temporary buffers are allocated as Shared in this debug runner.
        guard buffer.storageMode != .private else {
            throw DebugRunnerError.validationFailed(
                bufferName: name,
                reason: "Cannot validate private buffer directly in this simplified checker.")
            // A more complex checker could blit to a shared buffer first,
            // but requires careful command buffer management.
        }

        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: expectedElementCount)
        let bufferPointer = UnsafeBufferPointer(start: pointer, count: expectedElementCount)

        var firstNanIndex = -1
        var firstInfIndex = -1

        // Perform the check
        for i in 0..<expectedElementCount {
            let value = bufferPointer[i]
            if value.isNaN {
                if firstNanIndex == -1 { firstNanIndex = i }
            }
            if value.isInfinite {
                if firstInfIndex == -1 { firstInfIndex = i }
            }
            // Optimization: Stop checking if both found
            if firstNanIndex != -1 && firstInfIndex != -1 {
                break
            }
        }

        // Throw specific errors if issues found
        if firstNanIndex != -1 {
            print(
                "!!! Validation FAILED for buffer '\(name)': NaN detected at index \(firstNanIndex) (and possibly others)."
            )
            // Optionally print surrounding values here
            throw DebugRunnerError.nanDetected(bufferName: name, firstIndex: firstNanIndex)
        }
        if firstInfIndex != -1 {
            print(
                "!!! Validation FAILED for buffer '\(name)': Inf detected at index \(firstInfIndex) (and possibly others)."
            )
            throw DebugRunnerError.infDetected(bufferName: name, firstIndex: firstInfIndex)
        }

        print("--- Validation PASSED for buffer '\(name)'.")
    }

    // MARK: - Forward Pass with Validation

    func forward(tokenID: Int) -> MTLBuffer? {
        let pos = currentPosition

        // --- Input Validation ---
        guard pos < maxSequenceLength else {
            print("[DebugRunner] Error: KV Cache full")
            return nil
        }
        guard tokenID >= 0 && tokenID < config.vocabSize else {
            print("[DebugRunner] Error: Invalid token ID \(tokenID)")
            return nil
        }

        print("[DebugRunner] Forward pass for token \(tokenID) at position \(pos)...")

        // Wrap the core logic in a do-catch to handle validation errors
        do {
            // --- 1. Create Command Buffer ---
            // Create ONE command buffer for the whole pass *or* per validated step?
            // For this debug approach, let's create one per step to make validateBuffer simpler.
            // This is INEFFICIENT but easy for debugging.
            guard let initialCommandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                print("[DebugRunner] Error: No command buffer")
                return nil
            }
            initialCommandBuffer.label = "Debug Llama Forward Pass CB (Pos: \(pos))"
            var currentCommandBuffer = initialCommandBuffer  // Will be replaced after each validation

            // --- 2. Allocate Temporary Buffers (Use Shared for easy validation) ---
            let options: MTLResourceOptions = .storageModeShared
            let embeddingDim = config.embeddingDim
            let hiddenDim = config.hiddenDim
            let headDim = config.headDim
            let nHeads = config.numHeads
            let nKVHeads = config.numKeyValueHeads
            let vocabSize = config.vocabSize
            let f16Size = MemoryLayout<Float16>.stride
            let hiddenStateSizeBytes = embeddingDim * f16Size
            let qSizeBytes = embeddingDim * f16Size  // nHeads * headDim * f16Size
            let kvSizeBytes = nKVHeads * headDim * f16Size
            let ffnHiddenSizeBytes = hiddenDim * f16Size
            let logitsSizeBytes = vocabSize * f16Size  // Using F16 logits

            // Allocation... (Ensure error handling)
            guard
                let hiddenStateBuffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),
                let normBuffer1 = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),
                let residual1Buffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),
                let qBuffer = metalService.device.makeBuffer(length: qSizeBytes, options: options),
                let kBuffer = metalService.device.makeBuffer(length: kvSizeBytes, options: options),
                let vBuffer = metalService.device.makeBuffer(length: kvSizeBytes, options: options),
                let attnOutputBuffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),
                let attnProjBuffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),
                let residual2Buffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),
                let normBuffer2 = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),
                let ffnGateBuffer = metalService.device.makeBuffer(
                    length: ffnHiddenSizeBytes, options: options),
                let ffnUpBuffer = metalService.device.makeBuffer(
                    length: ffnHiddenSizeBytes, options: options),
                let ffnDownBuffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),
                let logitsBuffer = metalService.device.makeBuffer(
                    length: logitsSizeBytes, options: options)
            else { throw DebugRunnerError.bufferAllocationFailed("Temporary Buffers") }

            // Labels...
            hiddenStateBuffer.label = "HiddenState \(pos)"
            normBuffer1.label = "Norm1 Output \(pos)"
            residual1Buffer.label = "Residual1 Input \(pos)"
            qBuffer.label = "Q Buffer \(pos)"
            kBuffer.label = "K Buffer Temp \(pos)"
            vBuffer.label = "V Buffer Temp \(pos)"
            attnOutputBuffer.label = "Attn Output Raw \(pos)"
            attnProjBuffer.label = "Attn Proj Output \(pos)"
            residual2Buffer.label = "Residual2 Input \(pos)"
            normBuffer2.label = "Norm2 Output (FFN Input) \(pos)"
            ffnGateBuffer.label = "FFN Gate/SiLU \(pos)"
            ffnUpBuffer.label = "FFN Up/SwiGLU \(pos)"
            ffnDownBuffer.label = "FFN Down Output \(pos)"
            logitsBuffer.label = "Logits Output F16 \(pos)"

            // --- Utility function to get a new command buffer ---
            func nextCommandBuffer(labelSuffix: String) throws -> MTLCommandBuffer {
                guard let cb = metalService.commandQueue.makeCommandBuffer() else {
                    throw DebugRunnerError.commandEncoderCreationFailed(
                        "Failed to create command buffer for \(labelSuffix)")
                }
                cb.label = "Debug CB \(pos) - \(labelSuffix)"
                return cb
            }

            // --- 3. Embedding Lookup ---
            currentCommandBuffer = try nextCommandBuffer(labelSuffix: "Embedding")
            guard let blitEncoderEmbed = currentCommandBuffer.makeBlitCommandEncoder() else {
                throw DebugRunnerError.commandEncoderCreationFailed("Embedding Blit")
            }
            blitEncoderEmbed.label = "Embedding Blit Encoder \(pos)"
            let embeddingOffset = tokenID * embeddingDim * f16Size
            guard model.tokenEmbeddings.length >= embeddingOffset + hiddenStateSizeBytes else {
                throw DebugRunnerError.validationFailed(
                    bufferName: "tokenEmbeddings", reason: "Offset out of bounds")
            }
            blitEncoderEmbed.copy(
                from: model.tokenEmbeddings, sourceOffset: embeddingOffset, to: hiddenStateBuffer,
                destinationOffset: 0, size: hiddenStateSizeBytes)
            blitEncoderEmbed.endEncoding()
            print("  Encoded Embedding Lookup.")
            // VALIDATE EMBEDDING OUTPUT
            try validateBuffer(
                hiddenStateBuffer, name: hiddenStateBuffer.label!,
                expectedElementCount: embeddingDim, commandBuffer: currentCommandBuffer)

            // --- 4. Loop through Layers ---
            for layerIndex in 0..<config.numLayers {
                let layerLabel = "L\(layerIndex) P\(pos)"
                print("    Processing \(layerLabel)...")

                // --- Save Residual 1 ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "SaveRes1 L\(layerIndex)")
                guard let blitEncoderRes1 = currentCommandBuffer.makeBlitCommandEncoder() else {
                    throw DebugRunnerError.commandEncoderCreationFailed(
                        "Save Residual 1 Blit L\(layerIndex)")
                }
                blitEncoderRes1.label = "Save Residual 1 \(layerLabel)"
                blitEncoderRes1.copy(
                    from: hiddenStateBuffer, sourceOffset: 0, to: residual1Buffer,
                    destinationOffset: 0, size: hiddenStateSizeBytes)
                blitEncoderRes1.endEncoding()
                // No need to validate residual buffer itself immediately

                // --- a. Pre-Attention RMSNorm ---
                currentCommandBuffer = try nextCommandBuffer(
                    labelSuffix: "PreAttnNorm L\(layerIndex)")
                var success = metalService.encodeRMSNormF16(
                    commandBuffer: currentCommandBuffer, inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.blocks[layerIndex].attentionNormWeight,
                    outputBuffer: normBuffer1, rowCount: 1, elementCountPerRow: embeddingDim,
                    eps: config.rmsNormEps, label: "PreAttnNorm \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "PreAttnNorm \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded Pre-Attn RMSNorm.")
                try validateBuffer(
                    normBuffer1, name: normBuffer1.label!, expectedElementCount: embeddingDim,
                    commandBuffer: currentCommandBuffer)

                // --- b. QKV Projection ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "QKV Proj L\(layerIndex)")
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: currentCommandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.qWeight, outputC: qBuffer, rowsA: 1,
                    colsA: embeddingDim, rowsB: embeddingDim, colsB: embeddingDim,
                    label: "Q_Proj \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "Q_Proj \(layerLabel)", reason: "Encoding failed")
                }
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: currentCommandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.kWeight, outputC: kBuffer, rowsA: 1,
                    colsA: embeddingDim, rowsB: embeddingDim, colsB: kvSizeBytes / f16Size,
                    label: "K_Proj \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "K_Proj \(layerLabel)", reason: "Encoding failed")
                }
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: currentCommandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.vWeight, outputC: vBuffer, rowsA: 1,
                    colsA: embeddingDim, rowsB: embeddingDim, colsB: kvSizeBytes / f16Size,
                    label: "V_Proj \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "V_Proj \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded QKV Projections.")
                try validateBuffer(
                    qBuffer, name: qBuffer.label!, expectedElementCount: qSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)
                // Re-use CB for K and V validation
                try validateBuffer(
                    kBuffer, name: kBuffer.label!, expectedElementCount: kvSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)
                try validateBuffer(
                    vBuffer, name: vBuffer.label!, expectedElementCount: kvSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)

                // --- c. RoPE ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "RoPE L\(layerIndex)")
                success = metalService.applyRoPE(
                    commandBuffer: currentCommandBuffer, buffer: qBuffer,
                    ropeFrequencies: model.ropeFrequencies, config: config, posOffset: pos,
                    sequenceLength: 1, numHeads: nHeads, headDim: headDim)
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "RoPE Q \(layerLabel)", reason: "Encoding failed")
                }
                success = metalService.applyRoPE(
                    commandBuffer: currentCommandBuffer, buffer: kBuffer,
                    ropeFrequencies: model.ropeFrequencies, config: config, posOffset: pos,
                    sequenceLength: 1, numHeads: nKVHeads, headDim: headDim)
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "RoPE K \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded RoPE.")
                try validateBuffer(
                    qBuffer, name: "\(qBuffer.label!)_PostRoPE",
                    expectedElementCount: qSizeBytes / f16Size, commandBuffer: currentCommandBuffer)
                try validateBuffer(
                    kBuffer, name: "\(kBuffer.label!)_PostRoPE",
                    expectedElementCount: kvSizeBytes / f16Size, commandBuffer: currentCommandBuffer
                )

                // --- d. KV Cache Update ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "KV Cache L\(layerIndex)")
                guard let blitEncoderKV = currentCommandBuffer.makeBlitCommandEncoder() else {
                    throw DebugRunnerError.commandEncoderCreationFailed(
                        "KV Cache Blit L\(layerIndex)")
                }
                blitEncoderKV.label = "KV Cache Update Blit \(layerLabel)"
                let elementsPerKVEntry = nKVHeads * headDim
                let bytesPerKVEntry = elementsPerKVEntry * f16Size
                let layerOffsetBytes = layerIndex * maxSequenceLength * bytesPerKVEntry
                let posOffsetBytes = pos * bytesPerKVEntry
                let destinationOffsetK = layerOffsetBytes + posOffsetBytes
                let destinationOffsetV = layerOffsetBytes + posOffsetBytes
                guard kvCacheK.length >= destinationOffsetK + bytesPerKVEntry,
                    kvCacheV.length >= destinationOffsetV + bytesPerKVEntry
                else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "KV Cache", reason: "Offset out of bounds L\(layerIndex)")
                }
                blitEncoderKV.copy(
                    from: kBuffer, sourceOffset: 0, to: kvCacheK,
                    destinationOffset: destinationOffsetK, size: bytesPerKVEntry)
                blitEncoderKV.copy(
                    from: vBuffer, sourceOffset: 0, to: kvCacheV,
                    destinationOffset: destinationOffsetV, size: bytesPerKVEntry)
                blitEncoderKV.endEncoding()
                print("      Encoded KV Cache Update.")
                // Skip validating the *entire* KV cache for performance in debug. Focus on intermediates.

                // --- e/f/g/h/i/j Attention Calculation ---
                let currentSeqLen = pos + 1
                let scale = Float16(1.0 / sqrt(Float(config.headDim)))
                let kvSliceSizeBytes = currentSeqLen * nKVHeads * headDim * f16Size
                let repeatedKVSizeBytes = currentSeqLen * nHeads * headDim * f16Size
                let scoreSizeBytes = nHeads * currentSeqLen * f16Size

                // Allocate Attention Temp Buffers (ensure shared)
                guard
                    let kSlice = metalService.device.makeBuffer(
                        length: kvSliceSizeBytes, options: options),
                    let vSlice = metalService.device.makeBuffer(
                        length: kvSliceSizeBytes, options: options),
                    let kRepeated = metalService.device.makeBuffer(
                        length: repeatedKVSizeBytes, options: options),
                    let vRepeated = metalService.device.makeBuffer(
                        length: repeatedKVSizeBytes, options: options),
                    let attnScores = metalService.device.makeBuffer(
                        length: scoreSizeBytes, options: options)
                else {
                    throw DebugRunnerError.bufferAllocationFailed(
                        "Attention Temp Buffers L\(layerIndex)")
                }
                kSlice.label = "kSlice \(layerLabel)"
                vSlice.label = "vSlice \(layerLabel)"
                kRepeated.label = "kRepeated \(layerLabel)"
                vRepeated.label = "vRepeated \(layerLabel)"
                attnScores.label = "attnScores/Probs \(layerLabel)"

                // --- e. Get K/V Slice ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "KVSlice L\(layerIndex)")
                guard let blitEncoderSlice = currentCommandBuffer.makeBlitCommandEncoder() else {
                    throw DebugRunnerError.commandEncoderCreationFailed(
                        "KV Slice Blit L\(layerIndex)")
                }
                blitEncoderSlice.label = "KV Slice Blit \(layerLabel)"
                let sourceOffsetKV = layerOffsetBytes  // Offset to the start of the layer's cache
                blitEncoderSlice.copy(
                    from: kvCacheK, sourceOffset: sourceOffsetKV, to: kSlice, destinationOffset: 0,
                    size: kvSliceSizeBytes)
                blitEncoderSlice.copy(
                    from: kvCacheV, sourceOffset: sourceOffsetKV, to: vSlice, destinationOffset: 0,
                    size: kvSliceSizeBytes)
                blitEncoderSlice.endEncoding()
                print("      Encoded Get K/V Slice.")
                try validateBuffer(
                    kSlice, name: kSlice.label!, expectedElementCount: kvSliceSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)
                try validateBuffer(
                    vSlice, name: vSlice.label!, expectedElementCount: kvSliceSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)

                // --- f. GQA Repeat Heads ---
                currentCommandBuffer = try nextCommandBuffer(
                    labelSuffix: "GQA Repeat L\(layerIndex)")
                success = metalService.applyRepeatKVHeads(
                    sourceBuffer: kSlice, destinationBuffer: kRepeated, numKVHeads: nKVHeads,
                    numQueryGroups: config.numQueryGroups, headDim: headDim, seqLen: currentSeqLen,
                    commandBuffer: currentCommandBuffer)
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "Repeat K \(layerLabel)", reason: "Encoding failed")
                }
                success = metalService.applyRepeatKVHeads(
                    sourceBuffer: vSlice, destinationBuffer: vRepeated, numKVHeads: nKVHeads,
                    numQueryGroups: config.numQueryGroups, headDim: headDim, seqLen: currentSeqLen,
                    commandBuffer: currentCommandBuffer)
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "Repeat V \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded GQA Repeat Heads.")
                try validateBuffer(
                    kRepeated, name: kRepeated.label!,
                    expectedElementCount: repeatedKVSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)
                try validateBuffer(
                    vRepeated, name: vRepeated.label!,
                    expectedElementCount: repeatedKVSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)

                // --- g/h/i/j Attention Calculation (Corrected with Head Loop) ---
                // Temp buffers needed for head looping
                let headSizeBytes = headDim * f16Size
                let kSliceHeadSizeBytes = currentSeqLen * headDim * f16Size
                let scoreSliceSizeBytes = currentSeqLen * f16Size

                guard
                    let qHeadBuffer = metalService.device.makeBuffer(
                        length: headSizeBytes, options: options),
                    let kvSliceHeadBuffer = metalService.device.makeBuffer(
                        length: kSliceHeadSizeBytes, options: options),
                    let scoreSliceBuffer = metalService.device.makeBuffer(
                        length: scoreSliceSizeBytes, options: options)
                else {
                    throw DebugRunnerError.bufferAllocationFailed(
                        "Attention Head Loop Buffers L\(layerIndex)")
                }
                qHeadBuffer.label = "qHeadBuffer \(layerLabel)"
                kvSliceHeadBuffer.label = "kvSliceHeadBuffer \(layerLabel)"  // Used for K and V slices per head
                scoreSliceBuffer.label = "scoreSliceBuffer \(layerLabel)"  // Used for scores and probs per head

                // --- g. Attention Scores (Q @ K^T) --- Loop per head ---
                print("      Encoding Attention Scores (Looping \(nHeads) heads)...")
                for h in 0..<nHeads {
                    let headLabel = "\(layerLabel) H\(h)"

                    // Extract Q head
                    currentCommandBuffer = try nextCommandBuffer(
                        labelSuffix: "ExtractQ \(headLabel)")
                    guard let blitEncoderQHead = currentCommandBuffer.makeBlitCommandEncoder()
                    else {
                        throw DebugRunnerError.commandEncoderCreationFailed(
                            "Extract Q Head Blit \(headLabel)")
                    }
                    let qOffset = h * headSizeBytes
                    blitEncoderQHead.copy(
                        from: qBuffer, sourceOffset: qOffset, to: qHeadBuffer, destinationOffset: 0,
                        size: headSizeBytes)
                    blitEncoderQHead.endEncoding()
                    // Validate Q head (optional, might be slow)
                    // try validateBuffer(qHeadBuffer, name: "\(qHeadBuffer.label!) \(headLabel)", expectedElementCount: headDim, commandBuffer: currentCommandBuffer)

                    // Extract K slice for this head (K is transposed implicitly by MatMul flag)
                    // We need to gather the K values for this head across the sequence length
                    currentCommandBuffer = try nextCommandBuffer(
                        labelSuffix: "ExtractK \(headLabel)")
                    guard let blitEncoderKHead = currentCommandBuffer.makeBlitCommandEncoder()
                    else {
                        throw DebugRunnerError.commandEncoderCreationFailed(
                            "Extract K Head Blit \(headLabel)")
                    }
                    blitEncoderKHead.label = "Extract K \(headLabel)"
                    for t in 0..<currentSeqLen {
                        // Source: kRepeated [t, h, d]
                        let srcOffset = (t * nHeads * headDim + h * headDim) * f16Size
                        // Destination: kvSliceHeadBuffer [t, d] (acting as K^T buffer for this head)
                        let dstOffset = (t * headDim) * f16Size
                        blitEncoderKHead.copy(
                            from: kRepeated, sourceOffset: srcOffset, to: kvSliceHeadBuffer,
                            destinationOffset: dstOffset, size: headSizeBytes)
                    }
                    blitEncoderKHead.endEncoding()
                    // Validate K slice for head (optional)
                    // try validateBuffer(kvSliceHeadBuffer, name: "\(kvSliceHeadBuffer.label!)_K \(headLabel)", expectedElementCount: currentSeqLen * headDim, commandBuffer: currentCommandBuffer)

                    // Calculate Scores for head: Q[1, headDim] @ K^T[headDim, currentSeqLen] -> Score[1, currentSeqLen]
                    currentCommandBuffer = try nextCommandBuffer(
                        labelSuffix: "ScoreMM \(headLabel)")
                    success = metalService.encodeMPSMatrixMultiply(
                        commandBuffer: currentCommandBuffer,
                        inputA: qHeadBuffer,
                        inputB: kvSliceHeadBuffer,  // Contains K^T layout for this head
                        outputC: scoreSliceBuffer,
                        rowsA: 1, colsA: headDim,  // Q head
                        rowsB: currentSeqLen, colsB: headDim,  // K slice (implicitly transposed by flag)
                        transposeA: false, transposeB: true,  // Use transposeB = true
                        alpha: Double(scale), beta: 0.0,
                        label: "ScoreMatMul \(headLabel)"
                    )
                    guard success else {
                        throw DebugRunnerError.validationFailed(
                            bufferName: "Score MatMul \(headLabel)", reason: "Encoding failed")
                    }
                    // Validate raw scores for head (optional)
                    // try validateBuffer(scoreSliceBuffer, name: "\(scoreSliceBuffer.label!)_Raw \(headLabel)", expectedElementCount: currentSeqLen, commandBuffer: currentCommandBuffer)

                    // Copy head scores into the main attnScores buffer
                    currentCommandBuffer = try nextCommandBuffer(
                        labelSuffix: "CopyScore \(headLabel)")
                    guard let blitEncoderScore = currentCommandBuffer.makeBlitCommandEncoder()
                    else {
                        throw DebugRunnerError.commandEncoderCreationFailed(
                            "Copy Score Blit \(headLabel)")
                    }
                    blitEncoderScore.label = "Copy Score \(headLabel)"
                    let scoreDestOffset = h * currentSeqLen * f16Size
                    blitEncoderScore.copy(
                        from: scoreSliceBuffer, sourceOffset: 0, to: attnScores,
                        destinationOffset: scoreDestOffset, size: scoreSliceSizeBytes)
                    blitEncoderScore.endEncoding()
                }
                print("      Finished Encoding Attention Scores.")
                // Validate the *full* raw scores buffer after the loop
                currentCommandBuffer = try nextCommandBuffer(
                    labelSuffix: "AttnScores_PostLoop L\(layerIndex)")  // Need a CB to validate
                try validateBuffer(
                    attnScores, name: "\(attnScores.label!)_Raw_Full",
                    expectedElementCount: scoreSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)

                // --- i. Softmax (Applied to the full attnScores buffer) ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "Softmax L\(layerIndex)")
                success = metalService.encodeMPSSoftMax(
                    commandBuffer: currentCommandBuffer, inputMatrixBuffer: attnScores,
                    outputMatrixBuffer: attnScores, rows: nHeads, columns: currentSeqLen,
                    label: "Softmax \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "Softmax \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded Softmax.")
                try validateBuffer(
                    attnScores, name: "\(attnScores.label!)_Probs_Full",
                    expectedElementCount: scoreSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)

                // --- j. Scores @ V --- Loop per head ---
                print("      Encoding Attention Values (Looping \(nHeads) heads)...")
                // Zero out the output buffer before accumulating head results
                currentCommandBuffer = try nextCommandBuffer(
                    labelSuffix: "ZeroAttnOut L\(layerIndex)")
                guard let blitEncoderZeroAttnOut = currentCommandBuffer.makeBlitCommandEncoder()
                else {
                    throw DebugRunnerError.commandEncoderCreationFailed(
                        "Zero AttnOutput Blit L\(layerIndex)")
                }
                blitEncoderZeroAttnOut.fill(
                    buffer: attnOutputBuffer, range: 0..<attnOutputBuffer.length, value: 0)
                blitEncoderZeroAttnOut.endEncoding()
                // No validation needed for zeroing

                for h in 0..<nHeads {
                    let headLabel = "\(layerLabel) H\(h)"

                    // Extract Probabilities for head
                    currentCommandBuffer = try nextCommandBuffer(
                        labelSuffix: "ExtractProbs \(headLabel)")
                    guard let blitEncoderProbHead = currentCommandBuffer.makeBlitCommandEncoder()
                    else {
                        throw DebugRunnerError.commandEncoderCreationFailed(
                            "Extract Probs Blit \(headLabel)")
                    }
                    blitEncoderProbHead.label = "Extract Probs \(headLabel)"
                    let probSourceOffset = h * currentSeqLen * f16Size
                    blitEncoderProbHead.copy(
                        from: attnScores, sourceOffset: probSourceOffset, to: scoreSliceBuffer,
                        destinationOffset: 0, size: scoreSliceSizeBytes)
                    blitEncoderProbHead.endEncoding()
                    // Validate Probs head (optional)
                    // try validateBuffer(scoreSliceBuffer, name: "\(scoreSliceBuffer.label!)_Probs \(headLabel)", expectedElementCount: currentSeqLen, commandBuffer: currentCommandBuffer)

                    // Extract V slice for this head
                    currentCommandBuffer = try nextCommandBuffer(
                        labelSuffix: "ExtractV \(headLabel)")
                    guard let blitEncoderVHead = currentCommandBuffer.makeBlitCommandEncoder()
                    else {
                        throw DebugRunnerError.commandEncoderCreationFailed(
                            "Extract V Head Blit \(headLabel)")
                    }
                    blitEncoderVHead.label = "Extract V \(headLabel)"
                    for t in 0..<currentSeqLen {
                        // Source: vRepeated [t, h, d]
                        let srcOffset = (t * nHeads * headDim + h * headDim) * f16Size
                        // Destination: kvSliceHeadBuffer [t, d]
                        let dstOffset = (t * headDim) * f16Size
                        blitEncoderVHead.copy(
                            from: vRepeated, sourceOffset: srcOffset, to: kvSliceHeadBuffer,
                            destinationOffset: dstOffset, size: headSizeBytes)
                    }
                    blitEncoderVHead.endEncoding()
                    // Validate V slice for head (optional)
                    // try validateBuffer(kvSliceHeadBuffer, name: "\(kvSliceHeadBuffer.label!)_V \(headLabel)", expectedElementCount: currentSeqLen * headDim, commandBuffer: currentCommandBuffer)

                    // Calculate Value for head: Probs[1, currentSeqLen] @ V[currentSeqLen, headDim] -> Output[1, headDim]
                    currentCommandBuffer = try nextCommandBuffer(
                        labelSuffix: "ValueMM \(headLabel)")
                    success = metalService.encodeMPSMatrixMultiply(
                        commandBuffer: currentCommandBuffer,
                        inputA: scoreSliceBuffer,  // Probs for head
                        inputB: kvSliceHeadBuffer,  // V slice for head
                        outputC: qHeadBuffer,  // Re-use qHeadBuffer for output
                        rowsA: 1, colsA: currentSeqLen,
                        rowsB: currentSeqLen, colsB: headDim,
                        transposeA: false, transposeB: false,
                        label: "ValueMatMul \(headLabel)"
                    )
                    guard success else {
                        throw DebugRunnerError.validationFailed(
                            bufferName: "Value MatMul \(headLabel)", reason: "Encoding failed")
                    }
                    // Validate head output value (optional)
                    // try validateBuffer(qHeadBuffer, name: "\(qHeadBuffer.label!)_Value \(headLabel)", expectedElementCount: headDim, commandBuffer: currentCommandBuffer)

                    // Copy head result into the main attnOutputBuffer
                    currentCommandBuffer = try nextCommandBuffer(
                        labelSuffix: "CopyAttnOut \(headLabel)")
                    guard let blitEncoderAttnOut = currentCommandBuffer.makeBlitCommandEncoder()
                    else {
                        throw DebugRunnerError.commandEncoderCreationFailed(
                            "Copy AttnOutput Blit \(headLabel)")
                    }
                    blitEncoderAttnOut.label = "Copy AttnOutput \(headLabel)"
                    let attnDestOffset = h * headSizeBytes
                    blitEncoderAttnOut.copy(
                        from: qHeadBuffer, sourceOffset: 0, to: attnOutputBuffer,
                        destinationOffset: attnDestOffset, size: headSizeBytes)
                    blitEncoderAttnOut.endEncoding()
                }
                print("      Finished Encoding Attention Values.")
                // Validate the *full* attention output buffer after the loop
                currentCommandBuffer = try nextCommandBuffer(
                    labelSuffix: "AttnOutput_PostLoop L\(layerIndex)")  // Need a CB to validate
                try validateBuffer(
                    attnOutputBuffer, name: attnOutputBuffer.label!,
                    expectedElementCount: hiddenStateSizeBytes / f16Size,
                    commandBuffer: currentCommandBuffer)

                // --- k. Output Projection ---
                currentCommandBuffer = try nextCommandBuffer(
                    labelSuffix: "AttnOProj L\(layerIndex)")
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: currentCommandBuffer, inputA: attnOutputBuffer,
                    inputB: model.blocks[layerIndex].attention.oWeight, outputC: attnProjBuffer,
                    rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: embeddingDim,
                    label: "Attn_O_Proj \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "Attn_O_Proj \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded Attn Output Projection.")
                try validateBuffer(
                    attnProjBuffer, name: attnProjBuffer.label!, expectedElementCount: embeddingDim,
                    commandBuffer: currentCommandBuffer)

                // --- l. Residual Add 1 ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "ResAdd1 L\(layerIndex)")
                success = metalService.applyElementWiseAdd(
                    inputBufferA: residual1Buffer, inputBufferB: attnProjBuffer,
                    outputBufferC: hiddenStateBuffer, elementCount: embeddingDim,
                    commandBuffer: currentCommandBuffer)
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "Residual Add 1 \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded Residual Add 1.")
                try validateBuffer(
                    hiddenStateBuffer, name: "\(hiddenStateBuffer.label!)_PostRes1",
                    expectedElementCount: embeddingDim, commandBuffer: currentCommandBuffer)

                // --- Save Residual 2 ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "SaveRes2 L\(layerIndex)")
                guard let blitEncoderRes2 = currentCommandBuffer.makeBlitCommandEncoder() else {
                    throw DebugRunnerError.commandEncoderCreationFailed(
                        "Save Residual 2 Blit L\(layerIndex)")
                }
                blitEncoderRes2.label = "Save Residual 2 \(layerLabel)"
                blitEncoderRes2.copy(
                    from: hiddenStateBuffer, sourceOffset: 0, to: residual2Buffer,
                    destinationOffset: 0, size: hiddenStateSizeBytes)
                blitEncoderRes2.endEncoding()
                // No immediate validation needed

                // --- m. Pre-FFN RMSNorm ---
                currentCommandBuffer = try nextCommandBuffer(
                    labelSuffix: "PreFFNNorm L\(layerIndex)")
                success = metalService.encodeRMSNormF16(
                    commandBuffer: currentCommandBuffer, inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.blocks[layerIndex].ffnNormWeight, outputBuffer: normBuffer2,
                    rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                    label: "PreFFNNorm \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "PreFFNNorm \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded Pre-FFN RMSNorm.")
                try validateBuffer(
                    normBuffer2, name: normBuffer2.label!, expectedElementCount: embeddingDim,
                    commandBuffer: currentCommandBuffer)

                // --- n. MLP / SwiGLU ---
                // Gate Proj
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "FFNGate L\(layerIndex)")
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: currentCommandBuffer, inputA: normBuffer2,
                    inputB: model.blocks[layerIndex].mlp.gateWeight, outputC: ffnGateBuffer,
                    rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: hiddenDim,
                    label: "FFN_Gate_Proj \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "FFN_Gate_Proj \(layerLabel)", reason: "Encoding failed")
                }
                try validateBuffer(
                    ffnGateBuffer, name: "\(ffnGateBuffer.label!)_Raw",
                    expectedElementCount: hiddenDim, commandBuffer: currentCommandBuffer)

                // Up Proj
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "FFNUp L\(layerIndex)")
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: currentCommandBuffer, inputA: normBuffer2,
                    inputB: model.blocks[layerIndex].mlp.upWeight, outputC: ffnUpBuffer, rowsA: 1,
                    colsA: embeddingDim, rowsB: embeddingDim, colsB: hiddenDim,
                    label: "FFN_Up_Proj \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "FFN_Up_Proj \(layerLabel)", reason: "Encoding failed")
                }
                try validateBuffer(
                    ffnUpBuffer, name: "\(ffnUpBuffer.label!)_Raw", expectedElementCount: hiddenDim,
                    commandBuffer: currentCommandBuffer)

                // SiLU (In-place on Gate Buffer)
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "FFNSiLU L\(layerIndex)")
                success = metalService.applySILU(
                    inputBuffer: ffnGateBuffer, outputBuffer: ffnGateBuffer,
                    elementCount: hiddenDim, commandBuffer: currentCommandBuffer)
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "SiLU \(layerLabel)", reason: "Encoding failed")
                }
                try validateBuffer(
                    ffnGateBuffer, name: ffnGateBuffer.label!, expectedElementCount: hiddenDim,
                    commandBuffer: currentCommandBuffer)

                // Element-wise Multiply (Gate * Up -> Up)
                currentCommandBuffer = try nextCommandBuffer(
                    labelSuffix: "FFNMultiply L\(layerIndex)")
                success = metalService.applyElementWiseMul(
                    inputBufferA: ffnGateBuffer, inputBufferB: ffnUpBuffer,
                    outputBufferC: ffnUpBuffer, elementCount: hiddenDim,
                    commandBuffer: currentCommandBuffer)
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "ElemWise Mul \(layerLabel)", reason: "Encoding failed")
                }
                try validateBuffer(
                    ffnUpBuffer, name: ffnUpBuffer.label!, expectedElementCount: hiddenDim,
                    commandBuffer: currentCommandBuffer)

                // Down Proj
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "FFNDown L\(layerIndex)")
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: currentCommandBuffer, inputA: ffnUpBuffer,
                    inputB: model.blocks[layerIndex].mlp.downWeight, outputC: ffnDownBuffer,
                    rowsA: 1, colsA: hiddenDim, rowsB: hiddenDim, colsB: embeddingDim,
                    label: "FFN_Down_Proj \(layerLabel)")
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "FFN_Down_Proj \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded MLP/SwiGLU.")
                try validateBuffer(
                    ffnDownBuffer, name: ffnDownBuffer.label!, expectedElementCount: embeddingDim,
                    commandBuffer: currentCommandBuffer)

                // --- o. Residual Add 2 ---
                currentCommandBuffer = try nextCommandBuffer(labelSuffix: "ResAdd2 L\(layerIndex)")
                success = metalService.applyElementWiseAdd(
                    inputBufferA: residual2Buffer, inputBufferB: ffnDownBuffer,
                    outputBufferC: hiddenStateBuffer, elementCount: embeddingDim,
                    commandBuffer: currentCommandBuffer)
                guard success else {
                    throw DebugRunnerError.validationFailed(
                        bufferName: "Residual Add 2 \(layerLabel)", reason: "Encoding failed")
                }
                print("      Encoded Residual Add 2.")
                try validateBuffer(
                    hiddenStateBuffer, name: "\(hiddenStateBuffer.label!)_PostRes2",
                    expectedElementCount: embeddingDim, commandBuffer: currentCommandBuffer)

            }  // End layer loop
            print("    Finished layer loop.")

            // --- 5. Final RMSNorm ---
            currentCommandBuffer = try nextCommandBuffer(labelSuffix: "FinalNorm")
            var success = metalService.encodeRMSNormF16(
                commandBuffer: currentCommandBuffer, inputBuffer: hiddenStateBuffer,
                weightBuffer: model.finalNormWeight, outputBuffer: normBuffer1, rowCount: 1,
                elementCountPerRow: embeddingDim, eps: config.rmsNormEps, label: "FinalNorm")
            guard success else {
                throw DebugRunnerError.validationFailed(
                    bufferName: "FinalNorm", reason: "Encoding failed")
            }
            print("  Encoded Final RMSNorm.")
            try validateBuffer(
                normBuffer1, name: "FinalNorm Output", expectedElementCount: embeddingDim,
                commandBuffer: currentCommandBuffer)

            // --- 6. Output Projection (Logits) ---
            currentCommandBuffer = try nextCommandBuffer(labelSuffix: "OutputProj")
            success = metalService.encodeMPSMatrixMultiply(
                commandBuffer: currentCommandBuffer, inputA: normBuffer1,
                inputB: model.outputWeight, outputC: logitsBuffer, rowsA: 1, colsA: embeddingDim,
                rowsB: embeddingDim, colsB: vocabSize, label: "Output Projection")
            guard success else {
                throw DebugRunnerError.validationFailed(
                    bufferName: "Output Projection", reason: "Encoding failed")
            }
            print("  Encoded Output Projection.")
            try validateBuffer(
                logitsBuffer, name: logitsBuffer.label!, expectedElementCount: vocabSize,
                commandBuffer: currentCommandBuffer)

            // --- 7. Increment Position ---
            currentPosition += 1

            print(
                "[DebugRunner] Forward pass completed successfully up to validation point for pos \(pos)."
            )
            // --- 8. Return logits buffer ---
            return logitsBuffer

        } catch let error as DebugRunnerError {
            // Catch specific validation errors
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! DebugRunner HALTED due to validation error: \(error)")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            // Consider adding more context here, like currentPosition, layerIndex if available
            return nil
        } catch {
            // Catch unexpected errors during encoding/setup
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! DebugRunner HALTED due to unexpected error: \(error)")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return nil
        }
    }
}
