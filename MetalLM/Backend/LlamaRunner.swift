// MetalLM/Backend/LlamaRunner.swift

import Foundation
import Metal

// import MetalPerformanceShaders // Not needed directly here if using MetalService wrappers

// Error specific to the runner
enum LlamaRunnerError: Error {
    case kvCacheAllocationFailed
    case kvCacheOutOfBounds
    case invalidTokenID
    case bufferAllocationFailed(String)
    case commandEncoderCreationFailed(String)
    // Add other errors later
}

/// Manages the state and execution of a Llama model forward pass.
class LlamaRunner {
    private let metalService: MetalService
    private let model: LlamaModel
    public let config: LlamaConfig

    // KV Cache Buffers - Allocate for max sequence length
    private let kvCacheK: MTLBuffer
    private let kvCacheV: MTLBuffer

    // State for inference
    private(set) var currentPosition: Int = 0
    private let maxSequenceLength: Int

    // MARK: - Initialization & State

    /// Initializes the runner, allocating the KV cache.
    ///
    /// - Parameters:
    ///   - model: The loaded LlamaModel containing configuration and weights.
    ///   - metalService: The shared MetalService instance.
    /// - Throws: `LlamaRunnerError.kvCacheAllocationFailed` if buffers cannot be created.
    init(model: LlamaModel, metalService: MetalService) throws {
        self.model = model
        self.config = model.config  // Initialize the public config property
        self.metalService = metalService
        self.maxSequenceLength = model.config.sequenceLength

        // --- Calculate KV Cache Size ---
        let kvCacheElementCountPerBuffer =
            config.numLayers * config.sequenceLength * config.numKeyValueHeads * config.headDim
        let kvCacheSizeBytes = kvCacheElementCountPerBuffer * MemoryLayout<Float16>.stride

        guard kvCacheSizeBytes > 0 else {
            print("Error: Calculated KV Cache size is zero or negative.")
            throw LlamaRunnerError.kvCacheAllocationFailed
        }

        // --- Allocate Buffers ---
        let options: MTLResourceOptions = .storageModePrivate  // Prefer private
        print(
            "Attempting to allocate KV Cache (Private Storage): \(kvCacheSizeBytes * 2 / (1024*1024)) MB total..."
        )

        var tempCacheK: MTLBuffer?
        var tempCacheV: MTLBuffer?

        tempCacheK = metalService.device.makeBuffer(length: kvCacheSizeBytes, options: options)
        tempCacheV = metalService.device.makeBuffer(length: kvCacheSizeBytes, options: options)

        if tempCacheK == nil || tempCacheV == nil {
            print("Warning: Failed to allocate KV Cache with Private storage. Trying Shared...")
            let sharedOptions: MTLResourceOptions = .storageModeShared
            tempCacheK = metalService.device.makeBuffer(
                length: kvCacheSizeBytes, options: sharedOptions)
            tempCacheV = metalService.device.makeBuffer(
                length: kvCacheSizeBytes, options: sharedOptions)

            guard tempCacheK != nil, tempCacheV != nil else {
                print(
                    "Error: Failed to allocate KV Cache with Shared storage. Size: \(kvCacheSizeBytes) bytes per cache."
                )
                throw LlamaRunnerError.kvCacheAllocationFailed
            }
            print("KV Cache allocated with Shared storage.")
        } else {
            print("KV Cache allocated successfully (Private storage).")
        }

        self.kvCacheK = tempCacheK!
        self.kvCacheV = tempCacheV!

        // --- Set Labels ---
        self.kvCacheK.label =
            "KV_Cache_K (L:\(config.numLayers) S:\(config.sequenceLength) KVH:\(config.numKeyValueHeads) HD:\(config.headDim))"
        self.kvCacheV.label =
            "KV_Cache_V (L:\(config.numLayers) S:\(config.sequenceLength) KVH:\(config.numKeyValueHeads) HD:\(config.headDim))"

        // Initial state
        self.currentPosition = 0
    }

    /// Resets the inference state (primarily the current position).
    /// Note: This does NOT clear the buffer contents currently.
    func resetState() {
        currentPosition = 0
        print("LlamaRunner state reset (position = 0).")
        // TODO: Consider adding an option to zero out the KV cache buffers if needed.
        // This would require a Metal kernel (e.g., using fillBuffer).
    }

    // MARK: - Forward Pass

    func forward(tokenID: Int) -> MTLBuffer? {
        let pos = currentPosition

        return autoreleasepool {
            // Declare success ONCE for the whole function scope
            var success: Bool = false

            // --- Input Validation ---
            guard pos < maxSequenceLength else {
                print("Error: Position \(pos) exceeds maximum sequence length \(maxSequenceLength)")
                return nil
            }
            guard tokenID >= 0 && tokenID < config.vocabSize else {
                print(
                    "Error: Invalid token ID \(tokenID). Must be between 0 and \(config.vocabSize-1)"
                )
                return nil
            }

            print("Forward pass for token \(tokenID) at position \(pos)...")

            // --- 1. Create Command Buffer ---
            guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                print("Error: Failed to create command buffer")
                return nil
            }
            commandBuffer.label = "Llama Forward Pass CB (Pos: \(pos))"

            // --- 2. Allocate Temporary Buffers ---
            // (Keep buffer allocation logic as is)
            let options: MTLResourceOptions = .storageModeShared
            let embeddingDim = config.embeddingDim
            let hiddenDim = config.hiddenDim
            let headDim = config.headDim
            let nHeads = config.numHeads
            let nKVHeads = config.numKeyValueHeads
            let vocabSize = config.vocabSize
            let f16Size = MemoryLayout<Float16>.stride
            //            let f32Size = MemoryLayout<Float32>.stride

            let hiddenStateSizeBytes = embeddingDim * f16Size
            let qSizeBytes = embeddingDim * f16Size
            let kvSizeBytes = nKVHeads * headDim * f16Size
            let ffnHiddenSizeBytes = hiddenDim * f16Size
            // Let's explicitly use F16 for logits buffer in this MVP pass to match other temps
            // and avoid needing changes to encodeMPSMatrixMultiply for now.
            // We can change this later if F32 logits are strictly required.
            let logitsSizeBytes = vocabSize * f16Size  // Using F16 for now
            print("  NOTE: Using F16 for logits buffer in this pass.")
            // let logitsSizeBytes = vocabSize * f32Size // Use this if F32 needed later

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
                    length: logitsSizeBytes, options: options)  // F16 size
            else {
                print("Error: Failed to allocate temporary buffers")
                return nil
            }

            // (Keep buffer labels)
            hiddenStateBuffer.label = "HiddenState \(pos)"
            // ... etc ...
            logitsBuffer.label = "Logits Output F16 \(pos)"

            // --- 3. Embedding Lookup ---
            // (Keep embedding lookup logic as is)
            guard let blitEncoderEmbed = commandBuffer.makeBlitCommandEncoder() else {
                print("Error: Failed to create blit encoder for embedding lookup")
                return nil
            }
            // ... copy ...
            blitEncoderEmbed.endEncoding()
            print("  Encoded Embedding Lookup.")

            // --- 4. Loop through Layers ---
            for layerIndex in 0..<config.numLayers {
                let layerLabel = "L\(layerIndex) P\(pos)"
                print("    Processing \(layerLabel)...")

                // --- Save input for residual connection 1 ---
                guard let blitEncoderRes1 = commandBuffer.makeBlitCommandEncoder() else {
                    return nil
                }
                // ... copy hiddenStateBuffer to residual1Buffer ...
                blitEncoderRes1.endEncoding()

                // --- a. Input RMSNorm (Pre-Attention Norm) ---
                success = metalService.encodeRMSNormF16(
                    commandBuffer: commandBuffer, inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.blocks[layerIndex].attentionNormWeight,
                    outputBuffer: normBuffer1,
                    rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                    label: "PreAttnNorm \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Pre-Attn RMSNorm \(layerLabel)")
                    return nil
                }
                print("      Encoded Pre-Attn RMSNorm.")

                // --- b. QKV Projection (MatMul) ---
                // Input: normBuffer1 [1, embeddingDim]
                // Weights: Wq [embedDim, embedDim], Wk/Wv [embedDim, kvDim] (Assuming row-major storage)
                let kvDim = nKVHeads * headDim

                // Q = norm * Wq
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: normBuffer1, inputB: model.blocks[layerIndex].attention.qWeight,
                    outputC: qBuffer,
                    rowsA: 1, colsA: embeddingDim,  // Input A layout
                    rowsB: embeddingDim, colsB: embeddingDim,  // Input B (Weight) layout
                    transposeA: false, transposeB: false,  // Math operation
                    label: "Q_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Q Proj \(layerLabel)")
                    return nil
                }

                // K = norm * Wk
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: normBuffer1, inputB: model.blocks[layerIndex].attention.kWeight,
                    outputC: kBuffer,
                    rowsA: 1, colsA: embeddingDim,  // Input A layout
                    rowsB: embeddingDim, colsB: kvDim,  // Input B (Weight) layout
                    transposeA: false, transposeB: false,  // Math operation
                    label: "K_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding K Proj \(layerLabel)")
                    return nil
                }

                // V = norm * Wv
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: normBuffer1, inputB: model.blocks[layerIndex].attention.vWeight,
                    outputC: vBuffer,
                    rowsA: 1, colsA: embeddingDim,  // Input A layout
                    rowsB: embeddingDim, colsB: kvDim,  // Input B (Weight) layout
                    transposeA: false, transposeB: false,  // Math operation
                    label: "V_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding V Proj \(layerLabel)")
                    return nil
                }
                print("      Encoded QKV Projections.")

                // --- c. RoPE ---
                success = metalService.applyRoPE(  // Assuming applyRoPE returns Bool
                    commandBuffer: commandBuffer, buffer: qBuffer,
                    ropeFrequencies: model.ropeFrequencies,
                    config: config, posOffset: pos, sequenceLength: 1, numHeads: nHeads,
                    headDim: headDim)
                guard success else {
                    print("Error applying RoPE to Q \(layerLabel)")
                    return nil
                }

                success = metalService.applyRoPE(  // Assuming applyRoPE returns Bool
                    commandBuffer: commandBuffer, buffer: kBuffer,
                    ropeFrequencies: model.ropeFrequencies,
                    config: config, posOffset: pos, sequenceLength: 1, numHeads: nKVHeads,
                    headDim: headDim)
                guard success else {
                    print("Error applying RoPE to K \(layerLabel)")
                    return nil
                }
                print("      Encoded RoPE.")

                // --- d. KV Cache Update ---
                // (Keep KV Cache update logic as is)
                guard let blitEncoderKV = commandBuffer.makeBlitCommandEncoder() else {
                    print("Error creating blit encoder for KV cache update \(layerLabel)")
                    return nil
                }
                // ... copy kBuffer/vBuffer to kvCacheK/V ...
                blitEncoderKV.endEncoding()
                print("      Encoded KV Cache Update.")

                // --- e/f/g/h/i/j Attention Calculation ---
                let currentSeqLen = pos + 1
                let scale = Float16(1.0 / sqrt(Float(config.headDim)))
                let bytesPerKVEntry = nKVHeads * headDim * f16Size

                // Allocate buffers needed for attention (sizes depend on currentSeqLen)
                // *** TODO: Optimize buffer allocation later (move to init) ***
                let kvSliceSizeBytes = currentSeqLen * nKVHeads * headDim * f16Size
                let repeatedKVSizeBytes = currentSeqLen * nHeads * headDim * f16Size
                let scoreSizeBytes = nHeads * currentSeqLen * f16Size

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
                        length: scoreSizeBytes, options: options)  // Output of Q@K^T, input/output of Softmax
                else {
                    print("Error allocating attention buffers for \(layerLabel)")
                    return nil
                }

                kSlice.label = "kSlice \(layerLabel)"
                vSlice.label = "vSlice \(layerLabel)"
                kRepeated.label = "kRepeated \(layerLabel)"
                vRepeated.label = "vRepeated \(layerLabel)"
                attnScores.label = "attnScores/Probs \(layerLabel)"

                // --- e. Get K/V Slice from Cache (Blit) ---
                guard let blitEncoderSlice = commandBuffer.makeBlitCommandEncoder() else {
                    return nil
                }
                blitEncoderSlice.label = "KV Slice Blit \(layerLabel)"
                let layerOffsetBytes = layerIndex * maxSequenceLength * bytesPerKVEntry
                let sourceOffsetKV = layerOffsetBytes
                blitEncoderSlice.copy(
                    from: kvCacheK, sourceOffset: sourceOffsetKV, to: kSlice, destinationOffset: 0,
                    size: kvSliceSizeBytes)
                blitEncoderSlice.copy(
                    from: kvCacheV, sourceOffset: sourceOffsetKV, to: vSlice, destinationOffset: 0,
                    size: kvSliceSizeBytes)
                blitEncoderSlice.endEncoding()
                print("      Encoded Get K/V Slice.")

                // --- f. GQA Repeat Heads (Custom Kernel) ---
                success = metalService.applyRepeatKVHeads(
                    sourceBuffer: kSlice, destinationBuffer: kRepeated,
                    numKVHeads: nKVHeads, numQueryGroups: config.numQueryGroups, headDim: headDim,
                    seqLen: currentSeqLen,
                    commandBuffer: commandBuffer)
                guard success else {
                    print("Error repeating K heads \(layerLabel)")
                    return nil
                }
                success = metalService.applyRepeatKVHeads(
                    sourceBuffer: vSlice, destinationBuffer: vRepeated,
                    numKVHeads: nKVHeads, numQueryGroups: config.numQueryGroups, headDim: headDim,
                    seqLen: currentSeqLen,
                    commandBuffer: commandBuffer)
                guard success else {
                    print("Error repeating V heads \(layerLabel)")
                    return nil
                }
                print("      Encoded GQA Repeat Heads.")

                // --- g. Attention Scores (Q @ K^T * scale) ---
                // Using looping approach for MVP.
                let headSizeBytes = headDim * f16Size
                let kSliceHeadSizeBytes = currentSeqLen * headDim * f16Size  // Holds K[h] or V[h] slice
                let scoreSliceSizeBytes = currentSeqLen * f16Size  // Holds Score[h] or Probs[h] slice

                guard
                    let qHeadBuffer = metalService.device.makeBuffer(
                        length: headSizeBytes, options: options),
                    let kvSliceHeadBuffer = metalService.device.makeBuffer(
                        length: kSliceHeadSizeBytes, options: options),  // Reused for K[h] and V[h]
                    let scoreSliceBuffer = metalService.device.makeBuffer(
                        length: scoreSliceSizeBytes, options: options)  // Reused for Score[h] and Probs[h]
                else {
                    print("Error allocating per-head attention buffers for \(layerLabel)")
                    return nil
                }

                qHeadBuffer.label = "qHeadBuffer \(layerLabel)"
                kvSliceHeadBuffer.label = "kvSliceHeadBuffer \(layerLabel)"
                scoreSliceBuffer.label = "scoreSliceBuffer \(layerLabel)"

                print("      Encoding Attention Scores (Looping \(nHeads) heads)...")
                for h in 0..<nHeads {
                    // 1. Extract Q[h] -> qHeadBuffer
                    guard let blitEncoderQHead = commandBuffer.makeBlitCommandEncoder() else {
                        return nil
                    }
                    let qOffset = h * headSizeBytes
                    blitEncoderQHead.copy(
                        from: qBuffer, sourceOffset: qOffset, to: qHeadBuffer, destinationOffset: 0,
                        size: headSizeBytes)
                    blitEncoderQHead.endEncoding()

                    // 2. Extract K_repeated[h] -> kvSliceHeadBuffer (reuse buffer)
                    // TODO: Replace this loop with a strided Blit copy or dedicated kernel
                    guard let blitEncoderKHead = commandBuffer.makeBlitCommandEncoder() else {
                        return nil
                    }
                    blitEncoderKHead.label = "Extract K H\(h) \(layerLabel)"
                    for t in 0..<currentSeqLen {
                        let srcOffset = (t * nHeads * headDim + h * headDim) * f16Size
                        let dstOffset = (t * headDim) * f16Size
                        blitEncoderKHead.copy(
                            from: kRepeated, sourceOffset: srcOffset, to: kvSliceHeadBuffer,
                            destinationOffset: dstOffset, size: headSizeBytes)
                    }
                    blitEncoderKHead.endEncoding()

                    // 3. Calculate Q[h] @ K_slice[h]^T * scale -> scoreSliceBuffer
                    success = metalService.encodeMPSMatrixMultiply(
                        commandBuffer: commandBuffer,
                        inputA: qHeadBuffer,  // Layout [1, headDim]
                        inputB: kvSliceHeadBuffer,  // Layout [currentSeqLen, headDim]
                        outputC: scoreSliceBuffer,  // Layout [1, currentSeqLen]
                        rowsA: 1, colsA: headDim,
                        rowsB: currentSeqLen, colsB: headDim,
                        transposeA: false, transposeB: true,  // Q * K^T
                        alpha: Double(scale), beta: 0.0,  // Apply scale
                        label: "ScoreMatMul H\(h) \(layerLabel)"
                    )
                    guard success else {
                        print("Error encoding Score MatMul H\(h) \(layerLabel)")
                        return nil
                    }

                    // 4. Copy result scoreSliceBuffer into the correct row of attnScores
                    guard let blitEncoderScore = commandBuffer.makeBlitCommandEncoder() else {
                        return nil
                    }
                    blitEncoderScore.label = "Copy Score H\(h) \(layerLabel)"
                    let scoreDestOffset = h * currentSeqLen * f16Size  // Offset for head h in flat attnScores
                    blitEncoderScore.copy(
                        from: scoreSliceBuffer, sourceOffset: 0, to: attnScores,
                        destinationOffset: scoreDestOffset, size: scoreSliceSizeBytes)
                    blitEncoderScore.endEncoding()
                }
                print("      Finished Encoding Attention Scores.")

                // --- h. Masking (Implicit) --- Handled by currentSeqLen

                // --- i. Softmax (MPS) ---
                // Input: attnScores (shape [nHead, currentSeqLen])
                // Output: attnScores (in-place)
                success = metalService.encodeMPSSoftMax(
                    commandBuffer: commandBuffer, inputMatrixBuffer: attnScores,
                    outputMatrixBuffer: attnScores,  // In-place works
                    rows: nHeads, columns: currentSeqLen, label: "Softmax \(layerLabel)")
                guard success else {
                    print("Error encoding Softmax \(layerLabel)")
                    return nil
                }
                print("      Encoded Softmax.")

                // --- j. Scores @ V (MatMul) ---
                // Input Scores/Probs: attnScores [nHead, currentSeqLen] -> Use scoreSliceBuffer for Probs[h]
                // Input V_repeated: [currentSeqLen, nHead, headDim] -> Use kvSliceHeadBuffer for V_slice[h]
                // Output: attnOutputBuffer [1, nHead * headDim] -> Use qHeadBuffer for Output[h]

                print("      Encoding Attention Values (Looping \(nHeads) heads)...")
                // Zero out the final output buffer first (important if accumulating, though not needed here with beta=0 MatMuls)
                guard let blitEncoderZeroAttnOut = commandBuffer.makeBlitCommandEncoder() else {
                    return nil
                }
                blitEncoderZeroAttnOut.fill(
                    buffer: attnOutputBuffer, range: 0..<attnOutputBuffer.length, value: 0)  // Use 0 for F16
                blitEncoderZeroAttnOut.endEncoding()

                for h in 0..<nHeads {
                    // 1. Extract Probs for head h -> scoreSliceBuffer
                    guard let blitEncoderProbHead = commandBuffer.makeBlitCommandEncoder() else {
                        return nil
                    }
                    blitEncoderProbHead.label = "Extract Probs H\(h) \(layerLabel)"
                    let probSourceOffset = h * currentSeqLen * f16Size
                    blitEncoderProbHead.copy(
                        from: attnScores, sourceOffset: probSourceOffset, to: scoreSliceBuffer,
                        destinationOffset: 0, size: scoreSliceSizeBytes)
                    blitEncoderProbHead.endEncoding()

                    // 2. Extract V_repeated for head h -> kvSliceHeadBuffer (reuse buffer)
                    // TODO: Replace this loop with a strided Blit copy or dedicated kernel
                    guard let blitEncoderVHead = commandBuffer.makeBlitCommandEncoder() else {
                        return nil
                    }
                    blitEncoderVHead.label = "Extract V H\(h) \(layerLabel)"
                    for t in 0..<currentSeqLen {
                        let srcOffset = (t * nHeads * headDim + h * headDim) * f16Size
                        let dstOffset = (t * headDim) * f16Size
                        blitEncoderVHead.copy(
                            from: vRepeated, sourceOffset: srcOffset, to: kvSliceHeadBuffer,
                            destinationOffset: dstOffset, size: headSizeBytes)
                    }
                    blitEncoderVHead.endEncoding()

                    // 3. Calculate Probs[h] @ V_slice[h] -> qHeadBuffer (reuse buffer for output head)
                    //    Probs[h] is [1, currentSeqLen]
                    //    V_slice[h] is [currentSeqLen, headDim]
                    //    Result is [1, headDim]
                    success = metalService.encodeMPSMatrixMultiply(
                        commandBuffer: commandBuffer,
                        inputA: scoreSliceBuffer,  // Layout [1, currentSeqLen]
                        inputB: kvSliceHeadBuffer,  // Layout [currentSeqLen, headDim]
                        outputC: qHeadBuffer,  // Layout [1, headDim]
                        rowsA: 1, colsA: currentSeqLen,
                        rowsB: currentSeqLen, colsB: headDim,
                        transposeA: false, transposeB: false,  // Probs * V
                        alpha: 1.0, beta: 0.0,
                        label: "ValueMatMul H\(h) \(layerLabel)"
                    )
                    guard success else {
                        print("Error encoding Value MatMul H\(h) \(layerLabel)")
                        return nil
                    }

                    // 4. Copy result qHeadBuffer into the correct slice of attnOutputBuffer
                    guard let blitEncoderAttnOut = commandBuffer.makeBlitCommandEncoder() else {
                        return nil
                    }
                    blitEncoderAttnOut.label = "Copy AttnOutput H\(h) \(layerLabel)"
                    let attnDestOffset = h * headSizeBytes  // Offset for head h in flat attnOutputBuffer
                    blitEncoderAttnOut.copy(
                        from: qHeadBuffer, sourceOffset: 0, to: attnOutputBuffer,
                        destinationOffset: attnDestOffset, size: headSizeBytes)
                    blitEncoderAttnOut.endEncoding()
                }
                print("      Finished Encoding Attention Values.")

                // --- k. Output Projection (MatMul) ---
                // Input: attnOutputBuffer [1, embedDim], Weight: oWeight [embedDim, embedDim] -> Output: attnProjBuffer [1, embedDim]
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: attnOutputBuffer, inputB: model.blocks[layerIndex].attention.oWeight,
                    outputC: attnProjBuffer,
                    rowsA: 1, colsA: embeddingDim,
                    rowsB: embeddingDim, colsB: embeddingDim,
                    transposeA: false, transposeB: false,
                    label: "Attn_O_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Attn Output Proj \(layerLabel)")
                    return nil
                }
                print("      Encoded Attn Output Projection.")

                // --- l. Residual Add 1 ---
                // Input A: residual1Buffer, Input B: attnProjBuffer -> Output: hiddenStateBuffer
                success = metalService.applyElementWiseAdd(
                    inputBufferA: residual1Buffer, inputBufferB: attnProjBuffer,
                    outputBufferC: hiddenStateBuffer,
                    elementCount: embeddingDim, commandBuffer: commandBuffer)
                guard success else {
                    print("Error encoding Residual Add 1 \(layerLabel)")
                    return nil
                }
                print("      Encoded Residual Add 1.")

                // --- Save input for residual connection 2 ---
                guard let blitEncoderRes2 = commandBuffer.makeBlitCommandEncoder() else {
                    return nil
                }
                // ... copy hiddenStateBuffer to residual2Buffer ...
                blitEncoderRes2.endEncoding()

                // --- m. FFN RMSNorm (Pre-FFN Norm) ---
                success = metalService.encodeRMSNormF16(
                    commandBuffer: commandBuffer, inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.blocks[layerIndex].ffnNormWeight, outputBuffer: normBuffer2,
                    rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                    label: "PreFFNNorm \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Pre-FFN RMSNorm \(layerLabel)")
                    return nil
                }
                print("      Encoded Pre-FFN RMSNorm.")

                // --- n. MLP / SwiGLU ---
                // gate = normBuffer2[1, embedDim] * Wgate[embedDim, hiddenDim] -> ffnGateBuffer[1, hiddenDim]
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: normBuffer2, inputB: model.blocks[layerIndex].mlp.gateWeight,
                    outputC: ffnGateBuffer,
                    rowsA: 1, colsA: embeddingDim,  // Input A layout
                    rowsB: embeddingDim, colsB: hiddenDim,  // Input B (Weight) layout
                    transposeA: false, transposeB: false,  // Math operation
                    label: "FFN_Gate_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding FFN Gate Proj \(layerLabel)")
                    return nil
                }

                // up = normBuffer2[1, embedDim] * Wup[embedDim, hiddenDim] -> ffnUpBuffer[1, hiddenDim]
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: normBuffer2, inputB: model.blocks[layerIndex].mlp.upWeight,
                    outputC: ffnUpBuffer,
                    rowsA: 1, colsA: embeddingDim,  // Input A layout
                    rowsB: embeddingDim, colsB: hiddenDim,  // Input B (Weight) layout
                    transposeA: false, transposeB: false,  // Math operation
                    label: "FFN_Up_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding FFN Up Proj \(layerLabel)")
                    return nil
                }

                // SiLU(gate) -> ffnGateBuffer (in-place)
                success = metalService.applySILU(
                    inputBuffer: ffnGateBuffer,
                    outputBuffer: ffnGateBuffer,  // In-place
                    elementCount: hiddenDim,  // Use hiddenDim
                    commandBuffer: commandBuffer
                )
                guard success else {
                    print("Error applying SiLU \(layerLabel)")
                    return nil
                }

                // SiLU(gate) * up -> ffnUpBuffer (reuse ffnUpBuffer)
                success = metalService.applyElementWiseMul(
                    inputBufferA: ffnGateBuffer,  // Input is SiLU result
                    inputBufferB: ffnUpBuffer,  // Input is Up projection result
                    outputBufferC: ffnUpBuffer,  // Output overwrites Up projection result
                    elementCount: hiddenDim,  // Use hiddenDim
                    commandBuffer: commandBuffer
                )
                guard success else {
                    print("Error applying ElemWise Mul \(layerLabel)")
                    return nil
                }

                // Down = ffnUpBuffer[1, hiddenDim] * Wdown[hiddenDim, embedDim] -> ffnDownBuffer[1, embedDim]
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: ffnUpBuffer, inputB: model.blocks[layerIndex].mlp.downWeight,
                    outputC: ffnDownBuffer,
                    rowsA: 1, colsA: hiddenDim,  // Input A layout
                    rowsB: hiddenDim, colsB: embeddingDim,  // Input B (Weight) layout
                    transposeA: false, transposeB: false,  // Math operation
                    label: "FFN_Down_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding FFN Down Proj \(layerLabel)")
                    return nil
                }
                print("      Encoded MLP/SwiGLU.")

                // --- o. Residual Add 2 ---
                // Input A: residual2Buffer, Input B: ffnDownBuffer -> Output: hiddenStateBuffer
                success = metalService.applyElementWiseAdd(
                    inputBufferA: residual2Buffer, inputBufferB: ffnDownBuffer,
                    outputBufferC: hiddenStateBuffer,
                    elementCount: embeddingDim, commandBuffer: commandBuffer)
                guard success else {
                    print("Error encoding Residual Add 2 \(layerLabel)")
                    return nil
                }
                print("      Encoded Residual Add 2.")

            }  // End layer loop
            print("    Finished layer loop.")

            // --- 5. Final RMSNorm ---
            // Input: hiddenStateBuffer, Output: normBuffer1 (reuse)
            success = metalService.encodeRMSNormF16(
                commandBuffer: commandBuffer, inputBuffer: hiddenStateBuffer,
                weightBuffer: model.finalNormWeight, outputBuffer: normBuffer1,
                rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                label: "FinalNorm"
            )
            guard success else {
                print("Error encoding Final RMSNorm")
                return nil
            }
            print("  Encoded Final RMSNorm.")

            // --- 6. Output Projection (Logits) ---
            // Input: normBuffer1[1, embedDim], Weight: outputWeight[embedDim, vocabSize] -> Output: logitsBuffer[1, vocabSize]
            success = metalService.encodeMPSMatrixMultiply(
                commandBuffer: commandBuffer,
                inputA: normBuffer1, inputB: model.outputWeight, outputC: logitsBuffer,
                rowsA: 1, colsA: embeddingDim,  // Input A layout
                rowsB: embeddingDim, colsB: vocabSize,  // Input B (Weight) layout
                transposeA: false, transposeB: false,  // Math operation
                label: "Output Projection"
            )
            guard success else {
                print("Error encoding Output Projection")
                return nil
            }
            print("  Encoded Output Projection.")

            // --- 7. Commit and Wait ---
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // --- 8. Check for Errors ---
            if let error = commandBuffer.error {
                print("Error [Runner.forward]: Command buffer execution failed: \(error)")
                return nil
            }
            print("  Command buffer completed successfully for pos \(pos).")

            // --- 9. Increment Position ---
            currentPosition += 1

            // --- 10. Return logits buffer ---
            return logitsBuffer
        }  // End autoreleasepool
    }
}
