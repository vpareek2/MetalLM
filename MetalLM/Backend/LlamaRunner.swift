// MetalLM/Backend/LlamaRunner.swift

import Foundation
import Metal

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

    init(model: LlamaModel, metalService: MetalService) throws {
        self.model = model
        self.config = model.config
        self.metalService = metalService
        self.maxSequenceLength = model.config.sequenceLength

        let kvCacheElementCountPerBuffer =
            config.numLayers * config.sequenceLength * config.numKeyValueHeads * config.headDim
        let kvCacheSizeBytes = kvCacheElementCountPerBuffer * MemoryLayout<Float16>.stride

        guard kvCacheSizeBytes > 0 else { throw LlamaRunnerError.kvCacheAllocationFailed }

        let options: MTLResourceOptions = .storageModePrivate
        print(
            "Attempting to allocate KV Cache (Private Storage): \(kvCacheSizeBytes * 2 / (1024*1024)) MB total..."
        )

        var tempCacheK: MTLBuffer? = metalService.device.makeBuffer(
            length: kvCacheSizeBytes, options: options)
        var tempCacheV: MTLBuffer? = metalService.device.makeBuffer(
            length: kvCacheSizeBytes, options: options)

        if tempCacheK == nil || tempCacheV == nil {
            print("Warning: Failed to allocate KV Cache with Private storage. Trying Shared...")
            let sharedOptions: MTLResourceOptions = .storageModeShared
            tempCacheK = metalService.device.makeBuffer(
                length: kvCacheSizeBytes, options: sharedOptions)
            tempCacheV = metalService.device.makeBuffer(
                length: kvCacheSizeBytes, options: sharedOptions)
            guard tempCacheK != nil, tempCacheV != nil else {
                throw LlamaRunnerError.kvCacheAllocationFailed
            }
            print("KV Cache allocated with Shared storage.")
        } else {
            print("KV Cache allocated successfully (Private storage).")
        }

        self.kvCacheK = tempCacheK!
        self.kvCacheV = tempCacheV!
        self.kvCacheK.label =
            "KV_Cache_K (L:\(config.numLayers) S:\(config.sequenceLength) KVH:\(config.numKeyValueHeads) HD:\(config.headDim))"
        self.kvCacheV.label =
            "KV_Cache_V (L:\(config.numLayers) S:\(config.sequenceLength) KVH:\(config.numKeyValueHeads) HD:\(config.headDim))"
        self.currentPosition = 0
    }

    func resetState() {
        currentPosition = 0
        print("LlamaRunner state reset (position = 0).")
    }

    // MARK: - Forward Pass (Debug prints removed)

    func forward(tokenID: Int) -> MTLBuffer? {
        let pos = currentPosition

        return autoreleasepool { () -> MTLBuffer? in
            var success: Bool = false  // Declare once

            // --- Input Validation ---
            guard pos < maxSequenceLength else {
                print("Error: KV Cache full")
                return nil
            }
            guard tokenID >= 0 && tokenID < config.vocabSize else {
                print("Error: Invalid token ID")
                return nil
            }

            print("Forward pass for token \(tokenID) at position \(pos)...")

            // --- 1. Create Command Buffer ---
            guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                print("Error: No command buffer")
                return nil
            }
            commandBuffer.label = "Llama Forward Pass CB (Pos: \(pos))"

            // --- 2. Allocate Temporary Buffers ---
            let options: MTLResourceOptions = .storageModeShared  // Use Shared for easier debugging initially
            let embeddingDim = config.embeddingDim
            let hiddenDim = config.hiddenDim
            let headDim = config.headDim
            let nHeads = config.numHeads
            let nKVHeads = config.numKeyValueHeads
            let vocabSize = config.vocabSize
            let f16Size = MemoryLayout<Float16>.stride
            let hiddenStateSizeBytes = embeddingDim * f16Size
            let qSizeBytes = embeddingDim * f16Size
            let kvSizeBytes = nKVHeads * headDim * f16Size
            let ffnHiddenSizeBytes = hiddenDim * f16Size
            let logitsSizeBytes = vocabSize * f16Size  // Using F16 logits for now
            // print("  NOTE: Using F16 for logits buffer in this pass.") // Keep if desired

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
            else {
                print("Error: Failed to allocate temporary buffers")
                return nil as MTLBuffer? // Corrected return
            }

            // (Assign labels - this is useful for Frame Capture!)
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

            // --- 3. Embedding Lookup ---
            guard let blitEncoderEmbed = commandBuffer.makeBlitCommandEncoder() else {
                print("Error: No embedding blit encoder")
                return nil as MTLBuffer? // Corrected return
            }
            blitEncoderEmbed.label = "Embedding Blit Encoder \(pos)"
            let embeddingOffset = tokenID * embeddingDim * f16Size
            guard model.tokenEmbeddings.length >= embeddingOffset + hiddenStateSizeBytes else {
                print("Error: Embedding offset out of bounds")
                blitEncoderEmbed.endEncoding()
                return nil as MTLBuffer? // Corrected return
            }
            blitEncoderEmbed.copy(
                from: model.tokenEmbeddings, sourceOffset: embeddingOffset, to: hiddenStateBuffer,
                destinationOffset: 0, size: hiddenStateSizeBytes)
            blitEncoderEmbed.endEncoding()
            print("  Encoded Embedding Lookup.")

            // --- 4. Loop through Layers ---
            let layerCountForDebug = config.numLayers // Process all layers now
            print("    Processing \(layerCountForDebug) layer(s)...")
            for layerIndex in 0..<layerCountForDebug {
                let layerLabel = "L\(layerIndex) P\(pos)"
                print("    Processing \(layerLabel)...")

                // --- Save Residual 1 ---
                guard let blitEncoderRes1 = commandBuffer.makeBlitCommandEncoder() else {
                    return nil as MTLBuffer?
                }
                blitEncoderRes1.label = "Save Residual 1 \(layerLabel)"
                blitEncoderRes1.copy(
                    from: hiddenStateBuffer, sourceOffset: 0, to: residual1Buffer,
                    destinationOffset: 0, size: hiddenStateSizeBytes)
                blitEncoderRes1.endEncoding()

                // --- a. Pre-Attention RMSNorm ---
                success = metalService.encodeRMSNormF16(
                    commandBuffer: commandBuffer, inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.blocks[layerIndex].attentionNormWeight,
                    outputBuffer: normBuffer1, rowCount: 1, elementCountPerRow: embeddingDim,
                    eps: config.rmsNormEps, label: "PreAttnNorm \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Pre-Attn RMSNorm \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded Pre-Attn RMSNorm.")

                // --- b. QKV Projection ---
                let kvDim = nKVHeads * headDim
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.qWeight, outputC: qBuffer,
                    rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: embeddingDim,
                    label: "Q_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Q Proj \(layerLabel)")
                    return nil as MTLBuffer?
                }
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.kWeight, outputC: kBuffer,
                    rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: kvDim,
                    label: "K_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding K Proj \(layerLabel)")
                    return nil as MTLBuffer?
                }
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.vWeight, outputC: vBuffer,
                    rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: kvDim,
                    label: "V_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding V Proj \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded QKV Projections.")

                // --- c. RoPE ---
                success = metalService.applyRoPE(
                    commandBuffer: commandBuffer, buffer: qBuffer,
                    ropeFrequencies: model.ropeFrequencies, config: config, posOffset: pos,
                    sequenceLength: 1, numHeads: nHeads, headDim: headDim)
                guard success else {
                    print("Error applying RoPE to Q \(layerLabel)")
                    return nil as MTLBuffer?
                }
                success = metalService.applyRoPE(
                    commandBuffer: commandBuffer, buffer: kBuffer,
                    ropeFrequencies: model.ropeFrequencies, config: config, posOffset: pos,
                    sequenceLength: 1, numHeads: nKVHeads, headDim: headDim)
                guard success else {
                    print("Error applying RoPE to K \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded RoPE.")

                // --- d. KV Cache Update ---
                guard let blitEncoderKV = commandBuffer.makeBlitCommandEncoder() else {
                    return nil as MTLBuffer?
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
                    print("Error: KV Cache offset out of bounds")
                    blitEncoderKV.endEncoding()
                    return nil as MTLBuffer?
                }
                blitEncoderKV.copy(
                    from: kBuffer, sourceOffset: 0, to: kvCacheK,
                    destinationOffset: destinationOffsetK, size: bytesPerKVEntry)
                blitEncoderKV.copy(
                    from: vBuffer, sourceOffset: 0, to: kvCacheV,
                    destinationOffset: destinationOffsetV, size: bytesPerKVEntry)
                blitEncoderKV.endEncoding()
                print("      Encoded KV Cache Update.")

                // --- e/f/g/h/i/j Attention Calculation ---
                let currentSeqLen = pos + 1
                let scale = Float16(1.0 / sqrt(Float(config.headDim)))
                let kvSliceSizeBytes = currentSeqLen * nKVHeads * headDim * f16Size
                let repeatedKVSizeBytes = currentSeqLen * nHeads * headDim * f16Size
                let scoreSizeBytes = nHeads * currentSeqLen * f16Size
                let headSizeBytes = headDim * f16Size
                let kSliceHeadSizeBytes = currentSeqLen * headDim * f16Size
                let scoreSliceSizeBytes = currentSeqLen * f16Size

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
                        length: scoreSizeBytes, options: options),
                    let qHeadBuffer = metalService.device.makeBuffer(
                        length: headSizeBytes, options: options),
                    let kvSliceHeadBuffer = metalService.device.makeBuffer(
                        length: kSliceHeadSizeBytes, options: options),
                    let scoreSliceBuffer = metalService.device.makeBuffer(
                        length: scoreSliceSizeBytes, options: options)
                else {
                    print("Error allocating attention buffers")
                    return nil as MTLBuffer?
                }

                kSlice.label = "kSlice \(layerLabel)"
                vSlice.label = "vSlice \(layerLabel)"
                kRepeated.label = "kRepeated \(layerLabel)"
                vRepeated.label = "vRepeated \(layerLabel)"
                attnScores.label = "attnScores/Probs \(layerLabel)"
                qHeadBuffer.label = "qHeadBuffer \(layerLabel)"
                kvSliceHeadBuffer.label = "kvSliceHeadBuffer \(layerLabel)"
                scoreSliceBuffer.label = "scoreSliceBuffer \(layerLabel)"

                // --- e. Get K/V Slice ---
                guard let blitEncoderSlice = commandBuffer.makeBlitCommandEncoder() else {
                    return nil as MTLBuffer?
                }
                blitEncoderSlice.label = "KV Slice Blit \(layerLabel)"
                let sourceOffsetKV = layerOffsetBytes
                blitEncoderSlice.copy(
                    from: kvCacheK, sourceOffset: sourceOffsetKV, to: kSlice, destinationOffset: 0,
                    size: kvSliceSizeBytes)
                blitEncoderSlice.copy(
                    from: kvCacheV, sourceOffset: sourceOffsetKV, to: vSlice, destinationOffset: 0,
                    size: kvSliceSizeBytes)
                blitEncoderSlice.endEncoding()
                print("      Encoded Get K/V Slice.")

                // --- f. GQA Repeat Heads ---
                success = metalService.applyRepeatKVHeads(
                    sourceBuffer: kSlice, destinationBuffer: kRepeated, numKVHeads: nKVHeads,
                    numQueryGroups: config.numQueryGroups, headDim: headDim, seqLen: currentSeqLen,
                    commandBuffer: commandBuffer)
                guard success else {
                    print("Error repeating K heads \(layerLabel)")
                    return nil as MTLBuffer?
                }
                success = metalService.applyRepeatKVHeads(
                    sourceBuffer: vSlice, destinationBuffer: vRepeated, numKVHeads: nKVHeads,
                    numQueryGroups: config.numQueryGroups, headDim: headDim, seqLen: currentSeqLen,
                    commandBuffer: commandBuffer)
                guard success else {
                    print("Error repeating V heads \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded GQA Repeat Heads.")

                // --- g. Attention Scores ---
                print("      Encoding Attention Scores (Looping \(nHeads) heads)...")
                for h in 0..<nHeads {
                    guard let blitEncoderQHead = commandBuffer.makeBlitCommandEncoder() else {
                        return nil as MTLBuffer?
                    }
                    let qOffset = h * headSizeBytes
                    blitEncoderQHead.copy(
                        from: qBuffer, sourceOffset: qOffset, to: qHeadBuffer, destinationOffset: 0,
                        size: headSizeBytes)
                    blitEncoderQHead.endEncoding()

                    guard let blitEncoderKHead = commandBuffer.makeBlitCommandEncoder() else {
                        return nil as MTLBuffer?
                    }
                    blitEncoderKHead.label = "Extract K H\(h) \(layerLabel)"
                    for t in 0..<currentSeqLen
                    { /* ... Blit K_repeated[h] -> kvSliceHeadBuffer ... */
                        let srcOffset = (t * nHeads * headDim + h * headDim) * f16Size
                        let dstOffset = (t * headDim) * f16Size
                        blitEncoderKHead.copy(
                            from: kRepeated, sourceOffset: srcOffset, to: kvSliceHeadBuffer,
                            destinationOffset: dstOffset, size: headSizeBytes)
                    }
                    blitEncoderKHead.endEncoding()

                    success = metalService.encodeMPSMatrixMultiply(
                        commandBuffer: commandBuffer, inputA: qHeadBuffer,
                        inputB: kvSliceHeadBuffer, outputC: scoreSliceBuffer,
                        rowsA: 1, colsA: headDim, rowsB: currentSeqLen, colsB: headDim,
                        transposeA: false, transposeB: true, alpha: Double(scale), beta: 0.0,
                        label: "ScoreMatMul H\(h) \(layerLabel)"
                    )
                    guard success else {
                        print("Error encoding Score MatMul H\(h) \(layerLabel)")
                        return nil as MTLBuffer?
                    }

                    guard let blitEncoderScore = commandBuffer.makeBlitCommandEncoder() else {
                        return nil as MTLBuffer?
                    }
                    blitEncoderScore.label = "Copy Score H\(h) \(layerLabel)"
                    let scoreDestOffset = h * currentSeqLen * f16Size
                    blitEncoderScore.copy(
                        from: scoreSliceBuffer, sourceOffset: 0, to: attnScores,
                        destinationOffset: scoreDestOffset, size: scoreSliceSizeBytes)
                    blitEncoderScore.endEncoding()
                }
                print("      Finished Encoding Attention Scores.")

                // --- i. Softmax ---
                success = metalService.encodeMPSSoftMax(
                    commandBuffer: commandBuffer, inputMatrixBuffer: attnScores,
                    outputMatrixBuffer: attnScores, rows: nHeads, columns: currentSeqLen,
                    label: "Softmax \(layerLabel)")
                guard success else {
                    print("Error encoding Softmax \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded Softmax.")

                // --- j. Scores @ V ---
                print("      Encoding Attention Values (Looping \(nHeads) heads)...")
                guard let blitEncoderZeroAttnOut = commandBuffer.makeBlitCommandEncoder() else {
                    return nil as MTLBuffer?
                }
                blitEncoderZeroAttnOut.fill(
                    buffer: attnOutputBuffer, range: 0..<attnOutputBuffer.length, value: 0)
                blitEncoderZeroAttnOut.endEncoding()

                for h in 0..<nHeads {
                    guard let blitEncoderProbHead = commandBuffer.makeBlitCommandEncoder() else {
                        return nil as MTLBuffer?
                    }
                    blitEncoderProbHead.label = "Extract Probs H\(h) \(layerLabel)"
                    let probSourceOffset = h * currentSeqLen * f16Size
                    blitEncoderProbHead.copy(
                        from: attnScores, sourceOffset: probSourceOffset, to: scoreSliceBuffer,
                        destinationOffset: 0, size: scoreSliceSizeBytes)
                    blitEncoderProbHead.endEncoding()

                    guard let blitEncoderVHead = commandBuffer.makeBlitCommandEncoder() else {
                        return nil as MTLBuffer?
                    }
                    blitEncoderVHead.label = "Extract V H\(h) \(layerLabel)"
                    for t in 0..<currentSeqLen
                    { /* ... Blit V_repeated[h] -> kvSliceHeadBuffer ... */
                        let srcOffset = (t * nHeads * headDim + h * headDim) * f16Size
                        let dstOffset = (t * headDim) * f16Size
                        blitEncoderVHead.copy(
                            from: vRepeated, sourceOffset: srcOffset, to: kvSliceHeadBuffer,
                            destinationOffset: dstOffset, size: headSizeBytes)
                    }
                    blitEncoderVHead.endEncoding()

                    success = metalService.encodeMPSMatrixMultiply(
                        commandBuffer: commandBuffer, inputA: scoreSliceBuffer,
                        inputB: kvSliceHeadBuffer, outputC: qHeadBuffer,
                        rowsA: 1, colsA: currentSeqLen, rowsB: currentSeqLen, colsB: headDim,
                        transposeA: false, transposeB: false,
                        label: "ValueMatMul H\(h) \(layerLabel)"
                    )
                    guard success else {
                        print("Error encoding Value MatMul H\(h) \(layerLabel)")
                        return nil as MTLBuffer?
                    }

                    guard let blitEncoderAttnOut = commandBuffer.makeBlitCommandEncoder() else {
                        return nil as MTLBuffer?
                    }
                    blitEncoderAttnOut.label = "Copy AttnOutput H\(h) \(layerLabel)"
                    let attnDestOffset = h * headSizeBytes
                    blitEncoderAttnOut.copy(
                        from: qHeadBuffer, sourceOffset: 0, to: attnOutputBuffer,
                        destinationOffset: attnDestOffset, size: headSizeBytes)
                    blitEncoderAttnOut.endEncoding()
                }
                print("      Finished Encoding Attention Values.")

                // --- k. Output Projection ---
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: attnOutputBuffer,
                    inputB: model.blocks[layerIndex].attention.oWeight, outputC: attnProjBuffer,
                    rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: embeddingDim,
                    label: "Attn_O_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Attn Output Proj \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded Attn Output Projection.")

                // --- l. Residual Add 1 ---
                success = metalService.applyElementWiseAdd(
                    inputBufferA: residual1Buffer, inputBufferB: attnProjBuffer,
                    outputBufferC: hiddenStateBuffer, elementCount: embeddingDim,
                    commandBuffer: commandBuffer)
                guard success else {
                    print("Error encoding Residual Add 1 \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded Residual Add 1.")

                // --- Save Residual 2 ---
                guard let blitEncoderRes2 = commandBuffer.makeBlitCommandEncoder() else {
                    return nil as MTLBuffer?
                }
                blitEncoderRes2.label = "Save Residual 2 \(layerLabel)"
                blitEncoderRes2.copy(
                    from: hiddenStateBuffer, sourceOffset: 0, to: residual2Buffer,
                    destinationOffset: 0, size: hiddenStateSizeBytes)
                blitEncoderRes2.endEncoding()

                // --- m. Pre-FFN RMSNorm ---
                success = metalService.encodeRMSNormF16(
                    commandBuffer: commandBuffer, inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.blocks[layerIndex].ffnNormWeight, outputBuffer: normBuffer2,
                    rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                    label: "PreFFNNorm \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Pre-FFN RMSNorm \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded Pre-FFN RMSNorm.")

                // --- n. MLP / SwiGLU ---
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: normBuffer2, inputB: model.blocks[layerIndex].mlp.gateWeight,
                    outputC: ffnGateBuffer,
                    rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: hiddenDim,
                    label: "FFN_Gate_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding FFN Gate Proj \(layerLabel)")
                    return nil as MTLBuffer?
                }
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer,
                    inputA: normBuffer2, inputB: model.blocks[layerIndex].mlp.upWeight,
                    outputC: ffnUpBuffer,
                    rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: hiddenDim,
                    label: "FFN_Up_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding FFN Up Proj \(layerLabel)")
                    return nil as MTLBuffer?
                }

                success = metalService.applySILU(
                    inputBuffer: ffnGateBuffer, outputBuffer: ffnGateBuffer, elementCount: hiddenDim,
                    commandBuffer: commandBuffer)
                guard success else {
                    print("Error applying SiLU \(layerLabel)")
                    return nil as MTLBuffer?
                }

                success = metalService.applyElementWiseMul(
                    inputBufferA: ffnGateBuffer, inputBufferB: ffnUpBuffer,
                    outputBufferC: ffnUpBuffer, elementCount: hiddenDim, commandBuffer: commandBuffer
                )
                guard success else {
                    print("Error applying ElemWise Mul \(layerLabel)")
                    return nil as MTLBuffer?
                }

                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: ffnUpBuffer,
                    inputB: model.blocks[layerIndex].mlp.downWeight, outputC: ffnDownBuffer,
                    rowsA: 1, colsA: hiddenDim, rowsB: hiddenDim, colsB: embeddingDim,
                    label: "FFN_Down_Proj \(layerLabel)"
                )
                guard success else {
                    print("Error encoding FFN Down Proj \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded MLP/SwiGLU.")

                // --- o. Residual Add 2 ---
                success = metalService.applyElementWiseAdd(
                    inputBufferA: residual2Buffer, inputBufferB: ffnDownBuffer,
                    outputBufferC: hiddenStateBuffer, elementCount: embeddingDim,
                    commandBuffer: commandBuffer)
                guard success else {
                    print("Error encoding Residual Add 2 \(layerLabel)")
                    return nil as MTLBuffer?
                }
                print("      Encoded Residual Add 2.")

            }  // End layer loop
            print("    Finished layer loop.")

            // --- 5. Final RMSNorm ---
            success = metalService.encodeRMSNormF16(
                commandBuffer: commandBuffer, inputBuffer: hiddenStateBuffer,
                weightBuffer: model.finalNormWeight, outputBuffer: normBuffer1,
                rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                label: "FinalNorm"
            )
            guard success else {
                print("Error encoding Final RMSNorm")
                return nil as MTLBuffer?
            }
            print("  Encoded Final RMSNorm.")

            // --- 6. Output Projection (Logits) ---
            success = metalService.encodeMPSMatrixMultiply(
                commandBuffer: commandBuffer, inputA: normBuffer1, inputB: model.outputWeight,
                outputC: logitsBuffer,
                rowsA: 1, colsA: embeddingDim, rowsB: embeddingDim, colsB: vocabSize,
                label: "Output Projection"
            )
            guard success else {
                print("Error encoding Output Projection")
                return nil as MTLBuffer?
            }
            print("  Encoded Output Projection.")

            // --- 7. Commit and Wait ---
            print("  Committing final command buffer...")
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // --- 8. Check for Errors ---
            if let error = commandBuffer.error {
                print("Error [Runner.forward]: Command buffer execution failed: \(error)")
                return nil  // No cast needed here, final optional path
            }
            print("  Command buffer completed successfully for pos \(pos).")

            // --- 9. Increment Position ---
            currentPosition += 1

            // --- 10. Return logits buffer ---
            return logitsBuffer
        }  // End autoreleasepool
    }
}
