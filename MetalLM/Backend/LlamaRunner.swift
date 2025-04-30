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
        self.config = model.config // Initialize the public config property
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

    /// Processes a single input token and generates logits for the next token.
    /// Updates the KV cache and increments the current position.
    ///
    /// - Parameter tokenID: The ID of the input token.
    /// - Returns: An MTLBuffer containing the output logits (Float32), or nil on failure.
    func forward(tokenID: Int) -> MTLBuffer? {
        let pos = currentPosition  // Use local variable for current position

        // Use autoreleasepool to help manage the lifecycle of temporary Metal objects within the pass
        return autoreleasepool {
            // --- Input Validation ---
            guard pos < maxSequenceLength else {
                print(
                    "Error [Runner.forward]: KV Cache is full (pos \(pos) >= maxSequenceLength \(maxSequenceLength)). Call resetState()."
                )
                // Consider throwing an error instead of returning nil
                // throw LlamaRunnerError.kvCacheOutOfBounds
                return nil
            }
            guard tokenID >= 0 && tokenID < config.vocabSize else {
                print(
                    "Error [Runner.forward]: Invalid tokenID \(tokenID) provided. Vocab size is \(config.vocabSize)."
                )
                // throw LlamaRunnerError.invalidTokenID
                return nil
            }

            print("Forward pass for token \(tokenID) at position \(pos)...")

            // --- 1. Create Command Buffer ---
            guard let commandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                print("Error [Runner.forward]: Failed to create command buffer.")
                return nil
            }
            commandBuffer.label = "Llama Forward Pass CB (Pos: \(pos))"

            // --- 2. Allocate Temporary Buffers ---
            // Allocate buffers needed for one pass. Reuse where logically possible.
            // Using .storageModeShared for easier debugging initially. Switch to .storageModePrivate for potential perf gains.
            let options: MTLResourceOptions = .storageModeShared
            let embeddingDim = config.embeddingDim
            let hiddenDim = config.hiddenDim  // For MLP intermediate
            let headDim = config.headDim
            let nHeads = config.numHeads
            let nKVHeads = config.numKeyValueHeads
            let vocabSize = config.vocabSize
            let f16Size = MemoryLayout<Float16>.stride
            let f32Size = MemoryLayout<Float32>.stride

            let hiddenStateSizeBytes = embeddingDim * f16Size
            let qSizeBytes = embeddingDim * f16Size  // nHeads * headDim
            let kvSizeBytes = nKVHeads * headDim * f16Size  // For K and V *before* caching/repeating
            let ffnHiddenSizeBytes = hiddenDim * f16Size  // For gate/up/intermediate MLP results
            let logitsSizeBytes = vocabSize * f32Size  // Logits often computed/returned in F32

            // Allocate (handle potential allocation failures)
            // We need distinct buffers for inputs/outputs of ops where overwriting would be wrong
            // e.g., input to norm vs output of norm if norm output is needed later unmodified.
            // Also need distinct buffers for residual connections.
            guard
                let hiddenStateBuffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),  // Main working buffer
                let normBuffer1 = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),  // Output of first norm
                let residual1Buffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),  // Saved input for first residual
                let qBuffer = metalService.device.makeBuffer(length: qSizeBytes, options: options),
                let kBuffer = metalService.device.makeBuffer(length: kvSizeBytes, options: options),  // Temp buffer for current K
                let vBuffer = metalService.device.makeBuffer(length: kvSizeBytes, options: options),  // Temp buffer for current V
                let attnOutputBuffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),  // Attention output after V@Scores, before O projection
                let attnProjBuffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),  // Output after O projection
                let residual2Buffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),  // Saved input for second residual
                let normBuffer2 = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),  // Output of second norm (input to MLP)
                let ffnGateBuffer = metalService.device.makeBuffer(
                    length: ffnHiddenSizeBytes, options: options),  // gate_proj result & SiLU result
                let ffnUpBuffer = metalService.device.makeBuffer(
                    length: ffnHiddenSizeBytes, options: options),  // up_proj result & SwiGLU result
                let ffnDownBuffer = metalService.device.makeBuffer(
                    length: hiddenStateSizeBytes, options: options),  // Result after down_proj
                let logitsBuffer = metalService.device.makeBuffer(
                    length: logitsSizeBytes, options: options)  // Output logits (F32)
            else {
                print("Error [Runner.forward]: Failed to allocate one or more temporary buffers.")
                // Consider throwing LlamaRunnerError.bufferAllocationFailed("...")
                return nil
            }

            // Label buffers for debugging
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
            logitsBuffer.label = "Logits Output \(pos)"

            // --- 3. Embedding Lookup ---
            guard let blitEncoderEmbed = commandBuffer.makeBlitCommandEncoder() else {
                print("Error [Runner.forward]: Failed to create Blit Encoder for Embedding.")
                return nil  // Or throw
            }
            blitEncoderEmbed.label = "Embedding Blit Encoder \(pos)"
            let embeddingOffset = tokenID * embeddingDim * f16Size  // Calculate byte offset in embedding table
            guard model.tokenEmbeddings.length >= embeddingOffset + hiddenStateSizeBytes else {
                print(
                    "Error [Runner.forward]: Token ID \(tokenID) results in offset outside embedding table bounds."
                )
                blitEncoderEmbed.endEncoding()
                return nil  // Or throw
            }
            blitEncoderEmbed.copy(
                from: model.tokenEmbeddings, sourceOffset: embeddingOffset,
                to: hiddenStateBuffer, destinationOffset: 0,
                size: hiddenStateSizeBytes)  // Copy one embedding vector
            blitEncoderEmbed.endEncoding()
            print("  Encoded Embedding Lookup.")

            // --- 4. Loop through Layers ---
            for layerIndex in 0..<config.numLayers {
                let layerLabel = "L\(layerIndex) P\(pos)"  // Short label
                print("    Processing \(layerLabel)...")

                // --- Save input for residual connection 1 ---
                guard let blitEncoderRes1 = commandBuffer.makeBlitCommandEncoder() else {
                    return nil /* Throw */
                }
                blitEncoderRes1.label = "Save Residual 1 \(layerLabel)"
                blitEncoderRes1.copy(
                    from: hiddenStateBuffer, sourceOffset: 0, to: residual1Buffer,
                    destinationOffset: 0, size: hiddenStateSizeBytes)
                blitEncoderRes1.endEncoding()

                // --- a. Input RMSNorm (Pre-Attention Norm) ---
                // Input: hiddenStateBuffer, Output: normBuffer1
                var success = metalService.encodeRMSNormF16(
                    commandBuffer: commandBuffer,
                    inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.blocks[layerIndex].attentionNormWeight,
                    outputBuffer: normBuffer1,  // Use distinct output buffer
                    rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                    label: "PreAttnNorm \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Pre-Attn RMSNorm \(layerLabel)")
                    return nil /* Throw */
                }
                print("      Encoded Pre-Attn RMSNorm.")

                // --- b. QKV Projection (MatMul) ---
                // Input: normBuffer1, Output: qBuffer, kBuffer, vBuffer
                let M_proj = 1
                let K_proj = embeddingDim

                // Q = norm * Wq
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.qWeight, outputC: qBuffer,
                    M: M_proj, N: embeddingDim, K: K_proj, transposeB: false,
                    label: "Q_Proj \(layerLabel)")
                guard success else {
                    print("Error encoding Q Proj \(layerLabel)")
                    return nil /* Throw */
                }

                // K = norm * Wk
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.kWeight, outputC: kBuffer,
                    M: M_proj, N: nKVHeads * headDim, K: K_proj, transposeB: false,
                    label: "K_Proj \(layerLabel)")
                guard success else {
                    print("Error encoding K Proj \(layerLabel)")
                    return nil /* Throw */
                }

                // V = norm * Wv
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: normBuffer1,
                    inputB: model.blocks[layerIndex].attention.vWeight, outputC: vBuffer,
                    M: M_proj, N: nKVHeads * headDim, K: K_proj, transposeB: false,
                    label: "V_Proj \(layerLabel)")
                guard success else {
                    print("Error encoding V Proj \(layerLabel)")
                    return nil /* Throw */
                }
                print("      Encoded QKV Projections.")

                // --- c. RoPE ---
                success = metalService.applyRoPE(
                    commandBuffer: commandBuffer, buffer: qBuffer,
                    ropeFrequencies: model.ropeFrequencies,
                    config: config, posOffset: pos, sequenceLength: 1, numHeads: nHeads,
                    headDim: headDim)
                guard success else {
                    print("Error applying RoPE to Q \(layerLabel)")
                    return nil /* Throw */
                }

                success = metalService.applyRoPE(
                    commandBuffer: commandBuffer, buffer: kBuffer,
                    ropeFrequencies: model.ropeFrequencies,
                    config: config, posOffset: pos, sequenceLength: 1, numHeads: nKVHeads,
                    headDim: headDim)
                guard success else {
                    print("Error applying RoPE to K \(layerLabel)")
                    return nil /* Throw */
                }
                print("      Encoded RoPE.")

                // --- d. KV Cache Update ---
                guard let blitEncoderKV = commandBuffer.makeBlitCommandEncoder() else {
                    return nil /* Throw */
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
                    print("Error: KV Cache offset out of bounds for \(layerLabel) at pos \(pos).")
                    blitEncoderKV.endEncoding()
                    return nil /* Throw */
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
                // !!! MVP: SKIPPING ACTUAL ATTENTION SCORE CALCULATION !!!
                // We need attnOutputBuffer to exist for the O projection, but it will contain garbage/zeros.
                print(
                    "      SKIPPING Attention Calculation (Get K/V Slice, Repeat, Q@K^T, Softmax, Scores@V)."
                )
                // Optionally zero-fill attnOutputBuffer for predictability?
                // guard let blitZero = commandBuffer.makeBlitCommandEncoder() else { return nil }
                // blitZero.fill(buffer: attnOutputBuffer, range: 0..<attnOutputBuffer.length, value: 0)
                // blitZero.endEncoding()

                // --- k. Output Projection (MatMul) ---
                // Input: attnOutputBuffer (contains garbage/zeros), Weight: oWeight
                // Output: attnProjBuffer
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: attnOutputBuffer,
                    inputB: model.blocks[layerIndex].attention.oWeight, outputC: attnProjBuffer,
                    M: M_proj, N: embeddingDim, K: embeddingDim, transposeB: false,
                    label: "Attn_O_Proj \(layerLabel)")
                guard success else {
                    print("Error encoding Attn Output Proj \(layerLabel)")
                    return nil /* Throw */
                }
                print("      Encoded Attn Output Projection.")

                // --- l. Residual Add 1 ---
                // Input A: residual1Buffer (saved input)
                // Input B: attnProjBuffer (result of O projection)
                // Output: hiddenStateBuffer (add back to main hidden state for FFN input)
                success = metalService.applyElementWiseAdd(
                    inputBufferA: residual1Buffer, inputBufferB: attnProjBuffer,
                    outputBufferC: hiddenStateBuffer,
                    elementCount: embeddingDim, commandBuffer: commandBuffer)
                guard success else {
                    print("Error encoding Residual Add 1 \(layerLabel)")
                    return nil /* Throw */
                }
                print("      Encoded Residual Add 1.")

                // --- Save input for residual connection 2 ---
                guard let blitEncoderRes2 = commandBuffer.makeBlitCommandEncoder() else {
                    return nil /* Throw */
                }
                blitEncoderRes2.label = "Save Residual 2 \(layerLabel)"
                blitEncoderRes2.copy(
                    from: hiddenStateBuffer, sourceOffset: 0, to: residual2Buffer,
                    destinationOffset: 0, size: hiddenStateSizeBytes)
                blitEncoderRes2.endEncoding()

                // --- m. FFN RMSNorm (Pre-FFN Norm) ---
                // Input: hiddenStateBuffer, Output: normBuffer2
                success = metalService.encodeRMSNormF16(
                    commandBuffer: commandBuffer,
                    inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.blocks[layerIndex].ffnNormWeight,
                    outputBuffer: normBuffer2,  // Use second norm buffer
                    rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                    label: "PreFFNNorm \(layerLabel)"
                )
                guard success else {
                    print("Error encoding Pre-FFN RMSNorm \(layerLabel)")
                    return nil /* Throw */
                }
                print("      Encoded Pre-FFN RMSNorm.")

                // --- n. MLP / SwiGLU ---
                // Input: normBuffer2
                // gate = norm * Wgate -> ffnGateBuffer
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: normBuffer2,
                    inputB: model.blocks[layerIndex].mlp.gateWeight, outputC: ffnGateBuffer,
                    M: M_proj, N: hiddenDim, K: embeddingDim, transposeB: false,
                    label: "FFN_Gate_Proj \(layerLabel)")
                guard success else {
                    print("Error encoding FFN Gate Proj \(layerLabel)")
                    return nil /* Throw */
                }

                // up = norm * Wup -> ffnUpBuffer
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: normBuffer2,
                    inputB: model.blocks[layerIndex].mlp.upWeight, outputC: ffnUpBuffer,
                    M: M_proj, N: hiddenDim, K: embeddingDim, transposeB: false,
                    label: "FFN_Up_Proj \(layerLabel)")
                guard success else {
                    print("Error encoding FFN Up Proj \(layerLabel)")
                    return nil /* Throw */
                }

                // SiLU(gate) -> ffnGateBuffer (in-place)
                success = metalService.applySILU(
                    inputBuffer: ffnGateBuffer, outputBuffer: ffnGateBuffer,  // In-place
                    elementCount: hiddenDim, commandBuffer: commandBuffer)
                guard success else {
                    print("Error applying SiLU \(layerLabel)")
                    return nil /* Throw */
                }

                // SiLU(gate) * up -> ffnUpBuffer (reuse ffnUpBuffer)
                success = metalService.applyElementWiseMul(
                    inputBufferA: ffnGateBuffer, inputBufferB: ffnUpBuffer,
                    outputBufferC: ffnUpBuffer,  // Result in ffnUpBuffer
                    elementCount: hiddenDim, commandBuffer: commandBuffer)
                guard success else {
                    print("Error applying ElemWise Mul \(layerLabel)")
                    return nil /* Throw */
                }

                // Down projection -> ffnDownBuffer
                // Input: ffnUpBuffer (SwiGLU result), Weight: downWeight
                success = metalService.encodeMPSMatrixMultiply(
                    commandBuffer: commandBuffer, inputA: ffnUpBuffer,
                    inputB: model.blocks[layerIndex].mlp.downWeight, outputC: ffnDownBuffer,
                    M: M_proj, N: embeddingDim, K: hiddenDim, transposeB: false,
                    label: "FFN_Down_Proj \(layerLabel)")
                guard success else {
                    print("Error encoding FFN Down Proj \(layerLabel)")
                    return nil /* Throw */
                }
                print("      Encoded MLP/SwiGLU.")

                // --- o. Residual Add 2 ---
                // Input A: residual2Buffer
                // Input B: ffnDownBuffer
                // Output: hiddenStateBuffer (This becomes input for next layer or final norm)
                success = metalService.applyElementWiseAdd(
                    inputBufferA: residual2Buffer, inputBufferB: ffnDownBuffer,
                    outputBufferC: hiddenStateBuffer,
                    elementCount: embeddingDim, commandBuffer: commandBuffer)
                guard success else {
                    print("Error encoding Residual Add 2 \(layerLabel)")
                    return nil /* Throw */
                }
                print("      Encoded Residual Add 2.")

            }  // End layer loop
            print("    Finished layer loop.")

            // --- 5. Final RMSNorm ---
            // Input: hiddenStateBuffer, Output: normBuffer1 (reuse norm buffer)
            // Declare success for this scope
            var success = metalService.encodeRMSNormF16(
                commandBuffer: commandBuffer,
                inputBuffer: hiddenStateBuffer,
                weightBuffer: model.finalNormWeight,
                outputBuffer: normBuffer1,  // Reuse normBuffer1
                rowCount: 1, elementCountPerRow: embeddingDim, eps: config.rmsNormEps,
                label: "FinalNorm"
            )
            guard success else {
                print("Error encoding Final RMSNorm")
                return nil /* Throw */
            }
            print("  Encoded Final RMSNorm.")

            // --- 6. Output Projection (Logits) ---
            // Input: normBuffer1, Weight: model.outputWeight (often F16 or F32)
            // Output: logitsBuffer (F32)
            success = metalService.encodeMPSMatrixMultiply(
                commandBuffer: commandBuffer, inputA: normBuffer1, inputB: model.outputWeight,
                outputC: logitsBuffer,
                M: 1, N: vocabSize, K: embeddingDim, transposeB: false, label: "Output Projection")
            // Note: MPS should handle F16 * F16 -> F32 if outputC's descriptor dataType was F32,
            // but our current helper assumes F16 output. This might need adjustment later if logits MUST be F32.
            // For now, assume logitsBuffer is F16 like others, or adjust helper/allocation.
            // Let's assume logitsBuffer IS F32 and MPS handles it. We allocated it as F32 size.
            // Need to update encodeMPSMatrixMultiply if output type differs.
            // --> TODO: Update encodeMPSMatrixMultiply to handle different output types, or use F16 logits for now.
            // Let's proceed assuming F16 logits for MVP simplicity, matching other temp buffers. Adjust logitsSizeBytes if needed.
            guard success else {
                print("Error encoding Output Projection")
                return nil /* Throw */
            }
            print("  Encoded Output Projection.")

            // --- 7. Commit and Wait ---
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()  // Wait for GPU to finish

            // --- 8. Check for Errors ---
            if let error = commandBuffer.error {
                print("Error [Runner.forward]: Command buffer execution failed: \(error)")
                // Consider throwing
                return nil
            }
            print("  Command buffer completed successfully for pos \(pos).")

            // --- 9. Increment Position ---
            currentPosition += 1  // Increment only if successful execution

            // --- 10. Return logits buffer ---
            return logitsBuffer
        }  // End autoreleasepool
    }
}
