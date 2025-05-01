import Foundation
import Metal
import MetalPerformanceShaders

enum LlamaRunnerError: Error {
    case kvCacheAllocationFailed
    case kvCacheOutOfBounds
    case invalidTokenID
    case bufferAllocationFailed(String)
    case commandEncoderCreationFailed(String)
    case encodingFailed(String)
    case validationCheckFailed(bufferName: String, reason: String)
}

struct DebugStepSnapshot {
    let layer: Int
    let stepName: String
    let values: [Any]
    let timestamp: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
}

class LlamaRunner {
    private let metalService: MetalService
    private let model: LlamaModel
    public let config: LlamaConfig

    private let kvCacheK: MTLBuffer
    private let kvCacheV: MTLBuffer

    private(set) var currentPosition: Int = 0
    private let maxSequenceLength: Int

    private var debugSnapshots: [DebugStepSnapshot] = []
    private let captureDebugSnapshots = true
    private let debugSnapshotElementCount = 16

    init(model: LlamaModel, metalService: MetalService) throws {
        self.model = model
        self.config = model.config
        self.metalService = metalService
        self.maxSequenceLength = model.config.sequenceLength

        let kvCacheElementCountPerBuffer = config.numLayers * config.sequenceLength * config.numKeyValueHeads * config.headDim
        let kvCacheSizeBytes = kvCacheElementCountPerBuffer * MemoryLayout<Float16>.stride

        guard kvCacheSizeBytes > 0 else { throw LlamaRunnerError.kvCacheAllocationFailed }

        let options: MTLResourceOptions = .storageModePrivate
        print("Attempting to allocate KV Cache (Private Storage): \(kvCacheSizeBytes * 2 / (1024*1024)) MB total...")

        var tempCacheK: MTLBuffer? = metalService.device.makeBuffer(length: kvCacheSizeBytes, options: options)
        var tempCacheV: MTLBuffer? = metalService.device.makeBuffer(length: kvCacheSizeBytes, options: options)

        if tempCacheK == nil || tempCacheV == nil {
            print("Warning: Failed to allocate KV Cache with Private storage. Trying Shared...")
            let sharedOptions: MTLResourceOptions = .storageModeShared
            tempCacheK = metalService.device.makeBuffer(length: kvCacheSizeBytes, options: sharedOptions)
            tempCacheV = metalService.device.makeBuffer(length: kvCacheSizeBytes, options: sharedOptions)
            guard tempCacheK != nil, tempCacheV != nil else {
                throw LlamaRunnerError.kvCacheAllocationFailed
            }
            print("KV Cache allocated with Shared storage.")
        } else {
            print("KV Cache allocated successfully (Private storage).")
        }

        self.kvCacheK = tempCacheK!
        self.kvCacheV = tempCacheV!
        self.kvCacheK.label = "KV_Cache_K (L:\(config.numLayers) S:\(config.sequenceLength) KVH:\(config.numKeyValueHeads) HD:\(config.headDim))"
        self.kvCacheV.label = "KV_Cache_V (L:\(config.numLayers) S:\(config.sequenceLength) KVH:\(config.numKeyValueHeads) HD:\(config.headDim))"
        self.currentPosition = 0
    }

    func resetState() {
        currentPosition = 0
        print("LlamaRunner state reset (position = 0).")
    }

    private func validateBuffer(
        _ buffer: MTLBuffer,
        name: String,
        expectedType: GGUFDataType,
        elementCount: Int,
        dataType: GGUFDataType,
        commandBuffer: MTLCommandBuffer? // Make parameter optional
    ) throws {
        guard captureDebugSnapshots else { return }
        print("--- Validating Buffer (Runner): \(name) (Type: \(dataType))...")

        guard buffer.storageMode == .shared || buffer.storageMode == .managed else {
            print("--- Validation SKIPPED for private buffer '\(name)' ---")
            return
        }
        guard elementCount > 0 else {
            print("--- Validation SKIPPED for '\(name)': Zero element count provided.")
            return
        }

        // Create a new command buffer for this validation
        guard let validationCommandBuffer = metalService.commandQueue.makeCommandBuffer() else {
            throw LlamaRunnerError.commandEncoderCreationFailed("Validation Command Buffer for \(name)")
        }
        validationCommandBuffer.label = "Validation CB for \(name)"

        let elementSize: Int
        switch dataType {
        case .f32: elementSize = MemoryLayout<Float>.stride
        case .f16: elementSize = MemoryLayout<Float16>.stride
        case .i8: elementSize = MemoryLayout<Int8>.stride
        default:
            throw LlamaRunnerError.validationCheckFailed(bufferName: name, reason: "Unsupported validation type \(dataType)")
        }

        let actualElementCountInBuffer = buffer.length / elementSize
        let checkableElementCount = min(elementCount, actualElementCountInBuffer)
        guard checkableElementCount > 0 else {
            print("--- Validation SKIPPED for '\(name)': Zero checkable elements.")
            return
        }

        var nanCount = 0
        var infCount = 0
        var firstNanIndex = -1
        var firstInfIndex = -1
        let checkLimit = 10000
        let checkCountStart = min(checkableElementCount, checkLimit)
        let checkCountEnd = (checkableElementCount > checkLimit * 2) ? min(checkableElementCount - checkLimit, checkLimit) : 0
        var issueFound = false
        var reason = ""

        if buffer.storageMode == .managed {
            print("--- Synchronizing managed buffer '\(name)' before CPU access ---")
            guard let blitEncoder = validationCommandBuffer.makeBlitCommandEncoder() else {
                throw LlamaRunnerError.commandEncoderCreationFailed("Managed Buffer Sync Blit for \(name)")
            }
            blitEncoder.label = "Managed Buffer Sync Blit (\(name))"
            blitEncoder.synchronize(resource: buffer)
            blitEncoder.endEncoding()
        }
        validationCommandBuffer.commit()
        validationCommandBuffer.waitUntilCompleted()

        if let error = validationCommandBuffer.error {
            throw LlamaRunnerError.encodingFailed("\(name) encoding/sync failed: \(error.localizedDescription)")
        }
        print("--- Synchronization complete for '\(name)' ---")

        let pointer = buffer.contents()

        switch dataType {
        case .f32:
            let bufferPointer = pointer.bindMemory(to: Float.self, capacity: actualElementCountInBuffer)
            for i in 0..<checkCountStart {
                let value = bufferPointer[i]
                if value.isNaN { firstNanIndex = i; nanCount += 1; issueFound = true }
                if value.isInfinite { firstInfIndex = i; infCount += 1; issueFound = true }
                if issueFound { break }
            }
            if !issueFound && checkCountEnd > 0 {
                let startEndCheck = actualElementCountInBuffer - checkCountEnd
                for i in startEndCheck..<actualElementCountInBuffer {
                    let value = bufferPointer[i]
                    if value.isNaN { firstNanIndex = i; nanCount += 1; issueFound = true }
                    if value.isInfinite { firstInfIndex = i; infCount += 1; issueFound = true }
                    if issueFound { break }
                }
            }
        case .f16:
            let bufferPointer = pointer.bindMemory(to: Float16.self, capacity: actualElementCountInBuffer)
            for i in 0..<checkCountStart {
                let value = bufferPointer[i]
                if value.isNaN { firstNanIndex = i; nanCount += 1; issueFound = true }
                if value.isInfinite { firstInfIndex = i; infCount += 1; issueFound = true }
                if issueFound { break }
            }
            if !issueFound && checkCountEnd > 0 {
                let startEndCheck = actualElementCountInBuffer - checkCountEnd
                for i in startEndCheck..<actualElementCountInBuffer {
                    let value = bufferPointer[i]
                    if value.isNaN { firstNanIndex = i; nanCount += 1; issueFound = true }
                    if value.isInfinite { firstInfIndex = i; infCount += 1; issueFound = true }
                    if issueFound { break }
                }
            }
        default:
            print("--- Skipping NaN/Inf check for non-float type \(dataType) ---")
        }

        let validationType = checkableElementCount <= checkLimit * 2 ? "FULL (\(checkableElementCount)/\(actualElementCountInBuffer) elements)" : "Subset (first \(checkCountStart), last \(checkCountEnd))"

        if actualElementCountInBuffer > 0 {
            let sampleCount = min(checkableElementCount, debugSnapshotElementCount)
            print("  Sample (First \(sampleCount) elements):")
            switch dataType {
            case .f32:
                let bufferPointer = pointer.bindMemory(to: Float.self, capacity: actualElementCountInBuffer)
                let sample = Array(UnsafeBufferPointer(start: bufferPointer, count: sampleCount))
                print("  \(sample)")
            case .f16:
                let bufferPointer = pointer.bindMemory(to: Float16.self, capacity: actualElementCountInBuffer)
                let sample = Array(UnsafeBufferPointer(start: bufferPointer, count: sampleCount)).map { Float($0) }
                print("  \(sample)")
            default:
                print("  (Cannot print snippet for type \(dataType))")
            }
        }

        if issueFound {
            reason = "Validation FAILED (\(validationType)): "
            if nanCount > 0 && infCount > 0 {
                reason += "\(nanCount) NaNs (first @ \(firstNanIndex)), \(infCount) Infs (first @ \(firstInfIndex))."
            } else if nanCount > 0 {
                reason += "\(nanCount) NaNs (first @ \(firstNanIndex))."
            } else if infCount > 0 {
                reason += "\(infCount) Infs (first @ \(firstInfIndex))."
            }
            print("!!! \(reason) for buffer '\(name)' !!!")
            throw LlamaRunnerError.validationCheckFailed(bufferName: name, reason: reason)
        } else {
            print("--- Validation (\(validationType)) PASSED for buffer '\(name)'.")
        }

        if captureDebugSnapshots && actualElementCountInBuffer > 0 {
            let sampleCount = min(actualElementCountInBuffer, debugSnapshotElementCount)
            var capturedData: [Any] = []
            switch dataType {
            case .f32:
                let bufferPointer = pointer.bindMemory(to: Float.self, capacity: actualElementCountInBuffer)
                capturedData = Array(UnsafeBufferPointer(start: bufferPointer, count: sampleCount))
            case .f16:
                let bufferPointer = pointer.bindMemory(to: Float16.self, capacity: actualElementCountInBuffer)
                capturedData = Array(UnsafeBufferPointer(start: bufferPointer, count: sampleCount)).map { Float($0) }
            default:
                capturedData = ["Cannot capture type \(dataType)"]
            }
            let snapshot = DebugStepSnapshot(layer: -1, stepName: name, values: capturedData)
            debugSnapshots.append(snapshot)
        }
    }

    func printDebugSnapshots() {
        guard captureDebugSnapshots else { return }
        print("\n--- Debug Snapshots ---")
        if debugSnapshots.isEmpty {
            print("  (No snapshots captured)")
        } else {
            for snapshot in debugSnapshots {
                let valuesString = snapshot.values.map { "\($0)" }.joined(separator: ", ")
                print("  Layer \(snapshot.layer), Step: \(snapshot.stepName)")
                print("    Values (\(snapshot.values.count)): [\(valuesString)]")
            }
        }
        print("-----------------------\n")
    }

    func forward(tokenID: Int) -> MTLBuffer? {
        let pos = currentPosition

        return autoreleasepool { () -> MTLBuffer? in
            if captureDebugSnapshots { debugSnapshots.removeAll() }

            guard pos < maxSequenceLength else {
                print("Error: KV Cache full (pos=\(pos), max=\(maxSequenceLength))")
                return nil
            }
            guard tokenID >= 0 && tokenID < config.vocabSize else {
                print("Error: Invalid token ID (\(tokenID), vocabSize=\(config.vocabSize))")
                return nil
            }

            print("Forward pass for token \(tokenID) at position \(pos)...")

            do {
                let tempBufferOptions: MTLResourceOptions = .storageModeShared
                let embeddingDim = config.embeddingDim
                let hiddenDim = config.hiddenDim
                let headDim = config.headDim
                let nHeads = config.numHeads
                let nKVHeads = config.numKeyValueHeads
                let vocabSize = config.vocabSize
                let f16Size = MemoryLayout<Float16>.stride
                let f32Size = MemoryLayout<Float>.stride

                let hiddenStateSizeBytes = embeddingDim * f16Size
                let qSizeBytes = embeddingDim * f16Size
                let kvDim = nKVHeads * headDim
                let kvSizeBytes = kvDim * f16Size
                let ffnHiddenSizeBytes = hiddenDim * f16Size
                let logitsSizeBytes = vocabSize * f32Size

                guard
                    let hiddenStateBuffer = metalService.device.makeBuffer(length: hiddenStateSizeBytes, options: tempBufferOptions),
                    let normBuffer1 = metalService.device.makeBuffer(length: hiddenStateSizeBytes, options: tempBufferOptions),
                    let residual1Buffer = metalService.device.makeBuffer(length: hiddenStateSizeBytes, options: tempBufferOptions),
                    let qBuffer = metalService.device.makeBuffer(length: qSizeBytes, options: tempBufferOptions),
                    let kBuffer = metalService.device.makeBuffer(length: kvSizeBytes, options: tempBufferOptions),
                    let vBuffer = metalService.device.makeBuffer(length: kvSizeBytes, options: tempBufferOptions),
                    let attnOutputBuffer = metalService.device.makeBuffer(length: hiddenStateSizeBytes, options: tempBufferOptions),
                    let attnProjBuffer = metalService.device.makeBuffer(length: hiddenStateSizeBytes, options: tempBufferOptions),
                    let residual2Buffer = metalService.device.makeBuffer(length: hiddenStateSizeBytes, options: tempBufferOptions),
                    let normBuffer2 = metalService.device.makeBuffer(length: hiddenStateSizeBytes, options: tempBufferOptions),
                    let ffnGateBuffer = metalService.device.makeBuffer(length: ffnHiddenSizeBytes, options: tempBufferOptions),
                    let ffnUpBuffer = metalService.device.makeBuffer(length: ffnHiddenSizeBytes, options: tempBufferOptions),
                    let ffnDownBuffer = metalService.device.makeBuffer(length: hiddenStateSizeBytes, options: tempBufferOptions),
                    let logitsBuffer = metalService.device.makeBuffer(length: logitsSizeBytes, options: tempBufferOptions)
                else {
                    throw LlamaRunnerError.bufferAllocationFailed("Temporary Buffers")
                }

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
                logitsBuffer.label = "Logits Output F32 \(pos)"

                guard let validationCommandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                    throw LlamaRunnerError.commandEncoderCreationFailed("Validation Command Buffer")
                }
                validationCommandBuffer.label = "Output Weight Validation CB (Pos: \(pos))"
                let outputWeightElementCount = model.outputWeight.length / MemoryLayout<Float>.stride
                try validateBuffer(model.outputWeight, name: "Output Weight Matrix (Initial)", expectedType: .f32, elementCount: outputWeightElementCount, dataType: .f32, commandBuffer: validationCommandBuffer)

                // DEBUG: Enhanced CPU validation of model.tokenEmbeddings for tokenID
                if model.tokenEmbeddings.storageMode == .shared || model.tokenEmbeddings.storageMode == .managed {
                    print("--- Enhanced CPU Validation: Checking model.tokenEmbeddings for tokenID \(tokenID) ---")
                    let pointer = model.tokenEmbeddings.contents().assumingMemoryBound(to: Float16.self)
                    let rowStart = tokenID * embeddingDim
                    var nanCount = 0
                    var infCount = 0
                    var firstNanIndex = -1
                    var firstInfIndex = -1
                    for i in 0..<embeddingDim {
                        let value = pointer[rowStart + i]
                        if value.isNaN {
                            if firstNanIndex == -1 { firstNanIndex = i }
                            nanCount += 1
                        }
                        if value.isInfinite {
                            if firstInfIndex == -1 { firstInfIndex = i }
                            infCount += 1
                        }
                    }
                    if nanCount > 0 || infCount > 0 {
                        print("!!! Enhanced CPU Validation FAILED: \(nanCount) NaNs (first @ \(firstNanIndex)), \(infCount) Infs (first @ \(firstInfIndex)) in embedding row for tokenID \(tokenID) !!!")
                    } else {
                        print("--- Enhanced CPU Validation PASSED: No NaNs or Infs in embedding row for tokenID \(tokenID) ---")
                    }
                    // DEBUG: Specifically check index 38
                    let sourceIndex = rowStart + 38
                    let sourceValue = pointer[sourceIndex]
                    print("DEBUG: Source value at index 38 (absolute index \(sourceIndex)) for tokenID \(tokenID): \(sourceValue)")
                    if sourceValue.isNaN {
                        print("!!! NaN detected in model.tokenEmbeddings at index 38 for tokenID \(tokenID) !!!")
                    } else if sourceValue.isInfinite {
                        print("!!! Inf detected in model.tokenEmbeddings at index 38 for tokenID \(tokenID) !!!")
                    } else {
                        print("--- Source value at index 38 is valid: \(sourceValue) ---")
                    }
                } else {
                    print("--- Skipping enhanced CPU validation: model.tokenEmbeddings is private ---")
                }

                guard let embedCommandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                    throw LlamaRunnerError.commandEncoderCreationFailed("Embedding Command Buffer")
                }
                embedCommandBuffer.label = "Embedding Lookup CB (Pos: \(pos))"
                guard let blitEncoderEmbed = embedCommandBuffer.makeBlitCommandEncoder() else {
                    throw LlamaRunnerError.commandEncoderCreationFailed("Embedding Blit")
                }
                blitEncoderEmbed.label = "Embedding Blit Encoder \(pos)"
                let embeddingOffset = tokenID * embeddingDim * f16Size
                guard model.tokenEmbeddings.length >= embeddingOffset + hiddenStateSizeBytes else {
                    blitEncoderEmbed.endEncoding()
                    throw LlamaRunnerError.kvCacheOutOfBounds
                }
                blitEncoderEmbed.copy(from: model.tokenEmbeddings, sourceOffset: embeddingOffset, to: hiddenStateBuffer, destinationOffset: 0, size: hiddenStateSizeBytes)
                blitEncoderEmbed.endEncoding()
                embedCommandBuffer.commit()
                embedCommandBuffer.waitUntilCompleted()
                print("  Encoded Embedding Lookup.")

                // DEBUG: Check hiddenStateBuffer at index 38 after copy
                if hiddenStateBuffer.storageMode == .shared || hiddenStateBuffer.storageMode == .managed {
                    let destPointer = hiddenStateBuffer.contents().assumingMemoryBound(to: Float16.self)
                    let destValue = destPointer[38]
                    print("DEBUG: Value at index 38 in hiddenStateBuffer after copy: \(destValue)")
                    if destValue.isNaN {
                        print("!!! NaN detected in hiddenStateBuffer at index 38 after copy !!!")
                    } else if destValue.isInfinite {
                        print("!!! Inf detected in hiddenStateBuffer at index 38 after copy !!!")
                    } else {
                        print("--- Destination value at index 38 is valid: \(destValue) ---")
                    }
                } else {
                    print("--- Skipping hiddenStateBuffer debug check: Buffer is private ---")
                }

                guard let embedValidationCommandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                    throw LlamaRunnerError.commandEncoderCreationFailed("Embed Validation Command Buffer")
                }
                embedValidationCommandBuffer.label = "Embedding Validation CB (Pos: \(pos))"
                try validateBuffer(hiddenStateBuffer, name: "Initial Embedding", expectedType: .f16, elementCount: embeddingDim, dataType: .f16, commandBuffer: embedValidationCommandBuffer)

                for layerIndex in 0..<config.numLayers {
                    let layerLabel = "L\(layerIndex) P\(pos)"
                    print("    Processing \(layerLabel)...")

                    guard let layerCommandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                        throw LlamaRunnerError.commandEncoderCreationFailed("Layer \(layerIndex) Command Buffer")
                    }
                    layerCommandBuffer.label = "Layer \(layerIndex) Ops CB (Pos: \(pos))"

                    guard let blitEncoderRes1 = layerCommandBuffer.makeBlitCommandEncoder() else {
                        throw LlamaRunnerError.commandEncoderCreationFailed("Save Residual 1 Blit L\(layerIndex)")
                    }
                    blitEncoderRes1.label = "Save Residual 1 \(layerLabel)"
                    blitEncoderRes1.copy(from: hiddenStateBuffer, sourceOffset: 0, to: residual1Buffer, destinationOffset: 0, size: hiddenStateSizeBytes)
                    blitEncoderRes1.endEncoding()

                    guard metalService.encodeRMSNormF16(
                        commandBuffer: layerCommandBuffer,
                        inputBuffer: hiddenStateBuffer,
                        weightBuffer: model.blocks[layerIndex].attentionNormWeight,
                        outputBuffer: normBuffer1,
                        rowCount: 1,
                        elementCountPerRow: embeddingDim,
                        eps: config.rmsNormEps,
                        label: "PreAttnNorm \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Pre-Attn RMSNorm \(layerLabel)")
                    }
                    print("      Encoded Pre-Attn RMSNorm.")

                    guard metalService.encodeMPSMatrixMultiply(
                        commandBuffer: layerCommandBuffer,
                        inputA: normBuffer1,
                        inputB: model.blocks[layerIndex].attention.qWeight,
                        outputC: qBuffer,
                        rowsA: 1,
                        colsA: embeddingDim,
                        rowsB: embeddingDim,
                        colsB: embeddingDim,
                        label: "Q_Proj \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Q Proj \(layerLabel)")
                    }

                    guard metalService.encodeMPSMatrixMultiply(
                        commandBuffer: layerCommandBuffer,
                        inputA: normBuffer1,
                        inputB: model.blocks[layerIndex].attention.kWeight,
                        outputC: kBuffer,
                        rowsA: 1,
                        colsA: embeddingDim,
                        rowsB: embeddingDim,
                        colsB: kvDim,
                        label: "K_Proj \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("K Proj \(layerLabel)")
                    }

                    guard metalService.encodeMPSMatrixMultiply(
                        commandBuffer: layerCommandBuffer,
                        inputA: normBuffer1,
                        inputB: model.blocks[layerIndex].attention.vWeight,
                        outputC: vBuffer,
                        rowsA: 1,
                        colsA: embeddingDim,
                        rowsB: embeddingDim,
                        colsB: kvDim,
                        label: "V_Proj \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("V Proj \(layerLabel)")
                    }
                    print("      Encoded QKV Projections.")

                    guard metalService.applyRoPE(
                        commandBuffer: layerCommandBuffer,
                        buffer: qBuffer,
                        ropeFrequencies: model.ropeFrequencies,
                        config: config,
                        posOffset: pos,
                        sequenceLength: 1,
                        numHeads: nHeads,
                        headDim: headDim
                    ) else {
                        throw LlamaRunnerError.encodingFailed("RoPE Q \(layerLabel)")
                    }

                    guard metalService.applyRoPE(
                        commandBuffer: layerCommandBuffer,
                        buffer: kBuffer,
                        ropeFrequencies: model.ropeFrequencies,
                        config: config,
                        posOffset: pos,
                        sequenceLength: 1,
                        numHeads: nKVHeads,
                        headDim: headDim
                    ) else {
                        throw LlamaRunnerError.encodingFailed("RoPE K \(layerLabel)")
                    }
                    print("      Encoded RoPE.")

                    guard let blitEncoderKV = layerCommandBuffer.makeBlitCommandEncoder() else {
                        throw LlamaRunnerError.commandEncoderCreationFailed("KV Cache Blit L\(layerIndex)")
                    }
                    blitEncoderKV.label = "KV Cache Update Blit \(layerLabel)"
                    let elementsPerKVEntry = nKVHeads * headDim
                    let bytesPerKVEntry = elementsPerKVEntry * f16Size
                    let layerOffsetBytes = layerIndex * maxSequenceLength * bytesPerKVEntry
                    let posOffsetBytes = pos * bytesPerKVEntry
                    let destinationOffsetK = layerOffsetBytes + posOffsetBytes
                    let destinationOffsetV = layerOffsetBytes + posOffsetBytes
                    guard kvCacheK.length >= destinationOffsetK + bytesPerKVEntry,
                          kvCacheV.length >= destinationOffsetV + bytesPerKVEntry else {
                        blitEncoderKV.endEncoding()
                        throw LlamaRunnerError.kvCacheOutOfBounds
                    }
                    blitEncoderKV.copy(from: kBuffer, sourceOffset: 0, to: kvCacheK, destinationOffset: destinationOffsetK, size: bytesPerKVEntry)
                    blitEncoderKV.copy(from: vBuffer, sourceOffset: 0, to: kvCacheV, destinationOffset: destinationOffsetV, size: bytesPerKVEntry)
                    blitEncoderKV.endEncoding()
                    print("      Encoded KV Cache Update.")

                    let currentSeqLen = pos + 1
                    let scale = Float16(1.0 / sqrt(Float(config.headDim)))
                    let kvSliceSizeBytes = currentSeqLen * nKVHeads * headDim * f16Size
                    let repeatedKVSizeBytes = currentSeqLen * nHeads * headDim * f16Size
                    let scoreSizeBytes = nHeads * currentSeqLen * f16Size
                    let headSizeBytes = headDim * f16Size
                    let kSliceHeadSizeBytes = currentSeqLen * headDim * f16Size
                    let scoreSliceSizeBytes = currentSeqLen * f16Size

                    guard
                        let kSlice = metalService.device.makeBuffer(length: kvSliceSizeBytes, options: tempBufferOptions),
                        let vSlice = metalService.device.makeBuffer(length: kvSliceSizeBytes, options: tempBufferOptions),
                        let kRepeated = metalService.device.makeBuffer(length: repeatedKVSizeBytes, options: tempBufferOptions),
                        let vRepeated = metalService.device.makeBuffer(length: repeatedKVSizeBytes, options: tempBufferOptions),
                        let attnScores = metalService.device.makeBuffer(length: scoreSizeBytes, options: tempBufferOptions),
                        let qHeadBuffer = metalService.device.makeBuffer(length: headSizeBytes, options: tempBufferOptions),
                        let kvSliceHeadBuffer = metalService.device.makeBuffer(length: kSliceHeadSizeBytes, options: tempBufferOptions),
                        let scoreSliceBuffer = metalService.device.makeBuffer(length: scoreSliceSizeBytes, options: tempBufferOptions)
                    else {
                        throw LlamaRunnerError.bufferAllocationFailed("Attention Temp Buffers L\(layerIndex)")
                    }

                    kSlice.label = "kSlice \(layerLabel)"
                    vSlice.label = "vSlice \(layerLabel)"
                    kRepeated.label = "kRepeated \(layerLabel)"
                    vRepeated.label = "vRepeated \(layerLabel)"
                    attnScores.label = "attnScores/Probs \(layerLabel)"
                    qHeadBuffer.label = "qHeadBuffer \(layerLabel)"
                    kvSliceHeadBuffer.label = "kvSliceHeadBuffer \(layerLabel)"
                    scoreSliceBuffer.label = "scoreSliceBuffer \(layerLabel)"

                    guard let blitEncoderSlice = layerCommandBuffer.makeBlitCommandEncoder() else {
                        throw LlamaRunnerError.commandEncoderCreationFailed("KV Slice Blit L\(layerIndex)")
                    }
                    blitEncoderSlice.label = "KV Slice Blit \(layerLabel)"
                    let sourceOffsetKV = layerOffsetBytes
                    guard kvCacheK.length >= sourceOffsetKV + kvSliceSizeBytes,
                          kvCacheV.length >= sourceOffsetKV + kvSliceSizeBytes else {
                        blitEncoderSlice.endEncoding()
                        throw LlamaRunnerError.kvCacheOutOfBounds
                    }
                    blitEncoderSlice.copy(from: kvCacheK, sourceOffset: sourceOffsetKV, to: kSlice, destinationOffset: 0, size: kvSliceSizeBytes)
                    blitEncoderSlice.copy(from: kvCacheV, sourceOffset: sourceOffsetKV, to: vSlice, destinationOffset: 0, size: kvSliceSizeBytes)
                    blitEncoderSlice.endEncoding()
                    print("      Encoded Get K/V Slice.")

                    guard metalService.applyRepeatKVHeads(
                        sourceBuffer: kSlice,
                        destinationBuffer: kRepeated,
                        numKVHeads: nKVHeads,
                        numQueryGroups: config.numQueryGroups,
                        headDim: headDim,
                        seqLen: currentSeqLen,
                        commandBuffer: layerCommandBuffer
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Repeat K \(layerLabel)")
                    }

                    guard metalService.applyRepeatKVHeads(
                        sourceBuffer: vSlice,
                        destinationBuffer: vRepeated,
                        numKVHeads: nKVHeads,
                        numQueryGroups: config.numQueryGroups,
                        headDim: headDim,
                        seqLen: currentSeqLen,
                        commandBuffer: layerCommandBuffer
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Repeat V \(layerLabel)")
                    }
                    print("      Encoded GQA Repeat Heads.")

                    print("      Encoding Attention Scores (Looping \(nHeads) heads)...")
                    for h in 0..<nHeads {
                        let headLabel = "\(layerLabel) H\(h)"
                        guard let blitEncoderQHead = layerCommandBuffer.makeBlitCommandEncoder() else {
                            throw LlamaRunnerError.commandEncoderCreationFailed("Extract QHead Blit \(headLabel)")
                        }
                        let qOffset = h * headSizeBytes
                        guard qBuffer.length >= qOffset + headSizeBytes else {
                            blitEncoderQHead.endEncoding()
                            throw LlamaRunnerError.validationCheckFailed(bufferName: qBuffer.label!, reason: "Head slice out of bounds H\(h)")
                        }
                        blitEncoderQHead.label = "Extract Q H\(h) \(layerLabel)"
                        blitEncoderQHead.copy(from: qBuffer, sourceOffset: qOffset, to: qHeadBuffer, destinationOffset: 0, size: headSizeBytes)
                        blitEncoderQHead.endEncoding()

                        guard let blitEncoderKHead = layerCommandBuffer.makeBlitCommandEncoder() else {
                            throw LlamaRunnerError.commandEncoderCreationFailed("Extract KHead Blit H\(h) \(layerLabel)")
                        }
                        blitEncoderKHead.label = "Extract K H\(h) \(layerLabel)"
                        let srcStride = nHeads * headDim * f16Size
                        let dstStride = headDim * f16Size
                        let srcHeadOffset = h * headDim * f16Size
                        let expectedSrcSliceSize = currentSeqLen * srcStride
                        let expectedDstSliceSize = currentSeqLen * dstStride
                        guard kRepeated.length >= expectedSrcSliceSize,
                              kvSliceHeadBuffer.length >= expectedDstSliceSize else {
                            blitEncoderKHead.endEncoding()
                            throw LlamaRunnerError.validationCheckFailed(bufferName: kRepeated.label!, reason: "Slice out of bounds H\(h)")
                        }
                        for t in 0..<currentSeqLen {
                            let srcOffset = t * srcStride + srcHeadOffset
                            let dstOffset = t * dstStride
                            blitEncoderKHead.copy(from: kRepeated, sourceOffset: srcOffset, to: kvSliceHeadBuffer, destinationOffset: dstOffset, size: headSizeBytes)
                        }
                        blitEncoderKHead.endEncoding()

                        guard metalService.encodeMPSMatrixMultiply(
                            commandBuffer: layerCommandBuffer,
                            inputA: qHeadBuffer,
                            inputB: kvSliceHeadBuffer,
                            outputC: scoreSliceBuffer,
                            rowsA: 1, colsA: headDim,
                            rowsB: currentSeqLen, colsB: headDim,
                            transposeA: false, transposeB: true,
                            alpha: Double(scale), beta: 0.0,
                            label: "ScoreMatMul H\(h) \(layerLabel)"
                        ) else {
                            throw LlamaRunnerError.encodingFailed("Score MatMul H\(h) \(layerLabel)")
                        }

                        guard let blitEncoderScore = layerCommandBuffer.makeBlitCommandEncoder() else {
                            throw LlamaRunnerError.commandEncoderCreationFailed("Copy Score Blit H\(h) \(layerLabel)")
                        }
                        blitEncoderScore.label = "Copy Score H\(h) \(layerLabel)"
                        let scoreDestOffset = h * currentSeqLen * f16Size
                        guard attnScores.length >= scoreDestOffset + scoreSliceSizeBytes else {
                            blitEncoderScore.endEncoding()
                            throw LlamaRunnerError.validationCheckFailed(bufferName: attnScores.label!, reason: "Score destination out of bounds H\(h)")
                        }
                        blitEncoderScore.copy(from: scoreSliceBuffer, sourceOffset: 0, to: attnScores, destinationOffset: scoreDestOffset, size: scoreSliceSizeBytes)
                        blitEncoderScore.endEncoding()
                    }
                    print("      Finished Encoding Attention Scores.")

                    guard metalService.encodeMPSSoftMax(
                        commandBuffer: layerCommandBuffer,
                        inputMatrixBuffer: attnScores,
                        outputMatrixBuffer: attnScores,
                        rows: nHeads,
                        columns: currentSeqLen,
                        label: "Softmax \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Softmax \(layerLabel)")
                    }
                    print("      Encoded Softmax.")

                    print("      Encoding Attention Values (Looping \(nHeads) heads)...")
                    guard let blitEncoderZeroAttnOut = layerCommandBuffer.makeBlitCommandEncoder() else {
                        throw LlamaRunnerError.commandEncoderCreationFailed("Zero AttnOutput Blit L\(layerIndex)")
                    }
                    blitEncoderZeroAttnOut.label = "Zero AttnOutput \(layerLabel)"
                    blitEncoderZeroAttnOut.fill(buffer: attnOutputBuffer, range: 0..<attnOutputBuffer.length, value: 0)
                    blitEncoderZeroAttnOut.endEncoding()

                    let valueHeadOutputBuffer = qHeadBuffer
                    for h in 0..<nHeads {
                        let headLabel = "\(layerLabel) H\(h)"
                        guard let blitEncoderProbHead = layerCommandBuffer.makeBlitCommandEncoder() else {
                            throw LlamaRunnerError.commandEncoderCreationFailed("Extract Probs Blit \(headLabel)")
                        }
                        blitEncoderProbHead.label = "Extract Probs H\(h) \(layerLabel)"
                        let probSourceOffset = h * currentSeqLen * f16Size
                        guard attnScores.length >= probSourceOffset + scoreSliceSizeBytes else {
                            blitEncoderProbHead.endEncoding()
                            throw LlamaRunnerError.validationCheckFailed(bufferName: attnScores.label!, reason: "Probabilities source out of bounds H\(h)")
                        }
                        blitEncoderProbHead.copy(from: attnScores, sourceOffset: probSourceOffset, to: scoreSliceBuffer, destinationOffset: 0, size: scoreSliceSizeBytes)
                        blitEncoderProbHead.endEncoding()

                        guard let blitEncoderVHead = layerCommandBuffer.makeBlitCommandEncoder() else {
                            throw LlamaRunnerError.commandEncoderCreationFailed("Extract VHead Blit H\(h) \(layerLabel)")
                        }
                        blitEncoderVHead.label = "Extract V H\(h) \(layerLabel)"
                        let srcStride = nHeads * headDim * f16Size
                        let dstStride = headDim * f16Size
                        let srcHeadOffset = h * headDim * f16Size
                        let expectedSrcSliceSize = currentSeqLen * srcStride
                        let expectedDstSliceSize = currentSeqLen * dstStride
                        guard vRepeated.length >= expectedSrcSliceSize,
                              kvSliceHeadBuffer.length >= expectedDstSliceSize else {
                            blitEncoderVHead.endEncoding()
                            throw LlamaRunnerError.validationCheckFailed(bufferName: vRepeated.label!, reason: "V slice source out of bounds H\(h)")
                        }
                        for t in 0..<currentSeqLen {
                            let srcOffset = t * srcStride + srcHeadOffset
                            let dstOffset = t * dstStride
                            blitEncoderVHead.copy(from: vRepeated, sourceOffset: srcOffset, to: kvSliceHeadBuffer, destinationOffset: dstOffset, size: headSizeBytes)
                        }
                        blitEncoderVHead.endEncoding()

                        guard metalService.encodeMPSMatrixMultiply(
                            commandBuffer: layerCommandBuffer,
                            inputA: scoreSliceBuffer,
                            inputB: kvSliceHeadBuffer,
                            outputC: valueHeadOutputBuffer,
                            rowsA: 1, colsA: currentSeqLen,
                            rowsB: currentSeqLen, colsB: headDim,
                            transposeA: false, transposeB: false,
                            label: "ValueMatMul H\(h) \(layerLabel)"
                        ) else {
                            throw LlamaRunnerError.encodingFailed("Value MatMul H\(h) \(layerLabel)")
                        }

                        guard let blitEncoderAttnOut = layerCommandBuffer.makeBlitCommandEncoder() else {
                            throw LlamaRunnerError.commandEncoderCreationFailed("Copy AttnOutput Blit H\(h) \(layerLabel)")
                        }
                        blitEncoderAttnOut.label = "Copy AttnOutput H\(h) \(layerLabel)"
                        let attnDestOffset = h * headSizeBytes
                        guard attnOutputBuffer.length >= attnDestOffset + headSizeBytes else {
                            blitEncoderAttnOut.endEncoding()
                            throw LlamaRunnerError.validationCheckFailed(bufferName: attnOutputBuffer.label!, reason: "AttnOutput destination out of bounds H\(h)")
                        }
                        blitEncoderAttnOut.copy(from: valueHeadOutputBuffer, sourceOffset: 0, to: attnOutputBuffer, destinationOffset: attnDestOffset, size: headSizeBytes)
                        blitEncoderAttnOut.endEncoding()
                    }
                    print("      Finished Encoding Attention Values.")

                    guard metalService.encodeMPSMatrixMultiply(
                        commandBuffer: layerCommandBuffer,
                        inputA: attnOutputBuffer,
                        inputB: model.blocks[layerIndex].attention.oWeight,
                        outputC: attnProjBuffer,
                        rowsA: 1, colsA: embeddingDim,
                        rowsB: embeddingDim, colsB: embeddingDim,
                        label: "Attn_O_Proj \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Attn Output Proj \(layerLabel)")
                    }
                    print("      Encoded Attn Output Projection.")

                    guard metalService.applyElementWiseAdd(
                        inputBufferA: residual1Buffer,
                        inputBufferB: attnProjBuffer,
                        outputBufferC: hiddenStateBuffer,
                        elementCount: embeddingDim,
                        commandBuffer: layerCommandBuffer
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Residual Add 1 \(layerLabel)")
                    }
                    print("      Encoded Residual Add 1.")

                    guard let blitEncoderRes2 = layerCommandBuffer.makeBlitCommandEncoder() else {
                        throw LlamaRunnerError.commandEncoderCreationFailed("Save Residual 2 Blit L\(layerIndex)")
                    }
                    blitEncoderRes2.label = "Save Residual 2 \(layerLabel)"
                    blitEncoderRes2.copy(from: hiddenStateBuffer, sourceOffset: 0, to: residual2Buffer, destinationOffset: 0, size: hiddenStateSizeBytes)
                    blitEncoderRes2.endEncoding()

                    guard metalService.encodeRMSNormF16(
                        commandBuffer: layerCommandBuffer,
                        inputBuffer: hiddenStateBuffer,
                        weightBuffer: model.blocks[layerIndex].ffnNormWeight,
                        outputBuffer: normBuffer2,
                        rowCount: 1,
                        elementCountPerRow: embeddingDim,
                        eps: config.rmsNormEps,
                        label: "PreFFNNorm \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Pre-FFN RMSNorm \(layerLabel)")
                    }
                    print("      Encoded Pre-FFN RMSNorm.")

                    guard metalService.encodeMPSMatrixMultiply(
                        commandBuffer: layerCommandBuffer,
                        inputA: normBuffer2,
                        inputB: model.blocks[layerIndex].mlp.gateWeight,
                        outputC: ffnGateBuffer,
                        rowsA: 1, colsA: embeddingDim,
                        rowsB: embeddingDim, colsB: hiddenDim,
                        label: "FFN_Gate_Proj \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("FFN Gate Proj \(layerLabel)")
                    }

                    guard metalService.encodeMPSMatrixMultiply(
                        commandBuffer: layerCommandBuffer,
                        inputA: normBuffer2,
                        inputB: model.blocks[layerIndex].mlp.upWeight,
                        outputC: ffnUpBuffer,
                        rowsA: 1, colsA: embeddingDim,
                        rowsB: embeddingDim, colsB: hiddenDim,
                        label: "FFN_Up_Proj \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("FFN Up Proj \(layerLabel)")
                    }

                    guard metalService.applySILU(
                        inputBuffer: ffnGateBuffer,
                        outputBuffer: ffnGateBuffer,
                        elementCount: hiddenDim,
                        commandBuffer: layerCommandBuffer
                    ) else {
                        throw LlamaRunnerError.encodingFailed("SiLU \(layerLabel)")
                    }

                    guard metalService.applyElementWiseMul(
                        inputBufferA: ffnGateBuffer,
                        inputBufferB: ffnUpBuffer,
                        outputBufferC: ffnUpBuffer,
                        elementCount: hiddenDim,
                        commandBuffer: layerCommandBuffer
                    ) else {
                        throw LlamaRunnerError.encodingFailed("ElemWise Mul \(layerLabel)")
                    }

                    guard metalService.encodeMPSMatrixMultiply(
                        commandBuffer: layerCommandBuffer,
                        inputA: ffnUpBuffer,
                        inputB: model.blocks[layerIndex].mlp.downWeight,
                        outputC: ffnDownBuffer,
                        rowsA: 1, colsA: hiddenDim,
                        rowsB: hiddenDim, colsB: embeddingDim,
                        label: "FFN_Down_Proj \(layerLabel)"
                    ) else {
                        throw LlamaRunnerError.encodingFailed("FFN Down Proj \(layerLabel)")
                    }
                    print("      Encoded MLP/SwiGLU.")

                    guard metalService.applyElementWiseAdd(
                        inputBufferA: residual2Buffer,
                        inputBufferB: ffnDownBuffer,
                        outputBufferC: hiddenStateBuffer,
                        elementCount: embeddingDim,
                        commandBuffer: layerCommandBuffer
                    ) else {
                        throw LlamaRunnerError.encodingFailed("Residual Add 2 \(layerLabel)")
                    }
                    print("      Encoded Residual Add 2.")

                    layerCommandBuffer.commit()
                    layerCommandBuffer.waitUntilCompleted()

                    if let error = layerCommandBuffer.error {
                        throw LlamaRunnerError.encodingFailed("Layer \(layerIndex) operations failed: \(error.localizedDescription)")
                    }

                    guard let layerValidationCB = metalService.commandQueue.makeCommandBuffer() else {
                        throw LlamaRunnerError.commandEncoderCreationFailed("Layer \(layerIndex) Validation CB")
                    }
                    layerValidationCB.label = "Layer \(layerIndex) Validation CB (Pos: \(pos))"

                    if layerIndex >= config.numLayers - 2 {
                        try validateBuffer(
                            qBuffer, name: "L\(layerIndex) Q RoPE Out", expectedType: .f16,
                            elementCount: embeddingDim, dataType: .f16, commandBuffer: nil
                        )
                        try validateBuffer(
                            kBuffer, name: "L\(layerIndex) K RoPE Out", expectedType: .f16,
                            elementCount: kvDim, dataType: .f16, commandBuffer: nil
                        )
                        try validateBuffer(
                            vBuffer, name: "L\(layerIndex) V Proj Out", expectedType: .f16,
                            elementCount: kvDim, dataType: .f16, commandBuffer: nil
                        )
                        try validateBuffer(
                            attnOutputBuffer, name: "L\(layerIndex) AttnValueOut Full", expectedType: .f16,
                            elementCount: embeddingDim, dataType: .f16, commandBuffer: nil
                        )
                        try validateBuffer(
                            hiddenStateBuffer, name: "L\(layerIndex) HiddenState After Res2", expectedType: .f16,
                            elementCount: embeddingDim, dataType: .f16, commandBuffer: nil
                        )
                    }
                }

                guard let finalNormCommandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                    throw LlamaRunnerError.commandEncoderCreationFailed("Final Norm Command Buffer")
                }
                finalNormCommandBuffer.label = "Final Norm CB (Pos: \(pos))"
                guard metalService.encodeRMSNormF16(
                    commandBuffer: finalNormCommandBuffer,
                    inputBuffer: hiddenStateBuffer,
                    weightBuffer: model.finalNormWeight,
                    outputBuffer: normBuffer1,
                    rowCount: 1,
                    elementCountPerRow: embeddingDim,
                    eps: config.rmsNormEps,
                    label: "FinalNorm"
                ) else {
                    throw LlamaRunnerError.encodingFailed("Final RMSNorm")
                }
                print("  Encoded Final RMSNorm.")
                finalNormCommandBuffer.commit()
                finalNormCommandBuffer.waitUntilCompleted()

                guard let finalValidationCB = metalService.commandQueue.makeCommandBuffer() else {
                    throw LlamaRunnerError.commandEncoderCreationFailed("Final Validation CB")
                }
                finalValidationCB.label = "Final Validation CB (Pos: \(pos))"
                try validateBuffer(normBuffer1, name: "Final RMSNorm Output", expectedType: .f16, elementCount: embeddingDim, dataType: .f16, commandBuffer: finalValidationCB)

                guard let outputCommandBuffer = metalService.commandQueue.makeCommandBuffer() else {
                    throw LlamaRunnerError.commandEncoderCreationFailed("Output Projection Command Buffer")
                }
                outputCommandBuffer.label = "Output Projection CB (Pos: \(pos))"
                guard metalService.encodeMPSMatrixMultiply(
                    commandBuffer: outputCommandBuffer,
                    inputA: normBuffer1,
                    inputB: model.outputWeight,
                    outputC: logitsBuffer,
                    rowsA: 1, colsA: embeddingDim,
                    rowsB: embeddingDim, colsB: vocabSize,
                    label: "Output Projection"
                ) else {
                    throw LlamaRunnerError.encodingFailed("Output Projection")
                }
                print("  Encoded Output Projection.")
                outputCommandBuffer.commit()
                outputCommandBuffer.waitUntilCompleted()

                guard let logitsValidationCB = metalService.commandQueue.makeCommandBuffer() else {
                    throw LlamaRunnerError.commandEncoderCreationFailed("Logits Validation Command Buffer")
                }
                logitsValidationCB.label = "Logits Validation CB (Pos: \(pos))"
                try validateBuffer(logitsBuffer, name: "Output Logits (Post-GPU)", expectedType: .f32, elementCount: vocabSize, dataType: .f32, commandBuffer: logitsValidationCB)

                currentPosition += 1
                return logitsBuffer
            } catch let error as LlamaRunnerError {
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!! LlamaRunner HALTED due to error: \(error)")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                return nil
            } catch {
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!! LlamaRunner HALTED due to unexpected error: \(error)")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                return nil
            }
        }
    }
}
