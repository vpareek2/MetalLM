// MetalLM/Models/Llama3Architecture.swift

import Foundation
import Metal  // Required for MTLBuffer

// MARK: - Configuration Error

enum ConfigError: Error {
    case missingMetadata(key: String)
    case invalidMetadataType(key: String, expected: String)
    case configurationError(message: String)
}

// MARK: - Llama Configuration

/// Holds the hyperparameters and architectural details of a Llama model,
/// typically loaded from GGUF metadata.
struct LlamaConfig {
    // Core Dimensions
    let embeddingDim: Int  // n_embd: Dimension of token embeddings
    let hiddenDim: Int  // n_ff: Dimension of the intermediate layer in MLP/FFN
    let headDim: Int  // n_head_dim: Dimension of each attention head (calculated)
    let numLayers: Int  // n_layer: Number of transformer blocks
    let numHeads: Int  // n_head: Number of attention heads
    let numKeyValueHeads: Int  // n_kv_head: Number of key/value heads (for GQA/MQA)

    // Vocabulary & Context
    let vocabSize: Int  // Vocabulary size
    let sequenceLength: Int  // n_ctx: Maximum sequence length the model supports

    // Normalization
    let rmsNormEps: Float  // Epsilon value for RMSNorm layers

    // RoPE (Rotary Positional Embedding) Parameters - TODO: Verify GGUF keys
    let ropeDimensionCount: Int  // Defaults to headDim if not specified? Check GGUF keys like llama.rope.dimension_count
    let ropeFreqBase: Float  // Typically 10000.0 or 500000.0 etc. Check GGUF keys like llama.rope.freq_base

    // Calculated properties
    var numQueryGroups: Int { numHeads / numKeyValueHeads }  // For GQA/MQA grouping

    /// Initializes the configuration by parsing metadata from a GGUF file.
    /// - Parameter metadata: A dictionary containing key-value pairs from GGUF metadata.
    /// - Throws: `ConfigError` if required metadata is missing or has an incorrect type.
    init(metadata: [String: GGUFValue]) throws {
        // Helper functions for safe metadata extraction
        func getRequiredUInt64(_ key: String) throws -> UInt64 {
            guard let value = metadata[key]?.uint64 else {
                throw ConfigError.missingMetadata(key: key)
            }
            return value
        }

        func getRequiredFloat32(_ key: String) throws -> Float {
            guard case .float32(let value) = metadata[key] else {
                // Attempt fallback for uint32/uint64 if necessary? For now, require float32.
                throw ConfigError.invalidMetadataType(key: key, expected: "Float32")
            }
            return value
        }
        func getOptionalFloat32(_ key: String, default defaultValue: Float) -> Float {
            guard case .float32(let value) = metadata[key] else {
                return defaultValue
            }
            return value
        }

        // Extract required values
        self.embeddingDim = Int(try getRequiredUInt64("llama.embedding_length"))
        self.numLayers = Int(try getRequiredUInt64("llama.block_count"))
        self.numHeads = Int(try getRequiredUInt64("llama.attention.head_count"))
        self.numKeyValueHeads = Int(try getRequiredUInt64("llama.attention.head_count_kv"))
        self.hiddenDim = Int(try getRequiredUInt64("llama.feed_forward_length"))
        self.vocabSize = Int(try getRequiredUInt64("llama.vocab_size"))
        self.sequenceLength = Int(try getRequiredUInt64("llama.context_length"))
        self.rmsNormEps = try getRequiredFloat32("llama.attention.layer_norm_rms_epsilon")

        // --- RoPE Parameters ---
        // Use defaults or throw if missing, depending on strictness required.
        // Need to confirm the exact GGUF keys used by the models you target.
        // Example potential keys:
        self.ropeDimensionCount = Int(
            metadata["llama.rope.dimension_count"]?.uint64 ?? UInt64(embeddingDim / numHeads))  // Often defaults to head dimension
        self.ropeFreqBase = getOptionalFloat32("llama.rope.freq_base", default: 500000.0)  // Default updated based on dump

        // --- Validations and Derived Values ---
        guard embeddingDim % numHeads == 0 else {
            throw ConfigError.configurationError(
                message:
                    "Embedding dimension (\(embeddingDim)) must be divisible by number of heads (\(numHeads))."
            )
        }
        self.headDim = embeddingDim / numHeads

        guard numHeads % numKeyValueHeads == 0 else {
            throw ConfigError.configurationError(
                message:
                    "Number of heads (\(numHeads)) must be divisible by number of key/value heads (\(numKeyValueHeads))."
            )
        }

        // Print loaded config for verification during development
        print("--- LlamaConfig Initialized ---")
        print(
            String(
                format: "  Layers: %d, Embed Dim: %d, Hidden Dim: %d", numLayers, embeddingDim,
                hiddenDim))
        print(
            String(
                format: "  Heads: %d, KV Heads: %d, Head Dim: %d, Query Groups: %d", numHeads,
                numKeyValueHeads, headDim, numQueryGroups))
        print(
            String(
                format: "  Vocab: %d, Seq Len: %d, Norm Eps: %.8f", vocabSize, sequenceLength,
                rmsNormEps))
        print(
            String(format: "  RoPE Dim: %d, RoPE Freq Base: %.1f", ropeDimensionCount, ropeFreqBase)
        )
        print("-----------------------------")
    }
}

// MARK: - Model Layers

/// Represents the Attention mechanism weights for a single transformer block.
struct LlamaAttention {
    let qWeight: MTLBuffer  // Query projection weights (Shape: [embed_dim, n_head * head_dim])
    let kWeight: MTLBuffer  // Key projection weights   (Shape: [embed_dim, n_kv_head * head_dim])
    let vWeight: MTLBuffer  // Value projection weights (Shape: [embed_dim, n_kv_head * head_dim])
    let oWeight: MTLBuffer  // Output projection weights (Shape: [n_head * head_dim, embed_dim])

    // Note: RoPE is applied during computation, not stored as weights here.
    // Note: Attention norm weights are stored in LlamaTransformerBlock.
}

/// Represents the MLP (Feed-Forward Network) weights for a single transformer block.
struct LlamaMLP {
    let gateWeight: MTLBuffer  // Gate projection weights (for SwiGLU) (Shape: [embed_dim, hidden_dim])
    let upWeight: MTLBuffer  // Up projection weights (for SwiGLU)   (Shape: [embed_dim, hidden_dim])
    let downWeight: MTLBuffer  // Down projection weights             (Shape: [hidden_dim, embed_dim])

    // Note: FFN norm weights are stored in LlamaTransformerBlock.
}

/// Represents a single Transformer block in the Llama model.
struct LlamaTransformerBlock {
    // Normalization layers within the block
    let attentionNormWeight: MTLBuffer  // Weights for RMSNorm before attention
    let ffnNormWeight: MTLBuffer  // Weights for RMSNorm before MLP/FFN

    // Core mechanism weights
    let attention: LlamaAttention
    let mlp: LlamaMLP

    // Note: Residual connections are handled during computation.
}

// MARK: - Llama Model

/// Represents the complete Llama model structure, holding configuration and all weights.
/// Using a class for potential state management (like KV Cache) later.
class LlamaModel {
    let config: LlamaConfig

    // Input Embedding Layer
    let tokenEmbeddings: MTLBuffer  // Weight matrix for token embeddings (Shape: [vocab_size, embed_dim])

    // Transformer Blocks (Layers)
    let blocks: [LlamaTransformerBlock]

    // Final Output Layers
    let finalNormWeight: MTLBuffer  // Weights for the final RMSNorm after all blocks
    let outputWeight: MTLBuffer  // Output projection weights (often called lm_head) (Shape: [embed_dim, vocab_size])

    // TODO: Add KV Cache structure here later for inference state

    /// Initializes the LlamaModel with all its components.
    init(
        config: LlamaConfig,
        tokenEmbeddings: MTLBuffer,
        blocks: [LlamaTransformerBlock],
        finalNormWeight: MTLBuffer,
        outputWeight: MTLBuffer
    ) {
        precondition(
            blocks.count == config.numLayers,
            "Number of blocks (\(blocks.count)) must match config.numLayers (\(config.numLayers))")
        self.config = config
        self.tokenEmbeddings = tokenEmbeddings
        self.blocks = blocks
        self.finalNormWeight = finalNormWeight
        self.outputWeight = outputWeight

        print("--- LlamaModel Assembled ---")
        print("  Config loaded.")
        print("  Token Embeddings buffer assigned.")
        print("  \(blocks.count) Transformer Blocks assembled.")
        print("  Final Norm buffer assigned.")
        print("  Output Weight buffer assigned.")
        print("---------------------------")
    }

    // TODO: Add inference methods here later (e.g., forward pass)
}

