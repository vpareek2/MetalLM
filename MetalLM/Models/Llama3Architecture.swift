// MetalLM/Models/Llama3Architecture.swift

import Foundation
import Metal  // Required for MTLBuffer

// MARK: - Configuration Error

enum ConfigError: Error {
    case missingMetadata(key: String)
    case invalidMetadataType(key: String, expected: String)
    case configurationError(message: String)
    case unknownRopeScalingType(String)  // Added for RoPE
}

// MARK: - Llama Configuration

// Add enum for RoPE Scaling Type
enum RopeScalingType: String {
    case none = "none"
    case linear = "linear"
    case yarn = "yarn"
}

/// Holds the hyperparameters and architectural details of a Llama model,
/// typically loaded from GGUF metadata.
struct LlamaConfig {
    // Core Dimensions
    let embeddingDim: Int
    let hiddenDim: Int
    let headDim: Int
    let numLayers: Int
    let numHeads: Int
    let numKeyValueHeads: Int

    // Vocabulary & Context
    let vocabSize: Int
    let sequenceLength: Int  // Max context length

    // Normalization
    let rmsNormEps: Float

    // --- RoPE ---
    let ropeDimensionCount: Int
    let ropeFreqBase: Float
    // Scaling Parameters (Including YaRN)
    let ropeScalingType: RopeScalingType  // Type of scaling used
    let ropeScalingFactor: Float  // General scaling factor (used by linear/yarn)
    let ropeScalingOrigContextLength: Int  // Original context length for scaling calculations
    let ropeScalingFinetuned: Bool  // Whether the model was finetuned with scaling
    // YaRN Specific (often derived or have defaults)
    let ropeScalingBetaFast: Float  // YaRN param
    let ropeScalingBetaSlow: Float  // YaRN param
    // --- End RoPE ---

    // Calculated properties
    var numQueryGroups: Int { numHeads / numKeyValueHeads }

    /// Initializes the configuration by parsing metadata from a GGUF file.
    init(metadata: [String: GGUFValue]) throws {
        // --- Helper Functions ---
        func getRequiredUInt64(_ key: String) throws -> UInt64 {
            guard let value = metadata[key]?.uint64 else {
                throw ConfigError.missingMetadata(key: key)
            }
            return value
        }
        func getOptionalUInt64(_ key: String, default defaultValue: UInt64) -> UInt64 {
            return metadata[key]?.uint64 ?? defaultValue
        }
        func getRequiredUInt32(_ key: String) throws -> UInt32 {
            guard let value = metadata[key]?.uint32 else {
                throw ConfigError.missingMetadata(key: key)
            }
            return value
        }
        func getOptionalUInt32(_ key: String, default defaultValue: UInt32) -> UInt32 {
            return metadata[key]?.uint32 ?? defaultValue
        }
        func getRequiredFloat32(_ key: String) throws -> Float {
            guard case .float32(let value) = metadata[key] else {
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
        func getOptionalString(_ key: String, default defaultValue: String) -> String {
            return metadata[key]?.string ?? defaultValue
        }
        func getOptionalBool(_ key: String, default defaultValue: Bool) -> Bool {
            guard case .bool(let value) = metadata[key] else {
                return defaultValue
            }
            return value
        }
        // --- End Helper Functions ---

        // --- Extract Core Dimensions ---
        self.embeddingDim = Int(try getRequiredUInt64("llama.embedding_length"))
        self.numLayers = Int(try getRequiredUInt64("llama.block_count"))
        self.numHeads = Int(try getRequiredUInt64("llama.attention.head_count"))
        self.numKeyValueHeads = Int(try getRequiredUInt64("llama.attention.head_count_kv"))
        self.hiddenDim = Int(try getRequiredUInt64("llama.feed_forward_length"))
        self.vocabSize = Int(try getRequiredUInt64("llama.vocab_size"))
        self.sequenceLength = Int(try getRequiredUInt64("llama.context_length"))
        self.rmsNormEps = try getRequiredFloat32("llama.attention.layer_norm_rms_epsilon")

        // --- Extract RoPE Parameters ---
        let defaultRopeDim = embeddingDim / numHeads  // Default RoPE dim = head dim
        self.ropeDimensionCount = Int(
            getOptionalUInt64("llama.rope.dimension_count", default: UInt64(defaultRopeDim)))
        // Your GGUF dump showed 500k, let's keep that as default maybe? Or 10k? Check common Llama3 base.
        self.ropeFreqBase = getOptionalFloat32("llama.rope.freq_base", default: 500000.0)

        // --- Extract RoPE Scaling/YaRN Parameters ---
        let scalingTypeString = getOptionalString("llama.rope.scaling.type", default: "none")
        guard let scalingType = RopeScalingType(rawValue: scalingTypeString.lowercased()) else {
            throw ConfigError.unknownRopeScalingType(scalingTypeString)
        }
        self.ropeScalingType = scalingType

        // Factor applies to linear and yarn
        self.ropeScalingFactor = getOptionalFloat32("llama.rope.scaling.factor", default: 1.0)

        // Original context length - use current sequenceLength as default if missing
        self.ropeScalingOrigContextLength = Int(
            getOptionalUInt32(
                "llama.rope.scaling.original_context_length", default: UInt32(self.sequenceLength)))

        self.ropeScalingFinetuned = getOptionalBool("llama.rope.scaling.finetuned", default: false)

        // YaRN betas - use common defaults if missing (check llama.cpp defaults if unsure)
        self.ropeScalingBetaFast = getOptionalFloat32("llama.rope.scaling.beta_fast", default: 32.0)
        self.ropeScalingBetaSlow = getOptionalFloat32("llama.rope.scaling.beta_slow", default: 1.0)

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

        // --- Print Loaded Config ---
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
        print(
            String(
                format: "  RoPE Scaling: Type=%@, Factor=%.2f, OrigCtx=%d, Tuned=%@",
                ropeScalingType.rawValue, ropeScalingFactor, ropeScalingOrigContextLength,
                String(describing: ropeScalingFinetuned)))
        print(
            String(
                format: "  RoPE YaRN: BetaFast=%.1f, BetaSlow=%.1f", ropeScalingBetaFast,
                ropeScalingBetaSlow))
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

    // RoPE Frequencies
    let ropeFrequencies: MTLBuffer?  // Holds factors from rope_freqs.weight (optional)

    // TODO: Add KV Cache structure here later for inference state

    /// Initializes the LlamaModel with all its components.
    init(
        config: LlamaConfig,
        tokenEmbeddings: MTLBuffer,
        blocks: [LlamaTransformerBlock],
        finalNormWeight: MTLBuffer,
        outputWeight: MTLBuffer,
        ropeFrequencies: MTLBuffer?
    ) {
        precondition(
            blocks.count == config.numLayers,
            "Number of blocks (\(blocks.count)) must match config.numLayers (\(config.numLayers))")
        self.config = config
        self.tokenEmbeddings = tokenEmbeddings
        self.blocks = blocks
        self.finalNormWeight = finalNormWeight
        self.outputWeight = outputWeight
        self.ropeFrequencies = ropeFrequencies

        print("--- LlamaModel Assembled ---")
        print("  Config loaded.")
        print("  Token Embeddings buffer assigned.")
        print("  \(blocks.count) Transformer Blocks assembled.")
        print("  Final Norm buffer assigned.")
        print("  Output Weight buffer assigned.")
        print("  RoPE Frequencies buffer \(ropeFrequencies == nil ? "NOT " : "")assigned.")
        print("---------------------------")
    }

    // TODO: Add inference methods here later (e.g., forward pass)
}
