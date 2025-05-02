import Foundation

enum TokenizerError: Error {
    case ggufFileNotLoaded
    case metadataNotFound(String)
    case invalidTokenID
}

class Tokenizer {
    private let ggufFile: GGUFFile
    private let vocabulary: [String]
    private let merges: [(String, String, String)]
    private let bosTokenID: Int
    private let eosTokenID: Int
    private let vocabSize: Int

    init(ggufFile: GGUFFile) throws {
        self.ggufFile = ggufFile
        
        guard let tokensValue = ggufFile.metadata["tokenizer.ggml.tokens"],
              case let .array(tokensArray) = tokensValue,
              let tokens = tokensArray.compactMap({ $0.string }) as [String]? else {
            throw TokenizerError.metadataNotFound("tokenizer.ggml.tokens is not a string array")
        }
        self.vocabulary = tokens
        self.vocabSize = tokens.count
        
        guard let mergesValue = ggufFile.metadata["tokenizer.ggml.merges"],
              case let .array(mergesArray) = mergesValue,
              let mergesStrings = mergesArray.compactMap({ $0.string }) as [String]? else {
            throw TokenizerError.metadataNotFound("tokenizer.ggml.merges is not a string array")
        }
        self.merges = mergesStrings.map { merge in
            let parts = merge.split(separator: " ").map { String($0) }
            return (parts[0], parts[1], parts[0] + parts[1])
        }
        
        guard let bosValue = ggufFile.metadata["tokenizer.ggml.bos_token_id"],
              case let .uint32(bosInt) = bosValue else {
            throw TokenizerError.metadataNotFound("tokenizer.ggml.bos_token_id is not a uint32")
        }
        self.bosTokenID = Int(bosInt)
        
        guard let eosValue = ggufFile.metadata["tokenizer.ggml.eos_token_id"],
              case let .uint32(eosInt) = eosValue else {
            throw TokenizerError.metadataNotFound("tokenizer.ggml.eos_token_id is not a uint32")
        }
        self.eosTokenID = Int(eosInt)
        
        print("Tokenizer initialized with \(vocabSize) tokens, BOS=\(bosTokenID), EOS=\(eosTokenID), \(merges.count) merges")
    }

    func tokenize(_ text: String) -> [Int] {
        var tokens = [bosTokenID]
        var chars = Array(text.utf8).map { String(UnicodeScalar($0)) }
        
        while true {
            var bestScore: Float = -Float.infinity
            var bestMerge: (String, String, String)? = nil
            var bestIndex = 0
            
            for i in 0..<(chars.count - 1) {
                let pair = chars[i] + chars[i + 1]
                for merge in merges {
                    if merge.2 == pair {
                        let score = Float(vocabulary.firstIndex(of: merge.2) ?? vocabSize)
                        if score > bestScore {
                            bestScore = score
                            bestMerge = merge
                            bestIndex = i
                        }
                    }
                }
            }
            
            if let merge = bestMerge {
                chars[bestIndex] = merge.2
                chars.remove(at: bestIndex + 1)
            } else {
                break
            }
        }
        
        for token in chars {
            if let index = vocabulary.firstIndex(of: token) {
                tokens.append(index)
            } else if let unkIndex = vocabulary.firstIndex(of: "<unk>") {
                tokens.append(unkIndex)
            }
        }
        
        return tokens
    }

    func decode(_ tokenIDs: [Int]) -> String {
        return tokenIDs.compactMap { tokenID in
            guard tokenID >= 0 && tokenID < vocabSize else { return nil }
            return vocabulary[tokenID]
        }.joined()
    }

    func getVocabSize() -> Int {
        return vocabSize
    }

    func getBosTokenID() -> Int {
        return bosTokenID
    }

    func getEosTokenID() -> Int {
        return eosTokenID
    }
}
