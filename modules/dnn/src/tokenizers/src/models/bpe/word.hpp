#include <vector>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <unordered_map>

#ifndef __OPENCV_DNN_TOKENIZERS_MODELS_WORD.HPP__
#define __OPENCV_DNN_TOKENIZERS_MODELS_WORD.HPP__

namespace cv { namespace dnn { namespace tokenizer {

using Pair = std::pair<std::uint32_t, std::uint32_t>;

//Utility hash for Pair so that it can be used as a key in unordered_map
struct PairHash {
    std::size_t operator()(const Pair &p) const noexcept {
        // Simple combination hash (splitmix64 inspired)
        // https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
        std::size_t h1 = static_cast<std::size_t>(p.first);
        std::size_t h2 = static_cast<std::size_t>(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

class Word {
public: 
    Word() = default;
    explicit Word(std::size_t capacity) {
        symbols.reserve(capacity);
    }

    // Add a new UTF‑8 symbol (code‑point) and its byte‑length in the original
    // string. Handles the double‑linked‑list bookkeeping for prev/next indices.
    void add(std::uint32_t c, std::size_t byte_len);
    
    // Merge a **single** (c1,c2) pair into `replacement`. Returns a delta list
    // of pair‑frequency adjustments so the trainer can keep counts in sync.
    std::vector<std::pair<Pair, int>> merge(std::uint32_t c1, 
                                            std::uint32_t c2, 
                                            std::uint32_t replacement,
                                            std::size_t max_length);
                                            
    // Greedy merge of all possible pairs according to merges (rank → id).
    // Optional BPE‑dropout can be enabled by passing a probability in (0,1).
    void mergeAll(const std::unordered_map<Pair, std::pair<std::uint32_t, std::uint32_t>, PairHash>& merges,
                    std::optional<float> dropout=std::nullopt);

    // convenience helpers
    std::vector<std::uint32_t> getChars() const;
    std::vector<std::pair<std::size_t, std::size_t>> getOffsets() const;

private:
    struct Symbol {
        std::uint32_t c = 0;        // code-point / token id 
        std::ptrdiff_t prev = -1;   // index of previous symbol (-1 == none)
        std::ptrdiff_t next = -1;   // index of next symbol (-1 == none)
        std::size_t len = 0;        // length in bytes in original text

        void mergeWith(const Symbol &other, std::uint32_t new_c) {
            c = new_c;
            len += other.len;
            next = other.next;
        }
    };

    std::vector<Symbol> symbols;   // mutuable rope of symbols

};

}}}

#endif