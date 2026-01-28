#include "precomp.hpp"
#include <opencv2/dnn/tokenizer.hpp>
#include <map>
#include <vector>
#include <string>
#include <limits>

namespace cv {
namespace dnn {

// --- Base Class Definition ---
// Defining this here fixes the "Undefined symbols" for Tokenizer typeinfo and vtable.
Mat Tokenizer::tokensToMat(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        return Mat();
    }
    // Create a 1 x N matrix of type 32-bit Signed Integer.
    // We use .clone() so the Mat owns its data independently of the input vector.
    Mat m(1, (int)tokens.size(), CV_32S, (void*)tokens.data());
    return m.clone();
}

// --- BPE Implementation Class ---
class TokenizerBPE : public Tokenizer {
private:
    std::map<std::string, int> vocab;
    std::map<std::pair<std::string, std::string>, int> merge_ranks;

public:
    TokenizerBPE(const std::string& vocabPath) {
        if (!vocabPath.empty()) {
            load(vocabPath);
        }
    }

    void load(const std::string& path) override {
        FileStorage fs(path, FileStorage::READ);
        if (!fs.isOpened()) return;

        // Clear existing data if re-loading
        vocab.clear();
        merge_ranks.clear();

        // Load Vocab: {"token": id}
        FileNode v = fs["vocab"];
        for (FileNodeIterator it = v.begin(); it != v.end(); ++it) {
            vocab[(*it).name()] = (int)*it;
        }

        // Load Merges: ["char1 char2", "char3 char4"]
        FileNode m = fs["merges"];
        int rank = 0;
        for (FileNodeIterator it = m.begin(); it != m.end(); ++it) {
            std::string pair_str = (std::string)*it;
            size_t space = pair_str.find(' ');
            if (space != std::string::npos) {
                auto p = std::make_pair(pair_str.substr(0, space), pair_str.substr(space + 1));
                merge_ranks[p] = rank++;
            }
        }
    }

    std::vector<int> encode(const std::string& text) override {
        if (text.empty()) return {};

        // Start with individual characters
        std::vector<std::string> symbols;
        for (char c : text) {
            symbols.push_back(std::string(1, c));
        }



        while (symbols.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            int best_idx = -1;

            for (size_t i = 0; i < symbols.size() - 1; ++i) {
                auto p = std::make_pair(symbols[i], symbols[i+1]);
                if (merge_ranks.count(p) && merge_ranks[p] < best_rank) {
                    best_rank = merge_ranks[p];
                    best_idx = (int)i;
                }
            }

            if (best_idx == -1) break;

            // Merge the best pair
            std::vector<std::string> next_symbols;
            for (int i = 0; i < (int)symbols.size(); ++i) {
                if (i == best_idx) {
                    next_symbols.push_back(symbols[i] + symbols[i+1]);
                    i++;
                } else {
                    next_symbols.push_back(symbols[i]);
                }
            }
            symbols = next_symbols;
        }

        std::vector<int> ids;
        for (const auto& s : symbols) {
            ids.push_back(vocab.count(s) ? vocab[s] : -1);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& tokens) override {
        // Create an inverse map for decoding
        std::map<int, std::string> inv_vocab;
        for (auto const& [str, id] : vocab) {
            inv_vocab[id] = str;
        }

        std::string result;
        for (int id : tokens) {
            if (inv_vocab.count(id)) result += inv_vocab[id];
        }
        return result;
    }
};

// --- Factory Method ---
Ptr<Tokenizer> Tokenizer::createBPE(const std::string& vocabPath) {
    return makePtr<TokenizerBPE>(vocabPath);
}

} // namespace dnn
} // namespace cv