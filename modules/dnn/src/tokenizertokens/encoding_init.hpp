#include "load_tokenizer_bpe.hpp"
#include "encoding.hpp"
#include "encoding_registry.hpp"   

namespace cv { namespace dnn {namespace tokenizer {

static const std::string r50k_pat_str = R"REGEX('(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s)REGEX";

static const std::string cl100k_pat_str = R"REGEX('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s)REGEX";

static bool _reg_gpt2 = []() {
    auto mergeableRanks = dataGymToMergeableBpeRanks(
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
    );

    std::unordered_map<std::string, Rank> special = {
        {"<|endoftext|>", 50256}
    };

    registerEncoding(
        "gpt2",
        [mergeableRanks, special](){ return Encoding("gpts", r50k_pat_str, mergeableRanks, special, 50257);
        }
    );

    return true;
}();

static bool _reg_cl100k_base = []() {
    auto ranks = loadTokenizerBPE(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
    );

    // based on titokenizer settings 
    std::unordered_map<std::string, Rank> special = {
        {"<|endoftext|>", 100257},
        {"<|fim_prefix|>", 100258},
        {"<|fim_middle|>", 100259},
        {"<|fim_suffix|>", 100260},
        {"<|endofprompt|>", 100276},
    };
    registerEncoding(
        "cl100k_base",
        [ranks, special]() {
            return Encoding("cl100k_base", cl100k_pat_str, ranks, special);
        }  
    );
    return true;
}();




}}}

