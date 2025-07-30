#include "test_precomp.hpp"
#include "../src/tokenizer/core_bpe.hpp"
#include "../src/tokenizer/encoding.hpp"
#include "../src/tokenizer/utils.hpp"
#include "../include/opencv2/dnn/tokenizer.hpp"
#include "../src/tokenizer/gpt2_tokenizer_fast.hpp"
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <algorithm>

namespace opencv_test { namespace  {

using namespace cv::dnn::tokenizer;

template<typename tstring>
static std::string _tf_gpt2(tstring filename) {
    std::string basetestdir = getOpenCVExtraDir();
    size_t len = basetestdir.size();
    if (len > 0 && basetestdir[len-1] != '/' && basetestdir[len-1] != '\\')
        return (basetestdir + "/testdata/dnn/llm/gpt2") + filename;
    return (basetestdir + "testdata/dnn/llm/gpt2/") + filename;
}

template<typename tstring>
static std::string _tf_gpt4(tstring filename) {
    std::string basetestdir = getOpenCVExtraDir();
    size_t len = basetestdir.size();
    if (len > 0 && basetestdir[len-1] != '/' && basetestdir[len-1] != '\\')
        return (basetestdir + "/testdata/dnn/llm/gpt4") + filename;
    return (basetestdir + "testdata/dnn/llm/gpt4/") + filename;
}

template<typename tstring>
static std::string _tf_wikitext(tstring filename) {
    std::string basetestdir = getOpenCVExtraDir();
    size_t len = basetestdir.size();
    if (len > 0 && basetestdir[len-1] != '/' && basetestdir[len-1] != '\\')
        return (basetestdir + "/testdata/dnn/llm/wikitext") + filename;
    return (basetestdir + "testdata/dnn/llm/wikitext/") + filename;
}

TEST(EncodingBPE, EncodingOrdinary_GPT2) {
    std::string vocab_bpe = _tf_gpt2("vocab.bpe");
    GPT2TokenizerFast gpt2_tok = GPT2TokenizerFast::from_pretrained(vocab_bpe);
    std::vector<Rank> tokens = gpt2_tok.encodeOrdinary("hello world");
    std::vector<Rank> expected = {31373, 995}; // OpenAI GPT-2 tokens for "hello world"
    EXPECT_EQ(tokens, expected);
}

TEST(EncodingBPE, EncodingDecode_GPT2) {
    std::string vocab_bpe = _tf_gpt2("vocab.bpe");
    GPT2TokenizerFast gpt2_tok = GPT2TokenizerFast::from_pretrained(vocab_bpe);
    std::string sent = gpt2_tok.decode({31373, 995});
    std::string expected = "hello world";
    EXPECT_EQ(sent, expected);
}

TEST(EncodingBPE, EncodeWithAllowedSpecial_ALL) {
    std::string vocab_bpe = _tf_gpt2("vocab.bpe");
    GPT2TokenizerFast gpt2_tok = GPT2TokenizerFast::from_pretrained(vocab_bpe);
    // "__ALL__" is  sentinel for all special tokens
    std::unordered_set<std::string> allowedSpecial = {"__ALL__"};
    std::vector<int> tokens = gpt2_tok.encode("hello <|endoftext|>", allowedSpecial);
    std::vector<int> expected = {31373, 220, 50256}; // OpenAI GPT-2 tokens for this input
    EXPECT_EQ(tokens, expected);
}

TEST(EncodingBPE, SimpleRepeated_GPT2) {
    std::string vocab_bpe = _tf_gpt2("vocab.bpe");
    GPT2TokenizerFast gpt2_tok = GPT2TokenizerFast::from_pretrained(vocab_bpe);
    EXPECT_EQ(gpt2_tok.encode("0"), std::vector<int>({15}));
    EXPECT_EQ(gpt2_tok.encode("00"), std::vector<int>({405}));
    EXPECT_EQ(gpt2_tok.encode("000"), std::vector<int>({830}));
    EXPECT_EQ(gpt2_tok.encode("0000"), std::vector<int>({2388}));
    EXPECT_EQ(gpt2_tok.encode("00000"), std::vector<int>({20483}));
    EXPECT_EQ(gpt2_tok.encode("000000"), std::vector<int>({10535}));
    EXPECT_EQ(gpt2_tok.encode("0000000"), std::vector<int>({24598}));
    EXPECT_EQ(gpt2_tok.encode("00000000"), std::vector<int>({8269}));
    EXPECT_EQ(gpt2_tok.encode("000000000"), std::vector<int>({10535, 830}));
    EXPECT_EQ(gpt2_tok.encode("0000000000"), std::vector<int>({8269, 405}));
    EXPECT_EQ(gpt2_tok.encode("00000000000"), std::vector<int>({8269, 830}));
    EXPECT_EQ(gpt2_tok.encode("000000000000"), std::vector<int>({8269, 2388}));
    EXPECT_EQ(gpt2_tok.encode("0000000000000"), std::vector<int>({8269, 20483}));
    EXPECT_EQ(gpt2_tok.encode("00000000000000"), std::vector<int>({8269, 10535}));
    EXPECT_EQ(gpt2_tok.encode("000000000000000"), std::vector<int>({8269, 24598}));
    EXPECT_EQ(gpt2_tok.encode("0000000000000000"), std::vector<int>({25645}));
    EXPECT_EQ(gpt2_tok.encode("00000000000000000"), std::vector<int>({8269, 10535, 830}));
}

TEST(EncodingBPE, CatastrophicallyRepetitive_GPT2) {
    std::string vocab_bpe = _tf_gpt2("vocab.bpe");
    GPT2TokenizerFast gpt2_tok = GPT2TokenizerFast::from_pretrained(vocab_bpe);
    std::vector<std::string> chars = {"^", "0", "a", "'s", " ", "\n"};
    for (const auto& c : chars) {
        std::string big_value(c.size() == 1 ? 10000 : 10000 * c.size(), c[0]);
        if (c == "'s") big_value = std::string(10000, '\'') + std::string(10000, 's');
        EXPECT_EQ(big_value, gpt2_tok.decode(gpt2_tok.encode(big_value)));

        std::string with_space = " " + big_value;
        EXPECT_EQ(with_space, gpt2_tok.decode(gpt2_tok.encode(with_space)));

        std::string with_newline = big_value + "\n";
        EXPECT_EQ(with_newline, gpt2_tok.decode(gpt2_tok.encode(with_newline)));
    }
}

TEST(EncodingBPE, TrainAndEncodeDecode_Simple) {
    std::ifstream f(__FILE__);
    ASSERT_TRUE(f.is_open());
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string data = buffer.str();

    // Train encoding using train_bpe v1
    int vocab_size = 600;
    Tokenizer tok = Tokenizer::train_bpe_from_corpus(data, vocab_size, R50K_UTF8);

    // Encode and decode "hello world"
    std::vector<Rank> tokens = tok.encodeOrdinary("hello world");
    for (auto tok : tokens) std::cerr << tok << " ";
    std::cout << std::endl;
    std::vector<int> tokens_(tokens.begin(), tokens.end());
    std::string decoded = tok.decode(tokens_);

    EXPECT_EQ(decoded, "hello world");

    std::vector<std::uint8_t> bytes = tok.decodeBytes(tokens);
    std::string bytes_str(bytes.begin(), bytes.end());
    std::cerr << bytes_str << std::endl;
    std::string res = replaceGWithSpace(bytes_str);
    EXPECT_EQ(res, "hello world");
}

// [TODO] too slow to compute training on large data
// TEST(EncodingBPE, TrainOnWikiText) 
//     std::string wiki_train = _tf_wikitext("wiki.train.raw");
//     std::ifstream f(wiki_train);
//     ASSERT_TRUE(f.is_open());
//     std::stringstream buffer;
//     buffer << f.rdbuf();
//     std::string data = buffer.str();

//     // Vanilla gpt2
//     int vocab_size = 50257;
//     Tokenizer tok = Tokenizer::train_bpe_from_corpus(data, vocab_size, R50K_UTF8);

//     std::string test_str = "hello world";
//     std::vector<int> tokens = tok.encode(test_str);
//     std::string decoded = tok.decode(tokens);

//     EXPECT_EQ(decoded, test_str);
// }


TEST(EncodingBPE, TrainHuggingFaceStyle) {
    std::unordered_map<std::string, int> word_counts = {
        {"roses", 1},
        {"are", 2},
        {"red", 1},
        {"voilets", 1},
        {"blue", 1},
        {"BERT", 1},
        {"is", 2},
        {"big", 1},
        {"and", 1},
        {"so", 1},
        {"GPT-2", 1},
    };
    std::vector<std::string> words;
    for (const auto& kv : word_counts) {
        for (int i = 0; i < kv.second; ++i)
            words.push_back(kv.first);
    }

    Tokenizer tok = Tokenizer::train_bpe_from_corpus(words, 
                                                     30, 
                                                     R50K_UTF8, 
                                                     2, 
                                                     std::numeric_limits<int>::max(), 
                                                     true);

    std::unordered_map<std::string, int> expected_vocab = {
        {"-", 0}, {"2", 1}, {"B", 2}, {"E", 3}, {"G", 4}, {"P", 5}, {"R", 6},
        {"T", 7}, {"a", 8}, {"b", 9}, {"d", 10}, {"e", 11}, {"g", 12},
        {"i", 13}, {"l", 14}, {"n", 15}, {"o", 16}, {"r", 17}, {"s", 18},
        {"t", 19}, {"u", 20}, {"v", 21}, {"re", 22}, {"are", 23}, {"is", 24}
    };

    std::unordered_map<std::string, int> actual_vocab;
    for (const auto& kv : tok.encoding().getVocab()) {
        std::string s(kv.second.begin(), kv.second.end());
        actual_vocab[s] = kv.first;
    }
    EXPECT_EQ(actual_vocab, expected_vocab);

    std::map<std::pair<int,int>, int> expected_merges = {
        {{17, 11}, 22}, // 'r' + 'e'  -> 're'
        {{8, 22}, 23},  // 'a' + 're' -> 'are'
        {{13, 18}, 24}, // 'i' + 's'  -> 'is'
    };

    // 6. Check merges
    const auto& merges = tok.encoding().getMerges();
    for (const auto& kv : expected_merges) {
        auto it = merges.find(kv.first);
        ASSERT_TRUE(it != merges.end());
        EXPECT_EQ(it->second, kv.second);
    }
}

TEST(EncodingBPE, MaxTokenLengthDirectAssert) {
    std::unordered_map<std::string, int> word_counts = {
        {"sin", 2},
        {"Sin", 2},
        {"Lon", 2},
        {"Ano", 2},
        {"짧은한", 2},
        {"긴한글", 2},
        {"短字符", 2},
        {"长字符", 2},
        {"短い文", 2},
        {"長い文", 2},
        {"so", 2},
        {"GP", 2},
    };
    std::vector<std::string> words;
    for (const auto& kv : word_counts) {
        for (int i = 0; i < kv.second; ++i)
            words.push_back(kv.first);
    }

    int vocab_size = 40;
    int min_freq = 0;

    Tokenizer tok = Tokenizer::train_bpe_from_corpus(words, 
                                                     vocab_size, 
                                                     CL100K_BASE, 
                                                     min_freq, 
                                                     2, 
                                                     true);

    std::unordered_map<std::string, int> expected_vocab = {
        {"短", 12}, {"n", 6}, {"i", 5}, {"s", 8}, {"字符", 23}, {"長", 14}, {"긴", 17},
        {"い文", 22}, {"L", 2}, {"in", 21}, {"o", 7}, {"은한", 29}, {"S", 4}, {"P", 3},
        {"so", 27}, {"符", 13}, {"文", 11}, {"字", 10}, {"짧", 19}, {"GP", 25}, {"글", 16},
        {"G", 1}, {"An", 24}, {"长", 15}, {"A", 0}, {"Lo", 26}, {"긴한", 28}, {"い", 9},
        {"한", 20}, {"은", 18},
    };

    std::unordered_map<std::string, int> actual_vocab;
    for (const auto& kv : tok.encoding().getVocab()) {
        std::string s(kv.second.begin(), kv.second.end());
        actual_vocab[s] = kv.first;
    }

    EXPECT_EQ(actual_vocab, expected_vocab);
}

TEST(EncodingBPE, Encoding_GPT4) {
    std::string cl100k_base = _tf_gpt4("cl100k_base.tiktoken");
    Tokenizer tok = Tokenizer::from_pretrained("cl100k_base", cl100k_base);

    std::vector<int> tokens = tok.encode("hello world");
    std::vector<int> expected = {15339, 1917};
    EXPECT_EQ(tokens, expected);

    std::string sent = tok.decode({15339, 1917});
    std::string expec_str = "hello world";
    EXPECT_EQ(sent, expec_str);

    std::unordered_set<std::string> allowedSpecial = {"__ALL__"};
    std::vector<int> spec_tokens = tok.encode("hello <|endoftext|>", allowedSpecial);
    std::vector<int> expected_special = {15339, 220, 100257};
    EXPECT_EQ(spec_tokens, expected_special);

    Rank min = std::min(10000u, tok.encoding().maxTokenValue() - 1);
    for (Rank _token = 0; _token < min; _token++) {
        if (_token < 10) {
            std::vector<std::uint8_t> token_bytes = tok.decodeSingleTokenBytes(_token);
            std::string token_str(token_bytes.begin(), token_bytes.end());
            std::cout << "Token: " << _token << " | Decoded: " << token_str << std::endl;
        }
        Rank rank = tok.encodeSingleToken(tok.decodeSingleTokenBytes(_token));
        EXPECT_EQ(rank, _token);
    }
}

TEST(EncodingBPE, Tokenizer_GPT2) {
    std::string vocab_bpe = _tf_gpt2("vocab.bpe");
    Tokenizer tok = Tokenizer::from_pretrained("gpt2", vocab_bpe);
    auto ids = tok.encode("hello world");
    auto txt = tok.decode(ids);
    EXPECT_EQ(txt, "hello world");
}

TEST(EncodingBPE, GPT2Fast) {
    std::string vocab_bpe = _tf_gpt2("vocab.bpe");
    GPT2TokenizerFast tok = GPT2TokenizerFast::from_pretrained(vocab_bpe);
    std::vector<int> ids = tok.encode("hello world");
    std::vector<int> expected{31373, 995};
    EXPECT_EQ(ids, expected);
}

}}
