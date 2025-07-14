#include "test_precomp.hpp"
#include "../src/tokenizer/core_bpe.hpp"
#include "../src/tokenizer/encoding.hpp"
#include "../src/tokenizer/utils.hpp"
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace opencv_test { namespace  {

using namespace cv::dnn::tokenizer;

TEST(EncodingBPE, EncodingOrdinary_GPT2) {
    Encoding enc = getEncodingForGPT2("gpt2");
    std::vector<Rank> tokens = enc.encodeOrdinary("hello world");
    std::vector<Rank> expected = {31373, 995}; // OpenAI GPT-2 tokens for "hello world"
    EXPECT_EQ(tokens, expected);
}

TEST(EncodingBPE, EncodingDecode_GPT2) {
    Encoding enc = getEncodingForGPT2("gpt2");
    std::string sent = enc.decode({31373, 995});
    std::string expected = "hello world";
    EXPECT_EQ(sent, expected);
}

TEST(EncodingBPE, EncodeWithAllowedSpecial_ALL) {
    Encoding enc = getEncodingForGPT2("gpr2");
    // "__ALL__" is  sentinel for all special tokens
    std::unordered_set<std::string> allowedSpecial = {"__ALL__"};
    std::vector<Rank> tokens = enc.encode("hello <|endoftext|>", allowedSpecial);
    std::vector<Rank> expected = {31373, 220, 50256}; // OpenAI GPT-2 tokens for this input
    EXPECT_EQ(tokens, expected);
}

TEST(EncodingBPE, SimpleRepeated_GPT2) {
    Encoding enc = getEncodingForGPT2("gpt2");
    EXPECT_EQ(enc.encode("0"), std::vector<Rank>({15}));
    EXPECT_EQ(enc.encode("00"), std::vector<Rank>({405}));
    EXPECT_EQ(enc.encode("000"), std::vector<Rank>({830}));
    EXPECT_EQ(enc.encode("0000"), std::vector<Rank>({2388}));
    EXPECT_EQ(enc.encode("00000"), std::vector<Rank>({20483}));
    EXPECT_EQ(enc.encode("000000"), std::vector<Rank>({10535}));
    EXPECT_EQ(enc.encode("0000000"), std::vector<Rank>({24598}));
    EXPECT_EQ(enc.encode("00000000"), std::vector<Rank>({8269}));
    EXPECT_EQ(enc.encode("000000000"), std::vector<Rank>({10535, 830}));
    EXPECT_EQ(enc.encode("0000000000"), std::vector<Rank>({8269, 405}));
    EXPECT_EQ(enc.encode("00000000000"), std::vector<Rank>({8269, 830}));
    EXPECT_EQ(enc.encode("000000000000"), std::vector<Rank>({8269, 2388}));
    EXPECT_EQ(enc.encode("0000000000000"), std::vector<Rank>({8269, 20483}));
    EXPECT_EQ(enc.encode("00000000000000"), std::vector<Rank>({8269, 10535}));
    EXPECT_EQ(enc.encode("000000000000000"), std::vector<Rank>({8269, 24598}));
    EXPECT_EQ(enc.encode("0000000000000000"), std::vector<Rank>({25645}));
    EXPECT_EQ(enc.encode("00000000000000000"), std::vector<Rank>({8269, 10535, 830}));
}

TEST(EncodingBPE, CatastrophicallyRepetitive_GPT2) {
    Encoding enc = getEncodingForGPT2("gpt2");
    std::vector<std::string> chars = {"^", "0", "a", "'s", " ", "\n"};
    for (const auto& c : chars) {
        std::string big_value(c.size() == 1 ? 10000 : 10000 * c.size(), c[0]);
        if (c == "'s") big_value = std::string(10000, '\'') + std::string(10000, 's');
        EXPECT_EQ(big_value, enc.decode(enc.encode(big_value)));

        std::string with_space = " " + big_value;
        EXPECT_EQ(with_space, enc.decode(enc.encode(with_space)));

        std::string with_newline = big_value + "\n";
        EXPECT_EQ(with_newline, enc.decode(enc.encode(with_newline)));
    }
}

TEST(EncodingBPE, TrainAndEncodeDecode_Simple) {
    // Pattern string (same as your Python example)
    std::string gpt2_pattern = 
        "'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    // Read this source file as training data
    std::ifstream f(__FILE__);
    ASSERT_TRUE(f.is_open());
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string data = buffer.str();

    // Train encoding
    int vocab_size = 600;
    // When calling this constructor to create an Encoding 
    // internally it calls the train_bpe(text, vocabSize, /*verbose=*/false)
    // For now this is how we call an Encoder to train the tokenizer bpe.
    // TODO:
    // Might be more helpful to just create a default Encoder than call 
    // the a train function such as Encoding enc() then enc.train_bpe();
    Encoding enc(data, vocab_size, gpt2_pattern);

    // Encode and decode "hello world"
    std::vector<Rank> tokens = enc.encodeOrdinary("hello world");
    for (auto tok : tokens) std::cerr << tok << " ";
    std::cout << std::endl;
    std::string decoded = enc.decode(tokens);

    EXPECT_EQ(decoded, "hello world");

    std::vector<std::uint8_t> bytes = enc.decodeBytes(tokens);
    std::string bytes_str(bytes.begin(), bytes.end());
    std::cerr << bytes_str << std::endl;
    std::string res = replaceGWithSpace(bytes_str);
    EXPECT_EQ(res, "hello world");
}


TEST(EncodingBPE, TrainOnTaylorSwiftAndEncodeDecode) {
    std::string gpt2_pattern =
        "'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    // Read the Taylor Swift Wikipedia article as training data
    std::ifstream f("../modules/dnn/src/tokenizer/taylorswift.txt");
    ASSERT_TRUE(f.is_open());
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string data = buffer.str();

    int vocab_size = 512;
    Encoding enc(data, vocab_size, gpt2_pattern);

    std::string test_str = "hello world";
    std::vector<Rank> tokens = enc.encode(test_str);
    std::string decoded = enc.decode(tokens);

    EXPECT_EQ(decoded, test_str);
}


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

    Encoding enc(words, 30, R50K_UTF8, 2, std::numeric_limits<int>::max(), true);

    std::unordered_map<std::string, int> expected_vocab = {
        {"-", 0}, {"2", 1}, {"B", 2}, {"E", 3}, {"G", 4}, {"P", 5}, {"R", 6},
        {"T", 7}, {"a", 8}, {"b", 9}, {"d", 10}, {"e", 11}, {"g", 12},
        {"i", 13}, {"l", 14}, {"n", 15}, {"o", 16}, {"r", 17}, {"s", 18},
        {"t", 19}, {"u", 20}, {"v", 21}, {"re", 22}, {"are", 23}, {"is", 24}
    };

    std::unordered_map<std::string, int> actual_vocab;
    for (const auto& kv : enc.getVocab()) {
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
    const auto& merges = enc.getMerges();
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

    Encoding enc(words, vocab_size, CL100K_BASE, min_freq, 2, true);

    std::unordered_map<std::string, int> expected_vocab = {
        {"短", 12}, {"n", 6}, {"i", 5}, {"s", 8}, {"字符", 23}, {"長", 14}, {"긴", 17},
        {"い文", 22}, {"L", 2}, {"in", 21}, {"o", 7}, {"은한", 29}, {"S", 4}, {"P", 3},
        {"so", 27}, {"符", 13}, {"文", 11}, {"字", 10}, {"짧", 19}, {"GP", 25}, {"글", 16},
        {"G", 1}, {"An", 24}, {"长", 15}, {"A", 0}, {"Lo", 26}, {"긴한", 28}, {"い", 9},
        {"한", 20}, {"은", 18},
    };

    std::unordered_map<std::string, int> actual_vocab;
    for (const auto& kv : enc.getVocab()) {
        std::string s(kv.second.begin(), kv.second.end());
        actual_vocab[s] = kv.first;
    }
    EXPECT_EQ(actual_vocab, expected_vocab);
}

}}