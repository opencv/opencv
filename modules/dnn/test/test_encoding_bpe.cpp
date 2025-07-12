#include "test_precomp.hpp"
#include "../src/tokenizer/core_bpe.hpp"
#include "../src/tokenizer/encoding.hpp"
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





}}