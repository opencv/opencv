#include "test_precomp.hpp"
#include "../src/tokenizer/core_bpe.hpp"
#include "../src/tokenizer/encoding.hpp"
#include <fstream>
#include <sstream>

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
    std::string decoded = enc.decode(tokens);

    EXPECT_EQ(decoded, "hello world");

    std::vector<std::uint8_t> bytes = enc.decodeBytes(tokens);
    std::string bytes_str(bytes.begin(), bytes.end());
    std::string res = replaceGWithSpace(bytes_str);
    EXPECT_EQ(res, "hello world");
}

}}