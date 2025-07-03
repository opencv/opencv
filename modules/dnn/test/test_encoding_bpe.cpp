#include "test_precomp.hpp"
#include "../src/tokenizertokens/core_bpe.hpp"
#include "../src/tokenizertokens/encoding.hpp"

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

}}