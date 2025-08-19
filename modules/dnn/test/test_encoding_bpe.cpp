#include "test_precomp.hpp"
#include "../src/tokenizer/core_bpe.hpp"
#include "../src/tokenizer/encoding.hpp"
#include "../src/tokenizer/utils.hpp"
#include "../include/opencv2/dnn/tokenizer.hpp"
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


TEST(EncodingBPE, EncodingDecode_GPT2) {
   Tokenizer gpt2_tok = Tokenizer::load(_tf_gpt2(""));
    std::string sent = gpt2_tok.decode({31373, 995});
    std::string expected = "hello world";
    EXPECT_EQ(sent, expected);
}

TEST(EncodingBPE, EncodeWithAllowedSpecial_ALL) {
    Tokenizer gpt2_tok = Tokenizer::load(_tf_gpt2(""));
    // "__ALL__" is  sentinel for all special tokens
    std::unordered_set<std::string> allowedSpecial = {"__ALL__"};
    std::vector<int> tokens = gpt2_tok.encode("hello <|endoftext|>", allowedSpecial);
    std::vector<int> expected = {31373, 220, 50256}; // OpenAI GPT-2 tokens for this input
    EXPECT_EQ(tokens, expected);
}

TEST(EncodingBPE, SimpleRepeated_GPT2) {
    Tokenizer gpt2_tok = Tokenizer::load(_tf_gpt2(""));
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
    Tokenizer gpt2_tok = Tokenizer::load(_tf_gpt2(""));
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


TEST(EncodingBPE, Tokenizer_GPT2) {
    std::string gpt2_dir = _tf_gpt2("");
    Tokenizer tok = Tokenizer::load(gpt2_dir);
    auto ids = tok.encode("hello world");
    auto txt = tok.decode(ids);
    EXPECT_EQ(txt, "hello world");
}

TEST(EncodingBPE, GPT2Fast) {
    Tokenizer tok = Tokenizer::load(_tf_gpt2(""));
    std::vector<int> ids = tok.encode("hello world");
    std::vector<int> expected{31373, 995};
    EXPECT_EQ(ids, expected);
}

}}
