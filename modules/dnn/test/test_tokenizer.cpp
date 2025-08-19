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
 
TEST(Tokenizer_BPE, Encoding_GPT4) {
    std::string gpt4_dir = _tf_gpt4("");
    Tokenizer tok = Tokenizer::load(gpt4_dir);

    std::vector<int> tokens = tok.encode("hello world");
    std::vector<int> expected = {15339, 1917};
    EXPECT_EQ(tokens, expected);

    std::string sent = tok.decode({15339, 1917});
    std::string expec_str = "hello world";
    EXPECT_EQ(sent, expec_str);

}

TEST(Tokenizer_BPE, Tokenizer_GPT2) {
    // std::string vocab_bpe = _tf_gpt2("vocab.bpe");
    std::string gpt2_dir = _tf_gpt2("");
    Tokenizer tok = Tokenizer::load(gpt2_dir);
    auto ids = tok.encode("hello world");
    for (auto id : ids) std::cout << id << " ";
    auto txt = tok.decode(ids);
    EXPECT_EQ(txt, "hello world");
}

TEST(Tokenizer_BPE, Tokenizer_GPT2_Model) {
    std::string gpt2_dir = getOpenCVExtraDir() + "testdata/dnn/llm/gpt2/";
    Tokenizer tok = Tokenizer::load(gpt2_dir);
    auto ids = tok.encode("hello world");
    auto text = tok.decode(ids);
    EXPECT_EQ(text, "hello world");
}

}}
