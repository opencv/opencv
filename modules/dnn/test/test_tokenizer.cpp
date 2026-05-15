// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace  {

template<typename TString>
static String _tf(TString filename) {
    String basetestdir = getOpenCVExtraDir();
    size_t len = basetestdir.size();
    if(len > 0 && basetestdir[len-1] != '/' && basetestdir[len-1] != '\\')
        return (basetestdir + "/dnn/llm") + filename;
    return (basetestdir + "dnn/llm/") + filename;
}

TEST(Tokenizer_BPE, Tokenizer_GPT2_Tokens) {
    std::string gpt2_model = _tf("gpt2/config.json");
    Tokenizer tok = Tokenizer::load(gpt2_model);
    std::vector<int> tokens = tok.encode("hello world");
    std::vector<int> expected = {31373, 995};
    EXPECT_EQ(tokens, expected);
}

TEST(Tokenizer_BPE, Tokenizer_GPT4) {
    std::string gpt4_model = _tf("gpt4/config.json");
    Tokenizer tok = Tokenizer::load(gpt4_model);

    std::vector<int> tokens = tok.encode("hello world");
    std::vector<int> expected = {15339, 1917};
    EXPECT_EQ(tokens, expected);

    std::string sent = tok.decode({15339, 1917});
    std::string expec_str = "hello world";
    EXPECT_EQ(sent, expec_str);

}

TEST(Tokenizer_BPE, Tokenizer_GPT2) {
    std::string gpt2_model = _tf("gpt2/config.json");
    Tokenizer tok = Tokenizer::load(gpt2_model);
    auto ids = tok.encode("hello world");
    for (auto id : ids) std::cout << id << " ";
    std::cout << std::endl;
    auto txt = tok.decode(ids);
    EXPECT_EQ(txt, "hello world");

    // "Long characters" in Chinese
    auto ids_j = tok.encode("\xe9\x95\xbf\xe5\xad\x97\xe7\xac\xa6");
    std::string word = tok.decode(ids_j);
    std::cout << word << std::endl;
}

TEST(Tokenizer_BPE, Tokenizer_GPT2_Model) {
    std::string gpt2_model = _tf("gpt2/config.json");
    Tokenizer tok = Tokenizer::load(gpt2_model);
    auto ids = tok.encode("hello world");
    auto text = tok.decode(ids);
    EXPECT_EQ(text, "hello world");
}

TEST(Tokenizer_BPE, SimpleRepeated_GPT2) {
    Tokenizer gpt2_tok = Tokenizer::load(_tf("gpt2/config.json"));
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

TEST(Tokenizer_BPE, CatastrophicallyRepetitive_GPT2) {
    Tokenizer gpt2_tok = Tokenizer::load(_tf("gpt2/config.json"));
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

// ---- Qwen2.5 tests ----
// Ground truth generated with:
//   from transformers import AutoTokenizer
//   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
//   tok.encode(text)

TEST(Tokenizer_BPE, Tokenizer_Qwen2_5_English) {
    std::string model = _tf("qwen2.5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("Hello world"), (std::vector<int>{9707, 1879}));
}

TEST(Tokenizer_BPE, Tokenizer_Qwen2_5_Chinese) {
    std::string model = _tf("qwen2.5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    // 你好世界
    EXPECT_EQ(tok.encode("\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c"),
              (std::vector<int>{108386, 99489}));
}

TEST(Tokenizer_BPE, Tokenizer_Qwen2_5_Code) {
    std::string model = _tf("qwen2.5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("def hello(): print('hello')"),
              (std::vector<int>{750, 23811, 4555, 1173, 492, 14990, 863}));
}

TEST(Tokenizer_BPE, Tokenizer_Qwen2_5_Numbers) {
    std::string model = _tf("qwen2.5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("2024"), (std::vector<int>{17, 15, 17, 19}));
}

TEST(Tokenizer_BPE, Tokenizer_Qwen2_5_SpecialTokens) {
    std::string model = _tf("qwen2.5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    // <|im_start|>user\nHello<|im_end|>
    EXPECT_EQ(tok.encode("<|im_start|>user\nHello<|im_end|>"),
              (std::vector<int>{151644, 872, 198, 9707, 151645}));
}

TEST(Tokenizer_BPE, Tokenizer_Qwen2_5_Roundtrip) {
    std::string model = _tf("qwen2.5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    std::vector<std::string> cases = {
        "Hello world",
        "def hello(): print('hello')",
        "2024",
    };
    for (const auto& text : cases) {
        EXPECT_EQ(tok.decode(tok.encode(text)), text);
    }
}


TEST(Tokenizer_Gemma, Tokenizer_Gemma3_English) {
    std::string model = _tf("gemma3/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("Hello world"), (std::vector<int>{9259, 1902}));
}

TEST(Tokenizer_Gemma, Tokenizer_Gemma3_Phrase) {
    std::string model = _tf("gemma3/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("the quick brown fox"),
              (std::vector<int>{1437, 3823, 8864, 37423}));
}

TEST(Tokenizer_Gemma, Tokenizer_Gemma3_Mixed) {
    std::string model = _tf("gemma3/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("OpenCV"), (std::vector<int>{7084, 20741}));
}

TEST(Tokenizer_Gemma, Tokenizer_Gemma3_Numbers) {
    std::string model = _tf("gemma3/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("2024"), (std::vector<int>{236778, 236771, 236778, 236812}));
}

TEST(Tokenizer_Gemma, Tokenizer_Gemma3_SpecialTokens) {
    std::string model = _tf("gemma3/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("<bos>Hello<eos>"), (std::vector<int>{2, 9259, 1}));
}

TEST(Tokenizer_Gemma, Tokenizer_Gemma3_Roundtrip) {
    std::string model = _tf("gemma3/config.json");
    Tokenizer tok = Tokenizer::load(model);
    std::vector<std::string> cases = {
        "Hello world",
        "the quick brown fox",
        "OpenCV",
        "hello world",
    };
    for (const auto& text : cases) {
        EXPECT_EQ(tok.decode(tok.encode(text)), text);
    }
}

// Gemma2 tests (SentencePiece tokenizer)
TEST(Tokenizer_SentencePiece, Tokenizer_Gemma2_English) {
    std::string model = _tf("gemma2/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("Hello world"), (std::vector<int>{2, 4521, 2134}));
}

TEST(Tokenizer_SentencePiece, Tokenizer_Gemma2_Phrase) {
    std::string model = _tf("gemma2/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("the quick brown fox"),
              (std::vector<int>{2, 1175, 4320, 8426, 25341}));
}

TEST(Tokenizer_SentencePiece, Tokenizer_Gemma2_Mixed) {
    std::string model = _tf("gemma2/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("OpenCV"), (std::vector<int>{2, 6047, 17813}));
}

TEST(Tokenizer_SentencePiece, Tokenizer_Gemma2_Numbers) {
    std::string model = _tf("gemma2/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("2024"), (std::vector<int>{2, 235284, 235276, 235284, 235310}));
}

TEST(Tokenizer_SentencePiece, Tokenizer_Gemma2_Roundtrip) {
    std::string model = _tf("gemma2/config.json");
    Tokenizer tok = Tokenizer::load(model);
    std::vector<std::string> cases = {
        "Hello world",
        "the quick brown fox",
        "OpenCV",
        "hello world",
    };
    for (const auto& text : cases) {
        EXPECT_EQ(tok.decode(tok.encode(text)), text);
    }
}

}}
