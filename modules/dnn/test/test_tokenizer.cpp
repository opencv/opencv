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

// GPT-4's split regex must come from the embedded pre_tokenizer.pattern.Regex
// in tokenizer.json (CL100K-style whitespace-run handling), not from a
// family-name fallback. Ground truth generated with:
//   from tokenizers import Tokenizer
//   tok = Tokenizer.from_file("gpt4/tokenizer.json")
//   tok.encode("a\n\nb").ids
TEST(Tokenizer_BPE, Tokenizer_GPT4_WhitespaceSplit) {
    std::string gpt4_model = _tf("gpt4/config.json");
    Tokenizer tok = Tokenizer::load(gpt4_model);

    EXPECT_EQ(tok.encode("a b"), std::vector<int>({64, 293}));
    EXPECT_EQ(tok.encode("a\n\nb"), std::vector<int>({64, 271, 65}));
    EXPECT_EQ(tok.encode("a \n\n b"), std::vector<int>({64, 4815, 293}));
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

// ---- T5 tests (Unigram tokenizer) ----
// Ground truth generated with:
//   from tokenizers import Tokenizer
//   tok = Tokenizer.from_file("tokenizer.json")  # onnx-models/sentence-t5-base-onnx
//   tok.encode(text).ids

TEST(Tokenizer_Unigram, Tokenizer_T5_BasicEncode) {
    std::string model = _tf("t5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("hello world"), (std::vector<int>{21820, 296, 1}));
    EXPECT_EQ(tok.encode("Don't stop! Really?? ...ok."),
              (std::vector<int>{1008, 31, 17, 1190, 55, 11291, 8546, 3, 233, 1825, 5, 1}));
}

TEST(Tokenizer_Unigram, Tokenizer_T5_Numbers) {
    std::string model = _tf("t5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("Invoice #12345, total: $1,234.56"),
              (std::vector<int>{86, 23235, 7172, 2773, 2128, 6, 792, 10, 1970, 6, 2773, 12451, 948, 1}));
}

TEST(Tokenizer_Unigram, Tokenizer_T5_Whitespace) {
    std::string model = _tf("t5/config.json");
    Tokenizer tok = Tokenizer::load(model);

    EXPECT_EQ(tok.encode("helloworld"), (std::vector<int>{21820, 7276, 1}));

    std::vector<int> trailing = tok.encode("hello ");
    EXPECT_EQ(trailing, (std::vector<int>{21820, 1}));
    EXPECT_EQ(tok.decode(trailing), "hello");

    std::vector<int> leading = tok.encode(" hello");
    EXPECT_EQ(leading, (std::vector<int>{21820, 1}));
    EXPECT_EQ(tok.decode(leading), "hello");
}

TEST(Tokenizer_Unigram, Tokenizer_T5_UnicodeNormalization) {
    std::string model = _tf("t5/config.json");
    Tokenizer tok = Tokenizer::load(model);

    // "ＡＢＣ１２３"
    std::vector<int> fullwidth = tok.encode("\xef\xbc\xa1\xef\xbc\xa2\xef\xbc\xa3\xef\xbc\x91\xef\xbc\x92\xef\xbc\x93");
    EXPECT_EQ(fullwidth, (std::vector<int>{14213, 14574, 1}));
    EXPECT_EQ(tok.decode(fullwidth), "ABC123");

    // "ﬁle ﬂow"
    std::vector<int> ligature = tok.encode("\xef\xac\x81\x6c\x65\x20\xef\xac\x82\x6f\x77");
    EXPECT_EQ(ligature, (std::vector<int>{1042, 2537, 1}));
    EXPECT_EQ(tok.decode(ligature), "file flow");

    // "café näive" (NFC and NFD input forms both map to the same ids)
    std::vector<int> accents = tok.encode("\x63\x61\x66\xc3\xa9\x20\x6e\xc3\xa4\x69\x76\x65");
    EXPECT_EQ(accents, (std::vector<int>{11949, 3, 29, 1864, 757, 1}));
    EXPECT_EQ(tok.decode(accents), "caf\xc3\xa9 n\xc3\xa4ive");
}

TEST(Tokenizer_Unigram, Tokenizer_T5_UnknownChars) {
    std::string model = _tf("t5/config.json");
    Tokenizer tok = Tokenizer::load(model);

    // "こんにちは世界"
    EXPECT_EQ(tok.encode("\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe4\xb8\x96\xe7\x95\x8c"),
              (std::vector<int>{3, 2, 1}));

    // "hello 👋 world 🌍"
    std::vector<int> emoji = tok.encode("\x68\x65\x6c\x6c\x6f\x20\xf0\x9f\x91\x8b\x20\x77\x6f\x72\x6c\x64\x20\xf0\x9f\x8c\x8d");
    EXPECT_EQ(emoji, (std::vector<int>{21820, 3, 2, 296, 3, 2, 1}));
    EXPECT_EQ(tok.decode(emoji), "hello  world ");
}

TEST(Tokenizer_Unigram, Tokenizer_T5_Roundtrip) {
    std::string model = _tf("t5/config.json");
    Tokenizer tok = Tokenizer::load(model);
    std::vector<std::string> cases = {
        "hello world",
        "Invoice #12345, total: $1,234.56",
        "Don't stop! Really?? ...ok.",
        "helloworld",
    };
    for (const auto& text : cases) {
        EXPECT_EQ(tok.decode(tok.encode(text)), text);
    }
}

// ---- BERT tests (WordPiece tokenizer) ----
// Ground truth generated with:
//   from tokenizers import Tokenizer
//   tok = Tokenizer.from_file("tokenizer.json")  # bert-base-uncased
//   tok.encode(text).ids ; tok.decode(ids, skip_special_tokens=True)

TEST(Tokenizer_WordPiece, Tokenizer_Bert_BasicEncode) {
    std::string model = _tf("bert/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("hello world"), (std::vector<int>{101, 7592, 2088, 102}));
    EXPECT_EQ(tok.encode("Don't stop! Really?? ...ok."),
              (std::vector<int>{101, 2123, 1005, 1056, 2644, 999, 2428, 1029, 1029, 1012, 1012, 1012,
                                 7929, 1012, 102}));
}

TEST(Tokenizer_WordPiece, Tokenizer_Bert_Numbers) {
    std::string model = _tf("bert/config.json");
    Tokenizer tok = Tokenizer::load(model);
    EXPECT_EQ(tok.encode("Invoice #12345, total: $1,234.56"),
              (std::vector<int>{101, 1999, 6767, 6610, 1001, 13138, 19961, 1010, 2561, 1024, 1002, 1015,
                                 1010, 22018, 1012, 5179, 102}));
}

TEST(Tokenizer_WordPiece, Tokenizer_Bert_Whitespace) {
    std::string model = _tf("bert/config.json");
    Tokenizer tok = Tokenizer::load(model);

    EXPECT_EQ(tok.encode("helloworld"), (std::vector<int>{101, 7592, 11108, 102}));

    std::vector<int> trailing = tok.encode("hello ");
    EXPECT_EQ(trailing, (std::vector<int>{101, 7592, 102}));
    EXPECT_EQ(tok.decode(trailing), "hello");

    std::vector<int> leading = tok.encode(" hello");
    EXPECT_EQ(leading, (std::vector<int>{101, 7592, 102}));
    EXPECT_EQ(tok.decode(leading), "hello");
}

TEST(Tokenizer_WordPiece, Tokenizer_Bert_CaseAndSubword) {
    std::string model = _tf("bert/config.json");
    Tokenizer tok = Tokenizer::load(model);

    // do_lower_case: true
    std::vector<int> upper = tok.encode("HELLO WORLD");
    EXPECT_EQ(upper, (std::vector<int>{101, 7592, 2088, 102}));
    EXPECT_EQ(tok.decode(upper), "hello world");

    std::vector<int> mixed = tok.encode("OpenCV is Great");
    EXPECT_EQ(mixed, (std::vector<int>{101, 2330, 2278, 2615, 2003, 2307, 102}));
    EXPECT_EQ(tok.decode(mixed), "opencv is great");

    // long OOV word split fully into WordPiece subword units, no [UNK] fallback
    std::vector<int> unk = tok.encode("supercalifragilisticexpialidocious");
    EXPECT_EQ(unk, (std::vector<int>{101, 3565, 9289, 10128, 29181, 24411, 4588, 10288, 19312, 21273,
                                      10085, 6313, 102}));
    EXPECT_EQ(tok.decode(unk), "supercalifragilisticexpialidocious");

    std::vector<int> hyphen = tok.encode("state-of-the-art");
    EXPECT_EQ(hyphen, (std::vector<int>{101, 2110, 1011, 1997, 1011, 1996, 1011, 2396, 102}));
    EXPECT_EQ(tok.decode(hyphen), "state - of - the - art");
}

TEST(Tokenizer_WordPiece, Tokenizer_Bert_NonAsciiStripping) {
    std::string model = _tf("bert/config.json");
    Tokenizer tok = Tokenizer::load(model);

    // "こんにちは世界" -- BertNormalizer's handle_chinese_chars pads CJK codepoints with
    // spaces before WordPiece splitting; the trailing character falls back to [UNK] (100).
    EXPECT_EQ(tok.encode("\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe4\xb8\x96\xe7\x95\x8c"),
              (std::vector<int>{101, 1655, 30217, 30194, 30188, 30198, 1745, 100, 102}));

    // "hello 👋 world 🌍" -- emoji are stripped by the normalizer's control-char handling
    // and map to [UNK] (100)
    std::vector<int> emoji = tok.encode("\x68\x65\x6c\x6c\x6f\x20\xf0\x9f\x91\x8b\x20\x77\x6f\x72\x6c\x64\x20\xf0\x9f\x8c\x8d");
    EXPECT_EQ(emoji, (std::vector<int>{101, 7592, 100, 2088, 100, 102}));
    EXPECT_EQ(tok.decode(emoji), "hello world");

    // "café näive" -- BertNormalizer strips accents by default
    std::vector<int> accents = tok.encode("\x63\x61\x66\xc3\xa9\x20\x6e\xc3\xa4\x69\x76\x65");
    EXPECT_EQ(accents, (std::vector<int>{101, 7668, 15743, 102}));
    EXPECT_EQ(tok.decode(accents), "cafe naive");
}

TEST(Tokenizer_WordPiece, Tokenizer_Bert_Roundtrip) {
    std::string model = _tf("bert/config.json");
    Tokenizer tok = Tokenizer::load(model);
    std::vector<std::pair<std::string, std::string>> cases = {
        {"hello world", "hello world"},
        {"helloworld", "helloworld"},
        {"2024", "2024"},
        {"supercalifragilisticexpialidocious", "supercalifragilisticexpialidocious"},
    };
    for (const auto& c : cases) {
        EXPECT_EQ(tok.decode(tok.encode(c.first)), c.second);
    }
}

TEST(Tokenizer_WordPiece, Tokenizer_Bert_EncodePair) {
    std::string model = _tf("bert/config.json");
    Tokenizer tok = Tokenizer::load(model);

    std::vector<int> a = tok.encode("hello world");
    std::vector<int> b = tok.encode("OpenCV is Great");
    std::vector<int> pair = tok.encodePair("hello world", "OpenCV is Great");

    EXPECT_EQ(pair, (std::vector<int>{101, 7592, 2088, 102, 2330, 2278, 2615, 2003, 2307, 102}));

    std::vector<int> expected(a.begin(), a.end());
    expected.insert(expected.end(), b.begin() + 1, b.end());
    EXPECT_EQ(pair, expected);
}

TEST(Tokenizer_BPE, Tokenizer_EncodePair_Unsupported) {
    Tokenizer tok = Tokenizer::load(_tf("gpt2/config.json"));
    EXPECT_THROW(tok.encodePair("hello", "world"), cv::Exception);
}

TEST(Tokenizer_SentencePiece, Tokenizer_EncodePair_Unsupported) {
    Tokenizer tok = Tokenizer::load(_tf("gemma2/config.json"));
    EXPECT_THROW(tok.encodePair("hello", "world"), cv::Exception);
}

TEST(Tokenizer_Unigram, Tokenizer_EncodePair_Unsupported) {
    Tokenizer tok = Tokenizer::load(_tf("t5/config.json"));
    EXPECT_THROW(tok.encodePair("hello", "world"), cv::Exception);
}

TEST(Tokenizer_Unigram, Tokenizer_MalformedUtf8) {
    Tokenizer tok = Tokenizer::load(_tf("t5/config.json"));
    EXPECT_THROW(tok.encode("\xff"), cv::Exception);          // invalid lead byte
    EXPECT_THROW(tok.encode("\xc3"), cv::Exception);           // truncated 2-byte sequence
}

}}
