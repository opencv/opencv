#include "test_precomp.hpp"
#include "../src/tokenizer/core_bpe.hpp"
#include "../src/tokenizer/encoding.hpp"
#include "../src/tokenizer/utils.hpp"
#include "../include/opencv2/dnn/tokenizer.hpp"
#include <fstream>
#include <sstream>

namespace opencv_test { namespace  {

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

TEST(EncodingBPE_Example, CountingGPT4) {
    // Example: Using the CL100K_BASE encoding, the sentence
    // "tiktoken is great!" would be broken into the individual
    // token pieces ["t", "ik", "token", " is", " great", "!"].
    // In other words, the tokenizer converts raw text into a
    // sequence of sub‑strings (tokens) that the model can process.

    // An encoding defines the rules that map raw text into its corresponding
    // sequence of tokens.

    // Load an encoding 
    std::string cl100k_base = _tf_gpt4("");
    tokenizer::Tokenizer tok = tokenizer::Tokenizer::load(cl100k_base);

    // Turn text into tokens with encoding.encode()
    std::vector<int> tokens = tok.encode("tiktoken is great!");

    for (auto token : tokens) {
        std::cout << token << " "; 
    }
    // output: 83 1609 5963 374 2294 0
    std::cout << std::endl;
    
    // we can count tokens by counting the length if the vector returned by .encode()
    auto numOfTokensFromString = [](const std::string s, const std::string encodingName) -> int {
        if (encodingName == "cl100k_base")  {
            std::string cl100k_base = _tf_gpt4("");
            tokenizer::Tokenizer _tok = tokenizer::Tokenizer::load(cl100k_base);
            std::vector<int> tokens = _tok.encode(s);
            return tokens.size();
        } else if (encodingName == "gpt2") {
            auto tokenizer = tokenizer::Tokenizer::load(_tf_gpt2(""));
            tokenizer::Encoding _encoding = tokenizer.encoding();
            std::vector<uint32_t> tokens = _encoding.encode(s);
            return tokens.size();
        }
        return -1;
    };

    int num_of_tokens = numOfTokensFromString("tiktoken is great!", "cl100k_base");
    if (num_of_tokens != -1) std::cout << num_of_tokens << std::endl;

    // Turn tokens into text with encoding.decode()
    std::string sent = tok.decode({83, 1609, 5963, 374, 2294, 0});
    std::cout << sent << std::endl;



    using Message = std::map<std::string, std::string>;
    auto num_of_tokens_from_prompts = [&](const std::vector<Message>& messages,
                                         const std::string& model) -> int {
        std::string voacb_bpe = _tf_gpt2("vocab.bpe");
        std::string cl100k_base = _tf_gpt4("");
        tokenizer::Encoding encoding = (model == "gpt2")
                            ? tokenizer::Tokenizer::load(_tf_gpt2("")).encoding()
                            : tokenizer::Tokenizer::load(cl100k_base).encoding();
        int tokens_per_message = 3;
        int tokens_per_name = 1;
        if (model == "gpt2") {
            tokens_per_message = 2;
            tokens_per_name = 1;
        }
        int num_tokens = 0;
        for (const auto& message : messages) {
            num_tokens += tokens_per_message;
            for (const auto& kv : message) {
                num_tokens += encoding.encode(kv.second).size();
                if (kv.first == "name") {
                    num_tokens += tokens_per_name;
                }
            }
        }
        num_tokens += 3; // every reply is primed
        return num_tokens;
    };

    std::vector<Message> example_messages = {
        {{"role", "system"}, {"content", "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."}},
        {{"role", "system"}, {"name", "example_user"}, {"content", "New synergies will help drive top-line growth."}},
        {{"role", "system"}, {"name", "example_assistant"}, {"content", "Things working well together will increase revenue."}},
        {{"role", "system"}, {"name", "example_user"}, {"content", "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."}},
        {{"role", "system"}, {"name", "example_assistant"}, {"content", "Let's talk later when we're less busy about how to do better."}},
        {{"role", "user"}, {"content", "This late pivot means we don't have time to boil the ocean for the client deliverable."}}
    };

    std::vector<std::string> models = {
        "gtp2",
        "cl100_base", 
        "train_taylor"
    };
    std::cout << std::endl;
    for (const auto& model : models) {
        int cpp_token_count = num_of_tokens_from_prompts(example_messages, model);
        std::cout << model << std::endl;
        std::cout << cpp_token_count << " prompt tokens counted by num_of_tokens_from_prompts()." << std::endl;
    }

    /*
        gtp2
        129 prompt tokens counted by num_of_tokens_from_prompts().
        cl100_base
        129 prompt tokens counted by num_of_tokens_from_prompts().
        train_taylor
        348 prompt tokens counted by num_of_tokens_from_prompts().
    */

}

}}