#include "test_precomp.hpp"
#include "../src/tokenizer/core_bpe.hpp"
#include "../src/tokenizer/encoding.hpp"
#include "../src/tokenizer/utils.hpp"
#include <fstream>
#include <sstream>

namespace opencv_test { namespace  {

TEST(EncodingBPE_Example, CountingGPT4) {
    // Example: Using the CL100K_BASE encoding, the sentence
    // "tiktoken is great!" would be broken into the individual
    // token pieces ["t", "ik", "token", " is", " great", "!"].
    // In other words, the tokenizer converts raw text into a
    // sequence of sub‑strings (tokens) that the model can process.

    // An encoding defines the rules that map raw text into its corresponding
    // sequence of tokens.

    // Load an encoding 
    tokenizer::Encoding encoding = tokenizer::getEncodingForCl100k_base("cl100k_base");

    // Turn text into tokens with encoding.encode()
    std::vector<uint32_t> tokens = encoding.encode("tiktoken is great!");

    for (auto token : tokens) {
        std::cout << token << " "; 
    }
    // output: 83 1609 5963 374 2294 0
    std::cout << std::endl;
    
    // we can count tokens by counting the length if the vector returned by .encode()
    auto numOfTokensFromString = [](const std::string s, const std::string encodingName) -> int {
        if (encodingName == "cl100k_base")  {
            tokenizer::Encoding _encoding = tokenizer::getEncodingForCl100k_base("cl100k_base");
            std::vector<uint32_t> tokens = _encoding.encode(s);
            return tokens.size();
        } else if (encodingName == "gpt2") {
            tokenizer::Encoding _encoding = tokenizer::getEncodingForGPT2("gpt2");
            std::vector<uint32_t> tokens = _encoding.encode(s);
            return tokens.size();
        }
        return -1;
    };

    int num_of_tokens = numOfTokensFromString("tiktoken is great!", "cl100k_base");
    if (num_of_tokens != -1) std::cout << num_of_tokens << std::endl;

    // Turn tokens into text with encoding.decode()
    std::string sent = encoding.decode({83, 1609, 5963, 374, 2294, 0});
    std::cout << sent << std::endl;

    // For single tokens we can use decodeSingleTokenBytes() converting to the bytes it resembles 
    vector<uint32_t> listOfTokens{83, 1609, 5963, 374, 2294, 0};
    for (auto token : listOfTokens) {
        auto tmp = encoding.decodeSingleTokenBytes(token);
        std::string s(tmp.begin(), tmp.end());
        cout << s << " ";
    } // output: t ik token  is  great !
    std::cout << std::endl;

    // Compare encodings 
    // Encodings differ in their token‑splitting logic: one may partition words
    // another way, fold runs of whitespace together, or treat accented / non‑ASCII
    // uniquely.  Running the helper utilities above on a few test strings
    // lets us see those differences side‑by‑side.
    auto compareEncodings = [](const std::string sample) {
        auto printEncodingInfo = [](const tokenizer::Encoding& enc, const std::string& sample) {
            std::cout << std::endl << sample << std::endl;
            std::vector<uint32_t> token_integers = enc.encode(sample);
            int num_tokens = token_integers.size();
            std::vector<std::string> res;
            for (auto token : token_integers) {
                std::vector<uint8_t> sent_bytes = enc.decodeSingleTokenBytes(token);
                std::string sect(sent_bytes.begin(), sent_bytes.end());
                res.push_back(sect);
            }
            std::cout << std::endl;
            std::cout << "Encoding: " << enc.getName() << ": " << num_tokens << " tokens\n";
            std::cout << "token integers: ";
            for (auto ti : token_integers) {
                std::cout << ti << " ";
            }
            std::cout << std::endl;
            std::cout << "token bytes: ";
            for (auto sr : res) {
                std::cout << sr << " ";
            }
            std::cout << std::endl;
        };
        // cl100k_base -> gpt4
        tokenizer::Encoding enc = tokenizer::getEncodingForCl100k_base("cl100k_base");
        printEncodingInfo(enc, sample);
        /*
            output:
                antidisestablishmentarianism

                cl100k_base: 6 tokens
                token integers: 519 85342 34500 479 8997 2191 
                token bytes: ant idis establish ment arian ism
        */

        // r50k_base -> gpt2
        tokenizer::Encoding enc_gpt2 = tokenizer::getEncodingForGPT2("gpt2");
        printEncodingInfo(enc_gpt2, sample);
        /*
            output;
                antidisestablishmentarianism

                Encoding: gpt2: 5 tokens
                token integers: 415 29207 44390 3699 1042 
                token bytes: ant idis establishment arian ism
        */

        // train tokenizer on taylor swift text
        std::ifstream f("../modules/dnn/src/tokenizer/taylorswift.txt");
        ASSERT_TRUE(f.is_open());
        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string data = buffer.str();

        int vocab_size = 512;
        tokenizer::Encoding enc_trained(data, vocab_size, tokenizer::R50K_UTF8);
        printEncodingInfo(enc_trained, sample);
        /*
            output;
                antidisestablishmentarianism

                Encoding: trained_encoding: 16 tokens
                token integers: 267 116 384 357 433 97 98 108 357 104 109 379 266 498 357 109 
                token bytes: an t id is est a b l is h m ent ar ian is 
        */
    };
    compareEncodings("antidisestablishmentarianism");
    compareEncodings("2 + 2 = 4");
    compareEncodings("お誕生日おめでとう");
}

}}