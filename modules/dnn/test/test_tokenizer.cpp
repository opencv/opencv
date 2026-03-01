#include "test_precomp.hpp"
#include <opencv2/dnn/tokenizer.hpp>
#include <fstream>
#include <cstdio> // For std::remove

namespace opencv_test { namespace {

TEST(DNN_Tokenizer, BPE_FunctionalTest) {
    // 1. Create a dummy JSON file on disk
    std::string temp_file = "temp_bpe_vocab.json";
    std::ofstream out(temp_file);
    out << "{ \"vocab\": { \"h\":0, \"e\":1, \"l\":2, \"o\":3, \"he\":4, \"hel\":5, \"hello\":6 },"
        << "  \"merges\": [ \"h e\", \"he l\", \"hel l\", \"hell o\" ] }";
    out.close();

    // 2. Initialize Tokenizer with the PATH to the temp file
    // This triggers the 'load()' function in your implementation
    Ptr<dnn::Tokenizer> tokenizer = dnn::Tokenizer::createBPE(temp_file);

    // 3. Test Encoding
    std::string input = "hello";
    std::vector<int> ids = tokenizer->encode(input);

    // 4. Assert results
    // We expect "hello" -> ID 6.
    // If this fails with size 5, it means merges didn't happen.
    ASSERT_EQ(ids.size(), 1);
    EXPECT_EQ(ids[0], 6);

    // 5. Test Decoding
    std::string decoded = tokenizer->decode(ids);
    EXPECT_EQ(decoded, "hello");

    // 6. Cleanup
    std::remove(temp_file.c_str());
}

}} // namespace