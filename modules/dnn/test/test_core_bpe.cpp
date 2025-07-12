#include "test_precomp.hpp"
#include "../src/tokenizer/core_bpe.hpp"

namespace opencv_test { namespace  {

using namespace cv::dnn::tokenizer;

class Test_CoreBPE : public ::testing::Test {
public:
    static ByteVecRankMap makeRanks() {
        ByteVecRankMap ranks;
        ranks.emplace(ByteVec{'a', 'b'}, 0);
        ranks.emplace(ByteVec{'c', 'd'}, 1);
        return ranks;
    }
};

// Both following test cases bytePairSplit_Simple and BytePairSplit_Repeated are taken from the lib.rs file in tiktoken 
TEST_F(Test_CoreBPE, bytePairSplit_Simple) {
    auto ranks = makeRanks();
    ByteVec piece = {'a', 'b', 'c', 'd'};
    auto parts = bytePairSplit(piece, ranks);
    std::vector<ByteVec> expected = { 
        ByteVec{'a', 'b'},
        ByteVec{'c', 'd'}
    };
    EXPECT_EQ(parts, expected) << "bytePairSplit should split \"abcd\" into [\"ab\",\"cd\"]";
    // [PASSED]
}   



TEST_F(Test_CoreBPE, BytePairSplit_Repeated) {
    auto ranks = makeRanks();
    ByteVec piece = {'a', 'b', 'a', 'b'};
    auto parts = bytePairSplit(piece, ranks);
    std::vector<ByteVec> expected = {
        ByteVec{'a', 'b'},
        ByteVec{'a', 'b'}
    };
    EXPECT_EQ(parts, expected) << "bytePairEncode(\"abcd\") should yield [0,1]";
    // [PASSED]
}


TEST_F(Test_CoreBPE, EncodeOrdinary_Simple) {
    auto ranks = makeRanks();
    std::unordered_map<std::string, Rank> special;

    // We choose a tiny regex that first matches "ab" or "cd" as whole,
    // falling back to matching any single char (.)
    static const std::string PAT = R"((?:ab|cd)|.)";
    auto bpe = CoreBPE::create(
        ranks.begin(), ranks.end(),
        special.begin(), special.end(),
        PAT
    );
    std::vector<Rank> out = bpe.encodeOrdinary("abcd");
    // 3) Verify: we should get exactly two tokens [0,1]
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 0u);  // "ab" → token 0
    EXPECT_EQ(out[1], 1u);  // "cd" → token 1
}

}}