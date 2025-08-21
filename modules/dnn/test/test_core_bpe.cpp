// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace  {

class Test_CoreBPE : public ::testing::Test {
public:
    static ByteVecRankMap makeRanks() {
        ByteVecRankMap ranks;
        ranks.emplace(std::vector<std::uint8_t>{'a', 'b'}, 0);
        ranks.emplace(std::vector<std::uint8_t>{'c', 'd'}, 1);
        return ranks;
    }
};

// Both following test cases bytePairSplit_Simple and BytePairSplit_Repeated are taken from the lib.rs file in tiktoken 
TEST_F(Test_CoreBPE, bytePairSplit_Simple) {
    auto ranks = makeRanks();
    std::vector<std::uint8_t> piece = {'a', 'b', 'c', 'd'};
    auto parts = bytePairSplit(piece, ranks);
    std::vector<std::vector<std::uint8_t>> expected = { 
        std::vector<std::uint8_t>{'a', 'b'},
        std::vector<std::uint8_t>{'c', 'd'}
    };
    EXPECT_EQ(parts, expected) << "bytePairSplit should split \"abcd\" into [\"ab\",\"cd\"]";
}   

TEST_F(Test_CoreBPE, BytePairSplit_Repeated) {
    auto ranks = makeRanks();
    std::vector<std::uint8_t> piece = {'a', 'b', 'a', 'b'};
    auto parts = bytePairSplit(piece, ranks);
    std::vector<std::vector<std::uint8_t>> expected = {
        std::vector<std::uint8_t>{'a', 'b'},
        std::vector<std::uint8_t>{'a', 'b'}
    };
    EXPECT_EQ(parts, expected) << "bytePairEncode(\"abcd\") should yield [0,1]";
}

TEST_F(Test_CoreBPE, EncodeOrdinary_Simple) {
    auto ranks = makeRanks();
    std::unordered_map<std::string, uint32_t> special;

    // We choose a tiny regex that first matches "ab" or "cd" as whole,
    // falling back to matching any single char (.)
    static const std::string PAT = R"((?:ab|cd)|.)";
    CoreBPE bpe = CoreBPE(ranks, special, PAT);
    std::vector<uint32_t> out = bpe.encodeOrdinary("abcd");
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 0u);  // "ab" to token 0
    EXPECT_EQ(out[1], 1u);  // "cd" to token 1
}

}}