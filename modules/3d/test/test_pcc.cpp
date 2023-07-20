// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <fstream>
// TODO: when pcc.h put into 3d.hpp, this need to change
#include "../src/pcc.h"
#include "../src/pcc.cpp"

namespace opencv_test { namespace {

using namespace cv;

class PccEntropyCodingTest: public testing::Test
{
protected:
    void SetUp() override
    {
        // set test char vector to encode and decode
        testCharVector = {'a', 'a', 'b', 'b', 'b', 'c', '1', '2', '3'};
    }

public:
    std::vector<unsigned char> testCharVector;
    std::stringstream binaryStream;
    std::vector<unsigned char> restoreCharVector;
    EntropyCoder testEntropyCoder;
};

TEST_F(PccEntropyCodingTest, EntropyEncodingTest){
    EXPECT_NO_THROW(testEntropyCoder.encodeCharVectorToStream(testCharVector, binaryStream));
    binaryStream.seekg(0, std::ios::beg);
    EXPECT_NO_THROW(testEntropyCoder.decodeStreamToCharVector(binaryStream, restoreCharVector));
    EXPECT_EQ(testCharVector, restoreCharVector);
}

} // namespace
} // opencv_test