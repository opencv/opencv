// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../test_precomp.hpp"
#include "../src/api/gproto_priv.hpp"

namespace opencv_test {

template<typename T>
struct ProtoPtrTest : public ::testing::Test { using Type = T; };

using ProtoPtrTestTypes = ::testing::Types< cv::Mat
                                          , cv::UMat
                                          , cv::gapi::own::Mat
                                          , cv::Scalar
                                          , std::vector<int>
                                          , int
                                          >;

TYPED_TEST_CASE(ProtoPtrTest, ProtoPtrTestTypes);

TYPED_TEST(ProtoPtrTest, NonZero)
{
    typename TestFixture::Type value;
    const auto arg = cv::gout(value).front();
    const auto ptr = cv::gimpl::proto::ptr(arg);
    EXPECT_EQ(ptr, &value);
}

} // namespace opencv_test
