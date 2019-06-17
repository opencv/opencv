// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include <opencv2/gapi/own/scalar.hpp>

namespace opencv_test
{

TEST(Scalar, CreateEmpty)
{
    cv::gapi::own::Scalar s;

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(s[i], 0.0);
    }
}

TEST(Scalar, CreateFromVal)
{
    cv::gapi::own::Scalar s(5.0);

    EXPECT_EQ(s[0], 5.0);
    EXPECT_EQ(s[1], 0.0);
    EXPECT_EQ(s[2], 0.0);
    EXPECT_EQ(s[3], 0.0);
}

TEST(Scalar, CreateFromVals)
{
    cv::gapi::own::Scalar s(5.3, 3.3, 4.1, -2.0);

    EXPECT_EQ(s[0], 5.3);
    EXPECT_EQ(s[1], 3.3);
    EXPECT_EQ(s[2], 4.1);
    EXPECT_EQ(s[3], -2.0);
}

} // namespace opencv_test
