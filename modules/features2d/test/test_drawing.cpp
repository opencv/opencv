// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

Mat ref = (Mat_<uint8_t>(11, 11) << 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
                                    1,   1,   1,   1,  15,  54,  15,   1,   1,   1,   1,
                                    1,   1,   1,  76, 217, 217, 221,  81,   1,   1,   1,
                                    1,   1, 100, 224, 111,  57, 115, 225, 101,   1,   1,
                                    1,  44, 215, 100,   1,   1,   1, 101, 214,  44,   1,
                                    1,  54, 212,  57,   1,   1,   1,  55, 212,  55,   1,
                                    1,  40, 215, 104,   1,   1,   1, 105, 215,  40,   1,
                                    1,   1, 102, 221, 111,  55, 115, 222, 103,   1,   1,
                                    1,   1,   1,  76, 218, 217, 220,  81,   1,   1,   1,
                                    1,   1,   1,   1,  15,  55,  15,   1,   1,   1,   1,
                                    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1);

typedef testing::TestWithParam<int> Features2D;
TEST_P(Features2D, drawKeypoints)
{
    Mat inpImg(11, 11, GetParam(), Scalar::all(1)), outImg;

    std::vector<KeyPoint> keypoints(1, KeyPoint(5, 5, 1));
    drawKeypoints(inpImg, keypoints, outImg, Scalar::all(255));
    ASSERT_EQ(outImg.channels(), 3);

    Mat ref3;
    cvtColor(ref, ref3, COLOR_GRAY2BGR);
    ASSERT_EQ(countNonZero(outImg != ref3), 0);
}

TEST_P(Features2D, drawMatches)
{
    Mat inpImg1(11, 11, GetParam(), Scalar::all(1));
    Mat inpImg2(11, 11, GetParam(), Scalar::all(2)), outImg2, outImg;

    std::vector<KeyPoint> keypoints(1, KeyPoint(5, 5, 1));

    // Get outImg2 using drawKeypoints assuming that it works correctly (see the test above).
    drawKeypoints(inpImg2, keypoints, outImg2, Scalar::all(255));
    ASSERT_EQ(outImg2.channels(), 3);

    // Merge both references.
    Mat ref3, concattedRef;
    cvtColor(ref, ref3, COLOR_GRAY2BGR);
    hconcat(ref3, outImg2, concattedRef);

    std::vector<DMatch> matches;
    drawMatches(inpImg1, keypoints, inpImg2, keypoints, matches, outImg,
                Scalar::all(255), Scalar::all(255));
    ASSERT_EQ(outImg.channels(), 3);

    ASSERT_EQ(countNonZero(outImg != concattedRef), 0);
}
INSTANTIATE_TEST_CASE_P(/**/, Features2D, Values(CV_8UC1, CV_8UC3, CV_8UC4));

}} // namespace
