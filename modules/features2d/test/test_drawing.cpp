// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

static
Mat getReference_DrawKeypoint(int cn)
{
    static Mat ref = (Mat_<uint8_t>(11, 11) <<
        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
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
    Mat res;
    cvtColor(ref, res, (cn == 4) ? COLOR_GRAY2BGRA : COLOR_GRAY2BGR);
    return res;
}

typedef testing::TestWithParam<MatType> Features2D_drawKeypoints;
TEST_P(Features2D_drawKeypoints, Accuracy)
{
    const int cn = CV_MAT_CN(GetParam());
    Mat inpImg(11, 11, GetParam(), Scalar(1, 1, 1, 255)), outImg;

    std::vector<KeyPoint> keypoints(1, KeyPoint(5, 5, 1));
    drawKeypoints(inpImg, keypoints, outImg, Scalar::all(255));
    ASSERT_EQ(outImg.channels(), (cn == 4) ? 4 : 3);

    Mat ref_ = getReference_DrawKeypoint(cn);
    EXPECT_EQ(0, cv::norm(outImg, ref_, NORM_INF));
}
INSTANTIATE_TEST_CASE_P(/**/, Features2D_drawKeypoints, Values(CV_8UC1, CV_8UC3, CV_8UC4));

typedef testing::TestWithParam<tuple<MatType, MatType> > Features2D_drawMatches;
TEST_P(Features2D_drawMatches, Accuracy)
{
    Mat inpImg1(11, 11, get<0>(GetParam()), Scalar(1, 1, 1, 255));
    Mat inpImg2(11, 11, get<1>(GetParam()), Scalar(2, 2, 2, 255)), outImg2, outImg;

    std::vector<KeyPoint> keypoints(1, KeyPoint(5, 5, 1));

    // Get outImg2 using drawKeypoints assuming that it works correctly (see the test above).
    drawKeypoints(inpImg2, keypoints, outImg2, Scalar::all(255));
    ASSERT_EQ(outImg2.channels(), (inpImg2.channels() == 4) ? 4 : 3);

    // Merge both references.
    const int cn = max(3, max(inpImg1.channels(), inpImg2.channels()));
    if (cn == 4 && outImg2.channels() == 3)
        cvtColor(outImg2, outImg2, COLOR_BGR2BGRA);
    Mat ref_ = getReference_DrawKeypoint(cn);
    Mat concattedRef;
    hconcat(ref_, outImg2, concattedRef);

    std::vector<DMatch> matches;
    drawMatches(inpImg1, keypoints, inpImg2, keypoints, matches, outImg,
                Scalar::all(255), Scalar::all(255));
    ASSERT_EQ(outImg.channels(), cn);

    EXPECT_EQ(0, cv::norm(outImg, concattedRef, NORM_INF));
}
INSTANTIATE_TEST_CASE_P(/**/, Features2D_drawMatches, Combine(
    Values(CV_8UC1, CV_8UC3, CV_8UC4),
    Values(CV_8UC1, CV_8UC3, CV_8UC4)
));

}} // namespace
