// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ImageStitcher, setTransform)
{
    vector<Mat> images;
    images.push_back(imread(string(cvtest::TS::ptr()->get_data_path()) + "stitching/s1.jpg"));
    images.push_back(imread(string(cvtest::TS::ptr()->get_data_path()) + "stitching/s2.jpg"));

    Mat expected;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    EXPECT_TRUE(Stitcher::OK == stitcher->estimateTransform(images));
    EXPECT_TRUE(Stitcher::OK == stitcher->composePanorama(expected));

    Mat result;
    Ptr<Stitcher> another_stitcher = Stitcher::create(Stitcher::PANORAMA);
    EXPECT_TRUE(Stitcher::OK == another_stitcher->setTransform(images, stitcher->cameras()));
    EXPECT_TRUE(Stitcher::OK == another_stitcher->composePanorama(result));

    EXPECT_DOUBLE_EQ(cvtest::norm(expected, result, NORM_INF), .0);
}

}} // namespace opencv_test
