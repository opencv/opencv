/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

#if defined(HAVE_OPENCV_XFEATURES2D) && defined(OPENCV_ENABLE_NONFREE)

TEST(SurfFeaturesFinder, CanFindInROIs)
{
    Ptr<Feature2D> finder = xfeatures2d::SURF::create();
    Mat img  = imread(string(cvtest::TS::ptr()->get_data_path()) + "cv/shared/lena.png");

    vector<Rect> rois;
    rois.push_back(Rect(0, 0, img.cols / 2, img.rows / 2));
    rois.push_back(Rect(img.cols / 2, img.rows / 2, img.cols - img.cols / 2, img.rows - img.rows / 2));

    // construct mask
    Mat mask = Mat::zeros(img.size(), CV_8U);
    for (const Rect &roi : rois)
    {
        Mat(mask, roi) = 1;
    }

    detail::ImageFeatures roi_features;
    detail::computeImageFeatures(finder, img, roi_features, mask);

    int tl_rect_count = 0, br_rect_count = 0, bad_count = 0;
    for (const auto &keypoint : roi_features.keypoints)
    {
        // Workaround for https://github.com/opencv/opencv/issues/26016
        // To keep its behaviour, keypoint.pt casts to Point_<int>.
        if (rois[0].contains(Point_<int>(keypoint.pt)))
            tl_rect_count++;
        else if (rois[1].contains(Point_<int>(keypoint.pt)))
            br_rect_count++;
        else
            bad_count++;
    }

    EXPECT_GT(tl_rect_count, 0);
    EXPECT_GT(br_rect_count, 0);
    EXPECT_EQ(bad_count, 0);
}

#endif // HAVE_OPENCV_XFEATURES2D && OPENCV_ENABLE_NONFREE

TEST(ParallelFeaturesFinder, IsSameWithSerial)
{
    Ptr<Feature2D> para_finder = ORB::create();
    Ptr<Feature2D> serial_finder = ORB::create();
    Mat img  = imread(string(cvtest::TS::ptr()->get_data_path()) + "stitching/a3.png", IMREAD_GRAYSCALE);

    detail::ImageFeatures serial_features;
    detail::computeImageFeatures(serial_finder, img, serial_features);

    vector<Mat> imgs(50, img);
    vector<detail::ImageFeatures> para_features(imgs.size());
    detail::computeImageFeatures(para_finder, imgs, para_features);  // FIXIT This call doesn't use parallel_for_()

    // results must be the same
    Mat serial_descriptors;
    serial_features.descriptors.copyTo(serial_descriptors);
    for(size_t i = 0; i < para_features.size(); ++i)
    {
        SCOPED_TRACE(cv::format("i=%zu", i));
        EXPECT_EQ(serial_descriptors.size(), para_features[i].descriptors.size());
#if 0 // FIXIT ORB descriptors are not bit-exact (perhaps due internal parallel_for usage)
        ASSERT_EQ(0, cv::norm(u_serial_descriptors, para_features[i].descriptors, NORM_L1))
            << "serial_size=" << u_serial_descriptors.size()
            << " par_size=" << para_features[i].descriptors.size()
            << endl << u_serial_descriptors.getMat(ACCESS_READ)
            << endl << endl << para_features[i].descriptors.getMat(ACCESS_READ);
#endif
        EXPECT_EQ(serial_features.img_size, para_features[i].img_size);
        EXPECT_EQ(serial_features.keypoints.size(), para_features[i].keypoints.size());
    }
}

TEST(RangeMatcher, MatchesRangeOnly)
{
    Ptr<Feature2D> finder = ORB::create();

    Mat img0 = imread(string(cvtest::TS::ptr()->get_data_path()) + "stitching/a1.png", IMREAD_GRAYSCALE);
    Mat img1 = imread(string(cvtest::TS::ptr()->get_data_path()) + "stitching/a2.png", IMREAD_GRAYSCALE);
    Mat img2 = imread(string(cvtest::TS::ptr()->get_data_path()) + "stitching/a3.png", IMREAD_GRAYSCALE);

    vector<detail::ImageFeatures> features(3);

    computeImageFeatures(finder, img0, features[0]);
    computeImageFeatures(finder, img1, features[1]);
    computeImageFeatures(finder, img2, features[2]);

    vector<detail::MatchesInfo> pairwise_matches;
    Ptr<detail::FeaturesMatcher> matcher = makePtr<detail::BestOf2NearestRangeMatcher>(1);

    (*matcher)(features, pairwise_matches);

    // matches[1] will be image 0 and image 1, should have non-zero confidence
    EXPECT_NE(pairwise_matches[1].confidence, .0);

    // matches[2] will be image 0 and image 2, should have zero confidence due to range_width=1
    EXPECT_DOUBLE_EQ(pairwise_matches[2].confidence, .0);
}

}} // namespace
