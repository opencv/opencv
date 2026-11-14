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

TEST(Features2D_ORB, _1996)
{
    Ptr<FeatureDetector> fd = ORB::create(10000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    Ptr<DescriptorExtractor> de = fd;

    Mat image = imread(string(cvtest::TS::ptr()->get_data_path()) + "shared/lena.png");
    ASSERT_FALSE(image.empty());

    Mat roi(image.size(), CV_8UC1, Scalar(0));

    Point poly[] = {Point(100, 20), Point(300, 50), Point(400, 200), Point(10, 500)};
    fillConvexPoly(roi, poly, int(sizeof(poly) / sizeof(poly[0])), Scalar(255));

    std::vector<KeyPoint> keypoints;
    fd->detect(image, keypoints, roi);
    Mat descriptors;
    de->compute(image, keypoints, descriptors);

    //image.setTo(Scalar(255,255,255), roi);

    int roiViolations = 0;
    for(std::vector<KeyPoint>::const_iterator kp = keypoints.begin(); kp != keypoints.end(); ++kp)
    {
        int x = cvRound(kp->pt.x);
        int y = cvRound(kp->pt.y);

        ASSERT_LE(0, x);
        ASSERT_LE(0, y);
        ASSERT_GT(image.cols, x);
        ASSERT_GT(image.rows, y);

        // if (!roi.at<uchar>(y,x))
        // {
        //     roiViolations++;
        //     circle(image, kp->pt, 3, Scalar(0,0,255));
        // }
    }

    // if(roiViolations)
    // {
    //     imshow("img", image);
    //     waitKey();
    // }

    ASSERT_EQ(0, roiViolations);
}

TEST(Features2D_ORB, crash_5031)
{
    cv::Mat image = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC3);

    int nfeatures = 8000;
    float orbScaleFactor = 1.2f;
    int nlevels = 18;
    int edgeThreshold = 4;
    int firstLevel = 0;
    int WTA_K = 2;
    ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 47;
    int fastThreshold = 20;

    Ptr<ORB> orb = cv::ORB::create(nfeatures, orbScaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::KeyPoint kp;
    kp.pt.x = 443;
    kp.pt.y = 5;
    kp.size = 47;
    kp.angle = 53.4580612f;
    kp.response = 0.0000470733867f;
    kp.octave = 0;
    kp.class_id = -1;

    keypoints.push_back(kp);

    ASSERT_NO_THROW(orb->compute(image, keypoints, descriptors));
}


TEST(Features2D_ORB, regression_16197)
{
    Mat img(Size(72, 72), CV_8UC1, Scalar::all(0));
    Ptr<ORB> orbPtr = ORB::create();
    orbPtr->setNLevels(5);
    orbPtr->setFirstLevel(3);
    orbPtr->setScaleFactor(1.8);
    orbPtr->setPatchSize(8);
    orbPtr->setEdgeThreshold(8);

    std::vector<KeyPoint> kps;
    Mat fv;

    // exception in debug mode, crash in release
    ASSERT_NO_THROW(orbPtr->detectAndCompute(img, noArray(), kps, fv));
}

// https://github.com/opencv/opencv-python/issues/537
BIGDATA_TEST(Features2D_ORB, regression_opencv_python_537)  // memory usage: ~3 Gb
{
    applyTestTag(
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG,
        CV_TEST_TAG_MEMORY_6GB
    );

    const int width = 25000;
    const int height = 25000;
    Mat img(Size(width, height), CV_8UC1, Scalar::all(0));

    const int border = 23, num_lines = 23;
    for (int i = 0; i < num_lines; i++)
    {
        cv::Point2i point1(border + i * 100, border + i * 100);
        cv::Point2i point2(width - border - i * 100, height - border * i * 100);
        cv::line(img, point1, point2, 255, 1, LINE_AA);
    }

    Ptr<ORB> orbPtr = ORB::create(31);
    std::vector<KeyPoint> kps;
    Mat fv;
    ASSERT_NO_THROW(orbPtr->detectAndCompute(img, noArray(), kps, fv));
}

TEST(Features2D_ORB, MaskValue)
{
    Mat gray = imread(cvtest::findDataFile("features2d/tsukuba.png"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(gray.empty());

    cv::Rect roi(gray.cols/4, gray.rows/4, gray.cols/2, gray.rows/2);

    Mat mask255 = Mat::zeros(gray.size(), CV_8UC1);
    Mat mask1 = Mat::zeros(gray.size(), CV_8UC1);
    mask255(roi).setTo(255);
    mask1(roi).setTo(1);

    Ptr<ORB> orb = cv::ORB::create();

    vector<KeyPoint> keypoints_mask255, keypoints_mask1;
    Mat descriptors_mask255, descriptors_mask1;

    orb->detectAndCompute(gray, mask255, keypoints_mask255, descriptors_mask255, false);
    orb->detectAndCompute(gray, mask1, keypoints_mask1, descriptors_mask1, false);

    ASSERT_EQ(keypoints_mask255.size(), keypoints_mask1.size())
        << "Number of keypoints differs between mask values 255 and 1";

    Mat diff = descriptors_mask255 != descriptors_mask1;
    ASSERT_EQ(countNonZero(diff), 0);
}

TEST(Features2D_ORB, invalidCreateParameters)
{
    EXPECT_THROW(ORB::create(500, 1.0f), cv::Exception);
    EXPECT_THROW(ORB::create(500, 1.2f, 0), cv::Exception);
    EXPECT_THROW(ORB::create(500, 1.2f, 8, 31, 0, 5), cv::Exception);
    EXPECT_THROW(ORB::create(500, 1.2f, 8, 31, 0, 2,
                             static_cast<ORB::ScoreType>(-1)), cv::Exception);
    EXPECT_THROW(ORB::create(500, 1.2f, 8, 31, 0, 2,
                             ORB::HARRIS_SCORE, 1), cv::Exception);
}

TEST(Features2D_ORB, roi_submatrix_padding)
{
    // The keypoints and descriptors detected on a sub-matrix ROI must match
    // those detected on an isolated clone of the exact same region, rather
    // than reading pixels from the parent image outside the ROI boundaries.
    Mat parent(300, 300, CV_8UC1);
    RNG& rng = cvtest::TS::ptr()->get_rng();
    rng.fill(parent, RNG::UNIFORM, 0, 256);

    Rect roi(50, 50, 200, 200);
    Mat srcRoi = parent(roi);
    Mat srcClone = srcRoi.clone();

    Ptr<ORB> orb = cv::ORB::create();

    vector<KeyPoint> kpsRoi, kpsClone;
    Mat descRoi, descClone;

    orb->detectAndCompute(srcRoi, noArray(), kpsRoi, descRoi, false);
    orb->detectAndCompute(srcClone, noArray(), kpsClone, descClone, false);

    ASSERT_EQ(kpsRoi.size(), kpsClone.size())
        << "Keypoint count differs between submatrix ROI and isolated clone";
    ASSERT_GT(kpsRoi.size(), 0u)
        << "No keypoints were detected in test image";

    for (size_t i = 0; i < kpsRoi.size(); ++i)
    {
        EXPECT_EQ(kpsRoi[i].pt, kpsClone[i].pt) << "Keypoint position mismatch at index " << i;
        EXPECT_EQ(kpsRoi[i].response, kpsClone[i].response) << "Keypoint response mismatch at index " << i;
        EXPECT_EQ(kpsRoi[i].angle, kpsClone[i].angle) << "Keypoint angle mismatch at index " << i;
    }

    if (!descRoi.empty() && !descClone.empty())
    {
        EXPECT_EQ(0, cvtest::norm(descRoi, descClone, NORM_INF))
            << "Descriptors differ between submatrix ROI and isolated clone";
    }
}

}} // namespace
