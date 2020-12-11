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


}} // namespace
