/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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
#include "opencv2/core/utils/logger.hpp"

namespace opencv_test { namespace {

class CV_FastTest : public cvtest::BaseTest
{
public:
    CV_FastTest();
    ~CV_FastTest();
protected:
    void run(int);
};

CV_FastTest::CV_FastTest() {}
CV_FastTest::~CV_FastTest() {}

void CV_FastTest::run( int )
{
  for(int type=0; type <= 2; ++type) {
    Mat image1 = imread(string(ts->get_data_path()) + "inpaint/orig.png");
    Mat image2 = imread(string(ts->get_data_path()) + "cameracalibration/chess9.png");
    string xml = string(ts->get_data_path()) + format("fast/result%d.xml", type);

    if (image1.empty() || image2.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat gray1, gray2;
    cvtColor(image1, gray1, COLOR_BGR2GRAY);
    cvtColor(image2, gray2, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    FAST(gray1, keypoints1, 30, true, static_cast<FastFeatureDetector::DetectorType>(type));
    FAST(gray2, keypoints2, (type > 0 ? 30 : 20), true, static_cast<FastFeatureDetector::DetectorType>(type));

    for(size_t i = 0; i < keypoints1.size(); ++i)
    {
        const KeyPoint& kp = keypoints1[i];
        cv::circle(image1, kp.pt, cvRound(kp.size/2), Scalar(255, 0, 0));
    }

    for(size_t i = 0; i < keypoints2.size(); ++i)
    {
        const KeyPoint& kp = keypoints2[i];
        cv::circle(image2, kp.pt, cvRound(kp.size/2), Scalar(255, 0, 0));
    }

    Mat kps1(1, (int)(keypoints1.size() * sizeof(KeyPoint)), CV_8U, &keypoints1[0]);
    Mat kps2(1, (int)(keypoints2.size() * sizeof(KeyPoint)), CV_8U, &keypoints2[0]);

    FileStorage fs(xml, FileStorage::READ);
    if (!fs.isOpened())
    {
        fs.open(xml, FileStorage::WRITE);
        if (!fs.isOpened())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }
        fs << "exp_kps1" << kps1;
        fs << "exp_kps2" << kps2;
        fs.release();
        fs.open(xml, FileStorage::READ);
        if (!fs.isOpened())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }
    }

    Mat exp_kps1, exp_kps2;
    read( fs["exp_kps1"], exp_kps1, Mat() );
    read( fs["exp_kps2"], exp_kps2, Mat() );
    fs.release();

    if ( exp_kps1.size != kps1.size || 0 != cvtest::norm(exp_kps1, kps1, NORM_L2) ||
         exp_kps2.size != kps2.size || 0 != cvtest::norm(exp_kps2, kps2, NORM_L2))
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return;
    }

    /*cv::namedWindow("Img1"); cv::imshow("Img1", image1);
    cv::namedWindow("Img2"); cv::imshow("Img2", image2);
    cv::waitKey(0);*/
  }

  ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Features2d_FAST, regression) { CV_FastTest test; test.safe_run(); }

// #define DUMP_TEST_DATA

TEST(Features2d_FAST, noNMS)
{
    Mat img = imread(string(cvtest::TS::ptr()->get_data_path()) + "inpaint/orig.png", cv::IMREAD_GRAYSCALE);
    string xml = string(cvtest::TS::ptr()->get_data_path()) + "fast/result_no_nonmax.xml";

    vector<KeyPoint> keypoints;
    FAST(img, keypoints, 100, false, FastFeatureDetector::DetectorType::TYPE_9_16);
    Mat kps(1, (int)(keypoints.size() * sizeof(KeyPoint)), CV_8U, &keypoints[0]);

    Mat gt_kps;
    FileStorage fs(xml, FileStorage::READ);
#ifdef DUMP_TEST_DATA
    if (!fs.isOpened())
    {
        fs.open(xml, FileStorage::WRITE);
        fs << "exp_kps" << kps;
        fs.release();
        fs.open(xml, FileStorage::READ);
    }
#endif
    ASSERT_TRUE(fs.isOpened());
    fs["exp_kps"] >> gt_kps;
    fs.release();
    ASSERT_GT(gt_kps.total(), size_t(0));
    // std::cout << "Ground truth keypoints (gt_kps):\n" << cv::format(gt_kps, cv::Formatter::FMT_DEFAULT) << "\n"
    //           << "Computed keypoints (kps):\n" << cv::format(kps, cv::Formatter::FMT_DEFAULT) << std::endl;
    // std::cout << "Ground truth keypoints (gt_kps):\n" << gt_kps << "\n"
    //           << "Computed keypoints (kps):\n" << kps << std::endl;
    ASSERT_EQ( 0, cvtest::norm(gt_kps, kps, NORM_L2));
}

}} // namespace
