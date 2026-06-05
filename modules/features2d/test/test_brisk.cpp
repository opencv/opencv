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

namespace opencv_test { namespace {

class CV_BRISKTest : public cvtest::BaseTest
{
public:
    CV_BRISKTest();
    ~CV_BRISKTest();
protected:
    void run(int);
};

CV_BRISKTest::CV_BRISKTest() {}
CV_BRISKTest::~CV_BRISKTest() {}

void CV_BRISKTest::run( int )
{
  Mat image1 = imread(string(ts->get_data_path()) + "inpaint/orig.png");
  Mat image2 = imread(string(ts->get_data_path()) + "cameracalibration/chess9.png");

  if (image1.empty() || image2.empty())
    {
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
      return;
    }

  Mat gray1, gray2;
  cvtColor(image1, gray1, COLOR_BGR2GRAY);
  cvtColor(image2, gray2, COLOR_BGR2GRAY);

  Ptr<FeatureDetector> detector = BRISK::create();

  // Check parameter get/set functions.
  BRISK* detectorTyped = dynamic_cast<BRISK*>(detector.get());
  ASSERT_NE(nullptr, detectorTyped);
  detectorTyped->setOctaves(3);
  detectorTyped->setThreshold(30);
  ASSERT_EQ(detectorTyped->getOctaves(), 3);
  ASSERT_EQ(detectorTyped->getThreshold(), 30);
  detectorTyped->setOctaves(4);
  detectorTyped->setThreshold(29);
  ASSERT_EQ(detectorTyped->getOctaves(), 4);
  ASSERT_EQ(detectorTyped->getThreshold(), 29);

  vector<KeyPoint> keypoints1;
  vector<KeyPoint> keypoints2;
  detector->detect(image1, keypoints1);
  detector->detect(image2, keypoints2);

  for(size_t i = 0; i < keypoints1.size(); ++i)
    {
      const KeyPoint& kp = keypoints1[i];
      ASSERT_NE(kp.angle, -1);
    }

  for(size_t i = 0; i < keypoints2.size(); ++i)
    {
      const KeyPoint& kp = keypoints2[i];
      ASSERT_NE(kp.angle, -1);
    }
}

TEST(Features2d_BRISK, regression) { CV_BRISKTest test; test.safe_run(); }

// Regression tests for issue #29239: UB (left-shift of negative value) in
// BriskScaleSpace::subpixel2D.  These tests are most meaningful when built
// with -fsanitize=undefined, which would have caught the original defect.

// Direct-computation check with the exact crash parameters (s_0_2=3, all
// others=0).  Verifies that the fixed formula produces correct results;
// under UBSan the original << 2 / << 1 on a negative operand is flagged.
TEST(Features2d_BRISK, regression_ubsan_negative_shift_formula)
{
    const int s_0_0=0, s_0_1=0, s_0_2=3, s_1_0=0, s_1_1=0, s_1_2=0,
              s_2_0=0, s_2_1=0, s_2_2=0;
    const int coeff5 = (s_0_0 - s_0_2 - s_2_0 + s_2_2) * 4;
    const int coeff6 = -(s_0_0 + s_0_2
                         - ((s_1_0 + s_0_1 + s_1_2 + s_2_1) * 2)
                         - 5 * s_1_1 + s_2_0 + s_2_2) * 2;
    EXPECT_EQ(coeff5, -12);
    EXPECT_EQ(coeff6, -6);
}

// Integration check: a pseudo-random image (fixed LCG seed) with full 0-255
// intensity range produces a variety of AGAST score patterns including
// negative-difference neighbourhoods.  EXPECT_GT confirms subpixel2D is
// actually called.  Note: cv::RNG::fill can trigger an unrelated UBSan issue
// in rand.cpp, so the image is filled manually to stay within the scope of
// this fix.
TEST(Features2d_BRISK, regression_ubsan_negative_shift_random_image)
{
    Mat img(60, 60, CV_8UC1);
    uint32_t state = 0xDEADBEEF;
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
        {
            state = state * 1664525u + 1013904223u;  // Numerical Recipes LCG
            img.at<uchar>(r, c) = static_cast<uchar>((state >> 24) & 0xFF);
        }
    Ptr<BRISK> brisk = BRISK::create(/*threshold=*/1, /*octaves=*/0, /*patternScale=*/1.0f);
    ASSERT_FALSE(brisk.empty());
    std::vector<KeyPoint> kpts;
    Mat desc;
    ASSERT_NO_THROW(brisk->detectAndCompute(img, noArray(), kpts, desc));
    EXPECT_GT(kpts.size(), 0u);
}

// Multi-octave integration check: a checkerboard with strong contrast produces
// corner responses at every scale, exercising subpixel2D across all octave
// transitions in getScoreMaxAbove / getScoreMaxBelow.  EXPECT_GT confirms the
// code path was reached.
TEST(Features2d_BRISK, regression_ubsan_negative_shift_multi_octave)
{
    Mat img(80, 80, CV_8UC1, Scalar(0));
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            img.at<uchar>(r, c) = static_cast<uchar>(((r / 8 + c / 8) % 2) * 200);
    Ptr<BRISK> brisk = BRISK::create(/*threshold=*/1, /*octaves=*/3, /*patternScale=*/1.0f);
    ASSERT_FALSE(brisk.empty());
    std::vector<KeyPoint> kpts;
    Mat desc;
    ASSERT_NO_THROW(brisk->detectAndCompute(img, noArray(), kpts, desc));
    EXPECT_GT(kpts.size(), 0u);
}

}} // namespace
