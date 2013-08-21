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

using namespace cv;

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
  cvtColor(image1, gray1, CV_BGR2GRAY);
  cvtColor(image2, gray2, CV_BGR2GRAY);

  Ptr<FeatureDetector> detector = Algorithm::create<FeatureDetector>("Feature2D.BRISK");

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
