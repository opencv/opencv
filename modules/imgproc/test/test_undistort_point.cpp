/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
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
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test {
namespace {

class CV_UndistortPointTest : public cvtest::BaseTest {
 public:
  CV_UndistortPointTest();

 protected:
  void run(int);
};

CV_UndistortPointTest::CV_UndistortPointTest() {
}

void CV_UndistortPointTest::run(int) {
  int code = cvtest::TS::OK;
  const int col = 720;
  // const int row = 540;
  float camera_matrix_value[] = {437.8995, 0.0, 342.9241, 0.0, 438.8216,
                                 273.7163, 0.0, 0.0,      1.0};
  cv::Mat camera_interior(3, 3, CV_32F, camera_matrix_value);

  float camera_distort_value[] = {-0.34329, 0.11431, 0., 0., -0.017375};
  cv::Mat camera_distort(1, 5, CV_32F, camera_distort_value);

  float distort_points_value[] = {col, 0.};
  cv::Mat distort_pt(1, 2, CV_32F, distort_points_value);

  cv::Mat undistort_pt;
  distort_pt = distort_pt.reshape(2);
  cv::undistortPoints(distort_pt, undistort_pt, camera_interior,
                      camera_distort, cv::Mat(), camera_interior);
  distort_pt = distort_pt.reshape(1);
  ts->printf(cvtest::TS::LOG,
             "   distort point: [%.2f, %.2f]\n undistort point: [%.2f, %.2f]",
             distort_pt.at<float>(0), distort_pt.at<float>(1),
             undistort_pt.at<float>(0), undistort_pt.at<float>(1));
  if (fabs(distort_pt.at<float>(0) - undistort_pt.at<float>(0)) > col / 2) 
  {
    code = cvtest::TS::FAIL_INVALID_OUTPUT;
  }
  if (code < 0) 
  {
    ts->set_failed_test_info(code);
  }
}

TEST(Imgproc_undistortPoint, regression) {
  CV_UndistortPointTest test;
  test.safe_run();
}
}
}  // namespace
/* End of file. */
