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

#include <string>

using namespace std;

/* ///////////////////// simpleflow_test ///////////////////////// */

class CV_SimpleFlowTest : public cvtest::BaseTest
{
public:
    CV_SimpleFlowTest();
protected:
    void run(int);
};


CV_SimpleFlowTest::CV_SimpleFlowTest() {}

static bool readOpticalFlowFromFile(FILE* file, cv::Mat& flow) {
  char header[5];
  if (fread(header, 1, 4, file) < 4 && (string)header != "PIEH") {
    return false;
  }

  int cols, rows;
  if (fread(&cols, sizeof(int), 1, file) != 1||
      fread(&rows, sizeof(int), 1, file) != 1) {
    return false;
  }

  flow = cv::Mat::zeros(rows, cols, CV_32FC2);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      cv::Vec2f flow_at_point;
      if (fread(&(flow_at_point[0]), sizeof(float), 1, file) != 1 ||
          fread(&(flow_at_point[1]), sizeof(float), 1, file) != 1) {
        return false;
      }
      flow.at<cv::Vec2f>(i, j) = flow_at_point;
    }
  }

  return true;
}

static bool isFlowCorrect(float u) {
  return !cvIsNaN(u) && (fabs(u) < 1e9);
}

static float calc_rmse(cv::Mat flow1, cv::Mat flow2) {
  float sum = 0;
  int counter = 0;
  const int rows = flow1.rows;
  const int cols = flow1.cols;

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      cv::Vec2f flow1_at_point = flow1.at<cv::Vec2f>(y, x);
      cv::Vec2f flow2_at_point = flow2.at<cv::Vec2f>(y, x);

      float u1 = flow1_at_point[0];
      float v1 = flow1_at_point[1];
      float u2 = flow2_at_point[0];
      float v2 = flow2_at_point[1];

      if (isFlowCorrect(u1) && isFlowCorrect(u2) && isFlowCorrect(v1) && isFlowCorrect(v2)) {
        sum += (u1-u2)*(u1-u2) + (v1-v2)*(v1-v2);
        counter++;
      }
    }
  }
  return (float)sqrt(sum / (1e-9 + counter));
}

void CV_SimpleFlowTest::run(int) {
    const float MAX_RMSE = 0.6f;
    const string frame1_path = ts->get_data_path() + "optflow/RubberWhale1.png";
    const string frame2_path = ts->get_data_path() + "optflow/RubberWhale2.png";
    const string gt_flow_path = ts->get_data_path() + "optflow/RubberWhale.flo";

    cv::Mat frame1 = cv::imread(frame1_path);
    cv::Mat frame2 = cv::imread(frame2_path);

    if (frame1.empty()) {
      ts->printf(cvtest::TS::LOG, "could not read image %s\n", frame2_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    if (frame2.empty()) {
      ts->printf(cvtest::TS::LOG, "could not read image %s\n", frame2_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    if (frame1.rows != frame2.rows && frame1.cols != frame2.cols) {
      ts->printf(cvtest::TS::LOG, "images should be of equal sizes (%s and %s)",
                 frame1_path.c_str(), frame2_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    if (frame1.type() != 16 || frame2.type() != 16) {
      ts->printf(cvtest::TS::LOG, "images should be of equal type CV_8UC3 (%s and %s)",
                 frame1_path.c_str(), frame2_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    cv::Mat flow_gt;

    FILE* gt_flow_file = fopen(gt_flow_path.c_str(), "rb");
    if (gt_flow_file == NULL) {
      ts->printf(cvtest::TS::LOG, "could not read ground-thuth flow from file %s",
                 gt_flow_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    if (!readOpticalFlowFromFile(gt_flow_file, flow_gt)) {
      ts->printf(cvtest::TS::LOG, "error while reading flow data from file %s",
                 gt_flow_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }
    fclose(gt_flow_file);

    cv::Mat flow;
    cv::calcOpticalFlowSF(frame1, frame2, flow, 3, 2, 4);

    float rmse = calc_rmse(flow_gt, flow);

    ts->printf(cvtest::TS::LOG, "Optical flow estimation RMSE for SimpleFlow algorithm : %lf\n",
               rmse);

    if (rmse > MAX_RMSE) {
      ts->printf( cvtest::TS::LOG,
                 "Too big rmse error : %lf ( >= %lf )\n", rmse, MAX_RMSE);
      ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
      return;
    }
}


TEST(Video_OpticalFlowSimpleFlow, accuracy) { CV_SimpleFlowTest test; test.safe_run(); }

/* End of file. */
