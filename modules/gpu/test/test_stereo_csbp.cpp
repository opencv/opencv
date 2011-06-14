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

struct CV_GpuStereoCSBPTest : public cvtest::BaseTest
{
    void run(int )
    {
        cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "csstereobp/aloe-L.png");
        cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "csstereobp/aloe-R.png");

        cv::Mat img_template;

        if (cv::gpu::TargetArchs::builtWith(cv::gpu::FEATURE_SET_COMPUTE_20) &&
            cv::gpu::DeviceInfo().supports(cv::gpu::FEATURE_SET_COMPUTE_20))
            img_template = cv::imread(std::string(ts->get_data_path()) + "csstereobp/aloe-disp.png", CV_LOAD_IMAGE_GRAYSCALE);
        else
            img_template = cv::imread(std::string(ts->get_data_path()) + "csstereobp/aloe-disp_CC1X.png", CV_LOAD_IMAGE_GRAYSCALE);

        if (img_l.empty() || img_r.empty() || img_template.empty())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
            return;
        }

        {cv::Mat temp; cv::cvtColor(img_l, temp, CV_BGR2BGRA); cv::swap(temp, img_l);}
        {cv::Mat temp; cv::cvtColor(img_r, temp, CV_BGR2BGRA); cv::swap(temp, img_r);}

        cv::gpu::GpuMat disp;
        cv::gpu::StereoConstantSpaceBP bpm(128, 16, 4, 4);

        bpm(cv::gpu::GpuMat(img_l), cv::gpu::GpuMat(img_r), disp);

        //cv::imwrite(std::string(ts->get_data_path()) + "csstereobp/aloe-disp_CC1X.png", cv::Mat(disp));

        disp.convertTo(disp, img_template.type());

        double norm = cv::norm((cv::Mat)disp, img_template, cv::NORM_INF);
        if (norm >= 1.5)
        {
            ts->printf(cvtest::TS::LOG, "\nConstantSpaceStereoBP norm = %f\n", norm);
            ts->set_failed_test_info(cvtest::TS::FAIL_GENERIC);
            return;
        }

        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(StereoConstantSpaceBP, regression) { CV_GpuStereoCSBPTest test; test.safe_run(); }
