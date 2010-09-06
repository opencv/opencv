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

#include "gputest.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>
#include <string>


class CV_GpuMeanShiftTest : public CvTest
{
public:
    CV_GpuMeanShiftTest();

protected:
    void run(int);
};

CV_GpuMeanShiftTest::CV_GpuMeanShiftTest(): CvTest( "GPU-MeanShift", "MeanShift" ){}

void CV_GpuMeanShiftTest::run(int)
{
        int spatialRad = 30;
        int colorRad = 30;

        cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "meanshift/cones.png");
        cv::Mat img_template = cv::imread(std::string(ts->get_data_path()) + "meanshift/con_result.png");

        if (img.empty() || img_template.empty())
        {
            ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
            return;
        }

        cv::Mat rgba;
        cvtColor(img, rgba, CV_BGR2BGRA);

        cv::gpu::GpuMat res;
        cv::gpu::meanShiftFiltering( cv::gpu::GpuMat(rgba), res, spatialRad, colorRad );
        if (res.type() != CV_8UC4)
        {
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return;
        }

        cv::Mat result;
        res.download(result);

        uchar maxDiff = 0;
        for (int j = 0; j < result.rows; ++j)
        {
            const uchar* res_line = result.ptr<uchar>(j);
            const uchar* ref_line = img_template.ptr<uchar>(j);

            for (int i = 0; i < result.cols; ++i)
            {
                for (int k = 0; k < 3; ++k)
                {
                    const uchar& ch1 = res_line[result.channels()*i + k];
                    const uchar& ch2 = ref_line[img_template.channels()*i + k];
                    uchar diff = abs(ch1 - ch2);
                    if (maxDiff < diff)
                        maxDiff = diff;
                }
            }
        }
        if (maxDiff > 0) 
            ts->printf(CvTS::CONSOLE, "\nMeanShift maxDiff = %d\n", maxDiff);
        ts->set_failed_test_info((maxDiff == 0) ? CvTS::OK : CvTS::FAIL_GENERIC);
}

CV_GpuMeanShiftTest CV_GpuMeanShift_test;