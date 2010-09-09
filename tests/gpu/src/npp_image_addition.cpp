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
#include "highgui.h"

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuNppImageAdditionTest : public CvTest
{
public:
    CV_GpuNppImageAdditionTest();
    ~CV_GpuNppImageAdditionTest();

protected:
    void run(int);
};

CV_GpuNppImageAdditionTest::CV_GpuNppImageAdditionTest(): CvTest( "GPU-NppImageAddition", "add" )
{
}

CV_GpuNppImageAdditionTest::~CV_GpuNppImageAdditionTest() {}

void CV_GpuNppImageAdditionTest::run( int )
{
    cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-L.png", 0);
    cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-R.png", 0);

    if (img_l.empty() || img_r.empty())
    {
        ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
        return;
    }

    cv::Mat cpuAdd;
    cv::add(img_l, img_r, cpuAdd);

    GpuMat gpuL(img_l);
    GpuMat gpuR(img_r);
    GpuMat gpuAdd;
    cv::gpu::add(gpuL, gpuR, gpuAdd);

    //namedWindow("gpu");
    //imshow("gpu", gpuAdd);
    //namedWindow("cpu");
    //imshow("cpu", cpuAdd);
    //waitKey(1000);

    double ret = norm(cpuAdd, gpuAdd);

    if (ret < 1.0)
        ts->set_failed_test_info(CvTS::OK);
    else
    {
        ts->printf(CvTS::CONSOLE, "\nNorm: %f\n", ret);
        ts->set_failed_test_info(CvTS::FAIL_GENERIC);
    }
}

CV_GpuNppImageAdditionTest CV_GpuNppImageAddition_test;