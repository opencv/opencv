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

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct CV_AsyncGpuMatTest : public cvtest::BaseTest
{
    CV_AsyncGpuMatTest() {}

    void run(int)
    {
        CudaMem src(Mat::zeros(100, 100, CV_8UC1));

        GpuMat gpusrc;
        GpuMat gpudst0, gpudst1(100, 100, CV_8UC1);

        CudaMem cpudst0;
        CudaMem cpudst1;

        Stream stream0, stream1;

        stream0.enqueueUpload(src, gpusrc);
        bitwise_not(gpusrc, gpudst0, GpuMat(), stream0);
        stream0.enqueueDownload(gpudst0, cpudst0);

        stream1.enqueueMemSet(gpudst1, Scalar::all(128));
        stream1.enqueueDownload(gpudst1, cpudst1);

        stream0.waitForCompletion();
        stream1.waitForCompletion();

        Mat cpu_gold0(100, 100, CV_8UC1, Scalar::all(255));
        Mat cpu_gold1(100, 100, CV_8UC1, Scalar::all(128));

        if (norm(cpudst0, cpu_gold0, NORM_INF) > 0 || norm(cpudst1, cpu_gold1, NORM_INF) > 0)
            ts->set_failed_test_info(cvtest::TS::FAIL_GENERIC);
        else
            ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(GpuMat, async) { CV_AsyncGpuMatTest test; test.safe_run(); }
