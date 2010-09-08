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

#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuMatOpConvertToTest : public CvTest
{
    public:
        CV_GpuMatOpConvertToTest();
        ~CV_GpuMatOpConvertToTest();

    protected:
        void run(int);
};

CV_GpuMatOpConvertToTest::CV_GpuMatOpConvertToTest(): CvTest( "GPU-MatOperatorConvertTo", "convertTo" ) {}
CV_GpuMatOpConvertToTest::~CV_GpuMatOpConvertToTest() {}

void CV_GpuMatOpConvertToTest::run(int /* start_from */)
{
    const Size img_size(67, 35);

    const int types[] = {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F};
    const int types_num = sizeof(types) / sizeof(int);

    const char* types_str[] = {"CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32S", "CV_32F", "CV_64F"};

    bool passed = true;
    try
    {
        for (int i = 0; i < types_num && passed; ++i)
        {
            for (int j = 0; j < types_num && passed; ++j)
            {
                for (int c = 1; c < 2 && passed; ++c)
                {
                    const int src_type = CV_MAKETYPE(types[i], c);
                    const int dst_type = types[j];
                    const double alpha = (double)rand() / RAND_MAX * 2.0;
                    const double beta = (double)rand() / RAND_MAX * 150.0 - 75;

                    cv::RNG rng(*ts->get_rng());

                    Mat cpumatsrc(img_size, src_type);

                    rng.fill(cpumatsrc, RNG::UNIFORM, Scalar::all(0), Scalar::all(300));

                    GpuMat gpumatsrc(cpumatsrc);
                    Mat cpumatdst;
                    GpuMat gpumatdst;

                    cpumatsrc.convertTo(cpumatdst, dst_type, alpha, beta);
                    gpumatsrc.convertTo(gpumatdst, dst_type, alpha, beta);

                    double r = norm(cpumatdst, gpumatdst, NORM_INF);
                    if (r > 1)
                    {
                        ts->printf(CvTS::CONSOLE, 
                                   "\nFAILED: SRC_TYPE=%sC%d DST_TYPE=%s NORM = %d\n",
                                   types_str[i], c, types_str[j], r);
                        passed = false;
                    }
                }
            }
        }
    }
    catch(cv::Exception& e)
    {
        ts->printf(CvTS::CONSOLE, "\nERROR: %s\n", e.what());
    }
    ts->set_failed_test_info(passed ? CvTS::OK : CvTS::FAIL_GENERIC);
}

CV_GpuMatOpConvertToTest CV_GpuMatOpConvertToTest_test;

