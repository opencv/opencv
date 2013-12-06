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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// In no event shall contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include "opencv2/core/opencl/runtime/opencl_core.hpp" // for OpenCL types: cl_mem
#include "opencv2/core/ocl.hpp"

TEST(TestAPI, openCLExecuteKernelInterop)
{
    cv::RNG rng;
    Size sz(10000, 1);
    cv::Mat cpuMat = cvtest::randomMat(rng, sz, CV_32FC4, -10, 10, false);

    cv::ocl::oclMat gpuMat(cpuMat);
    cv::ocl::oclMat gpuMatDst(sz, CV_32FC4);

    const char* kernelStr =
"__kernel void test_kernel(__global float4* src, __global float4* dst) {\n"
"    int x = get_global_id(0);\n"
"    dst[x] = src[x];\n"
"}\n";

    cv::ocl::ProgramSource program("test_interop", kernelStr);

    using namespace std;
    vector<pair<size_t , const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *) &gpuMat.data ));
    args.push_back( make_pair( sizeof(cl_mem), (void *) &gpuMatDst.data ));

    size_t globalThreads[3] = { sz.width, 1, 1 };
    cv::ocl::openCLExecuteKernelInterop(
        gpuMat.clCxt,
        program,
        "test_kernel",
        globalThreads, NULL, args,
        -1, -1,
        "");

    cv::Mat dst;
    gpuMatDst.download(dst);

    EXPECT_LE(checkNorm(cpuMat, dst), 1e-3);
}

TEST(OCL_TestTAPI, performance)
{
    cv::RNG rng;
    cv::Mat src(1280,768,CV_8UC4), dst;
    rng.fill(src, RNG::UNIFORM, 0, 255);

    cv::UMat usrc, udst;
    src.copyTo(usrc);

    cv::ocl::oclMat osrc(src);
    cv::ocl::oclMat odst;

    int cvtcode = cv::COLOR_BGR2GRAY;
    int i, niters = 10;
    double t;

    cv::ocl::cvtColor(osrc, odst, cvtcode);
    cv::ocl::finish();
    t = (double)cv::getTickCount();
    for(i = 0; i < niters; i++)
    {
        cv::ocl::cvtColor(osrc, odst, cvtcode);
    }
    cv::ocl::finish();
    t = (double)cv::getTickCount() - t;
    printf("ocl exec time = %gms per iter\n", t*1000./niters/cv::getTickFrequency());

    cv::cvtColor(usrc, udst, cvtcode);
    cv::ocl::finish2();
    t = (double)cv::getTickCount();
    for(i = 0; i < niters; i++)
    {
        cv::cvtColor(usrc, udst, cvtcode);
    }
    cv::ocl::finish2();
    t = (double)cv::getTickCount() - t;
    printf("t-api exec time = %gms per iter\n", t*1000./niters/cv::getTickFrequency());

    cv::cvtColor(src, dst, cvtcode);
    t = (double)cv::getTickCount();
    for(i = 0; i < niters; i++)
    {
        cv::cvtColor(src, dst, cvtcode);
    }
    t = (double)cv::getTickCount() - t;
    printf("cpu exec time = %gms per iter\n", t*1000./niters/cv::getTickFrequency());
}
