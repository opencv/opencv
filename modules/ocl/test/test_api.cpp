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
#include "opencv2/ocl/cl_runtime/cl_runtime.hpp" // for OpenCL types & functions

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

// This test must be DISABLED by default!
// (We can't restore original context for other tests)
TEST(TestAPI, DISABLED_InitializationFromHandles)
{
#define MAX_PLATFORMS 16
    cl_platform_id platforms[MAX_PLATFORMS] = { NULL };
    cl_uint numPlatforms = 0;
    cl_int status = ::clGetPlatformIDs(MAX_PLATFORMS, &platforms[0], &numPlatforms);
    ASSERT_EQ(CL_SUCCESS, status) << "clGetPlatformIDs";
    ASSERT_NE(0, (int)numPlatforms);

    int selectedPlatform = 0;
    cl_platform_id platform = platforms[selectedPlatform];

    ASSERT_NE((void*)NULL, platform);

    cl_device_id device = NULL;
    status = ::clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    ASSERT_EQ(CL_SUCCESS, status) << "clGetDeviceIDs";
    ASSERT_NE((void*)NULL, device);

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0 };
    cl_context context = ::clCreateContext(cps, 1, &device, NULL, NULL, &status);
    ASSERT_EQ(CL_SUCCESS, status) << "clCreateContext";
    ASSERT_NE((void*)NULL, context);

    ASSERT_NO_THROW(cv::ocl::initializeContext(&platform, &context, &device));

    status = ::clReleaseContext(context);
    ASSERT_EQ(CL_SUCCESS, status) << "clReleaseContext";

#ifdef CL_VERSION_1_2
#if 1
    {
        cv::ocl::Context* ctx = cv::ocl::Context::getContext();
        ASSERT_NE((void*)NULL, ctx);
        if (ctx->supportsFeature(cv::ocl::FEATURE_CL_VER_1_2)) // device supports OpenCL 1.2+
        {
            status = ::clReleaseDevice(device);
            ASSERT_EQ(CL_SUCCESS, status) << "clReleaseDevice";
        }
    }
#else // code below doesn't work on Linux (SEGFAULTs on 1.1- devices are not handled via exceptions)
    try
    {
        status = ::clReleaseDevice(device); // NOTE This works only with !DEVICES! that supports OpenCL 1.2
        (void)status; // no check
    }
    catch (...)
    {
        // nothing, there is no problem
    }
#endif
#endif

    // print the name of current device
    cv::ocl::Context* ctx = cv::ocl::Context::getContext();
    ASSERT_NE((void*)NULL, ctx);
    const cv::ocl::DeviceInfo& deviceInfo = ctx->getDeviceInfo();
    std::cout << "Device name: " << deviceInfo.deviceName << std::endl;
    std::cout << "Platform name: " << deviceInfo.platform->platformName << std::endl;

    ASSERT_EQ(context, *(cl_context*)ctx->getOpenCLContextPtr());
    ASSERT_EQ(device, *(cl_device_id*)ctx->getOpenCLDeviceIDPtr());

    // do some calculations and check results
    cv::RNG rng;
    Size sz(100, 100);
    cv::Mat srcMat = cvtest::randomMat(rng, sz, CV_32FC4, -10, 10, false);
    cv::Mat dstMat;

    cv::ocl::oclMat srcGpuMat(srcMat);
    cv::ocl::oclMat dstGpuMat;

    cv::Scalar v = cv::Scalar::all(1);
    cv::add(srcMat, v, dstMat);
    cv::ocl::add(srcGpuMat, v, dstGpuMat);

    cv::Mat dstGpuMatMap;
    dstGpuMat.download(dstGpuMatMap);

    EXPECT_LE(checkNorm(dstMat, dstGpuMatMap), 1e-3);
}
