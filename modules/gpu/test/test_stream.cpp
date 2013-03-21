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

#ifdef HAVE_CUDA

using namespace cvtest;

#if CUDA_VERSION >= 5000

struct Async : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::CudaMem src;
    cv::gpu::GpuMat d_src;

    cv::gpu::CudaMem dst;
    cv::gpu::GpuMat d_dst;

    virtual void SetUp()
    {
        cv::gpu::DeviceInfo devInfo = GetParam();
        cv::gpu::setDevice(devInfo.deviceID());

        cv::Mat m = randomMat(cv::Size(128, 128), CV_8UC1);
        src.create(m.size(), m.type(), cv::gpu::CudaMem::ALLOC_PAGE_LOCKED);
        m.copyTo(src.createMatHeader());
    }
};

void checkMemSet(cv::gpu::Stream&, int status, void* userData)
{
    ASSERT_EQ(cudaSuccess, status);

    Async* test = reinterpret_cast<Async*>(userData);

    cv::Mat src = test->src;
    cv::Mat dst = test->dst;

    cv::Mat dst_gold = cv::Mat::zeros(src.size(), src.type());

    ASSERT_MAT_NEAR(dst_gold, dst, 0);
}

GPU_TEST_P(Async, MemSet)
{
    cv::gpu::Stream stream;

    d_dst.upload(src);

    stream.enqueueMemSet(d_dst, cv::Scalar::all(0));
    stream.enqueueDownload(d_dst, dst);

    Async* test = this;
    stream.enqueueHostCallback(checkMemSet, test);

    stream.waitForCompletion();
}

void checkConvert(cv::gpu::Stream&, int status, void* userData)
{
    ASSERT_EQ(cudaSuccess, status);

    Async* test = reinterpret_cast<Async*>(userData);

    cv::Mat src = test->src;
    cv::Mat dst = test->dst;

    cv::Mat dst_gold;
    src.convertTo(dst_gold, CV_32S);

    ASSERT_MAT_NEAR(dst_gold, dst, 0);
}

GPU_TEST_P(Async, Convert)
{
    cv::gpu::Stream stream;

    stream.enqueueUpload(src, d_src);
    stream.enqueueConvert(d_src, d_dst, CV_32S);
    stream.enqueueDownload(d_dst, dst);

    Async* test = this;
    stream.enqueueHostCallback(checkConvert, test);

    stream.waitForCompletion();
}

INSTANTIATE_TEST_CASE_P(GPU_Stream, Async, ALL_DEVICES);

#endif

#endif // HAVE_CUDA
