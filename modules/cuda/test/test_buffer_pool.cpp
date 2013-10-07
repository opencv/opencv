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

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace testing;
using namespace cv;
using namespace cv::cuda;

struct BufferPoolTest : TestWithParam<DeviceInfo>
{
};

namespace
{
    void func1(const GpuMat& src, GpuMat& dst, Stream& stream)
    {
        BufferPool pool(stream);

        GpuMat buf = pool.getBuffer(src.size(), CV_32FC(src.channels()));

        src.convertTo(buf, CV_32F, 1.0 / 255.0, stream);

        cuda::exp(buf, dst, stream);
    }

    void func2(const GpuMat& src, GpuMat& dst, Stream& stream)
    {
        BufferPool pool(stream);

        GpuMat buf1 = pool.getBuffer(saturate_cast<int>(src.rows * 0.5), saturate_cast<int>(src.cols * 0.5), src.type());

        cuda::resize(src, buf1, Size(), 0.5, 0.5, cv::INTER_NEAREST, stream);

        GpuMat buf2 = pool.getBuffer(buf1.size(), CV_32FC(buf1.channels()));

        func1(buf1, buf2, stream);

        GpuMat buf3 = pool.getBuffer(src.size(), buf2.type());

        cuda::resize(buf2, buf3, src.size(), 0, 0, cv::INTER_NEAREST, stream);

        buf3.convertTo(dst, CV_8U, stream);
    }
}

CUDA_TEST_P(BufferPoolTest, SimpleUsage)
{
    DeviceInfo devInfo = GetParam();
    setDevice(devInfo.deviceID());

    GpuMat src(200, 200, CV_8UC1);
    GpuMat dst;

    Stream stream;

    func2(src, dst, stream);

    stream.waitForCompletion();

    GpuMat buf, buf1, buf2, buf3;
    GpuMat dst_gold;

    cuda::resize(src, buf1, Size(), 0.5, 0.5, cv::INTER_NEAREST);
    buf1.convertTo(buf, CV_32F, 1.0 / 255.0);
    cuda::exp(buf, buf2);
    cuda::resize(buf2, buf3, src.size(), 0, 0, cv::INTER_NEAREST);
    buf3.convertTo(dst_gold, CV_8U);

    ASSERT_MAT_NEAR(dst_gold, dst, 0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Stream, BufferPoolTest, ALL_DEVICES);

#endif // HAVE_CUDA
