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

#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

#include "opencv2/cudaarithm.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace testing;
using namespace perf;
using namespace cv;
using namespace cv::cuda;

namespace
{
    void func1(const GpuMat& src, GpuMat& dst, Stream& stream)
    {
        BufferPool pool(stream);

        GpuMat buf = pool.getBuffer(src.size(), CV_32FC(src.channels()));

        src.convertTo(buf, CV_32F, 1.0 / 255.0, stream);

        cuda::exp(buf, dst, stream);
    }

    void func2(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        BufferPool pool(stream);

        GpuMat buf1 = pool.getBuffer(src1.size(), CV_32FC(src1.channels()));

        func1(src1, buf1, stream);

        GpuMat buf2 = pool.getBuffer(src2.size(), CV_32FC(src2.channels()));

        func1(src2, buf2, stream);

        cuda::add(buf1, buf2, dst, noArray(), -1, stream);
    }
}

PERF_TEST_P(Sz, BufferPool, CUDA_TYPICAL_MAT_SIZES)
{
    static bool first = true;

    const Size size = GetParam();

    const bool useBufferPool = PERF_RUN_CUDA();

    Mat host_src(size, CV_8UC1);
    declare.in(host_src, WARMUP_RNG);

    GpuMat src1(host_src), src2(host_src);
    GpuMat dst;

    setBufferPoolUsage(useBufferPool);
    if (useBufferPool && first)
    {
        setBufferPoolConfig(-1, 25 * 1024 * 1024, 2);
        first = false;
    }

    TEST_CYCLE()
    {
        func2(src1, src2, dst, Stream::Null());
    }

    Mat h_dst(dst);
    SANITY_CHECK(h_dst);
}

#endif
