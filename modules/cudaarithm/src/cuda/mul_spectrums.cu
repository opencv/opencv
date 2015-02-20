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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

//////////////////////////////////////////////////////////////////////////////
// mulSpectrums

namespace
{
    __device__ __forceinline__ float real(const float2& val)
    {
        return val.x;
    }

    __device__ __forceinline__ float imag(const float2& val)
    {
        return val.y;
    }

    __device__ __forceinline__ float2 cmul(const float2& a, const float2& b)
    {
        return make_float2((real(a) * real(b)) - (imag(a) * imag(b)),
                           (real(a) * imag(b)) + (imag(a) * real(b)));
    }

    __device__ __forceinline__ float2 conj(const float2& a)
    {
        return make_float2(real(a), -imag(a));
    }

    struct comlex_mul : binary_function<float2, float2, float2>
    {
        __device__ __forceinline__ float2 operator ()(const float2& a, const float2& b) const
        {
            return cmul(a, b);
        }
    };

    struct comlex_mul_conj : binary_function<float2, float2, float2>
    {
        __device__ __forceinline__ float2 operator ()(const float2& a, const float2& b) const
        {
            return cmul(a, conj(b));
        }
    };

    struct comlex_mul_scale : binary_function<float2, float2, float2>
    {
        float scale;

        __device__ __forceinline__ float2 operator ()(const float2& a, const float2& b) const
        {
            return scale * cmul(a, b);
        }
    };

    struct comlex_mul_conj_scale : binary_function<float2, float2, float2>
    {
        float scale;

        __device__ __forceinline__ float2 operator ()(const float2& a, const float2& b) const
        {
            return scale * cmul(a, conj(b));
        }
    };
}

void cv::cuda::mulSpectrums(InputArray _src1, InputArray _src2, OutputArray _dst, int flags, bool conjB, Stream& stream)
{
    (void) flags;

    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);

    CV_Assert( src1.type() == src2.type() && src1.type() == CV_32FC2 );
    CV_Assert( src1.size() == src2.size() );

    GpuMat dst = getOutputMat(_dst, src1.size(), CV_32FC2, stream);

    if (conjB)
        gridTransformBinary(globPtr<float2>(src1), globPtr<float2>(src2), globPtr<float2>(dst), comlex_mul_conj(), stream);
    else
        gridTransformBinary(globPtr<float2>(src1), globPtr<float2>(src2), globPtr<float2>(dst), comlex_mul(), stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::mulAndScaleSpectrums(InputArray _src1, InputArray _src2, OutputArray _dst, int flags, float scale, bool conjB, Stream& stream)
{
    (void) flags;

    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);

    CV_Assert( src1.type() == src2.type() && src1.type() == CV_32FC2);
    CV_Assert( src1.size() == src2.size() );

    GpuMat dst = getOutputMat(_dst, src1.size(), CV_32FC2, stream);

    if (conjB)
    {
        comlex_mul_conj_scale op;
        op.scale = scale;
        gridTransformBinary(globPtr<float2>(src1), globPtr<float2>(src2), globPtr<float2>(dst), op, stream);
    }
    else
    {
        comlex_mul_scale op;
        op.scale = scale;
        gridTransformBinary(globPtr<float2>(src1), globPtr<float2>(src2), globPtr<float2>(dst), op, stream);
    }

    syncOutput(dst, _dst, stream);
}

#endif
