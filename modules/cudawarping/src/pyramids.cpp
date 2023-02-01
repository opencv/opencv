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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::cuda::pyrDown(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::pyrUp(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

#else // HAVE_CUDA

//////////////////////////////////////////////////////////////////////////////
// pyrDown

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        template <typename T> void pyrDown_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    }
}}}

void cv::cuda::pyrDown(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace cv::cuda::device::imgproc;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[6][4] =
    {
        {pyrDown_gpu<uchar>      , 0 /*pyrDown_gpu<uchar2>*/ , pyrDown_gpu<uchar3>      , pyrDown_gpu<uchar4>      },
        {0 /*pyrDown_gpu<schar>*/, 0 /*pyrDown_gpu<schar2>*/ , 0 /*pyrDown_gpu<schar3>*/, 0 /*pyrDown_gpu<schar4>*/},
        {pyrDown_gpu<ushort>     , 0 /*pyrDown_gpu<ushort2>*/, pyrDown_gpu<ushort3>     , pyrDown_gpu<ushort4>     },
        {pyrDown_gpu<short>      , 0 /*pyrDown_gpu<short2>*/ , pyrDown_gpu<short3>      , pyrDown_gpu<short4>      },
        {pyrDown_gpu<int>        , 0 /*pyrDown_gpu<int2>*/   , pyrDown_gpu<int3>        , pyrDown_gpu<int4>        },
        {pyrDown_gpu<float>      , 0 /*pyrDown_gpu<float2>*/ , pyrDown_gpu<float3>      , pyrDown_gpu<float4>      }
    };

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.depth() <= CV_32F && src.channels() <= 4 );

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert( func != 0 );

    _dst.create((src.rows + 1) / 2, (src.cols + 1) / 2, src.type());
    GpuMat dst = _dst.getGpuMat();

    func(src, dst, StreamAccessor::getStream(stream));
}


//////////////////////////////////////////////////////////////////////////////
// pyrUp

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        template <typename T> void pyrUp_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    }
}}}

void cv::cuda::pyrUp(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace cv::cuda::device::imgproc;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[6][4] =
    {
        {pyrUp_gpu<uchar>      , 0 /*pyrUp_gpu<uchar2>*/ , pyrUp_gpu<uchar3>      , pyrUp_gpu<uchar4>      },
        {0 /*pyrUp_gpu<schar>*/, 0 /*pyrUp_gpu<schar2>*/ , 0 /*pyrUp_gpu<schar3>*/, 0 /*pyrUp_gpu<schar4>*/},
        {pyrUp_gpu<ushort>     , 0 /*pyrUp_gpu<ushort2>*/, pyrUp_gpu<ushort3>     , pyrUp_gpu<ushort4>     },
        {pyrUp_gpu<short>      , 0 /*pyrUp_gpu<short2>*/ , pyrUp_gpu<short3>      , pyrUp_gpu<short4>      },
        {0 /*pyrUp_gpu<int>*/  , 0 /*pyrUp_gpu<int2>*/   , 0 /*pyrUp_gpu<int3>*/  , 0 /*pyrUp_gpu<int4>*/  },
        {pyrUp_gpu<float>      , 0 /*pyrUp_gpu<float2>*/ , pyrUp_gpu<float3>      , pyrUp_gpu<float4>      }
    };

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.depth() <= CV_32F && src.channels() <= 4 );

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert( func != 0 );

    _dst.create(src.rows * 2, src.cols * 2, src.type());
    GpuMat dst = _dst.getGpuMat();

    func(src, dst, StreamAccessor::getStream(stream));
}

#endif