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

using namespace cv::cudev;

namespace
{
    struct ShiftMap
    {
        typedef int2 value_type;
        typedef int index_type;

        int top;
        int left;

        __device__ __forceinline__ int2 operator ()(int y, int x) const
        {
            return make_int2(x - left, y - top);
        }
    };

    struct ShiftMapSz : ShiftMap
    {
        int rows, cols;
    };
}

namespace cv { namespace cudev {

template <> struct PtrTraits<ShiftMapSz> : PtrTraitsBase<ShiftMapSz, ShiftMap>
{
};

}}

namespace
{
    template <typename T, int cn>
    void copyMakeBorderImpl(const GpuMat& src, GpuMat& dst, int top, int left, int borderMode, cv::Scalar borderValue, Stream& stream)
    {
        typedef typename MakeVec<T, cn>::type src_type;

        cv::Scalar_<T> borderValue_ = borderValue;
        const src_type brdVal = VecTraits<src_type>::make(borderValue_.val);

        ShiftMapSz map;
        map.top = top;
        map.left = left;
        map.rows = dst.rows;
        map.cols = dst.cols;

        switch (borderMode)
        {
        case cv::BORDER_CONSTANT:
            gridCopy(remapPtr(brdConstant(globPtr<src_type>(src), brdVal), map), globPtr<src_type>(dst), stream);
            break;
        case cv::BORDER_REPLICATE:
            gridCopy(remapPtr(brdReplicate(globPtr<src_type>(src)), map), globPtr<src_type>(dst), stream);
            break;
        case cv::BORDER_REFLECT:
            gridCopy(remapPtr(brdReflect(globPtr<src_type>(src)), map), globPtr<src_type>(dst), stream);
            break;
        case cv::BORDER_WRAP:
            gridCopy(remapPtr(brdWrap(globPtr<src_type>(src)), map), globPtr<src_type>(dst), stream);
            break;
        case cv::BORDER_REFLECT_101:
            gridCopy(remapPtr(brdReflect101(globPtr<src_type>(src)), map), globPtr<src_type>(dst), stream);
            break;
        };
    }
}

void cv::cuda::copyMakeBorder(InputArray _src, OutputArray _dst, int top, int bottom, int left, int right, int borderType, Scalar value, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, GpuMat& dst, int top, int left, int borderMode, cv::Scalar borderValue, Stream& stream);
    static const func_t funcs[6][4] =
    {
        {    copyMakeBorderImpl<uchar , 1>  ,     copyMakeBorderImpl<uchar , 2>  ,     copyMakeBorderImpl<uchar , 3>  ,     copyMakeBorderImpl<uchar , 4>  },
        {0 /*copyMakeBorderImpl<schar , 1>*/, 0 /*copyMakeBorderImpl<schar , 2>*/, 0 /*copyMakeBorderImpl<schar , 3>*/, 0 /*copyMakeBorderImpl<schar , 4>*/},
        {    copyMakeBorderImpl<ushort, 1>  , 0 /*copyMakeBorderImpl<ushort, 2>*/,     copyMakeBorderImpl<ushort, 3>  ,     copyMakeBorderImpl<ushort, 4>  },
        {    copyMakeBorderImpl<short , 1>  , 0 /*copyMakeBorderImpl<short , 2>*/,     copyMakeBorderImpl<short , 3>  ,     copyMakeBorderImpl<short , 4>  },
        {0 /*copyMakeBorderImpl<int   , 1>*/, 0 /*copyMakeBorderImpl<int   , 2>*/, 0 /*copyMakeBorderImpl<int   , 3>*/, 0 /*copyMakeBorderImpl<int   , 4>*/},
        {    copyMakeBorderImpl<float , 1>  , 0 /*copyMakeBorderImpl<float , 2>*/,     copyMakeBorderImpl<float , 3>  ,     copyMakeBorderImpl<float  ,4>  }
    };

    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();
    const int cn = src.channels();

    CV_Assert( depth <= CV_32F && cn <= 4 );
    CV_Assert( borderType == BORDER_REFLECT_101 || borderType == BORDER_REPLICATE || borderType == BORDER_CONSTANT || borderType == BORDER_REFLECT || borderType == BORDER_WRAP );

    _dst.create(src.rows + top + bottom, src.cols + left + right, src.type());
    GpuMat dst = _dst.getGpuMat();

    const func_t func = funcs[depth][cn - 1];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, dst, top, left, borderType, value, stream);
}

#endif
