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

////////////////////////////////////////////////////////////////////////
/// merge

namespace
{
    template <int cn, typename T> struct MergeFunc;

    template <typename T> struct MergeFunc<2, T>
    {
        static void call(const GpuMat* src, GpuMat& dst, Stream& stream)
        {
            gridMerge(zipPtr(globPtr<T>(src[0]), globPtr<T>(src[1])),
                    globPtr<typename MakeVec<T, 2>::type>(dst),
                    stream);
        }
    };

    template <typename T> struct MergeFunc<3, T>
    {
        static void call(const GpuMat* src, GpuMat& dst, Stream& stream)
        {
            gridMerge(zipPtr(globPtr<T>(src[0]), globPtr<T>(src[1]), globPtr<T>(src[2])),
                    globPtr<typename MakeVec<T, 3>::type>(dst),
                    stream);
        }
    };

    template <typename T> struct MergeFunc<4, T>
    {
        static void call(const GpuMat* src, GpuMat& dst, Stream& stream)
        {
            gridMerge(zipPtr(globPtr<T>(src[0]), globPtr<T>(src[1]), globPtr<T>(src[2]), globPtr<T>(src[3])),
                    globPtr<typename MakeVec<T, 4>::type>(dst),
                    stream);
        }
    };

    void mergeImpl(const GpuMat* src, size_t n, cv::OutputArray _dst, Stream& stream)
    {
        CV_Assert( src != 0 );
        CV_Assert( n > 0 && n <= 4 );

        const int depth = src[0].depth();
        const cv::Size size = src[0].size();

        for (size_t i = 0; i < n; ++i)
        {
            CV_Assert( src[i].size() == size );
            CV_Assert( src[i].depth() == depth );
            CV_Assert( src[i].channels() == 1 );
        }

        if (n == 1)
        {
            src[0].copyTo(_dst, stream);
        }
        else
        {
            typedef void (*func_t)(const GpuMat* src, GpuMat& dst, Stream& stream);
            static const func_t funcs[3][5] =
            {
                {MergeFunc<2, uchar>::call, MergeFunc<2, ushort>::call, MergeFunc<2, int>::call, 0, MergeFunc<2, double>::call},
                {MergeFunc<3, uchar>::call, MergeFunc<3, ushort>::call, MergeFunc<3, int>::call, 0, MergeFunc<3, double>::call},
                {MergeFunc<4, uchar>::call, MergeFunc<4, ushort>::call, MergeFunc<4, int>::call, 0, MergeFunc<4, double>::call}
            };

            const int channels = static_cast<int>(n);

            GpuMat dst = getOutputMat(_dst, size, CV_MAKE_TYPE(depth, channels), stream);

            const func_t func = funcs[channels - 2][CV_ELEM_SIZE(depth) / 2];

            if (func == 0)
                CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported channel count or data type");

            func(src, dst, stream);

            syncOutput(dst, _dst, stream);
        }
    }
}

void cv::cuda::merge(const GpuMat* src, size_t n, OutputArray dst, Stream& stream)
{
    mergeImpl(src, n, dst, stream);
}


void cv::cuda::merge(const std::vector<GpuMat>& src, OutputArray dst, Stream& stream)
{
    mergeImpl(&src[0], src.size(), dst, stream);
}

////////////////////////////////////////////////////////////////////////
/// split

namespace
{
    template <int cn, typename T> struct SplitFunc;

    template <typename T> struct SplitFunc<2, T>
    {
        static void call(const GpuMat& src, GpuMat* dst, Stream& stream)
        {
            GlobPtrSz<T> dstarr[2] =
            {
                globPtr<T>(dst[0]), globPtr<T>(dst[1])
            };

            gridSplit(globPtr<typename MakeVec<T, 2>::type>(src), dstarr, stream);
        }
    };

    template <typename T> struct SplitFunc<3, T>
    {
        static void call(const GpuMat& src, GpuMat* dst, Stream& stream)
        {
            GlobPtrSz<T> dstarr[3] =
            {
                globPtr<T>(dst[0]), globPtr<T>(dst[1]), globPtr<T>(dst[2])
            };

            gridSplit(globPtr<typename MakeVec<T, 3>::type>(src), dstarr, stream);
        }
    };

    template <typename T> struct SplitFunc<4, T>
    {
        static void call(const GpuMat& src, GpuMat* dst, Stream& stream)
        {
            GlobPtrSz<T> dstarr[4] =
            {
                globPtr<T>(dst[0]), globPtr<T>(dst[1]), globPtr<T>(dst[2]), globPtr<T>(dst[3])
            };

            gridSplit(globPtr<typename MakeVec<T, 4>::type>(src), dstarr, stream);
        }
    };

    void splitImpl(const GpuMat& src, GpuMat* dst, Stream& stream)
    {
        typedef void (*func_t)(const GpuMat& src, GpuMat* dst, Stream& stream);
        static const func_t funcs[3][5] =
        {
            {SplitFunc<2, uchar>::call, SplitFunc<2, ushort>::call, SplitFunc<2, int>::call, 0, SplitFunc<2, double>::call},
            {SplitFunc<3, uchar>::call, SplitFunc<3, ushort>::call, SplitFunc<3, int>::call, 0, SplitFunc<3, double>::call},
            {SplitFunc<4, uchar>::call, SplitFunc<4, ushort>::call, SplitFunc<4, int>::call, 0, SplitFunc<4, double>::call}
        };

        CV_Assert( dst != 0 );

        const int depth = src.depth();
        const int channels = src.channels();

        CV_Assert( channels <= 4 );

        if (channels == 0)
            return;

        if (channels == 1)
        {
            src.copyTo(dst[0], stream);
            return;
        }

        for (int i = 0; i < channels; ++i)
            dst[i].create(src.size(), depth);

        const func_t func = funcs[channels - 2][CV_ELEM_SIZE(depth) / 2];

        if (func == 0)
            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported channel count or data type");

        func(src, dst, stream);
    }
}

void cv::cuda::split(InputArray _src, GpuMat* dst, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);
    splitImpl(src, dst, stream);
}

void cv::cuda::split(InputArray _src, std::vector<GpuMat>& dst, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);
    dst.resize(src.channels());
    if (src.channels() > 0)
        splitImpl(src, &dst[0], stream);
}

#endif
