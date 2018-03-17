/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once

#ifndef OPENCV_CUDEV_GRID_HISTOGRAM_HPP
#define OPENCV_CUDEV_GRID_HISTOGRAM_HPP

#include "../common.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/gpumat.hpp"
#include "../ptr2d/mask.hpp"
#include "detail/histogram.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

template <int BIN_COUNT, class Policy, class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridHistogram_(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_Assert( deviceSupports(SHARED_ATOMICS) );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(1, BIN_COUNT);
    dst.setTo(0, stream);

    grid_histogram_detail::histogram<BIN_COUNT, Policy>(shrinkPtr(src),
                                                        dst[0],
                                                        shrinkPtr(mask),
                                                        rows, cols,
                                                        StreamAccessor::getStream(stream));
}

template <int BIN_COUNT, class Policy, class SrcPtr, typename ResType>
__host__ void gridHistogram_(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    CV_Assert( deviceSupports(SHARED_ATOMICS) );

    const int rows = getRows(src);
    const int cols = getCols(src);

    dst.create(1, BIN_COUNT);
    dst.setTo(0, stream);

    grid_histogram_detail::histogram<BIN_COUNT, Policy>(shrinkPtr(src),
                                                        dst[0],
                                                        WithOutMask(),
                                                        rows, cols,
                                                        StreamAccessor::getStream(stream));
}

// default policy

struct DefaultHistogramPolicy
{
    enum {
        block_size_x = 32,
        block_size_y = 8
    };
};

template <int BIN_COUNT, class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridHistogram(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridHistogram_<BIN_COUNT, DefaultHistogramPolicy>(src, dst, mask, stream);
}

template <int BIN_COUNT, class SrcPtr, typename ResType>
__host__ void gridHistogram(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    gridHistogram_<BIN_COUNT, DefaultHistogramPolicy>(src, dst, stream);
}

//! @}

}}

#endif
