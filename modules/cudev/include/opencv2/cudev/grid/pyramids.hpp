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

#ifndef __OPENCV_CUDEV_GRID_PYRAMIDS_HPP__
#define __OPENCV_CUDEV_GRID_PYRAMIDS_HPP__

#include "../common.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/gpumat.hpp"
#include "../ptr2d/extrapolation.hpp"
#include "detail/pyr_down.hpp"
#include "detail/pyr_up.hpp"

namespace cv { namespace cudev {

template <class Brd, class SrcPtr, typename DstType>
__host__ void gridPyrDown_(const SrcPtr& src, GpuMat_<DstType>& dst, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    dst.create(divUp(rows, 2), divUp(cols, 2));

    pyramids_detail::pyrDown<Brd>(shrinkPtr(src), shrinkPtr(dst), rows, cols, dst.rows, dst.cols, StreamAccessor::getStream(stream));
}

template <class SrcPtr, typename DstType>
__host__ void gridPyrDown(const SrcPtr& src, GpuMat_<DstType>& dst, Stream& stream = Stream::Null())
{
    gridPyrDown_<BrdReflect101>(src, dst, stream);
}

template <class SrcPtr, typename DstType>
__host__ void gridPyrUp(const SrcPtr& src, GpuMat_<DstType>& dst, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    dst.create(rows * 2, cols * 2);

    pyramids_detail::pyrUp(shrinkPtr(src), shrinkPtr(dst), rows, cols, dst.rows, dst.cols, StreamAccessor::getStream(stream));
}

}}

#endif
