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

#ifndef __OPENCV_CUDEV_GRID_REDUCE_HPP__
#define __OPENCV_CUDEV_GRID_REDUCE_HPP__

#include <limits>
#include "../common.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/gpumat.hpp"
#include "../ptr2d/mask.hpp"
#include "../ptr2d/transform.hpp"
#include "detail/reduce.hpp"
#include "detail/minmaxloc.hpp"

namespace cv { namespace cudev {

template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridCalcSum_(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    typedef typename PtrTraits<SrcPtr>::value_type src_type;

    CV_StaticAssert( unsigned(VecTraits<src_type>::cn) == unsigned(VecTraits<ResType>::cn), "" );

    dst.create(1, 1);
    dst.setTo(0, stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_reduce_detail::sum<Policy>(shrinkPtr(src),
                                    dst[0],
                                    shrinkPtr(mask),
                                    rows, cols,
                                    StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType>
__host__ void gridCalcSum_(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    typedef typename PtrTraits<SrcPtr>::value_type src_type;

    CV_StaticAssert( unsigned(VecTraits<src_type>::cn) == unsigned(VecTraits<ResType>::cn), "" );

    dst.create(1, 1);
    dst.setTo(0, stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    grid_reduce_detail::sum<Policy>(shrinkPtr(src),
                                    dst[0],
                                    WithOutMask(),
                                    rows, cols,
                                    StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridFindMinVal_(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    dst.create(1, 1);
    dst.setTo(Scalar::all(std::numeric_limits<ResType>::max()), stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_reduce_detail::minVal<Policy>(shrinkPtr(src),
                                       dst[0],
                                       shrinkPtr(mask),
                                       rows, cols,
                                       StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType>
__host__ void gridFindMinVal_(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    dst.create(1, 1);
    dst.setTo(Scalar::all(std::numeric_limits<ResType>::max()), stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    grid_reduce_detail::minVal<Policy>(shrinkPtr(src),
                                       dst[0],
                                       WithOutMask(),
                                       rows, cols,
                                       StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridFindMaxVal_(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    dst.create(1, 1);
    dst.setTo(Scalar::all(-std::numeric_limits<ResType>::max()), stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_reduce_detail::maxVal<Policy>(shrinkPtr(src),
                                       dst[0],
                                       shrinkPtr(mask),
                                       rows, cols,
                                       StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType>
__host__ void gridFindMaxVal_(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    dst.create(1, 1);
    dst.setTo(Scalar::all(-std::numeric_limits<ResType>::max()), stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    grid_reduce_detail::maxVal<Policy>(shrinkPtr(src),
                                       dst[0],
                                       WithOutMask(),
                                       rows, cols,
                                       StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridFindMinMaxVal_(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    dst.create(1, 2);
    dst.col(0).setTo(Scalar::all(std::numeric_limits<ResType>::max()), stream);
    dst.col(1).setTo(Scalar::all(-std::numeric_limits<ResType>::max()), stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_reduce_detail::minMaxVal<Policy>(shrinkPtr(src),
                                          dst[0],
                                          shrinkPtr(mask),
                                          rows, cols,
                                          StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType>
__host__ void gridFindMinMaxVal_(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    dst.create(1, 2);
    dst.col(0).setTo(Scalar::all(std::numeric_limits<ResType>::max()), stream);
    dst.col(1).setTo(Scalar::all(-std::numeric_limits<ResType>::max()), stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    grid_reduce_detail::minMaxVal<Policy>(shrinkPtr(src),
                                          dst[0],
                                          WithOutMask(),
                                          rows, cols,
                                          StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridMinMaxLoc_(const SrcPtr& src, GpuMat_<ResType>& valBuf, GpuMat_<int>& locBuf, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dim3 grid, block;
    grid_minmaxloc_detail::getLaunchCfg<Policy>(rows, cols, block, grid);

    valBuf.create(2, grid.x * grid.y);
    locBuf.create(2, grid.x * grid.y);

    grid_minmaxloc_detail::minMaxLoc<Policy>(shrinkPtr(src),
                                             valBuf[0], valBuf[1], locBuf[0], locBuf[1],
                                             shrinkPtr(mask),
                                             rows, cols,
                                             StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType>
__host__ void gridMinMaxLoc_(const SrcPtr& src, GpuMat_<ResType>& valBuf, GpuMat_<int>& locBuf, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    dim3 grid, block;
    grid_minmaxloc_detail::getLaunchCfg<Policy>(rows, cols, block, grid);

    valBuf.create(2, grid.x * grid.y);
    locBuf.create(2, grid.x * grid.y);

    grid_minmaxloc_detail::minMaxLoc<Policy>(shrinkPtr(src),
                                             valBuf[0], valBuf[1], locBuf[0], locBuf[1],
                                             WithOutMask(),
                                             rows, cols,
                                             StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridCountNonZero_(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    dst.create(1, 1);
    dst.setTo(0, stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    typedef typename PtrTraits<SrcPtr>::value_type src_type;
    not_equal_to<src_type> ne_op;
    const src_type zero = VecTraits<src_type>::all(0);

    grid_reduce_detail::sum<Policy>(shrinkPtr(transformPtr(src, bind2nd(ne_op, zero))),
                                    dst[0],
                                    shrinkPtr(mask),
                                    rows, cols,
                                    StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename ResType>
__host__ void gridCountNonZero_(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    dst.create(1, 1);
    dst.setTo(0, stream);

    const int rows = getRows(src);
    const int cols = getCols(src);

    typedef typename PtrTraits<SrcPtr>::value_type src_type;
    not_equal_to<src_type> ne_op;
    const src_type zero = VecTraits<src_type>::all(0);

    grid_reduce_detail::sum<Policy>(shrinkPtr(transformPtr(src, bind2nd(ne_op, zero))),
                                    dst[0],
                                    WithOutMask(),
                                    rows, cols,
                                    StreamAccessor::getStream(stream));
}

// default policy

struct DefaultGlobReducePolicy
{
    enum {
        block_size_x = 32,
        block_size_y = 8,

        patch_size_x = 4,
        patch_size_y = 4
    };
};

template <class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridCalcSum(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCalcSum_<DefaultGlobReducePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename ResType>
__host__ void gridCalcSum(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    gridCalcSum_<DefaultGlobReducePolicy>(src, dst, stream);
}

template <class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridFindMinVal(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridFindMinVal_<DefaultGlobReducePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename ResType>
__host__ void gridFindMinVal(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    gridFindMinVal_<DefaultGlobReducePolicy>(src, dst, stream);
}

template <class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridFindMaxVal(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridFindMaxVal_<DefaultGlobReducePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename ResType>
__host__ void gridFindMaxVal(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    gridFindMaxVal_<DefaultGlobReducePolicy>(src, dst, stream);
}

template <class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridFindMinMaxVal(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridFindMinMaxVal_<DefaultGlobReducePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename ResType>
__host__ void gridFindMinMaxVal(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    gridFindMinMaxVal_<DefaultGlobReducePolicy>(src, dst, stream);
}

template <class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridMinMaxLoc(const SrcPtr& src, GpuMat_<ResType>& valBuf, GpuMat_<int>& locBuf, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridMinMaxLoc_<DefaultGlobReducePolicy>(src, valBuf, locBuf, mask, stream);
}

template <class SrcPtr, typename ResType>
__host__ void gridMinMaxLoc(const SrcPtr& src, GpuMat_<ResType>& valBuf, GpuMat_<int>& locBuf, Stream& stream = Stream::Null())
{
    gridMinMaxLoc_<DefaultGlobReducePolicy>(src, valBuf, locBuf, stream);
}

template <class SrcPtr, typename ResType, class MaskPtr>
__host__ void gridCountNonZero(const SrcPtr& src, GpuMat_<ResType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCountNonZero_<DefaultGlobReducePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename ResType>
__host__ void gridCountNonZero(const SrcPtr& src, GpuMat_<ResType>& dst, Stream& stream = Stream::Null())
{
    gridCountNonZero_<DefaultGlobReducePolicy>(src, dst, stream);
}

}}

#endif
