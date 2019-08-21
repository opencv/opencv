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

#ifndef OPENCV_CUDEV_GRID_TRANSFORM_HPP
#define OPENCV_CUDEV_GRID_TRANSFORM_HPP

#include "../common.hpp"
#include "../util/tuple.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/gpumat.hpp"
#include "../ptr2d/glob.hpp"
#include "../ptr2d/mask.hpp"
#include "../ptr2d/zip.hpp"
#include "detail/transform.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

template <class Policy, class SrcPtr, typename DstType, class UnOp, class MaskPtr>
__host__ void gridTransformUnary_(const SrcPtr& src, GpuMat_<DstType>& dst, const UnOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_unary<Policy>(shrinkPtr(src), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class UnOp, class MaskPtr>
__host__ void gridTransformUnary_(const SrcPtr& src, const GlobPtrSz<DstType>& dst, const UnOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_transform_detail::transform_unary<Policy>(shrinkPtr(src), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class UnOp>
__host__ void gridTransformUnary_(const SrcPtr& src, GpuMat_<DstType>& dst, const UnOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    dst.create(rows, cols);

    grid_transform_detail::transform_unary<Policy>(shrinkPtr(src), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class UnOp>
__host__ void gridTransformUnary_(const SrcPtr& src, const GlobPtrSz<DstType>& dst, const UnOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );

    grid_transform_detail::transform_unary<Policy>(shrinkPtr(src), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, typename DstType, class BinOp, class MaskPtr>
__host__ void gridTransformBinary_(const SrcPtr1& src1, const SrcPtr2& src2, GpuMat_<DstType>& dst, const BinOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_binary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, typename DstType, class BinOp, class MaskPtr>
__host__ void gridTransformBinary_(const SrcPtr1& src1, const SrcPtr2& src2, const GlobPtrSz<DstType>& dst, const BinOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_transform_detail::transform_binary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, typename DstType, class BinOp>
__host__ void gridTransformBinary_(const SrcPtr1& src1, const SrcPtr2& src2, GpuMat_<DstType>& dst, const BinOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_binary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, typename DstType, class BinOp>
__host__ void gridTransformBinary_(const SrcPtr1& src1, const SrcPtr2& src2, const GlobPtrSz<DstType>& dst, const BinOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );

    grid_transform_detail::transform_binary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>& >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst))),
                                                   op,
                                                   shrinkPtr(mask),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1> >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst))),
                                                   op,
                                                   shrinkPtr(mask),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, class OpTuple>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>& >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst))),
                                                   op,
                                                   WithOutMask(),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, class OpTuple>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1> >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst))),
                                                   op,
                                                   WithOutMask(),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, typename D2, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>& >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst))),
                                                   op,
                                                   shrinkPtr(mask),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, typename D2, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2> >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(get<2>(dst)) == rows && getCols(get<2>(dst)) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst))),
                                                   op,
                                                   shrinkPtr(mask),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, typename D2, class OpTuple>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>& >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst))),
                                                   op,
                                                   WithOutMask(),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, typename D2, class OpTuple>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2> >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(get<2>(dst)) == rows && getCols(get<2>(dst)) == cols );

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst))),
                                                   op,
                                                   WithOutMask(),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, typename D2, typename D3, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>&, GpuMat_<D3>& >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);
    get<3>(dst).create(rows, cols);

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst), get<3>(dst))),
                                                   op,
                                                   shrinkPtr(mask),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, typename D2, typename D3, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2>, GlobPtrSz<D3> >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(get<2>(dst)) == rows && getCols(get<2>(dst)) == cols );
    CV_Assert( getRows(get<3>(dst)) == rows && getCols(get<3>(dst)) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst), get<3>(dst))),
                                                   op,
                                                   shrinkPtr(mask),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, typename D2, typename D3, class OpTuple>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>&, GpuMat_<D3>& >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);
    get<3>(dst).create(rows, cols);

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst), get<3>(dst))),
                                                   op,
                                                   WithOutMask(),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename D0, typename D1, typename D2, typename D3, class OpTuple>
__host__ void gridTransformTuple_(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2>, GlobPtrSz<D3> >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<OpTuple>::value == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(get<2>(dst)) == rows && getCols(get<2>(dst)) == cols );
    CV_Assert( getRows(get<3>(dst)) == rows && getCols(get<3>(dst)) == cols );

    grid_transform_detail::transform_tuple<Policy>(shrinkPtr(src),
                                                   shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst), get<3>(dst))),
                                                   op,
                                                   WithOutMask(),
                                                   rows, cols,
                                                   StreamAccessor::getStream(stream));
}

// Default Policy

struct DefaultTransformPolicy
{
    enum {
        block_size_x = 32,
        block_size_y = 8,
        shift = 4
    };
};

template <class SrcPtr, typename DstType, class Op, class MaskPtr>
__host__ void gridTransformUnary(const SrcPtr& src, GpuMat_<DstType>& dst, const Op& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformUnary_<DefaultTransformPolicy>(src, dst, op, mask, stream);
}

template <class SrcPtr, typename DstType, class Op, class MaskPtr>
__host__ void gridTransformUnary(const SrcPtr& src, const GlobPtrSz<DstType>& dst, const Op& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformUnary_<DefaultTransformPolicy>(src, dst, op, mask, stream);
}

template <class SrcPtr, typename DstType, class Op>
__host__ void gridTransformUnary(const SrcPtr& src, GpuMat_<DstType>& dst, const Op& op, Stream& stream = Stream::Null())
{
    gridTransformUnary_<DefaultTransformPolicy>(src, dst, op, stream);
}

template <class SrcPtr, typename DstType, class Op>
__host__ void gridTransformUnary(const SrcPtr& src, const GlobPtrSz<DstType>& dst, const Op& op, Stream& stream = Stream::Null())
{
    gridTransformUnary_<DefaultTransformPolicy>(src, dst, op, stream);
}

template <class SrcPtr1, class SrcPtr2, typename DstType, class Op, class MaskPtr>
__host__ void gridTransformBinary(const SrcPtr1& src1, const SrcPtr2& src2, GpuMat_<DstType>& dst, const Op& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformBinary_<DefaultTransformPolicy>(src1, src2, dst, op, mask, stream);
}

template <class SrcPtr1, class SrcPtr2, typename DstType, class Op, class MaskPtr>
__host__ void gridTransformBinary(const SrcPtr1& src1, const SrcPtr2& src2, const GlobPtrSz<DstType>& dst, const Op& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformBinary_<DefaultTransformPolicy>(src1, src2, dst, op, mask, stream);
}

template <class SrcPtr1, class SrcPtr2, typename DstType, class Op>
__host__ void gridTransformBinary(const SrcPtr1& src1, const SrcPtr2& src2, GpuMat_<DstType>& dst, const Op& op, Stream& stream = Stream::Null())
{
    gridTransformBinary_<DefaultTransformPolicy>(src1, src2, dst, op, stream);
}

template <class SrcPtr1, class SrcPtr2, typename DstType, class Op>
__host__ void gridTransformBinary(const SrcPtr1& src1, const SrcPtr2& src2, const GlobPtrSz<DstType>& dst, const Op& op, Stream& stream = Stream::Null())
{
    gridTransformBinary_<DefaultTransformPolicy>(src1, src2, dst, op, stream);
}

template <class SrcPtr, typename D0, typename D1, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>& >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, mask, stream);
}

template <class SrcPtr, typename D0, typename D1, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1> >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, mask, stream);
}

template <class SrcPtr, typename D0, typename D1, class OpTuple>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>& >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, stream);
}

template <class SrcPtr, typename D0, typename D1, class OpTuple>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1> >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, stream);
}

template <class SrcPtr, typename D0, typename D1, typename D2, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>& >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, mask, stream);
}

template <class SrcPtr, typename D0, typename D1, typename D2, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2> >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, mask, stream);
}

template <class SrcPtr, typename D0, typename D1, typename D2, class OpTuple>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>& >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, stream);
}

template <class SrcPtr, typename D0, typename D1, typename D2, class OpTuple>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2> >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, stream);
}

template <class SrcPtr, typename D0, typename D1, typename D2, typename D3, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>&, GpuMat_<D3>& >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, mask, stream);
}

template <class SrcPtr, typename D0, typename D1, typename D2, typename D3, class OpTuple, class MaskPtr>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2>, GlobPtrSz<D3> >& dst, const OpTuple& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, mask, stream);
}

template <class SrcPtr, typename D0, typename D1, typename D2, typename D3, class OpTuple>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>&, GpuMat_<D3>& >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, stream);
}

template <class SrcPtr, typename D0, typename D1, typename D2, typename D3, class OpTuple>
__host__ void gridTransformTuple(const SrcPtr& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2>, GlobPtrSz<D3> >& dst, const OpTuple& op, Stream& stream = Stream::Null())
{
    gridTransformTuple_<DefaultTransformPolicy>(src, dst, op, stream);
}

//! @}

}}

#endif
