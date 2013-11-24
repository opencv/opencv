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

#ifndef __OPENCV_CUDEV_GRID_COPY_HPP__
#define __OPENCV_CUDEV_GRID_COPY_HPP__

#include "../common.hpp"
#include "../util/tuple.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/gpumat.hpp"
#include "../ptr2d/glob.hpp"
#include "../ptr2d/mask.hpp"
#include "../ptr2d/zip.hpp"
#include "detail/copy.hpp"

namespace cv { namespace cudev {

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridCopy_(const SrcPtr& src, GpuMat_<DstType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(rows, cols);

    grid_copy_detail::copy<Policy>(shrinkPtr(src), shrinkPtr(dst), shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridCopy_(const SrcPtr& src, const GlobPtrSz<DstType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_copy_detail::copy<Policy>(shrinkPtr(src), shrinkPtr(dst), shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridCopy_(const SrcPtr& src, GpuMat_<DstType>& dst, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    dst.create(rows, cols);

    grid_copy_detail::copy<Policy>(shrinkPtr(src), shrinkPtr(dst), WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridCopy_(const SrcPtr& src, const GlobPtrSz<DstType>& dst, Stream& stream = Stream::Null())
{
    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );

    grid_copy_detail::copy<Policy>(shrinkPtr(src), shrinkPtr(dst), WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, class MaskPtr>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst))),
                                         shrinkPtr(mask),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, class MaskPtr>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1> >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst))),
                                         shrinkPtr(mask),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>& >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst))),
                                         WithOutMask(),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1> >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst))),
                                         WithOutMask(),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, typename D2, class MaskPtr>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst))),
                                         shrinkPtr(mask),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, typename D2, class MaskPtr>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2> >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(get<2>(dst)) == rows && getCols(get<2>(dst)) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst))),
                                         shrinkPtr(mask),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, typename D2>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>& >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst))),
                                         WithOutMask(),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, typename D2>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2> >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(get<2>(dst)) == rows && getCols(get<2>(dst)) == cols );

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst))),
                                         WithOutMask(),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, typename D2, typename D3, class MaskPtr>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>&, GpuMat_<D3>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);
    get<3>(dst).create(rows, cols);

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst), get<3>(dst))),
                                         shrinkPtr(mask),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, typename D2, typename D3, class MaskPtr>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2>, GlobPtrSz<D3> >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(get<2>(dst)) == rows && getCols(get<2>(dst)) == cols );
    CV_Assert( getRows(get<3>(dst)) == rows && getCols(get<3>(dst)) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst), get<3>(dst))),
                                         shrinkPtr(mask),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, typename D2, typename D3>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>&, GpuMat_<D3>& >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);
    get<3>(dst).create(rows, cols);

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst), get<3>(dst))),
                                         WithOutMask(),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename D0, typename D1, typename D2, typename D3>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2>, GlobPtrSz<D3> >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( tuple_size<SrcPtrTuple>::value == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(get<0>(dst)) == rows && getCols(get<0>(dst)) == cols );
    CV_Assert( getRows(get<1>(dst)) == rows && getCols(get<1>(dst)) == cols );
    CV_Assert( getRows(get<2>(dst)) == rows && getCols(get<2>(dst)) == cols );
    CV_Assert( getRows(get<3>(dst)) == rows && getCols(get<3>(dst)) == cols );

    grid_copy_detail::copy_tuple<Policy>(shrinkPtr(src),
                                         shrinkPtr(zipPtr(get<0>(dst), get<1>(dst), get<2>(dst), get<3>(dst))),
                                         WithOutMask(),
                                         rows, cols,
                                         StreamAccessor::getStream(stream));
}

// Default Policy

struct DefaultCopyPolicy
{
    enum {
        block_size_x = 32,
        block_size_y = 8
    };
};

template <class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridCopy(const SrcPtr& src, GpuMat_<DstType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridCopy(const SrcPtr& src, const GlobPtrSz<DstType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename DstType>
__host__ void gridCopy(const SrcPtr& src, GpuMat_<DstType>& dst, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, stream);
}

template <class SrcPtr, typename DstType>
__host__ void gridCopy(const SrcPtr& src, const GlobPtrSz<DstType>& dst, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, class MaskPtr>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, mask, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, class MaskPtr>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1> >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, mask, stream);
}

template <class SrcPtrTuple, typename D0, typename D1>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>& >& dst, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, stream);
}

template <class SrcPtrTuple, typename D0, typename D1>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1> >& dst, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, typename D2, class MaskPtr>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, mask, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, typename D2, class MaskPtr>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2> >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, mask, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, typename D2>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>& >& dst, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, typename D2>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2> >& dst, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, typename D2, typename D3, class MaskPtr>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>&, GpuMat_<D3>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, mask, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, typename D2, typename D3, class MaskPtr>
__host__ void gridCopy(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2>, GlobPtrSz<D3> >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, mask, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, typename D2, typename D3>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GpuMat_<D0>&, GpuMat_<D1>&, GpuMat_<D2>&, GpuMat_<D3>& >& dst, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, stream);
}

template <class SrcPtrTuple, typename D0, typename D1, typename D2, typename D3>
__host__ void gridCopy_(const SrcPtrTuple& src, const tuple< GlobPtrSz<D0>, GlobPtrSz<D1>, GlobPtrSz<D2>, GlobPtrSz<D3> >& dst, Stream& stream = Stream::Null())
{
    gridCopy_<DefaultCopyPolicy>(src, dst, stream);
}

}}

#endif
