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

#ifndef __OPENCV_CUDEV_GRID_SPLIT_MERGE_HPP__
#define __OPENCV_CUDEV_GRID_SPLIT_MERGE_HPP__

#include "../common.hpp"
#include "../util/tuple.hpp"
#include "../util/vec_traits.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/gpumat.hpp"
#include "../ptr2d/glob.hpp"
#include "../ptr2d/mask.hpp"
#include "detail/split_merge.hpp"

namespace cv { namespace cudev {

template <class Policy, class SrcPtrTuple, typename DstType, class MaskPtr>
__host__ void gridMerge_(const SrcPtrTuple& src, GpuMat_<DstType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<DstType>::cn == tuple_size<SrcPtrTuple>::value, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(rows, cols);

    grid_split_merge_detail::MergeImpl<VecTraits<DstType>::cn, Policy>::merge(shrinkPtr(src),
                                                                              shrinkPtr(dst),
                                                                              shrinkPtr(mask),
                                                                              rows, cols,
                                                                              StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename DstType, class MaskPtr>
__host__ void gridMerge_(const SrcPtrTuple& src, const GlobPtrSz<DstType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<DstType>::cn == tuple_size<SrcPtrTuple>::value, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_split_merge_detail::MergeImpl<VecTraits<DstType>::cn, Policy>::merge(shrinkPtr(src),
                                                                              shrinkPtr(dst),
                                                                              shrinkPtr(mask),
                                                                              rows, cols,
                                                                              StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename DstType>
__host__ void gridMerge_(const SrcPtrTuple& src, GpuMat_<DstType>& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<DstType>::cn == tuple_size<SrcPtrTuple>::value, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    dst.create(rows, cols);

    grid_split_merge_detail::MergeImpl<VecTraits<DstType>::cn, Policy>::merge(shrinkPtr(src),
                                                                              shrinkPtr(dst),
                                                                              WithOutMask(),
                                                                              rows, cols,
                                                                              StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtrTuple, typename DstType>
__host__ void gridMerge_(const SrcPtrTuple& src, const GlobPtrSz<DstType>& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<DstType>::cn == tuple_size<SrcPtrTuple>::value, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );

    grid_split_merge_detail::MergeImpl<VecTraits<DstType>::cn, Policy>::merge(shrinkPtr(src),
                                                                              shrinkPtr(dst),
                                                                              WithOutMask(),
                                                                              rows, cols,
                                                                              StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(get<0>(dst)), shrinkPtr(get<1>(dst)),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, GpuMat_<DstType> (&dst)[2], const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst[0].create(rows, cols);
    dst[1].create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, GlobPtrSz<DstType> (&dst)[2], const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst[0]) == rows && getCols(dst[0]) == cols );
    CV_Assert( getRows(dst[1]) == rows && getCols(dst[1]) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(get<0>(dst)), shrinkPtr(get<1>(dst)),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, GpuMat_<DstType> (&dst)[2], Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    dst[0].create(rows, cols);
    dst[1].create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, GlobPtrSz<DstType> (&dst)[2], Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 2, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst[0]) == rows && getCols(dst[0]) == cols );
    CV_Assert( getRows(dst[1]) == rows && getCols(dst[1]) == cols );

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(get<0>(dst)), shrinkPtr(get<1>(dst)), shrinkPtr(get<2>(dst)),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, GpuMat_<DstType> (&dst)[3], const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst[0].create(rows, cols);
    dst[1].create(rows, cols);
    dst[2].create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]), shrinkPtr(dst[2]),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, GlobPtrSz<DstType> (&dst)[3], const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst[0]) == rows && getCols(dst[0]) == cols );
    CV_Assert( getRows(dst[1]) == rows && getCols(dst[1]) == cols );
    CV_Assert( getRows(dst[2]) == rows && getCols(dst[2]) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]), shrinkPtr(dst[2]),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(get<0>(dst)), shrinkPtr(get<1>(dst)), shrinkPtr(get<2>(dst)),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, GpuMat_<DstType> (&dst)[3], Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    dst[0].create(rows, cols);
    dst[1].create(rows, cols);
    dst[2].create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]), shrinkPtr(dst[2]),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, GlobPtrSz<DstType> (&dst)[3], Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 3, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst[0]) == rows && getCols(dst[0]) == cols );
    CV_Assert( getRows(dst[1]) == rows && getCols(dst[1]) == cols );
    CV_Assert( getRows(dst[2]) == rows && getCols(dst[2]) == cols );

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]), shrinkPtr(dst[2]),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);
    get<3>(dst).create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(get<0>(dst)), shrinkPtr(get<1>(dst)), shrinkPtr(get<2>(dst)), shrinkPtr(get<3>(dst)),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, GpuMat_<DstType> (&dst)[4], const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst[0].create(rows, cols);
    dst[1].create(rows, cols);
    dst[2].create(rows, cols);
    dst[3].create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]), shrinkPtr(dst[2]), shrinkPtr(dst[3]),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit_(const SrcPtr& src, GlobPtrSz<DstType> (&dst)[4], const MaskPtr& mask, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst[0]) == rows && getCols(dst[0]) == cols );
    CV_Assert( getRows(dst[1]) == rows && getCols(dst[1]) == cols );
    CV_Assert( getRows(dst[2]) == rows && getCols(dst[2]) == cols );
    CV_Assert( getRows(dst[3]) == rows && getCols(dst[3]) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]), shrinkPtr(dst[2]), shrinkPtr(dst[3]),
                                           shrinkPtr(mask),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    get<0>(dst).create(rows, cols);
    get<1>(dst).create(rows, cols);
    get<2>(dst).create(rows, cols);
    get<3>(dst).create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(get<0>(dst)), shrinkPtr(get<1>(dst)), shrinkPtr(get<2>(dst)), shrinkPtr(get<3>(dst)),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, GpuMat_<DstType> (&dst)[4], Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    dst[0].create(rows, cols);
    dst[1].create(rows, cols);
    dst[2].create(rows, cols);
    dst[3].create(rows, cols);

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]), shrinkPtr(dst[2]), shrinkPtr(dst[3]),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr, typename DstType>
__host__ void gridSplit_(const SrcPtr& src, GlobPtrSz<DstType> (&dst)[4], Stream& stream = Stream::Null())
{
    CV_StaticAssert( VecTraits<typename PtrTraits<SrcPtr>::value_type>::cn == 4, "" );

    const int rows = getRows(src);
    const int cols = getCols(src);

    CV_Assert( getRows(dst[0]) == rows && getCols(dst[0]) == cols );
    CV_Assert( getRows(dst[1]) == rows && getCols(dst[1]) == cols );
    CV_Assert( getRows(dst[2]) == rows && getCols(dst[2]) == cols );
    CV_Assert( getRows(dst[3]) == rows && getCols(dst[3]) == cols );

    grid_split_merge_detail::split<Policy>(shrinkPtr(src),
                                           shrinkPtr(dst[0]), shrinkPtr(dst[1]), shrinkPtr(dst[2]), shrinkPtr(dst[3]),
                                           WithOutMask(),
                                           rows, cols,
                                           StreamAccessor::getStream(stream));
}

// Default Policy

struct DefaultSplitMergePolicy
{
    enum {
        block_size_x = 32,
        block_size_y = 8
    };
};

template <class SrcPtrTuple, typename DstType, class MaskPtr>
__host__ void gridMerge(const SrcPtrTuple& src, GpuMat_<DstType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridMerge_<DefaultSplitMergePolicy>(src, dst, mask, stream);
}

template <class SrcPtrTuple, typename DstType, class MaskPtr>
__host__ void gridMerge(const SrcPtrTuple& src, const GlobPtrSz<DstType>& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridMerge_<DefaultSplitMergePolicy>(src, dst, mask, stream);
}

template <class SrcPtrTuple, typename DstType>
__host__ void gridMerge(const SrcPtrTuple& src, GpuMat_<DstType>& dst, Stream& stream = Stream::Null())
{
    gridMerge_<DefaultSplitMergePolicy>(src, dst, stream);
}

template <class SrcPtrTuple, typename DstType>
__host__ void gridMerge(const SrcPtrTuple& src, const GlobPtrSz<DstType>& dst, Stream& stream = Stream::Null())
{
    gridMerge_<DefaultSplitMergePolicy>(src, dst, stream);
}

template <class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename DstType>
__host__ void gridSplit(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, stream);
}

template <class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename DstType>
__host__ void gridSplit(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, stream);
}

template <class SrcPtr, typename DstType, class MaskPtr>
__host__ void gridSplit(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename DstType>
__host__ void gridSplit(const SrcPtr& src, const tuple< GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>&, GpuMat_<DstType>& >& dst, Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, stream);
}

template <class SrcPtr, typename DstType, int COUNT, class MaskPtr>
__host__ void gridSplit(const SrcPtr& src, GpuMat_<DstType> (&dst)[COUNT], const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename DstType, int COUNT, class MaskPtr>
__host__ void gridSplit(const SrcPtr& src, GlobPtrSz<DstType> (&dst)[COUNT], const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, mask, stream);
}

template <class SrcPtr, typename DstType, int COUNT>
__host__ void gridSplit(const SrcPtr& src, GpuMat_<DstType> (&dst)[COUNT], Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, stream);
}

template <class SrcPtr, typename DstType, int COUNT>
__host__ void gridSplit(const SrcPtr& src, GlobPtrSz<DstType> (&dst)[COUNT], Stream& stream = Stream::Null())
{
    gridSplit_<DefaultSplitMergePolicy>(src, dst, stream);
}

}}

#endif
