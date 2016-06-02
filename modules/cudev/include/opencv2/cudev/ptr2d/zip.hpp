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

#ifndef __OPENCV_CUDEV_PTR2D_ZIP_HPP__
#define __OPENCV_CUDEV_PTR2D_ZIP_HPP__

#include "../common.hpp"
#include "../util/tuple.hpp"
#include "traits.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

template <class PtrTuple> struct ZipPtr;

template <class Ptr0, class Ptr1> struct ZipPtr< tuple<Ptr0, Ptr1> > : tuple<Ptr0, Ptr1>
{
    typedef tuple<typename PtrTraits<Ptr0>::value_type,
                  typename PtrTraits<Ptr1>::value_type> value_type;
    typedef typename PtrTraits<Ptr0>::index_type        index_type;

    __host__ __device__ __forceinline__ ZipPtr() {}
    __host__ __device__ __forceinline__ ZipPtr(const tuple<Ptr0, Ptr1>& t) : tuple<Ptr0, Ptr1>(t) {}

    __device__ __forceinline__ value_type operator ()(index_type y, index_type x) const
    {
        return make_tuple(cv::cudev::get<0>(*this)(y, x), cv::cudev::get<1>(*this)(y, x));
    }
};

template <class Ptr0, class Ptr1, class Ptr2> struct ZipPtr< tuple<Ptr0, Ptr1, Ptr2> > : tuple<Ptr0, Ptr1, Ptr2>
{
    typedef tuple<typename PtrTraits<Ptr0>::value_type,
                  typename PtrTraits<Ptr1>::value_type,
                  typename PtrTraits<Ptr2>::value_type> value_type;
    typedef typename PtrTraits<Ptr0>::index_type        index_type;

    __host__ __device__ __forceinline__ ZipPtr() {}
    __host__ __device__ __forceinline__ ZipPtr(const tuple<Ptr0, Ptr1, Ptr2>& t) : tuple<Ptr0, Ptr1, Ptr2>(t) {}

    __device__ __forceinline__ value_type operator ()(index_type y, index_type x) const
    {
        return make_tuple(cv::cudev::get<0>(*this)(y, x), cv::cudev::get<1>(*this)(y, x), cv::cudev::get<2>(*this)(y, x));
    }
};

template <class Ptr0, class Ptr1, class Ptr2, class Ptr3> struct ZipPtr< tuple<Ptr0, Ptr1, Ptr2, Ptr3> > : tuple<Ptr0, Ptr1, Ptr2, Ptr3>
{
    typedef tuple<typename PtrTraits<Ptr0>::value_type,
                  typename PtrTraits<Ptr1>::value_type,
                  typename PtrTraits<Ptr2>::value_type,
                  typename PtrTraits<Ptr3>::value_type> value_type;
    typedef typename PtrTraits<Ptr0>::index_type        index_type;

    __host__ __device__ __forceinline__ ZipPtr() {}
    __host__ __device__ __forceinline__ ZipPtr(const tuple<Ptr0, Ptr1, Ptr2, Ptr3>& t) : tuple<Ptr0, Ptr1, Ptr2, Ptr3>(t) {}

    __device__ __forceinline__ value_type operator ()(index_type y, index_type x) const
    {
        return make_tuple(cv::cudev::get<0>(*this)(y, x), cv::cudev::get<1>(*this)(y, x), cv::cudev::get<2>(*this)(y, x), cv::cudev::get<3>(*this)(y, x));
    }
};

template <class PtrTuple> struct ZipPtrSz : ZipPtr<PtrTuple>
{
    int rows, cols;

    __host__ __device__ __forceinline__ ZipPtrSz() {}
    __host__ __device__ __forceinline__ ZipPtrSz(const PtrTuple& t) : ZipPtr<PtrTuple>(t) {}
};

template <class Ptr0, class Ptr1>
__host__ ZipPtrSz< tuple<typename PtrTraits<Ptr0>::ptr_type, typename PtrTraits<Ptr1>::ptr_type> >
zipPtr(const Ptr0& ptr0, const Ptr1& ptr1)
{
    const int rows = getRows(ptr0);
    const int cols = getCols(ptr0);

    CV_Assert( getRows(ptr1) == rows && getCols(ptr1) == cols );

    ZipPtrSz< tuple<typename PtrTraits<Ptr0>::ptr_type, typename PtrTraits<Ptr1>::ptr_type> >
            z(make_tuple(shrinkPtr(ptr0), shrinkPtr(ptr1)));
    z.rows = rows;
    z.cols = cols;

    return z;
}

template <class Ptr0, class Ptr1, class Ptr2>
__host__ ZipPtrSz< tuple<typename PtrTraits<Ptr0>::ptr_type, typename PtrTraits<Ptr1>::ptr_type, typename PtrTraits<Ptr2>::ptr_type> >
zipPtr(const Ptr0& ptr0, const Ptr1& ptr1, const Ptr2& ptr2)
{
    const int rows = getRows(ptr0);
    const int cols = getCols(ptr0);

    CV_Assert( getRows(ptr1) == rows && getCols(ptr1) == cols );
    CV_Assert( getRows(ptr2) == rows && getCols(ptr2) == cols );

    ZipPtrSz< tuple<typename PtrTraits<Ptr0>::ptr_type, typename PtrTraits<Ptr1>::ptr_type, typename PtrTraits<Ptr2>::ptr_type> >
            z(make_tuple(shrinkPtr(ptr0), shrinkPtr(ptr1), shrinkPtr(ptr2)));
    z.rows = rows;
    z.cols = cols;

    return z;
}

template <class Ptr0, class Ptr1, class Ptr2, class Ptr3>
__host__ ZipPtrSz< tuple<typename PtrTraits<Ptr0>::ptr_type, typename PtrTraits<Ptr1>::ptr_type, typename PtrTraits<Ptr2>::ptr_type, typename PtrTraits<Ptr3>::ptr_type> >
zipPtr(const Ptr0& ptr0, const Ptr1& ptr1, const Ptr2& ptr2, const Ptr3& ptr3)
{
    const int rows = getRows(ptr0);
    const int cols = getCols(ptr0);

    CV_Assert( getRows(ptr1) == rows && getCols(ptr1) == cols );
    CV_Assert( getRows(ptr2) == rows && getCols(ptr2) == cols );
    CV_Assert( getRows(ptr3) == rows && getCols(ptr3) == cols );

    ZipPtrSz< tuple<typename PtrTraits<Ptr0>::ptr_type, typename PtrTraits<Ptr1>::ptr_type, typename PtrTraits<Ptr2>::ptr_type, typename PtrTraits<Ptr3>::ptr_type> >
            z(make_tuple(shrinkPtr(ptr0), shrinkPtr(ptr1), shrinkPtr(ptr2), shrinkPtr(ptr3)));
    z.rows = rows;
    z.cols = cols;

    return z;
}

template <class PtrTuple> struct PtrTraits< ZipPtrSz<PtrTuple> > : PtrTraitsBase<ZipPtrSz<PtrTuple>, ZipPtr<PtrTuple> >
{
};

//! @}

}}

#endif
