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

#ifndef __OPENCV_CUDEV_PTR2D_DERIV_HPP__
#define __OPENCV_CUDEV_PTR2D_DERIV_HPP__

#include "../common.hpp"
#include "../grid/copy.hpp"
#include "traits.hpp"
#include "gpumat.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// derivX

template <class SrcPtr> struct DerivXPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;

    __device__ __forceinline__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        return src(y, x + 1) - src(y, x - 1);
    }
};

template <class SrcPtr> struct DerivXPtrSz : DerivXPtr<SrcPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr>
__host__ DerivXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> derivXPtr(const SrcPtr& src)
{
    DerivXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> s;
    s.src = shrinkPtr(src);
    s.rows = getRows(src);
    s.cols = getCols(src);
    return s;
}

template <class SrcPtr> struct PtrTraits< DerivXPtrSz<SrcPtr> > : PtrTraitsBase<DerivXPtrSz<SrcPtr>, DerivXPtr<SrcPtr> >
{
};

// derivY

template <class SrcPtr> struct DerivYPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;

    __device__ __forceinline__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        return src(y + 1, x) - src(y - 1, x);
    }
};

template <class SrcPtr> struct DerivYPtrSz : DerivYPtr<SrcPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr>
__host__ DerivYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> derivYPtr(const SrcPtr& src)
{
    DerivYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> s;
    s.src = shrinkPtr(src);
    s.rows = getRows(src);
    s.cols = getCols(src);
    return s;
}

template <class SrcPtr> struct PtrTraits< DerivYPtrSz<SrcPtr> > : PtrTraitsBase<DerivYPtrSz<SrcPtr>, DerivYPtr<SrcPtr> >
{
};

// sobelX

template <class SrcPtr> struct SobelXPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        typename PtrTraits<SrcPtr>::value_type vals[6] =
        {
            src(y - 1, x - 1), src(y - 1, x + 1),
            src(y    , x - 1), src(y    , x + 1),
            src(y + 1, x - 1), src(y + 1, x + 1),
        };

        return (vals[1] - vals[0]) + 2 * (vals[3] - vals[2]) + (vals[5] - vals[4]);
    }
};

template <class SrcPtr> struct SobelXPtrSz : SobelXPtr<SrcPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr>
__host__ SobelXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> sobelXPtr(const SrcPtr& src)
{
    SobelXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> s;
    s.src = shrinkPtr(src);
    s.rows = getRows(src);
    s.cols = getCols(src);
    return s;
}

template <class SrcPtr> struct PtrTraits< SobelXPtrSz<SrcPtr> > : PtrTraitsBase<SobelXPtrSz<SrcPtr>, SobelXPtr<SrcPtr> >
{
};

// sobelY

template <class SrcPtr> struct SobelYPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        typename PtrTraits<SrcPtr>::value_type vals[6] =
        {
            src(y - 1, x - 1), src(y - 1, x), src(y - 1, x + 1),
            src(y + 1, x - 1), src(y + 1, x), src(y + 1, x + 1)
        };

        return (vals[3] - vals[0]) + 2 * (vals[4] - vals[1]) + (vals[5] - vals[2]);
    }
};

template <class SrcPtr> struct SobelYPtrSz : SobelYPtr<SrcPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr>
__host__ SobelYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> sobelYPtr(const SrcPtr& src)
{
    SobelYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> s;
    s.src = shrinkPtr(src);
    s.rows = getRows(src);
    s.cols = getCols(src);
    return s;
}

template <class SrcPtr> struct PtrTraits< SobelYPtrSz<SrcPtr> > : PtrTraitsBase<SobelYPtrSz<SrcPtr>, SobelYPtr<SrcPtr> >
{
};

// scharrX

template <class SrcPtr> struct ScharrXPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        typename PtrTraits<SrcPtr>::value_type vals[6] =
        {
            src(y - 1, x - 1), src(y - 1, x + 1),
            src(y    , x - 1), src(y    , x + 1),
            src(y + 1, x - 1), src(y + 1, x + 1),
        };

        return 3 * (vals[1] - vals[0]) + 10 * (vals[3] - vals[2]) + 3 * (vals[5] - vals[4]);
    }
};

template <class SrcPtr> struct ScharrXPtrSz : ScharrXPtr<SrcPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr>
__host__ ScharrXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> scharrXPtr(const SrcPtr& src)
{
    ScharrXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> s;
    s.src = shrinkPtr(src);
    s.rows = getRows(src);
    s.cols = getCols(src);
    return s;
}

template <class SrcPtr> struct PtrTraits< ScharrXPtrSz<SrcPtr> > : PtrTraitsBase<ScharrXPtrSz<SrcPtr>, ScharrXPtr<SrcPtr> >
{
};

// scharrY

template <class SrcPtr> struct ScharrYPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        typename PtrTraits<SrcPtr>::value_type vals[6] =
        {
            src(y - 1, x - 1), src(y - 1, x), src(y - 1, x + 1),
            src(y + 1, x - 1), src(y + 1, x), src(y + 1, x + 1)
        };

        return 3 * (vals[3] - vals[0]) + 10 * (vals[4] - vals[1]) + 3 * (vals[5] - vals[2]);
    }
};

template <class SrcPtr> struct ScharrYPtrSz : ScharrYPtr<SrcPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr>
__host__ ScharrYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> scharrYPtr(const SrcPtr& src)
{
    ScharrYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> s;
    s.src = shrinkPtr(src);
    s.rows = getRows(src);
    s.cols = getCols(src);
    return s;
}

template <class SrcPtr> struct PtrTraits< ScharrYPtrSz<SrcPtr> > : PtrTraitsBase<ScharrYPtrSz<SrcPtr>, ScharrYPtr<SrcPtr> >
{
};

// laplacian

template <int ksize, class SrcPtr> struct LaplacianPtr;

template <class SrcPtr> struct LaplacianPtr<1, SrcPtr>
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        typename PtrTraits<SrcPtr>::value_type vals[5] =
        {
                           src(y - 1, x),
            src(y, x - 1), src(y    , x), src(y, x + 1),
                           src(y + 1, x)
        };

        return (vals[0] + vals[1] + vals[3] + vals[4]) - 4 * vals[2];
    }
};

template <class SrcPtr> struct LaplacianPtr<3, SrcPtr>
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

   SrcPtr src;

   __device__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
   {
       typename PtrTraits<SrcPtr>::value_type vals[5] =
       {
           src(y - 1, x - 1),            src(y - 1, x + 1),
                              src(y, x),
           src(y + 1, x - 1),            src(y + 1, x + 1)
       };

       return 2 * (vals[0] + vals[1] + vals[3] + vals[4]) - 8 * vals[2];
   }
};

template <int ksize, class SrcPtr> struct LaplacianPtrSz : LaplacianPtr<ksize, SrcPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <int ksize, class SrcPtr>
__host__ LaplacianPtrSz<ksize, typename PtrTraits<SrcPtr>::ptr_type> laplacianPtr(const SrcPtr& src)
{
    LaplacianPtrSz<ksize, typename PtrTraits<SrcPtr>::ptr_type> ptr;
    ptr.src = shrinkPtr(src);
    ptr.rows = getRows(src);
    ptr.cols = getCols(src);
    return ptr;
}

template <int ksize, class SrcPtr> struct PtrTraits< LaplacianPtrSz<ksize, SrcPtr> > : PtrTraitsBase<LaplacianPtrSz<ksize, SrcPtr>, LaplacianPtr<ksize, SrcPtr> >
{
};

//! @}

}}

#endif
