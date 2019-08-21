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

#ifndef OPENCV_CUDEV_PTR2D_INTERPOLATION_HPP
#define OPENCV_CUDEV_PTR2D_INTERPOLATION_HPP

#include "../common.hpp"
#include "../util/vec_traits.hpp"
#include "../util/saturate_cast.hpp"
#include "../util/type_traits.hpp"
#include "../util/limits.hpp"
#include "traits.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// Nearest

template <class SrcPtr> struct NearestInterPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef float                                  index_type;

    SrcPtr src;

    __device__ __forceinline__ typename PtrTraits<SrcPtr>::value_type operator ()(float y, float x) const
    {
        return src(__float2int_rn(y), __float2int_rn(x));
    }
};

template <class SrcPtr> struct NearestInterPtrSz : NearestInterPtr<SrcPtr>
{
    int rows, cols;
};

template <class SrcPtr>
__host__ NearestInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> interNearest(const SrcPtr& src)
{
    NearestInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> i;
    i.src = shrinkPtr(src);
    i.rows = getRows(src);
    i.cols = getCols(src);
    return i;
}

template <class SrcPtr> struct PtrTraits< NearestInterPtrSz<SrcPtr> > : PtrTraitsBase<NearestInterPtrSz<SrcPtr>, NearestInterPtr<SrcPtr> >
{
};

// Linear

template <typename SrcPtr> struct LinearInterPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef float                                  index_type;

    SrcPtr src;

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(float y, float x) const
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;
        typedef typename VecTraits<src_type>::elem_type src_elem_type;
        typedef typename LargerType<float, src_elem_type>::type work_elem_type;
        typedef typename MakeVec<work_elem_type, VecTraits<src_type>::cn>::type work_type;

        work_type out = VecTraits<work_type>::all(0);

        const int x1 = __float2int_rd(x);
        const int y1 = __float2int_rd(y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;

        typename PtrTraits<SrcPtr>::value_type src_reg = src(y1, x1);
        out = out + src_reg * static_cast<work_elem_type>((x2 - x) * (y2 - y));

        src_reg = src(y1, x2);
        out = out + src_reg * static_cast<work_elem_type>((x - x1) * (y2 - y));

        src_reg = src(y2, x1);
        out = out + src_reg * static_cast<work_elem_type>((x2 - x) * (y - y1));

        src_reg = src(y2, x2);
        out = out + src_reg * static_cast<work_elem_type>((x - x1) * (y - y1));

        return saturate_cast<typename PtrTraits<SrcPtr>::value_type>(out);
    }
};

template <class SrcPtr> struct LinearInterPtrSz : LinearInterPtr<SrcPtr>
{
    int rows, cols;
};

template <class SrcPtr>
__host__ LinearInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> interLinear(const SrcPtr& src)
{
    LinearInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> i;
    i.src = shrinkPtr(src);
    i.rows = getRows(src);
    i.cols = getCols(src);
    return i;
}

template <class SrcPtr> struct PtrTraits< LinearInterPtrSz<SrcPtr> > : PtrTraitsBase<LinearInterPtrSz<SrcPtr>, LinearInterPtr<SrcPtr> >
{
};

// Cubic

template <typename SrcPtr> struct CubicInterPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef float                                  index_type;

    SrcPtr src;

    __device__ static float bicubicCoeff(float x_)
    {
        float x = ::fabsf(x_);
        if (x <= 1.0f)
        {
            return x * x * (1.5f * x - 2.5f) + 1.0f;
        }
        else if (x < 2.0f)
        {
            return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
        }
        else
        {
            return 0.0f;
        }
    }

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(float y, float x) const
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;
        typedef typename VecTraits<src_type>::elem_type src_elem_type;
        typedef typename LargerType<float, src_elem_type>::type work_elem_type;
        typedef typename MakeVec<work_elem_type, VecTraits<src_type>::cn>::type work_type;

        const float xmin = ::ceilf(x - 2.0f);
        const float xmax = ::floorf(x + 2.0f);

        const float ymin = ::ceilf(y - 2.0f);
        const float ymax = ::floorf(y + 2.0f);

        work_type sum = VecTraits<work_type>::all(0);
        float wsum = 0.0f;

        for (float cy = ymin; cy <= ymax; cy += 1.0f)
        {
            for (float cx = xmin; cx <= xmax; cx += 1.0f)
            {
                typename PtrTraits<SrcPtr>::value_type src_reg = src(__float2int_rd(cy), __float2int_rd(cx));
                const float w = bicubicCoeff(x - cx) * bicubicCoeff(y - cy);

                sum = sum + static_cast<work_elem_type>(w) * src_reg;
                wsum += w;
            }
        }

        work_type res = (wsum > numeric_limits<float>::epsilon()) ? VecTraits<work_type>::all(0) : sum / static_cast<work_elem_type>(wsum);

        return saturate_cast<typename PtrTraits<SrcPtr>::value_type>(res);
    }
};

template <class SrcPtr> struct CubicInterPtrSz : CubicInterPtr<SrcPtr>
{
    int rows, cols;
};

template <class SrcPtr>
__host__ CubicInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> interCubic(const SrcPtr& src)
{
    CubicInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> i;
    i.src = shrinkPtr(src);
    i.rows = getRows(src);
    i.cols = getCols(src);
    return i;
}

template <class SrcPtr> struct PtrTraits< CubicInterPtrSz<SrcPtr> > : PtrTraitsBase<CubicInterPtrSz<SrcPtr>, CubicInterPtr<SrcPtr> >
{
};

// IntegerArea

template <typename SrcPtr> struct IntegerAreaInterPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef float                                  index_type;

    SrcPtr src;
    int area_width, area_height;

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(float y, float x) const
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;
        typedef typename VecTraits<src_type>::elem_type src_elem_type;
        typedef typename LargerType<float, src_elem_type>::type work_elem_type;
        typedef typename MakeVec<work_elem_type, VecTraits<src_type>::cn>::type work_type;

        const int sx1 = __float2int_rd(x);
        const int sx2 = sx1 + area_width;

        const int sy1 = __float2int_rd(y);
        const int sy2 = sy1 + area_height;

        work_type out = VecTraits<work_type>::all(0);

        for (int dy = sy1; dy < sy2; ++dy)
        {
            for (int dx = sx1; dx < sx2; ++dx)
            {
                out = out + saturate_cast<work_type>(src(dy, dx));
            }
        }

        const work_elem_type scale = 1.0f / (area_width * area_height);

        return saturate_cast<typename PtrTraits<SrcPtr>::value_type>(out * scale);
    }
};

template <class SrcPtr> struct IntegerAreaInterPtrSz : IntegerAreaInterPtr<SrcPtr>
{
    int rows, cols;
};

template <class SrcPtr>
__host__ IntegerAreaInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> interArea(const SrcPtr& src, Size areaSize)
{
    IntegerAreaInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> i;
    i.src = shrinkPtr(src);
    i.area_width = areaSize.width;
    i.area_height = areaSize.height;
    i.rows = getRows(src);
    i.cols = getCols(src);
    return i;
}

template <class SrcPtr> struct PtrTraits< IntegerAreaInterPtrSz<SrcPtr> > : PtrTraitsBase<IntegerAreaInterPtrSz<SrcPtr>, IntegerAreaInterPtr<SrcPtr> >
{
};

// CommonArea

template <typename SrcPtr> struct CommonAreaInterPtr
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef float                                  index_type;

    SrcPtr src;
    float area_width, area_height;

    __device__ typename PtrTraits<SrcPtr>::value_type operator ()(float y, float x) const
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;
        typedef typename VecTraits<src_type>::elem_type src_elem_type;
        typedef typename LargerType<float, src_elem_type>::type work_elem_type;
        typedef typename MakeVec<work_elem_type, VecTraits<src_type>::cn>::type work_type;

        const float fsx1 = x;
        const float fsx2 = fsx1 + area_width;

        const int sx1 = __float2int_rd(fsx1);
        const int sx2 = __float2int_ru(fsx2);

        const float fsy1 = y;
        const float fsy2 = fsy1 + area_height;

        const int sy1 = __float2int_rd(fsy1);
        const int sy2 = __float2int_ru(fsy2);

        work_type out = VecTraits<work_type>::all(0);

        for (int dy = sy1; dy < sy2; ++dy)
        {
            for (int dx = sx1; dx < sx2; ++dx)
                out = out + saturate_cast<work_type>(src(dy, dx));

            if (sx1 > fsx1)
                out = out + saturate_cast<work_type>(src(dy, sx1 - 1)) * static_cast<work_elem_type>(sx1 - fsx1);

            if (sx2 < fsx2)
                out = out + saturate_cast<work_type>(src(dy, sx2)) * static_cast<work_elem_type>(fsx2 - sx2);
        }

        if (sy1 > fsy1)
        {
            for (int dx = sx1; dx < sx2; ++dx)
                out = out + saturate_cast<work_type>(src(sy1 - 1, dx)) * static_cast<work_elem_type>(sy1 - fsy1);
        }

        if (sy2 < fsy2)
        {
            for (int dx = sx1; dx < sx2; ++dx)
                out = out + saturate_cast<work_type>(src(sy2, dx)) * static_cast<work_elem_type>(fsy2 - sy2);
        }

        if ((sy1 > fsy1) && (sx1 > fsx1))
            out = out + saturate_cast<work_type>(src(sy1 - 1, sx1 - 1)) * static_cast<work_elem_type>((sy1 - fsy1) * (sx1 - fsx1));

        if ((sy1 > fsy1) && (sx2 < fsx2))
            out = out + saturate_cast<work_type>(src(sy1 - 1, sx2)) * static_cast<work_elem_type>((sy1 - fsy1) * (fsx2 - sx2));

        if ((sy2 < fsy2) && (sx2 < fsx2))
            out = out + saturate_cast<work_type>(src(sy2, sx2)) * static_cast<work_elem_type>((fsy2 - sy2) * (fsx2 - sx2));

        if ((sy2 < fsy2) && (sx1 > fsx1))
            out = out + saturate_cast<work_type>(src(sy2, sx1 - 1)) * static_cast<work_elem_type>((fsy2 - sy2) * (sx1 - fsx1));

        const work_elem_type scale = 1.0f / (area_width * area_height);

        return saturate_cast<typename PtrTraits<SrcPtr>::value_type>(out * scale);
    }
};

template <class SrcPtr> struct CommonAreaInterPtrSz : CommonAreaInterPtr<SrcPtr>
{
    int rows, cols;
};

template <class SrcPtr>
__host__ CommonAreaInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> interArea(const SrcPtr& src, Size2f areaSize)
{
    CommonAreaInterPtrSz<typename PtrTraits<SrcPtr>::ptr_type> i;
    i.src = shrinkPtr(src);
    i.area_width = areaSize.width;
    i.area_height = areaSize.height;
    i.rows = getRows(src);
    i.cols = getCols(src);
    return i;
}

template <class SrcPtr> struct PtrTraits< CommonAreaInterPtrSz<SrcPtr> > : PtrTraitsBase<CommonAreaInterPtrSz<SrcPtr>, CommonAreaInterPtr<SrcPtr> >
{
};

//! @}

}}

#endif
