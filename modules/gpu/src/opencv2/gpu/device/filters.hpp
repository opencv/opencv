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

#ifndef __OPENCV_GPU_FILTERS_HPP__
#define __OPENCV_GPU_FILTERS_HPP__

#include "saturate_cast.hpp"
#include "vec_traits.hpp"
#include "vec_math.hpp"

namespace cv { namespace gpu { namespace device 
{
    template <typename Ptr2D> struct PointFilter
    {
        typedef typename Ptr2D::elem_type elem_type;
        typedef float index_type;

        explicit __host__ __device__ __forceinline__ PointFilter(const Ptr2D& src_) : src(src_) {}
         
        __device__ __forceinline__ elem_type operator ()(float y, float x) const
        {
            return src(__float2int_rn(y), __float2int_rn(x));
        }

        const Ptr2D src;
    };

    template <typename Ptr2D> struct LinearFilter
    {
        typedef typename Ptr2D::elem_type elem_type;
        typedef float index_type;

        explicit __host__ __device__ __forceinline__ LinearFilter(const Ptr2D& src_) : src(src_) {}

        __device__ __forceinline__ elem_type operator ()(float y, float x) const
        {
            typedef typename TypeVec<float, VecTraits<elem_type>::cn>::vec_type work_type;

            work_type out = VecTraits<work_type>::all(0);

            const int x1 = __float2int_rd(x);
            const int y1 = __float2int_rd(y);
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;

            elem_type src_reg = src(y1, x1);
            out = out + src_reg * ((x2 - x) * (y2 - y));

            src_reg = src(y1, x2);
            out = out + src_reg * ((x - x1) * (y2 - y));

            src_reg = src(y2, x1);
            out = out + src_reg * ((x2 - x) * (y - y1));

            src_reg = src(y2, x2);
            out = out + src_reg * ((x - x1) * (y - y1));

            return saturate_cast<elem_type>(out);
        }

        const Ptr2D src;
    };

    template <typename Ptr2D> struct CubicFilter
    {
        typedef typename Ptr2D::elem_type elem_type;
        typedef float index_type;
        typedef typename TypeVec<float, VecTraits<elem_type>::cn>::vec_type work_type;

        explicit __host__ __device__ __forceinline__ CubicFilter(const Ptr2D& src_) : src(src_) {}
        
        static __device__ __forceinline__ work_type cubicInterpolate(const work_type& p0, const work_type& p1, const work_type& p2, const work_type& p3, float x) 
        {
            return p1 + 0.5f * x * (p2 - p0 + x * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + x * (3.0f * (p1 - p2) + p3 - p0)));
        }

        __device__ elem_type operator ()(float y, float x) const
        {
            const int xi = __float2int_rn(x);
            const int yi = __float2int_rn(y);
            
            work_type arr[4];
            
            arr[0] = cubicInterpolate(saturate_cast<work_type>(src(yi - 1, xi - 1)), saturate_cast<work_type>(src(yi - 1, xi)), saturate_cast<work_type>(src(yi - 1, xi + 1)), saturate_cast<work_type>(src(yi - 1, xi + 2)), x - xi);
            arr[1] = cubicInterpolate(saturate_cast<work_type>(src(yi    , xi - 1)), saturate_cast<work_type>(src(yi    , xi)), saturate_cast<work_type>(src(yi    , xi + 1)), saturate_cast<work_type>(src(yi    , xi + 2)), x - xi);
            arr[2] = cubicInterpolate(saturate_cast<work_type>(src(yi + 1, xi - 1)), saturate_cast<work_type>(src(yi + 1, xi)), saturate_cast<work_type>(src(yi + 1, xi + 1)), saturate_cast<work_type>(src(yi + 1, xi + 2)), x - xi);
            arr[3] = cubicInterpolate(saturate_cast<work_type>(src(yi + 2, xi - 1)), saturate_cast<work_type>(src(yi + 2, xi)), saturate_cast<work_type>(src(yi + 2, xi + 1)), saturate_cast<work_type>(src(yi + 2, xi + 2)), x - xi);
            
            return saturate_cast<elem_type>(cubicInterpolate(arr[0], arr[1], arr[2], arr[3], y - yi));
        }

        const Ptr2D src;
    };
}}} // namespace cv { namespace gpu { namespace device

#endif // __OPENCV_GPU_FILTERS_HPP__
