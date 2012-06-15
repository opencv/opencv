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
#include "type_traits.hpp"

namespace cv { namespace gpu { namespace device
{
    template <typename Ptr2D> struct PointFilter
    {
        typedef typename Ptr2D::elem_type elem_type;
        typedef float index_type;

        explicit __host__ __device__ __forceinline__ PointFilter(const Ptr2D& src_, float fx = 0.f, float fy = 0.f) : src(src_) {}

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

        explicit __host__ __device__ __forceinline__ LinearFilter(const Ptr2D& src_, float fx = 0.f, float fy = 0.f) : src(src_) {}

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

        explicit __host__ __device__ __forceinline__ CubicFilter(const Ptr2D& src_, float fx = 0.f, float fy = 0.f) : src(src_) {}

        static __device__ __forceinline__ float bicubicCoeff(float x_)
        {
            float x = fabsf(x_);
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

        __device__ elem_type operator ()(float y, float x) const
        {
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
                    const float w = bicubicCoeff(x - cx) * bicubicCoeff(y - cy);
                    sum = sum + w * src(__float2int_rd(cy), __float2int_rd(cx));
                    wsum += w;
                }
            }

            work_type res = (!wsum)? VecTraits<work_type>::all(0) : sum / wsum;

            return saturate_cast<elem_type>(res);
        }

        const Ptr2D src;
    };
    // for integer scaling
    template <typename Ptr2D> struct IntegerAreaFilter
    {
        typedef typename Ptr2D::elem_type elem_type;
        typedef float index_type;

        explicit __host__ __device__ __forceinline__ IntegerAreaFilter(const Ptr2D& src_, float scale_x_, float scale_y_)
            : src(src_), scale_x(scale_x_), scale_y(scale_y_), scale(1.f / (scale_x * scale_y)) {}

        __device__ __forceinline__ elem_type operator ()(float y, float x) const
        {
            float fsx1 = x * scale_x;
            float fsx2 = fsx1 + scale_x;

            int sx1 = __float2int_ru(fsx1);
            int sx2 = __float2int_rd(fsx2);

            float fsy1 = y * scale_y;
            float fsy2 = fsy1 + scale_y;

            int sy1 = __float2int_ru(fsy1);
            int sy2 = __float2int_rd(fsy2);

            typedef typename TypeVec<float, VecTraits<elem_type>::cn>::vec_type work_type;
            work_type out = VecTraits<work_type>::all(0.f);

            for(int dy = sy1; dy < sy2; ++dy)
                for(int dx = sx1; dx < sx2; ++dx)
                {
                    out = out + src(dy, dx) * scale;
                }

            return saturate_cast<elem_type>(out);
        }

        const Ptr2D src;
        float scale_x, scale_y ,scale;
    };

    template <typename Ptr2D> struct AreaFilter
    {
        typedef typename Ptr2D::elem_type elem_type;
        typedef float index_type;

        explicit __host__ __device__ __forceinline__ AreaFilter(const Ptr2D& src_, float scale_x_, float scale_y_)
            : src(src_), scale_x(scale_x_), scale_y(scale_y_){}

        __device__ __forceinline__ elem_type operator ()(float y, float x) const
        {
            float fsx1 = x * scale_x;
            float fsx2 = fsx1 + scale_x;

            int sx1 = __float2int_ru(fsx1);
            int sx2 = __float2int_rd(fsx2);

            float fsy1 = y * scale_y;
            float fsy2 = fsy1 + scale_y;

            int sy1 = __float2int_ru(fsy1);
            int sy2 = __float2int_rd(fsy2);

            float scale = 1.f / (fminf(scale_x, src.width - fsx1) * fminf(scale_y, src.height - fsy1));

            typedef typename TypeVec<float, VecTraits<elem_type>::cn>::vec_type work_type;
            work_type out = VecTraits<work_type>::all(0.f);

            for (int dy = sy1; dy < sy2; ++dy)
            {
                for (int dx = sx1; dx < sx2; ++dx)
                    out = out + src(dy, dx) * scale;

                if (sx1 > fsx1)
                    out = out + src(dy, (sx1 -1) ) * ((sx1 - fsx1) * scale);

                if (sx2 < fsx2)
                    out = out + src(dy, sx2) * ((fsx2 -sx2) * scale);
            }

            if (sy1 > fsy1)
                for (int dx = sx1; dx < sx2; ++dx)
                    out = out + src( (sy1 - 1) , dx) * ((sy1 -fsy1) * scale);

            if (sy2 < fsy2)
                for (int dx = sx1; dx < sx2; ++dx)
                    out = out + src(sy2, dx) * ((fsy2 -sy2) * scale);

            if ((sy1 > fsy1) &&  (sx1 > fsx1))
                out = out + src( (sy1 - 1) , (sx1 - 1)) * ((sy1 -fsy1) * (sx1 -fsx1) * scale);

            if ((sy1 > fsy1) &&  (sx2 < fsx2))
                out = out + src( (sy1 - 1) , sx2) * ((sy1 -fsy1) * (fsx2 -sx2) * scale);

            if ((sy2 < fsy2) &&  (sx2 < fsx2))
                out = out + src(sy2, sx2) * ((fsy2 -sy2) * (fsx2 -sx2) * scale);

            if ((sy2 < fsy2) &&  (sx1 > fsx1))
                out = out + src(sy2, (sx1 - 1)) * ((fsy2 -sy2) * (sx1 -fsx1) * scale);

            return saturate_cast<elem_type>(out);
        }

        const Ptr2D src;
        float scale_x, scale_y;
        int width, haight;
    };
}}} // namespace cv { namespace gpu { namespace device

#endif // __OPENCV_GPU_FILTERS_HPP__
