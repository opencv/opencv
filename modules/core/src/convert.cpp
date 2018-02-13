// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "convert.hpp"
#include "opencv2/core/openvx/ovx_defs.hpp"

namespace cv {

template <typename T, typename DT>
struct Cvt_SIMD
{
    int operator() (const T *, DT *, int) const
    {
        return 0;
    }
};

#if CV_SIMD128
// from uchar

template <>
struct Cvt_SIMD<uchar, schar>
{
    int operator() (const uchar * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_int16x8 v_src = v_reinterpret_as_s16(v_load_expand(src + x));
                v_store_low(dst + x, v_pack(v_src, v_src));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, ushort>
{
    int operator() (const uchar * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_load_expand(src + x));
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, short>
{
    int operator() (const uchar * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_int16x8 v_src = v_reinterpret_as_s16(v_load_expand(src + x));
                v_store(dst + x, v_src);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, int>
{
    int operator() (const uchar * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src = v_load_expand(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_reinterpret_as_s32(v_src1));
                v_store(dst + x + cWidth, v_reinterpret_as_s32(v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, float>
{
    int operator() (const uchar * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src = v_load_expand(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f32(v_reinterpret_as_s32(v_src1)));
                v_store(dst + x + cWidth, v_cvt_f32(v_reinterpret_as_s32(v_src2)));
            }
        }
        return x;
    }
};

// from schar

template <>
struct Cvt_SIMD<schar, uchar>
{
    int operator() (const schar * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_pack_u_store(dst + x, v_load_expand(src + x));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<schar, short>
{
    int operator() (const schar * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_load_expand(src + x));
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<schar, ushort>
{
    int operator() (const schar * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_int16x8 v_src = v_load_expand(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_pack_u(v_src1, v_src2));
            }
        }
        return x;
    }
};


template <>
struct Cvt_SIMD<schar, int>
{
    int operator() (const schar * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load_expand(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<schar, float>
{
    int operator() (const schar * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load_expand(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f32(v_src1));
                v_store(dst + x + cWidth, v_cvt_f32(v_src2));
            }
        }
        return x;
    }
};

// from ushort

template <>
struct Cvt_SIMD<ushort, uchar>
{
    int operator() (const ushort * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, schar>
{
    int operator() (const ushort * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_uint32x4 v_dst10, v_dst11, v_dst20, v_dst21;
                v_expand(v_src1, v_dst10, v_dst11);
                v_expand(v_src2, v_dst20, v_dst21);

                v_store(dst + x, v_pack(
                    v_pack(v_reinterpret_as_s32(v_dst10), v_reinterpret_as_s32(v_dst11)),
                    v_pack(v_reinterpret_as_s32(v_dst20), v_reinterpret_as_s32(v_dst21))));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, short>
{
    int operator() (const ushort * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_dst0, v_dst1;
                v_expand(v_src, v_dst0, v_dst1);
                v_store(dst + x, v_pack(v_reinterpret_as_s32(v_dst0), v_reinterpret_as_s32(v_dst1)));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, int>
{
    int operator() (const ushort * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_reinterpret_as_s32(v_src1));
                v_store(dst + x + cWidth, v_reinterpret_as_s32(v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, float>
{
    int operator() (const ushort * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f32(v_reinterpret_as_s32(v_src1)));
                v_store(dst + x + cWidth, v_cvt_f32(v_reinterpret_as_s32(v_src2)));
            }
        }
        return x;
    }
};


// from short

template <>
struct Cvt_SIMD<short, uchar>
{
    int operator() (const short * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack_u(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, schar>
{
    int operator() (const short * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, ushort>
{
    int operator() (const short * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_int16x8 v_src = v_load(src + x);
                v_int32x4 v_dst1, v_dst2;
                v_expand(v_src, v_dst1, v_dst2);
                v_store(dst + x, v_pack_u(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, int>
{
    int operator() (const short * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load(src + x);
                v_int32x4 v_dst1, v_dst2;
                v_expand(v_src, v_dst1, v_dst2);
                v_store(dst + x, v_dst1);
                v_store(dst + x + cWidth, v_dst2);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, float>
{
    int operator() (const short * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load(src + x);
                v_int32x4 v_dst1, v_dst2;
                v_expand(v_src, v_dst1, v_dst2);
                v_store(dst + x, v_cvt_f32(v_dst1));
                v_store(dst + x + cWidth, v_cvt_f32(v_dst2));
            }
        }
        return x;
    }
};

// from int

template <>
struct Cvt_SIMD<int, uchar>
{
    int operator() (const int * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_int32x4 v_src3 = v_load(src + x + cWidth * 2), v_src4 = v_load(src + x + cWidth * 3);
                v_uint16x8 v_dst1 = v_pack_u(v_src1, v_src2);
                v_uint16x8 v_dst2 = v_pack_u(v_src3, v_src4);
                v_store(dst + x, v_pack(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<int, schar>
{
    int operator() (const int * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_int32x4 v_src3 = v_load(src + x + cWidth * 2), v_src4 = v_load(src + x + cWidth * 3);
                v_int16x8 v_dst1 = v_pack(v_src1, v_src2);
                v_int16x8 v_dst2 = v_pack(v_src3, v_src4);
                v_store(dst + x, v_pack(v_dst1, v_dst2));
            }
        }
        return x;
    }
};


template <>
struct Cvt_SIMD<int, ushort>
{
    int operator() (const int * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack_u(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<int, short>
{
    int operator() (const int * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_load(src + x), v_src2 = v_load(src + x + cWidth);
                v_store(dst + x, v_pack(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<int, float>
{
    int operator() (const int * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_cvt_f32(v_load(src + x)));
        }
        return x;
    }
};

// from float

template <>
struct Cvt_SIMD<float, uchar>
{
    int operator() (const float * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1 = v_round(v_load(src + x));
                v_int32x4 v_src2 = v_round(v_load(src + x + cWidth));
                v_int32x4 v_src3 = v_round(v_load(src + x + cWidth * 2));
                v_int32x4 v_src4 = v_round(v_load(src + x + cWidth * 3));
                v_uint16x8 v_dst1 = v_pack_u(v_src1, v_src2);
                v_uint16x8 v_dst2 = v_pack_u(v_src3, v_src4);
                v_store(dst + x, v_pack(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, schar>
{
    int operator() (const float * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1 = v_round(v_load(src + x));
                v_int32x4 v_src2 = v_round(v_load(src + x + cWidth));
                v_int32x4 v_src3 = v_round(v_load(src + x + cWidth * 2));
                v_int32x4 v_src4 = v_round(v_load(src + x + cWidth * 3));
                v_int16x8 v_dst1 = v_pack(v_src1, v_src2);
                v_int16x8 v_dst2 = v_pack(v_src3, v_src4);
                v_store(dst + x, v_pack(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, ushort>
{
    int operator() (const float * src, ushort * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_round(v_load(src + x));
                v_int32x4 v_src2 = v_round(v_load(src + x + cWidth));
                v_store(dst + x, v_pack_u(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, short>
{
    int operator() (const float * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_round(v_load(src + x));
                v_int32x4 v_src2 = v_round(v_load(src + x + cWidth));
                v_store(dst + x, v_pack(v_src1, v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, int>
{
    int operator() (const float * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_round(v_load(src + x)));
        }
        return x;
    }
};
#if CV_SIMD128_64F
// from double

template <>
struct Cvt_SIMD<double, uchar>
{
    int operator() (const double * src, uchar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));
                v_float32x4 v_src2 = v_cvt_f32(v_load(src + x + cWidth * 2));
                v_float32x4 v_src3 = v_cvt_f32(v_load(src + x + cWidth * 3));

                v_src0 = v_combine_low(v_src0, v_src1);
                v_src1 = v_combine_low(v_src2, v_src3);

                v_int16x8 v_dst = v_pack(v_round(v_src0), v_round(v_src1));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, schar>
{
    int operator() (const double * src, schar * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));
                v_float32x4 v_src2 = v_cvt_f32(v_load(src + x + cWidth * 2));
                v_float32x4 v_src3 = v_cvt_f32(v_load(src + x + cWidth * 3));

                v_src0 = v_combine_low(v_src0, v_src1);
                v_src1 = v_combine_low(v_src2, v_src3);

                v_int16x8 v_dst = v_pack(v_round(v_src0), v_round(v_src1));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, ushort>
{
    int operator() (const double * src, ushort * dst, int width) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::Cvt_SIMD_f64u16_SSE41(src, dst, width);
#endif
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));
                v_float32x4 v_src2 = v_cvt_f32(v_load(src + x + cWidth * 2));
                v_float32x4 v_src3 = v_cvt_f32(v_load(src + x + cWidth * 3));

                v_src0 = v_combine_low(v_src0, v_src1);
                v_src1 = v_combine_low(v_src2, v_src3);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src0), v_round(v_src1));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, short>
{
    int operator() (const double * src, short * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));
                v_float32x4 v_src2 = v_cvt_f32(v_load(src + x + cWidth * 2));
                v_float32x4 v_src3 = v_cvt_f32(v_load(src + x + cWidth * 3));

                v_src0 = v_combine_low(v_src0, v_src1);
                v_src1 = v_combine_low(v_src2, v_src3);

                v_int16x8 v_dst = v_pack(v_round(v_src0), v_round(v_src1));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, int>
{
    int operator() (const double * src, int * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));

                v_store(dst + x, v_round(v_combine_low(v_src0, v_src1)));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<double, float>
{
    int operator() (const double * src, float * dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src0 = v_cvt_f32(v_load(src + x));
                v_float32x4 v_src1 = v_cvt_f32(v_load(src + x + cWidth));

                v_store(dst + x, v_combine_low(v_src0, v_src1));
            }
        }
        return x;
    }
};

// to double

template <>
struct Cvt_SIMD<uchar, double>
{
    int operator() (const uchar* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_uint16x8 v_src = v_load_expand(src + x);
                v_uint32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f64(v_reinterpret_as_s32(v_src1)));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_reinterpret_as_s32(v_src1)));
                v_store(dst + x + cWidth * 2, v_cvt_f64(v_reinterpret_as_s32(v_src2)));
                v_store(dst + x + cWidth * 3, v_cvt_f64_high(v_reinterpret_as_s32(v_src2)));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<schar, double>
{
    int operator() (const schar* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int16x8 v_src = v_load_expand(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_store(dst + x, v_cvt_f64(v_src1));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_src1));
                v_store(dst + x + cWidth * 2, v_cvt_f64(v_src2));
                v_store(dst + x + cWidth * 3, v_cvt_f64_high(v_src2));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, double>
{
    int operator() (const ushort* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_uint32x4 v_src = v_load_expand(src + x);

                v_store(dst + x, v_cvt_f64(v_reinterpret_as_s32(v_src)));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_reinterpret_as_s32(v_src)));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<short, double>
{
    int operator() (const short* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src = v_load_expand(src + x);

                v_store(dst + x, v_cvt_f64(v_src));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_src));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<int, double>
{
    int operator() (const int* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src = v_load(src + x);

                v_store(dst + x, v_cvt_f64(v_src));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_src));
            }
        }
        return x;
    }
};

template <>
struct Cvt_SIMD<float, double>
{
    int operator() (const float* src, double* dst, int width) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src = v_load(src + x);

                v_store(dst + x, v_cvt_f64(v_src));
                v_store(dst + x + cWidth, v_cvt_f64_high(v_src));
            }
        }
        return x;
    }
};
#endif // CV_SIMD128_64F
#endif // CV_SIMD128


#ifdef HAVE_OPENVX

template<typename T, typename DT>
static bool _openvx_cvt(const T* src, size_t sstep,
                        DT* dst, size_t dstep, Size continuousSize)
{
    using namespace ivx;

    if(!(continuousSize.width > 0 && continuousSize.height > 0))
    {
        return true;
    }

    //.height is for number of continuous pieces
    //.width  is for length of one piece
    Size imgSize = continuousSize;
    if(continuousSize.height == 1)
    {
        if(sstep / sizeof(T) == dstep / sizeof(DT) && sstep / sizeof(T) > 0 &&
           continuousSize.width % (sstep / sizeof(T)) == 0)
        {
            //continuous n-lines image
            imgSize.width  = sstep / sizeof(T);
            imgSize.height = continuousSize.width / (sstep / sizeof(T));
        }
        else
        {
            //1-row image with possibly incorrect step
            sstep = continuousSize.width * sizeof(T);
            dstep = continuousSize.width * sizeof(DT);
        }
    }

    int srcType = DataType<T>::type, dstType = DataType<DT>::type;

    if (ovx::skipSmallImages<VX_KERNEL_CONVERTDEPTH>(imgSize.width, imgSize.height))
        return false;

    try
    {
        Context context = ovx::getOpenVXContext();

        // Other conversions are marked as "experimental"
        if(context.vendorID() == VX_ID_KHRONOS &&
           !(srcType == CV_8U  && dstType == CV_16S) &&
           !(srcType == CV_16S && dstType == CV_8U))
        {
            return false;
        }

        Image srcImage = Image::createFromHandle(context, Image::matTypeToFormat(srcType),
                                                 Image::createAddressing(imgSize.width, imgSize.height,
                                                                         (vx_uint32)sizeof(T), (vx_uint32)sstep),
                                                 (void*)src);
        Image dstImage = Image::createFromHandle(context, Image::matTypeToFormat(dstType),
                                                 Image::createAddressing(imgSize.width, imgSize.height,
                                                                         (vx_uint32)sizeof(DT), (vx_uint32)dstep),
                                                 (void*)dst);

        IVX_CHECK_STATUS(vxuConvertDepth(context, srcImage, dstImage, VX_CONVERT_POLICY_SATURATE, 0));

#ifdef VX_VERSION_1_1
        //we should take user memory back before release
        //(it's not done automatically according to standard)
        srcImage.swapHandle(); dstImage.swapHandle();
#endif
    }
    catch (RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}

template<typename T, typename DT>
static bool openvx_cvt(const T* src, size_t sstep,
                       DT* dst, size_t dstep, Size size)
{
    (void)src; (void)sstep; (void)dst; (void)dstep; (void)size;
    return false;
}

#define DEFINE_OVX_CVT_SPECIALIZATION(T, DT) \
template<>                                                                    \
bool openvx_cvt(const T *src, size_t sstep, DT *dst, size_t dstep, Size size) \
{                                                                             \
    return _openvx_cvt<T, DT>(src, sstep, dst, dstep, size);                  \
}

DEFINE_OVX_CVT_SPECIALIZATION(uchar, ushort)
DEFINE_OVX_CVT_SPECIALIZATION(uchar, short)
DEFINE_OVX_CVT_SPECIALIZATION(uchar, int)
DEFINE_OVX_CVT_SPECIALIZATION(ushort, uchar)
DEFINE_OVX_CVT_SPECIALIZATION(ushort, int)
DEFINE_OVX_CVT_SPECIALIZATION(short, uchar)
DEFINE_OVX_CVT_SPECIALIZATION(short, int)
DEFINE_OVX_CVT_SPECIALIZATION(int, uchar)
DEFINE_OVX_CVT_SPECIALIZATION(int, ushort)
DEFINE_OVX_CVT_SPECIALIZATION(int, short)

#endif

template<typename T, typename DT> static void
cvt_( const T* src, size_t sstep,
      DT* dst, size_t dstep, Size size )
{
    CV_OVX_RUN(
        true,
        openvx_cvt(src, sstep, dst, dstep, size)
    )

    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    Cvt_SIMD<T, DT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]);
            t1 = saturate_cast<DT>(src[x+1]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]);
            t1 = saturate_cast<DT>(src[x+3]);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]);
    }
}

template<typename T> static void
cpy_( const T* src, size_t sstep, T* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
        memcpy(dst, src, size.width*sizeof(src[0]));
}


#if defined(HAVE_IPP)
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    CV_IPP_RUN(src && dst, CV_INSTRUMENT_FUN_IPP(ippiConvert_##ippFavor, src, (int)sstep, dst, (int)dstep, ippiSize(size.width, size.height)) >= 0) \
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CVT_FUNC_F2(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    CV_IPP_RUN(src && dst, CV_INSTRUMENT_FUN_IPP(ippiConvert_##ippFavor, src, (int)sstep, dst, (int)dstep, ippiSize(size.width, size.height), ippRndFinancial, 0) >= 0) \
    cvt_(src, sstep, dst, dstep, size); \
}
#else
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}
#define DEF_CVT_FUNC_F2 DEF_CVT_FUNC_F
#endif

#define DEF_CVT_FUNC(suffix, stype, dtype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CPY_FUNC(suffix, stype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         stype* dst, size_t dstep, Size size, double*) \
{ \
    cpy_(src, sstep, dst, dstep, size); \
}


DEF_CPY_FUNC(8u,     uchar)
DEF_CVT_FUNC_F(8s8u,   schar, uchar, 8s8u_C1Rs)
DEF_CVT_FUNC_F(16u8u,  ushort, uchar, 16u8u_C1R)
DEF_CVT_FUNC_F(16s8u,  short, uchar, 16s8u_C1R)
DEF_CVT_FUNC_F(32s8u,  int, uchar, 32s8u_C1R)
DEF_CVT_FUNC_F2(32f8u,  float, uchar, 32f8u_C1RSfs)
DEF_CVT_FUNC(64f8u,  double, uchar)

DEF_CVT_FUNC_F2(8u8s,   uchar, schar, 8u8s_C1RSfs)
DEF_CVT_FUNC_F2(16u8s,  ushort, schar, 16u8s_C1RSfs)
DEF_CVT_FUNC_F2(16s8s,  short, schar, 16s8s_C1RSfs)
DEF_CVT_FUNC_F(32s8s,  int, schar, 32s8s_C1R)
DEF_CVT_FUNC_F2(32f8s,  float, schar, 32f8s_C1RSfs)
DEF_CVT_FUNC(64f8s,  double, schar)

DEF_CVT_FUNC_F(8u16u,  uchar, ushort, 8u16u_C1R)
DEF_CVT_FUNC_F(8s16u,  schar, ushort, 8s16u_C1Rs)
DEF_CPY_FUNC(16u,    ushort)
DEF_CVT_FUNC_F(16s16u, short, ushort, 16s16u_C1Rs)
DEF_CVT_FUNC_F2(32s16u, int, ushort, 32s16u_C1RSfs)
DEF_CVT_FUNC_F2(32f16u, float, ushort, 32f16u_C1RSfs)
DEF_CVT_FUNC(64f16u, double, ushort)

DEF_CVT_FUNC_F(8u16s,  uchar, short, 8u16s_C1R)
DEF_CVT_FUNC_F(8s16s,  schar, short, 8s16s_C1R)
DEF_CVT_FUNC_F2(16u16s, ushort, short, 16u16s_C1RSfs)
DEF_CVT_FUNC_F2(32s16s, int, short, 32s16s_C1RSfs)
DEF_CVT_FUNC(32f16s, float, short)
DEF_CVT_FUNC(64f16s, double, short)

DEF_CVT_FUNC_F(8u32s,  uchar, int, 8u32s_C1R)
DEF_CVT_FUNC_F(8s32s,  schar, int, 8s32s_C1R)
DEF_CVT_FUNC_F(16u32s, ushort, int, 16u32s_C1R)
DEF_CVT_FUNC_F(16s32s, short, int, 16s32s_C1R)
DEF_CPY_FUNC(32s,    int)
DEF_CVT_FUNC_F2(32f32s, float, int, 32f32s_C1RSfs)
DEF_CVT_FUNC(64f32s, double, int)

DEF_CVT_FUNC_F(8u32f,  uchar, float, 8u32f_C1R)
DEF_CVT_FUNC_F(8s32f,  schar, float, 8s32f_C1R)
DEF_CVT_FUNC_F(16u32f, ushort, float, 16u32f_C1R)
DEF_CVT_FUNC_F(16s32f, short, float, 16s32f_C1R)
DEF_CVT_FUNC_F(32s32f, int, float, 32s32f_C1R)
DEF_CVT_FUNC(64f32f, double, float)

DEF_CVT_FUNC(8u64f,  uchar, double)
DEF_CVT_FUNC(8s64f,  schar, double)
DEF_CVT_FUNC(16u64f, ushort, double)
DEF_CVT_FUNC(16s64f, short, double)
DEF_CVT_FUNC(32s64f, int, double)
DEF_CVT_FUNC(32f64f, float, double)
DEF_CPY_FUNC(64s,    int64)


BinaryFunc getConvertFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtTab[][8] =
    {
        {
            (BinaryFunc)(cvt8u), (BinaryFunc)GET_OPTIMIZED(cvt8s8u), (BinaryFunc)GET_OPTIMIZED(cvt16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8u), (BinaryFunc)GET_OPTIMIZED(cvt32s8u), (BinaryFunc)GET_OPTIMIZED(cvt32f8u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u8s), (BinaryFunc)cvt8u, (BinaryFunc)GET_OPTIMIZED(cvt16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8s), (BinaryFunc)GET_OPTIMIZED(cvt32s8s), (BinaryFunc)GET_OPTIMIZED(cvt32f8s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16u), (BinaryFunc)GET_OPTIMIZED(cvt8s16u), (BinaryFunc)cvt16u,
            (BinaryFunc)GET_OPTIMIZED(cvt16s16u), (BinaryFunc)GET_OPTIMIZED(cvt32s16u), (BinaryFunc)GET_OPTIMIZED(cvt32f16u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16s), (BinaryFunc)GET_OPTIMIZED(cvt8s16s), (BinaryFunc)GET_OPTIMIZED(cvt16u16s),
            (BinaryFunc)cvt16u, (BinaryFunc)GET_OPTIMIZED(cvt32s16s), (BinaryFunc)GET_OPTIMIZED(cvt32f16s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32s), (BinaryFunc)GET_OPTIMIZED(cvt8s32s), (BinaryFunc)GET_OPTIMIZED(cvt16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32s), (BinaryFunc)cvt32s, (BinaryFunc)GET_OPTIMIZED(cvt32f32s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f32s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32f), (BinaryFunc)GET_OPTIMIZED(cvt8s32f), (BinaryFunc)GET_OPTIMIZED(cvt16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32f), (BinaryFunc)GET_OPTIMIZED(cvt32s32f), (BinaryFunc)cvt32s,
            (BinaryFunc)GET_OPTIMIZED(cvt64f32f), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u64f), (BinaryFunc)GET_OPTIMIZED(cvt8s64f), (BinaryFunc)GET_OPTIMIZED(cvt16u64f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s64f), (BinaryFunc)GET_OPTIMIZED(cvt32s64f), (BinaryFunc)GET_OPTIMIZED(cvt32f64f),
            (BinaryFunc)(cvt64s), 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

} // cv::

#ifdef HAVE_IPP
namespace cv
{
static bool ipp_convertTo(Mat &src, Mat &dst, double alpha, double beta)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    IppDataType srcDepth = ippiGetDataType(src.depth());
    IppDataType dstDepth = ippiGetDataType(dst.depth());
    int         channels = src.channels();

    if(src.dims == 0)
        return false;

    ::ipp::IwiImage iwSrc;
    ::ipp::IwiImage iwDst;

    try
    {
        IppHintAlgorithm mode = ippAlgHintFast;
        if(dstDepth == ipp64f ||
            (dstDepth == ipp32f && (srcDepth == ipp32s || srcDepth == ipp64f)) ||
            (dstDepth == ipp32s && (srcDepth == ipp32s || srcDepth == ipp64f)))
            mode = ippAlgHintAccurate;

        if(src.dims <= 2)
        {
            Size sz = getContinuousSize(src, dst, channels);

            iwSrc.Init(ippiSize(sz), srcDepth, 1, NULL, (void*)src.ptr(), src.step);
            iwDst.Init(ippiSize(sz), dstDepth, 1, NULL, (void*)dst.ptr(), dst.step);

            CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwSrc, iwDst, alpha, beta, ::ipp::IwiScaleParams(mode));
        }
        else
        {
            const Mat *arrays[] = {&src, &dst, NULL};
            uchar     *ptrs[2]  = {NULL};
            NAryMatIterator it(arrays, ptrs);

            iwSrc.Init(ippiSize(it.size, 1), srcDepth, channels);
            iwDst.Init(ippiSize(it.size, 1), dstDepth, channels);

            for(size_t i = 0; i < it.nplanes; i++, ++it)
            {
                iwSrc.m_ptr  = ptrs[0];
                iwDst.m_ptr  = ptrs[1];

                CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwSrc, iwDst, alpha, beta, ::ipp::IwiScaleParams(mode));
            }
        }
    }
    catch (::ipp::IwException)
    {
        return false;
    }
    return true;
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(alpha); CV_UNUSED(beta);
    return false;
#endif
}
} // cv::
#endif


void cv::Mat::convertTo(OutputArray _dst, int _type, double alpha, double beta) const
{
    CV_INSTRUMENT_REGION()

    bool noScale = fabs(alpha-1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;

    if( _type < 0 )
        _type = _dst.fixedType() ? _dst.type() : type();
    else
        _type = CV_MAKETYPE(CV_MAT_DEPTH(_type), channels());

    int sdepth = depth(), ddepth = CV_MAT_DEPTH(_type);
    if( sdepth == ddepth && noScale )
    {
        copyTo(_dst);
        return;
    }

    Mat src = *this;
    if( dims <= 2 )
        _dst.create( size(), _type );
    else
        _dst.create( dims, size, _type );
    Mat dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_convertTo(src, dst, alpha, beta ));

    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);
    double scale[] = {alpha, beta};
    int cn = channels();
    CV_Assert( func != 0 );

    if( dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);

        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, scale);
    }
}

//==================================================================================================

namespace cv {

// template for FP16 HW conversion function
template<typename T, typename DT> static void
cvtScaleHalf_( const T* src, size_t sstep, DT* dst, size_t dstep, Size size);

template<> void
cvtScaleHalf_<float, short>( const float* src, size_t sstep, short* dst, size_t dstep, Size size )
{
    CV_CPU_CALL_FP16_(cvtScaleHalf_SIMD32f16f, (src, sstep, dst, dstep, size));

#if !defined(CV_CPU_COMPILE_FP16)
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        for ( int x = 0; x < size.width; x++ )
        {
            dst[x] = convertFp16SW(src[x]);
        }
    }
#endif
}

template<> void
cvtScaleHalf_<short, float>( const short* src, size_t sstep, float* dst, size_t dstep, Size size )
{
    CV_CPU_CALL_FP16_(cvtScaleHalf_SIMD16f32f, (src, sstep, dst, dstep, size));

#if !defined(CV_CPU_COMPILE_FP16)
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        for ( int x = 0; x < size.width; x++ )
        {
            dst[x] = convertFp16SW(src[x]);
        }
    }
#endif
}

#define DEF_CVT_SCALE_FP16_FUNC(suffix, stype, dtype) \
static void cvtScaleHalf##suffix( const stype* src, size_t sstep, \
dtype* dst, size_t dstep, Size size) \
{ \
    cvtScaleHalf_<stype,dtype>(src, sstep, dst, dstep, size); \
}

DEF_CVT_SCALE_FP16_FUNC(32f16f, float, short)
DEF_CVT_SCALE_FP16_FUNC(16f32f, short, float)

static UnaryFunc getConvertFuncFp16(int ddepth)
{
    static UnaryFunc cvtTab[] =
    {
        0, 0, 0,
        (UnaryFunc)(cvtScaleHalf32f16f), 0, (UnaryFunc)(cvtScaleHalf16f32f),
        0, 0,
    };
    return cvtTab[CV_MAT_DEPTH(ddepth)];
}


#ifdef HAVE_OPENCL

static bool ocl_convertFp16( InputArray _src, OutputArray _dst, int ddepth )
{
    int type = _src.type(), cn = CV_MAT_CN(type);

    _dst.createSameSize( _src, CV_MAKETYPE(ddepth, cn) );
    int kercn = 1;
    int rowsPerWI = 1;
    String build_opt = format("-D HALF_SUPPORT -D dstT=%s -D srcT=%s -D rowsPerWI=%d%s",
                           ddepth == CV_16S ? "half" : "float",
                           ddepth == CV_16S ? "float" : "half",
                           rowsPerWI,
                           ddepth == CV_16S ? " -D FLOAT_TO_HALF " : "");
    ocl::Kernel k("convertFp16", ocl::core::halfconvert_oclsrc, build_opt);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    UMat dst = _dst.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    k.args(srcarg, dstarg);

    size_t globalsize[2] = { (size_t)src.cols * cn / kercn, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

} //cv::

void cv::convertFp16( InputArray _src, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    int ddepth = 0;
    switch( _src.depth() )
    {
    case CV_32F:
        ddepth = CV_16S;
        break;
    case CV_16S:
        ddepth = CV_32F;
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported input depth");
        return;
    }

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_convertFp16(_src, _dst, ddepth))

    Mat src = _src.getMat();

    int type = CV_MAKETYPE(ddepth, src.channels());
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();
    UnaryFunc func = getConvertFuncFp16(ddepth);
    int cn = src.channels();
    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, dst.data, dst.step, sz, 0);
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, ptrs[1], 1, sz, 0);
    }
}
