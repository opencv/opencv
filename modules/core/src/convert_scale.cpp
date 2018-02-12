// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "convert.hpp"

/****************************************************************************************\
*                                convertScale[Abs]                                       *
\****************************************************************************************/

namespace cv
{

template<typename T, typename DT, typename WT>
struct cvtScaleAbs_SIMD
{
    int operator () (const T *, DT *, int, WT, WT) const
    {
        return 0;
    }
};

#if CV_SIMD128

static inline void v_load_expand_from_u8_f32(const uchar* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    v_uint32x4 v_src0, v_src1;
    v_expand(v_load_expand(src), v_src0, v_src1);

    a = v_shift + v_scale * v_cvt_f32(v_reinterpret_as_s32(v_src0));
    b = v_shift + v_scale * v_cvt_f32(v_reinterpret_as_s32(v_src1));
}

static inline void v_load_expand_from_s8_f32(const schar* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    v_int32x4 v_src0, v_src1;
    v_expand(v_load_expand(src), v_src0, v_src1);

    a = v_shift + v_scale * v_cvt_f32(v_src0);
    b = v_shift + v_scale * v_cvt_f32(v_src1);
}

static inline void v_load_expand_from_u16_f32(const ushort* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    v_uint32x4 v_src0, v_src1;
    v_expand(v_load(src), v_src0, v_src1);

    a = v_shift + v_scale * v_cvt_f32(v_reinterpret_as_s32(v_src0));
    b = v_shift + v_scale * v_cvt_f32(v_reinterpret_as_s32(v_src1));
}

static inline void v_load_expand_from_s16_f32(const short* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    v_int32x4 v_src0, v_src1;
    v_expand(v_load(src), v_src0, v_src1);

    a = v_shift + v_scale * v_cvt_f32(v_src0);
    b = v_shift + v_scale * v_cvt_f32(v_src1);
}

static inline void v_load_expand_from_s32_f32(const int* src, const v_float32x4 &v_scale, const v_float32x4 &v_shift, v_float32x4 &a, v_float32x4 &b)
{
    a = v_shift + v_scale * v_cvt_f32(v_load(src));
    b = v_shift + v_scale * v_cvt_f32(v_load(src + v_int32x4::nlanes));
}

template <>
struct cvtScaleAbs_SIMD<uchar, uchar, float>
{
    int operator () (const uchar * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            const int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst_0, v_dst_1, v_dst_2, v_dst_3;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_dst_0, v_dst_1);
                v_load_expand_from_u8_f32(src + x + cWidth, v_scale, v_shift, v_dst_2, v_dst_3);
                v_dst_0 = v_abs(v_dst_0);
                v_dst_1 = v_abs(v_dst_1);
                v_dst_2 = v_abs(v_dst_2);
                v_dst_3 = v_abs(v_dst_3);

                v_int16x8 v_dsti_0 = v_pack(v_round(v_dst_0), v_round(v_dst_1));
                v_int16x8 v_dsti_1 = v_pack(v_round(v_dst_2), v_round(v_dst_3));
                v_store(dst + x, v_pack_u(v_dsti_0, v_dsti_1));
            }
        }
        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<schar, uchar, float>
{
    int operator () (const schar * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            const int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth*2; x += cWidth*2)
            {
                v_float32x4 v_dst_0, v_dst_1, v_dst_2, v_dst_3;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_dst_0, v_dst_1);
                v_load_expand_from_s8_f32(src + x + cWidth, v_scale, v_shift, v_dst_2, v_dst_3);
                v_dst_0 = v_abs(v_dst_0);
                v_dst_1 = v_abs(v_dst_1);
                v_dst_2 = v_abs(v_dst_2);
                v_dst_3 = v_abs(v_dst_3);

                v_uint16x8 v_dsti_0 = v_pack_u(v_round(v_dst_0), v_round(v_dst_1));
                v_uint16x8 v_dsti_1 = v_pack_u(v_round(v_dst_2), v_round(v_dst_3));
                v_store(dst + x, v_pack(v_dsti_0, v_dsti_1));
            }
        }
        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<ushort, uchar, float>
{
    int operator () (const ushort * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            const int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_dst0, v_dst1;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_dst0, v_dst1);
                v_dst0 = v_abs(v_dst0);
                v_dst1 = v_abs(v_dst1);

                v_int16x8 v_dst = v_pack(v_round(v_dst0), v_round(v_dst1));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<short, uchar, float>
{
    int operator () (const short * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            const int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_dst0, v_dst1;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_dst0, v_dst1);
                v_dst0 = v_abs(v_dst0);
                v_dst1 = v_abs(v_dst1);

                v_int16x8 v_dst = v_pack(v_round(v_dst0), v_round(v_dst1));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<int, uchar, float>
{
    int operator () (const int * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        v_float32x4 v_shift = v_setall_f32(shift);
        v_float32x4 v_scale = v_setall_f32(scale);
        const int cWidth = v_int32x4::nlanes;
        for (; x <= width - cWidth * 2; x += cWidth * 2)
        {
            v_float32x4 v_dst_0 = v_cvt_f32(v_load(src + x)) * v_scale;
            v_dst_0 = v_abs(v_dst_0 + v_shift);

            v_float32x4 v_dst_1 = v_cvt_f32(v_load(src + x + cWidth)) * v_scale;
            v_dst_1 = v_abs(v_dst_1 + v_shift);

            v_int16x8 v_dst = v_pack(v_round(v_dst_0), v_round(v_dst_1));
            v_pack_u_store(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<float, uchar, float>
{
    int operator () (const float * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;
        v_float32x4 v_shift = v_setall_f32(shift);
        v_float32x4 v_scale = v_setall_f32(scale);
        int cWidth = v_float32x4::nlanes;
        for (; x <= width - cWidth * 2; x += cWidth * 2)
        {
            v_float32x4 v_dst_0 = v_load(src + x) * v_scale;
            v_dst_0 = v_abs(v_dst_0 + v_shift);

            v_float32x4 v_dst_1 = v_load(src + x + cWidth) * v_scale;
            v_dst_1 = v_abs(v_dst_1 + v_shift);

            v_int16x8 v_dst = v_pack(v_round(v_dst_0), v_round(v_dst_1));
            v_pack_u_store(dst + x, v_dst);
        }
        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct cvtScaleAbs_SIMD<double, uchar, float>
{
    int operator () (const double * src, uchar * dst, int width,
        float scale, float shift) const
    {
        int x = 0;

        if (hasSIMD128())
        {
            v_float32x4 v_scale = v_setall_f32(scale);
            v_float32x4 v_shift = v_setall_f32(shift);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_src1, v_src2, v_dummy;
                v_recombine(v_cvt_f32(v_load(src + x)), v_cvt_f32(v_load(src + x + cWidth)), v_src1, v_dummy);
                v_recombine(v_cvt_f32(v_load(src + x + cWidth * 2)), v_cvt_f32(v_load(src + x + cWidth * 3)), v_src2, v_dummy);

                v_float32x4 v_dst1 = v_abs((v_src1 * v_scale) + v_shift);
                v_float32x4 v_dst2 = v_abs((v_src2 * v_scale) + v_shift);

                v_int16x8 v_dst_i = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_pack_u_store(dst + x, v_dst_i);
            }
        }

        return x;
    }
};
#endif // CV_SIMD128_64F

#endif

template<typename T, typename DT, typename WT> static void
cvtScaleAbs_( const T* src, size_t sstep,
              DT* dst, size_t dstep, Size size,
              WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    cvtScaleAbs_SIMD<T, DT, WT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width, scale, shift);

        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(std::abs(src[x]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+1]*scale + shift));
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(std::abs(src[x+2]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+3]*scale + shift));
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(std::abs(src[x]*scale + shift));
    }
}

template <typename T, typename DT, typename WT>
struct cvtScale_SIMD
{
    int operator () (const T *, DT *, int, WT, WT) const
    {
        return 0;
    }
};

#if CV_SIMD128

// from uchar

template <>
struct cvtScale_SIMD<uchar, uchar, float>
{
    int operator () (const uchar * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, schar, float>
{
    int operator () (const uchar * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, ushort, float>
{
    int operator () (const uchar * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_u8u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, short, float>
{
    int operator () (const uchar * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, int, float>
{
    int operator () (const uchar * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_round(v_src1));
                v_store(dst + x + cWidth, v_round(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, float, float>
{
    int operator () (const uchar * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

// from schar

template <>
struct cvtScale_SIMD<schar, uchar, float>
{
    int operator () (const schar * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, schar, float>
{
    int operator () (const schar * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, ushort, float>
{
    int operator () (const schar * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_s8u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, short, float>
{
    int operator () (const schar * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, int, float>
{
    int operator () (const schar * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_round(v_src1));
                v_store(dst + x + cWidth, v_round(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, float, float>
{
    int operator () (const schar * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s8_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

// from ushort

template <>
struct cvtScale_SIMD<ushort, uchar, float>
{
    int operator () (const ushort * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, schar, float>
{
    int operator () (const ushort * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, ushort, float>
{
    int operator () (const ushort * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_u16u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, short, float>
{
    int operator () (const ushort * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, int, float>
{
    int operator () (const ushort * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_round(v_src1));
                v_store(dst + x + cWidth, v_round(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, float, float>
{
    int operator () (const ushort * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_u16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

// from short

template <>
struct cvtScale_SIMD<short, uchar, float>
{
    int operator () (const short * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, schar, float>
{
    int operator () (const short * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, ushort, float>
{
    int operator () (const short * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_s16u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, short, float>
{
    int operator () (const short * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, float, float>
{
    int operator () (const short * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s16_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};

// from int

template <>
struct cvtScale_SIMD<int, uchar, float>
{
    int operator () (const int * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s32_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, schar, float>
{
    int operator () (const int * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s32_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, ushort, float>
{
    int operator () (const int * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_s32u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s32_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_uint16x8 v_dst = v_pack_u(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, short, float>
{
    int operator () (const int * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_src1, v_src2;
                v_load_expand_from_s32_f32(src + x, v_scale, v_shift, v_src1, v_src2);

                v_int16x8 v_dst = v_pack(v_round(v_src1), v_round(v_src2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct cvtScale_SIMD<int, int, double>
{
    int operator () (const int * src, int * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                double v_srcbuf[] = { (double)src[x], (double)src[x+1], (double)src[x+2], (double)src[x+3] };
                v_float64x2 v_src1 = v_shift + v_scale * v_load(v_srcbuf);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(v_srcbuf + 2);
                v_store(dst + x, v_combine_low(v_round(v_src1), v_round(v_src2)));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, float, double>
{
    int operator () (const int * src, float * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                double v_srcbuf[] = { (double)src[x], (double)src[x+1], (double)src[x+2], (double)src[x+3] };
                v_float64x2 v_src1 = v_shift + v_scale * v_load(v_srcbuf);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(v_srcbuf + 2);
                v_store(dst + x, v_combine_low(v_cvt_f32(v_src1), v_cvt_f32(v_src2)));
            }
        }
        return x;
    }
};
#endif //CV_SIMD128_64F

// from float

template <>
struct cvtScale_SIMD<float, uchar, float>
{
    int operator () (const float * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst1 = v_shift + v_scale * v_load(src + x);
                v_float32x4 v_dst2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_pack_u_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, schar, float>
{
    int operator () (const float * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst1 = v_shift + v_scale * v_load(src + x);
                v_float32x4 v_dst2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_store_low(dst + x, v_pack(v_dst, v_dst));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, ushort, float>
{
    int operator () (const float * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_f32u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst1 = v_shift + v_scale * v_load(src + x);
                v_float32x4 v_dst2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_uint16x8 v_dst = v_pack_u(v_round(v_dst1), v_round(v_dst2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, short, float>
{
    int operator () (const float * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_dst1 = v_shift + v_scale * v_load(src + x);
                v_float32x4 v_dst2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, int, float>
{
    int operator () (const float * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_round(v_load(src + x) * v_scale + v_shift));
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, float, float>
{
    int operator () (const float * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift), v_scale = v_setall_f32(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth; x += cWidth)
                v_store(dst + x, v_load(src + x) * v_scale + v_shift);
        }
        return x;
    }
};

#if CV_SIMD128_64F

static inline void v_load_scale_shift(const double* src, const v_float64x2& v_scale, const v_float64x2 &v_shift, v_float32x4& v_dst1, v_float32x4 &v_dst2)
{
    int cWidth = v_float64x2::nlanes;
    v_float64x2 v_src1 = v_shift + v_scale * v_load(src);
    v_float64x2 v_src2 = v_shift + v_scale * v_load(src + cWidth);
    v_float64x2 v_src3 = v_shift + v_scale * v_load(src + cWidth * 2);
    v_float64x2 v_src4 = v_shift + v_scale * v_load(src + cWidth * 3);
    v_dst1 = v_combine_low(v_cvt_f32(v_src1), v_cvt_f32(v_src2));
    v_dst2 = v_combine_low(v_cvt_f32(v_src3), v_cvt_f32(v_src4));
}

static inline void v_store_scale_shift_s32_to_f64(double *dst, const v_float64x2 &v_scale, const v_float64x2 &v_shift, const v_int32x4 &v1, const v_int32x4 &v2)
{
    v_float64x2 v_dst1 = v_shift + v_scale * v_cvt_f64(v1);
    v_float64x2 v_dst2 = v_shift + v_scale * v_cvt_f64_high(v1);
    v_float64x2 v_dst3 = v_shift + v_scale * v_cvt_f64(v2);
    v_float64x2 v_dst4 = v_shift + v_scale * v_cvt_f64_high(v2);

    v_store(dst, v_dst1);
    v_store(dst + v_float64x2::nlanes, v_dst2);
    v_store(dst + v_float64x2::nlanes * 2, v_dst3);
    v_store(dst + v_float64x2::nlanes * 3, v_dst4);
}

static inline void v_store_scale_shift_f32_to_f64(double *dst, const v_float64x2 &v_scale, const v_float64x2 &v_shift, const v_float32x4 &v1, const v_float32x4 &v2)
{
    v_float64x2 v_dst1 = v_shift + v_scale * v_cvt_f64(v1);
    v_float64x2 v_dst2 = v_shift + v_scale * v_cvt_f64_high(v1);
    v_float64x2 v_dst3 = v_shift + v_scale * v_cvt_f64(v2);
    v_float64x2 v_dst4 = v_shift + v_scale * v_cvt_f64_high(v2);

    v_store(dst, v_dst1);
    v_store(dst + v_float64x2::nlanes, v_dst2);
    v_store(dst + v_float64x2::nlanes * 2, v_dst3);
    v_store(dst + v_float64x2::nlanes * 3, v_dst4);
}

// from double

template <>
struct cvtScale_SIMD<double, uchar, float>
{
    int operator () (const double * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64((double)shift), v_scale = v_setall_f64((double)scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_dst1, v_dst2;
                v_load_scale_shift(src + x, v_scale, v_shift, v_dst1, v_dst2);
                v_pack_u_store(dst + x, v_pack(v_round(v_dst1), v_round(v_dst2)));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, schar, float>
{
    int operator () (const double * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64((double)shift), v_scale = v_setall_f64((double)scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_float32x4 v_dst1, v_dst2;
                v_load_scale_shift(src + x, v_scale, v_shift, v_dst1, v_dst2);
                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_pack_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, ushort, float>
{
    int operator () (const double * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
#if CV_TRY_SSE4_1
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::cvtScale_SIMD_f64u16f32_SSE41(src, dst, width, scale, shift);
#endif
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64((double)shift), v_scale = v_setall_f64((double)scale);
            int cWidth = v_uint16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_dst1, v_dst2;
                v_load_scale_shift(src + x, v_scale, v_shift, v_dst1, v_dst2);
                v_uint16x8 v_dst = v_pack_u(v_round(v_dst1), v_round(v_dst2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, short, float>
{
    int operator () (const double * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64((double)shift), v_scale = v_setall_f64((double)scale);
            int cWidth = v_int16x8::nlanes;
            for (; x <= width - cWidth; x += cWidth)
            {
                v_float32x4 v_dst1, v_dst2;
                v_load_scale_shift(src + x, v_scale, v_shift, v_dst1, v_dst2);
                v_int16x8 v_dst = v_pack(v_round(v_dst1), v_round(v_dst2));
                v_store(dst + x, v_dst);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, int, double>
{
    int operator () (const double * src, int * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float64x2 v_src1 = v_shift + v_scale * v_load(src + x);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(src + x + cWidth);

                v_store(dst + x, v_combine_low(v_round(v_src1), v_round(v_src2)));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, float, double>
{
    int operator () (const double * src, float * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float64x2 v_src1 = v_shift + v_scale * v_load(src + x);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(src + x + cWidth);
                v_float32x4 v_dst1 = v_cvt_f32(v_src1);
                v_float32x4 v_dst2 = v_cvt_f32(v_src2);

                v_store(dst + x, v_combine_low(v_dst1, v_dst2));
            }
        }
        return x;
    }
};

// to double

template <>
struct cvtScale_SIMD<uchar, double, double>
{
    int operator () (const uchar * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_uint32x4 v_src1, v_src2;
                v_expand(v_load_expand(src + x), v_src1, v_src2);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift
                    , v_reinterpret_as_s32(v_src1), v_reinterpret_as_s32(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, double, double>
{
    int operator () (const schar * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1, v_src2;
                v_expand(v_load_expand(src + x), v_src1, v_src2);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift, v_src1, v_src2);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, double, double>
{
    int operator () (const ushort * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_uint32x4 v_src1, v_src2;
                v_expand(v_load(src + x), v_src1, v_src2);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift
                    , v_reinterpret_as_s32(v_src1), v_reinterpret_as_s32(v_src2));
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<short, double, double>
{
    int operator () (const short * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 4; x += cWidth * 4)
            {
                v_int32x4 v_src1, v_src2;
                v_expand(v_load(src + x), v_src1, v_src2);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift, v_src1, v_src2);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<int, double, double>
{
    int operator () (const int * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_int32x4 v_src1 = v_load(src + x);
                v_int32x4 v_src2 = v_load(src + x + cWidth);
                v_store_scale_shift_s32_to_f64(dst + x, v_scale, v_shift, v_src1, v_src2);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<float, double, double>
{
    int operator () (const float * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float32x4::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float32x4 v_src1 = v_load(src + x);
                v_float32x4 v_src2 = v_load(src + x + cWidth);
                v_store_scale_shift_f32_to_f64(dst + x, v_scale, v_shift, v_src1, v_src2);
            }
        }
        return x;
    }
};

template <>
struct cvtScale_SIMD<double, double, double>
{
    int operator () (const double * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;
        if (hasSIMD128())
        {
            v_float64x2 v_shift = v_setall_f64(shift), v_scale = v_setall_f64(scale);
            int cWidth = v_float64x2::nlanes;
            for (; x <= width - cWidth * 2; x += cWidth * 2)
            {
                v_float64x2 v_src1 = v_shift + v_scale * v_load(src + x);
                v_float64x2 v_src2 = v_shift + v_scale * v_load(src + x + cWidth);
                v_store(dst + x, v_src1);
                v_store(dst + x + cWidth, v_src2);
            }
        }
        return x;
    }
};
#endif
#endif

template<typename T, typename DT, typename WT> static void
cvtScale_( const T* src, size_t sstep,
           DT* dst, size_t dstep, Size size,
           WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    cvtScale_SIMD<T, DT, WT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width, scale, shift);

        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]*scale + shift);
            t1 = saturate_cast<DT>(src[x+1]*scale + shift);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]*scale + shift);
            t1 = saturate_cast<DT>(src[x+3]*scale + shift);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif

        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]*scale + shift);
    }
}

template<> void
cvtScale_<short, int, float>( const short* src, size_t sstep,
           int* dst, size_t dstep, Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if CV_TRY_AVX2
        if (CV_CPU_HAS_SUPPORT_AVX2)
        {
            opt_AVX2::cvtScale_s16s32f32Line_AVX2(src, dst, scale, shift, size.width);
            continue;
        }
        #endif
        #if CV_SIMD128
        if (hasSIMD128())
        {
            v_float32x4 v_shift = v_setall_f32(shift);
            v_float32x4 v_scale = v_setall_f32(scale);
            int cWidth = v_int32x4::nlanes;
            for (; x <= size.width - cWidth * 2; x += cWidth * 2)
            {
                v_int16x8 v_src = v_load(src + x);
                v_int32x4 v_src1, v_src2;
                v_expand(v_src, v_src1, v_src2);
                v_float32x4 v_tmp1 = v_cvt_f32(v_src1);
                v_float32x4 v_tmp2 = v_cvt_f32(v_src2);

                v_tmp1 = v_tmp1 * v_scale + v_shift;
                v_tmp2 = v_tmp2 * v_scale + v_shift;

                v_store(dst + x, v_round(v_tmp1));
                v_store(dst + x + cWidth, v_round(v_tmp2));
            }
        }
        #endif

        for(; x < size.width; x++ )
            dst[x] = saturate_cast<int>(src[x]*scale + shift);
    }
}


//==================================================================================================

#define DEF_CVT_SCALE_ABS_FUNC(suffix, tfunc, stype, dtype, wtype) \
static void cvtScaleAbs##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    tfunc(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}


#define DEF_CVT_SCALE_FUNC(suffix, stype, dtype, wtype) \
static void cvtScale##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    cvtScale_(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

DEF_CVT_SCALE_ABS_FUNC(8u, cvtScaleAbs_, uchar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(8s8u, cvtScaleAbs_, schar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtScaleAbs_, ushort, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtScaleAbs_, short, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtScaleAbs_, int, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtScaleAbs_, float, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtScaleAbs_, double, uchar, float)


DEF_CVT_SCALE_FUNC(8u,     uchar, uchar, float)
DEF_CVT_SCALE_FUNC(8s8u,   schar, uchar, float)
DEF_CVT_SCALE_FUNC(16u8u,  ushort, uchar, float)
DEF_CVT_SCALE_FUNC(16s8u,  short, uchar, float)
DEF_CVT_SCALE_FUNC(32s8u,  int, uchar, float)
DEF_CVT_SCALE_FUNC(32f8u,  float, uchar, float)
DEF_CVT_SCALE_FUNC(64f8u,  double, uchar, float)

DEF_CVT_SCALE_FUNC(8u8s,   uchar, schar, float)
DEF_CVT_SCALE_FUNC(8s,     schar, schar, float)
DEF_CVT_SCALE_FUNC(16u8s,  ushort, schar, float)
DEF_CVT_SCALE_FUNC(16s8s,  short, schar, float)
DEF_CVT_SCALE_FUNC(32s8s,  int, schar, float)
DEF_CVT_SCALE_FUNC(32f8s,  float, schar, float)
DEF_CVT_SCALE_FUNC(64f8s,  double, schar, float)

DEF_CVT_SCALE_FUNC(8u16u,  uchar, ushort, float)
DEF_CVT_SCALE_FUNC(8s16u,  schar, ushort, float)
DEF_CVT_SCALE_FUNC(16u,    ushort, ushort, float)
DEF_CVT_SCALE_FUNC(16s16u, short, ushort, float)
DEF_CVT_SCALE_FUNC(32s16u, int, ushort, float)
DEF_CVT_SCALE_FUNC(32f16u, float, ushort, float)
DEF_CVT_SCALE_FUNC(64f16u, double, ushort, float)

DEF_CVT_SCALE_FUNC(8u16s,  uchar, short, float)
DEF_CVT_SCALE_FUNC(8s16s,  schar, short, float)
DEF_CVT_SCALE_FUNC(16u16s, ushort, short, float)
DEF_CVT_SCALE_FUNC(16s,    short, short, float)
DEF_CVT_SCALE_FUNC(32s16s, int, short, float)
DEF_CVT_SCALE_FUNC(32f16s, float, short, float)
DEF_CVT_SCALE_FUNC(64f16s, double, short, float)

DEF_CVT_SCALE_FUNC(8u32s,  uchar, int, float)
DEF_CVT_SCALE_FUNC(8s32s,  schar, int, float)
DEF_CVT_SCALE_FUNC(16u32s, ushort, int, float)
DEF_CVT_SCALE_FUNC(16s32s, short, int, float)
DEF_CVT_SCALE_FUNC(32s,    int, int, double)
DEF_CVT_SCALE_FUNC(32f32s, float, int, float)
DEF_CVT_SCALE_FUNC(64f32s, double, int, double)

DEF_CVT_SCALE_FUNC(8u32f,  uchar, float, float)
DEF_CVT_SCALE_FUNC(8s32f,  schar, float, float)
DEF_CVT_SCALE_FUNC(16u32f, ushort, float, float)
DEF_CVT_SCALE_FUNC(16s32f, short, float, float)
DEF_CVT_SCALE_FUNC(32s32f, int, float, double)
DEF_CVT_SCALE_FUNC(32f,    float, float, float)
DEF_CVT_SCALE_FUNC(64f32f, double, float, double)

DEF_CVT_SCALE_FUNC(8u64f,  uchar, double, double)
DEF_CVT_SCALE_FUNC(8s64f,  schar, double, double)
DEF_CVT_SCALE_FUNC(16u64f, ushort, double, double)
DEF_CVT_SCALE_FUNC(16s64f, short, double, double)
DEF_CVT_SCALE_FUNC(32s64f, int, double, double)
DEF_CVT_SCALE_FUNC(32f64f, float, double, double)
DEF_CVT_SCALE_FUNC(64f,    double, double, double)

static BinaryFunc getCvtScaleAbsFunc(int depth)
{
    static BinaryFunc cvtScaleAbsTab[] =
    {
        (BinaryFunc)cvtScaleAbs8u, (BinaryFunc)cvtScaleAbs8s8u, (BinaryFunc)cvtScaleAbs16u8u,
        (BinaryFunc)cvtScaleAbs16s8u, (BinaryFunc)cvtScaleAbs32s8u, (BinaryFunc)cvtScaleAbs32f8u,
        (BinaryFunc)cvtScaleAbs64f8u, 0
    };

    return cvtScaleAbsTab[depth];
}

BinaryFunc getConvertScaleFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtScaleTab[][8] =
    {
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8u),
            (BinaryFunc)cvtScale64f8u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u8s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8s),
            (BinaryFunc)cvtScale64f8s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16u),
            (BinaryFunc)cvtScale64f16u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u16s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16s),
            (BinaryFunc)cvtScale64f16s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f32s),
            (BinaryFunc)cvtScale64f32s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32f), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32f),
            (BinaryFunc)cvtScale64f32f, 0
        },
        {
            (BinaryFunc)cvtScale8u64f, (BinaryFunc)cvtScale8s64f, (BinaryFunc)cvtScale16u64f,
            (BinaryFunc)cvtScale16s64f, (BinaryFunc)cvtScale32s64f, (BinaryFunc)cvtScale32f64f,
            (BinaryFunc)cvtScale64f, 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtScaleTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

#ifdef HAVE_OPENCL

static bool ocl_convertScaleAbs( InputArray _src, OutputArray _dst, double alpha, double beta )
{
    const ocl::Device & d = ocl::Device::getDefault();

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    bool doubleSupport = d.doubleFPConfig() > 0;
    if (!doubleSupport && depth == CV_64F)
        return false;

    _dst.create(_src.size(), CV_8UC(cn));
    int kercn = 1;
    if (d.isIntel())
    {
        static const int vectorWidths[] = {4, 4, 4, 4, 4, 4, 4, -1};
        kercn = ocl::checkOptimalVectorWidth( vectorWidths, _src, _dst,
                                              noArray(), noArray(), noArray(),
                                              noArray(), noArray(), noArray(),
                                              noArray(), ocl::OCL_VECTOR_MAX);
    }
    else
        kercn = ocl::predictOptimalVectorWidthMax(_src, _dst);

    int rowsPerWI = d.isIntel() ? 4 : 1;
    char cvt[2][50];
    int wdepth = std::max(depth, CV_32F);
    String build_opt = format("-D OP_CONVERT_SCALE_ABS -D UNARY_OP -D dstT=%s -D srcT1=%s"
                         " -D workT=%s -D wdepth=%d -D convertToWT1=%s -D convertToDT=%s"
                         " -D workT1=%s -D rowsPerWI=%d%s",
                         ocl::typeToStr(CV_8UC(kercn)),
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::typeToStr(CV_MAKE_TYPE(wdepth, kercn)), wdepth,
                         ocl::convertTypeStr(depth, wdepth, kercn, cvt[0]),
                         ocl::convertTypeStr(wdepth, CV_8U, kercn, cvt[1]),
                         ocl::typeToStr(wdepth), rowsPerWI,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "");
    ocl::Kernel k("KF", ocl::core::arithm_oclsrc, build_opt);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    UMat dst = _dst.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    if (wdepth == CV_32F)
        k.args(srcarg, dstarg, (float)alpha, (float)beta);
    else if (wdepth == CV_64F)
        k.args(srcarg, dstarg, alpha, beta);

    size_t globalsize[2] = { (size_t)src.cols * cn / kercn, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

} //cv::


void cv::convertScaleAbs( InputArray _src, OutputArray _dst, double alpha, double beta )
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_convertScaleAbs(_src, _dst, alpha, beta))

    Mat src = _src.getMat();
    int cn = src.channels();
    double scale[] = {alpha, beta};
    _dst.create( src.dims, src.size, CV_8UC(cn) );
    Mat dst = _dst.getMat();
    BinaryFunc func = getCvtScaleAbsFunc(src.depth());
    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.ptr(), src.step, 0, 0, dst.ptr(), dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)it.size*cn, 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], 0, 0, 0, ptrs[1], 0, sz, scale );
    }
}

//==================================================================================================

namespace cv {

#ifdef HAVE_OPENCL

static bool ocl_normalize( InputArray _src, InputOutputArray _dst, InputArray _mask, int dtype,
                           double scale, double delta )
{
    UMat src = _src.getUMat();

    if( _mask.empty() )
        src.convertTo( _dst, dtype, scale, delta );
    else if (src.channels() <= 4)
    {
        const ocl::Device & dev = ocl::Device::getDefault();

        int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype),
                ddepth = CV_MAT_DEPTH(dtype), wdepth = std::max(CV_32F, std::max(sdepth, ddepth)),
                rowsPerWI = dev.isIntel() ? 4 : 1;

        float fscale = static_cast<float>(scale), fdelta = static_cast<float>(delta);
        bool haveScale = std::fabs(scale - 1) > DBL_EPSILON,
                haveZeroScale = !(std::fabs(scale) > DBL_EPSILON),
                haveDelta = std::fabs(delta) > DBL_EPSILON,
                doubleSupport = dev.doubleFPConfig() > 0;

        if (!haveScale && !haveDelta && stype == dtype)
        {
            _src.copyTo(_dst, _mask);
            return true;
        }
        if (haveZeroScale)
        {
            _dst.setTo(Scalar(delta), _mask);
            return true;
        }

        if ((sdepth == CV_64F || ddepth == CV_64F) && !doubleSupport)
            return false;

        char cvt[2][40];
        String opts = format("-D srcT=%s -D dstT=%s -D convertToWT=%s -D cn=%d -D rowsPerWI=%d"
                             " -D convertToDT=%s -D workT=%s%s%s%s -D srcT1=%s -D dstT1=%s",
                             ocl::typeToStr(stype), ocl::typeToStr(dtype),
                             ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]), cn,
                             rowsPerWI, ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]),
                             ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)),
                             doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                             haveScale ? " -D HAVE_SCALE" : "",
                             haveDelta ? " -D HAVE_DELTA" : "",
                             ocl::typeToStr(sdepth), ocl::typeToStr(ddepth));

        ocl::Kernel k("normalizek", ocl::core::normalize_oclsrc, opts);
        if (k.empty())
            return false;

        UMat mask = _mask.getUMat(), dst = _dst.getUMat();

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
                maskarg = ocl::KernelArg::ReadOnlyNoSize(mask),
                dstarg = ocl::KernelArg::ReadWrite(dst);

        if (haveScale)
        {
            if (haveDelta)
                k.args(srcarg, maskarg, dstarg, fscale, fdelta);
            else
                k.args(srcarg, maskarg, dstarg, fscale);
        }
        else
        {
            if (haveDelta)
                k.args(srcarg, maskarg, dstarg, fdelta);
            else
                k.args(srcarg, maskarg, dstarg);
        }

        size_t globalsize[2] = { (size_t)src.cols, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
        return k.run(2, globalsize, NULL, false);
    }
    else
    {
        UMat temp;
        src.convertTo( temp, dtype, scale, delta );
        temp.copyTo( _dst, _mask );
    }

    return true;
}

#endif

} // cv::

void cv::normalize( InputArray _src, InputOutputArray _dst, double a, double b,
                    int norm_type, int rtype, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    double scale = 1, shift = 0;
    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = MIN( a, b ), dmax = MAX( a, b );
        minMaxIdx( _src, &smin, &smax, 0, 0, _mask );
        scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
        shift = dmin - smin*scale;
    }
    else if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( _src, norm_type, _mask );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
        shift = 0;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );

    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    if( rtype < 0 )
        rtype = _dst.fixedType() ? _dst.depth() : depth;

    CV_OCL_RUN(_dst.isUMat(),
               ocl_normalize(_src, _dst, _mask, rtype, scale, shift))

    Mat src = _src.getMat();
    if( _mask.empty() )
        src.convertTo( _dst, rtype, scale, shift );
    else
    {
        Mat temp;
        src.convertTo( temp, rtype, scale, shift );
        temp.copyTo( _dst, _mask );
    }
}
