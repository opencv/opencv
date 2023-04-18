// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "convert.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

BinaryFunc getCvtScaleAbsFunc(int depth);
BinaryFunc getConvertScaleFunc(int sdepth, int ddepth);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

/****************************************************************************************\
*                                convertScale[Abs]                                       *
\****************************************************************************************/

template<typename _Ts, typename _Td> inline void
cvtabs_32f( const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
            Size size, float a, float b )
{
#if CV_SIMD
    v_float32 va = vx_setall_f32(a), vb = vx_setall_f32(b);
    const int VECSZ = v_float32::nlanes*2;
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if CV_SIMD
        for( ; j < size.width; j += VECSZ )
        {
            if( j > size.width - VECSZ )
            {
                if( j == 0 || src == (_Ts*)dst )
                    break;
                j = size.width - VECSZ;
            }
            v_float32 v0, v1;
            vx_load_pair_as(src + j, v0, v1);
            v0 = v_fma(v0, va, vb);
            v1 = v_fma(v1, va, vb);
            v_store_pair_as(dst + j, v_abs(v0), v_abs(v1));
        }
#endif
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(std::abs(src[j]*a + b));
    }
}

// variant for conversions 16f <-> ... w/o unrolling
template<typename _Ts, typename _Td> inline void
cvtabs1_32f( const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
             Size size, float a, float b )
{
#if CV_SIMD
    v_float32 va = vx_setall_f32(a), vb = vx_setall_f32(b);
    const int VECSZ = v_float32::nlanes*2;
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if CV_SIMD
        for( ; j < size.width; j += VECSZ )
        {
            if( j > size.width - VECSZ )
            {
                if( j == 0 || src == (_Ts*)dst )
                    break;
                j = size.width - VECSZ;
            }
            v_float32 v0;
            vx_load_as(src + j, v0);
            v0 = v_fma(v0, va, vb);
            v_store_as(dst + j, v_abs(v0));
        }
#endif
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(src[j]*a + b);
    }
}

template<typename _Ts, typename _Td> inline void
cvt_32f( const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
         Size size, float a, float b )
{
#if CV_SIMD
    v_float32 va = vx_setall_f32(a), vb = vx_setall_f32(b);
    const int VECSZ = v_float32::nlanes*2;
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if CV_SIMD
        for( ; j < size.width; j += VECSZ )
        {
            if( j > size.width - VECSZ )
            {
                if( j == 0 || src == (_Ts*)dst )
                    break;
                j = size.width - VECSZ;
            }
            v_float32 v0, v1;
            vx_load_pair_as(src + j, v0, v1);
            v0 = v_fma(v0, va, vb);
            v1 = v_fma(v1, va, vb);
            v_store_pair_as(dst + j, v0, v1);
        }
#endif
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(src[j]*a + b);
    }
}

// variant for conversions 16f <-> ... w/o unrolling
template<typename _Ts, typename _Td> inline void
cvt1_32f( const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
          Size size, float a, float b )
{
#if CV_SIMD
    v_float32 va = vx_setall_f32(a), vb = vx_setall_f32(b);
    const int VECSZ = v_float32::nlanes;
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if CV_SIMD
        for( ; j < size.width; j += VECSZ )
        {
            if( j > size.width - VECSZ )
            {
                if( j == 0 || src == (_Ts*)dst )
                    break;
                j = size.width - VECSZ;
            }
            v_float32 v0;
            vx_load_as(src + j, v0);
            v0 = v_fma(v0, va, vb);
            v_store_as(dst + j, v0);
        }
#endif
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(src[j]*a + b);
    }
}


template<typename _Ts, typename _Td> inline void
cvt_64f( const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
         Size size, double a, double b )
{
#if CV_SIMD_64F
    v_float64 va = vx_setall_f64(a), vb = vx_setall_f64(b);
    const int VECSZ = v_float64::nlanes*2;
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if CV_SIMD_64F
        for( ; j < size.width; j += VECSZ )
        {
            if( j > size.width - VECSZ )
            {
                if( j == 0 || src == (_Ts*)dst )
                    break;
                j = size.width - VECSZ;
            }
            v_float64 v0, v1;
            vx_load_pair_as(src + j, v0, v1);
            v0 = v_fma(v0, va, vb);
            v1 = v_fma(v1, va, vb);
            v_store_pair_as(dst + j, v0, v1);
        }
#endif
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(src[j]*a + b);
    }
}

//==================================================================================================

#define DEF_CVT_SCALE_ABS_FUNC(suffix, cvt, stype, dtype, wtype) \
static void cvtScaleAbs##suffix( const uchar* src_, size_t sstep, const uchar*, size_t, \
                                 uchar* dst_, size_t dstep, Size size, void* scale_) \
{ \
    const stype* src = (const stype*)src_; \
    dtype* dst = (dtype*)dst_; \
    double* scale = (double*)scale_; \
    cvt(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}


#define DEF_CVT_SCALE_FUNC(suffix, cvt, stype, dtype, wtype) \
static void cvtScale##suffix( const uchar* src_, size_t sstep, const uchar*, size_t, \
                              uchar* dst_, size_t dstep, Size size, void* scale_) \
{ \
    const stype* src = (const stype*)src_; \
    dtype* dst = (dtype*)dst_; \
    double* scale = (double*)scale_; \
    cvt(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

DEF_CVT_SCALE_ABS_FUNC(8u,    cvtabs_32f, uchar,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(8s8u,  cvtabs_32f, schar,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtabs_32f, ushort, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtabs_32f, short,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtabs_32f, int,    uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtabs_32f, float,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtabs_32f, double, uchar, float)

DEF_CVT_SCALE_FUNC(8u,     cvt_32f, uchar,  uchar, float)
DEF_CVT_SCALE_FUNC(8s8u,   cvt_32f, schar,  uchar, float)
DEF_CVT_SCALE_FUNC(16u8u,  cvt_32f, ushort, uchar, float)
DEF_CVT_SCALE_FUNC(16s8u,  cvt_32f, short,  uchar, float)
DEF_CVT_SCALE_FUNC(32s8u,  cvt_32f, int,    uchar, float)
DEF_CVT_SCALE_FUNC(32f8u,  cvt_32f, float,  uchar, float)
DEF_CVT_SCALE_FUNC(64f8u,  cvt_32f, double, uchar, float)
DEF_CVT_SCALE_FUNC(16f8u,  cvt_32f, float16_t, uchar, float)

DEF_CVT_SCALE_FUNC(8u8s,   cvt_32f, uchar,  schar, float)
DEF_CVT_SCALE_FUNC(8s,     cvt_32f, schar,  schar, float)
DEF_CVT_SCALE_FUNC(16u8s,  cvt_32f, ushort, schar, float)
DEF_CVT_SCALE_FUNC(16s8s,  cvt_32f, short,  schar, float)
DEF_CVT_SCALE_FUNC(32s8s,  cvt_32f, int,    schar, float)
DEF_CVT_SCALE_FUNC(32f8s,  cvt_32f, float,  schar, float)
DEF_CVT_SCALE_FUNC(64f8s,  cvt_32f, double, schar, float)
DEF_CVT_SCALE_FUNC(16f8s,  cvt_32f, float16_t, schar, float)

DEF_CVT_SCALE_FUNC(8u16u,  cvt_32f, uchar,  ushort, float)
DEF_CVT_SCALE_FUNC(8s16u,  cvt_32f, schar,  ushort, float)
DEF_CVT_SCALE_FUNC(16u,    cvt_32f, ushort, ushort, float)
DEF_CVT_SCALE_FUNC(16s16u, cvt_32f, short,  ushort, float)
DEF_CVT_SCALE_FUNC(32s16u, cvt_32f, int,    ushort, float)
DEF_CVT_SCALE_FUNC(32f16u, cvt_32f, float,  ushort, float)
DEF_CVT_SCALE_FUNC(64f16u, cvt_32f, double, ushort, float)
DEF_CVT_SCALE_FUNC(16f16u, cvt1_32f, float16_t, ushort, float)

DEF_CVT_SCALE_FUNC(8u16s,  cvt_32f, uchar,  short, float)
DEF_CVT_SCALE_FUNC(8s16s,  cvt_32f, schar,  short, float)
DEF_CVT_SCALE_FUNC(16u16s, cvt_32f, ushort, short, float)
DEF_CVT_SCALE_FUNC(16s,    cvt_32f, short,  short, float)
DEF_CVT_SCALE_FUNC(32s16s, cvt_32f, int,    short, float)
DEF_CVT_SCALE_FUNC(32f16s, cvt_32f, float,  short, float)
DEF_CVT_SCALE_FUNC(64f16s, cvt_32f, double, short, float)
DEF_CVT_SCALE_FUNC(16f16s, cvt1_32f, float16_t, short, float)

DEF_CVT_SCALE_FUNC(8u32s,  cvt_32f, uchar,  int, float)
DEF_CVT_SCALE_FUNC(8s32s,  cvt_32f, schar,  int, float)
DEF_CVT_SCALE_FUNC(16u32s, cvt_32f, ushort, int, float)
DEF_CVT_SCALE_FUNC(16s32s, cvt_32f, short,  int, float)
DEF_CVT_SCALE_FUNC(32s,    cvt_64f, int,    int, double)
DEF_CVT_SCALE_FUNC(32f32s, cvt_32f, float,  int, float)
DEF_CVT_SCALE_FUNC(64f32s, cvt_64f, double, int, double)
DEF_CVT_SCALE_FUNC(16f32s, cvt1_32f, float16_t, int, float)

DEF_CVT_SCALE_FUNC(8u32f,  cvt_32f, uchar,  float, float)
DEF_CVT_SCALE_FUNC(8s32f,  cvt_32f, schar,  float, float)
DEF_CVT_SCALE_FUNC(16u32f, cvt_32f, ushort, float, float)
DEF_CVT_SCALE_FUNC(16s32f, cvt_32f, short,  float, float)
DEF_CVT_SCALE_FUNC(32s32f, cvt_32f, int,    float, float)
DEF_CVT_SCALE_FUNC(32f,    cvt_32f, float,  float, float)
DEF_CVT_SCALE_FUNC(64f32f, cvt_64f, double, float, double)
DEF_CVT_SCALE_FUNC(16f32f, cvt1_32f, float16_t, float, float)

DEF_CVT_SCALE_FUNC(8u64f,  cvt_64f, uchar,  double, double)
DEF_CVT_SCALE_FUNC(8s64f,  cvt_64f, schar,  double, double)
DEF_CVT_SCALE_FUNC(16u64f, cvt_64f, ushort, double, double)
DEF_CVT_SCALE_FUNC(16s64f, cvt_64f, short,  double, double)
DEF_CVT_SCALE_FUNC(32s64f, cvt_64f, int,    double, double)
DEF_CVT_SCALE_FUNC(32f64f, cvt_64f, float,  double, double)
DEF_CVT_SCALE_FUNC(64f,    cvt_64f, double, double, double)
DEF_CVT_SCALE_FUNC(16f64f, cvt_64f, float16_t, double, double)

DEF_CVT_SCALE_FUNC(8u16f,  cvt1_32f, uchar,  float16_t, float)
DEF_CVT_SCALE_FUNC(8s16f,  cvt1_32f, schar,  float16_t, float)
DEF_CVT_SCALE_FUNC(16u16f, cvt1_32f, ushort, float16_t, float)
DEF_CVT_SCALE_FUNC(16s16f, cvt1_32f, short,  float16_t, float)
DEF_CVT_SCALE_FUNC(32s16f, cvt1_32f, int,    float16_t, float)
DEF_CVT_SCALE_FUNC(32f16f, cvt1_32f, float,  float16_t, float)
DEF_CVT_SCALE_FUNC(64f16f, cvt_64f,  double, float16_t, double)
DEF_CVT_SCALE_FUNC(16f,    cvt1_32f, float16_t, float16_t, float)

BinaryFunc getCvtScaleAbsFunc(int depth)
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
            (BinaryFunc)cvtScale64f8u, (BinaryFunc)cvtScale16f8u
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u8s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8s),
            (BinaryFunc)cvtScale64f8s, (BinaryFunc)cvtScale16f8s
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16u),
            (BinaryFunc)cvtScale64f16u, (BinaryFunc)cvtScale16f16u
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u16s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16s),
            (BinaryFunc)cvtScale64f16s, (BinaryFunc)cvtScale16f16s
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f32s),
            (BinaryFunc)cvtScale64f32s, (BinaryFunc)cvtScale16f32s
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32f), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32f),
            (BinaryFunc)cvtScale64f32f, (BinaryFunc)cvtScale16f32f
        },
        {
            (BinaryFunc)cvtScale8u64f, (BinaryFunc)cvtScale8s64f, (BinaryFunc)cvtScale16u64f,
            (BinaryFunc)cvtScale16s64f, (BinaryFunc)cvtScale32s64f, (BinaryFunc)cvtScale32f64f,
            (BinaryFunc)cvtScale64f, (BinaryFunc)cvtScale16f64f
        },
        {
            (BinaryFunc)cvtScale8u16f, (BinaryFunc)cvtScale8s16f, (BinaryFunc)cvtScale16u16f,
            (BinaryFunc)cvtScale16s16f, (BinaryFunc)cvtScale32s16f, (BinaryFunc)cvtScale32f16f,
            (BinaryFunc)cvtScale64f16f, (BinaryFunc)cvtScale16f
        },
    };

    return cvtScaleTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
