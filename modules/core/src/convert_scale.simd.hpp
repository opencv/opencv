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
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 va = vx_setall_f32(a), vb = vx_setall_f32(b);
    const int VECSZ = VTraits<v_float32>::vlanes()*2;
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
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

static void
cvtabs_32f( const bool* src_, size_t sstep,
            uchar* dst, size_t dstep,
            Size size, float a, float b )
{
    const uchar* src = (const uchar*)src_;
    uchar v0 = saturate_cast<uchar>(std::abs(b));
    uchar v1 = saturate_cast<uchar>(std::abs(a + b));
    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        for (int j = 0; j < size.width; j++)
            dst[j] = src[j] != 0 ? v1 : v0;
    }
}

template<typename _Ts, typename _Td> inline void
cvt_32f( const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
         Size size, float a, float b )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 va = vx_setall_f32(a), vb = vx_setall_f32(b);
    const int VECSZ = VTraits<v_float32>::vlanes()*2;
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 va = vx_setall_f32(a), vb = vx_setall_f32(b);
    const int VECSZ = VTraits<v_float32>::vlanes();
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
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
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    v_float64 va = vx_setall_f64(a), vb = vx_setall_f64(b);
    const int VECSZ = VTraits<v_float64>::vlanes()*2;
#endif
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
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

#define DEF_CVT_SCALE2BOOL_FUNC(suffix, stype, wtype) \
static void cvtScale##suffix( const uchar* src_, size_t sstep, const uchar*, size_t, \
                              uchar* dst, size_t dstep, Size size, void* scale_) \
{ \
    const stype* src = (const stype*)src_; \
    const double* scale = (const double*)scale_; \
    wtype a = (wtype)scale[0], b = (wtype)scale[1]; \
    sstep /= sizeof(src[0]); \
    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep ) \
        for (int j = 0; j < size.width; j++) \
            dst[j] = (bool)((wtype)src[j]*a + b != 0); \
}

#define DEF_CVT_SCALEBOOL2_FUNC(suffix, dtype, wtype) \
static void cvtScale##suffix( const uchar* src, size_t sstep, const uchar*, size_t, \
                              uchar* dst_, size_t dstep, Size size, void* scale_) \
{ \
    dtype* dst = (dtype*)dst_; \
    const double* scale = (const double*)scale_; \
    wtype a = (wtype)scale[0], b = (wtype)scale[1]; \
    dtype v0 = saturate_cast<dtype>(b), v1 = saturate_cast<dtype>(a + b); \
    dstep /= sizeof(dst[0]); \
    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep ) \
        for (int j = 0; j < size.width; j++) \
            dst[j] = src[j] != 0 ? v1 : v0; \
}

DEF_CVT_SCALE_ABS_FUNC(8u,    cvtabs_32f, uchar,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(8s8u,  cvtabs_32f, schar,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(8b8u,  cvtabs_32f, bool,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtabs_32f, ushort, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtabs_32f, short,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32u8u, cvtabs_32f, unsigned, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtabs_32f, int,    uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtabs_32f, float,  uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64u8u, cvtabs_32f, uint64_t, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64s8u, cvtabs_32f, int64_t, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtabs_32f, double, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16f8u, cvtabs_32f, hfloat, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16bf8u, cvtabs_32f, bfloat, uchar, float)

DEF_CVT_SCALE_FUNC(8u,     cvt_32f, uchar,  uchar, float)
DEF_CVT_SCALE_FUNC(8s8u,   cvt_32f, schar,  uchar, float)
DEF_CVT_SCALE_FUNC(16u8u,  cvt_32f, ushort, uchar, float)
DEF_CVT_SCALE_FUNC(16s8u,  cvt_32f, short,  uchar, float)
DEF_CVT_SCALE_FUNC(32u8u,  cvt_32f, unsigned, uchar, float)
DEF_CVT_SCALE_FUNC(32s8u,  cvt_32f, int,    uchar, float)
DEF_CVT_SCALE_FUNC(32f8u,  cvt_32f, float,  uchar, float)
DEF_CVT_SCALE_FUNC(64f8u,  cvt_32f, double, uchar, float)
DEF_CVT_SCALE_FUNC(64u8u,  cvt_32f, uint64_t, uchar, float)
DEF_CVT_SCALE_FUNC(64s8u,  cvt_32f, int64_t, uchar, float)
DEF_CVT_SCALE_FUNC(16f8u,  cvt_32f, hfloat, uchar, float)
DEF_CVT_SCALE_FUNC(16bf8u, cvt_32f, bfloat, uchar, float)

DEF_CVT_SCALE_FUNC(8u8s,   cvt_32f, uchar,  schar, float)
DEF_CVT_SCALE_FUNC(8s,     cvt_32f, schar,  schar, float)
DEF_CVT_SCALE_FUNC(16u8s,  cvt_32f, ushort, schar, float)
DEF_CVT_SCALE_FUNC(16s8s,  cvt_32f, short,  schar, float)
DEF_CVT_SCALE_FUNC(32u8s,  cvt_32f, unsigned, schar, float)
DEF_CVT_SCALE_FUNC(32s8s,  cvt_32f, int,    schar, float)
DEF_CVT_SCALE_FUNC(32f8s,  cvt_32f, float,  schar, float)
DEF_CVT_SCALE_FUNC(64f8s,  cvt_32f, double, schar, float)
DEF_CVT_SCALE_FUNC(64u8s,  cvt_32f, uint64_t, schar, float)
DEF_CVT_SCALE_FUNC(64s8s,  cvt_32f, int64_t, schar, float)
DEF_CVT_SCALE_FUNC(16f8s,  cvt_32f, hfloat, schar, float)
DEF_CVT_SCALE_FUNC(16bf8s, cvt_32f, bfloat, schar, float)

DEF_CVT_SCALE2BOOL_FUNC(8u8b, uchar, float)
DEF_CVT_SCALE2BOOL_FUNC(8s8b, schar, float)
DEF_CVT_SCALE2BOOL_FUNC(16u8b, ushort, float)
DEF_CVT_SCALE2BOOL_FUNC(16s8b, short, float)
DEF_CVT_SCALE2BOOL_FUNC(32u8b, unsigned, float)
DEF_CVT_SCALE2BOOL_FUNC(32s8b, int, float)
DEF_CVT_SCALE2BOOL_FUNC(32f8b, float, float)
DEF_CVT_SCALE2BOOL_FUNC(64f8b, double, float)
DEF_CVT_SCALE2BOOL_FUNC(64u8b, uint64_t, float)
DEF_CVT_SCALE2BOOL_FUNC(64s8b, int64_t, float)
DEF_CVT_SCALE2BOOL_FUNC(16f8b, hfloat, float)
DEF_CVT_SCALE2BOOL_FUNC(16bf8b, bfloat, float)

DEF_CVT_SCALE_FUNC(8u16u,  cvt_32f, uchar,  ushort, float)
DEF_CVT_SCALE_FUNC(8s16u,  cvt_32f, schar,  ushort, float)
DEF_CVT_SCALE_FUNC(16u,    cvt_32f, ushort, ushort, float)
DEF_CVT_SCALE_FUNC(16s16u, cvt_32f, short,  ushort, float)
DEF_CVT_SCALE_FUNC(32u16u, cvt_32f, unsigned, ushort, float)
DEF_CVT_SCALE_FUNC(32s16u, cvt_32f, int,    ushort, float)
DEF_CVT_SCALE_FUNC(32f16u, cvt_32f, float,  ushort, float)
DEF_CVT_SCALE_FUNC(64f16u, cvt_32f, double, ushort, float)
DEF_CVT_SCALE_FUNC(64u16u, cvt_32f, uint64_t, ushort, float)
DEF_CVT_SCALE_FUNC(64s16u, cvt_32f, int64_t, ushort, float)
DEF_CVT_SCALE_FUNC(16f16u, cvt1_32f, hfloat, ushort, float)
DEF_CVT_SCALE_FUNC(16bf16u, cvt1_32f, bfloat, ushort, float)

DEF_CVT_SCALE_FUNC(8u16s,  cvt_32f, uchar,  short, float)
DEF_CVT_SCALE_FUNC(8s16s,  cvt_32f, schar,  short, float)
DEF_CVT_SCALE_FUNC(16u16s, cvt_32f, ushort, short, float)
DEF_CVT_SCALE_FUNC(16s,    cvt_32f, short,  short, float)
DEF_CVT_SCALE_FUNC(32u16s, cvt_32f, unsigned, short, float)
DEF_CVT_SCALE_FUNC(32s16s, cvt_32f, int,    short, float)
DEF_CVT_SCALE_FUNC(32f16s, cvt_32f, float,  short, float)
DEF_CVT_SCALE_FUNC(64f16s, cvt_32f, double, short, float)
DEF_CVT_SCALE_FUNC(64u16s, cvt_32f, uint64_t, short, float)
DEF_CVT_SCALE_FUNC(64s16s, cvt_32f, int64_t, short, float)
DEF_CVT_SCALE_FUNC(16f16s, cvt1_32f, hfloat, short, float)
DEF_CVT_SCALE_FUNC(16bf16s, cvt1_32f, bfloat, short, float)

DEF_CVT_SCALE_FUNC(8u32u,  cvt_32f, uchar,  unsigned, float)
DEF_CVT_SCALE_FUNC(8s32u,  cvt_32f, schar,  unsigned, float)
DEF_CVT_SCALE_FUNC(16u32u, cvt_32f, ushort, unsigned, float)
DEF_CVT_SCALE_FUNC(16s32u, cvt_32f, short,  unsigned, float)
DEF_CVT_SCALE_FUNC(32u, cvt_32f, unsigned, unsigned, float)
DEF_CVT_SCALE_FUNC(32s32u, cvt_64f, int,    unsigned, double)
DEF_CVT_SCALE_FUNC(32f32u, cvt_32f, float,  unsigned, float)
DEF_CVT_SCALE_FUNC(64f32u, cvt_64f, double, unsigned, double)
DEF_CVT_SCALE_FUNC(64u32u, cvt_32f, uint64_t, unsigned, float)
DEF_CVT_SCALE_FUNC(64s32u, cvt_32f, int64_t, unsigned, float)
DEF_CVT_SCALE_FUNC(16f32u, cvt1_32f, hfloat, unsigned, float)
DEF_CVT_SCALE_FUNC(16bf32u, cvt1_32f, bfloat, unsigned, float)

DEF_CVT_SCALE_FUNC(8u32s,  cvt_32f, uchar,  int, float)
DEF_CVT_SCALE_FUNC(8s32s,  cvt_32f, schar,  int, float)
DEF_CVT_SCALE_FUNC(16u32s, cvt_32f, ushort, int, float)
DEF_CVT_SCALE_FUNC(16s32s, cvt_32f, short,  int, float)
DEF_CVT_SCALE_FUNC(32u32s, cvt_32f, unsigned, int, float)
DEF_CVT_SCALE_FUNC(32s,    cvt_64f, int,    int, double)
DEF_CVT_SCALE_FUNC(32f32s, cvt_32f, float,  int, float)
DEF_CVT_SCALE_FUNC(64f32s, cvt_64f, double, int, double)
DEF_CVT_SCALE_FUNC(64u32s, cvt_32f, uint64_t, int, float)
DEF_CVT_SCALE_FUNC(64s32s, cvt_32f, int64_t, int, float)
DEF_CVT_SCALE_FUNC(16f32s, cvt1_32f, hfloat, int, float)
DEF_CVT_SCALE_FUNC(16bf32s, cvt1_32f, bfloat, int, float)

DEF_CVT_SCALE_FUNC(8u32f,  cvt_32f, uchar,  float, float)
DEF_CVT_SCALE_FUNC(8s32f,  cvt_32f, schar,  float, float)
DEF_CVT_SCALE_FUNC(16u32f, cvt_32f, ushort, float, float)
DEF_CVT_SCALE_FUNC(16s32f, cvt_32f, short,  float, float)
DEF_CVT_SCALE_FUNC(32u32f, cvt_32f, unsigned, float, float)
DEF_CVT_SCALE_FUNC(32s32f, cvt_32f, int,    float, float)
DEF_CVT_SCALE_FUNC(32f,    cvt_32f, float,  float, float)
DEF_CVT_SCALE_FUNC(64f32f, cvt_64f, double, float, double)
DEF_CVT_SCALE_FUNC(64u32f, cvt_32f, uint64_t, float, float)
DEF_CVT_SCALE_FUNC(64s32f, cvt_32f, int64_t, float, float)
DEF_CVT_SCALE_FUNC(16f32f, cvt1_32f, hfloat, float, float)
DEF_CVT_SCALE_FUNC(16bf32f, cvt1_32f, bfloat, float, float)

DEF_CVT_SCALE_FUNC(8u64f,  cvt_64f, uchar,  double, double)
DEF_CVT_SCALE_FUNC(8s64f,  cvt_64f, schar,  double, double)
DEF_CVT_SCALE_FUNC(16u64f, cvt_64f, ushort, double, double)
DEF_CVT_SCALE_FUNC(16s64f, cvt_64f, short,  double, double)
DEF_CVT_SCALE_FUNC(32u64f, cvt_64f, unsigned, double, double)
DEF_CVT_SCALE_FUNC(32s64f, cvt_64f, int,    double, double)
DEF_CVT_SCALE_FUNC(32f64f, cvt_64f, float,  double, double)
DEF_CVT_SCALE_FUNC(64f,    cvt_64f, double, double, double)
DEF_CVT_SCALE_FUNC(64u64f, cvt_64f, uint64_t, double, double)
DEF_CVT_SCALE_FUNC(64s64f, cvt_64f, int64_t, double, double)
DEF_CVT_SCALE_FUNC(16f64f, cvt_64f, hfloat, double, double)
DEF_CVT_SCALE_FUNC(16bf64f, cvt_64f, bfloat, double, double)

DEF_CVT_SCALE_FUNC(8u64u,  cvt_64f, uchar,  uint64_t, double)
DEF_CVT_SCALE_FUNC(8s64u,  cvt_64f, schar,  uint64_t, double)
DEF_CVT_SCALE_FUNC(16u64u, cvt_64f, ushort, uint64_t, double)
DEF_CVT_SCALE_FUNC(16s64u, cvt_64f, short,  uint64_t, double)
DEF_CVT_SCALE_FUNC(32u64u, cvt_64f, unsigned, uint64_t, double)
DEF_CVT_SCALE_FUNC(32s64u, cvt_64f, int,    uint64_t, double)
DEF_CVT_SCALE_FUNC(32f64u, cvt_64f, float,  uint64_t, double)
DEF_CVT_SCALE_FUNC(64f64u, cvt_64f, double, uint64_t, double)
DEF_CVT_SCALE_FUNC(64u, cvt_64f, uint64_t, uint64_t, double)
DEF_CVT_SCALE_FUNC(64s64u, cvt_64f, int64_t, uint64_t, double)
DEF_CVT_SCALE_FUNC(16f64u, cvt_64f, hfloat, uint64_t, double)
DEF_CVT_SCALE_FUNC(16bf64u, cvt_64f, bfloat, uint64_t, double)

DEF_CVT_SCALE_FUNC(8u64s,  cvt_64f, uchar,  int64_t, double)
DEF_CVT_SCALE_FUNC(8s64s,  cvt_64f, schar,  int64_t, double)
DEF_CVT_SCALE_FUNC(16u64s, cvt_64f, ushort, int64_t, double)
DEF_CVT_SCALE_FUNC(16s64s, cvt_64f, short,  int64_t, double)
DEF_CVT_SCALE_FUNC(32u64s, cvt_64f, unsigned, int64_t, double)
DEF_CVT_SCALE_FUNC(32s64s, cvt_64f, int,    int64_t, double)
DEF_CVT_SCALE_FUNC(32f64s, cvt_64f, float,  int64_t, double)
DEF_CVT_SCALE_FUNC(64f64s, cvt_64f, double, int64_t, double)
DEF_CVT_SCALE_FUNC(64u64s, cvt_64f, uint64_t, int64_t, double)
DEF_CVT_SCALE_FUNC(64s, cvt_64f, int64_t, int64_t, double)
DEF_CVT_SCALE_FUNC(16f64s, cvt_64f, hfloat, int64_t, double)
DEF_CVT_SCALE_FUNC(16bf64s, cvt_64f, bfloat, int64_t, double)

DEF_CVT_SCALE_FUNC(8u16f,  cvt1_32f, uchar,  hfloat, float)
DEF_CVT_SCALE_FUNC(8s16f,  cvt1_32f, schar,  hfloat, float)
DEF_CVT_SCALE_FUNC(16u16f, cvt1_32f, ushort, hfloat, float)
DEF_CVT_SCALE_FUNC(16s16f, cvt1_32f, short,  hfloat, float)
DEF_CVT_SCALE_FUNC(32u16f, cvt1_32f, unsigned, hfloat, float)
DEF_CVT_SCALE_FUNC(32s16f, cvt1_32f, int,    hfloat, float)
DEF_CVT_SCALE_FUNC(32f16f, cvt1_32f, float,  hfloat, float)
DEF_CVT_SCALE_FUNC(64f16f, cvt1_32f, double, hfloat, float)
DEF_CVT_SCALE_FUNC(64u16f, cvt1_32f, uint64_t, hfloat, float)
DEF_CVT_SCALE_FUNC(64s16f, cvt1_32f, int64_t, hfloat, float)
DEF_CVT_SCALE_FUNC(16f,    cvt1_32f, hfloat, hfloat, float)
DEF_CVT_SCALE_FUNC(16bf16f, cvt1_32f, bfloat, hfloat, float)

DEF_CVT_SCALE_FUNC(8u16bf,  cvt1_32f, uchar,  bfloat, float)
DEF_CVT_SCALE_FUNC(8s16bf,  cvt1_32f, schar,  bfloat, float)
DEF_CVT_SCALE_FUNC(16u16bf, cvt1_32f, ushort, bfloat, float)
DEF_CVT_SCALE_FUNC(16s16bf, cvt1_32f, short,  bfloat, float)
DEF_CVT_SCALE_FUNC(32u16bf, cvt1_32f, unsigned, bfloat, float)
DEF_CVT_SCALE_FUNC(32s16bf, cvt1_32f, int,    bfloat, float)
DEF_CVT_SCALE_FUNC(32f16bf, cvt1_32f, float,  bfloat, float)
DEF_CVT_SCALE_FUNC(64f16bf, cvt1_32f, double, bfloat, float)
DEF_CVT_SCALE_FUNC(64u16bf, cvt1_32f, uint64_t, bfloat, float)
DEF_CVT_SCALE_FUNC(64s16bf, cvt1_32f, int64_t, bfloat, float)
DEF_CVT_SCALE_FUNC(16f16bf, cvt1_32f, hfloat, bfloat, float)
DEF_CVT_SCALE_FUNC(16bf, cvt1_32f, bfloat, bfloat, float)

DEF_CVT_SCALEBOOL2_FUNC(8b8u, uchar, float)
DEF_CVT_SCALEBOOL2_FUNC(8b8s, schar, float)
DEF_CVT_SCALEBOOL2_FUNC(8b, bool, float)
DEF_CVT_SCALEBOOL2_FUNC(8b16u, ushort, float)
DEF_CVT_SCALEBOOL2_FUNC(8b16s, short, float)
DEF_CVT_SCALEBOOL2_FUNC(8b32u, unsigned, float)
DEF_CVT_SCALEBOOL2_FUNC(8b32s, int, float)
DEF_CVT_SCALEBOOL2_FUNC(8b32f, float, float)
DEF_CVT_SCALEBOOL2_FUNC(8b64u, uint64_t, double)
DEF_CVT_SCALEBOOL2_FUNC(8b64s, int64_t, double)
DEF_CVT_SCALEBOOL2_FUNC(8b64f, double, double)
DEF_CVT_SCALEBOOL2_FUNC(8b16f, hfloat, float)
DEF_CVT_SCALEBOOL2_FUNC(8b16bf, bfloat, float)

BinaryFunc getCvtScaleAbsFunc(int depth)
{
    BinaryFunc func =
        depth == CV_8U ? (BinaryFunc)cvtScaleAbs8u :
        depth == CV_8S ? (BinaryFunc)cvtScaleAbs8s8u :
        depth == CV_Bool ? (BinaryFunc)cvtScaleAbs8b8u :
        depth == CV_16U ? (BinaryFunc)cvtScaleAbs16u8u :
        depth == CV_16S ? (BinaryFunc)cvtScaleAbs16s8u :
        depth == CV_16F ? (BinaryFunc)cvtScaleAbs16f8u :
        depth == CV_16BF ? (BinaryFunc)cvtScaleAbs16bf8u :
        depth == CV_32U ? (BinaryFunc)cvtScaleAbs32u8u :
        depth == CV_32S ? (BinaryFunc)cvtScaleAbs32s8u :
        depth == CV_32F ? (BinaryFunc)cvtScaleAbs32f8u :
        depth == CV_64U ? (BinaryFunc)cvtScaleAbs64u8u :
        depth == CV_64S ? (BinaryFunc)cvtScaleAbs64s8u :
        depth == CV_64F ? (BinaryFunc)cvtScaleAbs64f8u : 0;
    CV_Assert(func != 0);
    return func;
}

BinaryFunc getConvertScaleFunc(int sdepth_, int ddepth_)
{
    int sdepth = CV_MAT_DEPTH(sdepth_);
    int ddepth = CV_MAT_DEPTH(ddepth_);
    BinaryFunc func =
        ddepth == CV_8U ? (
            sdepth == CV_8U ? cvtScale8u :
            sdepth == CV_8S ? cvtScale8s8u :
            sdepth == CV_Bool ? cvtScale8b8u :
            sdepth == CV_16U ? cvtScale16u8u :
            sdepth == CV_16S ? cvtScale16s8u :
            sdepth == CV_32U ? cvtScale32u8u :
            sdepth == CV_32S ? cvtScale32s8u :
            sdepth == CV_32F ? cvtScale32f8u :
            sdepth == CV_64F ? cvtScale64f8u :
            sdepth == CV_16F ? cvtScale16f8u :
            sdepth == CV_16BF ? cvtScale16bf8u :
            sdepth == CV_64U ? cvtScale64u8u :
            sdepth == CV_64S ? cvtScale64s8u :
            0) :
        ddepth == CV_8S ? (
            sdepth == CV_8U ? cvtScale8u8s :
            sdepth == CV_8S ? cvtScale8s :
            sdepth == CV_Bool ? cvtScale8b8s :
            sdepth == CV_16U ? cvtScale16u8s :
            sdepth == CV_16S ? cvtScale16s8s :
            sdepth == CV_32U ? cvtScale32u8s :
            sdepth == CV_32S ? cvtScale32s8s :
            sdepth == CV_32F ? cvtScale32f8s :
            sdepth == CV_64F ? cvtScale64f8s :
            sdepth == CV_16F ? cvtScale16f8s :
            sdepth == CV_16BF ? cvtScale16bf8s :
            sdepth == CV_64U ? cvtScale64u8s :
            sdepth == CV_64S ? cvtScale64s8s :
            0) :
        ddepth == CV_16U ? (
            sdepth == CV_8U ? cvtScale8u16u :
            sdepth == CV_8S ? cvtScale8s16u :
            sdepth == CV_Bool ? cvtScale8b16u :
            sdepth == CV_16U ? cvtScale16u :
            sdepth == CV_16S ? cvtScale16s16u :
            sdepth == CV_32U ? cvtScale32u16u :
            sdepth == CV_32S ? cvtScale32s16u :
            sdepth == CV_32F ? cvtScale32f16u :
            sdepth == CV_64F ? cvtScale64f16u :
            sdepth == CV_16F ? cvtScale16f16u :
            sdepth == CV_16BF ? cvtScale16bf16u :
            sdepth == CV_64U ? cvtScale64u16u :
            sdepth == CV_64S ? cvtScale64s16u :
            0) :
        ddepth == CV_16S ? (
            sdepth == CV_8U ? cvtScale8u16s :
            sdepth == CV_8S ? cvtScale8s16s :
            sdepth == CV_Bool ? cvtScale8b16s :
            sdepth == CV_16U ? cvtScale16u16s :
            sdepth == CV_16S ? cvtScale16s :
            sdepth == CV_32U ? cvtScale32u16s :
            sdepth == CV_32S ? cvtScale32s16s :
            sdepth == CV_32F ? cvtScale32f16s :
            sdepth == CV_64F ? cvtScale64f16s :
            sdepth == CV_16F ? cvtScale16f16s :
            sdepth == CV_16BF ? cvtScale16bf16s :
            sdepth == CV_64U ? cvtScale64u16s :
            sdepth == CV_64S ? cvtScale64s16s :
            0) :
        ddepth == CV_32U ? (
            sdepth == CV_8U ? cvtScale8u32u :
            sdepth == CV_8S ? cvtScale8s32u :
            sdepth == CV_Bool ? cvtScale8b32u :
            sdepth == CV_16U ? cvtScale16u32u :
            sdepth == CV_16S ? cvtScale16s32u :
            sdepth == CV_32U ? cvtScale32u :
            sdepth == CV_32S ? cvtScale32s32u :
            sdepth == CV_32F ? cvtScale32f32u :
            sdepth == CV_64F ? cvtScale64f32u :
            sdepth == CV_16F ? cvtScale16f32u :
            sdepth == CV_16BF ? cvtScale16bf32u :
            sdepth == CV_64U ? cvtScale64u32u :
            sdepth == CV_64S ? cvtScale64s32u :

            0) :
        ddepth == CV_32S ? (
            sdepth == CV_8U ? cvtScale8u32s :
            sdepth == CV_8S ? cvtScale8s32s :
            sdepth == CV_Bool ? cvtScale8b32s :
            sdepth == CV_16U ? cvtScale16u32s :
            sdepth == CV_16S ? cvtScale16s32s :
            sdepth == CV_32U ? cvtScale32u32s :
            sdepth == CV_32S ? cvtScale32s :
            sdepth == CV_32F ? cvtScale32f32s :
            sdepth == CV_64F ? cvtScale64f32s :
            sdepth == CV_16F ? cvtScale16f32s :
            sdepth == CV_16BF ? cvtScale16bf32s :
            sdepth == CV_64U ? cvtScale64u32s :
            sdepth == CV_64S ? cvtScale64s32s :
            0) :
        ddepth == CV_32F ? (
            sdepth == CV_8U ? cvtScale8u32f :
            sdepth == CV_8S ? cvtScale8s32f :
            sdepth == CV_Bool ? cvtScale8b32f :
            sdepth == CV_16U ? cvtScale16u32f :
            sdepth == CV_16S ? cvtScale16s32f :
            sdepth == CV_32U ? cvtScale32u32f :
            sdepth == CV_32S ? cvtScale32s32f :
            sdepth == CV_32F ? cvtScale32f :
            sdepth == CV_64F ? cvtScale64f32f :
            sdepth == CV_16F ? cvtScale16f32f :
            sdepth == CV_16BF ? cvtScale16bf32f :
            sdepth == CV_64U ? cvtScale64u32f :
            sdepth == CV_64S ? cvtScale64s32f :
            0) :
        ddepth == CV_64F ? (
            sdepth == CV_8U ? cvtScale8u64f :
            sdepth == CV_8S ? cvtScale8s64f :
            sdepth == CV_Bool ? cvtScale8b64f :
            sdepth == CV_16U ? cvtScale16u64f :
            sdepth == CV_16S ? cvtScale16s64f :
            sdepth == CV_32U ? cvtScale32u64f :
            sdepth == CV_32S ? cvtScale32s64f :
            sdepth == CV_32F ? cvtScale32f64f :
            sdepth == CV_64F ? cvtScale64f :
            sdepth == CV_16F ? cvtScale16f64f :
            sdepth == CV_16BF ? cvtScale16bf64f :
            sdepth == CV_64U ? cvtScale64u64f :
            sdepth == CV_64S ? cvtScale64s64f :
            0) :
        ddepth == CV_16F ? (
            sdepth == CV_8U ? cvtScale8u16f :
            sdepth == CV_8S ? cvtScale8s16f :
            sdepth == CV_Bool ? cvtScale8b16f :
            sdepth == CV_16U ? cvtScale16u16f :
            sdepth == CV_16S ? cvtScale16s16f :
            sdepth == CV_32U ? cvtScale32u16f :
            sdepth == CV_32S ? cvtScale32s16f :
            sdepth == CV_32F ? cvtScale32f16f :
            sdepth == CV_64F ? cvtScale64f16f :
            sdepth == CV_16F ? cvtScale16f :
            sdepth == CV_16BF ? cvtScale16bf16f :
            sdepth == CV_64U ? cvtScale64u16f :
            sdepth == CV_64S ? cvtScale64s16f :
            0) :
        ddepth == CV_16BF ? (
            sdepth == CV_8U ? cvtScale8u16bf :
            sdepth == CV_8S ? cvtScale8s16bf :
            sdepth == CV_Bool ? cvtScale8b16bf :
            sdepth == CV_16U ? cvtScale16u16bf :
            sdepth == CV_16S ? cvtScale16s16bf :
            sdepth == CV_32U ? cvtScale32u16bf :
            sdepth == CV_32S ? cvtScale32s16bf :
            sdepth == CV_32F ? cvtScale32f16bf :
            sdepth == CV_64F ? cvtScale64f16bf :
            sdepth == CV_16F ? cvtScale16f16bf :
            sdepth == CV_16BF ? cvtScale16bf :
            sdepth == CV_64U ? cvtScale64u16bf :
            sdepth == CV_64S ? cvtScale64s16bf :
            0) :
        ddepth == CV_Bool ? (
            sdepth == CV_8U ? cvtScale8u8b :
            sdepth == CV_8S ? cvtScale8s8b :
            sdepth == CV_Bool ? cvtScale8b :
            sdepth == CV_16U ? cvtScale16u8b :
            sdepth == CV_16S ? cvtScale16s8b :
            sdepth == CV_32U ? cvtScale32u8b :
            sdepth == CV_32S ? cvtScale32s8b :
            sdepth == CV_32F ? cvtScale32f8b :
            sdepth == CV_64F ? cvtScale64f8b :
            sdepth == CV_16F ? cvtScale16f8b :
            sdepth == CV_16BF ? cvtScale16bf8b :
            sdepth == CV_64U ? cvtScale64u8b :
            sdepth == CV_64S ? cvtScale64s8b :
            0) :
        ddepth == CV_64U ? (
            sdepth == CV_8U ? cvtScale8u64u :
            sdepth == CV_8S ? cvtScale8s64u :
            sdepth == CV_Bool ? cvtScale8b64u :
            sdepth == CV_16U ? cvtScale16u64u :
            sdepth == CV_16S ? cvtScale16s64u :
            sdepth == CV_32U ? cvtScale32u64u :
            sdepth == CV_32S ? cvtScale32s64u :
            sdepth == CV_32F ? cvtScale32f64u :
            sdepth == CV_64F ? cvtScale64f64u :
            sdepth == CV_16F ? cvtScale16f64u :
            sdepth == CV_16BF ? cvtScale16bf64u :
            sdepth == CV_64U ? cvtScale64u :
            sdepth == CV_64S ? cvtScale64s64u :
            0) :
        ddepth == CV_64S ? (
            sdepth == CV_8U ? cvtScale8u64s :
            sdepth == CV_8S ? cvtScale8s64s :
            sdepth == CV_Bool ? cvtScale8b64s :
            sdepth == CV_16U ? cvtScale16u64s :
            sdepth == CV_16S ? cvtScale16s64s :
            sdepth == CV_32U ? cvtScale32u64s :
            sdepth == CV_32S ? cvtScale32s64s :
            sdepth == CV_32F ? cvtScale32f64s :
            sdepth == CV_64F ? cvtScale64f64s :
            sdepth == CV_16F ? cvtScale16f64s :
            sdepth == CV_16BF ? cvtScale16bf64s :
            sdepth == CV_64U ? cvtScale64u64s :
            sdepth == CV_64S ? cvtScale64s :
            0) :
        0;
    CV_Assert(func != 0);
    return func;
}

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
