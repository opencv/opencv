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

// variant for convrsions 16f <-> ... w/o unrolling
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

// variant for convrsions 16f <-> ... w/o unrolling
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

// without optimization
template<typename _Ts, typename _Td> inline void
cvtabs2_32f( const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
            Size size, float a, float b )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(std::abs(src[j]*a + b));
    }
}

// without optimization
template<typename _Ts, typename _Td> inline void
cvt2_32f(const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
    Size size, float a, float b)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for (int i = 0; i < size.height; i++, src += sstep, dst += dstep)
    {
        int j = 0;
        for (; j < size.width; j++)
            dst[j] = saturate_cast<_Td>(src[j] * a + b);
    }
}

// without optimization
template<typename _Ts, typename _Td> inline void
cvt2abs_64f(const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
    Size size, double a, double b)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for (int i = 0; i < size.height; i++, src += sstep, dst += dstep)
    {
        int j = 0;
        for (; j < size.width; j++)
            dst[j] = saturate_cast<_Td>(std::abs(src[j] * a + b));
    }
}

// without optimization
template<typename _Ts, typename _Td> inline void
cvt2_64f(const _Ts* src, size_t sstep, _Td* dst, size_t dstep,
    Size size, double a, double b)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for (int i = 0; i < size.height; i++, src += sstep, dst += dstep)
    {
        int j = 0;
        for (; j < size.width; j++)
            dst[j] = saturate_cast<_Td>(src[j] * a + b);
    }
}

//==================================================================================================

#define DEF_CVT_SCALE_ABS_FUNC(suffix, cvt, stype, dtype, wtype) \
static void cvtScaleAbs##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                                 dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    cvt(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}


#define DEF_CVT_SCALE_FUNC(suffix, cvt, stype, dtype, wtype) \
static void cvtScale##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                              dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    cvt(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

DEF_CVT_SCALE_ABS_FUNC(8u,    cvtabs_32f, uchar,     uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtabs_32f, ushort,    uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32u8u, cvtabs2_32f,uint,      uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64u8u, cvt2abs_64f,uint64_t,  uchar, double)
DEF_CVT_SCALE_ABS_FUNC(8s8u,  cvtabs_32f, schar,     uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtabs_32f, short,     uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtabs_32f, int,       uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64s8u, cvt2abs_64f,int64_t,   uchar, double)
//DEF_CVT_SCALE_ABS_FUNC(16f8u, cvtabs_32f, float16_t, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtabs_32f, float,     uchar, float)
/*
FIXME: This function should exist, and be utilized instead
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtabs_64f, double,   uchar, double)
*/
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtabs_32f, double,   uchar, float)

DEF_CVT_SCALE_FUNC(8u,     cvt_32f, uchar,     uchar, float)
DEF_CVT_SCALE_FUNC(16u8u,  cvt_32f, ushort,    uchar, float)
DEF_CVT_SCALE_FUNC(32u8u,  cvt2_32f,uint,      uchar, float)
DEF_CVT_SCALE_FUNC(64u8u,  cvt2_64f,uint64_t,  uchar, double)
DEF_CVT_SCALE_FUNC(8s8u,   cvt_32f, schar,     uchar, float)
DEF_CVT_SCALE_FUNC(16s8u,  cvt_32f, short,     uchar, float)
DEF_CVT_SCALE_FUNC(32s8u,  cvt_32f, int,       uchar, float)
DEF_CVT_SCALE_FUNC(64s8u,  cvt2_64f,int64_t,   uchar, double)
DEF_CVT_SCALE_FUNC(16f8u,  cvt_32f, float16_t, uchar, float)
DEF_CVT_SCALE_FUNC(32f8u,  cvt_32f, float,     uchar, float)
/*
FIXME: This function should exist, and be utilized instead
DEF_CVT_SCALE_FUNC(64f8u,  cvt_64f, double,    uchar, double)
*/
DEF_CVT_SCALE_FUNC(64f8u,  cvt_32f, double,    uchar, float)

DEF_CVT_SCALE_FUNC(8u16u,   cvt_32f, uchar,     ushort, float)
DEF_CVT_SCALE_FUNC(16u,     cvt_32f, ushort,    ushort, float)
DEF_CVT_SCALE_FUNC(32u16u,  cvt2_32f,uint,      ushort, float)
DEF_CVT_SCALE_FUNC(64u16u,  cvt2_64f,uint64_t,  ushort, double)
DEF_CVT_SCALE_FUNC(8s16u,   cvt_32f, schar,     ushort, float)
DEF_CVT_SCALE_FUNC(16s16u,  cvt_32f, short,     ushort, float)
DEF_CVT_SCALE_FUNC(32s16u,  cvt_32f, int,       ushort, float)
DEF_CVT_SCALE_FUNC(64s16u,  cvt2_64f,int64_t,   ushort, double)
DEF_CVT_SCALE_FUNC(16f16u,  cvt_32f, float16_t, ushort, float)
DEF_CVT_SCALE_FUNC(32f16u,  cvt_32f, float,     ushort, float)
/*
FIXME: This function should exist, and be utilized instead
DEF_CVT_SCALE_FUNC(64f16u,  cvt_64f, double,    ushort, double)
*/
DEF_CVT_SCALE_FUNC(64f16u,  cvt_32f, double,    ushort, float)

DEF_CVT_SCALE_FUNC(8u32u,   cvt2_32f,uchar,     uint, float)
DEF_CVT_SCALE_FUNC(16u32u,  cvt2_32f,ushort,    uint, float)
DEF_CVT_SCALE_FUNC(32u,     cvt2_64f,uint,      uint, double)
DEF_CVT_SCALE_FUNC(64u32u,  cvt2_64f,uint64_t,  uint, double)
DEF_CVT_SCALE_FUNC(8s32u,   cvt2_32f,schar,     uint, float)
DEF_CVT_SCALE_FUNC(16s32u,  cvt2_32f,short,     uint, float)
DEF_CVT_SCALE_FUNC(32s32u,  cvt2_32f,int,       uint, float)
DEF_CVT_SCALE_FUNC(64s32u,  cvt2_64f,int64_t,   uint, double)
DEF_CVT_SCALE_FUNC(16f32u,  cvt2_32f,float16_t, uint, float)
DEF_CVT_SCALE_FUNC(32f32u,  cvt2_32f,float,     uint, float)
DEF_CVT_SCALE_FUNC(64f32u,  cvt2_64f,double,    uint, double)

DEF_CVT_SCALE_FUNC(8u64u,   cvt2_64f,uchar,     uint64_t, double)
DEF_CVT_SCALE_FUNC(16u64u,  cvt2_64f,ushort,    uint64_t, double)
DEF_CVT_SCALE_FUNC(32u64u,  cvt2_64f,uint,      uint64_t, double)
DEF_CVT_SCALE_FUNC(64u,     cvt2_64f,uint64_t,  uint64_t, double)
DEF_CVT_SCALE_FUNC(8s64u,   cvt2_64f,schar,     uint64_t, double)
DEF_CVT_SCALE_FUNC(16s64u,  cvt2_64f,short,     uint64_t, double)
DEF_CVT_SCALE_FUNC(32s64u,  cvt2_64f,int,       uint64_t, double)
DEF_CVT_SCALE_FUNC(64s64u,  cvt2_64f,int64_t,   uint64_t, double)
DEF_CVT_SCALE_FUNC(16f64u,  cvt2_64f,float16_t, uint64_t, double)
DEF_CVT_SCALE_FUNC(32f64u,  cvt2_64f,float,     uint64_t, double)
DEF_CVT_SCALE_FUNC(64f64u,  cvt2_64f,double,    uint64_t, double)

DEF_CVT_SCALE_FUNC(8u8s,   cvt_32f, uchar,     schar, float)
DEF_CVT_SCALE_FUNC(16u8s,  cvt_32f, ushort,    schar, float)
DEF_CVT_SCALE_FUNC(32u8s,  cvt2_32f,uint,      schar, float)
DEF_CVT_SCALE_FUNC(64u8s,  cvt2_64f,uint64_t,  schar, double)
DEF_CVT_SCALE_FUNC(8s,     cvt_32f, schar,     schar, float)
DEF_CVT_SCALE_FUNC(16s8s,  cvt_32f, short,     schar, float)
DEF_CVT_SCALE_FUNC(32s8s,  cvt_32f, int,       schar, float)
DEF_CVT_SCALE_FUNC(64s8s,  cvt2_64f,int64_t,   schar, double)
DEF_CVT_SCALE_FUNC(16f8s,  cvt_32f, float16_t, schar, float)
DEF_CVT_SCALE_FUNC(32f8s,  cvt_32f, float,     schar, float)
/*
FIXME: This function should exist, and be utilized instead
DEF_CVT_SCALE_FUNC(64f8s,  cvt_64f, double,    schar, double)
*/
DEF_CVT_SCALE_FUNC(64f8s,  cvt_32f, double,    schar, float)

DEF_CVT_SCALE_FUNC(8u16s,   cvt_32f, uchar,     short, float)
DEF_CVT_SCALE_FUNC(16u16s,  cvt_32f, ushort,    short, float)
DEF_CVT_SCALE_FUNC(32u16s,  cvt2_32f,uint,      short, float)
DEF_CVT_SCALE_FUNC(64u16s,  cvt2_64f,uint64_t,  short, double)
DEF_CVT_SCALE_FUNC(8s16s,   cvt_32f, schar,     short, float)
DEF_CVT_SCALE_FUNC(16s,     cvt_32f, short,     short, float)
DEF_CVT_SCALE_FUNC(32s16s,  cvt_32f, int,       short, float)
DEF_CVT_SCALE_FUNC(64s16s,  cvt2_64f,int64_t,   short, double)
DEF_CVT_SCALE_FUNC(16f16s,  cvt_32f, float16_t, short, float)
DEF_CVT_SCALE_FUNC(32f16s,  cvt_32f, float,     short, float)
/*
FIXME: This function should exist, and be utilized instead
DEF_CVT_SCALE_FUNC(64f16s,  cvt_64f, double,    short, double)
*/
DEF_CVT_SCALE_FUNC(64f16s,  cvt_32f, double,    short, float)

DEF_CVT_SCALE_FUNC(8u32s,   cvt_32f, uchar,     int, float)
DEF_CVT_SCALE_FUNC(16u32s,  cvt_32f, ushort,    int, float)
DEF_CVT_SCALE_FUNC(32u32s,  cvt2_32f,uint,      int, float)
DEF_CVT_SCALE_FUNC(64u32s,  cvt2_64f,uint64_t,  int, double)
DEF_CVT_SCALE_FUNC(8s32s,   cvt_32f, schar,     int, float)
DEF_CVT_SCALE_FUNC(16s32s,  cvt_32f, short,     int, float)
DEF_CVT_SCALE_FUNC(32s,     cvt_64f, int,       int, double)
DEF_CVT_SCALE_FUNC(64s32s,  cvt2_64f,int64_t,   int, double)
DEF_CVT_SCALE_FUNC(16f32s,  cvt_32f, float16_t, int, float)
DEF_CVT_SCALE_FUNC(32f32s,  cvt_32f, float,     int, float)
DEF_CVT_SCALE_FUNC(64f32s,  cvt_64f, double,    int, double)

DEF_CVT_SCALE_FUNC(8u64s,   cvt2_64f,uchar,     int64_t, double)
DEF_CVT_SCALE_FUNC(16u64s,  cvt2_64f,ushort,    int64_t, double)
DEF_CVT_SCALE_FUNC(32u64s,  cvt2_64f,uint,      int64_t, double)
DEF_CVT_SCALE_FUNC(64u64s,  cvt2_64f,uint64_t,  int64_t, double)
DEF_CVT_SCALE_FUNC(8s64s,   cvt2_64f,schar,     int64_t, double)
DEF_CVT_SCALE_FUNC(16s64s,  cvt2_64f,short,     int64_t, double)
DEF_CVT_SCALE_FUNC(32s64s,  cvt2_64f,int,       int64_t, double)
DEF_CVT_SCALE_FUNC(64s,     cvt2_64f,int64_t,   int64_t, double)
DEF_CVT_SCALE_FUNC(16f64s,  cvt2_64f,float16_t, int64_t, double)
DEF_CVT_SCALE_FUNC(32f64s,  cvt2_64f,float,     int64_t, double)
DEF_CVT_SCALE_FUNC(64f64s,  cvt2_64f,double,    int64_t, double)

DEF_CVT_SCALE_FUNC(8u16f,   cvt1_32f,uchar,     float16_t, float)
DEF_CVT_SCALE_FUNC(16u16f,  cvt1_32f,ushort,    float16_t, float)
DEF_CVT_SCALE_FUNC(32u16f,  cvt2_32f,uint,      float16_t, float)
DEF_CVT_SCALE_FUNC(64u16f,  cvt2_64f,uint64_t,  float16_t, double)
DEF_CVT_SCALE_FUNC(8s16f,   cvt1_32f,schar,     float16_t, float)
DEF_CVT_SCALE_FUNC(16s16f,  cvt1_32f,short,     float16_t, float)
DEF_CVT_SCALE_FUNC(32s16f,  cvt1_32f,int,       float16_t, float)
DEF_CVT_SCALE_FUNC(64s16f,  cvt2_64f,int64_t,   float16_t, double)
DEF_CVT_SCALE_FUNC(16f,     cvt1_32f,float16_t, float16_t, float)
DEF_CVT_SCALE_FUNC(32f16f,  cvt1_32f,float,     float16_t, float)
DEF_CVT_SCALE_FUNC(64f16f,  cvt_64f, double,    float16_t, double)

DEF_CVT_SCALE_FUNC(8u32f,   cvt_32f, uchar,     float, float)
DEF_CVT_SCALE_FUNC(16u32f,  cvt_32f, ushort,    float, float)
DEF_CVT_SCALE_FUNC(32u32f,  cvt2_32f,uint,      float, float)
DEF_CVT_SCALE_FUNC(64u32f,  cvt2_64f,uint64_t,  float, double)
DEF_CVT_SCALE_FUNC(8s32f,   cvt_32f, schar,     float, float)
DEF_CVT_SCALE_FUNC(16s32f,  cvt_32f, short,     float, float)
DEF_CVT_SCALE_FUNC(32s32f,  cvt_32f, int,       float, float)
DEF_CVT_SCALE_FUNC(64s32f,  cvt2_64f,int64_t,   float, double)
DEF_CVT_SCALE_FUNC(16f32f,  cvt_32f, float16_t, float, float)
DEF_CVT_SCALE_FUNC(32f,     cvt_32f, float,     float, float)
DEF_CVT_SCALE_FUNC(64f32f,  cvt_64f, double,    float, double)

DEF_CVT_SCALE_FUNC(8u64f,   cvt_64f, uchar,     double, double)
DEF_CVT_SCALE_FUNC(16u64f,  cvt_64f, ushort,    double, double)
DEF_CVT_SCALE_FUNC(32u64f,  cvt2_64f,uint,      double, double)
DEF_CVT_SCALE_FUNC(64u64f,  cvt2_64f,uint64_t,  double, double)
DEF_CVT_SCALE_FUNC(8s64f,   cvt_64f, schar,     double, double)
DEF_CVT_SCALE_FUNC(16s64f,  cvt_64f, short,     double, double)
DEF_CVT_SCALE_FUNC(32s64f,  cvt_64f, int,       double, double)
DEF_CVT_SCALE_FUNC(64s64f,  cvt2_64f,int64_t,   double, double)
DEF_CVT_SCALE_FUNC(16f64f,  cvt_64f, float16_t, double, double)
DEF_CVT_SCALE_FUNC(32f64f,  cvt_64f, float,     double, double)
DEF_CVT_SCALE_FUNC(64f,     cvt_64f, double,    double, double)

static BinaryFunc getCvtScaleAbsFunc(int depth)
{
    static const std::map<int, BinaryFunc> cvtScaleAbsMap
    {
        {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs8u   )},
        {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs16u8u)},
        {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs32u8u)},
        {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs64u8u)},
        {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs8s8u )},
        {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs16s8u)},
        {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs32s8u)},
        {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs64s8u)},
        //{CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs16f8u)},
        {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs32f8u)},
        {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScaleAbs64f8u)},
    };

    return cvtScaleAbsMap.at(depth);
}

BinaryFunc getConvertScaleFunc(int sdepth, int ddepth)
{
    static const std::map<int, std::map<int, BinaryFunc> > cvtScaleMap
    {
        {CV_8U, {
            {CV_8U,  (BinaryFunc)(cvtScale8u)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u8u)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u8u)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u8u)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s8u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s8u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s8u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s8u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f8u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f8u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f8u)},
        }},
        {CV_16U, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u16u)},
            {CV_16U, (BinaryFunc)(cvtScale16u)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u16u)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u16u)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s16u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s16u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s16u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s16u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f16u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f16u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f16u)},
        }},
        {CV_32U, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u32u)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u32u)},
            {CV_32U, (BinaryFunc)(cvtScale32u)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u32u)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s32u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s32u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s32u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s32u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f32u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f32u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f32u)},
        }},
        {CV_64U, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u64u)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u64u)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u64u)},
            {CV_64U, (BinaryFunc)(cvtScale64u)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s64u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s64u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s64u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s64u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f64u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f64u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f64u)},
        }},
        {CV_8S, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u8s)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u8s)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u8s)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u8s)},
            {CV_8S,  (BinaryFunc)(cvtScale8s)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s8s)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s8s)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s8s)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f8s)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f8s)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f8s)},
        }},
        {CV_16S, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u16s)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u16s)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u16s)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u16s)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s16s)},
            {CV_16S, (BinaryFunc)(cvtScale16s)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s16s)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s16s)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f16s)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f16s)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f16s)},
        }},
        {CV_32S, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u32s)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u32s)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u32s)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u32s)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s32s)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s32s)},
            {CV_32S, (BinaryFunc)(cvtScale32s)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s32s)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f32s)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f32s)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f32s)},
        }},
        {CV_64S, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u64s)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u64s)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u64s)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u64s)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s64s)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s64s)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s64s)},
            {CV_64S, (BinaryFunc)(cvtScale64s)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f64s)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f64s)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f64s)},
        }},
        {CV_16F, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u16f)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u16f)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u16f)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u16f)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s16f)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s16f)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s16f)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s16f)},
            {CV_16F, (BinaryFunc)(cvtScale16f)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f16f)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f16f)},
        }},
        {CV_32F, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u32f)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u32f)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u32f)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u32f)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s32f)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s32f)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s32f)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s32f)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f32f)},
            {CV_32F, (BinaryFunc)(cvtScale32f)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvtScale64f32f)},
        }},
        {CV_64F, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED(cvtScale8u64f)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvtScale16u64f)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvtScale32u64f)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvtScale64u64f)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED(cvtScale8s64f)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvtScale16s64f)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvtScale32s64f)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvtScale64s64f)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvtScale16f64f)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvtScale32f64f)},
            {CV_64F, (BinaryFunc)(cvtScale64f)},
        }},
    };

    return cvtScaleMap.at(CV_MAT_DEPTH(ddepth)).at(CV_MAT_DEPTH(sdepth));
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
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_convertScaleAbs(_src, _dst, alpha, beta))

    Mat src = _src.getMat();
    int cn = src.channels();
    double scale[] = {alpha, beta};
    _dst.create( src.dims, src.size, CV_8UC(cn) );
    Mat dst = _dst.getMat();
    BinaryFunc func = getCvtScaleAbsFunc(src.depth());
    CV_Assert( func != NULL );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.ptr(), src.step, 0, 0, dst.ptr(), dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2] = {};
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
    CV_INSTRUMENT_REGION();

    double scale = 1, shift = 0;
    int type = _src.type(), depth = CV_MAT_DEPTH(type);

    if( rtype < 0 )
        rtype = _dst.fixedType() ? _dst.depth() : depth;

    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = MIN( a, b ), dmax = MAX( a, b );
        minMaxIdx( _src, &smin, &smax, 0, 0, _mask );
        scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
        if( rtype == CV_32F )
        {
            scale = (float)scale;
            shift = (float)dmin - (float)(smin*scale);
        }
        else
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
