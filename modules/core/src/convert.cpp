// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "convert.hpp"

namespace cv {

namespace hal {

void cvt16f32f( const float16_t* src, float* dst, int len )
{
    int j = 0;
#if CV_SIMD
    const int VECSZ = v_float32::nlanes;
    for( ; j < len; j += VECSZ )
    {
        if( j > len - VECSZ )
        {
            if( j == 0 )
                break;
            j = len - VECSZ;
        }
        v_store(dst + j, vx_load_expand(src + j));
    }
#endif
    for( ; j < len; j++ )
        dst[j] = (float)src[j];
}

void cvt32f16f( const float* src, float16_t* dst, int len )
{
    int j = 0;
#if CV_SIMD
    const int VECSZ = v_float32::nlanes;
    for( ; j < len; j += VECSZ )
    {
        if( j > len - VECSZ )
        {
            if( j == 0 )
                break;
            j = len - VECSZ;
        }
        v_pack_store(dst + j, vx_load(src + j));
    }
#endif
    for( ; j < len; j++ )
        dst[j] = float16_t(src[j]);
}

void addRNGBias32f( float* arr, const float* scaleBiasPairs, int len )
{
    // the loop is simple enough, so we let the compiler to vectorize it
    for( int i = 0; i < len; i++ )
        arr[i] += scaleBiasPairs[i*2 + 1];
}

void addRNGBias64f( double* arr, const double* scaleBiasPairs, int len )
{
    // the loop is simple enough, so we let the compiler to vectorize it
    for( int i = 0; i < len; i++ )
        arr[i] += scaleBiasPairs[i*2 + 1];
}

}

template<typename _Ts, typename _Td, typename _Twvec> inline void
cvt_( const _Ts* src, size_t sstep, _Td* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if CV_SIMD
        const int VECSZ = _Twvec::nlanes*2;
        for( ; j < size.width; j += VECSZ )
        {
            if( j > size.width - VECSZ )
            {
                if( j == 0 || src == (_Ts*)dst )
                    break;
                j = size.width - VECSZ;
            }
            _Twvec v0, v1;
            vx_load_pair_as(src + j, v0, v1);
            v_store_pair_as(dst + j, v0, v1);
        }
#endif
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(src[j]);
    }
}

// in order to reduce the code size, for (16f <-> ...) conversions
// we add a conversion function without loop unrolling
template<typename _Ts, typename _Td, typename _Twvec> inline void
cvt1_( const _Ts* src, size_t sstep, _Td* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if CV_SIMD
        const int VECSZ = _Twvec::nlanes;
        for( ; j < size.width; j += VECSZ )
        {
            if( j > size.width - VECSZ )
            {
                if( j == 0 || src == (_Ts*)dst )
                    break;
                j = size.width - VECSZ;
            }
            _Twvec v;
            vx_load_as(src + j, v);
            v_store_as(dst + j, v);
        }
        vx_cleanup();
#endif
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(src[j]);
    }
}

// without optimization
template<typename _Ts, typename _Td, typename _Twvec> inline void
cvt2_( const _Ts* src, size_t sstep, _Td* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
        for (; j < size.width; j++)
        {
            dst[j] = saturate_cast<_Td>(src[j]);
        }
    }
}

static void cvtCopy( const uchar* src, size_t sstep,
                     uchar* dst, size_t dstep, Size size, size_t elemsize)
{
    size_t len = size.width*elemsize;
    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        memcpy( dst, src, len );
    }
}

#define DEF_CVT_FUNC(suffix, cvtfunc, _Ts, _Td, _Twvec) \
static void cvt##suffix(const _Ts* src, size_t sstep, uchar*, size_t, \
                        _Td* dst, size_t dstep, Size size, void*) \
{ cvtfunc<_Ts, _Td, _Twvec>(src, sstep, dst, dstep, size); }

////////////////////// 8u -> ... ////////////////////////

DEF_CVT_FUNC(8u16u, cvt_,  uchar,  ushort,    v_uint16)
DEF_CVT_FUNC(8u32u, cvt2_, uchar,  uint,      v_uint32)
DEF_CVT_FUNC(8u64u, cvt2_, uchar,  uint64_t,  v_uint64)
DEF_CVT_FUNC(8u8s,  cvt_,  uchar,  schar,     v_int16)
DEF_CVT_FUNC(8u16s, cvt_,  uchar,  short,     v_int16)
DEF_CVT_FUNC(8u32s, cvt_,  uchar,  int,       v_int32)
DEF_CVT_FUNC(8u64s, cvt2_, uchar,  int64_t,   v_int64)
DEF_CVT_FUNC(8u32f, cvt_,  uchar,  float,     v_float32)
DEF_CVT_FUNC(8u64f, cvt_,  uchar,  double,    v_int32)
DEF_CVT_FUNC(8u16f, cvt1_, uchar,  float16_t, v_float32)

////////////////////// 16u -> ... ////////////////////////

DEF_CVT_FUNC(16u8u,  cvt_, ushort, uchar,     v_uint16)
DEF_CVT_FUNC(16u32u, cvt2_,ushort, uint,      v_uint32)
DEF_CVT_FUNC(16u64u, cvt2_,ushort, uint64_t,  v_uint64)
DEF_CVT_FUNC(16u8s,  cvt_, ushort, schar,     v_uint16)
DEF_CVT_FUNC(16u16s, cvt_, ushort, short,     v_int32)
DEF_CVT_FUNC(16u32s, cvt_, ushort, int,       v_int32)
DEF_CVT_FUNC(16u64s, cvt2_,ushort, int64_t,   v_int64)
DEF_CVT_FUNC(16u32f, cvt_, ushort, float,     v_float32)
DEF_CVT_FUNC(16u64f, cvt_, ushort, double,    v_int32)
DEF_CVT_FUNC(16u16f, cvt1_,ushort, float16_t, v_float32)

////////////////////// 32u -> ... ////////////////////////

DEF_CVT_FUNC(32u8u,  cvt2_,uint,   uchar,     v_uint32)
DEF_CVT_FUNC(32u16u, cvt2_,uint,   ushort,    v_uint32)
DEF_CVT_FUNC(32u64u, cvt2_,uint,   uint64_t,  v_uint64)
DEF_CVT_FUNC(32u8s,  cvt2_,uint,   schar,     v_int32)
DEF_CVT_FUNC(32u16s, cvt2_,uint,   short,     v_int32)
DEF_CVT_FUNC(32u32s, cvt2_,uint,   int,       v_int32)
DEF_CVT_FUNC(32u64s, cvt2_,uint,   int64_t,   v_int64)
DEF_CVT_FUNC(32u32f, cvt2_,uint,   float,     v_float32)
DEF_CVT_FUNC(32u64f, cvt2_,uint,   double,    v_int32)
DEF_CVT_FUNC(32u16f, cvt2_,uint,   float16_t, v_float32)

////////////////////// 64u -> ... ////////////////////////

DEF_CVT_FUNC(64u8u,  cvt2_,uint64_t, uchar,     v_uint64)
DEF_CVT_FUNC(64u16u, cvt2_,uint64_t, ushort,    v_uint64)
DEF_CVT_FUNC(64u32u, cvt2_,uint64_t, uint,      v_uint64)
DEF_CVT_FUNC(64u8s,  cvt2_,uint64_t, schar,     v_uint64)
DEF_CVT_FUNC(64u16s, cvt2_,uint64_t, short,     v_int64)
DEF_CVT_FUNC(64u32s, cvt2_,uint64_t, int,       v_int64)
DEF_CVT_FUNC(64u64s, cvt2_,uint64_t, int64_t,   v_int64)
DEF_CVT_FUNC(64u32f, cvt2_,uint64_t, float,     v_float32)
DEF_CVT_FUNC(64u64f, cvt2_,uint64_t, double,    v_int64)
DEF_CVT_FUNC(64u16f, cvt2_,uint64_t, float16_t, v_float32)

////////////////////// 8s -> ... ////////////////////////

DEF_CVT_FUNC(8s8u,  cvt_,  schar,  uchar,     v_int16)
DEF_CVT_FUNC(8s16u, cvt_,  schar,  ushort,    v_uint16)
DEF_CVT_FUNC(8s32u, cvt2_, schar,  uint,      v_uint32)
DEF_CVT_FUNC(8s64u, cvt2_, schar,  uint64_t,  v_uint64)
DEF_CVT_FUNC(8s16s, cvt_,  schar,  short,     v_int16)
DEF_CVT_FUNC(8s32s, cvt_,  schar,  int,       v_int32)
DEF_CVT_FUNC(8s64s, cvt2_, schar,  int64_t,   v_int64)
DEF_CVT_FUNC(8s32f, cvt_,  schar,  float,     v_float32)
DEF_CVT_FUNC(8s64f, cvt_,  schar,  double,    v_int32)
DEF_CVT_FUNC(8s16f, cvt1_, schar,  float16_t, v_float32)

////////////////////// 16s -> ... ////////////////////////

DEF_CVT_FUNC(16s8u,  cvt_, short, uchar,     v_int16)
DEF_CVT_FUNC(16s16u, cvt_, short, ushort,    v_int32)
DEF_CVT_FUNC(16s32u, cvt2_,short, uint,      v_int64)
DEF_CVT_FUNC(16s64u, cvt2_,short, uint64_t,  v_int64)
DEF_CVT_FUNC(16s8s,  cvt_, short, schar,     v_int16)
DEF_CVT_FUNC(16s32s, cvt_, short, int,       v_int32)
DEF_CVT_FUNC(16s64s, cvt2_,short, int64_t,   v_int64)
DEF_CVT_FUNC(16s32f, cvt_, short, float,     v_float32)
DEF_CVT_FUNC(16s64f, cvt_, short, double,    v_int32)
DEF_CVT_FUNC(16s16f, cvt1_,short, float16_t, v_float32)

////////////////////// 32s -> ... ////////////////////////

DEF_CVT_FUNC(32s8u,  cvt_, int, uchar,     v_int32)
DEF_CVT_FUNC(32s16u, cvt_, int, ushort,    v_int32)
DEF_CVT_FUNC(32s32u, cvt2_,int, uint,      v_int32)
DEF_CVT_FUNC(32s64u, cvt2_,int, uint64_t,  v_int64)
DEF_CVT_FUNC(32s8s,  cvt_, int, schar,     v_int32)
DEF_CVT_FUNC(32s16s, cvt_, int, short,     v_int32)
DEF_CVT_FUNC(32s64s, cvt2_,int, int64_t,   v_int64)
DEF_CVT_FUNC(32s32f, cvt_, int, float,     v_float32)
DEF_CVT_FUNC(32s64f, cvt_, int, double,    v_int32)
DEF_CVT_FUNC(32s16f, cvt1_,int, float16_t, v_float32)

////////////////////// 64s -> ... ////////////////////////

DEF_CVT_FUNC(64s8u,  cvt2_,int64_t, uchar,     v_int64)
DEF_CVT_FUNC(64s16u, cvt2_,int64_t, ushort,    v_int64)
DEF_CVT_FUNC(64s32u, cvt2_,int64_t, uint,      v_int64)
DEF_CVT_FUNC(64s64u, cvt2_,int64_t, uint64_t,  v_int64)
DEF_CVT_FUNC(64s8s,  cvt2_,int64_t, schar,     v_int64)
DEF_CVT_FUNC(64s16s, cvt2_,int64_t, short,     v_int64)
DEF_CVT_FUNC(64s32s, cvt2_,int64_t, int,       v_int64)
DEF_CVT_FUNC(64s32f, cvt2_,int64_t, float,     v_float32)
DEF_CVT_FUNC(64s64f, cvt2_,int64_t, double,    v_int32)
DEF_CVT_FUNC(64s16f, cvt2_,int64_t, float16_t, v_float32)

////////////////////// 32f -> ... ////////////////////////

DEF_CVT_FUNC(32f8u,  cvt_, float, uchar,     v_float32)
DEF_CVT_FUNC(32f16u, cvt_, float, ushort,    v_float32)
DEF_CVT_FUNC(32f32u, cvt2_,float, uint,      v_float32)
DEF_CVT_FUNC(32f64u, cvt2_,float, uint64_t,  v_float32)
DEF_CVT_FUNC(32f8s,  cvt_, float, schar,     v_float32)
DEF_CVT_FUNC(32f16s, cvt_, float, short,     v_float32)
DEF_CVT_FUNC(32f32s, cvt_, float, int,       v_float32)
DEF_CVT_FUNC(32f64s, cvt2_,float, int64_t,   v_float32)
DEF_CVT_FUNC(32f64f, cvt_, float, double,    v_float32)
DEF_CVT_FUNC(32f16f, cvt1_,float, float16_t, v_float32)

////////////////////// 64f -> ... ////////////////////////

DEF_CVT_FUNC(64f8u,  cvt_, double, uchar,     v_int32)
DEF_CVT_FUNC(64f16u, cvt_, double, ushort,    v_int32)
DEF_CVT_FUNC(64f32u, cvt2_,double, uint,      v_int32)
DEF_CVT_FUNC(64f64u, cvt2_,double, uint64_t,  v_int64)
DEF_CVT_FUNC(64f8s,  cvt_, double, schar,     v_int32)
DEF_CVT_FUNC(64f16s, cvt_, double, short,     v_int32)
DEF_CVT_FUNC(64f32s, cvt_, double, int,       v_int32)
DEF_CVT_FUNC(64f64s, cvt2_,double, int64_t,   v_int64)
DEF_CVT_FUNC(64f32f, cvt_, double, float,     v_float32)
DEF_CVT_FUNC(64f16f, cvt1_,double, float16_t, v_float32)

////////////////////// 16f -> ... ////////////////////////

DEF_CVT_FUNC(16f8u,  cvt_,  float16_t, uchar,    v_float32)
DEF_CVT_FUNC(16f16u, cvt1_, float16_t, ushort,   v_float32)
DEF_CVT_FUNC(16f32u, cvt2_, float16_t, uint,     v_float32)
DEF_CVT_FUNC(16f64u, cvt2_, float16_t, uint64_t, v_float32)
DEF_CVT_FUNC(16f8s,  cvt_,  float16_t, schar,    v_float32)
DEF_CVT_FUNC(16f16s, cvt1_, float16_t, short,    v_float32)
DEF_CVT_FUNC(16f32s, cvt1_, float16_t, int,      v_float32)
DEF_CVT_FUNC(16f64s, cvt2_, float16_t, int64_t,  v_float32)
DEF_CVT_FUNC(16f32f, cvt1_, float16_t, float,    v_float32)
DEF_CVT_FUNC(16f64f, cvt1_, float16_t, double,   v_float32)

///////////// "conversion" w/o conversion ///////////////

static void cvt8u(const uchar* src, size_t sstep, uchar*, size_t, uchar* dst, size_t dstep, Size size, void*)
{ cvtCopy(src, sstep, dst, dstep, size, 1); }

static void cvt16u(const ushort* src, size_t sstep, uchar*, size_t, ushort* dst, size_t dstep, Size size, void*)
{ cvtCopy((const uchar*)src, sstep, (uchar*)dst, dstep, size, 2); }

static void cvt32u(const uint* src, size_t sstep, uchar*, size_t, uint* dst, size_t dstep, Size size, void*)
{ cvtCopy((const uchar*)src, sstep, (uchar*)dst, dstep, size, 4); }

static void cvt64u(const uint64_t* src, size_t sstep, uchar*, size_t, uint64_t* dst, size_t dstep, Size size, void*)
{ cvtCopy((const uchar*)src, sstep, (uchar*)dst, dstep, size, 8); }

/* [TODO] Recover IPP calls
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
*/

BinaryFunc getConvertFunc(int sdepth, int ddepth)
{
    static const std::map<int, std::map<int, BinaryFunc> > cvtMap
    {
        {CV_8U, {
            {CV_8U,  (BinaryFunc)             (   cvt8u)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u8u)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u8u)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u8u)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s8u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s8u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s8u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s8u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f8u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f8u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f8u)},
        }},
        {CV_16U, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u16u)},
            {CV_16U, (BinaryFunc)             (   cvt16u)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u16u)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u16u)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s16u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s16u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s16u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s16u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f16u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f16u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f16u)},
        }},
        {CV_32U, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u32u)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u32u)},
            {CV_32U, (BinaryFunc)             (   cvt32u)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u32u)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s32u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s32u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s32u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s32u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f32u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f32u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f32u)},
        }},
        {CV_64U, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u64u)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u64u)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u64u)},
            {CV_64U, (BinaryFunc)             (   cvt64u)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s64u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s64u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s64u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s64u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f64u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f64u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f64u)},
        }},
        {CV_8S, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u8s)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u8s)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u8s)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u8s)},
            {CV_8S,  (BinaryFunc)             (   cvt8u)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s8s)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s8s)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s8s)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f8s)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f8s)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f8s)},
        }},
        {CV_16S, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u16s)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u16s)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u16s)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u16s)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s16s)},
            {CV_16S, (BinaryFunc)             (   cvt16u)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s16s)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s16s)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f16s)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f16s)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f16s)},
        }},
        {CV_32S, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u32s)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u32s)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u32s)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u32s)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s32s)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s32s)},
            {CV_32S, (BinaryFunc)             (   cvt32u)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s32s)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f32s)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f32s)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f32s)},
        }},
        {CV_64S, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u64s)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u64s)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u64s)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u64s)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s64s)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s64s)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s64s)},
            {CV_64S, (BinaryFunc)             (   cvt64u)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f64s)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f64s)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f64s)},
        }},
        {CV_16F, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u16f)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u16f)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u16f)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u16f)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s16f)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s16f)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s16f)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s16f)},
            {CV_16F, (BinaryFunc)             (   cvt16u)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f16f)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f16f)},
        }},
        {CV_32F, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u32f)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u32f)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u32f)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u32f)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s32f)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s32f)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s32f)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s32f)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f32f)},
            {CV_32F, (BinaryFunc)             (   cvt32u)},
            {CV_64F, (BinaryFunc)GET_OPTIMIZED(cvt64f32f)},
        }},
        {CV_64F, {
            {CV_8U,  (BinaryFunc)GET_OPTIMIZED( cvt8u64f)},
            {CV_16U, (BinaryFunc)GET_OPTIMIZED(cvt16u64f)},
            {CV_32U, (BinaryFunc)GET_OPTIMIZED(cvt32u64f)},
            {CV_64U, (BinaryFunc)GET_OPTIMIZED(cvt64u64f)},
            {CV_8S,  (BinaryFunc)GET_OPTIMIZED( cvt8s64f)},
            {CV_16S, (BinaryFunc)GET_OPTIMIZED(cvt16s64f)},
            {CV_32S, (BinaryFunc)GET_OPTIMIZED(cvt32s64f)},
            {CV_64S, (BinaryFunc)GET_OPTIMIZED(cvt64s64f)},
            {CV_16F, (BinaryFunc)GET_OPTIMIZED(cvt16f64f)},
            {CV_32F, (BinaryFunc)GET_OPTIMIZED(cvt32f64f)},
            {CV_64F, (BinaryFunc)             (   cvt64u)},
        }},
    };

    return cvtMap.at(CV_MAT_DEPTH(ddepth)).at(CV_MAT_DEPTH(sdepth));
}

#ifdef HAVE_OPENCL
static bool ocl_convertFp16( InputArray _src, OutputArray _dst, int sdepth, int ddepth )
{
    int type = _src.type(), cn = CV_MAT_CN(type);

    _dst.createSameSize( _src, CV_MAKETYPE(ddepth, cn) );
    int kercn = 1;
    int rowsPerWI = 1;
    String build_opt = format("-D HALF_SUPPORT -D srcT=%s -D dstT=%s -D rowsPerWI=%d%s",
                              sdepth == CV_32F ? "float" : "half",
                              sdepth == CV_32F ? "half" : "float",
                              rowsPerWI,
                              sdepth == CV_32F ? " -D FLOAT_TO_HALF " : "");
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

} // cv::

void cv::Mat::convertTo(OutputArray _dst, int _type, double alpha, double beta) const
{
    CV_INSTRUMENT_REGION();

    if( empty() )
    {
        _dst.release();
        return;
    }

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


    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);
    double scale[] = {alpha, beta};
    int cn = channels();
    if (func == NULL)
        CV_Error_(Error::StsNotImplemented, ("Convertor from %s to %s", depthToString(sdepth), depthToString(ddepth)));

    if( dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2] = {};
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, scale);
    }
}

//==================================================================================================

void cv::convertFp16( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int sdepth = _src.depth(), ddepth = 0;
    BinaryFunc func = 0;

    switch( sdepth )
    {
    case CV_32F:
        if(_dst.fixedType())
        {
            ddepth = _dst.depth();
            CV_Assert(ddepth == CV_16S || ddepth == CV_16F);
            CV_Assert(_dst.channels() == _src.channels());
        }
        else
            ddepth =  CV_16S;
        func = (BinaryFunc)cvt32f16f;
        break;
    case CV_16S:
    case CV_16F:
        ddepth = CV_32F;
        func = (BinaryFunc)cvt16f32f;
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported input depth");
        return;
    }

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_convertFp16(_src, _dst, sdepth, ddepth))

    Mat src = _src.getMat();

    int type = CV_MAKETYPE(ddepth, src.channels());
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();
    int cn = src.channels();

    CV_Assert( func != NULL );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, 0);
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2] = {};
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 0, 0, 0, ptrs[1], 0, sz, 0);
    }
}
