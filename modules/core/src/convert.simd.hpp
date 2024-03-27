// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "convert.hpp"

#if !defined(OPENCV_SUPRESS_WARNING_AVX2_WITHOUT_FP16C) && \
    (defined(__GNUC__) && defined(__AVX2__) && !defined(__F16C__))
#warning "Non-optimal compiler flags: AVX2 without FP16. Generated code is very slow. Consider adding '-mf16c' compiler option."
#endif

namespace cv {
namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void cvt16f32f(const hfloat* src, float* dst, int len);
void cvt32f16f(const float* src, hfloat* dst, int len);
void cvt16bf32f(const bfloat* src, float* dst, int len);
void cvt32f16bf(const float* src, bfloat* dst, int len);
void addRNGBias32f(float* arr, const float* scaleBiasPairs, int len, int cn);
void addRNGBias64f(double* arr, const double* scaleBiasPairs, int len, int cn);

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv::hal

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

BinaryFunc getConvertFunc(int sdepth, int ddepth);

CV_CPU_OPTIMIZATION_NAMESPACE_END

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

BinaryFunc getConvertFunc(int sdepth, int ddepth);

void cvt16f32f( const hfloat* src, float* dst, int len )
{
    CV_INSTRUMENT_REGION();
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
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

void cvt32f16f( const float* src, hfloat* dst, int len )
{
    CV_INSTRUMENT_REGION();
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
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
        dst[j] = hfloat(src[j]);
}

void cvt32f16bf( const float* src, bfloat* dst, int len )
{
    CV_INSTRUMENT_REGION();
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
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
        dst[j] = bfloat(src[j]);
}

void addRNGBias32f( float* arr, const float* scaleBiasPairs, int len, int cn )
{
    CV_INSTRUMENT_REGION();
    if (cn == 1) {
        float bias = scaleBiasPairs[1];
        for( int i = 0; i < len; i++ ) {
            arr[i] += bias;
        }
    } else {
        int k = 0;
        len *= cn;
        cn--;
        for( int i = 0; i < len; i++ ) {
            arr[i] += scaleBiasPairs[k*2 + 1];
            k = (k + 1) & ((k >= cn) - 1);
        }
    }
}

void addRNGBias64f( double* arr, const double* scaleBiasPairs, int len, int cn )
{
    CV_INSTRUMENT_REGION();
    if (cn == 1) {
        double bias = scaleBiasPairs[1];
        for( int i = 0; i < len; i++ ) {
            arr[i] += bias;
        }
    } else {
        int k = 0;
        len *= cn;
        cn--;
        for( int i = 0; i < len; i++ ) {
            arr[i] += scaleBiasPairs[k*2 + 1];
            k = (k + 1) & ((k >= cn) - 1);
        }
    }
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv::hal

// cv::
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

template<typename _Ts, typename _Td, typename _Twvec> static inline void
cvt_( const _Ts* src, size_t sstep, _Td* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int VECSZ = VTraits<_Twvec>::vlanes()*2;
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

template<typename _Ts, typename _Td, typename dummy> static inline void
cvt_64f( const _Ts* src, size_t sstep, _Td* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        const int VECSZ = VTraits<v_float64>::vlanes()*2;
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
            v_store_pair_as(dst + j, v0, v1);
        }
#endif
        for( ; j < size.width; j++ )
            dst[j] = saturate_cast<_Td>(src[j]);
    }
}

// in order to reduce the code size, for (16f <-> ...) conversions
// we add a conversion function without loop unrolling
template<typename _Ts, typename _Td, typename _Twvec> static inline void
cvt1_( const _Ts* src, size_t sstep, _Td* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep )
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int VECSZ = VTraits<_Twvec>::vlanes();
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
static void cvt##suffix(const uchar* src_, size_t sstep, const uchar*, size_t, \
                        uchar* dst_, size_t dstep, Size size, void*) \
{ \
    CV_INSTRUMENT_REGION(); \
    const _Ts* src = (const _Ts*)src_; \
    _Td* dst = (_Td*)dst_; \
    cvtfunc<_Ts, _Td, _Twvec>(src, sstep, dst, dstep, size); \
}

#define DEF_CVT2BOOL_FUNC(suffix, _Ts, shift) \
static void cvt##suffix(const uchar* src_, size_t sstep, const uchar*, size_t, \
                        uchar* dst, size_t dstep, Size size, void*) \
{ \
    CV_INSTRUMENT_REGION(); \
    const _Ts* src = (const _Ts*)src_; \
    sstep /= sizeof(src[0]); \
    \
    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep ) { \
        for ( int j = 0; j < size.width; j++ ) \
            dst[j] = (src[j]<<shift) != 0; \
    } \
}

#define DEF_CVTBOOL2_FUNC(suffix, _Td, scale) \
static void cvt##suffix(const uchar* src, size_t sstep, const uchar*, size_t, \
                        uchar* dst_, size_t dstep, Size size, void*) \
{ \
    CV_INSTRUMENT_REGION(); \
    _Td* dst = (_Td*)dst_; \
    dstep /= sizeof(dst[0]); \
    \
    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep ) { \
        for ( int j = 0; j < size.width; j++ ) \
            dst[j] = (_Td)((src[j] != 0)*scale); \
    } \
}

#define DEF_CVT_SCALAR_FUNC(suffix, _Ts, _Td) \
static void cvt##suffix(const uchar* src_, size_t sstep, const uchar*, size_t, \
                        uchar* dst_, size_t dstep, Size size, void*) \
{ \
    CV_INSTRUMENT_REGION(); \
    const _Ts* src = (const _Ts*)src_; \
    _Td* dst = (_Td*)dst_; \
    sstep /= sizeof(src[0]); \
    dstep /= sizeof(dst[0]); \
    \
    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep ) { \
        for ( int j = 0; j < size.width; j++ ) \
            dst[j] = saturate_cast<_Td>(src[j]); \
    } \
}

#define DEF_CVT_SCALAR_FUNC_S2U(suffix, _Ts, _Td, _Tw) \
static void cvt##suffix(const uchar* src_, size_t sstep, const uchar*, size_t, \
                        uchar* dst_, size_t dstep, Size size, void*) \
{ \
    CV_INSTRUMENT_REGION(); \
    const _Ts* src = (const _Ts*)src_; \
    _Td* dst = (_Td*)dst_; \
    sstep /= sizeof(src[0]); \
    dstep /= sizeof(dst[0]); \
    \
    for( int i = 0; i < size.height; i++, src += sstep, dst += dstep ) { \
        for ( int j = 0; j < size.width; j++ ) \
            dst[j] = saturate_cast<_Td>(std::max((_Tw)src[j], (_Tw)0)); \
    } \
}

////////////////////// 8u -> ... ////////////////////////

DEF_CVT_FUNC(8u8s,  cvt_,  uchar, schar,    v_int16)
DEF_CVT_FUNC(8u16s, cvt_,  uchar, short,    v_int16)
DEF_CVT_FUNC(8u32s, cvt_,  uchar, int,      v_int32)
DEF_CVT_FUNC(8u32f, cvt_,  uchar, float,    v_float32)
DEF_CVT_FUNC(8u64f, cvt_,  uchar, double,   v_int32)
DEF_CVT_SCALAR_FUNC(8u64s, uchar, int64_t)
DEF_CVT_FUNC(8u16f, cvt1_, uchar, hfloat, v_float32)
DEF_CVT_FUNC(8u16bf, cvt1_, uchar, bfloat, v_float32)
DEF_CVT2BOOL_FUNC(8u8b, uchar, 0)

////////////////////// 8s -> ... ////////////////////////

DEF_CVT_FUNC(8s8u,  cvt_,  schar, uchar,    v_int16)
DEF_CVT_FUNC(8s16u, cvt_,  schar, ushort,   v_uint16)
DEF_CVT_FUNC(8s16s, cvt_,  schar, short,    v_int16)
DEF_CVT_FUNC(8s32u, cvt_,  schar, unsigned, v_uint32)
DEF_CVT_FUNC(8s32s, cvt_,  schar, int,      v_int32)
DEF_CVT_FUNC(8s32f, cvt_,  schar, float,    v_float32)
DEF_CVT_FUNC(8s64f, cvt_,  schar, double,   v_int32)
DEF_CVT_FUNC(8s64u, cvt_,  schar, uint64_t, v_uint32)
DEF_CVT_FUNC(8s64s, cvt_,  schar, int64_t,  v_int32)
DEF_CVT_FUNC(8s16f, cvt1_, schar, hfloat, v_float32)
DEF_CVT_FUNC(8s16bf, cvt1_, schar, bfloat, v_float32)

////////////////////// 8b -> ... ////////////////////////

DEF_CVTBOOL2_FUNC(8b8u,  uchar, 1)
DEF_CVTBOOL2_FUNC(8b16s, short, 1)
DEF_CVTBOOL2_FUNC(8b32s, int, 1)
DEF_CVTBOOL2_FUNC(8b32f, float, 1)
DEF_CVTBOOL2_FUNC(8b64f, double, 1)
DEF_CVTBOOL2_FUNC(8b64s, int64_t, 1)
DEF_CVTBOOL2_FUNC(8b16f, uint16_t, 0x3c00) // hfloat(1.0f)
DEF_CVTBOOL2_FUNC(8b16bf, uint16_t, 0x3f80) // bfloat(1.0f)

////////////////////// 16u -> ... ////////////////////////

DEF_CVT_FUNC(16u8u,  cvt_, ushort, uchar,  v_uint16)
DEF_CVT_FUNC(16u8s,  cvt_, ushort, schar,  v_uint16)
DEF_CVT_FUNC(16u16s, cvt_, ushort, short,  v_int32)
DEF_CVT_FUNC(16u32s, cvt_, ushort, int,    v_int32)
DEF_CVT_FUNC(16u32f, cvt_, ushort, float,  v_float32)
DEF_CVT_FUNC(16u64f, cvt_, ushort, double, v_int32)
DEF_CVT_SCALAR_FUNC(16u64s, ushort, int64_t)
DEF_CVT_FUNC(16u16f, cvt1_,ushort, hfloat, v_float32)
DEF_CVT_FUNC(16u16bf, cvt1_, ushort, bfloat, v_float32)

////////////////////// 16s -> ... ////////////////////////

DEF_CVT_FUNC(16s8u,  cvt_, short, uchar,  v_int16)
DEF_CVT_FUNC(16s8s,  cvt_, short, schar,  v_int16)
DEF_CVT_FUNC(16s16u, cvt_, short, ushort, v_int32)
DEF_CVT_FUNC(16s32u, cvt_, short, unsigned, v_uint32)
DEF_CVT_FUNC(16s32s, cvt_, short, int,    v_int32)
DEF_CVT_FUNC(16s32f, cvt_, short, float,  v_float32)
DEF_CVT_FUNC(16s64f, cvt_, short, double, v_int32)
DEF_CVT_FUNC(16s64u, cvt_, short, uint64_t, v_uint32)
DEF_CVT_FUNC(16s64s, cvt_, short, int64_t, v_int32)
DEF_CVT_FUNC(16s16f, cvt1_,short, hfloat, v_float32)
DEF_CVT_FUNC(16s16bf, cvt1_, short, bfloat, v_float32)
DEF_CVT2BOOL_FUNC(16s8b, short, 0)

////////////////////// 32u -> ... ////////////////////////

DEF_CVT_FUNC(32u8u,  cvt_, unsigned, uchar,  v_uint32)
DEF_CVT_FUNC(32u8s,  cvt_, unsigned, schar,  v_int32)
DEF_CVT_FUNC(32u16u, cvt_, unsigned, ushort, v_uint32)
DEF_CVT_FUNC(32u16s, cvt_, unsigned, short,  v_int32)
DEF_CVT_SCALAR_FUNC(32u32s, unsigned, int)
DEF_CVT_FUNC(32u32f, cvt_, unsigned, float,  v_float32)
DEF_CVT_FUNC(32u64f, cvt_, unsigned, double, v_float32)
DEF_CVT_SCALAR_FUNC(32u64s, unsigned, int64_t)
DEF_CVT_FUNC(32u16f, cvt1_, unsigned, hfloat, v_float32)
DEF_CVT_FUNC(32u16bf, cvt1_, int, bfloat, v_float32)

////////////////////// 32s -> ... ////////////////////////

DEF_CVT_FUNC(32s8u,  cvt_, int, uchar,  v_int32)
DEF_CVT_FUNC(32s8s,  cvt_, int, schar,  v_int32)
DEF_CVT_FUNC(32s16u, cvt_, int, ushort, v_int32)
DEF_CVT_FUNC(32s16s, cvt_, int, short,  v_int32)
DEF_CVT_FUNC(32s32u, cvt_, int, unsigned, v_uint32)
DEF_CVT_FUNC(32s32f, cvt_, int, float,  v_float32)
DEF_CVT_FUNC(32s64f, cvt_, int, double, v_int32)
DEF_CVT_FUNC(32s64u, cvt_, int, uint64_t, v_uint32)
DEF_CVT_FUNC(32s64s, cvt_, int, int64_t, v_int32)
DEF_CVT_FUNC(32s16f, cvt1_,int, hfloat, v_float32)
DEF_CVT_FUNC(32s16bf, cvt1_, int, bfloat, v_float32)
DEF_CVT2BOOL_FUNC(32s8b, int, 0)

////////////////////// 32f -> ... ////////////////////////

DEF_CVT_FUNC(32f8u,  cvt_, float, uchar,  v_float32)
DEF_CVT_FUNC(32f8s,  cvt_, float, schar,  v_float32)
DEF_CVT_FUNC(32f16u, cvt_, float, ushort, v_float32)
DEF_CVT_FUNC(32f16s, cvt_, float, short,  v_float32)
DEF_CVT_FUNC(32f32u, cvt_, float, unsigned, v_float32)
DEF_CVT_FUNC(32f32s, cvt_, float, int,    v_float32)
DEF_CVT_FUNC(32f64f, cvt_, float, double, v_float32)
DEF_CVT_FUNC(32f64u, cvt_64f, float, uint64_t, v_float64)
DEF_CVT_FUNC(32f64s, cvt_64f, float, int64_t, v_float64)
DEF_CVT_FUNC(32f16f, cvt1_,float, hfloat, v_float32)
DEF_CVT_FUNC(32f16bf, cvt1_,float, bfloat, v_float32)
DEF_CVT2BOOL_FUNC(32f8b, int, 1)

////////////////////// 64f -> ... ////////////////////////

DEF_CVT_FUNC(64f8u,  cvt_, double, uchar,  v_int32)
DEF_CVT_FUNC(64f8s,  cvt_, double, schar,  v_int32)
DEF_CVT_FUNC(64f16u, cvt_, double, ushort, v_int32)
DEF_CVT_FUNC(64f16s, cvt_, double, short,  v_int32)
DEF_CVT_FUNC(64f32u, cvt_64f, double, unsigned, v_float32)
DEF_CVT_FUNC(64f32s, cvt_, double, int,    v_int32)
DEF_CVT_FUNC(64f32f, cvt_, double, float,  v_float32)
DEF_CVT_FUNC(64f64u, cvt_64f, double, uint64_t, v_float64)
DEF_CVT_FUNC(64f64s, cvt_64f, double, int64_t, v_float32)
DEF_CVT_FUNC(64f16f, cvt1_,double, hfloat, v_float32)
DEF_CVT_FUNC(64f16bf, cvt1_,double, bfloat, v_float32)
DEF_CVT2BOOL_FUNC(64f8b, int64_t, 1)

////////////////////// 16f -> ... ////////////////////////

DEF_CVT_FUNC(16f8u,  cvt_,  hfloat, uchar,  v_float32)
DEF_CVT_FUNC(16f8s,  cvt_,  hfloat, schar,  v_float32)
DEF_CVT_FUNC(16f16u, cvt1_, hfloat, ushort, v_float32)
DEF_CVT_FUNC(16f16s, cvt1_, hfloat, short,  v_float32)
DEF_CVT_FUNC(16f32u, cvt1_, hfloat, unsigned, v_float32)
DEF_CVT_FUNC(16f32s, cvt1_, hfloat, int,    v_float32)
DEF_CVT_FUNC(16f32f, cvt1_, hfloat, float,  v_float32)
DEF_CVT_FUNC(16f64f, cvt1_, hfloat, double, v_float32)
DEF_CVT_FUNC(16f64u, cvt1_, hfloat, uint64_t, v_float32)
DEF_CVT_FUNC(16f64s, cvt1_, hfloat, int64_t, v_float32)
DEF_CVT_FUNC(16f16bf, cvt1_, hfloat, bfloat, v_float32)
DEF_CVT2BOOL_FUNC(16f8b, short, 1)

////////////////////// 16bf -> ... ////////////////////////

DEF_CVT_FUNC(16bf8u,  cvt_,  bfloat, uchar,  v_float32)
DEF_CVT_FUNC(16bf8s,  cvt_,  bfloat, schar,  v_float32)
DEF_CVT_FUNC(16bf16u, cvt1_, bfloat, ushort, v_float32)
DEF_CVT_FUNC(16bf16s, cvt1_, bfloat, short,  v_float32)
DEF_CVT_FUNC(16bf32u, cvt1_, bfloat, unsigned, v_float32)
DEF_CVT_FUNC(16bf32s, cvt1_, bfloat, int,    v_float32)
DEF_CVT_FUNC(16bf32f, cvt1_, bfloat, float,  v_float32)
DEF_CVT_FUNC(16bf64f, cvt1_, bfloat, double, v_float32)
DEF_CVT_FUNC(16bf64u, cvt1_, bfloat, uint64_t, v_float32)
DEF_CVT_FUNC(16bf64s, cvt1_, bfloat, int64_t, v_float32)
DEF_CVT_FUNC(16bf16f, cvt1_, bfloat, hfloat, v_float32)

////////////////////// 64s -> ... ////////////////////////

DEF_CVT_FUNC(64s8u,  cvt_, int64_t, uchar,  v_int32)
DEF_CVT_FUNC(64s8s,  cvt_, int64_t, schar,  v_int32)
DEF_CVT_FUNC(64s16u, cvt_, int64_t, ushort, v_int32)
DEF_CVT_FUNC(64s16s, cvt_, int64_t, short,  v_int32)
DEF_CVT_FUNC(64s32u, cvt_, int64_t, unsigned, v_uint32)
DEF_CVT_FUNC(64s32s, cvt_, int64_t, int,    v_int32)
DEF_CVT_FUNC(64s32f, cvt_64f, int64_t, float,  v_float32)
DEF_CVT_FUNC(64s64f, cvt_64f, int64_t, double,  v_float64)
DEF_CVT_FUNC(64s64u, cvt_, int64_t, uint64_t, v_uint64)
DEF_CVT_FUNC(64s16f, cvt1_,int64_t, hfloat, v_float32)
DEF_CVT_FUNC(64s16bf, cvt1_, int64_t, bfloat, v_float32)
DEF_CVT2BOOL_FUNC(64s8b, int64_t, 0)

////////////////////// 64u -> ... ////////////////////////

DEF_CVT_FUNC(64u8u,  cvt_, uint64_t, uchar,  v_int32)
DEF_CVT_FUNC(64u8s,  cvt_, uint64_t, schar,  v_int32)
DEF_CVT_FUNC(64u16u, cvt_, uint64_t, ushort, v_int32)
DEF_CVT_FUNC(64u16s, cvt_, uint64_t, short,  v_int32)
DEF_CVT_FUNC(64u32u, cvt_, uint64_t, unsigned, v_uint32)
DEF_CVT_FUNC(64u32s, cvt_, uint64_t, int,   v_int32)
DEF_CVT_FUNC(64u32f, cvt_64f, uint64_t, float,  v_float64)
DEF_CVT_FUNC(64u64f, cvt_64f, uint64_t, double,  v_float64)
DEF_CVT_FUNC(64u16f, cvt1_,uint64_t, hfloat, v_float32)
DEF_CVT_FUNC(64u16bf, cvt1_, uint64_t, bfloat, v_float32)

///////////// "conversion" w/o conversion ///////////////

static void cvt8u(const uchar* src, size_t sstep, const uchar*, size_t, uchar* dst, size_t dstep, Size size, void*)
{ CV_INSTRUMENT_REGION(); cvtCopy(src, sstep, dst, dstep, size, 1); }

static void cvt16u(const uchar* src, size_t sstep, const uchar*, size_t, uchar* dst, size_t dstep, Size size, void*)
{ CV_INSTRUMENT_REGION(); cvtCopy((const uchar*)src, sstep, (uchar*)dst, dstep, size, 2); }

static void cvt32s(const uchar* src, size_t sstep, const uchar*, size_t, uchar* dst, size_t dstep, Size size, void*)
{ CV_INSTRUMENT_REGION(); cvtCopy((const uchar*)src, sstep, (uchar*)dst, dstep, size, 4); }

static void cvt64s(const uchar* src, size_t sstep, const uchar*, size_t, uchar* dst, size_t dstep, Size size, void*)
{ CV_INSTRUMENT_REGION(); cvtCopy((const uchar*)src, sstep, (uchar*)dst, dstep, size, 8); }

BinaryFunc getConvertFunc(int sdepth_, int ddepth_)
{
    int sdepth = CV_MAT_DEPTH(sdepth_);
    int ddepth = CV_MAT_DEPTH(ddepth_);
    BinaryFunc func =
        ddepth == CV_8U ? (
            sdepth == CV_8U ? cvt8u :
            sdepth == CV_8S ? cvt8s8u :
            sdepth == CV_16U ? cvt16u8u :
            sdepth == CV_16S ? cvt16s8u :
            sdepth == CV_32U ? cvt32u8u :
            sdepth == CV_32S ? cvt32s8u :
            sdepth == CV_32F ? cvt32f8u :
            sdepth == CV_64F ? cvt64f8u :
            sdepth == CV_16F ? cvt16f8u :
            sdepth == CV_16BF ? cvt16bf8u :
            sdepth == CV_Bool ? cvt8b8u :
            sdepth == CV_64U ? cvt64u8u :
            sdepth == CV_64S ? cvt64s8u :
            0) :
        ddepth == CV_8S ? (
            sdepth == CV_8U ? cvt8u8s :
            sdepth == CV_8S ? cvt8u :
            sdepth == CV_16U ? cvt16u8s :
            sdepth == CV_16S ? cvt16s8s :
            sdepth == CV_32U ? cvt32u8s :
            sdepth == CV_32S ? cvt32s8s :
            sdepth == CV_32F ? cvt32f8s :
            sdepth == CV_64F ? cvt64f8s :
            sdepth == CV_16F ? cvt16f8s :
            sdepth == CV_16BF ? cvt16bf8s :
            sdepth == CV_Bool ? cvt8b8u :
            sdepth == CV_64U ? cvt64u8s :
            sdepth == CV_64S ? cvt64s8s :
            0) :
        ddepth == CV_16U ? (
            sdepth == CV_8U ? cvt8u16s : // same as cvt8u16u
            sdepth == CV_8S ? cvt8s16u :
            sdepth == CV_16U ? cvt16u :
            sdepth == CV_16S ? cvt16s16u :
            sdepth == CV_32U ? cvt32u16u :
            sdepth == CV_32S ? cvt32s16u :
            sdepth == CV_32F ? cvt32f16u :
            sdepth == CV_64F ? cvt64f16u :
            sdepth == CV_16F ? cvt16f16u :
            sdepth == CV_16BF ? cvt16bf16u :
            sdepth == CV_Bool ? cvt8b16s :
            sdepth == CV_64U ? cvt64u16u :
            sdepth == CV_64S ? cvt64s16u :
            0) :
        ddepth == CV_16S ? (
            sdepth == CV_8U ? cvt8u16s :
            sdepth == CV_8S ? cvt8s16s :
            sdepth == CV_16U ? cvt16u16s :
            sdepth == CV_16S ? cvt16u :
            sdepth == CV_32U ? cvt32u16s :
            sdepth == CV_32S ? cvt32s16s :
            sdepth == CV_32F ? cvt32f16s :
            sdepth == CV_64F ? cvt64f16s :
            sdepth == CV_16F ? cvt16f16s :
            sdepth == CV_16BF ? cvt16bf16s :
            sdepth == CV_Bool ? cvt8b16s :
            sdepth == CV_64U ? cvt64u16s :
            sdepth == CV_64S ? cvt64s16s :
            0) :
        ddepth == CV_32U ? (
            sdepth == CV_8U ? cvt8u32s : // same as cvt8u32u
            sdepth == CV_8S ? cvt8s32u :
            sdepth == CV_16U ? cvt16u32s : // same as cvt16u32u
            sdepth == CV_16S ? cvt16s32u :
            sdepth == CV_32U ? cvt32s :
            sdepth == CV_32S ? cvt32s32u :
            sdepth == CV_32F ? cvt32f32u :
            sdepth == CV_64F ? cvt64f32u :
            sdepth == CV_16F ? cvt16f32u :
            sdepth == CV_16BF ? cvt16bf32u :
            sdepth == CV_Bool ? cvt8b32s :
            sdepth == CV_64U ? cvt64u32u :
            sdepth == CV_64S ? cvt64s32u :

            0) :
        ddepth == CV_32S ? (
            sdepth == CV_8U ? cvt8u32s :
            sdepth == CV_8S ? cvt8s32s :
            sdepth == CV_16U ? cvt16u32s :
            sdepth == CV_16S ? cvt16s32s :
            sdepth == CV_32U ? cvt32u32s :
            sdepth == CV_32S ? cvt32s :
            sdepth == CV_32F ? cvt32f32s :
            sdepth == CV_64F ? cvt64f32s :
            sdepth == CV_16F ? cvt16f32s :
            sdepth == CV_16BF ? cvt16bf32s :
            sdepth == CV_Bool ? cvt8b32s :
            sdepth == CV_64U ? cvt64u32s :
            sdepth == CV_64S ? cvt64s32s :
            0) :
        ddepth == CV_32F ? (
            sdepth == CV_8U ? cvt8u32f :
            sdepth == CV_8S ? cvt8s32f :
            sdepth == CV_16U ? cvt16u32f :
            sdepth == CV_16S ? cvt16s32f :
            sdepth == CV_32U ? cvt32u32f :
            sdepth == CV_32S ? cvt32s32f :
            sdepth == CV_32F ? cvt32s :
            sdepth == CV_64F ? cvt64f32f :
            sdepth == CV_16F ? cvt16f32f :
            sdepth == CV_16BF ? cvt16bf32f :
            sdepth == CV_Bool ? cvt8b32f :
            sdepth == CV_64U ? cvt64u32f :
            sdepth == CV_64S ? cvt64s32f :
            0) :
        ddepth == CV_64F ? (
            sdepth == CV_8U ? cvt8u64f :
            sdepth == CV_8S ? cvt8s64f :
            sdepth == CV_16U ? cvt16u64f :
            sdepth == CV_16S ? cvt16s64f :
            sdepth == CV_32U ? cvt32u64f :
            sdepth == CV_32S ? cvt32s64f :
            sdepth == CV_32F ? cvt32f64f :
            sdepth == CV_64F ? cvt64s :
            sdepth == CV_16F ? cvt16f64f :
            sdepth == CV_16BF ? cvt16bf64f :
            sdepth == CV_Bool ? cvt8b64f :
            sdepth == CV_64U ? cvt64u64f :
            sdepth == CV_64S ? cvt64s64f :
            0) :
        ddepth == CV_16F ? (
            sdepth == CV_8U ? cvt8u16f :
            sdepth == CV_8S ? cvt8s16f :
            sdepth == CV_16U ? cvt16u16f :
            sdepth == CV_16S ? cvt16s16f :
            sdepth == CV_32U ? cvt32u16f :
            sdepth == CV_32S ? cvt32s16f :
            sdepth == CV_32F ? cvt32f16f :
            sdepth == CV_64F ? cvt64f16f :
            sdepth == CV_16F ? cvt16u :
            sdepth == CV_16BF ? cvt16bf16f :
            sdepth == CV_Bool ? cvt8b16f :
            sdepth == CV_64U ? cvt64u16f :
            sdepth == CV_64S ? cvt64s16f :
            0) :
        ddepth == CV_16BF ? (
            sdepth == CV_8U ? cvt8u16bf :
            sdepth == CV_8S ? cvt8s16bf :
            sdepth == CV_16U ? cvt16u16bf :
            sdepth == CV_16S ? cvt16s16bf :
            sdepth == CV_32U ? cvt32u16bf :
            sdepth == CV_32S ? cvt32s16bf :
            sdepth == CV_32F ? cvt32f16bf :
            sdepth == CV_64F ? cvt64f16bf :
            sdepth == CV_16F ? cvt16f16bf :
            sdepth == CV_16BF ? cvt16u :
            sdepth == CV_Bool ? cvt8b16bf :
            sdepth == CV_64U ? cvt64u16bf :
            sdepth == CV_64S ? cvt64s16bf :
            0) :
        ddepth == CV_Bool ? (
            sdepth == CV_8U ? cvt8u8b :
            sdepth == CV_8S ? cvt8u8b :
            sdepth == CV_16U ? cvt16s8b :
            sdepth == CV_16S ? cvt16s8b :
            sdepth == CV_32U ? cvt32s8b :
            sdepth == CV_32S ? cvt32s8b :
            sdepth == CV_32F ? cvt32f8b :
            sdepth == CV_64F ? cvt64f8b :
            sdepth == CV_16F ? cvt16f8b :
            sdepth == CV_16BF ? cvt16f8b : // same as cvt16f8b
            sdepth == CV_Bool ? cvt8u :
            sdepth == CV_64U ? cvt64s8b :
            sdepth == CV_64S ? cvt64s8b :
            0) :
        ddepth == CV_64U ? (
            sdepth == CV_8U ? cvt8u64s : // same as cvt8u64u
            sdepth == CV_8S ? cvt8s64u :
            sdepth == CV_16U ? cvt16u64s : // same as cvt16u64u
            sdepth == CV_16S ? cvt16s64u :
            sdepth == CV_32U ? cvt32u64s : // same as cvt32u64u
            sdepth == CV_32S ? cvt32s64u :
            sdepth == CV_32F ? cvt32f64u :
            sdepth == CV_64F ? cvt64f64u :
            sdepth == CV_16F ? cvt16f64u :
            sdepth == CV_16BF ? cvt16bf64u :
            sdepth == CV_Bool ? cvt8b64s :
            sdepth == CV_64U ? cvt64s :
            sdepth == CV_64S ? cvt64s64u :
            0) :
        ddepth == CV_64S ? (
            sdepth == CV_8U ? cvt8u64s :
            sdepth == CV_8S ? cvt8s64s :
            sdepth == CV_16U ? cvt16u64s :
            sdepth == CV_16S ? cvt16s64s :
            sdepth == CV_32U ? cvt32u64s :
            sdepth == CV_32S ? cvt32s64s :
            sdepth == CV_32F ? cvt32f64s :
            sdepth == CV_64F ? cvt64f64s :
            sdepth == CV_16F ? cvt16f64s :
            sdepth == CV_16BF ? cvt16bf64s :
            sdepth == CV_Bool ? cvt8b64s :
            sdepth == CV_64U ? cvt64s :
            sdepth == CV_64S ? cvt64s :
            0) :
        0;
    CV_Assert(func != 0);
    return func;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
#endif
} // namespace
