// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {

typedef bool (*HasNonZeroFunc)(const uchar*, size_t);


CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

HasNonZeroFunc getHasNonZeroTab(int depth);


#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

template<typename T>
inline bool hasNonZero_(const T* src, size_t len )
{
    bool res = false;
    if (len > 0)
    {
        size_t i=0;
        #if CV_ENABLE_UNROLLED
        for(; !res && (i+4 <= len); i += 4 )
            res |= ((src[i] | src[i+1] | src[i+2] | src[i+3]) != 0);
        #endif
        for( ; !res && (i < len); i++ )
            res |= (src[i] != 0);
    }
    return res;
}

template<>
inline bool hasNonZero_(const float* src, size_t len )
{
    bool res = false;
    if (len > 0)
    {
        size_t i=0;
        if (sizeof(float) == sizeof(unsigned int))
        {
            #if CV_ENABLE_UNROLLED
            typedef unsigned int float_as_uint_t;
            const float_as_uint_t* src_as_ui = reinterpret_cast<const float_as_uint_t*>(src);
            for(; !res && (i+4 <= len); i += 4 )
            {
                const float_as_uint_t gathered = (src_as_ui[i] | src_as_ui[i+1] | src_as_ui[i+2] | src_as_ui[i+3]);
                res |= ((gathered<<1) != 0);//remove what would be the sign bit
            }
            #endif
        }
        for( ; !res && (i < len); i++ )
            res |= (src[i] != 0);
    }
    return res;
}

template<>
inline bool hasNonZero_(const double* src, size_t len )
{
    bool res = false;
    if (len > 0)
    {
        size_t i=0;
        if (sizeof(double) == sizeof(uint64_t))
        {
            #if CV_ENABLE_UNROLLED
            typedef uint64_t double_as_uint_t;
            const double_as_uint_t* src_as_ui = reinterpret_cast<const double_as_uint_t*>(src);
            for(; !res && (i+4 <= len); i += 4 )
            {
                const double_as_uint_t gathered = (src_as_ui[i] | src_as_ui[i+1] | src_as_ui[i+2] | src_as_ui[i+3]);
                res |= ((gathered<<1) != 0);//remove what would be the sign bit
            }
            #endif
        }
        for( ; !res && (i < len); i++ )
            res |= (src[i] != 0);
    }
    return res;
}

static bool hasNonZero8u( const uchar* src, size_t len )
{
    bool res = false;
    const uchar* srcEnd = src+len;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    typedef v_uint8 v_type;
    const v_type v_zero = vx_setzero_u8();
    constexpr const int unrollCount = 2;
    int step = VTraits<v_type>::vlanes() * unrollCount;
    int len0 = len & -step;
    const uchar* srcSimdEnd = src+len0;

    int countSIMD = static_cast<int>((srcSimdEnd-src)/step);
    while(!res && countSIMD--)
    {
        v_type v0 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v1 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        res = v_check_any((v_ne(v_or(v0, v1), v_zero)));
    }

    v_cleanup();
#endif
    return res || hasNonZero_(src, srcEnd-src);
}

static bool hasNonZero16u( const ushort* src, size_t len )
{
    bool res = false;
    const ushort* srcEnd = src+len;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    typedef v_uint16 v_type;
    const v_type v_zero = vx_setzero_u16();
    constexpr const int unrollCount = 4;
    int step = VTraits<v_type>::vlanes() * unrollCount;
    int len0 = len & -step;
    const ushort* srcSimdEnd = src+len0;

    int countSIMD = static_cast<int>((srcSimdEnd-src)/step);
    while(!res && countSIMD--)
    {
        v_type v0 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v1 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v2 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v3 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v0 = v_or(v0, v1);
        v2 = v_or(v2, v3);
        res = v_check_any((v_ne(v_or(v0, v2), v_zero)));
    }

    v_cleanup();
#endif
    return res || hasNonZero_(src, srcEnd-src);
}

static bool hasNonZero32s( const int* src, size_t len )
{
    bool res = false;
    const int* srcEnd = src+len;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    typedef v_int32 v_type;
    const v_type v_zero = vx_setzero_s32();
    constexpr const int unrollCount = 8;
    int step = VTraits<v_type>::vlanes() * unrollCount;
    int len0 = len & -step;
    const int* srcSimdEnd = src+len0;

    int countSIMD = static_cast<int>((srcSimdEnd-src)/step);
    while(!res && countSIMD--)
    {
        v_type v0 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v1 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v2 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v3 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v4 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v5 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v6 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v7 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v0 = v_or(v0, v1);
        v2 = v_or(v2, v3);
        v4 = v_or(v4, v5);
        v6 = v_or(v6, v7);

        v0 = v_or(v0, v2);
        v4 = v_or(v4, v6);
        res = v_check_any((v_ne(v_or(v0, v4), v_zero)));
    }

    v_cleanup();
#endif
    return res || hasNonZero_(src, srcEnd-src);
}

static bool hasNonZero32f( const float* src, size_t len )
{
    bool res = false;
    const float* srcEnd = src+len;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    typedef v_float32 v_type;
    const v_type v_zero = vx_setzero_f32();
    constexpr const int unrollCount = 8;
    int step = VTraits<v_type>::vlanes() * unrollCount;
    int len0 = len & -step;
    const float* srcSimdEnd = src+len0;

    int countSIMD = static_cast<int>((srcSimdEnd-src)/step);
    while(!res && countSIMD--)
    {
        v_type v0 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v1 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v2 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v3 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v4 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v5 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v6 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v7 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v0 = v_or(v0, v1);
        v2 = v_or(v2, v3);
        v4 = v_or(v4, v5);
        v6 = v_or(v6, v7);

        v0 = v_or(v0, v2);
        v4 = v_or(v4, v6);
        //res = v_check_any(((v0 | v4) != v_zero));//beware : (NaN != 0) returns "false" since != is mapped to _CMP_NEQ_OQ and not _CMP_NEQ_UQ
        res = !v_check_all((v_eq(v_or(v0, v4), v_zero)));
    }

    v_cleanup();
#endif
    return res || hasNonZero_(src, srcEnd-src);
}

static bool hasNonZero64f( const double* src, size_t len )
{
    bool res = false;
    const double* srcEnd = src+len;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    typedef v_float64 v_type;
    const v_type v_zero = vx_setzero_f64();
    constexpr const int unrollCount = 16;
    int step = VTraits<v_type>::vlanes() * unrollCount;
    int len0 = len & -step;
    const double* srcSimdEnd = src+len0;

    int countSIMD = static_cast<int>((srcSimdEnd-src)/step);
    while(!res && countSIMD--)
    {
        v_type v0 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v1 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v2 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v3 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v4 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v5 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v6 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v7 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v8 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v9 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v10 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v11 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v12 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v13 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v14 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v_type v15 = vx_load(src);
        src += VTraits<v_type>::vlanes();
        v0 = v_or(v0, v1);
        v2 = v_or(v2, v3);
        v4 = v_or(v4, v5);
        v6 = v_or(v6, v7);
        v8 = v_or(v8, v9);
        v10 = v_or(v10, v11);
        v12 = v_or(v12, v13);
        v14 = v_or(v14, v15);

        v0 = v_or(v0, v2);
        v4 = v_or(v4, v6);
        v8 = v_or(v8, v10);
        v12 = v_or(v12, v14);

        v0 = v_or(v0, v4);
        v8 = v_or(v8, v12);
        //res = v_check_any(((v0 | v8) != v_zero));//beware : (NaN != 0) returns "false" since != is mapped to _CMP_NEQ_OQ and not _CMP_NEQ_UQ
        res = !v_check_all((v_eq(v_or(v0, v8), v_zero)));
    }

    v_cleanup();
#endif
    return res || hasNonZero_(src, srcEnd-src);
}

HasNonZeroFunc getHasNonZeroTab(int depth)
{
    static HasNonZeroFunc hasNonZeroTab[CV_DEPTH_MAX] =
    {
        (HasNonZeroFunc)GET_OPTIMIZED(hasNonZero8u), (HasNonZeroFunc)GET_OPTIMIZED(hasNonZero8u),
        (HasNonZeroFunc)GET_OPTIMIZED(hasNonZero16u), (HasNonZeroFunc)GET_OPTIMIZED(hasNonZero16u),
        (HasNonZeroFunc)GET_OPTIMIZED(hasNonZero32s), (HasNonZeroFunc)GET_OPTIMIZED(hasNonZero32f),
        (HasNonZeroFunc)GET_OPTIMIZED(hasNonZero64f), 0
    };

    return hasNonZeroTab[depth];
}

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
