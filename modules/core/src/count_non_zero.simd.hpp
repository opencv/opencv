// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {

typedef int (*CountNonZeroFunc)(const void*, int);


CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

CountNonZeroFunc getCountNonZeroTab(int depth);


#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

template<typename T>
static int countNonZero_(const T* src, int len )
{
    int i=0, nz = 0;
    #if CV_ENABLE_UNROLLED
    for(; i <= len - 4; i += 4 )
        nz += (src[i] != 0) + (src[i+1] != 0) + (src[i+2] != 0) + (src[i+3] != 0);
    #endif
    for( ; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

#if (CV_SIMD || CV_SIMD_SCALABLE)

#if defined(CV_CPU_COMPILE_AVX512_SKX) || defined(CV_CPU_COMPILE_AVX512_ICL)
#define CV_COUNTNONZERO_AVX512 1
#else
#define CV_COUNTNONZERO_AVX512 0
#endif

#if CV_COUNTNONZERO_AVX512

static inline int cnz_popcount32(unsigned x)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int c = 0;
    for (; x; x &= x - 1)
        ++c;
    return c;
#endif
}

static inline int cnz_popcount64(uint64 x)
{
#if defined(__GNUC__) || defined(__clang__)
    return (int)__builtin_popcountll((unsigned long long)x);
#else
    int c = 0;
    for (; x; x &= x - 1)
        ++c;
    return c;
#endif
}

static inline int cnz_lane_pop(int m) { return cnz_popcount32((unsigned)m); }
static inline int cnz_lane_pop(int64 m) { return cnz_popcount64((uint64)m); }

template<typename VT> static inline int cnz_pop_ne(const VT& v, const VT& z);

template<> inline int cnz_pop_ne<v_uint8>(const v_uint8& v, const v_uint8& z)
{ return cnz_lane_pop(v_signmask(v_reinterpret_as_s8(v_ne(v, z)))); }

template<> inline int cnz_pop_ne<v_uint16>(const v_uint16& v, const v_uint16& z)
{ return cnz_lane_pop(v_signmask(v_reinterpret_as_s16(v_ne(v, z)))); }

template<> inline int cnz_pop_ne<v_int32>(const v_int32& v, const v_int32& z)
{ return cnz_lane_pop(v_signmask(v_ne(v, z))); }

template<> inline int cnz_pop_ne<v_float32>(const v_float32& v, const v_float32& z)
{ return cnz_lane_pop(v_signmask(v_ne(v, z))); }

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
template<> inline int cnz_pop_ne<v_float64>(const v_float64& v, const v_float64& z)
{ return cnz_lane_pop(v_signmask(v_ne(v, z))); }
#endif

#endif // CV_COUNTNONZERO_AVX512

struct cnz_pack16u
{
    static inline v_int8 eq(const ushort* src, int k, v_uint16 z)
    {
        const int n = VTraits<v_uint16>::vlanes();
        return v_pack(v_reinterpret_as_s16(v_eq(vx_load(src + k), z)),
                      v_reinterpret_as_s16(v_eq(vx_load(src + k + n), z)));
    }
};

struct cnz_pack32s
{
    static inline v_int8 eq(const int* src, int k, v_int32 z)
    {
        const int n = VTraits<v_int32>::vlanes();
        return v_pack(v_pack(v_eq(vx_load(src + k), z), v_eq(vx_load(src + k + n), z)),
                      v_pack(v_eq(vx_load(src + k + 2 * n), z), v_eq(vx_load(src + k + 3 * n), z)));
    }
};

struct cnz_pack32f
{
    static inline v_int8 eq(const float* src, int k, v_float32 z)
    {
        const int n = VTraits<v_float32>::vlanes();
        return v_pack(v_pack(v_reinterpret_as_s32(v_eq(vx_load(src + k), z)),
                             v_reinterpret_as_s32(v_eq(vx_load(src + k + n), z))),
                      v_pack(v_reinterpret_as_s32(v_eq(vx_load(src + k + 2 * n), z)),
                             v_reinterpret_as_s32(v_eq(vx_load(src + k + 3 * n), z))));
    }
};

template<typename PackOp, typename ST, typename ZVT>
static int countNonZeroVT_batched(const ST* src, int len, const ZVT& z)
{
    int i = 0, nz = 0;
    const int vl8 = VTraits<v_int8>::vlanes();
    const int len0 = len & -vl8;
    v_int8 v_one = vx_setall_s8(1);
    v_int32 v_sum32 = vx_setzero_s32();

    while (i < len0)
    {
        v_int16 v_sum16 = vx_setzero_s16();
        int j = i;
        while (j < std::min(len0, i + 32766 * VTraits<v_int16>::vlanes()))
        {
            v_int8 v_sum8 = vx_setzero_s8();
            int k = j;
            for (; k < std::min(len0, j + 127 * vl8); k += vl8)
                v_sum8 = v_add(v_sum8, v_and(v_one, PackOp::eq(src, k, z)));
            v_int16 part1, part2;
            v_expand(v_sum8, part1, part2);
            v_sum16 = v_add(v_sum16, v_add(part1, part2));
            j = k;
        }
        v_int32 part1, part2;
        v_expand(v_sum16, part1, part2);
        v_sum32 = v_add(v_sum32, v_add(part1, part2));
        i = j;
    }
    nz = i - v_reduce_sum(v_sum32);
    v_cleanup();
    return nz + countNonZero_(src + i, len - i);
}

static int countNonZeroVT_u8(const uchar* src, int len)
{
    int i = 0, nz = 0;
    const int len0 = len & -VTraits<v_uint8>::vlanes();
    v_uint8 v_zero = vx_setzero_u8();
    v_uint8 v_one = vx_setall_u8(1);
    v_uint32 v_sum32 = vx_setzero_u32();

    while (i < len0)
    {
        v_uint16 v_sum16 = vx_setzero_u16();
        int j = i;
        while (j < std::min(len0, i + 65280 * VTraits<v_uint16>::vlanes()))
        {
            v_uint8 v_sum8 = vx_setzero_u8();
            int k = j;
            for (; k < std::min(len0, j + 255 * VTraits<v_uint8>::vlanes()); k += VTraits<v_uint8>::vlanes())
                v_sum8 = v_add(v_sum8, v_and(v_one, v_eq(vx_load(src + k), v_zero)));
            v_uint16 part1, part2;
            v_expand(v_sum8, part1, part2);
            v_sum16 = v_add(v_sum16, v_add(part1, part2));
            j = k;
        }
        v_uint32 part1, part2;
        v_expand(v_sum16, part1, part2);
        v_sum32 = v_add(v_sum32, v_add(part1, part2));
        i = j;
    }
    nz = i - v_reduce_sum(v_sum32);
    v_cleanup();
    for (; i < len; i++)
        nz += src[i] != 0;
    return nz;
}

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
static int countNonZeroVT_f64(const double* src, int len)
{
    int nz = 0, i = 0;
    v_int64 sum1 = vx_setzero_s64();
    v_int64 sum2 = vx_setzero_s64();
    v_float64 zero = vx_setzero_f64();
    const int step = VTraits<v_float64>::vlanes() * 2;
    const int len0 = len & -step;

    for (i = 0; i < len0; i += step)
    {
        sum1 = v_add(sum1, v_reinterpret_as_s64(v_eq(vx_load(&src[i]), zero)));
        sum2 = v_add(sum2, v_reinterpret_as_s64(v_eq(vx_load(&src[i + step / 2]), zero)));
    }

    nz = i + (int)v_reduce_sum(v_add(sum1, sum2));
    v_cleanup();
    return nz + countNonZero_(src + i, len - i);
}
#endif

template<typename VT> struct cnz_vt_traits;

template<>
struct cnz_vt_traits<v_uint8>
{
    typedef uchar ST;
    static inline v_uint8 zero() { return vx_setzero_u8(); }
    static int legacy(const uchar* src, int len, const v_uint8&) { return countNonZeroVT_u8(src, len); }
};

template<>
struct cnz_vt_traits<v_uint16>
{
    typedef ushort ST;
    static inline v_uint16 zero() { return vx_setzero_u16(); }
    static int legacy(const ushort* src, int len, const v_uint16& z)
    { return countNonZeroVT_batched<cnz_pack16u, ushort, v_uint16>(src, len, z); }
};

template<>
struct cnz_vt_traits<v_int32>
{
    typedef int ST;
    static inline v_int32 zero() { return vx_setzero_s32(); }
    static int legacy(const int* src, int len, const v_int32& z)
    { return countNonZeroVT_batched<cnz_pack32s, int, v_int32>(src, len, z); }
};

template<>
struct cnz_vt_traits<v_float32>
{
    typedef float ST;
    static inline v_float32 zero() { return vx_setzero_f32(); }
    static int legacy(const float* src, int len, const v_float32& z)
    { return countNonZeroVT_batched<cnz_pack32f, float, v_float32>(src, len, z); }
};

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
template<>
struct cnz_vt_traits<v_float64>
{
    typedef double ST;
    static inline v_float64 zero() { return vx_setzero_f64(); }
    static int legacy(const double* src, int len, const v_float64&)
    { return countNonZeroVT_f64(src, len); }
};
#endif

template<typename VT, typename ST>
static int countNonZeroVT(const ST* src, int len, const VT& z)
{
#if CV_COUNTNONZERO_AVX512
    const int nlanes = VTraits<VT>::vlanes();
    const int step = nlanes * 2;
    const int len0 = len & -step;
    int i = 0, nz = 0;

    for (; i < len0; i += step)
    {
        nz += cnz_pop_ne<VT>(vx_load(src + i), z);
        nz += cnz_pop_ne<VT>(vx_load(src + i + nlanes), z);
    }
    for (; i <= len - nlanes; i += nlanes)
        nz += cnz_pop_ne<VT>(vx_load(src + i), z);

    v_cleanup();
    return nz + countNonZero_(src + i, len - i);
#else
    return cnz_vt_traits<VT>::legacy(src, len, z);
#endif
}

template<typename VT>
static int countNonZeroSimd(const void* src_ptr, int len)
{
    typedef cnz_vt_traits<VT> traits;
    const typename traits::ST* src = static_cast<const typename traits::ST*>(src_ptr);
    return countNonZeroVT<VT>(src, len, traits::zero());
}

#endif // CV_SIMD

template<typename VT, typename ST>
static int countNonZeroDepth(const void* src_ptr, int len)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    return countNonZeroSimd<VT>(src_ptr, len);
#else
    return countNonZero_(static_cast<const ST*>(src_ptr), len);
#endif
}

static int countNonZero8u( const void* src_ptr, int len )
{ return countNonZeroDepth<v_uint8, uchar>(src_ptr, len); }

static int countNonZero16u( const void* src_ptr, int len )
{ return countNonZeroDepth<v_uint16, ushort>(src_ptr, len); }

static int countNonZero32s( const void* src_ptr, int len )
{ return countNonZeroDepth<v_int32, int>(src_ptr, len); }

static int countNonZero32f( const void* src_ptr, int len )
{ return countNonZeroDepth<v_float32, float>(src_ptr, len); }

static int countNonZero64f( const void* src_ptr, int len )
{
#if (CV_SIMD || CV_SIMD_SCALABLE) && (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    return countNonZeroSimd<v_float64>(src_ptr, len);
#else
    return countNonZero_(static_cast<const double*>(src_ptr), len);
#endif
}

CountNonZeroFunc getCountNonZeroTab(int depth)
{
    static CountNonZeroFunc countNonZeroTab[CV_DEPTH_MAX] =
    {
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32s), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32f),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero64f), 0
    };

    return countNonZeroTab[depth];
}

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
