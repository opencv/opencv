// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {

typedef int (*CountNonZeroFunc)(const uchar*, int);


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

static int countNonZero8u( const uchar* src, int len )
{
    int i=0, nz = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_uint8>::vlanes();
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
#endif
    for( ; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

static int countNonZero16u( const ushort* src, int len )
{
    int i = 0, nz = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_int8>::vlanes();
    v_uint16 v_zero = vx_setzero_u16();
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
            for (; k < std::min(len0, j + 127 * VTraits<v_int8>::vlanes()); k += VTraits<v_int8>::vlanes())
                v_sum8 = v_add(v_sum8, v_and(v_one, v_pack(v_reinterpret_as_s16(v_eq(vx_load(src + k), v_zero)), v_reinterpret_as_s16(v_eq(vx_load(src + k + VTraits<v_uint16>::vlanes()), v_zero)))));
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
#endif
    return nz + countNonZero_(src + i, len - i);
}

static int countNonZero32s( const int* src, int len )
{
    int i = 0, nz = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_int8>::vlanes();
    v_int32 v_zero = vx_setzero_s32();
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
            for (; k < std::min(len0, j + 127 * VTraits<v_int8>::vlanes()); k += VTraits<v_int8>::vlanes())
                v_sum8 = v_add(v_sum8, v_and(v_one, v_pack(v_pack(v_eq(vx_load(src + k), v_zero), v_eq(vx_load(src + k + VTraits<v_int32>::vlanes()), v_zero)), v_pack(v_eq(vx_load(src + k + 2 * VTraits<v_int32>::vlanes()), v_zero), v_eq(vx_load(src + k + 3 * VTraits<v_int32>::vlanes()), v_zero)))));
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
#endif
    return nz + countNonZero_(src + i, len - i);
}

static int countNonZero32f( const float* src, int len )
{
    int i = 0, nz = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_int8>::vlanes();
    v_float32 v_zero = vx_setzero_f32();
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
            for (; k < std::min(len0, j + 127 * VTraits<v_int8>::vlanes()); k += VTraits<v_int8>::vlanes())
                v_sum8 = v_add(v_sum8, v_and(v_one, v_pack(v_pack(v_reinterpret_as_s32(v_eq(vx_load(src + k), v_zero)), v_reinterpret_as_s32(v_eq(vx_load(src + k + VTraits<v_float32>::vlanes()), v_zero))), v_pack(v_reinterpret_as_s32(v_eq(vx_load(src + k + 2 * VTraits<v_float32>::vlanes()), v_zero)), v_reinterpret_as_s32(v_eq(vx_load(src + k + 3 * VTraits<v_float32>::vlanes()), v_zero))))));
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
#endif
    return nz + countNonZero_(src + i, len - i);
}

static int countNonZero64f( const double* src, int len )
{
    int nz = 0, i = 0;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    v_int64 sum1 = vx_setzero_s64();
    v_int64 sum2 = vx_setzero_s64();
    v_float64 zero = vx_setzero_f64();
    int step = VTraits<v_float64>::vlanes() * 2;
    int len0 = len & -step;

    for(i = 0; i < len0; i += step )
        {
        sum1 = v_add(sum1, v_reinterpret_as_s64(v_eq(vx_load(&src[i]), zero)));
        sum2 = v_add(sum2, v_reinterpret_as_s64(v_eq(vx_load(&src[i + step / 2]), zero)));
        }

    // N.B the value is incremented by -1 (0xF...F) for each value
    nz = i + (int)v_reduce_sum(v_add(sum1, sum2));
    v_cleanup();
#endif
    return nz + countNonZero_(src + i, len - i);
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
