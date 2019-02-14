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
#if CV_SIMD
    int len0 = len & -v_uint8::nlanes;
    v_uint8 v_zero = vx_setzero_u8();
    v_uint8 v_one = vx_setall_u8(1);

    v_uint32 v_sum32 = vx_setzero_u32();
    while (i < len0)
    {
        v_uint16 v_sum16 = vx_setzero_u16();
        int j = i;
        while (j < std::min(len0, i + 65280 * v_uint16::nlanes))
        {
            v_uint8 v_sum8 = vx_setzero_u8();
            int k = j;
            for (; k < std::min(len0, j + 255 * v_uint8::nlanes); k += v_uint8::nlanes)
                v_sum8 += v_one & (vx_load(src + k) == v_zero);
            v_uint16 part1, part2;
            v_expand(v_sum8, part1, part2);
            v_sum16 += part1 + part2;
            j = k;
        }
        v_uint32 part1, part2;
        v_expand(v_sum16, part1, part2);
        v_sum32 += part1 + part2;
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
#if CV_SIMD
    int len0 = len & -v_int8::nlanes;
    v_uint16 v_zero = vx_setzero_u16();
    v_int8 v_one = vx_setall_s8(1);

    v_int32 v_sum32 = vx_setzero_s32();
    while (i < len0)
    {
        v_int16 v_sum16 = vx_setzero_s16();
        int j = i;
        while (j < std::min(len0, i + 32766 * v_int16::nlanes))
        {
            v_int8 v_sum8 = vx_setzero_s8();
            int k = j;
            for (; k < std::min(len0, j + 127 * v_int8::nlanes); k += v_int8::nlanes)
                v_sum8 += v_one & v_pack(v_reinterpret_as_s16(vx_load(src + k) == v_zero), v_reinterpret_as_s16(vx_load(src + k + v_uint16::nlanes) == v_zero));
            v_int16 part1, part2;
            v_expand(v_sum8, part1, part2);
            v_sum16 += part1 + part2;
            j = k;
        }
        v_int32 part1, part2;
        v_expand(v_sum16, part1, part2);
        v_sum32 += part1 + part2;
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
#if CV_SIMD
    int len0 = len & -v_int8::nlanes;
    v_int32 v_zero = vx_setzero_s32();
    v_int8 v_one = vx_setall_s8(1);

    v_int32 v_sum32 = vx_setzero_s32();
    while (i < len0)
    {
        v_int16 v_sum16 = vx_setzero_s16();
        int j = i;
        while (j < std::min(len0, i + 32766 * v_int16::nlanes))
        {
            v_int8 v_sum8 = vx_setzero_s8();
            int k = j;
            for (; k < std::min(len0, j + 127 * v_int8::nlanes); k += v_int8::nlanes)
                v_sum8 += v_one & v_pack(
                    v_pack(vx_load(src + k                    ) == v_zero, vx_load(src + k +   v_int32::nlanes) == v_zero),
                    v_pack(vx_load(src + k + 2*v_int32::nlanes) == v_zero, vx_load(src + k + 3*v_int32::nlanes) == v_zero)
                );
            v_int16 part1, part2;
            v_expand(v_sum8, part1, part2);
            v_sum16 += part1 + part2;
            j = k;
        }
        v_int32 part1, part2;
        v_expand(v_sum16, part1, part2);
        v_sum32 += part1 + part2;
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
#if CV_SIMD
    int len0 = len & -v_int8::nlanes;
    v_float32 v_zero = vx_setzero_f32();
    v_int8 v_one = vx_setall_s8(1);

    v_int32 v_sum32 = vx_setzero_s32();
    while (i < len0)
    {
        v_int16 v_sum16 = vx_setzero_s16();
        int j = i;
        while (j < std::min(len0, i + 32766 * v_int16::nlanes))
        {
            v_int8 v_sum8 = vx_setzero_s8();
            int k = j;
            for (; k < std::min(len0, j + 127 * v_int8::nlanes); k += v_int8::nlanes)
                v_sum8 += v_one & v_pack(
                    v_pack(v_reinterpret_as_s32(vx_load(src + k                      ) == v_zero), v_reinterpret_as_s32(vx_load(src + k +   v_float32::nlanes) == v_zero)),
                    v_pack(v_reinterpret_as_s32(vx_load(src + k + 2*v_float32::nlanes) == v_zero), v_reinterpret_as_s32(vx_load(src + k + 3*v_float32::nlanes) == v_zero))
                );
            v_int16 part1, part2;
            v_expand(v_sum8, part1, part2);
            v_sum16 += part1 + part2;
            j = k;
        }
        v_int32 part1, part2;
        v_expand(v_sum16, part1, part2);
        v_sum32 += part1 + part2;
        i = j;
    }
    nz = i - v_reduce_sum(v_sum32);
    v_cleanup();
#endif
    return nz + countNonZero_(src + i, len - i);
}

static int countNonZero64f( const double* src, int len )
{
    return countNonZero_(src, len);
}

CountNonZeroFunc getCountNonZeroTab(int depth)
{
    static CountNonZeroFunc countNonZeroTab[] =
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
