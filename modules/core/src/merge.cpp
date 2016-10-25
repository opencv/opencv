/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv { namespace hal {

#if CV_NEON
template<typename T> struct VMerge2;
template<typename T> struct VMerge3;
template<typename T> struct VMerge4;

#define MERGE2_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>{                                                       \
        void operator()(const data_type* src0, const data_type* src1,             \
                        data_type* dst){                                          \
            reg_type r;                                                           \
            r.val[0] = load_func(src0);                                           \
            r.val[1] = load_func(src1);                                           \
            store_func(dst, r);                                                   \
        }                                                                         \
    }

#define MERGE3_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>{                                                       \
        void operator()(const data_type* src0, const data_type* src1,             \
                        const data_type* src2, data_type* dst){                   \
            reg_type r;                                                           \
            r.val[0] = load_func(src0);                                           \
            r.val[1] = load_func(src1);                                           \
            r.val[2] = load_func(src2);                                           \
            store_func(dst, r);                                                   \
        }                                                                         \
    }

#define MERGE4_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>{                                                       \
        void operator()(const data_type* src0, const data_type* src1,             \
                        const data_type* src2, const data_type* src3,             \
                        data_type* dst){                                          \
            reg_type r;                                                           \
            r.val[0] = load_func(src0);                                           \
            r.val[1] = load_func(src1);                                           \
            r.val[2] = load_func(src2);                                           \
            r.val[3] = load_func(src3);                                           \
            store_func(dst, r);                                                   \
        }                                                                         \
    }

MERGE2_KERNEL_TEMPLATE(VMerge2, uchar ,  uint8x16x2_t, vld1q_u8 , vst2q_u8 );
MERGE2_KERNEL_TEMPLATE(VMerge2, ushort,  uint16x8x2_t, vld1q_u16, vst2q_u16);
MERGE2_KERNEL_TEMPLATE(VMerge2, int   ,   int32x4x2_t, vld1q_s32, vst2q_s32);
MERGE2_KERNEL_TEMPLATE(VMerge2, int64 ,   int64x1x2_t, vld1_s64 , vst2_s64 );

MERGE3_KERNEL_TEMPLATE(VMerge3, uchar ,  uint8x16x3_t, vld1q_u8 , vst3q_u8 );
MERGE3_KERNEL_TEMPLATE(VMerge3, ushort,  uint16x8x3_t, vld1q_u16, vst3q_u16);
MERGE3_KERNEL_TEMPLATE(VMerge3, int   ,   int32x4x3_t, vld1q_s32, vst3q_s32);
MERGE3_KERNEL_TEMPLATE(VMerge3, int64 ,   int64x1x3_t, vld1_s64 , vst3_s64 );

MERGE4_KERNEL_TEMPLATE(VMerge4, uchar ,  uint8x16x4_t, vld1q_u8 , vst4q_u8 );
MERGE4_KERNEL_TEMPLATE(VMerge4, ushort,  uint16x8x4_t, vld1q_u16, vst4q_u16);
MERGE4_KERNEL_TEMPLATE(VMerge4, int   ,   int32x4x4_t, vld1q_s32, vst4q_s32);
MERGE4_KERNEL_TEMPLATE(VMerge4, int64 ,   int64x1x4_t, vld1_s64 , vst4_s64 );

#elif CV_SSE2

template <typename T>
struct VMerge2
{
    VMerge2() : support(false) { }
    void operator()(const T *, const T *, T *) const { }

    bool support;
};

template <typename T>
struct VMerge3
{
    VMerge3() : support(false) { }
    void operator()(const T *, const T *, const T *, T *) const { }

    bool support;
};

template <typename T>
struct VMerge4
{
    VMerge4() : support(false) { }
    void operator()(const T *, const T *, const T *, const T *, T *) const { }

    bool support;
};

#define MERGE2_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_interleave, flavor, se) \
template <>                                                                                \
struct VMerge2<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VMerge2()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(se);                                                \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src0, const data_type * src1,                        \
                    data_type * dst) const                                                 \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((const cast_type *)(src0));                   \
        reg_type v_src1 = _mm_loadu_##flavor((const cast_type *)(src0 + ELEMS_IN_VEC));    \
        reg_type v_src2 = _mm_loadu_##flavor((const cast_type *)(src1));                   \
        reg_type v_src3 = _mm_loadu_##flavor((const cast_type *)(src1 + ELEMS_IN_VEC));    \
                                                                                           \
        _mm_interleave(v_src0, v_src1, v_src2, v_src3);                                    \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst), v_src0);                                   \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC), v_src1);                    \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 2), v_src2);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 3), v_src3);                \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

#define MERGE3_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_interleave, flavor, se) \
template <>                                                                                \
struct VMerge3<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VMerge3()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(se);                                                \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src0, const data_type * src1, const data_type * src2,\
                    data_type * dst) const                                                 \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((const cast_type *)(src0));                   \
        reg_type v_src1 = _mm_loadu_##flavor((const cast_type *)(src0 + ELEMS_IN_VEC));    \
        reg_type v_src2 = _mm_loadu_##flavor((const cast_type *)(src1));                   \
        reg_type v_src3 = _mm_loadu_##flavor((const cast_type *)(src1 + ELEMS_IN_VEC));    \
        reg_type v_src4 = _mm_loadu_##flavor((const cast_type *)(src2));                   \
        reg_type v_src5 = _mm_loadu_##flavor((const cast_type *)(src2 + ELEMS_IN_VEC));    \
                                                                                           \
        _mm_interleave(v_src0, v_src1, v_src2,                                             \
                       v_src3, v_src4, v_src5);                                            \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst), v_src0);                                   \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC), v_src1);                    \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 2), v_src2);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 3), v_src3);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 4), v_src4);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 5), v_src5);                \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

#define MERGE4_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_interleave, flavor, se) \
template <>                                                                                \
struct VMerge4<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VMerge4()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(se);                                                \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src0, const data_type * src1,                        \
                    const data_type * src2, const data_type * src3,                        \
                    data_type * dst) const                                                 \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((const cast_type *)(src0));                   \
        reg_type v_src1 = _mm_loadu_##flavor((const cast_type *)(src0 + ELEMS_IN_VEC));    \
        reg_type v_src2 = _mm_loadu_##flavor((const cast_type *)(src1));                   \
        reg_type v_src3 = _mm_loadu_##flavor((const cast_type *)(src1 + ELEMS_IN_VEC));    \
        reg_type v_src4 = _mm_loadu_##flavor((const cast_type *)(src2));                   \
        reg_type v_src5 = _mm_loadu_##flavor((const cast_type *)(src2 + ELEMS_IN_VEC));    \
        reg_type v_src6 = _mm_loadu_##flavor((const cast_type *)(src3));                   \
        reg_type v_src7 = _mm_loadu_##flavor((const cast_type *)(src3 + ELEMS_IN_VEC));    \
                                                                                           \
        _mm_interleave(v_src0, v_src1, v_src2, v_src3,                                     \
                       v_src4, v_src5, v_src6, v_src7);                                    \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst), v_src0);                                   \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC), v_src1);                    \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 2), v_src2);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 3), v_src3);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 4), v_src4);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 5), v_src5);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 6), v_src6);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 7), v_src7);                \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

MERGE2_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_interleave_epi8, si128, CV_CPU_SSE2);
MERGE3_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_interleave_epi8, si128, CV_CPU_SSE2);
MERGE4_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_interleave_epi8, si128, CV_CPU_SSE2);

#if CV_SSE4_1
MERGE2_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_interleave_epi16, si128, CV_CPU_SSE4_1);
MERGE3_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_interleave_epi16, si128, CV_CPU_SSE4_1);
MERGE4_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_interleave_epi16, si128, CV_CPU_SSE4_1);
#endif

MERGE2_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_interleave_ps, ps, CV_CPU_SSE2);
MERGE3_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_interleave_ps, ps, CV_CPU_SSE2);
MERGE4_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_interleave_ps, ps, CV_CPU_SSE2);

#endif

template<typename T> static void
merge_( const T** src, T* dst, int len, int cn )
{
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        const T* src0 = src[0];
        for( i = j = 0; i < len; i++, j += cn )
            dst[j] = src0[i];
    }
    else if( k == 2 )
    {
        const T *src0 = src[0], *src1 = src[1];
        i = j = 0;
#if CV_NEON
        if(cn == 2)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 2 * inc_i;

            VMerge2<T> vmerge;
            for( ; i < len - inc_i; i += inc_i, j += inc_j)
                vmerge(src0 + i, src1 + i, dst + j);
        }
#elif CV_SSE2
        if(cn == 2)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 2 * inc_i;

            VMerge2<T> vmerge;
            if (vmerge.support)
                for( ; i < len - inc_i; i += inc_i, j += inc_j)
                    vmerge(src0 + i, src1 + i, dst + j);
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const T *src0 = src[0], *src1 = src[1], *src2 = src[2];
        i = j = 0;
#if CV_NEON
        if(cn == 3)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 3 * inc_i;

            VMerge3<T> vmerge;
            for( ; i < len - inc_i; i += inc_i, j += inc_j)
                vmerge(src0 + i, src1 + i, src2 + i, dst + j);
        }
#elif CV_SSE2
        if(cn == 3)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 3 * inc_i;

            VMerge3<T> vmerge;
            if (vmerge.support)
                for( ; i < len - inc_i; i += inc_i, j += inc_j)
                    vmerge(src0 + i, src1 + i, src2 + i, dst + j);
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
            dst[j+2] = src2[i];
        }
    }
    else
    {
        const T *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        i = j = 0;
#if CV_NEON
        if(cn == 4)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 4 * inc_i;

            VMerge4<T> vmerge;
            for( ; i < len - inc_i; i += inc_i, j += inc_j)
                vmerge(src0 + i, src1 + i, src2 + i, src3 + i, dst + j);
        }
#elif CV_SSE2
        if(cn == 4)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 4 * inc_i;

            VMerge4<T> vmerge;
            if (vmerge.support)
                for( ; i < len - inc_i; i += inc_i, j += inc_j)
                    vmerge(src0 + i, src1 + i, src2 + i, src3 + i, dst + j);
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }

    for( ; k < cn; k += 4 )
    {
        const T *src0 = src[k], *src1 = src[k+1], *src2 = src[k+2], *src3 = src[k+3];
        for( i = 0, j = k; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
}


void merge8u(const uchar** src, uchar* dst, int len, int cn )
{
    CALL_HAL(merge8u, cv_hal_merge8u, src, dst, len, cn)
    merge_(src, dst, len, cn);
}

void merge16u(const ushort** src, ushort* dst, int len, int cn )
{
    CALL_HAL(merge16u, cv_hal_merge16u, src, dst, len, cn)
    merge_(src, dst, len, cn);
}

void merge32s(const int** src, int* dst, int len, int cn )
{
    CALL_HAL(merge32s, cv_hal_merge32s, src, dst, len, cn)
    merge_(src, dst, len, cn);
}

void merge64s(const int64** src, int64* dst, int len, int cn )
{
    CALL_HAL(merge64s, cv_hal_merge64s, src, dst, len, cn)
    merge_(src, dst, len, cn);
}

}}
