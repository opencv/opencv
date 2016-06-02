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
template<typename T> struct VSplit2;
template<typename T> struct VSplit3;
template<typename T> struct VSplit4;

#define SPLIT2_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>                                                        \
    {                                                                             \
        void operator()(const data_type* src, data_type* dst0,                    \
                        data_type* dst1) const                                    \
        {                                                                         \
            reg_type r = load_func(src);                                          \
            store_func(dst0, r.val[0]);                                           \
            store_func(dst1, r.val[1]);                                           \
        }                                                                         \
    }

#define SPLIT3_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>                                                        \
    {                                                                             \
        void operator()(const data_type* src, data_type* dst0, data_type* dst1,   \
                        data_type* dst2) const                                    \
        {                                                                         \
            reg_type r = load_func(src);                                          \
            store_func(dst0, r.val[0]);                                           \
            store_func(dst1, r.val[1]);                                           \
            store_func(dst2, r.val[2]);                                           \
        }                                                                         \
    }

#define SPLIT4_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>                                                        \
    {                                                                             \
        void operator()(const data_type* src, data_type* dst0, data_type* dst1,   \
                        data_type* dst2, data_type* dst3) const                   \
        {                                                                         \
            reg_type r = load_func(src);                                          \
            store_func(dst0, r.val[0]);                                           \
            store_func(dst1, r.val[1]);                                           \
            store_func(dst2, r.val[2]);                                           \
            store_func(dst3, r.val[3]);                                           \
        }                                                                         \
    }

SPLIT2_KERNEL_TEMPLATE(VSplit2, uchar ,  uint8x16x2_t, vld2q_u8 , vst1q_u8 );
SPLIT2_KERNEL_TEMPLATE(VSplit2, ushort,  uint16x8x2_t, vld2q_u16, vst1q_u16);
SPLIT2_KERNEL_TEMPLATE(VSplit2, int   ,   int32x4x2_t, vld2q_s32, vst1q_s32);
SPLIT2_KERNEL_TEMPLATE(VSplit2, int64 ,   int64x1x2_t, vld2_s64 , vst1_s64 );

SPLIT3_KERNEL_TEMPLATE(VSplit3, uchar ,  uint8x16x3_t, vld3q_u8 , vst1q_u8 );
SPLIT3_KERNEL_TEMPLATE(VSplit3, ushort,  uint16x8x3_t, vld3q_u16, vst1q_u16);
SPLIT3_KERNEL_TEMPLATE(VSplit3, int   ,   int32x4x3_t, vld3q_s32, vst1q_s32);
SPLIT3_KERNEL_TEMPLATE(VSplit3, int64 ,   int64x1x3_t, vld3_s64 , vst1_s64 );

SPLIT4_KERNEL_TEMPLATE(VSplit4, uchar ,  uint8x16x4_t, vld4q_u8 , vst1q_u8 );
SPLIT4_KERNEL_TEMPLATE(VSplit4, ushort,  uint16x8x4_t, vld4q_u16, vst1q_u16);
SPLIT4_KERNEL_TEMPLATE(VSplit4, int   ,   int32x4x4_t, vld4q_s32, vst1q_s32);
SPLIT4_KERNEL_TEMPLATE(VSplit4, int64 ,   int64x1x4_t, vld4_s64 , vst1_s64 );

#elif CV_SSE2

template <typename T>
struct VSplit2
{
    VSplit2() : support(false) { }
    void operator()(const T *, T *, T *) const { }

    bool support;
};

template <typename T>
struct VSplit3
{
    VSplit3() : support(false) { }
    void operator()(const T *, T *, T *, T *) const { }

    bool support;
};

template <typename T>
struct VSplit4
{
    VSplit4() : support(false) { }
    void operator()(const T *, T *, T *, T *, T *) const { }

    bool support;
};

#define SPLIT2_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_deinterleave, flavor)   \
template <>                                                                                \
struct VSplit2<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VSplit2()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(CV_CPU_SSE2);                                       \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src,                                                 \
                    data_type * dst0, data_type * dst1) const                              \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((cast_type const *)(src));                    \
        reg_type v_src1 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC));     \
        reg_type v_src2 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 2)); \
        reg_type v_src3 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 3)); \
                                                                                           \
        _mm_deinterleave(v_src0, v_src1, v_src2, v_src3);                                  \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst0), v_src0);                                  \
        _mm_storeu_##flavor((cast_type *)(dst0 + ELEMS_IN_VEC), v_src1);                   \
        _mm_storeu_##flavor((cast_type *)(dst1), v_src2);                                  \
        _mm_storeu_##flavor((cast_type *)(dst1 + ELEMS_IN_VEC), v_src3);                   \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

#define SPLIT3_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_deinterleave, flavor)   \
template <>                                                                                \
struct VSplit3<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VSplit3()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(CV_CPU_SSE2);                                       \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src,                                                 \
                    data_type * dst0, data_type * dst1, data_type * dst2) const            \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((cast_type const *)(src));                    \
        reg_type v_src1 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC));     \
        reg_type v_src2 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 2)); \
        reg_type v_src3 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 3)); \
        reg_type v_src4 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 4)); \
        reg_type v_src5 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 5)); \
                                                                                           \
        _mm_deinterleave(v_src0, v_src1, v_src2,                                           \
                         v_src3, v_src4, v_src5);                                          \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst0), v_src0);                                  \
        _mm_storeu_##flavor((cast_type *)(dst0 + ELEMS_IN_VEC), v_src1);                   \
        _mm_storeu_##flavor((cast_type *)(dst1), v_src2);                                  \
        _mm_storeu_##flavor((cast_type *)(dst1 + ELEMS_IN_VEC), v_src3);                   \
        _mm_storeu_##flavor((cast_type *)(dst2), v_src4);                                  \
        _mm_storeu_##flavor((cast_type *)(dst2 + ELEMS_IN_VEC), v_src5);                   \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

#define SPLIT4_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_deinterleave, flavor)   \
template <>                                                                                \
struct VSplit4<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VSplit4()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(CV_CPU_SSE2);                                       \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src, data_type * dst0, data_type * dst1,             \
                    data_type * dst2, data_type * dst3) const                              \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((cast_type const *)(src));                    \
        reg_type v_src1 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC));     \
        reg_type v_src2 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 2)); \
        reg_type v_src3 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 3)); \
        reg_type v_src4 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 4)); \
        reg_type v_src5 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 5)); \
        reg_type v_src6 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 6)); \
        reg_type v_src7 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 7)); \
                                                                                           \
        _mm_deinterleave(v_src0, v_src1, v_src2, v_src3,                                   \
                         v_src4, v_src5, v_src6, v_src7);                                  \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst0), v_src0);                                  \
        _mm_storeu_##flavor((cast_type *)(dst0 + ELEMS_IN_VEC), v_src1);                   \
        _mm_storeu_##flavor((cast_type *)(dst1), v_src2);                                  \
        _mm_storeu_##flavor((cast_type *)(dst1 + ELEMS_IN_VEC), v_src3);                   \
        _mm_storeu_##flavor((cast_type *)(dst2), v_src4);                                  \
        _mm_storeu_##flavor((cast_type *)(dst2 + ELEMS_IN_VEC), v_src5);                   \
        _mm_storeu_##flavor((cast_type *)(dst3), v_src6);                                  \
        _mm_storeu_##flavor((cast_type *)(dst3 + ELEMS_IN_VEC), v_src7);                   \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

SPLIT2_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_deinterleave_epi8, si128);
SPLIT2_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_deinterleave_epi16, si128);
SPLIT2_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_deinterleave_ps, ps);

SPLIT3_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_deinterleave_epi8, si128);
SPLIT3_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_deinterleave_epi16, si128);
SPLIT3_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_deinterleave_ps, ps);

SPLIT4_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_deinterleave_epi8, si128);
SPLIT4_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_deinterleave_epi16, si128);
SPLIT4_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_deinterleave_ps, ps);

#endif

template<typename T> static void
split_( const T* src, T** dst, int len, int cn )
{
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        T* dst0 = dst[0];

        if(cn == 1)
        {
            memcpy(dst0, src, len * sizeof(T));
        }
        else
        {
            for( i = 0, j = 0 ; i < len; i++, j += cn )
                dst0[i] = src[j];
        }
    }
    else if( k == 2 )
    {
        T *dst0 = dst[0], *dst1 = dst[1];
        i = j = 0;

#if CV_NEON
        if(cn == 2)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 2 * inc_i;

            VSplit2<T> vsplit;
            for( ; i < len - inc_i; i += inc_i, j += inc_j)
                vsplit(src + j, dst0 + i, dst1 + i);
        }
#elif CV_SSE2
        if (cn == 2)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 2 * inc_i;

            VSplit2<T> vsplit;
            if (vsplit.support)
            {
                for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                    vsplit(src + j, dst0 + i, dst1 + i);
            }
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst0[i] = src[j];
            dst1[i] = src[j+1];
        }
    }
    else if( k == 3 )
    {
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        i = j = 0;

#if CV_NEON
        if(cn == 3)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 3 * inc_i;

            VSplit3<T> vsplit;
            for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                vsplit(src + j, dst0 + i, dst1 + i, dst2 + i);
        }
#elif CV_SSE2
        if (cn == 3)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 3 * inc_i;

            VSplit3<T> vsplit;

            if (vsplit.support)
            {
                for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                    vsplit(src + j, dst0 + i, dst1 + i, dst2 + i);
            }
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst0[i] = src[j];
            dst1[i] = src[j+1];
            dst2[i] = src[j+2];
        }
    }
    else
    {
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        i = j = 0;

#if CV_NEON
        if(cn == 4)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 4 * inc_i;

            VSplit4<T> vsplit;
            for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                vsplit(src + j, dst0 + i, dst1 + i, dst2 + i, dst3 + i);
        }
#elif CV_SSE2
        if (cn == 4)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 4 * inc_i;

            VSplit4<T> vsplit;
            if (vsplit.support)
            {
                for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                    vsplit(src + j, dst0 + i, dst1 + i, dst2 + i, dst3 + i);
            }
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst0[i] = src[j]; dst1[i] = src[j+1];
            dst2[i] = src[j+2]; dst3[i] = src[j+3];
        }
    }

    for( ; k < cn; k += 4 )
    {
        T *dst0 = dst[k], *dst1 = dst[k+1], *dst2 = dst[k+2], *dst3 = dst[k+3];
        for( i = 0, j = k; i < len; i++, j += cn )
        {
            dst0[i] = src[j]; dst1[i] = src[j+1];
            dst2[i] = src[j+2]; dst3[i] = src[j+3];
        }
    }
}

void split8u(const uchar* src, uchar** dst, int len, int cn )
{
    CALL_HAL(split8u, cv_hal_split8u, src,dst, len, cn)
    split_(src, dst, len, cn);
}

void split16u(const ushort* src, ushort** dst, int len, int cn )
{
    CALL_HAL(split16u, cv_hal_split16u, src,dst, len, cn)
    split_(src, dst, len, cn);
}

void split32s(const int* src, int** dst, int len, int cn )
{
    CALL_HAL(split32s, cv_hal_split32s, src,dst, len, cn)
    split_(src, dst, len, cn);
}

void split64s(const int64* src, int64** dst, int len, int cn )
{
    CALL_HAL(split64s, cv_hal_split64s, src,dst, len, cn)
    split_(src, dst, len, cn);
}

}}
