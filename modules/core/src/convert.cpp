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
#include "opencl_kernels_core.hpp"

#ifdef __APPLE__
#undef CV_NEON
#define CV_NEON 0
#endif

namespace cv
{

/****************************************************************************************\
*                                       split & merge                                    *
\****************************************************************************************/

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

static void split8u(const uchar* src, uchar** dst, int len, int cn )
{
    split_(src, dst, len, cn);
}

static void split16u(const ushort* src, ushort** dst, int len, int cn )
{
    split_(src, dst, len, cn);
}

static void split32s(const int* src, int** dst, int len, int cn )
{
    split_(src, dst, len, cn);
}

static void split64s(const int64* src, int64** dst, int len, int cn )
{
    split_(src, dst, len, cn);
}

static void merge8u(const uchar** src, uchar* dst, int len, int cn )
{
    merge_(src, dst, len, cn);
}

static void merge16u(const ushort** src, ushort* dst, int len, int cn )
{
    merge_(src, dst, len, cn);
}

static void merge32s(const int** src, int* dst, int len, int cn )
{
    merge_(src, dst, len, cn);
}

static void merge64s(const int64** src, int64* dst, int len, int cn )
{
    merge_(src, dst, len, cn);
}

typedef void (*SplitFunc)(const uchar* src, uchar** dst, int len, int cn);
typedef void (*MergeFunc)(const uchar** src, uchar* dst, int len, int cn);

static SplitFunc getSplitFunc(int depth)
{
    static SplitFunc splitTab[] =
    {
        (SplitFunc)GET_OPTIMIZED(split8u), (SplitFunc)GET_OPTIMIZED(split8u), (SplitFunc)GET_OPTIMIZED(split16u), (SplitFunc)GET_OPTIMIZED(split16u),
        (SplitFunc)GET_OPTIMIZED(split32s), (SplitFunc)GET_OPTIMIZED(split32s), (SplitFunc)GET_OPTIMIZED(split64s), 0
    };

    return splitTab[depth];
}

static MergeFunc getMergeFunc(int depth)
{
    static MergeFunc mergeTab[] =
    {
        (MergeFunc)GET_OPTIMIZED(merge8u), (MergeFunc)GET_OPTIMIZED(merge8u), (MergeFunc)GET_OPTIMIZED(merge16u), (MergeFunc)GET_OPTIMIZED(merge16u),
        (MergeFunc)GET_OPTIMIZED(merge32s), (MergeFunc)GET_OPTIMIZED(merge32s), (MergeFunc)GET_OPTIMIZED(merge64s), 0
    };

    return mergeTab[depth];
}

}

void cv::split(const Mat& src, Mat* mv)
{
    int k, depth = src.depth(), cn = src.channels();
    if( cn == 1 )
    {
        src.copyTo(mv[0]);
        return;
    }

    SplitFunc func = getSplitFunc(depth);
    CV_Assert( func != 0 );

    int esz = (int)src.elemSize(), esz1 = (int)src.elemSize1();
    int blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &src;
    for( k = 0; k < cn; k++ )
    {
        mv[k].create(src.dims, src.size, depth);
        arrays[k+1] = &mv[k];
    }

    NAryMatIterator it(arrays, ptrs, cn+1);
    int total = (int)it.size, blocksize = cn <= 4 ? total : std::min(total, blocksize0);

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( int j = 0; j < total; j += blocksize )
        {
            int bsz = std::min(total - j, blocksize);
            func( ptrs[0], &ptrs[1], bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( k = 0; k < cn; k++ )
                    ptrs[k+1] += bsz*esz1;
            }
        }
    }
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_split( InputArray _m, OutputArrayOfArrays _mv )
{
    int type = _m.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;

    String dstargs, processelem, indexdecl;
    for (int i = 0; i < cn; ++i)
    {
        dstargs += format("DECLARE_DST_PARAM(%d)", i);
        indexdecl += format("DECLARE_INDEX(%d)", i);
        processelem += format("PROCESS_ELEM(%d)", i);
    }

    ocl::Kernel k("split", ocl::core::split_merge_oclsrc,
                  format("-D T=%s -D OP_SPLIT -D cn=%d -D DECLARE_DST_PARAMS=%s"
                         " -D PROCESS_ELEMS_N=%s -D DECLARE_INDEX_N=%s",
                         ocl::memopTypeToStr(depth), cn, dstargs.c_str(),
                         processelem.c_str(), indexdecl.c_str()));
    if (k.empty())
        return false;

    Size size = _m.size();
    _mv.create(cn, 1, depth);
    for (int i = 0; i < cn; ++i)
        _mv.create(size, depth, i);

    std::vector<UMat> dst;
    _mv.getUMatVector(dst);

    int argidx = k.set(0, ocl::KernelArg::ReadOnly(_m.getUMat()));
    for (int i = 0; i < cn; ++i)
        argidx = k.set(argidx, ocl::KernelArg::WriteOnlyNoSize(dst[i]));
    k.set(argidx, rowsPerWI);

    size_t globalsize[2] = { size.width, (size.height + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::split(InputArray _m, OutputArrayOfArrays _mv)
{
    CV_OCL_RUN(_m.dims() <= 2 && _mv.isUMatVector(),
               ocl_split(_m, _mv))

    Mat m = _m.getMat();
    if( m.empty() )
    {
        _mv.release();
        return;
    }

    CV_Assert( !_mv.fixedType() || _mv.empty() || _mv.type() == m.depth() );

    Size size = m.size();
    int depth = m.depth(), cn = m.channels();
    _mv.create(cn, 1, depth);
    for (int i = 0; i < cn; ++i)
        _mv.create(size, depth, i);

    std::vector<Mat> dst;
    _mv.getMatVector(dst);

    split(m, &dst[0]);
}

void cv::merge(const Mat* mv, size_t n, OutputArray _dst)
{
    CV_Assert( mv && n > 0 );

    int depth = mv[0].depth();
    bool allch1 = true;
    int k, cn = 0;
    size_t i;

    for( i = 0; i < n; i++ )
    {
        CV_Assert(mv[i].size == mv[0].size && mv[i].depth() == depth);
        allch1 = allch1 && mv[i].channels() == 1;
        cn += mv[i].channels();
    }

    CV_Assert( 0 < cn && cn <= CV_CN_MAX );
    _dst.create(mv[0].dims, mv[0].size, CV_MAKETYPE(depth, cn));
    Mat dst = _dst.getMat();

    if( n == 1 )
    {
        mv[0].copyTo(dst);
        return;
    }

    if( !allch1 )
    {
        AutoBuffer<int> pairs(cn*2);
        int j, ni=0;

        for( i = 0, j = 0; i < n; i++, j += ni )
        {
            ni = mv[i].channels();
            for( k = 0; k < ni; k++ )
            {
                pairs[(j+k)*2] = j + k;
                pairs[(j+k)*2+1] = j + k;
            }
        }
        mixChannels( mv, n, &dst, 1, &pairs[0], cn );
        return;
    }

    size_t esz = dst.elemSize(), esz1 = dst.elemSize1();
    int blocksize0 = (int)((BLOCK_SIZE + esz-1)/esz);
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &dst;
    for( k = 0; k < cn; k++ )
        arrays[k+1] = &mv[k];

    NAryMatIterator it(arrays, ptrs, cn+1);
    int total = (int)it.size, blocksize = cn <= 4 ? total : std::min(total, blocksize0);
    MergeFunc func = getMergeFunc(depth);

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( int j = 0; j < total; j += blocksize )
        {
            int bsz = std::min(total - j, blocksize);
            func( (const uchar**)&ptrs[1], ptrs[0], bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( int t = 0; t < cn; t++ )
                    ptrs[t+1] += bsz*esz1;
            }
        }
    }
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_merge( InputArrayOfArrays _mv, OutputArray _dst )
{
    std::vector<UMat> src, ksrc;
    _mv.getUMatVector(src);
    CV_Assert(!src.empty());

    int type = src[0].type(), depth = CV_MAT_DEPTH(type),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
    Size size = src[0].size();

    for (size_t i = 0, srcsize = src.size(); i < srcsize; ++i)
    {
        int itype = src[i].type(), icn = CV_MAT_CN(itype), idepth = CV_MAT_DEPTH(itype),
                esz1 = CV_ELEM_SIZE1(idepth);
        if (src[i].dims > 2)
            return false;

        CV_Assert(size == src[i].size() && depth == idepth);

        for (int cn = 0; cn < icn; ++cn)
        {
            UMat tsrc = src[i];
            tsrc.offset += cn * esz1;
            ksrc.push_back(tsrc);
        }
    }
    int dcn = (int)ksrc.size();

    String srcargs, processelem, cndecl, indexdecl;
    for (int i = 0; i < dcn; ++i)
    {
        srcargs += format("DECLARE_SRC_PARAM(%d)", i);
        processelem += format("PROCESS_ELEM(%d)", i);
        indexdecl += format("DECLARE_INDEX(%d)", i);
        cndecl += format(" -D scn%d=%d", i, ksrc[i].channels());
    }

    ocl::Kernel k("merge", ocl::core::split_merge_oclsrc,
                  format("-D OP_MERGE -D cn=%d -D T=%s -D DECLARE_SRC_PARAMS_N=%s"
                         " -D DECLARE_INDEX_N=%s -D PROCESS_ELEMS_N=%s%s",
                         dcn, ocl::memopTypeToStr(depth), srcargs.c_str(),
                         indexdecl.c_str(), processelem.c_str(), cndecl.c_str()));
    if (k.empty())
        return false;

    _dst.create(size, CV_MAKE_TYPE(depth, dcn));
    UMat dst = _dst.getUMat();

    int argidx = 0;
    for (int i = 0; i < dcn; ++i)
        argidx = k.set(argidx, ocl::KernelArg::ReadOnlyNoSize(ksrc[i]));
    argidx = k.set(argidx, ocl::KernelArg::WriteOnly(dst));
    k.set(argidx, rowsPerWI);

    size_t globalsize[2] = { dst.cols, (dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::merge(InputArrayOfArrays _mv, OutputArray _dst)
{
    CV_OCL_RUN(_mv.isUMatVector() && _dst.isUMat(),
               ocl_merge(_mv, _dst))

    std::vector<Mat> mv;
    _mv.getMatVector(mv);
    merge(!mv.empty() ? &mv[0] : 0, mv.size(), _dst);
}

/****************************************************************************************\
*                       Generalized split/merge: mixing channels                         *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
mixChannels_( const T** src, const int* sdelta,
              T** dst, const int* ddelta,
              int len, int npairs )
{
    int i, k;
    for( k = 0; k < npairs; k++ )
    {
        const T* s = src[k];
        T* d = dst[k];
        int ds = sdelta[k], dd = ddelta[k];
        if( s )
        {
            for( i = 0; i <= len - 2; i += 2, s += ds*2, d += dd*2 )
            {
                T t0 = s[0], t1 = s[ds];
                d[0] = t0; d[dd] = t1;
            }
            if( i < len )
                d[0] = s[0];
        }
        else
        {
            for( i = 0; i <= len - 2; i += 2, d += dd*2 )
                d[0] = d[dd] = 0;
            if( i < len )
                d[0] = 0;
        }
    }
}


static void mixChannels8u( const uchar** src, const int* sdelta,
                           uchar** dst, const int* ddelta,
                           int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels16u( const ushort** src, const int* sdelta,
                            ushort** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels32s( const int** src, const int* sdelta,
                            int** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels64s( const int64** src, const int* sdelta,
                            int64** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

typedef void (*MixChannelsFunc)( const uchar** src, const int* sdelta,
        uchar** dst, const int* ddelta, int len, int npairs );

static MixChannelsFunc getMixchFunc(int depth)
{
    static MixChannelsFunc mixchTab[] =
    {
        (MixChannelsFunc)mixChannels8u, (MixChannelsFunc)mixChannels8u, (MixChannelsFunc)mixChannels16u,
        (MixChannelsFunc)mixChannels16u, (MixChannelsFunc)mixChannels32s, (MixChannelsFunc)mixChannels32s,
        (MixChannelsFunc)mixChannels64s, 0
    };

    return mixchTab[depth];
}

}

void cv::mixChannels( const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts, const int* fromTo, size_t npairs )
{
    if( npairs == 0 )
        return;
    CV_Assert( src && nsrcs > 0 && dst && ndsts > 0 && fromTo && npairs > 0 );

    size_t i, j, k, esz1 = dst[0].elemSize1();
    int depth = dst[0].depth();

    AutoBuffer<uchar> buf((nsrcs + ndsts + 1)*(sizeof(Mat*) + sizeof(uchar*)) + npairs*(sizeof(uchar*)*2 + sizeof(int)*6));
    const Mat** arrays = (const Mat**)(uchar*)buf;
    uchar** ptrs = (uchar**)(arrays + nsrcs + ndsts);
    const uchar** srcs = (const uchar**)(ptrs + nsrcs + ndsts + 1);
    uchar** dsts = (uchar**)(srcs + npairs);
    int* tab = (int*)(dsts + npairs);
    int *sdelta = (int*)(tab + npairs*4), *ddelta = sdelta + npairs;

    for( i = 0; i < nsrcs; i++ )
        arrays[i] = &src[i];
    for( i = 0; i < ndsts; i++ )
        arrays[i + nsrcs] = &dst[i];
    ptrs[nsrcs + ndsts] = 0;

    for( i = 0; i < npairs; i++ )
    {
        int i0 = fromTo[i*2], i1 = fromTo[i*2+1];
        if( i0 >= 0 )
        {
            for( j = 0; j < nsrcs; i0 -= src[j].channels(), j++ )
                if( i0 < src[j].channels() )
                    break;
            CV_Assert(j < nsrcs && src[j].depth() == depth);
            tab[i*4] = (int)j; tab[i*4+1] = (int)(i0*esz1);
            sdelta[i] = src[j].channels();
        }
        else
        {
            tab[i*4] = (int)(nsrcs + ndsts); tab[i*4+1] = 0;
            sdelta[i] = 0;
        }

        for( j = 0; j < ndsts; i1 -= dst[j].channels(), j++ )
            if( i1 < dst[j].channels() )
                break;
        CV_Assert(i1 >= 0 && j < ndsts && dst[j].depth() == depth);
        tab[i*4+2] = (int)(j + nsrcs); tab[i*4+3] = (int)(i1*esz1);
        ddelta[i] = dst[j].channels();
    }

    NAryMatIterator it(arrays, ptrs, (int)(nsrcs + ndsts));
    int total = (int)it.size, blocksize = std::min(total, (int)((BLOCK_SIZE + esz1-1)/esz1));
    MixChannelsFunc func = getMixchFunc(depth);

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( k = 0; k < npairs; k++ )
        {
            srcs[k] = ptrs[tab[k*4]] + tab[k*4+1];
            dsts[k] = ptrs[tab[k*4+2]] + tab[k*4+3];
        }

        for( int t = 0; t < total; t += blocksize )
        {
            int bsz = std::min(total - t, blocksize);
            func( srcs, sdelta, dsts, ddelta, bsz, (int)npairs );

            if( t + blocksize < total )
                for( k = 0; k < npairs; k++ )
                {
                    srcs[k] += blocksize*sdelta[k]*esz1;
                    dsts[k] += blocksize*ddelta[k]*esz1;
                }
        }
    }
}

#ifdef HAVE_OPENCL

namespace cv {

static void getUMatIndex(const std::vector<UMat> & um, int cn, int & idx, int & cnidx)
{
    int totalChannels = 0;
    for (size_t i = 0, size = um.size(); i < size; ++i)
    {
        int ccn = um[i].channels();
        totalChannels += ccn;

        if (totalChannels == cn)
        {
            idx = (int)(i + 1);
            cnidx = 0;
            return;
        }
        else if (totalChannels > cn)
        {
            idx = (int)i;
            cnidx = i == 0 ? cn : (cn - totalChannels + ccn);
            return;
        }
    }

    idx = cnidx = -1;
}

static bool ocl_mixChannels(InputArrayOfArrays _src, InputOutputArrayOfArrays _dst,
                            const int* fromTo, size_t npairs)
{
    std::vector<UMat> src, dst;
    _src.getUMatVector(src);
    _dst.getUMatVector(dst);

    size_t nsrc = src.size(), ndst = dst.size();
    CV_Assert(nsrc > 0 && ndst > 0);

    Size size = src[0].size();
    int depth = src[0].depth(), esz = CV_ELEM_SIZE(depth),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;

    for (size_t i = 1, ssize = src.size(); i < ssize; ++i)
        CV_Assert(src[i].size() == size && src[i].depth() == depth);
    for (size_t i = 0, dsize = dst.size(); i < dsize; ++i)
        CV_Assert(dst[i].size() == size && dst[i].depth() == depth);

    String declsrc, decldst, declproc, declcn, indexdecl;
    std::vector<UMat> srcargs(npairs), dstargs(npairs);

    for (size_t i = 0; i < npairs; ++i)
    {
        int scn = fromTo[i<<1], dcn = fromTo[(i<<1) + 1];
        int src_idx, src_cnidx, dst_idx, dst_cnidx;

        getUMatIndex(src, scn, src_idx, src_cnidx);
        getUMatIndex(dst, dcn, dst_idx, dst_cnidx);

        CV_Assert(dst_idx >= 0 && src_idx >= 0);

        srcargs[i] = src[src_idx];
        srcargs[i].offset += src_cnidx * esz;

        dstargs[i] = dst[dst_idx];
        dstargs[i].offset += dst_cnidx * esz;

        declsrc += format("DECLARE_INPUT_MAT(%d)", i);
        decldst += format("DECLARE_OUTPUT_MAT(%d)", i);
        indexdecl += format("DECLARE_INDEX(%d)", i);
        declproc += format("PROCESS_ELEM(%d)", i);
        declcn += format(" -D scn%d=%d -D dcn%d=%d", i, src[src_idx].channels(), i, dst[dst_idx].channels());
    }

    ocl::Kernel k("mixChannels", ocl::core::mixchannels_oclsrc,
                  format("-D T=%s -D DECLARE_INPUT_MAT_N=%s -D DECLARE_OUTPUT_MAT_N=%s"
                         " -D PROCESS_ELEM_N=%s -D DECLARE_INDEX_N=%s%s",
                         ocl::memopTypeToStr(depth), declsrc.c_str(), decldst.c_str(),
                         declproc.c_str(), indexdecl.c_str(), declcn.c_str()));
    if (k.empty())
        return false;

    int argindex = 0;
    for (size_t i = 0; i < npairs; ++i)
        argindex = k.set(argindex, ocl::KernelArg::ReadOnlyNoSize(srcargs[i]));
    for (size_t i = 0; i < npairs; ++i)
        argindex = k.set(argindex, ocl::KernelArg::WriteOnlyNoSize(dstargs[i]));
    argindex = k.set(argindex, size.height);
    argindex = k.set(argindex, size.width);
    k.set(argindex, rowsPerWI);

    size_t globalsize[2] = { size.width, (size.height + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                 const int* fromTo, size_t npairs)
{
    if (npairs == 0 || fromTo == NULL)
        return;

    CV_OCL_RUN(dst.isUMatVector(),
               ocl_mixChannels(src, dst, fromTo, npairs))

    bool src_is_mat = src.kind() != _InputArray::STD_VECTOR_MAT &&
            src.kind() != _InputArray::STD_VECTOR_VECTOR &&
            src.kind() != _InputArray::STD_VECTOR_UMAT;
    bool dst_is_mat = dst.kind() != _InputArray::STD_VECTOR_MAT &&
            dst.kind() != _InputArray::STD_VECTOR_VECTOR &&
            dst.kind() != _InputArray::STD_VECTOR_UMAT;
    int i;
    int nsrc = src_is_mat ? 1 : (int)src.total();
    int ndst = dst_is_mat ? 1 : (int)dst.total();

    CV_Assert(nsrc > 0 && ndst > 0);
    cv::AutoBuffer<Mat> _buf(nsrc + ndst);
    Mat* buf = _buf;
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, fromTo, npairs);
}

void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                     const std::vector<int>& fromTo)
{
    if (fromTo.empty())
        return;

    CV_OCL_RUN(dst.isUMatVector(),
               ocl_mixChannels(src, dst, &fromTo[0], fromTo.size()>>1))

    bool src_is_mat = src.kind() != _InputArray::STD_VECTOR_MAT &&
            src.kind() != _InputArray::STD_VECTOR_VECTOR &&
            src.kind() != _InputArray::STD_VECTOR_UMAT;
    bool dst_is_mat = dst.kind() != _InputArray::STD_VECTOR_MAT &&
            dst.kind() != _InputArray::STD_VECTOR_VECTOR &&
            dst.kind() != _InputArray::STD_VECTOR_UMAT;
    int i;
    int nsrc = src_is_mat ? 1 : (int)src.total();
    int ndst = dst_is_mat ? 1 : (int)dst.total();

    CV_Assert(fromTo.size()%2 == 0 && nsrc > 0 && ndst > 0);
    cv::AutoBuffer<Mat> _buf(nsrc + ndst);
    Mat* buf = _buf;
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, &fromTo[0], fromTo.size()/2);
}

void cv::extractChannel(InputArray _src, OutputArray _dst, int coi)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert( 0 <= coi && coi < cn );
    int ch[] = { coi, 0 };

    if (ocl::useOpenCL() && _src.dims() <= 2 && _dst.isUMat())
    {
        UMat src = _src.getUMat();
        _dst.create(src.dims, &src.size[0], depth);
        UMat dst = _dst.getUMat();
        mixChannels(std::vector<UMat>(1, src), std::vector<UMat>(1, dst), ch, 1);
        return;
    }

    Mat src = _src.getMat();
    _dst.create(src.dims, &src.size[0], depth);
    Mat dst = _dst.getMat();
    mixChannels(&src, 1, &dst, 1, ch, 1);
}

void cv::insertChannel(InputArray _src, InputOutputArray _dst, int coi)
{
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);
    CV_Assert( _src.sameSize(_dst) && sdepth == ddepth );
    CV_Assert( 0 <= coi && coi < dcn && scn == 1 );

    int ch[] = { 0, coi };
    if (ocl::useOpenCL() && _src.dims() <= 2 && _dst.isUMat())
    {
        UMat src = _src.getUMat(), dst = _dst.getUMat();
        mixChannels(std::vector<UMat>(1, src), std::vector<UMat>(1, dst), ch, 1);
        return;
    }

    Mat src = _src.getMat(), dst = _dst.getMat();
    mixChannels(&src, 1, &dst, 1, ch, 1);
}

/****************************************************************************************\
*                                convertScale[Abs]                                       *
\****************************************************************************************/

namespace cv
{

template<typename T, typename DT, typename WT>
struct cvtScaleAbs_SIMD
{
    int operator () (const T *, DT *, int, WT, WT) const
    {
        return 0;
    }
};

#if CV_SSE2

template <>
struct cvtScaleAbs_SIMD<uchar, uchar, float>
{
    int operator () (const uchar * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift),
                v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for ( ; x <= width - 16; x += 16)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i *)(src + x));
                __m128i v_src12 = _mm_unpacklo_epi8(v_src, v_zero_i), v_src_34 = _mm_unpackhi_epi8(v_src, v_zero_i);
                __m128 v_dst1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src12, v_zero_i)), v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);
                __m128 v_dst2 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src12, v_zero_i)), v_scale), v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);
                __m128 v_dst3 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src_34, v_zero_i)), v_scale), v_shift);
                v_dst3 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst3), v_dst3);
                __m128 v_dst4 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src_34, v_zero_i)), v_scale), v_shift);
                v_dst4 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst4), v_dst4);

                __m128i v_dst_i = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2)),
                                                   _mm_packs_epi32(_mm_cvtps_epi32(v_dst3), _mm_cvtps_epi32(v_dst4)));
                _mm_storeu_si128((__m128i *)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<schar, uchar, float>
{
    int operator () (const schar * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift),
                v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for ( ; x <= width - 16; x += 16)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i *)(src + x));
                __m128i v_src_12 = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero_i, v_src), 8),
                        v_src_34 = _mm_srai_epi16(_mm_unpackhi_epi8(v_zero_i, v_src), 8);
                __m128 v_dst1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(
                    _mm_srai_epi32(_mm_unpacklo_epi16(v_zero_i, v_src_12), 16)), v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);
                __m128 v_dst2 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(
                    _mm_srai_epi32(_mm_unpackhi_epi16(v_zero_i, v_src_12), 16)), v_scale), v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);
                __m128 v_dst3 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(
                    _mm_srai_epi32(_mm_unpacklo_epi16(v_zero_i, v_src_34), 16)), v_scale), v_shift);
                v_dst3 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst3), v_dst3);
                __m128 v_dst4 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(
                    _mm_srai_epi32(_mm_unpackhi_epi16(v_zero_i, v_src_34), 16)), v_scale), v_shift);
                v_dst4 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst4), v_dst4);

                __m128i v_dst_i = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2)),
                                                   _mm_packs_epi32(_mm_cvtps_epi32(v_dst3), _mm_cvtps_epi32(v_dst4)));
                _mm_storeu_si128((__m128i *)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<ushort, uchar, float>
{
    int operator () (const ushort * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift),
                v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i *)(src + x));
                __m128 v_dst1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero_i)), v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);
                __m128 v_dst2 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero_i)), v_scale), v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);

                __m128i v_dst_i = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2)), v_zero_i);
                _mm_storel_epi64((__m128i *)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<short, uchar, float>
{
    int operator () (const short * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift),
                v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i *)(src + x));
                __m128 v_dst1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_src, v_src), 16)), v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);
                __m128 v_dst2 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_src, v_src), 16)), v_scale), v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);

                __m128i v_dst_i = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2)), v_zero_i);
                _mm_storel_epi64((__m128i *)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<int, uchar, float>
{
    int operator () (const int * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift),
                v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for ( ; x <= width - 8; x += 4)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i *)(src + x));
                __m128 v_dst1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);

                __m128i v_dst_i = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), v_zero_i), v_zero_i);
                _mm_storel_epi64((__m128i *)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<float, uchar, float>
{
    int operator () (const float * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift),
                v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for ( ; x <= width - 8; x += 4)
            {
                __m128 v_dst = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + x), v_scale), v_shift);
                v_dst = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst), v_dst);

                __m128i v_dst_i = _mm_packs_epi32(_mm_cvtps_epi32(v_dst), v_zero_i);
                _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst_i, v_zero_i));
            }
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<double, uchar, float>
{
    int operator () (const double * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift),
                v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for ( ; x <= width - 8; x += 8)
            {
                __m128 v_src1 = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)),
                                              _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
                __m128 v_src2 = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)),
                                              _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));

                __m128 v_dst1 = _mm_add_ps(_mm_mul_ps(v_src1, v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);

                __m128 v_dst2 = _mm_add_ps(_mm_mul_ps(v_src2, v_scale), v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);

                __m128i v_dst_i = _mm_packs_epi32(_mm_cvtps_epi32(v_dst1),
                                                  _mm_cvtps_epi32(v_dst2));

                _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst_i, v_zero_i));
            }
        }

        return x;
    }
};

#elif CV_NEON

template <>
struct cvtScaleAbs_SIMD<uchar, uchar, float>
{
    int operator () (const uchar * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for ( ; x <= width - 16; x += 16)
        {
            uint8x16_t v_src = vld1q_u8(src + x);
            uint16x8_t v_half = vmovl_u8(vget_low_u8(v_src));

            uint32x4_t v_quat = vmovl_u16(vget_low_u16(v_half));
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_u32(v_quat), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));

            v_quat = vmovl_u16(vget_high_u16(v_half));
            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_u32(v_quat), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));

            v_half = vmovl_u8(vget_high_u8(v_src));

            v_quat = vmovl_u16(vget_low_u16(v_half));
            float32x4_t v_dst_2 = vmulq_n_f32(vcvtq_f32_u32(v_quat), scale);
            v_dst_2 = vabsq_f32(vaddq_f32(v_dst_2, v_shift));

            v_quat = vmovl_u16(vget_high_u16(v_half));
            float32x4_t v_dst_3 = vmulq_n_f32(vcvtq_f32_u32(v_quat), scale);
            v_dst_3 = vabsq_f32(vaddq_f32(v_dst_3, v_shift));

            uint16x8_t v_dsti_0 = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_0)),
                vqmovn_u32(cv_vrndq_u32_f32(v_dst_1)));
            uint16x8_t v_dsti_1 = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_2)),
                vqmovn_u32(cv_vrndq_u32_f32(v_dst_3)));

            vst1q_u8(dst + x, vcombine_u8(vqmovn_u16(v_dsti_0), vqmovn_u16(v_dsti_1)));
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<schar, uchar, float>
{
    int operator () (const schar * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for ( ; x <= width - 16; x += 16)
        {
            int8x16_t v_src = vld1q_s8(src + x);
            int16x8_t v_half = vmovl_s8(vget_low_s8(v_src));

            int32x4_t v_quat = vmovl_s16(vget_low_s16(v_half));
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_s32(v_quat), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));

            v_quat = vmovl_s16(vget_high_s16(v_half));
            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_s32(v_quat), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));

            v_half = vmovl_s8(vget_high_s8(v_src));

            v_quat = vmovl_s16(vget_low_s16(v_half));
            float32x4_t v_dst_2 = vmulq_n_f32(vcvtq_f32_s32(v_quat), scale);
            v_dst_2 = vabsq_f32(vaddq_f32(v_dst_2, v_shift));

            v_quat = vmovl_s16(vget_high_s16(v_half));
            float32x4_t v_dst_3 = vmulq_n_f32(vcvtq_f32_s32(v_quat), scale);
            v_dst_3 = vabsq_f32(vaddq_f32(v_dst_3, v_shift));

            uint16x8_t v_dsti_0 = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_0)),
                vqmovn_u32(cv_vrndq_u32_f32(v_dst_1)));
            uint16x8_t v_dsti_1 = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_2)),
                vqmovn_u32(cv_vrndq_u32_f32(v_dst_3)));

            vst1q_u8(dst + x, vcombine_u8(vqmovn_u16(v_dsti_0), vqmovn_u16(v_dsti_1)));
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<ushort, uchar, float>
{
    int operator () (const ushort * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);

            uint32x4_t v_half = vmovl_u16(vget_low_u16(v_src));
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_u32(v_half), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));

            v_half = vmovl_u16(vget_high_u16(v_src));
            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_u32(v_half), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_0)),
                vqmovn_u32(cv_vrndq_u32_f32(v_dst_1)));

            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<short, uchar, float>
{
    int operator () (const short * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);

            int32x4_t v_half = vmovl_s16(vget_low_s16(v_src));
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_s32(v_half), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));

            v_half = vmovl_s16(vget_high_s16(v_src));
            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_s32(v_half), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_0)),
                vqmovn_u32(cv_vrndq_u32_f32(v_dst_1)));

            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<int, uchar, float>
{
    int operator () (const int * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(src + x)), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));
            uint16x4_t v_dsti_0 = vqmovn_u32(cv_vrndq_u32_f32(v_dst_0));

            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));
            uint16x4_t v_dsti_1 = vqmovn_u32(cv_vrndq_u32_f32(v_dst_1));

            uint16x8_t v_dst = vcombine_u16(v_dsti_0, v_dsti_1);
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScaleAbs_SIMD<float, uchar, float>
{
    int operator () (const float * src, uchar * dst, int width,
                     float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst_0 = vmulq_n_f32(vld1q_f32(src + x), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));
            uint16x4_t v_dsti_0 = vqmovn_u32(cv_vrndq_u32_f32(v_dst_0));

            float32x4_t v_dst_1 = vmulq_n_f32(vld1q_f32(src + x + 4), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));
            uint16x4_t v_dsti_1 = vqmovn_u32(cv_vrndq_u32_f32(v_dst_1));

            uint16x8_t v_dst = vcombine_u16(v_dsti_0, v_dsti_1);
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

#endif

template<typename T, typename DT, typename WT> static void
cvtScaleAbs_( const T* src, size_t sstep,
              DT* dst, size_t dstep, Size size,
              WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    cvtScaleAbs_SIMD<T, DT, WT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width, scale, shift);

        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(std::abs(src[x]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+1]*scale + shift));
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(std::abs(src[x+2]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+3]*scale + shift));
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(std::abs(src[x]*scale + shift));
    }
}

template <typename T, typename DT, typename WT>
struct cvtScale_SIMD
{
    int operator () (const T *, DT *, int, WT, WT) const
    {
        return 0;
    }
};

#if CV_SSE2

// from uchar

template <>
struct cvtScale_SIMD<uchar, uchar, float>
{
    int operator () (const uchar * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, schar, float>
{
    int operator () (const uchar * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if CV_SSE4_1

template <>
struct cvtScale_SIMD<uchar, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator () (const uchar * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                             _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <>
struct cvtScale_SIMD<uchar, short, float>
{
    int operator () (const uchar * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, int, float>
{
    int operator () (const uchar * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_si128((__m128i *)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i *)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, float, float>
{
    int operator () (const uchar * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_ps(dst + x, v_dst_0);
            _mm_storeu_ps(dst + x + 4, v_dst_1);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, double, double>
{
    int operator () (const uchar * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);

            __m128i v_src_s32 = _mm_unpacklo_epi16(v_src, v_zero);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);

            v_src_s32 = _mm_unpackhi_epi16(v_src, v_zero);
            v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x + 4, v_dst_0);
            _mm_storeu_pd(dst + x + 6, v_dst_1);
        }

        return x;
    }
};

// from schar

template <>
struct cvtScale_SIMD<schar, uchar, float>
{
    int operator () (const schar * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, schar, float>
{
    int operator () (const schar * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if CV_SSE4_1

template <>
struct cvtScale_SIMD<schar, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator () (const schar * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                             _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <>
struct cvtScale_SIMD<schar, short, float>
{
    int operator () (const schar * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, int, float>
{
    int operator () (const schar * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_si128((__m128i *)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i *)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, float, float>
{
    int operator () (const schar * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_ps(dst + x, v_dst_0);
            _mm_storeu_ps(dst + x + 4, v_dst_1);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, double, double>
{
    int operator () (const schar * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x)));
            v_src = _mm_srai_epi16(v_src, 8);

            __m128i v_src_s32 = _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);

            v_src_s32 = _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16);
            v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x + 4, v_dst_0);
            _mm_storeu_pd(dst + x + 6, v_dst_1);
        }

        return x;
    }
};

// from ushort

template <>
struct cvtScale_SIMD<ushort, uchar, float>
{
    int operator () (const ushort * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, schar, float>
{
    int operator () (const ushort * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if CV_SSE4_1

template <>
struct cvtScale_SIMD<ushort, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator () (const ushort * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                             _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <>
struct cvtScale_SIMD<ushort, short, float>
{
    int operator () (const ushort * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, int, float>
{
    int operator () (const ushort * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_si128((__m128i *)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i *)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, float, float>
{
    int operator () (const ushort * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_ps(dst + x, v_dst_0);
            _mm_storeu_ps(dst + x + 4, v_dst_1);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, double, double>
{
    int operator () (const ushort * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));

            __m128i v_src_s32 = _mm_unpacklo_epi16(v_src, v_zero);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);

            v_src_s32 = _mm_unpackhi_epi16(v_src, v_zero);
            v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x + 4, v_dst_0);
            _mm_storeu_pd(dst + x + 6, v_dst_1);
        }

        return x;
    }
};

// from short

template <>
struct cvtScale_SIMD<short, uchar, float>
{
    int operator () (const short * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<short, schar, float>
{
    int operator () (const short * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if CV_SSE4_1

template <>
struct cvtScale_SIMD<short, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator () (const short * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                             _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <>
struct cvtScale_SIMD<short, short, float>
{
    int operator () (const short * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<short, int, float>
{
    int operator () (const short * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_si128((__m128i *)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i *)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<short, float, float>
{
    int operator () (const short * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_ps(dst + x, v_dst_0);
            _mm_storeu_ps(dst + x + 4, v_dst_1);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<short, double, double>
{
    int operator () (const short * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));

            __m128i v_src_s32 = _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);

            v_src_s32 = _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16);
            v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x + 4, v_dst_0);
            _mm_storeu_pd(dst + x + 6, v_dst_1);
        }

        return x;
    }
};

// from int

template <>
struct cvtScale_SIMD<int, uchar, float>
{
    int operator () (const int * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            v_src = _mm_loadu_si128((__m128i const *)(src + x + 4));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<int, schar, float>
{
    int operator () (const int * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            v_src = _mm_loadu_si128((__m128i const *)(src + x + 4));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if CV_SSE4_1

template <>
struct cvtScale_SIMD<int, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator () (const int * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            v_src = _mm_loadu_si128((__m128i const *)(src + x + 4));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                             _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <>
struct cvtScale_SIMD<int, short, float>
{
    int operator () (const int * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            v_src = _mm_loadu_si128((__m128i const *)(src + x + 4));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<int, int, double>
{
    int operator () (const int * src, int * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 4; x += 4)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            v_src = _mm_srli_si128(v_src, 8);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            __m128 v_dst = _mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_dst_0)),
                                         _mm_castsi128_ps(_mm_cvtpd_epi32(v_dst_1)));

            _mm_storeu_si128((__m128i *)(dst + x), _mm_castps_si128(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<int, float, double>
{
    int operator () (const int * src, float * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 4; x += 4)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            v_src = _mm_srli_si128(v_src, 8);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            _mm_storeu_ps(dst + x, _mm_movelh_ps(_mm_cvtpd_ps(v_dst_0),
                                                 _mm_cvtpd_ps(v_dst_1)));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<int, double, double>
{
    int operator () (const int * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 4; x += 4)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            v_src = _mm_srli_si128(v_src, 8);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);
        }

        return x;
    }
};

// from float

template <>
struct cvtScale_SIMD<float, uchar, float>
{
    int operator () (const float * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, schar, float>
{
    int operator () (const float * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if CV_SSE4_1

template <>
struct cvtScale_SIMD<float, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator () (const float * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                             _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <>
struct cvtScale_SIMD<float, short, float>
{
    int operator () (const float * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, int, float>
{
    int operator () (const float * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            _mm_storeu_si128((__m128i *)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i *)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, float, float>
{
    int operator () (const float * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 4; x += 4)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);
            _mm_storeu_ps(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, double, double>
{
    int operator () (const float * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 4; x += 4)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtps_pd(v_src), v_scale), v_shift);
            v_src = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v_src), 8));
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtps_pd(v_src), v_scale), v_shift);

            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);
        }

        return x;
    }
};

// from double

template <>
struct cvtScale_SIMD<double, uchar, float>
{
    int operator () (const double * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)),
                                         _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)),
                                  _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<double, schar, float>
{
    int operator () (const double * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)),
                                         _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)),
                                  _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if CV_SSE4_1

template <>
struct cvtScale_SIMD<double, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator () (const double * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)),
                                         _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)),
                                  _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                             _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <>
struct cvtScale_SIMD<double, short, float>
{
    int operator () (const double * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)),
                                         _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)),
                                  _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<double, int, double>
{
    int operator () (const double * src, int * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 4; x += 4)
        {
            __m128d v_src = _mm_loadu_pd(src + x);
            __m128d v_dst0 = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);

            v_src = _mm_loadu_pd(src + x + 2);
            __m128d v_dst1 = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);

            __m128 v_dst = _mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_dst0)),
                                         _mm_castsi128_ps(_mm_cvtpd_epi32(v_dst1)));

            _mm_storeu_si128((__m128i *)(dst + x), _mm_castps_si128(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<double, float, double>
{
    int operator () (const double * src, float * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 4; x += 4)
        {
            __m128d v_src = _mm_loadu_pd(src + x);
            __m128d v_dst0 = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);

            v_src = _mm_loadu_pd(src + x + 2);
            __m128d v_dst1 = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);

            __m128 v_dst = _mm_movelh_ps(_mm_cvtpd_ps(v_dst0),
                                         _mm_cvtpd_ps(v_dst1));

            _mm_storeu_ps(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<double, double, double>
{
    int operator () (const double * src, double * dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for ( ; x <= width - 2; x += 2)
        {
            __m128d v_src = _mm_loadu_pd(src + x);
            __m128d v_dst = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst);
        }

        return x;
    }
};

#elif CV_NEON

// from uchar

template <>
struct cvtScale_SIMD<uchar, uchar, float>
{
    int operator () (const uchar * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, schar, float>
{
    int operator () (const uchar * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, ushort, float>
{
    int operator () (const uchar * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, short, float>
{
    int operator () (const uchar * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, int, float>
{
    int operator () (const uchar * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            vst1q_s32(dst + x, cv_vrndq_s32_f32(v_dst1));
            vst1q_s32(dst + x + 4, cv_vrndq_s32_f32(v_dst2));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<uchar, float, float>
{
    int operator () (const uchar * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift));
            vst1q_f32(dst + x + 4, vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift));
        }

        return x;
    }
};

// from schar

template <>
struct cvtScale_SIMD<schar, uchar, float>
{
    int operator () (const schar * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, schar, float>
{
    int operator () (const schar * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, ushort, float>
{
    int operator () (const schar * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, short, float>
{
    int operator () (const schar * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, int, float>
{
    int operator () (const schar * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            vst1q_s32(dst + x, cv_vrndq_s32_f32(v_dst1));
            vst1q_s32(dst + x + 4, cv_vrndq_s32_f32(v_dst2));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<schar, float, float>
{
    int operator () (const schar * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift));
            vst1q_f32(dst + x + 4, vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift));
        }

        return x;
    }
};

// from ushort

template <>
struct cvtScale_SIMD<ushort, uchar, float>
{
    int operator () (const ushort * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, schar, float>
{
    int operator () (const ushort * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, ushort, float>
{
    int operator () (const ushort * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, short, float>
{
    int operator () (const ushort * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, int, float>
{
    int operator () (const ushort * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            vst1q_s32(dst + x, cv_vrndq_s32_f32(v_dst1));
            vst1q_s32(dst + x + 4, cv_vrndq_s32_f32(v_dst2));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<ushort, float, float>
{
    int operator () (const ushort * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift));
            vst1q_f32(dst + x + 4, vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift));
        }

        return x;
    }
};

// from short

template <>
struct cvtScale_SIMD<short, uchar, float>
{
    int operator () (const short * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<short, schar, float>
{
    int operator () (const short * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<short, ushort, float>
{
    int operator () (const short * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<short, float, float>
{
    int operator () (const short * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift));
            vst1q_f32(dst + x + 4, vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift));
        }

        return x;
    }
};

// from int

template <>
struct cvtScale_SIMD<int, uchar, float>
{
    int operator () (const int * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x)), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<int, schar, float>
{
    int operator () (const int * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x)), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<int, ushort, float>
{
    int operator () (const int * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x)), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<int, short, float>
{
    int operator () (const int * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x)), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

// from float

template <>
struct cvtScale_SIMD<float, uchar, float>
{
    int operator () (const float * src, uchar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vld1q_f32(src + x + 4), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, schar, float>
{
    int operator () (const float * src, schar * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vld1q_f32(src + x + 4), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, ushort, float>
{
    int operator () (const float * src, ushort * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vld1q_f32(src + x + 4), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                            vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, short, float>
{
    int operator () (const float * src, short * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vld1q_f32(src + x + 4), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                            vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, int, float>
{
    int operator () (const float * src, int * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 4; x += 4)
            vst1q_s32(dst + x, cv_vrndq_s32_f32(vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift)));

        return x;
    }
};

template <>
struct cvtScale_SIMD<float, float, float>
{
    int operator () (const float * src, float * dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for ( ; x <= width - 4; x += 4)
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift));

        return x;
    }
};

#endif

template<typename T, typename DT, typename WT> static void
cvtScale_( const T* src, size_t sstep,
           DT* dst, size_t dstep, Size size,
           WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    cvtScale_SIMD<T, DT, WT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width, scale, shift);

        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]*scale + shift);
            t1 = saturate_cast<DT>(src[x+1]*scale + shift);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]*scale + shift);
            t1 = saturate_cast<DT>(src[x+3]*scale + shift);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif

        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]*scale + shift);
    }
}

//vz optimized template specialization
template<> void
cvtScale_<short, short, float>( const short* src, size_t sstep,
           short* dst, size_t dstep, Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if CV_SSE2
            if(USE_SSE2)
            {
                __m128 scale128 = _mm_set1_ps (scale);
                __m128 shift128 = _mm_set1_ps (shift);
                for(; x <= size.width - 8; x += 8 )
                {
                    __m128i r0 = _mm_loadl_epi64((const __m128i*)(src + x));
                    __m128i r1 = _mm_loadl_epi64((const __m128i*)(src + x + 4));
                    __m128 rf0 =_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
                    __m128 rf1 =_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));
                    rf0 = _mm_add_ps(_mm_mul_ps(rf0, scale128), shift128);
                    rf1 = _mm_add_ps(_mm_mul_ps(rf1, scale128), shift128);
                    r0 = _mm_cvtps_epi32(rf0);
                    r1 = _mm_cvtps_epi32(rf1);
                    r0 = _mm_packs_epi32(r0, r1);
                    _mm_storeu_si128((__m128i*)(dst + x), r0);
                }
            }
        #elif CV_NEON
        float32x4_t v_shift = vdupq_n_f32(shift);
        for(; x <= size.width - 8; x += 8 )
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_tmp1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src)));
            float32x4_t v_tmp2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src)));

            v_tmp1 = vaddq_f32(vmulq_n_f32(v_tmp1, scale), v_shift);
            v_tmp2 = vaddq_f32(vmulq_n_f32(v_tmp2, scale), v_shift);

            vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_tmp1)),
                                            vqmovn_s32(cv_vrndq_s32_f32(v_tmp2))));
        }
        #endif

        for(; x < size.width; x++ )
            dst[x] = saturate_cast<short>(src[x]*scale + shift);
    }
}

template<> void
cvtScale_<short, int, float>( const short* src, size_t sstep,
           int* dst, size_t dstep, Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;

        #if CV_AVX2
        if (USE_AVX2)
        {
            __m256 scale256 = _mm256_set1_ps(scale);
            __m256 shift256 = _mm256_set1_ps(shift);
            const int shuffle = 0xD8;

            for ( ; x <= size.width - 16; x += 16)
            {
                __m256i v_src = _mm256_loadu_si256((const __m256i *)(src + x));
                v_src = _mm256_permute4x64_epi64(v_src, shuffle);
                __m256i v_src_lo = _mm256_srai_epi32(_mm256_unpacklo_epi16(v_src, v_src), 16);
                __m256i v_src_hi = _mm256_srai_epi32(_mm256_unpackhi_epi16(v_src, v_src), 16);
                __m256 v_dst0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v_src_lo), scale256), shift256);
                __m256 v_dst1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v_src_hi), scale256), shift256);
                _mm256_storeu_si256((__m256i *)(dst + x), _mm256_cvtps_epi32(v_dst0));
                _mm256_storeu_si256((__m256i *)(dst + x + 8), _mm256_cvtps_epi32(v_dst1));
            }
        }
        #endif
        #if CV_SSE2
        if (USE_SSE2)//~5X
        {
            __m128 scale128 = _mm_set1_ps (scale);
            __m128 shift128 = _mm_set1_ps (shift);
            for(; x <= size.width - 8; x += 8 )
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)(src + x));

                __m128 rf0 =_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
                __m128 rf1 =_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(r0, r0), 16));
                rf0 = _mm_add_ps(_mm_mul_ps(rf0, scale128), shift128);
                rf1 = _mm_add_ps(_mm_mul_ps(rf1, scale128), shift128);

                _mm_storeu_si128((__m128i*)(dst + x), _mm_cvtps_epi32(rf0));
                _mm_storeu_si128((__m128i*)(dst + x + 4), _mm_cvtps_epi32(rf1));
            }
        }
        #elif CV_NEON
        float32x4_t v_shift = vdupq_n_f32(shift);
        for(; x <= size.width - 8; x += 8 )
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_tmp1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src)));
            float32x4_t v_tmp2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src)));

            v_tmp1 = vaddq_f32(vmulq_n_f32(v_tmp1, scale), v_shift);
            v_tmp2 = vaddq_f32(vmulq_n_f32(v_tmp2, scale), v_shift);

            vst1q_s32(dst + x, cv_vrndq_s32_f32(v_tmp1));
            vst1q_s32(dst + x + 4, cv_vrndq_s32_f32(v_tmp2));
        }
        #endif

        for(; x < size.width; x++ )
            dst[x] = saturate_cast<int>(src[x]*scale + shift);
    }
}

template <typename T, typename DT>
struct Cvt_SIMD
{
    int operator() (const T *, DT *, int) const
    {
        return 0;
    }
};

#if CV_SSE2

// from double

template <>
struct Cvt_SIMD<double, uchar>
{
    int operator() (const double * src, uchar * dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
            __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

            v_src0 = _mm_movelh_ps(v_src0, v_src1);
            v_src1 = _mm_movelh_ps(v_src2, v_src3);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_src0),
                                            _mm_cvtps_epi32(v_src1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packus_epi16(v_dst, v_dst));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<double, schar>
{
    int operator() (const double * src, schar * dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
            __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

            v_src0 = _mm_movelh_ps(v_src0, v_src1);
            v_src1 = _mm_movelh_ps(v_src2, v_src3);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_src0),
                                            _mm_cvtps_epi32(v_src1));
            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst, v_dst));
        }

        return x;
    }
};

#if CV_SSE4_1

template <>
struct Cvt_SIMD<double, ushort>
{
    bool haveSIMD;
    Cvt_SIMD() { haveSIMD = checkHardwareSupport(CV_CPU_SSE4_1); }

    int operator() (const double * src, ushort * dst, int width) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
            __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

            v_src0 = _mm_movelh_ps(v_src0, v_src1);
            v_src1 = _mm_movelh_ps(v_src2, v_src3);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_src0),
                                             _mm_cvtps_epi32(v_src1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

#endif // CV_SSE4_1

template <>
struct Cvt_SIMD<double, short>
{
    int operator() (const double * src, short * dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for ( ; x <= width - 8; x += 8)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
            __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

            v_src0 = _mm_movelh_ps(v_src0, v_src1);
            v_src1 = _mm_movelh_ps(v_src2, v_src3);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_src0),
                                            _mm_cvtps_epi32(v_src1));
            _mm_storeu_si128((__m128i *)(dst + x), v_dst);
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<double, int>
{
    int operator() (const double * src, int * dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for ( ; x <= width - 4; x += 4)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            v_src0 = _mm_movelh_ps(v_src0, v_src1);

            _mm_storeu_si128((__m128i *)(dst + x), _mm_cvtps_epi32(v_src0));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<double, float>
{
    int operator() (const double * src, float * dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for ( ; x <= width - 4; x += 4)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));

            _mm_storeu_ps(dst + x, _mm_movelh_ps(v_src0, v_src1));
        }

        return x;
    }
};


#elif CV_NEON

// from uchar

template <>
struct Cvt_SIMD<uchar, schar>
{
    int operator() (const uchar * src, schar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
            vst1_s8(dst + x, vqmovn_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + x)))));

        return x;
    }
};


template <>
struct Cvt_SIMD<uchar, ushort>
{
    int operator() (const uchar * src, ushort * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
            vst1q_u16(dst + x, vmovl_u8(vld1_u8(src + x)));

        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, short>
{
    int operator() (const uchar * src, short * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
            vst1q_s16(dst + x, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + x))));

        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, int>
{
    int operator() (const uchar * src, int * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            vst1q_s32(dst + x, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src))));
            vst1q_s32(dst + x + 4, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src))));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<uchar, float>
{
    int operator() (const uchar * src, float * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            vst1q_f32(dst + x, vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))));
            vst1q_f32(dst + x + 4, vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))));
        }

        return x;
    }
};

// from schar

template <>
struct Cvt_SIMD<schar, uchar>
{
    int operator() (const schar * src, uchar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
            vst1_u8(dst + x, vqmovun_s16(vmovl_s8(vld1_s8(src + x))));

        return x;
    }
};

template <>
struct Cvt_SIMD<schar, short>
{
    int operator() (const schar * src, short * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
            vst1q_s16(dst + x, vmovl_s8(vld1_s8(src + x)));

        return x;
    }
};

template <>
struct Cvt_SIMD<schar, ushort>
{
    int operator() (const schar * src, ushort * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            vst1q_u16(dst + x, vcombine_u16(vqmovun_s32(vmovl_s16(vget_low_s16(v_src))),
                                            vqmovun_s32(vmovl_s16(vget_high_s16(v_src)))));
        }

        return x;
    }
};


template <>
struct Cvt_SIMD<schar, int>
{
    int operator() (const schar * src, int * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            vst1q_s32(dst + x, vmovl_s16(vget_low_s16(v_src)));
            vst1q_s32(dst + x + 4, vmovl_s16(vget_high_s16(v_src)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<schar, float>
{
    int operator() (const schar * src, float * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            vst1q_f32(dst + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))));
            vst1q_f32(dst + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))));
        }

        return x;
    }
};

// from ushort

template <>
struct Cvt_SIMD<ushort, uchar>
{
    int operator() (const ushort * src, uchar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 16; x += 16)
        {
            uint16x8_t v_src1 = vld1q_u16(src + x), v_src2 = vld1q_u16(src + x + 8);
            vst1q_u8(dst + x, vcombine_u8(vqmovn_u16(v_src1), vqmovn_u16(v_src2)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, schar>
{
    int operator() (const ushort * src, schar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 16; x += 16)
        {
            uint16x8_t v_src1 = vld1q_u16(src + x), v_src2 = vld1q_u16(src + x + 8);
            int32x4_t v_dst10 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src1)));
            int32x4_t v_dst11 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src1)));
            int32x4_t v_dst20 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src2)));
            int32x4_t v_dst21 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src2)));

            vst1q_s8(dst + x, vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(v_dst10), vqmovn_s32(v_dst11))),
                                          vqmovn_s16(vcombine_s16(vqmovn_s32(v_dst20), vqmovn_s32(v_dst21)))));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, short>
{
    int operator() (const ushort * src, short * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            int32x4_t v_dst0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src)));
            int32x4_t v_dst1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src)));

            vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(v_dst0), vqmovn_s32(v_dst1)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, int>
{
    int operator() (const ushort * src, int * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            vst1q_s32(dst + x, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src))));
            vst1q_s32(dst + x + 4, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src))));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<ushort, float>
{
    int operator() (const ushort * src, float * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            vst1q_f32(dst + x, vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))));
            vst1q_f32(dst + x + 4, vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))));
        }

        return x;
    }
};

// from short

template <>
struct Cvt_SIMD<short, uchar>
{
    int operator() (const short * src, uchar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 16; x += 16)
        {
            int16x8_t v_src1 = vld1q_s16(src + x), v_src2 = vld1q_s16(src + x + 8);
            vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(v_src1), vqmovun_s16(v_src2)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<short, schar>
{
    int operator() (const short * src, schar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 16; x += 16)
        {
            int16x8_t v_src1 = vld1q_s16(src + x), v_src2 = vld1q_s16(src + x + 8);
            vst1q_s8(dst + x, vcombine_s8(vqmovn_s16(v_src1), vqmovn_s16(v_src2)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<short, ushort>
{
    int operator() (const short * src, ushort * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            uint16x4_t v_dst1 = vqmovun_s32(vmovl_s16(vget_low_s16(v_src)));
            uint16x4_t v_dst2 = vqmovun_s32(vmovl_s16(vget_high_s16(v_src)));
            vst1q_u16(dst + x, vcombine_u16(v_dst1, v_dst2));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<short, int>
{
    int operator() (const short * src, int * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            vst1q_s32(dst + x, vmovl_s16(vget_low_s16(v_src)));
            vst1q_s32(dst + x + 4, vmovl_s16(vget_high_s16(v_src)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<short, float>
{
    int operator() (const short * src, float * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            vst1q_f32(dst + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))));
            vst1q_f32(dst + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))));
        }

        return x;
    }
};

// from int

template <>
struct Cvt_SIMD<int, uchar>
{
    int operator() (const int * src, uchar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 16; x += 16)
        {
            int32x4_t v_src1 = vld1q_s32(src + x), v_src2 = vld1q_s32(src + x + 4);
            int32x4_t v_src3 = vld1q_s32(src + x + 8), v_src4 = vld1q_s32(src + x + 12);
            uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovun_s32(v_src1), vqmovun_s32(v_src2)));
            uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovun_s32(v_src3), vqmovun_s32(v_src4)));
            vst1q_u8(dst + x, vcombine_u8(v_dst1, v_dst2));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<int, schar>
{
    int operator() (const int * src, schar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 16; x += 16)
        {
            int32x4_t v_src1 = vld1q_s32(src + x), v_src2 = vld1q_s32(src + x + 4);
            int32x4_t v_src3 = vld1q_s32(src + x + 8), v_src4 = vld1q_s32(src + x + 12);
            int8x8_t v_dst1 = vqmovn_s16(vcombine_s16(vqmovn_s32(v_src1), vqmovn_s32(v_src2)));
            int8x8_t v_dst2 = vqmovn_s16(vcombine_s16(vqmovn_s32(v_src3), vqmovn_s32(v_src4)));
            vst1q_s8(dst + x, vcombine_s8(v_dst1, v_dst2));
        }

        return x;
    }
};


template <>
struct Cvt_SIMD<int, ushort>
{
    int operator() (const int * src, ushort * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            int32x4_t v_src1 = vld1q_s32(src + x), v_src2 = vld1q_s32(src + x + 4);
            vst1q_u16(dst + x, vcombine_u16(vqmovun_s32(v_src1), vqmovun_s32(v_src2)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<int, short>
{
    int operator() (const int * src, short * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            int32x4_t v_src1 = vld1q_s32(src + x), v_src2 = vld1q_s32(src + x + 4);
            vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(v_src1), vqmovn_s32(v_src2)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<int, float>
{
    int operator() (const int * src, float * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 4; x += 4)
            vst1q_f32(dst + x, vcvtq_f32_s32(vld1q_s32(src + x)));

        return x;
    }
};

// from float

template <>
struct Cvt_SIMD<float, uchar>
{
    int operator() (const float * src, uchar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 16; x += 16)
        {
            uint32x4_t v_src1 = cv_vrndq_u32_f32(vld1q_f32(src + x));
            uint32x4_t v_src2 = cv_vrndq_u32_f32(vld1q_f32(src + x + 4));
            uint32x4_t v_src3 = cv_vrndq_u32_f32(vld1q_f32(src + x + 8));
            uint32x4_t v_src4 = cv_vrndq_u32_f32(vld1q_f32(src + x + 12));
            uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(v_src1), vqmovn_u32(v_src2)));
            uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(v_src3), vqmovn_u32(v_src4)));
            vst1q_u8(dst + x, vcombine_u8(v_dst1, v_dst2));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<float, schar>
{
    int operator() (const float * src, schar * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 16; x += 16)
        {
            int32x4_t v_src1 = cv_vrndq_s32_f32(vld1q_f32(src + x));
            int32x4_t v_src2 = cv_vrndq_s32_f32(vld1q_f32(src + x + 4));
            int32x4_t v_src3 = cv_vrndq_s32_f32(vld1q_f32(src + x + 8));
            int32x4_t v_src4 = cv_vrndq_s32_f32(vld1q_f32(src + x + 12));
            int8x8_t v_dst1 = vqmovn_s16(vcombine_s16(vqmovn_s32(v_src1), vqmovn_s32(v_src2)));
            int8x8_t v_dst2 = vqmovn_s16(vcombine_s16(vqmovn_s32(v_src3), vqmovn_s32(v_src4)));
            vst1q_s8(dst + x, vcombine_s8(v_dst1, v_dst2));
        }

        return x;
    }
};


template <>
struct Cvt_SIMD<float, ushort>
{
    int operator() (const float * src, ushort * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 8; x += 8)
        {
            uint32x4_t v_src1 = cv_vrndq_u32_f32(vld1q_f32(src + x));
            uint32x4_t v_src2 = cv_vrndq_u32_f32(vld1q_f32(src + x + 4));
            vst1q_u16(dst + x, vcombine_u16(vqmovn_u32(v_src1), vqmovn_u32(v_src2)));
        }

        return x;
    }
};

template <>
struct Cvt_SIMD<float, int>
{
    int operator() (const float * src, int * dst, int width) const
    {
        int x = 0;

        for ( ; x <= width - 4; x += 4)
            vst1q_s32(dst + x, cv_vrndq_s32_f32(vld1q_f32(src + x)));

        return x;
    }
};

#endif

template<typename T, typename DT> static void
cvt_( const T* src, size_t sstep,
      DT* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    Cvt_SIMD<T, DT> vop;

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = vop(src, dst, size.width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]);
            t1 = saturate_cast<DT>(src[x+1]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]);
            t1 = saturate_cast<DT>(src[x+3]);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]);
    }
}

//vz optimized template specialization, test Core_ConvertScale/ElemWiseTest
template<>  void
cvt_<float, short>( const float* src, size_t sstep,
     short* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if   CV_SSE2
        if(USE_SSE2)
        {
            for( ; x <= size.width - 8; x += 8 )
            {
                __m128 src128 = _mm_loadu_ps (src + x);
                __m128i src_int128 = _mm_cvtps_epi32 (src128);

                src128 = _mm_loadu_ps (src + x + 4);
                __m128i src1_int128 = _mm_cvtps_epi32 (src128);

                src1_int128 = _mm_packs_epi32(src_int128, src1_int128);
                _mm_storeu_si128((__m128i*)(dst + x),src1_int128);
            }
        }
        #elif CV_NEON
        for( ; x <= size.width - 8; x += 8 )
        {
            float32x4_t v_src1 = vld1q_f32(src + x), v_src2 = vld1q_f32(src + x + 4);
            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_src1)),
                                           vqmovn_s32(cv_vrndq_s32_f32(v_src2)));
            vst1q_s16(dst + x, v_dst);
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<short>(src[x]);
    }

}


template<typename T> static void
cpy_( const T* src, size_t sstep, T* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
        memcpy(dst, src, size.width*sizeof(src[0]));
}

#define DEF_CVT_SCALE_ABS_FUNC(suffix, tfunc, stype, dtype, wtype) \
static void cvtScaleAbs##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    tfunc(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

#define DEF_CVT_SCALE_FUNC(suffix, stype, dtype, wtype) \
static void cvtScale##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    cvtScale_(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

#if defined(HAVE_IPP)
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    CV_IPP_CHECK()\
    {\
        if (src && dst)\
        {\
            if (ippiConvert_##ippFavor(src, (int)sstep, dst, (int)dstep, ippiSize(size.width, size.height)) >= 0) \
            {\
                CV_IMPL_ADD(CV_IMPL_IPP)\
                return; \
            }\
            setIppErrorStatus(); \
        }\
    }\
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CVT_FUNC_F2(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    CV_IPP_CHECK()\
    {\
        if (src && dst)\
        {\
            if (ippiConvert_##ippFavor(src, (int)sstep, dst, (int)dstep, ippiSize(size.width, size.height), ippRndFinancial, 0) >= 0) \
            {\
                CV_IMPL_ADD(CV_IMPL_IPP)\
                return; \
            }\
            setIppErrorStatus(); \
        }\
    }\
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


DEF_CVT_SCALE_ABS_FUNC(8u, cvtScaleAbs_, uchar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(8s8u, cvtScaleAbs_, schar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtScaleAbs_, ushort, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtScaleAbs_, short, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtScaleAbs_, int, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtScaleAbs_, float, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtScaleAbs_, double, uchar, float)

DEF_CVT_SCALE_FUNC(8u,     uchar, uchar, float)
DEF_CVT_SCALE_FUNC(8s8u,   schar, uchar, float)
DEF_CVT_SCALE_FUNC(16u8u,  ushort, uchar, float)
DEF_CVT_SCALE_FUNC(16s8u,  short, uchar, float)
DEF_CVT_SCALE_FUNC(32s8u,  int, uchar, float)
DEF_CVT_SCALE_FUNC(32f8u,  float, uchar, float)
DEF_CVT_SCALE_FUNC(64f8u,  double, uchar, float)

DEF_CVT_SCALE_FUNC(8u8s,   uchar, schar, float)
DEF_CVT_SCALE_FUNC(8s,     schar, schar, float)
DEF_CVT_SCALE_FUNC(16u8s,  ushort, schar, float)
DEF_CVT_SCALE_FUNC(16s8s,  short, schar, float)
DEF_CVT_SCALE_FUNC(32s8s,  int, schar, float)
DEF_CVT_SCALE_FUNC(32f8s,  float, schar, float)
DEF_CVT_SCALE_FUNC(64f8s,  double, schar, float)

DEF_CVT_SCALE_FUNC(8u16u,  uchar, ushort, float)
DEF_CVT_SCALE_FUNC(8s16u,  schar, ushort, float)
DEF_CVT_SCALE_FUNC(16u,    ushort, ushort, float)
DEF_CVT_SCALE_FUNC(16s16u, short, ushort, float)
DEF_CVT_SCALE_FUNC(32s16u, int, ushort, float)
DEF_CVT_SCALE_FUNC(32f16u, float, ushort, float)
DEF_CVT_SCALE_FUNC(64f16u, double, ushort, float)

DEF_CVT_SCALE_FUNC(8u16s,  uchar, short, float)
DEF_CVT_SCALE_FUNC(8s16s,  schar, short, float)
DEF_CVT_SCALE_FUNC(16u16s, ushort, short, float)
DEF_CVT_SCALE_FUNC(16s,    short, short, float)
DEF_CVT_SCALE_FUNC(32s16s, int, short, float)
DEF_CVT_SCALE_FUNC(32f16s, float, short, float)
DEF_CVT_SCALE_FUNC(64f16s, double, short, float)

DEF_CVT_SCALE_FUNC(8u32s,  uchar, int, float)
DEF_CVT_SCALE_FUNC(8s32s,  schar, int, float)
DEF_CVT_SCALE_FUNC(16u32s, ushort, int, float)
DEF_CVT_SCALE_FUNC(16s32s, short, int, float)
DEF_CVT_SCALE_FUNC(32s,    int, int, double)
DEF_CVT_SCALE_FUNC(32f32s, float, int, float)
DEF_CVT_SCALE_FUNC(64f32s, double, int, double)

DEF_CVT_SCALE_FUNC(8u32f,  uchar, float, float)
DEF_CVT_SCALE_FUNC(8s32f,  schar, float, float)
DEF_CVT_SCALE_FUNC(16u32f, ushort, float, float)
DEF_CVT_SCALE_FUNC(16s32f, short, float, float)
DEF_CVT_SCALE_FUNC(32s32f, int, float, double)
DEF_CVT_SCALE_FUNC(32f,    float, float, float)
DEF_CVT_SCALE_FUNC(64f32f, double, float, double)

DEF_CVT_SCALE_FUNC(8u64f,  uchar, double, double)
DEF_CVT_SCALE_FUNC(8s64f,  schar, double, double)
DEF_CVT_SCALE_FUNC(16u64f, ushort, double, double)
DEF_CVT_SCALE_FUNC(16s64f, short, double, double)
DEF_CVT_SCALE_FUNC(32s64f, int, double, double)
DEF_CVT_SCALE_FUNC(32f64f, float, double, double)
DEF_CVT_SCALE_FUNC(64f,    double, double, double)

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

static BinaryFunc getCvtScaleAbsFunc(int depth)
{
    static BinaryFunc cvtScaleAbsTab[] =
    {
        (BinaryFunc)cvtScaleAbs8u, (BinaryFunc)cvtScaleAbs8s8u, (BinaryFunc)cvtScaleAbs16u8u,
        (BinaryFunc)cvtScaleAbs16s8u, (BinaryFunc)cvtScaleAbs32s8u, (BinaryFunc)cvtScaleAbs32f8u,
        (BinaryFunc)cvtScaleAbs64f8u, 0
    };

    return cvtScaleAbsTab[depth];
}

BinaryFunc getConvertFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtTab[][8] =
    {
        {
            (BinaryFunc)(cvt8u), (BinaryFunc)GET_OPTIMIZED(cvt8s8u), (BinaryFunc)GET_OPTIMIZED(cvt16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8u), (BinaryFunc)GET_OPTIMIZED(cvt32s8u), (BinaryFunc)GET_OPTIMIZED(cvt32f8u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u8s), (BinaryFunc)cvt8u, (BinaryFunc)GET_OPTIMIZED(cvt16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8s), (BinaryFunc)GET_OPTIMIZED(cvt32s8s), (BinaryFunc)GET_OPTIMIZED(cvt32f8s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16u), (BinaryFunc)GET_OPTIMIZED(cvt8s16u), (BinaryFunc)cvt16u,
            (BinaryFunc)GET_OPTIMIZED(cvt16s16u), (BinaryFunc)GET_OPTIMIZED(cvt32s16u), (BinaryFunc)GET_OPTIMIZED(cvt32f16u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16s), (BinaryFunc)GET_OPTIMIZED(cvt8s16s), (BinaryFunc)GET_OPTIMIZED(cvt16u16s),
            (BinaryFunc)cvt16u, (BinaryFunc)GET_OPTIMIZED(cvt32s16s), (BinaryFunc)GET_OPTIMIZED(cvt32f16s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32s), (BinaryFunc)GET_OPTIMIZED(cvt8s32s), (BinaryFunc)GET_OPTIMIZED(cvt16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32s), (BinaryFunc)cvt32s, (BinaryFunc)GET_OPTIMIZED(cvt32f32s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f32s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32f), (BinaryFunc)GET_OPTIMIZED(cvt8s32f), (BinaryFunc)GET_OPTIMIZED(cvt16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32f), (BinaryFunc)GET_OPTIMIZED(cvt32s32f), (BinaryFunc)cvt32s,
            (BinaryFunc)GET_OPTIMIZED(cvt64f32f), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u64f), (BinaryFunc)GET_OPTIMIZED(cvt8s64f), (BinaryFunc)GET_OPTIMIZED(cvt16u64f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s64f), (BinaryFunc)GET_OPTIMIZED(cvt32s64f), (BinaryFunc)GET_OPTIMIZED(cvt32f64f),
            (BinaryFunc)(cvt64s), 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

static BinaryFunc getConvertScaleFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtScaleTab[][8] =
    {
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8u),
            (BinaryFunc)cvtScale64f8u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u8s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8s),
            (BinaryFunc)cvtScale64f8s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16u),
            (BinaryFunc)cvtScale64f16u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u16s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16s),
            (BinaryFunc)cvtScale64f16s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f32s),
            (BinaryFunc)cvtScale64f32s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32f), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32f),
            (BinaryFunc)cvtScale64f32f, 0
        },
        {
            (BinaryFunc)cvtScale8u64f, (BinaryFunc)cvtScale8s64f, (BinaryFunc)cvtScale16u64f,
            (BinaryFunc)cvtScale16s64f, (BinaryFunc)cvtScale32s64f, (BinaryFunc)cvtScale32f64f,
            (BinaryFunc)cvtScale64f, 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtScaleTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
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

    size_t globalsize[2] = { src.cols * cn / kercn, (src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

void cv::convertScaleAbs( InputArray _src, OutputArray _dst, double alpha, double beta )
{
    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_convertScaleAbs(_src, _dst, alpha, beta))

    Mat src = _src.getMat();
    int cn = src.channels();
    double scale[] = {alpha, beta};
    _dst.create( src.dims, src.size, CV_8UC(cn) );
    Mat dst = _dst.getMat();
    BinaryFunc func = getCvtScaleAbsFunc(src.depth());
    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.ptr(), src.step, 0, 0, dst.ptr(), dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)it.size*cn, 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], 0, 0, 0, ptrs[1], 0, sz, scale );
    }
}

void cv::Mat::convertTo(OutputArray _dst, int _type, double alpha, double beta) const
{
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

    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);
    double scale[] = {alpha, beta};
    int cn = channels();
    CV_Assert( func != 0 );

    if( dims <= 2 )
    {
        _dst.create( size(), _type );
        Mat dst = _dst.getMat();
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, scale );
    }
    else
    {
        _dst.create( dims, size, _type );
        Mat dst = _dst.getMat();
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, scale);
    }
}

/****************************************************************************************\
*                                    LUT Transform                                       *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
LUT8u_( const uchar* src, const T* lut, T* dst, int len, int cn, int lutcn )
{
    if( lutcn == 1 )
    {
        for( int i = 0; i < len*cn; i++ )
            dst[i] = lut[src[i]];
    }
    else
    {
        for( int i = 0; i < len*cn; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn+k];
    }
}

static void LUT8u_8u( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_8s( const uchar* src, const schar* lut, schar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16u( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16s( const uchar* src, const short* lut, short* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32s( const uchar* src, const int* lut, int* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32f( const uchar* src, const float* lut, float* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_64f( const uchar* src, const double* lut, double* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

typedef void (*LUTFunc)( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn );

static LUTFunc lutTab[] =
{
    (LUTFunc)LUT8u_8u, (LUTFunc)LUT8u_8s, (LUTFunc)LUT8u_16u, (LUTFunc)LUT8u_16s,
    (LUTFunc)LUT8u_32s, (LUTFunc)LUT8u_32f, (LUTFunc)LUT8u_64f, 0
};

#ifdef HAVE_OPENCL

static bool ocl_LUT(InputArray _src, InputArray _lut, OutputArray _dst)
{
    int lcn = _lut.channels(), dcn = _src.channels(), ddepth = _lut.depth();

    UMat src = _src.getUMat(), lut = _lut.getUMat();
    _dst.create(src.size(), CV_MAKETYPE(ddepth, dcn));
    UMat dst = _dst.getUMat();
    int kercn = lcn == 1 ? std::min(4, ocl::predictOptimalVectorWidth(_src, _dst)) : dcn;

    ocl::Kernel k("LUT", ocl::core::lut_oclsrc,
                  format("-D dcn=%d -D lcn=%d -D srcT=%s -D dstT=%s", kercn, lcn,
                         ocl::typeToStr(src.depth()), ocl::memopTypeToStr(ddepth)));
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::ReadOnlyNoSize(lut),
        ocl::KernelArg::WriteOnly(dst, dcn, kercn));

    size_t globalSize[2] = { dst.cols * dcn / kercn, (dst.rows + 3) / 4 };
    return k.run(2, globalSize, NULL, false);
}

#endif

#if defined(HAVE_IPP)
namespace ipp {

#if 0 // there are no performance benefits (PR #2653)
class IppLUTParallelBody_LUTC1 : public ParallelLoopBody
{
public:
    bool* ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    typedef IppStatus (*IppFn)(const Ipp8u* pSrc, int srcStep, void* pDst, int dstStep,
                          IppiSize roiSize, const void* pTable, int nBitSize);
    IppFn fn;

    int width;

    IppLUTParallelBody_LUTC1(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst)
    {
        width = dst.cols * dst.channels();

        size_t elemSize1 = CV_ELEM_SIZE1(dst.depth());

        fn =
                elemSize1 == 1 ? (IppFn)ippiLUTPalette_8u_C1R :
                elemSize1 == 4 ? (IppFn)ippiLUTPalette_8u32u_C1R :
                NULL;

        *ok = (fn != NULL);
    }

    void operator()( const cv::Range& range ) const
    {
        if (!*ok)
            return;

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        IppiSize sz = { width, dst.rows };

        CV_DbgAssert(fn != NULL);
        if (fn(src.data, (int)src.step[0], dst.data, (int)dst.step[0], sz, lut_.data, 8) < 0)
        {
            setIppErrorStatus();
            *ok = false;
        }
        CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
    }
private:
    IppLUTParallelBody_LUTC1(const IppLUTParallelBody_LUTC1&);
    IppLUTParallelBody_LUTC1& operator=(const IppLUTParallelBody_LUTC1&);
};
#endif

class IppLUTParallelBody_LUTCN : public ParallelLoopBody
{
public:
    bool *ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    int lutcn;

    uchar* lutBuffer;
    uchar* lutTable[4];

    IppLUTParallelBody_LUTCN(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst), lutBuffer(NULL)
    {
        lutcn = lut.channels();
        IppiSize sz256 = {256, 1};

        size_t elemSize1 = dst.elemSize1();
        CV_DbgAssert(elemSize1 == 1);
        lutBuffer = (uchar*)ippMalloc(256 * (int)elemSize1 * 4);
        lutTable[0] = lutBuffer + 0;
        lutTable[1] = lutBuffer + 1 * 256 * elemSize1;
        lutTable[2] = lutBuffer + 2 * 256 * elemSize1;
        lutTable[3] = lutBuffer + 3 * 256 * elemSize1;

        CV_DbgAssert(lutcn == 3 || lutcn == 4);
        if (lutcn == 3)
        {
            IppStatus status = ippiCopy_8u_C3P3R(lut.ptr(), (int)lut.step[0], lutTable, (int)lut.step[0], sz256);
            if (status < 0)
            {
                setIppErrorStatus();
                return;
            }
            CV_IMPL_ADD(CV_IMPL_IPP);
        }
        else if (lutcn == 4)
        {
            IppStatus status = ippiCopy_8u_C4P4R(lut.ptr(), (int)lut.step[0], lutTable, (int)lut.step[0], sz256);
            if (status < 0)
            {
                setIppErrorStatus();
                return;
            }
            CV_IMPL_ADD(CV_IMPL_IPP);
        }

        *ok = true;
    }

    ~IppLUTParallelBody_LUTCN()
    {
        if (lutBuffer != NULL)
            ippFree(lutBuffer);
        lutBuffer = NULL;
        lutTable[0] = NULL;
    }

    void operator()( const cv::Range& range ) const
    {
        if (!*ok)
            return;

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        if (lutcn == 3)
        {
            if (ippiLUTPalette_8u_C3R(
                    src.ptr(), (int)src.step[0], dst.ptr(), (int)dst.step[0],
                    ippiSize(dst.size()), lutTable, 8) >= 0)
            {
                CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
                return;
            }
        }
        else if (lutcn == 4)
        {
            if (ippiLUTPalette_8u_C4R(
                    src.ptr(), (int)src.step[0], dst.ptr(), (int)dst.step[0],
                    ippiSize(dst.size()), lutTable, 8) >= 0)
            {
                CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
                return;
            }
        }
        setIppErrorStatus();
        *ok = false;
    }
private:
    IppLUTParallelBody_LUTCN(const IppLUTParallelBody_LUTCN&);
    IppLUTParallelBody_LUTCN& operator=(const IppLUTParallelBody_LUTCN&);
};
} // namespace ipp
#endif // IPP

class LUTParallelBody : public ParallelLoopBody
{
public:
    bool* ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    LUTFunc func;

    LUTParallelBody(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst)
    {
        func = lutTab[lut.depth()];
        *ok = (func != NULL);
    }

    void operator()( const cv::Range& range ) const
    {
        CV_DbgAssert(*ok);

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        int cn = src.channels();
        int lutcn = lut_.channels();

        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        int len = (int)it.size;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], lut_.ptr(), ptrs[1], len, cn, lutcn);
    }
private:
    LUTParallelBody(const LUTParallelBody&);
    LUTParallelBody& operator=(const LUTParallelBody&);
};

}

void cv::LUT( InputArray _src, InputArray _lut, OutputArray _dst )
{
    int cn = _src.channels(), depth = _src.depth();
    int lutcn = _lut.channels();

    CV_Assert( (lutcn == cn || lutcn == 1) &&
        _lut.total() == 256 && _lut.isContinuous() &&
        (depth == CV_8U || depth == CV_8S) );

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_LUT(_src, _lut, _dst))

    Mat src = _src.getMat(), lut = _lut.getMat();
    _dst.create(src.dims, src.size, CV_MAKETYPE(_lut.depth(), cn));
    Mat dst = _dst.getMat();

    if (_src.dims() <= 2)
    {
        bool ok = false;
        Ptr<ParallelLoopBody> body;
#if defined(HAVE_IPP)
        CV_IPP_CHECK()
        {
            size_t elemSize1 = CV_ELEM_SIZE1(dst.depth());
#if 0 // there are no performance benefits (PR #2653)
            if (lutcn == 1)
            {
                ParallelLoopBody* p = new ipp::IppLUTParallelBody_LUTC1(src, lut, dst, &ok);
                body.reset(p);
            }
            else
#endif
            if ((lutcn == 3 || lutcn == 4) && elemSize1 == 1)
            {
                ParallelLoopBody* p = new ipp::IppLUTParallelBody_LUTCN(src, lut, dst, &ok);
                body.reset(p);
            }
        }
#endif
        if (body == NULL || ok == false)
        {
            ok = false;
            ParallelLoopBody* p = new LUTParallelBody(src, lut, dst, &ok);
            body.reset(p);
        }
        if (body != NULL && ok)
        {
            Range all(0, dst.rows);
            if (dst.total()>>18)
                parallel_for_(all, *body, (double)std::max((size_t)1, dst.total()>>16));
            else
                (*body)(all);
            if (ok)
                return;
        }
    }

    LUTFunc func = lutTab[lut.depth()];
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], lut.ptr(), ptrs[1], len, cn, lutcn);
}

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

        size_t globalsize[2] = { src.cols, (src.rows + rowsPerWI - 1) / rowsPerWI };
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

}

void cv::normalize( InputArray _src, InputOutputArray _dst, double a, double b,
                    int norm_type, int rtype, InputArray _mask )
{
    double scale = 1, shift = 0;
    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = MIN( a, b ), dmax = MAX( a, b );
        minMaxLoc( _src, &smin, &smax, 0, 0, _mask );
        scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
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

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if( rtype < 0 )
        rtype = _dst.fixedType() ? _dst.depth() : depth;
    _dst.createSameSize(_src, CV_MAKETYPE(rtype, cn));

    CV_OCL_RUN(_dst.isUMat(),
               ocl_normalize(_src, _dst, _mask, rtype, scale, shift))

    Mat src = _src.getMat(), dst = _dst.getMat();
    if( _mask.empty() )
        src.convertTo( dst, rtype, scale, shift );
    else
    {
        Mat temp;
        src.convertTo( temp, rtype, scale, shift );
        temp.copyTo( dst, _mask );
    }
}

CV_IMPL void
cvSplit( const void* srcarr, void* dstarr0, void* dstarr1, void* dstarr2, void* dstarr3 )
{
    void* dptrs[] = { dstarr0, dstarr1, dstarr2, dstarr3 };
    cv::Mat src = cv::cvarrToMat(srcarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += dptrs[i] != 0;
    CV_Assert( nz > 0 );
    std::vector<cv::Mat> dvec(nz);
    std::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( dptrs[i] != 0 )
        {
            dvec[j] = cv::cvarrToMat(dptrs[i]);
            CV_Assert( dvec[j].size() == src.size() );
            CV_Assert( dvec[j].depth() == src.depth() );
            CV_Assert( dvec[j].channels() == 1 );
            CV_Assert( i < src.channels() );
            pairs[j*2] = i;
            pairs[j*2+1] = j;
            j++;
        }
    }
    if( nz == src.channels() )
        cv::split( src, dvec );
    else
    {
        cv::mixChannels( &src, 1, &dvec[0], nz, &pairs[0], nz );
    }
}


CV_IMPL void
cvMerge( const void* srcarr0, const void* srcarr1, const void* srcarr2,
         const void* srcarr3, void* dstarr )
{
    const void* sptrs[] = { srcarr0, srcarr1, srcarr2, srcarr3 };
    cv::Mat dst = cv::cvarrToMat(dstarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += sptrs[i] != 0;
    CV_Assert( nz > 0 );
    std::vector<cv::Mat> svec(nz);
    std::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( sptrs[i] != 0 )
        {
            svec[j] = cv::cvarrToMat(sptrs[i]);
            CV_Assert( svec[j].size == dst.size &&
                svec[j].depth() == dst.depth() &&
                svec[j].channels() == 1 && i < dst.channels() );
            pairs[j*2] = j;
            pairs[j*2+1] = i;
            j++;
        }
    }

    if( nz == dst.channels() )
        cv::merge( svec, dst );
    else
    {
        cv::mixChannels( &svec[0], nz, &dst, 1, &pairs[0], nz );
    }
}


CV_IMPL void
cvMixChannels( const CvArr** src, int src_count,
               CvArr** dst, int dst_count,
               const int* from_to, int pair_count )
{
    cv::AutoBuffer<cv::Mat> buf(src_count + dst_count);

    int i;
    for( i = 0; i < src_count; i++ )
        buf[i] = cv::cvarrToMat(src[i]);
    for( i = 0; i < dst_count; i++ )
        buf[i+src_count] = cv::cvarrToMat(dst[i]);
    cv::mixChannels(&buf[0], src_count, &buf[src_count], dst_count, from_to, pair_count);
}

CV_IMPL void
cvConvertScaleAbs( const void* srcarr, void* dstarr,
                   double scale, double shift )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && dst.type() == CV_8UC(src.channels()));
    cv::convertScaleAbs( src, dst, scale, shift );
}

CV_IMPL void
cvConvertScale( const void* srcarr, void* dstarr,
                double scale, double shift )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size == dst.size && src.channels() == dst.channels() );
    src.convertTo(dst, dst.type(), scale, shift);
}

CV_IMPL void cvLUT( const void* srcarr, void* dstarr, const void* lutarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), lut = cv::cvarrToMat(lutarr);

    CV_Assert( dst.size() == src.size() && dst.type() == CV_MAKETYPE(lut.depth(), src.channels()) );
    cv::LUT( src, lut, dst );
}

CV_IMPL void cvNormalize( const CvArr* srcarr, CvArr* dstarr,
                          double a, double b, int norm_type, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    CV_Assert( dst.size() == src.size() && src.channels() == dst.channels() );
    cv::normalize( src, dst, a, b, norm_type, dst.type(), mask );
}

/* End of file. */
