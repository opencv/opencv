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
    int nz = 0;
    for( int i = 0; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

#undef SIMD_ONLY
#if (CV_SIMD || CV_SIMD_SCALABLE)
#define SIMD_ONLY(expr) expr
#else
#define SIMD_ONLY(expr)
#endif

#undef DEFINE_NONZERO_FUNC
#define DEFINE_NONZERO_FUNC(funcname, suffix, ssuffix, T, VT, ST, cmp_op, add_op, update_sum, scalar_cmp_op) \
static int funcname( const T* src, int len ) \
{ \
    int i = 0, nz = 0; \
    SIMD_ONLY( \
    const int vlanes = VTraits<VT>::vlanes(); \
    VT v_zero = vx_setzero_##suffix(); \
    VT v_1 = vx_setall_##suffix(1); \
    VT v_8 = vx_setall_##suffix(8); \
    ST v_sum0 = vx_setzero_##ssuffix(); \
    ST v_sum1 = v_sum0; \
    for (i = 0; i <= len - vlanes*8; i += vlanes*8) \
    { \
        VT x0 = vx_load(src + i); \
        VT x1 = vx_load(src + i + vlanes); \
        VT x2 = vx_load(src + i + vlanes*2); \
        VT x3 = vx_load(src + i + vlanes*3); \
        VT x4 = vx_load(src + i + vlanes*4); \
        VT x5 = vx_load(src + i + vlanes*5); \
        VT x6 = vx_load(src + i + vlanes*6); \
        VT x7 = vx_load(src + i + vlanes*7); \
        x0 = cmp_op(x0, v_zero); \
        x1 = cmp_op(x1, v_zero); \
        x2 = cmp_op(x2, v_zero); \
        x3 = cmp_op(x3, v_zero); \
        x4 = cmp_op(x4, v_zero); \
        x5 = cmp_op(x5, v_zero); \
        x6 = cmp_op(x6, v_zero); \
        x7 = cmp_op(x7, v_zero); \
        x0 = add_op(x0, x1); \
        x2 = add_op(x2, x3); \
        x4 = add_op(x4, x5); \
        x6 = add_op(x6, x7); \
        x0 = add_op(x0, x2); \
        x4 = add_op(x4, x6); \
        x0 = add_op(add_op(x0, x4), v_8); \
        update_sum(v_sum0, v_sum1, x0); \
    } \
    for (; i <= len - vlanes; i += vlanes) \
    { \
        VT x0 = vx_load(src + i); \
        x0 = add_op(cmp_op(x0, v_zero), v_1); \
        update_sum(v_sum0, v_sum1, x0); \
    } \
    nz += (int)v_reduce_sum(v_add(v_sum0, v_sum1)); \
    v_cleanup();) \
    for( ; i < len; i++ ) \
    { \
        nz += scalar_cmp_op(src[i]); \
    } \
    return nz; \
}

#undef CHECK_NZ_INT
#define CHECK_NZ_INT(x) ((x) != 0)
#undef CHECK_NZ_FP
#define CHECK_NZ_FP(x) ((x)*2 != 0)
#undef VEC_CMP_EQ_Z_FP16
#define VEC_CMP_EQ_Z_FP16(x, z) v_eq(v_add_wrap(x, x), z)
#undef VEC_CMP_EQ_Z_FP
#define VEC_CMP_EQ_Z_FP(x, z) v_eq(v_add(x, x), z)

#undef UPDATE_SUM_U8
#define UPDATE_SUM_U8(v_sum0, v_sum1, x0) \
    v_uint16 w0 = v_expand_low(x0); \
    v_uint16 w1 = v_expand_high(x0); \
    v_sum0 = v_add(v_sum0, v_expand_low(w0)); \
    v_sum1 = v_add(v_sum1, v_expand_high(w0)); \
    v_sum0 = v_add(v_sum0, v_expand_low(w1)); \
    v_sum1 = v_add(v_sum1, v_expand_high(w1))

#undef UPDATE_SUM_U16
#define UPDATE_SUM_U16(v_sum0, v_sum1, x0) \
    v_sum0 = v_add(v_sum0, v_expand_low(x0)); \
    v_sum1 = v_add(v_sum1, v_expand_high(x0))

#undef UPDATE_SUM_S32
#define UPDATE_SUM_S32(v_sum0, v_sum1, x0) \
    v_sum0 = v_add(v_sum0, x0)

DEFINE_NONZERO_FUNC(countNonZero8u, u8, u32, uchar, v_uint8, v_uint32, v_eq, v_add_wrap, UPDATE_SUM_U8, CHECK_NZ_INT)
DEFINE_NONZERO_FUNC(countNonZero16u, u16, u32, ushort, v_uint16, v_uint32, v_eq, v_add_wrap, UPDATE_SUM_U16, CHECK_NZ_INT)
DEFINE_NONZERO_FUNC(countNonZero32s, s32, s32, int, v_int32, v_int32, v_eq, v_add, UPDATE_SUM_S32, CHECK_NZ_INT)
DEFINE_NONZERO_FUNC(countNonZero32f, u32, u32, uint, v_uint32, v_uint32, VEC_CMP_EQ_Z_FP, v_add, UPDATE_SUM_S32, CHECK_NZ_FP)
DEFINE_NONZERO_FUNC(countNonZero16f, u16, u32, ushort, v_uint16, v_uint32, VEC_CMP_EQ_Z_FP16, v_add_wrap, UPDATE_SUM_U16, CHECK_NZ_FP)

#undef DEFINE_NONZERO_FUNC_NOSIMD
#define DEFINE_NONZERO_FUNC_NOSIMD(funcname, T) \
static int funcname(const T* src, int len) \
{ \
    return countNonZero_(src, len); \
}

DEFINE_NONZERO_FUNC_NOSIMD(countNonZero64s, int64)
DEFINE_NONZERO_FUNC_NOSIMD(countNonZero64f, double)

CountNonZeroFunc getCountNonZeroTab(int depth)
{
    static CountNonZeroFunc countNonZeroTab[CV_DEPTH_MAX] =
    {
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32s),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32f),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero64f),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16f),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16f), // for bf16 it's the same code as for f16
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero64s),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero64s),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32s),
        0
    };

    return countNonZeroTab[depth];
}

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
