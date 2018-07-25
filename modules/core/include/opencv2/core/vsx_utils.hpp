// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_VSX_UTILS_HPP
#define OPENCV_HAL_VSX_UTILS_HPP

#include "opencv2/core/cvdef.h"

#ifndef SKIP_INCLUDES
#   include <assert.h>
#endif

//! @addtogroup core_utils_vsx
//! @{
#if CV_VSX

#define __VSX_S16__(c, v) (c){v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v}
#define __VSX_S8__(c, v)  (c){v, v, v, v, v, v, v, v}
#define __VSX_S4__(c, v)  (c){v, v, v, v}
#define __VSX_S2__(c, v)  (c){v, v}

typedef __vector unsigned char vec_uchar16;
#define vec_uchar16_set(...) (vec_uchar16){__VA_ARGS__}
#define vec_uchar16_sp(c)    (__VSX_S16__(vec_uchar16, c))
#define vec_uchar16_c(v)     ((vec_uchar16)(v))
#define vec_uchar16_z        vec_uchar16_sp(0)

typedef __vector signed char vec_char16;
#define vec_char16_set(...) (vec_char16){__VA_ARGS__}
#define vec_char16_sp(c)    (__VSX_S16__(vec_char16, c))
#define vec_char16_c(v)     ((vec_char16)(v))
#define vec_char16_z        vec_char16_sp(0)

typedef __vector unsigned short vec_ushort8;
#define vec_ushort8_set(...) (vec_ushort8){__VA_ARGS__}
#define vec_ushort8_sp(c)    (__VSX_S8__(vec_ushort8, c))
#define vec_ushort8_c(v)     ((vec_ushort8)(v))
#define vec_ushort8_z        vec_ushort8_sp(0)

typedef __vector signed short vec_short8;
#define vec_short8_set(...) (vec_short8){__VA_ARGS__}
#define vec_short8_sp(c)    (__VSX_S8__(vec_short8, c))
#define vec_short8_c(v)     ((vec_short8)(v))
#define vec_short8_z        vec_short8_sp(0)

typedef __vector unsigned int vec_uint4;
#define vec_uint4_set(...) (vec_uint4){__VA_ARGS__}
#define vec_uint4_sp(c)    (__VSX_S4__(vec_uint4, c))
#define vec_uint4_c(v)     ((vec_uint4)(v))
#define vec_uint4_z        vec_uint4_sp(0)

typedef __vector signed int vec_int4;
#define vec_int4_set(...)  (vec_int4){__VA_ARGS__}
#define vec_int4_sp(c)     (__VSX_S4__(vec_int4, c))
#define vec_int4_c(v)      ((vec_int4)(v))
#define vec_int4_z         vec_int4_sp(0)

typedef __vector float vec_float4;
#define vec_float4_set(...)  (vec_float4){__VA_ARGS__}
#define vec_float4_sp(c)     (__VSX_S4__(vec_float4, c))
#define vec_float4_c(v)      ((vec_float4)(v))
#define vec_float4_z         vec_float4_sp(0)

typedef __vector unsigned long long vec_udword2;
#define vec_udword2_set(...) (vec_udword2){__VA_ARGS__}
#define vec_udword2_sp(c)    (__VSX_S2__(vec_udword2, c))
#define vec_udword2_c(v)     ((vec_udword2)(v))
#define vec_udword2_z        vec_udword2_sp(0)

typedef __vector signed long long vec_dword2;
#define vec_dword2_set(...) (vec_dword2){__VA_ARGS__}
#define vec_dword2_sp(c)    (__VSX_S2__(vec_dword2, c))
#define vec_dword2_c(v)     ((vec_dword2)(v))
#define vec_dword2_z        vec_dword2_sp(0)

typedef  __vector double vec_double2;
#define vec_double2_set(...) (vec_double2){__VA_ARGS__}
#define vec_double2_c(v)     ((vec_double2)(v))
#define vec_double2_sp(c)    (__VSX_S2__(vec_double2, c))
#define vec_double2_z        vec_double2_sp(0)

#define vec_bchar16           __vector __bool char
#define vec_bchar16_set(...) (vec_bchar16){__VA_ARGS__}
#define vec_bchar16_c(v)     ((vec_bchar16)(v))

#define vec_bshort8           __vector __bool short
#define vec_bshort8_set(...) (vec_bshort8){__VA_ARGS__}
#define vec_bshort8_c(v)     ((vec_bshort8)(v))

#define vec_bint4             __vector __bool int
#define vec_bint4_set(...)   (vec_bint4){__VA_ARGS__}
#define vec_bint4_c(v)       ((vec_bint4)(v))

#define vec_bdword2            __vector __bool long long
#define vec_bdword2_set(...)  (vec_bdword2){__VA_ARGS__}
#define vec_bdword2_c(v)      ((vec_bdword2)(v))

#define VSX_FINLINE(tp) extern inline tp __attribute__((always_inline))

#define VSX_REDIRECT_1RG(rt, rg, fnm, fn2)   \
VSX_FINLINE(rt) fnm(const rg& a) { return fn2(a); }

#define VSX_REDIRECT_2RG(rt, rg, fnm, fn2)   \
VSX_FINLINE(rt) fnm(const rg& a, const rg& b) { return fn2(a, b); }

/*
 * GCC VSX compatibility
**/
#if defined(__GNUG__) && !defined(__clang__)

// inline asm helper
#define VSX_IMPL_1RG(rt, rto, rg, rgo, opc, fnm) \
VSX_FINLINE(rt) fnm(const rg& a)                 \
{ rt rs; __asm__ __volatile__(#opc" %x0,%x1" : "="#rto (rs) : #rgo (a)); return rs; }

#define VSX_IMPL_1VRG(rt, rg, opc, fnm) \
VSX_FINLINE(rt) fnm(const rg& a)        \
{ rt rs; __asm__ __volatile__(#opc" %0,%1" : "=v" (rs) : "v" (a)); return rs; }

#define VSX_IMPL_2VRG_F(rt, rg, fopc, fnm)     \
VSX_FINLINE(rt) fnm(const rg& a, const rg& b)  \
{ rt rs; __asm__ __volatile__(fopc : "=v" (rs) : "v" (a), "v" (b)); return rs; }

#define VSX_IMPL_2VRG(rt, rg, opc, fnm) VSX_IMPL_2VRG_F(rt, rg, #opc" %0,%1,%2", fnm)

#if __GNUG__ < 7
// up to GCC 6 vec_mul only supports precisions and llong
#   ifdef vec_mul
#       undef vec_mul
#   endif
/*
 * there's no a direct instruction for supporting 16-bit multiplication in ISA 2.07,
 * XLC Implement it by using instruction "multiply even", "multiply odd" and "permute"
 * todo: Do I need to support 8-bit ?
**/
#   define VSX_IMPL_MULH(Tvec, Tcast)                                               \
    VSX_FINLINE(Tvec) vec_mul(const Tvec& a, const Tvec& b)                         \
    {                                                                               \
        static const vec_uchar16 even_perm = {0, 1, 16, 17, 4, 5, 20, 21,           \
                                              8, 9, 24, 25, 12, 13, 28, 29};        \
        return vec_perm(Tcast(vec_mule(a, b)), Tcast(vec_mulo(a, b)), even_perm);   \
    }
    VSX_IMPL_MULH(vec_short8,  vec_short8_c)
    VSX_IMPL_MULH(vec_ushort8, vec_ushort8_c)
    // vmuluwm can be used for unsigned or signed integers, that's what they said
    VSX_IMPL_2VRG(vec_int4,  vec_int4,  vmuluwm, vec_mul)
    VSX_IMPL_2VRG(vec_uint4, vec_uint4, vmuluwm, vec_mul)
    // redirect to GCC builtin vec_mul, since it already supports precisions and llong
    VSX_REDIRECT_2RG(vec_float4,  vec_float4,  vec_mul, __builtin_vec_mul)
    VSX_REDIRECT_2RG(vec_double2, vec_double2, vec_mul, __builtin_vec_mul)
    VSX_REDIRECT_2RG(vec_dword2,  vec_dword2,  vec_mul, __builtin_vec_mul)
    VSX_REDIRECT_2RG(vec_udword2, vec_udword2, vec_mul, __builtin_vec_mul)
#endif // __GNUG__ < 7

#if __GNUG__ < 6
/*
 * Instruction "compare greater than or equal" in ISA 2.07 only supports single
 * and double precision.
 * In XLC and new versions of GCC implement integers by using instruction "greater than" and NOR.
**/
#   ifdef vec_cmpge
#       undef vec_cmpge
#   endif
#   ifdef vec_cmple
#       undef vec_cmple
#   endif
#   define vec_cmple(a, b) vec_cmpge(b, a)
#   define VSX_IMPL_CMPGE(rt, rg, opc, fnm) \
    VSX_IMPL_2VRG_F(rt, rg, #opc" %0,%2,%1\n\t xxlnor %x0,%x0,%x0", fnm)

    VSX_IMPL_CMPGE(vec_bchar16, vec_char16,  vcmpgtsb, vec_cmpge)
    VSX_IMPL_CMPGE(vec_bchar16, vec_uchar16, vcmpgtub, vec_cmpge)
    VSX_IMPL_CMPGE(vec_bshort8, vec_short8,  vcmpgtsh, vec_cmpge)
    VSX_IMPL_CMPGE(vec_bshort8, vec_ushort8, vcmpgtuh, vec_cmpge)
    VSX_IMPL_CMPGE(vec_bint4,   vec_int4,    vcmpgtsw, vec_cmpge)
    VSX_IMPL_CMPGE(vec_bint4,   vec_uint4,   vcmpgtuw, vec_cmpge)
    VSX_IMPL_CMPGE(vec_bdword2, vec_dword2,  vcmpgtsd, vec_cmpge)
    VSX_IMPL_CMPGE(vec_bdword2, vec_udword2, vcmpgtud, vec_cmpge)

// redirect to GCC builtin cmpge, since it already supports precisions
    VSX_REDIRECT_2RG(vec_bint4,   vec_float4,  vec_cmpge, __builtin_vec_cmpge)
    VSX_REDIRECT_2RG(vec_bdword2, vec_double2, vec_cmpge, __builtin_vec_cmpge)

// up to gcc5 vec_nor doesn't support bool long long
#   undef vec_nor
    template<typename T>
    VSX_REDIRECT_2RG(T, T, vec_nor, __builtin_vec_nor)

    VSX_FINLINE(vec_bdword2) vec_nor(const vec_bdword2& a, const vec_bdword2& b)
    { return vec_bdword2_c(__builtin_vec_nor(vec_dword2_c(a), vec_dword2_c(b))); }

// vec_packs doesn't support double words in gcc4 and old versions of gcc5
#   undef vec_packs
    VSX_REDIRECT_2RG(vec_char16,  vec_short8,  vec_packs, __builtin_vec_packs)
    VSX_REDIRECT_2RG(vec_uchar16, vec_ushort8, vec_packs, __builtin_vec_packs)
    VSX_REDIRECT_2RG(vec_short8,  vec_int4,    vec_packs, __builtin_vec_packs)
    VSX_REDIRECT_2RG(vec_ushort8, vec_uint4,   vec_packs, __builtin_vec_packs)

    VSX_IMPL_2VRG_F(vec_int4,  vec_dword2,  "vpksdss %0,%2,%1", vec_packs)
    VSX_IMPL_2VRG_F(vec_uint4, vec_udword2, "vpkudus %0,%2,%1", vec_packs)
#endif // __GNUG__ < 6

#if __GNUG__ < 5
// vec_xxpermdi in gcc4 missing little-endian supports just like clang
#   define vec_permi(a, b, c) vec_xxpermdi(b, a, (3 ^ ((c & 1) << 1 | c >> 1)))
#else
#   define vec_permi vec_xxpermdi
#endif // __GNUG__ < 5

// shift left double by word immediate
#ifndef vec_sldw
#   define vec_sldw __builtin_vsx_xxsldwi
#endif

// vector population count
VSX_IMPL_1VRG(vec_uchar16, vec_uchar16, vpopcntb, vec_popcntu)
VSX_IMPL_1VRG(vec_uchar16, vec_char16,  vpopcntb, vec_popcntu)
VSX_IMPL_1VRG(vec_ushort8, vec_ushort8, vpopcnth, vec_popcntu)
VSX_IMPL_1VRG(vec_ushort8, vec_short8,  vpopcnth, vec_popcntu)
VSX_IMPL_1VRG(vec_uint4,   vec_uint4,   vpopcntw, vec_popcntu)
VSX_IMPL_1VRG(vec_uint4,   vec_int4,    vpopcntw, vec_popcntu)
VSX_IMPL_1VRG(vec_udword2, vec_udword2, vpopcntd, vec_popcntu)
VSX_IMPL_1VRG(vec_udword2, vec_dword2,  vpopcntd, vec_popcntu)

// converts between single and double-precision
VSX_REDIRECT_1RG(vec_float4,  vec_double2, vec_cvfo, __builtin_vsx_xvcvdpsp)
VSX_REDIRECT_1RG(vec_double2, vec_float4,  vec_cvfo, __builtin_vsx_xvcvspdp)

// converts word and doubleword to double-precision
#ifdef vec_ctd
#   undef vec_ctd
#endif
VSX_IMPL_1RG(vec_double2, wd, vec_int4,    wa, xvcvsxwdp, vec_ctdo)
VSX_IMPL_1RG(vec_double2, wd, vec_uint4,   wa, xvcvuxwdp, vec_ctdo)
VSX_IMPL_1RG(vec_double2, wd, vec_dword2,  wi, xvcvsxddp, vec_ctd)
VSX_IMPL_1RG(vec_double2, wd, vec_udword2, wi, xvcvuxddp, vec_ctd)

// converts word and doubleword to single-precision
#undef vec_ctf
VSX_IMPL_1RG(vec_float4, wf, vec_int4,    wa, xvcvsxwsp, vec_ctf)
VSX_IMPL_1RG(vec_float4, wf, vec_uint4,   wa, xvcvuxwsp, vec_ctf)
VSX_IMPL_1RG(vec_float4, wf, vec_dword2,  wi, xvcvsxdsp, vec_ctfo)
VSX_IMPL_1RG(vec_float4, wf, vec_udword2, wi, xvcvuxdsp, vec_ctfo)

// converts single and double precision to signed word
#undef vec_cts
VSX_IMPL_1RG(vec_int4,  wa, vec_double2, wd, xvcvdpsxws, vec_ctso)
VSX_IMPL_1RG(vec_int4,  wa, vec_float4,  wf, xvcvspsxws, vec_cts)

// converts single and double precision to unsigned word
#undef vec_ctu
VSX_IMPL_1RG(vec_uint4, wa, vec_double2, wd, xvcvdpuxws, vec_ctuo)
VSX_IMPL_1RG(vec_uint4, wa, vec_float4,  wf, xvcvspuxws, vec_ctu)

// converts single and double precision to signed doubleword
#ifdef vec_ctsl
#   undef vec_ctsl
#endif
VSX_IMPL_1RG(vec_dword2, wi, vec_double2, wd, xvcvdpsxds, vec_ctsl)
VSX_IMPL_1RG(vec_dword2, wi, vec_float4,  wf, xvcvspsxds, vec_ctslo)

// converts single and double precision to unsigned doubleword
#ifdef vec_ctul
#   undef vec_ctul
#endif
VSX_IMPL_1RG(vec_udword2, wi, vec_double2, wd, xvcvdpuxds, vec_ctul)
VSX_IMPL_1RG(vec_udword2, wi, vec_float4,  wf, xvcvspuxds, vec_ctulo)

// just in case if GCC doesn't define it
#ifndef vec_xl
#   define vec_xl vec_vsx_ld
#   define vec_xst vec_vsx_st
#endif

#endif // GCC VSX compatibility

/*
 * CLANG VSX compatibility
**/
#if defined(__clang__) && !defined(__IBMCPP__)

/*
 * CLANG doesn't support %x<n> in the inline asm template which fixes register number
 * when using any of the register constraints wa, wd, wf
 *
 * For more explanation checkout PowerPC and IBM RS6000 in https://gcc.gnu.org/onlinedocs/gcc/Machine-Constraints.html
 * Also there's already an open bug https://bugs.llvm.org/show_bug.cgi?id=31837
 *
 * So we're not able to use inline asm and only use built-in functions that CLANG supports
 * and use __builtin_convertvector if clang missng any of vector conversions built-in functions
*/

// convert vector helper
#define VSX_IMPL_CONVERT(rt, rg, fnm) \
VSX_FINLINE(rt) fnm(const rg& a) { return __builtin_convertvector(a, rt); }

#if __clang_major__ < 5
// implement vec_permi in a dirty way
#   define VSX_IMPL_CLANG_4_PERMI(Tvec)                                                 \
    VSX_FINLINE(Tvec) vec_permi(const Tvec& a, const Tvec& b, unsigned const char c)    \
    {                                                                                   \
        switch (c)                                                                      \
        {                                                                               \
        case 0:                                                                         \
            return vec_mergeh(a, b);                                                    \
        case 1:                                                                         \
            return vec_mergel(vec_mergeh(a, a), b);                                     \
        case 2:                                                                         \
            return vec_mergeh(vec_mergel(a, a), b);                                     \
        default:                                                                        \
            return vec_mergel(a, b);                                                    \
        }                                                                               \
    }
    VSX_IMPL_CLANG_4_PERMI(vec_udword2)
    VSX_IMPL_CLANG_4_PERMI(vec_dword2)
    VSX_IMPL_CLANG_4_PERMI(vec_double2)

// vec_xxsldwi is missing in clang 4
#   define vec_xxsldwi(a, b, c) vec_sld(a, b, (c) * 4)
#else
// vec_xxpermdi is missing little-endian supports in clang 4 just like gcc4
#   define vec_permi(a, b, c) vec_xxpermdi(b, a, (3 ^ ((c & 1) << 1 | c >> 1)))
#endif // __clang_major__ < 5

// shift left double by word immediate
#ifndef vec_sldw
#   define vec_sldw vec_xxsldwi
#endif

// Implement vec_rsqrt since clang only supports vec_rsqrte
#ifndef vec_rsqrt
    VSX_FINLINE(vec_float4) vec_rsqrt(const vec_float4& a)
    { return vec_div(vec_float4_sp(1), vec_sqrt(a)); }

    VSX_FINLINE(vec_double2) vec_rsqrt(const vec_double2& a)
    { return vec_div(vec_double2_sp(1), vec_sqrt(a)); }
#endif

// vec_promote missing support for doubleword
VSX_FINLINE(vec_dword2) vec_promote(long long a, int b)
{
    vec_dword2 ret = vec_dword2_z;
    ret[b & 1] = a;
    return ret;
}

VSX_FINLINE(vec_udword2) vec_promote(unsigned long long a, int b)
{
    vec_udword2 ret = vec_udword2_z;
    ret[b & 1] = a;
    return ret;
}

// vec_popcnt should return unsigned but clang has different thought just like gcc in vec_vpopcnt
#define VSX_IMPL_POPCNTU(Tvec, Tvec2, ucast)   \
VSX_FINLINE(Tvec) vec_popcntu(const Tvec2& a)  \
{ return ucast(vec_popcnt(a)); }
VSX_IMPL_POPCNTU(vec_uchar16, vec_char16, vec_uchar16_c);
VSX_IMPL_POPCNTU(vec_ushort8, vec_short8, vec_ushort8_c);
VSX_IMPL_POPCNTU(vec_uint4,   vec_int4,   vec_uint4_c);
// redirect unsigned types
VSX_REDIRECT_1RG(vec_uchar16, vec_uchar16, vec_popcntu, vec_popcnt)
VSX_REDIRECT_1RG(vec_ushort8, vec_ushort8, vec_popcntu, vec_popcnt)
VSX_REDIRECT_1RG(vec_uint4,   vec_uint4,   vec_popcntu, vec_popcnt)

// converts between single and double precision
VSX_REDIRECT_1RG(vec_float4,  vec_double2, vec_cvfo, __builtin_vsx_xvcvdpsp)
VSX_REDIRECT_1RG(vec_double2, vec_float4,  vec_cvfo, __builtin_vsx_xvcvspdp)

// converts word and doubleword to double-precision
#ifdef vec_ctd
#   undef vec_ctd
#endif
VSX_REDIRECT_1RG(vec_double2, vec_int4,  vec_ctdo, __builtin_vsx_xvcvsxwdp)
VSX_REDIRECT_1RG(vec_double2, vec_uint4, vec_ctdo, __builtin_vsx_xvcvuxwdp)

VSX_IMPL_CONVERT(vec_double2, vec_dword2,  vec_ctd)
VSX_IMPL_CONVERT(vec_double2, vec_udword2, vec_ctd)

// converts word and doubleword to single-precision
#if __clang_major__ > 4
#   undef vec_ctf
#endif
VSX_IMPL_CONVERT(vec_float4, vec_int4,    vec_ctf)
VSX_IMPL_CONVERT(vec_float4, vec_uint4,   vec_ctf)
VSX_REDIRECT_1RG(vec_float4, vec_dword2,  vec_ctfo, __builtin_vsx_xvcvsxdsp)
VSX_REDIRECT_1RG(vec_float4, vec_udword2, vec_ctfo, __builtin_vsx_xvcvuxdsp)

// converts single and double precision to signed word
#if __clang_major__ > 4
#   undef vec_cts
#endif
VSX_REDIRECT_1RG(vec_int4,  vec_double2, vec_ctso, __builtin_vsx_xvcvdpsxws)
VSX_IMPL_CONVERT(vec_int4,  vec_float4,  vec_cts)

// converts single and double precision to unsigned word
#if __clang_major__ > 4
#   undef vec_ctu
#endif
VSX_REDIRECT_1RG(vec_uint4, vec_double2, vec_ctuo, __builtin_vsx_xvcvdpuxws)
VSX_IMPL_CONVERT(vec_uint4, vec_float4,  vec_ctu)

// converts single and double precision to signed doubleword
#ifdef vec_ctsl
#   undef vec_ctsl
#endif
VSX_IMPL_CONVERT(vec_dword2, vec_double2, vec_ctsl)
// __builtin_convertvector unable to convert, xvcvspsxds is missing on it
VSX_FINLINE(vec_dword2) vec_ctslo(const vec_float4& a)
{ return vec_ctsl(vec_cvfo(a)); }

// converts single and double precision to unsigned doubleword
#ifdef vec_ctul
#   undef vec_ctul
#endif
VSX_IMPL_CONVERT(vec_udword2, vec_double2, vec_ctul)
// __builtin_convertvector unable to convert, xvcvspuxds is missing on it
VSX_FINLINE(vec_udword2) vec_ctulo(const vec_float4& a)
{ return vec_ctul(vec_cvfo(a)); }

#endif // CLANG VSX compatibility

/*
 * Common GCC, CLANG compatibility
**/
#if defined(__GNUG__) && !defined(__IBMCPP__)

#ifdef vec_cvf
#   undef vec_cvf
#endif

#define VSX_IMPL_CONV_EVEN_4_2(rt, rg, fnm, fn2) \
VSX_FINLINE(rt) fnm(const rg& a)                 \
{ return fn2(vec_sldw(a, a, 1)); }

VSX_IMPL_CONV_EVEN_4_2(vec_double2, vec_float4, vec_cvf,  vec_cvfo)
VSX_IMPL_CONV_EVEN_4_2(vec_double2, vec_int4,   vec_ctd,  vec_ctdo)
VSX_IMPL_CONV_EVEN_4_2(vec_double2, vec_uint4,  vec_ctd,  vec_ctdo)

VSX_IMPL_CONV_EVEN_4_2(vec_dword2,  vec_float4, vec_ctsl, vec_ctslo)
VSX_IMPL_CONV_EVEN_4_2(vec_udword2, vec_float4, vec_ctul, vec_ctulo)

#define VSX_IMPL_CONV_EVEN_2_4(rt, rg, fnm, fn2) \
VSX_FINLINE(rt) fnm(const rg& a)                 \
{                                                \
    rt v4 = fn2(a);                              \
    return vec_sldw(v4, v4, 3);                  \
}

VSX_IMPL_CONV_EVEN_2_4(vec_float4, vec_double2, vec_cvf, vec_cvfo)
VSX_IMPL_CONV_EVEN_2_4(vec_float4, vec_dword2,  vec_ctf, vec_ctfo)
VSX_IMPL_CONV_EVEN_2_4(vec_float4, vec_udword2, vec_ctf, vec_ctfo)

VSX_IMPL_CONV_EVEN_2_4(vec_int4,   vec_double2, vec_cts, vec_ctso)
VSX_IMPL_CONV_EVEN_2_4(vec_uint4,  vec_double2, vec_ctu, vec_ctuo)

// Only for Eigen!
/*
 * changing behavior of conversion intrinsics for gcc has effect on Eigen
 * so we redfine old behavior again only on gcc, clang
*/
#if !defined(__clang__) || __clang_major__ > 4
    // ignoring second arg since Eigen only truncates toward zero
#   define VSX_IMPL_CONV_2VARIANT(rt, rg, fnm, fn2)     \
    VSX_FINLINE(rt) fnm(const rg& a, int only_truncate) \
    {                                                   \
        assert(only_truncate == 0);                     \
        (void)only_truncate;                            \
        return fn2(a);                                  \
    }
    VSX_IMPL_CONV_2VARIANT(vec_int4,   vec_float4,  vec_cts, vec_cts)
    VSX_IMPL_CONV_2VARIANT(vec_float4, vec_int4,    vec_ctf, vec_ctf)
    // define vec_cts for converting double precision to signed doubleword
    // which isn't combitable with xlc but its okay since Eigen only use it for gcc
    VSX_IMPL_CONV_2VARIANT(vec_dword2, vec_double2, vec_cts, vec_ctsl)
#endif // Eigen

#endif // Common GCC, CLANG compatibility

/*
 * XLC VSX compatibility
**/
#if defined(__IBMCPP__)

// vector population count
#define vec_popcntu vec_popcnt

// overload and redirect with setting second arg to zero
// since we only support conversions without the second arg
#define VSX_IMPL_OVERLOAD_Z2(rt, rg, fnm) \
VSX_FINLINE(rt) fnm(const rg& a) { return fnm(a, 0); }

VSX_IMPL_OVERLOAD_Z2(vec_double2, vec_int4,    vec_ctd)
VSX_IMPL_OVERLOAD_Z2(vec_double2, vec_uint4,   vec_ctd)
VSX_IMPL_OVERLOAD_Z2(vec_double2, vec_dword2,  vec_ctd)
VSX_IMPL_OVERLOAD_Z2(vec_double2, vec_udword2, vec_ctd)

VSX_IMPL_OVERLOAD_Z2(vec_float4,  vec_int4,    vec_ctf)
VSX_IMPL_OVERLOAD_Z2(vec_float4,  vec_uint4,   vec_ctf)
VSX_IMPL_OVERLOAD_Z2(vec_float4,  vec_dword2,  vec_ctf)
VSX_IMPL_OVERLOAD_Z2(vec_float4,  vec_udword2, vec_ctf)

VSX_IMPL_OVERLOAD_Z2(vec_int4,    vec_double2, vec_cts)
VSX_IMPL_OVERLOAD_Z2(vec_int4,    vec_float4,  vec_cts)

VSX_IMPL_OVERLOAD_Z2(vec_uint4,   vec_double2, vec_ctu)
VSX_IMPL_OVERLOAD_Z2(vec_uint4,   vec_float4,  vec_ctu)

VSX_IMPL_OVERLOAD_Z2(vec_dword2,  vec_double2, vec_ctsl)
VSX_IMPL_OVERLOAD_Z2(vec_dword2,  vec_float4,  vec_ctsl)

VSX_IMPL_OVERLOAD_Z2(vec_udword2, vec_double2, vec_ctul)
VSX_IMPL_OVERLOAD_Z2(vec_udword2, vec_float4,  vec_ctul)

// fixme: implement conversions of odd-numbered elements in a dirty way
// since xlc doesn't support VSX registers operand in inline asm.
#define VSX_IMPL_CONV_ODD_4_2(rt, rg, fnm, fn2) \
VSX_FINLINE(rt) fnm(const rg& a) { return fn2(vec_sldw(a, a, 3)); }

VSX_IMPL_CONV_ODD_4_2(vec_double2, vec_float4, vec_cvfo,  vec_cvf)
VSX_IMPL_CONV_ODD_4_2(vec_double2, vec_int4,   vec_ctdo,  vec_ctd)
VSX_IMPL_CONV_ODD_4_2(vec_double2, vec_uint4,  vec_ctdo,  vec_ctd)

VSX_IMPL_CONV_ODD_4_2(vec_dword2,  vec_float4, vec_ctslo, vec_ctsl)
VSX_IMPL_CONV_ODD_4_2(vec_udword2, vec_float4, vec_ctulo, vec_ctul)

#define VSX_IMPL_CONV_ODD_2_4(rt, rg, fnm, fn2)  \
VSX_FINLINE(rt) fnm(const rg& a)                 \
{                                                \
    rt v4 = fn2(a);                              \
    return vec_sldw(v4, v4, 1);                  \
}

VSX_IMPL_CONV_ODD_2_4(vec_float4, vec_double2, vec_cvfo, vec_cvf)
VSX_IMPL_CONV_ODD_2_4(vec_float4, vec_dword2,  vec_ctfo, vec_ctf)
VSX_IMPL_CONV_ODD_2_4(vec_float4, vec_udword2, vec_ctfo, vec_ctf)

VSX_IMPL_CONV_ODD_2_4(vec_int4,   vec_double2, vec_ctso, vec_cts)
VSX_IMPL_CONV_ODD_2_4(vec_uint4,  vec_double2, vec_ctuo, vec_ctu)

#endif // XLC VSX compatibility

// ignore GCC warning that caused by -Wunused-but-set-variable in rare cases
#if defined(__GNUG__) && !defined(__clang__)
#   define VSX_UNUSED(Tvec) Tvec __attribute__((__unused__))
#else // CLANG, XLC
#   define VSX_UNUSED(Tvec) Tvec
#endif

// gcc can find his way in casting log int and XLC, CLANG ambiguous
#if defined(__clang__) || defined(__IBMCPP__)
    VSX_FINLINE(vec_udword2) vec_splats(uint64 v)
    { return vec_splats((unsigned long long) v); }

    VSX_FINLINE(vec_dword2) vec_splats(int64 v)
    { return vec_splats((long long) v); }

    VSX_FINLINE(vec_udword2) vec_promote(uint64 a, int b)
    { return vec_promote((unsigned long long) a, b); }

    VSX_FINLINE(vec_dword2) vec_promote(int64 a, int b)
    { return vec_promote((long long) a, b); }
#endif

/*
 * implement vsx_ld(offset, pointer), vsx_st(vector, offset, pointer)
 * load and set using offset depend on the pointer type
 *
 * implement vsx_ldf(offset, pointer), vsx_stf(vector, offset, pointer)
 * load and set using offset depend on fixed bytes size
 *
 * Note: In clang vec_xl and vec_xst fails to load unaligned addresses
 * so we are using vec_vsx_ld, vec_vsx_st instead
*/

#if defined(__clang__) && !defined(__IBMCPP__)
#   define vsx_ldf  vec_vsx_ld
#   define vsx_stf  vec_vsx_st
#else // GCC , XLC
#   define vsx_ldf  vec_xl
#   define vsx_stf  vec_xst
#endif

#define VSX_OFFSET(o, p) ((o) * sizeof(*(p)))
#define vsx_ld(o, p) vsx_ldf(VSX_OFFSET(o, p), p)
#define vsx_st(v, o, p) vsx_stf(v, VSX_OFFSET(o, p), p)

/*
 * implement vsx_ld2(offset, pointer), vsx_st2(vector, offset, pointer) to load and store double words
 * In GCC vec_xl and vec_xst it maps to vec_vsx_ld, vec_vsx_st which doesn't support long long
 * and in CLANG we are using vec_vsx_ld, vec_vsx_st because vec_xl, vec_xst fails to load unaligned addresses
 *
 * In XLC vec_xl and vec_xst fail to cast int64(long int) to long long
*/
#if (defined(__GNUG__) || defined(__clang__)) && !defined(__IBMCPP__)
    VSX_FINLINE(vec_udword2) vsx_ld2(long o, const uint64* p)
    { return vec_udword2_c(vsx_ldf(VSX_OFFSET(o, p), (unsigned int*)p)); }

    VSX_FINLINE(vec_dword2) vsx_ld2(long o, const int64* p)
    { return vec_dword2_c(vsx_ldf(VSX_OFFSET(o, p), (int*)p)); }

    VSX_FINLINE(void) vsx_st2(const vec_udword2& vec, long o, uint64* p)
    { vsx_stf(vec_uint4_c(vec), VSX_OFFSET(o, p), (unsigned int*)p); }

    VSX_FINLINE(void) vsx_st2(const vec_dword2& vec, long o, int64* p)
    { vsx_stf(vec_int4_c(vec), VSX_OFFSET(o, p), (int*)p); }
#else // XLC
    VSX_FINLINE(vec_udword2) vsx_ld2(long o, const uint64* p)
    { return vsx_ldf(VSX_OFFSET(o, p), (unsigned long long*)p); }

    VSX_FINLINE(vec_dword2) vsx_ld2(long o, const int64* p)
    { return vsx_ldf(VSX_OFFSET(o, p), (long long*)p); }

    VSX_FINLINE(void) vsx_st2(const vec_udword2& vec, long o, uint64* p)
    { vsx_stf(vec, VSX_OFFSET(o, p), (unsigned long long*)p); }

    VSX_FINLINE(void) vsx_st2(const vec_dword2& vec, long o, int64* p)
    { vsx_stf(vec, VSX_OFFSET(o, p), (long long*)p); }
#endif

// Store lower 8 byte
#define vec_st_l8(v, p) *((uint64*)(p)) = vec_extract(vec_udword2_c(v), 0)

// Store higher 8 byte
#define vec_st_h8(v, p) *((uint64*)(p)) = vec_extract(vec_udword2_c(v), 1)

// Load 64-bits of integer data to lower part
#define VSX_IMPL_LOAD_L8(Tvec, Tp)                  \
VSX_FINLINE(Tvec) vec_ld_l8(const Tp *p)            \
{ return ((Tvec)vec_promote(*((uint64*)p), 0)); }

VSX_IMPL_LOAD_L8(vec_uchar16, uchar)
VSX_IMPL_LOAD_L8(vec_char16,  schar)
VSX_IMPL_LOAD_L8(vec_ushort8, ushort)
VSX_IMPL_LOAD_L8(vec_short8,  short)
VSX_IMPL_LOAD_L8(vec_uint4,   uint)
VSX_IMPL_LOAD_L8(vec_int4,    int)
VSX_IMPL_LOAD_L8(vec_float4,  float)
VSX_IMPL_LOAD_L8(vec_udword2, uint64)
VSX_IMPL_LOAD_L8(vec_dword2,  int64)
VSX_IMPL_LOAD_L8(vec_double2, double)

// logical not
#define vec_not(a) vec_nor(a, a)

// power9 yaya
// not equal
#ifndef vec_cmpne
#   define vec_cmpne(a, b) vec_not(vec_cmpeq(a, b))
#endif

// absolute difference
#ifndef vec_absd
#   define vec_absd(a, b) vec_sub(vec_max(a, b), vec_min(a, b))
#endif

/*
 * Implement vec_unpacklu and vec_unpackhu
 * since vec_unpackl, vec_unpackh only support signed integers
**/
#define VSX_IMPL_UNPACKU(rt, rg, zero)      \
VSX_FINLINE(rt) vec_unpacklu(const rg& a)   \
{ return (rt)(vec_mergel(a, zero)); }       \
VSX_FINLINE(rt) vec_unpackhu(const rg& a)   \
{ return (rt)(vec_mergeh(a, zero));  }

VSX_IMPL_UNPACKU(vec_ushort8, vec_uchar16, vec_uchar16_z)
VSX_IMPL_UNPACKU(vec_uint4,   vec_ushort8, vec_ushort8_z)
VSX_IMPL_UNPACKU(vec_udword2, vec_uint4,   vec_uint4_z)

/*
 * Implement vec_mergesqe and vec_mergesqo
 * Merges the sequence values of even and odd elements of two vectors
*/
#define VSX_IMPL_PERM(rt, fnm, ...)            \
VSX_FINLINE(rt) fnm(const rt& a, const rt& b)  \
{ static const vec_uchar16 perm = {__VA_ARGS__}; return vec_perm(a, b, perm); }

// 16
#define perm16_mergesqe 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
#define perm16_mergesqo 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
VSX_IMPL_PERM(vec_uchar16, vec_mergesqe, perm16_mergesqe)
VSX_IMPL_PERM(vec_uchar16, vec_mergesqo, perm16_mergesqo)
VSX_IMPL_PERM(vec_char16,  vec_mergesqe, perm16_mergesqe)
VSX_IMPL_PERM(vec_char16,  vec_mergesqo, perm16_mergesqo)
// 8
#define perm8_mergesqe 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29
#define perm8_mergesqo 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31
VSX_IMPL_PERM(vec_ushort8, vec_mergesqe, perm8_mergesqe)
VSX_IMPL_PERM(vec_ushort8, vec_mergesqo, perm8_mergesqo)
VSX_IMPL_PERM(vec_short8,  vec_mergesqe, perm8_mergesqe)
VSX_IMPL_PERM(vec_short8,  vec_mergesqo, perm8_mergesqo)
// 4
#define perm4_mergesqe 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
#define perm4_mergesqo 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
VSX_IMPL_PERM(vec_uint4,  vec_mergesqe, perm4_mergesqe)
VSX_IMPL_PERM(vec_uint4,  vec_mergesqo, perm4_mergesqo)
VSX_IMPL_PERM(vec_int4,   vec_mergesqe, perm4_mergesqe)
VSX_IMPL_PERM(vec_int4,   vec_mergesqo, perm4_mergesqo)
VSX_IMPL_PERM(vec_float4, vec_mergesqe, perm4_mergesqe)
VSX_IMPL_PERM(vec_float4, vec_mergesqo, perm4_mergesqo)
// 2
VSX_REDIRECT_2RG(vec_double2, vec_double2, vec_mergesqe, vec_mergeh)
VSX_REDIRECT_2RG(vec_double2, vec_double2, vec_mergesqo, vec_mergel)
VSX_REDIRECT_2RG(vec_dword2,  vec_dword2,  vec_mergesqe, vec_mergeh)
VSX_REDIRECT_2RG(vec_dword2,  vec_dword2,  vec_mergesqo, vec_mergel)
VSX_REDIRECT_2RG(vec_udword2, vec_udword2, vec_mergesqe, vec_mergeh)
VSX_REDIRECT_2RG(vec_udword2, vec_udword2, vec_mergesqo, vec_mergel)

/*
 * Implement vec_mergesqh and vec_mergesql
 * Merges the sequence most and least significant halves of two vectors
*/
#define VSX_IMPL_MERGESQHL(Tvec)                                    \
VSX_FINLINE(Tvec) vec_mergesqh(const Tvec& a, const Tvec& b)        \
{ return (Tvec)vec_mergeh(vec_udword2_c(a), vec_udword2_c(b)); }    \
VSX_FINLINE(Tvec) vec_mergesql(const Tvec& a, const Tvec& b)        \
{ return (Tvec)vec_mergel(vec_udword2_c(a), vec_udword2_c(b)); }
VSX_IMPL_MERGESQHL(vec_uchar16)
VSX_IMPL_MERGESQHL(vec_char16)
VSX_IMPL_MERGESQHL(vec_ushort8)
VSX_IMPL_MERGESQHL(vec_short8)
VSX_IMPL_MERGESQHL(vec_uint4)
VSX_IMPL_MERGESQHL(vec_int4)
VSX_IMPL_MERGESQHL(vec_float4)
VSX_REDIRECT_2RG(vec_udword2, vec_udword2, vec_mergesqh, vec_mergeh)
VSX_REDIRECT_2RG(vec_udword2, vec_udword2, vec_mergesql, vec_mergel)
VSX_REDIRECT_2RG(vec_dword2,  vec_dword2,  vec_mergesqh, vec_mergeh)
VSX_REDIRECT_2RG(vec_dword2,  vec_dword2,  vec_mergesql, vec_mergel)
VSX_REDIRECT_2RG(vec_double2, vec_double2, vec_mergesqh, vec_mergeh)
VSX_REDIRECT_2RG(vec_double2, vec_double2, vec_mergesql, vec_mergel)


// 2 and 4 channels interleave for all types except 2 lanes
#define VSX_IMPL_ST_INTERLEAVE(Tp, Tvec)                                    \
VSX_FINLINE(void) vec_st_interleave(const Tvec& a, const Tvec& b, Tp* ptr)  \
{                                                                           \
    vsx_stf(vec_mergeh(a, b), 0, ptr);                                      \
    vsx_stf(vec_mergel(a, b), 16, ptr);                                     \
}                                                                           \
VSX_FINLINE(void) vec_st_interleave(const Tvec& a, const Tvec& b,           \
                                     const Tvec& c, const Tvec& d, Tp* ptr) \
{                                                                           \
    Tvec ac = vec_mergeh(a, c);                                             \
    Tvec bd = vec_mergeh(b, d);                                             \
    vsx_stf(vec_mergeh(ac, bd), 0, ptr);                                    \
    vsx_stf(vec_mergel(ac, bd), 16, ptr);                                   \
    ac = vec_mergel(a, c);                                                  \
    bd = vec_mergel(b, d);                                                  \
    vsx_stf(vec_mergeh(ac, bd), 32, ptr);                                   \
    vsx_stf(vec_mergel(ac, bd), 48, ptr);                                   \
}
VSX_IMPL_ST_INTERLEAVE(uchar,  vec_uchar16)
VSX_IMPL_ST_INTERLEAVE(schar,  vec_char16)
VSX_IMPL_ST_INTERLEAVE(ushort, vec_ushort8)
VSX_IMPL_ST_INTERLEAVE(short,  vec_short8)
VSX_IMPL_ST_INTERLEAVE(uint,   vec_uint4)
VSX_IMPL_ST_INTERLEAVE(int,    vec_int4)
VSX_IMPL_ST_INTERLEAVE(float,  vec_float4)

// 2 and 4 channels deinterleave for 16 lanes
#define VSX_IMPL_ST_DINTERLEAVE_8(Tp, Tvec)                                 \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b)      \
{                                                                           \
    Tvec v0 = vsx_ld(0, ptr);                                               \
    Tvec v1 = vsx_ld(16, ptr);                                              \
    a = vec_mergesqe(v0, v1);                                               \
    b = vec_mergesqo(v0, v1);                                               \
}                                                                           \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b,      \
                                       Tvec& c, Tvec& d)                    \
{                                                                           \
    Tvec v0 = vsx_ld(0, ptr);                                               \
    Tvec v1 = vsx_ld(16, ptr);                                              \
    Tvec v2 = vsx_ld(32, ptr);                                              \
    Tvec v3 = vsx_ld(48, ptr);                                              \
    Tvec m0 = vec_mergesqe(v0, v1);                                         \
    Tvec m1 = vec_mergesqe(v2, v3);                                         \
    a = vec_mergesqe(m0, m1);                                               \
    c = vec_mergesqo(m0, m1);                                               \
    m0 = vec_mergesqo(v0, v1);                                              \
    m1 = vec_mergesqo(v2, v3);                                              \
    b = vec_mergesqe(m0, m1);                                               \
    d = vec_mergesqo(m0, m1);                                               \
}
VSX_IMPL_ST_DINTERLEAVE_8(uchar, vec_uchar16)
VSX_IMPL_ST_DINTERLEAVE_8(schar, vec_char16)

// 2 and 4 channels deinterleave for 8 lanes
#define VSX_IMPL_ST_DINTERLEAVE_16(Tp, Tvec)                                \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b)      \
{                                                                           \
    Tvec v0 = vsx_ld(0, ptr);                                               \
    Tvec v1 = vsx_ld(8, ptr);                                               \
    a = vec_mergesqe(v0, v1);                                               \
    b = vec_mergesqo(v0, v1);                                               \
}                                                                           \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b,      \
                                       Tvec& c, Tvec& d)                    \
{                                                                           \
    Tvec v0 = vsx_ld(0, ptr);                                               \
    Tvec v1 = vsx_ld(8, ptr);                                               \
    Tvec m0 = vec_mergeh(v0, v1);                                           \
    Tvec m1 = vec_mergel(v0, v1);                                           \
    Tvec ab0 = vec_mergeh(m0, m1);                                          \
    Tvec cd0 = vec_mergel(m0, m1);                                          \
    v0 = vsx_ld(16, ptr);                                                   \
    v1 = vsx_ld(24, ptr);                                                   \
    m0 = vec_mergeh(v0, v1);                                                \
    m1 = vec_mergel(v0, v1);                                                \
    Tvec ab1 = vec_mergeh(m0, m1);                                          \
    Tvec cd1 = vec_mergel(m0, m1);                                          \
    a = vec_mergesqh(ab0, ab1);                                             \
    b = vec_mergesql(ab0, ab1);                                             \
    c = vec_mergesqh(cd0, cd1);                                             \
    d = vec_mergesql(cd0, cd1);                                             \
}
VSX_IMPL_ST_DINTERLEAVE_16(ushort, vec_ushort8)
VSX_IMPL_ST_DINTERLEAVE_16(short,  vec_short8)

// 2 and 4 channels deinterleave for 4 lanes
#define VSX_IMPL_ST_DINTERLEAVE_32(Tp, Tvec)                                \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b)      \
{                                                                           \
    a = vsx_ld(0, ptr);                                                     \
    b = vsx_ld(4, ptr);                                                     \
    Tvec m0 = vec_mergeh(a, b);                                             \
    Tvec m1 = vec_mergel(a, b);                                             \
    a = vec_mergeh(m0, m1);                                                 \
    b = vec_mergel(m0, m1);                                                 \
}                                                                           \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b,      \
                                       Tvec& c, Tvec& d)                    \
{                                                                           \
    Tvec v0 = vsx_ld(0, ptr);                                               \
    Tvec v1 = vsx_ld(4, ptr);                                               \
    Tvec v2 = vsx_ld(8, ptr);                                               \
    Tvec v3 = vsx_ld(12, ptr);                                              \
    Tvec m0 = vec_mergeh(v0, v2);                                           \
    Tvec m1 = vec_mergeh(v1, v3);                                           \
    a = vec_mergeh(m0, m1);                                                 \
    b = vec_mergel(m0, m1);                                                 \
    m0 = vec_mergel(v0, v2);                                                \
    m1 = vec_mergel(v1, v3);                                                \
    c = vec_mergeh(m0, m1);                                                 \
    d = vec_mergel(m0, m1);                                                 \
}
VSX_IMPL_ST_DINTERLEAVE_32(uint,  vec_uint4)
VSX_IMPL_ST_DINTERLEAVE_32(int,   vec_int4)
VSX_IMPL_ST_DINTERLEAVE_32(float, vec_float4)

// 2 and 4 channels interleave and deinterleave for 2 lanes
#define VSX_IMPL_ST_D_INTERLEAVE_64(Tp, Tvec, ld_func, st_func)             \
VSX_FINLINE(void) vec_st_interleave(const Tvec& a, const Tvec& b, Tp* ptr)  \
{                                                                           \
    st_func(vec_mergeh(a, b), 0, ptr);                                      \
    st_func(vec_mergel(a, b), 2, ptr);                                      \
}                                                                           \
VSX_FINLINE(void) vec_st_interleave(const Tvec& a, const Tvec& b,           \
                                     const Tvec& c, const Tvec& d, Tp* ptr) \
{                                                                           \
    st_func(vec_mergeh(a, b), 0, ptr);                                      \
    st_func(vec_mergeh(c, d), 2, ptr);                                      \
    st_func(vec_mergel(a, b), 4, ptr);                                      \
    st_func(vec_mergel(c, d), 6, ptr);                                      \
}                                                                           \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b)      \
{                                                                           \
    Tvec m0 = ld_func(0, ptr);                                              \
    Tvec m1 = ld_func(2, ptr);                                              \
    a = vec_mergeh(m0, m1);                                                 \
    b = vec_mergel(m0, m1);                                                 \
}                                                                           \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b,      \
                                       Tvec& c, Tvec& d)                    \
{                                                                           \
    Tvec v0 = ld_func(0, ptr);                                              \
    Tvec v1 = ld_func(2, ptr);                                              \
    Tvec v2 = ld_func(4, ptr);                                              \
    Tvec v3 = ld_func(6, ptr);                                              \
    a = vec_mergeh(v0, v2);                                                 \
    b = vec_mergel(v0, v2);                                                 \
    c = vec_mergeh(v1, v3);                                                 \
    d = vec_mergel(v1, v3);                                                 \
}
VSX_IMPL_ST_D_INTERLEAVE_64(int64,  vec_dword2,  vsx_ld2, vsx_st2)
VSX_IMPL_ST_D_INTERLEAVE_64(uint64, vec_udword2, vsx_ld2, vsx_st2)
VSX_IMPL_ST_D_INTERLEAVE_64(double, vec_double2, vsx_ld,  vsx_st)

/* 3 channels */
#define VSX_IMPL_ST_INTERLEAVE_3CH_16(Tp, Tvec)                                                   \
VSX_FINLINE(void) vec_st_interleave(const Tvec& a, const Tvec& b,                                 \
                                     const Tvec& c, Tp* ptr)                                      \
{                                                                                                 \
    static const vec_uchar16 a12 = {0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5};         \
    static const vec_uchar16 a123 = {0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15};    \
    vsx_st(vec_perm(vec_perm(a, b, a12), c, a123), 0, ptr);                                       \
    static const vec_uchar16 b12 = {21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26};       \
    static const vec_uchar16 b123 = {0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15};    \
    vsx_st(vec_perm(vec_perm(a, b, b12), c, b123), 16, ptr);                                      \
    static const vec_uchar16 c12 = {0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0};    \
    static const vec_uchar16 c123 = {26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31};   \
    vsx_st(vec_perm(vec_perm(a, b, c12), c, c123), 32, ptr);                                      \
}                                                                                                 \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b, Tvec& c)                   \
{                                                                                                 \
    Tvec v1 = vsx_ld(0, ptr);                                                                     \
    Tvec v2 = vsx_ld(16, ptr);                                                                    \
    Tvec v3 = vsx_ld(32, ptr);                                                                    \
    static const vec_uchar16 a12_perm = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0};  \
    static const vec_uchar16 a123_perm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29};  \
    a = vec_perm(vec_perm(v1, v2, a12_perm), v3, a123_perm);                                      \
    static const vec_uchar16 b12_perm = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0}; \
    static const vec_uchar16 b123_perm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30};  \
    b = vec_perm(vec_perm(v1, v2, b12_perm), v3, b123_perm);                                      \
    static const vec_uchar16 c12_perm = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0};  \
    static const vec_uchar16 c123_perm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31};  \
    c = vec_perm(vec_perm(v1, v2, c12_perm), v3, c123_perm);                                      \
}
VSX_IMPL_ST_INTERLEAVE_3CH_16(uchar, vec_uchar16)
VSX_IMPL_ST_INTERLEAVE_3CH_16(schar, vec_char16)

#define VSX_IMPL_ST_INTERLEAVE_3CH_8(Tp, Tvec)                                                    \
VSX_FINLINE(void) vec_st_interleave(const Tvec& a, const Tvec& b,                                 \
                                     const Tvec& c, Tp* ptr)                                      \
{                                                                                                 \
    static const vec_uchar16 a12 = {0, 1, 16, 17, 0, 0, 2, 3, 18, 19, 0, 0, 4, 5, 20, 21};        \
    static const vec_uchar16 a123 = {0, 1, 2, 3, 16, 17, 6, 7, 8, 9, 18, 19, 12, 13, 14, 15};     \
    vsx_st(vec_perm(vec_perm(a, b, a12), c, a123), 0, ptr);                                       \
    static const vec_uchar16 b12 = {0, 0, 6, 7, 22, 23, 0, 0, 8, 9, 24, 25, 0, 0, 10, 11};        \
    static const vec_uchar16 b123 = {20, 21, 2, 3, 4, 5, 22, 23, 8, 9, 10, 11, 24, 25, 14, 15};   \
    vsx_st(vec_perm(vec_perm(a, b, b12), c, b123), 8, ptr);                                       \
    static const vec_uchar16 c12 = {26, 27, 0, 0, 12, 13, 28, 29, 0, 0, 14, 15, 30, 31, 0, 0};    \
    static const vec_uchar16 c123 = {0, 1, 26, 27, 4, 5, 6, 7, 28, 29, 10, 11, 12, 13, 30, 31};   \
    vsx_st(vec_perm(vec_perm(a, b, c12), c, c123), 16, ptr);                                      \
}                                                                                                 \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b, Tvec& c)                   \
{                                                                                                 \
    Tvec v1 = vsx_ld(0, ptr);                                                                     \
    Tvec v2 = vsx_ld(8, ptr);                                                                     \
    Tvec v3 = vsx_ld(16, ptr);                                                                    \
    static const vec_uchar16 a12_perm = {0, 1, 6, 7, 12, 13, 18, 19, 24, 25, 30, 31, 0, 0, 0, 0}; \
    static const vec_uchar16 a123_perm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 26, 27};  \
    a = vec_perm(vec_perm(v1, v2, a12_perm), v3, a123_perm);                                      \
    static const vec_uchar16 b12_perm = {2, 3, 8, 9, 14, 15, 20, 21, 26, 27, 0, 0, 0, 0, 0, 0};   \
    static const vec_uchar16 b123_perm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 22, 23, 28, 29};  \
    b = vec_perm(vec_perm(v1, v2, b12_perm), v3, b123_perm);                                      \
    static const vec_uchar16 c12_perm = {4, 5, 10, 11, 16, 17, 22, 23, 28, 29, 0, 0, 0, 0, 0, 0}; \
    static const vec_uchar16 c123_perm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19, 24, 25, 30, 31};  \
    c = vec_perm(vec_perm(v1, v2, c12_perm), v3, c123_perm);                                      \
}
VSX_IMPL_ST_INTERLEAVE_3CH_8(ushort, vec_ushort8)
VSX_IMPL_ST_INTERLEAVE_3CH_8(short,  vec_short8)

#define VSX_IMPL_ST_INTERLEAVE_3CH_4(Tp, Tvec)                                                     \
VSX_FINLINE(void) vec_st_interleave(const Tvec& a, const Tvec& b,                                  \
                                     const Tvec& c, Tp* ptr)                                       \
{                                                                                                  \
    Tvec hbc = vec_mergeh(b, c);                                                                   \
    static const vec_uchar16 ahbc = {0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 4, 5, 6, 7};      \
    vsx_st(vec_perm(a, hbc, ahbc), 0, ptr);                                                        \
    Tvec lab = vec_mergel(a, b);                                                                   \
    vsx_st(vec_sld(lab, hbc, 8), 4, ptr);                                                          \
    static const vec_uchar16 clab = {8, 9, 10, 11, 24, 25, 26, 27, 28, 29, 30, 31, 12, 13, 14, 15};\
    vsx_st(vec_perm(c, lab, clab), 8, ptr);                                                        \
}                                                                                                  \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a, Tvec& b, Tvec& c)                    \
{                                                                                                  \
    Tvec v1 = vsx_ld(0, ptr);                                                                      \
    Tvec v2 = vsx_ld(4, ptr);                                                                      \
    Tvec v3 = vsx_ld(8, ptr);                                                                      \
    static const vec_uchar16 flp = {0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 30, 31};   \
    a = vec_perm(v1, vec_sld(v3, v2, 8), flp);                                                     \
    static const vec_uchar16 flp2 = {28, 29, 30, 31, 0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19};  \
    b = vec_perm(v2, vec_sld(v1, v3, 8), flp2);                                                    \
    c = vec_perm(vec_sld(v2, v1, 8), v3, flp);                                                     \
}
VSX_IMPL_ST_INTERLEAVE_3CH_4(uint,  vec_uint4)
VSX_IMPL_ST_INTERLEAVE_3CH_4(int,   vec_int4)
VSX_IMPL_ST_INTERLEAVE_3CH_4(float, vec_float4)

#define VSX_IMPL_ST_INTERLEAVE_3CH_2(Tp, Tvec, ld_func, st_func)     \
VSX_FINLINE(void) vec_st_interleave(const Tvec& a, const Tvec& b,    \
                                     const Tvec& c, Tp* ptr)         \
{                                                                    \
    st_func(vec_mergeh(a, b), 0, ptr);                               \
    st_func(vec_permi(c, a, 1), 2, ptr);                             \
    st_func(vec_mergel(b, c), 4, ptr);                               \
}                                                                    \
VSX_FINLINE(void) vec_ld_deinterleave(const Tp* ptr, Tvec& a,        \
                                       Tvec& b, Tvec& c)             \
{                                                                    \
    Tvec v1 = ld_func(0, ptr);                                       \
    Tvec v2 = ld_func(2, ptr);                                       \
    Tvec v3 = ld_func(4, ptr);                                       \
    a = vec_permi(v1, v2, 1);                                        \
    b = vec_permi(v1, v3, 2);                                        \
    c = vec_permi(v2, v3, 1);                                        \
}
VSX_IMPL_ST_INTERLEAVE_3CH_2(int64,  vec_dword2,  vsx_ld2, vsx_st2)
VSX_IMPL_ST_INTERLEAVE_3CH_2(uint64, vec_udword2, vsx_ld2, vsx_st2)
VSX_IMPL_ST_INTERLEAVE_3CH_2(double, vec_double2, vsx_ld,  vsx_st)

#endif // CV_VSX

//! @}

#endif // OPENCV_HAL_VSX_UTILS_HPP
