// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core/hal/intrin.hpp"

//=========================================
// Declare & Define & Dispatch in one step
//=========================================

// ARITHM_DISPATCHING_ONLY defined by arithm dispatch file

#undef ARITHM_DECLARATIONS_ONLY
#ifdef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
    #define ARITHM_DECLARATIONS_ONLY
#endif

#undef ARITHM_DEFINITIONS_ONLY
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && !defined(ARITHM_DISPATCHING_ONLY)
    #define ARITHM_DEFINITIONS_ONLY
#endif

///////////////////////////////////////////////////////////////////////////

namespace cv { namespace hal {

#ifndef ARITHM_DISPATCHING_ONLY
    CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
#endif

#if (defined ARITHM_DECLARATIONS_ONLY) || (defined ARITHM_DEFINITIONS_ONLY)

#undef DECLARE_SIMPLE_BINARY_OP
#define DECLARE_SIMPLE_BINARY_OP(opname, type) \
    void opname(const type* src1, size_t step1, const type* src2, size_t step2, \
                type* dst, size_t step, int width, int height)

#undef DECLARE_SIMPLE_BINARY_OP_ALLTYPES
#define DECLARE_SIMPLE_BINARY_OP_ALLTYPES(opname) \
    DECLARE_SIMPLE_BINARY_OP(opname##8u, uchar); \
    DECLARE_SIMPLE_BINARY_OP(opname##8s, schar); \
    DECLARE_SIMPLE_BINARY_OP(opname##16u, ushort); \
    DECLARE_SIMPLE_BINARY_OP(opname##16s, short); \
    DECLARE_SIMPLE_BINARY_OP(opname##32u, unsigned); \
    DECLARE_SIMPLE_BINARY_OP(opname##32s, int); \
    DECLARE_SIMPLE_BINARY_OP(opname##64u, uint64); \
    DECLARE_SIMPLE_BINARY_OP(opname##64s, int64); \
    DECLARE_SIMPLE_BINARY_OP(opname##16f, hfloat); \
    DECLARE_SIMPLE_BINARY_OP(opname##16bf, bfloat); \
    DECLARE_SIMPLE_BINARY_OP(opname##32f, float); \
    DECLARE_SIMPLE_BINARY_OP(opname##64f, double)

DECLARE_SIMPLE_BINARY_OP_ALLTYPES(add);
DECLARE_SIMPLE_BINARY_OP_ALLTYPES(sub);
DECLARE_SIMPLE_BINARY_OP_ALLTYPES(max);
DECLARE_SIMPLE_BINARY_OP_ALLTYPES(min);
DECLARE_SIMPLE_BINARY_OP_ALLTYPES(absdiff);

void and8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);
void or8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);
void xor8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);
void not8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);

#undef DECLARE_CMP_OP
#define DECLARE_CMP_OP(opname, type) \
    void opname(const type* src1, size_t step1, const type* src2, size_t step2, \
                uchar* dst, size_t step, int width, int height, int cmpop)

DECLARE_CMP_OP(cmp8u, uchar);
DECLARE_CMP_OP(cmp8s, schar);
DECLARE_CMP_OP(cmp16u, ushort);
DECLARE_CMP_OP(cmp16s, short);
DECLARE_CMP_OP(cmp32u, unsigned);
DECLARE_CMP_OP(cmp32s, int);
DECLARE_CMP_OP(cmp64u, uint64);
DECLARE_CMP_OP(cmp64s, int64);
DECLARE_CMP_OP(cmp16f, hfloat);
DECLARE_CMP_OP(cmp16bf, bfloat);
DECLARE_CMP_OP(cmp32f, float);
DECLARE_CMP_OP(cmp64f, double);

#undef DECLARE_SCALED_BINARY_OP
#define DECLARE_SCALED_BINARY_OP(opname, type, scale_arg) \
    void opname(const type* src1, size_t step1, const type* src2, size_t step2, \
                type* dst, size_t step, int width, int height, scale_arg)

#undef DECLARE_SCALED_BINARY_OP_ALLTYPES
#define DECLARE_SCALED_BINARY_OP_ALLTYPES(opname, scale_arg) \
    DECLARE_SCALED_BINARY_OP(opname##8u, uchar, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##8s, schar, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##16u, ushort, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##16s, short, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##32u, unsigned, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##32s, int, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##64u, uint64, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##64s, int64, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##16f, hfloat, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##16bf, bfloat, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##32f, float, scale_arg); \
    DECLARE_SCALED_BINARY_OP(opname##64f, double, scale_arg)

DECLARE_SCALED_BINARY_OP_ALLTYPES(mul, double);
DECLARE_SCALED_BINARY_OP_ALLTYPES(div, double);
DECLARE_SCALED_BINARY_OP_ALLTYPES(recip, double);
DECLARE_SCALED_BINARY_OP_ALLTYPES(addWeighted, double weights[3]);

#endif

#ifdef ARITHM_DEFINITIONS_ONLY

#if (CV_SIMD || CV_SIMD_SCALABLE)
#define SIMD_ONLY(expr) expr
#else
#define SIMD_ONLY(expr)
#endif

//=======================================
// Arithmetic and logical operations
// +, -, *, /, &, |, ^, ~, abs ...
//=======================================

///////////////////////////// Operations //////////////////////////////////

#undef DEFINE_SIMPLE_BINARY_OP
#undef DEFINE_SIMPLE_BINARY_OP_F16
#undef DEFINE_SIMPLE_BINARY_OP_NOSIMD

#define DEFINE_SIMPLE_BINARY_OP(opname, T1, Tvec, scalar_op, vec_op) \
void opname(const T1* src1, size_t step1, \
            const T1* src2, size_t step2, \
            T1* dst, size_t step, \
            int width, int height) \
{ \
    CV_INSTRUMENT_REGION(); \
    SIMD_ONLY(int simd_width = VTraits<Tvec>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step  /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == src1) | (dst == src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            vx_store(dst + x, vec_op(vx_load(src1 + x), vx_load(src2 + x))); \
        }) \
        for (; x < width; x++) \
            dst[x] = saturate_cast<T1>(scalar_op(src1[x], src2[x])); \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

#define DEFINE_SIMPLE_BINARY_OP_16F(opname, T1, scalar_op, vec_op) \
void opname(const T1* src1, size_t step1, \
            const T1* src2, size_t step2, \
            T1* dst, size_t step, \
            int width, int height) \
{ \
    CV_INSTRUMENT_REGION(); \
    SIMD_ONLY(int simd_width = VTraits<v_float32>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step  /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == src1) | (dst == src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            v_pack_store(dst + x, vec_op(vx_load_expand(src1 + x), vx_load_expand(src2 + x))); \
        }) \
        for (; x < width; x++) \
            dst[x] = T1(scalar_op((float)src1[x], (float)src2[x])); \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

#define DEFINE_SIMPLE_BINARY_OP_NOSIMD(opname, T1, worktype, scalar_op) \
void opname(const T1* src1, size_t step1, \
            const T1* src2, size_t step2, \
            T1* dst, size_t step, \
            int width, int height) \
{ \
    CV_INSTRUMENT_REGION(); \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step  /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        for (int x = 0; x < width; x++) \
            dst[x] = saturate_cast<T1>(scalar_op((worktype)src1[x], (worktype)src2[x])); \
    } \
}

#undef scalar_add
#define scalar_add(x, y) ((x) + (y))
#undef scalar_sub
#define scalar_sub(x, y) ((x) - (y))
#undef scalar_sub_u64
#define scalar_sub_u64(x, y) ((x) <= (y) ? 0 : (x) - (y))

#undef DEFINE_SIMPLE_BINARY_OP_64F
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
#define DEFINE_SIMPLE_BINARY_OP_64F(opname, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP(opname, double, v_float64, scalar_op, vec_op)
#else
#define DEFINE_SIMPLE_BINARY_OP_64F(opname, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP_NOSIMD(opname, double, double, scalar_op)
#endif

#undef DEFINE_SIMPLE_BINARY_OP_ALLTYPES
#define DEFINE_SIMPLE_BINARY_OP_ALLTYPES(opname, scalar_op, scalar_op_u64, vec_op) \
    DEFINE_SIMPLE_BINARY_OP(opname##8u, uchar, v_uint8, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP(opname##8s, schar, v_int8, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP(opname##16u, ushort, v_uint16, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP(opname##16s, short, v_int16, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP_NOSIMD(opname##32u, unsigned, int64, scalar_op) \
    DEFINE_SIMPLE_BINARY_OP(opname##32s, int, v_int32, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP_NOSIMD(opname##64u, uint64, uint64, scalar_op_u64) \
    DEFINE_SIMPLE_BINARY_OP_NOSIMD(opname##64s, int64, int64, scalar_op) \
    DEFINE_SIMPLE_BINARY_OP_16F(opname##16f, hfloat, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP_16F(opname##16bf, bfloat, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP(opname##32f, float, v_float32, scalar_op, vec_op) \
    DEFINE_SIMPLE_BINARY_OP_64F(opname##64f, scalar_op, vec_op)

DEFINE_SIMPLE_BINARY_OP_ALLTYPES(add, scalar_add, scalar_add, v_add)
DEFINE_SIMPLE_BINARY_OP_ALLTYPES(sub, scalar_sub, scalar_sub_u64, v_sub)
DEFINE_SIMPLE_BINARY_OP_ALLTYPES(max, std::max, std::max, v_max)
DEFINE_SIMPLE_BINARY_OP_ALLTYPES(min, std::min, std::min, v_min)

#undef scalar_absdiff
#define scalar_absdiff(x, y) std::abs((x) - (y))
#define scalar_absdiffu(x, y) (std::max((x), (y)) - std::min((x), (y)))

DEFINE_SIMPLE_BINARY_OP(absdiff8u, uchar, v_uint8, scalar_absdiff, v_absdiff)
DEFINE_SIMPLE_BINARY_OP(absdiff8s, schar, v_int8, scalar_absdiff, v_absdiffs)
DEFINE_SIMPLE_BINARY_OP(absdiff16u, ushort, v_uint16, scalar_absdiff, v_absdiff)
DEFINE_SIMPLE_BINARY_OP(absdiff16s, short, v_int16, scalar_absdiff, v_absdiffs)
DEFINE_SIMPLE_BINARY_OP_NOSIMD(absdiff32u, unsigned, unsigned, scalar_absdiffu)
DEFINE_SIMPLE_BINARY_OP_NOSIMD(absdiff32s, int, int, scalar_absdiff)
DEFINE_SIMPLE_BINARY_OP_NOSIMD(absdiff64u, uint64, uint64, scalar_absdiffu)
DEFINE_SIMPLE_BINARY_OP_NOSIMD(absdiff64s, int64, int64, scalar_absdiff)
DEFINE_SIMPLE_BINARY_OP_16F(absdiff16f, hfloat, scalar_absdiff, v_absdiff)
DEFINE_SIMPLE_BINARY_OP_16F(absdiff16bf, bfloat, scalar_absdiff, v_absdiff)
DEFINE_SIMPLE_BINARY_OP(absdiff32f, float, v_float32, scalar_absdiff, v_absdiff)
DEFINE_SIMPLE_BINARY_OP_64F(absdiff64f, scalar_absdiff, v_absdiff)

#undef DEFINE_BINARY_LOGIC_OP
#define DEFINE_BINARY_LOGIC_OP(opname, scalar_op, vec_op) \
void opname(const uchar* src1, size_t step1, \
            const uchar* src2, size_t step2, \
            uchar* dst, size_t step, \
            int width, int height) \
{ \
    CV_INSTRUMENT_REGION(); \
    int simd_width = VTraits<v_uint8>::vlanes(); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == src1) | (dst == src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            vx_store(dst + x, vec_op(vx_load(src1 + x), vx_load(src2 + x))); \
        } \
        for (; x < width; x++) \
            dst[x] = (uchar)(src1[x] scalar_op src2[x]); \
    } \
    vx_cleanup(); \
}

DEFINE_BINARY_LOGIC_OP(and8u, &, v_and)
DEFINE_BINARY_LOGIC_OP(or8u, |, v_or)
DEFINE_BINARY_LOGIC_OP(xor8u, ^, v_xor)

void not8u(const uchar* src1, size_t step1,
           const uchar*, size_t,
           uchar* dst, size_t step,
           int width, int height)
{
    CV_INSTRUMENT_REGION();
    int simd_width = VTraits<v_uint8>::vlanes();
    for (; --height >= 0; src1 += step1, dst += step) {
        int x = 0;
        for (; x < width; x += simd_width)
        {
            if (x + simd_width > width) {
                if (((x == 0) | (dst == src1)) != 0)
                    break;
                x = width - simd_width;
            }
            vx_store(dst + x, v_not(vx_load(src1 + x)));
        }
        for (; x < width; x++)
            dst[x] = (uchar)(~src1[x]);
    }
    vx_cleanup();
}

//=======================================
// Compare
//=======================================

#undef DEFINE_CMP_OP_8
#undef DEFINE_CMP_OP_16
#undef DEFINE_CMP_OP_16F
#undef DEFINE_CMP_OP_32
#undef DEFINE_CMP_OP_64

// comparison for 8-bit types
#define DEFINE_CMP_OP_8(opname, T1, Tvec, scalar_op, vec_op) \
static void opname(const T1* src1, size_t step1, \
                   const T1* src2, size_t step2, \
                   uchar* dst, size_t step, \
                   int width, int height) \
{ \
    SIMD_ONLY(int simd_width = VTraits<Tvec>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == (uchar*)src1) | (dst == (uchar*)src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            vx_store((T1*)(dst + x), vec_op(vx_load(src1 + x), vx_load(src2 + x))); \
        }) \
        for (; x < width; x++) \
            dst[x] = (uchar)-(int)(src1[x] scalar_op src2[x]); \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

// comparison for 16-bit integer types
#define DEFINE_CMP_OP_16(opname, T1, Tvec, scalar_op, vec_op) \
static void opname(const T1* src1, size_t step1, \
                   const T1* src2, size_t step2, \
                   uchar* dst, size_t step, \
                   int width, int height) \
{ \
    SIMD_ONLY(int simd_width = VTraits<Tvec>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (x == 0) \
                    break; \
                x = width - simd_width; \
            } \
            v_pack_store((schar*)(dst + x), v_reinterpret_as_s16(vec_op(vx_load(src1 + x), vx_load(src2 + x)))); \
        }) \
        for (; x < width; x++) \
            dst[x] = (uchar)-(int)(src1[x] scalar_op src2[x]); \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

// comparison for 16-bit floating-point types
#define DEFINE_CMP_OP_16F(opname, T1, scalar_op, vec_op) \
static void opname(const T1* src1, size_t step1, \
                   const T1* src2, size_t step2, \
                   uchar* dst, size_t step, \
                   int width, int height) \
{ \
    SIMD_ONLY(int simd_width = VTraits<v_float32>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width*2) \
        { \
            if (x + simd_width*2 > width) { \
                if (x == 0) \
                    break; \
                x = width - simd_width*2; \
            } \
            auto mask0 = v_reinterpret_as_s32(vec_op(vx_load_expand(src1 + x), \
                                                     vx_load_expand(src2 + x))); \
            auto mask1 = v_reinterpret_as_s32(vec_op(vx_load_expand(src1 + x + simd_width), \
                                                     vx_load_expand(src2 + x + simd_width))); \
            auto mask = v_pack(mask0, mask1); \
            v_pack_store((schar*)(dst + x), mask); \
        }) \
        for (; x < width; x++) \
            dst[x] = (uchar)-(int)((float)src1[x] scalar_op (float)src2[x]); \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

// comparison for 32-bit types
#define DEFINE_CMP_OP_32(opname, T1, Tvec, scalar_op, vec_op) \
static void opname(const T1* src1, size_t step1, \
                   const T1* src2, size_t step2, \
                   uchar* dst, size_t step, \
                   int width, int height) \
{ \
    SIMD_ONLY(int simd_width = VTraits<Tvec>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width*2) \
        { \
            if (x + simd_width*2 > width) { \
                if (x == 0) \
                    break; \
                x = width - simd_width*2; \
            } \
            auto mask0 = v_reinterpret_as_s32(vec_op(vx_load(src1 + x), \
                                                     vx_load(src2 + x))); \
            auto mask1 = v_reinterpret_as_s32(vec_op(vx_load(src1 + x + simd_width), \
                                                     vx_load(src2 + x + simd_width))); \
            auto mask = v_pack(mask0, mask1); \
            v_pack_store((schar*)(dst + x), mask); \
        }) \
        for (; x < width; x++) \
            dst[x] = (uchar)-(int)(src1[x] scalar_op src2[x]); \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

// comparison for 64-bit types; don't bother with SIMD here. Hope, compiler will do it
#define DEFINE_CMP_OP_64(opname, T1, scalar_op) \
static void opname(const T1* src1, size_t step1, \
                   const T1* src2, size_t step2, \
                   uchar* dst, size_t step, \
                   int width, int height) \
{ \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        for (int x = 0; x < width; x++) \
            dst[x] = (uchar)-(int)(src1[x] scalar_op src2[x]); \
    } \
}

#undef DEFINE_CMP_OP_ALLTYPES
#define DEFINE_CMP_OP_ALLTYPES(opname, scalar_op, vec_op) \
    DEFINE_CMP_OP_8(opname##8u, uchar, v_uint8, scalar_op, vec_op) \
    DEFINE_CMP_OP_8(opname##8s, schar, v_int8, scalar_op, vec_op) \
    DEFINE_CMP_OP_16(opname##16u, ushort, v_uint16, scalar_op, vec_op) \
    DEFINE_CMP_OP_16(opname##16s, short, v_int16, scalar_op, vec_op) \
    DEFINE_CMP_OP_32(opname##32u, unsigned, v_uint32, scalar_op, vec_op) \
    DEFINE_CMP_OP_32(opname##32s, int, v_int32, scalar_op, vec_op) \
    DEFINE_CMP_OP_64(opname##64u, uint64, scalar_op) \
    DEFINE_CMP_OP_64(opname##64s, int64, scalar_op) \
    DEFINE_CMP_OP_16F(opname##16f, hfloat, scalar_op, vec_op) \
    DEFINE_CMP_OP_16F(opname##16bf, bfloat, scalar_op, vec_op) \
    DEFINE_CMP_OP_32(opname##32f, float, v_float32, scalar_op, vec_op) \
    DEFINE_CMP_OP_64(opname##64f, double, scalar_op)

DEFINE_CMP_OP_ALLTYPES(cmpeq, ==, v_eq)
DEFINE_CMP_OP_ALLTYPES(cmpne, !=, v_ne)
DEFINE_CMP_OP_ALLTYPES(cmplt, <, v_lt)
DEFINE_CMP_OP_ALLTYPES(cmple, <=, v_le)

#undef DEFINE_CMP_OP
#define DEFINE_CMP_OP(suffix, type) \
void cmp##suffix(const type* src1, size_t step1, const type* src2, size_t step2, \
                 uchar* dst, size_t step, int width, int height, int cmpop) \
{ \
    CV_INSTRUMENT_REGION(); \
    switch(cmpop) \
    { \
    case CMP_LT: \
        cmplt##suffix(src1, step1, src2, step2, dst, step, width, height); \
        break; \
    case CMP_GT: \
        cmplt##suffix(src2, step2, src1, step1, dst, step, width, height); \
        break; \
    case CMP_LE: \
        cmple##suffix(src1, step1, src2, step2, dst, step, width, height); \
        break; \
    case CMP_GE: \
        cmple##suffix(src2, step2, src1, step1, dst, step, width, height); \
        break; \
    case CMP_EQ: \
        cmpeq##suffix(src1, step1, src2, step2, dst, step, width, height); \
        break; \
    default: \
        CV_Assert(cmpop == CMP_NE); \
        cmpne##suffix(src1, step1, src2, step2, dst, step, width, height); \
    } \
}

DEFINE_CMP_OP(8u, uchar)
DEFINE_CMP_OP(8s, schar)
DEFINE_CMP_OP(16u, ushort)
DEFINE_CMP_OP(16s, short)
DEFINE_CMP_OP(32u, unsigned)
DEFINE_CMP_OP(32s, int)
DEFINE_CMP_OP(64u, uint64)
DEFINE_CMP_OP(64s, int64)
DEFINE_CMP_OP(16f, hfloat)
DEFINE_CMP_OP(16bf, bfloat)
DEFINE_CMP_OP(32f, float)
DEFINE_CMP_OP(64f, double)

//=======================================
// Mul, Div, Recip, AddWeighted
//=======================================

#undef DEFINE_SCALED_OP_8
#undef DEFINE_SCALED_OP_16
#undef DEFINE_SCALED_OP_16F
#undef DEFINE_SCALED_OP_32
#undef DEFINE_SCALED_OP_64

#define DEFINE_SCALED_OP_8(opname, scale_arg, T1, Tvec, scalar_op, vec_op, init, pack_store_op, when_binary) \
void opname(const T1* src1, size_t step1, const T1* src2, size_t step2, \
            T1* dst, size_t step, int width, int height, scale_arg) \
{ \
    CV_INSTRUMENT_REGION(); \
    init(); \
    SIMD_ONLY(int simd_width = VTraits<Tvec>::vlanes()>>1;) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == src1) | (dst == src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            v_int16 i1 = v_reinterpret_as_s16(vx_load_expand(src1 + x)); \
            when_binary(v_int16 i2 = v_reinterpret_as_s16(vx_load_expand(src2 + x))); \
            v_float32 f1 = v_cvt_f32(v_expand_low(i1)); \
            when_binary(v_float32 f2 = v_cvt_f32(v_expand_low(i2))); \
            v_float32 g1 = vec_op(); \
            f1 = v_cvt_f32(v_expand_high(i1)); \
            when_binary(f2 = v_cvt_f32(v_expand_high(i2))); \
            v_float32 g2 = vec_op(); \
            i1 = v_pack(v_round(g1), v_round(g2)); \
            pack_store_op(dst + x, i1); \
        }) \
        for (; x < width; x++) { \
            float f1 = (float)src1[x]; \
            when_binary(float f2 = (float)src2[x]); \
            dst[x] = saturate_cast<T1>(scalar_op()); \
        } \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

#define DEFINE_SCALED_OP_16(opname, scale_arg, T1, Tvec, scalar_op, vec_op, init, pack_store_op, when_binary) \
void opname(const T1* src1, size_t step1, const T1* src2, size_t step2, \
            T1* dst, size_t step, int width, int height, scale_arg) \
{ \
    CV_INSTRUMENT_REGION(); \
    init() \
    SIMD_ONLY(int simd_width = VTraits<Tvec>::vlanes()>>1;) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == src1) | (dst == src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            v_int32 i1 = v_reinterpret_as_s32(vx_load_expand(src1 + x)); \
            when_binary(v_int32 i2 = v_reinterpret_as_s32(vx_load_expand(src2 + x))); \
            v_float32 f1 = v_cvt_f32(i1); \
            when_binary(v_float32 f2 = v_cvt_f32(i2)); \
            f1 = vec_op(); \
            i1 = v_round(f1); \
            pack_store_op(dst + x, i1); \
        }) \
        for (; x < width; x++) { \
            float f1 = (float)src1[x]; \
            when_binary(float f2 = (float)src2[x]); \
            dst[x] = saturate_cast<T1>(scalar_op()); \
        } \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

#define DEFINE_SCALED_OP_16F(opname, scale_arg, T1, scalar_op, vec_op, init, when_binary) \
void opname(const T1* src1, size_t step1, const T1* src2, size_t step2, \
            T1* dst, size_t step, int width, int height, scale_arg) \
{ \
    CV_INSTRUMENT_REGION(); \
    init() \
    SIMD_ONLY(int simd_width = VTraits<v_float32>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == src1) | (dst == src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            v_float32 f1 = vx_load_expand(src1 + x); \
            when_binary(v_float32 f2 = vx_load_expand(src2 + x)); \
            f1 = vec_op(); \
            v_pack_store(dst + x, f1); \
        }) \
        for (; x < width; x++) { \
            float f1 = (float)src1[x]; \
            when_binary(float f2 = (float)src2[x]); \
            dst[x] = saturate_cast<T1>(scalar_op()); \
        } \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

#define DEFINE_SCALED_OP_32(opname, scale_arg, T1, Tvec, scalar_op, vec_op, init, load_op, store_op, when_binary) \
void opname(const T1* src1, size_t step1, const T1* src2, size_t step2, \
            T1* dst, size_t step, int width, int height, scale_arg) \
{ \
    CV_INSTRUMENT_REGION(); \
    init() \
    SIMD_ONLY(int simd_width = VTraits<Tvec>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == src1) | (dst == src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            v_float32 f1 = load_op(src1 + x); \
            when_binary(v_float32 f2 = load_op(src2 + x)); \
            f1 = vec_op(); \
            store_op(dst + x, f1); \
        }) \
        for (; x < width; x++) { \
            float f1 = (float)src1[x]; \
            when_binary(float f2 = (float)src2[x]); \
            dst[x] = saturate_cast<T1>(scalar_op()); \
        } \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

#define DEFINE_SCALED_OP_64F_(opname, scale_arg, T1, Tvec, scalar_op, vec_op, init, when_binary) \
void opname(const T1* src1, size_t step1, const T1* src2, size_t step2, \
            T1* dst, size_t step, int width, int height, scale_arg) \
{ \
    CV_INSTRUMENT_REGION(); \
    init() \
    SIMD_ONLY(int simd_width = VTraits<Tvec>::vlanes();) \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        SIMD_ONLY(for (; x < width; x += simd_width) \
        { \
            if (x + simd_width > width) { \
                if (((x == 0) | (dst == src1) | (dst == src2)) != 0) \
                    break; \
                x = width - simd_width; \
            } \
            v_float64 f1 = vx_load(src1 + x); \
            when_binary(v_float64 f2 = vx_load(src2 + x)); \
            f1 = vec_op(); \
            v_store(dst + x, f1); \
        }) \
        for (; x < width; x++) { \
            double f1 = (double)src1[x]; \
            when_binary(double f2 = (double)src2[x]); \
            dst[x] = saturate_cast<T1>(scalar_op()); \
        } \
    } \
    SIMD_ONLY(vx_cleanup();) \
}

#define DEFINE_SCALED_OP_NOSIMD(opname, scale_arg, T1, worktype, scalar_op, init, when_binary) \
void opname(const T1* src1, size_t step1, const T1* src2, size_t step2, \
            T1* dst, size_t step, int width, int height, scale_arg) \
{ \
    CV_INSTRUMENT_REGION(); \
    init() \
    step1 /= sizeof(T1); \
    step2 /= sizeof(T1); \
    step /= sizeof(T1); \
    for (; --height >= 0; src1 += step1, src2 += step2, dst += step) { \
        int x = 0; \
        for (; x < width; x++) { \
            worktype f1 = (worktype)src1[x]; \
            when_binary(worktype f2 = (worktype)src2[x]); \
            dst[x] = saturate_cast<T1>(scalar_op()); \
        } \
    } \
}

#define init_muldiv_f32() \
    float sscale = (float)scale; \
    SIMD_ONLY(v_float32 vzero = vx_setzero_f32(); \
              v_float32 vscale = v_add(vx_setall_f32(sscale), vzero);)
#define init_addw_f32() \
    float sw1 = (float)weights[0]; \
    float sw2 = (float)weights[1]; \
    float sdelta = (float)weights[2];\
    SIMD_ONLY(v_float32 vw1 = vx_setall_f32(sw1); \
        v_float32 vw2 = vx_setall_f32(sw2); \
        v_float32 vdelta = vx_setall_f32(sdelta);)

#undef init_muldiv_nosimd_f32
#define init_muldiv_nosimd_f32() \
    float sscale = (float)scale;
#undef init_addw_nosimd_f32
#define init_addw_nosimd_f32() \
    float sw1 = (float)weights[0]; \
    float sw2 = (float)weights[1]; \
    float sdelta = (float)weights[2];

#undef init_muldiv_nosimd_f64
#undef init_addw_nosimd_f64
#define init_muldiv_nosimd_f64() \
    double sscale = scale;
#define init_addw_nosimd_f64() \
    double sw1 = weights[0]; \
    double sw2 = weights[1]; \
    double sdelta = weights[2];

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
#define DEFINE_SCALED_OP_64F(opname, scale_arg, scalar_op, vec_op, init, when_binary) \
    DEFINE_SCALED_OP_64F_(opname, scale_arg, double, v_float64, scalar_op, vec_op, init, when_binary)
#define init_muldiv_f64() \
    double sscale = (double)scale; \
    SIMD_ONLY(v_float64 vzero = vx_setzero_f64(); \
              v_float64 vscale = v_add(vx_setall_f64(sscale), vzero);)
#define init_addw_f64() \
    double sw1 = weights[0]; \
    double sw2 = weights[1]; \
    double sdelta = weights[2];\
    SIMD_ONLY(v_float64 vw1 = vx_setall_f64(sw1); \
        v_float64 vw2 = vx_setall_f64(sw2); \
        v_float64 vdelta = vx_setall_f64(sdelta);)
#else
#define DEFINE_SCALED_OP_64F(opname, scale_arg, scalar_op, vec_op, init, when_binary) \
    DEFINE_SCALED_OP_NOSIMD(opname, scale_arg, double, double, scalar_op, init, when_binary)
#define init_muldiv_f64() init_muldiv_nosimd_f64()
#define init_addw_f64() init_addw_nosimd_f64()
#endif

#undef scalar_mul
#undef vec_mul
#undef iscalar_div
#undef ivec_div
#undef fscalar_div
#undef fvec_div
#undef scalar_addw
#undef vec_addw
#define scalar_mul() ((f1)*(f2)*sscale)
#define vec_mul() v_mul(v_mul((f1), vscale), (f2))
#define iscalar_div() ((f2)!=0? (f1)*sscale/(f2) : 0)
#define ivec_div() v_select(v_eq((f2), vzero), vzero, v_div(v_mul((f1), vscale), (f2)))
#define fscalar_div() ((f1)*sscale/(f2))
#define fvec_div() v_div(v_mul((f1), vscale), (f2))
#define iscalar_recip() ((f1)!=0? sscale/(f1) : 0)
#define ivec_recip() v_select(v_eq((f1), vzero), vzero, v_div(vscale, (f1)))
#define fscalar_recip() (sscale/(f1))
#define fvec_recip() v_div(vscale, (f1))
#define scalar_addw() ((f1)*sw1 + (f2)*sw2 + sdelta)
#define vec_addw() v_fma((f1), vw1, v_fma((f2), vw2, vdelta))
#undef load_as_f32
#undef store_as_s32
#define load_as_f32(addr) v_cvt_f32(vx_load(addr))
#define store_as_s32(addr, x) v_store((addr), v_round(x))

#undef this_is_binary
#undef this_is_unary
#define this_is_binary(expr) expr
#define this_is_unary(expr)

#undef DEFINE_SCALED_OP_ALLTYPES
#define DEFINE_SCALED_OP_ALLTYPES(opname, scale_arg, iscalar_op, fscalar_op, ivec_op, fvec_op, init, when_binary) \
    DEFINE_SCALED_OP_8(opname##8u, scale_arg, uchar, v_uint8, iscalar_op, ivec_op, init##_f32, v_pack_u_store, when_binary) \
    DEFINE_SCALED_OP_8(opname##8s, scale_arg, schar, v_int8, iscalar_op, ivec_op, init##_f32, v_pack_store, when_binary) \
    DEFINE_SCALED_OP_16(opname##16u, scale_arg, ushort, v_uint16, iscalar_op, ivec_op, init##_f32, v_pack_u_store, when_binary) \
    DEFINE_SCALED_OP_16(opname##16s, scale_arg, short, v_int16, iscalar_op, ivec_op, init##_f32, v_pack_store, when_binary) \
    DEFINE_SCALED_OP_NOSIMD(opname##32u, scale_arg, unsigned, double, iscalar_op, init##_nosimd_f64, when_binary) \
    DEFINE_SCALED_OP_NOSIMD(opname##32s, scale_arg, int, double, iscalar_op, init##_nosimd_f64, when_binary) \
    DEFINE_SCALED_OP_NOSIMD(opname##64u, scale_arg, uint64, double, iscalar_op, init##_nosimd_f64, when_binary) \
    DEFINE_SCALED_OP_NOSIMD(opname##64s, scale_arg, int64, double, iscalar_op, init##_nosimd_f64, when_binary) \
    DEFINE_SCALED_OP_32(opname##32f, scale_arg, float, v_float32, fscalar_op, fvec_op, init##_f32, vx_load, v_store, when_binary) \
    DEFINE_SCALED_OP_64F(opname##64f, scale_arg, fscalar_op, fvec_op, init##_f64, when_binary) \
    DEFINE_SCALED_OP_16F(opname##16f, scale_arg, hfloat, fscalar_op, fvec_op, init##_f32, when_binary) \
    DEFINE_SCALED_OP_16F(opname##16bf, scale_arg, bfloat, fscalar_op, fvec_op, init##_f32, when_binary)


DEFINE_SCALED_OP_ALLTYPES(mul, double scale, scalar_mul, scalar_mul, vec_mul, vec_mul, init_muldiv, this_is_binary)
DEFINE_SCALED_OP_ALLTYPES(div, double scale, iscalar_div, fscalar_div, ivec_div, fvec_div, init_muldiv, this_is_binary)
DEFINE_SCALED_OP_ALLTYPES(addWeighted, double weights[3], scalar_addw, scalar_addw, vec_addw, vec_addw, init_addw, this_is_binary)
DEFINE_SCALED_OP_ALLTYPES(recip, double scale, iscalar_recip, fscalar_recip, ivec_recip, fvec_recip, init_muldiv, this_is_unary)

#endif

#ifdef ARITHM_DISPATCHING_ONLY

#undef DEFINE_BINARY_OP_DISPATCHER
#define DEFINE_BINARY_OP_DISPATCHER(opname, decl_type, type) \
void opname(const decl_type* src1, size_t step1, const decl_type* src2, size_t step2, \
            decl_type* dst, size_t step, int width, int height, void*) \
{ \
    CV_INSTRUMENT_REGION(); \
    CALL_HAL(opname, cv_hal_##opname, src1, step1, src2, step2, dst, step, width, height) \
    CV_CPU_DISPATCH(opname, ((const type*)src1, step1, (const type*)src2, step2, \
                            (type*)dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL); \
}

#define DEFINE_BINARY_OP_DISPATCHER_ALLTYPES(opname) \
    DEFINE_BINARY_OP_DISPATCHER(opname##8u, uchar, uchar) \
    DEFINE_BINARY_OP_DISPATCHER(opname##8s, schar, schar) \
    DEFINE_BINARY_OP_DISPATCHER(opname##16u, ushort, ushort) \
    DEFINE_BINARY_OP_DISPATCHER(opname##16s, short, short) \
    DEFINE_BINARY_OP_DISPATCHER(opname##32u, unsigned, unsigned) \
    DEFINE_BINARY_OP_DISPATCHER(opname##32s, int, int) \
    DEFINE_BINARY_OP_DISPATCHER(opname##64u, uint64, uint64) \
    DEFINE_BINARY_OP_DISPATCHER(opname##64s, int64, int64) \
    DEFINE_BINARY_OP_DISPATCHER(opname##16f, cv_hal_f16, hfloat) \
    DEFINE_BINARY_OP_DISPATCHER(opname##16bf, cv_hal_bf16, bfloat) \
    DEFINE_BINARY_OP_DISPATCHER(opname##32f, float, float) \
    DEFINE_BINARY_OP_DISPATCHER(opname##64f, double, double)

DEFINE_BINARY_OP_DISPATCHER_ALLTYPES(add)
DEFINE_BINARY_OP_DISPATCHER_ALLTYPES(sub)
DEFINE_BINARY_OP_DISPATCHER_ALLTYPES(max)
DEFINE_BINARY_OP_DISPATCHER_ALLTYPES(min)
DEFINE_BINARY_OP_DISPATCHER_ALLTYPES(absdiff)

DEFINE_BINARY_OP_DISPATCHER(and8u, uchar, uchar)
DEFINE_BINARY_OP_DISPATCHER(or8u, uchar, uchar)
DEFINE_BINARY_OP_DISPATCHER(xor8u, uchar, uchar)

void not8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
           uchar* dst, size_t step, int width, int height, void*)
{
    CV_INSTRUMENT_REGION();
    CALL_HAL(not8u, cv_hal_not8u, src1, step1, dst, step, width, height)
    CV_CPU_DISPATCH(not8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

#undef DEFINE_CMP_OP_DISPATCHER
#define DEFINE_CMP_OP_DISPATCHER(opname, decl_type, type) \
void opname(const decl_type* src1, size_t step1, const decl_type* src2, size_t step2, \
            uchar* dst, size_t step, int width, int height, void* params) \
{ \
    CV_INSTRUMENT_REGION(); \
    CV_CPU_DISPATCH(opname, ((const type*)src1, step1, (const type*)src2, step2, \
            dst, step, width, height, *(int*)params), CV_CPU_DISPATCH_MODES_ALL); \
}

DEFINE_CMP_OP_DISPATCHER(cmp8u, uchar, uchar)
DEFINE_CMP_OP_DISPATCHER(cmp8s, schar, schar)
DEFINE_CMP_OP_DISPATCHER(cmp16u, ushort, ushort)
DEFINE_CMP_OP_DISPATCHER(cmp16s, short, short)
DEFINE_CMP_OP_DISPATCHER(cmp32u, unsigned, unsigned)
DEFINE_CMP_OP_DISPATCHER(cmp32s, int, int)
DEFINE_CMP_OP_DISPATCHER(cmp64u, uint64, uint64)
DEFINE_CMP_OP_DISPATCHER(cmp64s, int64, int64)
DEFINE_CMP_OP_DISPATCHER(cmp16f, cv_hal_f16, hfloat)
DEFINE_CMP_OP_DISPATCHER(cmp16bf, cv_hal_bf16, bfloat)
DEFINE_CMP_OP_DISPATCHER(cmp32f, float, float)
DEFINE_CMP_OP_DISPATCHER(cmp64f, double, double)

#undef DEFINE_BINARY_OP_W_PARAMS_DISPATCHER
#define DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname, decl_type, type, read_params, paramname) \
void opname(const decl_type* src1, size_t step1, const decl_type* src2, size_t step2, \
            decl_type* dst, size_t step, int width, int height, void* params_) \
{ \
    CV_INSTRUMENT_REGION(); \
    read_params; \
    CALL_HAL(opname, cv_hal_##opname, src1, step1, src2, step2, dst, step, width, height, paramname) \
    CV_CPU_DISPATCH(opname, ((const type*)src1, step1, (const type*)src2, step2, \
                            (type*)dst, step, width, height, paramname), CV_CPU_DISPATCH_MODES_ALL); \
}

#undef DEFINE_BINARY_OP_W_PARAMS_DISPATCHER_ALLTYPES
#define DEFINE_BINARY_OP_W_PARAMS_DISPATCHER_ALLTYPES(opname, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##8u, uchar, uchar, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##8s, schar, schar, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##16u, ushort, ushort, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##16s, short, short, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##32u, unsigned, unsigned, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##32s, int, int, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##64u, uint64, uint64, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##64s, int64, int64, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##16f, cv_hal_f16, hfloat, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##16bf, cv_hal_bf16, bfloat, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##32f, float, float, read_params, paramname) \
    DEFINE_BINARY_OP_W_PARAMS_DISPATCHER(opname##64f, double, double, read_params, paramname)

DEFINE_BINARY_OP_W_PARAMS_DISPATCHER_ALLTYPES(mul, double scale = *(double*)params_, scale)
DEFINE_BINARY_OP_W_PARAMS_DISPATCHER_ALLTYPES(div, double scale = *(double*)params_, scale)
DEFINE_BINARY_OP_W_PARAMS_DISPATCHER_ALLTYPES(addWeighted, \
            double w[3]; \
            w[0]=((double*)params_)[0]; \
            w[1]=((double*)params_)[1]; \
            w[2]=((double*)params_)[2];, \
            w)

#undef DEFINE_UNARY_OP_W_PARAMS_DISPATCHER
#define DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(opname, decl_type, type, read_params, paramname) \
void opname(const decl_type* src1, size_t step1, const decl_type*, size_t, \
            decl_type* dst, size_t step, int width, int height, void* params_) \
{ \
    CV_INSTRUMENT_REGION(); \
    read_params; \
    CALL_HAL(opname, cv_hal_##opname, src1, step1, dst, step, width, height, paramname) \
    CV_CPU_DISPATCH(opname, ((const type*)src1, step1, nullptr, 0, \
                            (type*)dst, step, width, height, paramname), CV_CPU_DISPATCH_MODES_ALL); \
}

DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip8u, uchar, uchar, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip8s, schar, schar, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip16u, ushort, ushort, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip16s, short, short, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip32u, unsigned, unsigned, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip32s, int, int, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip64u, uint64, uint64, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip64s, int64, int64, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip16f, cv_hal_f16, hfloat, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip16bf, cv_hal_bf16, bfloat, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip32f, float, float, double scale = *(double*)params_, scale)
DEFINE_UNARY_OP_W_PARAMS_DISPATCHER(recip64f, double, double, double scale = *(double*)params_, scale)

#endif

#ifndef ARITHM_DISPATCHING_ONLY
    CV_CPU_OPTIMIZATION_NAMESPACE_END
#endif

}} // cv::hal::
