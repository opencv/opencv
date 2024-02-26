#include "ndsrvp_hal.hpp"

#define NDSRVP_BINOP_IMPL(hal, srctype, vtype, nlane, parallel, nonparallel)     \
    int ndsrvp_##hal(const srctype* src1_data, size_t src1_step,                 \
        const srctype* src2_data, size_t src2_step,                              \
        srctype* dst_data, size_t dst_step,                                      \
        int width, int height)                                                   \
    {                                                                            \
        src1_step /= sizeof(srctype);                                            \
        src2_step /= sizeof(srctype);                                            \
        dst_step /= sizeof(srctype);                                             \
                                                                                 \
        int i, j;                                                                \
        for (i = 0; i < height; ++i) {                                           \
            const srctype* src1_row = src1_data + (src1_step * i);               \
            const srctype* src2_row = src2_data + (src2_step * i);               \
            srctype* dst_row = dst_data + (dst_step * i);                        \
                                                                                 \
            j = 0;                                                               \
            for (; j + nlane <= width; j += nlane) {                             \
                vtype vs##nlane##_1 = *(vtype*)(src1_row + j);                   \
                vtype vs##nlane##_2 = *(vtype*)(src2_row + j);                   \
                                                                                 \
                *(vtype*)(dst_row + j) = parallel(vs##nlane##_1, vs##nlane##_2); \
            }                                                                    \
            for (; j < width; j++)                                               \
                dst_row[j] = nonparallel(src1_row[j], src2_row[j]);              \
        }                                                                        \
                                                                                 \
        return CV_HAL_ERROR_OK;                                                  \
    }

#define NDSRVP_BINOP_FUNC_BULK(FUNC, unsign0, sign0, unsign1, sign1) \
    FUNC(8, uchar, uint8x8_t, unsign0, unsign1)                      \
    FUNC(8, schar, int8x8_t, sign0, sign1)                           \
    FUNC(16, ushort, uint16x4_t, unsign0, unsign1)                   \
    FUNC(16, short, int16x4_t, sign0, sign1)                         \
    FUNC(32, int, int32x2_t, sign0, sign1)

#define NDSRVP_BINOP_NAME_BULK(op, para, nonpara, unsign, sign)                                      \
    NDSRVP_BINOP_IMPL(op##8u, uchar, uint8x8_t, 8, para##unsign##op##8, nonpara##unsign##op##8)      \
    NDSRVP_BINOP_IMPL(op##8s, schar, int8x8_t, 8, para##sign##op##8, nonpara##sign##op##8)           \
    NDSRVP_BINOP_IMPL(op##16u, ushort, uint16x4_t, 4, para##unsign##op##16, nonpara##unsign##op##16) \
    NDSRVP_BINOP_IMPL(op##16s, short, int16x4_t, 4, para##sign##op##16, nonpara##sign##op##16)       \
    NDSRVP_BINOP_IMPL(op##32s, int, int32x2_t, 2, para##sign##op##32, nonpara##sign##op##32)

// #### add ####

NDSRVP_BINOP_NAME_BULK(add, __nds__v_, __nds__, uk, k)

// #### sub ####

NDSRVP_BINOP_NAME_BULK(sub, __nds__v_, __nds__, uk, k)

// #### max ####

NDSRVP_BINOP_NAME_BULK(max, __nds__v_, __nds__, u, s)

// #### min ####

NDSRVP_BINOP_NAME_BULK(min, __nds__v_, __nds__, u, s)

// #### absdiff ####

#define FUNC_ABSDIFF(xlen, srctype, vtype, prefix0, prefix1)                                                            \
    inline vtype __ndsrvp__v_##prefix0##absdiff##xlen(vtype a, vtype b)                                                 \
    {                                                                                                                   \
        return __nds__v_##prefix1##sub##xlen(__nds__v_##prefix0##max##xlen(a, b), __nds__v_##prefix0##min##xlen(a, b)); \
    }                                                                                                                   \
    inline srctype __ndsrvp__##prefix0##absdiff##xlen(srctype a, srctype b)                                             \
    {                                                                                                                   \
        return __nds__##prefix1##sub##xlen(__nds__##prefix0##max##xlen(a, b), __nds__##prefix0##min##xlen(a, b));       \
    }

NDSRVP_BINOP_FUNC_BULK(FUNC_ABSDIFF, u, s, uk, k)

NDSRVP_BINOP_NAME_BULK(absdiff, __ndsrvp__v_, __ndsrvp__, u, s)
