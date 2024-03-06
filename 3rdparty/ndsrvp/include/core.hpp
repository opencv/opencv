#ifndef OPENCV_NDSRVP_CORE_HPP_INCLUDED
#define OPENCV_NDSRVP_CORE_HPP_INCLUDED

#define NDSRVP_BINOP_DECL(name, srctype, vtype, len)              \
    int ndsrvp_##name(const srctype* src1_data, size_t src1_step, \
        const srctype* src2_data, size_t src2_step,               \
        srctype* dst_data, size_t dst_step,                       \
        int width, int height);

#define NDSRVP_UNOP_DECL(name, srctype, vtype, len)             \
    int ndsrvp_##name(const srctype* src_data, size_t src_step, \
        srctype* dst_data, size_t dst_step,                     \
        int width, int height);

#define NDSRVP_BINOP_DECL_BULK(op)                    \
    NDSRVP_BINOP_DECL(op##8u, uchar, uint8x8_t, 8)    \
    NDSRVP_BINOP_DECL(op##8s, schar, int8x8_t, 8)     \
    NDSRVP_BINOP_DECL(op##16u, ushort, uint16x4_t, 4) \
    NDSRVP_BINOP_DECL(op##16s, short, int16x4_t, 4)   \
    NDSRVP_BINOP_DECL(op##32s, int, int32x2_t, 2)

// #### add ####

NDSRVP_BINOP_DECL_BULK(add)

#undef cv_hal_add8u
#define cv_hal_add8u ndsrvp_add8u

#undef cv_hal_add8s
#define cv_hal_add8s ndsrvp_add8s

#undef cv_hal_add16u
#define cv_hal_add16u ndsrvp_add16u

#undef cv_hal_add16s
#define cv_hal_add16s ndsrvp_add16s

#undef cv_hal_add32s
#define cv_hal_add32s ndsrvp_add32s

// #### sub ####

NDSRVP_BINOP_DECL_BULK(sub)

#undef cv_hal_sub8u
#define cv_hal_sub8u ndsrvp_sub8u

#undef cv_hal_sub8s
#define cv_hal_sub8s ndsrvp_sub8s

#undef cv_hal_sub16u
#define cv_hal_sub16u ndsrvp_sub16u

#undef cv_hal_sub16s
#define cv_hal_sub16s ndsrvp_sub16s

#undef cv_hal_sub32s
#define cv_hal_sub32s ndsrvp_sub32s

// #### max ####

NDSRVP_BINOP_DECL_BULK(max)

#undef cv_hal_max8u
#define cv_hal_max8u ndsrvp_max8u

#undef cv_hal_max8s
#define cv_hal_max8s ndsrvp_max8s

#undef cv_hal_max16u
#define cv_hal_max16u ndsrvp_max16u

#undef cv_hal_max16s
#define cv_hal_max16s ndsrvp_max16s

#undef cv_hal_max32s
#define cv_hal_max32s ndsrvp_max32s

#endif

// #### min ####

NDSRVP_BINOP_DECL_BULK(min)

#undef cv_hal_min8u
#define cv_hal_min8u ndsrvp_min8u

#undef cv_hal_min8s
#define cv_hal_min8s ndsrvp_min8s

#undef cv_hal_min16u
#define cv_hal_min16u ndsrvp_min16u

#undef cv_hal_min16s
#define cv_hal_min16s ndsrvp_min16s

#undef cv_hal_min32s
#define cv_hal_min32s ndsrvp_min32s

// #### absdiff ####

NDSRVP_BINOP_DECL_BULK(absdiff)

#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u ndsrvp_absdiff8u

#undef cv_hal_absdiff8s
#define cv_hal_absdiff8s ndsrvp_absdiff8s

#undef cv_hal_absdiff16u
#define cv_hal_absdiff16u ndsrvp_absdiff16u

#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s ndsrvp_absdiff16s

#undef cv_hal_absdiff32s
#define cv_hal_absdiff32s ndsrvp_absdiff32s

// #### bitwise ####

NDSRVP_BINOP_DECL(and8u, uchar, uint8x8_t, 8)
NDSRVP_BINOP_DECL(or8u, uchar, uint8x8_t, 8)
NDSRVP_BINOP_DECL(xor8u, uchar, uint8x8_t, 8)
NDSRVP_UNOP_DECL(not8u, uchar, uint8x8_t, 8)

#undef cv_hal_and8u
#define cv_hal_and8u ndsrvp_and8u

#undef cv_hal_or8u
#define cv_hal_or8u ndsrvp_or8u

#undef cv_hal_xor8u
#define cv_hal_xor8u ndsrvp_xor8u

#undef cv_hal_not8u
#define cv_hal_not8u ndsrvp_not8u
