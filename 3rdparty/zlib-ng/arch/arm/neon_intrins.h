#ifndef ARM_NEON_INTRINS_H
#define ARM_NEON_INTRINS_H

#if defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC))
/* arm64_neon.h is MSVC specific */
#  include <arm64_neon.h>
#else
#  include <arm_neon.h>
#endif

#if defined(ARM_NEON) && !defined(__aarch64__) && !defined(_M_ARM64) && !defined(_M_ARM64EC)
/* Compatibility shim for the _high family of functions */
#define vmull_high_u8(a, b) vmull_u8(vget_high_u8(a), vget_high_u8(b))
#define vmlal_high_u8(a, b, c) vmlal_u8(a, vget_high_u8(b), vget_high_u8(c))
#define vmlal_high_u16(a, b, c) vmlal_u16(a, vget_high_u16(b), vget_high_u16(c))
#define vaddw_high_u8(a, b) vaddw_u8(a, vget_high_u8(b))
#endif

#ifdef ARM_NEON

#define vqsubq_u16_x4_x1(out, a, b) do { \
    out.val[0] = vqsubq_u16(a.val[0], b); \
    out.val[1] = vqsubq_u16(a.val[1], b); \
    out.val[2] = vqsubq_u16(a.val[2], b); \
    out.val[3] = vqsubq_u16(a.val[3], b); \
} while (0)


#  ifndef ARM_NEON_HASLD4

static inline uint16x8x4_t vld1q_u16_x4(uint16_t const *a) {
    uint16x8x4_t ret = (uint16x8x4_t) {{
                          vld1q_u16(a),
                          vld1q_u16(a+8),
                          vld1q_u16(a+16),
                          vld1q_u16(a+24)}};
    return ret;
}

static inline uint8x16x4_t vld1q_u8_x4(uint8_t const *a) {
    uint8x16x4_t ret = (uint8x16x4_t) {{
                          vld1q_u8(a),
                          vld1q_u8(a+16),
                          vld1q_u8(a+32),
                          vld1q_u8(a+48)}};
    return ret;
}

static inline void vst1q_u16_x4(uint16_t *p, uint16x8x4_t a) {
    vst1q_u16(p, a.val[0]);
    vst1q_u16(p + 8, a.val[1]);
    vst1q_u16(p + 16, a.val[2]);
    vst1q_u16(p + 24, a.val[3]);
}
#  endif // HASLD4 check
#endif

#endif // include guard ARM_NEON_INTRINS_H
