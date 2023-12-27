#ifndef ARM_ACLE_INTRINS_H
#define ARM_ACLE_INTRINS_H

#include <stdint.h>
#ifdef _MSC_VER
#  include <intrin.h>
#elif defined(HAVE_ARM_ACLE_H)
#  include <arm_acle.h>
#endif

#ifdef ARM_ACLE
#if defined(__aarch64__)
#  define Z_TARGET_CRC Z_TARGET("+crc")
#else
#  define Z_TARGET_CRC
#endif
#endif

#ifdef ARM_SIMD
#ifdef _MSC_VER
typedef uint32_t uint16x2_t;

#define __uqsub16 _arm_uqsub16
#elif !defined(ARM_SIMD_INTRIN)
typedef uint32_t uint16x2_t;

static inline uint16x2_t __uqsub16(uint16x2_t __a, uint16x2_t __b) {
    uint16x2_t __c;
    __asm__ __volatile__("uqsub16 %0, %1, %2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}
#endif
#endif

#endif // include guard ARM_ACLE_INTRINS_H
