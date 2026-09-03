#ifndef ZARM_H
#define ZARM_H

#include "zutil.h"

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)

#  if defined(_MSC_VER)
#    include <arm64_neon.h>
#  else
#    include <arm_neon.h>
#  endif

#  define Z_ARM64_NEON 1

static inline unsigned char FAR *neon_copy_disjoint(unsigned char FAR *out,
                                                    unsigned char FAR *from,
                                                    unsigned len) {
    while (len >= 16) {
        vst1q_u8(out, vld1q_u8(from));
        out += 16;
        from += 16;
        len -= 16;
    }
    while (len--)
        *out++ = *from++;
    return out;
}

static inline unsigned char FAR *neon_copy_lz77(unsigned char FAR *out,
                                                unsigned dist, unsigned len) {
    unsigned char FAR *from = out - dist;

    if (dist >= 16) {
        while (len >= 16) {
            vst1q_u8(out, vld1q_u8(from));
            out += 16;
            from += 16;
            len -= 16;
        }
    }
    else if (len >= 16) {
        uint8x16_t pattern;
        unsigned step = (16 / dist) * dist;

        if (dist == 1) {
            pattern = vdupq_n_u8(*from);
        }
        else {
            unsigned char idx[16];
            unsigned i, k = 0;
            for (i = 0; i < 16; i++) {
                idx[i] = (unsigned char)k;
                if (++k == dist)
                    k = 0;
            }
            pattern = vqtbl1q_u8(vld1q_u8(from), vld1q_u8(idx));
        }

        do {
            vst1q_u8(out, pattern);
            out += step;
            len -= step;
        } while (len >= 16);

        from = out - dist;
    }

    while (len--)
        *out++ = *from++;

    return out;
}

#endif

#endif
