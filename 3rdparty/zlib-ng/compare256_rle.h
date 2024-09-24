/* compare256_rle.h -- 256 byte run-length encoding comparison
 * Copyright (C) 2022 Nathan Moinvaziri
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "fallback_builtins.h"

typedef uint32_t (*compare256_rle_func)(const uint8_t* src0, const uint8_t* src1);

/* ALIGNED, byte comparison */
static inline uint32_t compare256_rle_c(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        if (*src0 != *src1)
            return len;
        src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src1 += 1, len += 1;
    } while (len < 256);

    return 256;
}

#ifdef UNALIGNED_OK
/* 16-bit unaligned integer comparison */
static inline uint32_t compare256_rle_unaligned_16(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;
    uint16_t src0_cmp, src1_cmp;

    memcpy(&src0_cmp, src0, sizeof(src0_cmp));

    do {
        memcpy(&src1_cmp, src1, sizeof(src1_cmp));
        if (src0_cmp != src1_cmp)
            return len + (*src0 == *src1);
        src1 += 2, len += 2;
        memcpy(&src1_cmp, src1, sizeof(src1_cmp));
        if (src0_cmp != src1_cmp)
            return len + (*src0 == *src1);
        src1 += 2, len += 2;
        memcpy(&src1_cmp, src1, sizeof(src1_cmp));
        if (src0_cmp != src1_cmp)
            return len + (*src0 == *src1);
        src1 += 2, len += 2;
        memcpy(&src1_cmp, src1, sizeof(src1_cmp));
        if (src0_cmp != src1_cmp)
            return len + (*src0 == *src1);
        src1 += 2, len += 2;
    } while (len < 256);

    return 256;
}

#ifdef HAVE_BUILTIN_CTZ
/* 32-bit unaligned integer comparison */
static inline uint32_t compare256_rle_unaligned_32(const uint8_t *src0, const uint8_t *src1) {
    uint32_t sv, len = 0;
    uint16_t src0_cmp;

    memcpy(&src0_cmp, src0, sizeof(src0_cmp));
    sv = ((uint32_t)src0_cmp << 16) | src0_cmp;

    do {
        uint32_t mv, diff;

        memcpy(&mv, src1, sizeof(mv));

        diff = sv ^ mv;
        if (diff) {
            uint32_t match_byte = __builtin_ctz(diff) / 8;
            return len + match_byte;
        }

        src1 += 4, len += 4;
    } while (len < 256);

    return 256;
}

#endif

#if defined(UNALIGNED64_OK) && defined(HAVE_BUILTIN_CTZLL)
/* 64-bit unaligned integer comparison */
static inline uint32_t compare256_rle_unaligned_64(const uint8_t *src0, const uint8_t *src1) {
    uint32_t src0_cmp32, len = 0;
    uint16_t src0_cmp;
    uint64_t sv;

    memcpy(&src0_cmp, src0, sizeof(src0_cmp));
    src0_cmp32 = ((uint32_t)src0_cmp << 16) | src0_cmp;
    sv = ((uint64_t)src0_cmp32 << 32) | src0_cmp32;

    do {
        uint64_t mv, diff;

        memcpy(&mv, src1, sizeof(mv));

        diff = sv ^ mv;
        if (diff) {
            uint64_t match_byte = __builtin_ctzll(diff) / 8;
            return len + (uint32_t)match_byte;
        }

        src1 += 8, len += 8;
    } while (len < 256);

    return 256;
}

#endif

#endif

