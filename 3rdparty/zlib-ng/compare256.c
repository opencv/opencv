/* compare256.c -- 256 byte memory comparison with match length return
 * Copyright (C) 2020 Nathan Moinvaziri
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zutil_p.h"
#include "fallback_builtins.h"

/* ALIGNED, byte comparison */
static inline uint32_t compare256_c_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
    } while (len < 256);

    return 256;
}

Z_INTERNAL uint32_t compare256_c(const uint8_t *src0, const uint8_t *src1) {
    return compare256_c_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_c
#define COMPARE256          compare256_c_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_c
#define COMPARE256          compare256_c_static

#include "match_tpl.h"

#if defined(UNALIGNED_OK) && BYTE_ORDER == LITTLE_ENDIAN
/* 16-bit unaligned integer comparison */
static inline uint32_t compare256_unaligned_16_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        if (zng_memcmp_2(src0, src1) != 0)
            return len + (*src0 == *src1);
        src0 += 2, src1 += 2, len += 2;

        if (zng_memcmp_2(src0, src1) != 0)
            return len + (*src0 == *src1);
        src0 += 2, src1 += 2, len += 2;

        if (zng_memcmp_2(src0, src1) != 0)
            return len + (*src0 == *src1);
        src0 += 2, src1 += 2, len += 2;

        if (zng_memcmp_2(src0, src1) != 0)
            return len + (*src0 == *src1);
        src0 += 2, src1 += 2, len += 2;
    } while (len < 256);

    return 256;
}

Z_INTERNAL uint32_t compare256_unaligned_16(const uint8_t *src0, const uint8_t *src1) {
    return compare256_unaligned_16_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_unaligned_16
#define COMPARE256          compare256_unaligned_16_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_unaligned_16
#define COMPARE256          compare256_unaligned_16_static

#include "match_tpl.h"

#ifdef HAVE_BUILTIN_CTZ
/* 32-bit unaligned integer comparison */
static inline uint32_t compare256_unaligned_32_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        uint32_t sv, mv, diff;

        memcpy(&sv, src0, sizeof(sv));
        memcpy(&mv, src1, sizeof(mv));

        diff = sv ^ mv;
        if (diff) {
            uint32_t match_byte = __builtin_ctz(diff) / 8;
            return len + match_byte;
        }

        src0 += 4, src1 += 4, len += 4;
    } while (len < 256);

    return 256;
}

Z_INTERNAL uint32_t compare256_unaligned_32(const uint8_t *src0, const uint8_t *src1) {
    return compare256_unaligned_32_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_unaligned_32
#define COMPARE256          compare256_unaligned_32_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_unaligned_32
#define COMPARE256          compare256_unaligned_32_static

#include "match_tpl.h"

#endif

#if defined(UNALIGNED64_OK) && defined(HAVE_BUILTIN_CTZLL)
/* UNALIGNED64_OK, 64-bit integer comparison */
static inline uint32_t compare256_unaligned_64_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        uint64_t sv, mv, diff;

        memcpy(&sv, src0, sizeof(sv));
        memcpy(&mv, src1, sizeof(mv));

        diff = sv ^ mv;
        if (diff) {
            uint64_t match_byte = __builtin_ctzll(diff) / 8;
            return len + (uint32_t)match_byte;
        }

        src0 += 8, src1 += 8, len += 8;
    } while (len < 256);

    return 256;
}

Z_INTERNAL uint32_t compare256_unaligned_64(const uint8_t *src0, const uint8_t *src1) {
    return compare256_unaligned_64_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_unaligned_64
#define COMPARE256          compare256_unaligned_64_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_unaligned_64
#define COMPARE256          compare256_unaligned_64_static

#include "match_tpl.h"

#endif

#endif
