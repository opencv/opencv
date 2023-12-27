/* adler32.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011, 2016 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "functable.h"
#include "adler32_p.h"

/* ========================================================================= */
Z_INTERNAL uint32_t adler32_c(uint32_t adler, const uint8_t *buf, size_t len) {
    uint32_t sum2;
    unsigned n;

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (UNLIKELY(len == 1))
        return adler32_len_1(adler, buf, sum2);

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (UNLIKELY(buf == NULL))
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (UNLIKELY(len < 16))
        return adler32_len_16(adler, buf, len, sum2);

    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        len -= NMAX;
#ifdef UNROLL_MORE
        n = NMAX / 16;          /* NMAX is divisible by 16 */
#else
        n = NMAX / 8;           /* NMAX is divisible by 8 */
#endif
        do {
#ifdef UNROLL_MORE
            DO16(adler, sum2, buf);          /* 16 sums unrolled */
            buf += 16;
#else
            DO8(adler, sum2, buf, 0);         /* 8 sums unrolled */
            buf += 8;
#endif
        } while (--n);
        adler %= BASE;
        sum2 %= BASE;
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    return adler32_len_64(adler, buf, len, sum2);
}

#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(adler32_z)(unsigned long adler, const unsigned char *buf, size_t len) {
    return (unsigned long)functable.adler32((uint32_t)adler, buf, len);
}
#else
uint32_t Z_EXPORT PREFIX(adler32_z)(uint32_t adler, const unsigned char *buf, size_t len) {
    return functable.adler32(adler, buf, len);
}
#endif

/* ========================================================================= */
#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(adler32)(unsigned long adler, const unsigned char *buf, unsigned int len) {
    return (unsigned long)functable.adler32((uint32_t)adler, buf, len);
}
#else
uint32_t Z_EXPORT PREFIX(adler32)(uint32_t adler, const unsigned char *buf, uint32_t len) {
    return functable.adler32(adler, buf, len);
}
#endif

/* ========================================================================= */
static uint32_t adler32_combine_(uint32_t adler1, uint32_t adler2, z_off64_t len2) {
    uint32_t sum1;
    uint32_t sum2;
    unsigned rem;

    /* for negative len, return invalid adler32 as a clue for debugging */
    if (len2 < 0)
        return 0xffffffff;

    /* the derivation of this formula is left as an exercise for the reader */
    len2 %= BASE;                 /* assumes len2 >= 0 */
    rem = (unsigned)len2;
    sum1 = adler1 & 0xffff;
    sum2 = rem * sum1;
    sum2 %= BASE;
    sum1 += (adler2 & 0xffff) + BASE - 1;
    sum2 += ((adler1 >> 16) & 0xffff) + ((adler2 >> 16) & 0xffff) + BASE - rem;
    if (sum1 >= BASE) sum1 -= BASE;
    if (sum1 >= BASE) sum1 -= BASE;
    if (sum2 >= ((unsigned long)BASE << 1)) sum2 -= ((unsigned long)BASE << 1);
    if (sum2 >= BASE) sum2 -= BASE;
    return sum1 | (sum2 << 16);
}

/* ========================================================================= */
#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(adler32_combine)(unsigned long adler1, unsigned long adler2, z_off_t len2) {
    return (unsigned long)adler32_combine_((uint32_t)adler1, (uint32_t)adler2, len2);
}

unsigned long Z_EXPORT PREFIX4(adler32_combine)(unsigned long adler1, unsigned long adler2, z_off64_t len2) {
    return (unsigned long)adler32_combine_((uint32_t)adler1, (uint32_t)adler2, len2);
}
#else
uint32_t Z_EXPORT PREFIX4(adler32_combine)(uint32_t adler1, uint32_t adler2, z_off64_t len2) {
    return adler32_combine_(adler1, adler2, len2);
}
#endif
