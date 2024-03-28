/* adler32_p.h -- Private inline functions and macros shared with
 *                different computation of the Adler-32 checksum
 *                of a data stream.
 * Copyright (C) 1995-2011, 2016 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef ADLER32_P_H
#define ADLER32_P_H

#define BASE 65521U     /* largest prime smaller than 65536 */
#define NMAX 5552
/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */

#define DO1(sum1, sum2, buf, i)  {(sum1) += buf[(i)]; (sum2) += (sum1);}
#define DO2(sum1, sum2, buf, i)  {DO1(sum1, sum2, buf, i); DO1(sum1, sum2, buf, i+1);}
#define DO4(sum1, sum2, buf, i)  {DO2(sum1, sum2, buf, i); DO2(sum1, sum2, buf, i+2);}
#define DO8(sum1, sum2, buf, i)  {DO4(sum1, sum2, buf, i); DO4(sum1, sum2, buf, i+4);}
#define DO16(sum1, sum2, buf)    {DO8(sum1, sum2, buf, 0); DO8(sum1, sum2, buf, 8);}

static inline uint32_t adler32_len_1(uint32_t adler, const uint8_t *buf, uint32_t sum2) {
    adler += buf[0];
    adler %= BASE;
    sum2 += adler;
    sum2 %= BASE;
    return adler | (sum2 << 16);
}

static inline uint32_t adler32_len_16(uint32_t adler, const uint8_t *buf, size_t len, uint32_t sum2) {
    while (len) {
        --len;
        adler += *buf++;
        sum2 += adler;
    }
    adler %= BASE;
    sum2 %= BASE;            /* only added so many BASE's */
    /* return recombined sums */
    return adler | (sum2 << 16);
}

static inline uint32_t adler32_copy_len_16(uint32_t adler, const uint8_t *buf, uint8_t *dst, size_t len, uint32_t sum2) {
    while (len--) {
        *dst = *buf++;
        adler += *dst++;
        sum2 += adler;
    }
    adler %= BASE;
    sum2 %= BASE;            /* only added so many BASE's */
    /* return recombined sums */
    return adler | (sum2 << 16);
}

static inline uint32_t adler32_len_64(uint32_t adler, const uint8_t *buf, size_t len, uint32_t sum2) {
#ifdef UNROLL_MORE
    while (len >= 16) {
        len -= 16;
        DO16(adler, sum2, buf);
        buf += 16;
#else
    while (len >= 8) {
        len -= 8;
        DO8(adler, sum2, buf, 0);
        buf += 8;
#endif
    }
    /* Process tail (len < 16).  */
    return adler32_len_16(adler, buf, len, sum2);
}

#endif /* ADLER32_P_H */
