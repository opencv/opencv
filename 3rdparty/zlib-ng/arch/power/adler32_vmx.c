/* adler32_vmx.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011 Mark Adler
 * Copyright (C) 2017-2023 Mika T. Lindqvist <postmaster@raasu.org>
 * Copyright (C) 2021 Adam Stylinski <kungfujesus06@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef PPC_VMX
#include <altivec.h>
#include "zbuild.h"
#include "zendian.h"
#include "adler32_p.h"

#define vmx_zero()  (vec_splat_u32(0))

static inline void vmx_handle_head_or_tail(uint32_t *pair, const uint8_t *buf, size_t len) {
    unsigned int i;
    for (i = 0; i < len; ++i) {
        pair[0] += buf[i];
        pair[1] += pair[0];
    }
}

static void vmx_accum32(uint32_t *s, const uint8_t *buf, size_t len) {
    /* Different taps for the separable components of sums */
    const vector unsigned char t0 = {64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49};
    const vector unsigned char t1 = {48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33};
    const vector unsigned char t2 = {32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17};
    const vector unsigned char t3 = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    /* As silly and inefficient as it seems, creating 1 permutation vector to permute
     * a 2 element vector from a single load + a subsequent shift is just barely faster
     * than doing 2 indexed insertions into zero initialized vectors from unaligned memory. */
    const vector unsigned char s0_perm = {0, 1, 2, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
    const vector unsigned char shift_vec = vec_sl(vec_splat_u8(8), vec_splat_u8(2));
    vector unsigned int  adacc, s2acc;
    vector unsigned int pair_vec = vec_ld(0, s);
    adacc = vec_perm(pair_vec, pair_vec, s0_perm);
#if BYTE_ORDER == LITTLE_ENDIAN
    s2acc = vec_sro(pair_vec, shift_vec);
#else
    s2acc = vec_slo(pair_vec, shift_vec);
#endif

    vector unsigned int zero = vmx_zero();
    vector unsigned int s3acc = zero;
    vector unsigned int s3acc_0 = zero;
    vector unsigned int adacc_prev = adacc;
    vector unsigned int adacc_prev_0 = zero;

    vector unsigned int s2acc_0 = zero;
    vector unsigned int s2acc_1 = zero;
    vector unsigned int s2acc_2 = zero;

    /* Maintain a running sum of a second half, this might help use break yet another
     * data dependency bubble in the sum */
    vector unsigned int adacc_0 = zero;

    int num_iter = len / 4;
    int rem = len & 3;

    for (int i = 0; i < num_iter; ++i) {
        vector unsigned char d0 = vec_ld(0, buf);
        vector unsigned char d1 = vec_ld(16, buf);
        vector unsigned char d2 = vec_ld(32, buf);
        vector unsigned char d3 = vec_ld(48, buf);

        /* The core operation of the loop, basically
         * what is being unrolled below */
        adacc = vec_sum4s(d0, adacc);
        s3acc = vec_add(s3acc, adacc_prev);
        s3acc_0 = vec_add(s3acc_0, adacc_prev_0);
        s2acc = vec_msum(t0, d0, s2acc);

        /* interleave dependent sums in here */
        adacc_0 = vec_sum4s(d1, adacc_0);
        s2acc_0 = vec_msum(t1, d1, s2acc_0);
        adacc = vec_sum4s(d2, adacc);
        s2acc_1 = vec_msum(t2, d2, s2acc_1);
        s2acc_2 = vec_msum(t3, d3, s2acc_2);
        adacc_0 = vec_sum4s(d3, adacc_0);

        adacc_prev = adacc;
        adacc_prev_0 = adacc_0;
        buf += 64;
    }

    adacc = vec_add(adacc, adacc_0);
    s3acc = vec_add(s3acc, s3acc_0);
    s3acc = vec_sl(s3acc, vec_splat_u32(6));

    if (rem) {
        adacc_prev = vec_add(adacc_prev_0, adacc_prev);
        adacc_prev = vec_sl(adacc_prev, vec_splat_u32(4));
        while (rem--) {
            vector unsigned char d0 = vec_ld(0, buf);
            adacc = vec_sum4s(d0, adacc);
            s3acc = vec_add(s3acc, adacc_prev);
            s2acc = vec_msum(t3, d0, s2acc);
            adacc_prev = vec_sl(adacc, vec_splat_u32(4));
            buf += 16;
        }
    }


    /* Sum up independent second sums */
    s2acc = vec_add(s2acc, s2acc_0);
    s2acc_2 = vec_add(s2acc_1, s2acc_2);
    s2acc = vec_add(s2acc, s2acc_2);

    s2acc = vec_add(s2acc, s3acc);

    adacc = vec_add(adacc, vec_sld(adacc, adacc, 8));
    s2acc = vec_add(s2acc, vec_sld(s2acc, s2acc, 8));
    adacc = vec_add(adacc, vec_sld(adacc, adacc, 4));
    s2acc = vec_add(s2acc, vec_sld(s2acc, s2acc, 4));

    vec_ste(adacc, 0, s);
    vec_ste(s2acc, 0, s+1);
}

Z_INTERNAL uint32_t adler32_vmx(uint32_t adler, const uint8_t *buf, size_t len) {
    uint32_t sum2;
    uint32_t pair[16] ALIGNED_(16);
    memset(&pair[2], 0, 14);
    int n = NMAX;
    unsigned int done = 0, i;

    /* Split Adler-32 into component sums, it can be supplied by
     * the caller sites (e.g. in a PNG file).
     */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;
    pair[0] = adler;
    pair[1] = sum2;

    /* in case user likes doing a byte at a time, keep it fast */
    if (UNLIKELY(len == 1))
        return adler32_len_1(adler, buf, sum2);

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (UNLIKELY(buf == NULL))
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (UNLIKELY(len < 16))
        return adler32_len_16(adler, buf, len, sum2);

    // Align buffer
    unsigned int al = 0;
    if ((uintptr_t)buf & 0xf) {
        al = 16-((uintptr_t)buf & 0xf);
        if (al > len) {
            al=len;
        }
        vmx_handle_head_or_tail(pair, buf, al);

        done += al;
        /* Rather than rebasing, we can reduce the max sums for the
         * first round only */
        n -= al;
    }
    for (i = al; i < len; i += n) {
        int remaining = (int)(len-i);
        n = MIN(remaining, (i == al) ? n : NMAX);

        if (n < 16)
            break;

        vmx_accum32(pair, buf + i, n / 16);
        pair[0] %= BASE;
        pair[1] %= BASE;

        done += (n / 16) * 16;
    }

    /* Handle the tail elements. */
    if (done < len) {
        vmx_handle_head_or_tail(pair, (buf + done), len - done);
        pair[0] %= BASE;
        pair[1] %= BASE;
    }

    /* D = B * 65536 + A, see: https://en.wikipedia.org/wiki/Adler-32. */
    return (pair[1] << 16) | pair[0];
}
#endif
