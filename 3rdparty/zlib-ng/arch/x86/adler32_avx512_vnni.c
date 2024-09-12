/* adler32_avx512_vnni.c -- compute the Adler-32 checksum of a data stream
 * Based on Brian Bockelman's AVX2 version
 * Copyright (C) 1995-2011 Mark Adler
 * Authors:
 *   Adam Stylinski <kungfujesus06@gmail.com>
 *   Brian Bockelman <bockelman@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef X86_AVX512VNNI

#include "zbuild.h"
#include "adler32_p.h"
#include "arch_functions.h"
#include <immintrin.h>
#include "x86_intrins.h"
#include "adler32_avx512_p.h"
#include "adler32_avx2_p.h"

Z_INTERNAL uint32_t adler32_avx512_vnni(uint32_t adler, const uint8_t *src, size_t len) {
    if (src == NULL) return 1L;
    if (len == 0) return adler;

    uint32_t adler0, adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

rem_peel:
    if (len < 32)
        return adler32_ssse3(adler, src, len);

    if (len < 64)
        return adler32_avx2(adler, src, len);

    const __m512i dot2v = _mm512_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                                          38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                          56, 57, 58, 59, 60, 61, 62, 63, 64);

    const __m512i zero = _mm512_setzero_si512();
    __m512i vs1, vs2;

    while (len >= 64) {
        vs1 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler0));
        vs2 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler1));
        size_t k = MIN(len, NMAX);
        k -= k % 64;
        len -= k;
        __m512i vs1_0 = vs1;
        __m512i vs3 = _mm512_setzero_si512();
        /* We might get a tad bit more ILP here if we sum to a second register in the loop */
        __m512i vs2_1 = _mm512_setzero_si512();
        __m512i vbuf0, vbuf1;

        /* Remainder peeling */
        if (k % 128) {
            vbuf1 = _mm512_loadu_si512((__m512i*)src);

            src += 64;
            k -= 64;

            __m512i vs1_sad = _mm512_sad_epu8(vbuf1, zero);
            vs1 = _mm512_add_epi32(vs1, vs1_sad);
            vs3 = _mm512_add_epi32(vs3, vs1_0);
            vs2 = _mm512_dpbusd_epi32(vs2, vbuf1, dot2v);
            vs1_0 = vs1;
        }

        /* Manually unrolled this loop by 2 for an decent amount of ILP */
        while (k >= 128) {
            /*
               vs1 = adler + sum(c[i])
               vs2 = sum2 + 64 vs1 + sum( (64-i+1) c[i] )
            */
            vbuf0 = _mm512_loadu_si512((__m512i*)src);
            vbuf1 = _mm512_loadu_si512((__m512i*)(src + 64));
            src += 128;
            k -= 128;

            __m512i vs1_sad = _mm512_sad_epu8(vbuf0, zero);
            vs1 = _mm512_add_epi32(vs1, vs1_sad);
            vs3 = _mm512_add_epi32(vs3, vs1_0);
            /* multiply-add, resulting in 16 ints. Fuse with sum stage from prior versions, as we now have the dp
             * instructions to eliminate them */
            vs2 = _mm512_dpbusd_epi32(vs2, vbuf0, dot2v);

            vs3 = _mm512_add_epi32(vs3, vs1);
            vs1_sad = _mm512_sad_epu8(vbuf1, zero);
            vs1 = _mm512_add_epi32(vs1, vs1_sad);
            vs2_1 = _mm512_dpbusd_epi32(vs2_1, vbuf1, dot2v);
            vs1_0 = vs1;
        }

        vs3 = _mm512_slli_epi32(vs3, 6);
        vs2 = _mm512_add_epi32(vs2, vs3);
        vs2 = _mm512_add_epi32(vs2, vs2_1);

        adler0 = partial_hsum(vs1) % BASE;
        adler1 = _mm512_reduce_add_epu32(vs2) % BASE;
    }

    adler = adler0 | (adler1 << 16);

    /* Process tail (len < 64). */
    if (len) {
        goto rem_peel;
    }

    return adler;
}

Z_INTERNAL uint32_t adler32_fold_copy_avx512_vnni(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len) {
    if (src == NULL) return 1L;
    if (len == 0) return adler;

    uint32_t adler0, adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

rem_peel_copy:
    if (len < 32) {
        /* This handles the remaining copies, just call normal adler checksum after this */
        __mmask32 storemask = (0xFFFFFFFFUL >> (32 - len));
        __m256i copy_vec = _mm256_maskz_loadu_epi8(storemask, src);
        _mm256_mask_storeu_epi8(dst, storemask, copy_vec);

        return adler32_ssse3(adler, src, len);
    }

    const __m256i dot2v = _mm256_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);

    const __m256i zero = _mm256_setzero_si256();
    __m256i vs1, vs2;

    while (len >= 32) {
        vs1 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(adler0));
        vs2 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(adler1));
        size_t k = MIN(len, NMAX);
        k -= k % 32;
        len -= k;
        __m256i vs1_0 = vs1;
        __m256i vs3 = _mm256_setzero_si256();
        /* We might get a tad bit more ILP here if we sum to a second register in the loop */
        __m256i vs2_1 = _mm256_setzero_si256();
        __m256i vbuf0, vbuf1;

        /* Remainder peeling */
        if (k % 64) {
            vbuf1 = _mm256_loadu_si256((__m256i*)src);
            _mm256_storeu_si256((__m256i*)dst, vbuf1);
            dst += 32;

            src += 32;
            k -= 32;

            __m256i vs1_sad = _mm256_sad_epu8(vbuf1, zero);
            vs1 = _mm256_add_epi32(vs1, vs1_sad);
            vs3 = _mm256_add_epi32(vs3, vs1_0);
            vs2 = _mm256_dpbusd_epi32(vs2, vbuf1, dot2v);
            vs1_0 = vs1;
        }

        /* Manually unrolled this loop by 2 for an decent amount of ILP */
        while (k >= 64) {
            /*
               vs1 = adler + sum(c[i])
               vs2 = sum2 + 64 vs1 + sum( (64-i+1) c[i] )
            */
            vbuf0 = _mm256_loadu_si256((__m256i*)src);
            vbuf1 = _mm256_loadu_si256((__m256i*)(src + 32));
            _mm256_storeu_si256((__m256i*)dst, vbuf0);
            _mm256_storeu_si256((__m256i*)(dst + 32), vbuf1);
            dst += 64;
            src += 64;
            k -= 64;

            __m256i vs1_sad = _mm256_sad_epu8(vbuf0, zero);
            vs1 = _mm256_add_epi32(vs1, vs1_sad);
            vs3 = _mm256_add_epi32(vs3, vs1_0);
            /* multiply-add, resulting in 16 ints. Fuse with sum stage from prior versions, as we now have the dp
             * instructions to eliminate them */
            vs2 = _mm256_dpbusd_epi32(vs2, vbuf0, dot2v);

            vs3 = _mm256_add_epi32(vs3, vs1);
            vs1_sad = _mm256_sad_epu8(vbuf1, zero);
            vs1 = _mm256_add_epi32(vs1, vs1_sad);
            vs2_1 = _mm256_dpbusd_epi32(vs2_1, vbuf1, dot2v);
            vs1_0 = vs1;
        }

        vs3 = _mm256_slli_epi32(vs3, 5);
        vs2 = _mm256_add_epi32(vs2, vs3);
        vs2 = _mm256_add_epi32(vs2, vs2_1);

        adler0 = partial_hsum256(vs1) % BASE;
        adler1 = hsum256(vs2) % BASE;
    }

    adler = adler0 | (adler1 << 16);

    /* Process tail (len < 64). */
    if (len) {
        goto rem_peel_copy;
    }

    return adler;
}

#endif
