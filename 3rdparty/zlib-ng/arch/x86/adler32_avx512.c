/* adler32_avx512.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011 Mark Adler
 * Authors:
 *   Adam Stylinski <kungfujesus06@gmail.com>
 *   Brian Bockelman <bockelman@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef X86_AVX512

#include "../../zbuild.h"
#include "../../adler32_p.h"
#include "../../adler32_fold.h"
#include "../../cpu_features.h"
#include <immintrin.h>
#include "x86_intrins.h"
#include "adler32_avx512_p.h"

static inline uint32_t adler32_fold_copy_impl(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len, const int COPY) {
    if (src == NULL) return 1L;
    if (len == 0) return adler;

    uint32_t adler0, adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

rem_peel:
    if (len < 64) {
        /* This handles the remaining copies, just call normal adler checksum after this */
        if (COPY) {
            __mmask64 storemask = (0xFFFFFFFFFFFFFFFFUL >> (64 - len));
            __m512i copy_vec = _mm512_maskz_loadu_epi8(storemask, src);
            _mm512_mask_storeu_epi8(dst, storemask, copy_vec);
        }

#ifdef X86_AVX2
        return adler32_avx2(adler, src, len);
#elif defined(X86_SSSE3)
        return adler32_ssse3(adler, src, len);
#else
        return adler32_len_16(adler0, src, len, adler1);
#endif
    }

    __m512i vbuf, vs1_0, vs3;

    const __m512i dot2v = _mm512_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                                          38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                          56, 57, 58, 59, 60, 61, 62, 63, 64);
    const __m512i dot3v = _mm512_set1_epi16(1);
    const __m512i zero = _mm512_setzero_si512();
    size_t k;

    while (len >= 64) {
        __m512i vs1 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler0));
        __m512i vs2 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler1));
        vs1_0 = vs1;
        vs3 = _mm512_setzero_si512();

        k = MIN(len, NMAX);
        k -= k % 64;
        len -= k;

        while (k >= 64) {
            /*
               vs1 = adler + sum(c[i])
               vs2 = sum2 + 64 vs1 + sum( (64-i+1) c[i] )
            */
            vbuf = _mm512_loadu_si512(src);

            if (COPY) {
                _mm512_storeu_si512(dst, vbuf);
                dst += 64;
            }

            src += 64;
            k -= 64;

            __m512i vs1_sad = _mm512_sad_epu8(vbuf, zero);
            __m512i v_short_sum2 = _mm512_maddubs_epi16(vbuf, dot2v);
            vs1 = _mm512_add_epi32(vs1_sad, vs1);
            vs3 = _mm512_add_epi32(vs3, vs1_0);
            __m512i vsum2 = _mm512_madd_epi16(v_short_sum2, dot3v);
            vs2 = _mm512_add_epi32(vsum2, vs2);
            vs1_0 = vs1;
        }

        vs3 = _mm512_slli_epi32(vs3, 6);
        vs2 = _mm512_add_epi32(vs2, vs3);

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

Z_INTERNAL uint32_t adler32_fold_copy_avx512(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len) {
    return adler32_fold_copy_impl(adler, dst, src, len, 1);
}

Z_INTERNAL uint32_t adler32_avx512(uint32_t adler, const uint8_t *src, size_t len) {
    return adler32_fold_copy_impl(adler, NULL, src, len, 0);
}

#endif

