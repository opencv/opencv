/* adler32_avx2.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011 Mark Adler
 * Copyright (C) 2022 Adam Stylinski
 * Authors:
 *   Brian Bockelman <bockelman@gmail.com>
 *   Adam Stylinski <kungfujesus06@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef X86_AVX2

#include "zbuild.h"
#include <immintrin.h>
#include "adler32_p.h"
#include "adler32_avx2_p.h"
#include "x86_intrins.h"

extern uint32_t adler32_fold_copy_sse42(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
extern uint32_t adler32_ssse3(uint32_t adler, const uint8_t *src, size_t len);

static inline uint32_t adler32_fold_copy_impl(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len, const int COPY) {
    if (src == NULL) return 1L;
    if (len == 0) return adler;

    uint32_t adler0, adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

rem_peel:
    if (len < 16) {
        if (COPY) {
            return adler32_copy_len_16(adler0, src, dst, len, adler1);
        } else {
            return adler32_len_16(adler0, src, len, adler1);
        }
    } else if (len < 32) {
        if (COPY) {
            return adler32_fold_copy_sse42(adler, dst, src, len);
        } else {
            return adler32_ssse3(adler, src, len);
        }
    }

    __m256i vs1, vs2;

    const __m256i dot2v = _mm256_setr_epi8(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
                                           14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    const __m256i dot3v = _mm256_set1_epi16(1);
    const __m256i zero = _mm256_setzero_si256();

    while (len >= 32) {
        vs1 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(adler0));
        vs2 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(adler1));
        __m256i vs1_0 = vs1;
        __m256i vs3 = _mm256_setzero_si256();

        size_t k = MIN(len, NMAX);
        k -= k % 32;
        len -= k;

        while (k >= 32) {
            /*
               vs1 = adler + sum(c[i])
               vs2 = sum2 + 32 vs1 + sum( (32-i+1) c[i] )
            */
            __m256i vbuf = _mm256_loadu_si256((__m256i*)src);
            src += 32;
            k -= 32;

            __m256i vs1_sad = _mm256_sad_epu8(vbuf, zero); // Sum of abs diff, resulting in 2 x int32's

            if (COPY) {
                _mm256_storeu_si256((__m256i*)dst, vbuf);
                dst += 32;
            }
 
            vs1 = _mm256_add_epi32(vs1, vs1_sad);
            vs3 = _mm256_add_epi32(vs3, vs1_0);
            __m256i v_short_sum2 = _mm256_maddubs_epi16(vbuf, dot2v); // sum 32 uint8s to 16 shorts
            __m256i vsum2 = _mm256_madd_epi16(v_short_sum2, dot3v); // sum 16 shorts to 8 uint32s
            vs2 = _mm256_add_epi32(vsum2, vs2);
            vs1_0 = vs1;
        }

        /* Defer the multiplication with 32 to outside of the loop */
        vs3 = _mm256_slli_epi32(vs3, 5);
        vs2 = _mm256_add_epi32(vs2, vs3);

        /* The compiler is generating the following sequence for this integer modulus
         * when done the scalar way, in GPRs:

         adler = (s1_unpack[0] % BASE) + (s1_unpack[1] % BASE) + (s1_unpack[2] % BASE) + (s1_unpack[3] % BASE) +
                 (s1_unpack[4] % BASE) + (s1_unpack[5] % BASE) + (s1_unpack[6] % BASE) + (s1_unpack[7] % BASE);

         mov    $0x80078071,%edi // move magic constant into 32 bit register %edi
         ...
         vmovd  %xmm1,%esi // move vector lane 0 to 32 bit register %esi
         mov    %rsi,%rax  // zero-extend this value to 64 bit precision in %rax
         imul   %rdi,%rsi // do a signed multiplication with magic constant and vector element
         shr    $0x2f,%rsi // shift right by 47
         imul   $0xfff1,%esi,%esi // do a signed multiplication with value truncated to 32 bits with 0xfff1
         sub    %esi,%eax // subtract lower 32 bits of original vector value from modified one above
         ...
         // repeats for each element with vpextract instructions

         This is tricky with AVX2 for a number of reasons:
             1.) There's no 64 bit multiplication instruction, but there is a sequence to get there
             2.) There's ways to extend vectors to 64 bit precision, but no simple way to truncate
                 back down to 32 bit precision later (there is in AVX512)
             3.) Full width integer multiplications aren't cheap

         We can, however, do a relatively cheap sequence for horizontal sums.
         Then, we simply do the integer modulus on the resulting 64 bit GPR, on a scalar value. It was
         previously thought that casting to 64 bit precision was needed prior to the horizontal sum, but
         that is simply not the case, as NMAX is defined as the maximum number of scalar sums that can be
         performed on the maximum possible inputs before overflow
         */


         /* In AVX2-land, this trip through GPRs will probably be unavoidable, as there's no cheap and easy
          * conversion from 64 bit integer to 32 bit (needed for the inexpensive modulus with a constant).
          * This casting to 32 bit is cheap through GPRs (just register aliasing). See above for exactly
          * what the compiler is doing to avoid integer divisions. */
         adler0 = partial_hsum256(vs1) % BASE;
         adler1 = hsum256(vs2) % BASE;
    }

    adler = adler0 | (adler1 << 16);

    if (len) {
        goto rem_peel;
    }

    return adler;
}

Z_INTERNAL uint32_t adler32_avx2(uint32_t adler, const uint8_t *src, size_t len) {
    return adler32_fold_copy_impl(adler, NULL, src, len, 0);
}

Z_INTERNAL uint32_t adler32_fold_copy_avx2(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len) {
    return adler32_fold_copy_impl(adler, dst, src, len, 1);
}

#endif
