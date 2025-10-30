/* compare256_avx2.c -- AVX2 version of compare256
 * Copyright Mika T. Lindqvist  <postmaster@raasu.org>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zutil_p.h"
#include "deflate.h"
#include "fallback_builtins.h"

#if defined(X86_AVX2) && defined(HAVE_BUILTIN_CTZ)

#include <immintrin.h>
#ifdef _MSC_VER
#  include <nmmintrin.h>
#endif

static inline uint32_t compare256_avx2_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        __m256i ymm_src0, ymm_src1, ymm_cmp;
        ymm_src0 = _mm256_loadu_si256((__m256i*)src0);
        ymm_src1 = _mm256_loadu_si256((__m256i*)src1);
        ymm_cmp = _mm256_cmpeq_epi8(ymm_src0, ymm_src1); /* non-identical bytes = 00, identical bytes = FF */
        unsigned mask = (unsigned)_mm256_movemask_epi8(ymm_cmp);
        if (mask != 0xFFFFFFFF) {
            uint32_t match_byte = (uint32_t)__builtin_ctz(~mask); /* Invert bits so identical = 0 */
            return len + match_byte;
        }

        src0 += 32, src1 += 32, len += 32;

        ymm_src0 = _mm256_loadu_si256((__m256i*)src0);
        ymm_src1 = _mm256_loadu_si256((__m256i*)src1);
        ymm_cmp = _mm256_cmpeq_epi8(ymm_src0, ymm_src1);
        mask = (unsigned)_mm256_movemask_epi8(ymm_cmp);
        if (mask != 0xFFFFFFFF) {
            uint32_t match_byte = (uint32_t)__builtin_ctz(~mask);
            return len + match_byte;
        }

        src0 += 32, src1 += 32, len += 32;
    } while (len < 256);

    return 256;
}

Z_INTERNAL uint32_t compare256_avx2(const uint8_t *src0, const uint8_t *src1) {
    return compare256_avx2_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_avx2
#define COMPARE256          compare256_avx2_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_avx2
#define COMPARE256          compare256_avx2_static

#include "match_tpl.h"

#endif
