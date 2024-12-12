/* adler32_avx2_p.h -- adler32 avx2 utility functions
 * Copyright (C) 2022 Adam Stylinski
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef ADLER32_AVX2_P_H_
#define ADLER32_AVX2_P_H_

#if defined(X86_AVX2) || defined(X86_AVX512VNNI)

/* 32 bit horizontal sum, adapted from Agner Fog's vector library. */
static inline uint32_t hsum256(__m256i x) {
    __m128i sum1  = _mm_add_epi32(_mm256_extracti128_si256(x, 1),
                                  _mm256_castsi256_si128(x));
    __m128i sum2  = _mm_add_epi32(sum1, _mm_unpackhi_epi64(sum1, sum1));
    __m128i sum3  = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
    return (uint32_t)_mm_cvtsi128_si32(sum3);
}

static inline uint32_t partial_hsum256(__m256i x) {
    /* We need a permutation vector to extract every other integer. The
     * rest are going to be zeros */
    const __m256i perm_vec = _mm256_setr_epi32(0, 2, 4, 6, 1, 1, 1, 1);
    __m256i non_zero = _mm256_permutevar8x32_epi32(x, perm_vec);
    __m128i non_zero_sse = _mm256_castsi256_si128(non_zero);
    __m128i sum2  = _mm_add_epi32(non_zero_sse,_mm_unpackhi_epi64(non_zero_sse, non_zero_sse));
    __m128i sum3  = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
    return (uint32_t)_mm_cvtsi128_si32(sum3);
}
#endif

#endif
