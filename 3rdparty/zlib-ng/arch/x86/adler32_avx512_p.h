#ifndef AVX512_FUNCS_H
#define AVX512_FUNCS_H

#include <immintrin.h>
#include <stdint.h>
/* Written because *_add_epi32(a) sets off ubsan */
static inline uint32_t _mm512_reduce_add_epu32(__m512i x) {
    __m256i a = _mm512_extracti64x4_epi64(x, 1);
    __m256i b = _mm512_extracti64x4_epi64(x, 0);

    __m256i a_plus_b = _mm256_add_epi32(a, b);
    __m128i c = _mm256_extracti128_si256(a_plus_b, 1);
    __m128i d = _mm256_extracti128_si256(a_plus_b, 0);
    __m128i c_plus_d = _mm_add_epi32(c, d);

    __m128i sum1 = _mm_unpackhi_epi64(c_plus_d, c_plus_d);
    __m128i sum2 = _mm_add_epi32(sum1, c_plus_d);
    __m128i sum3 = _mm_shuffle_epi32(sum2, 0x01);
    __m128i sum4 = _mm_add_epi32(sum2, sum3);

    return _mm_cvtsi128_si32(sum4);
}

static inline uint32_t partial_hsum(__m512i x) {
    /* We need a permutation vector to extract every other integer. The
     * rest are going to be zeros. Marking this const so the compiler stands
     * a better chance of keeping this resident in a register through entire
     * loop execution. We certainly have enough zmm registers (32) */
    const __m512i perm_vec = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                               1, 1, 1, 1, 1,  1,  1,  1);

    __m512i non_zero = _mm512_permutexvar_epi32(perm_vec, x);

    /* From here, it's a simple 256 bit wide reduction sum */
    __m256i non_zero_avx = _mm512_castsi512_si256(non_zero);

    /* See Agner Fog's vectorclass for a decent reference. Essentially, phadd is
     * pretty slow, much slower than the longer instruction sequence below */
    __m128i sum1  = _mm_add_epi32(_mm256_extracti128_si256(non_zero_avx, 1),
                                  _mm256_castsi256_si128(non_zero_avx));
    __m128i sum2  = _mm_add_epi32(sum1,_mm_unpackhi_epi64(sum1, sum1));
    __m128i sum3  = _mm_add_epi32(sum2,_mm_shuffle_epi32(sum2, 1));
    return (uint32_t)_mm_cvtsi128_si32(sum3);
}

#endif
