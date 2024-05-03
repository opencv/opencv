/* chunkset_avx2.c -- AVX2 inline functions to copy small data chunks.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#include "zbuild.h"

#ifdef X86_AVX2
#include <immintrin.h>
#include "../generic/chunk_permute_table.h"

typedef __m256i chunk_t;

#define CHUNK_SIZE 32

#define HAVE_CHUNKMEMSET_2
#define HAVE_CHUNKMEMSET_4
#define HAVE_CHUNKMEMSET_8
#define HAVE_CHUNK_MAG

/* Populate don't cares so that this is a direct lookup (with some indirection into the permute table), because dist can
 * never be 0 - 2, we'll start with an offset, subtracting 3 from the input */
static const lut_rem_pair perm_idx_lut[29] = {
    { 0, 2},                /* 3 */
    { 0, 0},                /* don't care */
    { 1 * 32, 2},           /* 5 */
    { 2 * 32, 2},           /* 6 */
    { 3 * 32, 4},           /* 7 */
    { 0 * 32, 0},           /* don't care */
    { 4 * 32, 5},           /* 9 */
    { 5 * 32, 22},          /* 10 */
    { 6 * 32, 21},          /* 11 */
    { 7 * 32, 20},          /* 12 */
    { 8 * 32, 6},           /* 13 */
    { 9 * 32, 4},           /* 14 */
    {10 * 32, 2},           /* 15 */
    { 0 * 32, 0},           /* don't care */
    {11 * 32, 15},          /* 17 */
    {11 * 32 + 16, 14},     /* 18 */
    {11 * 32 + 16 * 2, 13}, /* 19 */
    {11 * 32 + 16 * 3, 12}, /* 20 */
    {11 * 32 + 16 * 4, 11}, /* 21 */
    {11 * 32 + 16 * 5, 10}, /* 22 */
    {11 * 32 + 16 * 6,  9}, /* 23 */
    {11 * 32 + 16 * 7,  8}, /* 24 */
    {11 * 32 + 16 * 8,  7}, /* 25 */
    {11 * 32 + 16 * 9,  6}, /* 26 */
    {11 * 32 + 16 * 10, 5}, /* 27 */
    {11 * 32 + 16 * 11, 4}, /* 28 */
    {11 * 32 + 16 * 12, 3}, /* 29 */
    {11 * 32 + 16 * 13, 2}, /* 30 */
    {11 * 32 + 16 * 14, 1}  /* 31 */
};

static inline void chunkmemset_2(uint8_t *from, chunk_t *chunk) {
    int16_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm256_set1_epi16(tmp);
}

static inline void chunkmemset_4(uint8_t *from, chunk_t *chunk) {
    int32_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm256_set1_epi32(tmp);
}

static inline void chunkmemset_8(uint8_t *from, chunk_t *chunk) {
    int64_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm256_set1_epi64x(tmp);
}

static inline void loadchunk(uint8_t const *s, chunk_t *chunk) {
    *chunk = _mm256_loadu_si256((__m256i *)s);
}

static inline void storechunk(uint8_t *out, chunk_t *chunk) {
    _mm256_storeu_si256((__m256i *)out, *chunk);
}

static inline chunk_t GET_CHUNK_MAG(uint8_t *buf, uint32_t *chunk_rem, uint32_t dist) {
    lut_rem_pair lut_rem = perm_idx_lut[dist - 3];
    __m256i ret_vec;
    /* While technically we only need to read 4 or 8 bytes into this vector register for a lot of cases, GCC is
     * compiling this to a shared load for all branches, preferring the simpler code.  Given that the buf value isn't in
     * GPRs to begin with the 256 bit load is _probably_ just as inexpensive */
    *chunk_rem = lut_rem.remval;

    /* See note in chunkset_ssse3.c for why this is ok */
    __msan_unpoison(buf + dist, 32 - dist);

    if (dist < 16) {
        /* This simpler case still requires us to shuffle in 128 bit lanes, so we must apply a static offset after
         * broadcasting the first vector register to both halves. This is _marginally_ faster than doing two separate
         * shuffles and combining the halves later */
        const __m256i permute_xform =
            _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16);
        __m256i perm_vec = _mm256_load_si256((__m256i*)(permute_table+lut_rem.idx));
        __m128i ret_vec0 = _mm_loadu_si128((__m128i*)buf);
        perm_vec = _mm256_add_epi8(perm_vec, permute_xform);
        ret_vec = _mm256_inserti128_si256(_mm256_castsi128_si256(ret_vec0), ret_vec0, 1);
        ret_vec = _mm256_shuffle_epi8(ret_vec, perm_vec);
    } else if (dist == 16) {
        __m128i ret_vec0 = _mm_loadu_si128((__m128i*)buf);
        return _mm256_inserti128_si256(_mm256_castsi128_si256(ret_vec0), ret_vec0, 1);
    } else {
        __m128i ret_vec0 = _mm_loadu_si128((__m128i*)buf);
        __m128i ret_vec1 = _mm_loadu_si128((__m128i*)(buf + 16));
        /* Take advantage of the fact that only the latter half of the 256 bit vector will actually differ */
        __m128i perm_vec1 = _mm_load_si128((__m128i*)(permute_table + lut_rem.idx));
        __m128i xlane_permutes = _mm_cmpgt_epi8(_mm_set1_epi8(16), perm_vec1);
        __m128i xlane_res  = _mm_shuffle_epi8(ret_vec0, perm_vec1);
        /* Since we can't wrap twice, we can simply keep the later half exactly how it is instead of having to _also_
         * shuffle those values */
        __m128i latter_half = _mm_blendv_epi8(ret_vec1, xlane_res, xlane_permutes);
        ret_vec = _mm256_inserti128_si256(_mm256_castsi128_si256(ret_vec0), latter_half, 1);
    }

    return ret_vec;
}

#define CHUNKSIZE        chunksize_avx2
#define CHUNKCOPY        chunkcopy_avx2
#define CHUNKUNROLL      chunkunroll_avx2
#define CHUNKMEMSET      chunkmemset_avx2
#define CHUNKMEMSET_SAFE chunkmemset_safe_avx2

#include "chunkset_tpl.h"

#define INFLATE_FAST     inflate_fast_avx2

#include "inffast_tpl.h"

#endif
