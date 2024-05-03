/* chunkset_ssse3.c -- SSSE3 inline functions to copy small data chunks.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"

/* This requires SSE2 support. While it's implicit with SSSE3, we can minimize
 * code size by sharing the chunkcopy functions, which will certainly compile
 * to identical machine code */
#if defined(X86_SSSE3) && defined(X86_SSE2)
#include <immintrin.h>
#include "../generic/chunk_permute_table.h"

typedef __m128i chunk_t;

#define CHUNK_SIZE 16

#define HAVE_CHUNKMEMSET_2
#define HAVE_CHUNKMEMSET_4
#define HAVE_CHUNKMEMSET_8
#define HAVE_CHUNK_MAG
#define HAVE_CHUNKCOPY
#define HAVE_CHUNKUNROLL

static const lut_rem_pair perm_idx_lut[13] = {
    {0, 1},      /* 3 */
    {0, 0},      /* don't care */
    {1 * 32, 1}, /* 5 */
    {2 * 32, 4}, /* 6 */
    {3 * 32, 2}, /* 7 */
    {0 * 32, 0}, /* don't care */
    {4 * 32, 7}, /* 9 */
    {5 * 32, 6}, /* 10 */
    {6 * 32, 5}, /* 11 */
    {7 * 32, 4}, /* 12 */
    {8 * 32, 3}, /* 13 */
    {9 * 32, 2}, /* 14 */
    {10 * 32, 1},/* 15 */
};


static inline void chunkmemset_2(uint8_t *from, chunk_t *chunk) {
    int16_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm_set1_epi16(tmp);
}

static inline void chunkmemset_4(uint8_t *from, chunk_t *chunk) {
    int32_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm_set1_epi32(tmp);
}

static inline void chunkmemset_8(uint8_t *from, chunk_t *chunk) {
    int64_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm_set1_epi64x(tmp);
}

static inline void loadchunk(uint8_t const *s, chunk_t *chunk) {
    *chunk = _mm_loadu_si128((__m128i *)s);
}

static inline void storechunk(uint8_t *out, chunk_t *chunk) {
    _mm_storeu_si128((__m128i *)out, *chunk);
}

static inline chunk_t GET_CHUNK_MAG(uint8_t *buf, uint32_t *chunk_rem, uint32_t dist) {
    lut_rem_pair lut_rem = perm_idx_lut[dist - 3];
    __m128i perm_vec, ret_vec;
    /* Important to note:
     * This is _not_ to subvert the memory sanitizer but to instead unpoison some
     * bytes we willingly and purposefully load uninitialized that we swizzle over
     * in a vector register, anyway.  If what we assume is wrong about what is used,
     * the memory sanitizer will still usefully flag it */
    __msan_unpoison(buf + dist, 16 - dist);
    ret_vec = _mm_loadu_si128((__m128i*)buf);
    *chunk_rem = lut_rem.remval;

    perm_vec = _mm_load_si128((__m128i*)(permute_table + lut_rem.idx));
    ret_vec = _mm_shuffle_epi8(ret_vec, perm_vec);

    return ret_vec;
}

extern uint8_t* chunkcopy_sse2(uint8_t *out, uint8_t const *from, unsigned len);
extern uint8_t* chunkunroll_sse2(uint8_t *out, unsigned *dist, unsigned *len);

#define CHUNKSIZE        chunksize_ssse3
#define CHUNKMEMSET      chunkmemset_ssse3
#define CHUNKMEMSET_SAFE chunkmemset_safe_ssse3
#define CHUNKCOPY        chunkcopy_sse2
#define CHUNKUNROLL      chunkunroll_sse2

#include "chunkset_tpl.h"

#define INFLATE_FAST     inflate_fast_ssse3

#include "inffast_tpl.h"

#endif
