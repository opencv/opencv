/* chunkset_neon.c -- NEON inline functions to copy small data chunks.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef ARM_NEON
#include "neon_intrins.h"
#include "../../zbuild.h"
#include "../generic/chunk_permute_table.h"

typedef uint8x16_t chunk_t;

#define CHUNK_SIZE 16

#define HAVE_CHUNKMEMSET_2
#define HAVE_CHUNKMEMSET_4
#define HAVE_CHUNKMEMSET_8
#define HAVE_CHUNK_MAG

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
    uint16_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = vreinterpretq_u8_u16(vdupq_n_u16(tmp));
}

static inline void chunkmemset_4(uint8_t *from, chunk_t *chunk) {
    uint32_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = vreinterpretq_u8_u32(vdupq_n_u32(tmp));
}

static inline void chunkmemset_8(uint8_t *from, chunk_t *chunk) {
    uint64_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = vreinterpretq_u8_u64(vdupq_n_u64(tmp));
}

#define CHUNKSIZE        chunksize_neon
#define CHUNKCOPY        chunkcopy_neon
#define CHUNKUNROLL      chunkunroll_neon
#define CHUNKMEMSET      chunkmemset_neon
#define CHUNKMEMSET_SAFE chunkmemset_safe_neon

static inline void loadchunk(uint8_t const *s, chunk_t *chunk) {
    *chunk = vld1q_u8(s);
}

static inline void storechunk(uint8_t *out, chunk_t *chunk) {
    vst1q_u8(out, *chunk);
}

static inline chunk_t GET_CHUNK_MAG(uint8_t *buf, uint32_t *chunk_rem, uint32_t dist) {
    lut_rem_pair lut_rem = perm_idx_lut[dist - 3];
    *chunk_rem = lut_rem.remval;

    /* See note in chunkset_ssse3.c for why this is ok */
    __msan_unpoison(buf + dist, 16 - dist);

    /* This version of table is only available on aarch64 */
#if defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__aarch64__)
    uint8x16_t ret_vec = vld1q_u8(buf);

    uint8x16_t perm_vec = vld1q_u8(permute_table + lut_rem.idx);
    return vqtbl1q_u8(ret_vec, perm_vec);
#else
    uint8x8_t ret0, ret1, a, b, perm_vec0, perm_vec1;
    perm_vec0 = vld1_u8(permute_table + lut_rem.idx);
    perm_vec1 = vld1_u8(permute_table + lut_rem.idx + 8);
    a = vld1_u8(buf);
    b = vld1_u8(buf + 8);
    ret0 = vtbl1_u8(a, perm_vec0);
    uint8x8x2_t ab = {{a, b}};
    ret1 = vtbl2_u8(ab, perm_vec1);
    return vcombine_u8(ret0, ret1);
#endif
}

#include "chunkset_tpl.h"

#define INFLATE_FAST     inflate_fast_neon

#include "inffast_tpl.h"

#endif
