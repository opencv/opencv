/* riscv_functions.h -- RISCV implementations for arch-specific functions.
 *
 * Copyright (C) 2023 SiFive, Inc. All rights reserved.
 * Contributed by Alex Chiang <alex.chiang@sifive.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef RISCV_FUNCTIONS_H_
#define RISCV_FUNCTIONS_H_

#ifdef RISCV_RVV
uint32_t adler32_rvv(uint32_t adler, const uint8_t *buf, size_t len);
uint32_t adler32_fold_copy_rvv(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
uint32_t chunksize_rvv(void);
uint8_t* chunkmemset_safe_rvv(uint8_t *out, unsigned dist, unsigned len, unsigned left);
uint32_t compare256_rvv(const uint8_t *src0, const uint8_t *src1);

uint32_t longest_match_rvv(deflate_state *const s, Pos cur_match);
uint32_t longest_match_slow_rvv(deflate_state *const s, Pos cur_match);
void slide_hash_rvv(deflate_state *s);
void inflate_fast_rvv(PREFIX3(stream) *strm, uint32_t start);
#endif

#ifdef DISABLE_RUNTIME_CPU_DETECTION
// RISCV - RVV
#  if defined(RISCV_RVV) && defined(__riscv_v) && defined(__linux__)
#    undef native_adler32
#    define native_adler32 adler32_rvv
#    undef native_adler32_fold_copy
#    define native_adler32_fold_copy adler32_fold_copy_rvv
#    undef native_chunkmemset_safe
#    define native_chunkmemset_safe chunkmemset_safe_rvv
#    undef native_chunksize
#    define native_chunksize chunksize_rvv
#    undef native_compare256
#    define native_compare256 compare256_rvv
#    undef native_inflate_fast
#    define native_inflate_fast inflate_fast_rvv
#    undef native_longest_match
#    define native_longest_match longest_match_rvv
#    undef native_longest_match_slow
#    define native_longest_match_slow longest_match_slow_rvv
#    undef native_slide_hash
#    define native_slide_hash slide_hash_rvv
#  endif
#endif

#endif /* RISCV_FUNCTIONS_H_ */
