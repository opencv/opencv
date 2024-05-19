/* cpu_features.h -- CPU architecture feature check
 * Copyright (C) 2017 Hans Kristian Rosbach
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef CPU_FEATURES_H_
#define CPU_FEATURES_H_

#include "adler32_fold.h"
#include "crc32_fold.h"

#if defined(X86_FEATURES)
#  include "arch/x86/x86_features.h"
#  include "fallback_builtins.h"
#elif defined(ARM_FEATURES)
#  include "arch/arm/arm_features.h"
#elif defined(PPC_FEATURES) || defined(POWER_FEATURES)
#  include "arch/power/power_features.h"
#elif defined(S390_FEATURES)
#  include "arch/s390/s390_features.h"
#elif defined(RISCV_FEATURES)
#  include "arch/riscv/riscv_features.h"
#endif

struct cpu_features {
#if defined(X86_FEATURES)
    struct x86_cpu_features x86;
#elif defined(ARM_FEATURES)
    struct arm_cpu_features arm;
#elif defined(PPC_FEATURES) || defined(POWER_FEATURES)
    struct power_cpu_features power;
#elif defined(S390_FEATURES)
    struct s390_cpu_features s390;
#elif defined(RISCV_FEATURES)
    struct riscv_cpu_features riscv;
#else
    char empty;
#endif
};

extern void cpu_check_features(struct cpu_features *features);

/* adler32 */
typedef uint32_t (*adler32_func)(uint32_t adler, const uint8_t *buf, size_t len);

extern uint32_t adler32_c(uint32_t adler, const uint8_t *buf, size_t len);
#ifdef ARM_NEON
extern uint32_t adler32_neon(uint32_t adler, const uint8_t *buf, size_t len);
#endif
#ifdef PPC_VMX
extern uint32_t adler32_vmx(uint32_t adler, const uint8_t *buf, size_t len);
#endif
#ifdef RISCV_RVV
extern uint32_t adler32_rvv(uint32_t adler, const uint8_t *buf, size_t len);
#endif
#ifdef X86_SSSE3
extern uint32_t adler32_ssse3(uint32_t adler, const uint8_t *buf, size_t len);
#endif
#ifdef X86_AVX2
extern uint32_t adler32_avx2(uint32_t adler, const uint8_t *buf, size_t len);
#endif
#ifdef X86_AVX512
extern uint32_t adler32_avx512(uint32_t adler, const uint8_t *buf, size_t len);
#endif
#ifdef X86_AVX512VNNI
extern uint32_t adler32_avx512_vnni(uint32_t adler, const uint8_t *buf, size_t len);
#endif
#ifdef POWER8_VSX
extern uint32_t adler32_power8(uint32_t adler, const uint8_t *buf, size_t len);
#endif

/* adler32 folding */
#ifdef RISCV_RVV
extern uint32_t adler32_fold_copy_rvv(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
#endif
#ifdef X86_SSE42
extern uint32_t adler32_fold_copy_sse42(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
#endif
#ifdef X86_AVX2
extern uint32_t adler32_fold_copy_avx2(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
#endif
#ifdef X86_AVX512
extern uint32_t adler32_fold_copy_avx512(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
#endif
#ifdef X86_AVX512VNNI
extern uint32_t adler32_fold_copy_avx512_vnni(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
#endif

/* CRC32 folding */
#ifdef X86_PCLMULQDQ_CRC
extern uint32_t crc32_fold_pclmulqdq_reset(crc32_fold *crc);
extern void     crc32_fold_pclmulqdq_copy(crc32_fold *crc, uint8_t *dst, const uint8_t *src, size_t len);
extern void     crc32_fold_pclmulqdq(crc32_fold *crc, const uint8_t *src, size_t len, uint32_t init_crc);
extern uint32_t crc32_fold_pclmulqdq_final(crc32_fold *crc);
extern uint32_t crc32_pclmulqdq(uint32_t crc32, const uint8_t *buf, size_t len);
#endif
#if defined(X86_PCLMULQDQ_CRC) && defined(X86_VPCLMULQDQ_CRC)
extern uint32_t crc32_fold_vpclmulqdq_reset(crc32_fold *crc);
extern void     crc32_fold_vpclmulqdq_copy(crc32_fold *crc, uint8_t *dst, const uint8_t *src, size_t len);
extern void     crc32_fold_vpclmulqdq(crc32_fold *crc, const uint8_t *src, size_t len, uint32_t init_crc);
extern uint32_t crc32_fold_vpclmulqdq_final(crc32_fold *crc);
extern uint32_t crc32_vpclmulqdq(uint32_t crc32, const uint8_t *buf, size_t len);
#endif

/* memory chunking */
extern uint32_t chunksize_c(void);
extern uint8_t* chunkmemset_safe_c(uint8_t *out, unsigned dist, unsigned len, unsigned left);
#ifdef X86_SSE2
extern uint32_t chunksize_sse2(void);
extern uint8_t* chunkmemset_safe_sse2(uint8_t *out, unsigned dist, unsigned len, unsigned left);
#endif
#ifdef X86_SSSE3
extern uint8_t* chunkmemset_safe_ssse3(uint8_t *out, unsigned dist, unsigned len, unsigned left);
#endif
#ifdef X86_AVX2
extern uint32_t chunksize_avx2(void);
extern uint8_t* chunkmemset_safe_avx2(uint8_t *out, unsigned dist, unsigned len, unsigned left);
#endif
#ifdef ARM_NEON
extern uint32_t chunksize_neon(void);
extern uint8_t* chunkmemset_safe_neon(uint8_t *out, unsigned dist, unsigned len, unsigned left);
#endif
#ifdef POWER8_VSX
extern uint32_t chunksize_power8(void);
extern uint8_t* chunkmemset_safe_power8(uint8_t *out, unsigned dist, unsigned len, unsigned left);
#endif
#ifdef RISCV_RVV
extern uint32_t chunksize_rvv(void);
extern uint8_t* chunkmemset_safe_rvv(uint8_t *out, unsigned dist, unsigned len, unsigned left);
#endif

#ifdef ZLIB_COMPAT
typedef struct z_stream_s z_stream;
#else
typedef struct zng_stream_s zng_stream;
#endif

/* inflate fast loop */
extern void inflate_fast_c(PREFIX3(stream) *strm, uint32_t start);
#ifdef X86_SSE2
extern void inflate_fast_sse2(PREFIX3(stream) *strm, uint32_t start);
#endif
#ifdef X86_SSSE3
extern void inflate_fast_ssse3(PREFIX3(stream) *strm, uint32_t start);
#endif
#ifdef X86_AVX2
extern void inflate_fast_avx2(PREFIX3(stream) *strm, uint32_t start);
#endif
#ifdef ARM_NEON
extern void inflate_fast_neon(PREFIX3(stream) *strm, uint32_t start);
#endif
#ifdef POWER8_VSX
extern void inflate_fast_power8(PREFIX3(stream) *strm, uint32_t start);
#endif
#ifdef RISCV_RVV
extern void inflate_fast_rvv(PREFIX3(stream) *strm, uint32_t start);
#endif

/* CRC32 */
typedef uint32_t (*crc32_func)(uint32_t crc32, const uint8_t *buf, size_t len);

extern uint32_t PREFIX(crc32_braid)(uint32_t crc, const uint8_t *buf, size_t len);
#ifdef ARM_ACLE
extern uint32_t crc32_acle(uint32_t crc, const uint8_t *buf, size_t len);
#elif defined(POWER8_VSX)
extern uint32_t crc32_power8(uint32_t crc, const uint8_t *buf, size_t len);
#elif defined(S390_CRC32_VX)
extern uint32_t crc32_s390_vx(uint32_t crc, const uint8_t *buf, size_t len);
#endif

/* compare256 */
typedef uint32_t (*compare256_func)(const uint8_t *src0, const uint8_t *src1);

extern uint32_t compare256_c(const uint8_t *src0, const uint8_t *src1);
#if defined(UNALIGNED_OK) && BYTE_ORDER == LITTLE_ENDIAN
extern uint32_t compare256_unaligned_16(const uint8_t *src0, const uint8_t *src1);
#ifdef HAVE_BUILTIN_CTZ
extern uint32_t compare256_unaligned_32(const uint8_t *src0, const uint8_t *src1);
#endif
#if defined(UNALIGNED64_OK) && defined(HAVE_BUILTIN_CTZLL)
extern uint32_t compare256_unaligned_64(const uint8_t *src0, const uint8_t *src1);
#endif
#endif
#if defined(X86_SSE2) && defined(HAVE_BUILTIN_CTZ)
extern uint32_t compare256_sse2(const uint8_t *src0, const uint8_t *src1);
#endif
#if defined(X86_AVX2) && defined(HAVE_BUILTIN_CTZ)
extern uint32_t compare256_avx2(const uint8_t *src0, const uint8_t *src1);
#endif
#if defined(ARM_NEON) && defined(HAVE_BUILTIN_CTZLL)
extern uint32_t compare256_neon(const uint8_t *src0, const uint8_t *src1);
#endif
#ifdef POWER9
extern uint32_t compare256_power9(const uint8_t *src0, const uint8_t *src1);
#endif
#ifdef RISCV_RVV
extern uint32_t compare256_rvv(const uint8_t *src0, const uint8_t *src1);
#endif

#ifdef DEFLATE_H_
/* insert_string */
extern void insert_string_c(deflate_state *const s, const uint32_t str, uint32_t count);
#ifdef X86_SSE42
extern void insert_string_sse42(deflate_state *const s, const uint32_t str, uint32_t count);
#elif defined(ARM_ACLE)
extern void insert_string_acle(deflate_state *const s, const uint32_t str, uint32_t count);
#endif

/* longest_match */
extern uint32_t longest_match_c(deflate_state *const s, Pos cur_match);
#if defined(UNALIGNED_OK) && BYTE_ORDER == LITTLE_ENDIAN
extern uint32_t longest_match_unaligned_16(deflate_state *const s, Pos cur_match);
#ifdef HAVE_BUILTIN_CTZ
extern uint32_t longest_match_unaligned_32(deflate_state *const s, Pos cur_match);
#endif
#if defined(UNALIGNED64_OK) && defined(HAVE_BUILTIN_CTZLL)
extern uint32_t longest_match_unaligned_64(deflate_state *const s, Pos cur_match);
#endif
#endif
#if defined(X86_SSE2) && defined(HAVE_BUILTIN_CTZ)
extern uint32_t longest_match_sse2(deflate_state *const s, Pos cur_match);
#endif
#if defined(X86_AVX2) && defined(HAVE_BUILTIN_CTZ)
extern uint32_t longest_match_avx2(deflate_state *const s, Pos cur_match);
#endif
#if defined(ARM_NEON) && defined(HAVE_BUILTIN_CTZLL)
extern uint32_t longest_match_neon(deflate_state *const s, Pos cur_match);
#endif
#ifdef POWER9
extern uint32_t longest_match_power9(deflate_state *const s, Pos cur_match);
#endif
#ifdef RISCV_RVV
extern uint32_t longest_match_rvv(deflate_state *const s, Pos cur_match);
#endif

/* longest_match_slow */
extern uint32_t longest_match_slow_c(deflate_state *const s, Pos cur_match);
#if defined(UNALIGNED_OK) && BYTE_ORDER == LITTLE_ENDIAN
extern uint32_t longest_match_slow_unaligned_16(deflate_state *const s, Pos cur_match);
extern uint32_t longest_match_slow_unaligned_32(deflate_state *const s, Pos cur_match);
#ifdef UNALIGNED64_OK
extern uint32_t longest_match_slow_unaligned_64(deflate_state *const s, Pos cur_match);
#endif
#endif
#if defined(X86_SSE2) && defined(HAVE_BUILTIN_CTZ)
extern uint32_t longest_match_slow_sse2(deflate_state *const s, Pos cur_match);
#endif
#if defined(X86_AVX2) && defined(HAVE_BUILTIN_CTZ)
extern uint32_t longest_match_slow_avx2(deflate_state *const s, Pos cur_match);
#endif
#if defined(ARM_NEON) && defined(HAVE_BUILTIN_CTZLL)
extern uint32_t longest_match_slow_neon(deflate_state *const s, Pos cur_match);
#endif
#ifdef POWER9
extern uint32_t longest_match_slow_power9(deflate_state *const s, Pos cur_match);
#endif
#ifdef RISCV_RVV
extern uint32_t longest_match_slow_rvv(deflate_state *const s, Pos cur_match);
#endif

/* quick_insert_string */
extern Pos quick_insert_string_c(deflate_state *const s, const uint32_t str);
#ifdef X86_SSE42
extern Pos quick_insert_string_sse42(deflate_state *const s, const uint32_t str);
#elif defined(ARM_ACLE)
extern Pos quick_insert_string_acle(deflate_state *const s, const uint32_t str);
#endif

/* slide_hash */
typedef void (*slide_hash_func)(deflate_state *s);

#ifdef X86_SSE2
extern void slide_hash_sse2(deflate_state *s);
#endif
#if defined(ARM_SIMD)
extern void slide_hash_armv6(deflate_state *s);
#endif
#if defined(ARM_NEON)
extern void slide_hash_neon(deflate_state *s);
#endif
#if defined(PPC_VMX)
extern void slide_hash_vmx(deflate_state *s);
#endif
#if defined(POWER8_VSX)
extern void slide_hash_power8(deflate_state *s);
#endif
#if defined(RISCV_RVV)
extern void slide_hash_rvv(deflate_state *s);
#endif
#ifdef X86_AVX2
extern void slide_hash_avx2(deflate_state *s);
#endif

/* update_hash */
extern uint32_t update_hash_c(deflate_state *const s, uint32_t h, uint32_t val);
#ifdef X86_SSE42
extern uint32_t update_hash_sse42(deflate_state *const s, uint32_t h, uint32_t val);
#elif defined(ARM_ACLE)
extern uint32_t update_hash_acle(deflate_state *const s, uint32_t h, uint32_t val);
#endif
#endif

#endif
