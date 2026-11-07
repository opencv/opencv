/* power_functions.h -- POWER implementations for arch-specific functions.
 * Copyright (C) 2020 Matheus Castanho <msc@linux.ibm.com>, IBM
 * Copyright (C) 2021 Mika T. Lindqvist <postmaster@raasu.org>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef POWER_FUNCTIONS_H_
#define POWER_FUNCTIONS_H_

#ifdef PPC_VMX
uint32_t adler32_vmx(uint32_t adler, const uint8_t *buf, size_t len);
void slide_hash_vmx(deflate_state *s);
#endif

#ifdef POWER8_VSX
uint32_t adler32_power8(uint32_t adler, const uint8_t *buf, size_t len);
uint32_t chunksize_power8(void);
uint8_t* chunkmemset_safe_power8(uint8_t *out, unsigned dist, unsigned len, unsigned left);
uint32_t crc32_power8(uint32_t crc, const uint8_t *buf, size_t len);
void slide_hash_power8(deflate_state *s);
void inflate_fast_power8(PREFIX3(stream) *strm, uint32_t start);
#endif

#ifdef POWER9
uint32_t compare256_power9(const uint8_t *src0, const uint8_t *src1);
uint32_t longest_match_power9(deflate_state *const s, Pos cur_match);
uint32_t longest_match_slow_power9(deflate_state *const s, Pos cur_match);
#endif


#ifdef DISABLE_RUNTIME_CPU_DETECTION
// Power - VMX
#  if defined(PPC_VMX) && defined(__ALTIVEC__)
#    undef native_adler32
#    define native_adler32 adler32_vmx
#    undef native_slide_hash
#    define native_slide_hash slide_hash_vmx
#  endif
// Power8 - VSX
#  if defined(POWER8_VSX) && defined(_ARCH_PWR8) && defined(__VSX__)
#    undef native_adler32
#    define native_adler32 adler32_power8
#    undef native_chunkmemset_safe
#    define native_chunkmemset_safe chunkmemset_safe_power8
#    undef native_chunksize
#    define native_chunksize chunksize_power8
#    undef native_inflate_fast
#    define native_inflate_fast inflate_fast_power8
#    undef native_slide_hash
#    define native_slide_hash slide_hash_power8
#  endif
#  if defined(POWER8_VSX_CRC32) && defined(_ARCH_PWR8) && defined(__VSX__)
#    undef native_crc32
#    define native_crc32 crc32_power8
#  endif
// Power9
#  if defined(POWER9) && defined(_ARCH_PWR9)
#    undef native_compare256
#    define native_compare256 compare256_power9
#    undef native_longest_match
#    define native_longest_match longest_match_power9
#    undef native_longest_match_slow
#    define native_longest_match_slow longest_match_slow_power9
#  endif
#endif

#endif /* POWER_FUNCTIONS_H_ */
