#ifndef CRC32_BRAID_P_H_
#define CRC32_BRAID_P_H_

#include "zendian.h"

/* Define N */
#ifdef Z_TESTN
#  define N Z_TESTN
#else
#  define N 5
#endif
#if N < 1 || N > 6
#  error N must be in 1..6
#endif

/*
  Define W and the associated z_word_t type. If W is not defined, then a
  braided calculation is not used, and the associated tables and code are not
  compiled.
 */
#ifdef Z_TESTW
#  if Z_TESTW-1 != -1
#    define W Z_TESTW
#  endif
#else
#  ifndef W
#    if defined(__x86_64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64) || defined(__powerpc64__)
#      define W 8
#    else
#      define W 4
#    endif
#  endif
#endif
#ifdef W
#  if W == 8
     typedef uint64_t z_word_t;
#  else
#    undef W
#    define W 4
     typedef uint32_t z_word_t;
#  endif
#endif

#if BYTE_ORDER == LITTLE_ENDIAN
#  define ZSWAPWORD(word) (word)
#  define BRAID_TABLE crc_braid_table
#elif BYTE_ORDER == BIG_ENDIAN
#  if W == 8
#    define ZSWAPWORD(word) ZSWAP64(word)
#  elif W == 4
#    define ZSWAPWORD(word) ZSWAP32(word)
#  endif
#  define BRAID_TABLE crc_braid_big_table
#else
#  error "No endian defined"
#endif

#define DO1 c = crc_table[(c ^ *buf++) & 0xff] ^ (c >> 8)
#define DO8 DO1; DO1; DO1; DO1; DO1; DO1; DO1; DO1

/* CRC polynomial. */
#define POLY 0xedb88320         /* p(x) reflected, with x^32 implied */

#endif /* CRC32_BRAID_P_H_ */
