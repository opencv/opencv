#ifndef CRC32_BRAID_P_H_
#define CRC32_BRAID_P_H_

#include "zbuild.h"
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
#    if defined(__x86_64__) || defined(__aarch64__) || defined(__powerpc64__)
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

/* CRC polynomial. */
#define POLY 0xedb88320         /* p(x) reflected, with x^32 implied */

extern uint32_t PREFIX(crc32_braid)(uint32_t crc, const uint8_t *buf, size_t len);

#endif /* CRC32_BRAID_P_H_ */
