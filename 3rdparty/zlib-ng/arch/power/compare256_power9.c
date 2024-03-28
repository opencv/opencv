/* compare256_power9.c - Power9 version of compare256
 * Copyright (C) 2019 Matheus Castanho <msc@linux.ibm.com>, IBM
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef POWER9
#include <altivec.h>
#include "../../zbuild.h"
#include "../../zendian.h"

/* Older versions of GCC misimplemented semantics for these bit counting builtins.
 * https://gcc.gnu.org/git/gitweb.cgi?p=gcc.git;h=3f30f2d1dbb3228b8468b26239fe60c2974ce2ac */
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 12)
#if BYTE_ORDER == LITTLE_ENDIAN
#  define zng_vec_vctzlsbb(vc, len) len = __builtin_vec_vctzlsbb(vc)
#else
#  define zng_vec_vctzlsbb(vc, len) len = __builtin_vec_vclzlsbb(vc)
#endif
#else
#  define zng_vec_vctzlsbb(vc, len) len = vec_cntlz_lsbb(vc)
#endif

static inline uint32_t compare256_power9_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0, cmplen;

    do {
        vector unsigned char vsrc0, vsrc1, vc;

        vsrc0 = *((vector unsigned char *)src0);
        vsrc1 = *((vector unsigned char *)src1);

        /* Compare 16 bytes at a time. Each byte of vc will be either
         * all ones or all zeroes, depending on the result of the comparison. */
        vc = (vector unsigned char)vec_cmpne(vsrc0, vsrc1);

        /* Since the index of matching bytes will contain only zeroes
         * on vc (since we used cmpne), counting the number of consecutive
         * bytes where LSB == 0 is the same as counting the length of the match. */
        zng_vec_vctzlsbb(vc, cmplen);
        if (cmplen != 16)
            return len + cmplen;

        src0 += 16, src1 += 16, len += 16;
    } while (len < 256);

   return 256;
}

Z_INTERNAL uint32_t compare256_power9(const uint8_t *src0, const uint8_t *src1) {
    return compare256_power9_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_power9
#define COMPARE256          compare256_power9_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_power9
#define COMPARE256          compare256_power9_static

#include "match_tpl.h"

#endif
