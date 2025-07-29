/* compare256_rvv.c - RVV version of compare256
 * Copyright (C) 2023 SiFive, Inc. All rights reserved.
 * Contributed by Alex Chiang <alex.chiang@sifive.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef RISCV_RVV

#include "zbuild.h"
#include "zutil_p.h"
#include "deflate.h"
#include "fallback_builtins.h"

#include <riscv_vector.h>

static inline uint32_t compare256_rvv_static(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;
    size_t vl;
    long found_diff;
    do {
        vl = __riscv_vsetvl_e8m4(256 - len);
        vuint8m4_t v_src0 = __riscv_vle8_v_u8m4(src0, vl);
        vuint8m4_t v_src1 = __riscv_vle8_v_u8m4(src1, vl);
        vbool2_t v_mask = __riscv_vmsne_vv_u8m4_b2(v_src0, v_src1, vl);
        found_diff = __riscv_vfirst_m_b2(v_mask, vl);
        if (found_diff >= 0)
            return len + (uint32_t)found_diff;
        src0 += vl, src1 += vl, len += vl;
    } while (len < 256);

    return 256;
}

Z_INTERNAL uint32_t compare256_rvv(const uint8_t *src0, const uint8_t *src1) {
    return compare256_rvv_static(src0, src1);
}

#define LONGEST_MATCH       longest_match_rvv
#define COMPARE256          compare256_rvv_static

#include "match_tpl.h"

#define LONGEST_MATCH_SLOW
#define LONGEST_MATCH       longest_match_slow_rvv
#define COMPARE256          compare256_rvv_static

#include "match_tpl.h"

#endif // RISCV_RVV
