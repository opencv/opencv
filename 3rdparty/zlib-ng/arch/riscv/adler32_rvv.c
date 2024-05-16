/* adler32_rvv.c - RVV version of adler32
 * Copyright (C) 2023 SiFive, Inc. All rights reserved.
 * Contributed by Alex Chiang <alex.chiang@sifive.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef RISCV_RVV

#include <riscv_vector.h>
#include <stdint.h>

#include "../../zbuild.h"
#include "../../adler32_p.h"

static inline uint32_t adler32_rvv_impl(uint32_t adler, uint8_t* restrict dst, const uint8_t *src, size_t len, int COPY) {
    /* split Adler-32 into component sums */
    uint32_t sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (len == 1) {
        if (COPY) memcpy(dst, src, 1);
        return adler32_len_1(adler, src, sum2);
    }

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (src == NULL)
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (len < 16) {
        if (COPY) memcpy(dst, src, len);
        return adler32_len_16(adler, src, len, sum2);
    }

    size_t left = len;
    size_t vl = __riscv_vsetvlmax_e8m1();
    vl = vl > 256 ? 256 : vl;
    vuint32m4_t v_buf32_accu = __riscv_vmv_v_x_u32m4(0, vl);
    vuint32m4_t v_adler32_prev_accu = __riscv_vmv_v_x_u32m4(0, vl);
    vuint16m2_t v_buf16_accu;

    /*
     * We accumulate 8-bit data, and to prevent overflow, we have to use a 32-bit accumulator.
     * However, adding 8-bit data into a 32-bit accumulator isn't efficient. We use 16-bit & 32-bit
     * accumulators to boost performance.
     *
     * The block_size is the largest multiple of vl that <= 256, because overflow would occur when
     * vl > 256 (255 * 256 <= UINT16_MAX).
     *
     * We accumulate 8-bit data into a 16-bit accumulator and then
     * move the data into the 32-bit accumulator at the last iteration.
     */
    size_t block_size = (256 / vl) * vl;
    size_t nmax_limit = (NMAX / block_size);
    size_t cnt = 0;
    while (left >= block_size) {
        v_buf16_accu = __riscv_vmv_v_x_u16m2(0, vl);
        size_t subprob = block_size;
        while (subprob > 0) {
            vuint8m1_t v_buf8 = __riscv_vle8_v_u8m1(src, vl);
            if (COPY) __riscv_vse8_v_u8m1(dst, v_buf8, vl);
            v_adler32_prev_accu = __riscv_vwaddu_wv_u32m4(v_adler32_prev_accu, v_buf16_accu, vl);
            v_buf16_accu = __riscv_vwaddu_wv_u16m2(v_buf16_accu, v_buf8, vl);
            src += vl;
            if (COPY) dst += vl;
            subprob -= vl;
        }
        v_adler32_prev_accu = __riscv_vmacc_vx_u32m4(v_adler32_prev_accu, block_size / vl, v_buf32_accu, vl);
        v_buf32_accu = __riscv_vwaddu_wv_u32m4(v_buf32_accu, v_buf16_accu, vl);
        left -= block_size;
        /* do modulo once each block of NMAX size */
        if (++cnt >= nmax_limit) {
            v_adler32_prev_accu = __riscv_vremu_vx_u32m4(v_adler32_prev_accu, BASE, vl);
            cnt = 0;
        }
    }
    /* the left len <= 256 now, we can use 16-bit accum safely */
    v_buf16_accu = __riscv_vmv_v_x_u16m2(0, vl);
    size_t res = left;
    while (left >= vl) {
        vuint8m1_t v_buf8 = __riscv_vle8_v_u8m1(src, vl);
        if (COPY) __riscv_vse8_v_u8m1(dst, v_buf8, vl);
        v_adler32_prev_accu = __riscv_vwaddu_wv_u32m4(v_adler32_prev_accu, v_buf16_accu, vl);
        v_buf16_accu = __riscv_vwaddu_wv_u16m2(v_buf16_accu, v_buf8, vl);
        src += vl;
        if (COPY) dst += vl;
        left -= vl;
    }
    v_adler32_prev_accu = __riscv_vmacc_vx_u32m4(v_adler32_prev_accu, res / vl, v_buf32_accu, vl);
    v_adler32_prev_accu = __riscv_vremu_vx_u32m4(v_adler32_prev_accu, BASE, vl);
    v_buf32_accu = __riscv_vwaddu_wv_u32m4(v_buf32_accu, v_buf16_accu, vl);

    vuint32m4_t v_seq = __riscv_vid_v_u32m4(vl);
    vuint32m4_t v_rev_seq = __riscv_vrsub_vx_u32m4(v_seq, vl, vl);
    vuint32m4_t v_sum32_accu = __riscv_vmul_vv_u32m4(v_buf32_accu, v_rev_seq, vl);

    v_sum32_accu = __riscv_vadd_vv_u32m4(v_sum32_accu, __riscv_vmul_vx_u32m4(v_adler32_prev_accu, vl, vl), vl);

    vuint32m1_t v_sum2_sum = __riscv_vmv_s_x_u32m1(0, vl);
    v_sum2_sum = __riscv_vredsum_vs_u32m4_u32m1(v_sum32_accu, v_sum2_sum, vl);
    uint32_t sum2_sum = __riscv_vmv_x_s_u32m1_u32(v_sum2_sum);

    sum2 += (sum2_sum + adler * (len - left));

    vuint32m1_t v_adler_sum = __riscv_vmv_s_x_u32m1(0, vl);
    v_adler_sum = __riscv_vredsum_vs_u32m4_u32m1(v_buf32_accu, v_adler_sum, vl);
    uint32_t adler_sum = __riscv_vmv_x_s_u32m1_u32(v_adler_sum);

    adler += adler_sum;

    while (left--) {
        if (COPY) *dst++ = *src;
        adler += *src++;
        sum2 += adler;
    }

    sum2 %= BASE;
    adler %= BASE;

    return adler | (sum2 << 16);
}

Z_INTERNAL uint32_t adler32_fold_copy_rvv(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len) {
    return adler32_rvv_impl(adler, dst, src, len, 1);
}

Z_INTERNAL uint32_t adler32_rvv(uint32_t adler, const uint8_t *buf, size_t len) {
    return adler32_rvv_impl(adler, NULL, buf, len, 0);
}

#endif // RISCV_RVV
