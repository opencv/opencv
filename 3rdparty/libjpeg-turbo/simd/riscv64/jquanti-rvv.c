/*
 * Integer Sample Conversion and Quantization (64-bit RVV 1.0)
 *
 * Copyright (C) 2022-2023, Institute of Software, Chinese Academy of Sciences.
 *                          Author:  Zhiyuan Tan
 * Copyright (C) 2025, Samsung Electronics Co., Ltd.
 *                     Author:  Filip Wasil
 * Copyright (C) 2026, D. R. Commander.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "../jsimdint.h"
#include <riscv_vector.h>


HIDDEN void
jsimd_convsamp_rvv(JSAMPARRAY sample_data, JDIMENSION start_col,
                   DCTELEM *workspace)
{
  /* The minimum register width (VLEN) for standard CPUs in RVV 1.0 is
   * 128 bits.  Thus, this should always be 8, meaning that only one pass is
   * required.
   */
  size_t vl = __riscv_vsetvl_e16m2(DCTSIZE);

  vuint8m1_t in0, in1, in2, in3, in4, in5, in6, in7;
  vint16m2_t row0, row1, row2, row3, row4, row5, row6, row7;

  in0 = __riscv_vle8_v_u8m1(sample_data[0] + start_col, vl);
  in1 = __riscv_vle8_v_u8m1(sample_data[1] + start_col, vl);
  in2 = __riscv_vle8_v_u8m1(sample_data[2] + start_col, vl);
  in3 = __riscv_vle8_v_u8m1(sample_data[3] + start_col, vl);
  in4 = __riscv_vle8_v_u8m1(sample_data[4] + start_col, vl);
  in5 = __riscv_vle8_v_u8m1(sample_data[5] + start_col, vl);
  in6 = __riscv_vle8_v_u8m1(sample_data[6] + start_col, vl);
  in7 = __riscv_vle8_v_u8m1(sample_data[7] + start_col, vl);

  row0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(in0, vl));
  row1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(in1, vl));
  row2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(in2, vl));
  row3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(in3, vl));
  row4 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(in4, vl));
  row5 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(in5, vl));
  row6 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(in6, vl));
  row7 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(in7, vl));

  row0 = __riscv_vsub_vx_i16m2(row0, CENTERJSAMPLE, vl);
  row1 = __riscv_vsub_vx_i16m2(row1, CENTERJSAMPLE, vl);
  row2 = __riscv_vsub_vx_i16m2(row2, CENTERJSAMPLE, vl);
  row3 = __riscv_vsub_vx_i16m2(row3, CENTERJSAMPLE, vl);
  row4 = __riscv_vsub_vx_i16m2(row4, CENTERJSAMPLE, vl);
  row5 = __riscv_vsub_vx_i16m2(row5, CENTERJSAMPLE, vl);
  row6 = __riscv_vsub_vx_i16m2(row6, CENTERJSAMPLE, vl);
  row7 = __riscv_vsub_vx_i16m2(row7, CENTERJSAMPLE, vl);

  __riscv_vse16_v_i16m2(workspace + 0 * DCTSIZE, row0, vl);
  __riscv_vse16_v_i16m2(workspace + 1 * DCTSIZE, row1, vl);
  __riscv_vse16_v_i16m2(workspace + 2 * DCTSIZE, row2, vl);
  __riscv_vse16_v_i16m2(workspace + 3 * DCTSIZE, row3, vl);
  __riscv_vse16_v_i16m2(workspace + 4 * DCTSIZE, row4, vl);
  __riscv_vse16_v_i16m2(workspace + 5 * DCTSIZE, row5, vl);
  __riscv_vse16_v_i16m2(workspace + 6 * DCTSIZE, row6, vl);
  __riscv_vse16_v_i16m2(workspace + 7 * DCTSIZE, row7, vl);
}


HIDDEN void
jsimd_quantize_rvv(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
  int coeffs_remaining = DCTSIZE2;

  vint16m4_t in, shift, out;
  vuint16m4_t recip, corr, temp;
  vuint32m8_t product;
  vbool4_t mask;

  while (coeffs_remaining > 0) {
    /* vl = the number of 16-bit elements that can be stored in a 4-deep RVV
     * register group, up to a maximum of coeffs_remaining.  For example, this
     * will be 64 if the register width (VLEN) is 256, meaning that all 64
     * coefficients in the block can be processed in one pass.
     */
    size_t vl = __riscv_vsetvl_e16m4(coeffs_remaining);

    in = __riscv_vle16_v_i16m4(workspace, vl);
    recip = __riscv_vle16_v_u16m4((UDCTELEM *)divisors, vl);
    corr = __riscv_vle16_v_u16m4((UDCTELEM *)divisors + DCTSIZE2, vl);
    shift = __riscv_vle16_v_i16m4(divisors + 3 * DCTSIZE2, vl);

    /* mask[i] = in[i] < 0 ? 1 : 0 */
    mask = __riscv_vmslt_vx_i16m4_b4(in, 0, vl);
    /* Compute absolute value. */
    in = __riscv_vneg_v_i16m4_mu(mask, in, in, vl);
    temp = __riscv_vreinterpret_v_i16m4_u16m4(in);

    temp = __riscv_vadd_vv_u16m4(temp, corr, vl);
    product = __riscv_vwmulu_vv_u32m8(temp, recip, vl);
    shift = __riscv_vadd_vx_i16m4(shift, sizeof(DCTELEM) * 8, vl);
    temp = __riscv_vreinterpret_v_i16m4_u16m4(shift);
    temp = __riscv_vnsrl_wv_u16m4(product, temp, vl);

    out = __riscv_vreinterpret_v_u16m4_i16m4(temp);
    /* Restore sign to original product. */
    out = __riscv_vneg_v_i16m4_mu(mask, out, out, vl);
    __riscv_vse16_v_i16m4(coef_block, out, vl);

    workspace += vl;
    divisors += vl;
    coef_block += vl;
    coeffs_remaining -= vl;
  }
}
