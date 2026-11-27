/*
 * Accurate Integer Forward DCT (64-bit RVV 1.0)
 *
 * Copyright (C) 2014, 2026, D. R. Commander.
 * Copyright (C) 2022-2023, Institute of Software, Chinese Academy of Sciences.
 *                          Author:  Zhiyuan Tan
 * Copyright (C) 2026 Chip Kerchner.
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
#include "jsimd_rvv.h"


#define F_0_298  2446   /* FIX(0.298631336) */
#define F_0_390  3196   /* FIX(0.390180644) */
#define F_0_541  4433   /* FIX(0.541196100) */
#define F_0_765  6270   /* FIX(0.765366865) */
#define F_0_899  7373   /* FIX(0.899976223) */
#define F_1_175  9633   /* FIX(1.175875602) */
#define F_1_501  12299  /* FIX(1.501321110) */
#define F_1_847  15137  /* FIX(1.847759065) */
#define F_1_961  16069  /* FIX(1.961570560) */
#define F_2_053  16819  /* FIX(2.053119869) */
#define F_2_562  20995  /* FIX(2.562915447) */
#define F_3_072  25172  /* FIX(3.072711026) */

#define CONST_BITS  13
#define PASS1_BITS  2
#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS)
#define ROUND_ADD(n)  (int32_t)1 << ((n) - 1)


#define DO_FDCT_COMMON_VLEN256(PASS) { \
  z1 = __riscv_vadd_vv_i16mf2(tmp12, tmp13, vl); \
  z1_32 = __riscv_vwmul_vx_i32m1(z1, F_0_541, vl); \
  z1_32 = __riscv_vadd_vx_i32m1(z1_32, ROUND_ADD(DESCALE_P##PASS), vl); \
  \
  out2_32 = __riscv_vwmacc_vx_i32m1(z1_32, F_0_765, tmp13, vl); \
  out2 = __riscv_vnsra_wx_i16mf2(out2_32, DESCALE_P##PASS, vl); \
  \
  out6_32 = __riscv_vwmacc_vx_i32m1(z1_32, -F_1_847, tmp12, vl); \
  out6 = __riscv_vnsra_wx_i16mf2(out6_32, DESCALE_P##PASS, vl); \
  \
  /* Odd part */ \
  z1 = __riscv_vadd_vv_i16mf2(tmp4, tmp7, vl); \
  z2 = __riscv_vadd_vv_i16mf2(tmp5, tmp6, vl); \
  z3 = __riscv_vadd_vv_i16mf2(tmp4, tmp6, vl); \
  z4 = __riscv_vadd_vv_i16mf2(tmp5, tmp7, vl); \
  z5 = __riscv_vadd_vv_i16mf2(z3, z4, vl); \
  z5_32 = __riscv_vwmul_vx_i32m1(z5, F_1_175, vl); \
  z5_32 = __riscv_vadd_vx_i32m1(z5_32, ROUND_ADD(DESCALE_P##PASS), vl); \
  \
  z1_32 = __riscv_vwmul_vx_i32m1(z1, -F_0_899, vl); \
  z2_32 = __riscv_vwmul_vx_i32m1(z2, -F_2_562, vl); \
  \
  z3_32 = __riscv_vwmacc_vx_i32m1(z5_32, -F_1_961, z3, vl); \
  z4_32 = __riscv_vwmacc_vx_i32m1(z5_32, -F_0_390, z4, vl); \
  \
  out7_32 = __riscv_vwmacc_vx_i32m1(z1_32, F_0_298, tmp4, vl); \
  out7_32 = __riscv_vadd_vv_i32m1(out7_32, z3_32, vl); \
  out7 = __riscv_vnsra_wx_i16mf2(out7_32, DESCALE_P##PASS, vl); \
  \
  out5_32 = __riscv_vwmacc_vx_i32m1(z2_32, F_2_053, tmp5, vl); \
  out5_32 = __riscv_vadd_vv_i32m1(out5_32, z4_32, vl); \
  out5 = __riscv_vnsra_wx_i16mf2(out5_32, DESCALE_P##PASS, vl); \
  \
  out3_32 = __riscv_vwmacc_vx_i32m1(z2_32, F_3_072, tmp6, vl); \
  out3_32 = __riscv_vadd_vv_i32m1(out3_32, z3_32, vl); \
  out3 = __riscv_vnsra_wx_i16mf2(out3_32, DESCALE_P##PASS, vl); \
  \
  out1_32 = __riscv_vwmacc_vx_i32m1(z1_32, F_1_501, tmp7, vl); \
  out1_32 = __riscv_vadd_vv_i32m1(out1_32, z4_32, vl); \
  out1 = __riscv_vnsra_wx_i16mf2(out1_32, DESCALE_P##PASS, vl); \
}


static void jsimd_fdct_islow_rvv_vlen256(DCTELEM *data)
{
  vint16mf2x4_t col0123, col4567;
  vint16mf2_t row0, row1, row2, row3, row4, row5, row6, row7,
    col0, col1, col2, col3, col4, col5, col6, col7,
    tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp10, tmp11, tmp12, tmp13,
    z1, z2, z3, z4, z5,
    out0, out1, out2, out3, out4, out5, out6, out7;
  vint32m1_t z1_32, z2_32, z3_32, z4_32, z5_32,
    out1_32, out2_32, out3_32, out5_32, out6_32, out7_32;

  /* VLEN >= 256, so this should always be 8. */
  size_t vl = __riscv_vsetvl_e16mf2(DCTSIZE),
    col_stride = DCTSIZE * sizeof(DCTELEM);

  /* Pass 1: process rows */

  /* Load and transpose row vectors to column vectors. */
  col0123 = __riscv_vlsseg4e16_v_i16mf2x4(data + 0, col_stride, vl);
  col4567 = __riscv_vlsseg4e16_v_i16mf2x4(data + 4, col_stride, vl);
  col0 = __riscv_vget_v_i16mf2x4_i16mf2(col0123, 0);
  col1 = __riscv_vget_v_i16mf2x4_i16mf2(col0123, 1);
  col2 = __riscv_vget_v_i16mf2x4_i16mf2(col0123, 2);
  col3 = __riscv_vget_v_i16mf2x4_i16mf2(col0123, 3);
  col4 = __riscv_vget_v_i16mf2x4_i16mf2(col4567, 0);
  col5 = __riscv_vget_v_i16mf2x4_i16mf2(col4567, 1);
  col6 = __riscv_vget_v_i16mf2x4_i16mf2(col4567, 2);
  col7 = __riscv_vget_v_i16mf2x4_i16mf2(col4567, 3);

  tmp0 = __riscv_vadd_vv_i16mf2(col0, col7, vl);
  tmp7 = __riscv_vsub_vv_i16mf2(col0, col7, vl);
  tmp1 = __riscv_vadd_vv_i16mf2(col1, col6, vl);
  tmp6 = __riscv_vsub_vv_i16mf2(col1, col6, vl);
  tmp2 = __riscv_vadd_vv_i16mf2(col2, col5, vl);
  tmp5 = __riscv_vsub_vv_i16mf2(col2, col5, vl);
  tmp3 = __riscv_vadd_vv_i16mf2(col3, col4, vl);
  tmp4 = __riscv_vsub_vv_i16mf2(col3, col4, vl);

  /* Even part */
  tmp10 = __riscv_vadd_vv_i16mf2(tmp0, tmp3, vl);
  tmp13 = __riscv_vsub_vv_i16mf2(tmp0, tmp3, vl);
  tmp11 = __riscv_vadd_vv_i16mf2(tmp1, tmp2, vl);
  tmp12 = __riscv_vsub_vv_i16mf2(tmp1, tmp2, vl);

  out0 = __riscv_vadd_vv_i16mf2(tmp10, tmp11, vl);
  out0 = __riscv_vsll_vx_i16mf2(out0, PASS1_BITS, vl);
  out4 = __riscv_vsub_vv_i16mf2(tmp10, tmp11, vl);
  out4 = __riscv_vsll_vx_i16mf2(out4, PASS1_BITS, vl);

  DO_FDCT_COMMON_VLEN256(1);

  /* Pass 2: process columns */

  /* Transpose column vectors to row vectors. */
  TRANSPOSE_8x8_VLEN256(out, row);

  tmp0 = __riscv_vadd_vv_i16mf2(row0, row7, vl);
  tmp7 = __riscv_vsub_vv_i16mf2(row0, row7, vl);
  tmp1 = __riscv_vadd_vv_i16mf2(row1, row6, vl);
  tmp6 = __riscv_vsub_vv_i16mf2(row1, row6, vl);
  tmp2 = __riscv_vadd_vv_i16mf2(row2, row5, vl);
  tmp5 = __riscv_vsub_vv_i16mf2(row2, row5, vl);
  tmp3 = __riscv_vadd_vv_i16mf2(row3, row4, vl);
  tmp4 = __riscv_vsub_vv_i16mf2(row3, row4, vl);

  /* Even part */
  tmp10 = __riscv_vadd_vv_i16mf2(tmp0, tmp3, vl);
  tmp13 = __riscv_vsub_vv_i16mf2(tmp0, tmp3, vl);
  tmp11 = __riscv_vadd_vv_i16mf2(tmp1, tmp2, vl);
  tmp12 = __riscv_vsub_vv_i16mf2(tmp1, tmp2, vl);
  tmp10 = __riscv_vadd_vx_i16mf2(tmp10, ROUND_ADD(PASS1_BITS), vl);

  out0 = __riscv_vadd_vv_i16mf2(tmp10, tmp11, vl);
  out0 = __riscv_vsra_vx_i16mf2(out0, PASS1_BITS, vl);
  out4 = __riscv_vsub_vv_i16mf2(tmp10, tmp11, vl);
  out4 = __riscv_vsra_vx_i16mf2(out4, PASS1_BITS, vl);

  DO_FDCT_COMMON_VLEN256(2);

  /* Store row vectors. */
  __riscv_vse16_v_i16mf2(data + 0 * DCTSIZE, out0, vl);
  __riscv_vse16_v_i16mf2(data + 1 * DCTSIZE, out1, vl);
  __riscv_vse16_v_i16mf2(data + 2 * DCTSIZE, out2, vl);
  __riscv_vse16_v_i16mf2(data + 3 * DCTSIZE, out3, vl);
  __riscv_vse16_v_i16mf2(data + 4 * DCTSIZE, out4, vl);
  __riscv_vse16_v_i16mf2(data + 5 * DCTSIZE, out5, vl);
  __riscv_vse16_v_i16mf2(data + 6 * DCTSIZE, out6, vl);
  __riscv_vse16_v_i16mf2(data + 7 * DCTSIZE, out7, vl);
}


#define DO_FDCT_COMMON(PASS) { \
  z1 = __riscv_vadd_vv_i16m1(tmp12, tmp13, vl); \
  z1_32 = __riscv_vwmul_vx_i32m2(z1, F_0_541, vl); \
  z1_32 = __riscv_vadd_vx_i32m2(z1_32, ROUND_ADD(DESCALE_P##PASS), vl); \
  \
  out2_32 = __riscv_vwmacc_vx_i32m2(z1_32, F_0_765, tmp13, vl); \
  out2 = __riscv_vnsra_wx_i16m1(out2_32, DESCALE_P##PASS, vl); \
  \
  out6_32 = __riscv_vwmacc_vx_i32m2(z1_32, -F_1_847, tmp12, vl); \
  out6 = __riscv_vnsra_wx_i16m1(out6_32, DESCALE_P##PASS, vl); \
  \
  /* Odd part */ \
  z1 = __riscv_vadd_vv_i16m1(tmp4, tmp7, vl); \
  z2 = __riscv_vadd_vv_i16m1(tmp5, tmp6, vl); \
  z3 = __riscv_vadd_vv_i16m1(tmp4, tmp6, vl); \
  z4 = __riscv_vadd_vv_i16m1(tmp5, tmp7, vl); \
  z5 = __riscv_vadd_vv_i16m1(z3, z4, vl); \
  z5_32 = __riscv_vwmul_vx_i32m2(z5, F_1_175, vl); \
  z5_32 = __riscv_vadd_vx_i32m2(z5_32, ROUND_ADD(DESCALE_P##PASS), vl); \
  \
  z1_32 = __riscv_vwmul_vx_i32m2(z1, -F_0_899, vl); \
  z2_32 = __riscv_vwmul_vx_i32m2(z2, -F_2_562, vl); \
  \
  z3_32 = __riscv_vwmacc_vx_i32m2(z5_32, -F_1_961, z3, vl); \
  z4_32 = __riscv_vwmacc_vx_i32m2(z5_32, -F_0_390, z4, vl); \
  \
  out7_32 = __riscv_vwmacc_vx_i32m2(z1_32, F_0_298, tmp4, vl); \
  out7_32 = __riscv_vadd_vv_i32m2(out7_32, z3_32, vl); \
  out7 = __riscv_vnsra_wx_i16m1(out7_32, DESCALE_P##PASS, vl); \
  \
  out5_32 = __riscv_vwmacc_vx_i32m2(z2_32, F_2_053, tmp5, vl); \
  out5_32 = __riscv_vadd_vv_i32m2(out5_32, z4_32, vl); \
  out5 = __riscv_vnsra_wx_i16m1(out5_32, DESCALE_P##PASS, vl); \
  \
  out3_32 = __riscv_vwmacc_vx_i32m2(z2_32, F_3_072, tmp6, vl); \
  out3_32 = __riscv_vadd_vv_i32m2(out3_32, z3_32, vl); \
  out3 = __riscv_vnsra_wx_i16m1(out3_32, DESCALE_P##PASS, vl); \
  \
  out1_32 = __riscv_vwmacc_vx_i32m2(z1_32, F_1_501, tmp7, vl); \
  out1_32 = __riscv_vadd_vv_i32m2(out1_32, z4_32, vl); \
  out1 = __riscv_vnsra_wx_i16m1(out1_32, DESCALE_P##PASS, vl); \
}


HIDDEN void
jsimd_fdct_islow_rvv(DCTELEM *data)
{
  vint16m1x4_t col0123, col4567;
  vint16m1_t row0, row1, row2, row3, row4, row5, row6, row7,
    col0, col1, col2, col3, col4, col5, col6, col7,
    tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp10, tmp11, tmp12, tmp13,
    z1, z2, z3, z4, z5,
    out0, out1, out2, out3, out4, out5, out6, out7;
  vint32m2_t z1_32, z2_32, z3_32, z4_32, z5_32,
    out1_32, out2_32, out3_32, out5_32, out6_32, out7_32;
  size_t vl, col_stride = DCTSIZE * sizeof(DCTELEM);

  if (__riscv_vsetvlmax_e16m1() >= DCTSIZE * 2) {
    jsimd_fdct_islow_rvv_vlen256(data);
    return;
  }

  /* The minimum register width (VLEN) for standard CPUs in RVV 1.0 is
   * 128 bits.  Thus, this should always be 8.
   */
  vl = __riscv_vsetvl_e16m1(DCTSIZE);

  /* Pass 1: process rows */

  /* Load and transpose row vectors to column vectors. */
  col0123 = __riscv_vlsseg4e16_v_i16m1x4(data + 0, col_stride, vl);
  col4567 = __riscv_vlsseg4e16_v_i16m1x4(data + 4, col_stride, vl);
  col0 = __riscv_vget_v_i16m1x4_i16m1(col0123, 0);
  col1 = __riscv_vget_v_i16m1x4_i16m1(col0123, 1);
  col2 = __riscv_vget_v_i16m1x4_i16m1(col0123, 2);
  col3 = __riscv_vget_v_i16m1x4_i16m1(col0123, 3);
  col4 = __riscv_vget_v_i16m1x4_i16m1(col4567, 0);
  col5 = __riscv_vget_v_i16m1x4_i16m1(col4567, 1);
  col6 = __riscv_vget_v_i16m1x4_i16m1(col4567, 2);
  col7 = __riscv_vget_v_i16m1x4_i16m1(col4567, 3);

  tmp0 = __riscv_vadd_vv_i16m1(col0, col7, vl);
  tmp7 = __riscv_vsub_vv_i16m1(col0, col7, vl);
  tmp1 = __riscv_vadd_vv_i16m1(col1, col6, vl);
  tmp6 = __riscv_vsub_vv_i16m1(col1, col6, vl);
  tmp2 = __riscv_vadd_vv_i16m1(col2, col5, vl);
  tmp5 = __riscv_vsub_vv_i16m1(col2, col5, vl);
  tmp3 = __riscv_vadd_vv_i16m1(col3, col4, vl);
  tmp4 = __riscv_vsub_vv_i16m1(col3, col4, vl);

  /* Even part */
  tmp10 = __riscv_vadd_vv_i16m1(tmp0, tmp3, vl);
  tmp13 = __riscv_vsub_vv_i16m1(tmp0, tmp3, vl);
  tmp11 = __riscv_vadd_vv_i16m1(tmp1, tmp2, vl);
  tmp12 = __riscv_vsub_vv_i16m1(tmp1, tmp2, vl);

  out0 = __riscv_vadd_vv_i16m1(tmp10, tmp11, vl);
  out0 = __riscv_vsll_vx_i16m1(out0, PASS1_BITS, vl);
  out4 = __riscv_vsub_vv_i16m1(tmp10, tmp11, vl);
  out4 = __riscv_vsll_vx_i16m1(out4, PASS1_BITS, vl);

  DO_FDCT_COMMON(1);

  /* Pass 2: process columns */

  /* Transpose column vectors to row vectors. */
  TRANSPOSE_8x8(out, row);

  tmp0 = __riscv_vadd_vv_i16m1(row0, row7, vl);
  tmp7 = __riscv_vsub_vv_i16m1(row0, row7, vl);
  tmp1 = __riscv_vadd_vv_i16m1(row1, row6, vl);
  tmp6 = __riscv_vsub_vv_i16m1(row1, row6, vl);
  tmp2 = __riscv_vadd_vv_i16m1(row2, row5, vl);
  tmp5 = __riscv_vsub_vv_i16m1(row2, row5, vl);
  tmp3 = __riscv_vadd_vv_i16m1(row3, row4, vl);
  tmp4 = __riscv_vsub_vv_i16m1(row3, row4, vl);

  /* Even part */
  tmp10 = __riscv_vadd_vv_i16m1(tmp0, tmp3, vl);
  tmp13 = __riscv_vsub_vv_i16m1(tmp0, tmp3, vl);
  tmp11 = __riscv_vadd_vv_i16m1(tmp1, tmp2, vl);
  tmp12 = __riscv_vsub_vv_i16m1(tmp1, tmp2, vl);
  tmp10 = __riscv_vadd_vx_i16m1(tmp10, ROUND_ADD(PASS1_BITS), vl);

  out0 = __riscv_vadd_vv_i16m1(tmp10, tmp11, vl);
  out0 = __riscv_vsra_vx_i16m1(out0, PASS1_BITS, vl);
  out4 = __riscv_vsub_vv_i16m1(tmp10, tmp11, vl);
  out4 = __riscv_vsra_vx_i16m1(out4, PASS1_BITS, vl);

  DO_FDCT_COMMON(2);

  /* Store row vectors. */
  __riscv_vse16_v_i16m1(data + 0 * DCTSIZE, out0, vl);
  __riscv_vse16_v_i16m1(data + 1 * DCTSIZE, out1, vl);
  __riscv_vse16_v_i16m1(data + 2 * DCTSIZE, out2, vl);
  __riscv_vse16_v_i16m1(data + 3 * DCTSIZE, out3, vl);
  __riscv_vse16_v_i16m1(data + 4 * DCTSIZE, out4, vl);
  __riscv_vse16_v_i16m1(data + 5 * DCTSIZE, out5, vl);
  __riscv_vse16_v_i16m1(data + 6 * DCTSIZE, out6, vl);
  __riscv_vse16_v_i16m1(data + 7 * DCTSIZE, out7, vl);
}
