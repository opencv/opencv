/*
 * Merged Upsampling/Color Conversion (64-bit RVV 1.0)
 *
 * Copyright (C) 2015, 2026, D. R. Commander.
 * Copyright (C) 2022-2023, Institute of Software, Chinese Academy of Sciences.
 *                          Author:  Zhiyuan Tan
 * Copyright (C) 2025, Samsung Electronics Co., Ltd.
 *                     Author:  Filip Wasil
 * Copyright (C) 2025-2026, Olaf Bernstein.
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

/* This file is included by jdmerge-rvv.c. */


HIDDEN void
jsimd_h2v1_merged_upsample_rvv(JDIMENSION output_width, JSAMPIMAGE input_buf,
                               JDIMENSION in_row_group_ctr,
                               JSAMPARRAY output_buf)
{
  JSAMPROW outptr, inptr0, inptr1, inptr2;

  vuint8m1x2_t y;
  vuint8m1_t ye, yo, cb, cr;
  vint16m2_t ye16, yo16, cb16, cb16_2, cr16, r_y16, g_y16, b_y16;
  vint16m2_t re16, ro16, ge16, go16, be16, bo16;
  vuint16m2_t r16, g16, b16;
  vint32m4_t g_y32;
  vuint8m2_t r, g, b;
#if RGB_PIXELSIZE == 3
  vuint8m2x3_t rgb;
#else
  vuint8m2_t a = __riscv_vmv_v_x_u8m2(0xFF, output_width);
  vuint8m2x4_t rgba;
#endif
  int cols_remaining = (int)output_width;

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  while (cols_remaining > 0) {
    /* vl = the number of 16-bit elements that can be stored in a 2-deep RVV
     * register group, up to a maximum of cols_remaining.  For example, this
     * will be 32 if the register width (VLEN) is 256 and there are at least 64
     * samples left in the input row.
     */
    size_t vl = __riscv_vsetvl_e16m2((cols_remaining + 1) / 2);

    /* Load 2 * vl adjacent Y samples into two 1-deep 8-bit-element register
     * groups, one group for the even-numbered columns and another for the
     * odd-numbered columns, and widen each register group into a 2-deep signed
     * 16-bit-element register group.  (We can safely load 2 * vl samples,
     * because vl is set based on the downsampled width.)
     */
    y = __riscv_vlseg2e8_v_u8m1x2(inptr0, vl);
    ye = __riscv_vget_v_u8m1x2_u8m1(y, 0);
    yo = __riscv_vget_v_u8m1x2_u8m1(y, 1);
    ye16 =
      __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(ye, vl));
    yo16 =
      __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(yo, vl));

    /* Load Cb and Cr components into 1-deep 8-bit-element register groups, and
     * widen each component register group to a 2-deep signed 16-bit-element
     * register group.
     */
    cb = __riscv_vle8_v_u8m1(inptr1, vl);
    cb16 =
      __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(cb, vl));
    cb16 = __riscv_vsub_vx_i16m2(cb16, CENTERJSAMPLE, vl);
    cr = __riscv_vle8_v_u8m1(inptr2, vl);
    cr16 =
      __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(cr, vl));
    cr16 = __riscv_vsub_vx_i16m2(cr16, CENTERJSAMPLE, vl);

    /* (128 has already been subtracted from Cb and Cr.)
     *
     * (Original)
     * R = Y                + 1.40200 * Cr
     * G = Y - 0.34414 * Cb - 0.71414 * Cr
     * B = Y + 1.77200 * Cb
     *
     * (This implementation)
     * R = Y                + 0.40200 * Cr + Cr
     * G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
     * B = Y - 0.22800 * Cb + Cb + Cb
     */

    /* R-Y */
    r_y16 = __riscv_vsll_vx_i16m2(cr16, 1, vl);                    /* 2 * Cr */
    r_y16 = __riscv_vmulh_vx_i16m2(r_y16, F_0_402, vl);
                                                      /* 2 * Cr * FIX(0.402) */
    r_y16 = __riscv_vadd_vx_i16m2(r_y16, 1, vl);  /* 2 * Cr * FIX(0.402) + 1 */
    r_y16 = __riscv_vsra_vx_i16m2(r_y16, 1, vl);          /* Cr * FIX(0.402) */
    r_y16 = __riscv_vadd_vv_i16m2(r_y16, cr16, vl);       /* Cr * FIX(1.402) */
    /* Add Y */
    re16 = __riscv_vadd_vv_i16m2(r_y16, ye16, vl);
    ro16 = __riscv_vadd_vv_i16m2(r_y16, yo16, vl);
    /* Range limit. */
    re16 = __riscv_vmax_vx_i16m2(re16, 0, vl);
    re16 = __riscv_vmin_vx_i16m2(re16, MAXJSAMPLE, vl);
    ro16 = __riscv_vmax_vx_i16m2(ro16, 0, vl);
    ro16 = __riscv_vmin_vx_i16m2(ro16, MAXJSAMPLE, vl);
    /* Combine even-numbered and odd-numbered output components.  An equivalent
     * vzip would be nice here.  Instead we shift the odd-numbered components
     * left by 8 bits, OR them with the even-numbered components, and
     * reinterpret the result as a 2-deep 8-bit-element register group.
     */
    r16 = __riscv_vsll_vx_u16m2(__riscv_vreinterpret_v_i16m2_u16m2(ro16), 8,
                                vl);
    r16 = __riscv_vor_vv_u16m2(r16, __riscv_vreinterpret_v_i16m2_u16m2(re16),
                               vl);
    r = __riscv_vreinterpret_v_u16m2_u8m2(r16);

    /* B-Y */
    cb16_2 = __riscv_vsll_vx_i16m2(cb16, 1, vl);                   /* 2 * Cb */
    b_y16 = __riscv_vmulh_vx_i16m2(cb16_2, -F_0_228, vl);
                                                     /* 2 * Cb * -FIX(0.228) */
    b_y16 = __riscv_vadd_vx_i16m2(b_y16, 1, vl);
                                                 /* 2 * Cb * -FIX(0.228) + 1 */
    b_y16 = __riscv_vsra_vx_i16m2(b_y16, 1, vl);         /* Cb * -FIX(0.228) */
    b_y16 = __riscv_vadd_vv_i16m2(b_y16, cb16_2, vl);     /* Cb * FIX(1.772) */
    /* Add Y */
    be16 = __riscv_vadd_vv_i16m2(b_y16, ye16, vl);
    bo16 = __riscv_vadd_vv_i16m2(b_y16, yo16, vl);
    /* Range limit. */
    be16 = __riscv_vmax_vx_i16m2(be16, 0, vl);
    be16 = __riscv_vmin_vx_i16m2(be16, MAXJSAMPLE, vl);
    bo16 = __riscv_vmax_vx_i16m2(bo16, 0, vl);
    bo16 = __riscv_vmin_vx_i16m2(bo16, MAXJSAMPLE, vl);
    /* Combine even-numbered and odd-numbered output components. */
    b16 = __riscv_vsll_vx_u16m2(__riscv_vreinterpret_v_i16m2_u16m2(bo16), 8,
                                vl);
    b16 = __riscv_vor_vv_u16m2(b16, __riscv_vreinterpret_v_i16m2_u16m2(be16),
                               vl);
    b = __riscv_vreinterpret_v_u16m2_u8m2(b16);

    /* G-Y */
    g_y32 = __riscv_vwmul_vx_i32m4(cb16, -F_0_344, vl);
    g_y32 = __riscv_vwmacc_vx_i32m4(g_y32, F_0_285, cr16, vl);
    g_y16 = __riscv_vnclip_wx_i16m2(g_y32, SCALEBITS, __RISCV_VXRM_RNU, vl);
    g_y16 = __riscv_vsub_vv_i16m2(g_y16, cr16, vl);
    /* Add Y */
    ge16 = __riscv_vadd_vv_i16m2(g_y16, ye16, vl);
    go16 = __riscv_vadd_vv_i16m2(g_y16, yo16, vl);
    /* Range limit. */
    ge16 = __riscv_vmax_vx_i16m2(ge16, 0, vl);
    ge16 = __riscv_vmin_vx_i16m2(ge16, MAXJSAMPLE, vl);
    go16 = __riscv_vmax_vx_i16m2(go16, 0, vl);
    go16 = __riscv_vmin_vx_i16m2(go16, MAXJSAMPLE, vl);
    /* Combine even-numbered and odd-numbered output components. */
    g16 = __riscv_vsll_vx_u16m2(__riscv_vreinterpret_v_i16m2_u16m2(go16), 8,
                                vl);
    g16 = __riscv_vor_vv_u16m2(g16, __riscv_vreinterpret_v_i16m2_u16m2(ge16),
                               vl);
    g = __riscv_vreinterpret_v_u16m2_u8m2(g16);

#if RGB_PIXELSIZE == 3
    /* Combine components into three 2-deep 8-bit-element register groups (one
     * group per component.)
     */
    rgb = __riscv_vset_v_u8m2_u8m2x3(rgb, RGB_RED, r);
    rgb = __riscv_vset_v_u8m2_u8m2x3(rgb, RGB_GREEN, g);
    rgb = __riscv_vset_v_u8m2_u8m2x3(rgb, RGB_BLUE, b);
    /* Pack vl pixels into memory. */
    __riscv_vsseg3e8_v_u8m2x3(outptr, rgb, MIN(2 * (int)vl, cols_remaining));
#else
    /* Combine components into four 2-deep 8-bit-element register groups (one
     * group per component.)
     */
    rgba = __riscv_vset_v_u8m2_u8m2x4(rgba, RGB_RED, r);
    rgba = __riscv_vset_v_u8m2_u8m2x4(rgba, RGB_GREEN, g);
    rgba = __riscv_vset_v_u8m2_u8m2x4(rgba, RGB_BLUE, b);
    rgba = __riscv_vset_v_u8m2_u8m2x4(rgba, RGB_ALPHA, a);
    /* Pack 2 * vl pixels into memory. */
    __riscv_vsseg4e8_v_u8m2x4(outptr, rgba, MIN(2 * (int)vl, cols_remaining));
#endif

    outptr += 2 * vl * RGB_PIXELSIZE;
    inptr0 += 2 * vl;
    inptr1 += vl;
    inptr2 += vl;
    cols_remaining -= 2 * vl;
  }
}


HIDDEN void
jsimd_h2v2_merged_upsample_rvv(JDIMENSION output_width, JSAMPIMAGE input_buf,
                               JDIMENSION in_row_group_ctr,
                               JSAMPARRAY output_buf)
{
  JSAMPROW inptr, outptr;

  inptr = input_buf[0][in_row_group_ctr];
  outptr = output_buf[0];

  input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2];
  jsimd_h2v1_merged_upsample_rvv(output_width, input_buf, in_row_group_ctr,
                                 output_buf);

  input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2 + 1];
  output_buf[0] = output_buf[1];
  jsimd_h2v1_merged_upsample_rvv(output_width, input_buf, in_row_group_ctr,
                                 output_buf);

  input_buf[0][in_row_group_ctr] = inptr;
  output_buf[0] = outptr;
}
