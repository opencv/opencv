/*
 * YCbCr-to-RGB Color Conversion (64-bit RVV 1.0)
 *
 * Copyright (C) 2022-2023, Institute of Software, Chinese Academy of Sciences.
 *                          Author:  Zhiyuan Tan
 * Copyright (C) 2025, Samsung Electronics Co., Ltd.
 *                     Author:  Filip Wasil
 * Copyright (C) 2025-2026, Olaf Bernstein.
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

/* This file is included by jdcolor-rvv.c */


HIDDEN void
jsimd_ycc_rgb_convert_rvv(JDIMENSION out_width, JSAMPIMAGE input_buf,
                          JDIMENSION input_row, JSAMPARRAY output_buf,
                          int num_rows)
{
  JSAMPROW outptr, inptr0, inptr1, inptr2;

  vuint8m2_t y, cb, cr;
  vint16m4_t y16, cb16, cb16_2, cr16, r16, g16, b16;
  vint32m8_t g32;
  vuint8m2_t r, g, b;
#if RGB_PIXELSIZE == 3
  vuint8m2x3_t rgb;
#else
  vuint8m2_t a = __riscv_vmv_v_x_u8m2(0xFF, out_width);
  vuint8m2x4_t rgba;
#endif

  while (--num_rows >= 0) {
    JDIMENSION cols_remaining = out_width;

    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;

    while (cols_remaining > 0) {
      /* vl = the number of 16-bit elements that can be stored in a 4-deep
       * RVV register group, up to a maximum of cols_remaining.  For example,
       * this will be 64 if the register width (VLEN) is 256 and there are at
       * least 64 samples left in the input row.
       */
      size_t vl = __riscv_vsetvl_e16m4(cols_remaining);

      /* Load Y, Cb, and Cr components into 2-deep 8-bit-element register
       * groups, and widen each component register group to a 4-deep signed
       * 16-bit-element register group.
       */
      y = __riscv_vle8_v_u8m2(inptr0, vl);
      y16 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(y, vl));
      cb = __riscv_vle8_v_u8m2(inptr1, vl);
      cb16 =
        __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(cb, vl));
      cb16 = __riscv_vsub_vx_i16m4(cb16, CENTERJSAMPLE, vl);
      cr = __riscv_vle8_v_u8m2(inptr2, vl);
      cr16 =
        __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(cr, vl));
      cr16 = __riscv_vsub_vx_i16m4(cr16, CENTERJSAMPLE, vl);

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

      r16 = __riscv_vsll_vx_i16m4(cr16, 1, vl);                    /* 2 * Cr */
      r16 = __riscv_vmulh_vx_i16m4(r16, F_0_402, vl);
                                                      /* 2 * Cr * FIX(0.402) */
      r16 = __riscv_vadd_vx_i16m4(r16, 1, vl);    /* 2 * Cr * FIX(0.402) + 1 */
      r16 = __riscv_vsra_vx_i16m4(r16, 1, vl);            /* Cr * FIX(0.402) */
      r16 = __riscv_vadd_vv_i16m4(r16, cr16, vl);         /* Cr * FIX(1.402) */
      r16 = __riscv_vadd_vv_i16m4(r16, y16, vl);      /* Y + Cr * FIX(1.402) */
      /* Range limit and narrow to 8-bit. */
      r16 = __riscv_vmax_vx_i16m4(r16, 0, vl);
      r = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(r16), 0,
                                  __RISCV_VXRM_RDN, vl);

      cb16_2 = __riscv_vsll_vx_i16m4(cb16, 1, vl);                 /* 2 * Cb */
      b16 = __riscv_vmulh_vx_i16m4(cb16_2, -F_0_228, vl);
                                                     /* 2 * Cb * -FIX(0.228) */
      b16 = __riscv_vadd_vx_i16m4(b16, 1, vl);
                                                 /* 2 * Cb * -FIX(0.228) + 1 */
      b16 = __riscv_vsra_vx_i16m4(b16, 1, vl);           /* Cb * -FIX(0.228) */
      b16 = __riscv_vadd_vv_i16m4(b16, cb16_2, vl);       /* Cb * FIX(1.772) */
      b16 = __riscv_vadd_vv_i16m4(b16, y16, vl);      /* Y + Cb * FIX(1.772) */
      /* Range limit and narrow to 8-bit. */
      b16 = __riscv_vmax_vx_i16m4(b16, 0, vl);
      b = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(b16), 0,
                                  __RISCV_VXRM_RDN, vl);

      g32 = __riscv_vwmul_vx_i32m8(cb16, -F_0_344, vl);
      g32 = __riscv_vwmacc_vx_i32m8(g32, F_0_285, cr16, vl);
      g16 = __riscv_vnclip_wx_i16m4(g32, SCALEBITS, __RISCV_VXRM_RNU, vl);
      g16 = __riscv_vadd_vv_i16m4(g16, y16, vl);
      g16 = __riscv_vsub_vv_i16m4(g16, cr16, vl);
      /* Range limit and narrow to 8-bit. */
      g16 = __riscv_vmax_vx_i16m4(g16, 0, vl);
      g = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(g16), 0,
                                  __RISCV_VXRM_RDN, vl);

#if RGB_PIXELSIZE == 3
      /* Combine components into three 2-deep 8-bit-element register groups
       * (one group per component.)
       */
      rgb = __riscv_vset_v_u8m2_u8m2x3(rgb, RGB_RED, r);
      rgb = __riscv_vset_v_u8m2_u8m2x3(rgb, RGB_GREEN, g);
      rgb = __riscv_vset_v_u8m2_u8m2x3(rgb, RGB_BLUE, b);
      /* Pack vl pixels into memory. */
      __riscv_vsseg3e8_v_u8m2x3(outptr, rgb, vl);
#else
      /* Combine components into four 2-deep 8-bit-element register groups (one
       * group per component.)
       */
      rgba = __riscv_vset_v_u8m2_u8m2x4(rgba, RGB_RED, r);
      rgba = __riscv_vset_v_u8m2_u8m2x4(rgba, RGB_GREEN, g);
      rgba = __riscv_vset_v_u8m2_u8m2x4(rgba, RGB_BLUE, b);
      rgba = __riscv_vset_v_u8m2_u8m2x4(rgba, RGB_ALPHA, a);
      /* Pack vl pixels into memory. */
      __riscv_vsseg4e8_v_u8m2x4(outptr, rgba, vl);
#endif

      outptr += vl * RGB_PIXELSIZE;
      inptr0 += vl;
      inptr1 += vl;
      inptr2 += vl;
      cols_remaining -= vl;
    }
  }
}
