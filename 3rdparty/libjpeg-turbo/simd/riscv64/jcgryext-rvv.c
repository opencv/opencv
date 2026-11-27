/*
 * RGB-to-Grayscale Color Conversion (64-bit RVV 1.0)
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

/* This file is included by jcgray-rvv.c */


HIDDEN void
jsimd_rgb_gray_convert_rvv(JDIMENSION img_width, JSAMPARRAY input_buf,
                           JSAMPIMAGE output_buf, JDIMENSION output_row,
                           int num_rows)
{
  JSAMPROW inptr, outptr;

#if RGB_PIXELSIZE == 3
  vuint8m2x3_t rgb;
#else
  vuint8m2x4_t rgb;
#endif
  vuint8m2_t r, g, b, y;
  vuint16m4_t r16, g16, b16, y16;
  vuint32m8_t y32;

  while (--num_rows >= 0) {
    JDIMENSION cols_remaining = img_width;

    inptr = *input_buf++;
    outptr = output_buf[0][output_row];
    output_row++;

    while (cols_remaining > 0) {
      /* vl = the number of 16-bit elements that can be stored in a 4-deep
       * RVV register group, up to a maximum of cols_remaining.  For example,
       * this will be 64 if the register width (VLEN) is 256 and there are at
       * least 64 pixels left in the row.
       */
      size_t vl = __riscv_vsetvl_e16m4(cols_remaining);

#if RGB_PIXELSIZE == 3
      /* Unpack vl pixels from memory into three 2-deep 8-bit-element register
       * groups (one group per component.)
       */
      rgb = __riscv_vlseg3e8_v_u8m2x3(inptr, vl);
      /* Extract each component into its own 2-deep 8-bit-element register
       * group, depending on the component index.
       */
      r = __riscv_vget_v_u8m2x3_u8m2(rgb, RGB_RED);
      g = __riscv_vget_v_u8m2x3_u8m2(rgb, RGB_GREEN);
      b = __riscv_vget_v_u8m2x3_u8m2(rgb, RGB_BLUE);
#else
      /* Unpack vl pixels from memory into four 2-deep 8-bit-element register
       * groups (one group per component.)
       */
      rgb = __riscv_vlseg4e8_v_u8m2x4(inptr, vl);
      /* Extract each component into its own 2-deep 8-bit-element register
       * group, depending on the component index.
       */
      r = __riscv_vget_v_u8m2x4_u8m2(rgb, RGB_RED);
      g = __riscv_vget_v_u8m2x4_u8m2(rgb, RGB_GREEN);
      b = __riscv_vget_v_u8m2x4_u8m2(rgb, RGB_BLUE);
#endif

      /* Widen each component register group to a 4-deep 16-bit-element
       * register group.
       */
      r16 = __riscv_vzext_vf2_u16m4(r, vl);
      g16 = __riscv_vzext_vf2_u16m4(g, vl);
      b16 = __riscv_vzext_vf2_u16m4(b, vl);

      /* Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B */

      y32 = __riscv_vwmulu_vx_u32m8(r16, F_0_299, vl);
      y32 = __riscv_vwmaccu_vx_u32m8(y32, F_0_587, g16, vl);
      y32 = __riscv_vwmaccu_vx_u32m8(y32, F_0_114, b16, vl);
      /* Narrow to 16-bit and round. */
      y16 = __riscv_vnclipu_wx_u16m4(y32, SCALEBITS, __RISCV_VXRM_RNU, vl);
      /* Narrow to 8-bit and store to memory. */
      y = __riscv_vncvt_x_x_w_u8m2(y16, vl);
      __riscv_vse8_v_u8m2(outptr, y, vl);

      inptr += vl * RGB_PIXELSIZE;
      outptr += vl;
      cols_remaining -= vl;
    }
  }
}
