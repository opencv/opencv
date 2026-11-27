/*
 * Downsampling (64-bit RVV 1.0)
 *
 * Copyright (C) 2015, 2018, 2026, D. R. Commander.
 * Copyright (C) 2022-2023, Institute of Software, Chinese Academy of Sciences.
 *                          Author:  Zhiyuan Tan
 * Copyright (C) 2025, Samsung Electronics Co., Ltd.
 *                     Author:  Filip Wasil
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
#include "../common/jcsample.h"


HIDDEN void
jsimd_h2v1_downsample_rvv(JDIMENSION image_width, int max_v_samp_factor,
                          JDIMENSION v_samp_factor, JDIMENSION width_in_blocks,
                          JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  JDIMENSION outrow, output_cols = width_in_blocks * DCTSIZE;
  JSAMPROW inptr, outptr;

  vuint8m2x2_t samples;
  vuint8m2_t even, odd, out;
  vuint16m4_t bias, out16;

  /* Generate alternating pattern of 0s and 1s by masking the first bit of the
   * element indices.
   */
  bias = __riscv_vid_v_u16m4(output_cols);
  bias = __riscv_vand_vx_u16m4(bias, 1, output_cols);

  expand_right_edge(input_data, max_v_samp_factor, image_width,
                    output_cols * 2);

  for (outrow = 0; outrow < v_samp_factor; outrow++) {
    JDIMENSION cols_remaining = output_cols;

    inptr = input_data[outrow];
    outptr = output_data[outrow];

    while (cols_remaining > 0) {
      /* vl = the number of 16-bit elements that can be stored in a 4-deep
       * RVV register group, up to a maximum of cols_remaining.  For example,
       * this will be 64 if the register width (VLEN) is 256 and there are at
       * least 64 samples left in the output row.
       */
      size_t vl = __riscv_vsetvl_e16m4(cols_remaining);

      /* Load 2 * vl adjacent samples into two 2-deep 8-bit-element register
       * groups, one group for the even-numbered columns and another for the
       * odd-numbered columns.  (We can safely load 2 * vl samples, because vl
       * is set based on the downsampled width.)
       */
      samples = __riscv_vlseg2e8_v_u8m2x2(inptr, vl);
      even = __riscv_vget_v_u8m2x2_u8m2(samples, 0);
      odd = __riscv_vget_v_u8m2x2_u8m2(samples, 1);

      /* Add adjacent sample values and bias. */
      out16 = __riscv_vwaddu_wv_u16m4(bias, even, vl);
      out16 = __riscv_vwaddu_wv_u16m4(out16, odd, vl);

      /* Divide total by 2, narrow to 8-bit, and store to memory. */
      out = __riscv_vnsrl_wx_u8m2(out16, 1, vl);
      __riscv_vse8_v_u8m2(outptr, out, vl);

      inptr += 2 * vl;
      outptr += vl;
      cols_remaining -= vl;
    }
  }
}


HIDDEN void
jsimd_h2v2_downsample_rvv(JDIMENSION image_width, int max_v_samp_factor,
                          JDIMENSION v_samp_factor, JDIMENSION width_in_blocks,
                          JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  JDIMENSION inrow, outrow, output_cols = width_in_blocks * DCTSIZE;
  JSAMPROW inptr0, inptr1, outptr;

  vuint8m2x2_t samples0, samples1;
  vuint8m2_t even0, odd0, even1, odd1, out;
  vuint16m4_t bias, out16;

  /* Generate alternating pattern of 1s and 2s by masking the first bit of the
   * element indices and adding 1.
   */
  bias = __riscv_vid_v_u16m4(output_cols);
  bias = __riscv_vand_vx_u16m4(bias, 1, output_cols);
  bias = __riscv_vadd_vx_u16m4(bias, 1, output_cols);

  expand_right_edge(input_data, max_v_samp_factor, image_width,
                    output_cols * 2);

  for (inrow = 0, outrow = 0; outrow < v_samp_factor; inrow += 2, outrow++) {
    JDIMENSION cols_remaining = output_cols;

    inptr0 = input_data[inrow];
    inptr1 = input_data[inrow + 1];
    outptr = output_data[outrow];

    while (cols_remaining > 0) {
      /* vl = the number of 16-bit elements that can be stored in a 4-deep
       * RVV register group, up to a maximum of cols_remaining.  For example,
       * this will be 64 if the register width (VLEN) is 256 and there are at
       * least 64 samples left in the output row.
       */
      size_t vl = __riscv_vsetvl_e16m4(cols_remaining);

      /* Load 2 * vl adjacent samples from each row into two 2-deep
       * 8-bit-element register groups, one group for the even-numbered columns
       * and another for the odd-numbered columns.  (We can safely load 2 * vl
       * samples, because vl is set based on the downsampled width.)
       */
      samples0 = __riscv_vlseg2e8_v_u8m2x2(inptr0, vl);
      samples1 = __riscv_vlseg2e8_v_u8m2x2(inptr1, vl);
      even0 = __riscv_vget_v_u8m2x2_u8m2(samples0, 0);
      odd0 = __riscv_vget_v_u8m2x2_u8m2(samples0, 1);
      even1 = __riscv_vget_v_u8m2x2_u8m2(samples1, 0);
      odd1 = __riscv_vget_v_u8m2x2_u8m2(samples1, 1);

      /* Add adjacent sample values and bias. */
      out16 = __riscv_vwaddu_wv_u16m4(bias, even0, vl);
      out16 = __riscv_vwaddu_wv_u16m4(out16, odd0, vl);
      out16 = __riscv_vwaddu_wv_u16m4(out16, even1, vl);
      out16 = __riscv_vwaddu_wv_u16m4(out16, odd1, vl);

      /* Divide total by 4, narrow to 8-bit, and store to memory. */
      out = __riscv_vnsrl_wx_u8m2(out16, 2, vl);
      __riscv_vse8_v_u8m2(outptr, out, vl);

      inptr0 += 2 * vl;
      inptr1 += 2 * vl;
      outptr += vl;
      cols_remaining -= vl;
    }
  }
}
