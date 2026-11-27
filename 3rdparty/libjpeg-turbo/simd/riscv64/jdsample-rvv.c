/*
 * Upsampling (64-bit RVV 1.0)
 *
 * Copyright (C) 2015, 2026, D. R. Commander.
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


HIDDEN void
jsimd_h2v1_fancy_upsample_rvv(int max_v_samp_factor,
                              JDIMENSION downsampled_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow;

  /* Suffix 'a' = adjacent input samples
   * Suffix 'o' = odd-numbered output components
   * Suffix 'e' = even-numbered output components
   */
  vuint8m4_t samples, samplesa, outo, oute;
  vuint16m8_t outo16, oute16;
  vuint8m4x2_t out;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    JDIMENSION cols_remaining = downsampled_width - 1;

    inptr = input_data[inrow];
    outptr = output_data[inrow];

    /* Special case for the first column */
    *outptr++ = *inptr;

    while (cols_remaining > 0) {
      /* vl = the number of 8-bit elements that can be stored in a 4-deep
       * RVV register group, up to a maximum of cols_remaining.  For example,
       * this will be 128 if the register width (VLEN) is 256 and there are at
       * least 128 samples left in the input row.
       */
      size_t vl = __riscv_vsetvl_e8m4(cols_remaining);

      /* Load samples (downsampled components) and adjacent samples into 4-deep
       * 8-bit-element register groups.
       */
      samples = __riscv_vle8_v_u8m4(inptr, vl);
      samplesa = __riscv_vslide1down_vx_u8m4(samples, inptr[vl], vl);

      /* Each sample is blended with a neighboring sample to make a 2x1 grid of
       * upsampled components, as follows:
       *
       * Left ("even") upsampled component =
       * 3/4 * sample +
       * 1/4 * left nearest neighbor sample
       *
       * Right ("odd") upsampled component =
       * 3/4 * sample +
       * 1/4 * right nearest neighbor sample
       *
       * Refer to the Neon implementation of this module for diagrams.
       *
       * We start with the odd-numbered upsampled components, because we've
       * already processed Column 0.  Thus, the even-numbered components
       * correspond to the next (adjacent) horizontal sample.
       *
       * Odd-numbered components are rounded toward infinity, and even-numbered
       * components are rounded toward 0.
       */

      /* outo[i] = (samples[i] * 3 + samplesa[i] + 2) / 4 */
      outo16 = __riscv_vwaddu_vx_u16m8(samplesa, 2, vl);
      outo16 = __riscv_vwmaccu_vx_u16m8(outo16, 3, samples, vl);
      /* Right shift by 2 (divide by 4) and truncate to 8-bit (no rounding.) */
      outo = __riscv_vnclipu_wx_u8m4(outo16, 2, __RISCV_VXRM_RDN, vl);

      /* oute[i] = (samplesa[i] * 3 + samples[i] + 1) / 4 */
      oute16 = __riscv_vwaddu_vx_u16m8(samples, 1, vl);
      oute16 = __riscv_vwmaccu_vx_u16m8(oute16, 3, samplesa, vl);
      /* Right shift by 2 (divide to 4) and truncate to 8-bit (no rounding.) */
      oute = __riscv_vnclipu_wx_u8m4(oute16, 2, __RISCV_VXRM_RDN, vl);

      /* Combine odd-numbered and even-numbered components into two 4-deep
       * 8-bit-element register groups.
       */
      out = __riscv_vcreate_v_u8m4x2(outo, oute);

      /* Interleave the components into memory. */
      __riscv_vsseg2e8_v_u8m4x2(outptr, out, vl);

      inptr += vl;
      outptr += 2 * vl;
      cols_remaining -= vl;
    }

    /* Special case for the last column */
    *outptr = *inptr;
  }
}


HIDDEN void
jsimd_h2v2_fancy_upsample_rvv(int max_v_samp_factor,
                              JDIMENSION downsampled_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr_1, inptr0, inptr1, outptr0, outptr1;
  int inrow, outrow;

  /* Suffix '_1' = previous input row
   * Suffix '0' = current input row
   * Suffix '1' = next input row
   * Suffix 'a' = adjacent input samples
   * Suffix 'o' = odd-numbered output components
   * Suffix 'e' = even-numbered output components
   * Suffix 'u' = upper output row
   * Suffix 'l' = lower output row
   */
  vuint8m2_t samples_1, samples_1a, samples0, samples0a, samples1, samples1a;
  vuint8m2_t outuo, outue, outlo, outle;
  vuint16m4_t outuo16, outue16, outlo16, outle16;
  vuint8m2x2_t outu, outl;

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {
    JDIMENSION cols_remaining = downsampled_width - 1;

    inptr_1 = input_data[inrow - 1];
    inptr0 = input_data[inrow];
    inptr1 = input_data[inrow + 1];
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    /* Special case for the first column */
    *outptr0++ = (((*inptr0) * 3 + (*inptr_1)) * 4 + 8) >> 4;
    *outptr1++ = (((*inptr0) * 3 + (*inptr1)) * 4 + 8) >> 4;

    while (cols_remaining > 0) {
      /* vl = the number of 8-bit elements that can be stored in a 2-deep
       * RVV register group, up to a maximum of cols_remaining.  For example,
       * this will be 64 if the register width (VLEN) is 256 and there are at
       * least 64 samples left in the input row.
       */
      size_t vl = __riscv_vsetvl_e8m2(cols_remaining);

      /* Load samples (downsampled components) and adjacent samples from the
       * previous, current, and next rows into 2-deep 8-bit-element register
       * groups.
       */
      samples_1 = __riscv_vle8_v_u8m2(inptr_1, vl);
      samples_1a = __riscv_vslide1down_vx_u8m2(samples_1, inptr_1[vl], vl);
      samples0 = __riscv_vle8_v_u8m2(inptr0, vl);
      samples0a = __riscv_vslide1down_vx_u8m2(samples0, inptr0[vl], vl);
      samples1 = __riscv_vle8_v_u8m2(inptr1, vl);
      samples1a = __riscv_vslide1down_vx_u8m2(samples1, inptr1[vl], vl);

      /* Each sample is blended with neighboring samples to make a 2x2 grid of
       * upsampled components, as follows:
       *
       * Upper left ("even") upsampled component =
       * 9/16 * sample +
       * 3/16 * upper nearest neighbor sample +
       * 3/16 * left nearest neighbor sample +
       * 1/16 * upper left diagonal nearest neighbor sample
       *
       * Upper right ("odd") upsampled component =
       * 9/16 * sample +
       * 3/16 * upper nearest neighbor sample +
       * 3/16 * right nearest neighbor sample +
       * 1/16 * upper right diagonal nearest neighbor sample
       *
       * Lower left ("even") upsampled component =
       * 9/16 * sample +
       * 3/16 * left nearest neighbor sample +
       * 3/16 * lower nearest neighbor sample +
       * 1/16 * lower left diagonal nearest neighbor sample
       *
       * Lower right ("odd") upsampled component =
       * 9/16 * sample +
       * 3/16 * right nearest neighbor sample +
       * 3/16 * lower nearest neighbor sample +
       * 1/16 * lower right diagonal nearest neighbor sample
       *
       * Refer to the Neon implementation of this module for diagrams.
       *
       * We start with the odd-numbered upsampled components, because we've
       * already processed Column 0.  Thus, the even-numbered components
       * correspond to the next (adjacent) horizontal sample.
       *
       * Odd-numbered components are rounded toward 0, and even-numbered
       * components are rounded toward infinity.
       */

      /* outuo[i] = ((samples0[i] * 3 + samples_1[i]) * 3 +
       *             (samples0a[i] * 3 + samples_1a[i]) + 7) / 16
       */
      outuo16 = __riscv_vwaddu_vx_u16m4(samples_1a, 7, vl);  /* bias */
      outuo16 = __riscv_vwmaccu_vx_u16m4(outuo16, 3, samples0a, vl);
      outuo16 = __riscv_vwmaccu_vx_u16m4(outuo16, 3, samples_1, vl);
      outuo16 = __riscv_vwmaccu_vx_u16m4(outuo16, 9, samples0, vl);
      outuo = __riscv_vnclipu_wx_u8m2(outuo16, 4, __RISCV_VXRM_RDN, vl);

      /* outue[i] = ((samples0a[i] * 3 + samples_1a[i]) * 3 +
       *             (samples0[i] * 3 + samples_1[i]) + 8) / 16
       */
      outue16 = __riscv_vwaddu_vx_u16m4(samples_1, 8, vl);  /* bias */
      outue16 = __riscv_vwmaccu_vx_u16m4(outue16, 3, samples0, vl);
      outue16 = __riscv_vwmaccu_vx_u16m4(outue16, 3, samples_1a, vl);
      outue16 = __riscv_vwmaccu_vx_u16m4(outue16, 9, samples0a, vl);
      outue = __riscv_vnclipu_wx_u8m2(outue16, 4, __RISCV_VXRM_RDN, vl);

      /* outlo[i] = ((samples0[i] * 3 + samples1[i]) * 3 +
       *             (samples0a[i] * 3 + samples1a[i]) + 7) / 16
       */
      outlo16 = __riscv_vwaddu_vx_u16m4(samples1a, 7, vl);  /* bias */
      outlo16 = __riscv_vwmaccu_vx_u16m4(outlo16, 3, samples0a, vl);
      outlo16 = __riscv_vwmaccu_vx_u16m4(outlo16, 3, samples1, vl);
      outlo16 = __riscv_vwmaccu_vx_u16m4(outlo16, 9, samples0, vl);
      outlo = __riscv_vnclipu_wx_u8m2(outlo16, 4, __RISCV_VXRM_RDN, vl);

      /* outle[i] = ((samples0a[i] * 3 + samples1a[i]) * 3 +
       *             (samples0[i] * 3 + samples1[i]) + 8) / 16
       */
      outle16 = __riscv_vwaddu_vx_u16m4(samples1, 8, vl);  /* bias */
      outle16 = __riscv_vwmaccu_vx_u16m4(outle16, 3, samples0, vl);
      outle16 = __riscv_vwmaccu_vx_u16m4(outle16, 3, samples1a, vl);
      outle16 = __riscv_vwmaccu_vx_u16m4(outle16, 9, samples0a, vl);
      outle = __riscv_vnclipu_wx_u8m2(outle16, 4, __RISCV_VXRM_RDN, vl);

      /* Combine odd-numbered and even-numbered components into two 2-deep
       * 8-bit-element register groups.
       */
      outu = __riscv_vcreate_v_u8m2x2(outuo, outue);
      outl = __riscv_vcreate_v_u8m2x2(outlo, outle);

      /* Interleave the components into memory. */
      __riscv_vsseg2e8_v_u8m2x2(outptr0, outu, vl);
      __riscv_vsseg2e8_v_u8m2x2(outptr1, outl, vl);

      inptr_1 += vl;
      inptr0 += vl;
      inptr1 += vl;
      outptr0 += 2 * vl;
      outptr1 += 2 * vl;
      cols_remaining -= vl;
    }

    /* Special case for the last column */
    *outptr0++ = (((*inptr0) * 3 + (*inptr_1)) * 4 + 7) >> 4;
    *outptr1++ = (((*inptr0) * 3 + (*inptr1)) * 4 + 7) >> 4;
  }
}


/* These are rarely used (mainly just for decompressing YCCK images.) */

HIDDEN void
jsimd_h2v1_upsample_rvv(int max_v_samp_factor, JDIMENSION output_width,
                        JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow;

  vuint8m4_t samples;
  vuint8m4x2_t out;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    int cols_remaining = (int)output_width;

    inptr = input_data[inrow];
    outptr = output_data[inrow];

    while (cols_remaining > 0) {
      /* vl = the number of 8-bit elements that can be stored in a 4-deep
       * RVV register group, up to a maximum of ceil(cols_remaining / 2).  For
       * example, this will be 128 if the register width (VLEN) is 256 and
       * there are at least 256 samples left in the output row.
       */
      size_t vl = __riscv_vsetvl_e8m4((cols_remaining + 1) / 2);

      /* Load samples (downsampled components) into a 4-deep 8-bit-element
       * register group.
       */
      samples = __riscv_vle8_v_u8m4(inptr, vl);

      /* Combine samples into two identical 4-deep 8-bit-element register
       * groups.
       */
      out = __riscv_vcreate_v_u8m4x2(samples, samples);

      /* Interleave the samples into memory, effectively duplicating each.
       * (This stores 2 * vl samples, which we can safely do because vl is set
       * based on the downsampled width.)
       */
      __riscv_vsseg2e8_v_u8m4x2(outptr, out, vl);

      inptr += vl;
      outptr += 2 * vl;
      cols_remaining -= 2 * vl;
    }
  }
}


HIDDEN void
jsimd_h2v2_upsample_rvv(int max_v_samp_factor, JDIMENSION output_width,
                        JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr0, outptr1;
  int inrow, outrow;

  vuint8m4_t samples;
  vuint8m4x2_t out;

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {
    int cols_remaining = (int)output_width;

    inptr = input_data[inrow];
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    while (cols_remaining > 0) {
      /* vl = the number of 8-bit elements that can be stored in a 4-deep
       * RVV register group, up to a maximum of ceil(cols_remaining / 2).  For
       * example, this will be 128 if the register width (VLEN) is 256 and
       * there are at least 256 samples left in the output row.
       */
      size_t vl = __riscv_vsetvl_e8m4((cols_remaining + 1) / 2);

      /* Load samples (downsampled components) into a 4-deep 8-bit-element
       * register group.
       */
      samples = __riscv_vle8_v_u8m4(inptr, vl);

      /* Combine samples into two identical 4-deep 8-bit-element register
       * groups.
       */
      out = __riscv_vcreate_v_u8m4x2(samples, samples);

      /* Interleave the samples into memory, effectively duplicating each.
       * (This stores 2 * vl samples from each row, which we can safely do
       * because vl is set based on the downsampled width.)
       */
      __riscv_vsseg2e8_v_u8m4x2(outptr0, out, vl);
      __riscv_vsseg2e8_v_u8m4x2(outptr1, out, vl);

      inptr += vl;
      outptr0 += 2 * vl;
      outptr1 += 2 * vl;
      cols_remaining -= 2 * vl;
    }
  }
}
