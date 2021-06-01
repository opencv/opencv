/*
 * jsimd_none.c
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2011, 2014, D. R. Commander.
 * Copyright (C) 2015-2016, 2018, Matthieu Darbois.
 * Copyright (C) 2020, Arm Limited.
 *
 * Based on the x86 SIMD extension for IJG JPEG library,
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
 * For conditions of distribution and use, see copyright notice in jsimdext.inc
 *
 * This file contains stubs for when there is no SIMD support available.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jsimd.h"
#include "jdct.h"
#include "jsimddct.h"

GLOBAL(int)
jsimd_can_rgb_ycc(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_rgb_gray(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_ycc_rgb(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_ycc_rgb565(void)
{
  return 0;
}

GLOBAL(int)
jsimd_c_can_null_convert(void)
{
  return 0;
}

GLOBAL(void)
jsimd_rgb_ycc_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                      JSAMPIMAGE output_buf, JDIMENSION output_row,
                      int num_rows)
{
}

GLOBAL(void)
jsimd_rgb_gray_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                       JSAMPIMAGE output_buf, JDIMENSION output_row,
                       int num_rows)
{
}

GLOBAL(void)
jsimd_ycc_rgb_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                      JDIMENSION input_row, JSAMPARRAY output_buf,
                      int num_rows)
{
}

GLOBAL(void)
jsimd_ycc_rgb565_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                         JDIMENSION input_row, JSAMPARRAY output_buf,
                         int num_rows)
{
}

GLOBAL(void)
jsimd_c_null_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                     JSAMPIMAGE output_buf, JDIMENSION output_row,
                     int num_rows)
{
}

GLOBAL(int)
jsimd_can_h2v2_downsample(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_h2v1_downsample(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_h2v2_smooth_downsample(void)
{
  return 0;
}

GLOBAL(void)
jsimd_h2v2_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
}

GLOBAL(void)
jsimd_h2v2_smooth_downsample(j_compress_ptr cinfo,
                             jpeg_component_info *compptr,
                             JSAMPARRAY input_data, JSAMPARRAY output_data)
{
}

GLOBAL(void)
jsimd_h2v1_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
}

GLOBAL(int)
jsimd_can_h2v2_upsample(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_h2v1_upsample(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_int_upsample(void)
{
  return 0;
}

GLOBAL(void)
jsimd_int_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                   JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
}

GLOBAL(void)
jsimd_h2v2_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
}

GLOBAL(void)
jsimd_h2v1_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
}

GLOBAL(int)
jsimd_can_h2v2_fancy_upsample(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_h2v1_fancy_upsample(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_h1v2_fancy_upsample(void)
{
  return 0;
}

GLOBAL(void)
jsimd_h2v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
}

GLOBAL(void)
jsimd_h2v1_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
}

GLOBAL(void)
jsimd_h1v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
}

GLOBAL(int)
jsimd_can_h2v2_merged_upsample(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_h2v1_merged_upsample(void)
{
  return 0;
}

GLOBAL(void)
jsimd_h2v2_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
}

GLOBAL(void)
jsimd_h2v1_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
}

GLOBAL(int)
jsimd_can_convsamp(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_convsamp_float(void)
{
  return 0;
}

GLOBAL(void)
jsimd_convsamp(JSAMPARRAY sample_data, JDIMENSION start_col,
               DCTELEM *workspace)
{
}

GLOBAL(void)
jsimd_convsamp_float(JSAMPARRAY sample_data, JDIMENSION start_col,
                     FAST_FLOAT *workspace)
{
}

GLOBAL(int)
jsimd_can_fdct_islow(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_fdct_ifast(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_fdct_float(void)
{
  return 0;
}

GLOBAL(void)
jsimd_fdct_islow(DCTELEM *data)
{
}

GLOBAL(void)
jsimd_fdct_ifast(DCTELEM *data)
{
}

GLOBAL(void)
jsimd_fdct_float(FAST_FLOAT *data)
{
}

GLOBAL(int)
jsimd_can_quantize(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_quantize_float(void)
{
  return 0;
}

GLOBAL(void)
jsimd_quantize(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
}

GLOBAL(void)
jsimd_quantize_float(JCOEFPTR coef_block, FAST_FLOAT *divisors,
                     FAST_FLOAT *workspace)
{
}

GLOBAL(int)
jsimd_can_idct_2x2(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_idct_4x4(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_idct_6x6(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_idct_12x12(void)
{
  return 0;
}

GLOBAL(void)
jsimd_idct_2x2(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
}

GLOBAL(void)
jsimd_idct_4x4(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
}

GLOBAL(void)
jsimd_idct_6x6(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
}

GLOBAL(void)
jsimd_idct_12x12(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
}

GLOBAL(int)
jsimd_can_idct_islow(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_idct_ifast(void)
{
  return 0;
}

GLOBAL(int)
jsimd_can_idct_float(void)
{
  return 0;
}

GLOBAL(void)
jsimd_idct_islow(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
}

GLOBAL(void)
jsimd_idct_ifast(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
}

GLOBAL(void)
jsimd_idct_float(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
}

GLOBAL(int)
jsimd_can_huff_encode_one_block(void)
{
  return 0;
}

GLOBAL(JOCTET *)
jsimd_huff_encode_one_block(void *state, JOCTET *buffer, JCOEFPTR block,
                            int last_dc_val, c_derived_tbl *dctbl,
                            c_derived_tbl *actbl)
{
  return NULL;
}

GLOBAL(int)
jsimd_can_encode_mcu_AC_first_prepare(void)
{
  return 0;
}

GLOBAL(void)
jsimd_encode_mcu_AC_first_prepare(const JCOEF *block,
                                  const int *jpeg_natural_order_start, int Sl,
                                  int Al, JCOEF *values, size_t *zerobits)
{
}

GLOBAL(int)
jsimd_can_encode_mcu_AC_refine_prepare(void)
{
  return 0;
}

GLOBAL(int)
jsimd_encode_mcu_AC_refine_prepare(const JCOEF *block,
                                   const int *jpeg_natural_order_start, int Sl,
                                   int Al, JCOEF *absvalues, size_t *bits)
{
  return 0;
}
