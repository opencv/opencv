/*
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2011, 2014-2016, 2018, 2020, 2022, 2025, D. R. Commander.
 * Copyright (C) 2014, Linaro Limited.
 * Copyright (C) 2015-2016, 2018, 2022, Matthieu Darbois.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 * Copyright (C) 2020, Arm Limited.
 * Copyright (C) 2022, Institute of Software, Chinese Academy of Sciences.
 *                     Author:  Zhiyuan Tan
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

#include "../src/jdct.h"
#include "../src/jchuff.h"
#include "jsimdconst.h"

/* Get a bitmask of supported SIMD instruction sets */

EXTERN(unsigned int) jpeg_simd_cpu_support(void);


/* RGB-to-YCbCr/Grayscale Color Conversion */

#define SET_SIMD_EXTRGB_COLOR_CONVERTER(cs, instrset) { \
  switch (cinfo->in_color_space) { \
    case JCS_EXT_RGB: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_extrgb_##cs##_convert_##instrset; \
      break; \
    case JCS_EXT_RGBX: \
    case JCS_EXT_RGBA: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_extrgbx_##cs##_convert_##instrset; \
      break; \
    case JCS_EXT_BGR: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_extbgr_##cs##_convert_##instrset; \
      break; \
    case JCS_EXT_BGRX: \
    case JCS_EXT_BGRA: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_extbgrx_##cs##_convert_##instrset; \
      break; \
    case JCS_EXT_XBGR: \
    case JCS_EXT_ABGR: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_extxbgr_##cs##_convert_##instrset; \
      break; \
    case JCS_EXT_XRGB: \
    case JCS_EXT_ARGB: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_extxrgb_##cs##_convert_##instrset; \
      break; \
    default: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_rgb_##cs##_convert_##instrset; \
  } \
}

#define DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(cs, instrset) \
EXTERN(void) jsimd_rgb_##cs##_convert_##instrset \
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, \
   JDIMENSION output_row, int num_rows); \
EXTERN(void) jsimd_extrgb_##cs##_convert_##instrset \
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, \
   JDIMENSION output_row, int num_rows); \
EXTERN(void) jsimd_extrgbx_##cs##_convert_##instrset \
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, \
   JDIMENSION output_row, int num_rows); \
EXTERN(void) jsimd_extbgr_##cs##_convert_##instrset \
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, \
   JDIMENSION output_row, int num_rows); \
EXTERN(void) jsimd_extbgrx_##cs##_convert_##instrset \
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, \
   JDIMENSION output_row, int num_rows); \
EXTERN(void) jsimd_extxbgr_##cs##_convert_##instrset \
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, \
   JDIMENSION output_row, int num_rows); \
EXTERN(void) jsimd_extxrgb_##cs##_convert_##instrset \
  (JDIMENSION img_width, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, \
   JDIMENSION output_row, int num_rows);

extern const int jconst_rgb_ycc_convert_avx2[];
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(ycc, avx2)
extern const int jconst_rgb_gray_convert_avx2[];
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(gray, avx2)

extern const int jconst_rgb_ycc_convert_sse2[];
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(ycc, sse2)
extern const int jconst_rgb_gray_convert_sse2[];
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(gray, sse2)

DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(ycc, mmx)
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(gray, mmx)

DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(ycc, neon)
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(gray, neon)

DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(ycc, altivec)
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(gray, altivec)

DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(ycc, rvv)
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(gray, rvv)

DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(ycc, mmi)
DEFINE_SIMD_EXTRGB_COLOR_CONVERTERS(gray, mmi)


/* YCbCr-to-RGB Color Conversion */

#define SET_SIMD_EXTRGB_COLOR_DECONVERTER(instrset) { \
  switch (cinfo->out_color_space) { \
    case JCS_EXT_RGB: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_ycc_extrgb_convert_##instrset; \
      break; \
    case JCS_EXT_RGBX: \
    case JCS_EXT_RGBA: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_ycc_extrgbx_convert_##instrset; \
      break; \
    case JCS_EXT_BGR: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_ycc_extbgr_convert_##instrset; \
      break; \
    case JCS_EXT_BGRX: \
    case JCS_EXT_BGRA: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_ycc_extbgrx_convert_##instrset; \
      break; \
    case JCS_EXT_XBGR: \
    case JCS_EXT_ABGR: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_ycc_extxbgr_convert_##instrset; \
      break; \
    case JCS_EXT_XRGB: \
    case JCS_EXT_ARGB: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_ycc_extxrgb_convert_##instrset; \
      break; \
    default: \
      cinfo->cconvert->color_convert_simd = \
        jsimd_ycc_rgb_convert_##instrset; \
  } \
}

#define DEFINE_SIMD_EXTRGB_COLOR_DECONVERTERS(instrset) \
EXTERN(void) jsimd_ycc_rgb_convert_##instrset \
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row, \
   JSAMPARRAY output_buf, int num_rows); \
EXTERN(void) jsimd_ycc_extrgb_convert_##instrset \
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row, \
   JSAMPARRAY output_buf, int num_rows); \
EXTERN(void) jsimd_ycc_extrgbx_convert_##instrset \
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row, \
   JSAMPARRAY output_buf, int num_rows); \
EXTERN(void) jsimd_ycc_extbgr_convert_##instrset \
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row, \
   JSAMPARRAY output_buf, int num_rows); \
EXTERN(void) jsimd_ycc_extbgrx_convert_##instrset \
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row, \
   JSAMPARRAY output_buf, int num_rows); \
EXTERN(void) jsimd_ycc_extxbgr_convert_##instrset \
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row, \
   JSAMPARRAY output_buf, int num_rows); \
EXTERN(void) jsimd_ycc_extxrgb_convert_##instrset \
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row, \
   JSAMPARRAY output_buf, int num_rows);

extern const int jconst_ycc_rgb_convert_avx2[];
DEFINE_SIMD_EXTRGB_COLOR_DECONVERTERS(avx2)

extern const int jconst_ycc_rgb_convert_sse2[];
DEFINE_SIMD_EXTRGB_COLOR_DECONVERTERS(sse2)

DEFINE_SIMD_EXTRGB_COLOR_DECONVERTERS(mmx)

DEFINE_SIMD_EXTRGB_COLOR_DECONVERTERS(neon)

DEFINE_SIMD_EXTRGB_COLOR_DECONVERTERS(altivec)

DEFINE_SIMD_EXTRGB_COLOR_DECONVERTERS(rvv)

DEFINE_SIMD_EXTRGB_COLOR_DECONVERTERS(mmi)


/* YCbCr-to-RGB565 Color Conversion */

EXTERN(void) jsimd_ycc_rgb565_convert_neon
  (JDIMENSION out_width, JSAMPIMAGE input_buf, JDIMENSION input_row,
   JSAMPARRAY output_buf, int num_rows);


/* Downsampling */

EXTERN(void) jsimd_h2v1_downsample_avx2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);
EXTERN(void) jsimd_h2v2_downsample_avx2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_sse2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);
EXTERN(void) jsimd_h2v2_downsample_sse2
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_mmx
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);
EXTERN(void) jsimd_h2v2_downsample_mmx
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_neon
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);
EXTERN(void) jsimd_h2v2_downsample_neon
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_altivec
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);
EXTERN(void) jsimd_h2v2_downsample_altivec
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v1_downsample_rvv
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);
EXTERN(void) jsimd_h2v2_downsample_rvv
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);

EXTERN(void) jsimd_h2v2_downsample_mmi
  (JDIMENSION image_width, int max_v_samp_factor, JDIMENSION v_samp_factor,
   JDIMENSION width_in_blocks, JSAMPARRAY input_data, JSAMPARRAY output_data);


/* Plain Upsampling */

EXTERN(void) jsimd_h2v1_upsample_avx2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_avx2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_sse2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_sse2
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_mmx
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_mmx
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_neon
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_neon
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_altivec
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_altivec
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_upsample_rvv
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_upsample_rvv
  (int max_v_samp_factor, JDIMENSION output_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);


/* Fancy (Smooth) Upsampling */

extern const int jconst_fancy_upsample_avx2[];
EXTERN(void) jsimd_h2v1_fancy_upsample_avx2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_avx2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

extern const int jconst_fancy_upsample_sse2[];
EXTERN(void) jsimd_h2v1_fancy_upsample_sse2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_sse2
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_mmx
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_mmx
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_neon
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_neon
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h1v2_fancy_upsample_neon
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_altivec
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_altivec
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_rvv
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_rvv
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);

EXTERN(void) jsimd_h2v1_fancy_upsample_mmi
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);
EXTERN(void) jsimd_h2v2_fancy_upsample_mmi
  (int max_v_samp_factor, JDIMENSION downsampled_width, JSAMPARRAY input_data,
   JSAMPARRAY *output_data_ptr);


/* Merged Upsampling/Color Conversion */

#define SET_SIMD_EXTRGB_MERGED_UPSAMPLER(samp, instrset) { \
  switch (cinfo->out_color_space) { \
    case JCS_EXT_RGB: \
      cinfo->upsample->merged_upsample_simd = \
        jsimd_##samp##_extrgb_merged_upsample_##instrset; \
      break; \
    case JCS_EXT_RGBX: \
    case JCS_EXT_RGBA: \
      cinfo->upsample->merged_upsample_simd = \
        jsimd_##samp##_extrgbx_merged_upsample_##instrset; \
      break; \
    case JCS_EXT_BGR: \
      cinfo->upsample->merged_upsample_simd = \
        jsimd_##samp##_extbgr_merged_upsample_##instrset; \
      break; \
    case JCS_EXT_BGRX: \
    case JCS_EXT_BGRA: \
      cinfo->upsample->merged_upsample_simd = \
        jsimd_##samp##_extbgrx_merged_upsample_##instrset; \
      break; \
    case JCS_EXT_XBGR: \
    case JCS_EXT_ABGR: \
      cinfo->upsample->merged_upsample_simd = \
        jsimd_##samp##_extxbgr_merged_upsample_##instrset; \
      break; \
    case JCS_EXT_XRGB: \
    case JCS_EXT_ARGB: \
      cinfo->upsample->merged_upsample_simd = \
        jsimd_##samp##_extxrgb_merged_upsample_##instrset; \
      break; \
    default: \
      cinfo->upsample->merged_upsample_simd = \
        jsimd_##samp##_merged_upsample_##instrset; \
  } \
}

#define DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(samp, instrset) \
EXTERN(void) jsimd_##samp##_merged_upsample_##instrset \
  (JDIMENSION output_width, JSAMPIMAGE input_buf, \
   JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf); \
EXTERN(void) jsimd_##samp##_extrgb_merged_upsample_##instrset \
  (JDIMENSION output_width, JSAMPIMAGE input_buf, \
   JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf); \
EXTERN(void) jsimd_##samp##_extrgbx_merged_upsample_##instrset \
  (JDIMENSION output_width, JSAMPIMAGE input_buf, \
   JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf); \
EXTERN(void) jsimd_##samp##_extbgr_merged_upsample_##instrset \
  (JDIMENSION output_width, JSAMPIMAGE input_buf, \
   JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf); \
EXTERN(void) jsimd_##samp##_extbgrx_merged_upsample_##instrset \
  (JDIMENSION output_width, JSAMPIMAGE input_buf, \
   JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf); \
EXTERN(void) jsimd_##samp##_extxbgr_merged_upsample_##instrset \
  (JDIMENSION output_width, JSAMPIMAGE input_buf, \
   JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf); \
EXTERN(void) jsimd_##samp##_extxrgb_merged_upsample_##instrset \
  (JDIMENSION output_width, JSAMPIMAGE input_buf, \
   JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf);

extern const int jconst_merged_upsample_avx2[];
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v1, avx2)
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v2, avx2)

extern const int jconst_merged_upsample_sse2[];
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v1, sse2)
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v2, sse2)

DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v1, mmx)
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v2, mmx)

DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v1, neon)
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v2, neon)

DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v1, altivec)
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v2, altivec)

DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v1, rvv)
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v2, rvv)

DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v1, mmi)
DEFINE_SIMD_EXTRGB_MERGED_UPSAMPLERS(h2v2, mmi)


/* Integer Sample Conversion */

EXTERN(void) jsimd_convsamp_avx2
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_sse2
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_mmx
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_neon
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_altivec
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);

EXTERN(void) jsimd_convsamp_rvv
  (JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace);


/* Floating Point Sample Conversion */

EXTERN(void) jsimd_convsamp_float_sse2
  (JSAMPARRAY sample_data, JDIMENSION start_col, FAST_FLOAT *workspace);

EXTERN(void) jsimd_convsamp_float_sse
  (JSAMPARRAY sample_data, JDIMENSION start_col, FAST_FLOAT *workspace);

EXTERN(void) jsimd_convsamp_float_3dnow
  (JSAMPARRAY sample_data, JDIMENSION start_col, FAST_FLOAT *workspace);


/* Integer Forward DCT */

extern const int jconst_fdct_islow_avx2[];
EXTERN(void) jsimd_fdct_islow_avx2(DCTELEM *data);

extern const int jconst_fdct_islow_sse2[];
EXTERN(void) jsimd_fdct_islow_sse2(DCTELEM *data);
extern const int jconst_fdct_ifast_sse2[];
EXTERN(void) jsimd_fdct_ifast_sse2(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_mmx(DCTELEM *data);
EXTERN(void) jsimd_fdct_ifast_mmx(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_neon(DCTELEM *data);
EXTERN(void) jsimd_fdct_ifast_neon(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_altivec(DCTELEM *data);
EXTERN(void) jsimd_fdct_ifast_altivec(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_rvv(DCTELEM *data);
EXTERN(void) jsimd_fdct_ifast_rvv(DCTELEM *data);

EXTERN(void) jsimd_fdct_islow_mmi(DCTELEM *data);
EXTERN(void) jsimd_fdct_ifast_mmi(DCTELEM *data);


/* Floating Point Forward DCT */

extern const int jconst_fdct_float_sse[];
EXTERN(void) jsimd_fdct_float_sse(FAST_FLOAT *data);

EXTERN(void) jsimd_fdct_float_3dnow(FAST_FLOAT *data);


/* Integer Quantization */

EXTERN(void) jsimd_quantize_avx2
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_sse2
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_mmx
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_neon
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_altivec
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_rvv
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);

EXTERN(void) jsimd_quantize_mmi
  (JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace);


/* Floating Point Quantization */

EXTERN(void) jsimd_quantize_float_sse2
  (JCOEFPTR coef_block, FAST_FLOAT *divisors, FAST_FLOAT *workspace);

EXTERN(void) jsimd_quantize_float_sse
  (JCOEFPTR coef_block, FAST_FLOAT *divisors, FAST_FLOAT *workspace);

EXTERN(void) jsimd_quantize_float_3dnow
  (JCOEFPTR coef_block, FAST_FLOAT *divisors, FAST_FLOAT *workspace);


/* Inverse DCT */

extern const int jconst_idct_islow_avx2[];
EXTERN(void) jsimd_idct_islow_avx2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

extern const int jconst_idct_islow_sse2[];
EXTERN(void) jsimd_idct_islow_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
extern const int jconst_idct_ifast_sse2[];
EXTERN(void) jsimd_idct_ifast_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
extern const int jconst_idct_float_sse2[];
EXTERN(void) jsimd_idct_float_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

extern const int jconst_idct_float_sse[];
EXTERN(void) jsimd_idct_float_sse
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_islow_mmx
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_ifast_mmx
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_float_3dnow
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_islow_neon
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_ifast_neon
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_islow_altivec
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_ifast_altivec
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_islow_rvv
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_ifast_rvv
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_islow_mmi
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_ifast_mmi
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);


/* Scaled Integer Inverse DCT */

extern const int jconst_idct_red_sse2[];
EXTERN(void) jsimd_idct_2x2_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_4x4_sse2
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_2x2_mmx
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_4x4_mmx
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);

EXTERN(void) jsimd_idct_2x2_neon
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);
EXTERN(void) jsimd_idct_4x4_neon
  (void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf,
   JDIMENSION output_col);


/* Huffman Encoding */

extern const int jconst_huff_encode_one_block[];
EXTERN(JOCTET *) jsimd_huff_encode_one_block_sse2
  (void *state, JOCTET *buffer, JCOEFPTR block, int last_dc_val, void *dctbl,
   void *actbl);

EXTERN(JOCTET *) jsimd_huff_encode_one_block_neon
  (void *state, JOCTET *buffer, JCOEFPTR block, int last_dc_val, void *dctbl,
   void *actbl);


/* Progressive Huffman Encoding */

EXTERN(void) jsimd_encode_mcu_AC_first_prepare_sse2
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *values, size_t *zerobits);
EXTERN(int) jsimd_encode_mcu_AC_refine_prepare_sse2
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *absvalues, size_t *bits);

EXTERN(void) jsimd_encode_mcu_AC_first_prepare_neon
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *values, size_t *zerobits);
EXTERN(int) jsimd_encode_mcu_AC_refine_prepare_neon
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *absvalues, size_t *bits);
