/*
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2011, 2013-2014, 2016, 2018, 2020, 2022-2026,
 *           D. R. Commander.
 * Copyright (C) 2011, Nokia Corporation and/or its subsidiary(-ies).
 * Copyright (C) 2015-2016, 2018, 2022, Matthieu Darbois.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 * Copyright (C) 2019, Google LLC.
 * Copyright (C) 2020, Arm Limited.
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
 *
 * This file contains the interface between the "normal" portions
 * of the library and the SIMD implementations.
 */

#include "jsimddct.h"
#include "jsimd.h"
#include "jsimdint.h"

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386

/*
 * In the PIC cases, we have no guarantee that constants will keep
 * their alignment. This macro allows us to verify it at runtime.
 */
#define IS_ALIGNED(ptr, order)  (((JUINTPTR)ptr & ((1 << order) - 1)) == 0)

#define IS_ALIGNED_SSE(ptr)  (IS_ALIGNED(ptr, 4)) /* 16 byte alignment */
#define IS_ALIGNED_AVX(ptr)  (IS_ALIGNED(ptr, 5)) /* 32 byte alignment */

#endif


/*
 * Check what SIMD accelerations are supported.
 */
LOCAL(void)
init_simd(j_common_ptr cinfo)
{
#ifndef NO_GETENV
  char env[2] = { 0 };
#endif
  unsigned int simd_support = cinfo->is_decompressor ?
                              ((j_decompress_ptr)cinfo)->master->simd_support :
                              ((j_compress_ptr)cinfo)->master->simd_support;
  unsigned int simd_huffman = cinfo->is_decompressor ?
                              ((j_decompress_ptr)cinfo)->master->simd_huffman :
                              ((j_compress_ptr)cinfo)->master->simd_huffman;

  if (simd_support != JSIMD_UNDEFINED)
    return;

  simd_support = jpeg_simd_cpu_support();

#ifndef NO_GETENV
  /* Force different settings through environment variables */
#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (!GETENV_S(env, 2, "JSIMD_FORCESSE2") && !strcmp(env, "1"))
    simd_support &= JSIMD_SSE2 | JSIMD_SSE;
#if SIMD_ARCHITECTURE == I386
  if (!GETENV_S(env, 2, "JSIMD_FORCESSE") && !strcmp(env, "1"))
    simd_support &= JSIMD_SSE | JSIMD_MMX;
  if (!GETENV_S(env, 2, "JSIMD_FORCEMMX") && !strcmp(env, "1"))
    simd_support &= JSIMD_MMX;
  if (!GETENV_S(env, 2, "JSIMD_FORCE3DNOW") && !strcmp(env, "1"))
    simd_support &= JSIMD_3DNOW | JSIMD_MMX;
#endif
#elif SIMD_ARCHITECTURE == ARM
  if (!GETENV_S(env, 2, "JSIMD_FORCENEON") && !strcmp(env, "1"))
    simd_support = JSIMD_NEON;
#elif SIMD_ARCHITECTURE == RISCV64
  if (!GETENV_S(env, 2, "JSIMD_FORCERVV") && !strcmp(env, "1"))
    simd_support = JSIMD_RVV;
#elif SIMD_ARCHITECTURE == MIPS64
  if (!GETENV_S(env, 2, "JSIMD_FORCEMMI") && !strcmp(env, "1"))
    simd_support = JSIMD_MMI;
#endif
  if (!GETENV_S(env, 2, "JSIMD_FORCENONE") && !strcmp(env, "1"))
    simd_support = 0;
#ifdef WITH_SIMDE
  if (!GETENV_S(env, 2, "JSIMD_HUFFENC") && !strcmp(env, "1"))
    simd_huffman = 1;
#else
  if (!GETENV_S(env, 2, "JSIMD_NOHUFFENC") && !strcmp(env, "1"))
    simd_huffman = 0;
#endif
#endif

  if (cinfo->is_decompressor) {
    ((j_decompress_ptr)cinfo)->master->simd_support = simd_support;
    ((j_decompress_ptr)cinfo)->master->simd_huffman = simd_huffman;
  } else {
    ((j_compress_ptr)cinfo)->master->simd_support = simd_support;
    ((j_compress_ptr)cinfo)->master->simd_huffman = simd_huffman;
  }
}


HIDDEN unsigned int
jsimd_set_rgb_ycc(j_compress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return JSIMD_NONE;
  if (!cinfo->cconvert)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_rgb_ycc_convert_avx2)) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(ycc, avx2);
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_rgb_ycc_convert_sse2)) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(ycc, sse2);
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(ycc, mmx);
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(ycc, neon);
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(ycc, altivec);
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(ycc, rvv);
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(ycc, mmi);
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_rgb_gray(j_compress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return JSIMD_NONE;
  if (!cinfo->cconvert)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_rgb_gray_convert_avx2)) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(gray, avx2);
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_rgb_gray_convert_sse2)) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(gray, sse2);
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(gray, mmx);
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(gray, neon);
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(gray, altivec);
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(gray, rvv);
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    SET_SIMD_EXTRGB_COLOR_CONVERTER(gray, mmi);
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_color_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                    JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  cinfo->cconvert->color_convert_simd(cinfo->image_width, input_buf,
                                      output_buf, output_row, num_rows);
}


HIDDEN unsigned int
jsimd_set_ycc_rgb(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return JSIMD_NONE;
  if (!cinfo->cconvert)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_ycc_rgb_convert_avx2)) {
    SET_SIMD_EXTRGB_COLOR_DECONVERTER(avx2);
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_ycc_rgb_convert_sse2)) {
    SET_SIMD_EXTRGB_COLOR_DECONVERTER(sse2);
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    SET_SIMD_EXTRGB_COLOR_DECONVERTER(mmx);
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    SET_SIMD_EXTRGB_COLOR_DECONVERTER(neon);
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    SET_SIMD_EXTRGB_COLOR_DECONVERTER(altivec);
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    SET_SIMD_EXTRGB_COLOR_DECONVERTER(rvv);
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    SET_SIMD_EXTRGB_COLOR_DECONVERTER(mmi);
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_ycc_rgb565(j_decompress_ptr cinfo)
{
#if SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->cconvert)
    return JSIMD_NONE;

  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->cconvert->color_convert_simd = jsimd_ycc_rgb565_convert_neon;
    return JSIMD_NEON;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_color_deconvert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                      JDIMENSION input_row, JSAMPARRAY output_buf,
                      int num_rows)
{
  cinfo->cconvert->color_convert_simd(cinfo->output_width, input_buf,
                                      input_row, output_buf, num_rows);
}


HIDDEN unsigned int
jsimd_set_h2v1_downsample(j_compress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->downsample)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_AVX2) {
    cinfo->downsample->h2v1_downsample_simd = jsimd_h2v1_downsample_avx2;
    return JSIMD_AVX2;
  }
  if (cinfo->master->simd_support & JSIMD_SSE2) {
    cinfo->downsample->h2v1_downsample_simd = jsimd_h2v1_downsample_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->downsample->h2v1_downsample_simd = jsimd_h2v1_downsample_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->downsample->h2v1_downsample_simd = jsimd_h2v1_downsample_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    cinfo->downsample->h2v1_downsample_simd = jsimd_h2v1_downsample_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    cinfo->downsample->h2v1_downsample_simd = jsimd_h2v1_downsample_rvv;
    return JSIMD_RVV;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h2v1_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  cinfo->downsample->h2v1_downsample_simd(cinfo->image_width,
                                          cinfo->max_v_samp_factor,
                                          compptr->v_samp_factor,
                                          compptr->width_in_blocks, input_data,
                                          output_data);
}


HIDDEN unsigned int
jsimd_set_h2v2_downsample(j_compress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->downsample)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_AVX2) {
    cinfo->downsample->h2v2_downsample_simd = jsimd_h2v2_downsample_avx2;
    return JSIMD_AVX2;
  }
  if (cinfo->master->simd_support & JSIMD_SSE2) {
    cinfo->downsample->h2v2_downsample_simd = jsimd_h2v2_downsample_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->downsample->h2v2_downsample_simd = jsimd_h2v2_downsample_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->downsample->h2v2_downsample_simd = jsimd_h2v2_downsample_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    cinfo->downsample->h2v2_downsample_simd = jsimd_h2v2_downsample_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    cinfo->downsample->h2v2_downsample_simd = jsimd_h2v2_downsample_rvv;
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    cinfo->downsample->h2v2_downsample_simd = jsimd_h2v2_downsample_mmi;
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h2v2_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  cinfo->downsample->h2v2_downsample_simd(cinfo->image_width,
                                          cinfo->max_v_samp_factor,
                                          compptr->v_samp_factor,
                                          compptr->width_in_blocks, input_data,
                                          output_data);
}


HIDDEN unsigned int
jsimd_set_h2v1_upsample(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->upsample)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_AVX2) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_upsample_avx2;
    return JSIMD_AVX2;
  }
  if (cinfo->master->simd_support & JSIMD_SSE2) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_upsample_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_upsample_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_upsample_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_upsample_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_upsample_rvv;
    return JSIMD_RVV;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h2v1_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  cinfo->upsample->h2v1_upsample_simd(cinfo->max_v_samp_factor,
                                      cinfo->output_width, input_data,
                                      output_data_ptr);
}


HIDDEN unsigned int
jsimd_set_h2v2_upsample(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->upsample)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_AVX2) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_upsample_avx2;
    return JSIMD_AVX2;
  }
  if (cinfo->master->simd_support & JSIMD_SSE2) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_upsample_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_upsample_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_upsample_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_upsample_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_upsample_rvv;
    return JSIMD_RVV;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h2v2_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  cinfo->upsample->h2v2_upsample_simd(cinfo->max_v_samp_factor,
                                      cinfo->output_width, input_data,
                                      output_data_ptr);
}


HIDDEN unsigned int
jsimd_set_h2v1_fancy_upsample(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->upsample)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_fancy_upsample_avx2)) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_fancy_upsample_avx2;
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_fancy_upsample_sse2)) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_fancy_upsample_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_fancy_upsample_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_fancy_upsample_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_fancy_upsample_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_fancy_upsample_rvv;
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    cinfo->upsample->h2v1_upsample_simd = jsimd_h2v1_fancy_upsample_mmi;
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h2v1_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  cinfo->upsample->h2v1_upsample_simd(cinfo->max_v_samp_factor,
                                      compptr->downsampled_width, input_data,
                                      output_data_ptr);
}


HIDDEN unsigned int
jsimd_set_h2v2_fancy_upsample(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->upsample)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_fancy_upsample_avx2)) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_fancy_upsample_avx2;
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_fancy_upsample_sse2)) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_fancy_upsample_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_fancy_upsample_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_fancy_upsample_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_fancy_upsample_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_fancy_upsample_rvv;
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    cinfo->upsample->h2v2_upsample_simd = jsimd_h2v2_fancy_upsample_mmi;
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h2v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  cinfo->upsample->h2v2_upsample_simd(cinfo->max_v_samp_factor,
                                      compptr->downsampled_width, input_data,
                                      output_data_ptr);
}


#if SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM

HIDDEN unsigned int
jsimd_set_h1v2_fancy_upsample(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->upsample)
    return JSIMD_NONE;

  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->upsample->h1v2_upsample_simd = jsimd_h1v2_fancy_upsample_neon;
    return JSIMD_NEON;
  }

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h1v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  cinfo->upsample->h1v2_upsample_simd(cinfo->max_v_samp_factor,
                                      compptr->downsampled_width, input_data,
                                      output_data_ptr);
}

#endif


HIDDEN unsigned int
jsimd_set_h2v1_merged_upsample(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->upsample)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_merged_upsample_avx2)) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v1, avx2);
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_merged_upsample_sse2)) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v1, sse2);
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v1, mmx);
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v1, neon);
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v1, altivec);
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v1, rvv);
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v1, mmi);
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h2v1_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  cinfo->upsample->merged_upsample_simd(cinfo->output_width, input_buf,
                                        in_row_group_ctr, output_buf);
}


HIDDEN unsigned int
jsimd_set_h2v2_merged_upsample(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (!cinfo->upsample)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_merged_upsample_avx2)) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v2, avx2);
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_merged_upsample_sse2)) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v2, sse2);
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v2, mmx);
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v2, neon);
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v2, altivec);
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v2, rvv);
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    SET_SIMD_EXTRGB_MERGED_UPSAMPLER(h2v2, mmi);
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_h2v2_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  cinfo->upsample->merged_upsample_simd(cinfo->output_width, input_buf,
                                        in_row_group_ctr, output_buf);
}


HIDDEN unsigned int
jsimd_set_convsamp(j_compress_ptr cinfo, convsamp_method_ptr *method)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (sizeof(DCTELEM) != 2)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_AVX2) {
    *method = jsimd_convsamp_avx2;
    return JSIMD_AVX2;
  }
  if (cinfo->master->simd_support & JSIMD_SSE2) {
    *method = jsimd_convsamp_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    *method = jsimd_convsamp_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    *method = jsimd_convsamp_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    *method = jsimd_convsamp_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    *method = jsimd_convsamp_rvv;
    return JSIMD_RVV;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_convsamp_float(j_compress_ptr cinfo,
                         float_convsamp_method_ptr *method)
{
  init_simd((j_common_ptr)cinfo);

  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (sizeof(FAST_FLOAT) != 4)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_SSE2) {
    *method = jsimd_convsamp_float_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_SSE) {
    *method = jsimd_convsamp_float_sse;
    return JSIMD_SSE;
  }
  if (cinfo->master->simd_support & JSIMD_3DNOW) {
    *method = jsimd_convsamp_float_3dnow;
    return JSIMD_3DNOW;
  }
#endif
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_fdct_islow(j_compress_ptr cinfo, forward_DCT_method_ptr *method)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(DCTELEM) != 2)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_fdct_islow_avx2)) {
    *method = jsimd_fdct_islow_avx2;
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_fdct_islow_sse2)) {
    *method = jsimd_fdct_islow_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    *method = jsimd_fdct_islow_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    *method = jsimd_fdct_islow_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    *method = jsimd_fdct_islow_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    *method = jsimd_fdct_islow_rvv;
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    *method = jsimd_fdct_islow_mmi;
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_fdct_ifast(j_compress_ptr cinfo, forward_DCT_method_ptr *method)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(DCTELEM) != 2)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_fdct_ifast_sse2)) {
    *method = jsimd_fdct_ifast_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    *method = jsimd_fdct_ifast_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    *method = jsimd_fdct_ifast_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    *method = jsimd_fdct_ifast_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    *method = jsimd_fdct_ifast_rvv;
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    *method = jsimd_fdct_ifast_mmi;
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_fdct_float(j_compress_ptr cinfo, float_DCT_method_ptr *method)
{
#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  init_simd((j_common_ptr)cinfo);

  if (sizeof(FAST_FLOAT) != 4)
    return JSIMD_NONE;

  if ((cinfo->master->simd_support & JSIMD_SSE) &&
      IS_ALIGNED_SSE(jconst_fdct_float_sse)) {
    *method = jsimd_fdct_float_sse;
    return JSIMD_SSE;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_3DNOW) {
    *method = jsimd_fdct_float_3dnow;
    return JSIMD_3DNOW;
  }
#endif
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_quantize(j_compress_ptr cinfo, quantize_method_ptr *method)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;
  if (sizeof(DCTELEM) != 2)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_AVX2) {
    *method = jsimd_quantize_avx2;
    return JSIMD_AVX2;
  }
  if (cinfo->master->simd_support & JSIMD_SSE2) {
    *method = jsimd_quantize_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    *method = jsimd_quantize_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    *method = jsimd_quantize_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    *method = jsimd_quantize_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    *method = jsimd_quantize_rvv;
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    *method = jsimd_quantize_mmi;
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_quantize_float(j_compress_ptr cinfo,
                         float_quantize_method_ptr *method)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;
  if (sizeof(FAST_FLOAT) != 4)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_SSE2) {
    *method = jsimd_quantize_float_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_SSE) {
    *method = jsimd_quantize_float_sse;
    return JSIMD_SSE;
  }
  if (cinfo->master->simd_support & JSIMD_3DNOW) {
    *method = jsimd_quantize_float_3dnow;
    return JSIMD_3DNOW;
  }
#endif
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_idct_islow(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;
  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return JSIMD_NONE;
  if (!cinfo->idct)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_AVX2) &&
      IS_ALIGNED_AVX(jconst_idct_islow_avx2)) {
    cinfo->idct->idct_simd = jsimd_idct_islow_avx2;
    return JSIMD_AVX2;
  }
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_idct_islow_sse2)) {
    cinfo->idct->idct_simd = jsimd_idct_islow_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->idct->idct_simd = jsimd_idct_islow_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->idct->idct_simd = jsimd_idct_islow_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    cinfo->idct->idct_simd = jsimd_idct_islow_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    cinfo->idct->idct_simd = jsimd_idct_islow_rvv;
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    cinfo->idct->idct_simd = jsimd_idct_islow_mmi;
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_idct_islow(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  cinfo->idct->idct_simd(compptr->dct_table, coef_block, output_buf,
                         output_col);
}


HIDDEN unsigned int
jsimd_set_idct_ifast(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;
  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (sizeof(IFAST_MULT_TYPE) != 2)
    return JSIMD_NONE;
  if (IFAST_SCALE_BITS != 2)
    return JSIMD_NONE;
  if (!cinfo->idct)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_idct_ifast_sse2)) {
    cinfo->idct->idct_simd = jsimd_idct_ifast_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->idct->idct_simd = jsimd_idct_ifast_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->idct->idct_simd = jsimd_idct_ifast_neon;
    return JSIMD_NEON;
  }
#elif SIMD_ARCHITECTURE == POWERPC
  if (cinfo->master->simd_support & JSIMD_ALTIVEC) {
    cinfo->idct->idct_simd = jsimd_idct_ifast_altivec;
    return JSIMD_ALTIVEC;
  }
#elif SIMD_ARCHITECTURE == RISCV64
  if (cinfo->master->simd_support & JSIMD_RVV) {
    cinfo->idct->idct_simd = jsimd_idct_ifast_rvv;
    return JSIMD_RVV;
  }
#elif SIMD_ARCHITECTURE == MIPS64
  if (cinfo->master->simd_support & JSIMD_MMI) {
    cinfo->idct->idct_simd = jsimd_idct_ifast_mmi;
    return JSIMD_MMI;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_idct_ifast(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  cinfo->idct->idct_simd(compptr->dct_table, coef_block, output_buf,
                         output_col);
}


HIDDEN unsigned int
jsimd_set_idct_float(j_decompress_ptr cinfo)
{
#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;
  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (sizeof(FAST_FLOAT) != 4)
    return JSIMD_NONE;
  if (sizeof(FLOAT_MULT_TYPE) != 4)
    return JSIMD_NONE;
  if (!cinfo->idct)
    return JSIMD_NONE;

  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_idct_float_sse2)) {
    cinfo->idct->idct_simd = jsimd_idct_float_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_SSE) &&
      IS_ALIGNED_SSE(jconst_idct_float_sse)) {
    cinfo->idct->idct_simd = jsimd_idct_float_sse;
    return JSIMD_SSE;
  }
  if (cinfo->master->simd_support & JSIMD_3DNOW) {
    cinfo->idct->idct_simd = jsimd_idct_float_3dnow;
    return JSIMD_3DNOW;
  }
#endif
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_idct_float(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  cinfo->idct->idct_simd(compptr->dct_table, coef_block, output_buf,
                         output_col);
}


HIDDEN unsigned int
jsimd_set_idct_2x2(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;
  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return JSIMD_NONE;
  if (!cinfo->idct)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_idct_red_sse2)) {
    cinfo->idct->idct_2x2_simd = jsimd_idct_2x2_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->idct->idct_2x2_simd = jsimd_idct_2x2_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->idct->idct_2x2_simd = jsimd_idct_2x2_neon;
    return JSIMD_NEON;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_idct_2x2(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  cinfo->idct->idct_2x2_simd(compptr->dct_table, coef_block, output_buf,
                             output_col);
}


HIDDEN unsigned int
jsimd_set_idct_4x4(j_decompress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;
  if (BITS_IN_JSAMPLE != 8)
    return JSIMD_NONE;
  if (sizeof(JDIMENSION) != 4)
    return JSIMD_NONE;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return JSIMD_NONE;
  if (!cinfo->idct)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      IS_ALIGNED_SSE(jconst_idct_red_sse2)) {
    cinfo->idct->idct_4x4_simd = jsimd_idct_4x4_sse2;
    return JSIMD_SSE2;
  }
#if SIMD_ARCHITECTURE == I386
  if (cinfo->master->simd_support & JSIMD_MMX) {
    cinfo->idct->idct_4x4_simd = jsimd_idct_4x4_mmx;
    return JSIMD_MMX;
  }
#endif
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if (cinfo->master->simd_support & JSIMD_NEON) {
    cinfo->idct->idct_4x4_simd = jsimd_idct_4x4_neon;
    return JSIMD_NEON;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN void
jsimd_idct_4x4(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  cinfo->idct->idct_4x4_simd(compptr->dct_table, coef_block, output_buf,
                             output_col);
}


HIDDEN unsigned int
jsimd_set_huff_encode_one_block(j_compress_ptr cinfo)
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;
  if (!cinfo->entropy)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      cinfo->master->simd_huffman &&
      IS_ALIGNED_SSE(jconst_huff_encode_one_block)) {
    cinfo->entropy->huff_encode_one_block_simd =
      jsimd_huff_encode_one_block_sse2;
    return JSIMD_SSE2;
  }
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if ((cinfo->master->simd_support & JSIMD_NEON) &&
      cinfo->master->simd_huffman) {
    cinfo->entropy->huff_encode_one_block_simd =
      jsimd_huff_encode_one_block_neon;
    return JSIMD_NEON;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_encode_mcu_AC_first_prepare(j_compress_ptr cinfo,
  void (**method) (const JCOEF *, const int *, int, int, UJCOEF *, size_t *))
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      cinfo->master->simd_huffman) {
    *method = jsimd_encode_mcu_AC_first_prepare_sse2;
    return JSIMD_SSE2;
  }
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if ((cinfo->master->simd_support & JSIMD_NEON) &&
      cinfo->master->simd_huffman) {
    *method = jsimd_encode_mcu_AC_first_prepare_neon;
    return JSIMD_NEON;
  }
#endif

  return JSIMD_NONE;
}


HIDDEN unsigned int
jsimd_set_encode_mcu_AC_refine_prepare(j_compress_ptr cinfo,
  int (**method) (const JCOEF *, const int *, int, int, UJCOEF *, size_t *))
{
  init_simd((j_common_ptr)cinfo);

  if (sizeof(JCOEF) != 2)
    return JSIMD_NONE;

#if SIMD_ARCHITECTURE == X86_64 || SIMD_ARCHITECTURE == I386
  if ((cinfo->master->simd_support & JSIMD_SSE2) &&
      cinfo->master->simd_huffman) {
    *method = jsimd_encode_mcu_AC_refine_prepare_sse2;
    return JSIMD_SSE2;
  }
#elif SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
  if ((cinfo->master->simd_support & JSIMD_NEON) &&
      cinfo->master->simd_huffman) {
    *method = jsimd_encode_mcu_AC_refine_prepare_neon;
    return JSIMD_NEON;
  }
#endif

  return JSIMD_NONE;
}
