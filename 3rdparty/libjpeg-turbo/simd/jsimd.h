/*
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2011, 2014, 2022, 2025, D. R. Commander.
 * Copyright (C) 2015-2016, 2018, 2022, Matthieu Darbois.
 * Copyright (C) 2020, Arm Limited.
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

#include "../src/jchuff.h"
#include "jsimdconst.h"


/* Color Conversion/Deconversion */
EXTERN(unsigned int) jsimd_set_rgb_ycc(j_compress_ptr cinfo);

EXTERN(unsigned int) jsimd_set_rgb_gray(j_compress_ptr cinfo);

EXTERN(void) jsimd_color_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                                 JSAMPIMAGE output_buf, JDIMENSION output_row,
                                 int num_rows);

EXTERN(unsigned int) jsimd_set_ycc_rgb(j_decompress_ptr cinfo);

EXTERN(unsigned int) jsimd_set_ycc_rgb565(j_decompress_ptr cinfo);

EXTERN(void) jsimd_color_deconvert(j_decompress_ptr cinfo,
                                   JSAMPIMAGE input_buf, JDIMENSION input_row,
                                   JSAMPARRAY output_buf, int num_rows);


/* Downsampling */
EXTERN(unsigned int) jsimd_set_h2v1_downsample(j_compress_ptr cinfo);
EXTERN(void) jsimd_h2v1_downsample(j_compress_ptr cinfo,
                                   jpeg_component_info *compptr,
                                   JSAMPARRAY input_data,
                                   JSAMPARRAY output_data);

EXTERN(unsigned int) jsimd_set_h2v2_downsample(j_compress_ptr cinfo);
EXTERN(void) jsimd_h2v2_downsample(j_compress_ptr cinfo,
                                   jpeg_component_info *compptr,
                                   JSAMPARRAY input_data,
                                   JSAMPARRAY output_data);


/* Plain Upsampling */
EXTERN(unsigned int) jsimd_set_h2v1_upsample(j_decompress_ptr cinfo);
EXTERN(void) jsimd_h2v1_upsample(j_decompress_ptr cinfo,
                                 jpeg_component_info *compptr,
                                 JSAMPARRAY input_data,
                                 JSAMPARRAY *output_data_ptr);

EXTERN(unsigned int) jsimd_set_h2v2_upsample(j_decompress_ptr cinfo);
EXTERN(void) jsimd_h2v2_upsample(j_decompress_ptr cinfo,
                                 jpeg_component_info *compptr,
                                 JSAMPARRAY input_data,
                                 JSAMPARRAY *output_data_ptr);


/* Fancy (Smooth) Upsampling */
EXTERN(unsigned int) jsimd_set_h2v1_fancy_upsample(j_decompress_ptr cinfo);
EXTERN(void) jsimd_h2v1_fancy_upsample(j_decompress_ptr cinfo,
                                       jpeg_component_info *compptr,
                                       JSAMPARRAY input_data,
                                       JSAMPARRAY *output_data_ptr);

EXTERN(unsigned int) jsimd_set_h2v2_fancy_upsample(j_decompress_ptr cinfo);
EXTERN(void) jsimd_h2v2_fancy_upsample(j_decompress_ptr cinfo,
                                       jpeg_component_info *compptr,
                                       JSAMPARRAY input_data,
                                       JSAMPARRAY *output_data_ptr);

#if SIMD_ARCHITECTURE == ARM64 || SIMD_ARCHITECTURE == ARM
EXTERN(unsigned int) jsimd_set_h1v2_fancy_upsample(j_decompress_ptr cinfo);
EXTERN(void) jsimd_h1v2_fancy_upsample(j_decompress_ptr cinfo,
                                       jpeg_component_info *compptr,
                                       JSAMPARRAY input_data,
                                       JSAMPARRAY *output_data_ptr);
#endif


/* Merged Upsampling/Color Conversion */
EXTERN(unsigned int) jsimd_set_h2v1_merged_upsample(j_decompress_ptr cinfo);
EXTERN(void) jsimd_h2v1_merged_upsample(j_decompress_ptr cinfo,
                                        JSAMPIMAGE input_buf,
                                        JDIMENSION in_row_group_ctr,
                                        JSAMPARRAY output_buf);

EXTERN(unsigned int) jsimd_set_h2v2_merged_upsample(j_decompress_ptr cinfo);
EXTERN(void) jsimd_h2v2_merged_upsample(j_decompress_ptr cinfo,
                                        JSAMPIMAGE input_buf,
                                        JDIMENSION in_row_group_ctr,
                                        JSAMPARRAY output_buf);


/* Huffman Encoding */
EXTERN(unsigned int) jsimd_set_huff_encode_one_block(j_compress_ptr cinfo);


/* Progressive Huffman Encoding */
EXTERN(unsigned int) jsimd_set_encode_mcu_AC_first_prepare
  (j_compress_ptr cinfo,
   void (**method) (const JCOEF *, const int *, int, int, UJCOEF *, size_t *));

EXTERN(unsigned int) jsimd_set_encode_mcu_AC_refine_prepare
  (j_compress_ptr cinfo,
   int (**method) (const JCOEF *, const int *, int, int, UJCOEF *, size_t *));
