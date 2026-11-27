/*
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2025, D. R. Commander.
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
#include "jsimdconst.h"


/* Sample Conversion */
EXTERN(unsigned int) jsimd_set_convsamp(j_compress_ptr cinfo,
                                        convsamp_method_ptr *method);

EXTERN(unsigned int) jsimd_set_convsamp_float
  (j_compress_ptr cinfo, float_convsamp_method_ptr *method);


/* Forward DCT */
EXTERN(unsigned int) jsimd_set_fdct_islow(j_compress_ptr cinfo,
                                          forward_DCT_method_ptr *method);

EXTERN(unsigned int) jsimd_set_fdct_ifast(j_compress_ptr cinfo,
                                          forward_DCT_method_ptr *method);

EXTERN(unsigned int) jsimd_set_fdct_float(j_compress_ptr cinfo,
                                          float_DCT_method_ptr *method);


/* Quantization */
EXTERN(unsigned int) jsimd_set_quantize(j_compress_ptr cinfo,
                                        quantize_method_ptr *method);

EXTERN(unsigned int) jsimd_set_quantize_float
  (j_compress_ptr cinfo, float_quantize_method_ptr *method);


/* Inverse DCT */
EXTERN(unsigned int) jsimd_set_idct_islow(j_decompress_ptr cinfo);
EXTERN(void) jsimd_idct_islow(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, JSAMPARRAY output_buf,
                              JDIMENSION output_col);

EXTERN(unsigned int) jsimd_set_idct_ifast(j_decompress_ptr cinfo);
EXTERN(void) jsimd_idct_ifast(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, JSAMPARRAY output_buf,
                              JDIMENSION output_col);

EXTERN(unsigned int) jsimd_set_idct_float(j_decompress_ptr cinfo);
EXTERN(void) jsimd_idct_float(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, JSAMPARRAY output_buf,
                              JDIMENSION output_col);


/* Scaled Integer Inverse DCT */
EXTERN(unsigned int) jsimd_set_idct_2x2(j_decompress_ptr cinfo);
EXTERN(void) jsimd_idct_2x2(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            JSAMPARRAY output_buf, JDIMENSION output_col);

EXTERN(unsigned int) jsimd_set_idct_4x4(j_decompress_ptr cinfo);
EXTERN(void) jsimd_idct_4x4(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            JSAMPARRAY output_buf, JDIMENSION output_col);
