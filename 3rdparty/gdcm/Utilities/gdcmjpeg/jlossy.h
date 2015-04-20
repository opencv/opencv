/*
 * jlossy.h
 *
 * Copyright (C) 1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This include file contains common declarations for the lossy (DCT-based)
 * JPEG codec modules.
 */

#ifndef JLOSSY_H
#define JLOSSY_H


/* Lossy-specific compression codec (compressor proper) */
typedef struct {
  struct jpeg_c_codec pub; /* public fields */


  /* Coefficient buffer control */
  JMETHOD(void, coef_start_pass, (j_compress_ptr cinfo, J_BUF_MODE pass_mode));
  /*  JMETHOD(boolean, coef_compress_data, (j_compress_ptr cinfo,
          JSAMPIMAGE input_buf));*/

  /* Pointer to data which is private to coef module */
  void *coef_private;


  /* Forward DCT (also controls coefficient quantization) */
  JMETHOD(void, fdct_start_pass, (j_compress_ptr cinfo));
  /* perhaps this should be an array??? */
  JMETHOD(void, fdct_forward_DCT, (j_compress_ptr cinfo,
           jpeg_component_info * compptr,
           JSAMPARRAY sample_data, JBLOCKROW coef_blocks,
           JDIMENSION start_row, JDIMENSION start_col,
           JDIMENSION num_blocks));

  /* Pointer to data which is private to fdct module */
  void *fdct_private;


  /* Entropy encoding */
  JMETHOD(boolean, entropy_encode_mcu, (j_compress_ptr cinfo,
          JBLOCKROW *MCU_data));

  /* Pointer to data which is private to entropy module */
  void *entropy_private;

} jpeg_lossy_c_codec;

typedef jpeg_lossy_c_codec * j_lossy_c_ptr;



typedef JMETHOD(void, inverse_DCT_method_ptr,
    (j_decompress_ptr cinfo, jpeg_component_info * compptr,
     JCOEFPTR coef_block,
     JSAMPARRAY output_buf, JDIMENSION output_col));

/* Lossy-specific decompression codec (decompressor proper) */
typedef struct {
  struct jpeg_d_codec pub; /* public fields */


  /* Coefficient buffer control */
  JMETHOD(void, coef_start_input_pass, (j_decompress_ptr cinfo));
  JMETHOD(void, coef_start_output_pass, (j_decompress_ptr cinfo));

  /* Pointer to array of coefficient virtual arrays, or NULL if none */
  jvirt_barray_ptr *coef_arrays;

  /* Pointer to data which is private to coef module */
  void *coef_private;


  /* Entropy decoding */
  JMETHOD(void, entropy_start_pass, (j_decompress_ptr cinfo));
  JMETHOD(boolean, entropy_decode_mcu, (j_decompress_ptr cinfo,
          JBLOCKROW *MCU_data));

  /* This is here to share code between baseline and progressive decoders; */
  /* other modules probably should not use it */
  boolean entropy_insufficient_data;  /* set TRUE after emitting warning */

  /* Pointer to data which is private to entropy module */
  void *entropy_private;


  /* Inverse DCT (also performs dequantization) */
  JMETHOD(void, idct_start_pass, (j_decompress_ptr cinfo));

  /* It is useful to allow each component to have a separate IDCT method. */
  inverse_DCT_method_ptr inverse_DCT[MAX_COMPONENTS];

  /* Pointer to data which is private to idct module */
  void *idct_private;

} jpeg_lossy_d_codec;

typedef jpeg_lossy_d_codec * j_lossy_d_ptr;


/* Compression module initialization routines */
EXTERN(void) jinit_lossy_c_codec JPP((j_compress_ptr cinfo));
EXTERN(void) jinit_c_coef_controller JPP((j_compress_ptr cinfo,
            boolean need_full_buffer));
EXTERN(void) jinit_forward_dct JPP((j_compress_ptr cinfo));
EXTERN(void) jinit_shuff_encoder JPP((j_compress_ptr cinfo));
EXTERN(void) jinit_phuff_encoder JPP((j_compress_ptr cinfo));

/* Decompression module initialization routines */
EXTERN(void) jinit_lossy_d_codec JPP((j_decompress_ptr cinfo));
EXTERN(void) jinit_d_coef_controller JPP((j_decompress_ptr cinfo,
            boolean need_full_buffer));
EXTERN(void) jinit_shuff_decoder JPP((j_decompress_ptr cinfo));
EXTERN(void) jinit_phuff_decoder JPP((j_decompress_ptr cinfo));
EXTERN(void) jinit_inverse_dct JPP((j_decompress_ptr cinfo));

#endif /* JLOSSY_H */
