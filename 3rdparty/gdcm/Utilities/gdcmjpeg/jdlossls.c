/*
 * jdlossls.c
 *
 * Copyright (C) 1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains the control logic for the lossless JPEG decompressor.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"


#ifdef D_LOSSLESS_SUPPORTED

/*
 * Compute output image dimensions and related values.
 */

METHODDEF(void)
calc_output_dimensions (j_decompress_ptr cinfo)
{
  /* Hardwire it to "no scaling" */
  cinfo->output_width = cinfo->image_width;
  cinfo->output_height = cinfo->image_height;
  /* jdinput.c has already initialized codec_data_unit to 1,
   * and has computed unscaled downsampled_width and downsampled_height.
   */
}


/*
 * Initialize for an input processing pass.
 */

METHODDEF(void)
start_input_pass (j_decompress_ptr cinfo)
{
  j_lossless_d_ptr losslsd = (j_lossless_d_ptr) cinfo->codec;

  (*losslsd->entropy_start_pass) (cinfo);
  (*losslsd->predict_start_pass) (cinfo);
  (*losslsd->scaler_start_pass) (cinfo);
  (*losslsd->diff_start_input_pass) (cinfo);
}


/*
 * Initialize the lossless decompression codec.
 * This is called only once, during master selection.
 */

GLOBAL(void)
jinit_lossless_d_codec(j_decompress_ptr cinfo)
{
  j_lossless_d_ptr losslsd;
  boolean use_c_buffer;

  /* Create subobject in permanent pool */
  losslsd = (j_lossless_d_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
        SIZEOF(jpeg_lossless_d_codec));
  cinfo->codec = (struct jpeg_d_codec *) losslsd;

  /* Initialize sub-modules */
  /* Entropy decoding: either Huffman or arithmetic coding. */
  if (cinfo->arith_code) {
#ifdef WITH_ARITHMETIC_PATCH
    jinit_arith_decoder(cinfo);
#else
    ERREXIT(cinfo, JERR_ARITH_NOTIMPL);
#endif
  } else {
    jinit_lhuff_decoder(cinfo);
  }

  /* Undifferencer */
  jinit_undifferencer(cinfo);

  /* Scaler */
  jinit_d_scaler(cinfo);

  use_c_buffer = cinfo->inputctl->has_multiple_scans || cinfo->buffered_image;
  jinit_d_diff_controller(cinfo, use_c_buffer);

  /* Initialize method pointers.
   *
   * Note: consume_data, start_output_pass and decompress_data are
   * assigned in jddiffct.c.
   */
  losslsd->pub.calc_output_dimensions = calc_output_dimensions;
  losslsd->pub.start_input_pass = start_input_pass;
}

#endif /* D_LOSSLESS_SUPPORTED */
