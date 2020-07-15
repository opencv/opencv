/*
 * jcinit.c
 *
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2003-2017 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains initialization logic for the JPEG compressor.
 * This routine is in charge of selecting the modules to be executed and
 * making an initialization call to each one.
 *
 * Logically, this code belongs in jcmaster.c.  It's split out because
 * linking this routine implies linking the entire compression library.
 * For a transcoding-only application, we want to be able to use jcmaster.c
 * without linking in the whole library.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


/*
 * Compute JPEG image dimensions and related values.
 * NOTE: this is exported for possible use by application.
 * Hence it mustn't do anything that can't be done twice.
 */

GLOBAL(void)
jpeg_calc_jpeg_dimensions (j_compress_ptr cinfo)
/* Do computations that are needed before master selection phase */
{
  /* Sanity check on input image dimensions to prevent overflow in
   * following calculations.
   * We do check jpeg_width and jpeg_height in initial_setup in jcmaster.c,
   * but image_width and image_height can come from arbitrary data,
   * and we need some space for multiplication by block_size.
   */
  if (((long) cinfo->image_width >> 24) || ((long) cinfo->image_height >> 24))
    ERREXIT1(cinfo, JERR_IMAGE_TOO_BIG, (unsigned int) JPEG_MAX_DIMENSION);

#ifdef DCT_SCALING_SUPPORTED

  /* Compute actual JPEG image dimensions and DCT scaling choices. */
  if (cinfo->scale_num >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/1 scaling */
    cinfo->jpeg_width = cinfo->image_width * cinfo->block_size;
    cinfo->jpeg_height = cinfo->image_height * cinfo->block_size;
    cinfo->min_DCT_h_scaled_size = 1;
    cinfo->min_DCT_v_scaled_size = 1;
  } else if (cinfo->scale_num * 2 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/2 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 2L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 2L);
    cinfo->min_DCT_h_scaled_size = 2;
    cinfo->min_DCT_v_scaled_size = 2;
  } else if (cinfo->scale_num * 3 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/3 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 3L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 3L);
    cinfo->min_DCT_h_scaled_size = 3;
    cinfo->min_DCT_v_scaled_size = 3;
  } else if (cinfo->scale_num * 4 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/4 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 4L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 4L);
    cinfo->min_DCT_h_scaled_size = 4;
    cinfo->min_DCT_v_scaled_size = 4;
  } else if (cinfo->scale_num * 5 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/5 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 5L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 5L);
    cinfo->min_DCT_h_scaled_size = 5;
    cinfo->min_DCT_v_scaled_size = 5;
  } else if (cinfo->scale_num * 6 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/6 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 6L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 6L);
    cinfo->min_DCT_h_scaled_size = 6;
    cinfo->min_DCT_v_scaled_size = 6;
  } else if (cinfo->scale_num * 7 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/7 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 7L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 7L);
    cinfo->min_DCT_h_scaled_size = 7;
    cinfo->min_DCT_v_scaled_size = 7;
  } else if (cinfo->scale_num * 8 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/8 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 8L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 8L);
    cinfo->min_DCT_h_scaled_size = 8;
    cinfo->min_DCT_v_scaled_size = 8;
  } else if (cinfo->scale_num * 9 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/9 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 9L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 9L);
    cinfo->min_DCT_h_scaled_size = 9;
    cinfo->min_DCT_v_scaled_size = 9;
  } else if (cinfo->scale_num * 10 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/10 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 10L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 10L);
    cinfo->min_DCT_h_scaled_size = 10;
    cinfo->min_DCT_v_scaled_size = 10;
  } else if (cinfo->scale_num * 11 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/11 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 11L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 11L);
    cinfo->min_DCT_h_scaled_size = 11;
    cinfo->min_DCT_v_scaled_size = 11;
  } else if (cinfo->scale_num * 12 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/12 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 12L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 12L);
    cinfo->min_DCT_h_scaled_size = 12;
    cinfo->min_DCT_v_scaled_size = 12;
  } else if (cinfo->scale_num * 13 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/13 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 13L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 13L);
    cinfo->min_DCT_h_scaled_size = 13;
    cinfo->min_DCT_v_scaled_size = 13;
  } else if (cinfo->scale_num * 14 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/14 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 14L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 14L);
    cinfo->min_DCT_h_scaled_size = 14;
    cinfo->min_DCT_v_scaled_size = 14;
  } else if (cinfo->scale_num * 15 >= cinfo->scale_denom * cinfo->block_size) {
    /* Provide block_size/15 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 15L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 15L);
    cinfo->min_DCT_h_scaled_size = 15;
    cinfo->min_DCT_v_scaled_size = 15;
  } else {
    /* Provide block_size/16 scaling */
    cinfo->jpeg_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * cinfo->block_size, 16L);
    cinfo->jpeg_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * cinfo->block_size, 16L);
    cinfo->min_DCT_h_scaled_size = 16;
    cinfo->min_DCT_v_scaled_size = 16;
  }

#else /* !DCT_SCALING_SUPPORTED */

  /* Hardwire it to "no scaling" */
  cinfo->jpeg_width = cinfo->image_width;
  cinfo->jpeg_height = cinfo->image_height;
  cinfo->min_DCT_h_scaled_size = DCTSIZE;
  cinfo->min_DCT_v_scaled_size = DCTSIZE;

#endif /* DCT_SCALING_SUPPORTED */
}


/*
 * Master selection of compression modules.
 * This is done once at the start of processing an image.  We determine
 * which modules will be used and give them appropriate initialization calls.
 */

GLOBAL(void)
jinit_compress_master (j_compress_ptr cinfo)
{
  long samplesperrow;
  JDIMENSION jd_samplesperrow;

  /* For now, precision must match compiled-in value... */
  if (cinfo->data_precision != BITS_IN_JSAMPLE)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Sanity check on input image dimensions */
  if (cinfo->image_height <= 0 || cinfo->image_width <= 0 ||
      cinfo->input_components <= 0)
    ERREXIT(cinfo, JERR_EMPTY_IMAGE);

  /* Width of an input scanline must be representable as JDIMENSION. */
  samplesperrow = (long) cinfo->image_width * (long) cinfo->input_components;
  jd_samplesperrow = (JDIMENSION) samplesperrow;
  if ((long) jd_samplesperrow != samplesperrow)
    ERREXIT(cinfo, JERR_WIDTH_OVERFLOW);

  /* Compute JPEG image dimensions and related values. */
  jpeg_calc_jpeg_dimensions(cinfo);

  /* Initialize master control (includes parameter checking/processing) */
  jinit_c_master_control(cinfo, FALSE /* full compression */);

  /* Preprocessing */
  if (! cinfo->raw_data_in) {
    jinit_color_converter(cinfo);
    jinit_downsampler(cinfo);
    jinit_c_prep_controller(cinfo, FALSE /* never need full buffer here */);
  }
  /* Forward DCT */
  jinit_forward_dct(cinfo);
  /* Entropy encoding: either Huffman or arithmetic coding. */
  if (cinfo->arith_code)
    jinit_arith_encoder(cinfo);
  else {
    jinit_huff_encoder(cinfo);
  }

  /* Need a full-image coefficient buffer in any multi-pass mode. */
  jinit_c_coef_controller(cinfo,
		(boolean) (cinfo->num_scans > 1 || cinfo->optimize_coding));
  jinit_c_main_controller(cinfo, FALSE /* never need full buffer here */);

  jinit_marker_writer(cinfo);

  /* We can now tell the memory manager to allocate virtual arrays. */
  (*cinfo->mem->realize_virt_arrays) ((j_common_ptr) cinfo);

  /* Write the datastream header (SOI) immediately.
   * Frame and scan headers are postponed till later.
   * This lets application insert special markers after the SOI.
   */
  (*cinfo->marker->write_file_header) (cinfo);
}
