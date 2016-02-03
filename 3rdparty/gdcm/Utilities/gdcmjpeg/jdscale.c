/*
 * jdscale.c
 *
 * Copyright (C) 1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains sample scaling for lossless JPEG.  This is a
 * combination of upscaling the undifferenced sample by 2^Pt and downscaling
 * the sample to fit into JSAMPLE.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"    /* Private declarations for lossless codec */


#ifdef D_LOSSLESS_SUPPORTED

/*
 * Private scaler object for lossless decoding.
 */

typedef struct {
  int scale_factor;
} scaler;

typedef scaler * scaler_ptr;


/*
 * Scalers for packing sample differences into JSAMPLEs.
 */

METHODDEF(void)
simple_upscale(j_decompress_ptr cinfo,
         JDIFFROW diff_buf, JSAMPROW output_buf,
         JDIMENSION width)
{
  j_lossless_d_ptr losslsd = (j_lossless_d_ptr) cinfo->codec;
  scaler_ptr scaler = (scaler_ptr) losslsd->scaler_private;
  int scale_factor = scaler->scale_factor;
  unsigned int xindex;

  for (xindex = 0; xindex < width; xindex++)
    output_buf[xindex] = (JSAMPLE) (diff_buf[xindex] << scale_factor);
}

METHODDEF(void)
simple_downscale(j_decompress_ptr cinfo,
     JDIFFROW diff_buf, JSAMPROW output_buf,
     JDIMENSION width)
{
  j_lossless_d_ptr losslsd = (j_lossless_d_ptr) cinfo->codec;
  scaler_ptr scaler = (scaler_ptr) losslsd->scaler_private;
  int scale_factor = scaler->scale_factor;
  unsigned int xindex;
  SHIFT_TEMPS

  for (xindex = 0; xindex < width; xindex++)
    output_buf[xindex] = (JSAMPLE) RIGHT_SHIFT(diff_buf[xindex], scale_factor);
}

METHODDEF(void)
noscale(j_decompress_ptr cinfo,
  JDIFFROW diff_buf, JSAMPROW output_buf,
  JDIMENSION width)
{
  unsigned int xindex;
  (void)cinfo;

  for (xindex = 0; xindex < width; xindex++)
    output_buf[xindex] = (JSAMPLE) diff_buf[xindex];
}


METHODDEF(void)
scaler_start_pass (j_decompress_ptr cinfo)
{
  j_lossless_d_ptr losslsd = (j_lossless_d_ptr) cinfo->codec;
  scaler_ptr scaler = (scaler_ptr) losslsd->scaler_private;
  int downscale;

  /*
   * Downscale by the difference in the input vs. output precision.  If the
   * output precision >= input precision, then do not downscale.
   */
  downscale = BITS_IN_JSAMPLE < cinfo->data_precision ?
    cinfo->data_precision - BITS_IN_JSAMPLE : 0;

  scaler->scale_factor = cinfo->Al - downscale;

  /* Set scaler functions based on scale_factor (positive = left shift) */
  if (scaler->scale_factor > 0)
    losslsd->scaler_scale = simple_upscale;
  else if (scaler->scale_factor < 0) {
    scaler->scale_factor = -scaler->scale_factor;
    losslsd->scaler_scale = simple_downscale;
  }
  else
    losslsd->scaler_scale = noscale;
}


GLOBAL(void)
jinit_d_scaler (j_decompress_ptr cinfo)
{
  j_lossless_d_ptr losslsd = (j_lossless_d_ptr) cinfo->codec;
  scaler_ptr scaler;

  scaler = (scaler_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
        SIZEOF(scaler));
  losslsd->scaler_private = (void *) scaler;
  losslsd->scaler_start_pass = scaler_start_pass;
}

#endif /* D_LOSSLESS_SUPPORTED */
