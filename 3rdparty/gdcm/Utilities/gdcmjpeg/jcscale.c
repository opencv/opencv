/*
 * jcscale.c
 *
 * Copyright (C) 1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains sample downscaling by 2^Pt for lossless JPEG.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"    /* Private declarations for lossless codec */


#ifdef C_LOSSLESS_SUPPORTED

METHODDEF(void)
simple_downscale(j_compress_ptr cinfo,
     JSAMPROW input_buf, JSAMPROW output_buf, JDIMENSION width)
{
  /* j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec; */
  unsigned int xindex;
  SHIFT_TEMPS

  for (xindex = 0; xindex < width; xindex++)
    output_buf[xindex] = (JSAMPLE) RIGHT_SHIFT(GETJSAMPLE(input_buf[xindex]),
                 cinfo->Al);
}


METHODDEF(void)
noscale(j_compress_ptr cinfo,
  JSAMPROW input_buf, JSAMPROW output_buf, JDIMENSION width)
{
  (void)cinfo;
  MEMCOPY(output_buf, input_buf, width * SIZEOF(JSAMPLE));
  return;
}


METHODDEF(void)
scaler_start_pass (j_compress_ptr cinfo)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;

  /* Set scaler function based on Pt */
  if (cinfo->Al)
    losslsc->scaler_scale = simple_downscale;
  else
    losslsc->scaler_scale = noscale;
}


GLOBAL(void)
jinit_c_scaler (j_compress_ptr cinfo)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;

  losslsc->scaler_start_pass = scaler_start_pass;
}

#endif /* C_LOSSLESS_SUPPORTED */
