/*
 * jcodec.c
 *
 * Copyright (C) 1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains utility functions for the JPEG codec(s).
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossy.h"
#include "jlossls.h"


/*
 * Initialize the compression codec.
 * This is called only once, during master selection.
 */

GLOBAL(void)
jinit_c_codec (j_compress_ptr cinfo)
{
  if (cinfo->process == JPROC_LOSSLESS) {
#ifdef C_LOSSLESS_SUPPORTED
    jinit_lossless_c_codec(cinfo);
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
  } else
    jinit_lossy_c_codec(cinfo);
}


/*
 * Initialize the decompression codec.
 * This is called only once, during master selection.
 */

GLOBAL(void)
jinit_d_codec (j_decompress_ptr cinfo)
{
  if (cinfo->process == JPROC_LOSSLESS) {
#ifdef D_LOSSLESS_SUPPORTED
    jinit_lossless_d_codec(cinfo);
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
  } else
    jinit_lossy_d_codec(cinfo);
}
