/*
 * jclossy.c
 *
 * Copyright (C) 1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains the control logic for the lossy JPEG compressor.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossy.h"


/*
 * Initialize for a processing pass.
 */

METHODDEF(void)
start_pass (j_compress_ptr cinfo, J_BUF_MODE pass_mode)
{
  j_lossy_c_ptr lossyc = (j_lossy_c_ptr) cinfo->codec;

  (*lossyc->fdct_start_pass) (cinfo);
  (*lossyc->coef_start_pass) (cinfo, pass_mode);
}


/*
 * Initialize the lossy compression codec.
 * This is called only once, during master selection.
 */

GLOBAL(void)
jinit_lossy_c_codec (j_compress_ptr cinfo)
{
  j_lossy_c_ptr lossyc;

  /* Create subobject in permanent pool */
  lossyc = (j_lossy_c_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
        SIZEOF(jpeg_lossy_c_codec));
  cinfo->codec = (struct jpeg_c_codec *) lossyc;

  /* Initialize sub-modules */

  /* Forward DCT */
  jinit_forward_dct(cinfo);
  /* Entropy encoding: either Huffman or arithmetic coding. */
  if (cinfo->arith_code) {
#ifdef WITH_ARITHMETIC_PATCH
    jinit_arith_encoder(cinfo);
#else
    ERREXIT(cinfo, JERR_ARITH_NOTIMPL);
#endif
  } else {
    if (cinfo->process == JPROC_PROGRESSIVE) {
#ifdef C_PROGRESSIVE_SUPPORTED
      jinit_phuff_encoder(cinfo);
#else
      ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
    } else
      jinit_shuff_encoder(cinfo);
  }

  /* Need a full-image coefficient buffer in any multi-pass mode. */
  jinit_c_coef_controller(cinfo,
        (boolean) (cinfo->num_scans > 1 ||
             cinfo->optimize_coding));

  /* Initialize method pointers.
   *
   * Note: entropy_start_pass and entropy_finish_pass are assigned in
   * jcshuff.c or jcphuff.c and compress_data is assigned in jccoefct.c.
   */
  lossyc->pub.start_pass = start_pass;
}
