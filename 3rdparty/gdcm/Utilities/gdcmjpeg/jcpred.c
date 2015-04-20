/*
 * jcpred.c
 *
 * Copyright (C) 1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains sample differencing for lossless JPEG.
 *
 * In order to avoid paying the performance penalty of having to check the
 * predictor being used and the row being processed for each call of the
 * undifferencer, and to promote optimization, we have separate differencing
 * functions for each case.
 *
 * We are able to avoid duplicating source code by implementing the predictors
 * and differencers as macros.  Each of the differencing functions are
 * simply wrappers around a DIFFERENCE macro with the appropriate PREDICTOR
 * macro passed as an argument.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"    /* Private declarations for lossless codec */


#ifdef C_LOSSLESS_SUPPORTED

/* Private predictor object */

typedef struct {
  /* MCU-rows left in the restart interval for each component */
  unsigned int restart_rows_to_go[MAX_COMPONENTS];
} c_predictor;

typedef c_predictor * c_pred_ptr;

/* Forward declarations */
LOCAL(void) reset_predictor
  JPP((j_compress_ptr cinfo, int ci));
METHODDEF(void) start_pass
  JPP((j_compress_ptr cinfo));


/* Predictor for the first column of the first row: 2^(P-Pt-1) */
#define INITIAL_PREDICTORx  (1 << (cinfo->data_precision - cinfo->Al - 1))

/* Predictor for the first column of the remaining rows: Rb */
#define INITIAL_PREDICTOR2  GETJSAMPLE(prev_row[0])


/*
 * 1-Dimensional differencer routine.
 *
 * This macro implements the 1-D horizontal predictor (1).  INITIAL_PREDICTOR
 * is used as the special case predictor for the first column, which must be
 * either INITIAL_PREDICTOR2 or INITIAL_PREDICTORx.  The remaining samples
 * use PREDICTOR1.
 */

#define DIFFERENCE_1D(INITIAL_PREDICTOR) \
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec; \
  c_pred_ptr pred = (c_pred_ptr) losslsc->pred_private; \
  boolean restart = FALSE; \
  unsigned int xindex; \
  int samp, Ra; \
 \
  samp = GETJSAMPLE(input_buf[0]); \
  diff_buf[0] = samp - INITIAL_PREDICTOR; \
 \
  for (xindex = 1; xindex < width; xindex++) { \
    Ra = samp; \
    samp = GETJSAMPLE(input_buf[xindex]); \
    diff_buf[xindex] = samp - PREDICTOR1; \
  } \
 \
  /* Account for restart interval (no-op if not using restarts) */ \
  if (cinfo->restart_interval) { \
    if (--(pred->restart_rows_to_go[ci]) == 0) { \
      reset_predictor(cinfo, ci); \
      restart = TRUE; \
    } \
  }


/*
 * 2-Dimensional differencer routine.
 *
 * This macro implements the 2-D horizontal predictors (#2-7).  PREDICTOR2 is
 * used as the special case predictor for the first column.  The remaining
 * samples use PREDICTOR, which is a function of Ra, Rb, Rc.
 *
 * Because prev_row and output_buf may point to the same storage area (in an
 * interleaved image with Vi=1, for example), we must take care to buffer Rb/Rc
 * before writing the current reconstructed sample value into output_buf.
 */

#define DIFFERENCE_2D(PREDICTOR) \
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec; \
  c_pred_ptr pred = (c_pred_ptr) losslsc->pred_private; \
  unsigned int xindex; \
  int samp, Ra, Rb, Rc; \
 \
  Rb = GETJSAMPLE(prev_row[0]); \
  samp = GETJSAMPLE(input_buf[0]); \
  diff_buf[0] = samp - PREDICTOR2; \
 \
  for (xindex = 1; xindex < width; xindex++) { \
    Rc = Rb; \
    Rb = GETJSAMPLE(prev_row[xindex]); \
    Ra = samp; \
    samp = GETJSAMPLE(input_buf[xindex]); \
    diff_buf[xindex] = samp - PREDICTOR; \
  } \
 \
  /* Account for restart interval (no-op if not using restarts) */ \
  if (cinfo->restart_interval) { \
    if (--pred->restart_rows_to_go[ci] == 0) \
      reset_predictor(cinfo, ci); \
  }


/*
 * Differencers for the all rows but the first in a scan or restart interval.
 * The first sample in the row is differenced using the vertical
 * predictor (2).  The rest of the samples are differenced using the
 * predictor specified in the scan header.
 */

METHODDEF(void)
jpeg_difference1(j_compress_ptr cinfo, int ci,
     JSAMPROW input_buf, JSAMPROW prev_row,
     JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_1D(INITIAL_PREDICTOR2);
}

METHODDEF(void)
jpeg_difference2(j_compress_ptr cinfo, int ci,
     JSAMPROW input_buf, JSAMPROW prev_row,
     JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR2);
}

METHODDEF(void)
jpeg_difference3(j_compress_ptr cinfo, int ci,
     JSAMPROW input_buf, JSAMPROW prev_row,
     JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR3);
}

METHODDEF(void)
jpeg_difference4(j_compress_ptr cinfo, int ci,
     JSAMPROW input_buf, JSAMPROW prev_row,
     JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR4);
}

METHODDEF(void)
jpeg_difference5(j_compress_ptr cinfo, int ci,
     JSAMPROW input_buf, JSAMPROW prev_row,
     JDIFFROW diff_buf, JDIMENSION width)
{
  SHIFT_TEMPS
  DIFFERENCE_2D(PREDICTOR5);
}

METHODDEF(void)
jpeg_difference6(j_compress_ptr cinfo, int ci,
     JSAMPROW input_buf, JSAMPROW prev_row,
     JDIFFROW diff_buf, JDIMENSION width)
{
  SHIFT_TEMPS
  DIFFERENCE_2D(PREDICTOR6);
}

METHODDEF(void)
jpeg_difference7(j_compress_ptr cinfo, int ci,
     JSAMPROW input_buf, JSAMPROW prev_row,
     JDIFFROW diff_buf, JDIMENSION width)
{
  SHIFT_TEMPS
  DIFFERENCE_2D(PREDICTOR7);
}


/*
 * Differencer for the first row in a scan or restart interval.  The first
 * sample in the row is differenced using the special predictor constant
 * x=2^(P-Pt-1).  The rest of the samples are differenced using the
 * 1-D horizontal predictor (1).
 */

METHODDEF(void)
jpeg_difference_first_row(j_compress_ptr cinfo, int ci,
     JSAMPROW input_buf, JSAMPROW prev_row,
     JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_1D(INITIAL_PREDICTORx);
  (void)prev_row;

  /*
   * Now that we have differenced the first row, we want to use the
   * differencer which corresponds to the predictor specified in the
   * scan header.
   *
   * Note that we don't to do this if we have just reset the predictor
   * for a new restart interval.
   */
  if (!restart) {
    switch (cinfo->Ss) {
    case 1:
      losslsc->predict_difference[ci] = jpeg_difference1;
      break;
    case 2:
      losslsc->predict_difference[ci] = jpeg_difference2;
      break;
    case 3:
      losslsc->predict_difference[ci] = jpeg_difference3;
      break;
    case 4:
      losslsc->predict_difference[ci] = jpeg_difference4;
      break;
    case 5:
      losslsc->predict_difference[ci] = jpeg_difference5;
      break;
    case 6:
      losslsc->predict_difference[ci] = jpeg_difference6;
      break;
    case 7:
      losslsc->predict_difference[ci] = jpeg_difference7;
      break;
    }
  }
}

/*
 * Reset predictor at the start of a pass or restart interval.
 */

LOCAL(void)
reset_predictor (j_compress_ptr cinfo, int ci)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;
  c_pred_ptr pred = (c_pred_ptr) losslsc->pred_private;

  /* Initialize restart counter */
  pred->restart_rows_to_go[ci] =
    cinfo->restart_interval / cinfo->MCUs_per_row;

  /* Set difference function to first row function */
  losslsc->predict_difference[ci] = jpeg_difference_first_row;
}


/*
 * Initialize for an input processing pass.
 */

METHODDEF(void)
start_pass (j_compress_ptr cinfo)
{
  /* j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec; */
  /* c_pred_ptr pred = (c_pred_ptr) losslsc->pred_private; */
  int ci;

  /* Check that the restart interval is an integer multiple of the number
   * of MCU in an MCU-row.
   */
  if (cinfo->restart_interval % cinfo->MCUs_per_row != 0)
    ERREXIT2(cinfo, JERR_BAD_RESTART,
       cinfo->restart_interval, cinfo->MCUs_per_row);

  /* Set predictors for start of pass */
  for (ci = 0; ci < cinfo->num_components; ci++)
    reset_predictor(cinfo, ci);
}


/*
 * Module initialization routine for the differencer.
 */

GLOBAL(void)
jinit_differencer (j_compress_ptr cinfo)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;
  c_pred_ptr pred;

  pred = (c_pred_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
        SIZEOF(c_predictor));
  losslsc->pred_private = (void *) pred;
  losslsc->predict_start_pass = start_pass;
}

#endif /* C_LOSSLESS_SUPPORTED */
