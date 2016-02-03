/*
 * jdpred.c
 *
 * Copyright (C) 1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains sample undifferencing (reconstruction) for lossless JPEG.
 *
 * In order to avoid paying the performance penalty of having to check the
 * predictor being used and the row being processed for each call of the
 * undifferencer, and to promote optimization, we have separate undifferencing
 * functions for each case.
 *
 * We are able to avoid duplicating source code by implementing the predictors
 * and undifferencers as macros.  Each of the undifferencing functions are
 * simply wrappers around an UNDIFFERENCE macro with the appropriate PREDICTOR
 * macro passed as an argument.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"    /* Private declarations for lossless codec */


#ifdef D_LOSSLESS_SUPPORTED

/* Predictor for the first column of the first row: 2^(P-Pt-1) */
#define INITIAL_PREDICTORx  (1 << (cinfo->data_precision - cinfo->Al - 1))

/* Predictor for the first column of the remaining rows: Rb */
#define INITIAL_PREDICTOR2  GETJSAMPLE(prev_row[0])


/*
 * 1-Dimensional undifferencer routine.
 *
 * This macro implements the 1-D horizontal predictor (1).  INITIAL_PREDICTOR
 * is used as the special case predictor for the first column, which must be
 * either INITIAL_PREDICTOR2 or INITIAL_PREDICTORx.  The remaining samples
 * use PREDICTOR1.
 *
 * The reconstructed sample is supposed to be calculated modulo 2^16, so we
 * logically AND the result with 0xFFFF.
*/

#define UNDIFFERENCE_1D(INITIAL_PREDICTOR) \
  unsigned int xindex; \
  int Ra; \
 \
  Ra = (diff_buf[0] + INITIAL_PREDICTOR) & 0xFFFF; \
  undiff_buf[0] = Ra; \
 \
  for (xindex = 1; xindex < width; xindex++) { \
    Ra = (diff_buf[xindex] + PREDICTOR1) & 0xFFFF; \
    undiff_buf[xindex] = Ra; \
  }

/*
 * 2-Dimensional undifferencer routine.
 *
 * This macro implements the 2-D horizontal predictors (#2-7).  PREDICTOR2 is
 * used as the special case predictor for the first column.  The remaining
 * samples use PREDICTOR, which is a function of Ra, Rb, Rc.
 *
 * Because prev_row and output_buf may point to the same storage area (in an
 * interleaved image with Vi=1, for example), we must take care to buffer Rb/Rc
 * before writing the current reconstructed sample value into output_buf.
 *
 * The reconstructed sample is supposed to be calculated modulo 2^16, so we
 * logically AND the result with 0xFFFF.
 */

#define UNDIFFERENCE_2D_BUG(PREDICTOR) \
  Rb = GETJSAMPLE(prev_row[0]); \
  Ra = (diff_buf[0] + PREDICTOR2) & 0xFFFF; \
  undiff_buf[0] = Ra; \
 \
  for (xindex = 1; xindex < width; xindex++) { \
    Rc = Rb; \
    Rb = GETJSAMPLE(prev_row[xindex]); \
    Ra = (diff_buf[xindex] + PREDICTOR) & 0xFFFF; \
    undiff_buf[xindex] = Ra; \
  }

#define UNDIFFERENCE_2D(PREDICTOR) \
  unsigned int xindex; \
  int Ra, Rb, Rc; \
 \
  Rb = GETJSAMPLE(prev_row[0]); \
  Ra = (diff_buf[0] + PREDICTOR2) & 0xFFFF; \
  undiff_buf[0] = Ra; \
 \
  for (xindex = 1; xindex < width; xindex++) { \
    Rc = Rb; \
    Rb = GETJSAMPLE(prev_row[xindex]); \
    Ra = (diff_buf[xindex] + PREDICTOR) & 0xFFFF; \
    undiff_buf[xindex] = Ra; \
  }


/*
 * Undifferencers for the all rows but the first in a scan or restart interval.
 * The first sample in the row is undifferenced using the vertical
 * predictor (2).  The rest of the samples are undifferenced using the
 * predictor specified in the scan header.
 */

METHODDEF(void)
jpeg_undifference1(j_decompress_ptr cinfo, int comp_index,
       JDIFFROW diff_buf, JDIFFROW prev_row,
       JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_1D(INITIAL_PREDICTOR2);
  (void)cinfo;(void)comp_index;(void)diff_buf;(void)prev_row;(void)undiff_buf;(void)width;
}

METHODDEF(void)
jpeg_undifference2(j_decompress_ptr cinfo, int comp_index,
       JDIFFROW diff_buf, JDIFFROW prev_row,
       JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR2);
  (void)cinfo;(void)comp_index;(void)diff_buf;(void)prev_row;(void)undiff_buf;(void)width;
}

METHODDEF(void)
jpeg_undifference3(j_decompress_ptr cinfo, int comp_index,
       JDIFFROW diff_buf, JDIFFROW prev_row,
       JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR3);
  (void)cinfo;(void)comp_index;(void)diff_buf;(void)prev_row;(void)undiff_buf;(void)width;
}

METHODDEF(void)
jpeg_undifference4(j_decompress_ptr cinfo, int comp_index,
       JDIFFROW diff_buf, JDIFFROW prev_row,
       JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR4);
  (void)cinfo;(void)comp_index;(void)diff_buf;(void)prev_row;(void)undiff_buf;(void)width;
}

METHODDEF(void)
jpeg_undifference5(j_decompress_ptr cinfo, int comp_index,
       JDIFFROW diff_buf, JDIFFROW prev_row,
       JDIFFROW undiff_buf, JDIMENSION width)
{
  SHIFT_TEMPS
  UNDIFFERENCE_2D(PREDICTOR5);
  (void)cinfo;(void)comp_index;(void)diff_buf;(void)prev_row;(void)undiff_buf;(void)width;
}

#ifdef SUPPORT_DICOMOBJECTS_BUG
/* uninitialized */
static int dicomobjectsbug = -1; /* 0 == nobug, 1 == bug */
#endif

METHODDEF(void)
jpeg_undifference6(j_decompress_ptr cinfo, int comp_index,
       JDIFFROW diff_buf, JDIFFROW prev_row,
       JDIFFROW undiff_buf, JDIMENSION width)
{
#ifdef SUPPORT_DICOMOBJECTS_BUG
  unsigned int xindex;
  int Ra, Rb, Rc;
  int min, max, temp;
  SHIFT_TEMPS
  if( dicomobjectsbug == -1 )
    {
    dicomobjectsbug = 0; /* no bug by default */

    Rb = GETJSAMPLE(prev_row[0]);
    Ra = (diff_buf[0] + PREDICTOR2) & 0xFFFF;
    undiff_buf[0] = Ra;
    temp = min = max = undiff_buf[0];

    for (xindex = 1; xindex < width; xindex++) {
      Rc = Rb;
      Rb = GETJSAMPLE(prev_row[xindex]);
      Ra = (diff_buf[xindex] + PREDICTOR6) & 0xFFFF;
      temp = Ra;
      min = temp < min ? temp : min;
      max = temp > max ? temp : max;
    }
    if( (max - min) > 50000) /* magic number */
      {
      dicomobjectsbug = 1;
      WARNMS(cinfo, JWRN_SIGNED_ARITH);
      }
    }
  if(dicomobjectsbug)
    {
    UNDIFFERENCE_2D_BUG(PREDICTOR6_BUG);
    }
  else
    {
    UNDIFFERENCE_2D_BUG(PREDICTOR6);
    }
#else
  SHIFT_TEMPS
  UNDIFFERENCE_2D(PREDICTOR6);
#endif
  (void)comp_index;(void)cinfo;
}

METHODDEF(void)
jpeg_undifference7(j_decompress_ptr cinfo, int comp_index,
       JDIFFROW diff_buf, JDIFFROW prev_row,
       JDIFFROW undiff_buf, JDIMENSION width)
{
  SHIFT_TEMPS
  UNDIFFERENCE_2D(PREDICTOR7);
  (void)cinfo;(void)comp_index;(void)diff_buf;(void)prev_row;(void)undiff_buf;(void)width;
}


/*
 * Undifferencer for the first row in a scan or restart interval.  The first
 * sample in the row is undifferenced using the special predictor constant
 * x=2^(P-Pt-1).  The rest of the samples are undifferenced using the
 * 1-D horizontal predictor (1).
 */

METHODDEF(void)
jpeg_undifference_first_row(j_decompress_ptr cinfo, int comp_index,
          JDIFFROW diff_buf, JDIFFROW prev_row,
          JDIFFROW undiff_buf, JDIMENSION width)
{
  j_lossless_d_ptr losslsd = (j_lossless_d_ptr) cinfo->codec;

  UNDIFFERENCE_1D(INITIAL_PREDICTORx);
  (void)prev_row;

  /*
   * Now that we have undifferenced the first row, we want to use the
   * undifferencer which corresponds to the predictor specified in the
   * scan header.
   */
  switch (cinfo->Ss) {
  case 1:
    losslsd->predict_undifference[comp_index] = jpeg_undifference1;
    break;
  case 2:
    losslsd->predict_undifference[comp_index] = jpeg_undifference2;
    break;
  case 3:
    losslsd->predict_undifference[comp_index] = jpeg_undifference3;
    break;
  case 4:
    losslsd->predict_undifference[comp_index] = jpeg_undifference4;
    break;
  case 5:
    losslsd->predict_undifference[comp_index] = jpeg_undifference5;
    break;
  case 6:
    losslsd->predict_undifference[comp_index] = jpeg_undifference6;
    break;
  case 7:
    losslsd->predict_undifference[comp_index] = jpeg_undifference7;
    break;
  }
}


/*
 * Initialize for an input processing pass.
 */

METHODDEF(void)
predict_start_pass (j_decompress_ptr cinfo)
{
  j_lossless_d_ptr losslsd = (j_lossless_d_ptr) cinfo->codec;
  int ci;

  /* Check that the scan parameters Ss, Se, Ah, Al are OK for lossless JPEG.
   *
   * Ss is the predictor selection value (psv).  Legal values for sequential
   * lossless JPEG are: 1 <= psv <= 7.
   *
   * Se and Ah are not used and should be zero.
   *
   * Al specifies the point transform (Pt).  Legal values are: 0 <= Pt <= 15.
   */
  if (cinfo->Ss < 1 || cinfo->Ss > 7 ||
      cinfo->Se != 0 || cinfo->Ah != 0 ||
      cinfo->Al > 15)        /* need not check for < 0 */
    ERREXIT4(cinfo, JERR_BAD_LOSSLESS,
       cinfo->Ss, cinfo->Se, cinfo->Ah, cinfo->Al);

  /* Set undifference functions to first row function */
  for (ci = 0; ci < cinfo->num_components; ci++)
    losslsd->predict_undifference[ci] = jpeg_undifference_first_row;
}


/*
 * Module initialization routine for the undifferencer.
 */

GLOBAL(void)
jinit_undifferencer (j_decompress_ptr cinfo)
{
  j_lossless_d_ptr losslsd = (j_lossless_d_ptr) cinfo->codec;

  losslsd->predict_start_pass = predict_start_pass;
  losslsd->predict_process_restart = predict_start_pass;
}

#endif /* D_LOSSLESS_SUPPORTED */
