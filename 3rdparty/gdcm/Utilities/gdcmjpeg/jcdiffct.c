/*
 * jcdiffct.c
 *
 * Copyright (C) 1994-1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains the difference buffer controller for compression.
 * This controller is the top level of the lossless JPEG compressor proper.
 * The difference buffer lies between prediction/differencing and entropy
 * encoding.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"    /* Private declarations for lossless codec */


#ifdef C_LOSSLESS_SUPPORTED

/* We use a full-image sample buffer when doing Huffman optimization,
 * and also for writing multiple-scan JPEG files.  In all cases, the
 * full-image buffer is filled during the first pass, and the scaling,
 * prediction and differencing steps are run during subsequent passes.
 */
#ifdef ENTROPY_OPT_SUPPORTED
#define FULL_SAMP_BUFFER_SUPPORTED
#else
#ifdef C_MULTISCAN_FILES_SUPPORTED
#define FULL_SAMP_BUFFER_SUPPORTED
#endif
#endif


/* Private buffer controller object */

typedef struct {
  JDIMENSION iMCU_row_num;  /* iMCU row # within image */
  JDIMENSION mcu_ctr;    /* counts MCUs processed in current row */
  int MCU_vert_offset;    /* counts MCU rows within iMCU row */
  int MCU_rows_per_iMCU_row;  /* number of such rows needed */

  JSAMPROW cur_row[MAX_COMPONENTS];  /* row of point transformed samples */
  JSAMPROW prev_row[MAX_COMPONENTS];  /* previous row of Pt'd samples */
  JDIFFARRAY diff_buf[MAX_COMPONENTS];  /* iMCU row of differences */

  /* In multi-pass modes, we need a virtual sample array for each component. */
  jvirt_sarray_ptr whole_image[MAX_COMPONENTS];
} c_diff_controller;

typedef c_diff_controller * c_diff_ptr;


/* Forward declarations */
METHODDEF(boolean) compress_data
    JPP((j_compress_ptr cinfo, JSAMPIMAGE input_buf));
#ifdef FULL_SAMP_BUFFER_SUPPORTED
METHODDEF(boolean) compress_first_pass
    JPP((j_compress_ptr cinfo, JSAMPIMAGE input_buf));
METHODDEF(boolean) compress_output
    JPP((j_compress_ptr cinfo, JSAMPIMAGE input_buf));
#endif


LOCAL(void)
start_iMCU_row (j_compress_ptr cinfo)
/* Reset within-iMCU-row counters for a new row */
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;
  c_diff_ptr diff = (c_diff_ptr) losslsc->diff_private;

  /* In an interleaved scan, an MCU row is the same as an iMCU row.
   * In a noninterleaved scan, an iMCU row has v_samp_factor MCU rows.
   * But at the bottom of the image, process only what's left.
   */
  if (cinfo->comps_in_scan > 1) {
    diff->MCU_rows_per_iMCU_row = 1;
  } else {
    if (diff->iMCU_row_num < (cinfo->total_iMCU_rows-1))
      diff->MCU_rows_per_iMCU_row = cinfo->cur_comp_info[0]->v_samp_factor;
    else
      diff->MCU_rows_per_iMCU_row = cinfo->cur_comp_info[0]->last_row_height;
  }

  diff->mcu_ctr = 0;
  diff->MCU_vert_offset = 0;
}


/*
 * Initialize for a processing pass.
 */

METHODDEF(void)
start_pass_diff (j_compress_ptr cinfo, J_BUF_MODE pass_mode)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;
  c_diff_ptr diff = (c_diff_ptr) losslsc->diff_private;

  diff->iMCU_row_num = 0;
  start_iMCU_row(cinfo);

  switch (pass_mode) {
  case JBUF_PASS_THRU:
    if (diff->whole_image[0] != NULL)
      ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    losslsc->pub.compress_data = compress_data;
    break;
#ifdef FULL_SAMP_BUFFER_SUPPORTED
  case JBUF_SAVE_AND_PASS:
    if (diff->whole_image[0] == NULL)
      ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    losslsc->pub.compress_data = compress_first_pass;
    break;
  case JBUF_CRANK_DEST:
    if (diff->whole_image[0] == NULL)
      ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    losslsc->pub.compress_data = compress_output;
    break;
#endif
  default:
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    break;
  }
}


#define SWAP_ROWS(rowa,rowb) {JSAMPROW temp; temp=rowa; rowa=rowb; rowb=temp;}

/*
 * Process some data in the single-pass case.
 * We process the equivalent of one fully interleaved MCU row ("iMCU" row)
 * per call, ie, v_samp_factor rows for each component in the image.
 * Returns TRUE if the iMCU row is completed, FALSE if suspended.
 *
 * NB: input_buf contains a plane for each component in image,
 * which we index according to the component's SOF position.
 */

METHODDEF(boolean)
compress_data (j_compress_ptr cinfo, JSAMPIMAGE input_buf)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;
  c_diff_ptr diff = (c_diff_ptr) losslsc->diff_private;
  JDIMENSION MCU_col_num;  /* index of current MCU within row */
  JDIMENSION MCU_count;    /* number of MCUs encoded */
  /* JDIMENSION last_MCU_col = cinfo->MCUs_per_row - 1; */
  JDIMENSION last_iMCU_row = cinfo->total_iMCU_rows - 1;
  int comp, ci, yoffset, samp_row, samp_rows, samps_across;
  jpeg_component_info *compptr;

  /* Loop to write as much as one whole iMCU row */
  for (yoffset = diff->MCU_vert_offset; yoffset < diff->MCU_rows_per_iMCU_row;
       yoffset++) {

    MCU_col_num = diff->mcu_ctr;

    /* Scale and predict each scanline of the MCU-row separately.
     *
     * Note: We only do this if we are at the start of a MCU-row, ie,
     * we don't want to reprocess a row suspended by the output.
     */
    if (MCU_col_num == 0) {
      for (comp = 0; comp < cinfo->comps_in_scan; comp++) {
  compptr = cinfo->cur_comp_info[comp];
  ci = compptr->component_index;
  if (diff->iMCU_row_num < last_iMCU_row)
    samp_rows = compptr->v_samp_factor;
  else {
    /* NB: can't use last_row_height here, since may not be set! */
    samp_rows = (int) (compptr->height_in_data_units % compptr->v_samp_factor);
    if (samp_rows == 0) samp_rows = compptr->v_samp_factor;
    else {
      /* Fill dummy difference rows at the bottom edge with zeros, which
       * will encode to the smallest amount of data.
       */
      for (samp_row = samp_rows; samp_row < compptr->v_samp_factor;
     samp_row++)
        MEMZERO(diff->diff_buf[ci][samp_row],
          jround_up((long) compptr->width_in_data_units,
        (long) compptr->h_samp_factor) * SIZEOF(JDIFF));
    }
  }
  samps_across = compptr->width_in_data_units;

  for (samp_row = 0; samp_row < samp_rows; samp_row++) {
    (*losslsc->scaler_scale) (cinfo,
            input_buf[ci][samp_row],
            diff->cur_row[ci], samps_across);
    (*losslsc->predict_difference[ci]) (cinfo, ci,
                diff->cur_row[ci],
                diff->prev_row[ci],
                diff->diff_buf[ci][samp_row],
                samps_across);
    SWAP_ROWS(diff->cur_row[ci], diff->prev_row[ci]);
  }
      }
    }

    /* Try to write the MCU-row (or remaining portion of suspended MCU-row). */
    MCU_count =
      (*losslsc->entropy_encode_mcus) (cinfo,
               diff->diff_buf, yoffset, MCU_col_num,
               cinfo->MCUs_per_row - MCU_col_num);
    if (MCU_count != cinfo->MCUs_per_row - MCU_col_num) {
      /* Suspension forced; update state counters and exit */
      diff->MCU_vert_offset = yoffset;
      diff->mcu_ctr += MCU_col_num;
      return FALSE;
    }

    /* Completed an MCU row, but perhaps not an iMCU row */
    diff->mcu_ctr = 0;
  }

  /* Completed the iMCU row, advance counters for next one */
  diff->iMCU_row_num++;
  start_iMCU_row(cinfo);
  return TRUE;
}


#ifdef FULL_SAMP_BUFFER_SUPPORTED

/*
 * Process some data in the first pass of a multi-pass case.
 * We process the equivalent of one fully interleaved MCU row ("iMCU" row)
 * per call, ie, v_samp_factor rows for each component in the image.
 * This amount of data is read from the source buffer and saved into the
 * virtual arrays.
 *
 * We must also emit the data to the compressor.  This is conveniently
 * done by calling compress_output() after we've loaded the current strip
 * of the virtual arrays.
 *
 * NB: input_buf contains a plane for each component in image.  All components
 * are loaded into the virtual arrays in this pass.  However, it may be that
 * only a subset of the components are emitted to the compressor during
 * this first pass; be careful about looking at the scan-dependent variables
 * (MCU dimensions, etc).
 */

METHODDEF(boolean)
compress_first_pass (j_compress_ptr cinfo, JSAMPIMAGE input_buf)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;
  c_diff_ptr diff = (c_diff_ptr) losslsc->diff_private;
  JDIMENSION last_iMCU_row = cinfo->total_iMCU_rows - 1;
  JDIMENSION samps_across;
  int ci, samp_row, samp_rows;
  JSAMPARRAY buffer[MAX_COMPONENTS];
  jpeg_component_info *compptr;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Align the virtual buffers for this component. */
    buffer[ci] = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr) cinfo, diff->whole_image[ci],
       diff->iMCU_row_num * compptr->v_samp_factor,
       (JDIMENSION) compptr->v_samp_factor, TRUE);

    /* Count non-dummy sample rows in this iMCU row. */
    if (diff->iMCU_row_num < last_iMCU_row)
      samp_rows = compptr->v_samp_factor;
    else {
      /* NB: can't use last_row_height here, since may not be set! */
      samp_rows = (int) (compptr->height_in_data_units % compptr->v_samp_factor);
      if (samp_rows == 0) samp_rows = compptr->v_samp_factor;
    }
    samps_across = compptr->width_in_data_units;

    /* Perform point transform scaling and prediction/differencing for all
     * non-dummy rows in this iMCU row.  Each call on these functions
     * process a complete row of samples.
     */
    for (samp_row = 0; samp_row < samp_rows; samp_row++) {
      MEMCOPY(buffer[ci][samp_row], input_buf[ci][samp_row],
        samps_across * SIZEOF(JSAMPLE));
    }
  }

  /* NB: compress_output will increment iMCU_row_num if successful.
   * A suspension return will result in redoing all the work above next time.
   */

  /* Emit data to the compressor, sharing code with subsequent passes */
  return compress_output(cinfo, input_buf);
}


/*
 * Process some data in subsequent passes of a multi-pass case.
 * We process the equivalent of one fully interleaved MCU row ("iMCU" row)
 * per call, ie, v_samp_factor rows for each component in the scan.
 * The data is obtained from the virtual arrays and fed to the compressor.
 * Returns TRUE if the iMCU row is completed, FALSE if suspended.
 *
 * NB: input_buf is ignored; it is likely to be a NULL pointer.
 */

METHODDEF(boolean)
compress_output (j_compress_ptr cinfo, JSAMPIMAGE input_buf)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;
  c_diff_ptr diff = (c_diff_ptr) losslsc->diff_private;
  /* JDIMENSION MCU_col_num; */  /* index of current MCU within row */
  /* JDIMENSION MCU_count; */  /* number of MCUs encoded */
  int comp, ci /* , yoffset */ ;
  JSAMPARRAY buffer[MAX_COMPONENTS];
  jpeg_component_info *compptr;
  (void)input_buf;

  /* Align the virtual buffers for the components used in this scan.
   * NB: during first pass, this is safe only because the buffers will
   * already be aligned properly, so jmemmgr.c won't need to do any I/O.
   */
  for (comp = 0; comp < cinfo->comps_in_scan; comp++) {
    compptr = cinfo->cur_comp_info[comp];
    ci = compptr->component_index;
    buffer[ci] = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr) cinfo, diff->whole_image[ci],
       diff->iMCU_row_num * compptr->v_samp_factor,
       (JDIMENSION) compptr->v_samp_factor, FALSE);
  }

  return compress_data(cinfo, buffer);
}

#endif /* FULL_SAMP_BUFFER_SUPPORTED */


/*
 * Initialize difference buffer controller.
 */

GLOBAL(void)
jinit_c_diff_controller (j_compress_ptr cinfo, boolean need_full_buffer)
{
  j_lossless_c_ptr losslsc = (j_lossless_c_ptr) cinfo->codec;
  c_diff_ptr diff;
  int ci, row;
  jpeg_component_info *compptr;

  diff = (c_diff_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
        SIZEOF(c_diff_controller));
  losslsc->diff_private = (void *) diff;
  losslsc->diff_start_pass = start_pass_diff;

  /* Create the prediction row buffers. */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    diff->cur_row[ci] = *(*cinfo->mem->alloc_sarray)
      ((j_common_ptr) cinfo, JPOOL_IMAGE,
       (JDIMENSION) jround_up((long) compptr->width_in_data_units,
            (long) compptr->h_samp_factor),
       (JDIMENSION) 1);
    diff->prev_row[ci] = *(*cinfo->mem->alloc_sarray)
      ((j_common_ptr) cinfo, JPOOL_IMAGE,
       (JDIMENSION) jround_up((long) compptr->width_in_data_units,
            (long) compptr->h_samp_factor),
       (JDIMENSION) 1);
  }

  /* Create the difference buffer. */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    diff->diff_buf[ci] = (*cinfo->mem->alloc_darray)
      ((j_common_ptr) cinfo, JPOOL_IMAGE,
       (JDIMENSION) jround_up((long) compptr->width_in_data_units,
            (long) compptr->h_samp_factor),
       (JDIMENSION) compptr->v_samp_factor);
    /* Prefill difference rows with zeros.  We do this because only actual
     * data is placed in the buffers during prediction/differencing, leaving
     * any dummy differences at the right edge as zeros, which will encode
     * to the smallest amount of data.
     */
    for (row = 0; row < compptr->v_samp_factor; row++)
      MEMZERO(diff->diff_buf[ci][row],
        jround_up((long) compptr->width_in_data_units,
      (long) compptr->h_samp_factor) * SIZEOF(JDIFF));
  }

  /* Create the sample buffer. */
  if (need_full_buffer) {
#ifdef FULL_SAMP_BUFFER_SUPPORTED
    /* Allocate a full-image virtual array for each component, */
    /* padded to a multiple of samp_factor differences in each direction. */
    int ci;
    jpeg_component_info *compptr;

    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
   ci++, compptr++) {
      diff->whole_image[ci] = (*cinfo->mem->request_virt_sarray)
  ((j_common_ptr) cinfo, JPOOL_IMAGE, FALSE,
   (JDIMENSION) jround_up((long) compptr->width_in_data_units,
        (long) compptr->h_samp_factor),
   (JDIMENSION) jround_up((long) compptr->height_in_data_units,
        (long) compptr->v_samp_factor),
   (JDIMENSION) compptr->v_samp_factor);
    }
#else
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
#endif
  } else
    diff->whole_image[0] = NULL; /* flag for no virtual arrays */
}

#endif /* C_LOSSLESS_SUPPORTED */
