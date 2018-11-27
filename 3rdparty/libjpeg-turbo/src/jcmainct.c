/*
 * jcmainct.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * It was modified by The libjpeg-turbo Project to include only code relevant
 * to libjpeg-turbo.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains the main buffer controller for compression.
 * The main buffer lies between the pre-processor and the JPEG
 * compressor proper; it holds downsampled data in the JPEG colorspace.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


/* Private buffer controller object */

typedef struct {
  struct jpeg_c_main_controller pub; /* public fields */

  JDIMENSION cur_iMCU_row;      /* number of current iMCU row */
  JDIMENSION rowgroup_ctr;      /* counts row groups received in iMCU row */
  boolean suspended;            /* remember if we suspended output */
  J_BUF_MODE pass_mode;         /* current operating mode */

  /* If using just a strip buffer, this points to the entire set of buffers
   * (we allocate one for each component).  In the full-image case, this
   * points to the currently accessible strips of the virtual arrays.
   */
  JSAMPARRAY buffer[MAX_COMPONENTS];
} my_main_controller;

typedef my_main_controller *my_main_ptr;


/* Forward declarations */
METHODDEF(void) process_data_simple_main
        (j_compress_ptr cinfo, JSAMPARRAY input_buf, JDIMENSION *in_row_ctr,
         JDIMENSION in_rows_avail);


/*
 * Initialize for a processing pass.
 */

METHODDEF(void)
start_pass_main (j_compress_ptr cinfo, J_BUF_MODE pass_mode)
{
  my_main_ptr main_ptr = (my_main_ptr) cinfo->main;

  /* Do nothing in raw-data mode. */
  if (cinfo->raw_data_in)
    return;

  if (pass_mode != JBUF_PASS_THRU)
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);

  main_ptr->cur_iMCU_row = 0;   /* initialize counters */
  main_ptr->rowgroup_ctr = 0;
  main_ptr->suspended = FALSE;
  main_ptr->pass_mode = pass_mode;      /* save mode for use by process_data */
  main_ptr->pub.process_data = process_data_simple_main;
}


/*
 * Process some data.
 * This routine handles the simple pass-through mode,
 * where we have only a strip buffer.
 */

METHODDEF(void)
process_data_simple_main (j_compress_ptr cinfo,
                          JSAMPARRAY input_buf, JDIMENSION *in_row_ctr,
                          JDIMENSION in_rows_avail)
{
  my_main_ptr main_ptr = (my_main_ptr) cinfo->main;

  while (main_ptr->cur_iMCU_row < cinfo->total_iMCU_rows) {
    /* Read input data if we haven't filled the main buffer yet */
    if (main_ptr->rowgroup_ctr < DCTSIZE)
      (*cinfo->prep->pre_process_data) (cinfo,
                                        input_buf, in_row_ctr, in_rows_avail,
                                        main_ptr->buffer, &main_ptr->rowgroup_ctr,
                                        (JDIMENSION) DCTSIZE);

    /* If we don't have a full iMCU row buffered, return to application for
     * more data.  Note that preprocessor will always pad to fill the iMCU row
     * at the bottom of the image.
     */
    if (main_ptr->rowgroup_ctr != DCTSIZE)
      return;

    /* Send the completed row to the compressor */
    if (! (*cinfo->coef->compress_data) (cinfo, main_ptr->buffer)) {
      /* If compressor did not consume the whole row, then we must need to
       * suspend processing and return to the application.  In this situation
       * we pretend we didn't yet consume the last input row; otherwise, if
       * it happened to be the last row of the image, the application would
       * think we were done.
       */
      if (! main_ptr->suspended) {
        (*in_row_ctr)--;
        main_ptr->suspended = TRUE;
      }
      return;
    }
    /* We did finish the row.  Undo our little suspension hack if a previous
     * call suspended; then mark the main buffer empty.
     */
    if (main_ptr->suspended) {
      (*in_row_ctr)++;
      main_ptr->suspended = FALSE;
    }
    main_ptr->rowgroup_ctr = 0;
    main_ptr->cur_iMCU_row++;
  }
}


/*
 * Initialize main buffer controller.
 */

GLOBAL(void)
jinit_c_main_controller (j_compress_ptr cinfo, boolean need_full_buffer)
{
  my_main_ptr main_ptr;
  int ci;
  jpeg_component_info *compptr;

  main_ptr = (my_main_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
                                sizeof(my_main_controller));
  cinfo->main = (struct jpeg_c_main_controller *) main_ptr;
  main_ptr->pub.start_pass = start_pass_main;

  /* We don't need to create a buffer in raw-data mode. */
  if (cinfo->raw_data_in)
    return;

  /* Create the buffer.  It holds downsampled data, so each component
   * may be of a different size.
   */
  if (need_full_buffer) {
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
  } else {
    /* Allocate a strip buffer for each component */
    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
         ci++, compptr++) {
      main_ptr->buffer[ci] = (*cinfo->mem->alloc_sarray)
        ((j_common_ptr) cinfo, JPOOL_IMAGE,
         compptr->width_in_blocks * DCTSIZE,
         (JDIMENSION) (compptr->v_samp_factor * DCTSIZE));
    }
  }
}
