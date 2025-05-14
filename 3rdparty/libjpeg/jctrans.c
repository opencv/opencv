/*
 * jctrans.c
 *
 * Copyright (C) 1995-1998, Thomas G. Lane.
 * Modified 2000-2020 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains library routines for transcoding compression,
 * that is, writing raw DCT coefficient arrays to an output JPEG file.
 * The routines in jcapimin.c will also be needed by a transcoder.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


/* Forward declarations */
LOCAL(void) transencode_master_selection
	JPP((j_compress_ptr cinfo, jvirt_barray_ptr * coef_arrays));
LOCAL(void) transencode_coef_controller
	JPP((j_compress_ptr cinfo, jvirt_barray_ptr * coef_arrays));


/*
 * Compression initialization for writing raw-coefficient data.
 * Before calling this, all parameters and a data destination must be set up.
 * Call jpeg_finish_compress() to actually write the data.
 *
 * The number of passed virtual arrays must match cinfo->num_components.
 * Note that the virtual arrays need not be filled or even realized at
 * the time write_coefficients is called; indeed, if the virtual arrays
 * were requested from this compression object's memory manager, they
 * typically will be realized during this routine and filled afterwards.
 */

GLOBAL(void)
jpeg_write_coefficients (j_compress_ptr cinfo, jvirt_barray_ptr * coef_arrays)
{
  if (cinfo->global_state != CSTATE_START)
    ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);
  /* Mark all tables to be written */
  jpeg_suppress_tables(cinfo, FALSE);
  /* (Re)initialize error mgr and destination modules */
  (*cinfo->err->reset_error_mgr) ((j_common_ptr) cinfo);
  (*cinfo->dest->init_destination) (cinfo);
  /* Perform master selection of active modules */
  transencode_master_selection(cinfo, coef_arrays);
  /* Wait for jpeg_finish_compress() call */
  cinfo->next_scanline = 0;	/* so jpeg_write_marker works */
  cinfo->global_state = CSTATE_WRCOEFS;
}


/*
 * Initialize the compression object with default parameters,
 * then copy from the source object all parameters needed for lossless
 * transcoding.  Parameters that can be varied without loss (such as
 * scan script and Huffman optimization) are left in their default states.
 */

GLOBAL(void)
jpeg_copy_critical_parameters (j_decompress_ptr srcinfo,
			       j_compress_ptr dstinfo)
{
  JQUANT_TBL ** qtblptr;
  jpeg_component_info *incomp, *outcomp;
  JQUANT_TBL *c_quant, *slot_quant;
  int tblno, ci, coefi;

  /* Safety check to ensure start_compress not called yet. */
  if (dstinfo->global_state != CSTATE_START)
    ERREXIT1(dstinfo, JERR_BAD_STATE, dstinfo->global_state);
  /* Copy fundamental image dimensions */
  dstinfo->image_width = srcinfo->image_width;
  dstinfo->image_height = srcinfo->image_height;
  dstinfo->input_components = srcinfo->num_components;
  dstinfo->in_color_space = srcinfo->jpeg_color_space;
  dstinfo->jpeg_width = srcinfo->output_width;
  dstinfo->jpeg_height = srcinfo->output_height;
  dstinfo->min_DCT_h_scaled_size = srcinfo->min_DCT_h_scaled_size;
  dstinfo->min_DCT_v_scaled_size = srcinfo->min_DCT_v_scaled_size;
  /* Initialize all parameters to default values */
  jpeg_set_defaults(dstinfo);
  /* jpeg_set_defaults may choose wrong colorspace, eg YCbCr if input is RGB.
   * Fix it to get the right header markers for the image colorspace.
   * Note: Entropy table assignment in jpeg_set_colorspace
   * depends on color_transform.
   * Adaption is also required for setting the appropriate
   * entropy coding mode dependent on image data precision.
   */
  dstinfo->color_transform = srcinfo->color_transform;
  jpeg_set_colorspace(dstinfo, srcinfo->jpeg_color_space);
  dstinfo->data_precision = srcinfo->data_precision;
  dstinfo->arith_code = srcinfo->data_precision > 8 ? TRUE : FALSE;
  dstinfo->CCIR601_sampling = srcinfo->CCIR601_sampling;
  /* Copy the source's quantization tables. */
  for (tblno = 0; tblno < NUM_QUANT_TBLS; tblno++) {
    if (srcinfo->quant_tbl_ptrs[tblno] != NULL) {
      qtblptr = & dstinfo->quant_tbl_ptrs[tblno];
      if (*qtblptr == NULL)
	*qtblptr = jpeg_alloc_quant_table((j_common_ptr) dstinfo);
      MEMCOPY((*qtblptr)->quantval,
	      srcinfo->quant_tbl_ptrs[tblno]->quantval,
	      SIZEOF((*qtblptr)->quantval));
      (*qtblptr)->sent_table = FALSE;
    }
  }
  /* Copy the source's per-component info.
   * Note we assume jpeg_set_defaults has allocated the dest comp_info array.
   */
  dstinfo->num_components = srcinfo->num_components;
  if (dstinfo->num_components < 1 || dstinfo->num_components > MAX_COMPONENTS)
    ERREXIT2(dstinfo, JERR_COMPONENT_COUNT, dstinfo->num_components,
	     MAX_COMPONENTS);
  for (ci = 0, incomp = srcinfo->comp_info, outcomp = dstinfo->comp_info;
       ci < dstinfo->num_components; ci++, incomp++, outcomp++) {
    outcomp->component_id = incomp->component_id;
    outcomp->h_samp_factor = incomp->h_samp_factor;
    outcomp->v_samp_factor = incomp->v_samp_factor;
    outcomp->quant_tbl_no = incomp->quant_tbl_no;
    /* Make sure saved quantization table for component matches the qtable
     * slot.  If not, the input file re-used this qtable slot.
     * IJG encoder currently cannot duplicate this.
     */
    tblno = outcomp->quant_tbl_no;
    if (tblno < 0 || tblno >= NUM_QUANT_TBLS ||
	srcinfo->quant_tbl_ptrs[tblno] == NULL)
      ERREXIT1(dstinfo, JERR_NO_QUANT_TABLE, tblno);
    slot_quant = srcinfo->quant_tbl_ptrs[tblno];
    c_quant = incomp->quant_table;
    if (c_quant != NULL) {
      for (coefi = 0; coefi < DCTSIZE2; coefi++) {
	if (c_quant->quantval[coefi] != slot_quant->quantval[coefi])
	  ERREXIT1(dstinfo, JERR_MISMATCHED_QUANT_TABLE, tblno);
      }
    }
    /* Note: we do not copy the source's entropy table assignments;
     * instead we rely on jpeg_set_colorspace to have made a suitable choice.
     */
  }
  /* Also copy JFIF version and resolution information, if available.
   * Strictly speaking this isn't "critical" info, but it's nearly
   * always appropriate to copy it if available.  In particular,
   * if the application chooses to copy JFIF 1.02 extension markers from
   * the source file, we need to copy the version to make sure we don't
   * emit a file that has 1.02 extensions but a claimed version of 1.01.
   */
  if (srcinfo->saw_JFIF_marker) {
    if (srcinfo->JFIF_major_version == 1 ||
	srcinfo->JFIF_major_version == 2) {
      dstinfo->JFIF_major_version = srcinfo->JFIF_major_version;
      dstinfo->JFIF_minor_version = srcinfo->JFIF_minor_version;
    }
    dstinfo->density_unit = srcinfo->density_unit;
    dstinfo->X_density = srcinfo->X_density;
    dstinfo->Y_density = srcinfo->Y_density;
  }
}


LOCAL(void)
jpeg_calc_trans_dimensions (j_compress_ptr cinfo)
/* Do computations that are needed before master selection phase */
{
  if (cinfo->min_DCT_h_scaled_size != cinfo->min_DCT_v_scaled_size)
    ERREXIT2(cinfo, JERR_BAD_DCTSIZE,
	     cinfo->min_DCT_h_scaled_size, cinfo->min_DCT_v_scaled_size);

  cinfo->block_size = cinfo->min_DCT_h_scaled_size;
}


/*
 * Master selection of compression modules for transcoding.
 * This substitutes for jcinit.c's initialization of the full compressor.
 */

LOCAL(void)
transencode_master_selection (j_compress_ptr cinfo,
			      jvirt_barray_ptr * coef_arrays)
{
  /* Do computations that are needed before master selection phase */
  jpeg_calc_trans_dimensions(cinfo);

  /* Initialize master control (includes parameter checking/processing) */
  jinit_c_master_control(cinfo, TRUE /* transcode only */);

  /* Entropy encoding: either Huffman or arithmetic coding. */
  if (cinfo->arith_code)
    jinit_arith_encoder(cinfo);
  else {
    jinit_huff_encoder(cinfo);
  }

  /* We need a special coefficient buffer controller. */
  transencode_coef_controller(cinfo, coef_arrays);

  jinit_marker_writer(cinfo);

  /* We can now tell the memory manager to allocate virtual arrays. */
  (*cinfo->mem->realize_virt_arrays) ((j_common_ptr) cinfo);

  /* Write the datastream header (SOI, JFIF) immediately.
   * Frame and scan headers are postponed till later.
   * This lets application insert special markers after the SOI.
   */
  (*cinfo->marker->write_file_header) (cinfo);
}


/*
 * The rest of this file is a special implementation of the coefficient
 * buffer controller.  This is similar to jccoefct.c, but it handles only
 * output from presupplied virtual arrays.  Furthermore, we generate any
 * dummy padding blocks on-the-fly rather than expecting them to be present
 * in the arrays.
 */

/* Private buffer controller object */

typedef struct {
  struct jpeg_c_coef_controller pub; /* public fields */

  JDIMENSION iMCU_row_num;	/* iMCU row # within image */
  JDIMENSION MCU_ctr;		/* counts MCUs processed in current row */
  int MCU_vert_offset;		/* counts MCU rows within iMCU row */
  int MCU_rows_per_iMCU_row;	/* number of such rows needed */

  /* Virtual block array for each component. */
  jvirt_barray_ptr * whole_image;

  /* Workspace for constructing dummy blocks at right/bottom edges. */
  JBLOCK dummy_buffer[C_MAX_BLOCKS_IN_MCU];
} my_coef_controller;

typedef my_coef_controller * my_coef_ptr;


LOCAL(void)
start_iMCU_row (j_compress_ptr cinfo)
/* Reset within-iMCU-row counters for a new row */
{
  my_coef_ptr coef = (my_coef_ptr) cinfo->coef;

  /* In an interleaved scan, an MCU row is the same as an iMCU row.
   * In a noninterleaved scan, an iMCU row has v_samp_factor MCU rows.
   * But at the bottom of the image, process only what's left.
   */
  if (cinfo->comps_in_scan > 1) {
    coef->MCU_rows_per_iMCU_row = 1;
  } else {
    if (coef->iMCU_row_num < (cinfo->total_iMCU_rows-1))
      coef->MCU_rows_per_iMCU_row = cinfo->cur_comp_info[0]->v_samp_factor;
    else
      coef->MCU_rows_per_iMCU_row = cinfo->cur_comp_info[0]->last_row_height;
  }

  coef->MCU_ctr = 0;
  coef->MCU_vert_offset = 0;
}


/*
 * Initialize for a processing pass.
 */

METHODDEF(void)
start_pass_coef (j_compress_ptr cinfo, J_BUF_MODE pass_mode)
{
  my_coef_ptr coef = (my_coef_ptr) cinfo->coef;

  if (pass_mode != JBUF_CRANK_DEST)
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);

  coef->iMCU_row_num = 0;
  start_iMCU_row(cinfo);
}


/*
 * Process some data.
 * We process the equivalent of one fully interleaved MCU row ("iMCU" row)
 * per call, ie, v_samp_factor block rows for each component in the scan.
 * The data is obtained from the virtual arrays and fed to the entropy coder.
 * Returns TRUE if the iMCU row is completed, FALSE if suspended.
 *
 * NB: input_buf is ignored; it is likely to be a NULL pointer.
 */

METHODDEF(boolean)
compress_output (j_compress_ptr cinfo, JSAMPIMAGE input_buf)
{
  my_coef_ptr coef = (my_coef_ptr) cinfo->coef;
  JDIMENSION MCU_col_num;	/* index of current MCU within row */
  JDIMENSION last_MCU_col = cinfo->MCUs_per_row - 1;
  JDIMENSION last_iMCU_row = cinfo->total_iMCU_rows - 1;
  int blkn, ci, xindex, yindex, yoffset, blockcnt;
  JDIMENSION start_col;
  JBLOCKARRAY buffer[MAX_COMPS_IN_SCAN];
  JBLOCKROW MCU_buffer[C_MAX_BLOCKS_IN_MCU];
  JBLOCKROW buffer_ptr;
  jpeg_component_info *compptr;

  /* Align the virtual buffers for the components used in this scan. */
  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    buffer[ci] = (*cinfo->mem->access_virt_barray)
      ((j_common_ptr) cinfo, coef->whole_image[compptr->component_index],
       coef->iMCU_row_num * compptr->v_samp_factor,
       (JDIMENSION) compptr->v_samp_factor, FALSE);
  }

  /* Loop to process one whole iMCU row */
  for (yoffset = coef->MCU_vert_offset; yoffset < coef->MCU_rows_per_iMCU_row;
       yoffset++) {
    for (MCU_col_num = coef->MCU_ctr; MCU_col_num <= last_MCU_col;
	 MCU_col_num++) {
      /* Construct list of pointers to DCT blocks belonging to this MCU */
      blkn = 0;			/* index of current DCT block within MCU */
      for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
	compptr = cinfo->cur_comp_info[ci];
	blockcnt = (MCU_col_num < last_MCU_col) ? compptr->MCU_width
						: compptr->last_col_width;
	start_col = MCU_col_num * compptr->MCU_width;
	for (yindex = 0; yindex < compptr->MCU_height; yindex++) {
	  if (coef->iMCU_row_num < last_iMCU_row ||
	      yoffset + yindex < compptr->last_row_height) {
	    /* Fill in pointers to real blocks in this row */
	    buffer_ptr = buffer[ci][yoffset + yindex] + start_col;
	    xindex = blockcnt;
	    do {
	      MCU_buffer[blkn++] = buffer_ptr++;
	    } while (--xindex);
	    /* Dummy blocks at right edge */
	    if ((xindex = compptr->MCU_width - blockcnt) == 0)
	      continue;
	  } else {
	    /* At bottom of image, need a whole row of dummy blocks */
	    xindex = compptr->MCU_width;
	  }
	  /* Fill in any dummy blocks needed in this row.
	   * Dummy blocks are filled in the same way as in jccoefct.c:
	   * all zeroes in the AC entries, DC entries equal to previous
	   * block's DC value.  The init routine has already zeroed the
	   * AC entries, so we need only set the DC entries correctly.
	   */
	  buffer_ptr = coef->dummy_buffer + blkn;
	  do {
	    buffer_ptr[0][0] = MCU_buffer[blkn-1][0][0];
	    MCU_buffer[blkn++] = buffer_ptr++;
	  } while (--xindex);
	}
      }
      /* Try to write the MCU. */
      if (! (*cinfo->entropy->encode_mcu) (cinfo, MCU_buffer)) {
	/* Suspension forced; update state counters and exit */
	coef->MCU_vert_offset = yoffset;
	coef->MCU_ctr = MCU_col_num;
	return FALSE;
      }
    }
    /* Completed an MCU row, but perhaps not an iMCU row */
    coef->MCU_ctr = 0;
  }
  /* Completed the iMCU row, advance counters for next one */
  coef->iMCU_row_num++;
  start_iMCU_row(cinfo);
  return TRUE;
}


/*
 * Initialize coefficient buffer controller.
 *
 * Each passed coefficient array must be the right size for that
 * coefficient: width_in_blocks wide and height_in_blocks high,
 * with unitheight at least v_samp_factor.
 */

LOCAL(void)
transencode_coef_controller (j_compress_ptr cinfo,
			     jvirt_barray_ptr * coef_arrays)
{
  my_coef_ptr coef;

  coef = (my_coef_ptr) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, SIZEOF(my_coef_controller));
  cinfo->coef = &coef->pub;
  coef->pub.start_pass = start_pass_coef;
  coef->pub.compress_data = compress_output;

  /* Save pointer to virtual arrays */
  coef->whole_image = coef_arrays;

  /* Pre-zero space for dummy DCT blocks */
  MEMZERO(coef->dummy_buffer, SIZEOF(coef->dummy_buffer));
}
