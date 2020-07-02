/*
 * jdcolor.c
 *
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2011-2019 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains output colorspace conversion routines.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


#if RANGE_BITS < 2
  /* Deliberate syntax err */
  Sorry, this code requires 2 or more range extension bits.
#endif


/* Private subobject */

typedef struct {
  struct jpeg_color_deconverter pub; /* public fields */

  /* Private state for YCbCr->RGB and BG_YCC->RGB conversion */
  int * Cr_r_tab;		/* => table for Cr to R conversion */
  int * Cb_b_tab;		/* => table for Cb to B conversion */
  INT32 * Cr_g_tab;		/* => table for Cr to G conversion */
  INT32 * Cb_g_tab;		/* => table for Cb to G conversion */

  /* Private state for RGB->Y conversion */
  INT32 * rgb_y_tab;		/* => table for RGB to Y conversion */
} my_color_deconverter;

typedef my_color_deconverter * my_cconvert_ptr;


/***************  YCbCr -> RGB conversion: most common case **************/
/*************** BG_YCC -> RGB conversion: less common case **************/
/***************    RGB -> Y   conversion: less common case **************/

/*
 * YCbCr is defined per Recommendation ITU-R BT.601-7 (03/2011),
 * previously known as Recommendation CCIR 601-1, except that Cb and Cr
 * are normalized to the range 0..MAXJSAMPLE rather than -0.5 .. 0.5.
 * sRGB (standard RGB color space) is defined per IEC 61966-2-1:1999.
 * sYCC (standard luma-chroma-chroma color space with extended gamut)
 * is defined per IEC 61966-2-1:1999 Amendment A1:2003 Annex F.
 * bg-sRGB and bg-sYCC (big gamut standard color spaces)
 * are defined per IEC 61966-2-1:1999 Amendment A1:2003 Annex G.
 * Note that the derived conversion coefficients given in some of these
 * documents are imprecise.  The general conversion equations are
 *
 *	R = Y + K * (1 - Kr) * Cr
 *	G = Y - K * (Kb * (1 - Kb) * Cb + Kr * (1 - Kr) * Cr) / (1 - Kr - Kb)
 *	B = Y + K * (1 - Kb) * Cb
 *
 *	Y = Kr * R + (1 - Kr - Kb) * G + Kb * B
 *
 * With Kr = 0.299 and Kb = 0.114 (derived according to SMPTE RP 177-1993
 * from the 1953 FCC NTSC primaries and CIE Illuminant C), K = 2 for sYCC,
 * the conversion equations to be implemented are therefore
 *
 *	R = Y + 1.402 * Cr
 *	G = Y - 0.344136286 * Cb - 0.714136286 * Cr
 *	B = Y + 1.772 * Cb
 *
 *	Y = 0.299 * R + 0.587 * G + 0.114 * B
 *
 * where Cb and Cr represent the incoming values less CENTERJSAMPLE.
 * For bg-sYCC, with K = 4, the equations are
 *
 *	R = Y + 2.804 * Cr
 *	G = Y - 0.688272572 * Cb - 1.428272572 * Cr
 *	B = Y + 3.544 * Cb
 *
 * To avoid floating-point arithmetic, we represent the fractional constants
 * as integers scaled up by 2^16 (about 4 digits precision); we have to divide
 * the products by 2^16, with appropriate rounding, to get the correct answer.
 * Notice that Y, being an integral input, does not contribute any fraction
 * so it need not participate in the rounding.
 *
 * For even more speed, we avoid doing any multiplications in the inner loop
 * by precalculating the constants times Cb and Cr for all possible values.
 * For 8-bit JSAMPLEs this is very reasonable (only 256 entries per table);
 * for 9-bit to 12-bit samples it is still acceptable.  It's not very
 * reasonable for 16-bit samples, but if you want lossless storage you
 * shouldn't be changing colorspace anyway.
 * The Cr=>R and Cb=>B values can be rounded to integers in advance; the
 * values for the G calculation are left scaled up, since we must add them
 * together before rounding.
 */

#define SCALEBITS	16	/* speediest right-shift on some machines */
#define ONE_HALF	((INT32) 1 << (SCALEBITS-1))
#define FIX(x)		((INT32) ((x) * (1L<<SCALEBITS) + 0.5))

/* We allocate one big table for RGB->Y conversion and divide it up into
 * three parts, instead of doing three alloc_small requests.  This lets us
 * use a single table base address, which can be held in a register in the
 * inner loops on many machines (more than can hold all three addresses,
 * anyway).
 */

#define R_Y_OFF		0			/* offset to R => Y section */
#define G_Y_OFF		(1*(MAXJSAMPLE+1))	/* offset to G => Y section */
#define B_Y_OFF		(2*(MAXJSAMPLE+1))	/* etc. */
#define TABLE_SIZE	(3*(MAXJSAMPLE+1))


/*
 * Initialize tables for YCbCr->RGB and BG_YCC->RGB colorspace conversion.
 */

LOCAL(void)
build_ycc_rgb_table (j_decompress_ptr cinfo)
/* Normal case, sYCC */
{
  my_cconvert_ptr cconvert = (my_cconvert_ptr) cinfo->cconvert;
  int i;
  INT32 x;
  SHIFT_TEMPS

  cconvert->Cr_r_tab = (int *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, (MAXJSAMPLE+1) * SIZEOF(int));
  cconvert->Cb_b_tab = (int *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, (MAXJSAMPLE+1) * SIZEOF(int));
  cconvert->Cr_g_tab = (INT32 *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, (MAXJSAMPLE+1) * SIZEOF(INT32));
  cconvert->Cb_g_tab = (INT32 *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, (MAXJSAMPLE+1) * SIZEOF(INT32));

  for (i = 0, x = -CENTERJSAMPLE; i <= MAXJSAMPLE; i++, x++) {
    /* i is the actual input pixel value, in the range 0..MAXJSAMPLE */
    /* The Cb or Cr value we are thinking of is x = i - CENTERJSAMPLE */
    /* Cr=>R value is nearest int to 1.402 * x */
    cconvert->Cr_r_tab[i] = (int) DESCALE(FIX(1.402) * x, SCALEBITS);
    /* Cb=>B value is nearest int to 1.772 * x */
    cconvert->Cb_b_tab[i] = (int) DESCALE(FIX(1.772) * x, SCALEBITS);
    /* Cr=>G value is scaled-up -0.714136286 * x */
    cconvert->Cr_g_tab[i] = (- FIX(0.714136286)) * x;
    /* Cb=>G value is scaled-up -0.344136286 * x */
    /* We also add in ONE_HALF so that need not do it in inner loop */
    cconvert->Cb_g_tab[i] = (- FIX(0.344136286)) * x + ONE_HALF;
  }
}


LOCAL(void)
build_bg_ycc_rgb_table (j_decompress_ptr cinfo)
/* Wide gamut case, bg-sYCC */
{
  my_cconvert_ptr cconvert = (my_cconvert_ptr) cinfo->cconvert;
  int i;
  INT32 x;
  SHIFT_TEMPS

  cconvert->Cr_r_tab = (int *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, (MAXJSAMPLE+1) * SIZEOF(int));
  cconvert->Cb_b_tab = (int *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, (MAXJSAMPLE+1) * SIZEOF(int));
  cconvert->Cr_g_tab = (INT32 *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, (MAXJSAMPLE+1) * SIZEOF(INT32));
  cconvert->Cb_g_tab = (INT32 *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, (MAXJSAMPLE+1) * SIZEOF(INT32));

  for (i = 0, x = -CENTERJSAMPLE; i <= MAXJSAMPLE; i++, x++) {
    /* i is the actual input pixel value, in the range 0..MAXJSAMPLE */
    /* The Cb or Cr value we are thinking of is x = i - CENTERJSAMPLE */
    /* Cr=>R value is nearest int to 2.804 * x */
    cconvert->Cr_r_tab[i] = (int) DESCALE(FIX(2.804) * x, SCALEBITS);
    /* Cb=>B value is nearest int to 3.544 * x */
    cconvert->Cb_b_tab[i] = (int) DESCALE(FIX(3.544) * x, SCALEBITS);
    /* Cr=>G value is scaled-up -1.428272572 * x */
    cconvert->Cr_g_tab[i] = (- FIX(1.428272572)) * x;
    /* Cb=>G value is scaled-up -0.688272572 * x */
    /* We also add in ONE_HALF so that need not do it in inner loop */
    cconvert->Cb_g_tab[i] = (- FIX(0.688272572)) * x + ONE_HALF;
  }
}


/*
 * Convert some rows of samples to the output colorspace.
 *
 * Note that we change from noninterleaved, one-plane-per-component format
 * to interleaved-pixel format.  The output buffer is therefore three times
 * as wide as the input buffer.
 *
 * A starting row offset is provided only for the input buffer.  The caller
 * can easily adjust the passed output_buf value to accommodate any row
 * offset required on that side.
 */

METHODDEF(void)
ycc_rgb_convert (j_decompress_ptr cinfo,
		 JSAMPIMAGE input_buf, JDIMENSION input_row,
		 JSAMPARRAY output_buf, int num_rows)
{
  my_cconvert_ptr cconvert = (my_cconvert_ptr) cinfo->cconvert;
  register int y, cb, cr;
  register JSAMPROW outptr;
  register JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;
  /* copy these pointers into registers if possible */
  register JSAMPLE * range_limit = cinfo->sample_range_limit;
  register int * Crrtab = cconvert->Cr_r_tab;
  register int * Cbbtab = cconvert->Cb_b_tab;
  register INT32 * Crgtab = cconvert->Cr_g_tab;
  register INT32 * Cbgtab = cconvert->Cb_g_tab;
  SHIFT_TEMPS

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      y  = GETJSAMPLE(inptr0[col]);
      cb = GETJSAMPLE(inptr1[col]);
      cr = GETJSAMPLE(inptr2[col]);
      /* Range-limiting is essential due to noise introduced by DCT losses,
       * for extended gamut (sYCC) and wide gamut (bg-sYCC) encodings.
       */
      outptr[RGB_RED]   = range_limit[y + Crrtab[cr]];
      outptr[RGB_GREEN] = range_limit[y +
			      ((int) RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
						 SCALEBITS))];
      outptr[RGB_BLUE]  = range_limit[y + Cbbtab[cb]];
      outptr += RGB_PIXELSIZE;
    }
  }
}


/**************** Cases other than YCC -> RGB ****************/


/*
 * Initialize for RGB->grayscale colorspace conversion.
 */

LOCAL(void)
build_rgb_y_table (j_decompress_ptr cinfo)
{
  my_cconvert_ptr cconvert = (my_cconvert_ptr) cinfo->cconvert;
  INT32 * rgb_y_tab;
  INT32 i;

  /* Allocate and fill in the conversion tables. */
  cconvert->rgb_y_tab = rgb_y_tab = (INT32 *) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, TABLE_SIZE * SIZEOF(INT32));

  for (i = 0; i <= MAXJSAMPLE; i++) {
    rgb_y_tab[i+R_Y_OFF] = FIX(0.299) * i;
    rgb_y_tab[i+G_Y_OFF] = FIX(0.587) * i;
    rgb_y_tab[i+B_Y_OFF] = FIX(0.114) * i + ONE_HALF;
  }
}


/*
 * Convert RGB to grayscale.
 */

METHODDEF(void)
rgb_gray_convert (j_decompress_ptr cinfo,
		  JSAMPIMAGE input_buf, JDIMENSION input_row,
		  JSAMPARRAY output_buf, int num_rows)
{
  my_cconvert_ptr cconvert = (my_cconvert_ptr) cinfo->cconvert;
  register int r, g, b;
  register INT32 * ctab = cconvert->rgb_y_tab;
  register JSAMPROW outptr;
  register JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      r = GETJSAMPLE(inptr0[col]);
      g = GETJSAMPLE(inptr1[col]);
      b = GETJSAMPLE(inptr2[col]);
      /* Y */
      outptr[col] = (JSAMPLE)
		((ctab[r+R_Y_OFF] + ctab[g+G_Y_OFF] + ctab[b+B_Y_OFF])
		 >> SCALEBITS);
    }
  }
}


/*
 * Convert some rows of samples to the output colorspace.
 * [R-G,G,B-G] to [R,G,B] conversion with modulo calculation
 * (inverse color transform).
 * This can be seen as an adaption of the general YCbCr->RGB
 * conversion equation with Kr = Kb = 0, while replacing the
 * normalization by modulo calculation.
 */

METHODDEF(void)
rgb1_rgb_convert (j_decompress_ptr cinfo,
		  JSAMPIMAGE input_buf, JDIMENSION input_row,
		  JSAMPARRAY output_buf, int num_rows)
{
  register int r, g, b;
  register JSAMPROW outptr;
  register JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      r = GETJSAMPLE(inptr0[col]);
      g = GETJSAMPLE(inptr1[col]);
      b = GETJSAMPLE(inptr2[col]);
      /* Assume that MAXJSAMPLE+1 is a power of 2, so that the MOD
       * (modulo) operator is equivalent to the bitmask operator AND.
       */
      outptr[RGB_RED]   = (JSAMPLE) ((r + g - CENTERJSAMPLE) & MAXJSAMPLE);
      outptr[RGB_GREEN] = (JSAMPLE) g;
      outptr[RGB_BLUE]  = (JSAMPLE) ((b + g - CENTERJSAMPLE) & MAXJSAMPLE);
      outptr += RGB_PIXELSIZE;
    }
  }
}


/*
 * [R-G,G,B-G] to grayscale conversion with modulo calculation
 * (inverse color transform).
 */

METHODDEF(void)
rgb1_gray_convert (j_decompress_ptr cinfo,
		   JSAMPIMAGE input_buf, JDIMENSION input_row,
		   JSAMPARRAY output_buf, int num_rows)
{
  my_cconvert_ptr cconvert = (my_cconvert_ptr) cinfo->cconvert;
  register int r, g, b;
  register INT32 * ctab = cconvert->rgb_y_tab;
  register JSAMPROW outptr;
  register JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      r = GETJSAMPLE(inptr0[col]);
      g = GETJSAMPLE(inptr1[col]);
      b = GETJSAMPLE(inptr2[col]);
      /* Assume that MAXJSAMPLE+1 is a power of 2, so that the MOD
       * (modulo) operator is equivalent to the bitmask operator AND.
       */
      r = (r + g - CENTERJSAMPLE) & MAXJSAMPLE;
      b = (b + g - CENTERJSAMPLE) & MAXJSAMPLE;
      /* Y */
      outptr[col] = (JSAMPLE)
		((ctab[r+R_Y_OFF] + ctab[g+G_Y_OFF] + ctab[b+B_Y_OFF])
		 >> SCALEBITS);
    }
  }
}


/*
 * Convert some rows of samples to the output colorspace.
 * No colorspace change, but conversion from separate-planes
 * to interleaved representation.
 */

METHODDEF(void)
rgb_convert (j_decompress_ptr cinfo,
	     JSAMPIMAGE input_buf, JDIMENSION input_row,
	     JSAMPARRAY output_buf, int num_rows)
{
  register JSAMPROW outptr;
  register JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      /* We can dispense with GETJSAMPLE() here */
      outptr[RGB_RED]   = inptr0[col];
      outptr[RGB_GREEN] = inptr1[col];
      outptr[RGB_BLUE]  = inptr2[col];
      outptr += RGB_PIXELSIZE;
    }
  }
}


/*
 * Color conversion for no colorspace change: just copy the data,
 * converting from separate-planes to interleaved representation.
 * We assume out_color_components == num_components.
 */

METHODDEF(void)
null_convert (j_decompress_ptr cinfo,
	      JSAMPIMAGE input_buf, JDIMENSION input_row,
	      JSAMPARRAY output_buf, int num_rows)
{
  register JSAMPROW outptr;
  register JSAMPROW inptr;
  register JDIMENSION count;
  register int num_comps = cinfo->num_components;
  JDIMENSION num_cols = cinfo->output_width;
  int ci;

  while (--num_rows >= 0) {
    /* It seems fastest to make a separate pass for each component. */
    for (ci = 0; ci < num_comps; ci++) {
      inptr = input_buf[ci][input_row];
      outptr = output_buf[0] + ci;
      for (count = num_cols; count > 0; count--) {
	*outptr = *inptr++;	/* don't need GETJSAMPLE() here */
	outptr += num_comps;
      }
    }
    input_row++;
    output_buf++;
  }
}


/*
 * Color conversion for grayscale: just copy the data.
 * This also works for YCC -> grayscale conversion, in which
 * we just copy the Y (luminance) component and ignore chrominance.
 */

METHODDEF(void)
grayscale_convert (j_decompress_ptr cinfo,
		   JSAMPIMAGE input_buf, JDIMENSION input_row,
		   JSAMPARRAY output_buf, int num_rows)
{
  jcopy_sample_rows(input_buf[0], (int) input_row, output_buf, 0,
		    num_rows, cinfo->output_width);
}


/*
 * Convert grayscale to RGB: just duplicate the graylevel three times.
 * This is provided to support applications that don't want to cope
 * with grayscale as a separate case.
 */

METHODDEF(void)
gray_rgb_convert (j_decompress_ptr cinfo,
		  JSAMPIMAGE input_buf, JDIMENSION input_row,
		  JSAMPARRAY output_buf, int num_rows)
{
  register JSAMPROW outptr;
  register JSAMPROW inptr;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    inptr = input_buf[0][input_row++];
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      /* We can dispense with GETJSAMPLE() here */
      outptr[RGB_RED] = outptr[RGB_GREEN] = outptr[RGB_BLUE] = inptr[col];
      outptr += RGB_PIXELSIZE;
    }
  }
}


/*
 * Convert some rows of samples to the output colorspace.
 * This version handles Adobe-style YCCK->CMYK conversion,
 * where we convert YCbCr to R=1-C, G=1-M, and B=1-Y using the
 * same conversion as above, while passing K (black) unchanged.
 * We assume build_ycc_rgb_table has been called.
 */

METHODDEF(void)
ycck_cmyk_convert (j_decompress_ptr cinfo,
		   JSAMPIMAGE input_buf, JDIMENSION input_row,
		   JSAMPARRAY output_buf, int num_rows)
{
  my_cconvert_ptr cconvert = (my_cconvert_ptr) cinfo->cconvert;
  register int y, cb, cr;
  register JSAMPROW outptr;
  register JSAMPROW inptr0, inptr1, inptr2, inptr3;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;
  /* copy these pointers into registers if possible */
  register JSAMPLE * range_limit = cinfo->sample_range_limit;
  register int * Crrtab = cconvert->Cr_r_tab;
  register int * Cbbtab = cconvert->Cb_b_tab;
  register INT32 * Crgtab = cconvert->Cr_g_tab;
  register INT32 * Cbgtab = cconvert->Cb_g_tab;
  SHIFT_TEMPS

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    inptr3 = input_buf[3][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      y  = GETJSAMPLE(inptr0[col]);
      cb = GETJSAMPLE(inptr1[col]);
      cr = GETJSAMPLE(inptr2[col]);
      /* Range-limiting is essential due to noise introduced by DCT losses,
       * and for extended gamut encodings (sYCC).
       */
      outptr[0] = range_limit[MAXJSAMPLE - (y + Crrtab[cr])];	/* red */
      outptr[1] = range_limit[MAXJSAMPLE - (y +			/* green */
			      ((int) RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
						 SCALEBITS)))];
      outptr[2] = range_limit[MAXJSAMPLE - (y + Cbbtab[cb])];	/* blue */
      /* K passes through unchanged */
      outptr[3] = inptr3[col];	/* don't need GETJSAMPLE here */
      outptr += 4;
    }
  }
}


/*
 * Empty method for start_pass.
 */

METHODDEF(void)
start_pass_dcolor (j_decompress_ptr cinfo)
{
  /* no work needed */
}


/*
 * Module initialization routine for output colorspace conversion.
 */

GLOBAL(void)
jinit_color_deconverter (j_decompress_ptr cinfo)
{
  my_cconvert_ptr cconvert;
  int ci;

  cconvert = (my_cconvert_ptr) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, SIZEOF(my_color_deconverter));
  cinfo->cconvert = &cconvert->pub;
  cconvert->pub.start_pass = start_pass_dcolor;

  /* Make sure num_components agrees with jpeg_color_space */
  switch (cinfo->jpeg_color_space) {
  case JCS_GRAYSCALE:
    if (cinfo->num_components != 1)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    break;

  case JCS_RGB:
  case JCS_YCbCr:
  case JCS_BG_RGB:
  case JCS_BG_YCC:
    if (cinfo->num_components != 3)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    break;

  case JCS_CMYK:
  case JCS_YCCK:
    if (cinfo->num_components != 4)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    break;

  default:			/* JCS_UNKNOWN can be anything */
    if (cinfo->num_components < 1)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
  }

  /* Support color transform only for RGB colorspaces */
  if (cinfo->color_transform &&
      cinfo->jpeg_color_space != JCS_RGB &&
      cinfo->jpeg_color_space != JCS_BG_RGB)
    ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);

  /* Set out_color_components and conversion method based on requested space.
   * Also clear the component_needed flags for any unused components,
   * so that earlier pipeline stages can avoid useless computation.
   */

  switch (cinfo->out_color_space) {
  case JCS_GRAYSCALE:
    cinfo->out_color_components = 1;
    switch (cinfo->jpeg_color_space) {
    case JCS_GRAYSCALE:
    case JCS_YCbCr:
    case JCS_BG_YCC:
      cconvert->pub.color_convert = grayscale_convert;
      /* For color->grayscale conversion, only the Y (0) component is needed */
      for (ci = 1; ci < cinfo->num_components; ci++)
	cinfo->comp_info[ci].component_needed = FALSE;
      break;
    case JCS_RGB:
      switch (cinfo->color_transform) {
      case JCT_NONE:
	cconvert->pub.color_convert = rgb_gray_convert;
	break;
      case JCT_SUBTRACT_GREEN:
	cconvert->pub.color_convert = rgb1_gray_convert;
	break;
      default:
	ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
      }
      build_rgb_y_table(cinfo);
      break;
    default:
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    }
    break;

  case JCS_RGB:
    cinfo->out_color_components = RGB_PIXELSIZE;
    switch (cinfo->jpeg_color_space) {
    case JCS_GRAYSCALE:
      cconvert->pub.color_convert = gray_rgb_convert;
      break;
    case JCS_YCbCr:
      cconvert->pub.color_convert = ycc_rgb_convert;
      build_ycc_rgb_table(cinfo);
      break;
    case JCS_BG_YCC:
      cconvert->pub.color_convert = ycc_rgb_convert;
      build_bg_ycc_rgb_table(cinfo);
      break;
    case JCS_RGB:
      switch (cinfo->color_transform) {
      case JCT_NONE:
	cconvert->pub.color_convert = rgb_convert;
	break;
      case JCT_SUBTRACT_GREEN:
	cconvert->pub.color_convert = rgb1_rgb_convert;
	break;
      default:
	ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
      }
      break;
    default:
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    }
    break;

  case JCS_BG_RGB:
    cinfo->out_color_components = RGB_PIXELSIZE;
    if (cinfo->jpeg_color_space != JCS_BG_RGB)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    switch (cinfo->color_transform) {
    case JCT_NONE:
      cconvert->pub.color_convert = rgb_convert;
      break;
    case JCT_SUBTRACT_GREEN:
      cconvert->pub.color_convert = rgb1_rgb_convert;
      break;
    default:
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    }
    break;

  case JCS_CMYK:
    cinfo->out_color_components = 4;
    switch (cinfo->jpeg_color_space) {
    case JCS_YCCK:
      cconvert->pub.color_convert = ycck_cmyk_convert;
      build_ycc_rgb_table(cinfo);
      break;
    case JCS_CMYK:
      cconvert->pub.color_convert = null_convert;
      break;
    default:
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    }
    break;

  default:		/* permit null conversion to same output space */
    if (cinfo->out_color_space != cinfo->jpeg_color_space)
      /* unsupported non-null conversion */
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    cinfo->out_color_components = cinfo->num_components;
    cconvert->pub.color_convert = null_convert;
  }

  if (cinfo->quantize_colors)
    cinfo->output_components = 1; /* single colormapped output component */
  else
    cinfo->output_components = cinfo->out_color_components;
}
