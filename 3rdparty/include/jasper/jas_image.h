/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2003 Michael David Adams.
 * All rights reserved.
 */

/* __START_OF_JASPER_LICENSE__
 * 
 * JasPer License Version 2.0
 * 
 * Copyright (c) 2001-2006 Michael David Adams
 * Copyright (c) 1999-2000 Image Power, Inc.
 * Copyright (c) 1999-2000 The University of British Columbia
 * 
 * All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person (the
 * "User") obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 * 
 * 1.  The above copyright notices and this permission notice (which
 * includes the disclaimer below) shall be included in all copies or
 * substantial portions of the Software.
 * 
 * 2.  The name of a copyright holder shall not be used to endorse or
 * promote products derived from the Software without specific prior
 * written permission.
 * 
 * THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS
 * LICENSE.  NO USE OF THE SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER
 * THIS DISCLAIMER.  THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
 * INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.  NO ASSURANCES ARE
 * PROVIDED BY THE COPYRIGHT HOLDERS THAT THE SOFTWARE DOES NOT INFRINGE
 * THE PATENT OR OTHER INTELLECTUAL PROPERTY RIGHTS OF ANY OTHER ENTITY.
 * EACH COPYRIGHT HOLDER DISCLAIMS ANY LIABILITY TO THE USER FOR CLAIMS
 * BROUGHT BY ANY OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL
 * PROPERTY RIGHTS OR OTHERWISE.  AS A CONDITION TO EXERCISING THE RIGHTS
 * GRANTED HEREUNDER, EACH USER HEREBY ASSUMES SOLE RESPONSIBILITY TO SECURE
 * ANY OTHER INTELLECTUAL PROPERTY RIGHTS NEEDED, IF ANY.  THE SOFTWARE
 * IS NOT FAULT-TOLERANT AND IS NOT INTENDED FOR USE IN MISSION-CRITICAL
 * SYSTEMS, SUCH AS THOSE USED IN THE OPERATION OF NUCLEAR FACILITIES,
 * AIRCRAFT NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL
 * SYSTEMS, DIRECT LIFE SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH
 * THE FAILURE OF THE SOFTWARE OR SYSTEM COULD LEAD DIRECTLY TO DEATH,
 * PERSONAL INJURY, OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH
 * RISK ACTIVITIES").  THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY
 * EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR HIGH RISK ACTIVITIES.
 * 
 * __END_OF_JASPER_LICENSE__
 */

/*
 * Image Class
 *
 * $Id: jas_image.h,v 1.2 2008-05-26 09:41:51 vp153 Exp $
 */

#ifndef JAS_IMAGE_H
#define JAS_IMAGE_H

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <jasper/jas_config.h>
#include <jasper/jas_stream.h>
#include <jasper/jas_seq.h>
#include <jasper/jas_cm.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************\
* Constants.
\******************************************************************************/

/*
 * Miscellaneous constants.
 */

/* The threshold at which image data is no longer stored in memory. */
#define JAS_IMAGE_INMEMTHRESH	(16 * 1024 * 1024)

/*
 * Component types
 */

#define	JAS_IMAGE_CT_UNKNOWN	0x10000
#define	JAS_IMAGE_CT_COLOR(n)	((n) & 0x7fff)
#define	JAS_IMAGE_CT_OPACITY	0x08000

#define	JAS_IMAGE_CT_RGB_R	0
#define	JAS_IMAGE_CT_RGB_G	1
#define	JAS_IMAGE_CT_RGB_B	2

#define	JAS_IMAGE_CT_YCBCR_Y	0
#define	JAS_IMAGE_CT_YCBCR_CB	1
#define	JAS_IMAGE_CT_YCBCR_CR	2

#define	JAS_IMAGE_CT_GRAY_Y	0

/******************************************************************************\
* Simple types.
\******************************************************************************/

/* Image coordinate. */
typedef int_fast32_t jas_image_coord_t;

/* Color space (e.g., RGB, YCbCr). */
typedef int_fast16_t jas_image_colorspc_t;

/* Component type (e.g., color, opacity). */
typedef int_fast32_t jas_image_cmpttype_t;

/* Component sample data format (e.g., real/integer, signedness, precision). */
typedef int_fast16_t jas_image_smpltype_t;

/******************************************************************************\
* Image class and supporting classes.
\******************************************************************************/

/* Image component class. */

typedef struct {

	jas_image_coord_t tlx_;
	/* The x-coordinate of the top-left corner of the component. */

	jas_image_coord_t tly_;
	/* The y-coordinate of the top-left corner of the component. */

	jas_image_coord_t hstep_;
	/* The horizontal sampling period in units of the reference grid. */

	jas_image_coord_t vstep_;
	/* The vertical sampling period in units of the reference grid. */

	jas_image_coord_t width_;
	/* The component width in samples. */

	jas_image_coord_t height_;
	/* The component height in samples. */

#ifdef FIX_ME
	int smpltype_;
#else
	int prec_;
	/* The precision of the sample data (i.e., the number of bits per
	sample).  If the samples are signed values, this quantity
	includes the sign bit. */

	int sgnd_;
	/* The signedness of the sample data. */
#endif

	jas_stream_t *stream_;
	/* The stream containing the component data. */

	int cps_;
	/* The number of characters per sample in the stream. */

	jas_image_cmpttype_t type_;
	/* The type of component (e.g., opacity, red, green, blue, luma). */

} jas_image_cmpt_t;

/* Image class. */

typedef struct {

	jas_image_coord_t tlx_;
	/* The x-coordinate of the top-left corner of the image bounding box. */

	jas_image_coord_t tly_;
	/* The y-coordinate of the top-left corner of the image bounding box. */

	jas_image_coord_t brx_;
	/* The x-coordinate of the bottom-right corner of the image bounding
	  box (plus one). */

	jas_image_coord_t bry_;
	/* The y-coordinate of the bottom-right corner of the image bounding
	  box (plus one). */

	int numcmpts_;
	/* The number of components. */

	int maxcmpts_;
	/* The maximum number of components that this image can have (i.e., the
	  allocated size of the components array). */

	jas_image_cmpt_t **cmpts_;
	/* Per-component information. */

	jas_clrspc_t clrspc_;

	jas_cmprof_t *cmprof_;

	bool inmem_;

} jas_image_t;

/* Component parameters class. */
/* This data type exists solely/mainly for the purposes of the
  jas_image_create function. */

typedef struct {

	jas_image_coord_t tlx;
	/* The x-coordinate of the top-left corner of the component. */

	jas_image_coord_t tly;
	/* The y-coordinate of the top-left corner of the component. */

	jas_image_coord_t hstep;
	/* The horizontal sampling period in units of the reference grid. */

	jas_image_coord_t vstep;
	/* The vertical sampling period in units of the reference grid. */

	jas_image_coord_t width;
	/* The width of the component in samples. */

	jas_image_coord_t height;
	/* The height of the component in samples. */

#ifdef FIX_ME
	int smpltype;
#else
	int prec;
	/* The precision of the component sample data. */

	int sgnd;
	/* The signedness of the component sample data. */
#endif

} jas_image_cmptparm_t;

/******************************************************************************\
* File format related classes.
\******************************************************************************/

#define	JAS_IMAGE_MAXFMTS	32
/* The maximum number of image data formats supported. */

/* Image format-dependent operations. */

typedef struct {

	jas_image_t *(*decode)(jas_stream_t *in, char *opts);
	/* Decode image data from a stream. */

	int (*encode)(jas_image_t *image, jas_stream_t *out, char *opts);
	/* Encode image data to a stream. */

	int (*validate)(jas_stream_t *in);
	/* Determine if stream data is in a particular format. */

} jas_image_fmtops_t;

/* Image format information. */

typedef struct {

	int id;
	/* The ID for this format. */

	char *name;
	/* The name by which this format is identified. */

	char *ext;
	/* The file name extension associated with this format. */

	char *desc;
	/* A brief description of the format. */

	jas_image_fmtops_t ops;
	/* The operations for this format. */

} jas_image_fmtinfo_t;

/******************************************************************************\
* Image operations.
\******************************************************************************/

/* Create an image. */
jas_image_t *jas_image_create(int numcmpts,
  jas_image_cmptparm_t *cmptparms, jas_clrspc_t clrspc);

/* Create an "empty" image. */
jas_image_t *jas_image_create0(void);

/* Clone an image. */
jas_image_t *jas_image_copy(jas_image_t *image);

/* Deallocate any resources associated with an image. */
void jas_image_destroy(jas_image_t *image);

/* Get the width of the image in units of the image reference grid. */
#define jas_image_width(image) \
	((image)->brx_ - (image)->tlx_)

/* Get the height of the image in units of the image reference grid. */
#define	jas_image_height(image) \
	((image)->bry_ - (image)->tly_)

/* Get the x-coordinate of the top-left corner of the image bounding box
  on the reference grid. */
#define jas_image_tlx(image) \
	((image)->tlx_)

/* Get the y-coordinate of the top-left corner of the image bounding box
  on the reference grid. */
#define jas_image_tly(image) \
	((image)->tly_)

/* Get the x-coordinate of the bottom-right corner of the image bounding box
  on the reference grid (plus one). */
#define jas_image_brx(image) \
	((image)->brx_)

/* Get the y-coordinate of the bottom-right corner of the image bounding box
  on the reference grid (plus one). */
#define jas_image_bry(image) \
	((image)->bry_)

/* Get the number of image components. */
#define	jas_image_numcmpts(image) \
	((image)->numcmpts_)

/* Get the color model used by the image. */
#define	jas_image_clrspc(image) \
	((image)->clrspc_)

/* Set the color model for an image. */
#define jas_image_setclrspc(image, clrspc) \
	((image)->clrspc_ = (clrspc))

#define jas_image_cmpttype(image, cmptno) \
	((image)->cmpts_[(cmptno)]->type_)
#define jas_image_setcmpttype(image, cmptno, type) \
	((image)->cmpts_[(cmptno)]->type_ = (type))

/* Get the width of a component. */
#define	jas_image_cmptwidth(image, cmptno) \
	((image)->cmpts_[cmptno]->width_)

/* Get the height of a component. */
#define	jas_image_cmptheight(image, cmptno) \
	((image)->cmpts_[cmptno]->height_)

/* Get the signedness of the sample data for a component. */
#define	jas_image_cmptsgnd(image, cmptno) \
	((image)->cmpts_[cmptno]->sgnd_)

/* Get the precision of the sample data for a component. */
#define	jas_image_cmptprec(image, cmptno) \
	((image)->cmpts_[cmptno]->prec_)

/* Get the horizontal subsampling factor for a component. */
#define	jas_image_cmpthstep(image, cmptno) \
	((image)->cmpts_[cmptno]->hstep_)

/* Get the vertical subsampling factor for a component. */
#define	jas_image_cmptvstep(image, cmptno) \
	((image)->cmpts_[cmptno]->vstep_)

/* Get the x-coordinate of the top-left corner of a component. */
#define	jas_image_cmpttlx(image, cmptno) \
	((image)->cmpts_[cmptno]->tlx_)

/* Get the y-coordinate of the top-left corner of a component. */
#define	jas_image_cmpttly(image, cmptno) \
	((image)->cmpts_[cmptno]->tly_)

/* Get the x-coordinate of the bottom-right corner of a component
  (plus "one"). */
#define	jas_image_cmptbrx(image, cmptno) \
	((image)->cmpts_[cmptno]->tlx_ + (image)->cmpts_[cmptno]->width_ * \
	  (image)->cmpts_[cmptno]->hstep_)

/* Get the y-coordinate of the bottom-right corner of a component
  (plus "one"). */
#define	jas_image_cmptbry(image, cmptno) \
	((image)->cmpts_[cmptno]->tly_ + (image)->cmpts_[cmptno]->height_ * \
	  (image)->cmpts_[cmptno]->vstep_)

/* Get the raw size of an image (i.e., the nominal size of the image without
  any compression. */
uint_fast32_t jas_image_rawsize(jas_image_t *image);

/* Create an image from a stream in some specified format. */
jas_image_t *jas_image_decode(jas_stream_t *in, int fmt, char *optstr);

/* Write an image to a stream in a specified format. */
int jas_image_encode(jas_image_t *image, jas_stream_t *out, int fmt,
  char *optstr);

/* Read a rectangular region of an image component. */
/* The position and size of the rectangular region to be read is specified
relative to the component's coordinate system. */
int jas_image_readcmpt(jas_image_t *image, int cmptno,
  jas_image_coord_t x, jas_image_coord_t y, jas_image_coord_t width, jas_image_coord_t height,
  jas_matrix_t *data);

/* Write a rectangular region of an image component. */
int jas_image_writecmpt(jas_image_t *image, int cmptno,
  jas_image_coord_t x, jas_image_coord_t y, jas_image_coord_t width, jas_image_coord_t height,
  jas_matrix_t *data);

/* Delete a component from an image. */
void jas_image_delcmpt(jas_image_t *image, int cmptno);

/* Add a component to an image. */
int jas_image_addcmpt(jas_image_t *image, int cmptno,
  jas_image_cmptparm_t *cmptparm);

/* Copy a component from one image to another. */
int jas_image_copycmpt(jas_image_t *dstimage, int dstcmptno,
  jas_image_t *srcimage, int srccmptno);

#define	JAS_IMAGE_CDT_GETSGND(dtype) (((dtype) >> 7) & 1)
#define	JAS_IMAGE_CDT_SETSGND(dtype) (((dtype) & 1) << 7)
#define	JAS_IMAGE_CDT_GETPREC(dtype) ((dtype) & 0x7f)
#define	JAS_IMAGE_CDT_SETPREC(dtype) ((dtype) & 0x7f)

#define	jas_image_cmptdtype(image, cmptno) \
	(JAS_IMAGE_CDT_SETSGND((image)->cmpts_[cmptno]->sgnd_) | JAS_IMAGE_CDT_SETPREC((image)->cmpts_[cmptno]->prec_))

int jas_image_depalettize(jas_image_t *image, int cmptno, int numlutents,
  int_fast32_t *lutents, int dtype, int newcmptno);

int jas_image_readcmptsample(jas_image_t *image, int cmptno, int x, int y);
void jas_image_writecmptsample(jas_image_t *image, int cmptno, int x, int y,
  int_fast32_t v);

int jas_image_getcmptbytype(jas_image_t *image, int ctype);

/******************************************************************************\
* Image format-related operations.
\******************************************************************************/

/* Clear the table of image formats. */
void jas_image_clearfmts(void);

/* Add entry to table of image formats. */
int jas_image_addfmt(int id, char *name, char *ext, char *desc,
  jas_image_fmtops_t *ops);

/* Get the ID for the image format with the specified name. */
int jas_image_strtofmt(char *s);

/* Get the name of the image format with the specified ID. */
char *jas_image_fmttostr(int fmt);

/* Lookup image format information by the format ID. */
jas_image_fmtinfo_t *jas_image_lookupfmtbyid(int id);

/* Lookup image format information by the format name. */
jas_image_fmtinfo_t *jas_image_lookupfmtbyname(const char *name);

/* Guess the format of an image file based on its name. */
int jas_image_fmtfromname(char *filename);

/* Get the format of image data in a stream. */
int jas_image_getfmt(jas_stream_t *in);


#define	jas_image_cmprof(image)	((image)->cmprof_)
int jas_image_ishomosamp(jas_image_t *image);
int jas_image_sampcmpt(jas_image_t *image, int cmptno, int newcmptno,
  jas_image_coord_t ho, jas_image_coord_t vo, jas_image_coord_t hs,
  jas_image_coord_t vs, int sgnd, int prec);
int jas_image_writecmpt2(jas_image_t *image, int cmptno, jas_image_coord_t x,
  jas_image_coord_t y, jas_image_coord_t width, jas_image_coord_t height,
  long *buf);
int jas_image_readcmpt2(jas_image_t *image, int cmptno, jas_image_coord_t x,
  jas_image_coord_t y, jas_image_coord_t width, jas_image_coord_t height,
  long *buf);

#define	jas_image_setcmprof(image, cmprof) ((image)->cmprof_ = cmprof)
jas_image_t *jas_image_chclrspc(jas_image_t *image, jas_cmprof_t *outprof,
  int intent);
void jas_image_dump(jas_image_t *image, FILE *out);

/******************************************************************************\
* Image format-dependent operations.
\******************************************************************************/

#if !defined(EXCLUDE_JPG_SUPPORT)
/* Format-dependent operations for JPG support. */
jas_image_t *jpg_decode(jas_stream_t *in, char *optstr);
int jpg_encode(jas_image_t *image, jas_stream_t *out, char *optstr);
int jpg_validate(jas_stream_t *in);
#endif

#if !defined(EXCLUDE_MIF_SUPPORT)
/* Format-dependent operations for MIF support. */
jas_image_t *mif_decode(jas_stream_t *in, char *optstr);
int mif_encode(jas_image_t *image, jas_stream_t *out, char *optstr);
int mif_validate(jas_stream_t *in);
#endif

#if !defined(EXCLUDE_PNM_SUPPORT)
/* Format-dependent operations for PNM support. */
jas_image_t *pnm_decode(jas_stream_t *in, char *optstr);
int pnm_encode(jas_image_t *image, jas_stream_t *out, char *optstr);
int pnm_validate(jas_stream_t *in);
#endif

#if !defined(EXCLUDE_RAS_SUPPORT)
/* Format-dependent operations for Sun Rasterfile support. */
jas_image_t *ras_decode(jas_stream_t *in, char *optstr);
int ras_encode(jas_image_t *image, jas_stream_t *out, char *optstr);
int ras_validate(jas_stream_t *in);
#endif

#if !defined(EXCLUDE_BMP_SUPPORT)
/* Format-dependent operations for BMP support. */
jas_image_t *bmp_decode(jas_stream_t *in, char *optstr);
int bmp_encode(jas_image_t *image, jas_stream_t *out, char *optstr);
int bmp_validate(jas_stream_t *in);
#endif

#if !defined(EXCLUDE_JP2_SUPPORT)
/* Format-dependent operations for JP2 support. */
jas_image_t *jp2_decode(jas_stream_t *in, char *optstr);
int jp2_encode(jas_image_t *image, jas_stream_t *out, char *optstr);
int jp2_validate(jas_stream_t *in);
#endif

#if !defined(EXCLUDE_JPC_SUPPORT)
/* Format-dependent operations for JPEG-2000 code stream support. */
jas_image_t *jpc_decode(jas_stream_t *in, char *optstr);
int jpc_encode(jas_image_t *image, jas_stream_t *out, char *optstr);
int jpc_validate(jas_stream_t *in);
#endif

#if !defined(EXCLUDE_PGX_SUPPORT)
/* Format-dependent operations for PGX support. */
jas_image_t *pgx_decode(jas_stream_t *in, char *optstr);
int pgx_encode(jas_image_t *image, jas_stream_t *out, char *optstr);
int pgx_validate(jas_stream_t *in);
#endif

#ifdef __cplusplus
}
#endif

#endif
