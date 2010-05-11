/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2002 Michael David Adams.
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
 * JP2 Library
 *
 * $Id: jp2_cod.h,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

#ifndef JP2_COD_H
#define JP2_COD_H

/******************************************************************************\
* Includes.
\******************************************************************************/

#include "jasper/jas_types.h"

/******************************************************************************\
* Macros.
\******************************************************************************/

#define	JP2_SPTOBPC(s, p) \
	((((p) - 1) & 0x7f) | (((s) & 1) << 7))

/******************************************************************************\
* Box class.
\******************************************************************************/

#define	JP2_BOX_HDRLEN(ext) ((ext) ? 16 : 8)

/* Box types. */
#define	JP2_BOX_JP		0x6a502020	/* Signature */
#define JP2_BOX_FTYP	0x66747970	/* File Type */
#define	JP2_BOX_JP2H	0x6a703268	/* JP2 Header */
#define	JP2_BOX_IHDR	0x69686472	/* Image Header */
#define	JP2_BOX_BPCC	0x62706363	/* Bits Per Component */
#define	JP2_BOX_COLR	0x636f6c72	/* Color Specification */
#define	JP2_BOX_PCLR	0x70636c72	/* Palette */
#define	JP2_BOX_CMAP	0x636d6170	/* Component Mapping */
#define	JP2_BOX_CDEF	0x63646566	/* Channel Definition */
#define	JP2_BOX_RES		0x72657320	/* Resolution */
#define	JP2_BOX_RESC	0x72657363	/* Capture Resolution */
#define	JP2_BOX_RESD	0x72657364	/* Default Display Resolution */
#define	JP2_BOX_JP2C	0x6a703263	/* Contiguous Code Stream */
#define	JP2_BOX_JP2I	0x6a703269	/* Intellectual Property */
#define	JP2_BOX_XML		0x786d6c20	/* XML */
#define	JP2_BOX_UUID	0x75756964	/* UUID */
#define	JP2_BOX_UINF	0x75696e66	/* UUID Info */
#define	JP2_BOX_ULST	0x75637374	/* UUID List */
#define	JP2_BOX_URL		0x75726c20	/* URL */

#define	JP2_BOX_SUPER	0x01
#define	JP2_BOX_NODATA	0x02

/* JP box data. */

#define	JP2_JP_MAGIC	0x0d0a870a
#define	JP2_JP_LEN		12

typedef struct {
	uint_fast32_t magic;
} jp2_jp_t;

/* FTYP box data. */

#define	JP2_FTYP_MAXCOMPATCODES	32
#define	JP2_FTYP_MAJVER		0x6a703220
#define	JP2_FTYP_MINVER		0
#define	JP2_FTYP_COMPATCODE		JP2_FTYP_MAJVER

typedef struct {
	uint_fast32_t majver;
	uint_fast32_t minver;
	uint_fast32_t numcompatcodes;
	uint_fast32_t compatcodes[JP2_FTYP_MAXCOMPATCODES];
} jp2_ftyp_t;

/* IHDR box data. */

#define	JP2_IHDR_COMPTYPE	7
#define	JP2_IHDR_BPCNULL	255

typedef struct {
	uint_fast32_t width;
	uint_fast32_t height;
	uint_fast16_t numcmpts;
	uint_fast8_t bpc;
	uint_fast8_t comptype;
	uint_fast8_t csunk;
	uint_fast8_t ipr;
} jp2_ihdr_t;

/* BPCC box data. */

typedef struct {
	uint_fast16_t numcmpts;
	uint_fast8_t *bpcs;
} jp2_bpcc_t;

/* COLR box data. */

#define	JP2_COLR_ENUM	1
#define	JP2_COLR_ICC	2
#define	JP2_COLR_PRI	0

#define	JP2_COLR_SRGB	16
#define	JP2_COLR_SGRAY	17
#define	JP2_COLR_SYCC	18

typedef struct {
	uint_fast8_t method;
	uint_fast8_t pri;
	uint_fast8_t approx;
	uint_fast32_t csid;
	uint_fast8_t *iccp;
	int iccplen;
	/* XXX - Someday we ought to add ICC profile data here. */
} jp2_colr_t;

/* PCLR box data. */

typedef struct {
	uint_fast16_t numlutents;
	uint_fast8_t numchans;
	int_fast32_t *lutdata;
	uint_fast8_t *bpc;
} jp2_pclr_t;

/* CDEF box per-channel data. */

#define JP2_CDEF_RGB_R	1
#define JP2_CDEF_RGB_G	2
#define JP2_CDEF_RGB_B	3

#define JP2_CDEF_YCBCR_Y	1
#define JP2_CDEF_YCBCR_CB	2
#define JP2_CDEF_YCBCR_CR	3

#define	JP2_CDEF_GRAY_Y	1

#define	JP2_CDEF_TYPE_COLOR	0
#define	JP2_CDEF_TYPE_OPACITY	1
#define	JP2_CDEF_TYPE_UNSPEC	65535
#define	JP2_CDEF_ASOC_ALL	0
#define	JP2_CDEF_ASOC_NONE	65535

typedef struct {
	uint_fast16_t channo;
	uint_fast16_t type;
	uint_fast16_t assoc;
} jp2_cdefchan_t;

/* CDEF box data. */

typedef struct {
	uint_fast16_t numchans;
	jp2_cdefchan_t *ents;
} jp2_cdef_t;

typedef struct {
	uint_fast16_t cmptno;
	uint_fast8_t map;
	uint_fast8_t pcol;
} jp2_cmapent_t;

typedef struct {
	uint_fast16_t numchans;
	jp2_cmapent_t *ents;
} jp2_cmap_t;

#define	JP2_CMAP_DIRECT		0
#define	JP2_CMAP_PALETTE	1

/* Generic box. */

struct jp2_boxops_s;
typedef struct {

	struct jp2_boxops_s *ops;
	struct jp2_boxinfo_s *info;

	uint_fast32_t type;

	/* The length of the box including the (variable-length) header. */
	uint_fast32_t len;

	/* The length of the box data. */
	uint_fast32_t datalen;

	union {
		jp2_jp_t jp;
		jp2_ftyp_t ftyp;
		jp2_ihdr_t ihdr;
		jp2_bpcc_t bpcc;
		jp2_colr_t colr;
		jp2_pclr_t pclr;
		jp2_cdef_t cdef;
		jp2_cmap_t cmap;
	} data;

} jp2_box_t;

typedef struct jp2_boxops_s {
	void (*init)(jp2_box_t *box);
	void (*destroy)(jp2_box_t *box);
	int (*getdata)(jp2_box_t *box, jas_stream_t *in);
	int (*putdata)(jp2_box_t *box, jas_stream_t *out);
	void (*dumpdata)(jp2_box_t *box, FILE *out);
} jp2_boxops_t;

/******************************************************************************\
*
\******************************************************************************/

typedef struct jp2_boxinfo_s {
	int type;
	char *name;
	int flags;
	jp2_boxops_t ops;
} jp2_boxinfo_t;

/******************************************************************************\
* Box class.
\******************************************************************************/

jp2_box_t *jp2_box_create(int type);
void jp2_box_destroy(jp2_box_t *box);
jp2_box_t *jp2_box_get(jas_stream_t *in);
int jp2_box_put(jp2_box_t *box, jas_stream_t *out);

#define JP2_DTYPETOBPC(dtype) \
  ((JAS_IMAGE_CDT_GETSGND(dtype) << 7) | (JAS_IMAGE_CDT_GETPREC(dtype) - 1))
#define	JP2_BPCTODTYPE(bpc) \
  (JAS_IMAGE_CDT_SETSGND(bpc >> 7) | JAS_IMAGE_CDT_SETPREC((bpc & 0x7f) + 1))

#define ICC_CS_RGB	0x52474220
#define ICC_CS_YCBCR	0x59436272
#define ICC_CS_GRAY	0x47524159

jp2_cdefchan_t *jp2_cdef_lookup(jp2_cdef_t *cdef, int channo);


#endif
