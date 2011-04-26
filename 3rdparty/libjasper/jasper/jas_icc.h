/*
 * Copyright (c) 2002-2003 Michael David Adams.
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

#ifndef JAS_ICC_H
#define	JAS_ICC_H

#include <jasper/jas_config.h>
#include <jasper/jas_types.h>
#include <jasper/jas_stream.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Profile file signature. */
#define	JAS_ICC_MAGIC		0x61637370

#define	JAS_ICC_HDRLEN	128

/* Profile/device class signatures. */
#define	JAS_ICC_CLAS_IN	0x73636e72 /* input device */
#define	JAS_ICC_CLAS_DPY	0x6d6e7472 /* display device */
#define	JAS_ICC_CLAS_OUT	0x70727472 /* output device */
#define	JAS_ICC_CLAS_LNK	0x6c696e6b /* device link */
#define	JAS_ICC_CLAS_CNV	0x73706163 /* color space conversion */
#define	JAS_ICC_CLAS_ABS	0x61627374 /* abstract */
#define	JAS_ICC_CLAS_NAM	0x6e6d636c /* named color */

/* Color space signatures. */
#define	JAS_ICC_COLORSPC_XYZ	0x58595a20 /* XYZ */
#define	JAS_ICC_COLORSPC_LAB	0x4c616220 /* LAB */
#define	JAS_ICC_COLORSPC_LUV	0x4c757620 /* LUV */
#define	JAS_ICC_COLORSPC_YCBCR	0x59436272 /* YCbCr */
#define	JAS_ICC_COLORSPC_YXY	0x59787920 /* Yxy */
#define	JAS_ICC_COLORSPC_RGB	0x52474220 /* RGB */
#define	JAS_ICC_COLORSPC_GRAY	0x47524159 /* Gray */
#define	JAS_ICC_COLORSPC_HSV	0x48535620 /* HSV */
#define	JAS_ICC_COLORSPC_HLS	0x484c5320 /* HLS */
#define	JAS_ICC_COLORSPC_CMYK	0x434d594b /* CMYK */
#define	JAS_ICC_COLORSPC_CMY	0x434d5920 /* CMY */
#define	JAS_ICC_COLORSPC_2	0x32434c52 /* 2 channel color */
#define	JAS_ICC_COLORSPC_3	0x33434c52 /* 3 channel color */
#define	JAS_ICC_COLORSPC_4	0x34434c52 /* 4 channel color */
#define	JAS_ICC_COLORSPC_5	0x35434c52 /* 5 channel color */
#define	JAS_ICC_COLORSPC_6	0x36434c52 /* 6 channel color */
#define	JAS_ICC_COLORSPC_7	0x37434c52 /* 7 channel color */
#define	JAS_ICC_COLORSPC_8	0x38434c52 /* 8 channel color */
#define	JAS_ICC_COLORSPC_9	0x39434c52 /* 9 channel color */
#define	JAS_ICC_COLORSPC_10	0x41434c52 /* 10 channel color */
#define	JAS_ICC_COLORSPC_11	0x42434c52 /* 11 channel color */
#define	JAS_ICC_COLORSPC_12	0x43434c52 /* 12 channel color */
#define	JAS_ICC_COLORSPC_13	0x44434c52 /* 13 channel color */
#define	JAS_ICC_COLORSPC_14	0x45434c52 /* 14 channel color */
#define	JAS_ICC_COLORSPC_15	0x46434c52 /* 15 channel color */

/* Profile connection color space (PCS) signatures. */
#define	JAS_ICC_REFCOLORSPC_XYZ		0x58595a20 /* CIE XYZ */
#define	JAS_ICC_REFCOLORSPC_LAB		0x4c616220 /* CIE Lab */

/* Primary platform signatures. */
#define	JAS_ICC_PLATFORM_APPL	0x4150504c /* Apple Computer */
#define	JAS_ICC_PLATFORM_MSFT	0x4d534654 /* Microsoft */
#define	JAS_ICC_PLATFORM_SGI	0x53474920 /* Silicon Graphics */
#define	JAS_ICC_PLATFORM_SUNW	0x53554e57 /* Sun Microsystems */
#define	JAS_ICC_PLATFORM_TGNT	0x54474e54 /* Taligent */

/* Profile flags. */
#define	JAS_ICC_FLAGS_EMBED	0x01 /* embedded */
#define	JAS_ICC_FLAGS_NOSEP	0x02 /* no separate use */

/* Attributes. */
#define	JAS_ICC_ATTR_TRANS	0x01 /* transparent */
#define	JAS_ICC_ATTR_MATTE	0x02 /* matte */

/* Rendering intents. */
#define	JAS_ICC_INTENT_PER	0 /* perceptual */
#define	JAS_ICC_INTENT_REL	1 /* relative colorimetric */
#define	JAS_ICC_INTENT_SAT	2 /* saturation */
#define	JAS_ICC_INTENT_ABS	3 /* absolute colorimetric */

/* Tag signatures. */
#define	JAS_ICC_TAG_ATOB0		0x41324230 /* */
#define	JAS_ICC_TAG_ATOB1		0x41324231 /* */
#define	JAS_ICC_TAG_ATOB2		0x41324232 /* */
#define	JAS_ICC_TAG_BLUMATCOL		0x6258595a /* */
#define	JAS_ICC_TAG_BLUTRC		0x62545243 /* */
#define	JAS_ICC_TAG_BTOA0		0x42324130 /* */
#define	JAS_ICC_TAG_BTOA1		0x42324131 /* */
#define	JAS_ICC_TAG_BTOA2		0x42324132 /* */
#define	JAS_ICC_TAG_CALTIME		0x63616c74 /* */
#define	JAS_ICC_TAG_CHARTARGET		0x74617267 /* */
#define	JAS_ICC_TAG_CPYRT		0x63707274 /* */
#define	JAS_ICC_TAG_CRDINFO		0x63726469 /* */
#define	JAS_ICC_TAG_DEVMAKERDESC	0x646d6e64 /* */
#define	JAS_ICC_TAG_DEVMODELDESC	0x646d6464 /* */
#define	JAS_ICC_TAG_DEVSET		0x64657673 /* */
#define	JAS_ICC_TAG_GAMUT		0x67616d74 /* */
#define	JAS_ICC_TAG_GRYTRC		0x6b545243 /* */
#define	JAS_ICC_TAG_GRNMATCOL		0x6758595a /* */
#define	JAS_ICC_TAG_GRNTRC		0x67545243 /* */
#define	JAS_ICC_TAG_LUM			0x6c756d69 /* */
#define	JAS_ICC_TAG_MEASURE		0x6d656173 /* */
#define	JAS_ICC_TAG_MEDIABLKPT		0x626b7074 /* */
#define	JAS_ICC_TAG_MEDIAWHIPT		0x77747074 /* */
#define	JAS_ICC_TAG_NAMCOLR		0x6e636f6c /* */
#define	JAS_ICC_TAG_NAMCOLR2		0x6e636c32 /* */
#define	JAS_ICC_TAG_OUTRESP		0x72657370 /* */
#define	JAS_ICC_TAG_PREVIEW0		0x70726530 /* */
#define	JAS_ICC_TAG_PREVIEW1		0x70726531 /* */
#define	JAS_ICC_TAG_PREVIEW2		0x70726532 /* */
#define	JAS_ICC_TAG_PROFDESC		0x64657363 /* */
#define	JAS_ICC_TAG_PROFSEQDESC		0x70736571 /* */
#define	JAS_ICC_TAG_PSDCRD0		0x70736430 /* */
#define	JAS_ICC_TAG_PSCRDD1		0x70736431 /* */
#define	JAS_ICC_TAG_PSCRDD2		0x70736432 /* */
#define	JAS_ICC_TAG_PSCRDD3		0x70736433 /* */
#define	JAS_ICC_TAG_PS2CSA		0x70733273 /* */
#define	JAS_ICC_TAG_PS2RENINTENT	0x70733269 /* */
#define	JAS_ICC_TAG_REDMATCOL		0x7258595a /* */
#define	JAS_ICC_TAG_REDTRC		0x72545243 /* */
#define	JAS_ICC_TAG_SCRNGDES		0x73637264 /* */
#define	JAS_ICC_TAG_SCRNG		0x7363726e /* */
#define	JAS_ICC_TAG_TECH		0x74656368 /* */
#define	JAS_ICC_TAG_UCRBG		0x62666420 /* */
#define	JAS_ICC_TAG_VIEWCONDDESC	0x76756564 /* */
#define	JAS_ICC_TAG_VIEWCOND		0x76696577 /* */

/* Type signatures. */
#define	JAS_ICC_TYPE_CRDINFO		0x63726469 /* CRD information */
#define	JAS_ICC_TYPE_CURV		0x63757276 /* curve */
#define	JAS_ICC_TYPE_DATA		0x64617461 /* data */
#define	JAS_ICC_TYPE_TIME		0x6474696d /* date/time */
#define	JAS_ICC_TYPE_DEVSET		0x64657673 /* device settings */
#define	JAS_ICC_TYPE_LUT16		0x6d667432 /* */
#define	JAS_ICC_TYPE_LUT8		0x6d667431 /* */
#define	JAS_ICC_TYPE_MEASURE		0x6d656173 /* */
#define	JAS_ICC_TYPE_NAMCOLR		0x6e636f6c /* */
#define	JAS_ICC_TYPE_NAMCOLR2		0x6e636c32 /* */
#define	JAS_ICC_TYPE_PROFSEQDESC	0x70736571 /* profile sequence description */
#define	JAS_ICC_TYPE_RESPCURVSET16	0x72637332 /* response curve set 16 */
#define	JAS_ICC_TYPE_SF32		0x73663332 /* signed 32-bit fixed-point */
#define	JAS_ICC_TYPE_SCRNG		0x7363726e /* screening */
#define	JAS_ICC_TYPE_SIG		0x73696720 /* signature */
#define	JAS_ICC_TYPE_TXTDESC		0x64657363 /* text description */
#define	JAS_ICC_TYPE_TXT		0x74657874 /* text */
#define	JAS_ICC_TYPE_UF32		0x75663332 /* unsigned 32-bit fixed-point */
#define	JAS_ICC_TYPE_UCRBG		0x62666420 /* */
#define	JAS_ICC_TYPE_UI16		0x75693136 /* */
#define	JAS_ICC_TYPE_UI32		0x75693332 /* */
#define	JAS_ICC_TYPE_UI8		0x75693038 /* */
#define	JAS_ICC_TYPE_UI64		0x75693634 /* */
#define	JAS_ICC_TYPE_VIEWCOND		0x76696577 /* */
#define	JAS_ICC_TYPE_XYZ		0x58595a20 /* XYZ */

typedef uint_fast8_t jas_iccuint8_t;
typedef uint_fast16_t jas_iccuint16_t;
typedef uint_fast32_t jas_iccuint32_t;
typedef int_fast32_t jas_iccsint32_t;
typedef int_fast32_t jas_iccs15fixed16_t;
typedef uint_fast32_t jas_iccu16fixed16_t;
typedef uint_fast64_t jas_iccuint64_t;
typedef uint_fast32_t jas_iccsig_t;

typedef jas_iccsig_t jas_icctagsig_t;
typedef jas_iccsig_t jas_icctagtype_t;
typedef jas_iccsig_t jas_iccattrname_t;

/* Date/time type. */
typedef struct {
	jas_iccuint16_t year;
	jas_iccuint16_t month;
	jas_iccuint16_t day;
	jas_iccuint16_t hour;
	jas_iccuint16_t min;
	jas_iccuint16_t sec;
} jas_icctime_t;

/* XYZ type. */
typedef struct {
	jas_iccs15fixed16_t x;
	jas_iccs15fixed16_t y;
	jas_iccs15fixed16_t z;
} jas_iccxyz_t;

/* Curve type. */
typedef struct {
	jas_iccuint32_t numents;
	jas_iccuint16_t *ents;
} jas_icccurv_t;

/* Text description type. */
typedef struct {
	jas_iccuint32_t asclen;
	char *ascdata; /* ASCII invariant description */
	jas_iccuint32_t uclangcode; /* Unicode language code */
	jas_iccuint32_t uclen; /* Unicode localizable description count */
	uchar *ucdata; /* Unicode localizable description */
	jas_iccuint16_t sccode; /* ScriptCode code */
	jas_iccuint8_t maclen; /* Localizable Macintosh description count */
	uchar macdata[69]; /* Localizable Macintosh description */
} jas_icctxtdesc_t;

/* Text type. */
typedef struct {
	char *string;	/* ASCII character string */
} jas_icctxt_t;

typedef struct {
	jas_iccuint8_t numinchans;
	jas_iccuint8_t numoutchans;
	jas_iccsint32_t e[3][3];
	jas_iccuint8_t clutlen;
	jas_iccuint8_t *clut;
	jas_iccuint16_t numintabents;
	jas_iccuint8_t **intabs;
	jas_iccuint8_t *intabsbuf;
	jas_iccuint16_t numouttabents;
	jas_iccuint8_t **outtabs;
	jas_iccuint8_t *outtabsbuf;
} jas_icclut8_t;

typedef struct {
	jas_iccuint8_t numinchans;
	jas_iccuint8_t numoutchans;
	jas_iccsint32_t e[3][3];
	jas_iccuint8_t clutlen;
	jas_iccuint16_t *clut;
	jas_iccuint16_t numintabents;
	jas_iccuint16_t **intabs;
	jas_iccuint16_t *intabsbuf;
	jas_iccuint16_t numouttabents;
	jas_iccuint16_t **outtabs;
	jas_iccuint16_t *outtabsbuf;
} jas_icclut16_t;

struct jas_iccattrval_s;

typedef struct {
	void (*destroy)(struct jas_iccattrval_s *);
	int (*copy)(struct jas_iccattrval_s *, struct jas_iccattrval_s *);
	int (*input)(struct jas_iccattrval_s *, jas_stream_t *, int);
	int (*output)(struct jas_iccattrval_s *, jas_stream_t *);
	int (*getsize)(struct jas_iccattrval_s *);
	void (*dump)(struct jas_iccattrval_s *, FILE *);
} jas_iccattrvalops_t;

/* Attribute value type (type and value information). */
typedef struct jas_iccattrval_s {
	int refcnt; /* reference count */
	jas_iccsig_t type; /* type */
	jas_iccattrvalops_t *ops; /* type-dependent operations */
	union {
		jas_iccxyz_t xyz;
		jas_icccurv_t curv;
		jas_icctxtdesc_t txtdesc;
		jas_icctxt_t txt;
		jas_icclut8_t lut8;
		jas_icclut16_t lut16;
	} data; /* value */
} jas_iccattrval_t;

/* Header type. */
typedef struct {
	jas_iccuint32_t size; /* profile size */
	jas_iccsig_t cmmtype; /* CMM type signature */
	jas_iccuint32_t version; /* profile version */
	jas_iccsig_t clas; /* profile/device class signature */
	jas_iccsig_t colorspc; /* color space of data */
	jas_iccsig_t refcolorspc; /* profile connection space */
	jas_icctime_t ctime; /* creation time */
	jas_iccsig_t magic; /* profile file signature */
	jas_iccsig_t platform; /* primary platform */
	jas_iccuint32_t flags; /* profile flags */
	jas_iccsig_t maker; /* device manufacturer signature */
	jas_iccsig_t model; /* device model signature */
	jas_iccuint64_t attr; /* device setup attributes */
	jas_iccsig_t intent; /* rendering intent */
	jas_iccxyz_t illum; /* illuminant */
	jas_iccsig_t creator; /* profile creator signature */
} jas_icchdr_t;

typedef struct {
	jas_iccsig_t name;
	jas_iccattrval_t *val;
} jas_iccattr_t;

typedef struct {
	int numattrs;
	int maxattrs;
	jas_iccattr_t *attrs;
} jas_iccattrtab_t;

typedef struct jas_icctagtabent_s {
	jas_iccuint32_t tag;
	jas_iccuint32_t off;
	jas_iccuint32_t len;
	void *data;
	struct jas_icctagtabent_s *first;
} jas_icctagtabent_t;

typedef struct {
	jas_iccuint32_t numents;
	jas_icctagtabent_t *ents;
} jas_icctagtab_t;

/* ICC profile type. */
typedef struct {
	jas_icchdr_t hdr;
	jas_icctagtab_t tagtab;
	jas_iccattrtab_t *attrtab;
} jas_iccprof_t;

typedef struct {
	jas_iccuint32_t type;
	jas_iccattrvalops_t ops;
} jas_iccattrvalinfo_t;

jas_iccprof_t *jas_iccprof_load(jas_stream_t *in);
int jas_iccprof_save(jas_iccprof_t *prof, jas_stream_t *out);
void jas_iccprof_destroy(jas_iccprof_t *prof);
jas_iccattrval_t *jas_iccprof_getattr(jas_iccprof_t *prof,
  jas_iccattrname_t name);
int jas_iccprof_setattr(jas_iccprof_t *prof, jas_iccattrname_t name,
  jas_iccattrval_t *val);
void jas_iccprof_dump(jas_iccprof_t *prof, FILE *out);
jas_iccprof_t *jas_iccprof_copy(jas_iccprof_t *prof);
int jas_iccprof_gethdr(jas_iccprof_t *prof, jas_icchdr_t *hdr);
int jas_iccprof_sethdr(jas_iccprof_t *prof, jas_icchdr_t *hdr);

void jas_iccattrval_destroy(jas_iccattrval_t *attrval);
void jas_iccattrval_dump(jas_iccattrval_t *attrval, FILE *out);
int jas_iccattrval_allowmodify(jas_iccattrval_t **attrval);
jas_iccattrval_t *jas_iccattrval_clone(jas_iccattrval_t *attrval);
jas_iccattrval_t *jas_iccattrval_create(jas_iccuint32_t type);

void jas_iccattrtab_dump(jas_iccattrtab_t *attrtab, FILE *out);

extern uchar jas_iccprofdata_srgb[];
extern int jas_iccprofdata_srgblen;
extern uchar jas_iccprofdata_sgray[];
extern int jas_iccprofdata_sgraylen;
jas_iccprof_t *jas_iccprof_createfrombuf(uchar *buf, int len);
jas_iccprof_t *jas_iccprof_createfromclrspc(int clrspc);

#ifdef __cplusplus
}
#endif

#endif
