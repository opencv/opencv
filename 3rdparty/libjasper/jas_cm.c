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

/*
 * Color Management
 *
 * $Id: jas_cm.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

#include <jasper/jas_config.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <jasper/jas_cm.h>
#include <jasper/jas_icc.h>
#include <jasper/jas_init.h>
#include <jasper/jas_stream.h>
#include <jasper/jas_malloc.h>
#include <jasper/jas_math.h>

static jas_cmprof_t *jas_cmprof_create(void);
static void jas_cmshapmatlut_cleanup(jas_cmshapmatlut_t *);
static jas_cmreal_t jas_cmshapmatlut_lookup(jas_cmshapmatlut_t *lut, jas_cmreal_t x);

static void jas_cmpxform_destroy(jas_cmpxform_t *pxform);
static jas_cmpxform_t *jas_cmpxform_copy(jas_cmpxform_t *pxform);

static void jas_cmshapmat_destroy(jas_cmpxform_t *pxform);
static int jas_cmshapmat_apply(jas_cmpxform_t *pxform, jas_cmreal_t *in,
  jas_cmreal_t *out, int cnt);

static int jas_cmputint(long **bufptr, int sgnd, int prec, long val);
static int jas_cmgetint(long **bufptr, int sgnd, int prec, long *val);
static int jas_cmpxformseq_append(jas_cmpxformseq_t *pxformseq,
  jas_cmpxformseq_t *othpxformseq);
static int jas_cmpxformseq_appendcnvt(jas_cmpxformseq_t *pxformseq,
  int, int);
static int jas_cmpxformseq_resize(jas_cmpxformseq_t *pxformseq, int n);

static int mono(jas_iccprof_t *prof, int op, jas_cmpxformseq_t **pxformseq);
static int triclr(jas_iccprof_t *prof, int op, jas_cmpxformseq_t **retpxformseq);

static void jas_cmpxformseq_destroy(jas_cmpxformseq_t *pxformseq);
static int jas_cmpxformseq_delete(jas_cmpxformseq_t *pxformseq, int i);
static jas_cmpxformseq_t *jas_cmpxformseq_create(void);
static jas_cmpxformseq_t *jas_cmpxformseq_copy(jas_cmpxformseq_t *pxformseq);
static int jas_cmshapmat_invmat(jas_cmreal_t out[3][4], jas_cmreal_t in[3][4]);
static int jas_cmpxformseq_insertpxform(jas_cmpxformseq_t *pxformseq,
  int i, jas_cmpxform_t *pxform);

#define	SEQFWD(intent)	(intent)
#define	SEQREV(intent)	(4 + (intent))
#define	SEQSIM(intent)	(8 + (intent))
#define	SEQGAM		12

#define fwdpxformseq(prof, intent) \
  (((prof)->pxformseqs[SEQFWD(intent)]) ? \
  ((prof)->pxformseqs[SEQFWD(intent)]) : \
  ((prof)->pxformseqs[SEQFWD(0)]))

#define revpxformseq(prof, intent) \
  (((prof)->pxformseqs[SEQREV(intent)]) ? \
  ((prof)->pxformseqs[SEQREV(intent)]) : \
  ((prof)->pxformseqs[SEQREV(0)]))

#define simpxformseq(prof, intent) \
  (((prof)->pxformseqs[SEQSIM(intent)]) ? \
  ((prof)->pxformseqs[SEQSIM(intent)]) : \
  ((prof)->pxformseqs[SEQSIM(0)]))

#define gampxformseq(prof)	((prof)->pxformseqs[SEQGAM])

static int icctoclrspc(int iccclrspc, int refflag);
static jas_cmpxform_t *jas_cmpxform_create0(void);
static jas_cmpxform_t *jas_cmpxform_createshapmat(void);
static void jas_cmshapmatlut_init(jas_cmshapmatlut_t *lut);
static int jas_cmshapmatlut_set(jas_cmshapmatlut_t *lut, jas_icccurv_t *curv);

static jas_cmpxformops_t shapmat_ops = {jas_cmshapmat_destroy, jas_cmshapmat_apply, 0};
static jas_cmprof_t *jas_cmprof_createsycc(void);

/******************************************************************************\
* Color profile class.
\******************************************************************************/

jas_cmprof_t *jas_cmprof_createfromclrspc(int clrspc)
{
	jas_iccprof_t *iccprof;
	jas_cmprof_t *prof;

	iccprof = 0;
	prof = 0;
	switch (clrspc) {
	case JAS_CLRSPC_SYCBCR:
		if (!(prof = jas_cmprof_createsycc()))
			goto error;
		break;
	default:
		if (!(iccprof = jas_iccprof_createfromclrspc(clrspc)))
			goto error;
		if (!(prof = jas_cmprof_createfromiccprof(iccprof)))
			goto error;
		jas_iccprof_destroy(iccprof);
		iccprof = 0;
		if (!jas_clrspc_isgeneric(clrspc))
			prof->clrspc = clrspc;
		break;
	}
	return prof;
error:
	if (iccprof)
		jas_iccprof_destroy(iccprof);
	return 0;
}

static jas_cmprof_t *jas_cmprof_createsycc()
{
	jas_cmprof_t *prof;
	jas_cmpxform_t *fwdpxform;
	jas_cmpxform_t *revpxform;
	jas_cmshapmat_t *fwdshapmat;
	jas_cmshapmat_t *revshapmat;
	int i;
	int j;

	if (!(prof = jas_cmprof_createfromclrspc(JAS_CLRSPC_SRGB)))
		goto error;
	prof->clrspc = JAS_CLRSPC_SYCBCR;
	assert(prof->numchans == 3 && prof->numrefchans == 3);
	assert(prof->refclrspc == JAS_CLRSPC_CIEXYZ);
	if (!(fwdpxform = jas_cmpxform_createshapmat()))
		goto error;
	fwdpxform->numinchans = 3;
	fwdpxform->numoutchans = 3;
	fwdshapmat = &fwdpxform->data.shapmat;
	fwdshapmat->mono = 0;
	fwdshapmat->order = 0;
	fwdshapmat->useluts = 0;
	fwdshapmat->usemat = 1;
	fwdshapmat->mat[0][0] = 1.0;
	fwdshapmat->mat[0][1] = 0.0;
	fwdshapmat->mat[0][2] = 1.402;
	fwdshapmat->mat[1][0] = 1.0;
	fwdshapmat->mat[1][1] = -0.34413;
	fwdshapmat->mat[1][2] = -0.71414;
	fwdshapmat->mat[2][0] = 1.0;
	fwdshapmat->mat[2][1] = 1.772;
	fwdshapmat->mat[2][2] = 0.0;
	fwdshapmat->mat[0][3] = -0.5 * (1.402);
	fwdshapmat->mat[1][3] = -0.5 * (-0.34413 - 0.71414);
	fwdshapmat->mat[2][3] = -0.5 * (1.772);
	if (!(revpxform = jas_cmpxform_createshapmat()))
		goto error;
	revpxform->numinchans = 3;
	revpxform->numoutchans = 3;
	revshapmat = &revpxform->data.shapmat;
	revshapmat->mono = 0;
	revshapmat->order = 1;
	revshapmat->useluts = 0;
	revshapmat->usemat = 1;
	jas_cmshapmat_invmat(revshapmat->mat, fwdshapmat->mat);

	for (i = 0; i < JAS_CMXFORM_NUMINTENTS; ++i) {
		j = SEQFWD(i);
		if (prof->pxformseqs[j]) {
			if (jas_cmpxformseq_insertpxform(prof->pxformseqs[j], 0,
			  fwdpxform))
				goto error;
		}
		j = SEQREV(i);
		if (prof->pxformseqs[j]) {
			if (jas_cmpxformseq_insertpxform(prof->pxformseqs[j],
			  -1, revpxform))
				goto error;
		}
	}

	jas_cmpxform_destroy(fwdpxform);
	jas_cmpxform_destroy(revpxform);
	return prof;
error:
	return 0;
}

jas_cmprof_t *jas_cmprof_createfromiccprof(jas_iccprof_t *iccprof)
{
	jas_cmprof_t *prof;
	jas_icchdr_t icchdr;
	jas_cmpxformseq_t *fwdpxformseq;
	jas_cmpxformseq_t *revpxformseq;

	prof = 0;
	fwdpxformseq = 0;
	revpxformseq = 0;

	if (!(prof = jas_cmprof_create()))
		goto error;
	jas_iccprof_gethdr(iccprof, &icchdr);
	if (!(prof->iccprof = jas_iccprof_copy(iccprof)))
		goto error;
	prof->clrspc = icctoclrspc(icchdr.colorspc, 0);
	prof->refclrspc = icctoclrspc(icchdr.refcolorspc, 1);
	prof->numchans = jas_clrspc_numchans(prof->clrspc);
	prof->numrefchans = jas_clrspc_numchans(prof->refclrspc);

	if (prof->numchans == 1) {
		if (mono(prof->iccprof, 0, &fwdpxformseq))
			goto error;
		if (mono(prof->iccprof, 1, &revpxformseq))
			goto error;
	} else if (prof->numchans == 3) {
		if (triclr(prof->iccprof, 0, &fwdpxformseq))
			goto error;
		if (triclr(prof->iccprof, 1, &revpxformseq))
			goto error;
	}
	prof->pxformseqs[SEQFWD(0)] = fwdpxformseq;
	prof->pxformseqs[SEQREV(0)] = revpxformseq;

#if 0
	if (prof->numchans > 1) {
		lut(prof->iccprof, 0, PER, &pxformseq);
		pxformseqs_set(prof, SEQFWD(PER), pxformseq);
		lut(prof->iccprof, 1, PER, &pxformseq);
		pxformseqs_set(prof, SEQREV(PER), pxformseq);
		lut(prof->iccprof, 0, CLR, &pxformseq);
		pxformseqs_set(prof, SEQREV(CLR), pxformseq);
		lut(prof->iccprof, 1, CLR, &pxformseq);
		pxformseqs_set(prof, SEQREV(CLR), pxformseq);
		lut(prof->iccprof, 0, SAT, &pxformseq);
		pxformseqs_set(prof, SEQREV(SAT), pxformseq);
		lut(prof->iccprof, 1, SAT, &pxformseq);
		pxformseqs_set(prof, SEQREV(SAT), pxformseq);
	}
#endif

	return prof;

error:
	if (fwdpxformseq) {
		jas_cmpxformseq_destroy(fwdpxformseq);
	}
	if (revpxformseq) {
		jas_cmpxformseq_destroy(revpxformseq);
	}
	if (prof) {
		jas_cmprof_destroy(prof);
	}

	return 0;
}

static jas_cmprof_t *jas_cmprof_create()
{
	int i;
	jas_cmprof_t *prof;
	if (!(prof = jas_malloc(sizeof(jas_cmprof_t))))
		return 0;
	memset(prof, 0, sizeof(jas_cmprof_t));
	prof->iccprof = 0;
	for (i = 0; i < JAS_CMPROF_NUMPXFORMSEQS; ++i)
		prof->pxformseqs[i] = 0;
	return prof;
}

void jas_cmprof_destroy(jas_cmprof_t *prof)
{ 
	int i;
	for (i = 0; i < JAS_CMPROF_NUMPXFORMSEQS; ++i) {
		if (prof->pxformseqs[i]) {
			jas_cmpxformseq_destroy(prof->pxformseqs[i]);
			prof->pxformseqs[i] = 0;
		}
	}
	if (prof->iccprof)
		jas_iccprof_destroy(prof->iccprof);
	jas_free(prof);
}

jas_cmprof_t *jas_cmprof_copy(jas_cmprof_t *prof)
{
	jas_cmprof_t *newprof;
	int i;

	if (!(newprof = jas_cmprof_create()))
		goto error;
	newprof->clrspc = prof->clrspc;
	newprof->numchans = prof->numchans;
	newprof->refclrspc = prof->refclrspc;
	newprof->numrefchans = prof->numrefchans;
	newprof->iccprof = jas_iccprof_copy(prof->iccprof);
	for (i = 0; i < JAS_CMPROF_NUMPXFORMSEQS; ++i) {
		if (prof->pxformseqs[i]) {
			if (!(newprof->pxformseqs[i] = jas_cmpxformseq_copy(prof->pxformseqs[i])))
				goto error;
		}
	}
	return newprof;
error:
	return 0;
}

/******************************************************************************\
* Transform class.
\******************************************************************************/

jas_cmxform_t *jas_cmxform_create(jas_cmprof_t *inprof, jas_cmprof_t *outprof,
  jas_cmprof_t *prfprof, int op, int intent, int optimize)
{
	jas_cmxform_t *xform;
	jas_cmpxformseq_t *inpxformseq;
	jas_cmpxformseq_t *outpxformseq;
	jas_cmpxformseq_t *altoutpxformseq;
	jas_cmpxformseq_t *prfpxformseq;
	int prfintent;

	/* Avoid compiler warnings about unused parameters. */
	optimize = 0;

	prfintent = intent;

	if (!(xform = jas_malloc(sizeof(jas_cmxform_t))))
		goto error;
	if (!(xform->pxformseq = jas_cmpxformseq_create()))
		goto error;

	switch (op) {
	case JAS_CMXFORM_OP_FWD:
		inpxformseq = fwdpxformseq(inprof, intent);
		outpxformseq = revpxformseq(outprof, intent);
		if (!inpxformseq || !outpxformseq)
			goto error;
		if (jas_cmpxformseq_append(xform->pxformseq, inpxformseq) ||
		  jas_cmpxformseq_appendcnvt(xform->pxformseq,
		  inprof->refclrspc, outprof->refclrspc) ||
		  jas_cmpxformseq_append(xform->pxformseq, outpxformseq))
			goto error;
		xform->numinchans = jas_clrspc_numchans(inprof->clrspc);
		xform->numoutchans = jas_clrspc_numchans(outprof->clrspc);
		break;
	case JAS_CMXFORM_OP_REV:
		outpxformseq = fwdpxformseq(outprof, intent);
		inpxformseq = revpxformseq(inprof, intent);
		if (!outpxformseq || !inpxformseq)
			goto error;
		if (jas_cmpxformseq_append(xform->pxformseq, outpxformseq) ||
		  jas_cmpxformseq_appendcnvt(xform->pxformseq,
		  outprof->refclrspc, inprof->refclrspc) ||
		  jas_cmpxformseq_append(xform->pxformseq, inpxformseq))
			goto error;
		xform->numinchans = jas_clrspc_numchans(outprof->clrspc);
		xform->numoutchans = jas_clrspc_numchans(inprof->clrspc);
		break;
	case JAS_CMXFORM_OP_PROOF:
		assert(prfprof);
		inpxformseq = fwdpxformseq(inprof, intent);
		prfpxformseq = fwdpxformseq(prfprof, prfintent);
		if (!inpxformseq || !prfpxformseq)
			goto error;
		outpxformseq = simpxformseq(outprof, intent);
		altoutpxformseq = 0;
		if (!outpxformseq) {
			outpxformseq = revpxformseq(outprof, intent);
			altoutpxformseq = fwdpxformseq(outprof, intent);
			if (!outpxformseq || !altoutpxformseq)
				goto error;
		}
		if (jas_cmpxformseq_append(xform->pxformseq, inpxformseq) ||
		  jas_cmpxformseq_appendcnvt(xform->pxformseq,
		  inprof->refclrspc, outprof->refclrspc))
			goto error;
		if (altoutpxformseq) {
			if (jas_cmpxformseq_append(xform->pxformseq, outpxformseq) ||
			  jas_cmpxformseq_append(xform->pxformseq, altoutpxformseq))
				goto error;
		} else {
			if (jas_cmpxformseq_append(xform->pxformseq, outpxformseq))
				goto error;
		}
		if (jas_cmpxformseq_appendcnvt(xform->pxformseq,
		  outprof->refclrspc, inprof->refclrspc) ||
		  jas_cmpxformseq_append(xform->pxformseq, prfpxformseq))
			goto error;
		xform->numinchans = jas_clrspc_numchans(inprof->clrspc);
		xform->numoutchans = jas_clrspc_numchans(prfprof->clrspc);
		break;
	case JAS_CMXFORM_OP_GAMUT:
		inpxformseq = fwdpxformseq(inprof, intent);
		outpxformseq = gampxformseq(outprof);
		if (!inpxformseq || !outpxformseq)
			goto error;
		if (jas_cmpxformseq_append(xform->pxformseq, inpxformseq) ||
		  jas_cmpxformseq_appendcnvt(xform->pxformseq,
		  inprof->refclrspc, outprof->refclrspc) ||
		  jas_cmpxformseq_append(xform->pxformseq, outpxformseq))
			goto error;
		xform->numinchans = jas_clrspc_numchans(inprof->clrspc);
		xform->numoutchans = 1;
		break;
	}
	return xform;
error:
	return 0;
}

#define	APPLYBUFSIZ	2048
int jas_cmxform_apply(jas_cmxform_t *xform, jas_cmpixmap_t *in, jas_cmpixmap_t *out)
{
	jas_cmcmptfmt_t *fmt;
	jas_cmreal_t buf[2][APPLYBUFSIZ];
	jas_cmpxformseq_t *pxformseq;
	int i;
	int j;
	int width;
	int height;
	int total;
	int n;
	jas_cmreal_t *inbuf;
	jas_cmreal_t *outbuf;
	jas_cmpxform_t *pxform;
	long *dataptr;
	int maxchans;
	int bufmax;
	int m;
	int bias;
	jas_cmreal_t scale;
	long v;
	jas_cmreal_t *bufptr;

	if (xform->numinchans > in->numcmpts || xform->numoutchans > out->numcmpts)
		goto error;

	fmt = &in->cmptfmts[0];
	width = fmt->width;
	height = fmt->height;
	for (i = 1; i < xform->numinchans; ++i) {
		fmt = &in->cmptfmts[i];
		if (fmt->width != width || fmt->height != height) {
			goto error;
		}
	}
	for (i = 0; i < xform->numoutchans; ++i) {
		fmt = &out->cmptfmts[i];
		if (fmt->width != width || fmt->height != height) {
			goto error;
		}
	}

	maxchans = 0;
	pxformseq = xform->pxformseq;
	for (i = 0; i < pxformseq->numpxforms; ++i) {
		pxform = pxformseq->pxforms[i];
		if (pxform->numinchans > maxchans) {
			maxchans = pxform->numinchans;
		}
		if (pxform->numoutchans > maxchans) {
			maxchans = pxform->numoutchans;
		}
	}
	bufmax = APPLYBUFSIZ / maxchans;
	assert(bufmax > 0);

	total = width * height;
	n = 0;
	while (n < total) {

		inbuf = &buf[0][0];
		m = JAS_MIN(total - n, bufmax);

		for (i = 0; i < xform->numinchans; ++i) {
			fmt = &in->cmptfmts[i];
			scale = (double)((1 << fmt->prec) - 1);
			bias = fmt->sgnd ? (1 << (fmt->prec - 1)) : 0;
			dataptr = &fmt->buf[n];
			bufptr = &inbuf[i];
			for (j = 0; j < m; ++j) {
				if (jas_cmgetint(&dataptr, fmt->sgnd, fmt->prec, &v))
					goto error;
				*bufptr = (v - bias) / scale;
				bufptr += xform->numinchans;
			}
		}

		inbuf = &buf[0][0];
		outbuf = inbuf;
		for (i = 0; i < pxformseq->numpxforms; ++i) {
			pxform = pxformseq->pxforms[i];
			if (pxform->numoutchans > pxform->numinchans) {
				outbuf = (inbuf == &buf[0][0]) ? &buf[1][0] : &buf[0][0];
			} else {
				outbuf = inbuf;
			}
			if ((*pxform->ops->apply)(pxform, inbuf, outbuf, m))
				goto error;
			inbuf = outbuf;
		}

		for (i = 0; i < xform->numoutchans; ++i) {
			fmt = &out->cmptfmts[i];
			scale = (double)((1 << fmt->prec) - 1);
			bias = fmt->sgnd ? (1 << (fmt->prec - 1)) : 0;
			bufptr = &outbuf[i];
			dataptr = &fmt->buf[n];
			for (j = 0; j < m; ++j) {
				v = (*bufptr) * scale + bias;
				bufptr += xform->numoutchans;
				if (jas_cmputint(&dataptr, fmt->sgnd, fmt->prec, v))
					goto error;
			}
		}
	
		n += m;
	}
	
	return 0;
error:
	return -1;
}

void jas_cmxform_destroy(jas_cmxform_t *xform)
{
	if (xform->pxformseq)
		jas_cmpxformseq_destroy(xform->pxformseq);
	jas_free(xform);
}

/******************************************************************************\
* Primitive transform sequence class.
\******************************************************************************/

static jas_cmpxformseq_t *jas_cmpxformseq_create()
{
	jas_cmpxformseq_t *pxformseq;
	pxformseq = 0;
	if (!(pxformseq = jas_malloc(sizeof(jas_cmpxformseq_t))))
		goto error;
	pxformseq->pxforms = 0;
	pxformseq->numpxforms = 0;
	pxformseq->maxpxforms = 0;
	if (jas_cmpxformseq_resize(pxformseq, 16))
		goto error;
	return pxformseq;
error:
	if (pxformseq)
		jas_cmpxformseq_destroy(pxformseq);
	return 0;
}

static jas_cmpxformseq_t *jas_cmpxformseq_copy(jas_cmpxformseq_t *pxformseq)
{
	jas_cmpxformseq_t *newpxformseq;

	if (!(newpxformseq = jas_cmpxformseq_create()))
		goto error;
	if (jas_cmpxformseq_append(newpxformseq, pxformseq))
		goto error;
	return newpxformseq;
error:
	return 0;
}

static void jas_cmpxformseq_destroy(jas_cmpxformseq_t *pxformseq)
{
	while (pxformseq->numpxforms > 0)
		jas_cmpxformseq_delete(pxformseq, pxformseq->numpxforms - 1);
	if (pxformseq->pxforms)
		jas_free(pxformseq->pxforms);
	jas_free(pxformseq);
}

static int jas_cmpxformseq_delete(jas_cmpxformseq_t *pxformseq, int i)
{
	assert(i >= 0 && i < pxformseq->numpxforms);
	if (i != pxformseq->numpxforms - 1)
		abort();
	jas_cmpxform_destroy(pxformseq->pxforms[i]);
	pxformseq->pxforms[i] = 0;
	--pxformseq->numpxforms;
	return 0;
}

static int jas_cmpxformseq_appendcnvt(jas_cmpxformseq_t *pxformseq,
  int dstclrspc, int srcclrspc)
{
	if (dstclrspc == srcclrspc)
		return 0;
	abort();
	/* Avoid compiler warnings about unused parameters. */
	pxformseq = 0;
	return -1;
}

static int jas_cmpxformseq_insertpxform(jas_cmpxformseq_t *pxformseq,
  int i, jas_cmpxform_t *pxform)
{
	jas_cmpxform_t *tmppxform;
	int n;
	if (i < 0)
		i = pxformseq->numpxforms;
	assert(i >= 0 && i <= pxformseq->numpxforms);
	if (pxformseq->numpxforms >= pxformseq->maxpxforms) {
		if (jas_cmpxformseq_resize(pxformseq, pxformseq->numpxforms +
		  16))
			goto error;
	}
	assert(pxformseq->numpxforms < pxformseq->maxpxforms);
	if (!(tmppxform = jas_cmpxform_copy(pxform)))
		goto error;
	n = pxformseq->numpxforms - i;
	if (n > 0) {
		memmove(&pxformseq->pxforms[i + 1], &pxformseq->pxforms[i],
		  n * sizeof(jas_cmpxform_t *));
	}
	pxformseq->pxforms[i] = tmppxform;
	++pxformseq->numpxforms;
	return 0;
error:
	return -1;
}

static int jas_cmpxformseq_append(jas_cmpxformseq_t *pxformseq,
  jas_cmpxformseq_t *othpxformseq)
{
	int n;
	int i;
	jas_cmpxform_t *pxform;
	jas_cmpxform_t *othpxform;
	n = pxformseq->numpxforms + othpxformseq->numpxforms;
	if (n > pxformseq->maxpxforms) {
		if (jas_cmpxformseq_resize(pxformseq, n))
			goto error;
	}
	for (i = 0; i < othpxformseq->numpxforms; ++i) {
		othpxform = othpxformseq->pxforms[i];
		if (!(pxform = jas_cmpxform_copy(othpxform)))
			goto error;
		pxformseq->pxforms[pxformseq->numpxforms] = pxform;
		++pxformseq->numpxforms;
	}
	return 0;
error:
	return -1;
}

static int jas_cmpxformseq_resize(jas_cmpxformseq_t *pxformseq, int n)
{
	jas_cmpxform_t **p;
	assert(n >= pxformseq->numpxforms);
	p = (!pxformseq->pxforms) ? jas_malloc(n * sizeof(jas_cmpxform_t *)) :
	  jas_realloc(pxformseq->pxforms, n * sizeof(jas_cmpxform_t *));
	if (!p) {
		return -1;
	}
	pxformseq->pxforms = p;
	pxformseq->maxpxforms = n;
	return 0;
}

/******************************************************************************\
* Primitive transform class.
\******************************************************************************/

static jas_cmpxform_t *jas_cmpxform_create0()
{
	jas_cmpxform_t *pxform;
	if (!(pxform = jas_malloc(sizeof(jas_cmpxform_t))))
		return 0;
	memset(pxform, 0, sizeof(jas_cmpxform_t));
	pxform->refcnt = 0;
	pxform->ops = 0;
	return pxform;
}

static void jas_cmpxform_destroy(jas_cmpxform_t *pxform)
{
	if (--pxform->refcnt <= 0) {
		(*pxform->ops->destroy)(pxform);
		jas_free(pxform);
	}
}

static jas_cmpxform_t *jas_cmpxform_copy(jas_cmpxform_t *pxform)
{
	++pxform->refcnt;
	return pxform;
}

/******************************************************************************\
* Shaper matrix class.
\******************************************************************************/

static jas_cmpxform_t *jas_cmpxform_createshapmat()
{
	int i;
	int j;
	jas_cmpxform_t *pxform;
	jas_cmshapmat_t *shapmat;
	if (!(pxform = jas_cmpxform_create0()))
		return 0;
	pxform->ops = &shapmat_ops;
	shapmat = &pxform->data.shapmat;
	shapmat->mono = 0;
	shapmat->order = 0;
	shapmat->useluts = 0;
	shapmat->usemat = 0;
	for (i = 0; i < 3; ++i)
		jas_cmshapmatlut_init(&shapmat->luts[i]);
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 4; ++j)
			shapmat->mat[i][j] = 0.0;
	}
	++pxform->refcnt;
	return pxform;
}

static void jas_cmshapmat_destroy(jas_cmpxform_t *pxform)
{
	jas_cmshapmat_t *shapmat = &pxform->data.shapmat;
	int i;
	for (i = 0; i < 3; ++i)
		jas_cmshapmatlut_cleanup(&shapmat->luts[i]);
}

static int jas_cmshapmat_apply(jas_cmpxform_t *pxform, jas_cmreal_t *in,
  jas_cmreal_t *out, int cnt)
{
	jas_cmshapmat_t *shapmat = &pxform->data.shapmat;
	jas_cmreal_t *src;
	jas_cmreal_t *dst;
	jas_cmreal_t a0;
	jas_cmreal_t a1;
	jas_cmreal_t a2;
	jas_cmreal_t b0;
	jas_cmreal_t b1;
	jas_cmreal_t b2;
	src = in;
	dst = out;
	if (!shapmat->mono) {
		while (--cnt >= 0) {
			a0 = *src++;
			a1 = *src++;
			a2 = *src++;
			if (!shapmat->order && shapmat->useluts) {
				a0 = jas_cmshapmatlut_lookup(&shapmat->luts[0], a0);
				a1 = jas_cmshapmatlut_lookup(&shapmat->luts[1], a1);
				a2 = jas_cmshapmatlut_lookup(&shapmat->luts[2], a2);
			}
			if (shapmat->usemat) {
				b0 = shapmat->mat[0][0] * a0
				  + shapmat->mat[0][1] * a1
				  + shapmat->mat[0][2] * a2
				  + shapmat->mat[0][3];
				b1 = shapmat->mat[1][0] * a0
				  + shapmat->mat[1][1] * a1
				  + shapmat->mat[1][2] * a2
				  + shapmat->mat[1][3];
				b2 = shapmat->mat[2][0] * a0
				  + shapmat->mat[2][1] * a1
				  + shapmat->mat[2][2] * a2
				  + shapmat->mat[2][3];
				a0 = b0;
				a1 = b1;
				a2 = b2;
			}
			if (shapmat->order && shapmat->useluts) {
				a0 = jas_cmshapmatlut_lookup(&shapmat->luts[0], a0);
				a1 = jas_cmshapmatlut_lookup(&shapmat->luts[1], a1);
				a2 = jas_cmshapmatlut_lookup(&shapmat->luts[2], a2);
			}
			*dst++ = a0;
			*dst++ = a1;
			*dst++ = a2;
		}
	} else {
		if (!shapmat->order) {
			while (--cnt >= 0) {
				a0 = *src++;
				if (shapmat->useluts)
					a0 = jas_cmshapmatlut_lookup(&shapmat->luts[0], a0);
				a2 = a0 * shapmat->mat[2][0];
				a1 = a0 * shapmat->mat[1][0];
				a0 = a0 * shapmat->mat[0][0];
				*dst++ = a0;
				*dst++ = a1;
				*dst++ = a2;
			}
		} else {
assert(0);
			while (--cnt >= 0) {
				a0 = *src++;
				src++;
				src++;
				a0 = a0 * shapmat->mat[0][0];
				if (shapmat->useluts)
					a0 = jas_cmshapmatlut_lookup(&shapmat->luts[0], a0);
				*dst++ = a0;
			}
		}
	}

	return 0;
}

static void jas_cmshapmatlut_init(jas_cmshapmatlut_t *lut)
{
	lut->data = 0;
	lut->size = 0;
}

static void jas_cmshapmatlut_cleanup(jas_cmshapmatlut_t *lut)
{
	if (lut->data) {
		jas_free(lut->data);
		lut->data = 0;
	}
	lut->size = 0;
}

static double gammafn(double x, double gamma)
{
	if (x == 0.0)
		return 0.0;
	return pow(x, gamma);
}

static int jas_cmshapmatlut_set(jas_cmshapmatlut_t *lut, jas_icccurv_t *curv)
{
	jas_cmreal_t gamma;
	int i;
	gamma = 0;
	jas_cmshapmatlut_cleanup(lut);
	if (curv->numents == 0) {
		lut->size = 2;
		if (!(lut->data = jas_malloc(lut->size * sizeof(jas_cmreal_t))))
			goto error;
		lut->data[0] = 0.0;
		lut->data[1] = 1.0;
	} else if (curv->numents == 1) {
		lut->size = 256;
		if (!(lut->data = jas_malloc(lut->size * sizeof(jas_cmreal_t))))
			goto error;
		gamma = curv->ents[0] / 256.0;
		for (i = 0; i < lut->size; ++i) {
			lut->data[i] = gammafn(i / (double) (lut->size - 1), gamma);
		}
	} else {
		lut->size = curv->numents;
		if (!(lut->data = jas_malloc(lut->size * sizeof(jas_cmreal_t))))
			goto error;
		for (i = 0; i < lut->size; ++i) {
			lut->data[i] = curv->ents[i] / 65535.0;
		}
	}
	return 0;
error:
	return -1;
}

static jas_cmreal_t jas_cmshapmatlut_lookup(jas_cmshapmatlut_t *lut, jas_cmreal_t x)
{
	jas_cmreal_t t;
	int lo;
	int hi;
	t = x * (lut->size - 1);
	lo = floor(t);
	if (lo < 0)
		return lut->data[0];
	hi = ceil(t);
	if (hi >= lut->size)
		return lut->data[lut->size - 1];
	return lut->data[lo] + (t - lo) * (lut->data[hi] - lut->data[lo]);
}

static int jas_cmshapmatlut_invert(jas_cmshapmatlut_t *invlut,
  jas_cmshapmatlut_t *lut, int n)
{
	int i;
	int j;
	int k;
	jas_cmreal_t ax;
	jas_cmreal_t ay;
	jas_cmreal_t bx;
	jas_cmreal_t by;
	jas_cmreal_t sx;
	jas_cmreal_t sy;
	assert(n >= 2);
	if (invlut->data) {
		jas_free(invlut->data);
		invlut->data = 0;
	}
	/* The sample values should be nondecreasing. */
	for (i = 1; i < lut->size; ++i) {
		if (lut->data[i - 1] > lut->data[i]) {
			assert(0);
			return -1;
		}
	}
	if (!(invlut->data = jas_malloc(n * sizeof(jas_cmreal_t))))
		return -1;
	invlut->size = n;
	for (i = 0; i < invlut->size; ++i) {
		sy = ((double) i) / (invlut->size - 1);
		sx = 1.0;
		for (j = 0; j < lut->size; ++j) {
			ay = lut->data[j];
			if (sy == ay) {
				for (k = j + 1; k < lut->size; ++k) {
					by = lut->data[k];
					if (by != sy)
						break;
#if 0
assert(0);
#endif
				}
				if (k < lut->size) {
					--k;
					ax = ((double) j) / (lut->size - 1);
					bx = ((double) k) / (lut->size - 1);
					sx = (ax + bx) / 2.0;
				}
				break;
			}
			if (j < lut->size - 1) {
				by = lut->data[j + 1];
				if (sy > ay && sy < by) {
					ax = ((double) j) / (lut->size - 1);
					bx = ((double) j + 1) / (lut->size - 1);
					sx = ax +
					  (sy - ay) / (by - ay) * (bx - ax);
					break;
				}
			}
		}
		invlut->data[i] = sx;
	}
#if 0
for (i=0;i<lut->size;++i)
	jas_eprintf("lut[%d]=%f ", i, lut->data[i]);
for (i=0;i<invlut->size;++i)
	jas_eprintf("invlut[%d]=%f ", i, invlut->data[i]);
#endif
	return 0;
}

static int jas_cmshapmat_invmat(jas_cmreal_t out[3][4], jas_cmreal_t in[3][4])
{
	jas_cmreal_t d;
	d = in[0][0] * (in[1][1] * in[2][2] - in[1][2] * in[2][1])
	  - in[0][1] * (in[1][0] * in[2][2] - in[1][2] * in[2][0])
	  + in[0][2] * (in[1][0] * in[2][1] - in[1][1] * in[2][0]);
#if 0
jas_eprintf("delta=%f\n", d);
#endif
	if (JAS_ABS(d) < 1e-6)
		return -1;
	out[0][0] = (in[1][1] * in[2][2] - in[1][2] * in[2][1]) / d;
	out[1][0] = -(in[1][0] * in[2][2] - in[1][2] * in[2][0]) / d;
	out[2][0] = (in[1][0] * in[2][1] - in[1][1] * in[2][0]) / d;
	out[0][1] = -(in[0][1] * in[2][2] - in[0][2] * in[2][1]) / d;
	out[1][1] = (in[0][0] * in[2][2] - in[0][2] * in[2][0]) / d;
	out[2][1] = -(in[0][0] * in[2][1] - in[0][1] * in[2][0]) / d;
	out[0][2] = (in[0][1] * in[1][2] - in[0][2] * in[1][1]) / d;
	out[1][2] = -(in[0][0] * in[1][2] - in[1][0] * in[0][2]) / d;
	out[2][2] = (in[0][0] * in[1][1] - in[0][1] * in[1][0]) / d;
	out[0][3] = -in[0][3];
	out[1][3] = -in[1][3];
	out[2][3] = -in[2][3];
#if 0
jas_eprintf("[ %f %f %f %f ]\n[ %f %f %f %f ]\n[ %f %f %f %f ]\n",
in[0][0], in[0][1], in[0][2], in[0][3],
in[1][0], in[1][1], in[1][2], in[1][3],
in[2][0], in[2][1], in[2][2], in[2][3]);
jas_eprintf("[ %f %f %f %f ]\n[ %f %f %f %f ]\n[ %f %f %f %f ]\n",
out[0][0], out[0][1], out[0][2], out[0][3],
out[1][0], out[1][1], out[1][2], out[1][3],
out[2][0], out[2][1], out[2][2], out[2][3]);
#endif
	return 0;
}

/******************************************************************************\
*
\******************************************************************************/

static int icctoclrspc(int iccclrspc, int refflag)
{
	if (refflag) {
		switch (iccclrspc) {
		case JAS_ICC_COLORSPC_XYZ:
			return JAS_CLRSPC_CIEXYZ;
		case JAS_ICC_COLORSPC_LAB:
			return JAS_CLRSPC_CIELAB;
		default:
			abort();
			break;
		}
	} else {
		switch (iccclrspc) {
		case JAS_ICC_COLORSPC_YCBCR:
			return JAS_CLRSPC_GENYCBCR;
		case JAS_ICC_COLORSPC_RGB:
			return JAS_CLRSPC_GENRGB;
		case JAS_ICC_COLORSPC_GRAY:
			return JAS_CLRSPC_GENGRAY;
		default:
			abort();
			break;
		}
	}
}

static int mono(jas_iccprof_t *iccprof, int op, jas_cmpxformseq_t **retpxformseq)
{
	jas_iccattrval_t *graytrc;
	jas_cmshapmat_t *shapmat;
	jas_cmpxform_t *pxform;
	jas_cmpxformseq_t *pxformseq;
	jas_cmshapmatlut_t lut;

	jas_cmshapmatlut_init(&lut);
	if (!(graytrc = jas_iccprof_getattr(iccprof, JAS_ICC_TAG_GRYTRC)) ||
	  graytrc->type != JAS_ICC_TYPE_CURV)
		goto error;
	if (!(pxform = jas_cmpxform_createshapmat()))
		goto error;
	shapmat = &pxform->data.shapmat;
	if (!(pxformseq = jas_cmpxformseq_create()))
		goto error;
	if (jas_cmpxformseq_insertpxform(pxformseq, -1, pxform))
		goto error;

	pxform->numinchans = 1;
	pxform->numoutchans = 3;

	shapmat->mono = 1;
	shapmat->useluts = 1;
	shapmat->usemat = 1;
	if (!op) {
		shapmat->order = 0;
		shapmat->mat[0][0] = 0.9642;
		shapmat->mat[1][0] = 1.0;
		shapmat->mat[2][0] = 0.8249;
		if (jas_cmshapmatlut_set(&shapmat->luts[0], &graytrc->data.curv))
			goto error;
	} else {
		shapmat->order = 1;
		shapmat->mat[0][0] = 1.0 / 0.9642;
		shapmat->mat[1][0] = 1.0;
		shapmat->mat[2][0] = 1.0 / 0.8249;
		jas_cmshapmatlut_init(&lut);
		if (jas_cmshapmatlut_set(&lut, &graytrc->data.curv))
			goto error;
		if (jas_cmshapmatlut_invert(&shapmat->luts[0], &lut, lut.size))
			goto error;
		jas_cmshapmatlut_cleanup(&lut);
	}
	jas_iccattrval_destroy(graytrc);
	jas_cmpxform_destroy(pxform);
	*retpxformseq = pxformseq;
	return 0;
error:
	return -1;
}

static int triclr(jas_iccprof_t *iccprof, int op, jas_cmpxformseq_t **retpxformseq)
{
	int i;
	jas_iccattrval_t *trcs[3];
	jas_iccattrval_t *cols[3];
	jas_cmshapmat_t *shapmat;
	jas_cmpxform_t *pxform;
	jas_cmpxformseq_t *pxformseq;
	jas_cmreal_t mat[3][4];
	jas_cmshapmatlut_t lut;

	pxform = 0;
	pxformseq = 0;
	for (i = 0; i < 3; ++i) {
		trcs[i] = 0;
		cols[i] = 0;
	}
	jas_cmshapmatlut_init(&lut);

	if (!(trcs[0] = jas_iccprof_getattr(iccprof, JAS_ICC_TAG_REDTRC)) ||
	  !(trcs[1] = jas_iccprof_getattr(iccprof, JAS_ICC_TAG_GRNTRC)) ||
	  !(trcs[2] = jas_iccprof_getattr(iccprof, JAS_ICC_TAG_BLUTRC)) ||
	  !(cols[0] = jas_iccprof_getattr(iccprof, JAS_ICC_TAG_REDMATCOL)) ||
	  !(cols[1] = jas_iccprof_getattr(iccprof, JAS_ICC_TAG_GRNMATCOL)) ||
	  !(cols[2] = jas_iccprof_getattr(iccprof, JAS_ICC_TAG_BLUMATCOL)))
		goto error;
	for (i = 0; i < 3; ++i) {
		if (trcs[i]->type != JAS_ICC_TYPE_CURV ||
		  cols[i]->type != JAS_ICC_TYPE_XYZ)
			goto error;
	}
	if (!(pxform = jas_cmpxform_createshapmat()))
		goto error;
	pxform->numinchans = 3;
	pxform->numoutchans = 3;
	shapmat = &pxform->data.shapmat;
	if (!(pxformseq = jas_cmpxformseq_create()))
		goto error;
	if (jas_cmpxformseq_insertpxform(pxformseq, -1, pxform))
		goto error;
	shapmat->mono = 0;
	shapmat->useluts = 1;
	shapmat->usemat = 1;
	if (!op) {
		shapmat->order = 0;
		for (i = 0; i < 3; ++i) {
			shapmat->mat[0][i] = cols[i]->data.xyz.x / 65536.0;
			shapmat->mat[1][i] = cols[i]->data.xyz.y / 65536.0;
			shapmat->mat[2][i] = cols[i]->data.xyz.z / 65536.0;
		}
		for (i = 0; i < 3; ++i)
			shapmat->mat[i][3] = 0.0;
		for (i = 0; i < 3; ++i) {
			if (jas_cmshapmatlut_set(&shapmat->luts[i], &trcs[i]->data.curv))
				goto error;
		}
	} else {
		shapmat->order = 1;
		for (i = 0; i < 3; ++i) {
			mat[0][i] = cols[i]->data.xyz.x / 65536.0;
			mat[1][i] = cols[i]->data.xyz.y / 65536.0;
			mat[2][i] = cols[i]->data.xyz.z / 65536.0;
		}
		for (i = 0; i < 3; ++i)
			mat[i][3] = 0.0;
		if (jas_cmshapmat_invmat(shapmat->mat, mat))
			goto error;
		for (i = 0; i < 3; ++i) {
			jas_cmshapmatlut_init(&lut);
			if (jas_cmshapmatlut_set(&lut, &trcs[i]->data.curv))
				goto error;
			if (jas_cmshapmatlut_invert(&shapmat->luts[i], &lut, lut.size))
				goto error;
			jas_cmshapmatlut_cleanup(&lut);
		}
	}
	for (i = 0; i < 3; ++i) {
		jas_iccattrval_destroy(trcs[i]);
		jas_iccattrval_destroy(cols[i]);
	}
	jas_cmpxform_destroy(pxform);
	*retpxformseq = pxformseq;
	return 0;

error:

	for (i = 0; i < 3; ++i) {
		if (trcs[i]) {
			jas_iccattrval_destroy(trcs[i]);
		}
		if (cols[i]) {
			jas_iccattrval_destroy(cols[i]);
		}
	}
	if (pxformseq) {
		jas_cmpxformseq_destroy(pxformseq);
	}
	if (pxform) {
		jas_cmpxform_destroy(pxform);
	}

	return -1;
}

static int jas_cmgetint(long **bufptr, int sgnd, int prec, long *val)
{
	long v;
	int m;
	v = **bufptr;
	if (sgnd) {
		m = (1 << (prec - 1));
		if (v < -m || v >= m)
			return -1;
	} else {
		if (v < 0 || v >= (1 << prec))
			return -1;
	}
	++(*bufptr);
	*val = v;
	return 0;
}

static int jas_cmputint(long **bufptr, int sgnd, int prec, long val)
{
	int m;
	if (sgnd) {
		m = (1 << (prec - 1));
		if (val < -m || val >= m)
			return -1;
	} else {
		if (val < 0 || val >= (1 << prec))
			return -1;
	}
	**bufptr = val;
	++(*bufptr);
	return 0;
}

int jas_clrspc_numchans(int clrspc)
{
	switch (jas_clrspc_fam(clrspc)) {
	case JAS_CLRSPC_FAM_XYZ:
	case JAS_CLRSPC_FAM_LAB:
	case JAS_CLRSPC_FAM_RGB:
	case JAS_CLRSPC_FAM_YCBCR:
		return 3;
		break;
	case JAS_CLRSPC_FAM_GRAY:
		return 1;
		break;
	default:
		abort();
		break;
	}
}

jas_iccprof_t *jas_iccprof_createfromcmprof(jas_cmprof_t *prof)
{
	return jas_iccprof_copy(prof->iccprof);
}
