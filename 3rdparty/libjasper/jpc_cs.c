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
 * JPEG-2000 Code Stream Library
 *
 * $Id: jpc_cs.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdlib.h>
#include <assert.h>
#include <ctype.h>

#include "jasper/jas_malloc.h"
#include "jasper/jas_debug.h"

#include "jpc_cs.h"

/******************************************************************************\
* Types.
\******************************************************************************/

/* Marker segment table entry. */
typedef struct {
	int id;
	char *name;
	jpc_msops_t ops;
} jpc_mstabent_t;

/******************************************************************************\
* Local prototypes.
\******************************************************************************/

static jpc_mstabent_t *jpc_mstab_lookup(int id);

static int jpc_poc_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_poc_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_poc_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static void jpc_poc_destroyparms(jpc_ms_t *ms);

static int jpc_unk_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_sot_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_siz_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_cod_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_coc_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_qcd_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_qcc_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_rgn_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_sop_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_ppm_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_ppt_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_crg_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);
static int jpc_com_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in);

static int jpc_sot_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_siz_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_cod_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_coc_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_qcd_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_qcc_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_rgn_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_unk_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_sop_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_ppm_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_ppt_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_crg_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);
static int jpc_com_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out);

static int jpc_sot_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_siz_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_cod_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_coc_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_qcd_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_qcc_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_rgn_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_unk_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_sop_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_ppm_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_ppt_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_crg_dumpparms(jpc_ms_t *ms, FILE *out);
static int jpc_com_dumpparms(jpc_ms_t *ms, FILE *out);

static void jpc_siz_destroyparms(jpc_ms_t *ms);
static void jpc_qcd_destroyparms(jpc_ms_t *ms);
static void jpc_qcc_destroyparms(jpc_ms_t *ms);
static void jpc_cod_destroyparms(jpc_ms_t *ms);
static void jpc_coc_destroyparms(jpc_ms_t *ms);
static void jpc_unk_destroyparms(jpc_ms_t *ms);
static void jpc_ppm_destroyparms(jpc_ms_t *ms);
static void jpc_ppt_destroyparms(jpc_ms_t *ms);
static void jpc_crg_destroyparms(jpc_ms_t *ms);
static void jpc_com_destroyparms(jpc_ms_t *ms);

static void jpc_qcx_destroycompparms(jpc_qcxcp_t *compparms);
static int jpc_qcx_getcompparms(jpc_qcxcp_t *compparms, jpc_cstate_t *cstate,
  jas_stream_t *in, uint_fast16_t len);
static int jpc_qcx_putcompparms(jpc_qcxcp_t *compparms, jpc_cstate_t *cstate,
  jas_stream_t *out);
static void jpc_cox_destroycompparms(jpc_coxcp_t *compparms);
static int jpc_cox_getcompparms(jpc_ms_t *ms, jpc_cstate_t *cstate,
  jas_stream_t *in, int prtflag, jpc_coxcp_t *compparms);
static int jpc_cox_putcompparms(jpc_ms_t *ms, jpc_cstate_t *cstate,
  jas_stream_t *out, int prtflag, jpc_coxcp_t *compparms);

/******************************************************************************\
* Global data.
\******************************************************************************/

static jpc_mstabent_t jpc_mstab[] = {
	{JPC_MS_SOC, "SOC", {0, 0, 0, 0}},
	{JPC_MS_SOT, "SOT", {0, jpc_sot_getparms, jpc_sot_putparms,
	  jpc_sot_dumpparms}},
	{JPC_MS_SOD, "SOD", {0, 0, 0, 0}},
	{JPC_MS_EOC, "EOC", {0, 0, 0, 0}},
	{JPC_MS_SIZ, "SIZ", {jpc_siz_destroyparms, jpc_siz_getparms,
	  jpc_siz_putparms, jpc_siz_dumpparms}},
	{JPC_MS_COD, "COD", {jpc_cod_destroyparms, jpc_cod_getparms,
	  jpc_cod_putparms, jpc_cod_dumpparms}},
	{JPC_MS_COC, "COC", {jpc_coc_destroyparms, jpc_coc_getparms,
	  jpc_coc_putparms, jpc_coc_dumpparms}},
	{JPC_MS_RGN, "RGN", {0, jpc_rgn_getparms, jpc_rgn_putparms,
	  jpc_rgn_dumpparms}},
	{JPC_MS_QCD, "QCD", {jpc_qcd_destroyparms, jpc_qcd_getparms,
	  jpc_qcd_putparms, jpc_qcd_dumpparms}},
	{JPC_MS_QCC, "QCC", {jpc_qcc_destroyparms, jpc_qcc_getparms,
	  jpc_qcc_putparms, jpc_qcc_dumpparms}},
	{JPC_MS_POC, "POC", {jpc_poc_destroyparms, jpc_poc_getparms,
	  jpc_poc_putparms, jpc_poc_dumpparms}},
	{JPC_MS_TLM, "TLM", {0, jpc_unk_getparms, jpc_unk_putparms, 0}},
	{JPC_MS_PLM, "PLM", {0, jpc_unk_getparms, jpc_unk_putparms, 0}},
	{JPC_MS_PPM, "PPM", {jpc_ppm_destroyparms, jpc_ppm_getparms,
	  jpc_ppm_putparms, jpc_ppm_dumpparms}},
	{JPC_MS_PPT, "PPT", {jpc_ppt_destroyparms, jpc_ppt_getparms,
	  jpc_ppt_putparms, jpc_ppt_dumpparms}},
	{JPC_MS_SOP, "SOP", {0, jpc_sop_getparms, jpc_sop_putparms,
	  jpc_sop_dumpparms}},
	{JPC_MS_EPH, "EPH", {0, 0, 0, 0}},
	{JPC_MS_CRG, "CRG", {0, jpc_crg_getparms, jpc_crg_putparms,
	  jpc_crg_dumpparms}},
	{JPC_MS_COM, "COM", {jpc_com_destroyparms, jpc_com_getparms,
	  jpc_com_putparms, jpc_com_dumpparms}},
	{-1, "UNKNOWN",  {jpc_unk_destroyparms, jpc_unk_getparms,
	  jpc_unk_putparms, jpc_unk_dumpparms}}
};

/******************************************************************************\
* Code stream manipulation functions.
\******************************************************************************/

/* Create a code stream state object. */
jpc_cstate_t *jpc_cstate_create()
{
	jpc_cstate_t *cstate;
	if (!(cstate = jas_malloc(sizeof(jpc_cstate_t)))) {
		return 0;
	}
	cstate->numcomps = 0;
	return cstate;
}

/* Destroy a code stream state object. */
void jpc_cstate_destroy(jpc_cstate_t *cstate)
{
	jas_free(cstate);
}

/* Read a marker segment from a stream. */
jpc_ms_t *jpc_getms(jas_stream_t *in, jpc_cstate_t *cstate)
{
	jpc_ms_t *ms;
	jpc_mstabent_t *mstabent;
	jas_stream_t *tmpstream;

	if (!(ms = jpc_ms_create(0))) {
		return 0;
	}

	/* Get the marker type. */
	if (jpc_getuint16(in, &ms->id) || ms->id < JPC_MS_MIN ||
	  ms->id > JPC_MS_MAX) {
		jpc_ms_destroy(ms);
		return 0;
	}

	mstabent = jpc_mstab_lookup(ms->id);
	ms->ops = &mstabent->ops;

	/* Get the marker segment length and parameters if present. */
	/* Note: It is tacitly assumed that a marker segment cannot have
	  parameters unless it has a length field.  That is, there cannot
	  be a parameters field without a length field and vice versa. */
	if (JPC_MS_HASPARMS(ms->id)) {
		/* Get the length of the marker segment. */
		if (jpc_getuint16(in, &ms->len) || ms->len < 3) {
			jpc_ms_destroy(ms);
			return 0;
		}
		/* Calculate the length of the marker segment parameters. */
		ms->len -= 2;
		/* Create and prepare a temporary memory stream from which to
		  read the marker segment parameters. */
		/* Note: This approach provides a simple way of ensuring that
		  we never read beyond the end of the marker segment (even if
		  the marker segment length is errantly set too small). */
		if (!(tmpstream = jas_stream_memopen(0, 0))) {
			jpc_ms_destroy(ms);
			return 0;
		}
		if (jas_stream_copy(tmpstream, in, ms->len) ||
		  jas_stream_seek(tmpstream, 0, SEEK_SET) < 0) {
			jas_stream_close(tmpstream);
			jpc_ms_destroy(ms);
			return 0;
		}
		/* Get the marker segment parameters. */
		if ((*ms->ops->getparms)(ms, cstate, tmpstream)) {
			ms->ops = 0;
			jpc_ms_destroy(ms);
			jas_stream_close(tmpstream);
			return 0;
		}

		if (jas_getdbglevel() > 0) {
			jpc_ms_dump(ms, stderr);
		}

		if (JAS_CAST(ulong, jas_stream_tell(tmpstream)) != ms->len) {
			jas_eprintf("warning: trailing garbage in marker segment (%ld bytes)\n",
			  ms->len - jas_stream_tell(tmpstream));
		}

		/* Close the temporary stream. */
		jas_stream_close(tmpstream);

	} else {
		/* There are no marker segment parameters. */
		ms->len = 0;

		if (jas_getdbglevel() > 0) {
			jpc_ms_dump(ms, stderr);
		}
	}

	/* Update the code stream state information based on the type of
	  marker segment read. */
	/* Note: This is a bit of a hack, but I'm not going to define another
	  type of virtual function for this one special case. */
	if (ms->id == JPC_MS_SIZ) {
		cstate->numcomps = ms->parms.siz.numcomps;
	}

	return ms;
}

/* Write a marker segment to a stream. */
int jpc_putms(jas_stream_t *out, jpc_cstate_t *cstate, jpc_ms_t *ms)
{
	jas_stream_t *tmpstream;
	int len;

	/* Output the marker segment type. */
	if (jpc_putuint16(out, ms->id)) {
		return -1;
	}

	/* Output the marker segment length and parameters if necessary. */
	if (ms->ops->putparms) {
		/* Create a temporary stream in which to buffer the
		  parameter data. */
		if (!(tmpstream = jas_stream_memopen(0, 0))) {
			return -1;
		}
		if ((*ms->ops->putparms)(ms, cstate, tmpstream)) {
			jas_stream_close(tmpstream);
			return -1;
		}
		/* Get the number of bytes of parameter data written. */
		if ((len = jas_stream_tell(tmpstream)) < 0) {
			jas_stream_close(tmpstream);
			return -1;
		}
		ms->len = len;
		/* Write the marker segment length and parameter data to
		  the output stream. */
		if (jas_stream_seek(tmpstream, 0, SEEK_SET) < 0 ||
		  jpc_putuint16(out, ms->len + 2) ||
		  jas_stream_copy(out, tmpstream, ms->len) < 0) {
			jas_stream_close(tmpstream);
			return -1;
		}
		/* Close the temporary stream. */
		jas_stream_close(tmpstream);
	}

	/* This is a bit of a hack, but I'm not going to define another
	  type of virtual function for this one special case. */
	if (ms->id == JPC_MS_SIZ) {
		cstate->numcomps = ms->parms.siz.numcomps;
	}

	if (jas_getdbglevel() > 0) {
		jpc_ms_dump(ms, stderr);
	}

	return 0;
}

/******************************************************************************\
* Marker segment operations.
\******************************************************************************/

/* Create a marker segment of the specified type. */
jpc_ms_t *jpc_ms_create(int type)
{
	jpc_ms_t *ms;
	jpc_mstabent_t *mstabent;

	if (!(ms = jas_malloc(sizeof(jpc_ms_t)))) {
		return 0;
	}
	ms->id = type;
	ms->len = 0;
	mstabent = jpc_mstab_lookup(ms->id);
	ms->ops = &mstabent->ops;
	memset(&ms->parms, 0, sizeof(jpc_msparms_t));
	return ms;
}

/* Destroy a marker segment. */
void jpc_ms_destroy(jpc_ms_t *ms)
{
	if (ms->ops && ms->ops->destroyparms) {
		(*ms->ops->destroyparms)(ms);
	}
	jas_free(ms);
}

/* Dump a marker segment to a stream for debugging. */
void jpc_ms_dump(jpc_ms_t *ms, FILE *out)
{
	jpc_mstabent_t *mstabent;
	mstabent = jpc_mstab_lookup(ms->id);
	fprintf(out, "type = 0x%04x (%s);", ms->id, mstabent->name);
	if (JPC_MS_HASPARMS(ms->id)) {
		fprintf(out, " len = %d;", ms->len + 2);
		if (ms->ops->dumpparms) {
			(*ms->ops->dumpparms)(ms, out);
		} else {
			fprintf(out, "\n");
		}
	} else {
		fprintf(out, "\n");
	}
}

/******************************************************************************\
* SOT marker segment operations.
\******************************************************************************/

static int jpc_sot_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_sot_t *sot = &ms->parms.sot;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	if (jpc_getuint16(in, &sot->tileno) ||
	  jpc_getuint32(in, &sot->len) ||
	  jpc_getuint8(in, &sot->partno) ||
	  jpc_getuint8(in, &sot->numparts)) {
		return -1;
	}
	if (jas_stream_eof(in)) {
		return -1;
	}
	return 0;
}

static int jpc_sot_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_sot_t *sot = &ms->parms.sot;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	if (jpc_putuint16(out, sot->tileno) ||
	  jpc_putuint32(out, sot->len) ||
	  jpc_putuint8(out, sot->partno) ||
	  jpc_putuint8(out, sot->numparts)) {
		return -1;
	}
	return 0;
}

static int jpc_sot_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_sot_t *sot = &ms->parms.sot;
	fprintf(out, "tileno = %d; len = %d; partno = %d; numparts = %d\n",
	  sot->tileno, sot->len, sot->partno, sot->numparts);
	return 0;
}

/******************************************************************************\
* SIZ marker segment operations.
\******************************************************************************/

static void jpc_siz_destroyparms(jpc_ms_t *ms)
{
	jpc_siz_t *siz = &ms->parms.siz;
	if (siz->comps) {
		jas_free(siz->comps);
	}
}

static int jpc_siz_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate,
  jas_stream_t *in)
{
	jpc_siz_t *siz = &ms->parms.siz;
	unsigned int i;
	uint_fast8_t tmp;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	if (jpc_getuint16(in, &siz->caps) ||
	  jpc_getuint32(in, &siz->width) ||
	  jpc_getuint32(in, &siz->height) ||
	  jpc_getuint32(in, &siz->xoff) ||
	  jpc_getuint32(in, &siz->yoff) ||
	  jpc_getuint32(in, &siz->tilewidth) ||
	  jpc_getuint32(in, &siz->tileheight) ||
	  jpc_getuint32(in, &siz->tilexoff) ||
	  jpc_getuint32(in, &siz->tileyoff) ||
	  jpc_getuint16(in, &siz->numcomps)) {
		return -1;
	}
	if (!siz->width || !siz->height || !siz->tilewidth ||
	  !siz->tileheight || !siz->numcomps) {
		return -1;
	}
	if (!(siz->comps = jas_malloc(siz->numcomps * sizeof(jpc_sizcomp_t)))) {
		return -1;
	}
	for (i = 0; i < siz->numcomps; ++i) {
		if (jpc_getuint8(in, &tmp) ||
		  jpc_getuint8(in, &siz->comps[i].hsamp) ||
		  jpc_getuint8(in, &siz->comps[i].vsamp)) {
			jas_free(siz->comps);
			return -1;
		}
		siz->comps[i].sgnd = (tmp >> 7) & 1;
		siz->comps[i].prec = (tmp & 0x7f) + 1;
	}
	if (jas_stream_eof(in)) {
		jas_free(siz->comps);
		return -1;
	}
	return 0;
}

static int jpc_siz_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_siz_t *siz = &ms->parms.siz;
	unsigned int i;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	assert(siz->width && siz->height && siz->tilewidth &&
	  siz->tileheight && siz->numcomps);
	if (jpc_putuint16(out, siz->caps) ||
	  jpc_putuint32(out, siz->width) ||
	  jpc_putuint32(out, siz->height) ||
	  jpc_putuint32(out, siz->xoff) ||
	  jpc_putuint32(out, siz->yoff) ||
	  jpc_putuint32(out, siz->tilewidth) ||
	  jpc_putuint32(out, siz->tileheight) ||
	  jpc_putuint32(out, siz->tilexoff) ||
	  jpc_putuint32(out, siz->tileyoff) ||
	  jpc_putuint16(out, siz->numcomps)) {
		return -1;
	}
	for (i = 0; i < siz->numcomps; ++i) {
		if (jpc_putuint8(out, ((siz->comps[i].sgnd & 1) << 7) |
		  ((siz->comps[i].prec - 1) & 0x7f)) ||
		  jpc_putuint8(out, siz->comps[i].hsamp) ||
		  jpc_putuint8(out, siz->comps[i].vsamp)) {
			return -1;
		}
	}
	return 0;
}

static int jpc_siz_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_siz_t *siz = &ms->parms.siz;
	unsigned int i;
	fprintf(out, "caps = 0x%02x;\n", siz->caps);
	fprintf(out, "width = %d; height = %d; xoff = %d; yoff = %d;\n",
	  siz->width, siz->height, siz->xoff, siz->yoff);
	fprintf(out, "tilewidth = %d; tileheight = %d; tilexoff = %d; "
	  "tileyoff = %d;\n", siz->tilewidth, siz->tileheight, siz->tilexoff,
	  siz->tileyoff);
	for (i = 0; i < siz->numcomps; ++i) {
		fprintf(out, "prec[%d] = %d; sgnd[%d] = %d; hsamp[%d] = %d; "
		  "vsamp[%d] = %d\n", i, siz->comps[i].prec, i,
		  siz->comps[i].sgnd, i, siz->comps[i].hsamp, i,
		  siz->comps[i].vsamp);
	}
	return 0;
}

/******************************************************************************\
* COD marker segment operations.
\******************************************************************************/

static void jpc_cod_destroyparms(jpc_ms_t *ms)
{
	jpc_cod_t *cod = &ms->parms.cod;
	jpc_cox_destroycompparms(&cod->compparms);
}

static int jpc_cod_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_cod_t *cod = &ms->parms.cod;
	if (jpc_getuint8(in, &cod->csty)) {
		return -1;
	}
	if (jpc_getuint8(in, &cod->prg) ||
	  jpc_getuint16(in, &cod->numlyrs) ||
	  jpc_getuint8(in, &cod->mctrans)) {
		return -1;
	}
	if (jpc_cox_getcompparms(ms, cstate, in,
	  (cod->csty & JPC_COX_PRT) != 0, &cod->compparms)) {
		return -1;
	}
	if (jas_stream_eof(in)) {
		jpc_cod_destroyparms(ms);
		return -1;
	}
	return 0;
}

static int jpc_cod_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_cod_t *cod = &ms->parms.cod;
	assert(cod->numlyrs > 0 && cod->compparms.numdlvls <= 32);
	assert(cod->compparms.numdlvls == cod->compparms.numrlvls - 1);
	if (jpc_putuint8(out, cod->compparms.csty) ||
	  jpc_putuint8(out, cod->prg) ||
	  jpc_putuint16(out, cod->numlyrs) ||
	  jpc_putuint8(out, cod->mctrans)) {
		return -1;
	}
	if (jpc_cox_putcompparms(ms, cstate, out,
	  (cod->csty & JPC_COX_PRT) != 0, &cod->compparms)) {
		return -1;
	}
	return 0;
}

static int jpc_cod_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_cod_t *cod = &ms->parms.cod;
	int i;
	fprintf(out, "csty = 0x%02x;\n", cod->compparms.csty);
	fprintf(out, "numdlvls = %d; qmfbid = %d; mctrans = %d\n",
	  cod->compparms.numdlvls, cod->compparms.qmfbid, cod->mctrans);
	fprintf(out, "prg = %d; numlyrs = %d;\n",
	  cod->prg, cod->numlyrs);
	fprintf(out, "cblkwidthval = %d; cblkheightval = %d; "
	  "cblksty = 0x%02x;\n", cod->compparms.cblkwidthval, cod->compparms.cblkheightval,
	  cod->compparms.cblksty);
	if (cod->csty & JPC_COX_PRT) {
		for (i = 0; i < cod->compparms.numrlvls; ++i) {
			jas_eprintf("prcwidth[%d] = %d, prcheight[%d] = %d\n",
			  i, cod->compparms.rlvls[i].parwidthval,
			  i, cod->compparms.rlvls[i].parheightval);
		}
	}
	return 0;
}

/******************************************************************************\
* COC marker segment operations.
\******************************************************************************/

static void jpc_coc_destroyparms(jpc_ms_t *ms)
{
	jpc_coc_t *coc = &ms->parms.coc;
	jpc_cox_destroycompparms(&coc->compparms);
}

static int jpc_coc_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_coc_t *coc = &ms->parms.coc;
	uint_fast8_t tmp;
	if (cstate->numcomps <= 256) {
		if (jpc_getuint8(in, &tmp)) {
			return -1;
		}
		coc->compno = tmp;
	} else {
		if (jpc_getuint16(in, &coc->compno)) {
			return -1;
		}
	}
	if (jpc_getuint8(in, &coc->compparms.csty)) {
		return -1;
	}
	if (jpc_cox_getcompparms(ms, cstate, in,
	  (coc->compparms.csty & JPC_COX_PRT) != 0, &coc->compparms)) {
		return -1;
	}
	if (jas_stream_eof(in)) {
		return -1;
	}
	return 0;
}

static int jpc_coc_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_coc_t *coc = &ms->parms.coc;
	assert(coc->compparms.numdlvls <= 32);
	if (cstate->numcomps <= 256) {
		if (jpc_putuint8(out, coc->compno)) {
			return -1;
		}
	} else {
		if (jpc_putuint16(out, coc->compno)) {
			return -1;
		}
	}
	if (jpc_putuint8(out, coc->compparms.csty)) {
		return -1;
	}
	if (jpc_cox_putcompparms(ms, cstate, out,
	  (coc->compparms.csty & JPC_COX_PRT) != 0, &coc->compparms)) {
		return -1;
	}
	return 0;
}

static int jpc_coc_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_coc_t *coc = &ms->parms.coc;
	fprintf(out, "compno = %d; csty = 0x%02x; numdlvls = %d;\n",
	  coc->compno, coc->compparms.csty, coc->compparms.numdlvls);
	fprintf(out, "cblkwidthval = %d; cblkheightval = %d; "
	  "cblksty = 0x%02x; qmfbid = %d;\n", coc->compparms.cblkwidthval,
	  coc->compparms.cblkheightval, coc->compparms.cblksty, coc->compparms.qmfbid);
	return 0;
}
/******************************************************************************\
* COD/COC marker segment operation helper functions.
\******************************************************************************/

static void jpc_cox_destroycompparms(jpc_coxcp_t *compparms)
{
	/* Eliminate compiler warning about unused variables. */
	compparms = 0;
}

static int jpc_cox_getcompparms(jpc_ms_t *ms, jpc_cstate_t *cstate,
  jas_stream_t *in, int prtflag, jpc_coxcp_t *compparms)
{
	uint_fast8_t tmp;
	int i;

	/* Eliminate compiler warning about unused variables. */
	ms = 0;
	cstate = 0;

	if (jpc_getuint8(in, &compparms->numdlvls) ||
	  jpc_getuint8(in, &compparms->cblkwidthval) ||
	  jpc_getuint8(in, &compparms->cblkheightval) ||
	  jpc_getuint8(in, &compparms->cblksty) ||
	  jpc_getuint8(in, &compparms->qmfbid)) {
		return -1;
	}
	compparms->numrlvls = compparms->numdlvls + 1;
	if (prtflag) {
		for (i = 0; i < compparms->numrlvls; ++i) {
			if (jpc_getuint8(in, &tmp)) {
				jpc_cox_destroycompparms(compparms);
				return -1;
			}
			compparms->rlvls[i].parwidthval = tmp & 0xf;
			compparms->rlvls[i].parheightval = (tmp >> 4) & 0xf;
		}
/* Sigh.  This bit should be in the same field in both COC and COD mrk segs. */
compparms->csty |= JPC_COX_PRT;
	} else {
	}
	if (jas_stream_eof(in)) {
		jpc_cox_destroycompparms(compparms);
		return -1;
	}
	return 0;
}

static int jpc_cox_putcompparms(jpc_ms_t *ms, jpc_cstate_t *cstate,
  jas_stream_t *out, int prtflag, jpc_coxcp_t *compparms)
{
	int i;
	assert(compparms->numdlvls <= 32);

	/* Eliminate compiler warning about unused variables. */
	ms = 0;
	cstate = 0;

	if (jpc_putuint8(out, compparms->numdlvls) ||
	  jpc_putuint8(out, compparms->cblkwidthval) ||
	  jpc_putuint8(out, compparms->cblkheightval) ||
	  jpc_putuint8(out, compparms->cblksty) ||
	  jpc_putuint8(out, compparms->qmfbid)) {
		return -1;
	}
	if (prtflag) {
		for (i = 0; i < compparms->numrlvls; ++i) {
			if (jpc_putuint8(out,
			  ((compparms->rlvls[i].parheightval & 0xf) << 4) |
			  (compparms->rlvls[i].parwidthval & 0xf))) {
				return -1;
			}
		}
	}
	return 0;
}

/******************************************************************************\
* RGN marker segment operations.
\******************************************************************************/

static int jpc_rgn_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_rgn_t *rgn = &ms->parms.rgn;
	uint_fast8_t tmp;
	if (cstate->numcomps <= 256) {
		if (jpc_getuint8(in, &tmp)) {
			return -1;
		}
		rgn->compno = tmp;
	} else {
		if (jpc_getuint16(in, &rgn->compno)) {
			return -1;
		}
	}
	if (jpc_getuint8(in, &rgn->roisty) ||
	  jpc_getuint8(in, &rgn->roishift)) {
		return -1;
	}
	return 0;
}

static int jpc_rgn_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_rgn_t *rgn = &ms->parms.rgn;
	if (cstate->numcomps <= 256) {
		if (jpc_putuint8(out, rgn->compno)) {
			return -1;
		}
	} else {
		if (jpc_putuint16(out, rgn->compno)) {
			return -1;
		}
	}
	if (jpc_putuint8(out, rgn->roisty) ||
	  jpc_putuint8(out, rgn->roishift)) {
		return -1;
	}
	return 0;
}

static int jpc_rgn_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_rgn_t *rgn = &ms->parms.rgn;
	fprintf(out, "compno = %d; roisty = %d; roishift = %d\n",
	  rgn->compno, rgn->roisty, rgn->roishift);
	return 0;
}

/******************************************************************************\
* QCD marker segment operations.
\******************************************************************************/

static void jpc_qcd_destroyparms(jpc_ms_t *ms)
{
	jpc_qcd_t *qcd = &ms->parms.qcd;
	jpc_qcx_destroycompparms(&qcd->compparms);
}

static int jpc_qcd_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_qcxcp_t *compparms = &ms->parms.qcd.compparms;
	return jpc_qcx_getcompparms(compparms, cstate, in, ms->len);
}

static int jpc_qcd_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_qcxcp_t *compparms = &ms->parms.qcd.compparms;
	return jpc_qcx_putcompparms(compparms, cstate, out);
}

static int jpc_qcd_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_qcd_t *qcd = &ms->parms.qcd;
	int i;
	fprintf(out, "qntsty = %d; numguard = %d; numstepsizes = %d\n",
	  (int) qcd->compparms.qntsty, qcd->compparms.numguard, qcd->compparms.numstepsizes);
	for (i = 0; i < qcd->compparms.numstepsizes; ++i) {
		fprintf(out, "expn[%d] = 0x%04x; mant[%d] = 0x%04x;\n",
		  i, (unsigned) JPC_QCX_GETEXPN(qcd->compparms.stepsizes[i]),
		  i, (unsigned) JPC_QCX_GETMANT(qcd->compparms.stepsizes[i]));
	}
	return 0;
}

/******************************************************************************\
* QCC marker segment operations.
\******************************************************************************/

static void jpc_qcc_destroyparms(jpc_ms_t *ms)
{
	jpc_qcc_t *qcc = &ms->parms.qcc;
	jpc_qcx_destroycompparms(&qcc->compparms);
}

static int jpc_qcc_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_qcc_t *qcc = &ms->parms.qcc;
	uint_fast8_t tmp;
	int len;
	len = ms->len;
	if (cstate->numcomps <= 256) {
		jpc_getuint8(in, &tmp);
		qcc->compno = tmp;
		--len;
	} else {
		jpc_getuint16(in, &qcc->compno);
		len -= 2;
	}
	if (jpc_qcx_getcompparms(&qcc->compparms, cstate, in, len)) {
		return -1;
	}
	if (jas_stream_eof(in)) {
		jpc_qcc_destroyparms(ms);
		return -1;
	}
	return 0;
}

static int jpc_qcc_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_qcc_t *qcc = &ms->parms.qcc;
	if (cstate->numcomps <= 256) {
		jpc_putuint8(out, qcc->compno);
	} else {
		jpc_putuint16(out, qcc->compno);
	}
	if (jpc_qcx_putcompparms(&qcc->compparms, cstate, out)) {
		return -1;
	}
	return 0;
}

static int jpc_qcc_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_qcc_t *qcc = &ms->parms.qcc;
	int i;
	fprintf(out, "compno = %d; qntsty = %d; numguard = %d; "
	  "numstepsizes = %d\n", qcc->compno, qcc->compparms.qntsty, qcc->compparms.numguard,
	  qcc->compparms.numstepsizes);
	for (i = 0; i < qcc->compparms.numstepsizes; ++i) {
		fprintf(out, "expn[%d] = 0x%04x; mant[%d] = 0x%04x;\n",
		  i, (unsigned) JPC_QCX_GETEXPN(qcc->compparms.stepsizes[i]),
		  i, (unsigned) JPC_QCX_GETMANT(qcc->compparms.stepsizes[i]));
	}
	return 0;
}

/******************************************************************************\
* QCD/QCC marker segment helper functions.
\******************************************************************************/

static void jpc_qcx_destroycompparms(jpc_qcxcp_t *compparms)
{
	if (compparms->stepsizes) {
		jas_free(compparms->stepsizes);
	}
}

static int jpc_qcx_getcompparms(jpc_qcxcp_t *compparms, jpc_cstate_t *cstate,
  jas_stream_t *in, uint_fast16_t len)
{
	uint_fast8_t tmp;
	int n;
	int i;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	n = 0;
	jpc_getuint8(in, &tmp);
	++n;
	compparms->qntsty = tmp & 0x1f;
	compparms->numguard = (tmp >> 5) & 7;
	switch (compparms->qntsty) {
	case JPC_QCX_SIQNT:
		compparms->numstepsizes = 1;
		break;
	case JPC_QCX_NOQNT:
		compparms->numstepsizes = (len - n);
		break;
	case JPC_QCX_SEQNT:
		/* XXX - this is a hack */
		compparms->numstepsizes = (len - n) / 2;
		break;
	}
	if (compparms->numstepsizes > 0) {
		compparms->stepsizes = jas_malloc(compparms->numstepsizes *
		  sizeof(uint_fast16_t));
		assert(compparms->stepsizes);
		for (i = 0; i < compparms->numstepsizes; ++i) {
			if (compparms->qntsty == JPC_QCX_NOQNT) {
				jpc_getuint8(in, &tmp);
				compparms->stepsizes[i] = JPC_QCX_EXPN(tmp >> 3);
			} else {
				jpc_getuint16(in, &compparms->stepsizes[i]);
			}
		}
	} else {
		compparms->stepsizes = 0;
	}
	if (jas_stream_error(in) || jas_stream_eof(in)) {
		jpc_qcx_destroycompparms(compparms);
		return -1;
	}
	return 0;
}

static int jpc_qcx_putcompparms(jpc_qcxcp_t *compparms, jpc_cstate_t *cstate,
  jas_stream_t *out)
{
	int i;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	jpc_putuint8(out, ((compparms->numguard & 7) << 5) | compparms->qntsty);
	for (i = 0; i < compparms->numstepsizes; ++i) {
		if (compparms->qntsty == JPC_QCX_NOQNT) {
			jpc_putuint8(out, JPC_QCX_GETEXPN(
			  compparms->stepsizes[i]) << 3);
		} else {
			jpc_putuint16(out, compparms->stepsizes[i]);
		}
	}
	return 0;
}

/******************************************************************************\
* SOP marker segment operations.
\******************************************************************************/

static int jpc_sop_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_sop_t *sop = &ms->parms.sop;

	/* Eliminate compiler warning about unused variable. */
	cstate = 0;

	if (jpc_getuint16(in, &sop->seqno)) {
		return -1;
	}
	return 0;
}

static int jpc_sop_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_sop_t *sop = &ms->parms.sop;

	/* Eliminate compiler warning about unused variable. */
	cstate = 0;

	if (jpc_putuint16(out, sop->seqno)) {
		return -1;
	}
	return 0;
}

static int jpc_sop_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_sop_t *sop = &ms->parms.sop;
	fprintf(out, "seqno = %d;\n", sop->seqno);
	return 0;
}

/******************************************************************************\
* PPM marker segment operations.
\******************************************************************************/

static void jpc_ppm_destroyparms(jpc_ms_t *ms)
{
	jpc_ppm_t *ppm = &ms->parms.ppm;
	if (ppm->data) {
		jas_free(ppm->data);
	}
}

static int jpc_ppm_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_ppm_t *ppm = &ms->parms.ppm;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	ppm->data = 0;

	if (ms->len < 1) {
		goto error;
	}
	if (jpc_getuint8(in, &ppm->ind)) {
		goto error;
	}

	ppm->len = ms->len - 1;
	if (ppm->len > 0) {
		if (!(ppm->data = jas_malloc(ppm->len * sizeof(unsigned char)))) {
			goto error;
		}
		if (JAS_CAST(uint, jas_stream_read(in, ppm->data, ppm->len)) != ppm->len) {
			goto error;
		}
	} else {
		ppm->data = 0;
	}
	return 0;

error:
	jpc_ppm_destroyparms(ms);
	return -1;
}

static int jpc_ppm_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_ppm_t *ppm = &ms->parms.ppm;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	if (JAS_CAST(uint, jas_stream_write(out, (char *) ppm->data, ppm->len)) != ppm->len) {
		return -1;
	}
	return 0;
}

static int jpc_ppm_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_ppm_t *ppm = &ms->parms.ppm;
	fprintf(out, "ind=%d; len = %d;\n", ppm->ind, ppm->len);
	if (ppm->len > 0) {
		fprintf(out, "data =\n");
		jas_memdump(out, ppm->data, ppm->len);
	}
	return 0;
}

/******************************************************************************\
* PPT marker segment operations.
\******************************************************************************/

static void jpc_ppt_destroyparms(jpc_ms_t *ms)
{
	jpc_ppt_t *ppt = &ms->parms.ppt;
	if (ppt->data) {
		jas_free(ppt->data);
	}
}

static int jpc_ppt_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_ppt_t *ppt = &ms->parms.ppt;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	ppt->data = 0;

	if (ms->len < 1) {
		goto error;
	}
	if (jpc_getuint8(in, &ppt->ind)) {
		goto error;
	}
	ppt->len = ms->len - 1;
	if (ppt->len > 0) {
		if (!(ppt->data = jas_malloc(ppt->len * sizeof(unsigned char)))) {
			goto error;
		}
		if (jas_stream_read(in, (char *) ppt->data, ppt->len) != JAS_CAST(int, ppt->len)) {
			goto error;
		}
	} else {
		ppt->data = 0;
	}
	return 0;

error:
	jpc_ppt_destroyparms(ms);
	return -1;
}

static int jpc_ppt_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_ppt_t *ppt = &ms->parms.ppt;

	/* Eliminate compiler warning about unused variable. */
	cstate = 0;

	if (jpc_putuint8(out, ppt->ind)) {
		return -1;
	}
	if (jas_stream_write(out, (char *) ppt->data, ppt->len) != JAS_CAST(int, ppt->len)) {
		return -1;
	}
	return 0;
}

static int jpc_ppt_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_ppt_t *ppt = &ms->parms.ppt;
	fprintf(out, "ind=%d; len = %d;\n", ppt->ind, ppt->len);
	if (ppt->len > 0) {
		fprintf(out, "data =\n");
		jas_memdump(out, ppt->data, ppt->len);
	}
	return 0;
}

/******************************************************************************\
* POC marker segment operations.
\******************************************************************************/

static void jpc_poc_destroyparms(jpc_ms_t *ms)
{
	jpc_poc_t *poc = &ms->parms.poc;
	if (poc->pchgs) {
		jas_free(poc->pchgs);
	}
}

static int jpc_poc_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_poc_t *poc = &ms->parms.poc;
	jpc_pocpchg_t *pchg;
	int pchgno;
	uint_fast8_t tmp;
	poc->numpchgs = (cstate->numcomps > 256) ? (ms->len / 9) :
	  (ms->len / 7);
	if (!(poc->pchgs = jas_malloc(poc->numpchgs * sizeof(jpc_pocpchg_t)))) {
		goto error;
	}
	for (pchgno = 0, pchg = poc->pchgs; pchgno < poc->numpchgs; ++pchgno,
	  ++pchg) {
		if (jpc_getuint8(in, &pchg->rlvlnostart)) {
			goto error;
		}
		if (cstate->numcomps > 256) {
			if (jpc_getuint16(in, &pchg->compnostart)) {
				goto error;
			}
		} else {
			if (jpc_getuint8(in, &tmp)) {
				goto error;
			};
			pchg->compnostart = tmp;
		}
		if (jpc_getuint16(in, &pchg->lyrnoend) ||
		  jpc_getuint8(in, &pchg->rlvlnoend)) {
			goto error;
		}
		if (cstate->numcomps > 256) {
			if (jpc_getuint16(in, &pchg->compnoend)) {
				goto error;
			}
		} else {
			if (jpc_getuint8(in, &tmp)) {
				goto error;
			}
			pchg->compnoend = tmp;
		}
		if (jpc_getuint8(in, &pchg->prgord)) {
			goto error;
		}
		if (pchg->rlvlnostart > pchg->rlvlnoend ||
		  pchg->compnostart > pchg->compnoend) {
			goto error;
		}
	}
	return 0;

error:
	jpc_poc_destroyparms(ms);
	return -1;
}

static int jpc_poc_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_poc_t *poc = &ms->parms.poc;
	jpc_pocpchg_t *pchg;
	int pchgno;
	for (pchgno = 0, pchg = poc->pchgs; pchgno < poc->numpchgs; ++pchgno,
	  ++pchg) {
		if (jpc_putuint8(out, pchg->rlvlnostart) ||
		  ((cstate->numcomps > 256) ?
		  jpc_putuint16(out, pchg->compnostart) :
		  jpc_putuint8(out, pchg->compnostart)) ||
		  jpc_putuint16(out, pchg->lyrnoend) ||
		  jpc_putuint8(out, pchg->rlvlnoend) ||
		  ((cstate->numcomps > 256) ?
		  jpc_putuint16(out, pchg->compnoend) :
		  jpc_putuint8(out, pchg->compnoend)) ||
		  jpc_putuint8(out, pchg->prgord)) {
			return -1;
		}
	}
	return 0;
}

static int jpc_poc_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_poc_t *poc = &ms->parms.poc;
	jpc_pocpchg_t *pchg;
	int pchgno;
	for (pchgno = 0, pchg = poc->pchgs; pchgno < poc->numpchgs;
	  ++pchgno, ++pchg) {
		fprintf(out, "po[%d] = %d; ", pchgno, pchg->prgord);
		fprintf(out, "cs[%d] = %d; ce[%d] = %d; ",
		  pchgno, pchg->compnostart, pchgno, pchg->compnoend);
		fprintf(out, "rs[%d] = %d; re[%d] = %d; ",
		  pchgno, pchg->rlvlnostart, pchgno, pchg->rlvlnoend);
		fprintf(out, "le[%d] = %d\n", pchgno, pchg->lyrnoend);
	}
	return 0;
}

/******************************************************************************\
* CRG marker segment operations.
\******************************************************************************/

static void jpc_crg_destroyparms(jpc_ms_t *ms)
{
	jpc_crg_t *crg = &ms->parms.crg;
	if (crg->comps) {
		jas_free(crg->comps);
	}
}

static int jpc_crg_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_crg_t *crg = &ms->parms.crg;
	jpc_crgcomp_t *comp;
	uint_fast16_t compno;
	crg->numcomps = cstate->numcomps;
	if (!(crg->comps = jas_malloc(cstate->numcomps * sizeof(uint_fast16_t)))) {
		return -1;
	}
	for (compno = 0, comp = crg->comps; compno < cstate->numcomps;
	  ++compno, ++comp) {
		if (jpc_getuint16(in, &comp->hoff) ||
		  jpc_getuint16(in, &comp->voff)) {
			jpc_crg_destroyparms(ms);
			return -1;
		}
	}
	return 0;
}

static int jpc_crg_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_crg_t *crg = &ms->parms.crg;
	int compno;
	jpc_crgcomp_t *comp;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	for (compno = 0, comp = crg->comps; compno < crg->numcomps; ++compno,
	  ++comp) {
		if (jpc_putuint16(out, comp->hoff) ||
		  jpc_putuint16(out, comp->voff)) {
			return -1;
		}
	}
	return 0;
}

static int jpc_crg_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_crg_t *crg = &ms->parms.crg;
	int compno;
	jpc_crgcomp_t *comp;
	for (compno = 0, comp = crg->comps; compno < crg->numcomps; ++compno,
	  ++comp) {
		fprintf(out, "hoff[%d] = %d; voff[%d] = %d\n", compno,
		  comp->hoff, compno, comp->voff);
	}
	return 0;
}

/******************************************************************************\
* Operations for COM marker segment.
\******************************************************************************/

static void jpc_com_destroyparms(jpc_ms_t *ms)
{
	jpc_com_t *com = &ms->parms.com;
	if (com->data) {
		jas_free(com->data);
	}
}

static int jpc_com_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_com_t *com = &ms->parms.com;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	if (jpc_getuint16(in, &com->regid)) {
		return -1;
	}
	com->len = ms->len - 2;
	if (com->len > 0) {
		if (!(com->data = jas_malloc(com->len))) {
			return -1;
		}
		if (jas_stream_read(in, com->data, com->len) != JAS_CAST(int, com->len)) {
			return -1;
		}
	} else {
		com->data = 0;
	}
	return 0;
}

static int jpc_com_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	jpc_com_t *com = &ms->parms.com;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	if (jpc_putuint16(out, com->regid)) {
		return -1;
	}
	if (jas_stream_write(out, com->data, com->len) != JAS_CAST(int, com->len)) {
		return -1;
	}
	return 0;
}

static int jpc_com_dumpparms(jpc_ms_t *ms, FILE *out)
{
	jpc_com_t *com = &ms->parms.com;
	unsigned int i;
	int printable;
	fprintf(out, "regid = %d;\n", com->regid);
	printable = 1;
	for (i = 0; i < com->len; ++i) {
		if (!isprint(com->data[i])) {
			printable = 0;
			break;
		}
	}
	if (printable) {
		fprintf(out, "data = ");
		fwrite(com->data, sizeof(char), com->len, out);
		fprintf(out, "\n");
	}
	return 0;
}

/******************************************************************************\
* Operations for unknown types of marker segments.
\******************************************************************************/

static void jpc_unk_destroyparms(jpc_ms_t *ms)
{
	jpc_unk_t *unk = &ms->parms.unk;
	if (unk->data) {
		jas_free(unk->data);
	}
}

static int jpc_unk_getparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *in)
{
	jpc_unk_t *unk = &ms->parms.unk;

	/* Eliminate compiler warning about unused variables. */
	cstate = 0;

	if (ms->len > 0) {
		if (!(unk->data = jas_malloc(ms->len * sizeof(unsigned char)))) {
			return -1;
		}
		if (jas_stream_read(in, (char *) unk->data, ms->len) != JAS_CAST(int, ms->len)) {
			jas_free(unk->data);
			return -1;
		}
		unk->len = ms->len;
	} else {
		unk->data = 0;
		unk->len = 0;
	}
	return 0;
}

static int jpc_unk_putparms(jpc_ms_t *ms, jpc_cstate_t *cstate, jas_stream_t *out)
{
	/* Eliminate compiler warning about unused variables. */
	cstate = 0;
	ms = 0;
	out = 0;

	/* If this function is called, we are trying to write an unsupported
	  type of marker segment.  Return with an error indication.  */
	return -1;
}

static int jpc_unk_dumpparms(jpc_ms_t *ms, FILE *out)
{
	unsigned int i;
	jpc_unk_t *unk = &ms->parms.unk;
	for (i = 0; i < unk->len; ++i) {
		fprintf(out, "%02x ", unk->data[i]);
	}
	return 0;
}

/******************************************************************************\
* Primitive I/O operations.
\******************************************************************************/

int jpc_getuint8(jas_stream_t *in, uint_fast8_t *val)
{
	int c;
	if ((c = jas_stream_getc(in)) == EOF) {
		return -1;
	}
	if (val) {
		*val = c;
	}
	return 0;
}

int jpc_putuint8(jas_stream_t *out, uint_fast8_t val)
{
	if (jas_stream_putc(out, val & 0xff) == EOF) {
		return -1;
	}
	return 0;
}

int jpc_getuint16(jas_stream_t *in, uint_fast16_t *val)
{
	uint_fast16_t v;
	int c;
	if ((c = jas_stream_getc(in)) == EOF) {
		return -1;
	}
	v = c;
	if ((c = jas_stream_getc(in)) == EOF) {
		return -1;
	}
	v = (v << 8) | c;
	if (val) {
		*val = v;
	}
	return 0;
}

int jpc_putuint16(jas_stream_t *out, uint_fast16_t val)
{
	if (jas_stream_putc(out, (val >> 8) & 0xff) == EOF ||
	  jas_stream_putc(out, val & 0xff) == EOF) {
		return -1;
	}
	return 0;
}

int jpc_getuint32(jas_stream_t *in, uint_fast32_t *val)
{
	uint_fast32_t v;
	int c;
	if ((c = jas_stream_getc(in)) == EOF) {
		return -1;
	}
	v = c;
	if ((c = jas_stream_getc(in)) == EOF) {
		return -1;
	}
	v = (v << 8) | c;
	if ((c = jas_stream_getc(in)) == EOF) {
		return -1;
	}
	v = (v << 8) | c;
	if ((c = jas_stream_getc(in)) == EOF) {
		return -1;
	}
	v = (v << 8) | c;
	if (val) {
		*val = v;
	}
	return 0;
}

int jpc_putuint32(jas_stream_t *out, uint_fast32_t val)
{
	if (jas_stream_putc(out, (val >> 24) & 0xff) == EOF ||
	  jas_stream_putc(out, (val >> 16) & 0xff) == EOF ||
	  jas_stream_putc(out, (val >> 8) & 0xff) == EOF ||
	  jas_stream_putc(out, val & 0xff) == EOF) {
		return -1;
	}
	return 0;
}

/******************************************************************************\
* Miscellany
\******************************************************************************/

static jpc_mstabent_t *jpc_mstab_lookup(int id)
{
	jpc_mstabent_t *mstabent;
	for (mstabent = jpc_mstab;; ++mstabent) {
		if (mstabent->id == id || mstabent->id < 0) {
			return mstabent;
		}
	}
	assert(0);
	return 0;
}

int jpc_validate(jas_stream_t *in)
{
	int n;
	int i;
	unsigned char buf[2];

	assert(JAS_STREAM_MAXPUTBACK >= 2);

	if ((n = jas_stream_read(in, (char *) buf, 2)) < 0) {
		return -1;
	}
	for (i = n - 1; i >= 0; --i) {
		if (jas_stream_ungetc(in, buf[i]) == EOF) {
			return -1;
		}
	}
	if (n < 2) {
		return -1;
	}
	if (buf[0] == (JPC_MS_SOC >> 8) && buf[1] == (JPC_MS_SOC & 0xff)) {
		return 0;
	}
	return -1;
}

int jpc_getdata(jas_stream_t *in, jas_stream_t *out, long len)
{
	return jas_stream_copy(out, in, len);
}

int jpc_putdata(jas_stream_t *out, jas_stream_t *in, long len)
{
	return jas_stream_copy(out, in, len);
}
