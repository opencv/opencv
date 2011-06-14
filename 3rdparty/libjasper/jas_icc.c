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

#include <assert.h>
#include <jasper/jas_config.h>
#include <jasper/jas_types.h>
#include <jasper/jas_malloc.h>
#include <jasper/jas_debug.h>
#include <jasper/jas_icc.h>
#include <jasper/jas_cm.h>
#include <jasper/jas_stream.h>
#include <jasper/jas_string.h>

#include <stdlib.h>
#include <ctype.h>

#define	jas_iccputuint8(out, val)	jas_iccputuint(out, 1, val)
#define	jas_iccputuint16(out, val)	jas_iccputuint(out, 2, val)
#define	jas_iccputsint32(out, val)	jas_iccputsint(out, 4, val)
#define	jas_iccputuint32(out, val)	jas_iccputuint(out, 4, val)
#define	jas_iccputuint64(out, val)	jas_iccputuint(out, 8, val)

static jas_iccattrval_t *jas_iccattrval_create0(void);

static int jas_iccgetuint(jas_stream_t *in, int n, ulonglong *val);
static int jas_iccgetuint8(jas_stream_t *in, jas_iccuint8_t *val);
static int jas_iccgetuint16(jas_stream_t *in, jas_iccuint16_t *val);
static int jas_iccgetsint32(jas_stream_t *in, jas_iccsint32_t *val);
static int jas_iccgetuint32(jas_stream_t *in, jas_iccuint32_t *val);
static int jas_iccgetuint64(jas_stream_t *in, jas_iccuint64_t *val);
static int jas_iccputuint(jas_stream_t *out, int n, ulonglong val);
static int jas_iccputsint(jas_stream_t *out, int n, longlong val);
static jas_iccprof_t *jas_iccprof_create(void);
static int jas_iccprof_readhdr(jas_stream_t *in, jas_icchdr_t *hdr);
static int jas_iccprof_writehdr(jas_stream_t *out, jas_icchdr_t *hdr);
static int jas_iccprof_gettagtab(jas_stream_t *in, jas_icctagtab_t *tagtab);
static void jas_iccprof_sorttagtab(jas_icctagtab_t *tagtab);
static int jas_iccattrtab_lookup(jas_iccattrtab_t *attrtab, jas_iccuint32_t name);
static jas_iccattrtab_t *jas_iccattrtab_copy(jas_iccattrtab_t *attrtab);
static jas_iccattrvalinfo_t *jas_iccattrvalinfo_lookup(jas_iccsig_t name);
static int jas_iccgettime(jas_stream_t *in, jas_icctime_t *time);
static int jas_iccgetxyz(jas_stream_t *in, jas_iccxyz_t *xyz);
static int jas_icctagtabent_cmp(const void *src, const void *dst);

static void jas_icccurv_destroy(jas_iccattrval_t *attrval);
static int jas_icccurv_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval);
static int jas_icccurv_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt);
static int jas_icccurv_getsize(jas_iccattrval_t *attrval);
static int jas_icccurv_output(jas_iccattrval_t *attrval, jas_stream_t *out);
static void jas_icccurv_dump(jas_iccattrval_t *attrval, FILE *out);

static void jas_icctxtdesc_destroy(jas_iccattrval_t *attrval);
static int jas_icctxtdesc_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval);
static int jas_icctxtdesc_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt);
static int jas_icctxtdesc_getsize(jas_iccattrval_t *attrval);
static int jas_icctxtdesc_output(jas_iccattrval_t *attrval, jas_stream_t *out);
static void jas_icctxtdesc_dump(jas_iccattrval_t *attrval, FILE *out);

static void jas_icctxt_destroy(jas_iccattrval_t *attrval);
static int jas_icctxt_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval);
static int jas_icctxt_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt);
static int jas_icctxt_getsize(jas_iccattrval_t *attrval);
static int jas_icctxt_output(jas_iccattrval_t *attrval, jas_stream_t *out);
static void jas_icctxt_dump(jas_iccattrval_t *attrval, FILE *out);

static int jas_iccxyz_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt);
static int jas_iccxyz_getsize(jas_iccattrval_t *attrval);
static int jas_iccxyz_output(jas_iccattrval_t *attrval, jas_stream_t *out);
static void jas_iccxyz_dump(jas_iccattrval_t *attrval, FILE *out);

static jas_iccattrtab_t *jas_iccattrtab_create(void);
static void jas_iccattrtab_destroy(jas_iccattrtab_t *tab);
static int jas_iccattrtab_resize(jas_iccattrtab_t *tab, int maxents);
static int jas_iccattrtab_add(jas_iccattrtab_t *attrtab, int i,
  jas_iccuint32_t name, jas_iccattrval_t *val);
static int jas_iccattrtab_replace(jas_iccattrtab_t *attrtab, int i,
  jas_iccuint32_t name, jas_iccattrval_t *val);
static void jas_iccattrtab_delete(jas_iccattrtab_t *attrtab, int i);
static long jas_iccpadtomult(long x, long y);
static int jas_iccattrtab_get(jas_iccattrtab_t *attrtab, int i,
  jas_iccattrname_t *name, jas_iccattrval_t **val);
static int jas_iccprof_puttagtab(jas_stream_t *out, jas_icctagtab_t *tagtab);

static void jas_icclut16_destroy(jas_iccattrval_t *attrval);
static int jas_icclut16_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval);
static int jas_icclut16_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt);
static int jas_icclut16_getsize(jas_iccattrval_t *attrval);
static int jas_icclut16_output(jas_iccattrval_t *attrval, jas_stream_t *out);
static void jas_icclut16_dump(jas_iccattrval_t *attrval, FILE *out);

static void jas_icclut8_destroy(jas_iccattrval_t *attrval);
static int jas_icclut8_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval);
static int jas_icclut8_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt);
static int jas_icclut8_getsize(jas_iccattrval_t *attrval);
static int jas_icclut8_output(jas_iccattrval_t *attrval, jas_stream_t *out);
static void jas_icclut8_dump(jas_iccattrval_t *attrval, FILE *out);

static int jas_iccputtime(jas_stream_t *out, jas_icctime_t *ctime);
static int jas_iccputxyz(jas_stream_t *out, jas_iccxyz_t *xyz);

static long jas_iccpowi(int x, int n);

static char *jas_iccsigtostr(int sig, char *buf);


jas_iccattrvalinfo_t jas_iccattrvalinfos[] = {
	{JAS_ICC_TYPE_CURV, {jas_icccurv_destroy, jas_icccurv_copy,
	  jas_icccurv_input, jas_icccurv_output, jas_icccurv_getsize,
	  jas_icccurv_dump}},
	{JAS_ICC_TYPE_XYZ, {0, 0, jas_iccxyz_input, jas_iccxyz_output,
	  jas_iccxyz_getsize, jas_iccxyz_dump}},
	{JAS_ICC_TYPE_TXTDESC, {jas_icctxtdesc_destroy,
	  jas_icctxtdesc_copy, jas_icctxtdesc_input, jas_icctxtdesc_output,
	  jas_icctxtdesc_getsize, jas_icctxtdesc_dump}},
	{JAS_ICC_TYPE_TXT, {jas_icctxt_destroy, jas_icctxt_copy,
	  jas_icctxt_input, jas_icctxt_output, jas_icctxt_getsize,
	  jas_icctxt_dump}},
	{JAS_ICC_TYPE_LUT8, {jas_icclut8_destroy, jas_icclut8_copy,
	  jas_icclut8_input, jas_icclut8_output, jas_icclut8_getsize,
	  jas_icclut8_dump}},
	{JAS_ICC_TYPE_LUT16, {jas_icclut16_destroy, jas_icclut16_copy,
	  jas_icclut16_input, jas_icclut16_output, jas_icclut16_getsize,
	  jas_icclut16_dump}},
	{0, {0, 0, 0, 0, 0, 0}}
};

typedef struct {
	jas_iccuint32_t tag;
	char *name;
} jas_icctaginfo_t;

/******************************************************************************\
* profile class
\******************************************************************************/

static jas_iccprof_t *jas_iccprof_create()
{
	jas_iccprof_t *prof;
	prof = 0;
	if (!(prof = jas_malloc(sizeof(jas_iccprof_t)))) {
		goto error;
	}
	if (!(prof->attrtab = jas_iccattrtab_create()))
		goto error;
	memset(&prof->hdr, 0, sizeof(jas_icchdr_t));
	prof->tagtab.numents = 0;
	prof->tagtab.ents = 0;
	return prof;
error:
	if (prof)
		jas_iccprof_destroy(prof);
	return 0;
}

jas_iccprof_t *jas_iccprof_copy(jas_iccprof_t *prof)
{
	jas_iccprof_t *newprof;
	newprof = 0;
	if (!(newprof = jas_iccprof_create()))
		goto error;
	newprof->hdr = prof->hdr;
	newprof->tagtab.numents = 0;
	newprof->tagtab.ents = 0;
	assert(newprof->attrtab);
	jas_iccattrtab_destroy(newprof->attrtab);
	if (!(newprof->attrtab = jas_iccattrtab_copy(prof->attrtab)))
		goto error;
	return newprof;
error:
	if (newprof)
		jas_iccprof_destroy(newprof);
	return 0;
}

void jas_iccprof_destroy(jas_iccprof_t *prof)
{
	if (prof->attrtab)
		jas_iccattrtab_destroy(prof->attrtab);
	if (prof->tagtab.ents)
		jas_free(prof->tagtab.ents);
	jas_free(prof);
}

void jas_iccprof_dump(jas_iccprof_t *prof, FILE *out)
{
	jas_iccattrtab_dump(prof->attrtab, out);
}

jas_iccprof_t *jas_iccprof_load(jas_stream_t *in)
{
	jas_iccprof_t *prof;
	int numtags;
	long curoff;
	long reloff;
	long prevoff;
	jas_iccsig_t type;
	jas_iccattrval_t *attrval;
	jas_iccattrval_t *prevattrval;
	jas_icctagtabent_t *tagtabent;
	jas_iccattrvalinfo_t *attrvalinfo;
	int i;
	int len;

	prof = 0;
	attrval = 0;

	if (!(prof = jas_iccprof_create())) {
		goto error;
	}

	if (jas_iccprof_readhdr(in, &prof->hdr)) {
		jas_eprintf("cannot get header\n");
		goto error;
	}
	if (jas_iccprof_gettagtab(in, &prof->tagtab)) {
		jas_eprintf("cannot get tab table\n");
		goto error;
	}
	jas_iccprof_sorttagtab(&prof->tagtab);

	numtags = prof->tagtab.numents;
	curoff = JAS_ICC_HDRLEN + 4 + 12 * numtags;
	prevoff = 0;
	prevattrval = 0;
	for (i = 0; i < numtags; ++i) {
		tagtabent = &prof->tagtab.ents[i];
		if (tagtabent->off == JAS_CAST(jas_iccuint32_t, prevoff)) {
			if (prevattrval) {
				if (!(attrval = jas_iccattrval_clone(prevattrval)))
					goto error;
				if (jas_iccprof_setattr(prof, tagtabent->tag, attrval))
					goto error;
				jas_iccattrval_destroy(attrval);
			} else {
#if 0
				jas_eprintf("warning: skipping unknown tag type\n");
#endif
			}
			continue;
		}
		reloff = tagtabent->off - curoff;
		if (reloff > 0) {
			if (jas_stream_gobble(in, reloff) != reloff)
				goto error;
			curoff += reloff;
		} else if (reloff < 0) {
			/* This should never happen since we read the tagged
			element data in a single pass. */
			abort();
		}
		prevoff = curoff;
		if (jas_iccgetuint32(in, &type)) {
			goto error;
		}
		if (jas_stream_gobble(in, 4) != 4) {
			goto error;
		}
		curoff += 8;
		if (!(attrvalinfo = jas_iccattrvalinfo_lookup(type))) {
#if 0
			jas_eprintf("warning: skipping unknown tag type\n");
#endif
			prevattrval = 0;
			continue;
		}
		if (!(attrval = jas_iccattrval_create(type))) {
			goto error;
		}
		len = tagtabent->len - 8;
		if ((*attrval->ops->input)(attrval, in, len)) {
			goto error;
		}
		curoff += len;
		if (jas_iccprof_setattr(prof, tagtabent->tag, attrval)) {
			goto error;
		}
		prevattrval = attrval; /* This is correct, but slimey. */
		jas_iccattrval_destroy(attrval);
		attrval = 0;
	}

	return prof;

error:
	if (prof)
		jas_iccprof_destroy(prof);
	if (attrval)
		jas_iccattrval_destroy(attrval);
	return 0;
}

int jas_iccprof_save(jas_iccprof_t *prof, jas_stream_t *out)
{
	long curoff;
	long reloff;
	long newoff;
	int i;
	int j;
	jas_icctagtabent_t *tagtabent;
	jas_icctagtabent_t *sharedtagtabent;
	jas_icctagtabent_t *tmptagtabent;
	jas_iccuint32_t attrname;
	jas_iccattrval_t *attrval;
	jas_icctagtab_t *tagtab;

	tagtab = &prof->tagtab;
	if (!(tagtab->ents = jas_malloc(prof->attrtab->numattrs *
	  sizeof(jas_icctagtabent_t))))
		goto error;
	tagtab->numents = prof->attrtab->numattrs;
	curoff = JAS_ICC_HDRLEN + 4 + 12 * tagtab->numents;
	for (i = 0; i < JAS_CAST(int, tagtab->numents); ++i) {
		tagtabent = &tagtab->ents[i];
		if (jas_iccattrtab_get(prof->attrtab, i, &attrname, &attrval))
			goto error;
		assert(attrval->ops->output);
		tagtabent->tag = attrname;
		tagtabent->data = &attrval->data;
		sharedtagtabent = 0;
		for (j = 0; j < i; ++j) {
			tmptagtabent = &tagtab->ents[j];
			if (tagtabent->data == tmptagtabent->data) {
				sharedtagtabent = tmptagtabent;
				break;
			}
		}
		if (sharedtagtabent) {
			tagtabent->off = sharedtagtabent->off;
			tagtabent->len = sharedtagtabent->len;
			tagtabent->first = sharedtagtabent;
		} else {
			tagtabent->off = curoff;
			tagtabent->len = (*attrval->ops->getsize)(attrval) + 8;
			tagtabent->first = 0;
			if (i < JAS_CAST(int, tagtab->numents - 1)) {
				curoff = jas_iccpadtomult(curoff + tagtabent->len, 4);
			} else {
				curoff += tagtabent->len;
			}
		}
		jas_iccattrval_destroy(attrval);
	}
	prof->hdr.size = curoff;
	if (jas_iccprof_writehdr(out, &prof->hdr))
		goto error;
	if (jas_iccprof_puttagtab(out, &prof->tagtab))
		goto error;
	curoff = JAS_ICC_HDRLEN + 4 + 12 * tagtab->numents;
	for (i = 0; i < JAS_CAST(int, tagtab->numents);) {
		tagtabent = &tagtab->ents[i];
		assert(curoff == JAS_CAST(long, tagtabent->off));
		if (jas_iccattrtab_get(prof->attrtab, i, &attrname, &attrval))
			goto error;
		if (jas_iccputuint32(out, attrval->type) || jas_stream_pad(out,
		  4, 0) != 4)
			goto error;
		if ((*attrval->ops->output)(attrval, out))
			goto error;
		jas_iccattrval_destroy(attrval);
		curoff += tagtabent->len;
		++i;
		while (i < JAS_CAST(int, tagtab->numents) &&
		  tagtab->ents[i].first)
			++i;
		newoff = (i < JAS_CAST(int, tagtab->numents)) ?
		  tagtab->ents[i].off : prof->hdr.size;
		reloff = newoff - curoff;
		assert(reloff >= 0);
		if (reloff > 0) {
			if (jas_stream_pad(out, reloff, 0) != reloff)
				goto error;
			curoff += reloff;
		}
	}	
	return 0;
error:
	/* XXX - need to free some resources here */
	return -1;
}

static int jas_iccprof_writehdr(jas_stream_t *out, jas_icchdr_t *hdr)
{
	if (jas_iccputuint32(out, hdr->size) ||
	  jas_iccputuint32(out, hdr->cmmtype) ||
	  jas_iccputuint32(out, hdr->version) ||
	  jas_iccputuint32(out, hdr->clas) ||
	  jas_iccputuint32(out, hdr->colorspc) ||
	  jas_iccputuint32(out, hdr->refcolorspc) ||
	  jas_iccputtime(out, &hdr->ctime) ||
	  jas_iccputuint32(out, hdr->magic) ||
	  jas_iccputuint32(out, hdr->platform) ||
	  jas_iccputuint32(out, hdr->flags) ||
	  jas_iccputuint32(out, hdr->maker) ||
	  jas_iccputuint32(out, hdr->model) ||
	  jas_iccputuint64(out, hdr->attr) ||
	  jas_iccputuint32(out, hdr->intent) ||
	  jas_iccputxyz(out, &hdr->illum) ||
	  jas_iccputuint32(out, hdr->creator) ||
	  jas_stream_pad(out, 44, 0) != 44)
		return -1;
	return 0;
}

static int jas_iccprof_puttagtab(jas_stream_t *out, jas_icctagtab_t *tagtab)
{
	int i;
	jas_icctagtabent_t *tagtabent;
	if (jas_iccputuint32(out, tagtab->numents))
		goto error;
	for (i = 0; i < JAS_CAST(int, tagtab->numents); ++i) {
		tagtabent = &tagtab->ents[i];
		if (jas_iccputuint32(out, tagtabent->tag) ||
		  jas_iccputuint32(out, tagtabent->off) ||
		  jas_iccputuint32(out, tagtabent->len))
			goto error;
	}
	return 0;
error:
	return -1;
}

static int jas_iccprof_readhdr(jas_stream_t *in, jas_icchdr_t *hdr)
{
	if (jas_iccgetuint32(in, &hdr->size) ||
	  jas_iccgetuint32(in, &hdr->cmmtype) ||
	  jas_iccgetuint32(in, &hdr->version) ||
	  jas_iccgetuint32(in, &hdr->clas) ||
	  jas_iccgetuint32(in, &hdr->colorspc) ||
	  jas_iccgetuint32(in, &hdr->refcolorspc) ||
	  jas_iccgettime(in, &hdr->ctime) ||
	  jas_iccgetuint32(in, &hdr->magic) ||
	  jas_iccgetuint32(in, &hdr->platform) ||
	  jas_iccgetuint32(in, &hdr->flags) ||
	  jas_iccgetuint32(in, &hdr->maker) ||
	  jas_iccgetuint32(in, &hdr->model) ||
	  jas_iccgetuint64(in, &hdr->attr) ||
	  jas_iccgetuint32(in, &hdr->intent) ||
	  jas_iccgetxyz(in, &hdr->illum) ||
	  jas_iccgetuint32(in, &hdr->creator) ||
	  jas_stream_gobble(in, 44) != 44)
		return -1;
	return 0;
}

static int jas_iccprof_gettagtab(jas_stream_t *in, jas_icctagtab_t *tagtab)
{
	int i;
	jas_icctagtabent_t *tagtabent;

	if (tagtab->ents) {
		jas_free(tagtab->ents);
		tagtab->ents = 0;
	}
	if (jas_iccgetuint32(in, &tagtab->numents))
		goto error;
	if (!(tagtab->ents = jas_malloc(tagtab->numents *
	  sizeof(jas_icctagtabent_t))))
		goto error;
	tagtabent = tagtab->ents;
	for (i = 0; i < JAS_CAST(long, tagtab->numents); ++i) {
		if (jas_iccgetuint32(in, &tagtabent->tag) ||
		jas_iccgetuint32(in, &tagtabent->off) ||
		jas_iccgetuint32(in, &tagtabent->len))
			goto error;
		++tagtabent;
	}
	return 0;
error:
	if (tagtab->ents) {
		jas_free(tagtab->ents);
		tagtab->ents = 0;
	}
	return -1;
}

jas_iccattrval_t *jas_iccprof_getattr(jas_iccprof_t *prof,
  jas_iccattrname_t name)
{
	int i;
	jas_iccattrval_t *attrval;
	if ((i = jas_iccattrtab_lookup(prof->attrtab, name)) < 0)
		goto error;
	if (!(attrval = jas_iccattrval_clone(prof->attrtab->attrs[i].val)))
		goto error;
	return attrval;
error:
	return 0;
}

int jas_iccprof_setattr(jas_iccprof_t *prof, jas_iccattrname_t name,
  jas_iccattrval_t *val)
{
	int i;
	if ((i = jas_iccattrtab_lookup(prof->attrtab, name)) >= 0) {
		if (val) {
			if (jas_iccattrtab_replace(prof->attrtab, i, name, val))
				goto error;
		} else {
			jas_iccattrtab_delete(prof->attrtab, i);
		}
	} else {
		if (val) {
			if (jas_iccattrtab_add(prof->attrtab, -1, name, val))
				goto error;
		} else {
			/* NOP */
		}
	}
	return 0;
error:
	return -1;
}

int jas_iccprof_gethdr(jas_iccprof_t *prof, jas_icchdr_t *hdr)
{
	*hdr = prof->hdr;
	return 0;
}

int jas_iccprof_sethdr(jas_iccprof_t *prof, jas_icchdr_t *hdr)
{
	prof->hdr = *hdr;
	return 0;
}

static void jas_iccprof_sorttagtab(jas_icctagtab_t *tagtab)
{
	qsort(tagtab->ents, tagtab->numents, sizeof(jas_icctagtabent_t),
	  jas_icctagtabent_cmp);
}

static int jas_icctagtabent_cmp(const void *src, const void *dst)
{
	jas_icctagtabent_t *srctagtabent = JAS_CAST(jas_icctagtabent_t *, src);
	jas_icctagtabent_t *dsttagtabent = JAS_CAST(jas_icctagtabent_t *, dst);
	if (srctagtabent->off > dsttagtabent->off) {
		return 1;
	} else if (srctagtabent->off < dsttagtabent->off) {
		return -1;
	}
	return 0;
}

static jas_iccattrvalinfo_t *jas_iccattrvalinfo_lookup(jas_iccsig_t type)
{
	jas_iccattrvalinfo_t *info;
	info = jas_iccattrvalinfos;
	for (info = jas_iccattrvalinfos; info->type; ++info) {
		if (info->type == type) {
			return info;
		}
	}
	return 0;
}

static int jas_iccgettime(jas_stream_t *in, jas_icctime_t *time)
{
	if (jas_iccgetuint16(in, &time->year) ||
	  jas_iccgetuint16(in, &time->month) ||
	  jas_iccgetuint16(in, &time->day) ||
	  jas_iccgetuint16(in, &time->hour) ||
	  jas_iccgetuint16(in, &time->min) ||
	  jas_iccgetuint16(in, &time->sec)) {
		return -1;
	}
	return 0;
}

static int jas_iccgetxyz(jas_stream_t *in, jas_iccxyz_t *xyz)
{
	if (jas_iccgetsint32(in, &xyz->x) ||
	  jas_iccgetsint32(in, &xyz->y) ||
	  jas_iccgetsint32(in, &xyz->z)) {
		return -1;
	}
	return 0;
}

static int jas_iccputtime(jas_stream_t *out, jas_icctime_t *time)
{
	jas_iccputuint16(out, time->year);
	jas_iccputuint16(out, time->month);
	jas_iccputuint16(out, time->day);
	jas_iccputuint16(out, time->hour);
	jas_iccputuint16(out, time->min);
	jas_iccputuint16(out, time->sec);
	return 0;
}

static int jas_iccputxyz(jas_stream_t *out, jas_iccxyz_t *xyz)
{
	jas_iccputuint32(out, xyz->x);
	jas_iccputuint32(out, xyz->y);
	jas_iccputuint32(out, xyz->z);
	return 0;
}

/******************************************************************************\
* attribute table class
\******************************************************************************/

static jas_iccattrtab_t *jas_iccattrtab_create()
{
	jas_iccattrtab_t *tab;
	tab = 0;
	if (!(tab = jas_malloc(sizeof(jas_iccattrtab_t))))
		goto error;
	tab->maxattrs = 0;
	tab->numattrs = 0;
	tab->attrs = 0;
	if (jas_iccattrtab_resize(tab, 32))
		goto error;
	return tab;
error:
	if (tab)
		jas_iccattrtab_destroy(tab);
	return 0;
}

static jas_iccattrtab_t *jas_iccattrtab_copy(jas_iccattrtab_t *attrtab)
{
	jas_iccattrtab_t *newattrtab;
	int i;
	if (!(newattrtab = jas_iccattrtab_create()))
		goto error;
	for (i = 0; i < attrtab->numattrs; ++i) {
		if (jas_iccattrtab_add(newattrtab, i, attrtab->attrs[i].name,
		  attrtab->attrs[i].val))
			goto error;
	}
	return newattrtab;
error:
	return 0;
}

static void jas_iccattrtab_destroy(jas_iccattrtab_t *tab)
{
	if (tab->attrs) {
		while (tab->numattrs > 0) {
			jas_iccattrtab_delete(tab, 0);
		}
		jas_free(tab->attrs);
	}
	jas_free(tab);
}

void jas_iccattrtab_dump(jas_iccattrtab_t *attrtab, FILE *out)
{
	int i;
	jas_iccattr_t *attr;
	jas_iccattrval_t *attrval;
	jas_iccattrvalinfo_t *info;
	char buf[16];
	fprintf(out, "numattrs=%d\n", attrtab->numattrs);
	fprintf(out, "---\n");
	for (i = 0; i < attrtab->numattrs; ++i) {
		attr = &attrtab->attrs[i];
		attrval = attr->val;
		info = jas_iccattrvalinfo_lookup(attrval->type);
		if (!info) abort();
		fprintf(out, "attrno=%d; attrname=\"%s\"(0x%08x); attrtype=\"%s\"(0x%08x)\n",
		  i,
		  jas_iccsigtostr(attr->name, &buf[0]),
		  (unsigned)attr->name,
		  jas_iccsigtostr(attrval->type, &buf[8]),
		  (unsigned)attrval->type
		  );
		jas_iccattrval_dump(attrval, out);
		fprintf(out, "---\n");
	}
}

static int jas_iccattrtab_resize(jas_iccattrtab_t *tab, int maxents)
{
	jas_iccattr_t *newattrs;
	assert(maxents >= tab->numattrs);
	newattrs = tab->attrs ? jas_realloc(tab->attrs, maxents *
	  sizeof(jas_iccattr_t)) : jas_malloc(maxents * sizeof(jas_iccattr_t));
	if (!newattrs)
		return -1;
	tab->attrs = newattrs;
	tab->maxattrs = maxents;
	return 0;
}

static int jas_iccattrtab_add(jas_iccattrtab_t *attrtab, int i,
  jas_iccuint32_t name, jas_iccattrval_t *val)
{
	int n;
	jas_iccattr_t *attr;
	jas_iccattrval_t *tmpattrval;
	tmpattrval = 0;
	if (i < 0) {
		i = attrtab->numattrs;
	}
	assert(i >= 0 && i <= attrtab->numattrs);
	if (attrtab->numattrs >= attrtab->maxattrs) {
		if (jas_iccattrtab_resize(attrtab, attrtab->numattrs + 32)) {
			goto error;
		}
	}
	if (!(tmpattrval = jas_iccattrval_clone(val)))
		goto error;
	n = attrtab->numattrs - i;
	if (n > 0)
		memmove(&attrtab->attrs[i + 1], &attrtab->attrs[i],
		  n * sizeof(jas_iccattr_t));
	attr = &attrtab->attrs[i];
	attr->name = name;
	attr->val = tmpattrval;
	++attrtab->numattrs;
	return 0;
error:
	if (tmpattrval)
		jas_iccattrval_destroy(tmpattrval);
	return -1;
}

static int jas_iccattrtab_replace(jas_iccattrtab_t *attrtab, int i,
  jas_iccuint32_t name, jas_iccattrval_t *val)
{
	jas_iccattrval_t *newval;
	jas_iccattr_t *attr;
	if (!(newval = jas_iccattrval_clone(val)))
		goto error;
	attr = &attrtab->attrs[i];
	jas_iccattrval_destroy(attr->val);
	attr->name = name;
	attr->val = newval;
	return 0;
error:
	return -1;
}

static void jas_iccattrtab_delete(jas_iccattrtab_t *attrtab, int i)
{
	int n;
	jas_iccattrval_destroy(attrtab->attrs[i].val);
	if ((n = attrtab->numattrs - i - 1) > 0)
		memmove(&attrtab->attrs[i], &attrtab->attrs[i + 1],
		  n * sizeof(jas_iccattr_t));
	--attrtab->numattrs;
}

static int jas_iccattrtab_get(jas_iccattrtab_t *attrtab, int i,
  jas_iccattrname_t *name, jas_iccattrval_t **val)
{
	jas_iccattr_t *attr;
	if (i < 0 || i >= attrtab->numattrs)
		goto error;
	attr = &attrtab->attrs[i];
	*name = attr->name;
	if (!(*val = jas_iccattrval_clone(attr->val)))
		goto error;
	return 0;
error:
	return -1;
}

static int jas_iccattrtab_lookup(jas_iccattrtab_t *attrtab,
  jas_iccuint32_t name)
{
	int i;
	jas_iccattr_t *attr;
	for (i = 0; i < attrtab->numattrs; ++i) {
		attr = &attrtab->attrs[i];
		if (attr->name == name)
			return i;
	}
	return -1;
}

/******************************************************************************\
* attribute value class
\******************************************************************************/

jas_iccattrval_t *jas_iccattrval_create(jas_iccuint32_t type)
{
	jas_iccattrval_t *attrval;
	jas_iccattrvalinfo_t *info;

	if (!(info = jas_iccattrvalinfo_lookup(type)))
		goto error;
	if (!(attrval = jas_iccattrval_create0()))
		goto error;
	attrval->ops = &info->ops;
	attrval->type = type;
	++attrval->refcnt;
	memset(&attrval->data, 0, sizeof(attrval->data));
	return attrval;
error:
	return 0;
}

jas_iccattrval_t *jas_iccattrval_clone(jas_iccattrval_t *attrval)
{
	++attrval->refcnt;
	return attrval;
}

void jas_iccattrval_destroy(jas_iccattrval_t *attrval)
{
#if 0
jas_eprintf("refcnt=%d\n", attrval->refcnt);
#endif
	if (--attrval->refcnt <= 0) {
		if (attrval->ops->destroy)
			(*attrval->ops->destroy)(attrval);
		jas_free(attrval);
	}
}

void jas_iccattrval_dump(jas_iccattrval_t *attrval, FILE *out)
{
	char buf[8];
	jas_iccsigtostr(attrval->type, buf);
	fprintf(out, "refcnt = %d; type = 0x%08x %s\n", attrval->refcnt,
	  (unsigned)attrval->type, jas_iccsigtostr(attrval->type, &buf[0]));
	if (attrval->ops->dump) {
		(*attrval->ops->dump)(attrval, out);
	}
}

int jas_iccattrval_allowmodify(jas_iccattrval_t **attrvalx)
{
	jas_iccattrval_t *newattrval;
	jas_iccattrval_t *attrval = *attrvalx;
	newattrval = 0;
	if (attrval->refcnt > 1) {
		if (!(newattrval = jas_iccattrval_create0()))
			goto error;
		newattrval->ops = attrval->ops;
		newattrval->type = attrval->type;
		++newattrval->refcnt;
		if (newattrval->ops->copy) {
			if ((*newattrval->ops->copy)(newattrval, attrval))
				goto error;
		} else {
			memcpy(&newattrval->data, &attrval->data,
			  sizeof(newattrval->data));
		}
		*attrvalx = newattrval;
	}
	return 0;
error:
	if (newattrval) {
		jas_free(newattrval);
	}
	return -1;
}

static jas_iccattrval_t *jas_iccattrval_create0()
{
	jas_iccattrval_t *attrval;
	if (!(attrval = jas_malloc(sizeof(jas_iccattrval_t))))
		return 0;
	memset(attrval, 0, sizeof(jas_iccattrval_t));
	attrval->refcnt = 0;
	attrval->ops = 0;
	attrval->type = 0;
	return attrval;
}

/******************************************************************************\
*
\******************************************************************************/

static int jas_iccxyz_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int len)
{
	if (len != 4 * 3) abort();
	return jas_iccgetxyz(in, &attrval->data.xyz);
}

static int jas_iccxyz_output(jas_iccattrval_t *attrval, jas_stream_t *out)
{
	jas_iccxyz_t *xyz = &attrval->data.xyz;
	if (jas_iccputuint32(out, xyz->x) ||
	  jas_iccputuint32(out, xyz->y) ||
	  jas_iccputuint32(out, xyz->z))
		return -1;
	return 0;
}

static int jas_iccxyz_getsize(jas_iccattrval_t *attrval)
{
	/* Avoid compiler warnings about unused parameters. */
	attrval = 0;

	return 12;
}

static void jas_iccxyz_dump(jas_iccattrval_t *attrval, FILE *out)
{
	jas_iccxyz_t *xyz = &attrval->data.xyz;
	fprintf(out, "(%f, %f, %f)\n", xyz->x / 65536.0, xyz->y / 65536.0, xyz->z / 65536.0);
}

/******************************************************************************\
* attribute table class
\******************************************************************************/

static void jas_icccurv_destroy(jas_iccattrval_t *attrval)
{
	jas_icccurv_t *curv = &attrval->data.curv;
	if (curv->ents)
		jas_free(curv->ents);
}

static int jas_icccurv_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval)
{
	/* Avoid compiler warnings about unused parameters. */
	attrval = 0;
	othattrval = 0;

	/* Not yet implemented. */
	abort();
	return -1;
}

static int jas_icccurv_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt)
{
	jas_icccurv_t *curv = &attrval->data.curv;
	unsigned int i;

	curv->numents = 0;
	curv->ents = 0;

	if (jas_iccgetuint32(in, &curv->numents))
		goto error;
	if (!(curv->ents = jas_malloc(curv->numents * sizeof(jas_iccuint16_t))))
		goto error;
	for (i = 0; i < curv->numents; ++i) {
		if (jas_iccgetuint16(in, &curv->ents[i]))
			goto error;
	}

	if (JAS_CAST(int, 4 + 2 * curv->numents) != cnt)
		goto error;
	return 0;

error:
	jas_icccurv_destroy(attrval);
	return -1;
}

static int jas_icccurv_getsize(jas_iccattrval_t *attrval)
{
	jas_icccurv_t *curv = &attrval->data.curv;
	return 4 + 2 * curv->numents;
}

static int jas_icccurv_output(jas_iccattrval_t *attrval, jas_stream_t *out)
{
	jas_icccurv_t *curv = &attrval->data.curv;
	unsigned int i;

	if (jas_iccputuint32(out, curv->numents))
		goto error;
	for (i = 0; i < curv->numents; ++i) {
		if (jas_iccputuint16(out, curv->ents[i]))
			goto error;
	}
	return 0;
error:
	return -1;
}

static void jas_icccurv_dump(jas_iccattrval_t *attrval, FILE *out)
{
	int i;
	jas_icccurv_t *curv = &attrval->data.curv;
	fprintf(out, "number of entires = %d\n", (int)curv->numents);
	if (curv->numents == 1) {
		fprintf(out, "gamma = %f\n", curv->ents[0] / 256.0);
	} else {
		for (i = 0; i < JAS_CAST(int, curv->numents); ++i) {
			if (i < 3 || i >= JAS_CAST(int, curv->numents) - 3) {
				fprintf(out, "entry[%d] = %f\n", i, curv->ents[i] / 65535.0);
			}
		}
	}
}

/******************************************************************************\
*
\******************************************************************************/

static void jas_icctxtdesc_destroy(jas_iccattrval_t *attrval)
{
	jas_icctxtdesc_t *txtdesc = &attrval->data.txtdesc;
	if (txtdesc->ascdata)
		jas_free(txtdesc->ascdata);
	if (txtdesc->ucdata)
		jas_free(txtdesc->ucdata);
}

static int jas_icctxtdesc_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval)
{
	jas_icctxtdesc_t *txtdesc = &attrval->data.txtdesc;

	/* Avoid compiler warnings about unused parameters. */
	attrval = 0;
	othattrval = 0;
	txtdesc = 0;

	/* Not yet implemented. */
	abort();
	return -1;
}

static int jas_icctxtdesc_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt)
{
	int n;
	int c;
	jas_icctxtdesc_t *txtdesc = &attrval->data.txtdesc;
	txtdesc->ascdata = 0;
	txtdesc->ucdata = 0;
	if (jas_iccgetuint32(in, &txtdesc->asclen))
		goto error;
	if (!(txtdesc->ascdata = jas_malloc(txtdesc->asclen)))
		goto error;
	if (jas_stream_read(in, txtdesc->ascdata, txtdesc->asclen) !=
	  JAS_CAST(int, txtdesc->asclen))
		goto error;
	txtdesc->ascdata[txtdesc->asclen - 1] = '\0';
	if (jas_iccgetuint32(in, &txtdesc->uclangcode) ||
	  jas_iccgetuint32(in, &txtdesc->uclen))
		goto error;
	if (!(txtdesc->ucdata = jas_malloc(txtdesc->uclen * 2)))
		goto error;
	if (jas_stream_read(in, txtdesc->ucdata, txtdesc->uclen * 2) !=
	  JAS_CAST(int, txtdesc->uclen * 2))
		goto error;
	if (jas_iccgetuint16(in, &txtdesc->sccode))
		goto error;
	if ((c = jas_stream_getc(in)) == EOF)
		goto error;
	txtdesc->maclen = c;
	if (jas_stream_read(in, txtdesc->macdata, 67) != 67)
		goto error;
	txtdesc->asclen = strlen(txtdesc->ascdata) + 1;
#define WORKAROUND_BAD_PROFILES
#ifdef WORKAROUND_BAD_PROFILES
	n = txtdesc->asclen + txtdesc->uclen * 2 + 15 + 67;
	if (n > cnt) {
		return -1;
	}
	if (n < cnt) {
		if (jas_stream_gobble(in, cnt - n) != cnt - n)
			goto error;
	}
#else
	if (txtdesc->asclen + txtdesc->uclen * 2 + 15 + 67 != cnt)
		return -1;
#endif
	return 0;
error:
	jas_icctxtdesc_destroy(attrval);
	return -1;
}

static int jas_icctxtdesc_getsize(jas_iccattrval_t *attrval)
{
	jas_icctxtdesc_t *txtdesc = &attrval->data.txtdesc;
	return strlen(txtdesc->ascdata) + 1 + txtdesc->uclen * 2 + 15 + 67;
}

static int jas_icctxtdesc_output(jas_iccattrval_t *attrval, jas_stream_t *out)
{
	jas_icctxtdesc_t *txtdesc = &attrval->data.txtdesc;
	if (jas_iccputuint32(out, txtdesc->asclen) ||
	  jas_stream_puts(out, txtdesc->ascdata) ||
	  jas_stream_putc(out, 0) == EOF ||
	  jas_iccputuint32(out, txtdesc->uclangcode) ||
	  jas_iccputuint32(out, txtdesc->uclen) ||
	  jas_stream_write(out, txtdesc->ucdata, txtdesc->uclen * 2) != JAS_CAST(int, txtdesc->uclen * 2) ||
	  jas_iccputuint16(out, txtdesc->sccode) ||
	  jas_stream_putc(out, txtdesc->maclen) == EOF)
		goto error;
	if (txtdesc->maclen > 0) {
		if (jas_stream_write(out, txtdesc->macdata, 67) != 67)
			goto error;
	} else {
		if (jas_stream_pad(out, 67, 0) != 67)
			goto error;
	}
	return 0;
error:
	return -1;
}

static void jas_icctxtdesc_dump(jas_iccattrval_t *attrval, FILE *out)
{
	jas_icctxtdesc_t *txtdesc = &attrval->data.txtdesc;
	fprintf(out, "ascii = \"%s\"\n", txtdesc->ascdata);
	fprintf(out, "uclangcode = %d; uclen = %d\n", (int)txtdesc->uclangcode,
	  (int)txtdesc->uclen);
	fprintf(out, "sccode = %d\n", (int)txtdesc->sccode);
	fprintf(out, "maclen = %d\n", txtdesc->maclen);
}

/******************************************************************************\
*
\******************************************************************************/

static void jas_icctxt_destroy(jas_iccattrval_t *attrval)
{
	jas_icctxt_t *txt = &attrval->data.txt;
	if (txt->string)
		jas_free(txt->string);
}

static int jas_icctxt_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval)
{
	jas_icctxt_t *txt = &attrval->data.txt;
	jas_icctxt_t *othtxt = &othattrval->data.txt;
	if (!(txt->string = jas_strdup(othtxt->string)))
		return -1;
	return 0;
}

static int jas_icctxt_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt)
{
	jas_icctxt_t *txt = &attrval->data.txt;
	txt->string = 0;
	if (!(txt->string = jas_malloc(cnt)))
		goto error;
	if (jas_stream_read(in, txt->string, cnt) != cnt)
		goto error;
	txt->string[cnt - 1] = '\0';
	if (JAS_CAST(int, strlen(txt->string)) + 1 != cnt)
		goto error;
	return 0;
error:
	if (txt->string)
		jas_free(txt->string);
	return -1;
}

static int jas_icctxt_getsize(jas_iccattrval_t *attrval)
{
	jas_icctxt_t *txt = &attrval->data.txt;
	return strlen(txt->string) + 1;
}

static int jas_icctxt_output(jas_iccattrval_t *attrval, jas_stream_t *out)
{
	jas_icctxt_t *txt = &attrval->data.txt;
	if (jas_stream_puts(out, txt->string) ||
	  jas_stream_putc(out, 0) == EOF)
		return -1;
	return 0;
}

static void jas_icctxt_dump(jas_iccattrval_t *attrval, FILE *out)
{
	jas_icctxt_t *txt = &attrval->data.txt;
	fprintf(out, "string = \"%s\"\n", txt->string);
}

/******************************************************************************\
*
\******************************************************************************/

static void jas_icclut8_destroy(jas_iccattrval_t *attrval)
{
	jas_icclut8_t *lut8 = &attrval->data.lut8;
	if (lut8->clut)
		jas_free(lut8->clut);
	if (lut8->intabs)
		jas_free(lut8->intabs);
	if (lut8->intabsbuf)
		jas_free(lut8->intabsbuf);
	if (lut8->outtabs)
		jas_free(lut8->outtabs);
	if (lut8->outtabsbuf)
		jas_free(lut8->outtabsbuf);
}

static int jas_icclut8_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval)
{
	jas_icclut8_t *lut8 = &attrval->data.lut8;
	/* Avoid compiler warnings about unused parameters. */
	attrval = 0;
	othattrval = 0;
	lut8 = 0;
	abort();
	return -1;
}

static int jas_icclut8_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt)
{
	int i;
	int j;
	int clutsize;
	jas_icclut8_t *lut8 = &attrval->data.lut8;
	lut8->clut = 0;
	lut8->intabs = 0;
	lut8->intabsbuf = 0;
	lut8->outtabs = 0;
	lut8->outtabsbuf = 0;
	if (jas_iccgetuint8(in, &lut8->numinchans) ||
	  jas_iccgetuint8(in, &lut8->numoutchans) ||
	  jas_iccgetuint8(in, &lut8->clutlen) ||
	  jas_stream_getc(in) == EOF)
		goto error;
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 3; ++j) {
			if (jas_iccgetsint32(in, &lut8->e[i][j]))
				goto error;
		}
	}
	if (jas_iccgetuint16(in, &lut8->numintabents) ||
	  jas_iccgetuint16(in, &lut8->numouttabents))
		goto error;
	clutsize = jas_iccpowi(lut8->clutlen, lut8->numinchans) * lut8->numoutchans;
	if (!(lut8->clut = jas_malloc(clutsize * sizeof(jas_iccuint8_t))) ||
	  !(lut8->intabsbuf = jas_malloc(lut8->numinchans *
	  lut8->numintabents * sizeof(jas_iccuint8_t))) ||
	  !(lut8->intabs = jas_malloc(lut8->numinchans *
	  sizeof(jas_iccuint8_t *))))
		goto error;
	for (i = 0; i < lut8->numinchans; ++i)
		lut8->intabs[i] = &lut8->intabsbuf[i * lut8->numintabents];
	if (!(lut8->outtabsbuf = jas_malloc(lut8->numoutchans *
	  lut8->numouttabents * sizeof(jas_iccuint8_t))) ||
	  !(lut8->outtabs = jas_malloc(lut8->numoutchans *
	  sizeof(jas_iccuint8_t *))))
		goto error;
	for (i = 0; i < lut8->numoutchans; ++i)
		lut8->outtabs[i] = &lut8->outtabsbuf[i * lut8->numouttabents];
	for (i = 0; i < lut8->numinchans; ++i) {
		for (j = 0; j < JAS_CAST(int, lut8->numintabents); ++j) {
			if (jas_iccgetuint8(in, &lut8->intabs[i][j]))
				goto error;
		}
	}
	for (i = 0; i < lut8->numoutchans; ++i) {
		for (j = 0; j < JAS_CAST(int, lut8->numouttabents); ++j) {
			if (jas_iccgetuint8(in, &lut8->outtabs[i][j]))
				goto error;
		}
	}
	for (i = 0; i < clutsize; ++i) {
		if (jas_iccgetuint8(in, &lut8->clut[i]))
			goto error;
	}
	if (JAS_CAST(int, 44 + lut8->numinchans * lut8->numintabents +
	  lut8->numoutchans * lut8->numouttabents +
	  jas_iccpowi(lut8->clutlen, lut8->numinchans) * lut8->numoutchans) !=
	  cnt)
		goto error;
	return 0;
error:
	jas_icclut8_destroy(attrval);
	return -1;
}

static int jas_icclut8_getsize(jas_iccattrval_t *attrval)
{
	jas_icclut8_t *lut8 = &attrval->data.lut8;
	return 44 + lut8->numinchans * lut8->numintabents +
	  lut8->numoutchans * lut8->numouttabents +
	  jas_iccpowi(lut8->clutlen, lut8->numinchans) * lut8->numoutchans;
}

static int jas_icclut8_output(jas_iccattrval_t *attrval, jas_stream_t *out)
{
	jas_icclut8_t *lut8 = &attrval->data.lut8;
	int i;
	int j;
	int n;
	lut8->clut = 0;
	lut8->intabs = 0;
	lut8->intabsbuf = 0;
	lut8->outtabs = 0;
	lut8->outtabsbuf = 0;
	if (jas_stream_putc(out, lut8->numinchans) == EOF ||
	  jas_stream_putc(out, lut8->numoutchans) == EOF ||
	  jas_stream_putc(out, lut8->clutlen) == EOF ||
	  jas_stream_putc(out, 0) == EOF)
		goto error;
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 3; ++j) {
			if (jas_iccputsint32(out, lut8->e[i][j]))
				goto error;
		}
	}
	if (jas_iccputuint16(out, lut8->numintabents) ||
	  jas_iccputuint16(out, lut8->numouttabents))
		goto error;
	n = lut8->numinchans * lut8->numintabents;
	for (i = 0; i < n; ++i) {
		if (jas_iccputuint8(out, lut8->intabsbuf[i]))
			goto error;
	}
	n = lut8->numoutchans * lut8->numouttabents;
	for (i = 0; i < n; ++i) {
		if (jas_iccputuint8(out, lut8->outtabsbuf[i]))
			goto error;
	}
	n = jas_iccpowi(lut8->clutlen, lut8->numinchans) * lut8->numoutchans;
	for (i = 0; i < n; ++i) {
		if (jas_iccputuint8(out, lut8->clut[i]))
			goto error;
	}
	return 0;
error:
	return -1;
}

static void jas_icclut8_dump(jas_iccattrval_t *attrval, FILE *out)
{
	jas_icclut8_t *lut8 = &attrval->data.lut8;
	int i;
	int j;
	fprintf(out, "numinchans=%d, numoutchans=%d, clutlen=%d\n",
	  lut8->numinchans, lut8->numoutchans, lut8->clutlen);
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 3; ++j) {
			fprintf(out, "e[%d][%d]=%f ", i, j, lut8->e[i][j] / 65536.0);
		}
		fprintf(out, "\n");
	}
	fprintf(out, "numintabents=%d, numouttabents=%d\n",
	  (int)lut8->numintabents, (int)lut8->numouttabents);
}

/******************************************************************************\
*
\******************************************************************************/

static void jas_icclut16_destroy(jas_iccattrval_t *attrval)
{
	jas_icclut16_t *lut16 = &attrval->data.lut16;
	if (lut16->clut)
		jas_free(lut16->clut);
	if (lut16->intabs)
		jas_free(lut16->intabs);
	if (lut16->intabsbuf)
		jas_free(lut16->intabsbuf);
	if (lut16->outtabs)
		jas_free(lut16->outtabs);
	if (lut16->outtabsbuf)
		jas_free(lut16->outtabsbuf);
}

static int jas_icclut16_copy(jas_iccattrval_t *attrval,
  jas_iccattrval_t *othattrval)
{
	/* Avoid compiler warnings about unused parameters. */
	attrval = 0;
	othattrval = 0;
	/* Not yet implemented. */
	abort();
	return -1;
}

static int jas_icclut16_input(jas_iccattrval_t *attrval, jas_stream_t *in,
  int cnt)
{
	int i;
	int j;
	int clutsize;
	jas_icclut16_t *lut16 = &attrval->data.lut16;
	lut16->clut = 0;
	lut16->intabs = 0;
	lut16->intabsbuf = 0;
	lut16->outtabs = 0;
	lut16->outtabsbuf = 0;
	if (jas_iccgetuint8(in, &lut16->numinchans) ||
	  jas_iccgetuint8(in, &lut16->numoutchans) ||
	  jas_iccgetuint8(in, &lut16->clutlen) ||
	  jas_stream_getc(in) == EOF)
		goto error;
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 3; ++j) {
			if (jas_iccgetsint32(in, &lut16->e[i][j]))
				goto error;
		}
	}
	if (jas_iccgetuint16(in, &lut16->numintabents) ||
	  jas_iccgetuint16(in, &lut16->numouttabents))
		goto error;
	clutsize = jas_iccpowi(lut16->clutlen, lut16->numinchans) * lut16->numoutchans;
	if (!(lut16->clut = jas_malloc(clutsize * sizeof(jas_iccuint16_t))) ||
	  !(lut16->intabsbuf = jas_malloc(lut16->numinchans *
	  lut16->numintabents * sizeof(jas_iccuint16_t))) ||
	  !(lut16->intabs = jas_malloc(lut16->numinchans *
	  sizeof(jas_iccuint16_t *))))
		goto error;
	for (i = 0; i < lut16->numinchans; ++i)
		lut16->intabs[i] = &lut16->intabsbuf[i * lut16->numintabents];
	if (!(lut16->outtabsbuf = jas_malloc(lut16->numoutchans *
	  lut16->numouttabents * sizeof(jas_iccuint16_t))) ||
	  !(lut16->outtabs = jas_malloc(lut16->numoutchans *
	  sizeof(jas_iccuint16_t *))))
		goto error;
	for (i = 0; i < lut16->numoutchans; ++i)
		lut16->outtabs[i] = &lut16->outtabsbuf[i * lut16->numouttabents];
	for (i = 0; i < lut16->numinchans; ++i) {
		for (j = 0; j < JAS_CAST(int, lut16->numintabents); ++j) {
			if (jas_iccgetuint16(in, &lut16->intabs[i][j]))
				goto error;
		}
	}
	for (i = 0; i < lut16->numoutchans; ++i) {
		for (j = 0; j < JAS_CAST(int, lut16->numouttabents); ++j) {
			if (jas_iccgetuint16(in, &lut16->outtabs[i][j]))
				goto error;
		}
	}
	for (i = 0; i < clutsize; ++i) {
		if (jas_iccgetuint16(in, &lut16->clut[i]))
			goto error;
	}
	if (JAS_CAST(int, 44 + 2 * (lut16->numinchans * lut16->numintabents +
          lut16->numoutchans * lut16->numouttabents +
          jas_iccpowi(lut16->clutlen, lut16->numinchans) *
	  lut16->numoutchans)) != cnt)
		goto error;
	return 0;
error:
	jas_icclut16_destroy(attrval);
	return -1;
}

static int jas_icclut16_getsize(jas_iccattrval_t *attrval)
{
	jas_icclut16_t *lut16 = &attrval->data.lut16;
	return 44 + 2 * (lut16->numinchans * lut16->numintabents +
	  lut16->numoutchans * lut16->numouttabents +
	  jas_iccpowi(lut16->clutlen, lut16->numinchans) * lut16->numoutchans);
}

static int jas_icclut16_output(jas_iccattrval_t *attrval, jas_stream_t *out)
{
	jas_icclut16_t *lut16 = &attrval->data.lut16;
	int i;
	int j;
	int n;
	if (jas_stream_putc(out, lut16->numinchans) == EOF ||
	  jas_stream_putc(out, lut16->numoutchans) == EOF ||
	  jas_stream_putc(out, lut16->clutlen) == EOF ||
	  jas_stream_putc(out, 0) == EOF)
		goto error;
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 3; ++j) {
			if (jas_iccputsint32(out, lut16->e[i][j]))
				goto error;
		}
	}
	if (jas_iccputuint16(out, lut16->numintabents) ||
	  jas_iccputuint16(out, lut16->numouttabents))
		goto error;
	n = lut16->numinchans * lut16->numintabents;
	for (i = 0; i < n; ++i) {
		if (jas_iccputuint16(out, lut16->intabsbuf[i]))
			goto error;
	}
	n = lut16->numoutchans * lut16->numouttabents;
	for (i = 0; i < n; ++i) {
		if (jas_iccputuint16(out, lut16->outtabsbuf[i]))
			goto error;
	}
	n = jas_iccpowi(lut16->clutlen, lut16->numinchans) * lut16->numoutchans;
	for (i = 0; i < n; ++i) {
		if (jas_iccputuint16(out, lut16->clut[i]))
			goto error;
	}
	return 0;
error:
	return -1;
}

static void jas_icclut16_dump(jas_iccattrval_t *attrval, FILE *out)
{
	jas_icclut16_t *lut16 = &attrval->data.lut16;
	int i;
	int j;
	fprintf(out, "numinchans=%d, numoutchans=%d, clutlen=%d\n",
	  lut16->numinchans, lut16->numoutchans, lut16->clutlen);
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 3; ++j) {
			fprintf(out, "e[%d][%d]=%f ", i, j, lut16->e[i][j] / 65536.0);
		}
		fprintf(out, "\n");
	}
	fprintf(out, "numintabents=%d, numouttabents=%d\n",
	  (int)lut16->numintabents, (int)lut16->numouttabents);
}

/******************************************************************************\
*
\******************************************************************************/

static int jas_iccgetuint(jas_stream_t *in, int n, ulonglong *val)
{
	int i;
	int c;
	ulonglong v;
	v = 0;
	for (i = n; i > 0; --i) {
		if ((c = jas_stream_getc(in)) == EOF)
			return -1;
		v = (v << 8) | c;
	}
	*val = v;
	return 0;
}

static int jas_iccgetuint8(jas_stream_t *in, jas_iccuint8_t *val)
{
	int c;
	if ((c = jas_stream_getc(in)) == EOF)
		return -1;
	*val = c;
	return 0;
}

static int jas_iccgetuint16(jas_stream_t *in, jas_iccuint16_t *val)
{
	ulonglong tmp;
	if (jas_iccgetuint(in, 2, &tmp))
		return -1;
	*val = tmp;
	return 0;
}

static int jas_iccgetsint32(jas_stream_t *in, jas_iccsint32_t *val)
{
	ulonglong tmp;
	if (jas_iccgetuint(in, 4, &tmp))
		return -1;
	*val = (tmp & 0x80000000) ? (-JAS_CAST(longlong, (((~tmp) &
	  0x7fffffff) + 1))) : JAS_CAST(longlong, tmp);
	return 0;
}

static int jas_iccgetuint32(jas_stream_t *in, jas_iccuint32_t *val)
{
	ulonglong tmp;
	if (jas_iccgetuint(in, 4, &tmp))
		return -1;
	*val = tmp;
	return 0;
}

static int jas_iccgetuint64(jas_stream_t *in, jas_iccuint64_t *val)
{
	ulonglong tmp;
	if (jas_iccgetuint(in, 8, &tmp))
		return -1;
	*val = tmp;
	return 0;
}

static int jas_iccputuint(jas_stream_t *out, int n, ulonglong val)
{
	int i;
	int c;
	for (i = n; i > 0; --i) {
		c = (val >> (8 * (i - 1))) & 0xff;
		if (jas_stream_putc(out, c) == EOF)
			return -1;
	}
	return 0;
}

static int jas_iccputsint(jas_stream_t *out, int n, longlong val)
{
	ulonglong tmp;
	tmp = (val < 0) ? (abort(), 0) : val;
	return jas_iccputuint(out, n, tmp);
}

/******************************************************************************\
*
\******************************************************************************/

static char *jas_iccsigtostr(int sig, char *buf)
{
	int n;
	int c;
	char *bufptr;
	bufptr = buf;
	for (n = 4; n > 0; --n) {
		c = (sig >> 24) & 0xff;
		if (isalpha(c) || isdigit(c)) {
			*bufptr++ = c;
		}
		sig <<= 8;
	}
	*bufptr = '\0';
	return buf;
}

static long jas_iccpadtomult(long x, long y)
{
	return ((x + y - 1) / y) * y;
}

static long jas_iccpowi(int x, int n)
{
	long y;
	y = 1;
	while (--n >= 0)
		y *= x;
	return y;
}


jas_iccprof_t *jas_iccprof_createfrombuf(uchar *buf, int len)
{
	jas_stream_t *in;
	jas_iccprof_t *prof;
	if (!(in = jas_stream_memopen(JAS_CAST(char *, buf), len)))
		goto error;
	if (!(prof = jas_iccprof_load(in)))
		goto error;
	jas_stream_close(in);
	return prof;
error:
	return 0;
}

jas_iccprof_t *jas_iccprof_createfromclrspc(int clrspc)
{
	jas_iccprof_t *prof;
	switch (clrspc) {
	case JAS_CLRSPC_SRGB:
		prof = jas_iccprof_createfrombuf(jas_iccprofdata_srgb,
		  jas_iccprofdata_srgblen);
		break;
	case JAS_CLRSPC_SGRAY:
		prof = jas_iccprof_createfrombuf(jas_iccprofdata_sgray,
		  jas_iccprofdata_sgraylen);
		break;
	default:
		prof = 0;
		break;
	}
	return prof;
}
