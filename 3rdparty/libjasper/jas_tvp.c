/*
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
 * Tag-Value Parser Library
 *
 * $Id: jas_tvp.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

#include "jasper/jas_malloc.h"
#include "jasper/jas_string.h"
#include "jasper/jas_tvp.h"

/******************************************************************************\
* Macros.
\******************************************************************************/

/* Is the specified character valid for a tag name? */
#define	JAS_TVP_ISTAG(x) \
	(isalpha(x) || (x) == '_' || isdigit(x))

/******************************************************************************\
* Code for creating and destroying a tag-value parser.
\******************************************************************************/

jas_tvparser_t *jas_tvparser_create(const char *s)
{
	jas_tvparser_t *tvp;
	if (!(tvp = jas_malloc(sizeof(jas_tvparser_t)))) {
		return 0;
	}
	if (!(tvp->buf = jas_strdup(s))) {
		jas_tvparser_destroy(tvp);
		return 0;
	}
	tvp->pos = tvp->buf;
	tvp->tag = 0;
	tvp->val = 0;
	return tvp;
}

void jas_tvparser_destroy(jas_tvparser_t *tvp)
{
	if (tvp->buf) {
		jas_free(tvp->buf);
	}
	jas_free(tvp);
}

/******************************************************************************\
* Main parsing code.
\******************************************************************************/

/* Get the next tag-value pair. */
int jas_tvparser_next(jas_tvparser_t *tvp)
{
	char *p;
	char *tag;
	char *val;

	/* Skip any leading whitespace. */
	p = tvp->pos;
	while (*p != '\0' && isspace(*p)) {
		++p;
	}

	/* Has the end of the input data been reached? */
	if (*p == '\0') {
		/* No more tags are present. */
		tvp->pos = p;
		return 1;
	}

	/* Does the tag name begin with a valid character? */
	if (!JAS_TVP_ISTAG(*p)) {
		return -1;
	}

	/* Remember where the tag name begins. */
	tag = p;

	/* Find the end of the tag name. */
	while (*p != '\0' && JAS_TVP_ISTAG(*p)) {
		++p;
	}

	/* Has the end of the input data been reached? */
	if (*p == '\0') {
		/* The value field is empty. */
		tvp->tag = tag;
		tvp->val = "";
		tvp->pos = p;
		return 0;
	}

	/* Is a value field not present? */
	if (*p != '=') {
		if (*p != '\0' && !isspace(*p)) {
			return -1;
		}
		*p++ = '\0';
		tvp->tag = tag;
		tvp->val = "";
		tvp->pos = p;
		return 0;
	}

	*p++ = '\0';

	val = p;
	while (*p != '\0' && !isspace(*p)) {
		++p;
	}

	if (*p != '\0') {
		*p++ = '\0';
	}

	tvp->pos = p;
	tvp->tag = tag;
	tvp->val = val;

	return 0;
}

/******************************************************************************\
* Code for querying the current tag/value.
\******************************************************************************/

/* Get the current tag. */
char *jas_tvparser_gettag(jas_tvparser_t *tvp)
{
	return tvp->tag;
}

/* Get the current value. */
char *jas_tvparser_getval(jas_tvparser_t *tvp)
{
	return tvp->val;
}

/******************************************************************************\
* Miscellaneous code.
\******************************************************************************/

/* Lookup a tag by name. */
jas_taginfo_t *jas_taginfos_lookup(jas_taginfo_t *taginfos, const char *name)
{
	jas_taginfo_t *taginfo;
	taginfo = taginfos;
	while (taginfo->id >= 0) {
		if (!strcmp(taginfo->name, name)) {
			return taginfo;
		}
		++taginfo;
	}
	return 0;
}

/* This function is simply for convenience. */
/* One can avoid testing for the special case of a null pointer, by
  using this function.   This function never returns a null pointer.  */
jas_taginfo_t *jas_taginfo_nonull(jas_taginfo_t *taginfo)
{
	static jas_taginfo_t invalidtaginfo = {
		-1, 0
	};
	
	return taginfo ? taginfo : &invalidtaginfo;
}
