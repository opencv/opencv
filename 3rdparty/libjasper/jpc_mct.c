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
 * Multicomponent Transform Code
 *
 * $Id: jpc_mct.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <assert.h>

#include "jasper/jas_seq.h"

#include "jpc_fix.h"
#include "jpc_mct.h"

/******************************************************************************\
* Code.
\******************************************************************************/

/* Compute the forward RCT. */

void jpc_rct(jas_matrix_t *c0, jas_matrix_t *c1, jas_matrix_t *c2)
{
	int numrows;
	int numcols;
	int i;
	int j;
	jpc_fix_t *c0p;
	jpc_fix_t *c1p;
	jpc_fix_t *c2p;

	numrows = jas_matrix_numrows(c0);
	numcols = jas_matrix_numcols(c0);

	/* All three matrices must have the same dimensions. */
	assert(jas_matrix_numrows(c1) == numrows && jas_matrix_numcols(c1) == numcols
	  && jas_matrix_numrows(c2) == numrows && jas_matrix_numcols(c2) == numcols);

	for (i = 0; i < numrows; i++) {
		c0p = jas_matrix_getref(c0, i, 0);
		c1p = jas_matrix_getref(c1, i, 0);
		c2p = jas_matrix_getref(c2, i, 0);
		for (j = numcols; j > 0; --j) {
			int r;
			int g;
			int b;
			int y;
			int u;
			int v;
			r = *c0p;
			g = *c1p;
			b = *c2p;
			y = (r + (g << 1) + b) >> 2;
			u = b - g;
			v = r - g;
			*c0p++ = y;
			*c1p++ = u;
			*c2p++ = v;
		}
	}
}

/* Compute the inverse RCT. */

void jpc_irct(jas_matrix_t *c0, jas_matrix_t *c1, jas_matrix_t *c2)
{
	int numrows;
	int numcols;
	int i;
	int j;
	jpc_fix_t *c0p;
	jpc_fix_t *c1p;
	jpc_fix_t *c2p;

	numrows = jas_matrix_numrows(c0);
	numcols = jas_matrix_numcols(c0);

	/* All three matrices must have the same dimensions. */
	assert(jas_matrix_numrows(c1) == numrows && jas_matrix_numcols(c1) == numcols
	  && jas_matrix_numrows(c2) == numrows && jas_matrix_numcols(c2) == numcols);

	for (i = 0; i < numrows; i++) {
		c0p = jas_matrix_getref(c0, i, 0);
		c1p = jas_matrix_getref(c1, i, 0);
		c2p = jas_matrix_getref(c2, i, 0);
		for (j = numcols; j > 0; --j) {
			int r;
			int g;
			int b;
			int y;
			int u;
			int v;
			y = *c0p;
			u = *c1p;
			v = *c2p;
			g = y - ((u + v) >> 2);
			r = v + g;
			b = u + g;
			*c0p++ = r;
			*c1p++ = g;
			*c2p++ = b;
		}
	}
}

void jpc_ict(jas_matrix_t *c0, jas_matrix_t *c1, jas_matrix_t *c2)
{
	int numrows;
	int numcols;
	int i;
	int j;
	jpc_fix_t r;
	jpc_fix_t g;
	jpc_fix_t b;
	jpc_fix_t y;
	jpc_fix_t u;
	jpc_fix_t v;
	jpc_fix_t *c0p;
	jpc_fix_t *c1p;
	jpc_fix_t *c2p;

	numrows = jas_matrix_numrows(c0);
	assert(jas_matrix_numrows(c1) == numrows && jas_matrix_numrows(c2) == numrows);
	numcols = jas_matrix_numcols(c0);
	assert(jas_matrix_numcols(c1) == numcols && jas_matrix_numcols(c2) == numcols);
	for (i = 0; i < numrows; ++i) {
		c0p = jas_matrix_getref(c0, i, 0);
		c1p = jas_matrix_getref(c1, i, 0);
		c2p = jas_matrix_getref(c2, i, 0);
		for (j = numcols; j > 0; --j) {
			r = *c0p;
			g = *c1p;
			b = *c2p;
			y = jpc_fix_add3(jpc_fix_mul(jpc_dbltofix(0.299), r), jpc_fix_mul(jpc_dbltofix(0.587), g),
			  jpc_fix_mul(jpc_dbltofix(0.114), b));
			u = jpc_fix_add3(jpc_fix_mul(jpc_dbltofix(-0.16875), r), jpc_fix_mul(jpc_dbltofix(-0.33126), g),
			  jpc_fix_mul(jpc_dbltofix(0.5), b));
			v = jpc_fix_add3(jpc_fix_mul(jpc_dbltofix(0.5), r), jpc_fix_mul(jpc_dbltofix(-0.41869), g),
			  jpc_fix_mul(jpc_dbltofix(-0.08131), b));
			*c0p++ = y;
			*c1p++ = u;
			*c2p++ = v;
		}
	}
}

void jpc_iict(jas_matrix_t *c0, jas_matrix_t *c1, jas_matrix_t *c2)
{
	int numrows;
	int numcols;
	int i;
	int j;
	jpc_fix_t r;
	jpc_fix_t g;
	jpc_fix_t b;
	jpc_fix_t y;
	jpc_fix_t u;
	jpc_fix_t v;
	jpc_fix_t *c0p;
	jpc_fix_t *c1p;
	jpc_fix_t *c2p;

	numrows = jas_matrix_numrows(c0);
	assert(jas_matrix_numrows(c1) == numrows && jas_matrix_numrows(c2) == numrows);
	numcols = jas_matrix_numcols(c0);
	assert(jas_matrix_numcols(c1) == numcols && jas_matrix_numcols(c2) == numcols);
	for (i = 0; i < numrows; ++i) {
		c0p = jas_matrix_getref(c0, i, 0);
		c1p = jas_matrix_getref(c1, i, 0);
		c2p = jas_matrix_getref(c2, i, 0);
		for (j = numcols; j > 0; --j) {
			y = *c0p;
			u = *c1p;
			v = *c2p;
			r = jpc_fix_add(y, jpc_fix_mul(jpc_dbltofix(1.402), v));
			g = jpc_fix_add3(y, jpc_fix_mul(jpc_dbltofix(-0.34413), u),
			  jpc_fix_mul(jpc_dbltofix(-0.71414), v));
			b = jpc_fix_add(y, jpc_fix_mul(jpc_dbltofix(1.772), u));
			*c0p++ = r;
			*c1p++ = g;
			*c2p++ = b;
		}
	}
}

jpc_fix_t jpc_mct_getsynweight(int mctid, int cmptno)
{
	jpc_fix_t synweight;

	synweight = JPC_FIX_ONE;
	switch (mctid) {
	case JPC_MCT_RCT:
		switch (cmptno) {
		case 0:
			synweight = jpc_dbltofix(sqrt(3.0));
			break;
		case 1:
			synweight = jpc_dbltofix(sqrt(0.6875));
			break;
		case 2:
			synweight = jpc_dbltofix(sqrt(0.6875));
			break;
		}
		break;
	case JPC_MCT_ICT:
		switch (cmptno) {
		case 0:
			synweight = jpc_dbltofix(sqrt(3.0000));
			break;
		case 1:
			synweight = jpc_dbltofix(sqrt(3.2584));
			break;
		case 2:
			synweight = jpc_dbltofix(sqrt(2.4755));
			break;
		}
		break;
#if 0
	default:
		synweight = JPC_FIX_ONE;
		break;
#endif
	}

	return synweight;
}
