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
 * $Id: jpc_t1cod.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "jasper/jas_types.h"
#include "jasper/jas_math.h"

#include "jpc_bs.h"
#include "jpc_dec.h"
#include "jpc_cs.h"
#include "jpc_mqcod.h"
#include "jpc_t1cod.h"
#include "jpc_tsfb.h"

double jpc_pow2i(int n);

/******************************************************************************\
* Global data.
\******************************************************************************/

int jpc_zcctxnolut[4 * 256];
int jpc_spblut[256];
int jpc_scctxnolut[256];
int jpc_magctxnolut[4096];

jpc_fix_t jpc_signmsedec[1 << JPC_NMSEDEC_BITS];
jpc_fix_t jpc_refnmsedec[1 << JPC_NMSEDEC_BITS];
jpc_fix_t jpc_signmsedec0[1 << JPC_NMSEDEC_BITS];
jpc_fix_t jpc_refnmsedec0[1 << JPC_NMSEDEC_BITS];

jpc_mqctx_t jpc_mqctxs[JPC_NUMCTXS];

/******************************************************************************\
*
\******************************************************************************/

void jpc_initmqctxs(void);

/******************************************************************************\
* Code.
\******************************************************************************/

int JPC_PASSTYPE(int passno)
{
	int passtype;
	switch (passno % 3) {
	case 0:
		passtype = JPC_CLNPASS;
		break;
	case 1:
		passtype = JPC_SIGPASS;
		break;
	case 2:
		passtype = JPC_REFPASS;
		break;
	default:
		passtype = -1;
		assert(0);
		break;
	}
	return passtype;
}

int JPC_NOMINALGAIN(int qmfbid, int numlvls, int lvlno, int orient)
{
	/* Avoid compiler warnings about unused parameters. */
	numlvls = 0;

if (qmfbid == JPC_COX_INS) {
	return 0;
}
	assert(qmfbid == JPC_COX_RFT);
	if (lvlno == 0) {
		assert(orient == JPC_TSFB_LL);
		return 0;
	} else {
		switch (orient) {
		case JPC_TSFB_LH:
		case JPC_TSFB_HL:
			return 1;
			break;
		case JPC_TSFB_HH:
			return 2;
			break;
		}
	}
	abort();
}

/******************************************************************************\
* Coding pass related functions.
\******************************************************************************/

int JPC_SEGTYPE(int passno, int firstpassno, int bypass)
{
	int passtype;
	if (bypass) {
		passtype = JPC_PASSTYPE(passno);
		if (passtype == JPC_CLNPASS) {
			return JPC_SEG_MQ;
		}
		return ((passno < firstpassno + 10) ? JPC_SEG_MQ : JPC_SEG_RAW);
	} else {
		return JPC_SEG_MQ;
	}
}

int JPC_SEGPASSCNT(int passno, int firstpassno, int numpasses, int bypass, int termall)
{
	int ret;
	int passtype;

	if (termall) {
		ret = 1;
	} else if (bypass) {
		if (passno < firstpassno + 10) {
			ret = 10 - (passno - firstpassno);
		} else {
			passtype = JPC_PASSTYPE(passno);
			switch (passtype) {
			case JPC_SIGPASS:
				ret = 2;
				break;
			case JPC_REFPASS:
				ret = 1;
				break;
			case JPC_CLNPASS:
				ret = 1;
				break;
			default:
				ret = -1;
				assert(0);
				break;
			}
		}
	} else {
		ret = JPC_PREC * 3 - 2;
	}
	ret = JAS_MIN(ret, numpasses - passno);
	return ret;
}

int JPC_ISTERMINATED(int passno, int firstpassno, int numpasses, int termall,
  int lazy)
{
	int ret;
	int n;
	if (passno - firstpassno == numpasses - 1) {
		ret = 1;
	} else {
		n = JPC_SEGPASSCNT(passno, firstpassno, numpasses, lazy, termall);
		ret = (n <= 1) ? 1 : 0;
	}

	return ret;
}

/******************************************************************************\
* Lookup table code.
\******************************************************************************/

void jpc_initluts()
{
	int i;
	int orient;
	int refine;
	float u;
	float v;
	float t;

/* XXX - hack */
jpc_initmqctxs();

	for (orient = 0; orient < 4; ++orient) {
		for (i = 0; i < 256; ++i) {
			jpc_zcctxnolut[(orient << 8) | i] = jpc_getzcctxno(i, orient);
		}
	}

	for (i = 0; i < 256; ++i) {
		jpc_spblut[i] = jpc_getspb(i << 4);
	}

	for (i = 0; i < 256; ++i) {
		jpc_scctxnolut[i] = jpc_getscctxno(i << 4);
	}

	for (refine = 0; refine < 2; ++refine) {
		for (i = 0; i < 2048; ++i) {
			jpc_magctxnolut[(refine << 11) + i] = jpc_getmagctxno((refine ? JPC_REFINE : 0) | i);
		}
	}

	for (i = 0; i < (1 << JPC_NMSEDEC_BITS); ++i) {
		t = i * jpc_pow2i(-JPC_NMSEDEC_FRACBITS);
		u = t;
		v = t - 1.5;
		jpc_signmsedec[i] = jpc_dbltofix(floor((u * u - v * v) * jpc_pow2i(JPC_NMSEDEC_FRACBITS) + 0.5) / jpc_pow2i(JPC_NMSEDEC_FRACBITS));
/* XXX - this calc is not correct */
		jpc_signmsedec0[i] = jpc_dbltofix(floor((u * u) * jpc_pow2i(JPC_NMSEDEC_FRACBITS) + 0.5) / jpc_pow2i(JPC_NMSEDEC_FRACBITS));
		u = t - 1.0;
		if (i & (1 << (JPC_NMSEDEC_BITS - 1))) {
			v = t - 1.5;
		} else {
			v = t - 0.5;
		}
		jpc_refnmsedec[i] = jpc_dbltofix(floor((u * u - v * v) * jpc_pow2i(JPC_NMSEDEC_FRACBITS) + 0.5) / jpc_pow2i(JPC_NMSEDEC_FRACBITS));
/* XXX - this calc is not correct */
		jpc_refnmsedec0[i] = jpc_dbltofix(floor((u * u) * jpc_pow2i(JPC_NMSEDEC_FRACBITS) + 0.5) / jpc_pow2i(JPC_NMSEDEC_FRACBITS));
	}
}

jpc_fix_t jpc_getsignmsedec_func(jpc_fix_t x, int bitpos)
{
	jpc_fix_t y;
	assert(!(x & (~JAS_ONES(bitpos + 1))));
	y = jpc_getsignmsedec_macro(x, bitpos);
	return y;
}

int jpc_getzcctxno(int f, int orient)
{
	int h;
	int v;
	int d;
	int n;
	int t;
	int hv;

	/* Avoid compiler warning. */
	n = 0;

	h = ((f & JPC_WSIG) != 0) + ((f & JPC_ESIG) != 0);
	v = ((f & JPC_NSIG) != 0) + ((f & JPC_SSIG) != 0);
	d = ((f & JPC_NWSIG) != 0) + ((f & JPC_NESIG) != 0) + ((f & JPC_SESIG) != 0) + ((f & JPC_SWSIG) != 0);
	switch (orient) {
	case JPC_TSFB_HL:
		t = h;
		h = v;
		v = t;
	case JPC_TSFB_LL:
	case JPC_TSFB_LH:
		if (!h) {
			if (!v) {
				if (!d) {
					n = 0;
				} else if (d == 1) {
					n = 1;
				} else {
					n = 2;
				}
			} else if (v == 1) {
				n = 3;
			} else {
				n = 4;
			}
		} else if (h == 1) {
			if (!v) {
				if (!d) {
					n = 5;
				} else {
					n = 6;
				}
			} else {
				n = 7;
			}
		} else {
			n = 8;
		}
		break;
	case JPC_TSFB_HH:
		hv = h + v;
		if (!d) {
			if (!hv) {
				n = 0;
			} else if (hv == 1) {
				n = 1;
			} else {
				n = 2;
			}
		} else if (d == 1) {
			if (!hv) {
				n = 3;
			} else if (hv == 1) {
				n = 4;
			} else {
				n = 5;
			}
		} else if (d == 2) {
			if (!hv) {
				n = 6;
			} else {
				n = 7;
			}
		} else {
			n = 8;
		}
		break;
	}
	assert(n < JPC_NUMZCCTXS);
	return JPC_ZCCTXNO + n;
}

int jpc_getspb(int f)
{
	int hc;
	int vc;
	int n;

	hc = JAS_MIN(((f & (JPC_ESIG | JPC_ESGN)) == JPC_ESIG) + ((f & (JPC_WSIG | JPC_WSGN)) == JPC_WSIG), 1) -
	  JAS_MIN(((f & (JPC_ESIG | JPC_ESGN)) == (JPC_ESIG | JPC_ESGN)) + ((f & (JPC_WSIG | JPC_WSGN)) == (JPC_WSIG | JPC_WSGN)), 1);
	vc = JAS_MIN(((f & (JPC_NSIG | JPC_NSGN)) == JPC_NSIG) + ((f & (JPC_SSIG | JPC_SSGN)) == JPC_SSIG), 1) -
	  JAS_MIN(((f & (JPC_NSIG | JPC_NSGN)) == (JPC_NSIG | JPC_NSGN)) + ((f & (JPC_SSIG | JPC_SSGN)) == (JPC_SSIG | JPC_SSGN)), 1);
	if (!hc && !vc) {
		n = 0;
	} else {
		n = (!(hc > 0 || (!hc && vc > 0)));
	}
	return n;
}

int jpc_getscctxno(int f)
{
	int hc;
	int vc;
	int n;

	/* Avoid compiler warning. */
	n = 0;

	hc = JAS_MIN(((f & (JPC_ESIG | JPC_ESGN)) == JPC_ESIG) + ((f & (JPC_WSIG | JPC_WSGN)) == JPC_WSIG),
	  1) - JAS_MIN(((f & (JPC_ESIG | JPC_ESGN)) == (JPC_ESIG | JPC_ESGN)) +
	  ((f & (JPC_WSIG | JPC_WSGN)) == (JPC_WSIG | JPC_WSGN)), 1);
	vc = JAS_MIN(((f & (JPC_NSIG | JPC_NSGN)) == JPC_NSIG) + ((f & (JPC_SSIG | JPC_SSGN)) == JPC_SSIG),
	  1) - JAS_MIN(((f & (JPC_NSIG | JPC_NSGN)) == (JPC_NSIG | JPC_NSGN)) +
	  ((f & (JPC_SSIG | JPC_SSGN)) == (JPC_SSIG | JPC_SSGN)), 1);
	assert(hc >= -1 && hc <= 1 && vc >= -1 && vc <= 1);
	if (hc < 0) {
		hc = -hc;
		vc = -vc;
	}
	if (!hc) {
		if (vc == -1) {
			n = 1;
		} else if (!vc) {
			n = 0;
		} else {
			n = 1;
		}
	} else if (hc == 1) {
		if (vc == -1) {
			n = 2;
		} else if (!vc) {
			n = 3;
		} else {
			n = 4;
		}
	}
	assert(n < JPC_NUMSCCTXS);
	return JPC_SCCTXNO + n;
}

int jpc_getmagctxno(int f)
{
	int n;

	if (!(f & JPC_REFINE)) {
		n = (f & (JPC_OTHSIGMSK)) ? 1 : 0;
	} else {
		n = 2;
	}

	assert(n < JPC_NUMMAGCTXS);
	return JPC_MAGCTXNO + n;
}

void jpc_initctxs(jpc_mqctx_t *ctxs)
{
	jpc_mqctx_t *ctx;
	int i;

	ctx = ctxs;
	for (i = 0; i < JPC_NUMCTXS; ++i) {
		ctx->mps = 0;
		switch (i) {
		case JPC_UCTXNO:
			ctx->ind = 46;
			break;
		case JPC_ZCCTXNO:
			ctx->ind = 4;
			break;
		case JPC_AGGCTXNO:
			ctx->ind = 3;
			break;
		default:
			ctx->ind = 0;
			break;
		}
		++ctx;
	}
}

void jpc_initmqctxs()
{
	jpc_initctxs(jpc_mqctxs);
}

/* Calculate the real quantity exp2(n), where x is an integer. */
double jpc_pow2i(int n)
{
	double x;
	double a;

	x = 1.0;
	if (n < 0) {
		a = 0.5;
		n = -n;
	} else {
		a = 2.0;
	}
	while (--n >= 0) {
		x *= a;
	}
	return x;
}
