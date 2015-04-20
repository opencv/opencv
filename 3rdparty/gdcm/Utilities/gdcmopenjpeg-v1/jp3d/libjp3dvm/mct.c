/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "opj_includes.h"

/* <summary> */
/* This table contains the norms of the basis function of the reversible MCT. */
/* </summary> */
static const double mct_norms[3] = { 1.732, .8292, .8292 };

/* <summary> */
/* This table contains the norms of the basis function of the irreversible MCT. */
/* </summary> */
static const double mct_norms_real[3] = { 1.732, 1.805, 1.573 };

/* <summary> */
/* Foward reversible MCT. */
/* </summary> */
void mct_encode(int *c0, int *c1, int *c2, int n) {
	int i;
	for (i = 0; i < n; i++) {
		int r, g, b, y, u, v;
		r = c0[i];
		g = c1[i];
		b = c2[i];
		y = (r + (g << 1) + b) >> 2;
		u = b - g;
		v = r - g;
		c0[i] = y;
		c1[i] = u;
		c2[i] = v;
	}
}

/* <summary> */
/* Inverse reversible MCT. */
/* </summary> */
void mct_decode(int *c0, int *c1, int *c2, int n) {
	int i;
	for (i = 0; i < n; i++) {
		int y, u, v, r, g, b;
		y = c0[i];
		u = c1[i];
		v = c2[i];
		g = y - ((u + v) >> 2);
		r = v + g;
		b = u + g;
		c0[i] = r;
		c1[i] = g;
		c2[i] = b;
	}
}

/* <summary> */
/* Get norm of basis function of reversible MCT. */
/* </summary> */
double mct_getnorm(int compno) {
	return mct_norms[compno];
}

/* <summary> */
/* Foward irreversible MCT. */
/* </summary> */
void mct_encode_real(int *c0, int *c1, int *c2, int n) {
	int i;
	for (i = 0; i < n; i++) {
		int r, g, b, y, u, v;
		r = c0[i];
		g = c1[i];
		b = c2[i];
		y = fix_mul(r, 2449) + fix_mul(g, 4809) + fix_mul(b, 934);
		u = -fix_mul(r, 1382) - fix_mul(g, 2714) + fix_mul(b, 4096);
		v = fix_mul(r, 4096) - fix_mul(g, 3430) - fix_mul(b, 666);
		c0[i] = y;
		c1[i] = u;
		c2[i] = v;
	}
}

/* <summary> */
/* Inverse irreversible MCT. */
/* </summary> */
void mct_decode_real(int *c0, int *c1, int *c2, int n) {
	int i;
	for (i = 0; i < n; i++) {
		int y, u, v, r, g, b;
		y = c0[i];
		u = c1[i];
		v = c2[i];
		r = y + fix_mul(v, 11485);
		g = y - fix_mul(u, 2819) - fix_mul(v, 5850);
		b = y + fix_mul(u, 14516);
		c0[i] = r;
		c1[i] = g;
		c2[i] = b;
	}
}

/* <summary> */
/* Get norm of basis function of irreversible MCT. */
/* </summary> */
double mct_getnorm_real(int compno) {
	return mct_norms_real[compno];
}
