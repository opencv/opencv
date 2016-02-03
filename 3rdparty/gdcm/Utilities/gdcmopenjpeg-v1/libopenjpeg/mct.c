/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
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

#ifdef __SSE__
#include <xmmintrin.h>
#endif

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
void mct_encode(
		int* restrict c0,
		int* restrict c1,
		int* restrict c2,
		int n)
{
	int i;
	for(i = 0; i < n; ++i) {
		int r = c0[i];
		int g = c1[i];
		int b = c2[i];
		int y = (r + (g * 2) + b) >> 2;
		int u = b - g;
		int v = r - g;
		c0[i] = y;
		c1[i] = u;
		c2[i] = v;
	}
}

/* <summary> */
/* Inverse reversible MCT. */
/* </summary> */
void mct_decode(
		int* restrict c0,
		int* restrict c1, 
		int* restrict c2, 
		int n)
{
	int i;
	for (i = 0; i < n; ++i) {
		int y = c0[i];
		int u = c1[i];
		int v = c2[i];
		int g = y - ((u + v) >> 2);
		int r = v + g;
		int b = u + g;
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
void mct_encode_real(
		int* restrict c0,
		int* restrict c1,
		int* restrict c2,
		int n)
{
	int i;
	for(i = 0; i < n; ++i) {
		int r = c0[i];
		int g = c1[i];
		int b = c2[i];
		int y =  fix_mul(r, 2449) + fix_mul(g, 4809) + fix_mul(b, 934);
		int u = -fix_mul(r, 1382) - fix_mul(g, 2714) + fix_mul(b, 4096);
		int v =  fix_mul(r, 4096) - fix_mul(g, 3430) - fix_mul(b, 666);
		c0[i] = y;
		c1[i] = u;
		c2[i] = v;
	}
}

/* <summary> */
/* Inverse irreversible MCT. */
/* </summary> */
void mct_decode_real(
		float* restrict c0,
		float* restrict c1,
		float* restrict c2,
		int n)
{
	int i;
#ifdef __SSE__
	__m128 vrv, vgu, vgv, vbu;
	vrv = _mm_set1_ps(1.402f);
	vgu = _mm_set1_ps(0.34413f);
	vgv = _mm_set1_ps(0.71414f);
	vbu = _mm_set1_ps(1.772f);
	for (i = 0; i < (n >> 3); ++i) {
		__m128 vy, vu, vv;
		__m128 vr, vg, vb;

		vy = _mm_load_ps(c0);
		vu = _mm_load_ps(c1);
		vv = _mm_load_ps(c2);
		vr = _mm_add_ps(vy, _mm_mul_ps(vv, vrv));
		vg = _mm_sub_ps(_mm_sub_ps(vy, _mm_mul_ps(vu, vgu)), _mm_mul_ps(vv, vgv));
		vb = _mm_add_ps(vy, _mm_mul_ps(vu, vbu));
		_mm_store_ps(c0, vr);
		_mm_store_ps(c1, vg);
		_mm_store_ps(c2, vb);
		c0 += 4;
		c1 += 4;
		c2 += 4;

		vy = _mm_load_ps(c0);
		vu = _mm_load_ps(c1);
		vv = _mm_load_ps(c2);
		vr = _mm_add_ps(vy, _mm_mul_ps(vv, vrv));
		vg = _mm_sub_ps(_mm_sub_ps(vy, _mm_mul_ps(vu, vgu)), _mm_mul_ps(vv, vgv));
		vb = _mm_add_ps(vy, _mm_mul_ps(vu, vbu));
		_mm_store_ps(c0, vr);
		_mm_store_ps(c1, vg);
		_mm_store_ps(c2, vb);
		c0 += 4;
		c1 += 4;
		c2 += 4;
	}
	n &= 7;
#endif
	for(i = 0; i < n; ++i) {
		float y = c0[i];
		float u = c1[i];
		float v = c2[i];
		float r = y + (v * 1.402f);
		float g = y - (u * 0.34413f) - (v * (0.71414f));
		float b = y + (u * 1.772f);
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
