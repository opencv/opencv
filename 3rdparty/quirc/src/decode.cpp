/* quirc -- QR-code recognition library
 * Copyright (C) 2010-2012 Daniel Beer <dlbeer@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "quirc_internal.h"

#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <bitset>
#include <iomanip>

using namespace std;
using namespace cv;

#define MAX_POLY       64

/************************************************************************
 * Galois fields
 */

struct galois_field {
	int p;
	const uint8_t *log;
	const uint8_t *exp;
};

static const uint8_t gf16_exp[16] = {
	0x01, 0x02, 0x04, 0x08, 0x03, 0x06, 0x0c, 0x0b,
	0x05, 0x0a, 0x07, 0x0e, 0x0f, 0x0d, 0x09, 0x01
};

static const uint8_t gf16_log[16] = {
	0x00, 0x0f, 0x01, 0x04, 0x02, 0x08, 0x05, 0x0a,
	0x03, 0x0e, 0x09, 0x07, 0x06, 0x0d, 0x0b, 0x0c
};


static const struct galois_field gf16 = {
	.p = 15,
	.log = gf16_log,
	.exp = gf16_exp
};

static const uint8_t gf256_exp[256] = {
	0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
	0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26,
	0x4c, 0x98, 0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9,
	0x8f, 0x03, 0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0,
	0x9d, 0x27, 0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35,
	0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23,
	0x46, 0x8c, 0x05, 0x0a, 0x14, 0x28, 0x50, 0xa0,
	0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1,
	0x5f, 0xbe, 0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc,
	0x65, 0xca, 0x89, 0x0f, 0x1e, 0x3c, 0x78, 0xf0,
	0xfd, 0xe7, 0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f,
	0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2,
	0xd9, 0xaf, 0x43, 0x86, 0x11, 0x22, 0x44, 0x88,
	0x0d, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce,
	0x81, 0x1f, 0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93,
	0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc,
	0x85, 0x17, 0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9,
	0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54,
	0xa8, 0x4d, 0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa,
	0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73,
	0xe6, 0xd1, 0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e,
	0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff,
	0xe3, 0xdb, 0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4,
	0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41,
	0x82, 0x19, 0x32, 0x64, 0xc8, 0x8d, 0x07, 0x0e,
	0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6,
	0x51, 0xa2, 0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef,
	0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x09,
	0x12, 0x24, 0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5,
	0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0x0b, 0x16,
	0x2c, 0x58, 0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83,
	0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x01
};

static const uint8_t gf256_log[256] = {
	0x00, 0xff, 0x01, 0x19, 0x02, 0x32, 0x1a, 0xc6,
	0x03, 0xdf, 0x33, 0xee, 0x1b, 0x68, 0xc7, 0x4b,
	0x04, 0x64, 0xe0, 0x0e, 0x34, 0x8d, 0xef, 0x81,
	0x1c, 0xc1, 0x69, 0xf8, 0xc8, 0x08, 0x4c, 0x71,
	0x05, 0x8a, 0x65, 0x2f, 0xe1, 0x24, 0x0f, 0x21,
	0x35, 0x93, 0x8e, 0xda, 0xf0, 0x12, 0x82, 0x45,
	0x1d, 0xb5, 0xc2, 0x7d, 0x6a, 0x27, 0xf9, 0xb9,
	0xc9, 0x9a, 0x09, 0x78, 0x4d, 0xe4, 0x72, 0xa6,
	0x06, 0xbf, 0x8b, 0x62, 0x66, 0xdd, 0x30, 0xfd,
	0xe2, 0x98, 0x25, 0xb3, 0x10, 0x91, 0x22, 0x88,
	0x36, 0xd0, 0x94, 0xce, 0x8f, 0x96, 0xdb, 0xbd,
	0xf1, 0xd2, 0x13, 0x5c, 0x83, 0x38, 0x46, 0x40,
	0x1e, 0x42, 0xb6, 0xa3, 0xc3, 0x48, 0x7e, 0x6e,
	0x6b, 0x3a, 0x28, 0x54, 0xfa, 0x85, 0xba, 0x3d,
	0xca, 0x5e, 0x9b, 0x9f, 0x0a, 0x15, 0x79, 0x2b,
	0x4e, 0xd4, 0xe5, 0xac, 0x73, 0xf3, 0xa7, 0x57,
	0x07, 0x70, 0xc0, 0xf7, 0x8c, 0x80, 0x63, 0x0d,
	0x67, 0x4a, 0xde, 0xed, 0x31, 0xc5, 0xfe, 0x18,
	0xe3, 0xa5, 0x99, 0x77, 0x26, 0xb8, 0xb4, 0x7c,
	0x11, 0x44, 0x92, 0xd9, 0x23, 0x20, 0x89, 0x2e,
	0x37, 0x3f, 0xd1, 0x5b, 0x95, 0xbc, 0xcf, 0xcd,
	0x90, 0x87, 0x97, 0xb2, 0xdc, 0xfc, 0xbe, 0x61,
	0xf2, 0x56, 0xd3, 0xab, 0x14, 0x2a, 0x5d, 0x9e,
	0x84, 0x3c, 0x39, 0x53, 0x47, 0x6d, 0x41, 0xa2,
	0x1f, 0x2d, 0x43, 0xd8, 0xb7, 0x7b, 0xa4, 0x76,
	0xc4, 0x17, 0x49, 0xec, 0x7f, 0x0c, 0x6f, 0xf6,
	0x6c, 0xa1, 0x3b, 0x52, 0x29, 0x9d, 0x55, 0xaa,
	0xfb, 0x60, 0x86, 0xb1, 0xbb, 0xcc, 0x3e, 0x5a,
	0xcb, 0x59, 0x5f, 0xb0, 0x9c, 0xa9, 0xa0, 0x51,
	0x0b, 0xf5, 0x16, 0xeb, 0x7a, 0x75, 0x2c, 0xd7,
	0x4f, 0xae, 0xd5, 0xe9, 0xe6, 0xe7, 0xad, 0xe8,
	0x74, 0xd6, 0xf4, 0xea, 0xa8, 0x50, 0x58, 0xaf
};
static const uint8_t gf_exp[256] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26,
        0x4c, 0x98, 0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9,
        0x8f, 0x03, 0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0,
        0x9d, 0x27, 0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35,
        0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23,
        0x46, 0x8c, 0x05, 0x0a, 0x14, 0x28, 0x50, 0xa0,
        0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1,
        0x5f, 0xbe, 0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc,
        0x65, 0xca, 0x89, 0x0f, 0x1e, 0x3c, 0x78, 0xf0,
        0xfd, 0xe7, 0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f,
        0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2,
        0xd9, 0xaf, 0x43, 0x86, 0x11, 0x22, 0x44, 0x88,
        0x0d, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce,
        0x81, 0x1f, 0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93,
        0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc,
        0x85, 0x17, 0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9,
        0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54,
        0xa8, 0x4d, 0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa,
        0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73,
        0xe6, 0xd1, 0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e,
        0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff,
        0xe3, 0xdb, 0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4,
        0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41,
        0x82, 0x19, 0x32, 0x64, 0xc8, 0x8d, 0x07, 0x0e,
        0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6,
        0x51, 0xa2, 0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef,
        0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x09,
        0x12, 0x24, 0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5,
        0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0x0b, 0x16,
        0x2c, 0x58, 0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83,
        0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x01
};
static const uint16_t after_mask_format [32]={
         0x5412,0x5125,0x5e7c,0x5b4b,0x45f9,  0x40ce,0x4f97,0x4aa0,0x77c4,0x72f3,
         0x7daa,0x789d,0x662f,0x6318,0x6c41,  0x6976,0x1689,0x13be,0x1ce7,0x19d0,
         0x0762,0x0255,0x0d0c,0x083b,0x355f,  0x3068,0x3f31,0x3a06,0x24b4,0x2183,
         0x2eda,0x2bed
};


static const uint8_t gf_log[256] = {
        0x00, 0xff, 0x01, 0x19, 0x02, 0x32, 0x1a, 0xc6,
        0x03, 0xdf, 0x33, 0xee, 0x1b, 0x68, 0xc7, 0x4b,
        0x04, 0x64, 0xe0, 0x0e, 0x34, 0x8d, 0xef, 0x81,
        0x1c, 0xc1, 0x69, 0xf8, 0xc8, 0x08, 0x4c, 0x71,
        0x05, 0x8a, 0x65, 0x2f, 0xe1, 0x24, 0x0f, 0x21,
        0x35, 0x93, 0x8e, 0xda, 0xf0, 0x12, 0x82, 0x45,
        0x1d, 0xb5, 0xc2, 0x7d, 0x6a, 0x27, 0xf9, 0xb9,
        0xc9, 0x9a, 0x09, 0x78, 0x4d, 0xe4, 0x72, 0xa6,
        0x06, 0xbf, 0x8b, 0x62, 0x66, 0xdd, 0x30, 0xfd,
        0xe2, 0x98, 0x25, 0xb3, 0x10, 0x91, 0x22, 0x88,
        0x36, 0xd0, 0x94, 0xce, 0x8f, 0x96, 0xdb, 0xbd,
        0xf1, 0xd2, 0x13, 0x5c, 0x83, 0x38, 0x46, 0x40,
        0x1e, 0x42, 0xb6, 0xa3, 0xc3, 0x48, 0x7e, 0x6e,
        0x6b, 0x3a, 0x28, 0x54, 0xfa, 0x85, 0xba, 0x3d,
        0xca, 0x5e, 0x9b, 0x9f, 0x0a, 0x15, 0x79, 0x2b,
        0x4e, 0xd4, 0xe5, 0xac, 0x73, 0xf3, 0xa7, 0x57,
        0x07, 0x70, 0xc0, 0xf7, 0x8c, 0x80, 0x63, 0x0d,
        0x67, 0x4a, 0xde, 0xed, 0x31, 0xc5, 0xfe, 0x18,
        0xe3, 0xa5, 0x99, 0x77, 0x26, 0xb8, 0xb4, 0x7c,
        0x11, 0x44, 0x92, 0xd9, 0x23, 0x20, 0x89, 0x2e,
        0x37, 0x3f, 0xd1, 0x5b, 0x95, 0xbc, 0xcf, 0xcd,
        0x90, 0x87, 0x97, 0xb2, 0xdc, 0xfc, 0xbe, 0x61,
        0xf2, 0x56, 0xd3, 0xab, 0x14, 0x2a, 0x5d, 0x9e,
        0x84, 0x3c, 0x39, 0x53, 0x47, 0x6d, 0x41, 0xa2,
        0x1f, 0x2d, 0x43, 0xd8, 0xb7, 0x7b, 0xa4, 0x76,
        0xc4, 0x17, 0x49, 0xec, 0x7f, 0x0c, 0x6f, 0xf6,
        0x6c, 0xa1, 0x3b, 0x52, 0x29, 0x9d, 0x55, 0xaa,
        0xfb, 0x60, 0x86, 0xb1, 0xbb, 0xcc, 0x3e, 0x5a,
        0xcb, 0x59, 0x5f, 0xb0, 0x9c, 0xa9, 0xa0, 0x51,
        0x0b, 0xf5, 0x16, 0xeb, 0x7a, 0x75, 0x2c, 0xd7,
        0x4f, 0xae, 0xd5, 0xe9, 0xe6, 0xe7, 0xad, 0xe8,
        0x74, 0xd6, 0xf4, 0xea, 0xa8, 0x50, 0x58, 0xaf
};
static const struct galois_field gf256 = {
	.p = 255,
	.log = gf256_log,
	.exp = gf256_exp
};

/************************************************************************
 * Polynomial operations
 */

static void poly_add(uint8_t *dst, const uint8_t *src, uint8_t c,
		     int shift, const struct galois_field *gf)
{
	int i;
	int log_c = gf->log[c];

	if (!c)
		return;

	for (i = 0; i < MAX_POLY; i++) {
		int p = i + shift;
		uint8_t v = src[i];

		if (p < 0 || p >= MAX_POLY)
			continue;
		if (!v)
			continue;

		dst[p] ^= gf->exp[(gf->log[v] + log_c) % gf->p];
	}
}

static uint8_t poly_eval(const uint8_t *s, uint8_t x,
			 const struct galois_field *gf)
{
	int i;
	uint8_t sum = 0;
	uint8_t log_x = gf->log[x];

	if (!x)
		return s[0];

	for (i = 0; i < MAX_POLY; i++) {
		uint8_t c = s[i];

		if (!c)
			continue;

		sum ^= gf->exp[(gf->log[c] + log_x * i) % gf->p];
	}

	return sum;
}

/************************************************************************
 * Berlekamp-Massey algorithm for finding error locator polynomials.
 */

static void berlekamp_massey(const uint8_t *s, int N,
			     const struct galois_field *gf,
			     uint8_t *sigma)
{
	uint8_t C[MAX_POLY];
	uint8_t B[MAX_POLY];
	int L = 0;
	int m = 1;
	uint8_t b = 1;
	int n;

	memset(B, 0, sizeof(B));
	memset(C, 0, sizeof(C));
	B[0] = 1;
	C[0] = 1;

	for (n = 0; n < N; n++) {
		uint8_t d = s[n];
		uint8_t mult;
		int i;

		for (i = 1; i <= L; i++) {
			if (!(C[i] && s[n - i]))
				continue;

			d ^= gf->exp[(gf->log[C[i]] +
				      gf->log[s[n - i]]) %
				     gf->p];
		}

		mult = gf->exp[(gf->p - gf->log[b] + gf->log[d]) % gf->p];

		if (!d) {
			m++;
		} else if (L * 2 <= n) {
			uint8_t T[MAX_POLY];

			memcpy(T, C, sizeof(T));
			poly_add(C, B, mult, m, gf);
			memcpy(B, T, sizeof(B));
			L = n + 1 - L;
			b = d;
			m = 1;
		} else {
			poly_add(C, B, mult, m, gf);
			m++;
		}
	}

	memcpy(sigma, C, MAX_POLY);
}

/************************************************************************
 * Code stream error correction
 *
 * Generator polynomial for GF(2^8) is x^8 + x^4 + x^3 + x^2 + 1
 */

static int block_syndromes(const uint8_t *data, int bs, int npar, uint8_t *s)
{
	int nonzero = 0;
	int i;

	memset(s, 0, MAX_POLY);

	for (i = 0; i < npar; i++) {
		int j;

		for (j = 0; j < bs; j++) {
			uint8_t c = data[bs - j - 1];

			if (!c)
				continue;

			s[i] ^= gf256_exp[((int)gf256_log[c] +
				    i * j) % 255];
		}

		if (s[i])
			nonzero = 1;
	}

	return nonzero;
}

static void eloc_poly(uint8_t *omega,
		      const uint8_t *s, const uint8_t *sigma,
		      int npar)
{
	int i;

	memset(omega, 0, MAX_POLY);

	for (i = 0; i < npar; i++) {
		const uint8_t a = sigma[i];
		const uint8_t log_a = gf256_log[a];
		int j;

		if (!a)
			continue;

		for (j = 0; j + 1 < MAX_POLY; j++) {
			const uint8_t b = s[j + 1];

			if (i + j >= npar)
				break;

			if (!b)
				continue;

			omega[i + j] ^=
			    gf256_exp[(log_a + gf256_log[b]) % 255];
		}
	}
}

static quirc_decode_error_t correct_block(uint8_t *data,
					  const struct quirc_rs_params *ecc)
{
	int npar = ecc->bs - ecc->dw;
	uint8_t s[MAX_POLY];
	uint8_t sigma[MAX_POLY];
	uint8_t sigma_deriv[MAX_POLY];
	uint8_t omega[MAX_POLY];
	int i;

	/* Compute syndrome vector */
	if (!block_syndromes(data, ecc->bs, npar, s))
		return QUIRC_SUCCESS;

	berlekamp_massey(s, npar, &gf256, sigma);

	/* Compute derivative of sigma */
	memset(sigma_deriv, 0, MAX_POLY);
	for (i = 0; i + 1 < MAX_POLY; i += 2)
		sigma_deriv[i] = sigma[i + 1];

	/* Compute error evaluator polynomial */
	eloc_poly(omega, s, sigma, npar - 1);

	/* Find error locations and magnitudes */
	for (i = 0; i < ecc->bs; i++) {
		uint8_t xinv = gf256_exp[255 - i];

		if (!poly_eval(sigma, xinv, &gf256)) {
			uint8_t sd_x = poly_eval(sigma_deriv, xinv, &gf256);
			uint8_t omega_x = poly_eval(omega, xinv, &gf256);
			uint8_t error = gf256_exp[(255 - gf256_log[sd_x] +
						   gf256_log[omega_x]) % 255];

			data[ecc->bs - i - 1] ^= error;
		}
	}

	if (block_syndromes(data, ecc->bs, npar, s))
		return QUIRC_ERROR_DATA_ECC;

	return QUIRC_SUCCESS;
}

/************************************************************************
 * Format value error correction
 *
 * Generator polynomial for GF(2^4) is x^4 + x + 1
 */
std::string D2B(uint16_t my_format){
    std::string f;
    for(int i=my_format;i>0;i=i>>1){
        if(i%2==1)
            f='1'+f;
        else
            f='0'+f;
    }
    return f;
}
#define FORMAT_MAX_ERROR        3
#define FORMAT_SYNDROMES        (FORMAT_MAX_ERROR * 2)
#define FORMAT_BITS             15

/*init_tables可以改进 通过初始化表的方式
 *对数与反对数
 *    gf_exp[n]  =a^n     //exp得到具体值
 *    gf_log[a^n]=n       //log得到系数
*/
// static const uint8_t gf16_exp[16] = {
//     0x01, 0x02, 0x04, 0x08, 0x03, 0x06, 0x0c, 0x0b,
//     0x05, 0x0a, 0x07, 0x0e, 0x0f, 0x0d, 0x09, 0x01
// };

/*multiplication in GF 乘法
 * params @ x , y
 * return x * y
 * EXP:
 *     a^x * a^y =a^(x+y)
 */
uint8_t gf_mul(uint8_t x,uint8_t y){
    if (x==0 || y==0)
        return 0;
    return gf_exp[(gf_log[x] + gf_log[y])%255];
}

/*除法*/
//def gf_div(x,y):
//if y==0:
//raise ZeroDivisionError()
//if x==0:
//return 0
//return gf_exp[(gf_log[x] + 255 - gf_log[y]) % 255]

/* exponentiation operator 幂
 * params @  x , power
 * return x^power
 * EXP:
 *     (a^n)^x =a^(x+n)
 * */
uint8_t gf_pow(uint8_t x , int power){
    return gf_exp[(gf_log[x] * power) % 255];
}

/*取反*/
//def gf_inverse(x):
//return gf_exp[255 - gf_log[x]] # gf_inverse(x) == gf_div(1, x)

/*多项式乘常数，每项都乘以x*/
//def gf_poly_scale(p,x):
//r = [0] * len(p)
//for i in range(0, len(p)):
//r[i] = gf_mul(p[i], x)
//return r

/*多项式加法，先拿一个赋满值，然后再拿另一个在对应的位置上进行加（亦或）*/
//def gf_poly_add(p,q):
//r = [0] * max(len(p),len(q))
//for i in range(0,len(p)):
//r[i+len(r)-len(p)] = p[i]
//for i in range(0,len(q)):
//r[i+len(r)-len(q)] ^= q[i]
//return r


enum OUTPUT{
    HEX,OCT,ALPHA
};
/* show_poly
 * params : const Mat& p(polynomial),OUTPUT o(output pattern)
 * return :output string
 * */
std::string show_poly(const Mat& p,OUTPUT o=HEX){
    std::string s;
    for(int i=0;i<p.cols;i++){
        char tmp[10];
        switch (o){
            case HEX:
                sprintf(tmp,"%02X",(int)p.ptr(0)[i]);//
                break;
            case OCT:
                sprintf(tmp,"%d",(int)p.ptr(0)[i]);
                break;
            case ALPHA:
                sprintf(tmp,"%d",gf_log[(int)p.ptr(0)[i]]%255);//%02X
                break;
        }
        s=" "+s;
        s=tmp+s;
    }
    s=s+'\n';
    return  s;
}


/* multiplication between two polys
 * params @ poly p , poly q
 * return @ result poly = p * q
 * 多项式乘法 乘法=加法
 *  先拿一个赋满值，然后再拿另一个在对应的位置上进行加（亦或）*/
/*
       10001001
    *  00101010
 ---------------
      10001001
^   10001001
^ 10001001
 ---------------
  1010001111010*/
Mat gf_poly_mul(const Mat &p,const Mat &q){
//    std::cout<<"p : "<<show_poly(p)<<"\nq : "<<show_poly(q)<<std::endl;
//    std::cout<<"p.length :"<< p.coeffi[0]<<p.coeffi[1]<<p.coeffi[2]<<endl;

/* multiplication == addition among items*/
    Mat r(1,p.cols+q.cols-1,CV_8UC1,Scalar(0));
    int len_p=p.cols;
    int len_q=q.cols;
    for(int j = 0; j<len_q;j++) {
        for (int i = 0; i < len_p; i++) {
//            std::cout << "(p_i : " << p.coeffi[i] << " q_j : " << q.coeffi[j] ;
//            std::cout << " )= " << D2B(gf_mul(p.coeffi[i]<<i, q.coeffi[j]<<j)) << std::endl;
            r.ptr(0)[i+j] ^= gf_mul(p.ptr(0)[i], q.ptr(0)[j]);
        }
        show_poly(r);
    }
    return r;
}



/* gf_poly_eval :
 *      evaluate a polynomial at a particular value of x, producing a scalar result
 * params @ poly 15bit format_Info,  uint8_t x(a scalar)
 * return @ result
 * 多项式估值
 * using the Horner's method here:
 * 01 x4 + 0f x3 + 36 x2 + 78 x + 40 = (((01 x + 0f) x + 36) x + 78) x + 40
 * doing this by simple addition and multiplication
 * */
uint8_t gf_poly_eval(const Mat& poly,uint8_t x){
    /*poly 从高项开始,mat的下标为次数*/
    /*Note the calculation begins at the high times of items,
     * That's to say , start from the large index in Mat
     * */
    int index=poly.cols-1;
    uint8_t y=poly.ptr(0)[index];
//    cout<<"x : \n"<<(int)x<<endl;
//    cout<<"gf_poly_eval : \n"<<(int)y<<" ";
    for(int i =index-1;i>=0;i--){
        y = gf_mul(x,y) ^ poly.ptr(0)[i];
//        cout<<(int)poly.ptr(0)[i]<<" ";
    }
    /*normal method*/
//    cout<<endl;
//    uint8_t y2=0;
//    for(int i =0;i<poly.cols;i++){
//        y2^=gf_mul(poly.ptr(0)[i],gf_pow(x,i));
//        if(poly.ptr(0)[i])//poly.cols-1-
//            cout<<(int)x<<"^"<<setw(3)<<i<<" : "<<setw(10)<<D2B(gf_pow(x,i))<<endl;
//    }
//    cout<<"y1 : "<<(int)y<<" y2 : "<<(int)y2<<endl;
    return y;
}

/*生成多项式*/
/* The generator can be implement by doing multiplication in GF.
 * params @ int nsym(number of the length)
 * return @ result
 * 多项式估值
 * g4(x) = (x - α0) (x - α1) (x - α2) (x - α3)
 * */
Mat rs_generator_poly(int nsym) {
    /*初始化第一项
     *initialize the first item*/
    Mat g(1,nsym+1,CV_8UC1,Scalar(0));
    g.ptr(0)[0]=1;
    g.ptr(0)[1]=1;
    for(int i =1;i<nsym;i++){
        /*之后的项
         *get n as item (x - α0)*/
        Mat n(1,2,CV_8UC1,Scalar(0));
        n.ptr(0)[0]=gf_pow(2, i);
        n.ptr(0)[1]=1;
        //cout<<n.coeffi[1]<<" "<<n.coeffi[0]<<endl;
        Mat r=gf_poly_mul(g, n);
        for(int j=0;j<r.cols;j++){
            g.ptr(0)[j]=r.ptr(0)[j];
        }
    }
    cout<<"rs_generator_poly : "<<show_poly(g)<<endl;
    return g;
}

/*gf_poly_div:
 *   This function is for getting the ECC for the data string ,which is implemented by doing a poly division.
 * params @ const Mat& dividend,const Mat& divisor
 * return @ ECC code
 * 除法获得纠错码
 *                             12 da df
 *               -----------------------
 *01 0f 36 78 40 ) 12 34 56 00 00 00 00
 *               ^ 12 ee 2b 23 f4
 *              -------------------------
 *                   da 7d 23 f4 00
 *                 ^ da a2 85 79 84
 *                  ---------------------
 *                      df a6 8d 84 00
 *                    ^ df 91 6b fc d9
 *                    -------------------
 *                         37 e6 78 d9
 */
Mat gf_poly_div(const Mat& dividend,const Mat& divisor) {
    /* Note that the processing starts from the item with high number of times, so item [total-i] is processed for the i-th round */
    /*注意 这里是从次数高的项开始处理，所以第i次 要处理第(total-i)项*/
    int times=dividend.cols-(divisor.cols-1);
    int dividend_len=dividend.cols-1;
    int divisor_len=divisor.cols-1;
    /*Mat.ptr(0)[i] stores the coeffient of the x^i*/
    Mat r=dividend.clone();
    for(int i =0;i<times;i++){
        uint16_t coef=r.ptr(0)[dividend_len-i];
        //cout<<"第"<<i<<"次:"<<hex<<coef<<endl;
        if(coef!=0){
            for (int j = 0; j < divisor.cols; ++j) {
                if(divisor.ptr(0)[divisor_len-j]!=0){
                    r.ptr(0)[dividend_len-i-j]^=gf_mul(divisor.ptr(0)[divisor_len-j], coef);
                }
            }
            //cout<<"r : "<<show_poly(r)<<endl;
        }
    }
    //cout<<"whole : \n"<<r<<endl;
    Mat ecc=r(Range(0,1),Range(0,10)).clone();
    //cout<<"ecc : \n"<<ecc<<endl;
    return ecc;
}

/*format_syndromes:
 *   test if the format corruption occurs
 *   evaluate the poly,stores in s[i] ,if all s[i] equals to 0,then no corruption.
 *
 *   Because we use BCD(hamming detection) here , so just simply doing a poly division.
 *   And check the value of the remainder if it is zero ,then no error.
 *
 * params @ uint16_t u, uint8_t *s,uint8_t *my_s, const Mat& my_f_ret()
 * return @ nonzero
 *
 * 有FORMAT_SYNDROMES个等式，计算每个等式的值，将之存放在s中
 * 如果全为0 那么无错误
 * 计算方法：
 *      x=2^0 2^1 2^2 ... 2^有(FORMAT_SYNDROMES-1) 带入多项式，计算结果值
 * */

static int format_syndromes(uint16_t u, uint8_t *s,uint8_t *my_s, const Mat& my_f_ret)
{
    //u=format
	int i;
	int nonzero = 0;
    int my_nonzero = 0;

    memset(s, 0, MAX_POLY);
    cout<<"D2B(u) : \n"<<D2B(u)<<endl;

    /*the original method*/
    /*第i个等式*/
	for (i = 0; i < FORMAT_SYNDROMES; i++) {
		int j;
		s[i] = 0;my_s[i]=0;
		/*获取每一项*/
		for (j = 0; j < FORMAT_BITS; j++) {
            if (u & (1 << j)) {
                s[i] ^= gf16_exp[((i + 1) * j) % 15];
            }
        }
        //cout<<endl;

/* WRONG !! WHY CAN'T I USE SYNDROMES TO FORMAT???*/
        //my_s[i]=gf_poly_eval(tmp, gf_pow(2,i));
		if (s[i])
			nonzero = 1;
	}

    /*my method*/
	/*format 解码用的是BCD*/
	/*the format generator*/
	Mat divisor=(Mat_<uint8_t >(1,11)<<1,0,1,0,0,1,1,0,1,1,1);
    cout<<"dividend:\n"<<my_f_ret<<endl;
    cout<<"divisor:\n"<<divisor<<endl;
    /*doing a poly division*/
    Mat remainder=gf_poly_div(my_f_ret,divisor);
    /*check the remainder*/
    std::cout<<"div : "<<show_poly(remainder)<<endl;

//	std::cout<<"syndromes:\n";
//    for (i = 0; i < FORMAT_SYNDROMES; i++) {
//        std::cout<<(int)s[i]<<" ";
//    }
//    std::cout<<std::endl;

    for(int i=0;i<remainder.cols;i++){
        if(remainder.ptr(0)[i]!=0)
            my_nonzero=1;
    }
    cout<<"nonzero : "<<nonzero<<"\nmy_nonzero : "<<my_nonzero<<endl;


	return nonzero;
}
/* some test codes
 * */
//	Mat p=(Mat_<uint8_t >(1,7)<<0,0,0,0,0x56,0x34,0x12);
//    Mat q=(Mat_<uint8_t >(1,5)<<0x40,0x78,0x36,0x0f,0x01);
//
//    std::cout<<"\np : "<<show_poly(p)<<std::endl;
//    std::cout<<"q : "<<show_poly(q)<<std::endl;

//    Mat r =gf_poly_mul(p,q);
//    std::cout<<"r : "<<show_poly(r)<<std::endl;

/* process in encoding
 * the code for generating the ECC of format
 * 生成format的ecc过程*/
//    /*padding for the dividend*/
//    Mat fomat(1,5,CV_8UC1,Scalar(0));//my_f_ret(1,Range(1,5));
//    for(int i =14;i>=10;i--){
//        fomat.ptr(0)[i-10]=my_f_ret.ptr(0)[14-i];
//    }
//    /*先入的小，打印在前面（真正顺序请用show_poly）*/
//    Mat fomat=(Mat_<uint8_t >(1,5)<<1,0,1,0,0);
//    /*项数平移*/
//    Mat pad(1,10,CV_8UC1,Scalar(0));
//
//    Mat dividend;
//    hconcat(pad,fomat,dividend);
//    /*get generator as the divisor*/
//    //Mat divisor =rs_generator_poly(10);
//    /*生成有特定的generator*/
//    Mat divisor=(Mat_<uint8_t >(1,11)<<1,1,1,0,1,1,0,0,1,0,1);
//    cout<<"dividend:\n"<<dividend<<endl;
//    cout<<"divisor:\n"<<divisor<<endl;
//
//    std::cout<<"div : "<<show_poly(gf_poly_div(dividend,divisor))<<endl;
//



/*hamming_weight:
 *      get the distance by counting the number of 1
 * params @ uint16_t x(input the XOR of two binary string)
 * return @ the distance
 * 异或后1的个数为汉明距离*/
int hamming_weight(uint16_t x){
    int weight=0;
    while(x>0){
        weight += x & 1;
        x >>= 1;
    }
    return weight;
}

/*hamming_detect:
 *      find the best matched string
 * params @ uint16_t fmt(input format)
 * return @ the index of matched component in the look-up table
 * 异或后1的个数为汉明距离*/

int hamming_detect(uint16_t fmt){
    int best_fmt = -1;
    int best_dist = 15;
    int test_dist;
    int index=0;
    for(;index<32;index++){
        /*取出*/
        uint16_t test_code=after_mask_format[index];
        test_dist=hamming_weight(fmt ^ test_code);
        /*更新
         * find the smallest distance*/
        if (test_dist < best_dist){
            best_dist = test_dist;
            best_fmt = index;
        }
        /*出现两个相同的距离，说明不对！
         * can't match two components*/
        else if(test_dist == best_dist) {
            best_fmt = -1;
        }
    }
    cout<<"best_fmt : "<<best_fmt<<" best_dist : "<<best_dist<<endl;
    return best_fmt;
}

/*correct_format:
 *  error correct
 *  params @ uint16_t *f_ret,const Mat& my_f_ret(my format info)
 *  return @
 * 格式码纠错*/
static quirc_decode_error_t correct_format(uint16_t *f_ret,const Mat& my_f_ret)
{
	/*original format 原序列*/
    std::cout<<"@@@correct_format@@@"<<endl;
	uint16_t u = *f_ret;

	int i;
	uint8_t s[MAX_POLY];
    uint8_t my_s[MAX_POLY];
	uint8_t sigma[MAX_POLY];

    /*ori: 110101100100011*/
    /*adjust several bits to check the correcting ability*/
    u^=4;u^=8;

    Mat tmp=my_f_ret.clone();
    tmp.ptr(0)[my_f_ret.cols-1-2]=1;tmp.ptr(0)[my_f_ret.cols-1-3]=1;
    cout<<"show_poly : \n"<<tmp<<endl;

    /*convert mat to uint16_t for easy bitwise operation*/
    uint16_t format_info=0;
    for(int i=0;i<my_f_ret.cols;i++){
        format_info |= tmp.ptr(0)[my_f_ret.cols-1-i]<<i;
    }
    cout<<"u       :"<<(int)u<<"\nformat_info:"<<(int)format_info<<endl;

    /*the original method*/
	/* Evaluate U (received codeword) at each of alpha_1 .. alpha_6
	 * to get S_1 .. S_6 (but we index them from 0).
	 */
	if (!format_syndromes(u, s,my_s,tmp))
		return QUIRC_SUCCESS;


    /*EEC ways*/
    berlekamp_massey(s, FORMAT_SYNDROMES, &gf16, sigma);
	/* Now, find the roots of the polynomial */
	for (i = 0; i < 15; i++)
		if (!poly_eval(sigma, gf16_exp[15 - i], &gf16))
			u ^= (1 << i);

    /*my method*/
    /*BCD ways*/
    int format_index=hamming_detect(format_info);

	if (format_index==-1||format_syndromes(u, s,my_s,tmp))
		return QUIRC_ERROR_FORMAT_ECC;

	cout<<"format_index : "<<D2B(after_mask_format[format_index]^0x5412)<<endl;

    cv::waitKey();
	*f_ret = u;
	return QUIRC_SUCCESS;
}

/************************************************************************
 * Decoder algorithm
 */

struct datastream {
	uint8_t		raw[QUIRC_MAX_PAYLOAD];
	int		data_bits;
	int		ptr;

	uint8_t         data[QUIRC_MAX_PAYLOAD];
};

static inline int grid_bit(const struct quirc_code *code, int x, int y)
{
	int p = y * code->size + x;

	return (code->cell_bitmap[p >> 3] >> (p & 7)) & 1;
}

#define FORMAT_LENGTH 15
/*读数据?
 * int which ??
 * 	*/
static quirc_decode_error_t read_format(const struct quirc_code *code,const struct QR_Info*info,
					struct quirc_data *data, int which)
{
	/*版本号小于等于6 version<=6*/
	int i;
	uint16_t format = 0;
    uint16_t my_format = 0;

    Mat mat_format(1,FORMAT_LENGTH,CV_8UC1,Scalar(0));

    uint16_t fdata;
	quirc_decode_error_t err;
    quirc_decode_error_t my_err;
	/*左下和右上的第一个格式码*/
	std::cout<<"format:";
	if (which) {
		/*下方的0-7*/
		for (i = 0; i < 7; i++){
		/*读出(code->size - 1 - i,8)*/
			format = (format << 1) |
				grid_bit(code, 8, code->size - 1 - i);
            my_format = (my_format << 1) |
                    (info->cell_bitmap.ptr<uint8_t>(code->size - 1 - i)[8]==0);
            std::cout<<grid_bit(code, 8, code->size - 1 - i)<<" ";

            mat_format.ptr(0)[i]=(info->cell_bitmap.ptr<uint8_t>(code->size - 1 - i)[8]==0);
        }
		/*右上方的7-14*/
		for (i = 0; i < 8; i++){
			format = (format << 1) |
				grid_bit(code, code->size - 8 + i, 8);

            my_format = (my_format << 1) |
                    (info->cell_bitmap.ptr<uint8_t>(8)[code->size - 8 + i]==0);

            mat_format.ptr(0)[7+i]=(info->cell_bitmap.ptr<uint8_t>(8)[code->size - 8 + i]==0);

            std::cout<<grid_bit(code, code->size - 8 + i, 8)<<" ";
        }
	} else
	/*左上方的第二个格式码*/ 
	{
		static const int xs[FORMAT_LENGTH] = {
			8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0
		};
		static const int ys[FORMAT_LENGTH] = {
			0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8
		};

		for (i = FORMAT_LENGTH-1; i >= 0; i--) {
            format = (format << 1) | grid_bit(code, xs[i], ys[i]);

            my_format = (my_format << 1) |
                    (info->cell_bitmap.ptr<uint8_t>(ys[i])[xs[i]]==0);

            mat_format.ptr(0)[FORMAT_LENGTH-1-i]=(info->cell_bitmap.ptr<uint8_t>(ys[i])[xs[i]]==0);


            std::cout << grid_bit(code, xs[i], ys[i]);
        }
		std::cout<<std::endl;
	}
    //cout<<"mat_format : \n"<<mat_format<<endl;

    //std::cout <<"\nmy_format:"<<my_format<<"\nmy_string:"<<f<<std::endl;

	/*unmask*/
	/*101010000010010*/
	Mat mask=(Mat_<uint8_t >(1,FORMAT_LENGTH)<<1,0,1,0,1,0,0,0,0,0,1,0,0,1,0);
    //cout<<"MASK : \n"<<mask<<endl;

//	for(int i=0;i<FORMAT_LENGTH;i++){
//	    mat_format.ptr(0)[i]^=mask.ptr(0)[i];
//	}

	format ^= 0x5412;


    std::cout <<"ori:"<<D2B(format)<<"\n"<<"adj:"<<D2B(my_format)<<std::endl;

    cout<<"FINAL : \n"<<mat_format<<endl;

	/*ECC 纠错*/
	err = correct_format(&format,mat_format);

    my_err = correct_format(&my_format,mat_format);
	if (err)
		return err;

    my_format ^=0x5412;
	/*EC level （1-2）+Mask(3-5) + EC for this string( 6-15) */
	/*去除纠错位*/
	fdata = format >> 10;
	/*纠错等级*/
	data->ecc_level = fdata >> 3;
	/*掩膜类型*/
	data->mask = fdata & 7;

	return QUIRC_SUCCESS;
}

static int mask_bit(int mask, int i, int j)
{
	switch (mask) {
	case 0: return !((i + j) % 2);
	case 1: return !(i % 2);
	case 2: return !(j % 3);
	case 3: return !((i + j) % 3);
	case 4: return !(((i / 2) + (j / 3)) % 2);
	case 5: return !((i * j) % 2 + (i * j) % 3);
	case 6: return !(((i * j) % 2 + (i * j) % 3) % 2);
	case 7: return !(((i * j) % 3 + (i + j) % 2) % 2);
	}

	return 0;
}

static int reserved_cell(int version, int i, int j)
{
	const struct quirc_version_info *ver = &quirc_version_db[version];
	int size = version * 4 + 17;
	int ai = -1, aj = -1, a;

	/* Finder + format: top left */
	if (i < 9 && j < 9)
		return 1;

	/* Finder + format: bottom left */
	if (i + 8 >= size && j < 9)
		return 1;

	/* Finder + format: top right */
	if (i < 9 && j + 8 >= size)
		return 1;

	/* Exclude timing patterns */
	if (i == 6 || j == 6)
		return 1;

	/* Exclude version info, if it exists. Version info sits adjacent to
	 * the top-right and bottom-left finders in three rows, bounded by
	 * the timing pattern.
	 */
	if (version >= 7) {
		if (i < 6 && j + 11 >= size)
			return 1;
		if (i + 11 >= size && j < 6)
			return 1;
	}

	/* Exclude alignment patterns */
	for (a = 0; a < QUIRC_MAX_ALIGNMENT && ver->apat[a]; a++) {
		int p = ver->apat[a];

		if (abs(p - i) < 3)
			ai = a;
		if (abs(p - j) < 3)
			aj = a;
	}

	if (ai >= 0 && aj >= 0) {
		a--;
		if (ai > 0 && ai < a)
			return 1;
		if (aj > 0 && aj < a)
			return 1;
		if (aj == a && ai == a)
			return 1;
	}

	return 0;
}

static void read_bit(const struct quirc_code *code,
		     struct quirc_data *data,
		     struct datastream *ds, int i, int j)
{
	int bitpos = ds->data_bits & 7;
	int bytepos = ds->data_bits >> 3;
	int v = grid_bit(code, j, i);

	if (mask_bit(data->mask, i, j))
		v ^= 1;

	if (v)
		ds->raw[bytepos] |= (0x80 >> bitpos);

	ds->data_bits++;
}

static void read_data(const struct quirc_code *code,
		      struct quirc_data *data,
		      struct datastream *ds)
{
	int y = code->size - 1;
	int x = code->size - 1;
	int dir = -1;

	while (x > 0) {
		if (x == 6)
			x--;

		if (!reserved_cell(data->version, y, x))
			read_bit(code, data, ds, y, x);

		if (!reserved_cell(data->version, y, x - 1))
			read_bit(code, data, ds, y, x - 1);

		y += dir;
		if (y < 0 || y >= code->size) {
			dir = -dir;
			x -= 2;
			y += dir;
		}
	}
}

static quirc_decode_error_t codestream_ecc(struct quirc_data *data,
					   struct datastream *ds)
{
	const struct quirc_version_info *ver =
		&quirc_version_db[data->version];
	const struct quirc_rs_params *sb_ecc = &ver->ecc[data->ecc_level];
	struct quirc_rs_params lb_ecc;
	const int lb_count =
	    (ver->data_bytes - sb_ecc->bs * sb_ecc->ns) / (sb_ecc->bs + 1);
	const int bc = lb_count + sb_ecc->ns;
	const int ecc_offset = sb_ecc->dw * bc + lb_count;
	int dst_offset = 0;
	int i;

	memcpy(&lb_ecc, sb_ecc, sizeof(lb_ecc));
	lb_ecc.dw++;
	lb_ecc.bs++;

	for (i = 0; i < bc; i++) {
		uint8_t *dst = ds->data + dst_offset;
		const struct quirc_rs_params *ecc =
		    (i < sb_ecc->ns) ? sb_ecc : &lb_ecc;
		const int num_ec = ecc->bs - ecc->dw;
		quirc_decode_error_t err;
		int j;

		for (j = 0; j < ecc->dw; j++)
			dst[j] = ds->raw[j * bc + i];
		for (j = 0; j < num_ec; j++)
			dst[ecc->dw + j] = ds->raw[ecc_offset + j * bc + i];

		err = correct_block(dst, ecc);
		if (err)
			return err;

		dst_offset += ecc->dw;
	}

	ds->data_bits = dst_offset * 8;

	return QUIRC_SUCCESS;
}

static inline int bits_remaining(const struct datastream *ds)
{
	return ds->data_bits - ds->ptr;
}

static int take_bits(struct datastream *ds, int len)
{
	int ret = 0;

	while (len && (ds->ptr < ds->data_bits)) {
		uint8_t b = ds->data[ds->ptr >> 3];
		int bitpos = ds->ptr & 7;

		ret <<= 1;
		if ((b << bitpos) & 0x80)
			ret |= 1;

		ds->ptr++;
		len--;
	}

	return ret;
}

static int numeric_tuple(struct quirc_data *data,
			 struct datastream *ds,
			 int bits, int digits)
{
	int tuple;
	int i;

	if (bits_remaining(ds) < bits)
		return -1;

	tuple = take_bits(ds, bits);

	for (i = digits - 1; i >= 0; i--) {
		data->payload[data->payload_len + i] = tuple % 10 + '0';
		tuple /= 10;
	}

	data->payload_len += digits;
	return 0;
}

static quirc_decode_error_t decode_numeric(struct quirc_data *data,
					   struct datastream *ds)
{
	int bits = 14;
	int count;

	if (data->version < 10)
		bits = 10;
	else if (data->version < 27)
		bits = 12;

	count = take_bits(ds, bits);
	if (data->payload_len + count + 1 > QUIRC_MAX_PAYLOAD)
		return QUIRC_ERROR_DATA_OVERFLOW;

	while (count >= 3) {
		if (numeric_tuple(data, ds, 10, 3) < 0)
			return QUIRC_ERROR_DATA_UNDERFLOW;
		count -= 3;
	}

	if (count >= 2) {
		if (numeric_tuple(data, ds, 7, 2) < 0)
			return QUIRC_ERROR_DATA_UNDERFLOW;
		count -= 2;
	}

	if (count) {
		if (numeric_tuple(data, ds, 4, 1) < 0)
			return QUIRC_ERROR_DATA_UNDERFLOW;
		count--;
	}

	return QUIRC_SUCCESS;
}

static int alpha_tuple(struct quirc_data *data,
		       struct datastream *ds,
		       int bits, int digits)
{
	int tuple;
	int i;

	if (bits_remaining(ds) < bits)
		return -1;

	tuple = take_bits(ds, bits);

	for (i = 0; i < digits; i++) {
		static const char *alpha_map =
			"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

		data->payload[data->payload_len + digits - i - 1] =
			alpha_map[tuple % 45];
		tuple /= 45;
	}

	data->payload_len += digits;
	return 0;
}

static quirc_decode_error_t decode_alpha(struct quirc_data *data,
					 struct datastream *ds)
{
	int bits = 13;
	int count;

	if (data->version < 10)
		bits = 9;
	else if (data->version < 27)
		bits = 11;

	count = take_bits(ds, bits);
	if (data->payload_len + count + 1 > QUIRC_MAX_PAYLOAD)
		return QUIRC_ERROR_DATA_OVERFLOW;

	while (count >= 2) {
		if (alpha_tuple(data, ds, 11, 2) < 0)
			return QUIRC_ERROR_DATA_UNDERFLOW;
		count -= 2;
	}

	if (count) {
		if (alpha_tuple(data, ds, 6, 1) < 0)
			return QUIRC_ERROR_DATA_UNDERFLOW;
		count--;
	}

	return QUIRC_SUCCESS;
}

static quirc_decode_error_t decode_byte(struct quirc_data *data,
					struct datastream *ds)
{
	int bits = 16;
	int count;
	int i;

	if (data->version < 10)
		bits = 8;

	count = take_bits(ds, bits);
	if (data->payload_len + count + 1 > QUIRC_MAX_PAYLOAD)
		return QUIRC_ERROR_DATA_OVERFLOW;
	if (bits_remaining(ds) < count * 8)
		return QUIRC_ERROR_DATA_UNDERFLOW;

	for (i = 0; i < count; i++)
		data->payload[data->payload_len++] = take_bits(ds, 8);

	return QUIRC_SUCCESS;
}

static quirc_decode_error_t decode_kanji(struct quirc_data *data,
					 struct datastream *ds)
{
	int bits = 12;
	int count;
	int i;

	if (data->version < 10)
		bits = 8;
	else if (data->version < 27)
		bits = 10;

	count = take_bits(ds, bits);
	if (data->payload_len + count * 2 + 1 > QUIRC_MAX_PAYLOAD)
		return QUIRC_ERROR_DATA_OVERFLOW;
	if (bits_remaining(ds) < count * 13)
		return QUIRC_ERROR_DATA_UNDERFLOW;

	for (i = 0; i < count; i++) {
		int d = take_bits(ds, 13);
		int msB = d / 0xc0;
		int lsB = d % 0xc0;
		int intermediate = (msB << 8) | lsB;
		uint16_t sjw;

		if (intermediate + 0x8140 <= 0x9ffc) {
			/* bytes are in the range 0x8140 to 0x9FFC */
			sjw = intermediate + 0x8140;
		} else {
			/* bytes are in the range 0xE040 to 0xEBBF */
			sjw = intermediate + 0xc140;
		}

		data->payload[data->payload_len++] = sjw >> 8;
		data->payload[data->payload_len++] = sjw & 0xff;
	}

	return QUIRC_SUCCESS;
}

static quirc_decode_error_t decode_eci(struct quirc_data *data,
				       struct datastream *ds)
{
	if (bits_remaining(ds) < 8)
		return QUIRC_ERROR_DATA_UNDERFLOW;

	data->eci = take_bits(ds, 8);

	if ((data->eci & 0xc0) == 0x80) {
		if (bits_remaining(ds) < 8)
			return QUIRC_ERROR_DATA_UNDERFLOW;

		data->eci = (data->eci << 8) | take_bits(ds, 8);
	} else if ((data->eci & 0xe0) == 0xc0) {
		if (bits_remaining(ds) < 16)
			return QUIRC_ERROR_DATA_UNDERFLOW;

		data->eci = (data->eci << 16) | take_bits(ds, 16);
	}

	return QUIRC_SUCCESS;
}

static quirc_decode_error_t decode_payload(struct quirc_data *data,
					   struct datastream *ds)
{
	while (bits_remaining(ds) >= 4) {
		quirc_decode_error_t err = QUIRC_SUCCESS;
		int type = take_bits(ds, 4);

		switch (type) {
		case QUIRC_DATA_TYPE_NUMERIC:
			err = decode_numeric(data, ds);
			break;

		case QUIRC_DATA_TYPE_ALPHA:
			err = decode_alpha(data, ds);
			break;

		case QUIRC_DATA_TYPE_BYTE:
			err = decode_byte(data, ds);
			break;

		case QUIRC_DATA_TYPE_KANJI:
			err = decode_kanji(data, ds);
			break;

		case 7:
			err = decode_eci(data, ds);
			break;

		default:
			goto done;
		}

		if (err)
			return err;

		if (!(type & (type - 1)) && (type > data->data_type))
			data->data_type = type;
	}
done:

	/* Add nul terminator to all payloads */
	if (data->payload_len >= (int) sizeof(data->payload))
		data->payload_len--;
	data->payload[data->payload_len] = 0;

	return QUIRC_SUCCESS;
}

quirc_decode_error_t quirc_decode(const struct quirc_code *code,const struct QR_Info*info,
				  struct quirc_data *data)
{
	quirc_decode_error_t err;
	struct datastream ds;

	if ((code->size - 17) % 4)
		return QUIRC_ERROR_INVALID_GRID_SIZE;

	memset(data, 0, sizeof(*data));
	memset(&ds, 0, sizeof(ds));

	/*版本号*/
	data->version = (code->size - 17) / 4;

	if (data->version < 1 ||
	    data->version > QUIRC_MAX_VERSION)
		return QUIRC_ERROR_INVALID_VERSION;

	/* Read format information -- try both locations */
	err = read_format(code, info,data, 0);
	if (err)
		err = read_format(code,info, data, 1);
	if (err)
		return err;
    std::cout<<std::endl;

	read_data(code, data, &ds);
	err = codestream_ecc(data, &ds);
	if (err)
		return err;

	err = decode_payload(data, &ds);
	if (err)
		return err;

	return QUIRC_SUCCESS;
}
