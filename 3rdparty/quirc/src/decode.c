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

#include <quirc_internal.h>

#include <string.h>
#include <stdlib.h>

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

#define FORMAT_MAX_ERROR        3
#define FORMAT_SYNDROMES        (FORMAT_MAX_ERROR * 2)
#define FORMAT_BITS             15

static int format_syndromes(uint16_t u, uint8_t *s)
{
	int i;
	int nonzero = 0;

	memset(s, 0, MAX_POLY);

	for (i = 0; i < FORMAT_SYNDROMES; i++) {
		int j;

		s[i] = 0;
		for (j = 0; j < FORMAT_BITS; j++)
			if (u & (1 << j))
				s[i] ^= gf16_exp[((i + 1) * j) % 15];

		if (s[i])
			nonzero = 1;
	}

	return nonzero;
}

static quirc_decode_error_t correct_format(uint16_t *f_ret)
{
	uint16_t u = *f_ret;
	int i;
	uint8_t s[MAX_POLY];
	uint8_t sigma[MAX_POLY];

	/* Evaluate U (received codeword) at each of alpha_1 .. alpha_6
	 * to get S_1 .. S_6 (but we index them from 0).
	 */
	if (!format_syndromes(u, s))
		return QUIRC_SUCCESS;

	berlekamp_massey(s, FORMAT_SYNDROMES, &gf16, sigma);

	/* Now, find the roots of the polynomial */
	for (i = 0; i < 15; i++)
		if (!poly_eval(sigma, gf16_exp[15 - i], &gf16))
			u ^= (1 << i);

	if (format_syndromes(u, s))
		return QUIRC_ERROR_FORMAT_ECC;

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

static quirc_decode_error_t read_format(const struct quirc_code *code,
					struct quirc_data *data, int which)
{
	int i;
	uint16_t format = 0;
	uint16_t fdata;
	quirc_decode_error_t err;

	if (which) {
		for (i = 0; i < 7; i++)
			format = (format << 1) |
				grid_bit(code, 8, code->size - 1 - i);
		for (i = 0; i < 8; i++)
			format = (format << 1) |
				grid_bit(code, code->size - 8 + i, 8);
	} else {
		static const int xs[15] = {
			8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0
		};
		static const int ys[15] = {
			0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8
		};

		for (i = 14; i >= 0; i--)
			format = (format << 1) | grid_bit(code, xs[i], ys[i]);
	}

	format ^= 0x5412;

	err = correct_format(&format);
	if (err)
		return err;

	fdata = format >> 10;
	data->ecc_level = fdata >> 3;
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

quirc_decode_error_t quirc_decode(const struct quirc_code *code,
				  struct quirc_data *data)
{
	quirc_decode_error_t err;
	struct datastream ds;

	if ((code->size - 17) % 4)
		return QUIRC_ERROR_INVALID_GRID_SIZE;

	memset(data, 0, sizeof(*data));
	memset(&ds, 0, sizeof(ds));

	data->version = (code->size - 17) / 4;

	if (data->version < 1 ||
	    data->version > QUIRC_MAX_VERSION)
		return QUIRC_ERROR_INVALID_VERSION;

	/* Read format information -- try both locations */
	err = read_format(code, data, 0);
	if (err)
		err = read_format(code, data, 1);
	if (err)
		return err;

	read_data(code, data, &ds);
	err = codestream_ecc(data, &ds);
	if (err)
		return err;

	err = decode_payload(data, &ds);
	if (err)
		return err;

	return QUIRC_SUCCESS;
}

void quirc_flip(struct quirc_code *code)
{
	struct quirc_code flipped;
	unsigned int offset = 0;
	int y;
	int x;

	memset(&flipped, 0, sizeof(flipped));
	for (y = 0; y < code->size; y++) {
		for (x = 0; x < code->size; x++) {
			if (grid_bit(code, y, x)) {
				flipped.cell_bitmap[offset >> 3u] |= (1u << (offset & 7u));
			}
			offset++;
		}
	}
	memcpy(&code->cell_bitmap, &flipped.cell_bitmap, sizeof(flipped.cell_bitmap));
}
