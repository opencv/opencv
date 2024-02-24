/*
 * zlib_compress.c - compress with a zlib wrapper
 *
 * Copyright 2016 Eric Biggers
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "deflate_compress.h"
#include "zlib_constants.h"

LIBDEFLATEAPI size_t
libdeflate_zlib_compress(struct libdeflate_compressor *c,
			 const void *in, size_t in_nbytes,
			 void *out, size_t out_nbytes_avail)
{
	u8 *out_next = out;
	u16 hdr;
	unsigned compression_level;
	unsigned level_hint;
	size_t deflate_size;

	if (out_nbytes_avail <= ZLIB_MIN_OVERHEAD)
		return 0;

	/* 2 byte header: CMF and FLG  */
	hdr = (ZLIB_CM_DEFLATE << 8) | (ZLIB_CINFO_32K_WINDOW << 12);
	compression_level = libdeflate_get_compression_level(c);
	if (compression_level < 2)
		level_hint = ZLIB_FASTEST_COMPRESSION;
	else if (compression_level < 6)
		level_hint = ZLIB_FAST_COMPRESSION;
	else if (compression_level < 8)
		level_hint = ZLIB_DEFAULT_COMPRESSION;
	else
		level_hint = ZLIB_SLOWEST_COMPRESSION;
	hdr |= level_hint << 6;
	hdr |= 31 - (hdr % 31);

	put_unaligned_be16(hdr, out_next);
	out_next += 2;

	/* Compressed data  */
	deflate_size = libdeflate_deflate_compress(c, in, in_nbytes, out_next,
					out_nbytes_avail - ZLIB_MIN_OVERHEAD);
	if (deflate_size == 0)
		return 0;
	out_next += deflate_size;

	/* ADLER32  */
	put_unaligned_be32(libdeflate_adler32(1, in, in_nbytes), out_next);
	out_next += 4;

	return out_next - (u8 *)out;
}

LIBDEFLATEAPI size_t
libdeflate_zlib_compress_bound(struct libdeflate_compressor *c,
			       size_t in_nbytes)
{
	return ZLIB_MIN_OVERHEAD +
	       libdeflate_deflate_compress_bound(c, in_nbytes);
}
