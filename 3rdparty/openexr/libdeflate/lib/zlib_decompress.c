/*
 * zlib_decompress.c - decompress with a zlib wrapper
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

#include "lib_common.h"
#include "zlib_constants.h"

LIBDEFLATEAPI enum libdeflate_result
libdeflate_zlib_decompress_ex(struct libdeflate_decompressor *d,
			      const void *in, size_t in_nbytes,
			      void *out, size_t out_nbytes_avail,
			      size_t *actual_in_nbytes_ret,
			      size_t *actual_out_nbytes_ret)
{
	const u8 *in_next = in;
	const u8 * const in_end = in_next + in_nbytes;
	u16 hdr;
	size_t actual_in_nbytes;
	size_t actual_out_nbytes;
	enum libdeflate_result result;

	if (in_nbytes < ZLIB_MIN_OVERHEAD)
		return LIBDEFLATE_BAD_DATA;

	/* 2 byte header: CMF and FLG  */
	hdr = get_unaligned_be16(in_next);
	in_next += 2;

	/* FCHECK */
	if ((hdr % 31) != 0)
		return LIBDEFLATE_BAD_DATA;

	/* CM */
	if (((hdr >> 8) & 0xF) != ZLIB_CM_DEFLATE)
		return LIBDEFLATE_BAD_DATA;

	/* CINFO */
	if ((hdr >> 12) > ZLIB_CINFO_32K_WINDOW)
		return LIBDEFLATE_BAD_DATA;

	/* FDICT */
	if ((hdr >> 5) & 1)
		return LIBDEFLATE_BAD_DATA;

	/* Compressed data  */
	result = libdeflate_deflate_decompress_ex(d, in_next,
					in_end - ZLIB_FOOTER_SIZE - in_next,
					out, out_nbytes_avail,
					&actual_in_nbytes, actual_out_nbytes_ret);
	if (result != LIBDEFLATE_SUCCESS)
		return result;

	if (actual_out_nbytes_ret)
		actual_out_nbytes = *actual_out_nbytes_ret;
	else
		actual_out_nbytes = out_nbytes_avail;

	in_next += actual_in_nbytes;

	/* ADLER32  */
	if (libdeflate_adler32(1, out, actual_out_nbytes) !=
	    get_unaligned_be32(in_next))
		return LIBDEFLATE_BAD_DATA;
	in_next += 4;

	if (actual_in_nbytes_ret)
		*actual_in_nbytes_ret = in_next - (u8 *)in;

	return LIBDEFLATE_SUCCESS;
}

LIBDEFLATEAPI enum libdeflate_result
libdeflate_zlib_decompress(struct libdeflate_decompressor *d,
			   const void *in, size_t in_nbytes,
			   void *out, size_t out_nbytes_avail,
			   size_t *actual_out_nbytes_ret)
{
	return libdeflate_zlib_decompress_ex(d, in, in_nbytes,
					     out, out_nbytes_avail,
					     NULL, actual_out_nbytes_ret);
}
