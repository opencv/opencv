/*
 * LZO 1x decompression
 * copyright (c) 2006 Reimar Doeffinger
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVUTIL_LZO_H
#define AVUTIL_LZO_H

/**
 * @defgroup lavu_lzo LZO
 * @ingroup lavu_crypto
 *
 * @{
 */

#include <stdint.h>

/** @name Error flags returned by av_lzo1x_decode
  * @{ */
/// end of the input buffer reached before decoding finished
#define AV_LZO_INPUT_DEPLETED 1
/// decoded data did not fit into output buffer
#define AV_LZO_OUTPUT_FULL 2
/// a reference to previously decoded data was wrong
#define AV_LZO_INVALID_BACKPTR 4
/// a non-specific error in the compressed bitstream
#define AV_LZO_ERROR 8
/** @} */

#define AV_LZO_INPUT_PADDING 8
#define AV_LZO_OUTPUT_PADDING 12

/**
 * @brief Decodes LZO 1x compressed data.
 * @param out output buffer
 * @param outlen size of output buffer, number of bytes left are returned here
 * @param in input buffer
 * @param inlen size of input buffer, number of bytes left are returned here
 * @return 0 on success, otherwise a combination of the error flags above
 *
 * Make sure all buffers are appropriately padded, in must provide
 * AV_LZO_INPUT_PADDING, out must provide AV_LZO_OUTPUT_PADDING additional bytes.
 */
int av_lzo1x_decode(void *out, int *outlen, const void *in, int *inlen);

/**
 * @brief deliberately overlapping memcpy implementation
 * @param dst destination buffer; must be padded with 12 additional bytes
 * @param back how many bytes back we start (the initial size of the overlapping window), must be > 0
 * @param cnt number of bytes to copy, must be >= 0
 *
 * cnt > back is valid, this will copy the bytes we just copied,
 * thus creating a repeating pattern with a period length of back.
 */
void av_memcpy_backptr(uint8_t *dst, int back, int cnt);

/**
 * @}
 */

#endif /* AVUTIL_LZO_H */
