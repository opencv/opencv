/*
 * Copyright (c) 2006 Ryan Martell. (rdm4@martellventures.com)
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

#ifndef AVUTIL_BASE64_H
#define AVUTIL_BASE64_H

#include <msc_stdint.h>

/**
 * Decodes the base64-encoded string in in and puts the decoded
 * data in out.
 *
 * @param out_size size in bytes of the out buffer, it should be at
 * least 3/4 of the length of in
 * @return the number of bytes written, or a negative value in case of
 * error
 */
int av_base64_decode(uint8_t *out, const char *in, int out_size);

/**
 * Encodes in base64 the data in in and puts the resulting string
 * in out.
 *
 * @param out_size size in bytes of the out string, it should be at
 * least ((in_size + 2) / 3) * 4 + 1
 * @param in_size size in bytes of the in buffer
 * @return the string containing the encoded data, or NULL in case of
 * error
 */
char *av_base64_encode(char *out, int out_size, const uint8_t *in, int in_size);

#endif /* AVUTIL_BASE64_H */
