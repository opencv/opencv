/*
 * Copyright (C) 2007 Michael Niedermayer <michaelni@gmx.at>
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

#ifndef AVUTIL_SHA1_H
#define AVUTIL_SHA1_H

#include <msc_stdint.h>

extern const int av_sha1_size;

struct AVSHA1;

/**
 * Initializes SHA-1 hashing.
 *
 * @param context pointer to the function context (of size av_sha_size)
 * @deprecated use av_sha_init() instead
 */
void av_sha1_init(struct AVSHA1* context);

/**
 * Updates hash value.
 *
 * @param context hash function context
 * @param data    input data to update hash with
 * @param len     input data length
 * @deprecated use av_sha_update() instead
 */
void av_sha1_update(struct AVSHA1* context, const uint8_t* data, unsigned int len);

/**
 * Finishes hashing and output digest value.
 *
 * @param context hash function context
 * @param digest  buffer where output digest value is stored
 * @deprecated use av_sha_final() instead
 */
void av_sha1_final(struct AVSHA1* context, uint8_t digest[20]);

#endif /* AVUTIL_SHA1_H */
