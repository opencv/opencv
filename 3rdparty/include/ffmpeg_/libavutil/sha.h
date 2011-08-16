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

#ifndef AVUTIL_SHA_H
#define AVUTIL_SHA_H

#include <stdint.h>

extern const int av_sha_size;

struct AVSHA;

/**
 * Initialize SHA-1 or SHA-2 hashing.
 *
 * @param context pointer to the function context (of size av_sha_size)
 * @param bits    number of bits in digest (SHA-1 - 160 bits, SHA-2 224 or 256 bits)
 * @return        zero if initialization succeeded, -1 otherwise
 */
int av_sha_init(struct AVSHA* context, int bits);

/**
 * Update hash value.
 *
 * @param context hash function context
 * @param data    input data to update hash with
 * @param len     input data length
 */
void av_sha_update(struct AVSHA* context, const uint8_t* data, unsigned int len);

/**
 * Finish hashing and output digest value.
 *
 * @param context hash function context
 * @param digest  buffer where output digest value is stored
 */
void av_sha_final(struct AVSHA* context, uint8_t *digest);

#endif /* AVUTIL_SHA_H */
