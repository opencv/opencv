/*
 * copyright (c) 2005 Michael Niedermayer <michaelni@gmx.at>
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

#ifndef AVUTIL_INTFLOAT_READWRITE_H
#define AVUTIL_INTFLOAT_READWRITE_H

#include <msc_stdint.h>
#include "attributes.h"

/* IEEE 80 bits extended float */
typedef struct AVExtFloat  {
    uint8_t exponent[2];
    uint8_t mantissa[8];
} AVExtFloat;

double av_int2dbl(int64_t v) av_const;
float av_int2flt(int32_t v) av_const;
double av_ext2dbl(const AVExtFloat ext) av_const;
int64_t av_dbl2int(double d) av_const;
int32_t av_flt2int(float d) av_const;
AVExtFloat av_dbl2ext(double d) av_const;

#endif /* AVUTIL_INTFLOAT_READWRITE_H */
