/*
 * copyright (c) 2006 Michael Niedermayer <michaelni@gmx.at>
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

/**
 * @file
 * common internal and external API header
 */

#ifndef AVUTIL_COMMON_H
#define AVUTIL_COMMON_H

#include <ctype.h>
#include <errno.h>
#ifdef _MSC_VER
#include <msc_inttypes.h>
#else
#include <inttypes.h>
#endif
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "attributes.h"

//rounded division & shift
#define RSHIFT(a,b) ((a) > 0 ? ((a) + ((1<<(b))>>1))>>(b) : ((a) + ((1<<(b))>>1)-1)>>(b))
/* assume b>0 */
#define ROUNDED_DIV(a,b) (((a)>0 ? (a) + ((b)>>1) : (a) - ((b)>>1))/(b))
#define FFABS(a) ((a) >= 0 ? (a) : (-(a)))
#define FFSIGN(a) ((a) > 0 ? 1 : -1)

#define FFMAX(a,b) ((a) > (b) ? (a) : (b))
#define FFMAX3(a,b,c) FFMAX(FFMAX(a,b),c)
#define FFMIN(a,b) ((a) > (b) ? (b) : (a))
#define FFMIN3(a,b,c) FFMIN(FFMIN(a,b),c)

#define FFSWAP(type,a,b) do{type SWAP_tmp= b; b= a; a= SWAP_tmp;}while(0)
#define FF_ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))
#define FFALIGN(x, a) (((x)+(a)-1)&~((a)-1))

/* misc math functions */
extern const uint8_t ff_log2_tab[256];

extern const uint8_t av_reverse[256];

static inline av_const int av_log2_c(unsigned int v)
{
    int n = 0;
    if (v & 0xffff0000) {
        v >>= 16;
        n += 16;
    }
    if (v & 0xff00) {
        v >>= 8;
        n += 8;
    }
    n += ff_log2_tab[v];

    return n;
}

static inline av_const int av_log2_16bit_c(unsigned int v)
{
    int n = 0;
    if (v & 0xff00) {
        v >>= 8;
        n += 8;
    }
    n += ff_log2_tab[v];

    return n;
}

#ifdef HAVE_AV_CONFIG_H
#   include "config.h"
#   include "intmath.h"
#endif

#ifndef av_log2
#   define av_log2       av_log2_c
#endif
#ifndef av_log2_16bit
#   define av_log2_16bit av_log2_16bit_c
#endif

/**
 * Clips a signed integer value into the amin-amax range.
 * @param a value to clip
 * @param amin minimum value of the clip range
 * @param amax maximum value of the clip range
 * @return clipped value
 */
static inline av_const int av_clip(int a, int amin, int amax)
{
    if      (a < amin) return amin;
    else if (a > amax) return amax;
    else               return a;
}

/**
 * Clips a signed integer value into the 0-255 range.
 * @param a value to clip
 * @return clipped value
 */
static inline av_const uint8_t av_clip_uint8(int a)
{
    if (a&(~0xFF)) return (-a)>>31;
    else           return a;
}

/**
 * Clips a signed integer value into the 0-65535 range.
 * @param a value to clip
 * @return clipped value
 */
static inline av_const uint16_t av_clip_uint16(int a)
{
    if (a&(~0xFFFF)) return (-a)>>31;
    else             return a;
}

/**
 * Clips a signed integer value into the -32768,32767 range.
 * @param a value to clip
 * @return clipped value
 */
static inline av_const int16_t av_clip_int16(int a)
{
    if ((a+0x8000) & ~0xFFFF) return (a>>31) ^ 0x7FFF;
    else                      return a;
}

/**
 * Clips a signed 64-bit integer value into the -2147483648,2147483647 range.
 * @param a value to clip
 * @return clipped value
 */
static inline av_const int32_t av_clipl_int32(int64_t a)
{
    if ((a+0x80000000u) & ~UINT64_C(0xFFFFFFFF)) return (a>>63) ^ 0x7FFFFFFF;
    else                                         return a;
}

/**
 * Clips a float value into the amin-amax range.
 * @param a value to clip
 * @param amin minimum value of the clip range
 * @param amax maximum value of the clip range
 * @return clipped value
 */
static inline av_const float av_clipf(float a, float amin, float amax)
{
    if      (a < amin) return amin;
    else if (a > amax) return amax;
    else               return a;
}

/** Computes ceil(log2(x)).
 * @param x value used to compute ceil(log2(x))
 * @return computed ceiling of log2(x)
 */
static inline av_const int av_ceil_log2(int x)
{
    return av_log2((x - 1) << 1);
}

#define MKTAG(a,b,c,d) (a | (b << 8) | (c << 16) | (d << 24))
#define MKBETAG(a,b,c,d) (d | (c << 8) | (b << 16) | (a << 24))

/*!
 * \def GET_UTF8(val, GET_BYTE, ERROR)
 * Converts a UTF-8 character (up to 4 bytes long) to its 32-bit UCS-4 encoded form
 * \param val is the output and should be of type uint32_t. It holds the converted
 * UCS-4 character and should be a left value.
 * \param GET_BYTE gets UTF-8 encoded bytes from any proper source. It can be
 * a function or a statement whose return value or evaluated value is of type
 * uint8_t. It will be executed up to 4 times for values in the valid UTF-8 range,
 * and up to 7 times in the general case.
 * \param ERROR action that should be taken when an invalid UTF-8 byte is returned
 * from GET_BYTE. It should be a statement that jumps out of the macro,
 * like exit(), goto, return, break, or continue.
 */
#define GET_UTF8(val, GET_BYTE, ERROR)\
    val= GET_BYTE;\
    {\
        int ones= 7 - av_log2(val ^ 255);\
        if(ones==1)\
            ERROR\
        val&= 127>>ones;\
        while(--ones > 0){\
            int tmp= GET_BYTE - 128;\
            if(tmp>>6)\
                ERROR\
            val= (val<<6) + tmp;\
        }\
    }

/*!
 * \def GET_UTF16(val, GET_16BIT, ERROR)
 * Converts a UTF-16 character (2 or 4 bytes) to its 32-bit UCS-4 encoded form
 * \param val is the output and should be of type uint32_t. It holds the converted
 * UCS-4 character and should be a left value.
 * \param GET_16BIT gets two bytes of UTF-16 encoded data converted to native endianness.
 * It can be a function or a statement whose return value or evaluated value is of type
 * uint16_t. It will be executed up to 2 times.
 * \param ERROR action that should be taken when an invalid UTF-16 surrogate is
 * returned from GET_BYTE. It should be a statement that jumps out of the macro,
 * like exit(), goto, return, break, or continue.
 */
#define GET_UTF16(val, GET_16BIT, ERROR)\
    val = GET_16BIT;\
    {\
        unsigned int hi = val - 0xD800;\
        if (hi < 0x800) {\
            val = GET_16BIT - 0xDC00;\
            if (val > 0x3FFU || hi > 0x3FFU)\
                ERROR\
            val += (hi<<10) + 0x10000;\
        }\
    }\

/*!
 * \def PUT_UTF8(val, tmp, PUT_BYTE)
 * Converts a 32-bit Unicode character to its UTF-8 encoded form (up to 4 bytes long).
 * \param val is an input-only argument and should be of type uint32_t. It holds
 * a UCS-4 encoded Unicode character that is to be converted to UTF-8. If
 * val is given as a function it is executed only once.
 * \param tmp is a temporary variable and should be of type uint8_t. It
 * represents an intermediate value during conversion that is to be
 * output by PUT_BYTE.
 * \param PUT_BYTE writes the converted UTF-8 bytes to any proper destination.
 * It could be a function or a statement, and uses tmp as the input byte.
 * For example, PUT_BYTE could be "*output++ = tmp;" PUT_BYTE will be
 * executed up to 4 times for values in the valid UTF-8 range and up to
 * 7 times in the general case, depending on the length of the converted
 * Unicode character.
 */
#define PUT_UTF8(val, tmp, PUT_BYTE)\
    {\
        int bytes, shift;\
        uint32_t in = val;\
        if (in < 0x80) {\
            tmp = in;\
            PUT_BYTE\
        } else {\
            bytes = (av_log2(in) + 4) / 5;\
            shift = (bytes - 1) * 6;\
            tmp = (256 - (256 >> bytes)) | (in >> shift);\
            PUT_BYTE\
            while (shift >= 6) {\
                shift -= 6;\
                tmp = 0x80 | ((in >> shift) & 0x3f);\
                PUT_BYTE\
            }\
        }\
    }

/*!
 * \def PUT_UTF16(val, tmp, PUT_16BIT)
 * Converts a 32-bit Unicode character to its UTF-16 encoded form (2 or 4 bytes).
 * \param val is an input-only argument and should be of type uint32_t. It holds
 * a UCS-4 encoded Unicode character that is to be converted to UTF-16. If
 * val is given as a function it is executed only once.
 * \param tmp is a temporary variable and should be of type uint16_t. It
 * represents an intermediate value during conversion that is to be
 * output by PUT_16BIT.
 * \param PUT_16BIT writes the converted UTF-16 data to any proper destination
 * in desired endianness. It could be a function or a statement, and uses tmp
 * as the input byte.  For example, PUT_BYTE could be "*output++ = tmp;"
 * PUT_BYTE will be executed 1 or 2 times depending on input character.
 */
#define PUT_UTF16(val, tmp, PUT_16BIT)\
    {\
        uint32_t in = val;\
        if (in < 0x10000) {\
            tmp = in;\
            PUT_16BIT\
        } else {\
            tmp = 0xD800 | ((in - 0x10000) >> 10);\
            PUT_16BIT\
            tmp = 0xDC00 | ((in - 0x10000) & 0x3FF);\
            PUT_16BIT\
        }\
    }\



#include "mem.h"

#ifdef HAVE_AV_CONFIG_H
#    include "internal.h"
#endif /* HAVE_AV_CONFIG_H */

#endif /* AVUTIL_COMMON_H */
