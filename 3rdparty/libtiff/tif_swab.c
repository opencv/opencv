/*
 * Copyright (c) 1988-1997 Sam Leffler
 * Copyright (c) 1991-1997 Silicon Graphics, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Sam Leffler and Silicon Graphics.
 *
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * IN NO EVENT SHALL SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

/*
 * TIFF Library Bit & Byte Swapping Support.
 *
 * XXX We assume short = 16-bits and long = 32-bits XXX
 */
#include "tiffiop.h"

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabShort)
void TIFFSwabShort(uint16_t *wp)
{
    register unsigned char *cp = (unsigned char *)wp;
    unsigned char t;
    assert(sizeof(uint16_t) == 2);
    t = cp[1];
    cp[1] = cp[0];
    cp[0] = t;
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabLong)
void TIFFSwabLong(uint32_t *lp)
{
    register unsigned char *cp = (unsigned char *)lp;
    unsigned char t;
    assert(sizeof(uint32_t) == 4);
    t = cp[3];
    cp[3] = cp[0];
    cp[0] = t;
    t = cp[2];
    cp[2] = cp[1];
    cp[1] = t;
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabLong8)
void TIFFSwabLong8(uint64_t *lp)
{
    register unsigned char *cp = (unsigned char *)lp;
    unsigned char t;
    assert(sizeof(uint64_t) == 8);
    t = cp[7];
    cp[7] = cp[0];
    cp[0] = t;
    t = cp[6];
    cp[6] = cp[1];
    cp[1] = t;
    t = cp[5];
    cp[5] = cp[2];
    cp[2] = t;
    t = cp[4];
    cp[4] = cp[3];
    cp[3] = t;
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabArrayOfShort)
void TIFFSwabArrayOfShort(register uint16_t *wp, tmsize_t n)
{
    register unsigned char *cp;
    register unsigned char t;
    assert(sizeof(uint16_t) == 2);
    /* XXX unroll loop some */
    while (n-- > 0)
    {
        cp = (unsigned char *)wp;
        t = cp[1];
        cp[1] = cp[0];
        cp[0] = t;
        wp++;
    }
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabArrayOfTriples)
void TIFFSwabArrayOfTriples(register uint8_t *tp, tmsize_t n)
{
    unsigned char *cp;
    unsigned char t;

    /* XXX unroll loop some */
    while (n-- > 0)
    {
        cp = (unsigned char *)tp;
        t = cp[2];
        cp[2] = cp[0];
        cp[0] = t;
        tp += 3;
    }
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabArrayOfLong)
void TIFFSwabArrayOfLong(register uint32_t *lp, tmsize_t n)
{
    register unsigned char *cp;
    register unsigned char t;
    assert(sizeof(uint32_t) == 4);
    /* XXX unroll loop some */
    while (n-- > 0)
    {
        cp = (unsigned char *)lp;
        t = cp[3];
        cp[3] = cp[0];
        cp[0] = t;
        t = cp[2];
        cp[2] = cp[1];
        cp[1] = t;
        lp++;
    }
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabArrayOfLong8)
void TIFFSwabArrayOfLong8(register uint64_t *lp, tmsize_t n)
{
    register unsigned char *cp;
    register unsigned char t;
    assert(sizeof(uint64_t) == 8);
    /* XXX unroll loop some */
    while (n-- > 0)
    {
        cp = (unsigned char *)lp;
        t = cp[7];
        cp[7] = cp[0];
        cp[0] = t;
        t = cp[6];
        cp[6] = cp[1];
        cp[1] = t;
        t = cp[5];
        cp[5] = cp[2];
        cp[2] = t;
        t = cp[4];
        cp[4] = cp[3];
        cp[3] = t;
        lp++;
    }
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabFloat)
void TIFFSwabFloat(float *fp)
{
    register unsigned char *cp = (unsigned char *)fp;
    unsigned char t;
    assert(sizeof(float) == 4);
    t = cp[3];
    cp[3] = cp[0];
    cp[0] = t;
    t = cp[2];
    cp[2] = cp[1];
    cp[1] = t;
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabArrayOfFloat)
void TIFFSwabArrayOfFloat(register float *fp, tmsize_t n)
{
    register unsigned char *cp;
    register unsigned char t;
    assert(sizeof(float) == 4);
    /* XXX unroll loop some */
    while (n-- > 0)
    {
        cp = (unsigned char *)fp;
        t = cp[3];
        cp[3] = cp[0];
        cp[0] = t;
        t = cp[2];
        cp[2] = cp[1];
        cp[1] = t;
        fp++;
    }
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabDouble)
void TIFFSwabDouble(double *dp)
{
    register unsigned char *cp = (unsigned char *)dp;
    unsigned char t;
    assert(sizeof(double) == 8);
    t = cp[7];
    cp[7] = cp[0];
    cp[0] = t;
    t = cp[6];
    cp[6] = cp[1];
    cp[1] = t;
    t = cp[5];
    cp[5] = cp[2];
    cp[2] = t;
    t = cp[4];
    cp[4] = cp[3];
    cp[3] = t;
}
#endif

#if defined(DISABLE_CHECK_TIFFSWABMACROS) || !defined(TIFFSwabArrayOfDouble)
void TIFFSwabArrayOfDouble(double *dp, tmsize_t n)
{
    register unsigned char *cp;
    register unsigned char t;
    assert(sizeof(double) == 8);
    /* XXX unroll loop some */
    while (n-- > 0)
    {
        cp = (unsigned char *)dp;
        t = cp[7];
        cp[7] = cp[0];
        cp[0] = t;
        t = cp[6];
        cp[6] = cp[1];
        cp[1] = t;
        t = cp[5];
        cp[5] = cp[2];
        cp[2] = t;
        t = cp[4];
        cp[4] = cp[3];
        cp[3] = t;
        dp++;
    }
}
#endif

/*
 * Bit reversal tables.  TIFFBitRevTable[<byte>] gives
 * the bit reversed value of <byte>.  Used in various
 * places in the library when the FillOrder requires
 * bit reversal of byte values (e.g. CCITT Fax 3
 * encoding/decoding).  TIFFNoBitRevTable is provided
 * for algorithms that want an equivalent table that
 * do not reverse bit values.
 */
static const unsigned char TIFFBitRevTable[256] = {
    0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0,
    0x30, 0xb0, 0x70, 0xf0, 0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
    0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8, 0x04, 0x84, 0x44, 0xc4,
    0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
    0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc,
    0x3c, 0xbc, 0x7c, 0xfc, 0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
    0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2, 0x0a, 0x8a, 0x4a, 0xca,
    0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
    0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6,
    0x36, 0xb6, 0x76, 0xf6, 0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
    0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe, 0x01, 0x81, 0x41, 0xc1,
    0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
    0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9,
    0x39, 0xb9, 0x79, 0xf9, 0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
    0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5, 0x0d, 0x8d, 0x4d, 0xcd,
    0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
    0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3,
    0x33, 0xb3, 0x73, 0xf3, 0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
    0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb, 0x07, 0x87, 0x47, 0xc7,
    0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
    0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf,
    0x3f, 0xbf, 0x7f, 0xff};
static const unsigned char TIFFNoBitRevTable[256] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23,
    0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b,
    0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53,
    0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b,
    0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x80, 0x81, 0x82, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b,
    0x9c, 0x9d, 0x9e, 0x9f, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0, 0xb1, 0xb2, 0xb3,
    0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb,
    0xcc, 0xcd, 0xce, 0xcf, 0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe0, 0xe1, 0xe2, 0xe3,
    0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb,
    0xfc, 0xfd, 0xfe, 0xff,
};

const unsigned char *TIFFGetBitRevTable(int reversed)
{
    return (reversed ? TIFFBitRevTable : TIFFNoBitRevTable);
}

void TIFFReverseBits(uint8_t *cp, tmsize_t n)
{
    for (; n > 8; n -= 8)
    {
        cp[0] = TIFFBitRevTable[cp[0]];
        cp[1] = TIFFBitRevTable[cp[1]];
        cp[2] = TIFFBitRevTable[cp[2]];
        cp[3] = TIFFBitRevTable[cp[3]];
        cp[4] = TIFFBitRevTable[cp[4]];
        cp[5] = TIFFBitRevTable[cp[5]];
        cp[6] = TIFFBitRevTable[cp[6]];
        cp[7] = TIFFBitRevTable[cp[7]];
        cp += 8;
    }
    while (n-- > 0)
    {
        *cp = TIFFBitRevTable[*cp];
        cp++;
    }
}
