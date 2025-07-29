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

#include "tiffiop.h"
#include <assert.h>
#ifdef THUNDER_SUPPORT
/*
 * TIFF Library.
 *
 * ThunderScan 4-bit Compression Algorithm Support
 */

/*
 * ThunderScan uses an encoding scheme designed for
 * 4-bit pixel values.  Data is encoded in bytes, with
 * each byte split into a 2-bit code word and a 6-bit
 * data value.  The encoding gives raw data, runs of
 * pixels, or pixel values encoded as a delta from the
 * previous pixel value.  For the latter, either 2-bit
 * or 3-bit delta values are used, with the deltas packed
 * into a single byte.
 */
#define THUNDER_DATA 0x3f /* mask for 6-bit data */
#define THUNDER_CODE 0xc0 /* mask for 2-bit code word */
/* code values */
#define THUNDER_RUN 0x00        /* run of pixels w/ encoded count */
#define THUNDER_2BITDELTAS 0x40 /* 3 pixels w/ encoded 2-bit deltas */
#define DELTA2_SKIP 2           /* skip code for 2-bit deltas */
#define THUNDER_3BITDELTAS 0x80 /* 2 pixels w/ encoded 3-bit deltas */
#define DELTA3_SKIP 4           /* skip code for 3-bit deltas */
#define THUNDER_RAW 0xc0        /* raw data encoded */

static const int twobitdeltas[4] = {0, 1, 0, -1};
static const int threebitdeltas[8] = {0, 1, 2, 3, 0, -3, -2, -1};

#define SETPIXEL(op, v)                                                        \
    {                                                                          \
        lastpixel = (v)&0xf;                                                   \
        if (npixels < maxpixels)                                               \
        {                                                                      \
            if (npixels++ & 1)                                                 \
                *op++ |= lastpixel;                                            \
            else                                                               \
                op[0] = (uint8_t)(lastpixel << 4);                             \
        }                                                                      \
    }

static int ThunderSetupDecode(TIFF *tif)
{
    static const char module[] = "ThunderSetupDecode";

    if (tif->tif_dir.td_bitspersample != 4)
    {
        TIFFErrorExtR(tif, module,
                      "Wrong bitspersample value (%d), Thunder decoder only "
                      "supports 4bits per sample.",
                      (int)tif->tif_dir.td_bitspersample);
        return 0;
    }

    return (1);
}

static int ThunderDecode(TIFF *tif, uint8_t *op, tmsize_t maxpixels)
{
    static const char module[] = "ThunderDecode";
    register unsigned char *bp;
    register tmsize_t cc;
    unsigned int lastpixel;
    tmsize_t npixels;

    bp = (unsigned char *)tif->tif_rawcp;
    cc = tif->tif_rawcc;
    lastpixel = 0;
    npixels = 0;
    while (cc > 0 && npixels < maxpixels)
    {
        int n, delta;

        n = *bp++;
        cc--;
        switch (n & THUNDER_CODE)
        {
            case THUNDER_RUN: /* pixel run */
                /*
                 * Replicate the last pixel n times,
                 * where n is the lower-order 6 bits.
                 */
                if (npixels & 1)
                {
                    op[0] |= lastpixel;
                    lastpixel = *op++;
                    npixels++;
                    n--;
                }
                else
                    lastpixel |= lastpixel << 4;
                npixels += n;
                if (npixels < maxpixels)
                {
                    for (; n > 0; n -= 2)
                        *op++ = (uint8_t)lastpixel;
                }
                if (n == -1)
                    *--op &= 0xf0;
                lastpixel &= 0xf;
                break;
            case THUNDER_2BITDELTAS: /* 2-bit deltas */
                if ((delta = ((n >> 4) & 3)) != DELTA2_SKIP)
                    SETPIXEL(op,
                             (unsigned)((int)lastpixel + twobitdeltas[delta]));
                if ((delta = ((n >> 2) & 3)) != DELTA2_SKIP)
                    SETPIXEL(op,
                             (unsigned)((int)lastpixel + twobitdeltas[delta]));
                if ((delta = (n & 3)) != DELTA2_SKIP)
                    SETPIXEL(op,
                             (unsigned)((int)lastpixel + twobitdeltas[delta]));
                break;
            case THUNDER_3BITDELTAS: /* 3-bit deltas */
                if ((delta = ((n >> 3) & 7)) != DELTA3_SKIP)
                    SETPIXEL(
                        op, (unsigned)((int)lastpixel + threebitdeltas[delta]));
                if ((delta = (n & 7)) != DELTA3_SKIP)
                    SETPIXEL(
                        op, (unsigned)((int)lastpixel + threebitdeltas[delta]));
                break;
            case THUNDER_RAW: /* raw data */
                SETPIXEL(op, n);
                break;
        }
    }
    tif->tif_rawcp = (uint8_t *)bp;
    tif->tif_rawcc = cc;
    if (npixels != maxpixels)
    {
        TIFFErrorExtR(tif, module,
                      "%s data at scanline %lu (%" PRIu64 " != %" PRIu64 ")",
                      npixels < maxpixels ? "Not enough" : "Too much",
                      (unsigned long)tif->tif_row, (uint64_t)npixels,
                      (uint64_t)maxpixels);
        return (0);
    }

    return (1);
}

static int ThunderDecodeRow(TIFF *tif, uint8_t *buf, tmsize_t occ, uint16_t s)
{
    static const char module[] = "ThunderDecodeRow";
    uint8_t *row = buf;

    (void)s;
    if (occ % tif->tif_scanlinesize)
    {
        TIFFErrorExtR(tif, module, "Fractional scanlines cannot be read");
        return (0);
    }
    while (occ > 0)
    {
        if (!ThunderDecode(tif, row, tif->tif_dir.td_imagewidth))
            return (0);
        occ -= tif->tif_scanlinesize;
        row += tif->tif_scanlinesize;
    }
    return (1);
}

int TIFFInitThunderScan(TIFF *tif, int scheme)
{
    (void)scheme;

    tif->tif_setupdecode = ThunderSetupDecode;
    tif->tif_decoderow = ThunderDecodeRow;
    tif->tif_decodestrip = ThunderDecodeRow;
    return (1);
}
#endif /* THUNDER_SUPPORT */
