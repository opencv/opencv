/*
 * Copyright (c) 1996-1997 Sam Leffler
 * Copyright (c) 1996 Pixar
 *
 * Permission to use, copy, modify, distribute, and sell this software and
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Pixar, Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Pixar, Sam Leffler and Silicon Graphics.
 *
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * IN NO EVENT SHALL PIXAR, SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

#include "tiffiop.h"
#ifdef PIXARLOG_SUPPORT

/*
 * TIFF Library.
 * PixarLog Compression Support
 *
 * Contributed by Dan McCoy.
 *
 * PixarLog film support uses the TIFF library to store companded
 * 11 bit values into a tiff file, which are compressed using the
 * zip compressor.
 *
 * The codec can take as input and produce as output 32-bit IEEE float values
 * as well as 16-bit or 8-bit unsigned integer values.
 *
 * On writing any of the above are converted into the internal
 * 11-bit log format.   In the case of  8 and 16 bit values, the
 * input is assumed to be unsigned linear color values that represent
 * the range 0-1.  In the case of IEEE values, the 0-1 range is assumed to
 * be the normal linear color range, in addition over 1 values are
 * accepted up to a value of about 25.0 to encode "hot" highlights and such.
 * The encoding is lossless for 8-bit values, slightly lossy for the
 * other bit depths.  The actual color precision should be better
 * than the human eye can perceive with extra room to allow for
 * error introduced by further image computation.  As with any quantized
 * color format, it is possible to perform image calculations which
 * expose the quantization error. This format should certainly be less
 * susceptible to such errors than standard 8-bit encodings, but more
 * susceptible than straight 16-bit or 32-bit encodings.
 *
 * On reading the internal format is converted to the desired output format.
 * The program can request which format it desires by setting the internal
 * pseudo tag TIFFTAG_PIXARLOGDATAFMT to one of these possible values:
 *  PIXARLOGDATAFMT_FLOAT     = provide IEEE float values.
 *  PIXARLOGDATAFMT_16BIT     = provide unsigned 16-bit integer values
 *  PIXARLOGDATAFMT_8BIT      = provide unsigned 8-bit integer values
 *
 * alternately PIXARLOGDATAFMT_8BITABGR provides unsigned 8-bit integer
 * values with the difference that if there are exactly three or four channels
 * (rgb or rgba) it swaps the channel order (bgr or abgr).
 *
 * PIXARLOGDATAFMT_11BITLOG provides the internal encoding directly
 * packed in 16-bit values.   However no tools are supplied for interpreting
 * these values.
 *
 * "hot" (over 1.0) areas written in floating point get clamped to
 * 1.0 in the integer data types.
 *
 * When the file is closed after writing, the bit depth and sample format
 * are set always to appear as if 8-bit data has been written into it.
 * That way a naive program unaware of the particulars of the encoding
 * gets the format it is most likely able to handle.
 *
 * The codec does it's own horizontal differencing step on the coded
 * values so the libraries predictor stuff should be turned off.
 * The codec also handle byte swapping the encoded values as necessary
 * since the library does not have the information necessary
 * to know the bit depth of the raw unencoded buffer.
 *
 * NOTE: This decoder does not appear to update tif_rawcp, and tif_rawcc.
 * This can cause problems with the implementation of CHUNKY_STRIP_READ_SUPPORT
 * as noted in http://trac.osgeo.org/gdal/ticket/3894.   FrankW - Jan'11
 */

#include "tif_predict.h"
#include "zlib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Tables for converting to/from 11 bit coded values */

#define TSIZE 2048   /* decode table size (11-bit tokens) */
#define TSIZEP1 2049 /* Plus one for slop */
#define ONE 1250     /* token value of 1.0 exactly */
#define RATIO 1.004  /* nominal ratio for log part */

#define CODE_MASK 0x7ff /* 11 bits. */

static float Fltsize;
static float LogK1, LogK2;

#define REPEAT(n, op)                                                          \
    {                                                                          \
        int i;                                                                 \
        i = n;                                                                 \
        do                                                                     \
        {                                                                      \
            i--;                                                               \
            op;                                                                \
        } while (i > 0);                                                       \
    }

static void horizontalAccumulateF(uint16_t *wp, int n, int stride, float *op,
                                  float *ToLinearF)
{
    register unsigned int cr, cg, cb, ca, mask;
    register float t0, t1, t2, t3;

    if (n >= stride)
    {
        mask = CODE_MASK;
        if (stride == 3)
        {
            t0 = ToLinearF[cr = (wp[0] & mask)];
            t1 = ToLinearF[cg = (wp[1] & mask)];
            t2 = ToLinearF[cb = (wp[2] & mask)];
            op[0] = t0;
            op[1] = t1;
            op[2] = t2;
            n -= 3;
            while (n > 0)
            {
                wp += 3;
                op += 3;
                n -= 3;
                t0 = ToLinearF[(cr += wp[0]) & mask];
                t1 = ToLinearF[(cg += wp[1]) & mask];
                t2 = ToLinearF[(cb += wp[2]) & mask];
                op[0] = t0;
                op[1] = t1;
                op[2] = t2;
            }
        }
        else if (stride == 4)
        {
            t0 = ToLinearF[cr = (wp[0] & mask)];
            t1 = ToLinearF[cg = (wp[1] & mask)];
            t2 = ToLinearF[cb = (wp[2] & mask)];
            t3 = ToLinearF[ca = (wp[3] & mask)];
            op[0] = t0;
            op[1] = t1;
            op[2] = t2;
            op[3] = t3;
            n -= 4;
            while (n > 0)
            {
                wp += 4;
                op += 4;
                n -= 4;
                t0 = ToLinearF[(cr += wp[0]) & mask];
                t1 = ToLinearF[(cg += wp[1]) & mask];
                t2 = ToLinearF[(cb += wp[2]) & mask];
                t3 = ToLinearF[(ca += wp[3]) & mask];
                op[0] = t0;
                op[1] = t1;
                op[2] = t2;
                op[3] = t3;
            }
        }
        else
        {
            REPEAT(stride, *op = ToLinearF[*wp & mask]; wp++; op++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride, wp[stride] += *wp; *op = ToLinearF[*wp & mask];
                       wp++; op++)
                n -= stride;
            }
        }
    }
}

static void horizontalAccumulate12(uint16_t *wp, int n, int stride, int16_t *op,
                                   float *ToLinearF)
{
    register unsigned int cr, cg, cb, ca, mask;
    register float t0, t1, t2, t3;

#define SCALE12 2048.0F
#define CLAMP12(t) (((t) < 3071) ? (uint16_t)(t) : 3071)

    if (n >= stride)
    {
        mask = CODE_MASK;
        if (stride == 3)
        {
            t0 = ToLinearF[cr = (wp[0] & mask)] * SCALE12;
            t1 = ToLinearF[cg = (wp[1] & mask)] * SCALE12;
            t2 = ToLinearF[cb = (wp[2] & mask)] * SCALE12;
            op[0] = CLAMP12(t0);
            op[1] = CLAMP12(t1);
            op[2] = CLAMP12(t2);
            n -= 3;
            while (n > 0)
            {
                wp += 3;
                op += 3;
                n -= 3;
                t0 = ToLinearF[(cr += wp[0]) & mask] * SCALE12;
                t1 = ToLinearF[(cg += wp[1]) & mask] * SCALE12;
                t2 = ToLinearF[(cb += wp[2]) & mask] * SCALE12;
                op[0] = CLAMP12(t0);
                op[1] = CLAMP12(t1);
                op[2] = CLAMP12(t2);
            }
        }
        else if (stride == 4)
        {
            t0 = ToLinearF[cr = (wp[0] & mask)] * SCALE12;
            t1 = ToLinearF[cg = (wp[1] & mask)] * SCALE12;
            t2 = ToLinearF[cb = (wp[2] & mask)] * SCALE12;
            t3 = ToLinearF[ca = (wp[3] & mask)] * SCALE12;
            op[0] = CLAMP12(t0);
            op[1] = CLAMP12(t1);
            op[2] = CLAMP12(t2);
            op[3] = CLAMP12(t3);
            n -= 4;
            while (n > 0)
            {
                wp += 4;
                op += 4;
                n -= 4;
                t0 = ToLinearF[(cr += wp[0]) & mask] * SCALE12;
                t1 = ToLinearF[(cg += wp[1]) & mask] * SCALE12;
                t2 = ToLinearF[(cb += wp[2]) & mask] * SCALE12;
                t3 = ToLinearF[(ca += wp[3]) & mask] * SCALE12;
                op[0] = CLAMP12(t0);
                op[1] = CLAMP12(t1);
                op[2] = CLAMP12(t2);
                op[3] = CLAMP12(t3);
            }
        }
        else
        {
            REPEAT(stride, t0 = ToLinearF[*wp & mask] * SCALE12;
                   *op = CLAMP12(t0); wp++; op++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride, wp[stride] += *wp;
                       t0 = ToLinearF[wp[stride] & mask] * SCALE12;
                       *op = CLAMP12(t0); wp++; op++)
                n -= stride;
            }
        }
    }
}

static void horizontalAccumulate16(uint16_t *wp, int n, int stride,
                                   uint16_t *op, uint16_t *ToLinear16)
{
    register unsigned int cr, cg, cb, ca, mask;

    if (n >= stride)
    {
        mask = CODE_MASK;
        if (stride == 3)
        {
            op[0] = ToLinear16[cr = (wp[0] & mask)];
            op[1] = ToLinear16[cg = (wp[1] & mask)];
            op[2] = ToLinear16[cb = (wp[2] & mask)];
            n -= 3;
            while (n > 0)
            {
                wp += 3;
                op += 3;
                n -= 3;
                op[0] = ToLinear16[(cr += wp[0]) & mask];
                op[1] = ToLinear16[(cg += wp[1]) & mask];
                op[2] = ToLinear16[(cb += wp[2]) & mask];
            }
        }
        else if (stride == 4)
        {
            op[0] = ToLinear16[cr = (wp[0] & mask)];
            op[1] = ToLinear16[cg = (wp[1] & mask)];
            op[2] = ToLinear16[cb = (wp[2] & mask)];
            op[3] = ToLinear16[ca = (wp[3] & mask)];
            n -= 4;
            while (n > 0)
            {
                wp += 4;
                op += 4;
                n -= 4;
                op[0] = ToLinear16[(cr += wp[0]) & mask];
                op[1] = ToLinear16[(cg += wp[1]) & mask];
                op[2] = ToLinear16[(cb += wp[2]) & mask];
                op[3] = ToLinear16[(ca += wp[3]) & mask];
            }
        }
        else
        {
            REPEAT(stride, *op = ToLinear16[*wp & mask]; wp++; op++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride, wp[stride] += *wp; *op = ToLinear16[*wp & mask];
                       wp++; op++)
                n -= stride;
            }
        }
    }
}

/*
 * Returns the log encoded 11-bit values with the horizontal
 * differencing undone.
 */
static void horizontalAccumulate11(uint16_t *wp, int n, int stride,
                                   uint16_t *op)
{
    register unsigned int cr, cg, cb, ca, mask;

    if (n >= stride)
    {
        mask = CODE_MASK;
        if (stride == 3)
        {
            op[0] = wp[0];
            op[1] = wp[1];
            op[2] = wp[2];
            cr = wp[0];
            cg = wp[1];
            cb = wp[2];
            n -= 3;
            while (n > 0)
            {
                wp += 3;
                op += 3;
                n -= 3;
                op[0] = (uint16_t)((cr += wp[0]) & mask);
                op[1] = (uint16_t)((cg += wp[1]) & mask);
                op[2] = (uint16_t)((cb += wp[2]) & mask);
            }
        }
        else if (stride == 4)
        {
            op[0] = wp[0];
            op[1] = wp[1];
            op[2] = wp[2];
            op[3] = wp[3];
            cr = wp[0];
            cg = wp[1];
            cb = wp[2];
            ca = wp[3];
            n -= 4;
            while (n > 0)
            {
                wp += 4;
                op += 4;
                n -= 4;
                op[0] = (uint16_t)((cr += wp[0]) & mask);
                op[1] = (uint16_t)((cg += wp[1]) & mask);
                op[2] = (uint16_t)((cb += wp[2]) & mask);
                op[3] = (uint16_t)((ca += wp[3]) & mask);
            }
        }
        else
        {
            REPEAT(stride, *op = *wp & mask; wp++; op++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride, wp[stride] += *wp; *op = *wp & mask; wp++; op++)
                n -= stride;
            }
        }
    }
}

static void horizontalAccumulate8(uint16_t *wp, int n, int stride,
                                  unsigned char *op, unsigned char *ToLinear8)
{
    register unsigned int cr, cg, cb, ca, mask;

    if (n >= stride)
    {
        mask = CODE_MASK;
        if (stride == 3)
        {
            op[0] = ToLinear8[cr = (wp[0] & mask)];
            op[1] = ToLinear8[cg = (wp[1] & mask)];
            op[2] = ToLinear8[cb = (wp[2] & mask)];
            n -= 3;
            while (n > 0)
            {
                n -= 3;
                wp += 3;
                op += 3;
                op[0] = ToLinear8[(cr += wp[0]) & mask];
                op[1] = ToLinear8[(cg += wp[1]) & mask];
                op[2] = ToLinear8[(cb += wp[2]) & mask];
            }
        }
        else if (stride == 4)
        {
            op[0] = ToLinear8[cr = (wp[0] & mask)];
            op[1] = ToLinear8[cg = (wp[1] & mask)];
            op[2] = ToLinear8[cb = (wp[2] & mask)];
            op[3] = ToLinear8[ca = (wp[3] & mask)];
            n -= 4;
            while (n > 0)
            {
                n -= 4;
                wp += 4;
                op += 4;
                op[0] = ToLinear8[(cr += wp[0]) & mask];
                op[1] = ToLinear8[(cg += wp[1]) & mask];
                op[2] = ToLinear8[(cb += wp[2]) & mask];
                op[3] = ToLinear8[(ca += wp[3]) & mask];
            }
        }
        else
        {
            REPEAT(stride, *op = ToLinear8[*wp & mask]; wp++; op++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride, wp[stride] += *wp; *op = ToLinear8[*wp & mask];
                       wp++; op++)
                n -= stride;
            }
        }
    }
}

static void horizontalAccumulate8abgr(uint16_t *wp, int n, int stride,
                                      unsigned char *op,
                                      unsigned char *ToLinear8)
{
    register unsigned int cr, cg, cb, ca, mask;
    register unsigned char t0, t1, t2, t3;

    if (n >= stride)
    {
        mask = CODE_MASK;
        if (stride == 3)
        {
            op[0] = 0;
            t1 = ToLinear8[cb = (wp[2] & mask)];
            t2 = ToLinear8[cg = (wp[1] & mask)];
            t3 = ToLinear8[cr = (wp[0] & mask)];
            op[1] = t1;
            op[2] = t2;
            op[3] = t3;
            n -= 3;
            while (n > 0)
            {
                n -= 3;
                wp += 3;
                op += 4;
                op[0] = 0;
                t1 = ToLinear8[(cb += wp[2]) & mask];
                t2 = ToLinear8[(cg += wp[1]) & mask];
                t3 = ToLinear8[(cr += wp[0]) & mask];
                op[1] = t1;
                op[2] = t2;
                op[3] = t3;
            }
        }
        else if (stride == 4)
        {
            t0 = ToLinear8[ca = (wp[3] & mask)];
            t1 = ToLinear8[cb = (wp[2] & mask)];
            t2 = ToLinear8[cg = (wp[1] & mask)];
            t3 = ToLinear8[cr = (wp[0] & mask)];
            op[0] = t0;
            op[1] = t1;
            op[2] = t2;
            op[3] = t3;
            n -= 4;
            while (n > 0)
            {
                n -= 4;
                wp += 4;
                op += 4;
                t0 = ToLinear8[(ca += wp[3]) & mask];
                t1 = ToLinear8[(cb += wp[2]) & mask];
                t2 = ToLinear8[(cg += wp[1]) & mask];
                t3 = ToLinear8[(cr += wp[0]) & mask];
                op[0] = t0;
                op[1] = t1;
                op[2] = t2;
                op[3] = t3;
            }
        }
        else
        {
            REPEAT(stride, *op = ToLinear8[*wp & mask]; wp++; op++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride, wp[stride] += *wp; *op = ToLinear8[*wp & mask];
                       wp++; op++)
                n -= stride;
            }
        }
    }
}

/*
 * State block for each open TIFF
 * file using PixarLog compression/decompression.
 */
typedef struct
{
    TIFFPredictorState predict;
    z_stream stream;
    tmsize_t tbuf_size; /* only set/used on reading for now */
    uint16_t *tbuf;
    uint16_t stride;
    int state;
    int user_datafmt;
    int quality;
#define PLSTATE_INIT 1

    TIFFVSetMethod vgetparent; /* super-class method */
    TIFFVSetMethod vsetparent; /* super-class method */

    float *ToLinearF;
    uint16_t *ToLinear16;
    unsigned char *ToLinear8;
    uint16_t *FromLT2;
    uint16_t *From14; /* Really for 16-bit data, but we shift down 2 */
    uint16_t *From8;

} PixarLogState;

static int PixarLogMakeTables(TIFF *tif, PixarLogState *sp)
{

    /*
     *    We make several tables here to convert between various external
     *    representations (float, 16-bit, and 8-bit) and the internal
     *    11-bit companded representation.  The 11-bit representation has two
     *    distinct regions.  A linear bottom end up through .018316 in steps
     *    of about .000073, and a region of constant ratio up to about 25.
     *    These floating point numbers are stored in the main table ToLinearF.
     *    All other tables are derived from this one.  The tables (and the
     *    ratios) are continuous at the internal seam.
     */

    int nlin, lt2size;
    int i, j;
    double b, c, linstep, v;
    float *ToLinearF;
    uint16_t *ToLinear16;
    unsigned char *ToLinear8;
    uint16_t *FromLT2;
    uint16_t *From14; /* Really for 16-bit data, but we shift down 2 */
    uint16_t *From8;

    c = log(RATIO);
    nlin = (int)(1. / c); /* nlin must be an integer */
    c = 1. / nlin;
    b = exp(-c * ONE); /* multiplicative scale factor [b*exp(c*ONE) = 1] */
    linstep = b * c * exp(1.);

    LogK1 = (float)(1. / c); /* if (v >= 2)  token = k1*log(v*k2) */
    LogK2 = (float)(1. / b);
    lt2size = (int)(2. / linstep) + 1;
    FromLT2 = (uint16_t *)_TIFFmallocExt(tif, lt2size * sizeof(uint16_t));
    From14 = (uint16_t *)_TIFFmallocExt(tif, 16384 * sizeof(uint16_t));
    From8 = (uint16_t *)_TIFFmallocExt(tif, 256 * sizeof(uint16_t));
    ToLinearF = (float *)_TIFFmallocExt(tif, TSIZEP1 * sizeof(float));
    ToLinear16 = (uint16_t *)_TIFFmallocExt(tif, TSIZEP1 * sizeof(uint16_t));
    ToLinear8 =
        (unsigned char *)_TIFFmallocExt(tif, TSIZEP1 * sizeof(unsigned char));
    if (FromLT2 == NULL || From14 == NULL || From8 == NULL ||
        ToLinearF == NULL || ToLinear16 == NULL || ToLinear8 == NULL)
    {
        if (FromLT2)
            _TIFFfreeExt(tif, FromLT2);
        if (From14)
            _TIFFfreeExt(tif, From14);
        if (From8)
            _TIFFfreeExt(tif, From8);
        if (ToLinearF)
            _TIFFfreeExt(tif, ToLinearF);
        if (ToLinear16)
            _TIFFfreeExt(tif, ToLinear16);
        if (ToLinear8)
            _TIFFfreeExt(tif, ToLinear8);
        sp->FromLT2 = NULL;
        sp->From14 = NULL;
        sp->From8 = NULL;
        sp->ToLinearF = NULL;
        sp->ToLinear16 = NULL;
        sp->ToLinear8 = NULL;
        return 0;
    }

    j = 0;

    for (i = 0; i < nlin; i++)
    {
        v = i * linstep;
        ToLinearF[j++] = (float)v;
    }

    for (i = nlin; i < TSIZE; i++)
        ToLinearF[j++] = (float)(b * exp(c * i));

    ToLinearF[2048] = ToLinearF[2047];

    for (i = 0; i < TSIZEP1; i++)
    {
        v = ToLinearF[i] * 65535.0 + 0.5;
        ToLinear16[i] = (v > 65535.0) ? 65535 : (uint16_t)v;
        v = ToLinearF[i] * 255.0 + 0.5;
        ToLinear8[i] = (v > 255.0) ? 255 : (unsigned char)v;
    }

    j = 0;
    for (i = 0; i < lt2size; i++)
    {
        if ((i * linstep) * (i * linstep) > ToLinearF[j] * ToLinearF[j + 1])
            j++;
        FromLT2[i] = (uint16_t)j;
    }

    /*
     * Since we lose info anyway on 16-bit data, we set up a 14-bit
     * table and shift 16-bit values down two bits on input.
     * saves a little table space.
     */
    j = 0;
    for (i = 0; i < 16384; i++)
    {
        while ((i / 16383.) * (i / 16383.) > ToLinearF[j] * ToLinearF[j + 1])
            j++;
        From14[i] = (uint16_t)j;
    }

    j = 0;
    for (i = 0; i < 256; i++)
    {
        while ((i / 255.) * (i / 255.) > ToLinearF[j] * ToLinearF[j + 1])
            j++;
        From8[i] = (uint16_t)j;
    }

    Fltsize = (float)(lt2size / 2);

    sp->ToLinearF = ToLinearF;
    sp->ToLinear16 = ToLinear16;
    sp->ToLinear8 = ToLinear8;
    sp->FromLT2 = FromLT2;
    sp->From14 = From14;
    sp->From8 = From8;

    return 1;
}

#define DecoderState(tif) ((PixarLogState *)(tif)->tif_data)
#define EncoderState(tif) ((PixarLogState *)(tif)->tif_data)

static int PixarLogEncode(TIFF *tif, uint8_t *bp, tmsize_t cc, uint16_t s);
static int PixarLogDecode(TIFF *tif, uint8_t *op, tmsize_t occ, uint16_t s);

#define PIXARLOGDATAFMT_UNKNOWN -1

static int PixarLogGuessDataFmt(TIFFDirectory *td)
{
    int guess = PIXARLOGDATAFMT_UNKNOWN;
    int format = td->td_sampleformat;

    /* If the user didn't tell us his datafmt,
     * take our best guess from the bitspersample.
     */
    switch (td->td_bitspersample)
    {
        case 32:
            if (format == SAMPLEFORMAT_IEEEFP)
                guess = PIXARLOGDATAFMT_FLOAT;
            break;
        case 16:
            if (format == SAMPLEFORMAT_VOID || format == SAMPLEFORMAT_UINT)
                guess = PIXARLOGDATAFMT_16BIT;
            break;
        case 12:
            if (format == SAMPLEFORMAT_VOID || format == SAMPLEFORMAT_INT)
                guess = PIXARLOGDATAFMT_12BITPICIO;
            break;
        case 11:
            if (format == SAMPLEFORMAT_VOID || format == SAMPLEFORMAT_UINT)
                guess = PIXARLOGDATAFMT_11BITLOG;
            break;
        case 8:
            if (format == SAMPLEFORMAT_VOID || format == SAMPLEFORMAT_UINT)
                guess = PIXARLOGDATAFMT_8BIT;
            break;
    }

    return guess;
}

static tmsize_t multiply_ms(tmsize_t m1, tmsize_t m2)
{
    return _TIFFMultiplySSize(NULL, m1, m2, NULL);
}

static tmsize_t add_ms(tmsize_t m1, tmsize_t m2)
{
    assert(m1 >= 0 && m2 >= 0);
    /* if either input is zero, assume overflow already occurred */
    if (m1 == 0 || m2 == 0)
        return 0;
    else if (m1 > TIFF_TMSIZE_T_MAX - m2)
        return 0;

    return m1 + m2;
}

static int PixarLogFixupTags(TIFF *tif)
{
    (void)tif;
    return (1);
}

static int PixarLogSetupDecode(TIFF *tif)
{
    static const char module[] = "PixarLogSetupDecode";
    TIFFDirectory *td = &tif->tif_dir;
    PixarLogState *sp = DecoderState(tif);
    tmsize_t tbuf_size;
    uint32_t strip_height;

    assert(sp != NULL);

    /* This function can possibly be called several times by */
    /* PredictorSetupDecode() if this function succeeds but */
    /* PredictorSetup() fails */
    if ((sp->state & PLSTATE_INIT) != 0)
        return 1;

    strip_height = td->td_rowsperstrip;
    if (strip_height > td->td_imagelength)
        strip_height = td->td_imagelength;

    /* Make sure no byte swapping happens on the data
     * after decompression. */
    tif->tif_postdecode = _TIFFNoPostDecode;

    /* for some reason, we can't do this in TIFFInitPixarLog */

    sp->stride =
        (td->td_planarconfig == PLANARCONFIG_CONTIG ? td->td_samplesperpixel
                                                    : 1);
    tbuf_size = multiply_ms(
        multiply_ms(multiply_ms(sp->stride, td->td_imagewidth), strip_height),
        sizeof(uint16_t));
    /* add one more stride in case input ends mid-stride */
    tbuf_size = add_ms(tbuf_size, sizeof(uint16_t) * sp->stride);
    if (tbuf_size == 0)
        return (0); /* TODO: this is an error return without error report
                       through TIFFErrorExt */
    sp->tbuf = (uint16_t *)_TIFFmallocExt(tif, tbuf_size);
    if (sp->tbuf == NULL)
        return (0);
    sp->tbuf_size = tbuf_size;
    if (sp->user_datafmt == PIXARLOGDATAFMT_UNKNOWN)
        sp->user_datafmt = PixarLogGuessDataFmt(td);
    if (sp->user_datafmt == PIXARLOGDATAFMT_UNKNOWN)
    {
        _TIFFfreeExt(tif, sp->tbuf);
        sp->tbuf = NULL;
        sp->tbuf_size = 0;
        TIFFErrorExtR(tif, module,
                      "PixarLog compression can't handle bits depth/data "
                      "format combination (depth: %" PRIu16 ")",
                      td->td_bitspersample);
        return (0);
    }

    if (inflateInit(&sp->stream) != Z_OK)
    {
        _TIFFfreeExt(tif, sp->tbuf);
        sp->tbuf = NULL;
        sp->tbuf_size = 0;
        TIFFErrorExtR(tif, module, "%s",
                      sp->stream.msg ? sp->stream.msg : "(null)");
        return (0);
    }
    else
    {
        sp->state |= PLSTATE_INIT;
        return (1);
    }
}

/*
 * Setup state for decoding a strip.
 */
static int PixarLogPreDecode(TIFF *tif, uint16_t s)
{
    static const char module[] = "PixarLogPreDecode";
    PixarLogState *sp = DecoderState(tif);

    (void)s;
    assert(sp != NULL);
    sp->stream.next_in = tif->tif_rawdata;
    assert(sizeof(sp->stream.avail_in) == 4); /* if this assert gets raised,
         we need to simplify this code to reflect a ZLib that is likely updated
         to deal with 8byte memory sizes, though this code will respond
         appropriately even before we simplify it */
    sp->stream.avail_in = (uInt)tif->tif_rawcc;
    if ((tmsize_t)sp->stream.avail_in != tif->tif_rawcc)
    {
        TIFFErrorExtR(tif, module, "ZLib cannot deal with buffers this size");
        return (0);
    }
    return (inflateReset(&sp->stream) == Z_OK);
}

static int PixarLogDecode(TIFF *tif, uint8_t *op, tmsize_t occ, uint16_t s)
{
    static const char module[] = "PixarLogDecode";
    TIFFDirectory *td = &tif->tif_dir;
    PixarLogState *sp = DecoderState(tif);
    tmsize_t i;
    tmsize_t nsamples;
    int llen;
    uint16_t *up;

    switch (sp->user_datafmt)
    {
        case PIXARLOGDATAFMT_FLOAT:
            nsamples = occ / sizeof(float); /* XXX float == 32 bits */
            break;
        case PIXARLOGDATAFMT_16BIT:
        case PIXARLOGDATAFMT_12BITPICIO:
        case PIXARLOGDATAFMT_11BITLOG:
            nsamples = occ / sizeof(uint16_t); /* XXX uint16_t == 16 bits */
            break;
        case PIXARLOGDATAFMT_8BIT:
        case PIXARLOGDATAFMT_8BITABGR:
            nsamples = occ;
            break;
        default:
            TIFFErrorExtR(tif, module,
                          "%" PRIu16 " bit input not supported in PixarLog",
                          td->td_bitspersample);
            return 0;
    }

    llen = sp->stride * td->td_imagewidth;

    (void)s;
    assert(sp != NULL);

    sp->stream.next_in = tif->tif_rawcp;
    sp->stream.avail_in = (uInt)tif->tif_rawcc;

    sp->stream.next_out = (unsigned char *)sp->tbuf;
    assert(sizeof(sp->stream.avail_out) == 4); /* if this assert gets raised,
         we need to simplify this code to reflect a ZLib that is likely updated
         to deal with 8byte memory sizes, though this code will respond
         appropriately even before we simplify it */
    sp->stream.avail_out = (uInt)(nsamples * sizeof(uint16_t));
    if (sp->stream.avail_out != nsamples * sizeof(uint16_t))
    {
        TIFFErrorExtR(tif, module, "ZLib cannot deal with buffers this size");
        return (0);
    }
    /* Check that we will not fill more than what was allocated */
    if ((tmsize_t)sp->stream.avail_out > sp->tbuf_size)
    {
        TIFFErrorExtR(tif, module, "sp->stream.avail_out > sp->tbuf_size");
        return (0);
    }
    do
    {
        int state = inflate(&sp->stream, Z_PARTIAL_FLUSH);
        if (state == Z_STREAM_END)
        {
            break; /* XXX */
        }
        if (state == Z_DATA_ERROR)
        {
            TIFFErrorExtR(
                tif, module, "Decoding error at scanline %" PRIu32 ", %s",
                tif->tif_row, sp->stream.msg ? sp->stream.msg : "(null)");
            return (0);
        }
        if (state != Z_OK)
        {
            TIFFErrorExtR(tif, module, "ZLib error: %s",
                          sp->stream.msg ? sp->stream.msg : "(null)");
            return (0);
        }
    } while (sp->stream.avail_out > 0);

    /* hopefully, we got all the bytes we needed */
    if (sp->stream.avail_out != 0)
    {
        TIFFErrorExtR(tif, module,
                      "Not enough data at scanline %" PRIu32
                      " (short %u bytes)",
                      tif->tif_row, sp->stream.avail_out);
        return (0);
    }

    tif->tif_rawcp = sp->stream.next_in;
    tif->tif_rawcc = sp->stream.avail_in;

    up = sp->tbuf;
    /* Swap bytes in the data if from a different endian machine. */
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfShort(up, nsamples);

    /*
     * if llen is not an exact multiple of nsamples, the decode operation
     * may overflow the output buffer, so truncate it enough to prevent
     * that but still salvage as much data as possible.
     */
    if (nsamples % llen)
    {
        TIFFWarningExtR(tif, module,
                        "stride %d is not a multiple of sample count, "
                        "%" TIFF_SSIZE_FORMAT ", data truncated.",
                        llen, nsamples);
        nsamples -= nsamples % llen;
    }

    for (i = 0; i < nsamples; i += llen, up += llen)
    {
        switch (sp->user_datafmt)
        {
            case PIXARLOGDATAFMT_FLOAT:
                horizontalAccumulateF(up, llen, sp->stride, (float *)op,
                                      sp->ToLinearF);
                op += llen * sizeof(float);
                break;
            case PIXARLOGDATAFMT_16BIT:
                horizontalAccumulate16(up, llen, sp->stride, (uint16_t *)op,
                                       sp->ToLinear16);
                op += llen * sizeof(uint16_t);
                break;
            case PIXARLOGDATAFMT_12BITPICIO:
                horizontalAccumulate12(up, llen, sp->stride, (int16_t *)op,
                                       sp->ToLinearF);
                op += llen * sizeof(int16_t);
                break;
            case PIXARLOGDATAFMT_11BITLOG:
                horizontalAccumulate11(up, llen, sp->stride, (uint16_t *)op);
                op += llen * sizeof(uint16_t);
                break;
            case PIXARLOGDATAFMT_8BIT:
                horizontalAccumulate8(up, llen, sp->stride, (unsigned char *)op,
                                      sp->ToLinear8);
                op += llen * sizeof(unsigned char);
                break;
            case PIXARLOGDATAFMT_8BITABGR:
                horizontalAccumulate8abgr(up, llen, sp->stride,
                                          (unsigned char *)op, sp->ToLinear8);
                op += llen * sizeof(unsigned char);
                break;
            default:
                TIFFErrorExtR(tif, module, "Unsupported bits/sample: %" PRIu16,
                              td->td_bitspersample);
                return (0);
        }
    }

    return (1);
}

static int PixarLogSetupEncode(TIFF *tif)
{
    static const char module[] = "PixarLogSetupEncode";
    TIFFDirectory *td = &tif->tif_dir;
    PixarLogState *sp = EncoderState(tif);
    tmsize_t tbuf_size;

    assert(sp != NULL);

    /* for some reason, we can't do this in TIFFInitPixarLog */

    sp->stride =
        (td->td_planarconfig == PLANARCONFIG_CONTIG ? td->td_samplesperpixel
                                                    : 1);
    tbuf_size =
        multiply_ms(multiply_ms(multiply_ms(sp->stride, td->td_imagewidth),
                                td->td_rowsperstrip),
                    sizeof(uint16_t));
    if (tbuf_size == 0)
        return (0); /* TODO: this is an error return without error report
                       through TIFFErrorExt */
    sp->tbuf = (uint16_t *)_TIFFmallocExt(tif, tbuf_size);
    if (sp->tbuf == NULL)
        return (0);
    if (sp->user_datafmt == PIXARLOGDATAFMT_UNKNOWN)
        sp->user_datafmt = PixarLogGuessDataFmt(td);
    if (sp->user_datafmt == PIXARLOGDATAFMT_UNKNOWN)
    {
        TIFFErrorExtR(tif, module,
                      "PixarLog compression can't handle %" PRIu16
                      " bit linear encodings",
                      td->td_bitspersample);
        return (0);
    }

    if (deflateInit(&sp->stream, sp->quality) != Z_OK)
    {
        TIFFErrorExtR(tif, module, "%s",
                      sp->stream.msg ? sp->stream.msg : "(null)");
        return (0);
    }
    else
    {
        sp->state |= PLSTATE_INIT;
        return (1);
    }
}

/*
 * Reset encoding state at the start of a strip.
 */
static int PixarLogPreEncode(TIFF *tif, uint16_t s)
{
    static const char module[] = "PixarLogPreEncode";
    PixarLogState *sp = EncoderState(tif);

    (void)s;
    assert(sp != NULL);
    sp->stream.next_out = tif->tif_rawdata;
    assert(sizeof(sp->stream.avail_out) == 4); /* if this assert gets raised,
         we need to simplify this code to reflect a ZLib that is likely updated
         to deal with 8byte memory sizes, though this code will respond
         appropriately even before we simplify it */
    sp->stream.avail_out = (uInt)tif->tif_rawdatasize;
    if ((tmsize_t)sp->stream.avail_out != tif->tif_rawdatasize)
    {
        TIFFErrorExtR(tif, module, "ZLib cannot deal with buffers this size");
        return (0);
    }
    return (deflateReset(&sp->stream) == Z_OK);
}

static void horizontalDifferenceF(float *ip, int n, int stride, uint16_t *wp,
                                  uint16_t *FromLT2)
{
    int32_t r1, g1, b1, a1, r2, g2, b2, a2, mask;
    float fltsize = Fltsize;

#define CLAMP(v)                                                               \
    ((v < (float)0.)     ? 0                                                   \
     : (v < (float)2.)   ? FromLT2[(int)(v * fltsize)]                         \
     : (v > (float)24.2) ? 2047                                                \
                         : LogK1 * log(v * LogK2) + 0.5)

    mask = CODE_MASK;
    if (n >= stride)
    {
        if (stride == 3)
        {
            r2 = wp[0] = (uint16_t)CLAMP(ip[0]);
            g2 = wp[1] = (uint16_t)CLAMP(ip[1]);
            b2 = wp[2] = (uint16_t)CLAMP(ip[2]);
            n -= 3;
            while (n > 0)
            {
                n -= 3;
                wp += 3;
                ip += 3;
                r1 = (int32_t)CLAMP(ip[0]);
                wp[0] = (uint16_t)((r1 - r2) & mask);
                r2 = r1;
                g1 = (int32_t)CLAMP(ip[1]);
                wp[1] = (uint16_t)((g1 - g2) & mask);
                g2 = g1;
                b1 = (int32_t)CLAMP(ip[2]);
                wp[2] = (uint16_t)((b1 - b2) & mask);
                b2 = b1;
            }
        }
        else if (stride == 4)
        {
            r2 = wp[0] = (uint16_t)CLAMP(ip[0]);
            g2 = wp[1] = (uint16_t)CLAMP(ip[1]);
            b2 = wp[2] = (uint16_t)CLAMP(ip[2]);
            a2 = wp[3] = (uint16_t)CLAMP(ip[3]);
            n -= 4;
            while (n > 0)
            {
                n -= 4;
                wp += 4;
                ip += 4;
                r1 = (int32_t)CLAMP(ip[0]);
                wp[0] = (uint16_t)((r1 - r2) & mask);
                r2 = r1;
                g1 = (int32_t)CLAMP(ip[1]);
                wp[1] = (uint16_t)((g1 - g2) & mask);
                g2 = g1;
                b1 = (int32_t)CLAMP(ip[2]);
                wp[2] = (uint16_t)((b1 - b2) & mask);
                b2 = b1;
                a1 = (int32_t)CLAMP(ip[3]);
                wp[3] = (uint16_t)((a1 - a2) & mask);
                a2 = a1;
            }
        }
        else
        {
            REPEAT(stride, wp[0] = (uint16_t)CLAMP(ip[0]); wp++; ip++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride,
                       wp[0] = (uint16_t)(((int32_t)CLAMP(ip[0]) -
                                           (int32_t)CLAMP(ip[-stride])) &
                                          mask);
                       wp++; ip++)
                n -= stride;
            }
        }
    }
}

static void horizontalDifference16(unsigned short *ip, int n, int stride,
                                   unsigned short *wp, uint16_t *From14)
{
    register int r1, g1, b1, a1, r2, g2, b2, a2, mask;

/* assumption is unsigned pixel values */
#undef CLAMP
#define CLAMP(v) From14[(v) >> 2]

    mask = CODE_MASK;
    if (n >= stride)
    {
        if (stride == 3)
        {
            r2 = wp[0] = CLAMP(ip[0]);
            g2 = wp[1] = CLAMP(ip[1]);
            b2 = wp[2] = CLAMP(ip[2]);
            n -= 3;
            while (n > 0)
            {
                n -= 3;
                wp += 3;
                ip += 3;
                r1 = CLAMP(ip[0]);
                wp[0] = (uint16_t)((r1 - r2) & mask);
                r2 = r1;
                g1 = CLAMP(ip[1]);
                wp[1] = (uint16_t)((g1 - g2) & mask);
                g2 = g1;
                b1 = CLAMP(ip[2]);
                wp[2] = (uint16_t)((b1 - b2) & mask);
                b2 = b1;
            }
        }
        else if (stride == 4)
        {
            r2 = wp[0] = CLAMP(ip[0]);
            g2 = wp[1] = CLAMP(ip[1]);
            b2 = wp[2] = CLAMP(ip[2]);
            a2 = wp[3] = CLAMP(ip[3]);
            n -= 4;
            while (n > 0)
            {
                n -= 4;
                wp += 4;
                ip += 4;
                r1 = CLAMP(ip[0]);
                wp[0] = (uint16_t)((r1 - r2) & mask);
                r2 = r1;
                g1 = CLAMP(ip[1]);
                wp[1] = (uint16_t)((g1 - g2) & mask);
                g2 = g1;
                b1 = CLAMP(ip[2]);
                wp[2] = (uint16_t)((b1 - b2) & mask);
                b2 = b1;
                a1 = CLAMP(ip[3]);
                wp[3] = (uint16_t)((a1 - a2) & mask);
                a2 = a1;
            }
        }
        else
        {
            REPEAT(stride, wp[0] = CLAMP(ip[0]); wp++; ip++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride,
                       wp[0] = (uint16_t)((CLAMP(ip[0]) - CLAMP(ip[-stride])) &
                                          mask);
                       wp++; ip++)
                n -= stride;
            }
        }
    }
}

static void horizontalDifference8(unsigned char *ip, int n, int stride,
                                  unsigned short *wp, uint16_t *From8)
{
    register int r1, g1, b1, a1, r2, g2, b2, a2, mask;

#undef CLAMP
#define CLAMP(v) (From8[(v)])

    mask = CODE_MASK;
    if (n >= stride)
    {
        if (stride == 3)
        {
            r2 = wp[0] = CLAMP(ip[0]);
            g2 = wp[1] = CLAMP(ip[1]);
            b2 = wp[2] = CLAMP(ip[2]);
            n -= 3;
            while (n > 0)
            {
                n -= 3;
                r1 = CLAMP(ip[3]);
                wp[3] = (uint16_t)((r1 - r2) & mask);
                r2 = r1;
                g1 = CLAMP(ip[4]);
                wp[4] = (uint16_t)((g1 - g2) & mask);
                g2 = g1;
                b1 = CLAMP(ip[5]);
                wp[5] = (uint16_t)((b1 - b2) & mask);
                b2 = b1;
                wp += 3;
                ip += 3;
            }
        }
        else if (stride == 4)
        {
            r2 = wp[0] = CLAMP(ip[0]);
            g2 = wp[1] = CLAMP(ip[1]);
            b2 = wp[2] = CLAMP(ip[2]);
            a2 = wp[3] = CLAMP(ip[3]);
            n -= 4;
            while (n > 0)
            {
                n -= 4;
                r1 = CLAMP(ip[4]);
                wp[4] = (uint16_t)((r1 - r2) & mask);
                r2 = r1;
                g1 = CLAMP(ip[5]);
                wp[5] = (uint16_t)((g1 - g2) & mask);
                g2 = g1;
                b1 = CLAMP(ip[6]);
                wp[6] = (uint16_t)((b1 - b2) & mask);
                b2 = b1;
                a1 = CLAMP(ip[7]);
                wp[7] = (uint16_t)((a1 - a2) & mask);
                a2 = a1;
                wp += 4;
                ip += 4;
            }
        }
        else
        {
            REPEAT(stride, wp[0] = CLAMP(ip[0]); wp++; ip++)
            n -= stride;
            while (n > 0)
            {
                REPEAT(stride,
                       wp[0] = (uint16_t)((CLAMP(ip[0]) - CLAMP(ip[-stride])) &
                                          mask);
                       wp++; ip++)
                n -= stride;
            }
        }
    }
}

/*
 * Encode a chunk of pixels.
 */
static int PixarLogEncode(TIFF *tif, uint8_t *bp, tmsize_t cc, uint16_t s)
{
    static const char module[] = "PixarLogEncode";
    TIFFDirectory *td = &tif->tif_dir;
    PixarLogState *sp = EncoderState(tif);
    tmsize_t i;
    tmsize_t n;
    int llen;
    unsigned short *up;

    (void)s;

    switch (sp->user_datafmt)
    {
        case PIXARLOGDATAFMT_FLOAT:
            n = cc / sizeof(float); /* XXX float == 32 bits */
            break;
        case PIXARLOGDATAFMT_16BIT:
        case PIXARLOGDATAFMT_12BITPICIO:
        case PIXARLOGDATAFMT_11BITLOG:
            n = cc / sizeof(uint16_t); /* XXX uint16_t == 16 bits */
            break;
        case PIXARLOGDATAFMT_8BIT:
        case PIXARLOGDATAFMT_8BITABGR:
            n = cc;
            break;
        default:
            TIFFErrorExtR(tif, module,
                          "%" PRIu16 " bit input not supported in PixarLog",
                          td->td_bitspersample);
            return 0;
    }

    llen = sp->stride * td->td_imagewidth;
    /* Check against the number of elements (of size uint16_t) of sp->tbuf */
    if (n > ((tmsize_t)td->td_rowsperstrip * llen))
    {
        TIFFErrorExtR(tif, module, "Too many input bytes provided");
        return 0;
    }

    for (i = 0, up = sp->tbuf; i < n; i += llen, up += llen)
    {
        switch (sp->user_datafmt)
        {
            case PIXARLOGDATAFMT_FLOAT:
                horizontalDifferenceF((float *)bp, llen, sp->stride, up,
                                      sp->FromLT2);
                bp += llen * sizeof(float);
                break;
            case PIXARLOGDATAFMT_16BIT:
                horizontalDifference16((uint16_t *)bp, llen, sp->stride, up,
                                       sp->From14);
                bp += llen * sizeof(uint16_t);
                break;
            case PIXARLOGDATAFMT_8BIT:
                horizontalDifference8((unsigned char *)bp, llen, sp->stride, up,
                                      sp->From8);
                bp += llen * sizeof(unsigned char);
                break;
            default:
                TIFFErrorExtR(tif, module,
                              "%" PRIu16 " bit input not supported in PixarLog",
                              td->td_bitspersample);
                return 0;
        }
    }

    sp->stream.next_in = (unsigned char *)sp->tbuf;
    assert(sizeof(sp->stream.avail_in) == 4); /* if this assert gets raised,
         we need to simplify this code to reflect a ZLib that is likely updated
         to deal with 8byte memory sizes, though this code will respond
         appropriately even before we simplify it */
    sp->stream.avail_in = (uInt)(n * sizeof(uint16_t));
    if ((sp->stream.avail_in / sizeof(uint16_t)) != (uInt)n)
    {
        TIFFErrorExtR(tif, module, "ZLib cannot deal with buffers this size");
        return (0);
    }

    do
    {
        if (deflate(&sp->stream, Z_NO_FLUSH) != Z_OK)
        {
            TIFFErrorExtR(tif, module, "Encoder error: %s",
                          sp->stream.msg ? sp->stream.msg : "(null)");
            return (0);
        }
        if (sp->stream.avail_out == 0)
        {
            tif->tif_rawcc = tif->tif_rawdatasize;
            if (!TIFFFlushData1(tif))
                return 0;
            sp->stream.next_out = tif->tif_rawdata;
            sp->stream.avail_out =
                (uInt)tif
                    ->tif_rawdatasize; /* this is a safe typecast, as check is
                                          made already in PixarLogPreEncode */
        }
    } while (sp->stream.avail_in > 0);
    return (1);
}

/*
 * Finish off an encoded strip by flushing the last
 * string and tacking on an End Of Information code.
 */

static int PixarLogPostEncode(TIFF *tif)
{
    static const char module[] = "PixarLogPostEncode";
    PixarLogState *sp = EncoderState(tif);
    int state;

    sp->stream.avail_in = 0;

    do
    {
        state = deflate(&sp->stream, Z_FINISH);
        switch (state)
        {
            case Z_STREAM_END:
            case Z_OK:
                if ((tmsize_t)sp->stream.avail_out != tif->tif_rawdatasize)
                {
                    tif->tif_rawcc =
                        tif->tif_rawdatasize - sp->stream.avail_out;
                    if (!TIFFFlushData1(tif))
                        return 0;
                    sp->stream.next_out = tif->tif_rawdata;
                    sp->stream.avail_out =
                        (uInt)tif->tif_rawdatasize; /* this is a safe typecast,
                                                       as check is made already
                                                       in PixarLogPreEncode */
                }
                break;
            default:
                TIFFErrorExtR(tif, module, "ZLib error: %s",
                              sp->stream.msg ? sp->stream.msg : "(null)");
                return (0);
        }
    } while (state != Z_STREAM_END);
    return (1);
}

static void PixarLogClose(TIFF *tif)
{
    PixarLogState *sp = (PixarLogState *)tif->tif_data;
    TIFFDirectory *td = &tif->tif_dir;

    assert(sp != 0);
    /* In a really sneaky (and really incorrect, and untruthful, and
     * troublesome, and error-prone) maneuver that completely goes against
     * the spirit of TIFF, and breaks TIFF, on close, we covertly
     * modify both bitspersample and sampleformat in the directory to
     * indicate 8-bit linear.  This way, the decode "just works" even for
     * readers that don't know about PixarLog, or how to set
     * the PIXARLOGDATFMT pseudo-tag.
     */

    if (sp->state & PLSTATE_INIT)
    {
        /* We test the state to avoid an issue such as in
         * http://bugzilla.maptools.org/show_bug.cgi?id=2604
         * What appends in that case is that the bitspersample is 1 and
         * a TransferFunction is set. The size of the TransferFunction
         * depends on 1<<bitspersample. So if we increase it, an access
         * out of the buffer will happen at directory flushing.
         * Another option would be to clear those targs.
         */
        td->td_bitspersample = 8;
        td->td_sampleformat = SAMPLEFORMAT_UINT;
    }
}

static void PixarLogCleanup(TIFF *tif)
{
    PixarLogState *sp = (PixarLogState *)tif->tif_data;

    assert(sp != 0);

    (void)TIFFPredictorCleanup(tif);

    tif->tif_tagmethods.vgetfield = sp->vgetparent;
    tif->tif_tagmethods.vsetfield = sp->vsetparent;

    if (sp->FromLT2)
        _TIFFfreeExt(tif, sp->FromLT2);
    if (sp->From14)
        _TIFFfreeExt(tif, sp->From14);
    if (sp->From8)
        _TIFFfreeExt(tif, sp->From8);
    if (sp->ToLinearF)
        _TIFFfreeExt(tif, sp->ToLinearF);
    if (sp->ToLinear16)
        _TIFFfreeExt(tif, sp->ToLinear16);
    if (sp->ToLinear8)
        _TIFFfreeExt(tif, sp->ToLinear8);
    if (sp->state & PLSTATE_INIT)
    {
        if (tif->tif_mode == O_RDONLY)
            inflateEnd(&sp->stream);
        else
            deflateEnd(&sp->stream);
    }
    if (sp->tbuf)
        _TIFFfreeExt(tif, sp->tbuf);
    _TIFFfreeExt(tif, sp);
    tif->tif_data = NULL;

    _TIFFSetDefaultCompressionState(tif);
}

static int PixarLogVSetField(TIFF *tif, uint32_t tag, va_list ap)
{
    static const char module[] = "PixarLogVSetField";
    PixarLogState *sp = (PixarLogState *)tif->tif_data;
    int result;

    switch (tag)
    {
        case TIFFTAG_PIXARLOGQUALITY:
            sp->quality = (int)va_arg(ap, int);
            if (tif->tif_mode != O_RDONLY && (sp->state & PLSTATE_INIT))
            {
                if (deflateParams(&sp->stream, sp->quality,
                                  Z_DEFAULT_STRATEGY) != Z_OK)
                {
                    TIFFErrorExtR(tif, module, "ZLib error: %s",
                                  sp->stream.msg ? sp->stream.msg : "(null)");
                    return (0);
                }
            }
            return (1);
        case TIFFTAG_PIXARLOGDATAFMT:
            sp->user_datafmt = (int)va_arg(ap, int);
            /* Tweak the TIFF header so that the rest of libtiff knows what
             * size of data will be passed between app and library, and
             * assume that the app knows what it is doing and is not
             * confused by these header manipulations...
             */
            switch (sp->user_datafmt)
            {
                case PIXARLOGDATAFMT_8BIT:
                case PIXARLOGDATAFMT_8BITABGR:
                    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
                    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
                    break;
                case PIXARLOGDATAFMT_11BITLOG:
                    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);
                    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
                    break;
                case PIXARLOGDATAFMT_12BITPICIO:
                    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);
                    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
                    break;
                case PIXARLOGDATAFMT_16BIT:
                    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);
                    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
                    break;
                case PIXARLOGDATAFMT_FLOAT:
                    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
                    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT,
                                 SAMPLEFORMAT_IEEEFP);
                    break;
            }
            /*
             * Must recalculate sizes should bits/sample change.
             */
            tif->tif_tilesize =
                isTiled(tif) ? TIFFTileSize(tif) : (tmsize_t)(-1);
            tif->tif_scanlinesize = TIFFScanlineSize(tif);
            result = 1; /* NB: pseudo tag */
            break;
        default:
            result = (*sp->vsetparent)(tif, tag, ap);
    }
    return (result);
}

static int PixarLogVGetField(TIFF *tif, uint32_t tag, va_list ap)
{
    PixarLogState *sp = (PixarLogState *)tif->tif_data;

    switch (tag)
    {
        case TIFFTAG_PIXARLOGQUALITY:
            *va_arg(ap, int *) = sp->quality;
            break;
        case TIFFTAG_PIXARLOGDATAFMT:
            *va_arg(ap, int *) = sp->user_datafmt;
            break;
        default:
            return (*sp->vgetparent)(tif, tag, ap);
    }
    return (1);
}

static const TIFFField pixarlogFields[] = {
    {TIFFTAG_PIXARLOGDATAFMT, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, FALSE, FALSE, "", NULL},
    {TIFFTAG_PIXARLOGQUALITY, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, FALSE, FALSE, "", NULL}};

int TIFFInitPixarLog(TIFF *tif, int scheme)
{
    static const char module[] = "TIFFInitPixarLog";

    PixarLogState *sp;

    (void)scheme;
    assert(scheme == COMPRESSION_PIXARLOG);

    /*
     * Merge codec-specific tag information.
     */
    if (!_TIFFMergeFields(tif, pixarlogFields, TIFFArrayCount(pixarlogFields)))
    {
        TIFFErrorExtR(tif, module,
                      "Merging PixarLog codec-specific tags failed");
        return 0;
    }

    /*
     * Allocate state block so tag methods have storage to record values.
     */
    tif->tif_data = (uint8_t *)_TIFFmallocExt(tif, sizeof(PixarLogState));
    if (tif->tif_data == NULL)
        goto bad;
    sp = (PixarLogState *)tif->tif_data;
    _TIFFmemset(sp, 0, sizeof(*sp));
    sp->stream.data_type = Z_BINARY;
    sp->user_datafmt = PIXARLOGDATAFMT_UNKNOWN;

    /*
     * Install codec methods.
     */
    tif->tif_fixuptags = PixarLogFixupTags;
    tif->tif_setupdecode = PixarLogSetupDecode;
    tif->tif_predecode = PixarLogPreDecode;
    tif->tif_decoderow = PixarLogDecode;
    tif->tif_decodestrip = PixarLogDecode;
    tif->tif_decodetile = PixarLogDecode;
    tif->tif_setupencode = PixarLogSetupEncode;
    tif->tif_preencode = PixarLogPreEncode;
    tif->tif_postencode = PixarLogPostEncode;
    tif->tif_encoderow = PixarLogEncode;
    tif->tif_encodestrip = PixarLogEncode;
    tif->tif_encodetile = PixarLogEncode;
    tif->tif_close = PixarLogClose;
    tif->tif_cleanup = PixarLogCleanup;

    /* Override SetField so we can handle our private pseudo-tag */
    sp->vgetparent = tif->tif_tagmethods.vgetfield;
    tif->tif_tagmethods.vgetfield = PixarLogVGetField; /* hook for codec tags */
    sp->vsetparent = tif->tif_tagmethods.vsetfield;
    tif->tif_tagmethods.vsetfield = PixarLogVSetField; /* hook for codec tags */

    /* Default values for codec-specific fields */
    sp->quality = Z_DEFAULT_COMPRESSION; /* default comp. level */
    sp->state = 0;

    /* we don't wish to use the predictor,
     * the default is none, which predictor value 1
     */
    (void)TIFFPredictorInit(tif);

    /*
     * build the companding tables
     */
    PixarLogMakeTables(tif, sp);

    return (1);
bad:
    TIFFErrorExtR(tif, module, "No space for PixarLog state block");
    return (0);
}
#endif /* PIXARLOG_SUPPORT */
