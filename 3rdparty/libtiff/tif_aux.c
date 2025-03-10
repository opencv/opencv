/*
 * Copyright (c) 1991-1997 Sam Leffler
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
 * TIFF Library.
 *
 * Auxiliary Support Routines.
 */
#include "tif_predict.h"
#include "tiffiop.h"
#include <float.h>
#include <math.h>

uint32_t _TIFFMultiply32(TIFF *tif, uint32_t first, uint32_t second,
                         const char *where)
{
    if (second && first > UINT32_MAX / second)
    {
        TIFFErrorExtR(tif, where, "Integer overflow in %s", where);
        return 0;
    }

    return first * second;
}

uint64_t _TIFFMultiply64(TIFF *tif, uint64_t first, uint64_t second,
                         const char *where)
{
    if (second && first > UINT64_MAX / second)
    {
        TIFFErrorExtR(tif, where, "Integer overflow in %s", where);
        return 0;
    }

    return first * second;
}

tmsize_t _TIFFMultiplySSize(TIFF *tif, tmsize_t first, tmsize_t second,
                            const char *where)
{
    if (first <= 0 || second <= 0)
    {
        if (tif != NULL && where != NULL)
        {
            TIFFErrorExtR(tif, where,
                          "Invalid argument to _TIFFMultiplySSize() in %s",
                          where);
        }
        return 0;
    }

    if (first > TIFF_TMSIZE_T_MAX / second)
    {
        if (tif != NULL && where != NULL)
        {
            TIFFErrorExtR(tif, where, "Integer overflow in %s", where);
        }
        return 0;
    }
    return first * second;
}

tmsize_t _TIFFCastUInt64ToSSize(TIFF *tif, uint64_t val, const char *module)
{
    if (val > (uint64_t)TIFF_TMSIZE_T_MAX)
    {
        if (tif != NULL && module != NULL)
        {
            TIFFErrorExtR(tif, module, "Integer overflow");
        }
        return 0;
    }
    return (tmsize_t)val;
}

void *_TIFFCheckRealloc(TIFF *tif, void *buffer, tmsize_t nmemb,
                        tmsize_t elem_size, const char *what)
{
    void *cp = NULL;
    tmsize_t count = _TIFFMultiplySSize(tif, nmemb, elem_size, NULL);
    /*
     * Check for integer overflow.
     */
    if (count != 0)
    {
        cp = _TIFFreallocExt(tif, buffer, count);
    }

    if (cp == NULL)
    {
        TIFFErrorExtR(tif, tif->tif_name,
                      "Failed to allocate memory for %s "
                      "(%" TIFF_SSIZE_FORMAT " elements of %" TIFF_SSIZE_FORMAT
                      " bytes each)",
                      what, nmemb, elem_size);
    }

    return cp;
}

void *_TIFFCheckMalloc(TIFF *tif, tmsize_t nmemb, tmsize_t elem_size,
                       const char *what)
{
    return _TIFFCheckRealloc(tif, NULL, nmemb, elem_size, what);
}

static int TIFFDefaultTransferFunction(TIFF *tif, TIFFDirectory *td)
{
    uint16_t **tf = td->td_transferfunction;
    tmsize_t i, n, nbytes;

    tf[0] = tf[1] = tf[2] = 0;
    if (td->td_bitspersample >= sizeof(tmsize_t) * 8 - 2)
        return 0;

    n = ((tmsize_t)1) << td->td_bitspersample;
    nbytes = n * sizeof(uint16_t);
    tf[0] = (uint16_t *)_TIFFmallocExt(tif, nbytes);
    if (tf[0] == NULL)
        return 0;
    tf[0][0] = 0;
    for (i = 1; i < n; i++)
    {
        double t = (double)i / ((double)n - 1.);
        tf[0][i] = (uint16_t)floor(65535. * pow(t, 2.2) + .5);
    }

    if (td->td_samplesperpixel - td->td_extrasamples > 1)
    {
        tf[1] = (uint16_t *)_TIFFmallocExt(tif, nbytes);
        if (tf[1] == NULL)
            goto bad;
        _TIFFmemcpy(tf[1], tf[0], nbytes);
        tf[2] = (uint16_t *)_TIFFmallocExt(tif, nbytes);
        if (tf[2] == NULL)
            goto bad;
        _TIFFmemcpy(tf[2], tf[0], nbytes);
    }
    return 1;

bad:
    if (tf[0])
        _TIFFfreeExt(tif, tf[0]);
    if (tf[1])
        _TIFFfreeExt(tif, tf[1]);
    if (tf[2])
        _TIFFfreeExt(tif, tf[2]);
    tf[0] = tf[1] = tf[2] = 0;
    return 0;
}

static int TIFFDefaultRefBlackWhite(TIFF *tif, TIFFDirectory *td)
{
    int i;

    td->td_refblackwhite = (float *)_TIFFmallocExt(tif, 6 * sizeof(float));
    if (td->td_refblackwhite == NULL)
        return 0;
    if (td->td_photometric == PHOTOMETRIC_YCBCR)
    {
        /*
         * YCbCr (Class Y) images must have the ReferenceBlackWhite
         * tag set. Fix the broken images, which lacks that tag.
         */
        td->td_refblackwhite[0] = 0.0F;
        td->td_refblackwhite[1] = td->td_refblackwhite[3] =
            td->td_refblackwhite[5] = 255.0F;
        td->td_refblackwhite[2] = td->td_refblackwhite[4] = 128.0F;
    }
    else
    {
        /*
         * Assume RGB (Class R)
         */
        for (i = 0; i < 3; i++)
        {
            td->td_refblackwhite[2 * i + 0] = 0;
            td->td_refblackwhite[2 * i + 1] =
                (float)((1L << td->td_bitspersample) - 1L);
        }
    }
    return 1;
}

/*
 * Like TIFFGetField, but return any default
 * value if the tag is not present in the directory.
 *
 * NB:	We use the value in the directory, rather than
 *	explicit values so that defaults exist only one
 *	place in the library -- in TIFFDefaultDirectory.
 */
int TIFFVGetFieldDefaulted(TIFF *tif, uint32_t tag, va_list ap)
{
    TIFFDirectory *td = &tif->tif_dir;

    if (TIFFVGetField(tif, tag, ap))
        return (1);
    switch (tag)
    {
        case TIFFTAG_SUBFILETYPE:
            *va_arg(ap, uint32_t *) = td->td_subfiletype;
            return (1);
        case TIFFTAG_BITSPERSAMPLE:
            *va_arg(ap, uint16_t *) = td->td_bitspersample;
            return (1);
        case TIFFTAG_THRESHHOLDING:
            *va_arg(ap, uint16_t *) = td->td_threshholding;
            return (1);
        case TIFFTAG_FILLORDER:
            *va_arg(ap, uint16_t *) = td->td_fillorder;
            return (1);
        case TIFFTAG_ORIENTATION:
            *va_arg(ap, uint16_t *) = td->td_orientation;
            return (1);
        case TIFFTAG_SAMPLESPERPIXEL:
            *va_arg(ap, uint16_t *) = td->td_samplesperpixel;
            return (1);
        case TIFFTAG_ROWSPERSTRIP:
            *va_arg(ap, uint32_t *) = td->td_rowsperstrip;
            return (1);
        case TIFFTAG_MINSAMPLEVALUE:
            *va_arg(ap, uint16_t *) = td->td_minsamplevalue;
            return (1);
        case TIFFTAG_MAXSAMPLEVALUE:
        {
            uint16_t maxsamplevalue;
            /* td_bitspersample=1 is always set in TIFFDefaultDirectory().
             * Therefore, td_maxsamplevalue has to be re-calculated in
             * TIFFGetFieldDefaulted(). */
            if (td->td_bitspersample > 0)
            {
                /* This shift operation into a uint16_t limits the value to
                 * 65535 even if td_bitspersamle is > 16 */
                if (td->td_bitspersample <= 16)
                {
                    maxsamplevalue = (1 << td->td_bitspersample) -
                                     1; /* 2**(BitsPerSample) - 1 */
                }
                else
                {
                    maxsamplevalue = 65535;
                }
            }
            else
            {
                maxsamplevalue = 0;
            }
            *va_arg(ap, uint16_t *) = maxsamplevalue;
            return (1);
        }
        case TIFFTAG_PLANARCONFIG:
            *va_arg(ap, uint16_t *) = td->td_planarconfig;
            return (1);
        case TIFFTAG_RESOLUTIONUNIT:
            *va_arg(ap, uint16_t *) = td->td_resolutionunit;
            return (1);
        case TIFFTAG_PREDICTOR:
        {
            TIFFPredictorState *sp = (TIFFPredictorState *)tif->tif_data;
            if (sp == NULL)
            {
                TIFFErrorExtR(
                    tif, tif->tif_name,
                    "Cannot get \"Predictor\" tag as plugin is not configured");
                *va_arg(ap, uint16_t *) = 0;
                return 0;
            }
            *va_arg(ap, uint16_t *) = (uint16_t)sp->predictor;
            return 1;
        }
        case TIFFTAG_DOTRANGE:
            *va_arg(ap, uint16_t *) = 0;
            *va_arg(ap, uint16_t *) = (1 << td->td_bitspersample) - 1;
            return (1);
        case TIFFTAG_INKSET:
            *va_arg(ap, uint16_t *) = INKSET_CMYK;
            return 1;
        case TIFFTAG_NUMBEROFINKS:
            *va_arg(ap, uint16_t *) = 4;
            return (1);
        case TIFFTAG_EXTRASAMPLES:
            *va_arg(ap, uint16_t *) = td->td_extrasamples;
            *va_arg(ap, const uint16_t **) = td->td_sampleinfo;
            return (1);
        case TIFFTAG_MATTEING:
            *va_arg(ap, uint16_t *) =
                (td->td_extrasamples == 1 &&
                 td->td_sampleinfo[0] == EXTRASAMPLE_ASSOCALPHA);
            return (1);
        case TIFFTAG_TILEDEPTH:
            *va_arg(ap, uint32_t *) = td->td_tiledepth;
            return (1);
        case TIFFTAG_DATATYPE:
            *va_arg(ap, uint16_t *) = td->td_sampleformat - 1;
            return (1);
        case TIFFTAG_SAMPLEFORMAT:
            *va_arg(ap, uint16_t *) = td->td_sampleformat;
            return (1);
        case TIFFTAG_IMAGEDEPTH:
            *va_arg(ap, uint32_t *) = td->td_imagedepth;
            return (1);
        case TIFFTAG_YCBCRCOEFFICIENTS:
        {
            /* defaults are from CCIR Recommendation 601-1 */
            static const float ycbcrcoeffs[] = {0.299f, 0.587f, 0.114f};
            *va_arg(ap, const float **) = ycbcrcoeffs;
            return 1;
        }
        case TIFFTAG_YCBCRSUBSAMPLING:
            *va_arg(ap, uint16_t *) = td->td_ycbcrsubsampling[0];
            *va_arg(ap, uint16_t *) = td->td_ycbcrsubsampling[1];
            return (1);
        case TIFFTAG_YCBCRPOSITIONING:
            *va_arg(ap, uint16_t *) = td->td_ycbcrpositioning;
            return (1);
        case TIFFTAG_WHITEPOINT:
        {
            /* TIFF 6.0 specification tells that it is no default
               value for the WhitePoint, but AdobePhotoshop TIFF
               Technical Note tells that it should be CIE D50. */
            static const float whitepoint[] = {
                D50_X0 / (D50_X0 + D50_Y0 + D50_Z0),
                D50_Y0 / (D50_X0 + D50_Y0 + D50_Z0)};
            *va_arg(ap, const float **) = whitepoint;
            return 1;
        }
        case TIFFTAG_TRANSFERFUNCTION:
            if (!td->td_transferfunction[0] &&
                !TIFFDefaultTransferFunction(tif, td))
            {
                TIFFErrorExtR(tif, tif->tif_name,
                              "No space for \"TransferFunction\" tag");
                return (0);
            }
            *va_arg(ap, const uint16_t **) = td->td_transferfunction[0];
            if (td->td_samplesperpixel - td->td_extrasamples > 1)
            {
                *va_arg(ap, const uint16_t **) = td->td_transferfunction[1];
                *va_arg(ap, const uint16_t **) = td->td_transferfunction[2];
            }
            return (1);
        case TIFFTAG_REFERENCEBLACKWHITE:
            if (!td->td_refblackwhite && !TIFFDefaultRefBlackWhite(tif, td))
                return (0);
            *va_arg(ap, const float **) = td->td_refblackwhite;
            return (1);
    }
    return 0;
}

/*
 * Like TIFFGetField, but return any default
 * value if the tag is not present in the directory.
 */
int TIFFGetFieldDefaulted(TIFF *tif, uint32_t tag, ...)
{
    int ok;
    va_list ap;

    va_start(ap, tag);
    ok = TIFFVGetFieldDefaulted(tif, tag, ap);
    va_end(ap);
    return (ok);
}

struct _Int64Parts
{
    int32_t low, high;
};

typedef union
{
    struct _Int64Parts part;
    int64_t value;
} _Int64;

float _TIFFUInt64ToFloat(uint64_t ui64)
{
    _Int64 i;

    i.value = ui64;
    if (i.part.high >= 0)
    {
        return (float)i.value;
    }
    else
    {
        long double df;
        df = (long double)i.value;
        df += 18446744073709551616.0; /* adding 2**64 */
        return (float)df;
    }
}

double _TIFFUInt64ToDouble(uint64_t ui64)
{
    _Int64 i;

    i.value = ui64;
    if (i.part.high >= 0)
    {
        return (double)i.value;
    }
    else
    {
        long double df;
        df = (long double)i.value;
        df += 18446744073709551616.0; /* adding 2**64 */
        return (double)df;
    }
}

float _TIFFClampDoubleToFloat(double val)
{
    if (val > FLT_MAX)
        return FLT_MAX;
    if (val < -FLT_MAX)
        return -FLT_MAX;
    return (float)val;
}

uint32_t _TIFFClampDoubleToUInt32(double val)
{
    if (val < 0)
        return 0;
    if (val > 0xFFFFFFFFU || val != val)
        return 0xFFFFFFFFU;
    return (uint32_t)val;
}

int _TIFFSeekOK(TIFF *tif, toff_t off)
{
    /* Huge offsets, especially -1 / UINT64_MAX, can cause issues */
    /* See http://bugzilla.maptools.org/show_bug.cgi?id=2726 */
    return off <= (~(uint64_t)0) / 2 && TIFFSeekFile(tif, off, SEEK_SET) == off;
}
