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
 * TIFF Library.
 *
 * Directory Tag Get & Set Routines.
 * (and also some miscellaneous stuff)
 */
#include "tiffiop.h"
#include <float.h> /*--: for Rational2Double */
#include <limits.h>

/*
 * These are used in the backwards compatibility code...
 */
#define DATATYPE_VOID 0   /* !untyped data */
#define DATATYPE_INT 1    /* !signed integer data */
#define DATATYPE_UINT 2   /* !unsigned integer data */
#define DATATYPE_IEEEFP 3 /* !IEEE floating point data */

static void setByteArray(TIFF *tif, void **vpp, const void *vp, size_t nmemb,
                         size_t elem_size)
{
    if (*vpp)
    {
        _TIFFfreeExt(tif, *vpp);
        *vpp = 0;
    }
    if (vp)
    {
        tmsize_t bytes = _TIFFMultiplySSize(NULL, nmemb, elem_size, NULL);
        if (bytes)
            *vpp = (void *)_TIFFmallocExt(tif, bytes);
        if (*vpp)
            _TIFFmemcpy(*vpp, vp, bytes);
    }
}
void _TIFFsetByteArray(void **vpp, const void *vp, uint32_t n)
{
    setByteArray(NULL, vpp, vp, n, 1);
}
void _TIFFsetByteArrayExt(TIFF *tif, void **vpp, const void *vp, uint32_t n)
{
    setByteArray(tif, vpp, vp, n, 1);
}

static void _TIFFsetNString(TIFF *tif, char **cpp, const char *cp, uint32_t n)
{
    setByteArray(tif, (void **)cpp, cp, n, 1);
}

void _TIFFsetShortArray(uint16_t **wpp, const uint16_t *wp, uint32_t n)
{
    setByteArray(NULL, (void **)wpp, wp, n, sizeof(uint16_t));
}
void _TIFFsetShortArrayExt(TIFF *tif, uint16_t **wpp, const uint16_t *wp,
                           uint32_t n)
{
    setByteArray(tif, (void **)wpp, wp, n, sizeof(uint16_t));
}

void _TIFFsetLongArray(uint32_t **lpp, const uint32_t *lp, uint32_t n)
{
    setByteArray(NULL, (void **)lpp, lp, n, sizeof(uint32_t));
}
void _TIFFsetLongArrayExt(TIFF *tif, uint32_t **lpp, const uint32_t *lp,
                          uint32_t n)
{
    setByteArray(tif, (void **)lpp, lp, n, sizeof(uint32_t));
}

static void _TIFFsetLong8Array(TIFF *tif, uint64_t **lpp, const uint64_t *lp,
                               uint32_t n)
{
    setByteArray(tif, (void **)lpp, lp, n, sizeof(uint64_t));
}

void _TIFFsetFloatArray(float **fpp, const float *fp, uint32_t n)
{
    setByteArray(NULL, (void **)fpp, fp, n, sizeof(float));
}
void _TIFFsetFloatArrayExt(TIFF *tif, float **fpp, const float *fp, uint32_t n)
{
    setByteArray(tif, (void **)fpp, fp, n, sizeof(float));
}

void _TIFFsetDoubleArray(double **dpp, const double *dp, uint32_t n)
{
    setByteArray(NULL, (void **)dpp, dp, n, sizeof(double));
}
void _TIFFsetDoubleArrayExt(TIFF *tif, double **dpp, const double *dp,
                            uint32_t n)
{
    setByteArray(tif, (void **)dpp, dp, n, sizeof(double));
}

static void setDoubleArrayOneValue(TIFF *tif, double **vpp, double value,
                                   size_t nmemb)
{
    if (*vpp)
        _TIFFfreeExt(tif, *vpp);
    *vpp = _TIFFmallocExt(tif, nmemb * sizeof(double));
    if (*vpp)
    {
        while (nmemb--)
            ((double *)*vpp)[nmemb] = value;
    }
}

/*
 * Install extra samples information.
 */
static int setExtraSamples(TIFF *tif, va_list ap, uint32_t *v)
{
/* XXX: Unassociated alpha data == 999 is a known Corel Draw bug, see below */
#define EXTRASAMPLE_COREL_UNASSALPHA 999

    uint16_t *va;
    uint32_t i;
    TIFFDirectory *td = &tif->tif_dir;
    static const char module[] = "setExtraSamples";

    *v = (uint16_t)va_arg(ap, uint16_vap);
    if ((uint16_t)*v > td->td_samplesperpixel)
        return 0;
    va = va_arg(ap, uint16_t *);
    if (*v > 0 && va == NULL) /* typically missing param */
        return 0;
    for (i = 0; i < *v; i++)
    {
        if (va[i] > EXTRASAMPLE_UNASSALPHA)
        {
            /*
             * XXX: Corel Draw is known to produce incorrect
             * ExtraSamples tags which must be patched here if we
             * want to be able to open some of the damaged TIFF
             * files:
             */
            if (va[i] == EXTRASAMPLE_COREL_UNASSALPHA)
                va[i] = EXTRASAMPLE_UNASSALPHA;
            else
                return 0;
        }
    }

    if (td->td_transferfunction[0] != NULL &&
        (td->td_samplesperpixel - *v > 1) &&
        !(td->td_samplesperpixel - td->td_extrasamples > 1))
    {
        TIFFWarningExtR(tif, module,
                        "ExtraSamples tag value is changing, "
                        "but TransferFunction was read with a different value. "
                        "Canceling it");
        TIFFClrFieldBit(tif, FIELD_TRANSFERFUNCTION);
        _TIFFfreeExt(tif, td->td_transferfunction[0]);
        td->td_transferfunction[0] = NULL;
    }

    td->td_extrasamples = (uint16_t)*v;
    _TIFFsetShortArrayExt(tif, &td->td_sampleinfo, va, td->td_extrasamples);
    return 1;

#undef EXTRASAMPLE_COREL_UNASSALPHA
}

/*
 * Count ink names separated by \0.  Returns
 * zero if the ink names are not as expected.
 */
static uint16_t countInkNamesString(TIFF *tif, uint32_t slen, const char *s)
{
    uint16_t i = 0;

    if (slen > 0)
    {
        const char *ep = s + slen;
        const char *cp = s;
        do
        {
            for (; cp < ep && *cp != '\0'; cp++)
            {
            }
            if (cp >= ep)
                goto bad;
            cp++; /* skip \0 */
            i++;
        } while (cp < ep);
        return (i);
    }
bad:
    TIFFErrorExtR(tif, "TIFFSetField",
                  "%s: Invalid InkNames value; no null at given buffer end "
                  "location %" PRIu32 ", after %" PRIu16 " ink",
                  tif->tif_name, slen, i);
    return (0);
}

static int _TIFFVSetField(TIFF *tif, uint32_t tag, va_list ap)
{
    static const char module[] = "_TIFFVSetField";

    TIFFDirectory *td = &tif->tif_dir;
    int status = 1;
    uint32_t v32, v;
    double dblval;
    char *s;
    const TIFFField *fip = TIFFFindField(tif, tag, TIFF_ANY);
    uint32_t standard_tag = tag;
    if (fip == NULL) /* cannot happen since OkToChangeTag() already checks it */
        return 0;
    /*
     * We want to force the custom code to be used for custom
     * fields even if the tag happens to match a well known
     * one - important for reinterpreted handling of standard
     * tag values in custom directories (i.e. EXIF)
     */
    if (fip->field_bit == FIELD_CUSTOM)
    {
        standard_tag = 0;
    }

    switch (standard_tag)
    {
        case TIFFTAG_SUBFILETYPE:
            td->td_subfiletype = (uint32_t)va_arg(ap, uint32_t);
            break;
        case TIFFTAG_IMAGEWIDTH:
            td->td_imagewidth = (uint32_t)va_arg(ap, uint32_t);
            break;
        case TIFFTAG_IMAGELENGTH:
            td->td_imagelength = (uint32_t)va_arg(ap, uint32_t);
            break;
        case TIFFTAG_BITSPERSAMPLE:
            td->td_bitspersample = (uint16_t)va_arg(ap, uint16_vap);
            /*
             * If the data require post-decoding processing to byte-swap
             * samples, set it up here.  Note that since tags are required
             * to be ordered, compression code can override this behavior
             * in the setup method if it wants to roll the post decoding
             * work in with its normal work.
             */
            if (tif->tif_flags & TIFF_SWAB)
            {
                if (td->td_bitspersample == 8)
                    tif->tif_postdecode = _TIFFNoPostDecode;
                else if (td->td_bitspersample == 16)
                    tif->tif_postdecode = _TIFFSwab16BitData;
                else if (td->td_bitspersample == 24)
                    tif->tif_postdecode = _TIFFSwab24BitData;
                else if (td->td_bitspersample == 32)
                    tif->tif_postdecode = _TIFFSwab32BitData;
                else if (td->td_bitspersample == 64)
                    tif->tif_postdecode = _TIFFSwab64BitData;
                else if (td->td_bitspersample == 128) /* two 64's */
                    tif->tif_postdecode = _TIFFSwab64BitData;
            }
            break;
        case TIFFTAG_COMPRESSION:
            v = (uint16_t)va_arg(ap, uint16_vap);
            /*
             * If we're changing the compression scheme, notify the
             * previous module so that it can cleanup any state it's
             * setup.
             */
            if (TIFFFieldSet(tif, FIELD_COMPRESSION))
            {
                if ((uint32_t)td->td_compression == v)
                    break;
                (*tif->tif_cleanup)(tif);
                tif->tif_flags &= ~TIFF_CODERSETUP;
            }
            /*
             * Setup new compression routine state.
             */
            if ((status = TIFFSetCompressionScheme(tif, v)) != 0)
                td->td_compression = (uint16_t)v;
            else
                status = 0;
            break;
        case TIFFTAG_PHOTOMETRIC:
            td->td_photometric = (uint16_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_THRESHHOLDING:
            td->td_threshholding = (uint16_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_FILLORDER:
            v = (uint16_t)va_arg(ap, uint16_vap);
            if (v != FILLORDER_LSB2MSB && v != FILLORDER_MSB2LSB)
                goto badvalue;
            td->td_fillorder = (uint16_t)v;
            break;
        case TIFFTAG_ORIENTATION:
            v = (uint16_t)va_arg(ap, uint16_vap);
            if (v < ORIENTATION_TOPLEFT || ORIENTATION_LEFTBOT < v)
                goto badvalue;
            else
                td->td_orientation = (uint16_t)v;
            break;
        case TIFFTAG_SAMPLESPERPIXEL:
            v = (uint16_t)va_arg(ap, uint16_vap);
            if (v == 0)
                goto badvalue;
            if (v != td->td_samplesperpixel)
            {
                /* See http://bugzilla.maptools.org/show_bug.cgi?id=2500 */
                if (td->td_sminsamplevalue != NULL)
                {
                    TIFFWarningExtR(tif, module,
                                    "SamplesPerPixel tag value is changing, "
                                    "but SMinSampleValue tag was read with a "
                                    "different value. Canceling it");
                    TIFFClrFieldBit(tif, FIELD_SMINSAMPLEVALUE);
                    _TIFFfreeExt(tif, td->td_sminsamplevalue);
                    td->td_sminsamplevalue = NULL;
                }
                if (td->td_smaxsamplevalue != NULL)
                {
                    TIFFWarningExtR(tif, module,
                                    "SamplesPerPixel tag value is changing, "
                                    "but SMaxSampleValue tag was read with a "
                                    "different value. Canceling it");
                    TIFFClrFieldBit(tif, FIELD_SMAXSAMPLEVALUE);
                    _TIFFfreeExt(tif, td->td_smaxsamplevalue);
                    td->td_smaxsamplevalue = NULL;
                }
                /* Test if 3 transfer functions instead of just one are now
                   needed See http://bugzilla.maptools.org/show_bug.cgi?id=2820
                 */
                if (td->td_transferfunction[0] != NULL &&
                    (v - td->td_extrasamples > 1) &&
                    !(td->td_samplesperpixel - td->td_extrasamples > 1))
                {
                    TIFFWarningExtR(tif, module,
                                    "SamplesPerPixel tag value is changing, "
                                    "but TransferFunction was read with a "
                                    "different value. Canceling it");
                    TIFFClrFieldBit(tif, FIELD_TRANSFERFUNCTION);
                    _TIFFfreeExt(tif, td->td_transferfunction[0]);
                    td->td_transferfunction[0] = NULL;
                }
            }
            td->td_samplesperpixel = (uint16_t)v;
            break;
        case TIFFTAG_ROWSPERSTRIP:
            v32 = (uint32_t)va_arg(ap, uint32_t);
            if (v32 == 0)
                goto badvalue32;
            td->td_rowsperstrip = v32;
            if (!TIFFFieldSet(tif, FIELD_TILEDIMENSIONS))
            {
                td->td_tilelength = v32;
                td->td_tilewidth = td->td_imagewidth;
            }
            break;
        case TIFFTAG_MINSAMPLEVALUE:
            td->td_minsamplevalue = (uint16_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_MAXSAMPLEVALUE:
            td->td_maxsamplevalue = (uint16_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_SMINSAMPLEVALUE:
            if (tif->tif_flags & TIFF_PERSAMPLE)
                _TIFFsetDoubleArrayExt(tif, &td->td_sminsamplevalue,
                                       va_arg(ap, double *),
                                       td->td_samplesperpixel);
            else
                setDoubleArrayOneValue(tif, &td->td_sminsamplevalue,
                                       va_arg(ap, double),
                                       td->td_samplesperpixel);
            break;
        case TIFFTAG_SMAXSAMPLEVALUE:
            if (tif->tif_flags & TIFF_PERSAMPLE)
                _TIFFsetDoubleArrayExt(tif, &td->td_smaxsamplevalue,
                                       va_arg(ap, double *),
                                       td->td_samplesperpixel);
            else
                setDoubleArrayOneValue(tif, &td->td_smaxsamplevalue,
                                       va_arg(ap, double),
                                       td->td_samplesperpixel);
            break;
        case TIFFTAG_XRESOLUTION:
            dblval = va_arg(ap, double);
            if (dblval != dblval || dblval < 0)
                goto badvaluedouble;
            td->td_xresolution = _TIFFClampDoubleToFloat(dblval);
            break;
        case TIFFTAG_YRESOLUTION:
            dblval = va_arg(ap, double);
            if (dblval != dblval || dblval < 0)
                goto badvaluedouble;
            td->td_yresolution = _TIFFClampDoubleToFloat(dblval);
            break;
        case TIFFTAG_PLANARCONFIG:
            v = (uint16_t)va_arg(ap, uint16_vap);
            if (v != PLANARCONFIG_CONTIG && v != PLANARCONFIG_SEPARATE)
                goto badvalue;
            td->td_planarconfig = (uint16_t)v;
            break;
        case TIFFTAG_XPOSITION:
            td->td_xposition = _TIFFClampDoubleToFloat(va_arg(ap, double));
            break;
        case TIFFTAG_YPOSITION:
            td->td_yposition = _TIFFClampDoubleToFloat(va_arg(ap, double));
            break;
        case TIFFTAG_RESOLUTIONUNIT:
            v = (uint16_t)va_arg(ap, uint16_vap);
            if (v < RESUNIT_NONE || RESUNIT_CENTIMETER < v)
                goto badvalue;
            td->td_resolutionunit = (uint16_t)v;
            break;
        case TIFFTAG_PAGENUMBER:
            td->td_pagenumber[0] = (uint16_t)va_arg(ap, uint16_vap);
            td->td_pagenumber[1] = (uint16_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_HALFTONEHINTS:
            td->td_halftonehints[0] = (uint16_t)va_arg(ap, uint16_vap);
            td->td_halftonehints[1] = (uint16_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_COLORMAP:
            v32 = (uint32_t)(1L << td->td_bitspersample);
            _TIFFsetShortArrayExt(tif, &td->td_colormap[0],
                                  va_arg(ap, uint16_t *), v32);
            _TIFFsetShortArrayExt(tif, &td->td_colormap[1],
                                  va_arg(ap, uint16_t *), v32);
            _TIFFsetShortArrayExt(tif, &td->td_colormap[2],
                                  va_arg(ap, uint16_t *), v32);
            break;
        case TIFFTAG_EXTRASAMPLES:
            if (!setExtraSamples(tif, ap, &v))
                goto badvalue;
            break;
        case TIFFTAG_MATTEING:
            td->td_extrasamples = (((uint16_t)va_arg(ap, uint16_vap)) != 0);
            if (td->td_extrasamples)
            {
                uint16_t sv = EXTRASAMPLE_ASSOCALPHA;
                _TIFFsetShortArrayExt(tif, &td->td_sampleinfo, &sv, 1);
            }
            break;
        case TIFFTAG_TILEWIDTH:
            v32 = (uint32_t)va_arg(ap, uint32_t);
            if (v32 % 16)
            {
                if (tif->tif_mode != O_RDONLY)
                    goto badvalue32;
                TIFFWarningExtR(
                    tif, tif->tif_name,
                    "Nonstandard tile width %" PRIu32 ", convert file", v32);
            }
            td->td_tilewidth = v32;
            tif->tif_flags |= TIFF_ISTILED;
            break;
        case TIFFTAG_TILELENGTH:
            v32 = (uint32_t)va_arg(ap, uint32_t);
            if (v32 % 16)
            {
                if (tif->tif_mode != O_RDONLY)
                    goto badvalue32;
                TIFFWarningExtR(
                    tif, tif->tif_name,
                    "Nonstandard tile length %" PRIu32 ", convert file", v32);
            }
            td->td_tilelength = v32;
            tif->tif_flags |= TIFF_ISTILED;
            break;
        case TIFFTAG_TILEDEPTH:
            v32 = (uint32_t)va_arg(ap, uint32_t);
            if (v32 == 0)
                goto badvalue32;
            td->td_tiledepth = v32;
            break;
        case TIFFTAG_DATATYPE:
            v = (uint16_t)va_arg(ap, uint16_vap);
            switch (v)
            {
                case DATATYPE_VOID:
                    v = SAMPLEFORMAT_VOID;
                    break;
                case DATATYPE_INT:
                    v = SAMPLEFORMAT_INT;
                    break;
                case DATATYPE_UINT:
                    v = SAMPLEFORMAT_UINT;
                    break;
                case DATATYPE_IEEEFP:
                    v = SAMPLEFORMAT_IEEEFP;
                    break;
                default:
                    goto badvalue;
            }
            td->td_sampleformat = (uint16_t)v;
            break;
        case TIFFTAG_SAMPLEFORMAT:
            v = (uint16_t)va_arg(ap, uint16_vap);
            if (v < SAMPLEFORMAT_UINT || SAMPLEFORMAT_COMPLEXIEEEFP < v)
                goto badvalue;
            td->td_sampleformat = (uint16_t)v;

            /*  Try to fix up the SWAB function for complex data. */
            if (td->td_sampleformat == SAMPLEFORMAT_COMPLEXINT &&
                td->td_bitspersample == 32 &&
                tif->tif_postdecode == _TIFFSwab32BitData)
                tif->tif_postdecode = _TIFFSwab16BitData;
            else if ((td->td_sampleformat == SAMPLEFORMAT_COMPLEXINT ||
                      td->td_sampleformat == SAMPLEFORMAT_COMPLEXIEEEFP) &&
                     td->td_bitspersample == 64 &&
                     tif->tif_postdecode == _TIFFSwab64BitData)
                tif->tif_postdecode = _TIFFSwab32BitData;
            break;
        case TIFFTAG_IMAGEDEPTH:
            td->td_imagedepth = (uint32_t)va_arg(ap, uint32_t);
            break;
        case TIFFTAG_SUBIFD:
            if ((tif->tif_flags & TIFF_INSUBIFD) == 0)
            {
                td->td_nsubifd = (uint16_t)va_arg(ap, uint16_vap);
                _TIFFsetLong8Array(tif, &td->td_subifd,
                                   (uint64_t *)va_arg(ap, uint64_t *),
                                   (uint32_t)td->td_nsubifd);
            }
            else
            {
                TIFFErrorExtR(tif, module, "%s: Sorry, cannot nest SubIFDs",
                              tif->tif_name);
                status = 0;
            }
            break;
        case TIFFTAG_YCBCRPOSITIONING:
            td->td_ycbcrpositioning = (uint16_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_YCBCRSUBSAMPLING:
            td->td_ycbcrsubsampling[0] = (uint16_t)va_arg(ap, uint16_vap);
            td->td_ycbcrsubsampling[1] = (uint16_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_TRANSFERFUNCTION:
        {
            uint32_t i;
            v = (td->td_samplesperpixel - td->td_extrasamples) > 1 ? 3 : 1;
            for (i = 0; i < v; i++)
                _TIFFsetShortArrayExt(tif, &td->td_transferfunction[i],
                                      va_arg(ap, uint16_t *),
                                      1U << td->td_bitspersample);
            break;
        }
        case TIFFTAG_REFERENCEBLACKWHITE:
            /* XXX should check for null range */
            _TIFFsetFloatArrayExt(tif, &td->td_refblackwhite,
                                  va_arg(ap, float *), 6);
            break;
        case TIFFTAG_INKNAMES:
        {
            v = (uint16_t)va_arg(ap, uint16_vap);
            s = va_arg(ap, char *);
            uint16_t ninksinstring;
            ninksinstring = countInkNamesString(tif, v, s);
            status = ninksinstring > 0;
            if (ninksinstring > 0)
            {
                _TIFFsetNString(tif, &td->td_inknames, s, v);
                td->td_inknameslen = v;
                /* Set NumberOfInks to the value ninksinstring */
                if (TIFFFieldSet(tif, FIELD_NUMBEROFINKS))
                {
                    if (td->td_numberofinks != ninksinstring)
                    {
                        TIFFErrorExtR(
                            tif, module,
                            "Warning %s; Tag %s:\n  Value %" PRIu16
                            " of NumberOfInks is different from the number of "
                            "inks %" PRIu16
                            ".\n  -> NumberOfInks value adapted to %" PRIu16 "",
                            tif->tif_name, fip->field_name, td->td_numberofinks,
                            ninksinstring, ninksinstring);
                        td->td_numberofinks = ninksinstring;
                    }
                }
                else
                {
                    td->td_numberofinks = ninksinstring;
                    TIFFSetFieldBit(tif, FIELD_NUMBEROFINKS);
                }
                if (TIFFFieldSet(tif, FIELD_SAMPLESPERPIXEL))
                {
                    if (td->td_numberofinks != td->td_samplesperpixel)
                    {
                        TIFFErrorExtR(tif, module,
                                      "Warning %s; Tag %s:\n  Value %" PRIu16
                                      " of NumberOfInks is different from the "
                                      "SamplesPerPixel value %" PRIu16 "",
                                      tif->tif_name, fip->field_name,
                                      td->td_numberofinks,
                                      td->td_samplesperpixel);
                    }
                }
            }
        }
        break;
        case TIFFTAG_NUMBEROFINKS:
            v = (uint16_t)va_arg(ap, uint16_vap);
            /* If InkNames already set also NumberOfInks is set accordingly and
             * should be equal */
            if (TIFFFieldSet(tif, FIELD_INKNAMES))
            {
                if (v != td->td_numberofinks)
                {
                    TIFFErrorExtR(
                        tif, module,
                        "Error %s; Tag %s:\n  It is not possible to set the "
                        "value %" PRIu32
                        " for NumberOfInks\n  which is different from the "
                        "number of inks in the InkNames tag (%" PRIu16 ")",
                        tif->tif_name, fip->field_name, v, td->td_numberofinks);
                    /* Do not set / overwrite number of inks already set by
                     * InkNames case accordingly. */
                    status = 0;
                }
            }
            else
            {
                td->td_numberofinks = (uint16_t)v;
                if (TIFFFieldSet(tif, FIELD_SAMPLESPERPIXEL))
                {
                    if (td->td_numberofinks != td->td_samplesperpixel)
                    {
                        TIFFErrorExtR(tif, module,
                                      "Warning %s; Tag %s:\n  Value %" PRIu32
                                      " of NumberOfInks is different from the "
                                      "SamplesPerPixel value %" PRIu16 "",
                                      tif->tif_name, fip->field_name, v,
                                      td->td_samplesperpixel);
                    }
                }
            }
            break;
        case TIFFTAG_PERSAMPLE:
            v = (uint16_t)va_arg(ap, uint16_vap);
            if (v == PERSAMPLE_MULTI)
                tif->tif_flags |= TIFF_PERSAMPLE;
            else
                tif->tif_flags &= ~TIFF_PERSAMPLE;
            break;
        default:
        {
            TIFFTagValue *tv;
            int tv_size, iCustom;

            /*
             * This can happen if multiple images are open with different
             * codecs which have private tags.  The global tag information
             * table may then have tags that are valid for one file but not
             * the other. If the client tries to set a tag that is not valid
             * for the image's codec then we'll arrive here.  This
             * happens, for example, when tiffcp is used to convert between
             * compression schemes and codec-specific tags are blindly copied.
             *
             * This also happens when a FIELD_IGNORE tag is written.
             */
            if (fip->field_bit == FIELD_IGNORE)
            {
                TIFFErrorExtR(
                    tif, module,
                    "%s: Ignored %stag \"%s\" (not supported by libtiff)",
                    tif->tif_name, isPseudoTag(tag) ? "pseudo-" : "",
                    fip->field_name);
                status = 0;
                break;
            }
            if (fip->field_bit != FIELD_CUSTOM)
            {
                TIFFErrorExtR(
                    tif, module,
                    "%s: Invalid %stag \"%s\" (not supported by codec)",
                    tif->tif_name, isPseudoTag(tag) ? "pseudo-" : "",
                    fip->field_name);
                status = 0;
                break;
            }

            /*
             * Find the existing entry for this custom value.
             */
            tv = NULL;
            for (iCustom = 0; iCustom < td->td_customValueCount; iCustom++)
            {
                if (td->td_customValues[iCustom].info->field_tag == tag)
                {
                    tv = td->td_customValues + iCustom;
                    if (tv->value != NULL)
                    {
                        _TIFFfreeExt(tif, tv->value);
                        tv->value = NULL;
                    }
                    break;
                }
            }

            /*
             * Grow the custom list if the entry was not found.
             */
            if (tv == NULL)
            {
                TIFFTagValue *new_customValues;

                td->td_customValueCount++;
                new_customValues = (TIFFTagValue *)_TIFFreallocExt(
                    tif, td->td_customValues,
                    sizeof(TIFFTagValue) * td->td_customValueCount);
                if (!new_customValues)
                {
                    TIFFErrorExtR(tif, module,
                                  "%s: Failed to allocate space for list of "
                                  "custom values",
                                  tif->tif_name);
                    status = 0;
                    goto end;
                }

                td->td_customValues = new_customValues;

                tv = td->td_customValues + (td->td_customValueCount - 1);
                tv->info = fip;
                tv->value = NULL;
                tv->count = 0;
            }

            /*
             * Set custom value ... save a copy of the custom tag value.
             */
            /*--: Rational2Double: For Rationals evaluate "set_field_type" to
             * determine internal storage size. */
            tv_size = TIFFFieldSetGetSize(fip);
            if (tv_size == 0)
            {
                status = 0;
                TIFFErrorExtR(tif, module, "%s: Bad field type %d for \"%s\"",
                              tif->tif_name, fip->field_type, fip->field_name);
                goto end;
            }

            if (fip->field_type == TIFF_ASCII)
            {
                uint32_t ma;
                const char *mb;
                if (fip->field_passcount)
                {
                    assert(fip->field_writecount == TIFF_VARIABLE2);
                    ma = (uint32_t)va_arg(ap, uint32_t);
                    mb = (const char *)va_arg(ap, const char *);
                }
                else
                {
                    mb = (const char *)va_arg(ap, const char *);
                    size_t len = strlen(mb) + 1;
                    if (len >= 0x80000000U)
                    {
                        status = 0;
                        TIFFErrorExtR(tif, module,
                                      "%s: Too long string value for \"%s\". "
                                      "Maximum supported is 2147483647 bytes",
                                      tif->tif_name, fip->field_name);
                        goto end;
                    }
                    ma = (uint32_t)len;
                }
                tv->count = ma;
                setByteArray(tif, &tv->value, mb, ma, 1);
            }
            else
            {
                if (fip->field_passcount)
                {
                    if (fip->field_writecount == TIFF_VARIABLE2)
                        tv->count = (uint32_t)va_arg(ap, uint32_t);
                    else
                        tv->count = (int)va_arg(ap, int);
                }
                else if (fip->field_writecount == TIFF_VARIABLE ||
                         fip->field_writecount == TIFF_VARIABLE2)
                    tv->count = 1;
                else if (fip->field_writecount == TIFF_SPP)
                    tv->count = td->td_samplesperpixel;
                else
                    tv->count = fip->field_writecount;

                if (tv->count == 0)
                {
                    status = 0;
                    TIFFErrorExtR(tif, module,
                                  "%s: Null count for \"%s\" (type "
                                  "%d, writecount %d, passcount %d)",
                                  tif->tif_name, fip->field_name,
                                  fip->field_type, fip->field_writecount,
                                  fip->field_passcount);
                    goto end;
                }

                tv->value = _TIFFCheckMalloc(tif, tv->count, tv_size,
                                             "custom tag binary object");
                if (!tv->value)
                {
                    status = 0;
                    goto end;
                }

                if (fip->field_tag == TIFFTAG_DOTRANGE &&
                    strcmp(fip->field_name, "DotRange") == 0)
                {
                    /* TODO: This is an evil exception and should not have been
                       handled this way ... likely best if we move it into
                       the directory structure with an explicit field in
                       libtiff 4.1 and assign it a FIELD_ value */
                    uint16_t v2[2];
                    v2[0] = (uint16_t)va_arg(ap, int);
                    v2[1] = (uint16_t)va_arg(ap, int);
                    _TIFFmemcpy(tv->value, &v2, 4);
                }

                else if (fip->field_passcount ||
                         fip->field_writecount == TIFF_VARIABLE ||
                         fip->field_writecount == TIFF_VARIABLE2 ||
                         fip->field_writecount == TIFF_SPP || tv->count > 1)
                {
                    /*--: Rational2Double: For Rationals tv_size is set above to
                     * 4 or 8 according to fip->set_field_type! */
                    _TIFFmemcpy(tv->value, va_arg(ap, void *),
                                tv->count * tv_size);
                    /* Test here for too big values for LONG8, SLONG8 in
                     * ClassicTIFF and delete custom field from custom list */
                    if (!(tif->tif_flags & TIFF_BIGTIFF))
                    {
                        if (tv->info->field_type == TIFF_LONG8)
                        {
                            uint64_t *pui64 = (uint64_t *)tv->value;
                            for (int i = 0; i < tv->count; i++)
                            {
                                if (pui64[i] > 0xffffffffu)
                                {
                                    TIFFErrorExtR(
                                        tif, module,
                                        "%s: Bad LONG8 value %" PRIu64
                                        " at %d. array position for \"%s\" tag "
                                        "%d in ClassicTIFF. Tag won't be "
                                        "written to file",
                                        tif->tif_name, pui64[i], i,
                                        fip->field_name, tag);
                                    goto badvalueifd8long8;
                                }
                            }
                        }
                        else if (tv->info->field_type == TIFF_SLONG8)
                        {
                            int64_t *pi64 = (int64_t *)tv->value;
                            for (int i = 0; i < tv->count; i++)
                            {
                                if (pi64[i] > 2147483647 ||
                                    pi64[i] < (-2147483647 - 1))
                                {
                                    TIFFErrorExtR(
                                        tif, module,
                                        "%s: Bad SLONG8 value %" PRIi64
                                        " at %d. array position for \"%s\" tag "
                                        "%d in ClassicTIFF. Tag won't be "
                                        "written to file",
                                        tif->tif_name, pi64[i], i,
                                        fip->field_name, tag);
                                    goto badvalueifd8long8;
                                }
                            }
                        }
                    }
                }
                else
                {
                    char *val = (char *)tv->value;
                    assert(tv->count == 1);

                    switch (fip->field_type)
                    {
                        case TIFF_BYTE:
                        case TIFF_UNDEFINED:
                        {
                            uint8_t v2 = (uint8_t)va_arg(ap, int);
                            _TIFFmemcpy(val, &v2, tv_size);
                        }
                        break;
                        case TIFF_SBYTE:
                        {
                            int8_t v2 = (int8_t)va_arg(ap, int);
                            _TIFFmemcpy(val, &v2, tv_size);
                        }
                        break;
                        case TIFF_SHORT:
                        {
                            uint16_t v2 = (uint16_t)va_arg(ap, int);
                            _TIFFmemcpy(val, &v2, tv_size);
                        }
                        break;
                        case TIFF_SSHORT:
                        {
                            int16_t v2 = (int16_t)va_arg(ap, int);
                            _TIFFmemcpy(val, &v2, tv_size);
                        }
                        break;
                        case TIFF_LONG:
                        case TIFF_IFD:
                        {
                            uint32_t v2 = va_arg(ap, uint32_t);
                            _TIFFmemcpy(val, &v2, tv_size);
                        }
                        break;
                        case TIFF_SLONG:
                        {
                            int32_t v2 = va_arg(ap, int32_t);
                            _TIFFmemcpy(val, &v2, tv_size);
                        }
                        break;
                        case TIFF_LONG8:
                        case TIFF_IFD8:
                        {
                            uint64_t v2 = va_arg(ap, uint64_t);
                            _TIFFmemcpy(val, &v2, tv_size);
                            /* Test here for too big values for ClassicTIFF and
                             * delete custom field from custom list */
                            if (!(tif->tif_flags & TIFF_BIGTIFF) &&
                                (v2 > 0xffffffffu))
                            {
                                TIFFErrorExtR(
                                    tif, module,
                                    "%s: Bad LONG8 or IFD8 value %" PRIu64
                                    " for \"%s\" tag %d in ClassicTIFF. Tag "
                                    "won't be written to file",
                                    tif->tif_name, v2, fip->field_name, tag);
                                goto badvalueifd8long8;
                            }
                        }
                        break;
                        case TIFF_SLONG8:
                        {
                            int64_t v2 = va_arg(ap, int64_t);
                            _TIFFmemcpy(val, &v2, tv_size);
                            /* Test here for too big values for ClassicTIFF and
                             * delete custom field from custom list */
                            if (!(tif->tif_flags & TIFF_BIGTIFF) &&
                                ((v2 > 2147483647) || (v2 < (-2147483647 - 1))))
                            {
                                TIFFErrorExtR(
                                    tif, module,
                                    "%s: Bad SLONG8 value %" PRIi64
                                    " for \"%s\" tag %d in ClassicTIFF. Tag "
                                    "won't be written to file",
                                    tif->tif_name, v2, fip->field_name, tag);
                                goto badvalueifd8long8;
                            }
                        }
                        break;
                        case TIFF_RATIONAL:
                        case TIFF_SRATIONAL:
                            /*-- Rational2Double: For Rationals tv_size is set
                             * above to 4 or 8 according to fip->set_field_type!
                             */
                            {
                                if (tv_size == 8)
                                {
                                    double v2 = va_arg(ap, double);
                                    _TIFFmemcpy(val, &v2, tv_size);
                                }
                                else
                                {
                                    /*-- default should be tv_size == 4 */
                                    float v3 = (float)va_arg(ap, double);
                                    _TIFFmemcpy(val, &v3, tv_size);
                                    /*-- ToDo: After Testing, this should be
                                     * removed and tv_size==4 should be set as
                                     * default. */
                                    if (tv_size != 4)
                                    {
                                        TIFFErrorExtR(
                                            tif, module,
                                            "Rational2Double: .set_field_type "
                                            "in not 4 but %d",
                                            tv_size);
                                    }
                                }
                            }
                            break;
                        case TIFF_FLOAT:
                        {
                            float v2 =
                                _TIFFClampDoubleToFloat(va_arg(ap, double));
                            _TIFFmemcpy(val, &v2, tv_size);
                        }
                        break;
                        case TIFF_DOUBLE:
                        {
                            double v2 = va_arg(ap, double);
                            _TIFFmemcpy(val, &v2, tv_size);
                        }
                        break;
                        default:
                            _TIFFmemset(val, 0, tv_size);
                            status = 0;
                            break;
                    }
                }
            }
        }
    }
    if (status)
    {
        const TIFFField *fip2 = TIFFFieldWithTag(tif, tag);
        if (fip2)
            TIFFSetFieldBit(tif, fip2->field_bit);
        tif->tif_flags |= TIFF_DIRTYDIRECT;
    }

end:
    va_end(ap);
    return (status);
badvalue:
{
    const TIFFField *fip2 = TIFFFieldWithTag(tif, tag);
    TIFFErrorExtR(tif, module, "%s: Bad value %" PRIu32 " for \"%s\" tag",
                  tif->tif_name, v, fip2 ? fip2->field_name : "Unknown");
    va_end(ap);
}
    return (0);
badvalue32:
{
    const TIFFField *fip2 = TIFFFieldWithTag(tif, tag);
    TIFFErrorExtR(tif, module, "%s: Bad value %" PRIu32 " for \"%s\" tag",
                  tif->tif_name, v32, fip2 ? fip2->field_name : "Unknown");
    va_end(ap);
}
    return (0);
badvaluedouble:
{
    const TIFFField *fip2 = TIFFFieldWithTag(tif, tag);
    TIFFErrorExtR(tif, module, "%s: Bad value %f for \"%s\" tag", tif->tif_name,
                  dblval, fip2 ? fip2->field_name : "Unknown");
    va_end(ap);
}
    return (0);
badvalueifd8long8:
{
    /* Error message issued already above. */
    TIFFTagValue *tv2 = NULL;
    int iCustom2, iC2;
    /* Find the existing entry for this custom value. */
    for (iCustom2 = 0; iCustom2 < td->td_customValueCount; iCustom2++)
    {
        if (td->td_customValues[iCustom2].info->field_tag == tag)
        {
            tv2 = td->td_customValues + (iCustom2);
            break;
        }
    }
    if (tv2 != NULL)
    {
        /* Remove custom field from custom list */
        if (tv2->value != NULL)
        {
            _TIFFfreeExt(tif, tv2->value);
            tv2->value = NULL;
        }
        /* Shorten list and close gap in customValues list.
         * Re-allocation of td_customValues not necessary here. */
        td->td_customValueCount--;
        for (iC2 = iCustom2; iC2 < td->td_customValueCount; iC2++)
        {
            td->td_customValues[iC2] = td->td_customValues[iC2 + 1];
        }
    }
    else
    {
        assert(0);
    }
    va_end(ap);
}
    return (0);
} /*-- _TIFFVSetField() --*/

/*
 * Return 1/0 according to whether or not
 * it is permissible to set the tag's value.
 * Note that we allow ImageLength to be changed
 * so that we can append and extend to images.
 * Any other tag may not be altered once writing
 * has commenced, unless its value has no effect
 * on the format of the data that is written.
 */
static int OkToChangeTag(TIFF *tif, uint32_t tag)
{
    const TIFFField *fip = TIFFFindField(tif, tag, TIFF_ANY);
    if (!fip)
    { /* unknown tag */
        TIFFErrorExtR(tif, "TIFFSetField", "%s: Unknown %stag %" PRIu32,
                      tif->tif_name, isPseudoTag(tag) ? "pseudo-" : "", tag);
        return (0);
    }
    if (tag != TIFFTAG_IMAGELENGTH && (tif->tif_flags & TIFF_BEENWRITING) &&
        !fip->field_oktochange)
    {
        /*
         * Consult info table to see if tag can be changed
         * after we've started writing.  We only allow changes
         * to those tags that don't/shouldn't affect the
         * compression and/or format of the data.
         */
        TIFFErrorExtR(tif, "TIFFSetField",
                      "%s: Cannot modify tag \"%s\" while writing",
                      tif->tif_name, fip->field_name);
        return (0);
    }
    return (1);
}

/*
 * Record the value of a field in the
 * internal directory structure.  The
 * field will be written to the file
 * when/if the directory structure is
 * updated.
 */
int TIFFSetField(TIFF *tif, uint32_t tag, ...)
{
    va_list ap;
    int status;

    va_start(ap, tag);
    status = TIFFVSetField(tif, tag, ap);
    va_end(ap);
    return (status);
}

/*
 * Clear the contents of the field in the internal structure.
 */
int TIFFUnsetField(TIFF *tif, uint32_t tag)
{
    const TIFFField *fip = TIFFFieldWithTag(tif, tag);
    TIFFDirectory *td = &tif->tif_dir;

    if (!fip)
        return 0;

    if (fip->field_bit != FIELD_CUSTOM)
        TIFFClrFieldBit(tif, fip->field_bit);
    else
    {
        TIFFTagValue *tv = NULL;
        int i;

        for (i = 0; i < td->td_customValueCount; i++)
        {

            tv = td->td_customValues + i;
            if (tv->info->field_tag == tag)
                break;
        }

        if (i < td->td_customValueCount)
        {
            _TIFFfreeExt(tif, tv->value);
            for (; i < td->td_customValueCount - 1; i++)
            {
                td->td_customValues[i] = td->td_customValues[i + 1];
            }
            td->td_customValueCount--;
        }
    }

    tif->tif_flags |= TIFF_DIRTYDIRECT;

    return (1);
}

/*
 * Like TIFFSetField, but taking a varargs
 * parameter list.  This routine is useful
 * for building higher-level interfaces on
 * top of the library.
 */
int TIFFVSetField(TIFF *tif, uint32_t tag, va_list ap)
{
    return OkToChangeTag(tif, tag)
               ? (*tif->tif_tagmethods.vsetfield)(tif, tag, ap)
               : 0;
}

static int _TIFFVGetField(TIFF *tif, uint32_t tag, va_list ap)
{
    TIFFDirectory *td = &tif->tif_dir;
    int ret_val = 1;
    uint32_t standard_tag = tag;
    const TIFFField *fip = TIFFFindField(tif, tag, TIFF_ANY);
    if (fip == NULL) /* cannot happen since TIFFGetField() already checks it */
        return 0;

    /*
     * We want to force the custom code to be used for custom
     * fields even if the tag happens to match a well known
     * one - important for reinterpreted handling of standard
     * tag values in custom directories (i.e. EXIF)
     */
    if (fip->field_bit == FIELD_CUSTOM)
    {
        standard_tag = 0;
    }

    switch (standard_tag)
    {
        case TIFFTAG_SUBFILETYPE:
            *va_arg(ap, uint32_t *) = td->td_subfiletype;
            break;
        case TIFFTAG_IMAGEWIDTH:
            *va_arg(ap, uint32_t *) = td->td_imagewidth;
            break;
        case TIFFTAG_IMAGELENGTH:
            *va_arg(ap, uint32_t *) = td->td_imagelength;
            break;
        case TIFFTAG_BITSPERSAMPLE:
            *va_arg(ap, uint16_t *) = td->td_bitspersample;
            break;
        case TIFFTAG_COMPRESSION:
            *va_arg(ap, uint16_t *) = td->td_compression;
            break;
        case TIFFTAG_PHOTOMETRIC:
            *va_arg(ap, uint16_t *) = td->td_photometric;
            break;
        case TIFFTAG_THRESHHOLDING:
            *va_arg(ap, uint16_t *) = td->td_threshholding;
            break;
        case TIFFTAG_FILLORDER:
            *va_arg(ap, uint16_t *) = td->td_fillorder;
            break;
        case TIFFTAG_ORIENTATION:
            *va_arg(ap, uint16_t *) = td->td_orientation;
            break;
        case TIFFTAG_SAMPLESPERPIXEL:
            *va_arg(ap, uint16_t *) = td->td_samplesperpixel;
            break;
        case TIFFTAG_ROWSPERSTRIP:
            *va_arg(ap, uint32_t *) = td->td_rowsperstrip;
            break;
        case TIFFTAG_MINSAMPLEVALUE:
            *va_arg(ap, uint16_t *) = td->td_minsamplevalue;
            break;
        case TIFFTAG_MAXSAMPLEVALUE:
            *va_arg(ap, uint16_t *) = td->td_maxsamplevalue;
            break;
        case TIFFTAG_SMINSAMPLEVALUE:
            if (tif->tif_flags & TIFF_PERSAMPLE)
                *va_arg(ap, double **) = td->td_sminsamplevalue;
            else
            {
                /* libtiff historically treats this as a single value. */
                uint16_t i;
                double v = td->td_sminsamplevalue[0];
                for (i = 1; i < td->td_samplesperpixel; ++i)
                    if (td->td_sminsamplevalue[i] < v)
                        v = td->td_sminsamplevalue[i];
                *va_arg(ap, double *) = v;
            }
            break;
        case TIFFTAG_SMAXSAMPLEVALUE:
            if (tif->tif_flags & TIFF_PERSAMPLE)
                *va_arg(ap, double **) = td->td_smaxsamplevalue;
            else
            {
                /* libtiff historically treats this as a single value. */
                uint16_t i;
                double v = td->td_smaxsamplevalue[0];
                for (i = 1; i < td->td_samplesperpixel; ++i)
                    if (td->td_smaxsamplevalue[i] > v)
                        v = td->td_smaxsamplevalue[i];
                *va_arg(ap, double *) = v;
            }
            break;
        case TIFFTAG_XRESOLUTION:
            *va_arg(ap, float *) = td->td_xresolution;
            break;
        case TIFFTAG_YRESOLUTION:
            *va_arg(ap, float *) = td->td_yresolution;
            break;
        case TIFFTAG_PLANARCONFIG:
            *va_arg(ap, uint16_t *) = td->td_planarconfig;
            break;
        case TIFFTAG_XPOSITION:
            *va_arg(ap, float *) = td->td_xposition;
            break;
        case TIFFTAG_YPOSITION:
            *va_arg(ap, float *) = td->td_yposition;
            break;
        case TIFFTAG_RESOLUTIONUNIT:
            *va_arg(ap, uint16_t *) = td->td_resolutionunit;
            break;
        case TIFFTAG_PAGENUMBER:
            *va_arg(ap, uint16_t *) = td->td_pagenumber[0];
            *va_arg(ap, uint16_t *) = td->td_pagenumber[1];
            break;
        case TIFFTAG_HALFTONEHINTS:
            *va_arg(ap, uint16_t *) = td->td_halftonehints[0];
            *va_arg(ap, uint16_t *) = td->td_halftonehints[1];
            break;
        case TIFFTAG_COLORMAP:
            *va_arg(ap, const uint16_t **) = td->td_colormap[0];
            *va_arg(ap, const uint16_t **) = td->td_colormap[1];
            *va_arg(ap, const uint16_t **) = td->td_colormap[2];
            break;
        case TIFFTAG_STRIPOFFSETS:
        case TIFFTAG_TILEOFFSETS:
            _TIFFFillStriles(tif);
            *va_arg(ap, const uint64_t **) = td->td_stripoffset_p;
            if (td->td_stripoffset_p == NULL)
                ret_val = 0;
            break;
        case TIFFTAG_STRIPBYTECOUNTS:
        case TIFFTAG_TILEBYTECOUNTS:
            _TIFFFillStriles(tif);
            *va_arg(ap, const uint64_t **) = td->td_stripbytecount_p;
            if (td->td_stripbytecount_p == NULL)
                ret_val = 0;
            break;
        case TIFFTAG_MATTEING:
            *va_arg(ap, uint16_t *) =
                (td->td_extrasamples == 1 &&
                 td->td_sampleinfo[0] == EXTRASAMPLE_ASSOCALPHA);
            break;
        case TIFFTAG_EXTRASAMPLES:
            *va_arg(ap, uint16_t *) = td->td_extrasamples;
            *va_arg(ap, const uint16_t **) = td->td_sampleinfo;
            break;
        case TIFFTAG_TILEWIDTH:
            *va_arg(ap, uint32_t *) = td->td_tilewidth;
            break;
        case TIFFTAG_TILELENGTH:
            *va_arg(ap, uint32_t *) = td->td_tilelength;
            break;
        case TIFFTAG_TILEDEPTH:
            *va_arg(ap, uint32_t *) = td->td_tiledepth;
            break;
        case TIFFTAG_DATATYPE:
            switch (td->td_sampleformat)
            {
                case SAMPLEFORMAT_UINT:
                    *va_arg(ap, uint16_t *) = DATATYPE_UINT;
                    break;
                case SAMPLEFORMAT_INT:
                    *va_arg(ap, uint16_t *) = DATATYPE_INT;
                    break;
                case SAMPLEFORMAT_IEEEFP:
                    *va_arg(ap, uint16_t *) = DATATYPE_IEEEFP;
                    break;
                case SAMPLEFORMAT_VOID:
                    *va_arg(ap, uint16_t *) = DATATYPE_VOID;
                    break;
            }
            break;
        case TIFFTAG_SAMPLEFORMAT:
            *va_arg(ap, uint16_t *) = td->td_sampleformat;
            break;
        case TIFFTAG_IMAGEDEPTH:
            *va_arg(ap, uint32_t *) = td->td_imagedepth;
            break;
        case TIFFTAG_SUBIFD:
            *va_arg(ap, uint16_t *) = td->td_nsubifd;
            *va_arg(ap, const uint64_t **) = td->td_subifd;
            break;
        case TIFFTAG_YCBCRPOSITIONING:
            *va_arg(ap, uint16_t *) = td->td_ycbcrpositioning;
            break;
        case TIFFTAG_YCBCRSUBSAMPLING:
            *va_arg(ap, uint16_t *) = td->td_ycbcrsubsampling[0];
            *va_arg(ap, uint16_t *) = td->td_ycbcrsubsampling[1];
            break;
        case TIFFTAG_TRANSFERFUNCTION:
            *va_arg(ap, const uint16_t **) = td->td_transferfunction[0];
            if (td->td_samplesperpixel - td->td_extrasamples > 1)
            {
                *va_arg(ap, const uint16_t **) = td->td_transferfunction[1];
                *va_arg(ap, const uint16_t **) = td->td_transferfunction[2];
            }
            else
            {
                *va_arg(ap, const uint16_t **) = NULL;
                *va_arg(ap, const uint16_t **) = NULL;
            }
            break;
        case TIFFTAG_REFERENCEBLACKWHITE:
            *va_arg(ap, const float **) = td->td_refblackwhite;
            break;
        case TIFFTAG_INKNAMES:
            *va_arg(ap, const char **) = td->td_inknames;
            break;
        case TIFFTAG_NUMBEROFINKS:
            *va_arg(ap, uint16_t *) = td->td_numberofinks;
            break;
        default:
        {
            int i;

            /*
             * This can happen if multiple images are open
             * with different codecs which have private
             * tags.  The global tag information table may
             * then have tags that are valid for one file
             * but not the other. If the client tries to
             * get a tag that is not valid for the image's
             * codec then we'll arrive here.
             */
            if (fip->field_bit != FIELD_CUSTOM)
            {
                TIFFErrorExtR(tif, "_TIFFVGetField",
                              "%s: Invalid %stag \"%s\" "
                              "(not supported by codec)",
                              tif->tif_name, isPseudoTag(tag) ? "pseudo-" : "",
                              fip->field_name);
                ret_val = 0;
                break;
            }

            /*
             * Do we have a custom value?
             */
            ret_val = 0;
            for (i = 0; i < td->td_customValueCount; i++)
            {
                TIFFTagValue *tv = td->td_customValues + i;

                if (tv->info->field_tag != tag)
                    continue;

                if (fip->field_passcount)
                {
                    if (fip->field_readcount == TIFF_VARIABLE2)
                        *va_arg(ap, uint32_t *) = (uint32_t)tv->count;
                    else /* Assume TIFF_VARIABLE */
                        *va_arg(ap, uint16_t *) = (uint16_t)tv->count;
                    *va_arg(ap, const void **) = tv->value;
                    ret_val = 1;
                }
                else if (fip->field_tag == TIFFTAG_DOTRANGE &&
                         strcmp(fip->field_name, "DotRange") == 0)
                {
                    /* TODO: This is an evil exception and should not have been
                       handled this way ... likely best if we move it into
                       the directory structure with an explicit field in
                       libtiff 4.1 and assign it a FIELD_ value */
                    *va_arg(ap, uint16_t *) = ((uint16_t *)tv->value)[0];
                    *va_arg(ap, uint16_t *) = ((uint16_t *)tv->value)[1];
                    ret_val = 1;
                }
                else
                {
                    if (fip->field_type == TIFF_ASCII ||
                        fip->field_readcount == TIFF_VARIABLE ||
                        fip->field_readcount == TIFF_VARIABLE2 ||
                        fip->field_readcount == TIFF_SPP || tv->count > 1)
                    {
                        *va_arg(ap, void **) = tv->value;
                        ret_val = 1;
                    }
                    else
                    {
                        char *val = (char *)tv->value;
                        assert(tv->count == 1);
                        switch (fip->field_type)
                        {
                            case TIFF_BYTE:
                            case TIFF_UNDEFINED:
                                *va_arg(ap, uint8_t *) = *(uint8_t *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SBYTE:
                                *va_arg(ap, int8_t *) = *(int8_t *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SHORT:
                                *va_arg(ap, uint16_t *) = *(uint16_t *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SSHORT:
                                *va_arg(ap, int16_t *) = *(int16_t *)val;
                                ret_val = 1;
                                break;
                            case TIFF_LONG:
                            case TIFF_IFD:
                                *va_arg(ap, uint32_t *) = *(uint32_t *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SLONG:
                                *va_arg(ap, int32_t *) = *(int32_t *)val;
                                ret_val = 1;
                                break;
                            case TIFF_LONG8:
                            case TIFF_IFD8:
                                *va_arg(ap, uint64_t *) = *(uint64_t *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SLONG8:
                                *va_arg(ap, int64_t *) = *(int64_t *)val;
                                ret_val = 1;
                                break;
                            case TIFF_RATIONAL:
                            case TIFF_SRATIONAL:
                            {
                                /*-- Rational2Double: For Rationals evaluate
                                 * "set_field_type" to determine internal
                                 * storage size and return value size. */
                                int tv_size = TIFFFieldSetGetSize(fip);
                                if (tv_size == 8)
                                {
                                    *va_arg(ap, double *) = *(double *)val;
                                    ret_val = 1;
                                }
                                else
                                {
                                    /*-- default should be tv_size == 4  */
                                    *va_arg(ap, float *) = *(float *)val;
                                    ret_val = 1;
                                    /*-- ToDo: After Testing, this should be
                                     * removed and tv_size==4 should be set as
                                     * default. */
                                    if (tv_size != 4)
                                    {
                                        TIFFErrorExtR(
                                            tif, "_TIFFVGetField",
                                            "Rational2Double: .set_field_type "
                                            "in not 4 but %d",
                                            tv_size);
                                    }
                                }
                            }
                            break;
                            case TIFF_FLOAT:
                                *va_arg(ap, float *) = *(float *)val;
                                ret_val = 1;
                                break;
                            case TIFF_DOUBLE:
                                *va_arg(ap, double *) = *(double *)val;
                                ret_val = 1;
                                break;
                            default:
                                ret_val = 0;
                                break;
                        }
                    }
                }
                break;
            }
        }
    }
    return (ret_val);
}

/*
 * Return the value of a field in the
 * internal directory structure.
 */
int TIFFGetField(TIFF *tif, uint32_t tag, ...)
{
    int status;
    va_list ap;

    va_start(ap, tag);
    status = TIFFVGetField(tif, tag, ap);
    va_end(ap);
    return (status);
}

/*
 * Like TIFFGetField, but taking a varargs
 * parameter list.  This routine is useful
 * for building higher-level interfaces on
 * top of the library.
 */
int TIFFVGetField(TIFF *tif, uint32_t tag, va_list ap)
{
    const TIFFField *fip = TIFFFindField(tif, tag, TIFF_ANY);
    return (fip && (isPseudoTag(tag) || TIFFFieldSet(tif, fip->field_bit))
                ? (*tif->tif_tagmethods.vgetfield)(tif, tag, ap)
                : 0);
}

#define CleanupField(member)                                                   \
    {                                                                          \
        if (td->member)                                                        \
        {                                                                      \
            _TIFFfreeExt(tif, td->member);                                     \
            td->member = 0;                                                    \
        }                                                                      \
    }

/*
 * Release storage associated with a directory.
 */
void TIFFFreeDirectory(TIFF *tif)
{
    TIFFDirectory *td = &tif->tif_dir;
    int i;

    _TIFFmemset(td->td_fieldsset, 0, sizeof(td->td_fieldsset));
    CleanupField(td_sminsamplevalue);
    CleanupField(td_smaxsamplevalue);
    CleanupField(td_colormap[0]);
    CleanupField(td_colormap[1]);
    CleanupField(td_colormap[2]);
    CleanupField(td_sampleinfo);
    CleanupField(td_subifd);
    CleanupField(td_inknames);
    CleanupField(td_refblackwhite);
    CleanupField(td_transferfunction[0]);
    CleanupField(td_transferfunction[1]);
    CleanupField(td_transferfunction[2]);
    CleanupField(td_stripoffset_p);
    CleanupField(td_stripbytecount_p);
    td->td_stripoffsetbyteallocsize = 0;
    TIFFClrFieldBit(tif, FIELD_YCBCRSUBSAMPLING);
    TIFFClrFieldBit(tif, FIELD_YCBCRPOSITIONING);

    /* Cleanup custom tag values */
    for (i = 0; i < td->td_customValueCount; i++)
    {
        if (td->td_customValues[i].value)
            _TIFFfreeExt(tif, td->td_customValues[i].value);
    }

    td->td_customValueCount = 0;
    CleanupField(td_customValues);

    _TIFFmemset(&(td->td_stripoffset_entry), 0, sizeof(TIFFDirEntry));
    _TIFFmemset(&(td->td_stripbytecount_entry), 0, sizeof(TIFFDirEntry));

    /* Reset some internal parameters for IFD data size checking. */
    tif->tif_dir.td_dirdatasize_read = 0;
    tif->tif_dir.td_dirdatasize_write = 0;
    if (tif->tif_dir.td_dirdatasize_offsets != NULL)
    {
        _TIFFfreeExt(tif, tif->tif_dir.td_dirdatasize_offsets);
        tif->tif_dir.td_dirdatasize_offsets = NULL;
        tif->tif_dir.td_dirdatasize_Noffsets = 0;
    }
    tif->tif_dir.td_iswrittentofile = FALSE;
}
#undef CleanupField

/*
 * Client Tag extension support (from Niles Ritter).
 */
static TIFFExtendProc _TIFFextender = (TIFFExtendProc)NULL;

TIFFExtendProc TIFFSetTagExtender(TIFFExtendProc extender)
{
    TIFFExtendProc prev = _TIFFextender;
    _TIFFextender = extender;
    return (prev);
}

/*
 * Setup for a new directory.  Should we automatically call
 * TIFFWriteDirectory() if the current one is dirty?
 *
 * The newly created directory will not exist on the file till
 * TIFFWriteDirectory(), TIFFFlush() or TIFFClose() is called.
 */
int TIFFCreateDirectory(TIFF *tif)
{
    /* Free previously allocated memory and setup default values. */
    TIFFFreeDirectory(tif);
    TIFFDefaultDirectory(tif);
    tif->tif_diroff = 0;
    tif->tif_nextdiroff = 0;
    tif->tif_curoff = 0;
    tif->tif_row = (uint32_t)-1;
    tif->tif_curstrip = (uint32_t)-1;
    tif->tif_dir.td_iswrittentofile = FALSE;

    return 0;
}

int TIFFCreateCustomDirectory(TIFF *tif, const TIFFFieldArray *infoarray)
{
    /* Free previously allocated memory and setup default values. */
    TIFFFreeDirectory(tif);
    TIFFDefaultDirectory(tif);

    /*
     * Reset the field definitions to match the application provided list.
     * Hopefully TIFFDefaultDirectory() won't have done anything irreversible
     * based on it's assumption this is an image directory.
     */
    _TIFFSetupFields(tif, infoarray);

    tif->tif_diroff = 0;
    tif->tif_nextdiroff = 0;
    tif->tif_curoff = 0;
    tif->tif_row = (uint32_t)-1;
    tif->tif_curstrip = (uint32_t)-1;
    /* invalidate directory index */
    tif->tif_curdir = TIFF_NON_EXISTENT_DIR_NUMBER;
    /* invalidate IFD loop lists */
    _TIFFCleanupIFDOffsetAndNumberMaps(tif);
    /* To be able to return from SubIFD or custom-IFD to main-IFD */
    tif->tif_setdirectory_force_absolute = TRUE;

    return 0;
}

int TIFFCreateEXIFDirectory(TIFF *tif)
{
    const TIFFFieldArray *exifFieldArray;
    exifFieldArray = _TIFFGetExifFields();
    return TIFFCreateCustomDirectory(tif, exifFieldArray);
}

/*
 * Creates the EXIF GPS custom directory
 */
int TIFFCreateGPSDirectory(TIFF *tif)
{
    const TIFFFieldArray *gpsFieldArray;
    gpsFieldArray = _TIFFGetGpsFields();
    return TIFFCreateCustomDirectory(tif, gpsFieldArray);
}

/*
 * Setup a default directory structure.
 */
int TIFFDefaultDirectory(TIFF *tif)
{
    register TIFFDirectory *td = &tif->tif_dir;
    const TIFFFieldArray *tiffFieldArray;

    tiffFieldArray = _TIFFGetFields();
    _TIFFSetupFields(tif, tiffFieldArray);

    _TIFFmemset(td, 0, sizeof(*td));
    td->td_fillorder = FILLORDER_MSB2LSB;
    td->td_bitspersample = 1;
    td->td_threshholding = THRESHHOLD_BILEVEL;
    td->td_orientation = ORIENTATION_TOPLEFT;
    td->td_samplesperpixel = 1;
    td->td_rowsperstrip = (uint32_t)-1;
    td->td_tilewidth = 0;
    td->td_tilelength = 0;
    td->td_tiledepth = 1;
#ifdef STRIPBYTECOUNTSORTED_UNUSED
    td->td_stripbytecountsorted = 1; /* Our own arrays always sorted. */
#endif
    td->td_resolutionunit = RESUNIT_INCH;
    td->td_sampleformat = SAMPLEFORMAT_UINT;
    td->td_imagedepth = 1;
    td->td_ycbcrsubsampling[0] = 2;
    td->td_ycbcrsubsampling[1] = 2;
    td->td_ycbcrpositioning = YCBCRPOSITION_CENTERED;
    tif->tif_postdecode = _TIFFNoPostDecode;
    tif->tif_foundfield = NULL;
    tif->tif_tagmethods.vsetfield = _TIFFVSetField;
    tif->tif_tagmethods.vgetfield = _TIFFVGetField;
    tif->tif_tagmethods.printdir = NULL;
    /* additional default values */
    td->td_planarconfig = PLANARCONFIG_CONTIG;
    td->td_compression = COMPRESSION_NONE;
    td->td_subfiletype = 0;
    td->td_minsamplevalue = 0;
    /* td_bitspersample=1 is always set in TIFFDefaultDirectory().
     * Therefore, td_maxsamplevalue has to be re-calculated in
     * TIFFGetFieldDefaulted(). */
    td->td_maxsamplevalue = 1; /* Default for td_bitspersample=1 */
    td->td_extrasamples = 0;
    td->td_sampleinfo = NULL;

    /*
     *  Give client code a chance to install their own
     *  tag extensions & methods, prior to compression overloads,
     *  but do some prior cleanup first.
     * (http://trac.osgeo.org/gdal/ticket/5054)
     */
    if (tif->tif_nfieldscompat > 0)
    {
        uint32_t i;

        for (i = 0; i < tif->tif_nfieldscompat; i++)
        {
            if (tif->tif_fieldscompat[i].allocated_size)
                _TIFFfreeExt(tif, tif->tif_fieldscompat[i].fields);
        }
        _TIFFfreeExt(tif, tif->tif_fieldscompat);
        tif->tif_nfieldscompat = 0;
        tif->tif_fieldscompat = NULL;
    }
    if (_TIFFextender)
        (*_TIFFextender)(tif);
    (void)TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    /*
     * NB: The directory is marked dirty as a result of setting
     * up the default compression scheme.  However, this really
     * isn't correct -- we want TIFF_DIRTYDIRECT to be set only
     * if the user does something.  We could just do the setup
     * by hand, but it seems better to use the normal mechanism
     * (i.e. TIFFSetField).
     */
    tif->tif_flags &= ~TIFF_DIRTYDIRECT;

    /*
     * As per http://bugzilla.remotesensing.org/show_bug.cgi?id=19
     * we clear the ISTILED flag when setting up a new directory.
     * Should we also be clearing stuff like INSUBIFD?
     */
    tif->tif_flags &= ~TIFF_ISTILED;

    return (1);
}

static int TIFFAdvanceDirectory(TIFF *tif, uint64_t *nextdiroff, uint64_t *off,
                                tdir_t *nextdirnum)
{
    static const char module[] = "TIFFAdvanceDirectory";

    /* Add this directory to the directory list, if not already in. */
    if (!_TIFFCheckDirNumberAndOffset(tif, *nextdirnum, *nextdiroff))
    {
        TIFFErrorExtR(tif, module,
                      "Starting directory %u at offset 0x%" PRIx64 " (%" PRIu64
                      ") might cause an IFD loop",
                      *nextdirnum, *nextdiroff, *nextdiroff);
        *nextdiroff = 0;
        *nextdirnum = 0;
        return (0);
    }

    if (isMapped(tif))
    {
        uint64_t poff = *nextdiroff;
        if (!(tif->tif_flags & TIFF_BIGTIFF))
        {
            tmsize_t poffa, poffb, poffc, poffd;
            uint16_t dircount;
            uint32_t nextdir32;
            poffa = (tmsize_t)poff;
            poffb = poffa + sizeof(uint16_t);
            if (((uint64_t)poffa != poff) || (poffb < poffa) ||
                (poffb < (tmsize_t)sizeof(uint16_t)) || (poffb > tif->tif_size))
            {
                TIFFErrorExtR(tif, module,
                              "%s:%d: %s: Error fetching directory count",
                              __FILE__, __LINE__, tif->tif_name);
                *nextdiroff = 0;
                return (0);
            }
            _TIFFmemcpy(&dircount, tif->tif_base + poffa, sizeof(uint16_t));
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort(&dircount);
            poffc = poffb + dircount * 12;
            poffd = poffc + sizeof(uint32_t);
            if ((poffc < poffb) || (poffc < dircount * 12) || (poffd < poffc) ||
                (poffd < (tmsize_t)sizeof(uint32_t)) || (poffd > tif->tif_size))
            {
                TIFFErrorExtR(tif, module, "Error fetching directory link");
                return (0);
            }
            if (off != NULL)
                *off = (uint64_t)poffc;
            _TIFFmemcpy(&nextdir32, tif->tif_base + poffc, sizeof(uint32_t));
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong(&nextdir32);
            *nextdiroff = nextdir32;
        }
        else
        {
            tmsize_t poffa, poffb, poffc, poffd;
            uint64_t dircount64;
            uint16_t dircount16;
            if (poff > (uint64_t)TIFF_TMSIZE_T_MAX - sizeof(uint64_t))
            {
                TIFFErrorExtR(tif, module,
                              "%s:%d: %s: Error fetching directory count",
                              __FILE__, __LINE__, tif->tif_name);
                return (0);
            }
            poffa = (tmsize_t)poff;
            poffb = poffa + sizeof(uint64_t);
            if (poffb > tif->tif_size)
            {
                TIFFErrorExtR(tif, module,
                              "%s:%d: %s: Error fetching directory count",
                              __FILE__, __LINE__, tif->tif_name);
                return (0);
            }
            _TIFFmemcpy(&dircount64, tif->tif_base + poffa, sizeof(uint64_t));
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&dircount64);
            if (dircount64 > 0xFFFF)
            {
                TIFFErrorExtR(tif, module,
                              "Sanity check on directory count failed");
                return (0);
            }
            dircount16 = (uint16_t)dircount64;
            if (poffb > TIFF_TMSIZE_T_MAX - (tmsize_t)(dircount16 * 20) -
                            (tmsize_t)sizeof(uint64_t))
            {
                TIFFErrorExtR(tif, module, "Error fetching directory link");
                return (0);
            }
            poffc = poffb + dircount16 * 20;
            poffd = poffc + sizeof(uint64_t);
            if (poffd > tif->tif_size)
            {
                TIFFErrorExtR(tif, module, "Error fetching directory link");
                return (0);
            }
            if (off != NULL)
                *off = (uint64_t)poffc;
            _TIFFmemcpy(nextdiroff, tif->tif_base + poffc, sizeof(uint64_t));
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(nextdiroff);
        }
    }
    else
    {
        if (!(tif->tif_flags & TIFF_BIGTIFF))
        {
            uint16_t dircount;
            uint32_t nextdir32;
            if (!SeekOK(tif, *nextdiroff) ||
                !ReadOK(tif, &dircount, sizeof(uint16_t)))
            {
                TIFFErrorExtR(tif, module,
                              "%s:%d: %s: Error fetching directory count",
                              __FILE__, __LINE__, tif->tif_name);
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort(&dircount);
            if (off != NULL)
                *off = TIFFSeekFile(tif, dircount * 12, SEEK_CUR);
            else
                (void)TIFFSeekFile(tif, dircount * 12, SEEK_CUR);
            if (!ReadOK(tif, &nextdir32, sizeof(uint32_t)))
            {
                TIFFErrorExtR(tif, module, "%s: Error fetching directory link",
                              tif->tif_name);
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong(&nextdir32);
            *nextdiroff = nextdir32;
        }
        else
        {
            uint64_t dircount64;
            uint16_t dircount16;
            if (!SeekOK(tif, *nextdiroff) ||
                !ReadOK(tif, &dircount64, sizeof(uint64_t)))
            {
                TIFFErrorExtR(tif, module,
                              "%s:%d: %s: Error fetching directory count",
                              __FILE__, __LINE__, tif->tif_name);
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&dircount64);
            if (dircount64 > 0xFFFF)
            {
                TIFFErrorExtR(tif, module,
                              "%s:%d: %s: Error fetching directory count",
                              __FILE__, __LINE__, tif->tif_name);
                return (0);
            }
            dircount16 = (uint16_t)dircount64;
            if (off != NULL)
                *off = TIFFSeekFile(tif, dircount16 * 20, SEEK_CUR);
            else
                (void)TIFFSeekFile(tif, dircount16 * 20, SEEK_CUR);
            if (!ReadOK(tif, nextdiroff, sizeof(uint64_t)))
            {
                TIFFErrorExtR(tif, module, "%s: Error fetching directory link",
                              tif->tif_name);
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(nextdiroff);
        }
    }
    if (*nextdiroff != 0)
    {
        (*nextdirnum)++;
        /* Check next directory for IFD looping and if so, set it as last
         * directory. */
        if (!_TIFFCheckDirNumberAndOffset(tif, *nextdirnum, *nextdiroff))
        {
            TIFFWarningExtR(
                tif, module,
                "the next directory %u at offset 0x%" PRIx64 " (%" PRIu64
                ") might be an IFD loop. Treating directory %d as "
                "last directory",
                *nextdirnum, *nextdiroff, *nextdiroff, (int)(*nextdirnum) - 1);
            *nextdiroff = 0;
            (*nextdirnum)--;
        }
    }
    return (1);
}

/*
 * Count the number of directories in a file.
 */
tdir_t TIFFNumberOfDirectories(TIFF *tif)
{
    uint64_t nextdiroff;
    tdir_t nextdirnum;
    tdir_t n;
    if (!(tif->tif_flags & TIFF_BIGTIFF))
        nextdiroff = tif->tif_header.classic.tiff_diroff;
    else
        nextdiroff = tif->tif_header.big.tiff_diroff;
    nextdirnum = 0;
    n = 0;
    while (nextdiroff != 0 &&
           TIFFAdvanceDirectory(tif, &nextdiroff, NULL, &nextdirnum))
    {
        ++n;
    }
    /* Update number of main-IFDs in file. */
    tif->tif_curdircount = n;
    return (n);
}

/*
 * Set the n-th directory as the current directory.
 * NB: Directories are numbered starting at 0.
 */
int TIFFSetDirectory(TIFF *tif, tdir_t dirn)
{
    uint64_t nextdiroff;
    tdir_t nextdirnum = 0;
    tdir_t n;

    if (tif->tif_setdirectory_force_absolute)
    {
        /* tif_setdirectory_force_absolute=1 will force parsing the main IFD
         * chain from the beginning, thus IFD directory list needs to be cleared
         * from possible SubIFD offsets.
         */
        _TIFFCleanupIFDOffsetAndNumberMaps(tif); /* invalidate IFD loop lists */
    }

    /* Even faster path, if offset is available within IFD loop hash list. */
    if (!tif->tif_setdirectory_force_absolute &&
        _TIFFGetOffsetFromDirNumber(tif, dirn, &nextdiroff))
    {
        /* Set parameters for following TIFFReadDirectory() below. */
        tif->tif_nextdiroff = nextdiroff;
        tif->tif_curdir = dirn;
        /* Reset to relative stepping */
        tif->tif_setdirectory_force_absolute = FALSE;
    }
    else
    {

        /* Fast path when we just advance relative to the current directory:
         * start at the current dir offset and continue to seek from there.
         * Check special cases when relative is not allowed:
         * - jump back from SubIFD or custom directory
         * - right after TIFFWriteDirectory() jump back to that directory
         *   using TIFFSetDirectory() */
        const int relative = (dirn >= tif->tif_curdir) &&
                             (tif->tif_diroff != 0) &&
                             !tif->tif_setdirectory_force_absolute;

        if (relative)
        {
            nextdiroff = tif->tif_diroff;
            dirn -= tif->tif_curdir;
            nextdirnum = tif->tif_curdir;
        }
        else if (!(tif->tif_flags & TIFF_BIGTIFF))
            nextdiroff = tif->tif_header.classic.tiff_diroff;
        else
            nextdiroff = tif->tif_header.big.tiff_diroff;

        /* Reset to relative stepping */
        tif->tif_setdirectory_force_absolute = FALSE;

        for (n = dirn; n > 0 && nextdiroff != 0; n--)
            if (!TIFFAdvanceDirectory(tif, &nextdiroff, NULL, &nextdirnum))
                return (0);
        /* If the n-th directory could not be reached (does not exist),
         * return here without touching anything further. */
        if (nextdiroff == 0 || n > 0)
            return (0);

        tif->tif_nextdiroff = nextdiroff;

        /* Set curdir to the actual directory index. */
        if (relative)
            tif->tif_curdir += dirn - n;
        else
            tif->tif_curdir = dirn - n;
    }

    /* The -1 decrement is because TIFFReadDirectory will increment
     * tif_curdir after successfully reading the directory. */
    if (tif->tif_curdir == 0)
        tif->tif_curdir = TIFF_NON_EXISTENT_DIR_NUMBER;
    else
        tif->tif_curdir--;

    tdir_t curdir = tif->tif_curdir;

    int retval = TIFFReadDirectory(tif);

    if (!retval && tif->tif_curdir == curdir)
    {
        /* If tif_curdir has not be incremented, TIFFFetchDirectory() in
         * TIFFReadDirectory() has failed and tif_curdir shall be set
         * specifically. */
        tif->tif_curdir = TIFF_NON_EXISTENT_DIR_NUMBER;
    }
    return (retval);
}

/*
 * Set the current directory to be the directory
 * located at the specified file offset.  This interface
 * is used mainly to access directories linked with
 * the SubIFD tag (e.g. thumbnail images).
 */
int TIFFSetSubDirectory(TIFF *tif, uint64_t diroff)
{
    /* Match nextdiroff and curdir for consistent IFD-loop checking.
     * Only with TIFFSetSubDirectory() the IFD list can be corrupted with
     * invalid offsets within the main IFD tree. In the case of several subIFDs
     * of a main image, there are two possibilities that are not even mutually
     * exclusive. a.) The subIFD tag contains an array with all offsets of the
     * subIFDs. b.) The SubIFDs are concatenated with their NextIFD parameters.
     * (refer to
     * https://www.awaresystems.be/imaging/tiff/specification/TIFFPM6.pdf.)
     */
    int retval;
    uint32_t curdir = 0;
    int8_t probablySubIFD = 0;
    if (diroff == 0)
    {
        /* Special case to set tif_diroff=0, which is done in
         * TIFFReadDirectory() below to indicate that the currently read IFD is
         * treated as a new, fresh IFD. */
        tif->tif_curdir = TIFF_NON_EXISTENT_DIR_NUMBER;
        tif->tif_dir.td_iswrittentofile = FALSE;
    }
    else
    {
        if (!_TIFFGetDirNumberFromOffset(tif, diroff, &curdir))
        {
            /* Non-existing offsets might point to a SubIFD or invalid IFD.*/
            probablySubIFD = 1;
        }
        /* -1 because TIFFReadDirectory() will increment tif_curdir. */
        if (curdir >= 1)
            tif->tif_curdir = curdir - 1;
        else
            tif->tif_curdir = TIFF_NON_EXISTENT_DIR_NUMBER;
    }
    curdir = tif->tif_curdir;

    tif->tif_nextdiroff = diroff;
    retval = TIFFReadDirectory(tif);

    /* tif_curdir is incremented in TIFFReadDirectory(), but if it has not been
     * incremented, TIFFFetchDirectory() has failed there and tif_curdir shall
     * be set specifically. */
    if (!retval && diroff != 0 && tif->tif_curdir == curdir)
    {
        tif->tif_curdir = TIFF_NON_EXISTENT_DIR_NUMBER;
    }

    if (probablySubIFD)
    {
        if (retval)
        {
            /* Reset IFD list to start new one for SubIFD chain and also start
             * SubIFD chain with tif_curdir=0 for IFD loop checking. */
            /* invalidate IFD loop lists */
            _TIFFCleanupIFDOffsetAndNumberMaps(tif);
            tif->tif_curdir = 0; /* first directory of new chain */
            /* add this offset to new IFD list */
            _TIFFCheckDirNumberAndOffset(tif, tif->tif_curdir, diroff);
        }
        /* To be able to return from SubIFD or custom-IFD to main-IFD */
        tif->tif_setdirectory_force_absolute = TRUE;
    }

    return (retval);
}

/*
 * Return file offset of the current directory.
 */
uint64_t TIFFCurrentDirOffset(TIFF *tif) { return (tif->tif_diroff); }

/*
 * Return an indication of whether or not we are
 * at the last directory in the file.
 */
int TIFFLastDirectory(TIFF *tif) { return (tif->tif_nextdiroff == 0); }

/*
 * Unlink the specified directory from the directory chain.
 * Note: First directory starts with number dirn=1.
 * This is different to TIFFSetDirectory() where the first directory starts with
 * zero.
 */
int TIFFUnlinkDirectory(TIFF *tif, tdir_t dirn)
{
    static const char module[] = "TIFFUnlinkDirectory";
    uint64_t nextdir;
    tdir_t nextdirnum;
    uint64_t off;
    tdir_t n;

    if (tif->tif_mode == O_RDONLY)
    {
        TIFFErrorExtR(tif, module,
                      "Can not unlink directory in read-only file");
        return (0);
    }
    if (dirn == 0)
    {
        TIFFErrorExtR(tif, module,
                      "For TIFFUnlinkDirectory() first directory starts with "
                      "number 1 and not 0");
        return (0);
    }
    /*
     * Go to the directory before the one we want
     * to unlink and nab the offset of the link
     * field we'll need to patch.
     */
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        nextdir = tif->tif_header.classic.tiff_diroff;
        off = 4;
    }
    else
    {
        nextdir = tif->tif_header.big.tiff_diroff;
        off = 8;
    }
    nextdirnum = 0; /* First directory is dirn=0 */

    for (n = dirn - 1; n > 0; n--)
    {
        if (nextdir == 0)
        {
            TIFFErrorExtR(tif, module, "Directory %u does not exist", dirn);
            return (0);
        }
        if (!TIFFAdvanceDirectory(tif, &nextdir, &off, &nextdirnum))
            return (0);
    }
    /*
     * Advance to the directory to be unlinked and fetch
     * the offset of the directory that follows.
     */
    if (!TIFFAdvanceDirectory(tif, &nextdir, NULL, &nextdirnum))
        return (0);
    /*
     * Go back and patch the link field of the preceding
     * directory to point to the offset of the directory
     * that follows.
     */
    (void)TIFFSeekFile(tif, off, SEEK_SET);
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        uint32_t nextdir32;
        nextdir32 = (uint32_t)nextdir;
        assert((uint64_t)nextdir32 == nextdir);
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&nextdir32);
        if (!WriteOK(tif, &nextdir32, sizeof(uint32_t)))
        {
            TIFFErrorExtR(tif, module, "Error writing directory link");
            return (0);
        }
    }
    else
    {
        /* Need local swap because nextdir has to be used unswapped below. */
        uint64_t nextdir64 = nextdir;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8(&nextdir64);
        if (!WriteOK(tif, &nextdir64, sizeof(uint64_t)))
        {
            TIFFErrorExtR(tif, module, "Error writing directory link");
            return (0);
        }
    }

    /* For dirn=1 (first directory) also update the libtiff internal
     * base offset variables. */
    if (dirn == 1)
    {
        if (!(tif->tif_flags & TIFF_BIGTIFF))
            tif->tif_header.classic.tiff_diroff = (uint32_t)nextdir;
        else
            tif->tif_header.big.tiff_diroff = nextdir;
    }

    /*
     * Leave directory state setup safely.  We don't have
     * facilities for doing inserting and removing directories,
     * so it's safest to just invalidate everything.  This
     * means that the caller can only append to the directory
     * chain.
     */
    (*tif->tif_cleanup)(tif);
    if ((tif->tif_flags & TIFF_MYBUFFER) && tif->tif_rawdata)
    {
        _TIFFfreeExt(tif, tif->tif_rawdata);
        tif->tif_rawdata = NULL;
        tif->tif_rawcc = 0;
        tif->tif_rawdataoff = 0;
        tif->tif_rawdataloaded = 0;
    }
    tif->tif_flags &= ~(TIFF_BEENWRITING | TIFF_BUFFERSETUP | TIFF_POSTENCODE |
                        TIFF_BUF4WRITE);
    TIFFFreeDirectory(tif);
    TIFFDefaultDirectory(tif);
    tif->tif_diroff = 0;     /* force link on next write */
    tif->tif_nextdiroff = 0; /* next write must be at end */
    tif->tif_lastdiroff = 0; /* will be updated on next link */
    tif->tif_curoff = 0;
    tif->tif_row = (uint32_t)-1;
    tif->tif_curstrip = (uint32_t)-1;
    tif->tif_curdir = TIFF_NON_EXISTENT_DIR_NUMBER;
    if (tif->tif_curdircount > 0)
        tif->tif_curdircount--;
    else
        tif->tif_curdircount = TIFF_NON_EXISTENT_DIR_NUMBER;
    _TIFFCleanupIFDOffsetAndNumberMaps(tif); /* invalidate IFD loop lists */
    return (1);
}
