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
 * TIFF Library
 *
 * Read and return a packed RGBA image.
 */
#include "tiffiop.h"
#include <limits.h>
#include <stdio.h>

static int gtTileContig(TIFFRGBAImage *, uint32_t *, uint32_t, uint32_t);
static int gtTileSeparate(TIFFRGBAImage *, uint32_t *, uint32_t, uint32_t);
static int gtStripContig(TIFFRGBAImage *, uint32_t *, uint32_t, uint32_t);
static int gtStripSeparate(TIFFRGBAImage *, uint32_t *, uint32_t, uint32_t);
static int PickContigCase(TIFFRGBAImage *);
static int PickSeparateCase(TIFFRGBAImage *);

static int BuildMapUaToAa(TIFFRGBAImage *img);
static int BuildMapBitdepth16To8(TIFFRGBAImage *img);

static const char photoTag[] = "PhotometricInterpretation";

/*
 * Helper constants used in Orientation tag handling
 */
#define FLIP_VERTICALLY 0x01
#define FLIP_HORIZONTALLY 0x02

#define EMSG_BUF_SIZE 1024

/*
 * Color conversion constants. We will define display types here.
 */

static const TIFFDisplay display_sRGB = {
    {/* XYZ -> luminance matrix */
     {3.2410F, -1.5374F, -0.4986F},
     {-0.9692F, 1.8760F, 0.0416F},
     {0.0556F, -0.2040F, 1.0570F}},
    100.0F,
    100.0F,
    100.0F, /* Light o/p for reference white */
    255,
    255,
    255, /* Pixel values for ref. white */
    1.0F,
    1.0F,
    1.0F, /* Residual light o/p for black pixel */
    2.4F,
    2.4F,
    2.4F, /* Gamma values for the three guns */
};

/*
 * Check the image to see if TIFFReadRGBAImage can deal with it.
 * 1/0 is returned according to whether or not the image can
 * be handled.  If 0 is returned, emsg contains the reason
 * why it is being rejected.
 */
int TIFFRGBAImageOK(TIFF *tif, char emsg[EMSG_BUF_SIZE])
{
    TIFFDirectory *td = &tif->tif_dir;
    uint16_t photometric;
    int colorchannels;

    if (!tif->tif_decodestatus)
    {
        snprintf(emsg, EMSG_BUF_SIZE,
                 "Sorry, requested compression method is not configured");
        return (0);
    }
    switch (td->td_bitspersample)
    {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
            break;
        default:
            snprintf(emsg, EMSG_BUF_SIZE,
                     "Sorry, can not handle images with %" PRIu16
                     "-bit samples",
                     td->td_bitspersample);
            return (0);
    }
    if (td->td_sampleformat == SAMPLEFORMAT_IEEEFP)
    {
        snprintf(
            emsg, EMSG_BUF_SIZE,
            "Sorry, can not handle images with IEEE floating-point samples");
        return (0);
    }
    colorchannels = td->td_samplesperpixel - td->td_extrasamples;
    if (!TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric))
    {
        switch (colorchannels)
        {
            case 1:
                photometric = PHOTOMETRIC_MINISBLACK;
                break;
            case 3:
                photometric = PHOTOMETRIC_RGB;
                break;
            default:
                snprintf(emsg, EMSG_BUF_SIZE, "Missing needed %s tag",
                         photoTag);
                return (0);
        }
    }
    switch (photometric)
    {
        case PHOTOMETRIC_MINISWHITE:
        case PHOTOMETRIC_MINISBLACK:
        case PHOTOMETRIC_PALETTE:
            if (td->td_planarconfig == PLANARCONFIG_CONTIG &&
                td->td_samplesperpixel != 1 && td->td_bitspersample < 8)
            {
                snprintf(
                    emsg, EMSG_BUF_SIZE,
                    "Sorry, can not handle contiguous data with %s=%" PRIu16
                    ", "
                    "and %s=%" PRIu16 " and Bits/Sample=%" PRIu16 "",
                    photoTag, photometric, "Samples/pixel",
                    td->td_samplesperpixel, td->td_bitspersample);
                return (0);
            }
            /*
             * We should likely validate that any extra samples are either
             * to be ignored, or are alpha, and if alpha we should try to use
             * them.  But for now we won't bother with this.
             */
            break;
        case PHOTOMETRIC_YCBCR:
            /*
             * TODO: if at all meaningful and useful, make more complete
             * support check here, or better still, refactor to let supporting
             * code decide whether there is support and what meaningful
             * error to return
             */
            break;
        case PHOTOMETRIC_RGB:
            if (colorchannels < 3)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, can not handle RGB image with %s=%d",
                         "Color channels", colorchannels);
                return (0);
            }
            break;
        case PHOTOMETRIC_SEPARATED:
        {
            uint16_t inkset;
            TIFFGetFieldDefaulted(tif, TIFFTAG_INKSET, &inkset);
            if (inkset != INKSET_CMYK)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, can not handle separated image with %s=%d",
                         "InkSet", inkset);
                return 0;
            }
            if (td->td_samplesperpixel < 4)
            {
                snprintf(
                    emsg, EMSG_BUF_SIZE,
                    "Sorry, can not handle separated image with %s=%" PRIu16,
                    "Samples/pixel", td->td_samplesperpixel);
                return 0;
            }
            break;
        }
        case PHOTOMETRIC_LOGL:
            if (td->td_compression != COMPRESSION_SGILOG)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, LogL data must have %s=%d", "Compression",
                         COMPRESSION_SGILOG);
                return (0);
            }
            break;
        case PHOTOMETRIC_LOGLUV:
            if (td->td_compression != COMPRESSION_SGILOG &&
                td->td_compression != COMPRESSION_SGILOG24)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, LogLuv data must have %s=%d or %d",
                         "Compression", COMPRESSION_SGILOG,
                         COMPRESSION_SGILOG24);
                return (0);
            }
            if (td->td_planarconfig != PLANARCONFIG_CONTIG)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, can not handle LogLuv images with %s=%" PRIu16,
                         "Planarconfiguration", td->td_planarconfig);
                return (0);
            }
            if (td->td_samplesperpixel != 3 || colorchannels != 3)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, can not handle image with %s=%" PRIu16
                         ", %s=%d",
                         "Samples/pixel", td->td_samplesperpixel,
                         "colorchannels", colorchannels);
                return 0;
            }
            break;
        case PHOTOMETRIC_CIELAB:
            if (td->td_samplesperpixel != 3 || colorchannels != 3 ||
                (td->td_bitspersample != 8 && td->td_bitspersample != 16))
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, can not handle image with %s=%" PRIu16
                         ", %s=%d and %s=%" PRIu16,
                         "Samples/pixel", td->td_samplesperpixel,
                         "colorchannels", colorchannels, "Bits/sample",
                         td->td_bitspersample);
                return 0;
            }
            break;
        default:
            snprintf(emsg, EMSG_BUF_SIZE,
                     "Sorry, can not handle image with %s=%" PRIu16, photoTag,
                     photometric);
            return (0);
    }
    return (1);
}

void TIFFRGBAImageEnd(TIFFRGBAImage *img)
{
    if (img->Map)
    {
        _TIFFfreeExt(img->tif, img->Map);
        img->Map = NULL;
    }
    if (img->BWmap)
    {
        _TIFFfreeExt(img->tif, img->BWmap);
        img->BWmap = NULL;
    }
    if (img->PALmap)
    {
        _TIFFfreeExt(img->tif, img->PALmap);
        img->PALmap = NULL;
    }
    if (img->ycbcr)
    {
        _TIFFfreeExt(img->tif, img->ycbcr);
        img->ycbcr = NULL;
    }
    if (img->cielab)
    {
        _TIFFfreeExt(img->tif, img->cielab);
        img->cielab = NULL;
    }
    if (img->UaToAa)
    {
        _TIFFfreeExt(img->tif, img->UaToAa);
        img->UaToAa = NULL;
    }
    if (img->Bitdepth16To8)
    {
        _TIFFfreeExt(img->tif, img->Bitdepth16To8);
        img->Bitdepth16To8 = NULL;
    }

    if (img->redcmap)
    {
        _TIFFfreeExt(img->tif, img->redcmap);
        _TIFFfreeExt(img->tif, img->greencmap);
        _TIFFfreeExt(img->tif, img->bluecmap);
        img->redcmap = img->greencmap = img->bluecmap = NULL;
    }
}

static int isCCITTCompression(TIFF *tif)
{
    uint16_t compress;
    TIFFGetField(tif, TIFFTAG_COMPRESSION, &compress);
    return (compress == COMPRESSION_CCITTFAX3 ||
            compress == COMPRESSION_CCITTFAX4 ||
            compress == COMPRESSION_CCITTRLE ||
            compress == COMPRESSION_CCITTRLEW);
}

int TIFFRGBAImageBegin(TIFFRGBAImage *img, TIFF *tif, int stop,
                       char emsg[EMSG_BUF_SIZE])
{
    uint16_t *sampleinfo;
    uint16_t extrasamples;
    uint16_t planarconfig;
    uint16_t compress;
    int colorchannels;
    uint16_t *red_orig, *green_orig, *blue_orig;
    int n_color;

    if (!TIFFRGBAImageOK(tif, emsg))
        return 0;

    /* Initialize to normal values */
    img->row_offset = 0;
    img->col_offset = 0;
    img->redcmap = NULL;
    img->greencmap = NULL;
    img->bluecmap = NULL;
    img->Map = NULL;
    img->BWmap = NULL;
    img->PALmap = NULL;
    img->ycbcr = NULL;
    img->cielab = NULL;
    img->UaToAa = NULL;
    img->Bitdepth16To8 = NULL;
    img->req_orientation = ORIENTATION_BOTLEFT; /* It is the default */

    img->tif = tif;
    img->stoponerr = stop;
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &img->bitspersample);
    switch (img->bitspersample)
    {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
            break;
        default:
            snprintf(emsg, EMSG_BUF_SIZE,
                     "Sorry, can not handle images with %" PRIu16
                     "-bit samples",
                     img->bitspersample);
            goto fail_return;
    }
    img->alpha = 0;
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &img->samplesperpixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_EXTRASAMPLES, &extrasamples,
                          &sampleinfo);
    if (extrasamples >= 1)
    {
        switch (sampleinfo[0])
        {
            case EXTRASAMPLE_UNSPECIFIED: /* Workaround for some images without
                                           */
                if (img->samplesperpixel >
                    3) /* correct info about alpha channel */
                    img->alpha = EXTRASAMPLE_ASSOCALPHA;
                break;
            case EXTRASAMPLE_ASSOCALPHA: /* data is pre-multiplied */
            case EXTRASAMPLE_UNASSALPHA: /* data is not pre-multiplied */
                img->alpha = sampleinfo[0];
                break;
        }
    }

#ifdef DEFAULT_EXTRASAMPLE_AS_ALPHA
    if (!TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &img->photometric))
        img->photometric = PHOTOMETRIC_MINISWHITE;

    if (extrasamples == 0 && img->samplesperpixel == 4 &&
        img->photometric == PHOTOMETRIC_RGB)
    {
        img->alpha = EXTRASAMPLE_ASSOCALPHA;
        extrasamples = 1;
    }
#endif

    colorchannels = img->samplesperpixel - extrasamples;
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &compress);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planarconfig);
    if (!TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &img->photometric))
    {
        switch (colorchannels)
        {
            case 1:
                if (isCCITTCompression(tif))
                    img->photometric = PHOTOMETRIC_MINISWHITE;
                else
                    img->photometric = PHOTOMETRIC_MINISBLACK;
                break;
            case 3:
                img->photometric = PHOTOMETRIC_RGB;
                break;
            default:
                snprintf(emsg, EMSG_BUF_SIZE, "Missing needed %s tag",
                         photoTag);
                goto fail_return;
        }
    }
    switch (img->photometric)
    {
        case PHOTOMETRIC_PALETTE:
            if (!TIFFGetField(tif, TIFFTAG_COLORMAP, &red_orig, &green_orig,
                              &blue_orig))
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Missing required \"Colormap\" tag");
                goto fail_return;
            }

            /* copy the colormaps so we can modify them */
            n_color = (1U << img->bitspersample);
            img->redcmap =
                (uint16_t *)_TIFFmallocExt(tif, sizeof(uint16_t) * n_color);
            img->greencmap =
                (uint16_t *)_TIFFmallocExt(tif, sizeof(uint16_t) * n_color);
            img->bluecmap =
                (uint16_t *)_TIFFmallocExt(tif, sizeof(uint16_t) * n_color);
            if (!img->redcmap || !img->greencmap || !img->bluecmap)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Out of memory for colormap copy");
                goto fail_return;
            }

            _TIFFmemcpy(img->redcmap, red_orig, n_color * 2);
            _TIFFmemcpy(img->greencmap, green_orig, n_color * 2);
            _TIFFmemcpy(img->bluecmap, blue_orig, n_color * 2);

            /* fall through... */
        case PHOTOMETRIC_MINISWHITE:
        case PHOTOMETRIC_MINISBLACK:
            if (planarconfig == PLANARCONFIG_CONTIG &&
                img->samplesperpixel != 1 && img->bitspersample < 8)
            {
                snprintf(
                    emsg, EMSG_BUF_SIZE,
                    "Sorry, can not handle contiguous data with %s=%" PRIu16
                    ", "
                    "and %s=%" PRIu16 " and Bits/Sample=%" PRIu16,
                    photoTag, img->photometric, "Samples/pixel",
                    img->samplesperpixel, img->bitspersample);
                goto fail_return;
            }
            break;
        case PHOTOMETRIC_YCBCR:
            /* It would probably be nice to have a reality check here. */
            if (planarconfig == PLANARCONFIG_CONTIG)
                /* can rely on libjpeg to convert to RGB */
                /* XXX should restore current state on exit */
                switch (compress)
                {
                    case COMPRESSION_JPEG:
                        /*
                         * TODO: when complete tests verify complete
                         * desubsampling and YCbCr handling, remove use of
                         * TIFFTAG_JPEGCOLORMODE in favor of tif_getimage.c
                         * native handling
                         */
                        TIFFSetField(tif, TIFFTAG_JPEGCOLORMODE,
                                     JPEGCOLORMODE_RGB);
                        img->photometric = PHOTOMETRIC_RGB;
                        break;
                    default:
                        /* do nothing */;
                        break;
                }
            /*
             * TODO: if at all meaningful and useful, make more complete
             * support check here, or better still, refactor to let supporting
             * code decide whether there is support and what meaningful
             * error to return
             */
            break;
        case PHOTOMETRIC_RGB:
            if (colorchannels < 3)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, can not handle RGB image with %s=%d",
                         "Color channels", colorchannels);
                goto fail_return;
            }
            break;
        case PHOTOMETRIC_SEPARATED:
        {
            uint16_t inkset;
            TIFFGetFieldDefaulted(tif, TIFFTAG_INKSET, &inkset);
            if (inkset != INKSET_CMYK)
            {
                snprintf(
                    emsg, EMSG_BUF_SIZE,
                    "Sorry, can not handle separated image with %s=%" PRIu16,
                    "InkSet", inkset);
                goto fail_return;
            }
            if (img->samplesperpixel < 4)
            {
                snprintf(
                    emsg, EMSG_BUF_SIZE,
                    "Sorry, can not handle separated image with %s=%" PRIu16,
                    "Samples/pixel", img->samplesperpixel);
                goto fail_return;
            }
        }
        break;
        case PHOTOMETRIC_LOGL:
            if (compress != COMPRESSION_SGILOG)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, LogL data must have %s=%d", "Compression",
                         COMPRESSION_SGILOG);
                goto fail_return;
            }
            TIFFSetField(tif, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_8BIT);
            img->photometric = PHOTOMETRIC_MINISBLACK; /* little white lie */
            img->bitspersample = 8;
            break;
        case PHOTOMETRIC_LOGLUV:
            if (compress != COMPRESSION_SGILOG &&
                compress != COMPRESSION_SGILOG24)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, LogLuv data must have %s=%d or %d",
                         "Compression", COMPRESSION_SGILOG,
                         COMPRESSION_SGILOG24);
                goto fail_return;
            }
            if (planarconfig != PLANARCONFIG_CONTIG)
            {
                snprintf(emsg, EMSG_BUF_SIZE,
                         "Sorry, can not handle LogLuv images with %s=%" PRIu16,
                         "Planarconfiguration", planarconfig);
                return (0);
            }
            TIFFSetField(tif, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_8BIT);
            img->photometric = PHOTOMETRIC_RGB; /* little white lie */
            img->bitspersample = 8;
            break;
        case PHOTOMETRIC_CIELAB:
            break;
        default:
            snprintf(emsg, EMSG_BUF_SIZE,
                     "Sorry, can not handle image with %s=%" PRIu16, photoTag,
                     img->photometric);
            goto fail_return;
    }
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &img->width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &img->height);
    TIFFGetFieldDefaulted(tif, TIFFTAG_ORIENTATION, &img->orientation);
    img->isContig =
        !(planarconfig == PLANARCONFIG_SEPARATE && img->samplesperpixel > 1);
    if (img->isContig)
    {
        if (!PickContigCase(img))
        {
            snprintf(emsg, EMSG_BUF_SIZE, "Sorry, can not handle image");
            goto fail_return;
        }
    }
    else
    {
        if (!PickSeparateCase(img))
        {
            snprintf(emsg, EMSG_BUF_SIZE, "Sorry, can not handle image");
            goto fail_return;
        }
    }
    return 1;

fail_return:
    TIFFRGBAImageEnd(img);
    return 0;
}

int TIFFRGBAImageGet(TIFFRGBAImage *img, uint32_t *raster, uint32_t w,
                     uint32_t h)
{
    if (img->get == NULL)
    {
        TIFFErrorExtR(img->tif, TIFFFileName(img->tif),
                      "No \"get\" routine setup");
        return (0);
    }
    if (img->put.any == NULL)
    {
        TIFFErrorExtR(
            img->tif, TIFFFileName(img->tif),
            "No \"put\" routine setupl; probably can not handle image format");
        return (0);
    }
    return (*img->get)(img, raster, w, h);
}

/*
 * Read the specified image into an ABGR-format rastertaking in account
 * specified orientation.
 */
int TIFFReadRGBAImageOriented(TIFF *tif, uint32_t rwidth, uint32_t rheight,
                              uint32_t *raster, int orientation, int stop)
{
    char emsg[EMSG_BUF_SIZE] = "";
    TIFFRGBAImage img;
    int ok;

    if (TIFFRGBAImageOK(tif, emsg) && TIFFRGBAImageBegin(&img, tif, stop, emsg))
    {
        img.req_orientation = (uint16_t)orientation;
        /* XXX verify rwidth and rheight against width and height */
        ok = TIFFRGBAImageGet(&img, raster + (rheight - img.height) * rwidth,
                              rwidth, img.height);
        TIFFRGBAImageEnd(&img);
    }
    else
    {
        TIFFErrorExtR(tif, TIFFFileName(tif), "%s", emsg);
        ok = 0;
    }
    return (ok);
}

/*
 * Read the specified image into an ABGR-format raster. Use bottom left
 * origin for raster by default.
 */
int TIFFReadRGBAImage(TIFF *tif, uint32_t rwidth, uint32_t rheight,
                      uint32_t *raster, int stop)
{
    return TIFFReadRGBAImageOriented(tif, rwidth, rheight, raster,
                                     ORIENTATION_BOTLEFT, stop);
}

static int setorientation(TIFFRGBAImage *img)
{
    switch (img->orientation)
    {
        case ORIENTATION_TOPLEFT:
        case ORIENTATION_LEFTTOP:
            if (img->req_orientation == ORIENTATION_TOPRIGHT ||
                img->req_orientation == ORIENTATION_RIGHTTOP)
                return FLIP_HORIZONTALLY;
            else if (img->req_orientation == ORIENTATION_BOTRIGHT ||
                     img->req_orientation == ORIENTATION_RIGHTBOT)
                return FLIP_HORIZONTALLY | FLIP_VERTICALLY;
            else if (img->req_orientation == ORIENTATION_BOTLEFT ||
                     img->req_orientation == ORIENTATION_LEFTBOT)
                return FLIP_VERTICALLY;
            else
                return 0;
        case ORIENTATION_TOPRIGHT:
        case ORIENTATION_RIGHTTOP:
            if (img->req_orientation == ORIENTATION_TOPLEFT ||
                img->req_orientation == ORIENTATION_LEFTTOP)
                return FLIP_HORIZONTALLY;
            else if (img->req_orientation == ORIENTATION_BOTRIGHT ||
                     img->req_orientation == ORIENTATION_RIGHTBOT)
                return FLIP_VERTICALLY;
            else if (img->req_orientation == ORIENTATION_BOTLEFT ||
                     img->req_orientation == ORIENTATION_LEFTBOT)
                return FLIP_HORIZONTALLY | FLIP_VERTICALLY;
            else
                return 0;
        case ORIENTATION_BOTRIGHT:
        case ORIENTATION_RIGHTBOT:
            if (img->req_orientation == ORIENTATION_TOPLEFT ||
                img->req_orientation == ORIENTATION_LEFTTOP)
                return FLIP_HORIZONTALLY | FLIP_VERTICALLY;
            else if (img->req_orientation == ORIENTATION_TOPRIGHT ||
                     img->req_orientation == ORIENTATION_RIGHTTOP)
                return FLIP_VERTICALLY;
            else if (img->req_orientation == ORIENTATION_BOTLEFT ||
                     img->req_orientation == ORIENTATION_LEFTBOT)
                return FLIP_HORIZONTALLY;
            else
                return 0;
        case ORIENTATION_BOTLEFT:
        case ORIENTATION_LEFTBOT:
            if (img->req_orientation == ORIENTATION_TOPLEFT ||
                img->req_orientation == ORIENTATION_LEFTTOP)
                return FLIP_VERTICALLY;
            else if (img->req_orientation == ORIENTATION_TOPRIGHT ||
                     img->req_orientation == ORIENTATION_RIGHTTOP)
                return FLIP_HORIZONTALLY | FLIP_VERTICALLY;
            else if (img->req_orientation == ORIENTATION_BOTRIGHT ||
                     img->req_orientation == ORIENTATION_RIGHTBOT)
                return FLIP_HORIZONTALLY;
            else
                return 0;
        default: /* NOTREACHED */
            return 0;
    }
}

/*
 * Get an tile-organized image that has
 *	PlanarConfiguration contiguous if SamplesPerPixel > 1
 * or
 *	SamplesPerPixel == 1
 */
static int gtTileContig(TIFFRGBAImage *img, uint32_t *raster, uint32_t w,
                        uint32_t h)
{
    TIFF *tif = img->tif;
    tileContigRoutine put = img->put.contig;
    uint32_t col, row, y, rowstoread;
    tmsize_t pos;
    uint32_t tw, th;
    unsigned char *buf = NULL;
    int32_t fromskew, toskew;
    uint32_t nrow;
    int ret = 1, flip;
    uint32_t this_tw, tocol;
    int32_t this_toskew, leftmost_toskew;
    int32_t leftmost_fromskew;
    uint32_t leftmost_tw;
    tmsize_t bufsize;

    bufsize = TIFFTileSize(tif);
    if (bufsize == 0)
    {
        TIFFErrorExtR(tif, TIFFFileName(tif), "%s", "No space for tile buffer");
        return (0);
    }

    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tw);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &th);

    flip = setorientation(img);
    if (flip & FLIP_VERTICALLY)
    {
        if ((tw + w) > INT_MAX)
        {
            TIFFErrorExtR(tif, TIFFFileName(tif), "%s",
                          "unsupported tile size (too wide)");
            return (0);
        }
        y = h - 1;
        toskew = -(int32_t)(tw + w);
    }
    else
    {
        if (tw > (INT_MAX + w))
        {
            TIFFErrorExtR(tif, TIFFFileName(tif), "%s",
                          "unsupported tile size (too wide)");
            return (0);
        }
        y = 0;
        toskew = -(int32_t)(tw - w);
    }

    /*
     *	Leftmost tile is clipped on left side if col_offset > 0.
     */
    leftmost_fromskew = img->col_offset % tw;
    leftmost_tw = tw - leftmost_fromskew;
    leftmost_toskew = toskew + leftmost_fromskew;
    for (row = 0; ret != 0 && row < h; row += nrow)
    {
        rowstoread = th - (row + img->row_offset) % th;
        nrow = (row + rowstoread > h ? h - row : rowstoread);
        fromskew = leftmost_fromskew;
        this_tw = leftmost_tw;
        this_toskew = leftmost_toskew;
        tocol = 0;
        col = img->col_offset;
        while (tocol < w)
        {
            if (_TIFFReadTileAndAllocBuffer(tif, (void **)&buf, bufsize, col,
                                            row + img->row_offset, 0,
                                            0) == (tmsize_t)(-1) &&
                (buf == NULL || img->stoponerr))
            {
                ret = 0;
                break;
            }
            pos = ((row + img->row_offset) % th) * TIFFTileRowSize(tif) +
                  ((tmsize_t)fromskew * img->samplesperpixel);
            if (tocol + this_tw > w)
            {
                /*
                 * Rightmost tile is clipped on right side.
                 */
                fromskew = tw - (w - tocol);
                this_tw = tw - fromskew;
                this_toskew = toskew + fromskew;
            }
            tmsize_t roffset = (tmsize_t)y * w + tocol;
            (*put)(img, raster + roffset, tocol, y, this_tw, nrow, fromskew,
                   this_toskew, buf + pos);
            tocol += this_tw;
            col += this_tw;
            /*
             * After the leftmost tile, tiles are no longer clipped on left
             * side.
             */
            fromskew = 0;
            this_tw = tw;
            this_toskew = toskew;
        }

        y += ((flip & FLIP_VERTICALLY) ? -(int32_t)nrow : (int32_t)nrow);
    }
    _TIFFfreeExt(img->tif, buf);

    if (flip & FLIP_HORIZONTALLY)
    {
        uint32_t line;

        for (line = 0; line < h; line++)
        {
            uint32_t *left = raster + (line * w);
            uint32_t *right = left + w - 1;

            while (left < right)
            {
                uint32_t temp = *left;
                *left = *right;
                *right = temp;
                left++;
                right--;
            }
        }
    }

    return (ret);
}

/*
 * Get an tile-organized image that has
 *	 SamplesPerPixel > 1
 *	 PlanarConfiguration separated
 * We assume that all such images are RGB.
 */
static int gtTileSeparate(TIFFRGBAImage *img, uint32_t *raster, uint32_t w,
                          uint32_t h)
{
    TIFF *tif = img->tif;
    tileSeparateRoutine put = img->put.separate;
    uint32_t col, row, y, rowstoread;
    tmsize_t pos;
    uint32_t tw, th;
    unsigned char *buf = NULL;
    unsigned char *p0 = NULL;
    unsigned char *p1 = NULL;
    unsigned char *p2 = NULL;
    unsigned char *pa = NULL;
    tmsize_t tilesize;
    tmsize_t bufsize;
    int32_t fromskew, toskew;
    int alpha = img->alpha;
    uint32_t nrow;
    int ret = 1, flip;
    uint16_t colorchannels;
    uint32_t this_tw, tocol;
    int32_t this_toskew, leftmost_toskew;
    int32_t leftmost_fromskew;
    uint32_t leftmost_tw;

    tilesize = TIFFTileSize(tif);
    bufsize =
        _TIFFMultiplySSize(tif, alpha ? 4 : 3, tilesize, "gtTileSeparate");
    if (bufsize == 0)
    {
        return (0);
    }

    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tw);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &th);

    flip = setorientation(img);
    if (flip & FLIP_VERTICALLY)
    {
        if ((tw + w) > INT_MAX)
        {
            TIFFErrorExtR(tif, TIFFFileName(tif), "%s",
                          "unsupported tile size (too wide)");
            return (0);
        }
        y = h - 1;
        toskew = -(int32_t)(tw + w);
    }
    else
    {
        if (tw > (INT_MAX + w))
        {
            TIFFErrorExtR(tif, TIFFFileName(tif), "%s",
                          "unsupported tile size (too wide)");
            return (0);
        }
        y = 0;
        toskew = -(int32_t)(tw - w);
    }

    switch (img->photometric)
    {
        case PHOTOMETRIC_MINISWHITE:
        case PHOTOMETRIC_MINISBLACK:
        case PHOTOMETRIC_PALETTE:
            colorchannels = 1;
            break;

        default:
            colorchannels = 3;
            break;
    }

    /*
     *	Leftmost tile is clipped on left side if col_offset > 0.
     */
    leftmost_fromskew = img->col_offset % tw;
    leftmost_tw = tw - leftmost_fromskew;
    leftmost_toskew = toskew + leftmost_fromskew;
    for (row = 0; ret != 0 && row < h; row += nrow)
    {
        rowstoread = th - (row + img->row_offset) % th;
        nrow = (row + rowstoread > h ? h - row : rowstoread);
        fromskew = leftmost_fromskew;
        this_tw = leftmost_tw;
        this_toskew = leftmost_toskew;
        tocol = 0;
        col = img->col_offset;
        while (tocol < w)
        {
            if (buf == NULL)
            {
                if (_TIFFReadTileAndAllocBuffer(tif, (void **)&buf, bufsize,
                                                col, row + img->row_offset, 0,
                                                0) == (tmsize_t)(-1) &&
                    (buf == NULL || img->stoponerr))
                {
                    ret = 0;
                    break;
                }
                p0 = buf;
                if (colorchannels == 1)
                {
                    p2 = p1 = p0;
                    pa = (alpha ? (p0 + 3 * tilesize) : NULL);
                }
                else
                {
                    p1 = p0 + tilesize;
                    p2 = p1 + tilesize;
                    pa = (alpha ? (p2 + tilesize) : NULL);
                }
            }
            else if (TIFFReadTile(tif, p0, col, row + img->row_offset, 0, 0) ==
                         (tmsize_t)(-1) &&
                     img->stoponerr)
            {
                ret = 0;
                break;
            }
            if (colorchannels > 1 &&
                TIFFReadTile(tif, p1, col, row + img->row_offset, 0, 1) ==
                    (tmsize_t)(-1) &&
                img->stoponerr)
            {
                ret = 0;
                break;
            }
            if (colorchannels > 1 &&
                TIFFReadTile(tif, p2, col, row + img->row_offset, 0, 2) ==
                    (tmsize_t)(-1) &&
                img->stoponerr)
            {
                ret = 0;
                break;
            }
            if (alpha &&
                TIFFReadTile(tif, pa, col, row + img->row_offset, 0,
                             colorchannels) == (tmsize_t)(-1) &&
                img->stoponerr)
            {
                ret = 0;
                break;
            }

            pos = ((row + img->row_offset) % th) * TIFFTileRowSize(tif) +
                  ((tmsize_t)fromskew * img->samplesperpixel);
            if (tocol + this_tw > w)
            {
                /*
                 * Rightmost tile is clipped on right side.
                 */
                fromskew = tw - (w - tocol);
                this_tw = tw - fromskew;
                this_toskew = toskew + fromskew;
            }
            tmsize_t roffset = (tmsize_t)y * w + tocol;
            (*put)(img, raster + roffset, tocol, y, this_tw, nrow, fromskew,
                   this_toskew, p0 + pos, p1 + pos, p2 + pos,
                   (alpha ? (pa + pos) : NULL));
            tocol += this_tw;
            col += this_tw;
            /*
             * After the leftmost tile, tiles are no longer clipped on left
             * side.
             */
            fromskew = 0;
            this_tw = tw;
            this_toskew = toskew;
        }

        y += ((flip & FLIP_VERTICALLY) ? -(int32_t)nrow : (int32_t)nrow);
    }

    if (flip & FLIP_HORIZONTALLY)
    {
        uint32_t line;

        for (line = 0; line < h; line++)
        {
            uint32_t *left = raster + (line * w);
            uint32_t *right = left + w - 1;

            while (left < right)
            {
                uint32_t temp = *left;
                *left = *right;
                *right = temp;
                left++;
                right--;
            }
        }
    }

    _TIFFfreeExt(img->tif, buf);
    return (ret);
}

/*
 * Get a strip-organized image that has
 *	PlanarConfiguration contiguous if SamplesPerPixel > 1
 * or
 *	SamplesPerPixel == 1
 */
static int gtStripContig(TIFFRGBAImage *img, uint32_t *raster, uint32_t w,
                         uint32_t h)
{
    TIFF *tif = img->tif;
    tileContigRoutine put = img->put.contig;
    uint32_t row, y, nrow, nrowsub, rowstoread;
    tmsize_t pos;
    unsigned char *buf = NULL;
    uint32_t rowsperstrip;
    uint16_t subsamplinghor, subsamplingver;
    uint32_t imagewidth = img->width;
    tmsize_t scanline;
    int32_t fromskew, toskew;
    int ret = 1, flip;
    tmsize_t maxstripsize;

    TIFFGetFieldDefaulted(tif, TIFFTAG_YCBCRSUBSAMPLING, &subsamplinghor,
                          &subsamplingver);
    if (subsamplingver == 0)
    {
        TIFFErrorExtR(tif, TIFFFileName(tif),
                      "Invalid vertical YCbCr subsampling");
        return (0);
    }

    maxstripsize = TIFFStripSize(tif);

    flip = setorientation(img);
    if (flip & FLIP_VERTICALLY)
    {
        if (w > INT_MAX)
        {
            TIFFErrorExtR(tif, TIFFFileName(tif), "Width overflow");
            return (0);
        }
        y = h - 1;
        toskew = -(int32_t)(w + w);
    }
    else
    {
        y = 0;
        toskew = -(int32_t)(w - w);
    }

    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsperstrip);

    scanline = TIFFScanlineSize(tif);
    fromskew = (w < imagewidth ? imagewidth - w : 0);
    for (row = 0; row < h; row += nrow)
    {
        uint32_t temp;
        rowstoread = rowsperstrip - (row + img->row_offset) % rowsperstrip;
        nrow = (row + rowstoread > h ? h - row : rowstoread);
        nrowsub = nrow;
        if ((nrowsub % subsamplingver) != 0)
            nrowsub += subsamplingver - nrowsub % subsamplingver;
        temp = (row + img->row_offset) % rowsperstrip + nrowsub;
        if (scanline > 0 && temp > (size_t)(TIFF_TMSIZE_T_MAX / scanline))
        {
            TIFFErrorExtR(tif, TIFFFileName(tif),
                          "Integer overflow in gtStripContig");
            return 0;
        }
        if (_TIFFReadEncodedStripAndAllocBuffer(
                tif, TIFFComputeStrip(tif, row + img->row_offset, 0),
                (void **)(&buf), maxstripsize,
                temp * scanline) == (tmsize_t)(-1) &&
            (buf == NULL || img->stoponerr))
        {
            ret = 0;
            break;
        }

        pos = ((row + img->row_offset) % rowsperstrip) * scanline +
              ((tmsize_t)img->col_offset * img->samplesperpixel);
        tmsize_t roffset = (tmsize_t)y * w;
        (*put)(img, raster + roffset, 0, y, w, nrow, fromskew, toskew,
               buf + pos);
        y += ((flip & FLIP_VERTICALLY) ? -(int32_t)nrow : (int32_t)nrow);
    }

    if (flip & FLIP_HORIZONTALLY)
    {
        uint32_t line;

        for (line = 0; line < h; line++)
        {
            uint32_t *left = raster + (line * w);
            uint32_t *right = left + w - 1;

            while (left < right)
            {
                uint32_t temp = *left;
                *left = *right;
                *right = temp;
                left++;
                right--;
            }
        }
    }

    _TIFFfreeExt(img->tif, buf);
    return (ret);
}

/*
 * Get a strip-organized image with
 *	 SamplesPerPixel > 1
 *	 PlanarConfiguration separated
 * We assume that all such images are RGB.
 */
static int gtStripSeparate(TIFFRGBAImage *img, uint32_t *raster, uint32_t w,
                           uint32_t h)
{
    TIFF *tif = img->tif;
    tileSeparateRoutine put = img->put.separate;
    unsigned char *buf = NULL;
    unsigned char *p0 = NULL, *p1 = NULL, *p2 = NULL, *pa = NULL;
    uint32_t row, y, nrow, rowstoread;
    tmsize_t pos;
    tmsize_t scanline;
    uint32_t rowsperstrip, offset_row;
    uint32_t imagewidth = img->width;
    tmsize_t stripsize;
    tmsize_t bufsize;
    int32_t fromskew, toskew;
    int alpha = img->alpha;
    int ret = 1, flip;
    uint16_t colorchannels;

    stripsize = TIFFStripSize(tif);
    bufsize =
        _TIFFMultiplySSize(tif, alpha ? 4 : 3, stripsize, "gtStripSeparate");
    if (bufsize == 0)
    {
        return (0);
    }

    flip = setorientation(img);
    if (flip & FLIP_VERTICALLY)
    {
        if (w > INT_MAX)
        {
            TIFFErrorExtR(tif, TIFFFileName(tif), "Width overflow");
            return (0);
        }
        y = h - 1;
        toskew = -(int32_t)(w + w);
    }
    else
    {
        y = 0;
        toskew = -(int32_t)(w - w);
    }

    switch (img->photometric)
    {
        case PHOTOMETRIC_MINISWHITE:
        case PHOTOMETRIC_MINISBLACK:
        case PHOTOMETRIC_PALETTE:
            colorchannels = 1;
            break;

        default:
            colorchannels = 3;
            break;
    }

    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsperstrip);
    scanline = TIFFScanlineSize(tif);
    fromskew = (w < imagewidth ? imagewidth - w : 0);
    for (row = 0; row < h; row += nrow)
    {
        uint32_t temp;
        rowstoread = rowsperstrip - (row + img->row_offset) % rowsperstrip;
        nrow = (row + rowstoread > h ? h - row : rowstoread);
        offset_row = row + img->row_offset;
        temp = (row + img->row_offset) % rowsperstrip + nrow;
        if (scanline > 0 && temp > (size_t)(TIFF_TMSIZE_T_MAX / scanline))
        {
            TIFFErrorExtR(tif, TIFFFileName(tif),
                          "Integer overflow in gtStripSeparate");
            return 0;
        }
        if (buf == NULL)
        {
            if (_TIFFReadEncodedStripAndAllocBuffer(
                    tif, TIFFComputeStrip(tif, offset_row, 0), (void **)&buf,
                    bufsize, temp * scanline) == (tmsize_t)(-1) &&
                (buf == NULL || img->stoponerr))
            {
                ret = 0;
                break;
            }
            p0 = buf;
            if (colorchannels == 1)
            {
                p2 = p1 = p0;
                pa = (alpha ? (p0 + 3 * stripsize) : NULL);
            }
            else
            {
                p1 = p0 + stripsize;
                p2 = p1 + stripsize;
                pa = (alpha ? (p2 + stripsize) : NULL);
            }
        }
        else if (TIFFReadEncodedStrip(tif, TIFFComputeStrip(tif, offset_row, 0),
                                      p0, temp * scanline) == (tmsize_t)(-1) &&
                 img->stoponerr)
        {
            ret = 0;
            break;
        }
        if (colorchannels > 1 &&
            TIFFReadEncodedStrip(tif, TIFFComputeStrip(tif, offset_row, 1), p1,
                                 temp * scanline) == (tmsize_t)(-1) &&
            img->stoponerr)
        {
            ret = 0;
            break;
        }
        if (colorchannels > 1 &&
            TIFFReadEncodedStrip(tif, TIFFComputeStrip(tif, offset_row, 2), p2,
                                 temp * scanline) == (tmsize_t)(-1) &&
            img->stoponerr)
        {
            ret = 0;
            break;
        }
        if (alpha)
        {
            if (TIFFReadEncodedStrip(
                    tif, TIFFComputeStrip(tif, offset_row, colorchannels), pa,
                    temp * scanline) == (tmsize_t)(-1) &&
                img->stoponerr)
            {
                ret = 0;
                break;
            }
        }

        pos = ((row + img->row_offset) % rowsperstrip) * scanline +
              ((tmsize_t)img->col_offset * img->samplesperpixel);
        tmsize_t roffset = (tmsize_t)y * w;
        (*put)(img, raster + roffset, 0, y, w, nrow, fromskew, toskew, p0 + pos,
               p1 + pos, p2 + pos, (alpha ? (pa + pos) : NULL));
        y += ((flip & FLIP_VERTICALLY) ? -(int32_t)nrow : (int32_t)nrow);
    }

    if (flip & FLIP_HORIZONTALLY)
    {
        uint32_t line;

        for (line = 0; line < h; line++)
        {
            uint32_t *left = raster + (line * w);
            uint32_t *right = left + w - 1;

            while (left < right)
            {
                uint32_t temp = *left;
                *left = *right;
                *right = temp;
                left++;
                right--;
            }
        }
    }

    _TIFFfreeExt(img->tif, buf);
    return (ret);
}

/*
 * The following routines move decoded data returned
 * from the TIFF library into rasters filled with packed
 * ABGR pixels (i.e. suitable for passing to lrecwrite.)
 *
 * The routines have been created according to the most
 * important cases and optimized.  PickContigCase and
 * PickSeparateCase analyze the parameters and select
 * the appropriate "get" and "put" routine to use.
 */
#define REPEAT8(op)                                                            \
    REPEAT4(op);                                                               \
    REPEAT4(op)
#define REPEAT4(op)                                                            \
    REPEAT2(op);                                                               \
    REPEAT2(op)
#define REPEAT2(op)                                                            \
    op;                                                                        \
    op
#define CASE8(x, op)                                                           \
    switch (x)                                                                 \
    {                                                                          \
        case 7:                                                                \
            op; /*-fallthrough*/                                               \
        case 6:                                                                \
            op; /*-fallthrough*/                                               \
        case 5:                                                                \
            op; /*-fallthrough*/                                               \
        case 4:                                                                \
            op; /*-fallthrough*/                                               \
        case 3:                                                                \
            op; /*-fallthrough*/                                               \
        case 2:                                                                \
            op; /*-fallthrough*/                                               \
        case 1:                                                                \
            op;                                                                \
    }
#define CASE4(x, op)                                                           \
    switch (x)                                                                 \
    {                                                                          \
        case 3:                                                                \
            op; /*-fallthrough*/                                               \
        case 2:                                                                \
            op; /*-fallthrough*/                                               \
        case 1:                                                                \
            op;                                                                \
    }
#define NOP

#define UNROLL8(w, op1, op2)                                                   \
    {                                                                          \
        uint32_t _x;                                                           \
        for (_x = w; _x >= 8; _x -= 8)                                         \
        {                                                                      \
            op1;                                                               \
            REPEAT8(op2);                                                      \
        }                                                                      \
        if (_x > 0)                                                            \
        {                                                                      \
            op1;                                                               \
            CASE8(_x, op2);                                                    \
        }                                                                      \
    }
#define UNROLL4(w, op1, op2)                                                   \
    {                                                                          \
        uint32_t _x;                                                           \
        for (_x = w; _x >= 4; _x -= 4)                                         \
        {                                                                      \
            op1;                                                               \
            REPEAT4(op2);                                                      \
        }                                                                      \
        if (_x > 0)                                                            \
        {                                                                      \
            op1;                                                               \
            CASE4(_x, op2);                                                    \
        }                                                                      \
    }
#define UNROLL2(w, op1, op2)                                                   \
    {                                                                          \
        uint32_t _x;                                                           \
        for (_x = w; _x >= 2; _x -= 2)                                         \
        {                                                                      \
            op1;                                                               \
            REPEAT2(op2);                                                      \
        }                                                                      \
        if (_x)                                                                \
        {                                                                      \
            op1;                                                               \
            op2;                                                               \
        }                                                                      \
    }

#define SKEW(r, g, b, skew)                                                    \
    {                                                                          \
        r += skew;                                                             \
        g += skew;                                                             \
        b += skew;                                                             \
    }
#define SKEW4(r, g, b, a, skew)                                                \
    {                                                                          \
        r += skew;                                                             \
        g += skew;                                                             \
        b += skew;                                                             \
        a += skew;                                                             \
    }

#define A1 (((uint32_t)0xffL) << 24)
#define PACK(r, g, b)                                                          \
    ((uint32_t)(r) | ((uint32_t)(g) << 8) | ((uint32_t)(b) << 16) | A1)
#define PACK4(r, g, b, a)                                                      \
    ((uint32_t)(r) | ((uint32_t)(g) << 8) | ((uint32_t)(b) << 16) |            \
     ((uint32_t)(a) << 24))
#define W2B(v) (((v) >> 8) & 0xff)
/* TODO: PACKW should have be made redundant in favor of Bitdepth16To8 LUT */
#define PACKW(r, g, b)                                                         \
    ((uint32_t)W2B(r) | ((uint32_t)W2B(g) << 8) | ((uint32_t)W2B(b) << 16) | A1)
#define PACKW4(r, g, b, a)                                                     \
    ((uint32_t)W2B(r) | ((uint32_t)W2B(g) << 8) | ((uint32_t)W2B(b) << 16) |   \
     ((uint32_t)W2B(a) << 24))

#define DECLAREContigPutFunc(name)                                             \
    static void name(TIFFRGBAImage *img, uint32_t *cp, uint32_t x, uint32_t y, \
                     uint32_t w, uint32_t h, int32_t fromskew, int32_t toskew, \
                     unsigned char *pp)

/*
 * 8-bit palette => colormap/RGB
 */
DECLAREContigPutFunc(put8bitcmaptile)
{
    uint32_t **PALmap = img->PALmap;
    int samplesperpixel = img->samplesperpixel;

    (void)y;
    for (; h > 0; --h)
    {
        for (x = w; x > 0; --x)
        {
            *cp++ = PALmap[*pp][0];
            pp += samplesperpixel;
        }
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 4-bit palette => colormap/RGB
 */
DECLAREContigPutFunc(put4bitcmaptile)
{
    uint32_t **PALmap = img->PALmap;

    (void)x;
    (void)y;
    fromskew /= 2;
    for (; h > 0; --h)
    {
        uint32_t *bw;
        UNROLL2(w, bw = PALmap[*pp++], *cp++ = *bw++);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 2-bit palette => colormap/RGB
 */
DECLAREContigPutFunc(put2bitcmaptile)
{
    uint32_t **PALmap = img->PALmap;

    (void)x;
    (void)y;
    fromskew /= 4;
    for (; h > 0; --h)
    {
        uint32_t *bw;
        UNROLL4(w, bw = PALmap[*pp++], *cp++ = *bw++);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 1-bit palette => colormap/RGB
 */
DECLAREContigPutFunc(put1bitcmaptile)
{
    uint32_t **PALmap = img->PALmap;

    (void)x;
    (void)y;
    fromskew /= 8;
    for (; h > 0; --h)
    {
        uint32_t *bw;
        UNROLL8(w, bw = PALmap[*pp++], *cp++ = *bw++);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 8-bit greyscale => colormap/RGB
 */
DECLAREContigPutFunc(putgreytile)
{
    int samplesperpixel = img->samplesperpixel;
    uint32_t **BWmap = img->BWmap;

    (void)y;
    for (; h > 0; --h)
    {
        for (x = w; x > 0; --x)
        {
            *cp++ = BWmap[*pp][0];
            pp += samplesperpixel;
        }
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 8-bit greyscale with associated alpha => colormap/RGBA
 */
DECLAREContigPutFunc(putagreytile)
{
    int samplesperpixel = img->samplesperpixel;
    uint32_t **BWmap = img->BWmap;

    (void)y;
    for (; h > 0; --h)
    {
        for (x = w; x > 0; --x)
        {
            *cp++ = BWmap[*pp][0] & ((uint32_t) * (pp + 1) << 24 | ~A1);
            pp += samplesperpixel;
        }
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 16-bit greyscale => colormap/RGB
 */
DECLAREContigPutFunc(put16bitbwtile)
{
    int samplesperpixel = img->samplesperpixel;
    uint32_t **BWmap = img->BWmap;

    (void)y;
    for (; h > 0; --h)
    {
        uint16_t *wp = (uint16_t *)pp;

        for (x = w; x > 0; --x)
        {
            /* use high order byte of 16bit value */

            *cp++ = BWmap[*wp >> 8][0];
            pp += 2 * samplesperpixel;
            wp += samplesperpixel;
        }
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 1-bit bilevel => colormap/RGB
 */
DECLAREContigPutFunc(put1bitbwtile)
{
    uint32_t **BWmap = img->BWmap;

    (void)x;
    (void)y;
    fromskew /= 8;
    for (; h > 0; --h)
    {
        uint32_t *bw;
        UNROLL8(w, bw = BWmap[*pp++], *cp++ = *bw++);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 2-bit greyscale => colormap/RGB
 */
DECLAREContigPutFunc(put2bitbwtile)
{
    uint32_t **BWmap = img->BWmap;

    (void)x;
    (void)y;
    fromskew /= 4;
    for (; h > 0; --h)
    {
        uint32_t *bw;
        UNROLL4(w, bw = BWmap[*pp++], *cp++ = *bw++);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 4-bit greyscale => colormap/RGB
 */
DECLAREContigPutFunc(put4bitbwtile)
{
    uint32_t **BWmap = img->BWmap;

    (void)x;
    (void)y;
    fromskew /= 2;
    for (; h > 0; --h)
    {
        uint32_t *bw;
        UNROLL2(w, bw = BWmap[*pp++], *cp++ = *bw++);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 8-bit packed samples, no Map => RGB
 */
DECLAREContigPutFunc(putRGBcontig8bittile)
{
    int samplesperpixel = img->samplesperpixel;

    (void)x;
    (void)y;
    fromskew *= samplesperpixel;
    for (; h > 0; --h)
    {
        UNROLL8(w, NOP, *cp++ = PACK(pp[0], pp[1], pp[2]);
                pp += samplesperpixel);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 8-bit packed samples => RGBA w/ associated alpha
 * (known to have Map == NULL)
 */
DECLAREContigPutFunc(putRGBAAcontig8bittile)
{
    int samplesperpixel = img->samplesperpixel;

    (void)x;
    (void)y;
    fromskew *= samplesperpixel;
    for (; h > 0; --h)
    {
        UNROLL8(w, NOP, *cp++ = PACK4(pp[0], pp[1], pp[2], pp[3]);
                pp += samplesperpixel);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 8-bit packed samples => RGBA w/ unassociated alpha
 * (known to have Map == NULL)
 */
DECLAREContigPutFunc(putRGBUAcontig8bittile)
{
    int samplesperpixel = img->samplesperpixel;
    (void)y;
    fromskew *= samplesperpixel;
    for (; h > 0; --h)
    {
        uint32_t r, g, b, a;
        uint8_t *m;
        for (x = w; x > 0; --x)
        {
            a = pp[3];
            m = img->UaToAa + ((size_t)a << 8);
            r = m[pp[0]];
            g = m[pp[1]];
            b = m[pp[2]];
            *cp++ = PACK4(r, g, b, a);
            pp += samplesperpixel;
        }
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 16-bit packed samples => RGB
 */
DECLAREContigPutFunc(putRGBcontig16bittile)
{
    int samplesperpixel = img->samplesperpixel;
    uint16_t *wp = (uint16_t *)pp;
    (void)y;
    fromskew *= samplesperpixel;
    for (; h > 0; --h)
    {
        for (x = w; x > 0; --x)
        {
            *cp++ = PACK(img->Bitdepth16To8[wp[0]], img->Bitdepth16To8[wp[1]],
                         img->Bitdepth16To8[wp[2]]);
            wp += samplesperpixel;
        }
        cp += toskew;
        wp += fromskew;
    }
}

/*
 * 16-bit packed samples => RGBA w/ associated alpha
 * (known to have Map == NULL)
 */
DECLAREContigPutFunc(putRGBAAcontig16bittile)
{
    int samplesperpixel = img->samplesperpixel;
    uint16_t *wp = (uint16_t *)pp;
    (void)y;
    fromskew *= samplesperpixel;
    for (; h > 0; --h)
    {
        for (x = w; x > 0; --x)
        {
            *cp++ = PACK4(img->Bitdepth16To8[wp[0]], img->Bitdepth16To8[wp[1]],
                          img->Bitdepth16To8[wp[2]], img->Bitdepth16To8[wp[3]]);
            wp += samplesperpixel;
        }
        cp += toskew;
        wp += fromskew;
    }
}

/*
 * 16-bit packed samples => RGBA w/ unassociated alpha
 * (known to have Map == NULL)
 */
DECLAREContigPutFunc(putRGBUAcontig16bittile)
{
    int samplesperpixel = img->samplesperpixel;
    uint16_t *wp = (uint16_t *)pp;
    (void)y;
    fromskew *= samplesperpixel;
    for (; h > 0; --h)
    {
        uint32_t r, g, b, a;
        uint8_t *m;
        for (x = w; x > 0; --x)
        {
            a = img->Bitdepth16To8[wp[3]];
            m = img->UaToAa + ((size_t)a << 8);
            r = m[img->Bitdepth16To8[wp[0]]];
            g = m[img->Bitdepth16To8[wp[1]]];
            b = m[img->Bitdepth16To8[wp[2]]];
            *cp++ = PACK4(r, g, b, a);
            wp += samplesperpixel;
        }
        cp += toskew;
        wp += fromskew;
    }
}

/*
 * 8-bit packed CMYK samples w/o Map => RGB
 *
 * NB: The conversion of CMYK->RGB is *very* crude.
 */
DECLAREContigPutFunc(putRGBcontig8bitCMYKtile)
{
    int samplesperpixel = img->samplesperpixel;
    uint16_t r, g, b, k;

    (void)x;
    (void)y;
    fromskew *= samplesperpixel;
    for (; h > 0; --h)
    {
        UNROLL8(w, NOP, k = 255 - pp[3]; r = (k * (255 - pp[0])) / 255;
                g = (k * (255 - pp[1])) / 255; b = (k * (255 - pp[2])) / 255;
                *cp++ = PACK(r, g, b); pp += samplesperpixel);
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 8-bit packed CMYK samples w/Map => RGB
 *
 * NB: The conversion of CMYK->RGB is *very* crude.
 */
DECLAREContigPutFunc(putRGBcontig8bitCMYKMaptile)
{
    int samplesperpixel = img->samplesperpixel;
    TIFFRGBValue *Map = img->Map;
    uint16_t r, g, b, k;

    (void)y;
    fromskew *= samplesperpixel;
    for (; h > 0; --h)
    {
        for (x = w; x > 0; --x)
        {
            k = 255 - pp[3];
            r = (k * (255 - pp[0])) / 255;
            g = (k * (255 - pp[1])) / 255;
            b = (k * (255 - pp[2])) / 255;
            *cp++ = PACK(Map[r], Map[g], Map[b]);
            pp += samplesperpixel;
        }
        pp += fromskew;
        cp += toskew;
    }
}

#define DECLARESepPutFunc(name)                                                \
    static void name(TIFFRGBAImage *img, uint32_t *cp, uint32_t x, uint32_t y, \
                     uint32_t w, uint32_t h, int32_t fromskew, int32_t toskew, \
                     unsigned char *r, unsigned char *g, unsigned char *b,     \
                     unsigned char *a)

/*
 * 8-bit unpacked samples => RGB
 */
DECLARESepPutFunc(putRGBseparate8bittile)
{
    (void)img;
    (void)x;
    (void)y;
    (void)a;
    for (; h > 0; --h)
    {
        UNROLL8(w, NOP, *cp++ = PACK(*r++, *g++, *b++));
        SKEW(r, g, b, fromskew);
        cp += toskew;
    }
}

/*
 * 8-bit unpacked samples => RGBA w/ associated alpha
 */
DECLARESepPutFunc(putRGBAAseparate8bittile)
{
    (void)img;
    (void)x;
    (void)y;
    for (; h > 0; --h)
    {
        UNROLL8(w, NOP, *cp++ = PACK4(*r++, *g++, *b++, *a++));
        SKEW4(r, g, b, a, fromskew);
        cp += toskew;
    }
}

/*
 * 8-bit unpacked CMYK samples => RGBA
 */
DECLARESepPutFunc(putCMYKseparate8bittile)
{
    (void)img;
    (void)y;
    for (; h > 0; --h)
    {
        uint32_t rv, gv, bv, kv;
        for (x = w; x > 0; --x)
        {
            kv = 255 - *a++;
            rv = (kv * (255 - *r++)) / 255;
            gv = (kv * (255 - *g++)) / 255;
            bv = (kv * (255 - *b++)) / 255;
            *cp++ = PACK4(rv, gv, bv, 255);
        }
        SKEW4(r, g, b, a, fromskew);
        cp += toskew;
    }
}

/*
 * 8-bit unpacked samples => RGBA w/ unassociated alpha
 */
DECLARESepPutFunc(putRGBUAseparate8bittile)
{
    (void)img;
    (void)y;
    for (; h > 0; --h)
    {
        uint32_t rv, gv, bv, av;
        uint8_t *m;
        for (x = w; x > 0; --x)
        {
            av = *a++;
            m = img->UaToAa + ((size_t)av << 8);
            rv = m[*r++];
            gv = m[*g++];
            bv = m[*b++];
            *cp++ = PACK4(rv, gv, bv, av);
        }
        SKEW4(r, g, b, a, fromskew);
        cp += toskew;
    }
}

/*
 * 16-bit unpacked samples => RGB
 */
DECLARESepPutFunc(putRGBseparate16bittile)
{
    uint16_t *wr = (uint16_t *)r;
    uint16_t *wg = (uint16_t *)g;
    uint16_t *wb = (uint16_t *)b;
    (void)img;
    (void)y;
    (void)a;
    for (; h > 0; --h)
    {
        for (x = 0; x < w; x++)
            *cp++ = PACK(img->Bitdepth16To8[*wr++], img->Bitdepth16To8[*wg++],
                         img->Bitdepth16To8[*wb++]);
        SKEW(wr, wg, wb, fromskew);
        cp += toskew;
    }
}

/*
 * 16-bit unpacked samples => RGBA w/ associated alpha
 */
DECLARESepPutFunc(putRGBAAseparate16bittile)
{
    uint16_t *wr = (uint16_t *)r;
    uint16_t *wg = (uint16_t *)g;
    uint16_t *wb = (uint16_t *)b;
    uint16_t *wa = (uint16_t *)a;
    (void)img;
    (void)y;
    for (; h > 0; --h)
    {
        for (x = 0; x < w; x++)
            *cp++ = PACK4(img->Bitdepth16To8[*wr++], img->Bitdepth16To8[*wg++],
                          img->Bitdepth16To8[*wb++], img->Bitdepth16To8[*wa++]);
        SKEW4(wr, wg, wb, wa, fromskew);
        cp += toskew;
    }
}

/*
 * 16-bit unpacked samples => RGBA w/ unassociated alpha
 */
DECLARESepPutFunc(putRGBUAseparate16bittile)
{
    uint16_t *wr = (uint16_t *)r;
    uint16_t *wg = (uint16_t *)g;
    uint16_t *wb = (uint16_t *)b;
    uint16_t *wa = (uint16_t *)a;
    (void)img;
    (void)y;
    for (; h > 0; --h)
    {
        uint32_t r2, g2, b2, a2;
        uint8_t *m;
        for (x = w; x > 0; --x)
        {
            a2 = img->Bitdepth16To8[*wa++];
            m = img->UaToAa + ((size_t)a2 << 8);
            r2 = m[img->Bitdepth16To8[*wr++]];
            g2 = m[img->Bitdepth16To8[*wg++]];
            b2 = m[img->Bitdepth16To8[*wb++]];
            *cp++ = PACK4(r2, g2, b2, a2);
        }
        SKEW4(wr, wg, wb, wa, fromskew);
        cp += toskew;
    }
}

/*
 * 8-bit packed CIE L*a*b 1976 samples => RGB
 */
DECLAREContigPutFunc(putcontig8bitCIELab8)
{
    float X, Y, Z;
    uint32_t r, g, b;
    (void)y;
    fromskew *= 3;
    for (; h > 0; --h)
    {
        for (x = w; x > 0; --x)
        {
            TIFFCIELabToXYZ(img->cielab, (unsigned char)pp[0],
                            (signed char)pp[1], (signed char)pp[2], &X, &Y, &Z);
            TIFFXYZToRGB(img->cielab, X, Y, Z, &r, &g, &b);
            *cp++ = PACK(r, g, b);
            pp += 3;
        }
        cp += toskew;
        pp += fromskew;
    }
}

/*
 * 16-bit packed CIE L*a*b 1976 samples => RGB
 */
DECLAREContigPutFunc(putcontig8bitCIELab16)
{
    float X, Y, Z;
    uint32_t r, g, b;
    uint16_t *wp = (uint16_t *)pp;
    (void)y;
    fromskew *= 3;
    for (; h > 0; --h)
    {
        for (x = w; x > 0; --x)
        {
            TIFFCIELab16ToXYZ(img->cielab, (uint16_t)wp[0], (int16_t)wp[1],
                              (int16_t)wp[2], &X, &Y, &Z);
            TIFFXYZToRGB(img->cielab, X, Y, Z, &r, &g, &b);
            *cp++ = PACK(r, g, b);
            wp += 3;
        }
        cp += toskew;
        wp += fromskew;
    }
}

/*
 * YCbCr -> RGB conversion and packing routines.
 */

#define YCbCrtoRGB(dst, Y)                                                     \
    {                                                                          \
        uint32_t r, g, b;                                                      \
        TIFFYCbCrtoRGB(img->ycbcr, (Y), Cb, Cr, &r, &g, &b);                   \
        dst = PACK(r, g, b);                                                   \
    }

/*
 * 8-bit packed YCbCr samples w/ 4,4 subsampling => RGB
 */
DECLAREContigPutFunc(putcontig8bitYCbCr44tile)
{
    uint32_t *cp1 = cp + w + toskew;
    uint32_t *cp2 = cp1 + w + toskew;
    uint32_t *cp3 = cp2 + w + toskew;
    int32_t incr = 3 * w + 4 * toskew;

    (void)y;
    /* adjust fromskew */
    fromskew = (fromskew / 4) * (4 * 2 + 2);
    if ((h & 3) == 0 && (w & 3) == 0)
    {
        for (; h >= 4; h -= 4)
        {
            x = w >> 2;
            do
            {
                int32_t Cb = pp[16];
                int32_t Cr = pp[17];

                YCbCrtoRGB(cp[0], pp[0]);
                YCbCrtoRGB(cp[1], pp[1]);
                YCbCrtoRGB(cp[2], pp[2]);
                YCbCrtoRGB(cp[3], pp[3]);
                YCbCrtoRGB(cp1[0], pp[4]);
                YCbCrtoRGB(cp1[1], pp[5]);
                YCbCrtoRGB(cp1[2], pp[6]);
                YCbCrtoRGB(cp1[3], pp[7]);
                YCbCrtoRGB(cp2[0], pp[8]);
                YCbCrtoRGB(cp2[1], pp[9]);
                YCbCrtoRGB(cp2[2], pp[10]);
                YCbCrtoRGB(cp2[3], pp[11]);
                YCbCrtoRGB(cp3[0], pp[12]);
                YCbCrtoRGB(cp3[1], pp[13]);
                YCbCrtoRGB(cp3[2], pp[14]);
                YCbCrtoRGB(cp3[3], pp[15]);

                cp += 4;
                cp1 += 4;
                cp2 += 4;
                cp3 += 4;
                pp += 18;
            } while (--x);
            cp += incr;
            cp1 += incr;
            cp2 += incr;
            cp3 += incr;
            pp += fromskew;
        }
    }
    else
    {
        while (h > 0)
        {
            for (x = w; x > 0;)
            {
                int32_t Cb = pp[16];
                int32_t Cr = pp[17];
                switch (x)
                {
                    default:
                        switch (h)
                        {
                            default:
                                YCbCrtoRGB(cp3[3], pp[15]); /* FALLTHROUGH */
                            case 3:
                                YCbCrtoRGB(cp2[3], pp[11]); /* FALLTHROUGH */
                            case 2:
                                YCbCrtoRGB(cp1[3], pp[7]); /* FALLTHROUGH */
                            case 1:
                                YCbCrtoRGB(cp[3], pp[3]); /* FALLTHROUGH */
                        }                                 /* FALLTHROUGH */
                    case 3:
                        switch (h)
                        {
                            default:
                                YCbCrtoRGB(cp3[2], pp[14]); /* FALLTHROUGH */
                            case 3:
                                YCbCrtoRGB(cp2[2], pp[10]); /* FALLTHROUGH */
                            case 2:
                                YCbCrtoRGB(cp1[2], pp[6]); /* FALLTHROUGH */
                            case 1:
                                YCbCrtoRGB(cp[2], pp[2]); /* FALLTHROUGH */
                        }                                 /* FALLTHROUGH */
                    case 2:
                        switch (h)
                        {
                            default:
                                YCbCrtoRGB(cp3[1], pp[13]); /* FALLTHROUGH */
                            case 3:
                                YCbCrtoRGB(cp2[1], pp[9]); /* FALLTHROUGH */
                            case 2:
                                YCbCrtoRGB(cp1[1], pp[5]); /* FALLTHROUGH */
                            case 1:
                                YCbCrtoRGB(cp[1], pp[1]); /* FALLTHROUGH */
                        }                                 /* FALLTHROUGH */
                    case 1:
                        switch (h)
                        {
                            default:
                                YCbCrtoRGB(cp3[0], pp[12]); /* FALLTHROUGH */
                            case 3:
                                YCbCrtoRGB(cp2[0], pp[8]); /* FALLTHROUGH */
                            case 2:
                                YCbCrtoRGB(cp1[0], pp[4]); /* FALLTHROUGH */
                            case 1:
                                YCbCrtoRGB(cp[0], pp[0]); /* FALLTHROUGH */
                        }                                 /* FALLTHROUGH */
                }
                if (x < 4)
                {
                    cp += x;
                    cp1 += x;
                    cp2 += x;
                    cp3 += x;
                    x = 0;
                }
                else
                {
                    cp += 4;
                    cp1 += 4;
                    cp2 += 4;
                    cp3 += 4;
                    x -= 4;
                }
                pp += 18;
            }
            if (h <= 4)
                break;
            h -= 4;
            cp += incr;
            cp1 += incr;
            cp2 += incr;
            cp3 += incr;
            pp += fromskew;
        }
    }
}

/*
 * 8-bit packed YCbCr samples w/ 4,2 subsampling => RGB
 */
DECLAREContigPutFunc(putcontig8bitYCbCr42tile)
{
    uint32_t *cp1 = cp + w + toskew;
    int32_t incr = 2 * toskew + w;

    (void)y;
    fromskew = (fromskew / 4) * (4 * 2 + 2);
    if ((w & 3) == 0 && (h & 1) == 0)
    {
        for (; h >= 2; h -= 2)
        {
            x = w >> 2;
            do
            {
                int32_t Cb = pp[8];
                int32_t Cr = pp[9];

                YCbCrtoRGB(cp[0], pp[0]);
                YCbCrtoRGB(cp[1], pp[1]);
                YCbCrtoRGB(cp[2], pp[2]);
                YCbCrtoRGB(cp[3], pp[3]);
                YCbCrtoRGB(cp1[0], pp[4]);
                YCbCrtoRGB(cp1[1], pp[5]);
                YCbCrtoRGB(cp1[2], pp[6]);
                YCbCrtoRGB(cp1[3], pp[7]);

                cp += 4;
                cp1 += 4;
                pp += 10;
            } while (--x);
            cp += incr;
            cp1 += incr;
            pp += fromskew;
        }
    }
    else
    {
        while (h > 0)
        {
            for (x = w; x > 0;)
            {
                int32_t Cb = pp[8];
                int32_t Cr = pp[9];
                switch (x)
                {
                    default:
                        switch (h)
                        {
                            default:
                                YCbCrtoRGB(cp1[3], pp[7]); /* FALLTHROUGH */
                            case 1:
                                YCbCrtoRGB(cp[3], pp[3]); /* FALLTHROUGH */
                        }                                 /* FALLTHROUGH */
                    case 3:
                        switch (h)
                        {
                            default:
                                YCbCrtoRGB(cp1[2], pp[6]); /* FALLTHROUGH */
                            case 1:
                                YCbCrtoRGB(cp[2], pp[2]); /* FALLTHROUGH */
                        }                                 /* FALLTHROUGH */
                    case 2:
                        switch (h)
                        {
                            default:
                                YCbCrtoRGB(cp1[1], pp[5]); /* FALLTHROUGH */
                            case 1:
                                YCbCrtoRGB(cp[1], pp[1]); /* FALLTHROUGH */
                        }                                 /* FALLTHROUGH */
                    case 1:
                        switch (h)
                        {
                            default:
                                YCbCrtoRGB(cp1[0], pp[4]); /* FALLTHROUGH */
                            case 1:
                                YCbCrtoRGB(cp[0], pp[0]); /* FALLTHROUGH */
                        }                                 /* FALLTHROUGH */
                }
                if (x < 4)
                {
                    cp += x;
                    cp1 += x;
                    x = 0;
                }
                else
                {
                    cp += 4;
                    cp1 += 4;
                    x -= 4;
                }
                pp += 10;
            }
            if (h <= 2)
                break;
            h -= 2;
            cp += incr;
            cp1 += incr;
            pp += fromskew;
        }
    }
}

/*
 * 8-bit packed YCbCr samples w/ 4,1 subsampling => RGB
 */
DECLAREContigPutFunc(putcontig8bitYCbCr41tile)
{
    (void)y;
    fromskew = (fromskew / 4) * (4 * 1 + 2);
    do
    {
        x = w >> 2;
        while (x > 0)
        {
            int32_t Cb = pp[4];
            int32_t Cr = pp[5];

            YCbCrtoRGB(cp[0], pp[0]);
            YCbCrtoRGB(cp[1], pp[1]);
            YCbCrtoRGB(cp[2], pp[2]);
            YCbCrtoRGB(cp[3], pp[3]);

            cp += 4;
            pp += 6;
            x--;
        }

        if ((w & 3) != 0)
        {
            int32_t Cb = pp[4];
            int32_t Cr = pp[5];

            switch ((w & 3))
            {
                case 3:
                    YCbCrtoRGB(cp[2], pp[2]); /*-fallthrough*/
                case 2:
                    YCbCrtoRGB(cp[1], pp[1]); /*-fallthrough*/
                case 1:
                    YCbCrtoRGB(cp[0], pp[0]); /*-fallthrough*/
                case 0:
                    break;
            }

            cp += (w & 3);
            pp += 6;
        }

        cp += toskew;
        pp += fromskew;
    } while (--h);
}

/*
 * 8-bit packed YCbCr samples w/ 2,2 subsampling => RGB
 */
DECLAREContigPutFunc(putcontig8bitYCbCr22tile)
{
    uint32_t *cp2;
    int32_t incr = 2 * toskew + w;
    (void)y;
    fromskew = (fromskew / 2) * (2 * 2 + 2);
    cp2 = cp + w + toskew;
    while (h >= 2)
    {
        x = w;
        while (x >= 2)
        {
            uint32_t Cb = pp[4];
            uint32_t Cr = pp[5];
            YCbCrtoRGB(cp[0], pp[0]);
            YCbCrtoRGB(cp[1], pp[1]);
            YCbCrtoRGB(cp2[0], pp[2]);
            YCbCrtoRGB(cp2[1], pp[3]);
            cp += 2;
            cp2 += 2;
            pp += 6;
            x -= 2;
        }
        if (x == 1)
        {
            uint32_t Cb = pp[4];
            uint32_t Cr = pp[5];
            YCbCrtoRGB(cp[0], pp[0]);
            YCbCrtoRGB(cp2[0], pp[2]);
            cp++;
            cp2++;
            pp += 6;
        }
        cp += incr;
        cp2 += incr;
        pp += fromskew;
        h -= 2;
    }
    if (h == 1)
    {
        x = w;
        while (x >= 2)
        {
            uint32_t Cb = pp[4];
            uint32_t Cr = pp[5];
            YCbCrtoRGB(cp[0], pp[0]);
            YCbCrtoRGB(cp[1], pp[1]);
            cp += 2;
            cp2 += 2;
            pp += 6;
            x -= 2;
        }
        if (x == 1)
        {
            uint32_t Cb = pp[4];
            uint32_t Cr = pp[5];
            YCbCrtoRGB(cp[0], pp[0]);
        }
    }
}

/*
 * 8-bit packed YCbCr samples w/ 2,1 subsampling => RGB
 */
DECLAREContigPutFunc(putcontig8bitYCbCr21tile)
{
    (void)y;
    fromskew = (fromskew / 2) * (2 * 1 + 2);
    do
    {
        x = w >> 1;
        while (x > 0)
        {
            int32_t Cb = pp[2];
            int32_t Cr = pp[3];

            YCbCrtoRGB(cp[0], pp[0]);
            YCbCrtoRGB(cp[1], pp[1]);

            cp += 2;
            pp += 4;
            x--;
        }

        if ((w & 1) != 0)
        {
            int32_t Cb = pp[2];
            int32_t Cr = pp[3];

            YCbCrtoRGB(cp[0], pp[0]);

            cp += 1;
            pp += 4;
        }

        cp += toskew;
        pp += fromskew;
    } while (--h);
}

/*
 * 8-bit packed YCbCr samples w/ 1,2 subsampling => RGB
 */
DECLAREContigPutFunc(putcontig8bitYCbCr12tile)
{
    uint32_t *cp2;
    int32_t incr = 2 * toskew + w;
    (void)y;
    fromskew = (fromskew / 1) * (1 * 2 + 2);
    cp2 = cp + w + toskew;
    while (h >= 2)
    {
        x = w;
        do
        {
            uint32_t Cb = pp[2];
            uint32_t Cr = pp[3];
            YCbCrtoRGB(cp[0], pp[0]);
            YCbCrtoRGB(cp2[0], pp[1]);
            cp++;
            cp2++;
            pp += 4;
        } while (--x);
        cp += incr;
        cp2 += incr;
        pp += fromskew;
        h -= 2;
    }
    if (h == 1)
    {
        x = w;
        do
        {
            uint32_t Cb = pp[2];
            uint32_t Cr = pp[3];
            YCbCrtoRGB(cp[0], pp[0]);
            cp++;
            pp += 4;
        } while (--x);
    }
}

/*
 * 8-bit packed YCbCr samples w/ no subsampling => RGB
 */
DECLAREContigPutFunc(putcontig8bitYCbCr11tile)
{
    (void)y;
    fromskew = (fromskew / 1) * (1 * 1 + 2);
    do
    {
        x = w; /* was x = w>>1; patched 2000/09/25 warmerda@home.com */
        do
        {
            int32_t Cb = pp[1];
            int32_t Cr = pp[2];

            YCbCrtoRGB(*cp++, pp[0]);

            pp += 3;
        } while (--x);
        cp += toskew;
        pp += fromskew;
    } while (--h);
}

/*
 * 8-bit packed YCbCr samples w/ no subsampling => RGB
 */
DECLARESepPutFunc(putseparate8bitYCbCr11tile)
{
    (void)y;
    (void)a;
    /* TODO: naming of input vars is still off, change obfuscating declaration
     * inside define, or resolve obfuscation */
    for (; h > 0; --h)
    {
        x = w;
        do
        {
            uint32_t dr, dg, db;
            TIFFYCbCrtoRGB(img->ycbcr, *r++, *g++, *b++, &dr, &dg, &db);
            *cp++ = PACK(dr, dg, db);
        } while (--x);
        SKEW(r, g, b, fromskew);
        cp += toskew;
    }
}
#undef YCbCrtoRGB

static int isInRefBlackWhiteRange(float f)
{
    return f > (float)(-0x7FFFFFFF + 128) && f < (float)0x7FFFFFFF;
}

static int initYCbCrConversion(TIFFRGBAImage *img)
{
    static const char module[] = "initYCbCrConversion";

    float *luma, *refBlackWhite;

    if (img->ycbcr == NULL)
    {
        img->ycbcr = (TIFFYCbCrToRGB *)_TIFFmallocExt(
            img->tif, TIFFroundup_32(sizeof(TIFFYCbCrToRGB), sizeof(long)) +
                          4 * 256 * sizeof(TIFFRGBValue) +
                          2 * 256 * sizeof(int) + 3 * 256 * sizeof(int32_t));
        if (img->ycbcr == NULL)
        {
            TIFFErrorExtR(img->tif, module,
                          "No space for YCbCr->RGB conversion state");
            return (0);
        }
    }

    TIFFGetFieldDefaulted(img->tif, TIFFTAG_YCBCRCOEFFICIENTS, &luma);
    TIFFGetFieldDefaulted(img->tif, TIFFTAG_REFERENCEBLACKWHITE,
                          &refBlackWhite);

    /* Do some validation to avoid later issues. Detect NaN for now */
    /* and also if lumaGreen is zero since we divide by it later */
    if (luma[0] != luma[0] || luma[1] != luma[1] || luma[1] == 0.0 ||
        luma[2] != luma[2])
    {
        TIFFErrorExtR(img->tif, module,
                      "Invalid values for YCbCrCoefficients tag");
        return (0);
    }

    if (!isInRefBlackWhiteRange(refBlackWhite[0]) ||
        !isInRefBlackWhiteRange(refBlackWhite[1]) ||
        !isInRefBlackWhiteRange(refBlackWhite[2]) ||
        !isInRefBlackWhiteRange(refBlackWhite[3]) ||
        !isInRefBlackWhiteRange(refBlackWhite[4]) ||
        !isInRefBlackWhiteRange(refBlackWhite[5]))
    {
        TIFFErrorExtR(img->tif, module,
                      "Invalid values for ReferenceBlackWhite tag");
        return (0);
    }

    if (TIFFYCbCrToRGBInit(img->ycbcr, luma, refBlackWhite) < 0)
        return (0);
    return (1);
}

static tileContigRoutine initCIELabConversion(TIFFRGBAImage *img)
{
    static const char module[] = "initCIELabConversion";

    float *whitePoint;
    float refWhite[3];

    TIFFGetFieldDefaulted(img->tif, TIFFTAG_WHITEPOINT, &whitePoint);
    if (whitePoint[1] == 0.0f)
    {
        TIFFErrorExtR(img->tif, module, "Invalid value for WhitePoint tag.");
        return NULL;
    }

    if (!img->cielab)
    {
        img->cielab = (TIFFCIELabToRGB *)_TIFFmallocExt(
            img->tif, sizeof(TIFFCIELabToRGB));
        if (!img->cielab)
        {
            TIFFErrorExtR(img->tif, module,
                          "No space for CIE L*a*b*->RGB conversion state.");
            return NULL;
        }
    }

    refWhite[1] = 100.0F;
    refWhite[0] = whitePoint[0] / whitePoint[1] * refWhite[1];
    refWhite[2] =
        (1.0F - whitePoint[0] - whitePoint[1]) / whitePoint[1] * refWhite[1];
    if (TIFFCIELabToRGBInit(img->cielab, &display_sRGB, refWhite) < 0)
    {
        TIFFErrorExtR(img->tif, module,
                      "Failed to initialize CIE L*a*b*->RGB conversion state.");
        _TIFFfreeExt(img->tif, img->cielab);
        return NULL;
    }

    if (img->bitspersample == 8)
        return putcontig8bitCIELab8;
    else if (img->bitspersample == 16)
        return putcontig8bitCIELab16;
    return NULL;
}

/*
 * Greyscale images with less than 8 bits/sample are handled
 * with a table to avoid lots of shifts and masks.  The table
 * is setup so that put*bwtile (below) can retrieve 8/bitspersample
 * pixel values simply by indexing into the table with one
 * number.
 */
static int makebwmap(TIFFRGBAImage *img)
{
    TIFFRGBValue *Map = img->Map;
    int bitspersample = img->bitspersample;
    int nsamples = 8 / bitspersample;
    int i;
    uint32_t *p;

    if (nsamples == 0)
        nsamples = 1;

    img->BWmap = (uint32_t **)_TIFFmallocExt(
        img->tif,
        256 * sizeof(uint32_t *) + (256 * nsamples * sizeof(uint32_t)));
    if (img->BWmap == NULL)
    {
        TIFFErrorExtR(img->tif, TIFFFileName(img->tif),
                      "No space for B&W mapping table");
        return (0);
    }
    p = (uint32_t *)(img->BWmap + 256);
    for (i = 0; i < 256; i++)
    {
        TIFFRGBValue c;
        img->BWmap[i] = p;
        switch (bitspersample)
        {
#define GREY(x)                                                                \
    c = Map[x];                                                                \
    *p++ = PACK(c, c, c);
            case 1:
                GREY(i >> 7);
                GREY((i >> 6) & 1);
                GREY((i >> 5) & 1);
                GREY((i >> 4) & 1);
                GREY((i >> 3) & 1);
                GREY((i >> 2) & 1);
                GREY((i >> 1) & 1);
                GREY(i & 1);
                break;
            case 2:
                GREY(i >> 6);
                GREY((i >> 4) & 3);
                GREY((i >> 2) & 3);
                GREY(i & 3);
                break;
            case 4:
                GREY(i >> 4);
                GREY(i & 0xf);
                break;
            case 8:
            case 16:
                GREY(i);
                break;
        }
#undef GREY
    }
    return (1);
}

/*
 * Construct a mapping table to convert from the range
 * of the data samples to [0,255] --for display.  This
 * process also handles inverting B&W images when needed.
 */
static int setupMap(TIFFRGBAImage *img)
{
    int32_t x, range;

    range = (int32_t)((1L << img->bitspersample) - 1);

    /* treat 16 bit the same as eight bit */
    if (img->bitspersample == 16)
        range = (int32_t)255;

    img->Map = (TIFFRGBValue *)_TIFFmallocExt(
        img->tif, (range + 1) * sizeof(TIFFRGBValue));
    if (img->Map == NULL)
    {
        TIFFErrorExtR(img->tif, TIFFFileName(img->tif),
                      "No space for photometric conversion table");
        return (0);
    }
    if (img->photometric == PHOTOMETRIC_MINISWHITE)
    {
        for (x = 0; x <= range; x++)
            img->Map[x] = (TIFFRGBValue)(((range - x) * 255) / range);
    }
    else
    {
        for (x = 0; x <= range; x++)
            img->Map[x] = (TIFFRGBValue)((x * 255) / range);
    }
    if (img->bitspersample <= 16 &&
        (img->photometric == PHOTOMETRIC_MINISBLACK ||
         img->photometric == PHOTOMETRIC_MINISWHITE))
    {
        /*
         * Use photometric mapping table to construct
         * unpacking tables for samples <= 8 bits.
         */
        if (!makebwmap(img))
            return (0);
        /* no longer need Map, free it */
        _TIFFfreeExt(img->tif, img->Map);
        img->Map = NULL;
    }
    return (1);
}

static int checkcmap(TIFFRGBAImage *img)
{
    uint16_t *r = img->redcmap;
    uint16_t *g = img->greencmap;
    uint16_t *b = img->bluecmap;
    long n = 1L << img->bitspersample;

    while (n-- > 0)
        if (*r++ >= 256 || *g++ >= 256 || *b++ >= 256)
            return (16);
    return (8);
}

static void cvtcmap(TIFFRGBAImage *img)
{
    uint16_t *r = img->redcmap;
    uint16_t *g = img->greencmap;
    uint16_t *b = img->bluecmap;
    long i;

    for (i = (1L << img->bitspersample) - 1; i >= 0; i--)
    {
#define CVT(x) ((uint16_t)((x) >> 8))
        r[i] = CVT(r[i]);
        g[i] = CVT(g[i]);
        b[i] = CVT(b[i]);
#undef CVT
    }
}

/*
 * Palette images with <= 8 bits/sample are handled
 * with a table to avoid lots of shifts and masks.  The table
 * is setup so that put*cmaptile (below) can retrieve 8/bitspersample
 * pixel values simply by indexing into the table with one
 * number.
 */
static int makecmap(TIFFRGBAImage *img)
{
    int bitspersample = img->bitspersample;
    int nsamples = 8 / bitspersample;
    uint16_t *r = img->redcmap;
    uint16_t *g = img->greencmap;
    uint16_t *b = img->bluecmap;
    uint32_t *p;
    int i;

    img->PALmap = (uint32_t **)_TIFFmallocExt(
        img->tif,
        256 * sizeof(uint32_t *) + (256 * nsamples * sizeof(uint32_t)));
    if (img->PALmap == NULL)
    {
        TIFFErrorExtR(img->tif, TIFFFileName(img->tif),
                      "No space for Palette mapping table");
        return (0);
    }
    p = (uint32_t *)(img->PALmap + 256);
    for (i = 0; i < 256; i++)
    {
        TIFFRGBValue c;
        img->PALmap[i] = p;
#define CMAP(x)                                                                \
    c = (TIFFRGBValue)x;                                                       \
    *p++ = PACK(r[c] & 0xff, g[c] & 0xff, b[c] & 0xff);
        switch (bitspersample)
        {
            case 1:
                CMAP(i >> 7);
                CMAP((i >> 6) & 1);
                CMAP((i >> 5) & 1);
                CMAP((i >> 4) & 1);
                CMAP((i >> 3) & 1);
                CMAP((i >> 2) & 1);
                CMAP((i >> 1) & 1);
                CMAP(i & 1);
                break;
            case 2:
                CMAP(i >> 6);
                CMAP((i >> 4) & 3);
                CMAP((i >> 2) & 3);
                CMAP(i & 3);
                break;
            case 4:
                CMAP(i >> 4);
                CMAP(i & 0xf);
                break;
            case 8:
                CMAP(i);
                break;
        }
#undef CMAP
    }
    return (1);
}

/*
 * Construct any mapping table used
 * by the associated put routine.
 */
static int buildMap(TIFFRGBAImage *img)
{
    switch (img->photometric)
    {
        case PHOTOMETRIC_RGB:
        case PHOTOMETRIC_YCBCR:
        case PHOTOMETRIC_SEPARATED:
            if (img->bitspersample == 8)
                break;
            /* fall through... */
        case PHOTOMETRIC_MINISBLACK:
        case PHOTOMETRIC_MINISWHITE:
            if (!setupMap(img))
                return (0);
            break;
        case PHOTOMETRIC_PALETTE:
            /*
             * Convert 16-bit colormap to 8-bit (unless it looks
             * like an old-style 8-bit colormap).
             */
            if (checkcmap(img) == 16)
                cvtcmap(img);
            else
                TIFFWarningExtR(img->tif, TIFFFileName(img->tif),
                                "Assuming 8-bit colormap");
            /*
             * Use mapping table and colormap to construct
             * unpacking tables for samples < 8 bits.
             */
            if (img->bitspersample <= 8 && !makecmap(img))
                return (0);
            break;
    }
    return (1);
}

/*
 * Select the appropriate conversion routine for packed data.
 */
static int PickContigCase(TIFFRGBAImage *img)
{
    img->get = TIFFIsTiled(img->tif) ? gtTileContig : gtStripContig;
    img->put.contig = NULL;
    switch (img->photometric)
    {
        case PHOTOMETRIC_RGB:
            switch (img->bitspersample)
            {
                case 8:
                    if (img->alpha == EXTRASAMPLE_ASSOCALPHA &&
                        img->samplesperpixel >= 4)
                        img->put.contig = putRGBAAcontig8bittile;
                    else if (img->alpha == EXTRASAMPLE_UNASSALPHA &&
                             img->samplesperpixel >= 4)
                    {
                        if (BuildMapUaToAa(img))
                            img->put.contig = putRGBUAcontig8bittile;
                    }
                    else if (img->samplesperpixel >= 3)
                        img->put.contig = putRGBcontig8bittile;
                    break;
                case 16:
                    if (img->alpha == EXTRASAMPLE_ASSOCALPHA &&
                        img->samplesperpixel >= 4)
                    {
                        if (BuildMapBitdepth16To8(img))
                            img->put.contig = putRGBAAcontig16bittile;
                    }
                    else if (img->alpha == EXTRASAMPLE_UNASSALPHA &&
                             img->samplesperpixel >= 4)
                    {
                        if (BuildMapBitdepth16To8(img) && BuildMapUaToAa(img))
                            img->put.contig = putRGBUAcontig16bittile;
                    }
                    else if (img->samplesperpixel >= 3)
                    {
                        if (BuildMapBitdepth16To8(img))
                            img->put.contig = putRGBcontig16bittile;
                    }
                    break;
            }
            break;
        case PHOTOMETRIC_SEPARATED:
            if (img->samplesperpixel >= 4 && buildMap(img))
            {
                if (img->bitspersample == 8)
                {
                    if (!img->Map)
                        img->put.contig = putRGBcontig8bitCMYKtile;
                    else
                        img->put.contig = putRGBcontig8bitCMYKMaptile;
                }
            }
            break;
        case PHOTOMETRIC_PALETTE:
            if (buildMap(img))
            {
                switch (img->bitspersample)
                {
                    case 8:
                        img->put.contig = put8bitcmaptile;
                        break;
                    case 4:
                        img->put.contig = put4bitcmaptile;
                        break;
                    case 2:
                        img->put.contig = put2bitcmaptile;
                        break;
                    case 1:
                        img->put.contig = put1bitcmaptile;
                        break;
                }
            }
            break;
        case PHOTOMETRIC_MINISWHITE:
        case PHOTOMETRIC_MINISBLACK:
            if (buildMap(img))
            {
                switch (img->bitspersample)
                {
                    case 16:
                        img->put.contig = put16bitbwtile;
                        break;
                    case 8:
                        if (img->alpha && img->samplesperpixel == 2)
                            img->put.contig = putagreytile;
                        else
                            img->put.contig = putgreytile;
                        break;
                    case 4:
                        img->put.contig = put4bitbwtile;
                        break;
                    case 2:
                        img->put.contig = put2bitbwtile;
                        break;
                    case 1:
                        img->put.contig = put1bitbwtile;
                        break;
                }
            }
            break;
        case PHOTOMETRIC_YCBCR:
            if ((img->bitspersample == 8) && (img->samplesperpixel == 3))
            {
                if (initYCbCrConversion(img) != 0)
                {
                    /*
                     * The 6.0 spec says that subsampling must be
                     * one of 1, 2, or 4, and that vertical subsampling
                     * must always be <= horizontal subsampling; so
                     * there are only a few possibilities and we just
                     * enumerate the cases.
                     * Joris: added support for the [1,2] case, nonetheless, to
                     * accommodate some OJPEG files
                     */
                    uint16_t SubsamplingHor;
                    uint16_t SubsamplingVer;
                    TIFFGetFieldDefaulted(img->tif, TIFFTAG_YCBCRSUBSAMPLING,
                                          &SubsamplingHor, &SubsamplingVer);
                    switch ((SubsamplingHor << 4) | SubsamplingVer)
                    {
                        case 0x44:
                            img->put.contig = putcontig8bitYCbCr44tile;
                            break;
                        case 0x42:
                            img->put.contig = putcontig8bitYCbCr42tile;
                            break;
                        case 0x41:
                            img->put.contig = putcontig8bitYCbCr41tile;
                            break;
                        case 0x22:
                            img->put.contig = putcontig8bitYCbCr22tile;
                            break;
                        case 0x21:
                            img->put.contig = putcontig8bitYCbCr21tile;
                            break;
                        case 0x12:
                            img->put.contig = putcontig8bitYCbCr12tile;
                            break;
                        case 0x11:
                            img->put.contig = putcontig8bitYCbCr11tile;
                            break;
                    }
                }
            }
            break;
        case PHOTOMETRIC_CIELAB:
            if (img->samplesperpixel == 3 && buildMap(img))
            {
                if (img->bitspersample == 8 || img->bitspersample == 16)
                    img->put.contig = initCIELabConversion(img);
                break;
            }
    }
    return ((img->get != NULL) && (img->put.contig != NULL));
}

/*
 * Select the appropriate conversion routine for unpacked data.
 *
 * NB: we assume that unpacked single channel data is directed
 *	 to the "packed routines.
 */
static int PickSeparateCase(TIFFRGBAImage *img)
{
    img->get = TIFFIsTiled(img->tif) ? gtTileSeparate : gtStripSeparate;
    img->put.separate = NULL;
    switch (img->photometric)
    {
        case PHOTOMETRIC_MINISWHITE:
        case PHOTOMETRIC_MINISBLACK:
            /* greyscale images processed pretty much as RGB by gtTileSeparate
             */
        case PHOTOMETRIC_RGB:
            switch (img->bitspersample)
            {
                case 8:
                    if (img->alpha == EXTRASAMPLE_ASSOCALPHA)
                        img->put.separate = putRGBAAseparate8bittile;
                    else if (img->alpha == EXTRASAMPLE_UNASSALPHA)
                    {
                        if (BuildMapUaToAa(img))
                            img->put.separate = putRGBUAseparate8bittile;
                    }
                    else
                        img->put.separate = putRGBseparate8bittile;
                    break;
                case 16:
                    if (img->alpha == EXTRASAMPLE_ASSOCALPHA)
                    {
                        if (BuildMapBitdepth16To8(img))
                            img->put.separate = putRGBAAseparate16bittile;
                    }
                    else if (img->alpha == EXTRASAMPLE_UNASSALPHA)
                    {
                        if (BuildMapBitdepth16To8(img) && BuildMapUaToAa(img))
                            img->put.separate = putRGBUAseparate16bittile;
                    }
                    else
                    {
                        if (BuildMapBitdepth16To8(img))
                            img->put.separate = putRGBseparate16bittile;
                    }
                    break;
            }
            break;
        case PHOTOMETRIC_SEPARATED:
            if (img->bitspersample == 8 && img->samplesperpixel == 4)
            {
                img->alpha =
                    1; // Not alpha, but seems like the only way to get 4th band
                img->put.separate = putCMYKseparate8bittile;
            }
            break;
        case PHOTOMETRIC_YCBCR:
            if ((img->bitspersample == 8) && (img->samplesperpixel == 3))
            {
                if (initYCbCrConversion(img) != 0)
                {
                    uint16_t hs, vs;
                    TIFFGetFieldDefaulted(img->tif, TIFFTAG_YCBCRSUBSAMPLING,
                                          &hs, &vs);
                    switch ((hs << 4) | vs)
                    {
                        case 0x11:
                            img->put.separate = putseparate8bitYCbCr11tile;
                            break;
                            /* TODO: add other cases here */
                    }
                }
            }
            break;
    }
    return ((img->get != NULL) && (img->put.separate != NULL));
}

static int BuildMapUaToAa(TIFFRGBAImage *img)
{
    static const char module[] = "BuildMapUaToAa";
    uint8_t *m;
    uint16_t na, nv;
    assert(img->UaToAa == NULL);
    img->UaToAa = _TIFFmallocExt(img->tif, 65536);
    if (img->UaToAa == NULL)
    {
        TIFFErrorExtR(img->tif, module, "Out of memory");
        return (0);
    }
    m = img->UaToAa;
    for (na = 0; na < 256; na++)
    {
        for (nv = 0; nv < 256; nv++)
            *m++ = (uint8_t)((nv * na + 127) / 255);
    }
    return (1);
}

static int BuildMapBitdepth16To8(TIFFRGBAImage *img)
{
    static const char module[] = "BuildMapBitdepth16To8";
    uint8_t *m;
    uint32_t n;
    assert(img->Bitdepth16To8 == NULL);
    img->Bitdepth16To8 = _TIFFmallocExt(img->tif, 65536);
    if (img->Bitdepth16To8 == NULL)
    {
        TIFFErrorExtR(img->tif, module, "Out of memory");
        return (0);
    }
    m = img->Bitdepth16To8;
    for (n = 0; n < 65536; n++)
        *m++ = (uint8_t)((n + 128) / 257);
    return (1);
}

/*
 * Read a whole strip off data from the file, and convert to RGBA form.
 * If this is the last strip, then it will only contain the portion of
 * the strip that is actually within the image space.  The result is
 * organized in bottom to top form.
 */

int TIFFReadRGBAStrip(TIFF *tif, uint32_t row, uint32_t *raster)

{
    return TIFFReadRGBAStripExt(tif, row, raster, 0);
}

int TIFFReadRGBAStripExt(TIFF *tif, uint32_t row, uint32_t *raster,
                         int stop_on_error)

{
    char emsg[EMSG_BUF_SIZE] = "";
    TIFFRGBAImage img;
    int ok;
    uint32_t rowsperstrip, rows_to_read;

    if (TIFFIsTiled(tif))
    {
        TIFFErrorExtR(tif, TIFFFileName(tif),
                      "Can't use TIFFReadRGBAStrip() with tiled file.");
        return (0);
    }

    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsperstrip);
    if ((row % rowsperstrip) != 0)
    {
        TIFFErrorExtR(
            tif, TIFFFileName(tif),
            "Row passed to TIFFReadRGBAStrip() must be first in a strip.");
        return (0);
    }

    if (TIFFRGBAImageOK(tif, emsg) &&
        TIFFRGBAImageBegin(&img, tif, stop_on_error, emsg))
    {

        img.row_offset = row;
        img.col_offset = 0;

        if (row + rowsperstrip > img.height)
            rows_to_read = img.height - row;
        else
            rows_to_read = rowsperstrip;

        ok = TIFFRGBAImageGet(&img, raster, img.width, rows_to_read);

        TIFFRGBAImageEnd(&img);
    }
    else
    {
        TIFFErrorExtR(tif, TIFFFileName(tif), "%s", emsg);
        ok = 0;
    }

    return (ok);
}

/*
 * Read a whole tile off data from the file, and convert to RGBA form.
 * The returned RGBA data is organized from bottom to top of tile,
 * and may include zeroed areas if the tile extends off the image.
 */

int TIFFReadRGBATile(TIFF *tif, uint32_t col, uint32_t row, uint32_t *raster)

{
    return TIFFReadRGBATileExt(tif, col, row, raster, 0);
}

int TIFFReadRGBATileExt(TIFF *tif, uint32_t col, uint32_t row, uint32_t *raster,
                        int stop_on_error)
{
    char emsg[EMSG_BUF_SIZE] = "";
    TIFFRGBAImage img;
    int ok;
    uint32_t tile_xsize, tile_ysize;
    uint32_t read_xsize, read_ysize;
    uint32_t i_row;

    /*
     * Verify that our request is legal - on a tile file, and on a
     * tile boundary.
     */

    if (!TIFFIsTiled(tif))
    {
        TIFFErrorExtR(tif, TIFFFileName(tif),
                      "Can't use TIFFReadRGBATile() with striped file.");
        return (0);
    }

    TIFFGetFieldDefaulted(tif, TIFFTAG_TILEWIDTH, &tile_xsize);
    TIFFGetFieldDefaulted(tif, TIFFTAG_TILELENGTH, &tile_ysize);
    if ((col % tile_xsize) != 0 || (row % tile_ysize) != 0)
    {
        TIFFErrorExtR(tif, TIFFFileName(tif),
                      "Row/col passed to TIFFReadRGBATile() must be top"
                      "left corner of a tile.");
        return (0);
    }

    /*
     * Setup the RGBA reader.
     */

    if (!TIFFRGBAImageOK(tif, emsg) ||
        !TIFFRGBAImageBegin(&img, tif, stop_on_error, emsg))
    {
        TIFFErrorExtR(tif, TIFFFileName(tif), "%s", emsg);
        return (0);
    }

    /*
     * The TIFFRGBAImageGet() function doesn't allow us to get off the
     * edge of the image, even to fill an otherwise valid tile.  So we
     * figure out how much we can read, and fix up the tile buffer to
     * a full tile configuration afterwards.
     */

    if (row + tile_ysize > img.height)
        read_ysize = img.height - row;
    else
        read_ysize = tile_ysize;

    if (col + tile_xsize > img.width)
        read_xsize = img.width - col;
    else
        read_xsize = tile_xsize;

    /*
     * Read the chunk of imagery.
     */

    img.row_offset = row;
    img.col_offset = col;

    ok = TIFFRGBAImageGet(&img, raster, read_xsize, read_ysize);

    TIFFRGBAImageEnd(&img);

    /*
     * If our read was incomplete we will need to fix up the tile by
     * shifting the data around as if a full tile of data is being returned.
     *
     * This is all the more complicated because the image is organized in
     * bottom to top format.
     */

    if (read_xsize == tile_xsize && read_ysize == tile_ysize)
        return (ok);

    for (i_row = 0; i_row < read_ysize; i_row++)
    {
        memmove(raster + (size_t)(tile_ysize - i_row - 1) * tile_xsize,
                raster + (size_t)(read_ysize - i_row - 1) * read_xsize,
                read_xsize * sizeof(uint32_t));
        _TIFFmemset(raster + (size_t)(tile_ysize - i_row - 1) * tile_xsize +
                        read_xsize,
                    0, sizeof(uint32_t) * (tile_xsize - read_xsize));
    }

    for (i_row = read_ysize; i_row < tile_ysize; i_row++)
    {
        _TIFFmemset(raster + (size_t)(tile_ysize - i_row - 1) * tile_xsize, 0,
                    sizeof(uint32_t) * tile_xsize);
    }

    return (ok);
}
