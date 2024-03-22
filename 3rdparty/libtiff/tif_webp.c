/*
 * Copyright (c) 2018, Mapbox
 * Author: <norman.barker at mapbox.com>
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
#ifdef WEBP_SUPPORT
/*
 * TIFF Library.
 *
 * WEBP Compression Support
 *
 */

#include "webp/decode.h"
#include "webp/encode.h"

#include <stdbool.h>
#include <stdio.h>

#define LSTATE_INIT_DECODE 0x01
#define LSTATE_INIT_ENCODE 0x02
/*
 * State block for each open TIFF
 * file using WEBP compression/decompression.
 */
typedef struct
{
    uint16_t nSamples; /* number of samples per pixel */

    int lossless;         /* lossy/lossless compression */
    int lossless_exact;   /* lossless exact mode. If TRUE, R,G,B values in areas
                             with alpha = 0 will be preserved */
    int quality_level;    /* compression level */
    WebPPicture sPicture; /* WebP Picture */
    WebPConfig sEncoderConfig;  /* WebP encoder config */
    uint8_t *pBuffer;           /* buffer to hold raw data on encoding */
    unsigned int buffer_offset; /* current offset into the buffer */
    unsigned int buffer_size;

    WebPIDecoder *psDecoder;  /* WebPIDecoder */
    WebPDecBuffer sDecBuffer; /* Decoder buffer */
    int last_y;               /* Last row decoded */

    int state; /* state flags */

    TIFFVGetMethod vgetparent; /* super-class method */
    TIFFVSetMethod vsetparent; /* super-class method */
} WebPState;

#define LState(tif) ((WebPState *)(tif)->tif_data)
#define DecoderState(tif) LState(tif)
#define EncoderState(tif) LState(tif)

static int TWebPEncode(TIFF *tif, uint8_t *bp, tmsize_t cc, uint16_t s);
static int TWebPDecode(TIFF *tif, uint8_t *op, tmsize_t occ, uint16_t s);

static int TWebPDatasetWriter(const uint8_t *data, size_t data_size,
                              const WebPPicture *const picture)
{
    static const char module[] = "TWebPDatasetWriter";
    TIFF *tif = (TIFF *)(picture->custom_ptr);

    if ((tif->tif_rawcc + (tmsize_t)data_size) > tif->tif_rawdatasize)
    {
        TIFFErrorExtR(
            tif, module, "Buffer too small by %" TIFF_SIZE_FORMAT " bytes.",
            (size_t)(tif->tif_rawcc + data_size - tif->tif_rawdatasize));
        return 0;
    }
    else
    {
        _TIFFmemcpy(tif->tif_rawcp, data, data_size);
        tif->tif_rawcc += data_size;
        tif->tif_rawcp += data_size;
        return 1;
    }
}

/*
 * Encode a chunk of pixels.
 */
static int TWebPEncode(TIFF *tif, uint8_t *bp, tmsize_t cc, uint16_t s)
{
    static const char module[] = "TWebPEncode";
    WebPState *sp = EncoderState(tif);
    (void)s;

    assert(sp != NULL);
    assert(sp->state == LSTATE_INIT_ENCODE);

    if ((uint64_t)sp->buffer_offset + (uint64_t)cc > sp->buffer_size)
    {
        TIFFErrorExtR(tif, module, "Too many bytes to be written");
        return 0;
    }

    memcpy(sp->pBuffer + sp->buffer_offset, bp, cc);
    sp->buffer_offset += (unsigned)cc;

    return 1;
}

static int TWebPDecode(TIFF *tif, uint8_t *op, tmsize_t occ, uint16_t s)
{
    static const char module[] = "WebPDecode";
    VP8StatusCode status = VP8_STATUS_OK;
    WebPState *sp = DecoderState(tif);
    uint32_t segment_width, segment_height;
    bool decode_whole_strile = false;

    (void)s;

    assert(sp != NULL);
    assert(sp->state == LSTATE_INIT_DECODE);

    if (sp->psDecoder == NULL)
    {
        TIFFDirectory *td = &tif->tif_dir;
        uint32_t buffer_size;

        if (isTiled(tif))
        {
            segment_width = td->td_tilewidth;
            segment_height = td->td_tilelength;
        }
        else
        {
            segment_width = td->td_imagewidth;
            segment_height = td->td_imagelength - tif->tif_row;
            if (segment_height > td->td_rowsperstrip)
                segment_height = td->td_rowsperstrip;
        }

        int webp_width, webp_height;
        if (!WebPGetInfo(tif->tif_rawcp,
                         (uint64_t)tif->tif_rawcc > UINT32_MAX
                             ? UINT32_MAX
                             : (uint32_t)tif->tif_rawcc,
                         &webp_width, &webp_height))
        {
            TIFFErrorExtR(tif, module, "WebPGetInfo() failed");
            return 0;
        }
        if ((uint32_t)webp_width != segment_width ||
            (uint32_t)webp_height != segment_height)
        {
            TIFFErrorExtR(
                tif, module, "WebP blob dimension is %dx%d. Expected %ux%u",
                webp_width, webp_height, segment_width, segment_height);
            return 0;
        }

#if WEBP_DECODER_ABI_VERSION >= 0x0002
        WebPDecoderConfig config;
        if (!WebPInitDecoderConfig(&config))
        {
            TIFFErrorExtR(tif, module, "WebPInitDecoderConfig() failed");
            return 0;
        }

        const bool bWebPGetFeaturesOK =
            WebPGetFeatures(tif->tif_rawcp,
                            (uint64_t)tif->tif_rawcc > UINT32_MAX
                                ? UINT32_MAX
                                : (uint32_t)tif->tif_rawcc,
                            &config.input) == VP8_STATUS_OK;

        WebPFreeDecBuffer(&config.output);

        if (!bWebPGetFeaturesOK)
        {
            TIFFErrorExtR(tif, module, "WebPInitDecoderConfig() failed");
            return 0;
        }

        const int webp_bands = config.input.has_alpha ? 4 : 3;
        if (webp_bands != sp->nSamples &&
            /* We accept the situation where the WebP blob has only 3 bands,
             * whereas the raster is 4 bands. This can happen when the alpha
             * channel is fully opaque, and WebP decoding works fine in that
             * situation.
             */
            !(webp_bands == 3 && sp->nSamples == 4))
        {
            TIFFErrorExtR(tif, module,
                          "WebP blob band count is %d. Expected %d", webp_bands,
                          sp->nSamples);
            return 0;
        }
#endif

        buffer_size = segment_width * segment_height * sp->nSamples;
        if (occ == (tmsize_t)buffer_size)
        {
            /* If decoding the whole strip/tile, we can directly use the */
            /* output buffer */
            decode_whole_strile = true;
        }
        else if (sp->pBuffer == NULL || buffer_size > sp->buffer_size)
        {
            if (sp->pBuffer != NULL)
            {
                _TIFFfreeExt(tif, sp->pBuffer);
                sp->pBuffer = NULL;
            }

            sp->pBuffer = _TIFFmallocExt(tif, buffer_size);
            if (!sp->pBuffer)
            {
                TIFFErrorExtR(tif, module, "Cannot allocate buffer");
                return 0;
            }
            sp->buffer_size = buffer_size;
        }

        sp->last_y = 0;

        WebPInitDecBuffer(&sp->sDecBuffer);

        sp->sDecBuffer.is_external_memory = 1;
        sp->sDecBuffer.width = segment_width;
        sp->sDecBuffer.height = segment_height;
        sp->sDecBuffer.u.RGBA.rgba = decode_whole_strile ? op : sp->pBuffer;
        sp->sDecBuffer.u.RGBA.stride = segment_width * sp->nSamples;
        sp->sDecBuffer.u.RGBA.size = buffer_size;

        if (sp->nSamples > 3)
        {
            sp->sDecBuffer.colorspace = MODE_RGBA;
        }
        else
        {
            sp->sDecBuffer.colorspace = MODE_RGB;
        }

        sp->psDecoder = WebPINewDecoder(&sp->sDecBuffer);

        if (sp->psDecoder == NULL)
        {
            TIFFErrorExtR(tif, module, "Unable to allocate WebP decoder.");
            return 0;
        }
    }

    if (occ % sp->sDecBuffer.u.RGBA.stride)
    {
        TIFFErrorExtR(tif, module, "Fractional scanlines cannot be read");
        return 0;
    }

    status = WebPIAppend(sp->psDecoder, tif->tif_rawcp, tif->tif_rawcc);

    if (status != VP8_STATUS_OK && status != VP8_STATUS_SUSPENDED)
    {
        if (status == VP8_STATUS_INVALID_PARAM)
        {
            TIFFErrorExtR(tif, module, "Invalid parameter used.");
        }
        else if (status == VP8_STATUS_OUT_OF_MEMORY)
        {
            TIFFErrorExtR(tif, module, "Out of memory.");
        }
        else
        {
            TIFFErrorExtR(tif, module, "Unrecognized error.");
        }
        return 0;
    }
    else
    {
        int current_y, stride;
        uint8_t *buf;

        /* Returns the RGB/A image decoded so far */
        buf = WebPIDecGetRGB(sp->psDecoder, &current_y, NULL, NULL, &stride);

        if ((buf != NULL) &&
            (occ <= (tmsize_t)stride * (current_y - sp->last_y)))
        {
            const int numberOfExpectedLines =
                (int)(occ / sp->sDecBuffer.u.RGBA.stride);
            if (decode_whole_strile)
            {
                if (current_y != numberOfExpectedLines)
                {
                    TIFFErrorExtR(tif, module,
                                  "Unable to decode WebP data: less lines than "
                                  "expected.");
                    return 0;
                }
            }
            else
            {
                memcpy(op, buf + (sp->last_y * stride), occ);
            }

            tif->tif_rawcp += tif->tif_rawcc;
            tif->tif_rawcc = 0;
            sp->last_y += numberOfExpectedLines;

            if (decode_whole_strile)
            {
                /* We can now free the decoder as we're completely done */
                if (sp->psDecoder != NULL)
                {
                    WebPIDelete(sp->psDecoder);
                    WebPFreeDecBuffer(&sp->sDecBuffer);
                    sp->psDecoder = NULL;
                }
            }
            return 1;
        }
        else
        {
            TIFFErrorExtR(tif, module, "Unable to decode WebP data.");
            return 0;
        }
    }
}

static int TWebPFixupTags(TIFF *tif)
{
    (void)tif;
    if (tif->tif_dir.td_planarconfig != PLANARCONFIG_CONTIG)
    {
        static const char module[] = "TWebPFixupTags";
        TIFFErrorExtR(tif, module,
                      "TIFF WEBP requires data to be stored contiguously in "
                      "RGB e.g. RGBRGBRGB "
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
                      "or RGBARGBARGBA"
#endif
        );
        return 0;
    }
    return 1;
}

static int TWebPSetupDecode(TIFF *tif)
{
    static const char module[] = "WebPSetupDecode";
    uint16_t nBitsPerSample = tif->tif_dir.td_bitspersample;
    uint16_t sampleFormat = tif->tif_dir.td_sampleformat;

    WebPState *sp = DecoderState(tif);
    assert(sp != NULL);

    sp->nSamples = tif->tif_dir.td_samplesperpixel;

    /* check band count */
    if (sp->nSamples != 3
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
        && sp->nSamples != 4
#endif
    )
    {
        TIFFErrorExtR(tif, module,
                      "WEBP driver doesn't support %d bands. Must be 3 (RGB) "
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
                      "or 4 (RGBA) "
#endif
                      "bands.",
                      sp->nSamples);
        return 0;
    }

    /* check bits per sample and data type */
    if ((nBitsPerSample != 8) && (sampleFormat != 1))
    {
        TIFFErrorExtR(tif, module, "WEBP driver requires 8 bit unsigned data");
        return 0;
    }

    /* if we were last encoding, terminate this mode */
    if (sp->state & LSTATE_INIT_ENCODE)
    {
        WebPPictureFree(&sp->sPicture);
        if (sp->pBuffer != NULL)
        {
            _TIFFfreeExt(tif, sp->pBuffer);
            sp->pBuffer = NULL;
        }
        sp->buffer_offset = 0;
        sp->state = 0;
    }

    sp->state |= LSTATE_INIT_DECODE;

    return 1;
}

/*
 * Setup state for decoding a strip.
 */
static int TWebPPreDecode(TIFF *tif, uint16_t s)
{
    static const char module[] = "TWebPPreDecode";
    uint32_t segment_width, segment_height;
    WebPState *sp = DecoderState(tif);
    TIFFDirectory *td = &tif->tif_dir;
    (void)s;
    assert(sp != NULL);

    if (isTiled(tif))
    {
        segment_width = td->td_tilewidth;
        segment_height = td->td_tilelength;
    }
    else
    {
        segment_width = td->td_imagewidth;
        segment_height = td->td_imagelength - tif->tif_row;
        if (segment_height > td->td_rowsperstrip)
            segment_height = td->td_rowsperstrip;
    }

    if (segment_width > 16383 || segment_height > 16383)
    {
        TIFFErrorExtR(tif, module,
                      "WEBP maximum image dimensions are 16383 x 16383.");
        return 0;
    }

    if ((sp->state & LSTATE_INIT_DECODE) == 0)
        tif->tif_setupdecode(tif);

    if (sp->psDecoder != NULL)
    {
        WebPIDelete(sp->psDecoder);
        WebPFreeDecBuffer(&sp->sDecBuffer);
        sp->psDecoder = NULL;
    }

    return 1;
}

static int TWebPSetupEncode(TIFF *tif)
{
    static const char module[] = "WebPSetupEncode";
    uint16_t nBitsPerSample = tif->tif_dir.td_bitspersample;
    uint16_t sampleFormat = tif->tif_dir.td_sampleformat;

    WebPState *sp = EncoderState(tif);
    assert(sp != NULL);

    sp->nSamples = tif->tif_dir.td_samplesperpixel;

    /* check band count */
    if (sp->nSamples != 3
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
        && sp->nSamples != 4
#endif
    )
    {
        TIFFErrorExtR(tif, module,
                      "WEBP driver doesn't support %d bands. Must be 3 (RGB) "
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
                      "or 4 (RGBA) "
#endif
                      "bands.",
                      sp->nSamples);
        return 0;
    }

    /* check bits per sample and data type */
    if ((nBitsPerSample != 8) || (sampleFormat != SAMPLEFORMAT_UINT))
    {
        TIFFErrorExtR(tif, module, "WEBP driver requires 8 bit unsigned data");
        return 0;
    }

    if (sp->state & LSTATE_INIT_DECODE)
    {
        WebPIDelete(sp->psDecoder);
        WebPFreeDecBuffer(&sp->sDecBuffer);
        sp->psDecoder = NULL;
        sp->last_y = 0;
        sp->state = 0;
    }

    sp->state |= LSTATE_INIT_ENCODE;

    if (!WebPPictureInit(&sp->sPicture))
    {
        TIFFErrorExtR(tif, module, "Error initializing WebP picture.");
        return 0;
    }

    if (!WebPConfigInitInternal(&sp->sEncoderConfig, WEBP_PRESET_DEFAULT,
                                (float)sp->quality_level,
                                WEBP_ENCODER_ABI_VERSION))
    {
        TIFFErrorExtR(tif, module,
                      "Error creating WebP encoder configuration.");
        return 0;
    }

// WebPConfigInitInternal above sets lossless to false
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
    sp->sEncoderConfig.lossless = sp->lossless;
    if (sp->lossless)
    {
        sp->sPicture.use_argb = 1;
#if WEBP_ENCODER_ABI_VERSION >= 0x0209
        sp->sEncoderConfig.exact = sp->lossless_exact;
#endif
    }
#endif

    if (!WebPValidateConfig(&sp->sEncoderConfig))
    {
        TIFFErrorExtR(tif, module, "Error with WebP encoder configuration.");
        return 0;
    }

    return 1;
}

/*
 * Reset encoding state at the start of a strip.
 */
static int TWebPPreEncode(TIFF *tif, uint16_t s)
{
    static const char module[] = "TWebPPreEncode";
    uint32_t segment_width, segment_height;
    WebPState *sp = EncoderState(tif);
    TIFFDirectory *td = &tif->tif_dir;

    (void)s;

    assert(sp != NULL);
    if (sp->state != LSTATE_INIT_ENCODE)
        tif->tif_setupencode(tif);

    /*
     * Set encoding parameters for this strip/tile.
     */
    if (isTiled(tif))
    {
        segment_width = td->td_tilewidth;
        segment_height = td->td_tilelength;
    }
    else
    {
        segment_width = td->td_imagewidth;
        segment_height = td->td_imagelength - tif->tif_row;
        if (segment_height > td->td_rowsperstrip)
            segment_height = td->td_rowsperstrip;
    }

    if (segment_width > 16383 || segment_height > 16383)
    {
        TIFFErrorExtR(tif, module,
                      "WEBP maximum image dimensions are 16383 x 16383.");
        return 0;
    }

    /* set up buffer for raw data */
    /* given above check and that nSamples <= 4, buffer_size is <= 1 GB */
    sp->buffer_size = segment_width * segment_height * sp->nSamples;

    if (sp->pBuffer != NULL)
    {
        _TIFFfreeExt(tif, sp->pBuffer);
        sp->pBuffer = NULL;
    }

    sp->pBuffer = _TIFFmallocExt(tif, sp->buffer_size);
    if (!sp->pBuffer)
    {
        TIFFErrorExtR(tif, module, "Cannot allocate buffer");
        return 0;
    }
    sp->buffer_offset = 0;

    sp->sPicture.width = segment_width;
    sp->sPicture.height = segment_height;
    sp->sPicture.writer = TWebPDatasetWriter;
    sp->sPicture.custom_ptr = tif;

    return 1;
}

/*
 * Finish off an encoded strip by flushing it.
 */
static int TWebPPostEncode(TIFF *tif)
{
    static const char module[] = "WebPPostEncode";
    int64_t stride;
    WebPState *sp = EncoderState(tif);
    assert(sp != NULL);

    assert(sp->state == LSTATE_INIT_ENCODE);

    stride = (int64_t)sp->sPicture.width * sp->nSamples;

#if WEBP_ENCODER_ABI_VERSION >= 0x0100
    if (sp->nSamples == 4)
    {
        if (!WebPPictureImportRGBA(&sp->sPicture, sp->pBuffer, (int)stride))
        {
            TIFFErrorExtR(tif, module, "WebPPictureImportRGBA() failed");
            return 0;
        }
    }
    else
#endif
        if (!WebPPictureImportRGB(&sp->sPicture, sp->pBuffer, (int)stride))
    {
        TIFFErrorExtR(tif, module, "WebPPictureImportRGB() failed");
        return 0;
    }

    if (!WebPEncode(&sp->sEncoderConfig, &sp->sPicture))
    {

#if WEBP_ENCODER_ABI_VERSION >= 0x0100
        const char *pszErrorMsg = NULL;
        switch (sp->sPicture.error_code)
        {
            case VP8_ENC_ERROR_OUT_OF_MEMORY:
                pszErrorMsg = "Out of memory";
                break;
            case VP8_ENC_ERROR_BITSTREAM_OUT_OF_MEMORY:
                pszErrorMsg = "Out of memory while flushing bits";
                break;
            case VP8_ENC_ERROR_NULL_PARAMETER:
                pszErrorMsg = "A pointer parameter is NULL";
                break;
            case VP8_ENC_ERROR_INVALID_CONFIGURATION:
                pszErrorMsg = "Configuration is invalid";
                break;
            case VP8_ENC_ERROR_BAD_DIMENSION:
                pszErrorMsg = "Picture has invalid width/height";
                break;
            case VP8_ENC_ERROR_PARTITION0_OVERFLOW:
                pszErrorMsg = "Partition is bigger than 512k. Try using less "
                              "SEGMENTS, or increase PARTITION_LIMIT value";
                break;
            case VP8_ENC_ERROR_PARTITION_OVERFLOW:
                pszErrorMsg = "Partition is bigger than 16M";
                break;
            case VP8_ENC_ERROR_BAD_WRITE:
                pszErrorMsg = "Error while fludshing bytes";
                break;
            case VP8_ENC_ERROR_FILE_TOO_BIG:
                pszErrorMsg = "File is bigger than 4G";
                break;
            case VP8_ENC_ERROR_USER_ABORT:
                pszErrorMsg = "User interrupted";
                break;
            default:
                TIFFErrorExtR(tif, module,
                              "WebPEncode returned an unknown error code: %d",
                              sp->sPicture.error_code);
                pszErrorMsg = "Unknown WebP error type.";
                break;
        }
        TIFFErrorExtR(tif, module, "WebPEncode() failed : %s", pszErrorMsg);
#else
        TIFFErrorExtR(tif, module, "Error in WebPEncode()");
#endif
        return 0;
    }

    sp->sPicture.custom_ptr = NULL;

    if (!TIFFFlushData1(tif))
    {
        TIFFErrorExtR(tif, module, "Error flushing TIFF WebP encoder.");
        return 0;
    }

    return 1;
}

static void TWebPCleanup(TIFF *tif)
{
    WebPState *sp = LState(tif);

    assert(sp != 0);

    tif->tif_tagmethods.vgetfield = sp->vgetparent;
    tif->tif_tagmethods.vsetfield = sp->vsetparent;

    if (sp->state & LSTATE_INIT_ENCODE)
    {
        WebPPictureFree(&sp->sPicture);
    }

    if (sp->psDecoder != NULL)
    {
        WebPIDelete(sp->psDecoder);
        WebPFreeDecBuffer(&sp->sDecBuffer);
        sp->psDecoder = NULL;
        sp->last_y = 0;
    }

    if (sp->pBuffer != NULL)
    {
        _TIFFfreeExt(tif, sp->pBuffer);
        sp->pBuffer = NULL;
    }

    _TIFFfreeExt(tif, tif->tif_data);
    tif->tif_data = NULL;

    _TIFFSetDefaultCompressionState(tif);
}

static int TWebPVSetField(TIFF *tif, uint32_t tag, va_list ap)
{
    static const char module[] = "WebPVSetField";
    WebPState *sp = LState(tif);

    switch (tag)
    {
        case TIFFTAG_WEBP_LEVEL:
            sp->quality_level = (int)va_arg(ap, int);
            if (sp->quality_level <= 0 || sp->quality_level > 100.0f)
            {
                TIFFWarningExtR(tif, module,
                                "WEBP_LEVEL should be between 1 and 100");
            }
            return 1;
        case TIFFTAG_WEBP_LOSSLESS:
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
            sp->lossless = va_arg(ap, int);
            if (sp->lossless)
            {
                sp->quality_level = 100;
            }
            return 1;
#else
            TIFFErrorExtR(
                tif, module,
                "Need to upgrade WEBP driver, this version doesn't support "
                "lossless compression.");
            return 0;
#endif
        case TIFFTAG_WEBP_LOSSLESS_EXACT:
#if WEBP_ENCODER_ABI_VERSION >= 0x0209
            sp->lossless_exact = va_arg(ap, int);
            return 1;
#else
            TIFFErrorExtR(
                tif, module,
                "Need to upgrade WEBP driver, this version doesn't support "
                "lossless compression.");
            return 0;
#endif
        default:
            return (*sp->vsetparent)(tif, tag, ap);
    }
    /*NOTREACHED*/
}

static int TWebPVGetField(TIFF *tif, uint32_t tag, va_list ap)
{
    WebPState *sp = LState(tif);

    switch (tag)
    {
        case TIFFTAG_WEBP_LEVEL:
            *va_arg(ap, int *) = sp->quality_level;
            break;
        case TIFFTAG_WEBP_LOSSLESS:
            *va_arg(ap, int *) = sp->lossless;
            break;
        case TIFFTAG_WEBP_LOSSLESS_EXACT:
            *va_arg(ap, int *) = sp->lossless_exact;
            break;
        default:
            return (*sp->vgetparent)(tif, tag, ap);
    }
    return 1;
}

static const TIFFField TWebPFields[] = {
    {TIFFTAG_WEBP_LEVEL, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE, "WEBP quality", NULL},
    {TIFFTAG_WEBP_LOSSLESS, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE, "WEBP lossless/lossy",
     NULL},
    {TIFFTAG_WEBP_LOSSLESS_EXACT, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE, "WEBP exact lossless",
     NULL},
};

int TIFFInitWebP(TIFF *tif, int scheme)
{
    static const char module[] = "TIFFInitWebP";
    WebPState *sp;

    (void)scheme;
    assert(scheme == COMPRESSION_WEBP);

    /*
     * Merge codec-specific tag information.
     */
    if (!_TIFFMergeFields(tif, TWebPFields, TIFFArrayCount(TWebPFields)))
    {
        TIFFErrorExtR(tif, module, "Merging WebP codec-specific tags failed");
        return 0;
    }

    /*
     * Allocate state block so tag methods have storage to record values.
     */
    tif->tif_data = (uint8_t *)_TIFFmallocExt(tif, sizeof(WebPState));
    if (tif->tif_data == NULL)
        goto bad;
    sp = LState(tif);

    /*
     * Override parent get/set field methods.
     */
    sp->vgetparent = tif->tif_tagmethods.vgetfield;
    tif->tif_tagmethods.vgetfield = TWebPVGetField; /* hook for codec tags */
    sp->vsetparent = tif->tif_tagmethods.vsetfield;
    tif->tif_tagmethods.vsetfield = TWebPVSetField; /* hook for codec tags */

    /* Default values for codec-specific fields */
    sp->quality_level = 75; /* default comp. level */
    sp->lossless = 0;       /* default to false */
    sp->lossless_exact = 1; /* exact lossless mode (if lossless enabled) */
    sp->state = 0;
    sp->nSamples = 0;
    sp->psDecoder = NULL;
    sp->last_y = 0;

    sp->buffer_offset = 0;
    sp->pBuffer = NULL;

    /*
     * Install codec methods.
     * Notes:
     * encoderow is not supported
     */
    tif->tif_fixuptags = TWebPFixupTags;
    tif->tif_setupdecode = TWebPSetupDecode;
    tif->tif_predecode = TWebPPreDecode;
    tif->tif_decoderow = TWebPDecode;
    tif->tif_decodestrip = TWebPDecode;
    tif->tif_decodetile = TWebPDecode;
    tif->tif_setupencode = TWebPSetupEncode;
    tif->tif_preencode = TWebPPreEncode;
    tif->tif_postencode = TWebPPostEncode;
    tif->tif_encoderow = TWebPEncode;
    tif->tif_encodestrip = TWebPEncode;
    tif->tif_encodetile = TWebPEncode;
    tif->tif_cleanup = TWebPCleanup;

    return 1;
bad:
    TIFFErrorExtR(tif, module, "No space for WebP state block");
    return 0;
}

#endif /* WEBP_SUPPORT */
