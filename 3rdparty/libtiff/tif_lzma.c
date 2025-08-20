/*
 * Copyright (c) 2010, Andrey Kiselev <dron@ak4719.spb.edu>
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
#ifdef LZMA_SUPPORT
/*
 * TIFF Library.
 *
 * LZMA2 Compression Support
 *
 * You need an LZMA2 SDK to link with. See http://tukaani.org/xz/ for details.
 *
 * The codec is derived from ZLIB codec (tif_zip.c).
 */

#include "lzma.h"
#include "tif_predict.h"

#include <stdio.h>

/*
 * State block for each open TIFF file using LZMA2 compression/decompression.
 */
typedef struct
{
    TIFFPredictorState predict;
    int read_error; /* whether a read error has occurred, and which should cause
                       further reads in the same strip/tile to be aborted */
    lzma_stream stream;
    lzma_filter filters[LZMA_FILTERS_MAX + 1];
    lzma_options_delta opt_delta; /* delta filter options */
    lzma_options_lzma opt_lzma;   /* LZMA2 filter options */
    int preset;                   /* compression level */
    lzma_check check;             /* type of the integrity check */
    int state;                    /* state flags */
#define LSTATE_INIT_DECODE 0x01
#define LSTATE_INIT_ENCODE 0x02

    TIFFVGetMethod vgetparent; /* super-class method */
    TIFFVSetMethod vsetparent; /* super-class method */
} LZMAState;

#define GetLZMAState(tif) ((LZMAState *)(tif)->tif_data)
#define LZMADecoderState(tif) GetLZMAState(tif)
#define LZMAEncoderState(tif) GetLZMAState(tif)

static int LZMAEncode(TIFF *tif, uint8_t *bp, tmsize_t cc, uint16_t s);
static int LZMADecode(TIFF *tif, uint8_t *op, tmsize_t occ, uint16_t s);

static const char *LZMAStrerror(lzma_ret ret)
{
    switch (ret)
    {
        case LZMA_OK:
            return "operation completed successfully";
        case LZMA_STREAM_END:
            return "end of stream was reached";
        case LZMA_NO_CHECK:
            return "input stream has no integrity check";
        case LZMA_UNSUPPORTED_CHECK:
            return "cannot calculate the integrity check";
        case LZMA_GET_CHECK:
            return "integrity check type is now available";
        case LZMA_MEM_ERROR:
            return "cannot allocate memory";
        case LZMA_MEMLIMIT_ERROR:
            return "memory usage limit was reached";
        case LZMA_FORMAT_ERROR:
            return "file format not recognized";
        case LZMA_OPTIONS_ERROR:
            return "invalid or unsupported options";
        case LZMA_DATA_ERROR:
            return "data is corrupt";
        case LZMA_BUF_ERROR:
            return "no progress is possible (stream is truncated or corrupt)";
        case LZMA_PROG_ERROR:
            return "programming error";
        default:
            return "unidentified liblzma error";
    }
}

static int LZMAFixupTags(TIFF *tif)
{
    (void)tif;
    return 1;
}

static int LZMASetupDecode(TIFF *tif)
{
    LZMAState *sp = LZMADecoderState(tif);

    assert(sp != NULL);

    /* if we were last encoding, terminate this mode */
    if (sp->state & LSTATE_INIT_ENCODE)
    {
        lzma_end(&sp->stream);
        sp->state = 0;
    }

    sp->state |= LSTATE_INIT_DECODE;
    return 1;
}

/*
 * Setup state for decoding a strip.
 */
static int LZMAPreDecode(TIFF *tif, uint16_t s)
{
    static const char module[] = "LZMAPreDecode";
    LZMAState *sp = LZMADecoderState(tif);
    lzma_ret ret;

    (void)s;
    assert(sp != NULL);

    if ((sp->state & LSTATE_INIT_DECODE) == 0)
        tif->tif_setupdecode(tif);

    sp->stream.next_in = tif->tif_rawdata;
    sp->stream.avail_in = (size_t)tif->tif_rawcc;
    if ((tmsize_t)sp->stream.avail_in != tif->tif_rawcc)
    {
        TIFFErrorExtR(tif, module,
                      "Liblzma cannot deal with buffers this size");
        return 0;
    }

    /*
     * Disable memory limit when decoding. UINT64_MAX is a flag to disable
     * the limit, we are passing (uint64_t)-1 which should be the same.
     */
    ret = lzma_stream_decoder(&sp->stream, (uint64_t)-1, 0);
    if (ret != LZMA_OK)
    {
        TIFFErrorExtR(tif, module, "Error initializing the stream decoder, %s",
                      LZMAStrerror(ret));
        return 0;
    }

    sp->read_error = 0;

    return 1;
}

static int LZMADecode(TIFF *tif, uint8_t *op, tmsize_t occ, uint16_t s)
{
    static const char module[] = "LZMADecode";
    LZMAState *sp = LZMADecoderState(tif);

    (void)s;
    assert(sp != NULL);
    assert(sp->state == LSTATE_INIT_DECODE);

    if (sp->read_error)
    {
        memset(op, 0, (size_t)occ);
        TIFFErrorExtR(tif, module,
                      "LZMADecode: Scanline %" PRIu32 " cannot be read due to "
                      "previous error",
                      tif->tif_row);
        return 0;
    }

    sp->stream.next_in = tif->tif_rawcp;
    sp->stream.avail_in = (size_t)tif->tif_rawcc;

    sp->stream.next_out = op;
    sp->stream.avail_out = (size_t)occ;
    if ((tmsize_t)sp->stream.avail_out != occ)
    {
        // read_error not set here as this is a usage issue that can be
        // recovered in a following call.
        memset(op, 0, (size_t)occ);
        TIFFErrorExtR(tif, module,
                      "Liblzma cannot deal with buffers this size");
        return 0;
    }

    do
    {
        /*
         * Save the current stream state to properly recover from the
         * decoding errors later.
         */
        const uint8_t *next_in = sp->stream.next_in;
        size_t avail_in = sp->stream.avail_in;

        lzma_ret ret = lzma_code(&sp->stream, LZMA_RUN);
        if (ret == LZMA_STREAM_END)
            break;
        if (ret == LZMA_MEMLIMIT_ERROR)
        {
            lzma_ret r =
                lzma_stream_decoder(&sp->stream, lzma_memusage(&sp->stream), 0);
            if (r != LZMA_OK)
            {
                sp->read_error = 1;
                memset(op, 0, (size_t)occ);
                TIFFErrorExtR(tif, module,
                              "Error initializing the stream decoder, %s",
                              LZMAStrerror(r));
                break;
            }
            sp->stream.next_in = next_in;
            sp->stream.avail_in = avail_in;
            continue;
        }
        if (ret != LZMA_OK)
        {
            TIFFErrorExtR(tif, module,
                          "Decoding error at scanline %" PRIu32 ", %s",
                          tif->tif_row, LZMAStrerror(ret));
            break;
        }
    } while (sp->stream.avail_out > 0);
    if (sp->stream.avail_out != 0)
    {
        sp->read_error = 1;
        memset(sp->stream.next_out, 0, sp->stream.avail_out);
        TIFFErrorExtR(tif, module,
                      "Not enough data at scanline %" PRIu32
                      " (short %" TIFF_SIZE_FORMAT " bytes)",
                      tif->tif_row, sp->stream.avail_out);
        return 0;
    }

    tif->tif_rawcp = (uint8_t *)sp->stream.next_in; /* cast away const */
    tif->tif_rawcc = sp->stream.avail_in;

    return 1;
}

static int LZMASetupEncode(TIFF *tif)
{
    LZMAState *sp = LZMAEncoderState(tif);

    assert(sp != NULL);
    if (sp->state & LSTATE_INIT_DECODE)
    {
        lzma_end(&sp->stream);
        sp->state = 0;
    }

    sp->state |= LSTATE_INIT_ENCODE;
    return 1;
}

/*
 * Reset encoding state at the start of a strip.
 */
static int LZMAPreEncode(TIFF *tif, uint16_t s)
{
    static const char module[] = "LZMAPreEncode";
    LZMAState *sp = LZMAEncoderState(tif);
    lzma_ret ret;

    (void)s;
    assert(sp != NULL);
    if (sp->state != LSTATE_INIT_ENCODE)
        tif->tif_setupencode(tif);

    sp->stream.next_out = tif->tif_rawdata;
    sp->stream.avail_out = (size_t)tif->tif_rawdatasize;
    if ((tmsize_t)sp->stream.avail_out != tif->tif_rawdatasize)
    {
        TIFFErrorExtR(tif, module,
                      "Liblzma cannot deal with buffers this size");
        return 0;
    }
    ret = lzma_stream_encoder(&sp->stream, sp->filters, sp->check);
    if (ret != LZMA_OK)
    {
        TIFFErrorExtR(tif, module, "Error in lzma_stream_encoder(): %s",
                      LZMAStrerror(ret));
        return 0;
    }
    return 1;
}

/*
 * Encode a chunk of pixels.
 */
static int LZMAEncode(TIFF *tif, uint8_t *bp, tmsize_t cc, uint16_t s)
{
    static const char module[] = "LZMAEncode";
    LZMAState *sp = LZMAEncoderState(tif);

    assert(sp != NULL);
    assert(sp->state == LSTATE_INIT_ENCODE);

    (void)s;
    sp->stream.next_in = bp;
    sp->stream.avail_in = (size_t)cc;
    if ((tmsize_t)sp->stream.avail_in != cc)
    {
        TIFFErrorExtR(tif, module,
                      "Liblzma cannot deal with buffers this size");
        return 0;
    }
    do
    {
        lzma_ret ret = lzma_code(&sp->stream, LZMA_RUN);
        if (ret != LZMA_OK)
        {
            TIFFErrorExtR(tif, module,
                          "Encoding error at scanline %" PRIu32 ", %s",
                          tif->tif_row, LZMAStrerror(ret));
            return 0;
        }
        if (sp->stream.avail_out == 0)
        {
            tif->tif_rawcc = tif->tif_rawdatasize;
            if (!TIFFFlushData1(tif))
                return 0;
            sp->stream.next_out = tif->tif_rawdata;
            sp->stream.avail_out =
                (size_t)
                    tif->tif_rawdatasize; /* this is a safe typecast, as check
                                             is made already in LZMAPreEncode */
        }
    } while (sp->stream.avail_in > 0);
    return 1;
}

/*
 * Finish off an encoded strip by flushing the last
 * string and tacking on an End Of Information code.
 */
static int LZMAPostEncode(TIFF *tif)
{
    static const char module[] = "LZMAPostEncode";
    LZMAState *sp = LZMAEncoderState(tif);
    lzma_ret ret;

    sp->stream.avail_in = 0;
    do
    {
        ret = lzma_code(&sp->stream, LZMA_FINISH);
        switch (ret)
        {
            case LZMA_STREAM_END:
            case LZMA_OK:
                if ((tmsize_t)sp->stream.avail_out != tif->tif_rawdatasize)
                {
                    tif->tif_rawcc =
                        tif->tif_rawdatasize - sp->stream.avail_out;
                    if (!TIFFFlushData1(tif))
                        return 0;
                    sp->stream.next_out = tif->tif_rawdata;
                    sp->stream.avail_out =
                        (size_t)
                            tif->tif_rawdatasize; /* this is a safe typecast, as
                                                     check is made already in
                                                     ZIPPreEncode */
                }
                break;
            default:
                TIFFErrorExtR(tif, module, "Liblzma error: %s",
                              LZMAStrerror(ret));
                return 0;
        }
    } while (ret != LZMA_STREAM_END);
    return 1;
}

static void LZMACleanup(TIFF *tif)
{
    LZMAState *sp = GetLZMAState(tif);

    assert(sp != 0);

    (void)TIFFPredictorCleanup(tif);

    tif->tif_tagmethods.vgetfield = sp->vgetparent;
    tif->tif_tagmethods.vsetfield = sp->vsetparent;

    if (sp->state)
    {
        lzma_end(&sp->stream);
        sp->state = 0;
    }
    _TIFFfreeExt(tif, sp);
    tif->tif_data = NULL;

    _TIFFSetDefaultCompressionState(tif);
}

static int LZMAVSetField(TIFF *tif, uint32_t tag, va_list ap)
{
    static const char module[] = "LZMAVSetField";
    LZMAState *sp = GetLZMAState(tif);

    switch (tag)
    {
        case TIFFTAG_LZMAPRESET:
            sp->preset = (int)va_arg(ap, int);
            lzma_lzma_preset(&sp->opt_lzma, sp->preset);
            if (sp->state & LSTATE_INIT_ENCODE)
            {
                lzma_ret ret =
                    lzma_stream_encoder(&sp->stream, sp->filters, sp->check);
                if (ret != LZMA_OK)
                {
                    TIFFErrorExtR(tif, module, "Liblzma error: %s",
                                  LZMAStrerror(ret));
                }
            }
            return 1;
        default:
            return (*sp->vsetparent)(tif, tag, ap);
    }
    /*NOTREACHED*/
}

static int LZMAVGetField(TIFF *tif, uint32_t tag, va_list ap)
{
    LZMAState *sp = GetLZMAState(tif);

    switch (tag)
    {
        case TIFFTAG_LZMAPRESET:
            *va_arg(ap, int *) = sp->preset;
            break;
        default:
            return (*sp->vgetparent)(tif, tag, ap);
    }
    return 1;
}

static const TIFFField lzmaFields[] = {
    {TIFFTAG_LZMAPRESET, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE,
     "LZMA2 Compression Preset", NULL},
};

int TIFFInitLZMA(TIFF *tif, int scheme)
{
    static const char module[] = "TIFFInitLZMA";
    LZMAState *sp;
    lzma_stream tmp_stream = LZMA_STREAM_INIT;

    (void)scheme;
    assert(scheme == COMPRESSION_LZMA);

    /*
     * Merge codec-specific tag information.
     */
    if (!_TIFFMergeFields(tif, lzmaFields, TIFFArrayCount(lzmaFields)))
    {
        TIFFErrorExtR(tif, module, "Merging LZMA2 codec-specific tags failed");
        return 0;
    }

    /*
     * Allocate state block so tag methods have storage to record values.
     */
    tif->tif_data = (uint8_t *)_TIFFmallocExt(tif, sizeof(LZMAState));
    if (tif->tif_data == NULL)
        goto bad;
    sp = GetLZMAState(tif);
    memcpy(&sp->stream, &tmp_stream, sizeof(lzma_stream));

    /*
     * Override parent get/set field methods.
     */
    sp->vgetparent = tif->tif_tagmethods.vgetfield;
    tif->tif_tagmethods.vgetfield = LZMAVGetField; /* hook for codec tags */
    sp->vsetparent = tif->tif_tagmethods.vsetfield;
    tif->tif_tagmethods.vsetfield = LZMAVSetField; /* hook for codec tags */

    /* Default values for codec-specific fields */
    sp->preset = LZMA_PRESET_DEFAULT; /* default comp. level */
    sp->check = LZMA_CHECK_NONE;
    sp->state = 0;

    /* Data filters. So far we are using delta and LZMA2 filters only. */
    sp->opt_delta.type = LZMA_DELTA_TYPE_BYTE;
    /*
     * The sample size in bytes seems to be reasonable distance for delta
     * filter.
     */
    sp->opt_delta.dist = (tif->tif_dir.td_bitspersample % 8)
                             ? 1
                             : tif->tif_dir.td_bitspersample / 8;
    sp->filters[0].id = LZMA_FILTER_DELTA;
    sp->filters[0].options = &sp->opt_delta;

    lzma_lzma_preset(&sp->opt_lzma, sp->preset);
    sp->filters[1].id = LZMA_FILTER_LZMA2;
    sp->filters[1].options = &sp->opt_lzma;

    sp->filters[2].id = LZMA_VLI_UNKNOWN;
    sp->filters[2].options = NULL;

    /*
     * Install codec methods.
     */
    tif->tif_fixuptags = LZMAFixupTags;
    tif->tif_setupdecode = LZMASetupDecode;
    tif->tif_predecode = LZMAPreDecode;
    tif->tif_decoderow = LZMADecode;
    tif->tif_decodestrip = LZMADecode;
    tif->tif_decodetile = LZMADecode;
    tif->tif_setupencode = LZMASetupEncode;
    tif->tif_preencode = LZMAPreEncode;
    tif->tif_postencode = LZMAPostEncode;
    tif->tif_encoderow = LZMAEncode;
    tif->tif_encodestrip = LZMAEncode;
    tif->tif_encodetile = LZMAEncode;
    tif->tif_cleanup = LZMACleanup;
    /*
     * Setup predictor setup.
     */
    (void)TIFFPredictorInit(tif);
    return 1;
bad:
    TIFFErrorExtR(tif, module, "No space for LZMA2 state block");
    return 0;
}
#endif /* LZMA_SUPPORT */
