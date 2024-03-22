/*
 * Copyright (c) 2018, Even Rouault
 * Author: <even.rouault at spatialys.com>
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
#ifdef LERC_SUPPORT
/*
 * TIFF Library.
 *
 * LERC Compression Support
 *
 */

#include "Lerc_c_api.h"
#include "zlib.h"
#ifdef ZSTD_SUPPORT
#include "zstd.h"
#endif

#if LIBDEFLATE_SUPPORT
#include "libdeflate.h"
#endif
#define LIBDEFLATE_MAX_COMPRESSION_LEVEL 12

#include <assert.h>

#define LSTATE_INIT_DECODE 0x01
#define LSTATE_INIT_ENCODE 0x02

#ifndef LERC_AT_LEAST_VERSION
#define LERC_AT_LEAST_VERSION(maj, min, patch) 0
#endif

/*
 * State block for each open TIFF file using LERC compression/decompression.
 */
typedef struct
{
    double maxzerror; /* max z error */
    int lerc_version;
    int additional_compression;
    int zstd_compress_level; /* zstd */
    int zipquality;          /* deflate */
    int state;               /* state flags */

    uint32_t segment_width;
    uint32_t segment_height;

    unsigned int uncompressed_size;
    unsigned int uncompressed_alloc;
    uint8_t *uncompressed_buffer;
    unsigned int uncompressed_offset;

    unsigned int mask_size;
    uint8_t *mask_buffer;

    unsigned int compressed_size;
    void *compressed_buffer;

#if LIBDEFLATE_SUPPORT
    struct libdeflate_decompressor *libdeflate_dec;
    struct libdeflate_compressor *libdeflate_enc;
#endif

    TIFFVGetMethod vgetparent; /* super-class method */
    TIFFVSetMethod vsetparent; /* super-class method */
} LERCState;

#define LState(tif) ((LERCState *)(tif)->tif_data)
#define DecoderState(tif) LState(tif)
#define EncoderState(tif) LState(tif)

static int LERCEncode(TIFF *tif, uint8_t *bp, tmsize_t cc, uint16_t s);
static int LERCDecode(TIFF *tif, uint8_t *op, tmsize_t occ, uint16_t s);

static int LERCFixupTags(TIFF *tif)
{
    (void)tif;
    return 1;
}

static int LERCSetupDecode(TIFF *tif)
{
    LERCState *sp = DecoderState(tif);

    assert(sp != NULL);

    /* if we were last encoding, terminate this mode */
    if (sp->state & LSTATE_INIT_ENCODE)
    {
        sp->state = 0;
    }

    sp->state |= LSTATE_INIT_DECODE;
    return 1;
}

static int GetLercDataType(TIFF *tif)
{
    TIFFDirectory *td = &tif->tif_dir;
    static const char module[] = "GetLercDataType";

    if (td->td_sampleformat == SAMPLEFORMAT_INT && td->td_bitspersample == 8)
    {
        return 0;
    }

    if (td->td_sampleformat == SAMPLEFORMAT_UINT && td->td_bitspersample == 8)
    {
        return 1;
    }

    if (td->td_sampleformat == SAMPLEFORMAT_INT && td->td_bitspersample == 16)
    {
        return 2;
    }

    if (td->td_sampleformat == SAMPLEFORMAT_UINT && td->td_bitspersample == 16)
    {
        return 3;
    }

    if (td->td_sampleformat == SAMPLEFORMAT_INT && td->td_bitspersample == 32)
    {
        return 4;
    }

    if (td->td_sampleformat == SAMPLEFORMAT_UINT && td->td_bitspersample == 32)
    {
        return 5;
    }

    if (td->td_sampleformat == SAMPLEFORMAT_IEEEFP &&
        td->td_bitspersample == 32)
    {
        return 6;
    }

    if (td->td_sampleformat == SAMPLEFORMAT_IEEEFP &&
        td->td_bitspersample == 64)
    {
        return 7;
    }

    TIFFErrorExtR(
        tif, module,
        "Unsupported combination of SampleFormat and td_bitspersample");
    return -1;
}

static int SetupUncompressedBuffer(TIFF *tif, LERCState *sp, const char *module)
{
    TIFFDirectory *td = &tif->tif_dir;
    uint64_t new_size_64;
    uint64_t new_alloc_64;
    unsigned int new_size;
    unsigned int new_alloc;

    sp->uncompressed_offset = 0;

    if (isTiled(tif))
    {
        sp->segment_width = td->td_tilewidth;
        sp->segment_height = td->td_tilelength;
    }
    else
    {
        sp->segment_width = td->td_imagewidth;
        sp->segment_height = td->td_imagelength - tif->tif_row;
        if (sp->segment_height > td->td_rowsperstrip)
            sp->segment_height = td->td_rowsperstrip;
    }

    new_size_64 = (uint64_t)sp->segment_width * sp->segment_height *
                  (td->td_bitspersample / 8);
    if (td->td_planarconfig == PLANARCONFIG_CONTIG)
    {
        new_size_64 *= td->td_samplesperpixel;
    }

    new_size = (unsigned int)new_size_64;
    sp->uncompressed_size = new_size;

    /* add some margin as we are going to use it also to store deflate/zstd
     * compressed data */
    new_alloc_64 = 100 + new_size_64 + new_size_64 / 3;
#ifdef ZSTD_SUPPORT
    {
        size_t zstd_max = ZSTD_compressBound((size_t)new_size_64);
        if (new_alloc_64 < zstd_max)
        {
            new_alloc_64 = zstd_max;
        }
    }
#endif
    new_alloc = (unsigned int)new_alloc_64;
    if (new_alloc != new_alloc_64)
    {
        TIFFErrorExtR(tif, module, "Too large uncompressed strip/tile");
        _TIFFfreeExt(tif, sp->uncompressed_buffer);
        sp->uncompressed_buffer = 0;
        sp->uncompressed_alloc = 0;
        return 0;
    }

    if (sp->uncompressed_alloc < new_alloc)
    {
        _TIFFfreeExt(tif, sp->uncompressed_buffer);
        sp->uncompressed_buffer = _TIFFmallocExt(tif, new_alloc);
        if (!sp->uncompressed_buffer)
        {
            TIFFErrorExtR(tif, module, "Cannot allocate buffer");
            _TIFFfreeExt(tif, sp->uncompressed_buffer);
            sp->uncompressed_buffer = 0;
            sp->uncompressed_alloc = 0;
            return 0;
        }
        sp->uncompressed_alloc = new_alloc;
    }

    if ((td->td_planarconfig == PLANARCONFIG_CONTIG &&
         td->td_extrasamples > 0 &&
         td->td_sampleinfo[td->td_extrasamples - 1] == EXTRASAMPLE_UNASSALPHA &&
         GetLercDataType(tif) == 1) ||
        (td->td_sampleformat == SAMPLEFORMAT_IEEEFP &&
         (td->td_planarconfig == PLANARCONFIG_SEPARATE ||
          td->td_samplesperpixel == 1) &&
         (td->td_bitspersample == 32 || td->td_bitspersample == 64)))
    {
        unsigned int mask_size = sp->segment_width * sp->segment_height;
        if (sp->mask_size < mask_size)
        {
            void *mask_buffer =
                _TIFFreallocExt(tif, sp->mask_buffer, mask_size);
            if (mask_buffer == NULL)
            {
                TIFFErrorExtR(tif, module, "Cannot allocate buffer");
                sp->mask_size = 0;
                _TIFFfreeExt(tif, sp->uncompressed_buffer);
                sp->uncompressed_buffer = 0;
                sp->uncompressed_alloc = 0;
                return 0;
            }
            sp->mask_buffer = (uint8_t *)mask_buffer;
            sp->mask_size = mask_size;
        }
    }

    return 1;
}

/*
 * Setup state for decoding a strip.
 */
static int LERCPreDecode(TIFF *tif, uint16_t s)
{
    static const char module[] = "LERCPreDecode";
    lerc_status lerc_ret;
    TIFFDirectory *td = &tif->tif_dir;
    LERCState *sp = DecoderState(tif);
    int lerc_data_type;
    unsigned int infoArray[8];
    unsigned nomask_bands = td->td_samplesperpixel;
    int ndims;
    int use_mask = 0;
    uint8_t *lerc_data = tif->tif_rawcp;
    unsigned int lerc_data_size = (unsigned int)tif->tif_rawcc;

    (void)s;
    assert(sp != NULL);
    if (sp->state != LSTATE_INIT_DECODE)
        tif->tif_setupdecode(tif);

    lerc_data_type = GetLercDataType(tif);
    if (lerc_data_type < 0)
        return 0;

    if (!SetupUncompressedBuffer(tif, sp, module))
        return 0;

    if (sp->additional_compression != LERC_ADD_COMPRESSION_NONE)
    {
        if (sp->compressed_size < sp->uncompressed_alloc)
        {
            _TIFFfreeExt(tif, sp->compressed_buffer);
            sp->compressed_buffer = _TIFFmallocExt(tif, sp->uncompressed_alloc);
            if (!sp->compressed_buffer)
            {
                sp->compressed_size = 0;
                return 0;
            }
            sp->compressed_size = sp->uncompressed_alloc;
        }
    }

    if (sp->additional_compression == LERC_ADD_COMPRESSION_DEFLATE)
    {
#if LIBDEFLATE_SUPPORT
        enum libdeflate_result res;
        size_t lerc_data_sizet = 0;
        if (sp->libdeflate_dec == NULL)
        {
            sp->libdeflate_dec = libdeflate_alloc_decompressor();
            if (sp->libdeflate_dec == NULL)
            {
                TIFFErrorExtR(tif, module, "Cannot allocate decompressor");
                return 0;
            }
        }

        res = libdeflate_zlib_decompress(
            sp->libdeflate_dec, tif->tif_rawcp, (size_t)tif->tif_rawcc,
            sp->compressed_buffer, sp->compressed_size, &lerc_data_sizet);
        if (res != LIBDEFLATE_SUCCESS)
        {
            TIFFErrorExtR(tif, module, "Decoding error at scanline %lu",
                          (unsigned long)tif->tif_row);
            return 0;
        }
        assert(lerc_data_sizet == (unsigned int)lerc_data_sizet);
        lerc_data = sp->compressed_buffer;
        lerc_data_size = (unsigned int)lerc_data_sizet;
#else
        z_stream strm;
        int zlib_ret;

        memset(&strm, 0, sizeof(strm));
        strm.zalloc = NULL;
        strm.zfree = NULL;
        strm.opaque = NULL;
        zlib_ret = inflateInit(&strm);
        if (zlib_ret != Z_OK)
        {
            TIFFErrorExtR(tif, module, "inflateInit() failed");
            inflateEnd(&strm);
            return 0;
        }

        strm.avail_in = (uInt)tif->tif_rawcc;
        strm.next_in = tif->tif_rawcp;
        strm.avail_out = sp->compressed_size;
        strm.next_out = sp->compressed_buffer;
        zlib_ret = inflate(&strm, Z_FINISH);
        if (zlib_ret != Z_STREAM_END && zlib_ret != Z_OK)
        {
            TIFFErrorExtR(tif, module, "inflate() failed");
            inflateEnd(&strm);
            return 0;
        }
        lerc_data = sp->compressed_buffer;
        lerc_data_size = sp->compressed_size - strm.avail_out;
        inflateEnd(&strm);
#endif
    }
    else if (sp->additional_compression == LERC_ADD_COMPRESSION_ZSTD)
    {
#ifdef ZSTD_SUPPORT
        size_t zstd_ret;

        zstd_ret = ZSTD_decompress(sp->compressed_buffer, sp->compressed_size,
                                   tif->tif_rawcp, tif->tif_rawcc);
        if (ZSTD_isError(zstd_ret))
        {
            TIFFErrorExtR(tif, module, "Error in ZSTD_decompress(): %s",
                          ZSTD_getErrorName(zstd_ret));
            return 0;
        }

        lerc_data = sp->compressed_buffer;
        lerc_data_size = (unsigned int)zstd_ret;
#else
        TIFFErrorExtR(tif, module, "ZSTD support missing");
        return 0;
#endif
    }
    else if (sp->additional_compression != LERC_ADD_COMPRESSION_NONE)
    {
        TIFFErrorExtR(tif, module, "Unhandled additional compression");
        return 0;
    }

    lerc_ret =
        lerc_getBlobInfo(lerc_data, lerc_data_size, infoArray, NULL, 8, 0);
    if (lerc_ret != 0)
    {
        TIFFErrorExtR(tif, module, "lerc_getBlobInfo() failed");
        return 0;
    }

    /* If the configuration is compatible of a LERC mask, and that the */
    /* LERC info has dim == samplesperpixel - 1, then there is a LERC */
    /* mask. */
    if (td->td_planarconfig == PLANARCONFIG_CONTIG && td->td_extrasamples > 0 &&
        td->td_sampleinfo[td->td_extrasamples - 1] == EXTRASAMPLE_UNASSALPHA &&
        GetLercDataType(tif) == 1 &&
        infoArray[2] == td->td_samplesperpixel - 1U)
    {
        use_mask = 1;
        nomask_bands--;
    }
    else if (td->td_sampleformat == SAMPLEFORMAT_IEEEFP &&
             (td->td_planarconfig == PLANARCONFIG_SEPARATE ||
              td->td_samplesperpixel == 1) &&
             (td->td_bitspersample == 32 || td->td_bitspersample == 64))
    {
        use_mask = 1;
    }

    ndims = td->td_planarconfig == PLANARCONFIG_CONTIG ? nomask_bands : 1;

    /* Info returned in infoArray is { version, dataType, nDim, nCols,
        nRows, nBands, nValidPixels, blobSize } */
    if (infoArray[0] != (unsigned)sp->lerc_version)
    {
        TIFFWarningExtR(tif, module,
                        "Unexpected version number: %d. Expected: %d",
                        infoArray[0], sp->lerc_version);
    }
    if (infoArray[1] != (unsigned)lerc_data_type)
    {
        TIFFErrorExtR(tif, module, "Unexpected dataType: %d. Expected: %d",
                      infoArray[1], lerc_data_type);
        return 0;
    }
    if (infoArray[2] != (unsigned)ndims)
    {
        TIFFErrorExtR(tif, module, "Unexpected nDim: %d. Expected: %d",
                      infoArray[2], ndims);
        return 0;
    }
    if (infoArray[3] != sp->segment_width)
    {
        TIFFErrorExtR(tif, module, "Unexpected nCols: %d. Expected: %du",
                      infoArray[3], sp->segment_width);
        return 0;
    }
    if (infoArray[4] != sp->segment_height)
    {
        TIFFErrorExtR(tif, module, "Unexpected nRows: %d. Expected: %u",
                      infoArray[4], sp->segment_height);
        return 0;
    }
    if (infoArray[5] != 1)
    {
        TIFFErrorExtR(tif, module, "Unexpected nBands: %d. Expected: %d",
                      infoArray[5], 1);
        return 0;
    }
    if (infoArray[7] != lerc_data_size)
    {
        TIFFErrorExtR(tif, module, "Unexpected blobSize: %d. Expected: %u",
                      infoArray[7], lerc_data_size);
        return 0;
    }

    lerc_ret = lerc_decode(lerc_data, lerc_data_size,
#if LERC_AT_LEAST_VERSION(3, 0, 0)
                           use_mask ? 1 : 0,
#endif
                           use_mask ? sp->mask_buffer : NULL, ndims,
                           sp->segment_width, sp->segment_height, 1,
                           lerc_data_type, sp->uncompressed_buffer);
    if (lerc_ret != 0)
    {
        TIFFErrorExtR(tif, module, "lerc_decode() failed");
        return 0;
    }

    /* Interleave alpha mask with other samples. */
    if (use_mask && GetLercDataType(tif) == 1)
    {
        unsigned src_stride =
            (td->td_samplesperpixel - 1) * (td->td_bitspersample / 8);
        unsigned dst_stride =
            td->td_samplesperpixel * (td->td_bitspersample / 8);
        unsigned i = sp->segment_width * sp->segment_height;
        /* Operate from end to begin to be able to move in place */
        while (i > 0 && i > nomask_bands)
        {
            i--;
            sp->uncompressed_buffer[i * dst_stride + td->td_samplesperpixel -
                                    1] = 255 * sp->mask_buffer[i];
            memcpy(sp->uncompressed_buffer + i * dst_stride,
                   sp->uncompressed_buffer + i * src_stride, src_stride);
        }
        /* First pixels must use memmove due to overlapping areas */
        while (i > 0)
        {
            i--;
            sp->uncompressed_buffer[i * dst_stride + td->td_samplesperpixel -
                                    1] = 255 * sp->mask_buffer[i];
            memmove(sp->uncompressed_buffer + i * dst_stride,
                    sp->uncompressed_buffer + i * src_stride, src_stride);
        }
    }
    else if (use_mask && td->td_sampleformat == SAMPLEFORMAT_IEEEFP)
    {
        const unsigned nb_pixels = sp->segment_width * sp->segment_height;
        unsigned i;
#if WORDS_BIGENDIAN
        const unsigned char nan_bytes[] = {0x7f, 0xc0, 0, 0};
#else
        const unsigned char nan_bytes[] = {0, 0, 0xc0, 0x7f};
#endif
        float nan_float32;
        memcpy(&nan_float32, nan_bytes, 4);

        if (td->td_bitspersample == 32)
        {
            for (i = 0; i < nb_pixels; i++)
            {
                if (sp->mask_buffer[i] == 0)
                    ((float *)sp->uncompressed_buffer)[i] = nan_float32;
            }
        }
        else
        {
            const double nan_float64 = nan_float32;
            for (i = 0; i < nb_pixels; i++)
            {
                if (sp->mask_buffer[i] == 0)
                    ((double *)sp->uncompressed_buffer)[i] = nan_float64;
            }
        }
    }

    return 1;
}

/*
 * Decode a strip, tile or scanline.
 */
static int LERCDecode(TIFF *tif, uint8_t *op, tmsize_t occ, uint16_t s)
{
    static const char module[] = "LERCDecode";
    LERCState *sp = DecoderState(tif);

    (void)s;
    assert(sp != NULL);
    assert(sp->state == LSTATE_INIT_DECODE);

    if (sp->uncompressed_buffer == 0)
    {
        TIFFErrorExtR(tif, module, "Uncompressed buffer not allocated");
        return 0;
    }

    if ((uint64_t)sp->uncompressed_offset + (uint64_t)occ >
        sp->uncompressed_size)
    {
        TIFFErrorExtR(tif, module, "Too many bytes read");
        return 0;
    }

    memcpy(op, sp->uncompressed_buffer + sp->uncompressed_offset, occ);
    sp->uncompressed_offset += (unsigned)occ;

    return 1;
}

static int LERCSetupEncode(TIFF *tif)
{
    LERCState *sp = EncoderState(tif);

    assert(sp != NULL);
    if (sp->state & LSTATE_INIT_DECODE)
    {
        sp->state = 0;
    }

    sp->state |= LSTATE_INIT_ENCODE;

    return 1;
}

/*
 * Reset encoding state at the start of a strip.
 */
static int LERCPreEncode(TIFF *tif, uint16_t s)
{
    static const char module[] = "LERCPreEncode";
    LERCState *sp = EncoderState(tif);
    int lerc_data_type;

    (void)s;
    assert(sp != NULL);
    if (sp->state != LSTATE_INIT_ENCODE)
        tif->tif_setupencode(tif);

    lerc_data_type = GetLercDataType(tif);
    if (lerc_data_type < 0)
        return 0;

    if (!SetupUncompressedBuffer(tif, sp, module))
        return 0;

    return 1;
}

/*
 * Encode a chunk of pixels.
 */
static int LERCEncode(TIFF *tif, uint8_t *bp, tmsize_t cc, uint16_t s)
{
    static const char module[] = "LERCEncode";
    LERCState *sp = EncoderState(tif);

    (void)s;
    assert(sp != NULL);
    assert(sp->state == LSTATE_INIT_ENCODE);

    if ((uint64_t)sp->uncompressed_offset + (uint64_t)cc >
        sp->uncompressed_size)
    {
        TIFFErrorExtR(tif, module, "Too many bytes written");
        return 0;
    }

    memcpy(sp->uncompressed_buffer + sp->uncompressed_offset, bp, cc);
    sp->uncompressed_offset += (unsigned)cc;

    return 1;
}

/*
 * Finish off an encoded strip by flushing it.
 */
static int LERCPostEncode(TIFF *tif)
{
    lerc_status lerc_ret;
    static const char module[] = "LERCPostEncode";
    LERCState *sp = EncoderState(tif);
    unsigned int numBytes = 0;
    unsigned int numBytesWritten = 0;
    TIFFDirectory *td = &tif->tif_dir;
    int use_mask = 0;
    unsigned dst_nbands = td->td_samplesperpixel;

    if (sp->uncompressed_offset != sp->uncompressed_size)
    {
        TIFFErrorExtR(tif, module, "Unexpected number of bytes in the buffer");
        return 0;
    }

    /* Extract alpha mask (if containing only 0 and 255 values, */
    /* and compact array of regular bands */
    if (td->td_planarconfig == PLANARCONFIG_CONTIG && td->td_extrasamples > 0 &&
        td->td_sampleinfo[td->td_extrasamples - 1] == EXTRASAMPLE_UNASSALPHA &&
        GetLercDataType(tif) == 1)
    {
        const unsigned dst_stride =
            (td->td_samplesperpixel - 1) * (td->td_bitspersample / 8);
        const unsigned src_stride =
            td->td_samplesperpixel * (td->td_bitspersample / 8);
        unsigned i = 0;
        const unsigned nb_pixels = sp->segment_width * sp->segment_height;

        use_mask = 1;
        for (i = 0; i < nb_pixels; i++)
        {
            int v = sp->uncompressed_buffer[i * src_stride +
                                            td->td_samplesperpixel - 1];
            if (v != 0 && v != 255)
            {
                use_mask = 0;
                break;
            }
        }

        if (use_mask)
        {
            dst_nbands--;
            /* First pixels must use memmove due to overlapping areas */
            for (i = 0; i < dst_nbands && i < nb_pixels; i++)
            {
                memmove(sp->uncompressed_buffer + i * dst_stride,
                        sp->uncompressed_buffer + i * src_stride, dst_stride);
                sp->mask_buffer[i] =
                    sp->uncompressed_buffer[i * src_stride +
                                            td->td_samplesperpixel - 1];
            }
            for (; i < nb_pixels; i++)
            {
                memcpy(sp->uncompressed_buffer + i * dst_stride,
                       sp->uncompressed_buffer + i * src_stride, dst_stride);
                sp->mask_buffer[i] =
                    sp->uncompressed_buffer[i * src_stride +
                                            td->td_samplesperpixel - 1];
            }
        }
    }
    else if (td->td_sampleformat == SAMPLEFORMAT_IEEEFP &&
             (td->td_planarconfig == PLANARCONFIG_SEPARATE ||
              dst_nbands == 1) &&
             (td->td_bitspersample == 32 || td->td_bitspersample == 64))
    {
        /* Check for NaN values */
        unsigned i;
        const unsigned nb_pixels = sp->segment_width * sp->segment_height;
        if (td->td_bitspersample == 32)
        {
            for (i = 0; i < nb_pixels; i++)
            {
                const float val = ((float *)sp->uncompressed_buffer)[i];
                if (val != val)
                {
                    use_mask = 1;
                    break;
                }
            }
        }
        else
        {
            for (i = 0; i < nb_pixels; i++)
            {
                const double val = ((double *)sp->uncompressed_buffer)[i];
                if (val != val)
                {
                    use_mask = 1;
                    break;
                }
            }
        }

        if (use_mask)
        {
            if (td->td_bitspersample == 32)
            {
                for (i = 0; i < nb_pixels; i++)
                {
                    const float val = ((float *)sp->uncompressed_buffer)[i];
                    sp->mask_buffer[i] = (val == val) ? 255 : 0;
                }
            }
            else
            {
                for (i = 0; i < nb_pixels; i++)
                {
                    const double val = ((double *)sp->uncompressed_buffer)[i];
                    sp->mask_buffer[i] = (val == val) ? 255 : 0;
                }
            }
        }
    }

#if 0
        lerc_ret = lerc_computeCompressedSize(
            sp->uncompressed_buffer,
            sp->lerc_version,
            GetLercDataType(tif),
            td->td_planarconfig == PLANARCONFIG_CONTIG ?
                dst_nbands : 1,
            sp->segment_width,
            sp->segment_height,
            1,
            use_mask ? sp->mask_buffer : NULL,
            sp->maxzerror,
            &numBytes);
        if( lerc_ret != 0 )
        {
            TIFFErrorExtR(tif, module,
                         "lerc_computeCompressedSize() failed");
            return 0;
        }
#else
    numBytes = sp->uncompressed_alloc;
#endif

    if (sp->compressed_size < numBytes)
    {
        _TIFFfreeExt(tif, sp->compressed_buffer);
        sp->compressed_buffer = _TIFFmallocExt(tif, numBytes);
        if (!sp->compressed_buffer)
        {
            sp->compressed_size = 0;
            return 0;
        }
        sp->compressed_size = numBytes;
    }

    lerc_ret = lerc_encodeForVersion(
        sp->uncompressed_buffer, sp->lerc_version, GetLercDataType(tif),
        td->td_planarconfig == PLANARCONFIG_CONTIG ? dst_nbands : 1,
        sp->segment_width, sp->segment_height, 1,
#if LERC_AT_LEAST_VERSION(3, 0, 0)
        use_mask ? 1 : 0,
#endif
        use_mask ? sp->mask_buffer : NULL, sp->maxzerror, sp->compressed_buffer,
        sp->compressed_size, &numBytesWritten);
    if (lerc_ret != 0)
    {
        TIFFErrorExtR(tif, module, "lerc_encode() failed");
        return 0;
    }
    assert(numBytesWritten < numBytes);

    if (sp->additional_compression == LERC_ADD_COMPRESSION_DEFLATE)
    {
#if LIBDEFLATE_SUPPORT
        if (sp->libdeflate_enc == NULL)
        {
            /* To get results as good as zlib, we ask for an extra */
            /* level of compression */
            sp->libdeflate_enc = libdeflate_alloc_compressor(
                sp->zipquality == Z_DEFAULT_COMPRESSION ? 7
                : sp->zipquality >= 6 && sp->zipquality <= 9
                    ? sp->zipquality + 1
                    : sp->zipquality);
            if (sp->libdeflate_enc == NULL)
            {
                TIFFErrorExtR(tif, module, "Cannot allocate compressor");
                return 0;
            }
        }

        /* Should not happen normally */
        if (libdeflate_zlib_compress_bound(
                sp->libdeflate_enc, numBytesWritten) > sp->uncompressed_alloc)
        {
            TIFFErrorExtR(tif, module,
                          "Output buffer for libdeflate too small");
            return 0;
        }

        tif->tif_rawcc = libdeflate_zlib_compress(
            sp->libdeflate_enc, sp->compressed_buffer, numBytesWritten,
            sp->uncompressed_buffer, sp->uncompressed_alloc);

        if (tif->tif_rawcc == 0)
        {
            TIFFErrorExtR(tif, module, "Encoder error at scanline %lu",
                          (unsigned long)tif->tif_row);
            return 0;
        }
#else
        z_stream strm;
        int zlib_ret;
        int cappedQuality = sp->zipquality;
        if (cappedQuality > Z_BEST_COMPRESSION)
            cappedQuality = Z_BEST_COMPRESSION;

        memset(&strm, 0, sizeof(strm));
        strm.zalloc = NULL;
        strm.zfree = NULL;
        strm.opaque = NULL;
        zlib_ret = deflateInit(&strm, cappedQuality);
        if (zlib_ret != Z_OK)
        {
            TIFFErrorExtR(tif, module, "deflateInit() failed");
            return 0;
        }

        strm.avail_in = numBytesWritten;
        strm.next_in = sp->compressed_buffer;
        strm.avail_out = sp->uncompressed_alloc;
        strm.next_out = sp->uncompressed_buffer;
        zlib_ret = deflate(&strm, Z_FINISH);
        if (zlib_ret == Z_STREAM_END)
        {
            tif->tif_rawcc = sp->uncompressed_alloc - strm.avail_out;
        }
        deflateEnd(&strm);
        if (zlib_ret != Z_STREAM_END)
        {
            TIFFErrorExtR(tif, module, "deflate() failed");
            return 0;
        }
#endif
        {
            int ret;
            uint8_t *tif_rawdata_backup = tif->tif_rawdata;
            tif->tif_rawdata = sp->uncompressed_buffer;
            ret = TIFFFlushData1(tif);
            tif->tif_rawdata = tif_rawdata_backup;
            if (!ret)
            {
                return 0;
            }
        }
    }
    else if (sp->additional_compression == LERC_ADD_COMPRESSION_ZSTD)
    {
#ifdef ZSTD_SUPPORT
        size_t zstd_ret = ZSTD_compress(
            sp->uncompressed_buffer, sp->uncompressed_alloc,
            sp->compressed_buffer, numBytesWritten, sp->zstd_compress_level);
        if (ZSTD_isError(zstd_ret))
        {
            TIFFErrorExtR(tif, module, "Error in ZSTD_compress(): %s",
                          ZSTD_getErrorName(zstd_ret));
            return 0;
        }

        {
            int ret;
            uint8_t *tif_rawdata_backup = tif->tif_rawdata;
            tif->tif_rawdata = sp->uncompressed_buffer;
            tif->tif_rawcc = zstd_ret;
            ret = TIFFFlushData1(tif);
            tif->tif_rawdata = tif_rawdata_backup;
            if (!ret)
            {
                return 0;
            }
        }
#else
        TIFFErrorExtR(tif, module, "ZSTD support missing");
        return 0;
#endif
    }
    else if (sp->additional_compression != LERC_ADD_COMPRESSION_NONE)
    {
        TIFFErrorExtR(tif, module, "Unhandled additional compression");
        return 0;
    }
    else
    {
        int ret;
        uint8_t *tif_rawdata_backup = tif->tif_rawdata;
        tif->tif_rawdata = sp->compressed_buffer;
        tif->tif_rawcc = numBytesWritten;
        ret = TIFFFlushData1(tif);
        tif->tif_rawdata = tif_rawdata_backup;
        if (!ret)
            return 0;
    }

    return 1;
}

static void LERCCleanup(TIFF *tif)
{
    LERCState *sp = LState(tif);

    assert(sp != 0);

    tif->tif_tagmethods.vgetfield = sp->vgetparent;
    tif->tif_tagmethods.vsetfield = sp->vsetparent;

    _TIFFfreeExt(tif, sp->uncompressed_buffer);
    _TIFFfreeExt(tif, sp->compressed_buffer);
    _TIFFfreeExt(tif, sp->mask_buffer);

#if LIBDEFLATE_SUPPORT
    if (sp->libdeflate_dec)
        libdeflate_free_decompressor(sp->libdeflate_dec);
    if (sp->libdeflate_enc)
        libdeflate_free_compressor(sp->libdeflate_enc);
#endif

    _TIFFfreeExt(tif, sp);
    tif->tif_data = NULL;

    _TIFFSetDefaultCompressionState(tif);
}

static const TIFFField LERCFields[] = {
    {TIFFTAG_LERC_PARAMETERS, TIFF_VARIABLE2, TIFF_VARIABLE2, TIFF_LONG, 0,
     TIFF_SETGET_C32_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, FALSE, TRUE,
     "LercParameters", NULL},
    {TIFFTAG_LERC_MAXZERROR, 0, 0, TIFF_ANY, 0, TIFF_SETGET_DOUBLE,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE, "LercMaximumError",
     NULL},
    {TIFFTAG_LERC_VERSION, 0, 0, TIFF_ANY, 0, TIFF_SETGET_UINT32,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, FALSE, FALSE, "LercVersion", NULL},
    {TIFFTAG_LERC_ADD_COMPRESSION, 0, 0, TIFF_ANY, 0, TIFF_SETGET_UINT32,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, FALSE, FALSE,
     "LercAdditionalCompression", NULL},
    {TIFFTAG_ZSTD_LEVEL, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE,
     "ZSTD zstd_compress_level", NULL},
    {TIFFTAG_ZIPQUALITY, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
     TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE, "", NULL},
};

static int LERCVSetFieldBase(TIFF *tif, uint32_t tag, ...)
{
    LERCState *sp = LState(tif);
    int ret;
    va_list ap;
    va_start(ap, tag);
    ret = (*sp->vsetparent)(tif, tag, ap);
    va_end(ap);
    return ret;
}

static int LERCVSetField(TIFF *tif, uint32_t tag, va_list ap)
{
    static const char module[] = "LERCVSetField";
    LERCState *sp = LState(tif);

    switch (tag)
    {
        case TIFFTAG_LERC_PARAMETERS:
        {
            uint32_t count = va_arg(ap, int);
            int *params = va_arg(ap, int *);
            if (count < 2)
            {
                TIFFErrorExtR(tif, module,
                              "Invalid count for LercParameters: %u", count);
                return 0;
            }
            sp->lerc_version = params[0];
            sp->additional_compression = params[1];
            return LERCVSetFieldBase(tif, TIFFTAG_LERC_PARAMETERS, count,
                                     params);
        }
        case TIFFTAG_LERC_MAXZERROR:
            sp->maxzerror = va_arg(ap, double);
            return 1;
        case TIFFTAG_LERC_VERSION:
        {
            int params[2] = {0, 0};
            int version = va_arg(ap, int);
            if (version != LERC_VERSION_2_4)
            {
                TIFFErrorExtR(tif, module, "Invalid value for LercVersion: %d",
                              version);
                return 0;
            }
            sp->lerc_version = version;
            params[0] = sp->lerc_version;
            params[1] = sp->additional_compression;
            return LERCVSetFieldBase(tif, TIFFTAG_LERC_PARAMETERS, 2, params);
        }
        case TIFFTAG_LERC_ADD_COMPRESSION:
        {
            int params[2] = {0, 0};
            int additional_compression = va_arg(ap, int);
#ifndef ZSTD_SUPPORT
            if (additional_compression == LERC_ADD_COMPRESSION_ZSTD)
            {
                TIFFErrorExtR(tif, module,
                              "LERC_ZSTD requested, but ZSTD not available");
                return 0;
            }
#endif
            if (additional_compression != LERC_ADD_COMPRESSION_NONE &&
                additional_compression != LERC_ADD_COMPRESSION_DEFLATE &&
                additional_compression != LERC_ADD_COMPRESSION_ZSTD)
            {
                TIFFErrorExtR(tif, module,
                              "Invalid value for LercAdditionalCompression: %d",
                              additional_compression);
                return 0;
            }
            sp->additional_compression = additional_compression;
            params[0] = sp->lerc_version;
            params[1] = sp->additional_compression;
            return LERCVSetFieldBase(tif, TIFFTAG_LERC_PARAMETERS, 2, params);
        }
#ifdef ZSTD_SUPPORT
        case TIFFTAG_ZSTD_LEVEL:
        {
            sp->zstd_compress_level = (int)va_arg(ap, int);
            if (sp->zstd_compress_level <= 0 ||
                sp->zstd_compress_level > ZSTD_maxCLevel())
            {
                TIFFWarningExtR(tif, module,
                                "ZSTD_LEVEL should be between 1 and %d",
                                ZSTD_maxCLevel());
            }
            return 1;
        }
#endif
        case TIFFTAG_ZIPQUALITY:
        {
            sp->zipquality = (int)va_arg(ap, int);
            if (sp->zipquality < Z_DEFAULT_COMPRESSION ||
                sp->zipquality > LIBDEFLATE_MAX_COMPRESSION_LEVEL)
            {
                TIFFErrorExtR(
                    tif, module,
                    "Invalid ZipQuality value. Should be in [-1,%d] range",
                    LIBDEFLATE_MAX_COMPRESSION_LEVEL);
                return 0;
            }

#if LIBDEFLATE_SUPPORT
            if (sp->libdeflate_enc)
            {
                libdeflate_free_compressor(sp->libdeflate_enc);
                sp->libdeflate_enc = NULL;
            }
#endif

            return (1);
        }
        default:
            return (*sp->vsetparent)(tif, tag, ap);
    }
    /*NOTREACHED*/
}

static int LERCVGetField(TIFF *tif, uint32_t tag, va_list ap)
{
    LERCState *sp = LState(tif);

    switch (tag)
    {
        case TIFFTAG_LERC_MAXZERROR:
            *va_arg(ap, double *) = sp->maxzerror;
            break;
        case TIFFTAG_LERC_VERSION:
            *va_arg(ap, int *) = sp->lerc_version;
            break;
        case TIFFTAG_LERC_ADD_COMPRESSION:
            *va_arg(ap, int *) = sp->additional_compression;
            break;
        case TIFFTAG_ZSTD_LEVEL:
            *va_arg(ap, int *) = sp->zstd_compress_level;
            break;
        case TIFFTAG_ZIPQUALITY:
            *va_arg(ap, int *) = sp->zipquality;
            break;
        default:
            return (*sp->vgetparent)(tif, tag, ap);
    }
    return 1;
}

int TIFFInitLERC(TIFF *tif, int scheme)
{
    static const char module[] = "TIFFInitLERC";
    LERCState *sp;

    (void)scheme;
    assert(scheme == COMPRESSION_LERC);

    /*
     * Merge codec-specific tag information.
     */
    if (!_TIFFMergeFields(tif, LERCFields, TIFFArrayCount(LERCFields)))
    {
        TIFFErrorExtR(tif, module, "Merging LERC codec-specific tags failed");
        return 0;
    }

    /*
     * Allocate state block so tag methods have storage to record values.
     */
    tif->tif_data = (uint8_t *)_TIFFcallocExt(tif, 1, sizeof(LERCState));
    if (tif->tif_data == NULL)
        goto bad;
    sp = LState(tif);

    /*
     * Override parent get/set field methods.
     */
    sp->vgetparent = tif->tif_tagmethods.vgetfield;
    tif->tif_tagmethods.vgetfield = LERCVGetField; /* hook for codec tags */
    sp->vsetparent = tif->tif_tagmethods.vsetfield;
    tif->tif_tagmethods.vsetfield = LERCVSetField; /* hook for codec tags */

    /*
     * Install codec methods.
     */
    tif->tif_fixuptags = LERCFixupTags;
    tif->tif_setupdecode = LERCSetupDecode;
    tif->tif_predecode = LERCPreDecode;
    tif->tif_decoderow = LERCDecode;
    tif->tif_decodestrip = LERCDecode;
    tif->tif_decodetile = LERCDecode;
    tif->tif_setupencode = LERCSetupEncode;
    tif->tif_preencode = LERCPreEncode;
    tif->tif_postencode = LERCPostEncode;
    tif->tif_encoderow = LERCEncode;
    tif->tif_encodestrip = LERCEncode;
    tif->tif_encodetile = LERCEncode;
    tif->tif_cleanup = LERCCleanup;

    /* Default values for codec-specific fields */
    TIFFSetField(tif, TIFFTAG_LERC_VERSION, LERC_VERSION_2_4);
    TIFFSetField(tif, TIFFTAG_LERC_ADD_COMPRESSION, LERC_ADD_COMPRESSION_NONE);
    sp->maxzerror = 0.0;
    sp->zstd_compress_level = 9;            /* default comp. level */
    sp->zipquality = Z_DEFAULT_COMPRESSION; /* default comp. level */
    sp->state = 0;

    return 1;
bad:
    TIFFErrorExtR(tif, module, "No space for LERC state block");
    return 0;
}
#endif /* LERC_SUPPORT */
