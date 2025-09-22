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
 * Scanline-oriented Write Support
 */
#include "tiffiop.h"
#include <stdio.h>

#define STRIPINCR 20 /* expansion factor on strip array */

#define WRITECHECKSTRIPS(tif, module)                                          \
    (((tif)->tif_flags & TIFF_BEENWRITING) || TIFFWriteCheck((tif), 0, module))
#define WRITECHECKTILES(tif, module)                                           \
    (((tif)->tif_flags & TIFF_BEENWRITING) || TIFFWriteCheck((tif), 1, module))
#define BUFFERCHECK(tif)                                                       \
    ((((tif)->tif_flags & TIFF_BUFFERSETUP) && tif->tif_rawdata) ||            \
     TIFFWriteBufferSetup((tif), NULL, (tmsize_t)-1))

static int TIFFGrowStrips(TIFF *tif, uint32_t delta, const char *module);
static int TIFFAppendToStrip(TIFF *tif, uint32_t strip, uint8_t *data,
                             tmsize_t cc);

int TIFFWriteScanline(TIFF *tif, void *buf, uint32_t row, uint16_t sample)
{
    static const char module[] = "TIFFWriteScanline";
    register TIFFDirectory *td;
    int status, imagegrew = 0;
    uint32_t strip;

    if (!WRITECHECKSTRIPS(tif, module))
        return (-1);
    /*
     * Handle delayed allocation of data buffer.  This
     * permits it to be sized more intelligently (using
     * directory information).
     */
    if (!BUFFERCHECK(tif))
        return (-1);
    tif->tif_flags |= TIFF_BUF4WRITE; /* not strictly sure this is right*/

    td = &tif->tif_dir;
    /*
     * Extend image length if needed
     * (but only for PlanarConfig=1).
     */
    if (row >= td->td_imagelength)
    { /* extend image */
        if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
        {
            TIFFErrorExtR(
                tif, module,
                "Can not change \"ImageLength\" when using separate planes");
            return (-1);
        }
        td->td_imagelength = row + 1;
        imagegrew = 1;
    }
    /*
     * Calculate strip and check for crossings.
     */
    if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
    {
        if (sample >= td->td_samplesperpixel)
        {
            TIFFErrorExtR(tif, module, "%lu: Sample out of range, max %lu",
                          (unsigned long)sample,
                          (unsigned long)td->td_samplesperpixel);
            return (-1);
        }
        strip = sample * td->td_stripsperimage + row / td->td_rowsperstrip;
    }
    else
        strip = row / td->td_rowsperstrip;
    /*
     * Check strip array to make sure there's space. We don't support
     * dynamically growing files that have data organized in separate
     * bitplanes because it's too painful.  In that case we require that
     * the imagelength be set properly before the first write (so that the
     * strips array will be fully allocated above).
     */
    if (strip >= td->td_nstrips && !TIFFGrowStrips(tif, 1, module))
        return (-1);
    if (strip != tif->tif_curstrip)
    {
        /*
         * Changing strips -- flush any data present.
         */
        if (!TIFFFlushData(tif))
            return (-1);
        tif->tif_curstrip = strip;
        /*
         * Watch out for a growing image.  The value of strips/image
         * will initially be 1 (since it can't be deduced until the
         * imagelength is known).
         */
        if (strip >= td->td_stripsperimage && imagegrew)
            td->td_stripsperimage =
                TIFFhowmany_32(td->td_imagelength, td->td_rowsperstrip);
        if (td->td_stripsperimage == 0)
        {
            TIFFErrorExtR(tif, module, "Zero strips per image");
            return (-1);
        }
        tif->tif_row = (strip % td->td_stripsperimage) * td->td_rowsperstrip;
        if ((tif->tif_flags & TIFF_CODERSETUP) == 0)
        {
            if (!(*tif->tif_setupencode)(tif))
                return (-1);
            tif->tif_flags |= TIFF_CODERSETUP;
        }

        tif->tif_rawcc = 0;
        tif->tif_rawcp = tif->tif_rawdata;

        /* this informs TIFFAppendToStrip() we have changed strip */
        tif->tif_curoff = 0;

        if (!(*tif->tif_preencode)(tif, sample))
            return (-1);
        tif->tif_flags |= TIFF_POSTENCODE;
    }
    /*
     * Ensure the write is either sequential or at the
     * beginning of a strip (or that we can randomly
     * access the data -- i.e. no encoding).
     */
    if (row != tif->tif_row)
    {
        if (row < tif->tif_row)
        {
            /*
             * Moving backwards within the same strip:
             * backup to the start and then decode
             * forward (below).
             */
            tif->tif_row =
                (strip % td->td_stripsperimage) * td->td_rowsperstrip;
            tif->tif_rawcp = tif->tif_rawdata;
        }
        /*
         * Seek forward to the desired row.
         */
        if (!(*tif->tif_seek)(tif, row - tif->tif_row))
            return (-1);
        tif->tif_row = row;
    }

    /* swab if needed - note that source buffer will be altered */
    tif->tif_postdecode(tif, (uint8_t *)buf, tif->tif_scanlinesize);

    status = (*tif->tif_encoderow)(tif, (uint8_t *)buf, tif->tif_scanlinesize,
                                   sample);

    /* we are now poised at the beginning of the next row */
    tif->tif_row = row + 1;
    return (status);
}

/* Make sure that at the first attempt of rewriting a tile/strip, we will have
 */
/* more bytes available in the output buffer than the previous byte count, */
/* so that TIFFAppendToStrip() will detect the overflow when it is called the
 * first */
/* time if the new compressed tile is bigger than the older one. (GDAL #4771) */
static int _TIFFReserveLargeEnoughWriteBuffer(TIFF *tif, uint32_t strip_or_tile)
{
    TIFFDirectory *td = &tif->tif_dir;
    if (td->td_stripbytecount_p[strip_or_tile] > 0)
    {
        /* The +1 is to ensure at least one extra bytes */
        /* The +4 is because the LZW encoder flushes 4 bytes before the limit */
        uint64_t safe_buffer_size =
            (uint64_t)(td->td_stripbytecount_p[strip_or_tile] + 1 + 4);
        if (tif->tif_rawdatasize <= (tmsize_t)safe_buffer_size)
        {
            if (!(TIFFWriteBufferSetup(
                    tif, NULL,
                    (tmsize_t)TIFFroundup_64(safe_buffer_size, 1024))))
                return 0;
        }
    }
    return 1;
}

/*
 * Encode the supplied data and write it to the
 * specified strip.
 *
 * NB: Image length must be setup before writing.
 */
tmsize_t TIFFWriteEncodedStrip(TIFF *tif, uint32_t strip, void *data,
                               tmsize_t cc)
{
    static const char module[] = "TIFFWriteEncodedStrip";
    TIFFDirectory *td = &tif->tif_dir;
    uint16_t sample;

    if (!WRITECHECKSTRIPS(tif, module))
        return ((tmsize_t)-1);
    /*
     * Check strip array to make sure there's space.
     * We don't support dynamically growing files that
     * have data organized in separate bitplanes because
     * it's too painful.  In that case we require that
     * the imagelength be set properly before the first
     * write (so that the strips array will be fully
     * allocated above).
     */
    if (strip >= td->td_nstrips)
    {
        if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
        {
            TIFFErrorExtR(
                tif, module,
                "Can not grow image by strips when using separate planes");
            return ((tmsize_t)-1);
        }
        if (!TIFFGrowStrips(tif, 1, module))
            return ((tmsize_t)-1);
        td->td_stripsperimage =
            TIFFhowmany_32(td->td_imagelength, td->td_rowsperstrip);
    }
    /*
     * Handle delayed allocation of data buffer.  This
     * permits it to be sized according to the directory
     * info.
     */
    if (!BUFFERCHECK(tif))
        return ((tmsize_t)-1);

    tif->tif_flags |= TIFF_BUF4WRITE;

    tif->tif_curstrip = strip;

    /* this informs TIFFAppendToStrip() we have changed or reset strip */
    tif->tif_curoff = 0;

    if (!_TIFFReserveLargeEnoughWriteBuffer(tif, strip))
    {
        return ((tmsize_t)(-1));
    }

    tif->tif_rawcc = 0;
    tif->tif_rawcp = tif->tif_rawdata;

    if (td->td_stripsperimage == 0)
    {
        TIFFErrorExtR(tif, module, "Zero strips per image");
        return ((tmsize_t)-1);
    }

    tif->tif_row = (strip % td->td_stripsperimage) * td->td_rowsperstrip;
    if ((tif->tif_flags & TIFF_CODERSETUP) == 0)
    {
        if (!(*tif->tif_setupencode)(tif))
            return ((tmsize_t)-1);
        tif->tif_flags |= TIFF_CODERSETUP;
    }

    tif->tif_flags &= ~TIFF_POSTENCODE;

    /* shortcut to avoid an extra memcpy() */
    if (td->td_compression == COMPRESSION_NONE)
    {
        /* swab if needed - note that source buffer will be altered */
        tif->tif_postdecode(tif, (uint8_t *)data, cc);

        if (!isFillOrder(tif, td->td_fillorder) &&
            (tif->tif_flags & TIFF_NOBITREV) == 0)
            TIFFReverseBits((uint8_t *)data, cc);

        if (cc > 0 && !TIFFAppendToStrip(tif, strip, (uint8_t *)data, cc))
            return ((tmsize_t)-1);
        return (cc);
    }

    sample = (uint16_t)(strip / td->td_stripsperimage);
    if (!(*tif->tif_preencode)(tif, sample))
        return ((tmsize_t)-1);

    /* swab if needed - note that source buffer will be altered */
    tif->tif_postdecode(tif, (uint8_t *)data, cc);

    if (!(*tif->tif_encodestrip)(tif, (uint8_t *)data, cc, sample))
        return ((tmsize_t)-1);
    if (!(*tif->tif_postencode)(tif))
        return ((tmsize_t)-1);
    if (!isFillOrder(tif, td->td_fillorder) &&
        (tif->tif_flags & TIFF_NOBITREV) == 0)
        TIFFReverseBits(tif->tif_rawdata, tif->tif_rawcc);
    if (tif->tif_rawcc > 0 &&
        !TIFFAppendToStrip(tif, strip, tif->tif_rawdata, tif->tif_rawcc))
        return ((tmsize_t)-1);
    tif->tif_rawcc = 0;
    tif->tif_rawcp = tif->tif_rawdata;
    return (cc);
}

/*
 * Write the supplied data to the specified strip.
 *
 * NB: Image length must be setup before writing.
 */
tmsize_t TIFFWriteRawStrip(TIFF *tif, uint32_t strip, void *data, tmsize_t cc)
{
    static const char module[] = "TIFFWriteRawStrip";
    TIFFDirectory *td = &tif->tif_dir;

    if (!WRITECHECKSTRIPS(tif, module))
        return ((tmsize_t)-1);
    /*
     * Check strip array to make sure there's space.
     * We don't support dynamically growing files that
     * have data organized in separate bitplanes because
     * it's too painful.  In that case we require that
     * the imagelength be set properly before the first
     * write (so that the strips array will be fully
     * allocated above).
     */
    if (strip >= td->td_nstrips)
    {
        if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
        {
            TIFFErrorExtR(
                tif, module,
                "Can not grow image by strips when using separate planes");
            return ((tmsize_t)-1);
        }
        /*
         * Watch out for a growing image.  The value of
         * strips/image will initially be 1 (since it
         * can't be deduced until the imagelength is known).
         */
        if (strip >= td->td_stripsperimage)
            td->td_stripsperimage =
                TIFFhowmany_32(td->td_imagelength, td->td_rowsperstrip);
        if (!TIFFGrowStrips(tif, 1, module))
            return ((tmsize_t)-1);
    }

    if (tif->tif_curstrip != strip)
    {
        tif->tif_curstrip = strip;

        /* this informs TIFFAppendToStrip() we have changed or reset strip */
        tif->tif_curoff = 0;
    }

    if (td->td_stripsperimage == 0)
    {
        TIFFErrorExtR(tif, module, "Zero strips per image");
        return ((tmsize_t)-1);
    }
    tif->tif_row = (strip % td->td_stripsperimage) * td->td_rowsperstrip;
    return (TIFFAppendToStrip(tif, strip, (uint8_t *)data, cc) ? cc
                                                               : (tmsize_t)-1);
}

/*
 * Write and compress a tile of data.  The
 * tile is selected by the (x,y,z,s) coordinates.
 */
tmsize_t TIFFWriteTile(TIFF *tif, void *buf, uint32_t x, uint32_t y, uint32_t z,
                       uint16_t s)
{
    if (!TIFFCheckTile(tif, x, y, z, s))
        return ((tmsize_t)(-1));
    /*
     * NB: A tile size of -1 is used instead of tif_tilesize knowing
     *     that TIFFWriteEncodedTile will clamp this to the tile size.
     *     This is done because the tile size may not be defined until
     *     after the output buffer is setup in TIFFWriteBufferSetup.
     */
    return (TIFFWriteEncodedTile(tif, TIFFComputeTile(tif, x, y, z, s), buf,
                                 (tmsize_t)(-1)));
}

/*
 * Encode the supplied data and write it to the
 * specified tile.  There must be space for the
 * data.  The function clamps individual writes
 * to a tile to the tile size, but does not (and
 * can not) check that multiple writes to the same
 * tile do not write more than tile size data.
 *
 * NB: Image length must be setup before writing; this
 *     interface does not support automatically growing
 *     the image on each write (as TIFFWriteScanline does).
 */
tmsize_t TIFFWriteEncodedTile(TIFF *tif, uint32_t tile, void *data, tmsize_t cc)
{
    static const char module[] = "TIFFWriteEncodedTile";
    TIFFDirectory *td;
    uint16_t sample;
    uint32_t howmany32;

    if (!WRITECHECKTILES(tif, module))
        return ((tmsize_t)(-1));
    td = &tif->tif_dir;
    if (tile >= td->td_nstrips)
    {
        TIFFErrorExtR(tif, module, "Tile %lu out of range, max %lu",
                      (unsigned long)tile, (unsigned long)td->td_nstrips);
        return ((tmsize_t)(-1));
    }
    /*
     * Handle delayed allocation of data buffer.  This
     * permits it to be sized more intelligently (using
     * directory information).
     */
    if (!BUFFERCHECK(tif))
        return ((tmsize_t)(-1));

    tif->tif_flags |= TIFF_BUF4WRITE;

    tif->tif_curtile = tile;

    /* this informs TIFFAppendToStrip() we have changed or reset tile */
    tif->tif_curoff = 0;

    if (!_TIFFReserveLargeEnoughWriteBuffer(tif, tile))
    {
        return ((tmsize_t)(-1));
    }

    tif->tif_rawcc = 0;
    tif->tif_rawcp = tif->tif_rawdata;

    /*
     * Compute tiles per row & per column to compute
     * current row and column
     */
    howmany32 = TIFFhowmany_32(td->td_imagelength, td->td_tilelength);
    if (howmany32 == 0)
    {
        TIFFErrorExtR(tif, module, "Zero tiles");
        return ((tmsize_t)(-1));
    }
    tif->tif_row = (tile % howmany32) * td->td_tilelength;
    howmany32 = TIFFhowmany_32(td->td_imagewidth, td->td_tilewidth);
    if (howmany32 == 0)
    {
        TIFFErrorExtR(tif, module, "Zero tiles");
        return ((tmsize_t)(-1));
    }
    tif->tif_col = (tile % howmany32) * td->td_tilewidth;

    if ((tif->tif_flags & TIFF_CODERSETUP) == 0)
    {
        if (!(*tif->tif_setupencode)(tif))
            return ((tmsize_t)(-1));
        tif->tif_flags |= TIFF_CODERSETUP;
    }
    tif->tif_flags &= ~TIFF_POSTENCODE;

    /*
     * Clamp write amount to the tile size.  This is mostly
     * done so that callers can pass in some large number
     * (e.g. -1) and have the tile size used instead.
     */
    if (cc < 1 || cc > tif->tif_tilesize)
        cc = tif->tif_tilesize;

    /* shortcut to avoid an extra memcpy() */
    if (td->td_compression == COMPRESSION_NONE)
    {
        /* swab if needed - note that source buffer will be altered */
        tif->tif_postdecode(tif, (uint8_t *)data, cc);

        if (!isFillOrder(tif, td->td_fillorder) &&
            (tif->tif_flags & TIFF_NOBITREV) == 0)
            TIFFReverseBits((uint8_t *)data, cc);

        if (cc > 0 && !TIFFAppendToStrip(tif, tile, (uint8_t *)data, cc))
            return ((tmsize_t)-1);
        return (cc);
    }

    sample = (uint16_t)(tile / td->td_stripsperimage);
    if (!(*tif->tif_preencode)(tif, sample))
        return ((tmsize_t)(-1));
    /* swab if needed - note that source buffer will be altered */
    tif->tif_postdecode(tif, (uint8_t *)data, cc);

    if (!(*tif->tif_encodetile)(tif, (uint8_t *)data, cc, sample))
        return ((tmsize_t)-1);
    if (!(*tif->tif_postencode)(tif))
        return ((tmsize_t)(-1));
    if (!isFillOrder(tif, td->td_fillorder) &&
        (tif->tif_flags & TIFF_NOBITREV) == 0)
        TIFFReverseBits((uint8_t *)tif->tif_rawdata, tif->tif_rawcc);
    if (tif->tif_rawcc > 0 &&
        !TIFFAppendToStrip(tif, tile, tif->tif_rawdata, tif->tif_rawcc))
        return ((tmsize_t)(-1));
    tif->tif_rawcc = 0;
    tif->tif_rawcp = tif->tif_rawdata;
    return (cc);
}

/*
 * Write the supplied data to the specified strip.
 * There must be space for the data; we don't check
 * if strips overlap!
 *
 * NB: Image length must be setup before writing; this
 *     interface does not support automatically growing
 *     the image on each write (as TIFFWriteScanline does).
 */
tmsize_t TIFFWriteRawTile(TIFF *tif, uint32_t tile, void *data, tmsize_t cc)
{
    static const char module[] = "TIFFWriteRawTile";

    if (!WRITECHECKTILES(tif, module))
        return ((tmsize_t)(-1));
    if (tile >= tif->tif_dir.td_nstrips)
    {
        TIFFErrorExtR(tif, module, "Tile %lu out of range, max %lu",
                      (unsigned long)tile,
                      (unsigned long)tif->tif_dir.td_nstrips);
        return ((tmsize_t)(-1));
    }
    return (TIFFAppendToStrip(tif, tile, (uint8_t *)data, cc) ? cc
                                                              : (tmsize_t)(-1));
}

#define isUnspecified(tif, f)                                                  \
    (TIFFFieldSet(tif, f) && (tif)->tif_dir.td_imagelength == 0)

int TIFFSetupStrips(TIFF *tif)
{
    TIFFDirectory *td = &tif->tif_dir;

    if (isTiled(tif))
        td->td_stripsperimage = isUnspecified(tif, FIELD_TILEDIMENSIONS)
                                    ? td->td_samplesperpixel
                                    : TIFFNumberOfTiles(tif);
    else
        td->td_stripsperimage = isUnspecified(tif, FIELD_ROWSPERSTRIP)
                                    ? td->td_samplesperpixel
                                    : TIFFNumberOfStrips(tif);
    td->td_nstrips = td->td_stripsperimage;
    /* TIFFWriteDirectoryTagData has a limitation to 0x80000000U bytes */
    if (td->td_nstrips >=
        0x80000000U / ((tif->tif_flags & TIFF_BIGTIFF) ? 0x8U : 0x4U))
    {
        TIFFErrorExtR(tif, "TIFFSetupStrips",
                      "Too large Strip/Tile Offsets/ByteCounts arrays");
        return 0;
    }
    if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
        td->td_stripsperimage /= td->td_samplesperpixel;

    if (td->td_stripoffset_p != NULL)
        _TIFFfreeExt(tif, td->td_stripoffset_p);
    td->td_stripoffset_p = (uint64_t *)_TIFFCheckMalloc(
        tif, td->td_nstrips, sizeof(uint64_t), "for \"StripOffsets\" array");
    if (td->td_stripbytecount_p != NULL)
        _TIFFfreeExt(tif, td->td_stripbytecount_p);
    td->td_stripbytecount_p = (uint64_t *)_TIFFCheckMalloc(
        tif, td->td_nstrips, sizeof(uint64_t), "for \"StripByteCounts\" array");
    if (td->td_stripoffset_p == NULL || td->td_stripbytecount_p == NULL)
        return (0);
    /*
     * Place data at the end-of-file
     * (by setting offsets to zero).
     */
    _TIFFmemset(td->td_stripoffset_p, 0, td->td_nstrips * sizeof(uint64_t));
    _TIFFmemset(td->td_stripbytecount_p, 0, td->td_nstrips * sizeof(uint64_t));
    TIFFSetFieldBit(tif, FIELD_STRIPOFFSETS);
    TIFFSetFieldBit(tif, FIELD_STRIPBYTECOUNTS);
    return (1);
}
#undef isUnspecified

/*
 * Verify file is writable and that the directory
 * information is setup properly.  In doing the latter
 * we also "freeze" the state of the directory so
 * that important information is not changed.
 */
int TIFFWriteCheck(TIFF *tif, int tiles, const char *module)
{
    if (tif->tif_mode == O_RDONLY)
    {
        TIFFErrorExtR(tif, module, "File not open for writing");
        return (0);
    }
    if (tiles ^ isTiled(tif))
    {
        TIFFErrorExtR(tif, module,
                      tiles ? "Can not write tiles to a striped image"
                            : "Can not write scanlines to a tiled image");
        return (0);
    }

    _TIFFFillStriles(tif);

    /*
     * On the first write verify all the required information
     * has been setup and initialize any data structures that
     * had to wait until directory information was set.
     * Note that a lot of our work is assumed to remain valid
     * because we disallow any of the important parameters
     * from changing after we start writing (i.e. once
     * TIFF_BEENWRITING is set, TIFFSetField will only allow
     * the image's length to be changed).
     */
    if (!TIFFFieldSet(tif, FIELD_IMAGEDIMENSIONS))
    {
        TIFFErrorExtR(tif, module,
                      "Must set \"ImageWidth\" before writing data");
        return (0);
    }
    if (tif->tif_dir.td_stripoffset_p == NULL && !TIFFSetupStrips(tif))
    {
        tif->tif_dir.td_nstrips = 0;
        TIFFErrorExtR(tif, module, "No space for %s arrays",
                      isTiled(tif) ? "tile" : "strip");
        return (0);
    }
    if (isTiled(tif))
    {
        tif->tif_tilesize = TIFFTileSize(tif);
        if (tif->tif_tilesize == 0)
            return (0);
    }
    else
        tif->tif_tilesize = (tmsize_t)(-1);
    tif->tif_scanlinesize = TIFFScanlineSize(tif);
    if (tif->tif_scanlinesize == 0)
        return (0);
    tif->tif_flags |= TIFF_BEENWRITING;

    if (tif->tif_dir.td_stripoffset_entry.tdir_tag != 0 &&
        tif->tif_dir.td_stripoffset_entry.tdir_count == 0 &&
        tif->tif_dir.td_stripoffset_entry.tdir_type == 0 &&
        tif->tif_dir.td_stripoffset_entry.tdir_offset.toff_long8 == 0 &&
        tif->tif_dir.td_stripbytecount_entry.tdir_tag != 0 &&
        tif->tif_dir.td_stripbytecount_entry.tdir_count == 0 &&
        tif->tif_dir.td_stripbytecount_entry.tdir_type == 0 &&
        tif->tif_dir.td_stripbytecount_entry.tdir_offset.toff_long8 == 0 &&
        !(tif->tif_flags & TIFF_DIRTYDIRECT))
    {
        TIFFForceStrileArrayWriting(tif);
    }

    return (1);
}

/*
 * Setup the raw data buffer used for encoding.
 */
int TIFFWriteBufferSetup(TIFF *tif, void *bp, tmsize_t size)
{
    static const char module[] = "TIFFWriteBufferSetup";

    if (tif->tif_rawdata)
    {
        if (tif->tif_flags & TIFF_MYBUFFER)
        {
            _TIFFfreeExt(tif, tif->tif_rawdata);
            tif->tif_flags &= ~TIFF_MYBUFFER;
        }
        tif->tif_rawdata = NULL;
    }
    if (size == (tmsize_t)(-1))
    {
        size = (isTiled(tif) ? tif->tif_tilesize : TIFFStripSize(tif));

        /* Adds 10% margin for cases where compression would expand a bit */
        if (size < TIFF_TMSIZE_T_MAX - size / 10)
            size += size / 10;
        /*
         * Make raw data buffer at least 8K
         */
        if (size < 8 * 1024)
            size = 8 * 1024;
        bp = NULL; /* NB: force malloc */
    }
    if (bp == NULL)
    {
        bp = _TIFFmallocExt(tif, size);
        if (bp == NULL)
        {
            TIFFErrorExtR(tif, module, "No space for output buffer");
            return (0);
        }
        tif->tif_flags |= TIFF_MYBUFFER;
    }
    else
        tif->tif_flags &= ~TIFF_MYBUFFER;
    tif->tif_rawdata = (uint8_t *)bp;
    tif->tif_rawdatasize = size;
    tif->tif_rawcc = 0;
    tif->tif_rawcp = tif->tif_rawdata;
    tif->tif_flags |= TIFF_BUFFERSETUP;
    return (1);
}

/*
 * Grow the strip data structures by delta strips.
 */
static int TIFFGrowStrips(TIFF *tif, uint32_t delta, const char *module)
{
    TIFFDirectory *td = &tif->tif_dir;
    uint64_t *new_stripoffset;
    uint64_t *new_stripbytecount;

    assert(td->td_planarconfig == PLANARCONFIG_CONTIG);
    new_stripoffset = (uint64_t *)_TIFFreallocExt(
        tif, td->td_stripoffset_p, (td->td_nstrips + delta) * sizeof(uint64_t));
    new_stripbytecount = (uint64_t *)_TIFFreallocExt(
        tif, td->td_stripbytecount_p,
        (td->td_nstrips + delta) * sizeof(uint64_t));
    if (new_stripoffset == NULL || new_stripbytecount == NULL)
    {
        if (new_stripoffset)
            _TIFFfreeExt(tif, new_stripoffset);
        if (new_stripbytecount)
            _TIFFfreeExt(tif, new_stripbytecount);
        td->td_nstrips = 0;
        TIFFErrorExtR(tif, module, "No space to expand strip arrays");
        return (0);
    }
    td->td_stripoffset_p = new_stripoffset;
    td->td_stripbytecount_p = new_stripbytecount;
    _TIFFmemset(td->td_stripoffset_p + td->td_nstrips, 0,
                delta * sizeof(uint64_t));
    _TIFFmemset(td->td_stripbytecount_p + td->td_nstrips, 0,
                delta * sizeof(uint64_t));
    td->td_nstrips += delta;
    tif->tif_flags |= TIFF_DIRTYDIRECT;

    return (1);
}

/*
 * Append the data to the specified strip.
 */
static int TIFFAppendToStrip(TIFF *tif, uint32_t strip, uint8_t *data,
                             tmsize_t cc)
{
    static const char module[] = "TIFFAppendToStrip";
    TIFFDirectory *td = &tif->tif_dir;
    uint64_t m;
    int64_t old_byte_count = -1;

    if (tif->tif_curoff == 0)
        tif->tif_lastvalidoff = 0;

    if (td->td_stripoffset_p[strip] == 0 || tif->tif_curoff == 0)
    {
        assert(td->td_nstrips > 0);

        if (td->td_stripbytecount_p[strip] != 0 &&
            td->td_stripoffset_p[strip] != 0 &&
            td->td_stripbytecount_p[strip] >= (uint64_t)cc)
        {
            /*
             * There is already tile data on disk, and the new tile
             * data we have will fit in the same space.  The only
             * aspect of this that is risky is that there could be
             * more data to append to this strip before we are done
             * depending on how we are getting called.
             */
            if (!SeekOK(tif, td->td_stripoffset_p[strip]))
            {
                TIFFErrorExtR(tif, module, "Seek error at scanline %lu",
                              (unsigned long)tif->tif_row);
                return (0);
            }

            tif->tif_lastvalidoff =
                td->td_stripoffset_p[strip] + td->td_stripbytecount_p[strip];
        }
        else
        {
            /*
             * Seek to end of file, and set that as our location to
             * write this strip.
             */
            td->td_stripoffset_p[strip] = TIFFSeekFile(tif, 0, SEEK_END);
            tif->tif_flags |= TIFF_DIRTYSTRIP;
        }

        tif->tif_curoff = td->td_stripoffset_p[strip];

        /*
         * We are starting a fresh strip/tile, so set the size to zero.
         */
        old_byte_count = td->td_stripbytecount_p[strip];
        td->td_stripbytecount_p[strip] = 0;
    }

    m = tif->tif_curoff + cc;
    if (!(tif->tif_flags & TIFF_BIGTIFF))
        m = (uint32_t)m;
    if ((m < tif->tif_curoff) || (m < (uint64_t)cc))
    {
        TIFFErrorExtR(tif, module, "Maximum TIFF file size exceeded");
        return (0);
    }

    if (tif->tif_lastvalidoff != 0 && m > tif->tif_lastvalidoff &&
        td->td_stripbytecount_p[strip] > 0)
    {
        /* Ouch: we have detected that we are rewriting in place a strip/tile */
        /* with several calls to TIFFAppendToStrip(). The first call was with */
        /* a size smaller than the previous size of the strip/tile, so we */
        /* opted to rewrite in place, but a following call causes us to go */
        /* outsize of the strip/tile area, so we have to finally go for a */
        /* append-at-end-of-file strategy, and start by moving what we already
         */
        /* wrote. */
        tmsize_t tempSize;
        void *temp;
        uint64_t offsetRead;
        uint64_t offsetWrite;
        uint64_t toCopy = td->td_stripbytecount_p[strip];

        if (toCopy < 1024 * 1024)
            tempSize = (tmsize_t)toCopy;
        else
            tempSize = 1024 * 1024;

        offsetRead = td->td_stripoffset_p[strip];
        offsetWrite = TIFFSeekFile(tif, 0, SEEK_END);

        m = offsetWrite + toCopy + cc;
        if (!(tif->tif_flags & TIFF_BIGTIFF) && m != (uint32_t)m)
        {
            TIFFErrorExtR(tif, module, "Maximum TIFF file size exceeded");
            return (0);
        }

        temp = _TIFFmallocExt(tif, tempSize);
        if (temp == NULL)
        {
            TIFFErrorExtR(tif, module, "No space for output buffer");
            return (0);
        }

        tif->tif_flags |= TIFF_DIRTYSTRIP;

        td->td_stripoffset_p[strip] = offsetWrite;
        td->td_stripbytecount_p[strip] = 0;

        /* Move data written by previous calls to us at end of file */
        while (toCopy > 0)
        {
            if (!SeekOK(tif, offsetRead))
            {
                TIFFErrorExtR(tif, module, "Seek error");
                _TIFFfreeExt(tif, temp);
                return (0);
            }
            if (!ReadOK(tif, temp, tempSize))
            {
                TIFFErrorExtR(tif, module, "Cannot read");
                _TIFFfreeExt(tif, temp);
                return (0);
            }
            if (!SeekOK(tif, offsetWrite))
            {
                TIFFErrorExtR(tif, module, "Seek error");
                _TIFFfreeExt(tif, temp);
                return (0);
            }
            if (!WriteOK(tif, temp, tempSize))
            {
                TIFFErrorExtR(tif, module, "Cannot write");
                _TIFFfreeExt(tif, temp);
                return (0);
            }
            offsetRead += tempSize;
            offsetWrite += tempSize;
            td->td_stripbytecount_p[strip] += tempSize;
            toCopy -= tempSize;
        }
        _TIFFfreeExt(tif, temp);

        /* Append the data of this call */
        offsetWrite += cc;
        m = offsetWrite;
    }

    if (!WriteOK(tif, data, cc))
    {
        TIFFErrorExtR(tif, module, "Write error at scanline %lu",
                      (unsigned long)tif->tif_row);
        return (0);
    }
    tif->tif_curoff = m;
    td->td_stripbytecount_p[strip] += cc;

    if ((int64_t)td->td_stripbytecount_p[strip] != old_byte_count)
        tif->tif_flags |= TIFF_DIRTYSTRIP;

    return (1);
}

/*
 * Internal version of TIFFFlushData that can be
 * called by ``encodestrip routines'' w/o concern
 * for infinite recursion.
 */
int TIFFFlushData1(TIFF *tif)
{
    if (tif->tif_rawcc > 0 && tif->tif_flags & TIFF_BUF4WRITE)
    {
        if (!isFillOrder(tif, tif->tif_dir.td_fillorder) &&
            (tif->tif_flags & TIFF_NOBITREV) == 0)
            TIFFReverseBits((uint8_t *)tif->tif_rawdata, tif->tif_rawcc);
        if (!TIFFAppendToStrip(
                tif, isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip,
                tif->tif_rawdata, tif->tif_rawcc))
        {
            /* We update those variables even in case of error since there's */
            /* code that doesn't really check the return code of this */
            /* function */
            tif->tif_rawcc = 0;
            tif->tif_rawcp = tif->tif_rawdata;
            return (0);
        }
        tif->tif_rawcc = 0;
        tif->tif_rawcp = tif->tif_rawdata;
    }
    return (1);
}

/*
 * Set the current write offset.  This should only be
 * used to set the offset to a known previous location
 * (very carefully), or to 0 so that the next write gets
 * appended to the end of the file.
 */
void TIFFSetWriteOffset(TIFF *tif, toff_t off)
{
    tif->tif_curoff = off;
    tif->tif_lastvalidoff = 0;
}
