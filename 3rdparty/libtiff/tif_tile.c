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
 * Tiled Image Support Routines.
 */
#include "tiffiop.h"

/*
 * Compute which tile an (x,y,z,s) value is in.
 */
uint32_t TIFFComputeTile(TIFF *tif, uint32_t x, uint32_t y, uint32_t z,
                         uint16_t s)
{
    TIFFDirectory *td = &tif->tif_dir;
    uint32_t dx = td->td_tilewidth;
    uint32_t dy = td->td_tilelength;
    uint32_t dz = td->td_tiledepth;
    uint32_t tile = 1;

    if (td->td_imagedepth == 1)
        z = 0;
    if (dx == (uint32_t)-1)
        dx = td->td_imagewidth;
    if (dy == (uint32_t)-1)
        dy = td->td_imagelength;
    if (dz == (uint32_t)-1)
        dz = td->td_imagedepth;
    if (dx != 0 && dy != 0 && dz != 0)
    {
        uint32_t xpt = TIFFhowmany_32(td->td_imagewidth, dx);
        uint32_t ypt = TIFFhowmany_32(td->td_imagelength, dy);
        uint32_t zpt = TIFFhowmany_32(td->td_imagedepth, dz);
        uint32_t xpt_ypt = _TIFFMultiply32(tif, xpt, ypt, "TIFFComputeTile");
        uint32_t xpt_ypt_zpt =
            _TIFFMultiply32(tif, xpt_ypt, zpt, "TIFFComputeTile");
        uint64_t z_offset;
        uint64_t y_offset;
        uint64_t tile64;

        if ((xpt_ypt == 0 && xpt != 0 && ypt != 0) ||
            (xpt_ypt_zpt == 0 && xpt_ypt != 0 && zpt != 0))
            return (0);

        z_offset = _TIFFMultiply64(tif, xpt_ypt, z / dz, "TIFFComputeTile");
        y_offset = _TIFFMultiply64(tif, xpt, y / dy, "TIFFComputeTile");
        if ((z_offset == 0 && xpt_ypt != 0 && (z / dz) != 0) ||
            (y_offset == 0 && xpt != 0 && (y / dy) != 0))
            return (0);
        tile64 = _TIFFAdd64(tif, z_offset, y_offset, "TIFFComputeTile");
        if (tile64 == 0 && (z_offset != 0 || y_offset != 0))
            return (0);
        tile64 = _TIFFAdd64(tif, tile64, x / dx, "TIFFComputeTile");
        if (tile64 == 0 && (z_offset != 0 || y_offset != 0 || (x / dx) != 0))
            return (0);
        if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
        {
            uint64_t sample_offset;
            if (s >= td->td_samplesperpixel)
            {
                TIFFErrorExtR(
                    tif, "TIFFComputeTile", "%lu: Sample out of range, max %lu",
                    (unsigned long)s, (unsigned long)td->td_samplesperpixel);
                return (0);
            }
            sample_offset =
                _TIFFMultiply64(tif, xpt_ypt_zpt, s, "TIFFComputeTile");
            if (sample_offset == 0 && xpt_ypt_zpt != 0 && s != 0)
                return (0);
            tile64 = _TIFFAdd64(tif, sample_offset, tile64, "TIFFComputeTile");
            if (tile64 == 0 && (sample_offset != 0 || z_offset != 0 ||
                                y_offset != 0 || (x / dx) != 0))
                return (0);
        }
        tile = _TIFFCastUInt64ToUInt32(tif, tile64, "TIFFComputeTile");
        if (tile == 0 && tile64 != 0)
            return (0);
    }
    return (tile);
}

/*
 * Check an (x,y,z,s) coordinate
 * against the image bounds.
 */
int TIFFCheckTile(TIFF *tif, uint32_t x, uint32_t y, uint32_t z, uint16_t s)
{
    TIFFDirectory *td = &tif->tif_dir;

    if (x >= td->td_imagewidth)
    {
        TIFFErrorExtR(tif, tif->tif_name, "%lu: Col out of range, max %lu",
                      (unsigned long)x, (unsigned long)(td->td_imagewidth - 1));
        return (0);
    }
    if (y >= td->td_imagelength)
    {
        TIFFErrorExtR(tif, tif->tif_name, "%lu: Row out of range, max %lu",
                      (unsigned long)y,
                      (unsigned long)(td->td_imagelength - 1));
        return (0);
    }
    if (z >= td->td_imagedepth)
    {
        TIFFErrorExtR(tif, tif->tif_name, "%lu: Depth out of range, max %lu",
                      (unsigned long)z, (unsigned long)(td->td_imagedepth - 1));
        return (0);
    }
    if (td->td_planarconfig == PLANARCONFIG_SEPARATE &&
        s >= td->td_samplesperpixel)
    {
        TIFFErrorExtR(tif, tif->tif_name, "%lu: Sample out of range, max %lu",
                      (unsigned long)s,
                      (unsigned long)(td->td_samplesperpixel - 1));
        return (0);
    }
    return (1);
}

/*
 * Compute how many tiles are in an image.
 */
uint32_t TIFFNumberOfTiles(TIFF *tif)
{
    TIFFDirectory *td = &tif->tif_dir;
    uint32_t dx = td->td_tilewidth;
    uint32_t dy = td->td_tilelength;
    uint32_t dz = td->td_tiledepth;
    uint32_t ntiles;

    if (dx == (uint32_t)-1)
        dx = td->td_imagewidth;
    if (dy == (uint32_t)-1)
        dy = td->td_imagelength;
    if (dz == (uint32_t)-1)
        dz = td->td_imagedepth;
    ntiles =
        (dx == 0 || dy == 0 || dz == 0)
            ? 0
            : _TIFFMultiply32(
                  tif,
                  _TIFFMultiply32(tif, TIFFhowmany_32(td->td_imagewidth, dx),
                                  TIFFhowmany_32(td->td_imagelength, dy),
                                  "TIFFNumberOfTiles"),
                  TIFFhowmany_32(td->td_imagedepth, dz), "TIFFNumberOfTiles");
    if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
        ntiles = _TIFFMultiply32(tif, ntiles, td->td_samplesperpixel,
                                 "TIFFNumberOfTiles");
    return (ntiles);
}

/*
 * Compute the # bytes in each row of a tile.
 */
uint64_t TIFFTileRowSize64(TIFF *tif)
{
    static const char module[] = "TIFFTileRowSize64";
    TIFFDirectory *td = &tif->tif_dir;
    uint64_t rowsize;
    uint64_t tilerowsize;

    if (td->td_tilelength == 0)
    {
        TIFFErrorExtR(tif, module, "Tile length is zero");
        return 0;
    }
    if (td->td_tilewidth == 0)
    {
        TIFFErrorExtR(tif, module, "Tile width is zero");
        return (0);
    }
    rowsize = _TIFFMultiply64(tif, td->td_bitspersample, td->td_tilewidth,
                              "TIFFTileRowSize");
    if (td->td_planarconfig == PLANARCONFIG_CONTIG)
    {
        if (td->td_samplesperpixel == 0)
        {
            TIFFErrorExtR(tif, module, "Samples per pixel is zero");
            return 0;
        }
        rowsize = _TIFFMultiply64(tif, rowsize, td->td_samplesperpixel,
                                  "TIFFTileRowSize");
    }
    tilerowsize = TIFFhowmany8_64(rowsize);
    if (tilerowsize == 0)
    {
        TIFFErrorExtR(tif, module, "Computed tile row size is zero");
        return 0;
    }
    return (tilerowsize);
}
tmsize_t TIFFTileRowSize(TIFF *tif)
{
    static const char module[] = "TIFFTileRowSize";
    uint64_t m;
    m = TIFFTileRowSize64(tif);
    return _TIFFCastUInt64ToSSize(tif, m, module);
}

/*
 * Compute the # bytes in a variable length, row-aligned tile.
 */
uint64_t TIFFVTileSize64(TIFF *tif, uint32_t nrows)
{
    return _TIFFStrileSize64(tif, nrows, /* isStrip = */ FALSE);
}

tmsize_t TIFFVTileSize(TIFF *tif, uint32_t nrows)
{
    static const char module[] = "TIFFVTileSize";
    uint64_t m;
    m = TIFFVTileSize64(tif, nrows);
    return _TIFFCastUInt64ToSSize(tif, m, module);
}

/*
 * Compute the # bytes in a row-aligned tile.
 */
uint64_t TIFFTileSize64(TIFF *tif)
{
    return (TIFFVTileSize64(tif, tif->tif_dir.td_tilelength));
}
tmsize_t TIFFTileSize(TIFF *tif)
{
    static const char module[] = "TIFFTileSize";
    uint64_t m;
    m = TIFFTileSize64(tif);
    return _TIFFCastUInt64ToSSize(tif, m, module);
}

/*
 * Compute a default tile size based on the image
 * characteristics and a requested value.  If a
 * request is <1 then we choose a size according
 * to certain heuristics.
 */
void TIFFDefaultTileSize(TIFF *tif, uint32_t *tw, uint32_t *th)
{
    (*tif->tif_deftilesize)(tif, tw, th);
}

void _TIFFDefaultTileSize(TIFF *tif, uint32_t *tw, uint32_t *th)
{
    (void)tif;
    if (*(int32_t *)tw < 1)
        *tw = 256;
    if (*(int32_t *)th < 1)
        *th = 256;
    /* roundup to a multiple of 16 per the spec */
    if (*tw & 0xf)
        *tw = TIFFroundup_32(*tw, 16);
    if (*th & 0xf)
        *th = TIFFroundup_32(*th, 16);
}
