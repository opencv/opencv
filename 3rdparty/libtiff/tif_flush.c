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
 */
#include "tiffiop.h"

int TIFFFlush(TIFF *tif)
{
    if (tif->tif_mode == O_RDONLY)
        return 1;

    if (!TIFFFlushData(tif))
        return (0);

    /* In update (r+) mode we try to detect the case where
       only the strip/tile map has been altered, and we try to
       rewrite only that portion of the directory without
       making any other changes */

    if ((tif->tif_flags & TIFF_DIRTYSTRIP) &&
        !(tif->tif_flags & TIFF_DIRTYDIRECT) && tif->tif_mode == O_RDWR)
    {
        if (TIFFForceStrileArrayWriting(tif))
            return 1;
    }

    if ((tif->tif_flags & (TIFF_DIRTYDIRECT | TIFF_DIRTYSTRIP)) &&
        !TIFFRewriteDirectory(tif))
        return (0);

    return (1);
}

/*
 * This is an advanced writing function that must be used in a particular
 * sequence, and together with TIFFDeferStrileArrayWriting(),
 * to make its intended effect. Its aim is to force the writing of
 * the [Strip/Tile][Offsets/ByteCounts] arrays at the end of the file, when
 * they have not yet been rewritten.
 *
 * The typical sequence of calls is:
 * TIFFOpen()
 * [ TIFFCreateDirectory(tif) ]
 * Set fields with calls to TIFFSetField(tif, ...)
 * TIFFDeferStrileArrayWriting(tif)
 * TIFFWriteCheck(tif, ...)
 * TIFFWriteDirectory(tif)
 * ... potentially create other directories and come back to the above directory
 * TIFFForceStrileArrayWriting(tif)
 *
 * Returns 1 in case of success, 0 otherwise.
 */
int TIFFForceStrileArrayWriting(TIFF *tif)
{
    static const char module[] = "TIFFForceStrileArrayWriting";
    const int isTiled = TIFFIsTiled(tif);

    if (tif->tif_mode == O_RDONLY)
    {
        TIFFErrorExtR(tif, tif->tif_name, "File opened in read-only mode");
        return 0;
    }
    if (tif->tif_diroff == 0)
    {
        TIFFErrorExtR(tif, module, "Directory has not yet been written");
        return 0;
    }
    if ((tif->tif_flags & TIFF_DIRTYDIRECT) != 0)
    {
        TIFFErrorExtR(tif, module,
                      "Directory has changes other than the strile arrays. "
                      "TIFFRewriteDirectory() should be called instead");
        return 0;
    }

    if (!(tif->tif_flags & TIFF_DIRTYSTRIP))
    {
        if (!(tif->tif_dir.td_stripoffset_entry.tdir_tag != 0 &&
              tif->tif_dir.td_stripoffset_entry.tdir_count == 0 &&
              tif->tif_dir.td_stripoffset_entry.tdir_type == 0 &&
              tif->tif_dir.td_stripoffset_entry.tdir_offset.toff_long8 == 0 &&
              tif->tif_dir.td_stripbytecount_entry.tdir_tag != 0 &&
              tif->tif_dir.td_stripbytecount_entry.tdir_count == 0 &&
              tif->tif_dir.td_stripbytecount_entry.tdir_type == 0 &&
              tif->tif_dir.td_stripbytecount_entry.tdir_offset.toff_long8 == 0))
        {
            TIFFErrorExtR(tif, module,
                          "Function not called together with "
                          "TIFFDeferStrileArrayWriting()");
            return 0;
        }

        if (tif->tif_dir.td_stripoffset_p == NULL && !TIFFSetupStrips(tif))
            return 0;
    }

    if (_TIFFRewriteField(tif,
                          isTiled ? TIFFTAG_TILEOFFSETS : TIFFTAG_STRIPOFFSETS,
                          TIFF_LONG8, tif->tif_dir.td_nstrips,
                          tif->tif_dir.td_stripoffset_p) &&
        _TIFFRewriteField(
            tif, isTiled ? TIFFTAG_TILEBYTECOUNTS : TIFFTAG_STRIPBYTECOUNTS,
            TIFF_LONG8, tif->tif_dir.td_nstrips,
            tif->tif_dir.td_stripbytecount_p))
    {
        tif->tif_flags &= ~TIFF_DIRTYSTRIP;
        tif->tif_flags &= ~TIFF_BEENWRITING;
        return 1;
    }

    return 0;
}

/*
 * Flush buffered data to the file.
 *
 * Frank Warmerdam'2000: I modified this to return 1 if TIFF_BEENWRITING
 * is not set, so that TIFFFlush() will proceed to write out the directory.
 * The documentation says returning 1 is an error indicator, but not having
 * been writing isn't exactly a an error.  Hopefully this doesn't cause
 * problems for other people.
 */
int TIFFFlushData(TIFF *tif)
{
    if ((tif->tif_flags & TIFF_BEENWRITING) == 0)
        return (1);
    if (tif->tif_flags & TIFF_POSTENCODE)
    {
        tif->tif_flags &= ~TIFF_POSTENCODE;
        if (!(*tif->tif_postencode)(tif))
            return (0);
    }
    return (TIFFFlushData1(tif));
}
