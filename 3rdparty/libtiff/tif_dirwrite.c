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
 * Directory Write Support Routines.
 */
#include "tiffiop.h"
#include <float.h> /*--: for Rational2Double */
#include <math.h>  /*--: for Rational2Double */

#ifdef HAVE_IEEEFP
#define TIFFCvtNativeToIEEEFloat(tif, n, fp)
#define TIFFCvtNativeToIEEEDouble(tif, n, dp)
#else
/* If your machine does not support IEEE floating point then you will need to
 * add support to tif_machdep.c to convert between the native format and
 * IEEE format. */
extern void TIFFCvtNativeToIEEEFloat(TIFF *tif, uint32_t n, float *fp);
extern void TIFFCvtNativeToIEEEDouble(TIFF *tif, uint32_t n, double *dp);
#endif

static int TIFFWriteDirectorySec(TIFF *tif, int isimage, int imagedone,
                                 uint64_t *pdiroff);

static int TIFFWriteDirectoryTagSampleformatArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  double *value);

static int TIFFWriteDirectoryTagAscii(TIFF *tif, uint32_t *ndir,
                                      TIFFDirEntry *dir, uint16_t tag,
                                      uint32_t count, char *value);
static int TIFFWriteDirectoryTagUndefinedArray(TIFF *tif, uint32_t *ndir,
                                               TIFFDirEntry *dir, uint16_t tag,
                                               uint32_t count, uint8_t *value);
static int TIFFWriteDirectoryTagByteArray(TIFF *tif, uint32_t *ndir,
                                          TIFFDirEntry *dir, uint16_t tag,
                                          uint32_t count, uint8_t *value);
static int TIFFWriteDirectoryTagSbyteArray(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, int8_t *value);
static int TIFFWriteDirectoryTagShort(TIFF *tif, uint32_t *ndir,
                                      TIFFDirEntry *dir, uint16_t tag,
                                      uint16_t value);
static int TIFFWriteDirectoryTagShortArray(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, uint16_t *value);
static int TIFFWriteDirectoryTagShortPerSample(TIFF *tif, uint32_t *ndir,
                                               TIFFDirEntry *dir, uint16_t tag,
                                               uint16_t value);
static int TIFFWriteDirectoryTagSshortArray(TIFF *tif, uint32_t *ndir,
                                            TIFFDirEntry *dir, uint16_t tag,
                                            uint32_t count, int16_t *value);
static int TIFFWriteDirectoryTagLong(TIFF *tif, uint32_t *ndir,
                                     TIFFDirEntry *dir, uint16_t tag,
                                     uint32_t value);
static int TIFFWriteDirectoryTagLongArray(TIFF *tif, uint32_t *ndir,
                                          TIFFDirEntry *dir, uint16_t tag,
                                          uint32_t count, uint32_t *value);
static int TIFFWriteDirectoryTagSlongArray(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, int32_t *value);
static int TIFFWriteDirectoryTagLong8Array(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, uint64_t *value);
static int TIFFWriteDirectoryTagSlong8Array(TIFF *tif, uint32_t *ndir,
                                            TIFFDirEntry *dir, uint16_t tag,
                                            uint32_t count, int64_t *value);
static int TIFFWriteDirectoryTagRational(TIFF *tif, uint32_t *ndir,
                                         TIFFDirEntry *dir, uint16_t tag,
                                         double value);
static int TIFFWriteDirectoryTagRationalArray(TIFF *tif, uint32_t *ndir,
                                              TIFFDirEntry *dir, uint16_t tag,
                                              uint32_t count, float *value);
static int TIFFWriteDirectoryTagSrationalArray(TIFF *tif, uint32_t *ndir,
                                               TIFFDirEntry *dir, uint16_t tag,
                                               uint32_t count, float *value);
static int TIFFWriteDirectoryTagFloatArray(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, float *value);
static int TIFFWriteDirectoryTagDoubleArray(TIFF *tif, uint32_t *ndir,
                                            TIFFDirEntry *dir, uint16_t tag,
                                            uint32_t count, double *value);
static int TIFFWriteDirectoryTagIfdArray(TIFF *tif, uint32_t *ndir,
                                         TIFFDirEntry *dir, uint16_t tag,
                                         uint32_t count, uint32_t *value);
static int TIFFWriteDirectoryTagShortLong(TIFF *tif, uint32_t *ndir,
                                          TIFFDirEntry *dir, uint16_t tag,
                                          uint32_t value);
static int TIFFWriteDirectoryTagLongLong8Array(TIFF *tif, uint32_t *ndir,
                                               TIFFDirEntry *dir, uint16_t tag,
                                               uint32_t count, uint64_t *value);
static int TIFFWriteDirectoryTagIfdIfd8Array(TIFF *tif, uint32_t *ndir,
                                             TIFFDirEntry *dir, uint16_t tag,
                                             uint32_t count, uint64_t *value);
static int TIFFWriteDirectoryTagColormap(TIFF *tif, uint32_t *ndir,
                                         TIFFDirEntry *dir);
static int TIFFWriteDirectoryTagTransferfunction(TIFF *tif, uint32_t *ndir,
                                                 TIFFDirEntry *dir);
static int TIFFWriteDirectoryTagSubifd(TIFF *tif, uint32_t *ndir,
                                       TIFFDirEntry *dir);

static int TIFFWriteDirectoryTagCheckedAscii(TIFF *tif, uint32_t *ndir,
                                             TIFFDirEntry *dir, uint16_t tag,
                                             uint32_t count, char *value);
static int TIFFWriteDirectoryTagCheckedUndefinedArray(TIFF *tif, uint32_t *ndir,
                                                      TIFFDirEntry *dir,
                                                      uint16_t tag,
                                                      uint32_t count,
                                                      uint8_t *value);
static int TIFFWriteDirectoryTagCheckedByteArray(TIFF *tif, uint32_t *ndir,
                                                 TIFFDirEntry *dir,
                                                 uint16_t tag, uint32_t count,
                                                 uint8_t *value);
static int TIFFWriteDirectoryTagCheckedSbyteArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  int8_t *value);
static int TIFFWriteDirectoryTagCheckedShort(TIFF *tif, uint32_t *ndir,
                                             TIFFDirEntry *dir, uint16_t tag,
                                             uint16_t value);
static int TIFFWriteDirectoryTagCheckedShortArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  uint16_t *value);
static int TIFFWriteDirectoryTagCheckedSshortArray(TIFF *tif, uint32_t *ndir,
                                                   TIFFDirEntry *dir,
                                                   uint16_t tag, uint32_t count,
                                                   int16_t *value);
static int TIFFWriteDirectoryTagCheckedLong(TIFF *tif, uint32_t *ndir,
                                            TIFFDirEntry *dir, uint16_t tag,
                                            uint32_t value);
static int TIFFWriteDirectoryTagCheckedLongArray(TIFF *tif, uint32_t *ndir,
                                                 TIFFDirEntry *dir,
                                                 uint16_t tag, uint32_t count,
                                                 uint32_t *value);
static int TIFFWriteDirectoryTagCheckedSlongArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  int32_t *value);
static int TIFFWriteDirectoryTagCheckedLong8Array(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  uint64_t *value);
static int TIFFWriteDirectoryTagCheckedSlong8Array(TIFF *tif, uint32_t *ndir,
                                                   TIFFDirEntry *dir,
                                                   uint16_t tag, uint32_t count,
                                                   int64_t *value);
static int TIFFWriteDirectoryTagCheckedRational(TIFF *tif, uint32_t *ndir,
                                                TIFFDirEntry *dir, uint16_t tag,
                                                double value);
static int TIFFWriteDirectoryTagCheckedRationalArray(TIFF *tif, uint32_t *ndir,
                                                     TIFFDirEntry *dir,
                                                     uint16_t tag,
                                                     uint32_t count,
                                                     float *value);
static int TIFFWriteDirectoryTagCheckedSrationalArray(TIFF *tif, uint32_t *ndir,
                                                      TIFFDirEntry *dir,
                                                      uint16_t tag,
                                                      uint32_t count,
                                                      float *value);

/*--: Rational2Double: New functions to support true double-precision for custom
 * rational tag types. */
static int TIFFWriteDirectoryTagRationalDoubleArray(TIFF *tif, uint32_t *ndir,
                                                    TIFFDirEntry *dir,
                                                    uint16_t tag,
                                                    uint32_t count,
                                                    double *value);
static int TIFFWriteDirectoryTagSrationalDoubleArray(TIFF *tif, uint32_t *ndir,
                                                     TIFFDirEntry *dir,
                                                     uint16_t tag,
                                                     uint32_t count,
                                                     double *value);
static int
TIFFWriteDirectoryTagCheckedRationalDoubleArray(TIFF *tif, uint32_t *ndir,
                                                TIFFDirEntry *dir, uint16_t tag,
                                                uint32_t count, double *value);
static int TIFFWriteDirectoryTagCheckedSrationalDoubleArray(
    TIFF *tif, uint32_t *ndir, TIFFDirEntry *dir, uint16_t tag, uint32_t count,
    double *value);
static void DoubleToRational(double value, uint32_t *num, uint32_t *denom);
static void DoubleToSrational(double value, int32_t *num, int32_t *denom);

static int TIFFWriteDirectoryTagCheckedFloatArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  float *value);
static int TIFFWriteDirectoryTagCheckedDoubleArray(TIFF *tif, uint32_t *ndir,
                                                   TIFFDirEntry *dir,
                                                   uint16_t tag, uint32_t count,
                                                   double *value);
static int TIFFWriteDirectoryTagCheckedIfdArray(TIFF *tif, uint32_t *ndir,
                                                TIFFDirEntry *dir, uint16_t tag,
                                                uint32_t count,
                                                uint32_t *value);
static int TIFFWriteDirectoryTagCheckedIfd8Array(TIFF *tif, uint32_t *ndir,
                                                 TIFFDirEntry *dir,
                                                 uint16_t tag, uint32_t count,
                                                 uint64_t *value);

static int TIFFWriteDirectoryTagData(TIFF *tif, uint32_t *ndir,
                                     TIFFDirEntry *dir, uint16_t tag,
                                     uint16_t datatype, uint32_t count,
                                     uint32_t datalength, void *data);

static int TIFFLinkDirectory(TIFF *);

/*
 * Write the contents of the current directory
 * to the specified file.  This routine doesn't
 * handle overwriting a directory with auxiliary
 * storage that's been changed.
 */
int TIFFWriteDirectory(TIFF *tif)
{
    return TIFFWriteDirectorySec(tif, TRUE, TRUE, NULL);
}

/*
 * This is an advanced writing function that must be used in a particular
 * sequence, and generally together with TIFFForceStrileArrayWriting(),
 * to make its intended effect. Its aim is to modify the location
 * where the [Strip/Tile][Offsets/ByteCounts] arrays are located in the file.
 * More precisely, when TIFFWriteCheck() will be called, the tag entries for
 * those arrays will be written with type = count = offset = 0 as a temporary
 * value.
 *
 * Its effect is only valid for the current directory, and before
 * TIFFWriteDirectory() is first called, and  will be reset when
 * changing directory.
 *
 * The typical sequence of calls is:
 * TIFFOpen()
 * [ TIFFCreateDirectory(tif) ]
 * Set fields with calls to TIFFSetField(tif, ...)
 * TIFFDeferStrileArrayWriting(tif)
 * TIFFWriteCheck(tif, ...)
 * TIFFWriteDirectory(tif)
 * ... potentially create other directories and come back to the above directory
 * TIFFForceStrileArrayWriting(tif): emit the arrays at the end of file
 *
 * Returns 1 in case of success, 0 otherwise.
 */
int TIFFDeferStrileArrayWriting(TIFF *tif)
{
    static const char module[] = "TIFFDeferStrileArrayWriting";
    if (tif->tif_mode == O_RDONLY)
    {
        TIFFErrorExtR(tif, tif->tif_name, "File opened in read-only mode");
        return 0;
    }
    if (tif->tif_diroff != 0)
    {
        TIFFErrorExtR(tif, module, "Directory has already been written");
        return 0;
    }

    tif->tif_dir.td_deferstrilearraywriting = TRUE;
    return 1;
}

/*
 * Similar to TIFFWriteDirectory(), writes the directory out
 * but leaves all data structures in memory so that it can be
 * written again.  This will make a partially written TIFF file
 * readable before it is successfully completed/closed.
 */
int TIFFCheckpointDirectory(TIFF *tif)
{
    int rc;
    /* Setup the strips arrays, if they haven't already been. */
    if (tif->tif_dir.td_stripoffset_p == NULL)
        (void)TIFFSetupStrips(tif);
    rc = TIFFWriteDirectorySec(tif, TRUE, FALSE, NULL);
    (void)TIFFSetWriteOffset(tif, TIFFSeekFile(tif, 0, SEEK_END));
    return rc;
}

int TIFFWriteCustomDirectory(TIFF *tif, uint64_t *pdiroff)
{
    return TIFFWriteDirectorySec(tif, FALSE, FALSE, pdiroff);
}

/*
 * Similar to TIFFWriteDirectorySec(), but if the directory has already
 * been written once, it is relocated to the end of the file, in case it
 * has changed in size.  Note that this will result in the loss of the
 * previously used directory space.
 */

static int TIFFRewriteDirectorySec(TIFF *tif, int isimage, int imagedone,
                                   uint64_t *pdiroff)
{
    static const char module[] = "TIFFRewriteDirectory";

    /* We don't need to do anything special if it hasn't been written. */
    if (tif->tif_diroff == 0)
        return TIFFWriteDirectory(tif);

    /*
     * Find and zero the pointer to this directory, so that TIFFLinkDirectory
     * will cause it to be added after this directories current pre-link.
     */
    uint64_t torewritediroff = tif->tif_diroff;

    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        if (tif->tif_header.classic.tiff_diroff == tif->tif_diroff)
        {
            tif->tif_header.classic.tiff_diroff = 0;
            tif->tif_diroff = 0;

            TIFFSeekFile(tif, 4, SEEK_SET);
            if (!WriteOK(tif, &(tif->tif_header.classic.tiff_diroff), 4))
            {
                TIFFErrorExtR(tif, tif->tif_name, "Error updating TIFF header");
                return (0);
            }
        }
        else if (tif->tif_diroff > 0xFFFFFFFFU)
        {
            TIFFErrorExtR(tif, module,
                          "tif->tif_diroff exceeds 32 bit range allowed for "
                          "Classic TIFF");
            return (0);
        }
        else
        {
            uint32_t nextdir;
            nextdir = tif->tif_header.classic.tiff_diroff;
            while (1)
            {
                uint16_t dircount;
                uint32_t nextnextdir;

                if (!SeekOK(tif, nextdir) || !ReadOK(tif, &dircount, 2))
                {
                    TIFFErrorExtR(tif, module,
                                  "Error fetching directory count");
                    return (0);
                }
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(&dircount);
                (void)TIFFSeekFile(tif, nextdir + 2 + dircount * 12, SEEK_SET);
                if (!ReadOK(tif, &nextnextdir, 4))
                {
                    TIFFErrorExtR(tif, module, "Error fetching directory link");
                    return (0);
                }
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(&nextnextdir);
                if (nextnextdir == tif->tif_diroff)
                {
                    uint32_t m;
                    m = 0;
                    (void)TIFFSeekFile(tif, nextdir + 2 + dircount * 12,
                                       SEEK_SET);
                    if (!WriteOK(tif, &m, 4))
                    {
                        TIFFErrorExtR(tif, module,
                                      "Error writing directory link");
                        return (0);
                    }
                    tif->tif_diroff = 0;
                    /* Force a full-traversal to reach the zeroed pointer */
                    tif->tif_lastdiroff = 0;
                    break;
                }
                nextdir = nextnextdir;
            }
        }
        /* Remove skipped offset from IFD loop directory list. */
        _TIFFRemoveEntryFromDirectoryListByOffset(tif, torewritediroff);
    }
    else
    {
        if (tif->tif_header.big.tiff_diroff == tif->tif_diroff)
        {
            tif->tif_header.big.tiff_diroff = 0;
            tif->tif_diroff = 0;

            TIFFSeekFile(tif, 8, SEEK_SET);
            if (!WriteOK(tif, &(tif->tif_header.big.tiff_diroff), 8))
            {
                TIFFErrorExtR(tif, tif->tif_name, "Error updating TIFF header");
                return (0);
            }
        }
        else
        {
            uint64_t nextdir;
            nextdir = tif->tif_header.big.tiff_diroff;
            while (1)
            {
                uint64_t dircount64;
                uint16_t dircount;
                uint64_t nextnextdir;

                if (!SeekOK(tif, nextdir) || !ReadOK(tif, &dircount64, 8))
                {
                    TIFFErrorExtR(tif, module,
                                  "Error fetching directory count");
                    return (0);
                }
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(&dircount64);
                if (dircount64 > 0xFFFF)
                {
                    TIFFErrorExtR(tif, module,
                                  "Sanity check on tag count failed, likely "
                                  "corrupt TIFF");
                    return (0);
                }
                dircount = (uint16_t)dircount64;
                (void)TIFFSeekFile(tif, nextdir + 8 + dircount * 20, SEEK_SET);
                if (!ReadOK(tif, &nextnextdir, 8))
                {
                    TIFFErrorExtR(tif, module, "Error fetching directory link");
                    return (0);
                }
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(&nextnextdir);
                if (nextnextdir == tif->tif_diroff)
                {
                    uint64_t m;
                    m = 0;
                    (void)TIFFSeekFile(tif, nextdir + 8 + dircount * 20,
                                       SEEK_SET);
                    if (!WriteOK(tif, &m, 8))
                    {
                        TIFFErrorExtR(tif, module,
                                      "Error writing directory link");
                        return (0);
                    }
                    tif->tif_diroff = 0;
                    /* Force a full-traversal to reach the zeroed pointer */
                    tif->tif_lastdiroff = 0;
                    break;
                }
                nextdir = nextnextdir;
            }
        }
        /* Remove skipped offset from IFD loop directory list. */
        _TIFFRemoveEntryFromDirectoryListByOffset(tif, torewritediroff);
    }

    /*
     * Now use TIFFWriteDirectorySec() normally.
     */
    return TIFFWriteDirectorySec(tif, isimage, imagedone, pdiroff);
} /*-- TIFFRewriteDirectorySec() --*/

/*
 * Similar to TIFFWriteDirectory(), but if the directory has already
 * been written once, it is relocated to the end of the file, in case it
 * has changed in size.  Note that this will result in the loss of the
 * previously used directory space.
 */
int TIFFRewriteDirectory(TIFF *tif)
{
    return TIFFRewriteDirectorySec(tif, TRUE, TRUE, NULL);
}

static int TIFFWriteDirectorySec(TIFF *tif, int isimage, int imagedone,
                                 uint64_t *pdiroff)
{
    static const char module[] = "TIFFWriteDirectorySec";
    uint32_t ndir;
    TIFFDirEntry *dir;
    uint32_t dirsize;
    void *dirmem;
    uint32_t m;
    if (tif->tif_mode == O_RDONLY)
        return (1);

    _TIFFFillStriles(tif);

    /*
     * Clear write state so that subsequent images with
     * different characteristics get the right buffers
     * setup for them.
     */
    if (imagedone)
    {
        if (tif->tif_flags & TIFF_POSTENCODE)
        {
            tif->tif_flags &= ~TIFF_POSTENCODE;
            if (!(*tif->tif_postencode)(tif))
            {
                TIFFErrorExtR(tif, module,
                              "Error post-encoding before directory write");
                return (0);
            }
        }
        (*tif->tif_close)(tif); /* shutdown encoder */
        /*
         * Flush any data that might have been written
         * by the compression close+cleanup routines.  But
         * be careful not to write stuff if we didn't add data
         * in the previous steps as the "rawcc" data may well be
         * a previously read tile/strip in mixed read/write mode.
         */
        if (tif->tif_rawcc > 0 && (tif->tif_flags & TIFF_BEENWRITING) != 0)
        {
            if (!TIFFFlushData1(tif))
            {
                TIFFErrorExtR(tif, module,
                              "Error flushing data before directory write");
                return (0);
            }
        }
        if ((tif->tif_flags & TIFF_MYBUFFER) && tif->tif_rawdata)
        {
            _TIFFfreeExt(tif, tif->tif_rawdata);
            tif->tif_rawdata = NULL;
            tif->tif_rawcp = NULL;
            tif->tif_rawcc = 0;
            tif->tif_rawdatasize = 0;
            tif->tif_rawdataoff = 0;
            tif->tif_rawdataloaded = 0;
        }
        tif->tif_flags &= ~(TIFF_BEENWRITING | TIFF_BUFFERSETUP);
    }

    if (TIFFFieldSet(tif, FIELD_COMPRESSION) &&
        (tif->tif_dir.td_compression == COMPRESSION_DEFLATE))
    {
        TIFFWarningExtR(tif, module,
                        "Creating TIFF with legacy Deflate codec identifier, "
                        "COMPRESSION_ADOBE_DEFLATE is more widely supported");
    }
    dir = NULL;
    dirmem = NULL;
    dirsize = 0;
    while (1)
    {
        /* The first loop only determines "ndir" and uses TIFFLinkDirectory() to
         * set the offset at which the IFD is to be written to the file.
         * The second loop writes IFD entries to the file. */
        ndir = 0;
        if (dir == NULL)
            tif->tif_dir.td_dirdatasize_write = 0;
        if (isimage)
        {
            /*-- Step 1: Process named tags for an image with FIELD bits
             *           associated. --*/
            if (TIFFFieldSet(tif, FIELD_IMAGEDIMENSIONS))
            {
                if (!TIFFWriteDirectoryTagShortLong(tif, &ndir, dir,
                                                    TIFFTAG_IMAGEWIDTH,
                                                    tif->tif_dir.td_imagewidth))
                    goto bad;
                if (!TIFFWriteDirectoryTagShortLong(
                        tif, &ndir, dir, TIFFTAG_IMAGELENGTH,
                        tif->tif_dir.td_imagelength))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_TILEDIMENSIONS))
            {
                if (!TIFFWriteDirectoryTagShortLong(tif, &ndir, dir,
                                                    TIFFTAG_TILEWIDTH,
                                                    tif->tif_dir.td_tilewidth))
                    goto bad;
                if (!TIFFWriteDirectoryTagShortLong(tif, &ndir, dir,
                                                    TIFFTAG_TILELENGTH,
                                                    tif->tif_dir.td_tilelength))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_RESOLUTION))
            {
                if (!TIFFWriteDirectoryTagRational(tif, &ndir, dir,
                                                   TIFFTAG_XRESOLUTION,
                                                   tif->tif_dir.td_xresolution))
                    goto bad;
                if (!TIFFWriteDirectoryTagRational(tif, &ndir, dir,
                                                   TIFFTAG_YRESOLUTION,
                                                   tif->tif_dir.td_yresolution))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_POSITION))
            {
                if (!TIFFWriteDirectoryTagRational(tif, &ndir, dir,
                                                   TIFFTAG_XPOSITION,
                                                   tif->tif_dir.td_xposition))
                    goto bad;
                if (!TIFFWriteDirectoryTagRational(tif, &ndir, dir,
                                                   TIFFTAG_YPOSITION,
                                                   tif->tif_dir.td_yposition))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_SUBFILETYPE))
            {
                if (!TIFFWriteDirectoryTagLong(tif, &ndir, dir,
                                               TIFFTAG_SUBFILETYPE,
                                               tif->tif_dir.td_subfiletype))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_BITSPERSAMPLE))
            {
                if (!TIFFWriteDirectoryTagShortPerSample(
                        tif, &ndir, dir, TIFFTAG_BITSPERSAMPLE,
                        tif->tif_dir.td_bitspersample))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_COMPRESSION))
            {
                if (!TIFFWriteDirectoryTagShort(tif, &ndir, dir,
                                                TIFFTAG_COMPRESSION,
                                                tif->tif_dir.td_compression))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_PHOTOMETRIC))
            {
                if (!TIFFWriteDirectoryTagShort(tif, &ndir, dir,
                                                TIFFTAG_PHOTOMETRIC,
                                                tif->tif_dir.td_photometric))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_THRESHHOLDING))
            {
                if (!TIFFWriteDirectoryTagShort(tif, &ndir, dir,
                                                TIFFTAG_THRESHHOLDING,
                                                tif->tif_dir.td_threshholding))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_FILLORDER))
            {
                if (!TIFFWriteDirectoryTagShort(tif, &ndir, dir,
                                                TIFFTAG_FILLORDER,
                                                tif->tif_dir.td_fillorder))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_ORIENTATION))
            {
                if (!TIFFWriteDirectoryTagShort(tif, &ndir, dir,
                                                TIFFTAG_ORIENTATION,
                                                tif->tif_dir.td_orientation))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_SAMPLESPERPIXEL))
            {
                if (!TIFFWriteDirectoryTagShort(
                        tif, &ndir, dir, TIFFTAG_SAMPLESPERPIXEL,
                        tif->tif_dir.td_samplesperpixel))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_ROWSPERSTRIP))
            {
                if (!TIFFWriteDirectoryTagShortLong(
                        tif, &ndir, dir, TIFFTAG_ROWSPERSTRIP,
                        tif->tif_dir.td_rowsperstrip))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_MINSAMPLEVALUE))
            {
                if (!TIFFWriteDirectoryTagShortPerSample(
                        tif, &ndir, dir, TIFFTAG_MINSAMPLEVALUE,
                        tif->tif_dir.td_minsamplevalue))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_MAXSAMPLEVALUE))
            {
                if (!TIFFWriteDirectoryTagShortPerSample(
                        tif, &ndir, dir, TIFFTAG_MAXSAMPLEVALUE,
                        tif->tif_dir.td_maxsamplevalue))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_PLANARCONFIG))
            {
                if (!TIFFWriteDirectoryTagShort(tif, &ndir, dir,
                                                TIFFTAG_PLANARCONFIG,
                                                tif->tif_dir.td_planarconfig))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_RESOLUTIONUNIT))
            {
                if (!TIFFWriteDirectoryTagShort(tif, &ndir, dir,
                                                TIFFTAG_RESOLUTIONUNIT,
                                                tif->tif_dir.td_resolutionunit))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_PAGENUMBER))
            {
                if (!TIFFWriteDirectoryTagShortArray(
                        tif, &ndir, dir, TIFFTAG_PAGENUMBER, 2,
                        &tif->tif_dir.td_pagenumber[0]))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_STRIPBYTECOUNTS))
            {
                if (!isTiled(tif))
                {
                    if (!TIFFWriteDirectoryTagLongLong8Array(
                            tif, &ndir, dir, TIFFTAG_STRIPBYTECOUNTS,
                            tif->tif_dir.td_nstrips,
                            tif->tif_dir.td_stripbytecount_p))
                        goto bad;
                }
                else
                {
                    if (!TIFFWriteDirectoryTagLongLong8Array(
                            tif, &ndir, dir, TIFFTAG_TILEBYTECOUNTS,
                            tif->tif_dir.td_nstrips,
                            tif->tif_dir.td_stripbytecount_p))
                        goto bad;
                }
            }
            if (TIFFFieldSet(tif, FIELD_STRIPOFFSETS))
            {
                if (!isTiled(tif))
                {
                    /* td_stripoffset_p might be NULL in an odd OJPEG case. See
                     *  tif_dirread.c around line 3634.
                     * XXX: OJPEG hack.
                     * If a) compression is OJPEG, b) it's not a tiled TIFF,
                     * and c) the number of strips is 1,
                     * then we tolerate the absence of stripoffsets tag,
                     * because, presumably, all required data is in the
                     * JpegInterchangeFormat stream.
                     * We can get here when using tiffset on such a file.
                     * See http://bugzilla.maptools.org/show_bug.cgi?id=2500
                     */
                    if (tif->tif_dir.td_stripoffset_p != NULL &&
                        !TIFFWriteDirectoryTagLongLong8Array(
                            tif, &ndir, dir, TIFFTAG_STRIPOFFSETS,
                            tif->tif_dir.td_nstrips,
                            tif->tif_dir.td_stripoffset_p))
                        goto bad;
                }
                else
                {
                    if (!TIFFWriteDirectoryTagLongLong8Array(
                            tif, &ndir, dir, TIFFTAG_TILEOFFSETS,
                            tif->tif_dir.td_nstrips,
                            tif->tif_dir.td_stripoffset_p))
                        goto bad;
                }
            }
            if (TIFFFieldSet(tif, FIELD_COLORMAP))
            {
                if (!TIFFWriteDirectoryTagColormap(tif, &ndir, dir))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_EXTRASAMPLES))
            {
                if (tif->tif_dir.td_extrasamples)
                {
                    uint16_t na;
                    uint16_t *nb;
                    TIFFGetFieldDefaulted(tif, TIFFTAG_EXTRASAMPLES, &na, &nb);
                    if (!TIFFWriteDirectoryTagShortArray(
                            tif, &ndir, dir, TIFFTAG_EXTRASAMPLES, na, nb))
                        goto bad;
                }
            }
            if (TIFFFieldSet(tif, FIELD_SAMPLEFORMAT))
            {
                if (!TIFFWriteDirectoryTagShortPerSample(
                        tif, &ndir, dir, TIFFTAG_SAMPLEFORMAT,
                        tif->tif_dir.td_sampleformat))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_SMINSAMPLEVALUE))
            {
                if (!TIFFWriteDirectoryTagSampleformatArray(
                        tif, &ndir, dir, TIFFTAG_SMINSAMPLEVALUE,
                        tif->tif_dir.td_samplesperpixel,
                        tif->tif_dir.td_sminsamplevalue))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_SMAXSAMPLEVALUE))
            {
                if (!TIFFWriteDirectoryTagSampleformatArray(
                        tif, &ndir, dir, TIFFTAG_SMAXSAMPLEVALUE,
                        tif->tif_dir.td_samplesperpixel,
                        tif->tif_dir.td_smaxsamplevalue))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_IMAGEDEPTH))
            {
                if (!TIFFWriteDirectoryTagLong(tif, &ndir, dir,
                                               TIFFTAG_IMAGEDEPTH,
                                               tif->tif_dir.td_imagedepth))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_TILEDEPTH))
            {
                if (!TIFFWriteDirectoryTagLong(tif, &ndir, dir,
                                               TIFFTAG_TILEDEPTH,
                                               tif->tif_dir.td_tiledepth))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_HALFTONEHINTS))
            {
                if (!TIFFWriteDirectoryTagShortArray(
                        tif, &ndir, dir, TIFFTAG_HALFTONEHINTS, 2,
                        &tif->tif_dir.td_halftonehints[0]))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_YCBCRSUBSAMPLING))
            {
                if (!TIFFWriteDirectoryTagShortArray(
                        tif, &ndir, dir, TIFFTAG_YCBCRSUBSAMPLING, 2,
                        &tif->tif_dir.td_ycbcrsubsampling[0]))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_YCBCRPOSITIONING))
            {
                if (!TIFFWriteDirectoryTagShort(
                        tif, &ndir, dir, TIFFTAG_YCBCRPOSITIONING,
                        tif->tif_dir.td_ycbcrpositioning))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_REFBLACKWHITE))
            {
                if (!TIFFWriteDirectoryTagRationalArray(
                        tif, &ndir, dir, TIFFTAG_REFERENCEBLACKWHITE, 6,
                        tif->tif_dir.td_refblackwhite))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_TRANSFERFUNCTION))
            {
                if (!TIFFWriteDirectoryTagTransferfunction(tif, &ndir, dir))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_INKNAMES))
            {
                if (!TIFFWriteDirectoryTagAscii(
                        tif, &ndir, dir, TIFFTAG_INKNAMES,
                        tif->tif_dir.td_inknameslen, tif->tif_dir.td_inknames))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_NUMBEROFINKS))
            {
                if (!TIFFWriteDirectoryTagShort(tif, &ndir, dir,
                                                TIFFTAG_NUMBEROFINKS,
                                                tif->tif_dir.td_numberofinks))
                    goto bad;
            }
            if (TIFFFieldSet(tif, FIELD_SUBIFD))
            {
                if (!TIFFWriteDirectoryTagSubifd(tif, &ndir, dir))
                    goto bad;
            }
            /*-- Step 2: Process named tags for an image with FIELD bits
                         added by a codec.
                         Attention: There is only code for some field_types,
                         which are actually used by current codecs. --*/
            {
                uint32_t n;
                for (n = 0; n < tif->tif_nfields; n++)
                {
                    const TIFFField *o;
                    o = tif->tif_fields[n];
                    if ((o->field_bit >= FIELD_CODEC) &&
                        (TIFFFieldSet(tif, o->field_bit)))
                    {
                        switch (o->set_get_field_type)
                        {
                            case TIFF_SETGET_ASCII:
                            {
                                uint32_t pa;
                                char *pb;
                                assert(o->field_type == TIFF_ASCII);
                                assert(o->field_readcount == TIFF_VARIABLE);
                                assert(o->field_passcount == 0);
                                TIFFGetField(tif, o->field_tag, &pb);
                                pa = (uint32_t)(strlen(pb) + 1);
                                if (!TIFFWriteDirectoryTagAscii(
                                        tif, &ndir, dir, (uint16_t)o->field_tag,
                                        pa, pb))
                                    goto bad;
                            }
                            break;
                            case TIFF_SETGET_UINT16:
                            {
                                uint16_t p;
                                assert(o->field_type == TIFF_SHORT);
                                assert(o->field_readcount == 1);
                                assert(o->field_passcount == 0);
                                TIFFGetField(tif, o->field_tag, &p);
                                if (!TIFFWriteDirectoryTagShort(
                                        tif, &ndir, dir, (uint16_t)o->field_tag,
                                        p))
                                    goto bad;
                            }
                            break;
                            case TIFF_SETGET_UINT32:
                            {
                                uint32_t p;
                                assert(o->field_type == TIFF_LONG);
                                assert(o->field_readcount == 1);
                                assert(o->field_passcount == 0);
                                TIFFGetField(tif, o->field_tag, &p);
                                if (!TIFFWriteDirectoryTagLong(
                                        tif, &ndir, dir, (uint16_t)o->field_tag,
                                        p))
                                    goto bad;
                            }
                            break;
                            case TIFF_SETGET_C32_UINT8:
                            {
                                uint32_t pa;
                                void *pb;
                                assert(o->field_type == TIFF_UNDEFINED);
                                assert(o->field_readcount == TIFF_VARIABLE2);
                                assert(o->field_passcount == 1);
                                TIFFGetField(tif, o->field_tag, &pa, &pb);
                                if (!TIFFWriteDirectoryTagUndefinedArray(
                                        tif, &ndir, dir, (uint16_t)o->field_tag,
                                        pa, pb))
                                    goto bad;
                            }
                            break;
                            default:
                                TIFFErrorExtR(
                                    tif, module,
                                    "Cannot write tag %" PRIu32 " (%s)",
                                    TIFFFieldTag(o),
                                    o->field_name ? o->field_name : "unknown");
                                goto bad;
                        }
                    }
                }
            }
        }
        /*-- Step 3: Process custom tags without FIELD bit for an image
         *           or for custom IFDs (e.g. EXIF) with !isimage. --*/
        for (m = 0; m < (uint32_t)(tif->tif_dir.td_customValueCount); m++)
        {
            uint16_t tag =
                (uint16_t)tif->tif_dir.td_customValues[m].info->field_tag;
            uint32_t count = tif->tif_dir.td_customValues[m].count;
            switch (tif->tif_dir.td_customValues[m].info->field_type)
            {
                case TIFF_ASCII:
                    if (!TIFFWriteDirectoryTagAscii(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_UNDEFINED:
                    if (!TIFFWriteDirectoryTagUndefinedArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_BYTE:
                    if (!TIFFWriteDirectoryTagByteArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_SBYTE:
                    if (!TIFFWriteDirectoryTagSbyteArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_SHORT:
                    if (!TIFFWriteDirectoryTagShortArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_SSHORT:
                    if (!TIFFWriteDirectoryTagSshortArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_LONG:
                    if (!TIFFWriteDirectoryTagLongArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_SLONG:
                    if (!TIFFWriteDirectoryTagSlongArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_LONG8:
                    if (!TIFFWriteDirectoryTagLong8Array(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_SLONG8:
                    if (!TIFFWriteDirectoryTagSlong8Array(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_RATIONAL:
                {
                    /*-- Rational2Double: For Rationals evaluate
                     * "set_get_field_type" to determine internal storage size.
                     */
                    int tv_size;
                    tv_size = TIFFFieldSetGetSize(
                        tif->tif_dir.td_customValues[m].info);
                    if (tv_size == 8)
                    {
                        if (!TIFFWriteDirectoryTagRationalDoubleArray(
                                tif, &ndir, dir, tag, count,
                                tif->tif_dir.td_customValues[m].value))
                            goto bad;
                    }
                    else
                    {
                        /*-- default should be tv_size == 4 */
                        if (!TIFFWriteDirectoryTagRationalArray(
                                tif, &ndir, dir, tag, count,
                                tif->tif_dir.td_customValues[m].value))
                            goto bad;
                        /*-- ToDo: After Testing, this should be removed and
                         * tv_size==4 should be set as default. */
                        if (tv_size != 4)
                        {
                            TIFFErrorExtR(
                                tif, "TIFFLib: _TIFFWriteDirectorySec()",
                                "Rational2Double: .set_get_field_type is "
                                "not 4 but %d",
                                tv_size);
                        }
                    }
                }
                break;
                case TIFF_SRATIONAL:
                {
                    /*-- Rational2Double: For Rationals evaluate
                     * "set_get_field_type" to determine internal storage size.
                     */
                    int tv_size;
                    tv_size = TIFFFieldSetGetSize(
                        tif->tif_dir.td_customValues[m].info);
                    if (tv_size == 8)
                    {
                        if (!TIFFWriteDirectoryTagSrationalDoubleArray(
                                tif, &ndir, dir, tag, count,
                                tif->tif_dir.td_customValues[m].value))
                            goto bad;
                    }
                    else
                    {
                        /*-- default should be tv_size == 4 */
                        if (!TIFFWriteDirectoryTagSrationalArray(
                                tif, &ndir, dir, tag, count,
                                tif->tif_dir.td_customValues[m].value))
                            goto bad;
                        /*-- ToDo: After Testing, this should be removed and
                         * tv_size==4 should be set as default. */
                        if (tv_size != 4)
                        {
                            TIFFErrorExtR(
                                tif, "TIFFLib: _TIFFWriteDirectorySec()",
                                "Rational2Double: .set_get_field_type is "
                                "not 4 but %d",
                                tv_size);
                        }
                    }
                }
                break;
                case TIFF_FLOAT:
                    if (!TIFFWriteDirectoryTagFloatArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_DOUBLE:
                    if (!TIFFWriteDirectoryTagDoubleArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_IFD:
                    if (!TIFFWriteDirectoryTagIfdArray(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                case TIFF_IFD8:
                    if (!TIFFWriteDirectoryTagIfdIfd8Array(
                            tif, &ndir, dir, tag, count,
                            tif->tif_dir.td_customValues[m].value))
                        goto bad;
                    break;
                default:
                    assert(0); /* we should never get here */
                    break;
            }
        }
        /* "break" if IFD has been written above in second pass.*/
        if (dir != NULL)
            break;

        /* Evaluate IFD data size: Finally, add the size of the IFD tag entries
         * themselves. */
        if (!(tif->tif_flags & TIFF_BIGTIFF))
            tif->tif_dir.td_dirdatasize_write += 2 + ndir * 12 + 4;
        else
            tif->tif_dir.td_dirdatasize_write += 8 + ndir * 20 + 8;

        /* Setup a new directory within first pass. */
        dir = _TIFFmallocExt(tif, ndir * sizeof(TIFFDirEntry));
        if (dir == NULL)
        {
            TIFFErrorExtR(tif, module, "Out of memory");
            goto bad;
        }
        if (isimage)
        {
            /* Check, weather the IFD to be written is new or an already written
             * IFD can be overwritten or needs to be re-written to a different
             * location in the file because the IFD is extended with additional
             * tags or the IFD data size is increased.
             * - tif_diroff == 0, if a new directory has to be linked.
             * - tif_diroff != 0, IFD has been re-read from file and will be
             *   overwritten or re-written.
             */
            if (tif->tif_diroff == 0)
            {
                if (!TIFFLinkDirectory(tif))
                    goto bad;
            }
            else if (tif->tif_dir.td_dirdatasize_write >
                     tif->tif_dir.td_dirdatasize_read)
            {
                if (dir != NULL)
                {
                    _TIFFfreeExt(tif, dir);
                    dir = NULL;
                }
                if (!TIFFRewriteDirectorySec(tif, isimage, imagedone, pdiroff))
                    goto bad;
                return (1);
            }
        }
        else
        {
            /* For !isimage, which means custom-IFD like EXIFIFD or
             * checkpointing an IFD, determine whether to overwrite or append at
             * the end of the file.
             */
            if (!((tif->tif_dir.td_dirdatasize_read > 0) &&
                  (tif->tif_dir.td_dirdatasize_write <=
                   tif->tif_dir.td_dirdatasize_read)))
            {
                /* Append at end of file and increment to an even offset. */
                tif->tif_diroff =
                    (TIFFSeekFile(tif, 0, SEEK_END) + 1) & (~((toff_t)1));
            }
        }
        /* Return IFD offset */
        if (pdiroff != NULL)
            *pdiroff = tif->tif_diroff;
        if (!(tif->tif_flags & TIFF_BIGTIFF))
            dirsize = 2 + ndir * 12 + 4;
        else
            dirsize = 8 + ndir * 20 + 8;
        /* Append IFD data stright after the IFD tag entries.
         * Data that does not fit into an IFD tag entry is written to the file
         * in the second pass of the while loop. That offset is stored in "dir".
         */
        tif->tif_dataoff = tif->tif_diroff + dirsize;
        if (!(tif->tif_flags & TIFF_BIGTIFF))
            tif->tif_dataoff = (uint32_t)tif->tif_dataoff;
        if ((tif->tif_dataoff < tif->tif_diroff) ||
            (tif->tif_dataoff < (uint64_t)dirsize))
        {
            TIFFErrorExtR(tif, module, "Maximum TIFF file size exceeded");
            goto bad;
        }
        if (tif->tif_dataoff & 1)
            tif->tif_dataoff++;
    } /* while() */
    if (isimage)
    {
        /* For SubIFDs remember offset of SubIFD tag within main IFD.
         * However, might be already done in TIFFWriteDirectoryTagSubifd() if
         * there are more than one SubIFD. */
        if (TIFFFieldSet(tif, FIELD_SUBIFD) && (tif->tif_subifdoff == 0))
        {
            uint32_t na;
            TIFFDirEntry *nb;
            for (na = 0, nb = dir;; na++, nb++)
            {
                if (na == ndir)
                {
                    TIFFErrorExtR(tif, module, "Cannot find SubIFD tag");
                    goto bad;
                }
                if (nb->tdir_tag == TIFFTAG_SUBIFD)
                    break;
            }
            if (!(tif->tif_flags & TIFF_BIGTIFF))
                tif->tif_subifdoff = tif->tif_diroff + 2 + na * 12 + 8;
            else
                tif->tif_subifdoff = tif->tif_diroff + 8 + na * 20 + 12;
        }
    }
    /* Copy/swab IFD entries from "dir" into "dirmem",
     * which is then written to file. */
    dirmem = _TIFFmallocExt(tif, dirsize);
    if (dirmem == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        goto bad;
    }
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        uint8_t *n;
        uint32_t nTmp;
        TIFFDirEntry *o;
        n = dirmem;
        *(uint16_t *)n = (uint16_t)ndir;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabShort((uint16_t *)n);
        n += 2;
        o = dir;
        for (m = 0; m < ndir; m++)
        {
            *(uint16_t *)n = o->tdir_tag;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort((uint16_t *)n);
            n += 2;
            *(uint16_t *)n = o->tdir_type;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort((uint16_t *)n);
            n += 2;
            nTmp = (uint32_t)o->tdir_count;
            _TIFFmemcpy(n, &nTmp, 4);
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong((uint32_t *)n);
            n += 4;
            /* This is correct. The data has been */
            /* swabbed previously in TIFFWriteDirectoryTagData */
            _TIFFmemcpy(n, &o->tdir_offset, 4);
            n += 4;
            o++;
        }
        nTmp = (uint32_t)tif->tif_nextdiroff;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&nTmp);
        _TIFFmemcpy(n, &nTmp, 4);
    }
    else
    {
        uint8_t *n;
        TIFFDirEntry *o;
        n = dirmem;
        *(uint64_t *)n = ndir;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8((uint64_t *)n);
        n += 8;
        o = dir;
        for (m = 0; m < ndir; m++)
        {
            *(uint16_t *)n = o->tdir_tag;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort((uint16_t *)n);
            n += 2;
            *(uint16_t *)n = o->tdir_type;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort((uint16_t *)n);
            n += 2;
            _TIFFmemcpy(n, &o->tdir_count, 8);
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8((uint64_t *)n);
            n += 8;
            _TIFFmemcpy(n, &o->tdir_offset, 8);
            n += 8;
            o++;
        }
        _TIFFmemcpy(n, &tif->tif_nextdiroff, 8);
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8((uint64_t *)n);
    }
    _TIFFfreeExt(tif, dir);
    dir = NULL;
    if (!SeekOK(tif, tif->tif_diroff))
    {
        TIFFErrorExtR(tif, module,
                      "IO error writing directory at seek to offset");
        goto bad;
    }
    if (!WriteOK(tif, dirmem, (tmsize_t)dirsize))
    {
        TIFFErrorExtR(tif, module, "IO error writing directory");
        goto bad;
    }
    _TIFFfreeExt(tif, dirmem);

    /* Increment tif_curdir if IFD wasn't already written to file and no error
     * occurred during IFD writing above. */
    if (isimage && !tif->tif_dir.td_iswrittentofile)
    {
        if (!((tif->tif_flags & TIFF_INSUBIFD) &&
              !(TIFFFieldSet(tif, FIELD_SUBIFD))))
        {
            /*-- Normal main-IFD case --*/
            if (tif->tif_curdircount != TIFF_NON_EXISTENT_DIR_NUMBER)
            {
                tif->tif_curdir = tif->tif_curdircount;
            }
            else
            {
                /*ToDo SU: NEW_IFD_CURDIR_INCREMENTING:  Delete this
                 * unexpected case after some testing time. */
                /* Attention: tif->tif_curdircount is already set within
                 * TIFFNumberOfDirectories() */
                tif->tif_curdircount = TIFFNumberOfDirectories(tif);
                tif->tif_curdir = tif->tif_curdircount;
                TIFFErrorExtR(
                    tif, module,
                    "tif_curdircount is TIFF_NON_EXISTENT_DIR_NUMBER, "
                    "not expected !! Line %d",
                    __LINE__);
                goto bad;
            }
        }
        else
        {
            /*-- SubIFD case -- */
            /* tif_curdir is always set to 0 for all SubIFDs. */
            tif->tif_curdir = 0;
        }
    }
    /* Increment tif_curdircount only if main-IFD of an image was not already
     * present on file. */
    /* Check in combination with (... && !(TIFFFieldSet(tif, FIELD_SUBIFD)))
     * is necessary here because TIFF_INSUBIFD was already set above for the
     * next SubIFD when this main-IFD (with FIELD_SUBIFD) is currently being
     * written. */
    if (isimage && !tif->tif_dir.td_iswrittentofile &&
        !((tif->tif_flags & TIFF_INSUBIFD) &&
          !(TIFFFieldSet(tif, FIELD_SUBIFD))))
        tif->tif_curdircount++;

    tif->tif_dir.td_iswrittentofile = TRUE;

    /* Reset SubIFD writing stage after last SubIFD has been written. */
    if (imagedone && (tif->tif_flags & TIFF_INSUBIFD) && tif->tif_nsubifd == 0)
        tif->tif_flags &= ~TIFF_INSUBIFD;

    /* Add or update this directory to the IFD list. */
    if (!_TIFFCheckDirNumberAndOffset(tif, tif->tif_curdir, tif->tif_diroff))
    {
        TIFFErrorExtR(tif, module,
                      "Starting directory %u at offset 0x%" PRIx64 " (%" PRIu64
                      ") might cause an IFD loop",
                      tif->tif_curdir, tif->tif_diroff, tif->tif_diroff);
    }

    if (imagedone)
    {
        TIFFFreeDirectory(tif);
        tif->tif_flags &= ~TIFF_DIRTYDIRECT;
        tif->tif_flags &= ~TIFF_DIRTYSTRIP;
        /* Reset directory-related state for subsequent directories. */
        TIFFCreateDirectory(tif);
    }
    else
    {
        /* IFD is only checkpointed to file (or a custom IFD like EXIF is
         * written), thus set IFD data size written to file. */
        tif->tif_dir.td_dirdatasize_read = tif->tif_dir.td_dirdatasize_write;
    }
    return (1);
bad:
    if (dir != NULL)
        _TIFFfreeExt(tif, dir);
    if (dirmem != NULL)
        _TIFFfreeExt(tif, dirmem);
    return (0);
}

static int8_t TIFFClampDoubleToInt8(double val)
{
    if (val > 127)
        return 127;
    if (val < -128 || val != val)
        return -128;
    return (int8_t)val;
}

static int16_t TIFFClampDoubleToInt16(double val)
{
    if (val > 32767)
        return 32767;
    if (val < -32768 || val != val)
        return -32768;
    return (int16_t)val;
}

static int32_t TIFFClampDoubleToInt32(double val)
{
    if (val > 0x7FFFFFFF)
        return 0x7FFFFFFF;
    if (val < -0x7FFFFFFF - 1 || val != val)
        return -0x7FFFFFFF - 1;
    return (int32_t)val;
}

static uint8_t TIFFClampDoubleToUInt8(double val)
{
    if (val < 0)
        return 0;
    if (val > 255 || val != val)
        return 255;
    return (uint8_t)val;
}

static uint16_t TIFFClampDoubleToUInt16(double val)
{
    if (val < 0)
        return 0;
    if (val > 65535 || val != val)
        return 65535;
    return (uint16_t)val;
}

static uint32_t TIFFClampDoubleToUInt32(double val)
{
    if (val < 0)
        return 0;
    if (val > 0xFFFFFFFFU || val != val)
        return 0xFFFFFFFFU;
    return (uint32_t)val;
}

static int TIFFWriteDirectoryTagSampleformatArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  double *value)
{
    static const char module[] = "TIFFWriteDirectoryTagSampleformatArray";
    void *conv;
    uint32_t i;
    int ok;
    conv = _TIFFmallocExt(tif, count * sizeof(double));
    if (conv == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }

    switch (tif->tif_dir.td_sampleformat)
    {
        case SAMPLEFORMAT_IEEEFP:
            if (tif->tif_dir.td_bitspersample <= 32)
            {
                for (i = 0; i < count; ++i)
                    ((float *)conv)[i] = _TIFFClampDoubleToFloat(value[i]);
                ok = TIFFWriteDirectoryTagFloatArray(tif, ndir, dir, tag, count,
                                                     (float *)conv);
            }
            else
            {
                ok = TIFFWriteDirectoryTagDoubleArray(tif, ndir, dir, tag,
                                                      count, value);
            }
            break;
        case SAMPLEFORMAT_INT:
            if (tif->tif_dir.td_bitspersample <= 8)
            {
                for (i = 0; i < count; ++i)
                    ((int8_t *)conv)[i] = TIFFClampDoubleToInt8(value[i]);
                ok = TIFFWriteDirectoryTagSbyteArray(tif, ndir, dir, tag, count,
                                                     (int8_t *)conv);
            }
            else if (tif->tif_dir.td_bitspersample <= 16)
            {
                for (i = 0; i < count; ++i)
                    ((int16_t *)conv)[i] = TIFFClampDoubleToInt16(value[i]);
                ok = TIFFWriteDirectoryTagSshortArray(tif, ndir, dir, tag,
                                                      count, (int16_t *)conv);
            }
            else
            {
                for (i = 0; i < count; ++i)
                    ((int32_t *)conv)[i] = TIFFClampDoubleToInt32(value[i]);
                ok = TIFFWriteDirectoryTagSlongArray(tif, ndir, dir, tag, count,
                                                     (int32_t *)conv);
            }
            break;
        case SAMPLEFORMAT_UINT:
            if (tif->tif_dir.td_bitspersample <= 8)
            {
                for (i = 0; i < count; ++i)
                    ((uint8_t *)conv)[i] = TIFFClampDoubleToUInt8(value[i]);
                ok = TIFFWriteDirectoryTagByteArray(tif, ndir, dir, tag, count,
                                                    (uint8_t *)conv);
            }
            else if (tif->tif_dir.td_bitspersample <= 16)
            {
                for (i = 0; i < count; ++i)
                    ((uint16_t *)conv)[i] = TIFFClampDoubleToUInt16(value[i]);
                ok = TIFFWriteDirectoryTagShortArray(tif, ndir, dir, tag, count,
                                                     (uint16_t *)conv);
            }
            else
            {
                for (i = 0; i < count; ++i)
                    ((uint32_t *)conv)[i] = TIFFClampDoubleToUInt32(value[i]);
                ok = TIFFWriteDirectoryTagLongArray(tif, ndir, dir, tag, count,
                                                    (uint32_t *)conv);
            }
            break;
        default:
            ok = 0;
    }

    _TIFFfreeExt(tif, conv);
    return (ok);
}

static int TIFFWriteDirectoryTagAscii(TIFF *tif, uint32_t *ndir,
                                      TIFFDirEntry *dir, uint16_t tag,
                                      uint32_t count, char *value)
{
    return (
        TIFFWriteDirectoryTagCheckedAscii(tif, ndir, dir, tag, count, value));
}

static int TIFFWriteDirectoryTagUndefinedArray(TIFF *tif, uint32_t *ndir,
                                               TIFFDirEntry *dir, uint16_t tag,
                                               uint32_t count, uint8_t *value)
{
    return (TIFFWriteDirectoryTagCheckedUndefinedArray(tif, ndir, dir, tag,
                                                       count, value));
}

static int TIFFWriteDirectoryTagByteArray(TIFF *tif, uint32_t *ndir,
                                          TIFFDirEntry *dir, uint16_t tag,
                                          uint32_t count, uint8_t *value)
{
    return (TIFFWriteDirectoryTagCheckedByteArray(tif, ndir, dir, tag, count,
                                                  value));
}

static int TIFFWriteDirectoryTagSbyteArray(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, int8_t *value)
{
    return (TIFFWriteDirectoryTagCheckedSbyteArray(tif, ndir, dir, tag, count,
                                                   value));
}

static int TIFFWriteDirectoryTagShort(TIFF *tif, uint32_t *ndir,
                                      TIFFDirEntry *dir, uint16_t tag,
                                      uint16_t value)
{
    return (TIFFWriteDirectoryTagCheckedShort(tif, ndir, dir, tag, value));
}

static int TIFFWriteDirectoryTagShortArray(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, uint16_t *value)
{
    return (TIFFWriteDirectoryTagCheckedShortArray(tif, ndir, dir, tag, count,
                                                   value));
}

static int TIFFWriteDirectoryTagShortPerSample(TIFF *tif, uint32_t *ndir,
                                               TIFFDirEntry *dir, uint16_t tag,
                                               uint16_t value)
{
    static const char module[] = "TIFFWriteDirectoryTagShortPerSample";
    uint16_t *m;
    uint16_t *na;
    uint16_t nb;
    int o;
    if (dir == NULL)
    {
        /* only evaluate IFD data size and inc. ndir */
        return (TIFFWriteDirectoryTagCheckedShortArray(
            tif, ndir, dir, tag, tif->tif_dir.td_samplesperpixel, NULL));
    }
    m = _TIFFmallocExt(tif, tif->tif_dir.td_samplesperpixel * sizeof(uint16_t));
    if (m == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }
    for (na = m, nb = 0; nb < tif->tif_dir.td_samplesperpixel; na++, nb++)
        *na = value;
    o = TIFFWriteDirectoryTagCheckedShortArray(
        tif, ndir, dir, tag, tif->tif_dir.td_samplesperpixel, m);
    _TIFFfreeExt(tif, m);
    return (o);
}

static int TIFFWriteDirectoryTagSshortArray(TIFF *tif, uint32_t *ndir,
                                            TIFFDirEntry *dir, uint16_t tag,
                                            uint32_t count, int16_t *value)
{
    return (TIFFWriteDirectoryTagCheckedSshortArray(tif, ndir, dir, tag, count,
                                                    value));
}

static int TIFFWriteDirectoryTagLong(TIFF *tif, uint32_t *ndir,
                                     TIFFDirEntry *dir, uint16_t tag,
                                     uint32_t value)
{
    return (TIFFWriteDirectoryTagCheckedLong(tif, ndir, dir, tag, value));
}

static int TIFFWriteDirectoryTagLongArray(TIFF *tif, uint32_t *ndir,
                                          TIFFDirEntry *dir, uint16_t tag,
                                          uint32_t count, uint32_t *value)
{
    return (TIFFWriteDirectoryTagCheckedLongArray(tif, ndir, dir, tag, count,
                                                  value));
}

static int TIFFWriteDirectoryTagSlongArray(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, int32_t *value)
{
    return (TIFFWriteDirectoryTagCheckedSlongArray(tif, ndir, dir, tag, count,
                                                   value));
}

/************************************************************************/
/*                 TIFFWriteDirectoryTagLong8Array()                    */
/*                                                                      */
/*      Write either Long8 or Long array depending on file type.        */
/************************************************************************/
static int TIFFWriteDirectoryTagLong8Array(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, uint64_t *value)
{
    static const char module[] = "TIFFWriteDirectoryTagLong8Array";
    uint64_t *ma;
    uint32_t mb;
    uint32_t *p;
    uint32_t *q;
    int o;

    /* is this just a counting pass? */
    if (dir == NULL)
    {
        /* only evaluate IFD data size and inc. ndir */
        return (TIFFWriteDirectoryTagCheckedLong8Array(tif, ndir, dir, tag,
                                                       count, value));
    }

    /* We always write Long8 for BigTIFF, no checking needed. */
    if (tif->tif_flags & TIFF_BIGTIFF)
        return (TIFFWriteDirectoryTagCheckedLong8Array(tif, ndir, dir, tag,
                                                       count, value));

    /*
    ** For classic tiff we want to verify everything is in range for long
    ** and convert to long format.
    */
    p = _TIFFmallocExt(tif, count * sizeof(uint32_t));
    if (p == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }

    for (q = p, ma = value, mb = 0; mb < count; ma++, mb++, q++)
    {
        if (*ma > 0xFFFFFFFF)
        {
            TIFFErrorExtR(tif, module,
                          "Attempt to write unsigned long value %" PRIu64
                          " larger than 0xFFFFFFFF for tag %d in Classic TIFF "
                          "file. TIFF file writing aborted",
                          *ma, tag);
            _TIFFfreeExt(tif, p);
            return (0);
        }
        *q = (uint32_t)(*ma);
    }

    o = TIFFWriteDirectoryTagCheckedLongArray(tif, ndir, dir, tag, count, p);
    _TIFFfreeExt(tif, p);

    return (o);
}

/************************************************************************/
/*                 TIFFWriteDirectoryTagSlong8Array()                   */
/*                                                                      */
/*      Write either SLong8 or SLong array depending on file type.      */
/************************************************************************/
static int TIFFWriteDirectoryTagSlong8Array(TIFF *tif, uint32_t *ndir,
                                            TIFFDirEntry *dir, uint16_t tag,
                                            uint32_t count, int64_t *value)
{
    static const char module[] = "TIFFWriteDirectoryTagSlong8Array";
    int64_t *ma;
    uint32_t mb;
    int32_t *p;
    int32_t *q;
    int o;

    /* is this just a counting pass? */
    if (dir == NULL)
    {
        /* only evaluate IFD data size and inc. ndir */
        return (TIFFWriteDirectoryTagCheckedSlong8Array(tif, ndir, dir, tag,
                                                        count, value));
    }
    /* We always write SLong8 for BigTIFF, no checking needed. */
    if (tif->tif_flags & TIFF_BIGTIFF)
        return (TIFFWriteDirectoryTagCheckedSlong8Array(tif, ndir, dir, tag,
                                                        count, value));

    /*
    ** For classic tiff we want to verify everything is in range for signed-long
    ** and convert to signed-long format.
    */
    p = _TIFFmallocExt(tif, count * sizeof(uint32_t));
    if (p == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }

    for (q = p, ma = value, mb = 0; mb < count; ma++, mb++, q++)
    {
        if (*ma > (2147483647))
        {
            TIFFErrorExtR(tif, module,
                          "Attempt to write signed long value %" PRIi64
                          " larger than 0x7FFFFFFF (2147483647) for tag %d in "
                          "Classic TIFF file. TIFF writing to file aborted",
                          *ma, tag);
            _TIFFfreeExt(tif, p);
            return (0);
        }
        else if (*ma < (-2147483647 - 1))
        {
            TIFFErrorExtR(tif, module,
                          "Attempt to write signed long value %" PRIi64
                          " smaller than 0x80000000 (-2147483648) for tag %d "
                          "in Classic TIFF file. TIFF writing to file aborted",
                          *ma, tag);
            _TIFFfreeExt(tif, p);
            return (0);
        }
        *q = (int32_t)(*ma);
    }

    o = TIFFWriteDirectoryTagCheckedSlongArray(tif, ndir, dir, tag, count, p);
    _TIFFfreeExt(tif, p);

    return (o);
}

static int TIFFWriteDirectoryTagRational(TIFF *tif, uint32_t *ndir,
                                         TIFFDirEntry *dir, uint16_t tag,
                                         double value)
{
    return (TIFFWriteDirectoryTagCheckedRational(tif, ndir, dir, tag, value));
}

static int TIFFWriteDirectoryTagRationalArray(TIFF *tif, uint32_t *ndir,
                                              TIFFDirEntry *dir, uint16_t tag,
                                              uint32_t count, float *value)
{
    return (TIFFWriteDirectoryTagCheckedRationalArray(tif, ndir, dir, tag,
                                                      count, value));
}

static int TIFFWriteDirectoryTagSrationalArray(TIFF *tif, uint32_t *ndir,
                                               TIFFDirEntry *dir, uint16_t tag,
                                               uint32_t count, float *value)
{
    return (TIFFWriteDirectoryTagCheckedSrationalArray(tif, ndir, dir, tag,
                                                       count, value));
}

/*-- Rational2Double: additional write functions */
static int TIFFWriteDirectoryTagRationalDoubleArray(TIFF *tif, uint32_t *ndir,
                                                    TIFFDirEntry *dir,
                                                    uint16_t tag,
                                                    uint32_t count,
                                                    double *value)
{
    return (TIFFWriteDirectoryTagCheckedRationalDoubleArray(tif, ndir, dir, tag,
                                                            count, value));
}

static int TIFFWriteDirectoryTagSrationalDoubleArray(TIFF *tif, uint32_t *ndir,
                                                     TIFFDirEntry *dir,
                                                     uint16_t tag,
                                                     uint32_t count,
                                                     double *value)
{
    return (TIFFWriteDirectoryTagCheckedSrationalDoubleArray(
        tif, ndir, dir, tag, count, value));
}

static int TIFFWriteDirectoryTagFloatArray(TIFF *tif, uint32_t *ndir,
                                           TIFFDirEntry *dir, uint16_t tag,
                                           uint32_t count, float *value)
{
    return (TIFFWriteDirectoryTagCheckedFloatArray(tif, ndir, dir, tag, count,
                                                   value));
}

static int TIFFWriteDirectoryTagDoubleArray(TIFF *tif, uint32_t *ndir,
                                            TIFFDirEntry *dir, uint16_t tag,
                                            uint32_t count, double *value)
{
    return (TIFFWriteDirectoryTagCheckedDoubleArray(tif, ndir, dir, tag, count,
                                                    value));
}

static int TIFFWriteDirectoryTagIfdArray(TIFF *tif, uint32_t *ndir,
                                         TIFFDirEntry *dir, uint16_t tag,
                                         uint32_t count, uint32_t *value)
{
    return (TIFFWriteDirectoryTagCheckedIfdArray(tif, ndir, dir, tag, count,
                                                 value));
}

static int TIFFWriteDirectoryTagShortLong(TIFF *tif, uint32_t *ndir,
                                          TIFFDirEntry *dir, uint16_t tag,
                                          uint32_t value)
{
    if (value <= 0xFFFF)
        return (TIFFWriteDirectoryTagCheckedShort(tif, ndir, dir, tag,
                                                  (uint16_t)value));
    else
        return (TIFFWriteDirectoryTagCheckedLong(tif, ndir, dir, tag, value));
}

static int _WriteAsType(TIFF *tif, uint64_t strile_size,
                        uint64_t uncompressed_threshold)
{
    const uint16_t compression = tif->tif_dir.td_compression;
    if (compression == COMPRESSION_NONE)
    {
        return strile_size > uncompressed_threshold;
    }
    else if (compression == COMPRESSION_JPEG ||
             compression == COMPRESSION_LZW ||
             compression == COMPRESSION_ADOBE_DEFLATE ||
             compression == COMPRESSION_DEFLATE ||
             compression == COMPRESSION_LZMA ||
             compression == COMPRESSION_LERC ||
             compression == COMPRESSION_ZSTD ||
             compression == COMPRESSION_WEBP || compression == COMPRESSION_JXL)
    {
        /* For a few select compression types, we assume that in the worst */
        /* case the compressed size will be 10 times the uncompressed size. */
        /* This is overly pessismistic ! */
        return strile_size >= uncompressed_threshold / 10;
    }
    return 1;
}

static int WriteAsLong8(TIFF *tif, uint64_t strile_size)
{
    return _WriteAsType(tif, strile_size, 0xFFFFFFFFU);
}

static int WriteAsLong4(TIFF *tif, uint64_t strile_size)
{
    return _WriteAsType(tif, strile_size, 0xFFFFU);
}

/************************************************************************/
/*                TIFFWriteDirectoryTagLongLong8Array()                 */
/*                                                                      */
/*      Write out LONG8 array and write a SHORT/LONG/LONG8 depending    */
/*      on strile size and Classic/BigTIFF mode.                        */
/************************************************************************/

static int TIFFWriteDirectoryTagLongLong8Array(TIFF *tif, uint32_t *ndir,
                                               TIFFDirEntry *dir, uint16_t tag,
                                               uint32_t count, uint64_t *value)
{
    static const char module[] = "TIFFWriteDirectoryTagLongLong8Array";
    int o;
    int write_aslong4;

    if (tif->tif_dir.td_deferstrilearraywriting)
    {
        if (dir == NULL)
        {
            /* This is just a counting pass to count IFD entries.
             * For deferstrilearraywriting no extra bytes will be written
             * into IFD space. */
            (*ndir)++;
            return 1;
        }
        return TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_NOTYPE, 0, 0,
                                         NULL);
    }

    if (tif->tif_flags & TIFF_BIGTIFF)
    {
        int write_aslong8 = 1;
        /* In the case of ByteCounts array, we may be able to write them on LONG
         * if the strip/tilesize is not too big. Also do that for count > 1 in
         * the case someone would want to create a single-strip file with a
         * growing height, in which case using LONG8 will be safer. */
        if (count > 1 && tag == TIFFTAG_STRIPBYTECOUNTS)
        {
            write_aslong8 = WriteAsLong8(tif, TIFFStripSize64(tif));
        }
        else if (count > 1 && tag == TIFFTAG_TILEBYTECOUNTS)
        {
            write_aslong8 = WriteAsLong8(tif, TIFFTileSize64(tif));
        }
        if (write_aslong8)
        {
            return TIFFWriteDirectoryTagCheckedLong8Array(tif, ndir, dir, tag,
                                                          count, value);
        }
    }

    write_aslong4 = 1;
    if (count > 1 && tag == TIFFTAG_STRIPBYTECOUNTS)
    {
        write_aslong4 = WriteAsLong4(tif, TIFFStripSize64(tif));
    }
    else if (count > 1 && tag == TIFFTAG_TILEBYTECOUNTS)
    {
        write_aslong4 = WriteAsLong4(tif, TIFFTileSize64(tif));
    }
    if (write_aslong4)
    {
        /*
        ** For classic tiff we want to verify everything is in range for LONG
        ** and convert to long format.
        */

        uint32_t *p = _TIFFmallocExt(tif, count * sizeof(uint32_t));
        uint32_t *q;
        uint64_t *ma;
        uint32_t mb;

        if (p == NULL)
        {
            TIFFErrorExtR(tif, module, "Out of memory");
            return (0);
        }

        for (q = p, ma = value, mb = 0; mb < count; ma++, mb++, q++)
        {
            if (*ma > 0xFFFFFFFF)
            {
                TIFFErrorExtR(tif, module,
                              "Attempt to write value larger than 0xFFFFFFFF "
                              "in LONG array.");
                _TIFFfreeExt(tif, p);
                return (0);
            }
            *q = (uint32_t)(*ma);
        }

        o = TIFFWriteDirectoryTagCheckedLongArray(tif, ndir, dir, tag, count,
                                                  p);
        _TIFFfreeExt(tif, p);
    }
    else
    {
        uint16_t *p = _TIFFmallocExt(tif, count * sizeof(uint16_t));
        uint16_t *q;
        uint64_t *ma;
        uint32_t mb;

        if (p == NULL)
        {
            TIFFErrorExtR(tif, module, "Out of memory");
            return (0);
        }

        for (q = p, ma = value, mb = 0; mb < count; ma++, mb++, q++)
        {
            if (*ma > 0xFFFF)
            {
                /* Should not happen normally given the check we did before */
                TIFFErrorExtR(tif, module,
                              "Attempt to write value larger than 0xFFFF in "
                              "SHORT array.");
                _TIFFfreeExt(tif, p);
                return (0);
            }
            *q = (uint16_t)(*ma);
        }

        o = TIFFWriteDirectoryTagCheckedShortArray(tif, ndir, dir, tag, count,
                                                   p);
        _TIFFfreeExt(tif, p);
    }

    return (o);
}

/************************************************************************/
/*                 TIFFWriteDirectoryTagIfdIfd8Array()                  */
/*                                                                      */
/*      Write either IFD8 or IFD array depending on file type.          */
/************************************************************************/

static int TIFFWriteDirectoryTagIfdIfd8Array(TIFF *tif, uint32_t *ndir,
                                             TIFFDirEntry *dir, uint16_t tag,
                                             uint32_t count, uint64_t *value)
{
    static const char module[] = "TIFFWriteDirectoryTagIfdIfd8Array";
    uint64_t *ma;
    uint32_t mb;
    uint32_t *p;
    uint32_t *q;
    int o;

    /* We always write IFD8 for BigTIFF, no checking needed. */
    if (tif->tif_flags & TIFF_BIGTIFF)
        return TIFFWriteDirectoryTagCheckedIfd8Array(tif, ndir, dir, tag, count,
                                                     value);

    /*
    ** For classic tiff we want to verify everything is in range for IFD
    ** and convert to long format.
    */

    p = _TIFFmallocExt(tif, count * sizeof(uint32_t));
    if (p == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }

    for (q = p, ma = value, mb = 0; mb < count; ma++, mb++, q++)
    {
        if (*ma > 0xFFFFFFFF)
        {
            TIFFErrorExtR(tif, module,
                          "Attempt to write value larger than 0xFFFFFFFF in "
                          "Classic TIFF file.");
            _TIFFfreeExt(tif, p);
            return (0);
        }
        *q = (uint32_t)(*ma);
    }

    o = TIFFWriteDirectoryTagCheckedIfdArray(tif, ndir, dir, tag, count, p);
    _TIFFfreeExt(tif, p);

    return (o);
}

/*
 * Auxiliary function to determine the IFD data size to be written to the file.
 * The IFD data size is finally the size of the IFD tag entries plus the IFD
 * data that is written directly after the IFD tag entries.
 */
static void EvaluateIFDdatasizeWrite(TIFF *tif, uint32_t count,
                                     uint32_t typesize, uint32_t *ndir)
{
    uint64_t datalength = (uint64_t)count * typesize;
    if (datalength > ((tif->tif_flags & TIFF_BIGTIFF) ? 0x8U : 0x4U))
    {
        /* LibTIFF increments write address to an even offset, thus datalength
         * written is also incremented. */
        if (datalength & 1)
            datalength++;
        tif->tif_dir.td_dirdatasize_write += datalength;
    }
    (*ndir)++;
}

static int TIFFWriteDirectoryTagColormap(TIFF *tif, uint32_t *ndir,
                                         TIFFDirEntry *dir)
{
    static const char module[] = "TIFFWriteDirectoryTagColormap";
    uint32_t m;
    uint16_t *n;
    int o;
    m = (1 << tif->tif_dir.td_bitspersample);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, 3 * m, sizeof(uint16_t), ndir);
        return 1;
    }

    n = _TIFFmallocExt(tif, 3 * m * sizeof(uint16_t));
    if (n == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }
    _TIFFmemcpy(&n[0], tif->tif_dir.td_colormap[0], m * sizeof(uint16_t));
    _TIFFmemcpy(&n[m], tif->tif_dir.td_colormap[1], m * sizeof(uint16_t));
    _TIFFmemcpy(&n[2 * m], tif->tif_dir.td_colormap[2], m * sizeof(uint16_t));
    o = TIFFWriteDirectoryTagCheckedShortArray(tif, ndir, dir, TIFFTAG_COLORMAP,
                                               3 * m, n);
    _TIFFfreeExt(tif, n);
    return (o);
}

static int TIFFWriteDirectoryTagTransferfunction(TIFF *tif, uint32_t *ndir,
                                                 TIFFDirEntry *dir)
{
    static const char module[] = "TIFFWriteDirectoryTagTransferfunction";
    uint32_t m;
    uint16_t n;
    uint16_t *o;
    int p;
    /* TIFFTAG_TRANSFERFUNCTION expects (1 or 3) pointer to arrays with
     *  (1 << BitsPerSample) * uint16_t values.
     */
    m = (1 << tif->tif_dir.td_bitspersample);
    /* clang-format off */
    n = (tif->tif_dir.td_samplesperpixel - tif->tif_dir.td_extrasamples) > 1 ? 3 : 1;
    /* clang-format on */

    /* Check for proper number of transferfunctions */
    for (int i = 0; i < n; i++)
    {
        if (tif->tif_dir.td_transferfunction[i] == NULL)
        {
            TIFFWarningExtR(tif, module,
                            "Too few TransferFunctions provided. Tag "
                            "not written to file");
            return (1); /* Not an error; only tag is not written. */
        }
    }
    /*
     * Check if the table can be written as a single column,
     * or if it must be written as 3 columns.  Note that we
     * write a 3-column tag if there are 2 samples/pixel and
     * a single column of data won't suffice--hmm.
     */
    if (n == 3)
    {
        if (!_TIFFmemcmp(tif->tif_dir.td_transferfunction[0],
                         tif->tif_dir.td_transferfunction[2],
                         m * sizeof(uint16_t)) &&
            !_TIFFmemcmp(tif->tif_dir.td_transferfunction[0],
                         tif->tif_dir.td_transferfunction[1],
                         m * sizeof(uint16_t)))
            n = 1;
    }
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, n * m, 2, ndir);
        return 1;
    }

    o = _TIFFmallocExt(tif, n * m * sizeof(uint16_t));
    if (o == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }
    _TIFFmemcpy(&o[0], tif->tif_dir.td_transferfunction[0],
                m * sizeof(uint16_t));
    if (n > 1)
        _TIFFmemcpy(&o[m], tif->tif_dir.td_transferfunction[1],
                    m * sizeof(uint16_t));
    if (n > 2)
        _TIFFmemcpy(&o[2 * m], tif->tif_dir.td_transferfunction[2],
                    m * sizeof(uint16_t));
    p = TIFFWriteDirectoryTagCheckedShortArray(
        tif, ndir, dir, TIFFTAG_TRANSFERFUNCTION, n * m, o);
    _TIFFfreeExt(tif, o);
    return (p);
}

static int TIFFWriteDirectoryTagSubifd(TIFF *tif, uint32_t *ndir,
                                       TIFFDirEntry *dir)
{
    static const char module[] = "TIFFWriteDirectoryTagSubifd";
    uint64_t m;
    int n;
    if (tif->tif_dir.td_nsubifd == 0)
        return (1);
    m = tif->tif_dataoff;
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        uint32_t *o;
        uint64_t *pa;
        uint32_t *pb;
        uint16_t p;
        o = _TIFFmallocExt(tif, tif->tif_dir.td_nsubifd * sizeof(uint32_t));
        if (o == NULL)
        {
            TIFFErrorExtR(tif, module, "Out of memory");
            return (0);
        }
        pa = tif->tif_dir.td_subifd;
        pb = o;
        for (p = 0; p < tif->tif_dir.td_nsubifd; p++)
        {
            assert(pa != 0);

            /* Could happen if an classicTIFF has a SubIFD of type LONG8 (which
             * is illegal) */
            if (*pa > 0xFFFFFFFFUL)
            {
                TIFFErrorExtR(tif, module, "Illegal value for SubIFD tag");
                _TIFFfreeExt(tif, o);
                return (0);
            }
            *pb++ = (uint32_t)(*pa++);
        }
        n = TIFFWriteDirectoryTagCheckedIfdArray(tif, ndir, dir, TIFFTAG_SUBIFD,
                                                 tif->tif_dir.td_nsubifd, o);
        _TIFFfreeExt(tif, o);
    }
    else
        n = TIFFWriteDirectoryTagCheckedIfd8Array(
            tif, ndir, dir, TIFFTAG_SUBIFD, tif->tif_dir.td_nsubifd,
            tif->tif_dir.td_subifd);

    if (dir == NULL)
        /* Just have evaluated IFD data size and incremented ndir
         * above in sub-functions. */
        return (n);

    if (!n)
        return (0);
    /*
     * Total hack: if this directory includes a SubIFD
     * tag then force the next <n> directories to be
     * written as ``sub directories'' of this one.  This
     * is used to write things like thumbnails and
     * image masks that one wants to keep out of the
     * normal directory linkage access mechanism.
     */
    tif->tif_flags |= TIFF_INSUBIFD;
    tif->tif_nsubifd = tif->tif_dir.td_nsubifd;
    if (tif->tif_dir.td_nsubifd == 1)
        tif->tif_subifdoff = 0;
    else
        tif->tif_subifdoff = m;
    return (1);
}

static int TIFFWriteDirectoryTagCheckedAscii(TIFF *tif, uint32_t *ndir,
                                             TIFFDirEntry *dir, uint16_t tag,
                                             uint32_t count, char *value)
{
    assert(sizeof(char) == 1);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 1, ndir);
        return 1;
    }
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_ASCII, count,
                                      count, value));
}

static int TIFFWriteDirectoryTagCheckedUndefinedArray(TIFF *tif, uint32_t *ndir,
                                                      TIFFDirEntry *dir,
                                                      uint16_t tag,
                                                      uint32_t count,
                                                      uint8_t *value)
{
    assert(sizeof(uint8_t) == 1);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 1, ndir);
        return 1;
    }
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_UNDEFINED,
                                      count, count, value));
}

static int TIFFWriteDirectoryTagCheckedByteArray(TIFF *tif, uint32_t *ndir,
                                                 TIFFDirEntry *dir,
                                                 uint16_t tag, uint32_t count,
                                                 uint8_t *value)
{
    assert(sizeof(uint8_t) == 1);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 1, ndir);
        return 1;
    }
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_BYTE, count,
                                      count, value));
}

static int TIFFWriteDirectoryTagCheckedSbyteArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  int8_t *value)
{
    assert(sizeof(int8_t) == 1);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 1, ndir);
        return 1;
    }
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_SBYTE, count,
                                      count, value));
}

static int TIFFWriteDirectoryTagCheckedShort(TIFF *tif, uint32_t *ndir,
                                             TIFFDirEntry *dir, uint16_t tag,
                                             uint16_t value)
{
    uint16_t m;
    assert(sizeof(uint16_t) == 2);
    if (dir == NULL)
    {
        /* No additional data to IFD data size just increment ndir. */
        (*ndir)++;
        return 1;
    }
    m = value;
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabShort(&m);
    return (
        TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_SHORT, 1, 2, &m));
}

static int TIFFWriteDirectoryTagCheckedShortArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  uint16_t *value)
{
    assert(count < 0x80000000);
    assert(sizeof(uint16_t) == 2);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 2, ndir);
        return 1;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfShort(value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_SHORT, count,
                                      count * 2, value));
}

static int TIFFWriteDirectoryTagCheckedSshortArray(TIFF *tif, uint32_t *ndir,
                                                   TIFFDirEntry *dir,
                                                   uint16_t tag, uint32_t count,
                                                   int16_t *value)
{
    assert(count < 0x80000000);
    assert(sizeof(int16_t) == 2);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 2, ndir);
        return 1;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfShort((uint16_t *)value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_SSHORT, count,
                                      count * 2, value));
}

static int TIFFWriteDirectoryTagCheckedLong(TIFF *tif, uint32_t *ndir,
                                            TIFFDirEntry *dir, uint16_t tag,
                                            uint32_t value)
{
    uint32_t m;
    assert(sizeof(uint32_t) == 4);
    if (dir == NULL)
    {
        /* No additional data to IFD data size just increment ndir. */
        (*ndir)++;
        return 1;
    }
    m = value;
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabLong(&m);
    return (
        TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_LONG, 1, 4, &m));
}

static int TIFFWriteDirectoryTagCheckedLongArray(TIFF *tif, uint32_t *ndir,
                                                 TIFFDirEntry *dir,
                                                 uint16_t tag, uint32_t count,
                                                 uint32_t *value)
{
    assert(count < 0x40000000);
    assert(sizeof(uint32_t) == 4);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 4, ndir);
        return 1;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong(value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_LONG, count,
                                      count * 4, value));
}

static int TIFFWriteDirectoryTagCheckedSlongArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  int32_t *value)
{
    assert(count < 0x40000000);
    assert(sizeof(int32_t) == 4);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 4, ndir);
        return 1;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong((uint32_t *)value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_SLONG, count,
                                      count * 4, value));
}

static int TIFFWriteDirectoryTagCheckedLong8Array(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  uint64_t *value)
{
    assert(count < 0x20000000);
    assert(sizeof(uint64_t) == 8);
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        TIFFErrorExtR(tif, "TIFFWriteDirectoryTagCheckedLong8Array",
                      "LONG8 not allowed for ClassicTIFF");
        return (0);
    }
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 8, ndir);
        return 1;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong8(value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_LONG8, count,
                                      count * 8, value));
}

static int TIFFWriteDirectoryTagCheckedSlong8Array(TIFF *tif, uint32_t *ndir,
                                                   TIFFDirEntry *dir,
                                                   uint16_t tag, uint32_t count,
                                                   int64_t *value)
{
    assert(count < 0x20000000);
    assert(sizeof(int64_t) == 8);
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        TIFFErrorExtR(tif, "TIFFWriteDirectoryTagCheckedSlong8Array",
                      "SLONG8 not allowed for ClassicTIFF");
        return (0);
    }
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 8, ndir);
        return 1;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong8((uint64_t *)value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_SLONG8, count,
                                      count * 8, value));
}

static int TIFFWriteDirectoryTagCheckedRational(TIFF *tif, uint32_t *ndir,
                                                TIFFDirEntry *dir, uint16_t tag,
                                                double value)
{
    static const char module[] = "TIFFWriteDirectoryTagCheckedRational";
    uint32_t m[2];
    assert(sizeof(uint32_t) == 4);
    if (value < 0)
    {
        TIFFErrorExtR(tif, module, "Negative value is illegal");
        return 0;
    }
    else if (value != value)
    {
        TIFFErrorExtR(tif, module, "Not-a-number value is illegal");
        return 0;
    }

    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        tif->tif_dir.td_dirdatasize_write +=
            (tif->tif_flags & TIFF_BIGTIFF) ? 0 : 0x8U;
        (*ndir)++;
        return 1;
    }

    DoubleToRational(value, &m[0], &m[1]);

    if (tif->tif_flags & TIFF_SWAB)
    {
        TIFFSwabLong(&m[0]);
        TIFFSwabLong(&m[1]);
    }
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_RATIONAL, 1, 8,
                                      &m[0]));
}

static int TIFFWriteDirectoryTagCheckedRationalArray(TIFF *tif, uint32_t *ndir,
                                                     TIFFDirEntry *dir,
                                                     uint16_t tag,
                                                     uint32_t count,
                                                     float *value)
{
    static const char module[] = "TIFFWriteDirectoryTagCheckedRationalArray";
    uint32_t *m;
    float *na;
    uint32_t *nb;
    uint32_t nc;
    int o;
    assert(sizeof(uint32_t) == 4);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count * 2, sizeof(uint32_t), ndir);
        return 1;
    }
    m = _TIFFmallocExt(tif, count * 2 * sizeof(uint32_t));
    if (m == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }
    for (na = value, nb = m, nc = 0; nc < count; na++, nb += 2, nc++)
    {
        DoubleToRational(*na, &nb[0], &nb[1]);
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong(m, count * 2);
    o = TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_RATIONAL, count,
                                  count * 8, &m[0]);
    _TIFFfreeExt(tif, m);
    return (o);
}

static int TIFFWriteDirectoryTagCheckedSrationalArray(TIFF *tif, uint32_t *ndir,
                                                      TIFFDirEntry *dir,
                                                      uint16_t tag,
                                                      uint32_t count,
                                                      float *value)
{
    static const char module[] = "TIFFWriteDirectoryTagCheckedSrationalArray";
    int32_t *m;
    float *na;
    int32_t *nb;
    uint32_t nc;
    int o;
    assert(sizeof(int32_t) == 4);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count * 2, sizeof(int32_t), ndir);
        return 1;
    }
    m = _TIFFmallocExt(tif, count * 2 * sizeof(int32_t));
    if (m == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }
    for (na = value, nb = m, nc = 0; nc < count; na++, nb += 2, nc++)
    {
        DoubleToSrational(*na, &nb[0], &nb[1]);
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong((uint32_t *)m, count * 2);
    o = TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_SRATIONAL, count,
                                  count * 8, &m[0]);
    _TIFFfreeExt(tif, m);
    return (o);
}

/*-- Rational2Double: additional write functions for double arrays */
static int
TIFFWriteDirectoryTagCheckedRationalDoubleArray(TIFF *tif, uint32_t *ndir,
                                                TIFFDirEntry *dir, uint16_t tag,
                                                uint32_t count, double *value)
{
    static const char module[] =
        "TIFFWriteDirectoryTagCheckedRationalDoubleArray";
    uint32_t *m;
    double *na;
    uint32_t *nb;
    uint32_t nc;
    int o;
    assert(sizeof(uint32_t) == 4);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count * 2, sizeof(uint32_t), ndir);
        return 1;
    }
    m = _TIFFmallocExt(tif, count * 2 * sizeof(uint32_t));
    if (m == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }
    for (na = value, nb = m, nc = 0; nc < count; na++, nb += 2, nc++)
    {
        DoubleToRational(*na, &nb[0], &nb[1]);
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong(m, count * 2);
    o = TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_RATIONAL, count,
                                  count * 8, &m[0]);
    _TIFFfreeExt(tif, m);
    return (o);
} /*-- TIFFWriteDirectoryTagCheckedRationalDoubleArray() ------- */

static int TIFFWriteDirectoryTagCheckedSrationalDoubleArray(
    TIFF *tif, uint32_t *ndir, TIFFDirEntry *dir, uint16_t tag, uint32_t count,
    double *value)
{
    static const char module[] =
        "TIFFWriteDirectoryTagCheckedSrationalDoubleArray";
    int32_t *m;
    double *na;
    int32_t *nb;
    uint32_t nc;
    int o;
    assert(sizeof(int32_t) == 4);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count * 2, sizeof(int32_t), ndir);
        return 1;
    }
    m = _TIFFmallocExt(tif, count * 2 * sizeof(int32_t));
    if (m == NULL)
    {
        TIFFErrorExtR(tif, module, "Out of memory");
        return (0);
    }
    for (na = value, nb = m, nc = 0; nc < count; na++, nb += 2, nc++)
    {
        DoubleToSrational(*na, &nb[0], &nb[1]);
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong((uint32_t *)m, count * 2);
    o = TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_SRATIONAL, count,
                                  count * 8, &m[0]);
    _TIFFfreeExt(tif, m);
    return (o);
} /*--- TIFFWriteDirectoryTagCheckedSrationalDoubleArray() -------- */

/** -----  Rational2Double: Double To Rational Conversion
----------------------------------------------------------
* There is a mathematical theorem to convert real numbers into a rational
(integer fraction) number.
* This is called "continuous fraction" which uses the Euclidean algorithm to
find the greatest common divisor (GCD).
*  (ref. e.g. https://de.wikipedia.org/wiki/Kettenbruch or
https://en.wikipedia.org/wiki/Continued_fraction
*             https://en.wikipedia.org/wiki/Euclidean_algorithm)
* The following functions implement the
* - ToRationalEuclideanGCD()		auxiliary function which mainly
implements euclidean GCD
* - DoubleToRational()			conversion function for un-signed
rationals
* - DoubleToSrational()			conversion function for signed rationals
------------------------------------------------------------------------------------------------------------------*/

/**---- ToRationalEuclideanGCD() -----------------------------------------
* Calculates the rational fractional of a double input value
* using the Euclidean algorithm to find the greatest common divisor (GCD)
------------------------------------------------------------------------*/
static void ToRationalEuclideanGCD(double value, int blnUseSignedRange,
                                   int blnUseSmallRange, uint64_t *ullNum,
                                   uint64_t *ullDenom)
{
    /* Internally, the integer variables can be bigger than the external ones,
     * as long as the result will fit into the external variable size.
     */
    uint64_t numSum[3] = {0, 1, 0}, denomSum[3] = {1, 0, 0};
    uint64_t aux, bigNum, bigDenom;
    uint64_t returnLimit;
    int i;
    uint64_t nMax;
    double fMax;
    unsigned long maxDenom;
    /*-- nMax and fMax defines the initial accuracy of the starting fractional,
     *   or better, the highest used integer numbers used within the starting
     * fractional (bigNum/bigDenom). There are two approaches, which can
     * accidentally lead to different accuracies just depending on the value.
     *   Therefore, blnUseSmallRange steers this behavior.
     *   For long long nMax = ((9223372036854775807-1)/2); for long nMax =
     * ((2147483647-1)/2);
     */
    if (blnUseSmallRange)
    {
        nMax = (uint64_t)((2147483647 - 1) / 2); /* for ULONG range */
    }
    else
    {
        nMax = ((9223372036854775807 - 1) / 2); /* for ULLONG range */
    }
    fMax = (double)nMax;

    /*-- For the Euclidean GCD define the denominator range, so that it stays
     * within size of unsigned long variables. maxDenom should be LONG_MAX for
     * negative values and ULONG_MAX for positive ones. Also the final returned
     * value of ullNum and ullDenom is limited according to signed- or
     * unsigned-range.
     */
    if (blnUseSignedRange)
    {
        maxDenom = 2147483647UL; /*LONG_MAX = 0x7FFFFFFFUL*/
        returnLimit = maxDenom;
    }
    else
    {
        maxDenom = 0xFFFFFFFFUL; /*ULONG_MAX = 0xFFFFFFFFUL*/
        returnLimit = maxDenom;
    }

    /*-- First generate a rational fraction (bigNum/bigDenom) which represents
     *the value as a rational number with the highest accuracy. Therefore,
     *uint64_t (uint64_t) is needed. This rational fraction is then reduced
     *using the Euclidean algorithm to find the greatest common divisor (GCD).
     *   bigNum   = big numinator of value without fraction (or cut residual
     *fraction) bigDenom = big denominator of value
     *-- Break-criteria so that uint64_t cast to "bigNum" introduces no error
     *and bigDenom has no overflow, and stop with enlargement of fraction when
     *the double-value of it reaches an integer number without fractional part.
     */
    bigDenom = 1;
    while ((value != floor(value)) && (value < fMax) && (bigDenom < nMax))
    {
        bigDenom <<= 1;
        value *= 2;
    }
    bigNum = (uint64_t)value;

    /*-- Start Euclidean algorithm to find the greatest common divisor (GCD) --
     */
#define MAX_ITERATIONS 64
    for (i = 0; i < MAX_ITERATIONS; i++)
    {
        uint64_t val;
        /* if bigDenom is not zero, calculate integer part of fraction. */
        if (bigDenom == 0)
        {
            break;
        }
        val = bigNum / bigDenom;

        /* Set bigDenom to reminder of bigNum/bigDenom and bigNum to previous
         * denominator bigDenom. */
        aux = bigNum;
        bigNum = bigDenom;
        bigDenom = aux % bigDenom;

        /* calculate next denominator and check for its given maximum */
        aux = val;
        if (denomSum[1] * val + denomSum[0] >= maxDenom)
        {
            aux = (maxDenom - denomSum[0]) / denomSum[1];
            if (aux * 2 >= val || denomSum[1] >= maxDenom)
                i = (MAX_ITERATIONS +
                     1); /* exit but execute rest of for-loop */
            else
                break;
        }
        /* calculate next numerator to numSum2 and save previous one to numSum0;
         * numSum1 just copy of numSum2. */
        numSum[2] = aux * numSum[1] + numSum[0];
        numSum[0] = numSum[1];
        numSum[1] = numSum[2];
        /* calculate next denominator to denomSum2 and save previous one to
         * denomSum0; denomSum1 just copy of denomSum2. */
        denomSum[2] = aux * denomSum[1] + denomSum[0];
        denomSum[0] = denomSum[1];
        denomSum[1] = denomSum[2];
    }

    /*-- Check and adapt for final variable size and return values; reduces
     * internal accuracy; denominator is kept in ULONG-range with maxDenom -- */
    while (numSum[1] > returnLimit || denomSum[1] > returnLimit)
    {
        numSum[1] = numSum[1] / 2;
        denomSum[1] = denomSum[1] / 2;
    }

    /* return values */
    *ullNum = numSum[1];
    *ullDenom = denomSum[1];

} /*-- ToRationalEuclideanGCD() -------------- */

/**---- DoubleToRational() -----------------------------------------------
* Calculates the rational fractional of a double input value
* for UN-SIGNED rationals,
* using the Euclidean algorithm to find the greatest common divisor (GCD)
------------------------------------------------------------------------*/
static void DoubleToRational(double value, uint32_t *num, uint32_t *denom)
{
    /*---- UN-SIGNED RATIONAL ---- */
    double dblDiff, dblDiff2;
    uint64_t ullNum, ullDenom, ullNum2, ullDenom2;

    /*-- Check for negative values. If so it is an error. */
    /* Test written that way to catch NaN */
    if (!(value >= 0))
    {
        *num = *denom = 0;
        TIFFErrorExt(0, "TIFFLib: DoubleToRational()",
                     " Negative Value for Unsigned Rational given.");
        return;
    }

    /*-- Check for too big numbers (> ULONG_MAX) -- */
    if (value > 0xFFFFFFFFUL)
    {
        *num = 0xFFFFFFFFU;
        *denom = 0;
        return;
    }
    /*-- Check for easy integer numbers -- */
    if (value == (uint32_t)(value))
    {
        *num = (uint32_t)value;
        *denom = 1;
        return;
    }
    /*-- Check for too small numbers for "unsigned long" type rationals -- */
    if (value < 1.0 / (double)0xFFFFFFFFUL)
    {
        *num = 0;
        *denom = 0xFFFFFFFFU;
        return;
    }

    /*-- There are two approaches using the Euclidean algorithm,
     *   which can accidentally lead to different accuracies just depending on
     * the value. Try both and define which one was better.
     */
    ToRationalEuclideanGCD(value, FALSE, FALSE, &ullNum, &ullDenom);
    ToRationalEuclideanGCD(value, FALSE, TRUE, &ullNum2, &ullDenom2);
    /*-- Double-Check, that returned values fit into ULONG :*/
    if (ullNum > 0xFFFFFFFFUL || ullDenom > 0xFFFFFFFFUL ||
        ullNum2 > 0xFFFFFFFFUL || ullDenom2 > 0xFFFFFFFFUL)
    {
        TIFFErrorExt(0, "TIFFLib: DoubleToRational()",
                     " Num or Denom exceeds ULONG: val=%14.6f, num=%12" PRIu64
                     ", denom=%12" PRIu64 " | num2=%12" PRIu64
                     ", denom2=%12" PRIu64 "",
                     value, ullNum, ullDenom, ullNum2, ullDenom2);
        assert(0);
    }

    /* Check, which one has higher accuracy and take that. */
    dblDiff = fabs(value - ((double)ullNum / (double)ullDenom));
    dblDiff2 = fabs(value - ((double)ullNum2 / (double)ullDenom2));
    if (dblDiff < dblDiff2)
    {
        *num = (uint32_t)ullNum;
        *denom = (uint32_t)ullDenom;
    }
    else
    {
        *num = (uint32_t)ullNum2;
        *denom = (uint32_t)ullDenom2;
    }
} /*-- DoubleToRational() -------------- */

/**---- DoubleToSrational() -----------------------------------------------
* Calculates the rational fractional of a double input value
* for SIGNED rationals,
* using the Euclidean algorithm to find the greatest common divisor (GCD)
------------------------------------------------------------------------*/
static void DoubleToSrational(double value, int32_t *num, int32_t *denom)
{
    /*---- SIGNED RATIONAL ----*/
    int neg = 1;
    double dblDiff, dblDiff2;
    uint64_t ullNum, ullDenom, ullNum2, ullDenom2;

    /*-- Check for negative values and use then the positive one for internal
     * calculations, but take the sign into account before returning. */
    if (value < 0)
    {
        neg = -1;
        value = -value;
    }

    /*-- Check for too big numbers (> LONG_MAX) -- */
    if (value > 0x7FFFFFFFL)
    {
        *num = 0x7FFFFFFFL;
        *denom = 0;
        return;
    }
    /*-- Check for easy numbers -- */
    if (value == (int32_t)(value))
    {
        *num = (int32_t)(neg * value);
        *denom = 1;
        return;
    }
    /*-- Check for too small numbers for "long" type rationals -- */
    if (value < 1.0 / (double)0x7FFFFFFFL)
    {
        *num = 0;
        *denom = 0x7FFFFFFFL;
        return;
    }

    /*-- There are two approaches using the Euclidean algorithm,
     *   which can accidentally lead to different accuracies just depending on
     * the value. Try both and define which one was better. Furthermore, set
     * behavior of ToRationalEuclideanGCD() to the range of signed-long.
     */
    ToRationalEuclideanGCD(value, TRUE, FALSE, &ullNum, &ullDenom);
    ToRationalEuclideanGCD(value, TRUE, TRUE, &ullNum2, &ullDenom2);
    /*-- Double-Check, that returned values fit into LONG :*/
    if (ullNum > 0x7FFFFFFFL || ullDenom > 0x7FFFFFFFL ||
        ullNum2 > 0x7FFFFFFFL || ullDenom2 > 0x7FFFFFFFL)
    {
        TIFFErrorExt(0, "TIFFLib: DoubleToSrational()",
                     " Num or Denom exceeds LONG: val=%14.6f, num=%12" PRIu64
                     ", denom=%12" PRIu64 " | num2=%12" PRIu64
                     ", denom2=%12" PRIu64 "",
                     neg * value, ullNum, ullDenom, ullNum2, ullDenom2);
        assert(0);
    }

    /* Check, which one has higher accuracy and take that. */
    dblDiff = fabs(value - ((double)ullNum / (double)ullDenom));
    dblDiff2 = fabs(value - ((double)ullNum2 / (double)ullDenom2));
    if (dblDiff < dblDiff2)
    {
        *num = (int32_t)(neg * (long)ullNum);
        *denom = (int32_t)ullDenom;
    }
    else
    {
        *num = (int32_t)(neg * (long)ullNum2);
        *denom = (int32_t)ullDenom2;
    }
} /*-- DoubleToSrational() --------------*/

static int TIFFWriteDirectoryTagCheckedFloatArray(TIFF *tif, uint32_t *ndir,
                                                  TIFFDirEntry *dir,
                                                  uint16_t tag, uint32_t count,
                                                  float *value)
{
    assert(count < 0x40000000);
    assert(sizeof(float) == 4);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 4, ndir);
        return 1;
    }
    TIFFCvtNativeToIEEEFloat(tif, count, value);
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfFloat(value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_FLOAT, count,
                                      count * 4, value));
}

static int TIFFWriteDirectoryTagCheckedDoubleArray(TIFF *tif, uint32_t *ndir,
                                                   TIFFDirEntry *dir,
                                                   uint16_t tag, uint32_t count,
                                                   double *value)
{
    assert(count < 0x20000000);
    assert(sizeof(double) == 8);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 8, ndir);
        return 1;
    }
    TIFFCvtNativeToIEEEDouble(tif, count, value);
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfDouble(value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_DOUBLE, count,
                                      count * 8, value));
}

static int TIFFWriteDirectoryTagCheckedIfdArray(TIFF *tif, uint32_t *ndir,
                                                TIFFDirEntry *dir, uint16_t tag,
                                                uint32_t count, uint32_t *value)
{
    assert(count < 0x40000000);
    assert(sizeof(uint32_t) == 4);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 4, ndir);
        return 1;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong(value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_IFD, count,
                                      count * 4, value));
}

static int TIFFWriteDirectoryTagCheckedIfd8Array(TIFF *tif, uint32_t *ndir,
                                                 TIFFDirEntry *dir,
                                                 uint16_t tag, uint32_t count,
                                                 uint64_t *value)
{
    assert(count < 0x20000000);
    assert(sizeof(uint64_t) == 8);
    assert(tif->tif_flags & TIFF_BIGTIFF);
    if (dir == NULL) /* Just evaluate IFD data size and increment ndir. */
    {
        EvaluateIFDdatasizeWrite(tif, count, 8, ndir);
        return 1;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong8(value, count);
    return (TIFFWriteDirectoryTagData(tif, ndir, dir, tag, TIFF_IFD8, count,
                                      count * 8, value));
}

static int TIFFWriteDirectoryTagData(TIFF *tif, uint32_t *ndir,
                                     TIFFDirEntry *dir, uint16_t tag,
                                     uint16_t datatype, uint32_t count,
                                     uint32_t datalength, void *data)
{
    static const char module[] = "TIFFWriteDirectoryTagData";
    uint32_t m;
    m = 0;
    while (m < (*ndir))
    {
        assert(dir[m].tdir_tag != tag);
        if (dir[m].tdir_tag > tag)
            break;
        m++;
    }
    if (m < (*ndir))
    {
        uint32_t n;
        for (n = *ndir; n > m; n--)
            dir[n] = dir[n - 1];
    }
    dir[m].tdir_tag = tag;
    dir[m].tdir_type = datatype;
    dir[m].tdir_count = count;
    dir[m].tdir_offset.toff_long8 = 0;
    if (datalength <= ((tif->tif_flags & TIFF_BIGTIFF) ? 0x8U : 0x4U))
    {
        if (data && datalength)
        {
            _TIFFmemcpy(&dir[m].tdir_offset, data, datalength);
        }
    }
    else
    {
        uint64_t na, nb;
        na = tif->tif_dataoff;
        nb = na + datalength;
        if (!(tif->tif_flags & TIFF_BIGTIFF))
            nb = (uint32_t)nb;
        if ((nb < na) || (nb < datalength))
        {
            TIFFErrorExtR(tif, module, "Maximum TIFF file size exceeded");
            return (0);
        }
        if (!SeekOK(tif, na))
        {
            TIFFErrorExtR(tif, module, "IO error writing tag data");
            return (0);
        }
        if (datalength >= 0x80000000UL)
        {
            TIFFErrorExtR(tif, module,
                          "libtiff does not allow writing more than 2147483647 "
                          "bytes in a tag");
            return (0);
        }
        if (!WriteOK(tif, data, (tmsize_t)datalength))
        {
            TIFFErrorExtR(tif, module, "IO error writing tag data");
            return (0);
        }
        tif->tif_dataoff = nb;
        if (tif->tif_dataoff & 1)
            tif->tif_dataoff++;
        if (!(tif->tif_flags & TIFF_BIGTIFF))
        {
            uint32_t o;
            o = (uint32_t)na;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong(&o);
            _TIFFmemcpy(&dir[m].tdir_offset, &o, 4);
        }
        else
        {
            dir[m].tdir_offset.toff_long8 = na;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&dir[m].tdir_offset.toff_long8);
        }
    }
    (*ndir)++;
    return (1);
}

/*
 * Link the current directory into the directory chain for the file.
 */
static int TIFFLinkDirectory(TIFF *tif)
{
    static const char module[] = "TIFFLinkDirectory";

    tif->tif_diroff = (TIFFSeekFile(tif, 0, SEEK_END) + 1) & (~((toff_t)1));

    /*
     * Handle SubIFDs
     */
    if (tif->tif_flags & TIFF_INSUBIFD)
    {
        if (!(tif->tif_flags & TIFF_BIGTIFF))
        {
            uint32_t m;
            m = (uint32_t)tif->tif_diroff;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong(&m);
            (void)TIFFSeekFile(tif, tif->tif_subifdoff, SEEK_SET);
            if (!WriteOK(tif, &m, 4))
            {
                TIFFErrorExtR(tif, module,
                              "Error writing SubIFD directory link");
                return (0);
            }

            /*
             * Advance to the next SubIFD or, if this is
             * the last one configured, reverting back to the
             * normal directory linkage is done in TIFFWriteDirectorySec()
             * by tif->tif_flags &= ~TIFF_INSUBIFD;.
             */
            if (--tif->tif_nsubifd)
                tif->tif_subifdoff += 4;
            return (1);
        }
        else
        {
            uint64_t m;
            m = tif->tif_diroff;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&m);
            (void)TIFFSeekFile(tif, tif->tif_subifdoff, SEEK_SET);
            if (!WriteOK(tif, &m, 8))
            {
                TIFFErrorExtR(tif, module,
                              "Error writing SubIFD directory link");
                return (0);
            }

            /*
             * Advance to the next SubIFD or, if this is
             * the last one configured, reverting back to the
             * normal directory linkage is done in TIFFWriteDirectorySec()
             * by tif->tif_flags &= ~TIFF_INSUBIFD;.
             */
            if (--tif->tif_nsubifd)
                tif->tif_subifdoff += 8;
            return (1);
        }
    }

    /*
     * Handle main-IFDs
     */
    tdir_t ndir = 1; /* count current number of main-IFDs */
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        uint32_t m;
        uint32_t nextdir;
        m = (uint32_t)(tif->tif_diroff);
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&m);
        if (tif->tif_header.classic.tiff_diroff == 0)
        {
            /*
             * First directory, overwrite offset in header.
             */
            tif->tif_header.classic.tiff_diroff = (uint32_t)tif->tif_diroff;
            tif->tif_lastdiroff = tif->tif_diroff;
            (void)TIFFSeekFile(tif, 4, SEEK_SET);
            if (!WriteOK(tif, &m, 4))
            {
                TIFFErrorExtR(tif, tif->tif_name, "Error writing TIFF header");
                return (0);
            }
            if (!tif->tif_dir.td_iswrittentofile)
                tif->tif_curdircount = 0;
            return (1);
        }
        /*
         * Not the first directory, search to the last and append.
         */
        tdir_t dirn = 0;
        if (tif->tif_lastdiroff != 0 &&
            _TIFFGetDirNumberFromOffset(tif, tif->tif_lastdiroff, &dirn))
        {
            /* Start searching from the lastely written IFD. Thus get its IFD
             * number. */
            nextdir = (uint32_t)tif->tif_lastdiroff;
            ndir = dirn + 1;
        }
        else
        {
            nextdir = tif->tif_header.classic.tiff_diroff;
            ndir = 1; /* start searching from the first IFD */
        }

        while (1)
        {
            uint16_t dircount;
            uint32_t nextnextdir;

            if (!SeekOK(tif, nextdir) || !ReadOK(tif, &dircount, 2))
            {
                TIFFErrorExtR(tif, module, "Error fetching directory count");
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort(&dircount);
            (void)TIFFSeekFile(tif, nextdir + 2 + dircount * 12, SEEK_SET);
            if (!ReadOK(tif, &nextnextdir, 4))
            {
                TIFFErrorExtR(tif, module, "Error fetching directory link");
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong(&nextnextdir);
            if (nextnextdir == 0)
            {
                (void)TIFFSeekFile(tif, nextdir + 2 + dircount * 12, SEEK_SET);
                if (!WriteOK(tif, &m, 4))
                {
                    TIFFErrorExtR(tif, module, "Error writing directory link");
                    return (0);
                }
                tif->tif_lastdiroff = tif->tif_diroff;
                break;
            }
            nextdir = nextnextdir;
            ndir++;
        }
    }
    else
    {
        /*- BigTIFF -*/
        uint64_t m;
        uint64_t nextdir;
        m = tif->tif_diroff;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8(&m);
        if (tif->tif_header.big.tiff_diroff == 0)
        {
            /*
             * First directory, overwrite offset in header.
             */
            tif->tif_header.big.tiff_diroff = tif->tif_diroff;
            tif->tif_lastdiroff = tif->tif_diroff;
            (void)TIFFSeekFile(tif, 8, SEEK_SET);
            if (!WriteOK(tif, &m, 8))
            {
                TIFFErrorExtR(tif, tif->tif_name, "Error writing TIFF header");
                return (0);
            }
            if (!tif->tif_dir.td_iswrittentofile)
                tif->tif_curdircount = 0;
            return (1);
        }
        /*
         * Not the first directory, search to the last and append.
         */
        tdir_t dirn = 0;
        if (tif->tif_lastdiroff != 0 &&
            _TIFFGetDirNumberFromOffset(tif, tif->tif_lastdiroff, &dirn))
        {
            /* Start searching from the lastely written IFD. Thus get its IFD
             * number. */
            nextdir = tif->tif_lastdiroff;
            ndir = dirn + 1;
        }
        else
        {
            nextdir = tif->tif_header.big.tiff_diroff;
            ndir = 1; /* start searching from the first IFD */
        }
        while (1)
        {
            uint64_t dircount64;
            uint16_t dircount;
            uint64_t nextnextdir;

            if (!SeekOK(tif, nextdir) || !ReadOK(tif, &dircount64, 8))
            {
                TIFFErrorExtR(tif, module, "Error fetching directory count");
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&dircount64);
            if (dircount64 > 0xFFFF)
            {
                TIFFErrorExtR(tif, module,
                              "Sanity check on tag count failed, "
                              "likely corrupt TIFF");
                return (0);
            }
            dircount = (uint16_t)dircount64;
            (void)TIFFSeekFile(tif, nextdir + 8 + dircount * 20, SEEK_SET);
            if (!ReadOK(tif, &nextnextdir, 8))
            {
                TIFFErrorExtR(tif, module, "Error fetching directory link");
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&nextnextdir);
            if (nextnextdir == 0)
            {
                (void)TIFFSeekFile(tif, nextdir + 8 + dircount * 20, SEEK_SET);
                if (!WriteOK(tif, &m, 8))
                {
                    TIFFErrorExtR(tif, module, "Error writing directory link");
                    return (0);
                }
                tif->tif_lastdiroff = tif->tif_diroff;
                break;
            }
            nextdir = nextnextdir;
            ndir++;
        }
    }
    /* Offset of next IFD is written to file.
     * Update number of main-IFDs in file.
     * However, tif_curdircount shall count only newly written main-IFDs with
     * entries and not only number of linked offsets! Thus, tif_curdircount is
     * incremented at the end of TIFFWriteDirectorySec().
     * TIFF_NON_EXISTENT_DIR_NUMBER means 'dont know number of IFDs'
     * 0 means 'empty file opened for writing, but no IFD written yet' */
    if (!tif->tif_dir.td_iswrittentofile && !(tif->tif_flags & TIFF_INSUBIFD))
    {
        tif->tif_curdircount = ndir;
    }
    return (1);
}

/************************************************************************/
/*                          TIFFRewriteField()                          */
/*                                                                      */
/*      Rewrite a field in the directory on disk without regard to      */
/*      updating the TIFF directory structure in memory.  Currently     */
/*      only supported for field that already exist in the on-disk      */
/*      directory.  Mainly used for updating stripoffset /              */
/*      stripbytecount values after the directory is already on         */
/*      disk.                                                           */
/*                                                                      */
/*      Returns zero on failure, and one on success.                    */
/************************************************************************/

int _TIFFRewriteField(TIFF *tif, uint16_t tag, TIFFDataType in_datatype,
                      tmsize_t count, void *data)
{
    static const char module[] = "TIFFResetField";
    /* const TIFFField* fip = NULL; */
    uint16_t dircount;
    tmsize_t dirsize;
    uint8_t direntry_raw[20];
    uint16_t entry_tag = 0;
    uint16_t entry_type = 0;
    uint64_t entry_count = 0;
    uint64_t entry_offset = 0;
    int value_in_entry = 0;
    uint64_t read_offset;
    uint8_t *buf_to_write = NULL;
    TIFFDataType datatype;

    /* -------------------------------------------------------------------- */
    /*      Find field definition.                                          */
    /* -------------------------------------------------------------------- */
    /*fip =*/TIFFFindField(tif, tag, TIFF_ANY);

    /* -------------------------------------------------------------------- */
    /*      Do some checking this is a straight forward case.               */
    /* -------------------------------------------------------------------- */
    if (isMapped(tif))
    {
        TIFFErrorExtR(tif, module,
                      "Memory mapped files not currently supported for "
                      "this operation.");
        return 0;
    }

    if (tif->tif_diroff == 0)
    {
        TIFFErrorExtR(
            tif, module,
            "Attempt to reset field on directory not already on disk.");
        return 0;
    }

    /* -------------------------------------------------------------------- */
    /*      Read the directory entry count.                                 */
    /* -------------------------------------------------------------------- */
    if (!SeekOK(tif, tif->tif_diroff))
    {
        TIFFErrorExtR(tif, module, "%s: Seek error accessing TIFF directory",
                      tif->tif_name);
        return 0;
    }

    read_offset = tif->tif_diroff;

    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        if (!ReadOK(tif, &dircount, sizeof(uint16_t)))
        {
            TIFFErrorExtR(tif, module, "%s: Can not read TIFF directory count",
                          tif->tif_name);
            return 0;
        }
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabShort(&dircount);
        dirsize = 12;
        read_offset += 2;
    }
    else
    {
        uint64_t dircount64;
        if (!ReadOK(tif, &dircount64, sizeof(uint64_t)))
        {
            TIFFErrorExtR(tif, module, "%s: Can not read TIFF directory count",
                          tif->tif_name);
            return 0;
        }
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8(&dircount64);
        dircount = (uint16_t)dircount64;
        dirsize = 20;
        read_offset += 8;
    }

    /* -------------------------------------------------------------------- */
    /*      Read through directory to find target tag.                      */
    /* -------------------------------------------------------------------- */
    while (dircount > 0)
    {
        if (!ReadOK(tif, direntry_raw, dirsize))
        {
            TIFFErrorExtR(tif, module, "%s: Can not read TIFF directory entry.",
                          tif->tif_name);
            return 0;
        }

        memcpy(&entry_tag, direntry_raw + 0, sizeof(uint16_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabShort(&entry_tag);

        if (entry_tag == tag)
            break;

        read_offset += dirsize;
    }

    if (entry_tag != tag)
    {
        TIFFErrorExtR(tif, module, "%s: Could not find tag %" PRIu16 ".",
                      tif->tif_name, tag);
        return 0;
    }

    /* -------------------------------------------------------------------- */
    /*      Extract the type, count and offset for this entry.              */
    /* -------------------------------------------------------------------- */
    memcpy(&entry_type, direntry_raw + 2, sizeof(uint16_t));
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabShort(&entry_type);

    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        uint32_t value;

        memcpy(&value, direntry_raw + 4, sizeof(uint32_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&value);
        entry_count = value;

        memcpy(&value, direntry_raw + 8, sizeof(uint32_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&value);
        entry_offset = value;
    }
    else
    {
        memcpy(&entry_count, direntry_raw + 4, sizeof(uint64_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8(&entry_count);

        memcpy(&entry_offset, direntry_raw + 12, sizeof(uint64_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8(&entry_offset);
    }

    /* -------------------------------------------------------------------- */
    /*      When a dummy tag was written due to TIFFDeferStrileArrayWriting() */
    /* -------------------------------------------------------------------- */
    if (entry_offset == 0 && entry_count == 0 && entry_type == 0)
    {
        if (tag == TIFFTAG_TILEOFFSETS || tag == TIFFTAG_STRIPOFFSETS)
        {
            entry_type =
                (tif->tif_flags & TIFF_BIGTIFF) ? TIFF_LONG8 : TIFF_LONG;
        }
        else
        {
            int write_aslong8 = 1;
            if (count > 1 && tag == TIFFTAG_STRIPBYTECOUNTS)
            {
                write_aslong8 = WriteAsLong8(tif, TIFFStripSize64(tif));
            }
            else if (count > 1 && tag == TIFFTAG_TILEBYTECOUNTS)
            {
                write_aslong8 = WriteAsLong8(tif, TIFFTileSize64(tif));
            }
            if (write_aslong8)
            {
                entry_type = TIFF_LONG8;
            }
            else
            {
                int write_aslong4 = 1;
                if (count > 1 && tag == TIFFTAG_STRIPBYTECOUNTS)
                {
                    write_aslong4 = WriteAsLong4(tif, TIFFStripSize64(tif));
                }
                else if (count > 1 && tag == TIFFTAG_TILEBYTECOUNTS)
                {
                    write_aslong4 = WriteAsLong4(tif, TIFFTileSize64(tif));
                }
                if (write_aslong4)
                {
                    entry_type = TIFF_LONG;
                }
                else
                {
                    entry_type = TIFF_SHORT;
                }
            }
        }
    }

    /* -------------------------------------------------------------------- */
    /*      What data type do we want to write this as?                     */
    /* -------------------------------------------------------------------- */
    if (TIFFDataWidth(in_datatype) == 8 && !(tif->tif_flags & TIFF_BIGTIFF))
    {
        if (in_datatype == TIFF_LONG8)
            datatype = entry_type == TIFF_SHORT ? TIFF_SHORT : TIFF_LONG;
        else if (in_datatype == TIFF_SLONG8)
            datatype = TIFF_SLONG;
        else if (in_datatype == TIFF_IFD8)
            datatype = TIFF_IFD;
        else
            datatype = in_datatype;
    }
    else
    {
        if (in_datatype == TIFF_LONG8 &&
            (entry_type == TIFF_SHORT || entry_type == TIFF_LONG ||
             entry_type == TIFF_LONG8))
            datatype = entry_type;
        else if (in_datatype == TIFF_SLONG8 &&
                 (entry_type == TIFF_SLONG || entry_type == TIFF_SLONG8))
            datatype = entry_type;
        else if (in_datatype == TIFF_IFD8 &&
                 (entry_type == TIFF_IFD || entry_type == TIFF_IFD8))
            datatype = entry_type;
        else
            datatype = in_datatype;
    }

    /* -------------------------------------------------------------------- */
    /*      Prepare buffer of actual data to write.  This includes          */
    /*      swabbing as needed.                                             */
    /* -------------------------------------------------------------------- */
    buf_to_write = (uint8_t *)_TIFFCheckMalloc(
        tif, count, TIFFDataWidth(datatype), "for field buffer.");
    if (!buf_to_write)
        return 0;

    if (datatype == in_datatype)
        memcpy(buf_to_write, data, count * TIFFDataWidth(datatype));
    else if (datatype == TIFF_SLONG && in_datatype == TIFF_SLONG8)
    {
        tmsize_t i;

        for (i = 0; i < count; i++)
        {
            ((int32_t *)buf_to_write)[i] = (int32_t)((int64_t *)data)[i];
            if ((int64_t)((int32_t *)buf_to_write)[i] != ((int64_t *)data)[i])
            {
                _TIFFfreeExt(tif, buf_to_write);
                TIFFErrorExtR(tif, module,
                              "Value exceeds 32bit range of output type.");
                return 0;
            }
        }
    }
    else if ((datatype == TIFF_LONG && in_datatype == TIFF_LONG8) ||
             (datatype == TIFF_IFD && in_datatype == TIFF_IFD8))
    {
        tmsize_t i;

        for (i = 0; i < count; i++)
        {
            ((uint32_t *)buf_to_write)[i] = (uint32_t)((uint64_t *)data)[i];
            if ((uint64_t)((uint32_t *)buf_to_write)[i] !=
                ((uint64_t *)data)[i])
            {
                _TIFFfreeExt(tif, buf_to_write);
                TIFFErrorExtR(tif, module,
                              "Value exceeds 32bit range of output type.");
                return 0;
            }
        }
    }
    else if (datatype == TIFF_SHORT && in_datatype == TIFF_LONG8)
    {
        tmsize_t i;

        for (i = 0; i < count; i++)
        {
            ((uint16_t *)buf_to_write)[i] = (uint16_t)((uint64_t *)data)[i];
            if ((uint64_t)((uint16_t *)buf_to_write)[i] !=
                ((uint64_t *)data)[i])
            {
                _TIFFfreeExt(tif, buf_to_write);
                TIFFErrorExtR(tif, module,
                              "Value exceeds 16bit range of output type.");
                return 0;
            }
        }
    }
    else
    {
        TIFFErrorExtR(tif, module, "Unhandled type conversion.");
        return 0;
    }

    if (TIFFDataWidth(datatype) > 1 && (tif->tif_flags & TIFF_SWAB))
    {
        if (TIFFDataWidth(datatype) == 2)
            TIFFSwabArrayOfShort((uint16_t *)buf_to_write, count);
        else if (TIFFDataWidth(datatype) == 4)
            TIFFSwabArrayOfLong((uint32_t *)buf_to_write, count);
        else if (TIFFDataWidth(datatype) == 8)
            TIFFSwabArrayOfLong8((uint64_t *)buf_to_write, count);
    }

    /* -------------------------------------------------------------------- */
    /*      Is this a value that fits into the directory entry?             */
    /* -------------------------------------------------------------------- */
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        if (TIFFDataWidth(datatype) * count <= 4)
        {
            entry_offset = read_offset + 8;
            value_in_entry = 1;
        }
    }
    else
    {
        if (TIFFDataWidth(datatype) * count <= 8)
        {
            entry_offset = read_offset + 12;
            value_in_entry = 1;
        }
    }

    if ((tag == TIFFTAG_TILEOFFSETS || tag == TIFFTAG_STRIPOFFSETS) &&
        tif->tif_dir.td_stripoffset_entry.tdir_count == 0 &&
        tif->tif_dir.td_stripoffset_entry.tdir_type == 0 &&
        tif->tif_dir.td_stripoffset_entry.tdir_offset.toff_long8 == 0)
    {
        tif->tif_dir.td_stripoffset_entry.tdir_type = datatype;
        tif->tif_dir.td_stripoffset_entry.tdir_count = count;
    }
    else if ((tag == TIFFTAG_TILEBYTECOUNTS ||
              tag == TIFFTAG_STRIPBYTECOUNTS) &&
             tif->tif_dir.td_stripbytecount_entry.tdir_count == 0 &&
             tif->tif_dir.td_stripbytecount_entry.tdir_type == 0 &&
             tif->tif_dir.td_stripbytecount_entry.tdir_offset.toff_long8 == 0)
    {
        tif->tif_dir.td_stripbytecount_entry.tdir_type = datatype;
        tif->tif_dir.td_stripbytecount_entry.tdir_count = count;
    }

    /* -------------------------------------------------------------------- */
    /*      If the tag type, and count match, then we just write it out     */
    /*      over the old values without altering the directory entry at     */
    /*      all.                                                            */
    /* -------------------------------------------------------------------- */
    if (entry_count == (uint64_t)count && entry_type == (uint16_t)datatype)
    {
        if (!SeekOK(tif, entry_offset))
        {
            _TIFFfreeExt(tif, buf_to_write);
            TIFFErrorExtR(tif, module,
                          "%s: Seek error accessing TIFF directory",
                          tif->tif_name);
            return 0;
        }
        if (!WriteOK(tif, buf_to_write, count * TIFFDataWidth(datatype)))
        {
            _TIFFfreeExt(tif, buf_to_write);
            TIFFErrorExtR(tif, module, "Error writing directory link");
            return (0);
        }

        _TIFFfreeExt(tif, buf_to_write);
        return 1;
    }

    /* -------------------------------------------------------------------- */
    /*      Otherwise, we write the new tag data at the end of the file.    */
    /* -------------------------------------------------------------------- */
    if (!value_in_entry)
    {
        entry_offset = TIFFSeekFile(tif, 0, SEEK_END);

        if (!WriteOK(tif, buf_to_write, count * TIFFDataWidth(datatype)))
        {
            _TIFFfreeExt(tif, buf_to_write);
            TIFFErrorExtR(tif, module, "Error writing directory link");
            return (0);
        }
    }
    else
    {
        if (count * TIFFDataWidth(datatype) == 4)
        {
            uint32_t value;
            memcpy(&value, buf_to_write, count * TIFFDataWidth(datatype));
            entry_offset = value;
        }
        else
        {
            memcpy(&entry_offset, buf_to_write,
                   count * TIFFDataWidth(datatype));
        }
    }

    _TIFFfreeExt(tif, buf_to_write);
    buf_to_write = 0;

    /* -------------------------------------------------------------------- */
    /*      Adjust the directory entry.                                     */
    /* -------------------------------------------------------------------- */
    entry_type = datatype;
    entry_count = (uint64_t)count;
    memcpy(direntry_raw + 2, &entry_type, sizeof(uint16_t));
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabShort((uint16_t *)(direntry_raw + 2));

    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        uint32_t value;

        value = (uint32_t)entry_count;
        memcpy(direntry_raw + 4, &value, sizeof(uint32_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong((uint32_t *)(direntry_raw + 4));

        value = (uint32_t)entry_offset;
        memcpy(direntry_raw + 8, &value, sizeof(uint32_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong((uint32_t *)(direntry_raw + 8));
    }
    else
    {
        memcpy(direntry_raw + 4, &entry_count, sizeof(uint64_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8((uint64_t *)(direntry_raw + 4));

        memcpy(direntry_raw + 12, &entry_offset, sizeof(uint64_t));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8((uint64_t *)(direntry_raw + 12));
    }

    /* -------------------------------------------------------------------- */
    /*      Write the directory entry out to disk.                          */
    /* -------------------------------------------------------------------- */
    if (!SeekOK(tif, read_offset))
    {
        TIFFErrorExtR(tif, module, "%s: Seek error accessing TIFF directory",
                      tif->tif_name);
        return 0;
    }

    if (!WriteOK(tif, direntry_raw, dirsize))
    {
        TIFFErrorExtR(tif, module, "%s: Can not write TIFF directory entry.",
                      tif->tif_name);
        return 0;
    }

    return 1;
}
