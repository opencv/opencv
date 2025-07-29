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
 * Directory Read Support Routines.
 */

/* Suggested pending improvements:
 * - add a field 'field_info' to the TIFFDirEntry structure, and set that with
 *   the pointer to the appropriate TIFFField structure early on in
 *   TIFFReadDirectory, so as to eliminate current possibly repetitive lookup.
 */

#include "tiffconf.h"
#include "tiffiop.h"
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define FAILED_FII ((uint32_t)-1)

#ifdef HAVE_IEEEFP
#define TIFFCvtIEEEFloatToNative(tif, n, fp)
#define TIFFCvtIEEEDoubleToNative(tif, n, dp)
#else
extern void TIFFCvtIEEEFloatToNative(TIFF *, uint32_t, float *);
extern void TIFFCvtIEEEDoubleToNative(TIFF *, uint32_t, double *);
#endif

enum TIFFReadDirEntryErr
{
    TIFFReadDirEntryErrOk = 0,
    TIFFReadDirEntryErrCount = 1,
    TIFFReadDirEntryErrType = 2,
    TIFFReadDirEntryErrIo = 3,
    TIFFReadDirEntryErrRange = 4,
    TIFFReadDirEntryErrPsdif = 5,
    TIFFReadDirEntryErrSizesan = 6,
    TIFFReadDirEntryErrAlloc = 7,
};

static enum TIFFReadDirEntryErr
TIFFReadDirEntryByte(TIFF *tif, TIFFDirEntry *direntry, uint8_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntrySbyte(TIFF *tif, TIFFDirEntry *direntry, int8_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryShort(TIFF *tif, TIFFDirEntry *direntry, uint16_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntrySshort(TIFF *tif, TIFFDirEntry *direntry, int16_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryLong(TIFF *tif, TIFFDirEntry *direntry, uint32_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntrySlong(TIFF *tif, TIFFDirEntry *direntry, int32_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryLong8(TIFF *tif, TIFFDirEntry *direntry, uint64_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntrySlong8(TIFF *tif, TIFFDirEntry *direntry, int64_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryFloat(TIFF *tif, TIFFDirEntry *direntry, float *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryDouble(TIFF *tif, TIFFDirEntry *direntry, double *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryIfd8(TIFF *tif, TIFFDirEntry *direntry, uint64_t *value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryArray(TIFF *tif, TIFFDirEntry *direntry, uint32_t *count,
                      uint32_t desttypesize, void **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryByteArray(TIFF *tif, TIFFDirEntry *direntry, uint8_t **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntrySbyteArray(TIFF *tif, TIFFDirEntry *direntry, int8_t **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryShortArray(TIFF *tif, TIFFDirEntry *direntry, uint16_t **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntrySshortArray(TIFF *tif, TIFFDirEntry *direntry, int16_t **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryLongArray(TIFF *tif, TIFFDirEntry *direntry, uint32_t **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntrySlongArray(TIFF *tif, TIFFDirEntry *direntry, int32_t **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryLong8Array(TIFF *tif, TIFFDirEntry *direntry, uint64_t **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntrySlong8Array(TIFF *tif, TIFFDirEntry *direntry, int64_t **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryFloatArray(TIFF *tif, TIFFDirEntry *direntry, float **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryDoubleArray(TIFF *tif, TIFFDirEntry *direntry, double **value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryIfd8Array(TIFF *tif, TIFFDirEntry *direntry, uint64_t **value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryPersampleShort(TIFF *tif, TIFFDirEntry *direntry,
                               uint16_t *value);

static void TIFFReadDirEntryCheckedByte(TIFF *tif, TIFFDirEntry *direntry,
                                        uint8_t *value);
static void TIFFReadDirEntryCheckedSbyte(TIFF *tif, TIFFDirEntry *direntry,
                                         int8_t *value);
static void TIFFReadDirEntryCheckedShort(TIFF *tif, TIFFDirEntry *direntry,
                                         uint16_t *value);
static void TIFFReadDirEntryCheckedSshort(TIFF *tif, TIFFDirEntry *direntry,
                                          int16_t *value);
static void TIFFReadDirEntryCheckedLong(TIFF *tif, TIFFDirEntry *direntry,
                                        uint32_t *value);
static void TIFFReadDirEntryCheckedSlong(TIFF *tif, TIFFDirEntry *direntry,
                                         int32_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedLong8(TIFF *tif, TIFFDirEntry *direntry,
                             uint64_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedSlong8(TIFF *tif, TIFFDirEntry *direntry,
                              int64_t *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedRational(TIFF *tif, TIFFDirEntry *direntry,
                                double *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedSrational(TIFF *tif, TIFFDirEntry *direntry,
                                 double *value);
static void TIFFReadDirEntryCheckedFloat(TIFF *tif, TIFFDirEntry *direntry,
                                         float *value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedDouble(TIFF *tif, TIFFDirEntry *direntry, double *value);
#if 0
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedRationalDirect(TIFF *tif, TIFFDirEntry *direntry,
                                      TIFFRational_t *value);
#endif
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteSbyte(int8_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteShort(uint16_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteSshort(int16_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteLong(uint32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteSlong(int32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteLong8(uint64_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteSlong8(int64_t value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteByte(uint8_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteShort(uint16_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteSshort(int16_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteLong(uint32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteSlong(int32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteLong8(uint64_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteSlong8(int64_t value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortSbyte(int8_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortSshort(int16_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortLong(uint32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortSlong(int32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortLong8(uint64_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortSlong8(int64_t value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortShort(uint16_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortLong(uint32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortSlong(int32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortLong8(uint64_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortSlong8(int64_t value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSbyte(int8_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSshort(int16_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSlong(int32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongLong8(uint64_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSlong8(int64_t value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongLong(uint32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongLong8(uint64_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongSlong8(int64_t value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Sbyte(int8_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Sshort(int16_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Slong(int32_t value);
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Slong8(int64_t value);

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlong8Long8(uint64_t value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryData(TIFF *tif, uint64_t offset,
                                                     tmsize_t size, void *dest);
static void TIFFReadDirEntryOutputErr(TIFF *tif, enum TIFFReadDirEntryErr err,
                                      const char *module, const char *tagname,
                                      int recover);

static void TIFFReadDirectoryCheckOrder(TIFF *tif, TIFFDirEntry *dir,
                                        uint16_t dircount);
static TIFFDirEntry *TIFFReadDirectoryFindEntry(TIFF *tif, TIFFDirEntry *dir,
                                                uint16_t dircount,
                                                uint16_t tagid);
static void TIFFReadDirectoryFindFieldInfo(TIFF *tif, uint16_t tagid,
                                           uint32_t *fii);

static int EstimateStripByteCounts(TIFF *tif, TIFFDirEntry *dir,
                                   uint16_t dircount);
static void MissingRequired(TIFF *, const char *);
static int CheckDirCount(TIFF *, TIFFDirEntry *, uint32_t);
static uint16_t TIFFFetchDirectory(TIFF *tif, uint64_t diroff,
                                   TIFFDirEntry **pdir, uint64_t *nextdiroff);
static int TIFFFetchNormalTag(TIFF *, TIFFDirEntry *, int recover);
static int TIFFFetchStripThing(TIFF *tif, TIFFDirEntry *dir, uint32_t nstrips,
                               uint64_t **lpp);
static int TIFFFetchSubjectDistance(TIFF *, TIFFDirEntry *);
static void ChopUpSingleUncompressedStrip(TIFF *);
static void TryChopUpUncompressedBigTiff(TIFF *);
static uint64_t TIFFReadUInt64(const uint8_t *value);
static int _TIFFGetMaxColorChannels(uint16_t photometric);

static int _TIFFFillStrilesInternal(TIFF *tif, int loadStripByteCount);

typedef union _UInt64Aligned_t
{
    double d;
    uint64_t l;
    uint32_t i[2];
    uint16_t s[4];
    uint8_t c[8];
} UInt64Aligned_t;

/*
  Unaligned safe copy of a uint64_t value from an octet array.
*/
static uint64_t TIFFReadUInt64(const uint8_t *value)
{
    UInt64Aligned_t result;

    result.c[0] = value[0];
    result.c[1] = value[1];
    result.c[2] = value[2];
    result.c[3] = value[3];
    result.c[4] = value[4];
    result.c[5] = value[5];
    result.c[6] = value[6];
    result.c[7] = value[7];

    return result.l;
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryByte(TIFF *tif, TIFFDirEntry *direntry, uint8_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_UNDEFINED: /* Support to read TIFF_UNDEFINED with
                                field_readcount==1 */
            TIFFReadDirEntryCheckedByte(tif, direntry, value);
            return (TIFFReadDirEntryErrOk);
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeByteSbyte(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeByteShort(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeByteSshort(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeByteLong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeByteSlong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeByteLong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeByteSlong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        default:
            return (TIFFReadDirEntryErrType);
    }
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntrySbyte(TIFF *tif, TIFFDirEntry *direntry, int8_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_UNDEFINED: /* Support to read TIFF_UNDEFINED with
                                field_readcount==1 */
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSbyteByte(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            TIFFReadDirEntryCheckedSbyte(tif, direntry, value);
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSbyteShort(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSbyteSshort(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSbyteLong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSbyteSlong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeSbyteLong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeSbyteSlong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int8_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        default:
            return (TIFFReadDirEntryErrType);
    }
} /*-- TIFFReadDirEntrySbyte() --*/

static enum TIFFReadDirEntryErr
TIFFReadDirEntryShort(TIFF *tif, TIFFDirEntry *direntry, uint16_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            *value = (uint16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeShortSbyte(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
            TIFFReadDirEntryCheckedShort(tif, direntry, value);
            return (TIFFReadDirEntryErrOk);
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeShortSshort(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeShortLong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeShortSlong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeShortLong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeShortSlong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        default:
            return (TIFFReadDirEntryErrType);
    }
} /*-- TIFFReadDirEntryShort() --*/

static enum TIFFReadDirEntryErr
TIFFReadDirEntrySshort(TIFF *tif, TIFFDirEntry *direntry, int16_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            *value = (int16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            *value = (int16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSshortShort(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
            TIFFReadDirEntryCheckedSshort(tif, direntry, value);
            return (TIFFReadDirEntryErrOk);
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSshortLong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSshortSlong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeSshortLong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeSshortSlong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int16_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        default:
            return (TIFFReadDirEntryErrType);
    }
} /*-- TIFFReadDirEntrySshort() --*/

static enum TIFFReadDirEntryErr
TIFFReadDirEntryLong(TIFF *tif, TIFFDirEntry *direntry, uint32_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            *value = (uint32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeLongSbyte(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            *value = (uint32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeLongSshort(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
            TIFFReadDirEntryCheckedLong(tif, direntry, value);
            return (TIFFReadDirEntryErrOk);
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeLongSlong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeLongLong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeLongSlong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        default:
            return (TIFFReadDirEntryErrType);
    }
} /*-- TIFFReadDirEntryLong() --*/

static enum TIFFReadDirEntryErr
TIFFReadDirEntrySlong(TIFF *tif, TIFFDirEntry *direntry, int32_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            *value = (int32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            *value = (int32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            *value = (int32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            *value = (int32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeSlongLong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
            TIFFReadDirEntryCheckedSlong(tif, direntry, value);
            return (TIFFReadDirEntryErrOk);
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeSlongLong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeSlongSlong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int32_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        default:
            return (TIFFReadDirEntryErrType);
    }
} /*-- TIFFReadDirEntrySlong() --*/

static enum TIFFReadDirEntryErr
TIFFReadDirEntryLong8(TIFF *tif, TIFFDirEntry *direntry, uint64_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            *value = (uint64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeLong8Sbyte(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            *value = (uint64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeLong8Sshort(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            *value = (uint64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            err = TIFFReadDirEntryCheckRangeLong8Slong(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, value);
            return (err);
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeLong8Slong8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (uint64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        default:
            return (TIFFReadDirEntryErrType);
    }
} /*-- TIFFReadDirEntryLong8() --*/

static enum TIFFReadDirEntryErr
TIFFReadDirEntrySlong8(TIFF *tif, TIFFDirEntry *direntry, int64_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            *value = (int64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            *value = (int64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            *value = (int64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            *value = (int64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            *value = (int64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            *value = (int64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            err = TIFFReadDirEntryCheckRangeSlong8Long8(m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (int64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, value);
            return (err);
        default:
            return (TIFFReadDirEntryErrType);
    }
} /*-- TIFFReadDirEntrySlong8() --*/

static enum TIFFReadDirEntryErr
TIFFReadDirEntryFloat(TIFF *tif, TIFFDirEntry *direntry, float *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
#if defined(__WIN32__) && (_MSC_VER < 1500)
            /*
             * XXX: MSVC 6.0 does not support conversion
             * of 64-bit integers into floating point
             * values.
             */
            *value = _TIFFUInt64ToFloat(m);
#else
            *value = (float)m;
#endif
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_RATIONAL:
        {
            double m;
            err = TIFFReadDirEntryCheckedRational(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SRATIONAL:
        {
            double m;
            err = TIFFReadDirEntryCheckedSrational(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_FLOAT:
            TIFFReadDirEntryCheckedFloat(tif, direntry, value);
            return (TIFFReadDirEntryErrOk);
        case TIFF_DOUBLE:
        {
            double m;
            err = TIFFReadDirEntryCheckedDouble(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            if ((m > FLT_MAX) || (m < -FLT_MAX))
                return (TIFFReadDirEntryErrRange);
            *value = (float)m;
            return (TIFFReadDirEntryErrOk);
        }
        default:
            return (TIFFReadDirEntryErrType);
    }
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryDouble(TIFF *tif, TIFFDirEntry *direntry, double *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t m;
            TIFFReadDirEntryCheckedByte(tif, direntry, &m);
            *value = (double)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
        {
            int8_t m;
            TIFFReadDirEntryCheckedSbyte(tif, direntry, &m);
            *value = (double)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SHORT:
        {
            uint16_t m;
            TIFFReadDirEntryCheckedShort(tif, direntry, &m);
            *value = (double)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
        {
            int16_t m;
            TIFFReadDirEntryCheckedSshort(tif, direntry, &m);
            *value = (double)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            *value = (double)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
        {
            int32_t m;
            TIFFReadDirEntryCheckedSlong(tif, direntry, &m);
            *value = (double)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        {
            uint64_t m;
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
#if defined(__WIN32__) && (_MSC_VER < 1500)
            /*
             * XXX: MSVC 6.0 does not support conversion
             * of 64-bit integers into floating point
             * values.
             */
            *value = _TIFFUInt64ToDouble(m);
#else
            *value = (double)m;
#endif
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
        {
            int64_t m;
            err = TIFFReadDirEntryCheckedSlong8(tif, direntry, &m);
            if (err != TIFFReadDirEntryErrOk)
                return (err);
            *value = (double)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_RATIONAL:
            err = TIFFReadDirEntryCheckedRational(tif, direntry, value);
            return (err);
        case TIFF_SRATIONAL:
            err = TIFFReadDirEntryCheckedSrational(tif, direntry, value);
            return (err);
        case TIFF_FLOAT:
        {
            float m;
            TIFFReadDirEntryCheckedFloat(tif, direntry, &m);
            *value = (double)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_DOUBLE:
            err = TIFFReadDirEntryCheckedDouble(tif, direntry, value);
            return (err);
        default:
            return (TIFFReadDirEntryErrType);
    }
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryIfd8(TIFF *tif, TIFFDirEntry *direntry, uint64_t *value)
{
    enum TIFFReadDirEntryErr err;
    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);
    switch (direntry->tdir_type)
    {
        case TIFF_LONG:
        case TIFF_IFD:
        {
            uint32_t m;
            TIFFReadDirEntryCheckedLong(tif, direntry, &m);
            *value = (uint64_t)m;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_LONG8:
        case TIFF_IFD8:
            err = TIFFReadDirEntryCheckedLong8(tif, direntry, value);
            return (err);
        default:
            return (TIFFReadDirEntryErrType);
    }
}

#define INITIAL_THRESHOLD (1024 * 1024)
#define THRESHOLD_MULTIPLIER 10
#define MAX_THRESHOLD                                                          \
    (THRESHOLD_MULTIPLIER * THRESHOLD_MULTIPLIER * THRESHOLD_MULTIPLIER *      \
     INITIAL_THRESHOLD)

static enum TIFFReadDirEntryErr TIFFReadDirEntryDataAndRealloc(TIFF *tif,
                                                               uint64_t offset,
                                                               tmsize_t size,
                                                               void **pdest)
{
#if SIZEOF_SIZE_T == 8
    tmsize_t threshold = INITIAL_THRESHOLD;
#endif
    tmsize_t already_read = 0;

    assert(!isMapped(tif));

    if (!SeekOK(tif, offset))
        return (TIFFReadDirEntryErrIo);

    /* On 64 bit processes, read first a maximum of 1 MB, then 10 MB, etc */
    /* so as to avoid allocating too much memory in case the file is too */
    /* short. We could ask for the file size, but this might be */
    /* expensive with some I/O layers (think of reading a gzipped file) */
    /* Restrict to 64 bit processes, so as to avoid reallocs() */
    /* on 32 bit processes where virtual memory is scarce.  */
    while (already_read < size)
    {
        void *new_dest;
        tmsize_t bytes_read;
        tmsize_t to_read = size - already_read;
#if SIZEOF_SIZE_T == 8
        if (to_read >= threshold && threshold < MAX_THRESHOLD)
        {
            to_read = threshold;
            threshold *= THRESHOLD_MULTIPLIER;
        }
#endif

        new_dest =
            (uint8_t *)_TIFFreallocExt(tif, *pdest, already_read + to_read);
        if (new_dest == NULL)
        {
            TIFFErrorExtR(tif, tif->tif_name,
                          "Failed to allocate memory for %s "
                          "(%" TIFF_SSIZE_FORMAT
                          " elements of %" TIFF_SSIZE_FORMAT " bytes each)",
                          "TIFFReadDirEntryArray", (tmsize_t)1,
                          already_read + to_read);
            return TIFFReadDirEntryErrAlloc;
        }
        *pdest = new_dest;

        bytes_read = TIFFReadFile(tif, (char *)*pdest + already_read, to_read);
        already_read += bytes_read;
        if (bytes_read != to_read)
        {
            return TIFFReadDirEntryErrIo;
        }
    }
    return TIFFReadDirEntryErrOk;
}

/* Caution: if raising that value, make sure int32 / uint32 overflows can't
 * occur elsewhere */
#define MAX_SIZE_TAG_DATA 2147483647U

static enum TIFFReadDirEntryErr
TIFFReadDirEntryArrayWithLimit(TIFF *tif, TIFFDirEntry *direntry,
                               uint32_t *count, uint32_t desttypesize,
                               void **value, uint64_t maxcount)
{
    int typesize;
    uint32_t datasize;
    void *data;
    uint64_t target_count64;
    int original_datasize_clamped;
    typesize = TIFFDataWidth(direntry->tdir_type);

    target_count64 =
        (direntry->tdir_count > maxcount) ? maxcount : direntry->tdir_count;

    if ((target_count64 == 0) || (typesize == 0))
    {
        *value = 0;
        return (TIFFReadDirEntryErrOk);
    }
    (void)desttypesize;

    /* We just want to know if the original tag size is more than 4 bytes
     * (classic TIFF) or 8 bytes (BigTIFF)
     */
    original_datasize_clamped =
        ((direntry->tdir_count > 10) ? 10 : (int)direntry->tdir_count) *
        typesize;

    /*
     * As a sanity check, make sure we have no more than a 2GB tag array
     * in either the current data type or the dest data type.  This also
     * avoids problems with overflow of tmsize_t on 32bit systems.
     */
    if ((uint64_t)(MAX_SIZE_TAG_DATA / typesize) < target_count64)
        return (TIFFReadDirEntryErrSizesan);
    if ((uint64_t)(MAX_SIZE_TAG_DATA / desttypesize) < target_count64)
        return (TIFFReadDirEntryErrSizesan);

    *count = (uint32_t)target_count64;
    datasize = (*count) * typesize;
    assert((tmsize_t)datasize > 0);

    if (isMapped(tif) && datasize > (uint64_t)tif->tif_size)
        return TIFFReadDirEntryErrIo;

    if (!isMapped(tif) && (((tif->tif_flags & TIFF_BIGTIFF) && datasize > 8) ||
                           (!(tif->tif_flags & TIFF_BIGTIFF) && datasize > 4)))
    {
        data = NULL;
    }
    else
    {
        data = _TIFFCheckMalloc(tif, *count, typesize, "ReadDirEntryArray");
        if (data == 0)
            return (TIFFReadDirEntryErrAlloc);
    }
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        /* Only the condition on original_datasize_clamped. The second
         * one is implied, but Coverity Scan cannot see it. */
        if (original_datasize_clamped <= 4 && datasize <= 4)
            _TIFFmemcpy(data, &direntry->tdir_offset, datasize);
        else
        {
            enum TIFFReadDirEntryErr err;
            uint32_t offset = direntry->tdir_offset.toff_long;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong(&offset);
            if (isMapped(tif))
                err = TIFFReadDirEntryData(tif, (uint64_t)offset,
                                           (tmsize_t)datasize, data);
            else
                err = TIFFReadDirEntryDataAndRealloc(tif, (uint64_t)offset,
                                                     (tmsize_t)datasize, &data);
            if (err != TIFFReadDirEntryErrOk)
            {
                _TIFFfreeExt(tif, data);
                return (err);
            }
        }
    }
    else
    {
        /* See above comment for the Classic TIFF case */
        if (original_datasize_clamped <= 8 && datasize <= 8)
            _TIFFmemcpy(data, &direntry->tdir_offset, datasize);
        else
        {
            enum TIFFReadDirEntryErr err;
            uint64_t offset = direntry->tdir_offset.toff_long8;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&offset);
            if (isMapped(tif))
                err = TIFFReadDirEntryData(tif, (uint64_t)offset,
                                           (tmsize_t)datasize, data);
            else
                err = TIFFReadDirEntryDataAndRealloc(tif, (uint64_t)offset,
                                                     (tmsize_t)datasize, &data);
            if (err != TIFFReadDirEntryErrOk)
            {
                _TIFFfreeExt(tif, data);
                return (err);
            }
        }
    }
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryArray(TIFF *tif, TIFFDirEntry *direntry, uint32_t *count,
                      uint32_t desttypesize, void **value)
{
    return TIFFReadDirEntryArrayWithLimit(tif, direntry, count, desttypesize,
                                          value, ~((uint64_t)0));
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryByteArray(TIFF *tif, TIFFDirEntry *direntry, uint8_t **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    uint8_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_ASCII:
        case TIFF_UNDEFINED:
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 1, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_ASCII:
        case TIFF_UNDEFINED:
        case TIFF_BYTE:
            *value = (uint8_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        case TIFF_SBYTE:
        {
            int8_t *m;
            uint32_t n;
            m = (int8_t *)origdata;
            for (n = 0; n < count; n++)
            {
                err = TIFFReadDirEntryCheckRangeByteSbyte(*m);
                if (err != TIFFReadDirEntryErrOk)
                {
                    _TIFFfreeExt(tif, origdata);
                    return (err);
                }
                m++;
            }
            *value = (uint8_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        }
    }
    data = (uint8_t *)_TIFFmallocExt(tif, count);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_SHORT:
        {
            uint16_t *ma;
            uint8_t *mb;
            uint32_t n;
            ma = (uint16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(ma);
                err = TIFFReadDirEntryCheckRangeByteShort(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint8_t)(*ma++);
            }
        }
        break;
        case TIFF_SSHORT:
        {
            int16_t *ma;
            uint8_t *mb;
            uint32_t n;
            ma = (int16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)ma);
                err = TIFFReadDirEntryCheckRangeByteSshort(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint8_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG:
        {
            uint32_t *ma;
            uint8_t *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                err = TIFFReadDirEntryCheckRangeByteLong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint8_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG:
        {
            int32_t *ma;
            uint8_t *mb;
            uint32_t n;
            ma = (int32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)ma);
                err = TIFFReadDirEntryCheckRangeByteSlong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint8_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG8:
        {
            uint64_t *ma;
            uint8_t *mb;
            uint32_t n;
            ma = (uint64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(ma);
                err = TIFFReadDirEntryCheckRangeByteLong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint8_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG8:
        {
            int64_t *ma;
            uint8_t *mb;
            uint32_t n;
            ma = (int64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)ma);
                err = TIFFReadDirEntryCheckRangeByteSlong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint8_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    if (err != TIFFReadDirEntryErrOk)
    {
        _TIFFfreeExt(tif, data);
        return (err);
    }
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntrySbyteArray(TIFF *tif, TIFFDirEntry *direntry, int8_t **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    int8_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_UNDEFINED:
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 1, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_UNDEFINED:
        case TIFF_BYTE:
        {
            uint8_t *m;
            uint32_t n;
            m = (uint8_t *)origdata;
            for (n = 0; n < count; n++)
            {
                err = TIFFReadDirEntryCheckRangeSbyteByte(*m);
                if (err != TIFFReadDirEntryErrOk)
                {
                    _TIFFfreeExt(tif, origdata);
                    return (err);
                }
                m++;
            }
            *value = (int8_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SBYTE:
            *value = (int8_t *)origdata;
            return (TIFFReadDirEntryErrOk);
    }
    data = (int8_t *)_TIFFmallocExt(tif, count);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_SHORT:
        {
            uint16_t *ma;
            int8_t *mb;
            uint32_t n;
            ma = (uint16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(ma);
                err = TIFFReadDirEntryCheckRangeSbyteShort(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int8_t)(*ma++);
            }
        }
        break;
        case TIFF_SSHORT:
        {
            int16_t *ma;
            int8_t *mb;
            uint32_t n;
            ma = (int16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)ma);
                err = TIFFReadDirEntryCheckRangeSbyteSshort(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int8_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG:
        {
            uint32_t *ma;
            int8_t *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                err = TIFFReadDirEntryCheckRangeSbyteLong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int8_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG:
        {
            int32_t *ma;
            int8_t *mb;
            uint32_t n;
            ma = (int32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)ma);
                err = TIFFReadDirEntryCheckRangeSbyteSlong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int8_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG8:
        {
            uint64_t *ma;
            int8_t *mb;
            uint32_t n;
            ma = (uint64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(ma);
                err = TIFFReadDirEntryCheckRangeSbyteLong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int8_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG8:
        {
            int64_t *ma;
            int8_t *mb;
            uint32_t n;
            ma = (int64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)ma);
                err = TIFFReadDirEntryCheckRangeSbyteSlong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int8_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    if (err != TIFFReadDirEntryErrOk)
    {
        _TIFFfreeExt(tif, data);
        return (err);
    }
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryShortArray(TIFF *tif, TIFFDirEntry *direntry, uint16_t **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    uint16_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 2, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_SHORT:
            *value = (uint16_t *)origdata;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfShort(*value, count);
            return (TIFFReadDirEntryErrOk);
        case TIFF_SSHORT:
        {
            int16_t *m;
            uint32_t n;
            m = (int16_t *)origdata;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)m);
                err = TIFFReadDirEntryCheckRangeShortSshort(*m);
                if (err != TIFFReadDirEntryErrOk)
                {
                    _TIFFfreeExt(tif, origdata);
                    return (err);
                }
                m++;
            }
            *value = (uint16_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        }
    }
    data = (uint16_t *)_TIFFmallocExt(tif, count * 2);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t *ma;
            uint16_t *mb;
            uint32_t n;
            ma = (uint8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (uint16_t)(*ma++);
        }
        break;
        case TIFF_SBYTE:
        {
            int8_t *ma;
            uint16_t *mb;
            uint32_t n;
            ma = (int8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                err = TIFFReadDirEntryCheckRangeShortSbyte(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint16_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG:
        {
            uint32_t *ma;
            uint16_t *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                err = TIFFReadDirEntryCheckRangeShortLong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint16_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG:
        {
            int32_t *ma;
            uint16_t *mb;
            uint32_t n;
            ma = (int32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)ma);
                err = TIFFReadDirEntryCheckRangeShortSlong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint16_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG8:
        {
            uint64_t *ma;
            uint16_t *mb;
            uint32_t n;
            ma = (uint64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(ma);
                err = TIFFReadDirEntryCheckRangeShortLong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint16_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG8:
        {
            int64_t *ma;
            uint16_t *mb;
            uint32_t n;
            ma = (int64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)ma);
                err = TIFFReadDirEntryCheckRangeShortSlong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint16_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    if (err != TIFFReadDirEntryErrOk)
    {
        _TIFFfreeExt(tif, data);
        return (err);
    }
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntrySshortArray(TIFF *tif, TIFFDirEntry *direntry, int16_t **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    int16_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 2, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_SHORT:
        {
            uint16_t *m;
            uint32_t n;
            m = (uint16_t *)origdata;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(m);
                err = TIFFReadDirEntryCheckRangeSshortShort(*m);
                if (err != TIFFReadDirEntryErrOk)
                {
                    _TIFFfreeExt(tif, origdata);
                    return (err);
                }
                m++;
            }
            *value = (int16_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SSHORT:
            *value = (int16_t *)origdata;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfShort((uint16_t *)(*value), count);
            return (TIFFReadDirEntryErrOk);
    }
    data = (int16_t *)_TIFFmallocExt(tif, count * 2);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t *ma;
            int16_t *mb;
            uint32_t n;
            ma = (uint8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (int16_t)(*ma++);
        }
        break;
        case TIFF_SBYTE:
        {
            int8_t *ma;
            int16_t *mb;
            uint32_t n;
            ma = (int8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (int16_t)(*ma++);
        }
        break;
        case TIFF_LONG:
        {
            uint32_t *ma;
            int16_t *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                err = TIFFReadDirEntryCheckRangeSshortLong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int16_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG:
        {
            int32_t *ma;
            int16_t *mb;
            uint32_t n;
            ma = (int32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)ma);
                err = TIFFReadDirEntryCheckRangeSshortSlong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int16_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG8:
        {
            uint64_t *ma;
            int16_t *mb;
            uint32_t n;
            ma = (uint64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(ma);
                err = TIFFReadDirEntryCheckRangeSshortLong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int16_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG8:
        {
            int64_t *ma;
            int16_t *mb;
            uint32_t n;
            ma = (int64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)ma);
                err = TIFFReadDirEntryCheckRangeSshortSlong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int16_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    if (err != TIFFReadDirEntryErrOk)
    {
        _TIFFfreeExt(tif, data);
        return (err);
    }
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryLongArray(TIFF *tif, TIFFDirEntry *direntry, uint32_t **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    uint32_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 4, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_LONG:
            *value = (uint32_t *)origdata;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong(*value, count);
            return (TIFFReadDirEntryErrOk);
        case TIFF_SLONG:
        {
            int32_t *m;
            uint32_t n;
            m = (int32_t *)origdata;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)m);
                err = TIFFReadDirEntryCheckRangeLongSlong(*m);
                if (err != TIFFReadDirEntryErrOk)
                {
                    _TIFFfreeExt(tif, origdata);
                    return (err);
                }
                m++;
            }
            *value = (uint32_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        }
    }
    data = (uint32_t *)_TIFFmallocExt(tif, count * 4);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t *ma;
            uint32_t *mb;
            uint32_t n;
            ma = (uint8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (uint32_t)(*ma++);
        }
        break;
        case TIFF_SBYTE:
        {
            int8_t *ma;
            uint32_t *mb;
            uint32_t n;
            ma = (int8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                err = TIFFReadDirEntryCheckRangeLongSbyte(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint32_t)(*ma++);
            }
        }
        break;
        case TIFF_SHORT:
        {
            uint16_t *ma;
            uint32_t *mb;
            uint32_t n;
            ma = (uint16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(ma);
                *mb++ = (uint32_t)(*ma++);
            }
        }
        break;
        case TIFF_SSHORT:
        {
            int16_t *ma;
            uint32_t *mb;
            uint32_t n;
            ma = (int16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)ma);
                err = TIFFReadDirEntryCheckRangeLongSshort(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint32_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG8:
        {
            uint64_t *ma;
            uint32_t *mb;
            uint32_t n;
            ma = (uint64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(ma);
                err = TIFFReadDirEntryCheckRangeLongLong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint32_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG8:
        {
            int64_t *ma;
            uint32_t *mb;
            uint32_t n;
            ma = (int64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)ma);
                err = TIFFReadDirEntryCheckRangeLongSlong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint32_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    if (err != TIFFReadDirEntryErrOk)
    {
        _TIFFfreeExt(tif, data);
        return (err);
    }
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntrySlongArray(TIFF *tif, TIFFDirEntry *direntry, int32_t **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    int32_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 4, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_LONG:
        {
            uint32_t *m;
            uint32_t n;
            m = (uint32_t *)origdata;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)m);
                err = TIFFReadDirEntryCheckRangeSlongLong(*m);
                if (err != TIFFReadDirEntryErrOk)
                {
                    _TIFFfreeExt(tif, origdata);
                    return (err);
                }
                m++;
            }
            *value = (int32_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG:
            *value = (int32_t *)origdata;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong((uint32_t *)(*value), count);
            return (TIFFReadDirEntryErrOk);
    }
    data = (int32_t *)_TIFFmallocExt(tif, count * 4);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t *ma;
            int32_t *mb;
            uint32_t n;
            ma = (uint8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (int32_t)(*ma++);
        }
        break;
        case TIFF_SBYTE:
        {
            int8_t *ma;
            int32_t *mb;
            uint32_t n;
            ma = (int8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (int32_t)(*ma++);
        }
        break;
        case TIFF_SHORT:
        {
            uint16_t *ma;
            int32_t *mb;
            uint32_t n;
            ma = (uint16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(ma);
                *mb++ = (int32_t)(*ma++);
            }
        }
        break;
        case TIFF_SSHORT:
        {
            int16_t *ma;
            int32_t *mb;
            uint32_t n;
            ma = (int16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)ma);
                *mb++ = (int32_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG8:
        {
            uint64_t *ma;
            int32_t *mb;
            uint32_t n;
            ma = (uint64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(ma);
                err = TIFFReadDirEntryCheckRangeSlongLong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int32_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG8:
        {
            int64_t *ma;
            int32_t *mb;
            uint32_t n;
            ma = (int64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)ma);
                err = TIFFReadDirEntryCheckRangeSlongSlong8(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (int32_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    if (err != TIFFReadDirEntryErrOk)
    {
        _TIFFfreeExt(tif, data);
        return (err);
    }
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryLong8ArrayWithLimit(TIFF *tif, TIFFDirEntry *direntry,
                                    uint64_t **value, uint64_t maxcount)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    uint64_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArrayWithLimit(tif, direntry, &count, 8, &origdata,
                                         maxcount);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_LONG8:
            *value = (uint64_t *)origdata;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong8(*value, count);
            return (TIFFReadDirEntryErrOk);
        case TIFF_SLONG8:
        {
            int64_t *m;
            uint32_t n;
            m = (int64_t *)origdata;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)m);
                err = TIFFReadDirEntryCheckRangeLong8Slong8(*m);
                if (err != TIFFReadDirEntryErrOk)
                {
                    _TIFFfreeExt(tif, origdata);
                    return (err);
                }
                m++;
            }
            *value = (uint64_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        }
    }
    data = (uint64_t *)_TIFFmallocExt(tif, count * 8);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t *ma;
            uint64_t *mb;
            uint32_t n;
            ma = (uint8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (uint64_t)(*ma++);
        }
        break;
        case TIFF_SBYTE:
        {
            int8_t *ma;
            uint64_t *mb;
            uint32_t n;
            ma = (int8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                err = TIFFReadDirEntryCheckRangeLong8Sbyte(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint64_t)(*ma++);
            }
        }
        break;
        case TIFF_SHORT:
        {
            uint16_t *ma;
            uint64_t *mb;
            uint32_t n;
            ma = (uint16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(ma);
                *mb++ = (uint64_t)(*ma++);
            }
        }
        break;
        case TIFF_SSHORT:
        {
            int16_t *ma;
            uint64_t *mb;
            uint32_t n;
            ma = (int16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)ma);
                err = TIFFReadDirEntryCheckRangeLong8Sshort(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint64_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG:
        {
            uint32_t *ma;
            uint64_t *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                *mb++ = (uint64_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG:
        {
            int32_t *ma;
            uint64_t *mb;
            uint32_t n;
            ma = (int32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)ma);
                err = TIFFReadDirEntryCheckRangeLong8Slong(*ma);
                if (err != TIFFReadDirEntryErrOk)
                    break;
                *mb++ = (uint64_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    if (err != TIFFReadDirEntryErrOk)
    {
        _TIFFfreeExt(tif, data);
        return (err);
    }
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryLong8Array(TIFF *tif, TIFFDirEntry *direntry, uint64_t **value)
{
    return TIFFReadDirEntryLong8ArrayWithLimit(tif, direntry, value,
                                               ~((uint64_t)0));
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntrySlong8Array(TIFF *tif, TIFFDirEntry *direntry, int64_t **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    int64_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 8, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_LONG8:
        {
            uint64_t *m;
            uint32_t n;
            m = (uint64_t *)origdata;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(m);
                err = TIFFReadDirEntryCheckRangeSlong8Long8(*m);
                if (err != TIFFReadDirEntryErrOk)
                {
                    _TIFFfreeExt(tif, origdata);
                    return (err);
                }
                m++;
            }
            *value = (int64_t *)origdata;
            return (TIFFReadDirEntryErrOk);
        }
        case TIFF_SLONG8:
            *value = (int64_t *)origdata;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong8((uint64_t *)(*value), count);
            return (TIFFReadDirEntryErrOk);
    }
    data = (int64_t *)_TIFFmallocExt(tif, count * 8);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t *ma;
            int64_t *mb;
            uint32_t n;
            ma = (uint8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (int64_t)(*ma++);
        }
        break;
        case TIFF_SBYTE:
        {
            int8_t *ma;
            int64_t *mb;
            uint32_t n;
            ma = (int8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (int64_t)(*ma++);
        }
        break;
        case TIFF_SHORT:
        {
            uint16_t *ma;
            int64_t *mb;
            uint32_t n;
            ma = (uint16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(ma);
                *mb++ = (int64_t)(*ma++);
            }
        }
        break;
        case TIFF_SSHORT:
        {
            int16_t *ma;
            int64_t *mb;
            uint32_t n;
            ma = (int16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)ma);
                *mb++ = (int64_t)(*ma++);
            }
        }
        break;
        case TIFF_LONG:
        {
            uint32_t *ma;
            int64_t *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                *mb++ = (int64_t)(*ma++);
            }
        }
        break;
        case TIFF_SLONG:
        {
            int32_t *ma;
            int64_t *mb;
            uint32_t n;
            ma = (int32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)ma);
                *mb++ = (int64_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryFloatArray(TIFF *tif, TIFFDirEntry *direntry, float **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    float *data;
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
        case TIFF_RATIONAL:
        case TIFF_SRATIONAL:
        case TIFF_FLOAT:
        case TIFF_DOUBLE:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 4, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_FLOAT:
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong((uint32_t *)origdata, count);
            TIFFCvtIEEEDoubleToNative(tif, count, (float *)origdata);
            *value = (float *)origdata;
            return (TIFFReadDirEntryErrOk);
    }
    data = (float *)_TIFFmallocExt(tif, count * sizeof(float));
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t *ma;
            float *mb;
            uint32_t n;
            ma = (uint8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (float)(*ma++);
        }
        break;
        case TIFF_SBYTE:
        {
            int8_t *ma;
            float *mb;
            uint32_t n;
            ma = (int8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (float)(*ma++);
        }
        break;
        case TIFF_SHORT:
        {
            uint16_t *ma;
            float *mb;
            uint32_t n;
            ma = (uint16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(ma);
                *mb++ = (float)(*ma++);
            }
        }
        break;
        case TIFF_SSHORT:
        {
            int16_t *ma;
            float *mb;
            uint32_t n;
            ma = (int16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)ma);
                *mb++ = (float)(*ma++);
            }
        }
        break;
        case TIFF_LONG:
        {
            uint32_t *ma;
            float *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                *mb++ = (float)(*ma++);
            }
        }
        break;
        case TIFF_SLONG:
        {
            int32_t *ma;
            float *mb;
            uint32_t n;
            ma = (int32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)ma);
                *mb++ = (float)(*ma++);
            }
        }
        break;
        case TIFF_LONG8:
        {
            uint64_t *ma;
            float *mb;
            uint32_t n;
            ma = (uint64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(ma);
#if defined(__WIN32__) && (_MSC_VER < 1500)
                /*
                 * XXX: MSVC 6.0 does not support
                 * conversion of 64-bit integers into
                 * floating point values.
                 */
                *mb++ = _TIFFUInt64ToFloat(*ma++);
#else
                *mb++ = (float)(*ma++);
#endif
            }
        }
        break;
        case TIFF_SLONG8:
        {
            int64_t *ma;
            float *mb;
            uint32_t n;
            ma = (int64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)ma);
                *mb++ = (float)(*ma++);
            }
        }
        break;
        case TIFF_RATIONAL:
        {
            uint32_t *ma;
            uint32_t maa;
            uint32_t mab;
            float *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                maa = *ma++;
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                mab = *ma++;
                if (mab == 0)
                    *mb++ = 0.0;
                else
                    *mb++ = (float)maa / (float)mab;
            }
        }
        break;
        case TIFF_SRATIONAL:
        {
            uint32_t *ma;
            int32_t maa;
            uint32_t mab;
            float *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                maa = *(int32_t *)ma;
                ma++;
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                mab = *ma++;
                if (mab == 0)
                    *mb++ = 0.0;
                else
                    *mb++ = (float)maa / (float)mab;
            }
        }
        break;
        case TIFF_DOUBLE:
        {
            double *ma;
            float *mb;
            uint32_t n;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong8((uint64_t *)origdata, count);
            TIFFCvtIEEEDoubleToNative(tif, count, (double *)origdata);
            ma = (double *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                double val = *ma++;
                if (val > FLT_MAX)
                    val = FLT_MAX;
                else if (val < -FLT_MAX)
                    val = -FLT_MAX;
                *mb++ = (float)val;
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryDoubleArray(TIFF *tif, TIFFDirEntry *direntry, double **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    double *data;
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        case TIFF_SBYTE:
        case TIFF_SHORT:
        case TIFF_SSHORT:
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_LONG8:
        case TIFF_SLONG8:
        case TIFF_RATIONAL:
        case TIFF_SRATIONAL:
        case TIFF_FLOAT:
        case TIFF_DOUBLE:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 8, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_DOUBLE:
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong8((uint64_t *)origdata, count);
            TIFFCvtIEEEDoubleToNative(tif, count, (double *)origdata);
            *value = (double *)origdata;
            return (TIFFReadDirEntryErrOk);
    }
    data = (double *)_TIFFmallocExt(tif, count * sizeof(double));
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_BYTE:
        {
            uint8_t *ma;
            double *mb;
            uint32_t n;
            ma = (uint8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (double)(*ma++);
        }
        break;
        case TIFF_SBYTE:
        {
            int8_t *ma;
            double *mb;
            uint32_t n;
            ma = (int8_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (double)(*ma++);
        }
        break;
        case TIFF_SHORT:
        {
            uint16_t *ma;
            double *mb;
            uint32_t n;
            ma = (uint16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort(ma);
                *mb++ = (double)(*ma++);
            }
        }
        break;
        case TIFF_SSHORT:
        {
            int16_t *ma;
            double *mb;
            uint32_t n;
            ma = (int16_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabShort((uint16_t *)ma);
                *mb++ = (double)(*ma++);
            }
        }
        break;
        case TIFF_LONG:
        {
            uint32_t *ma;
            double *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                *mb++ = (double)(*ma++);
            }
        }
        break;
        case TIFF_SLONG:
        {
            int32_t *ma;
            double *mb;
            uint32_t n;
            ma = (int32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong((uint32_t *)ma);
                *mb++ = (double)(*ma++);
            }
        }
        break;
        case TIFF_LONG8:
        {
            uint64_t *ma;
            double *mb;
            uint32_t n;
            ma = (uint64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(ma);
#if defined(__WIN32__) && (_MSC_VER < 1500)
                /*
                 * XXX: MSVC 6.0 does not support
                 * conversion of 64-bit integers into
                 * floating point values.
                 */
                *mb++ = _TIFFUInt64ToDouble(*ma++);
#else
                *mb++ = (double)(*ma++);
#endif
            }
        }
        break;
        case TIFF_SLONG8:
        {
            int64_t *ma;
            double *mb;
            uint32_t n;
            ma = (int64_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8((uint64_t *)ma);
                *mb++ = (double)(*ma++);
            }
        }
        break;
        case TIFF_RATIONAL:
        {
            uint32_t *ma;
            uint32_t maa;
            uint32_t mab;
            double *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                maa = *ma++;
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                mab = *ma++;
                if (mab == 0)
                    *mb++ = 0.0;
                else
                    *mb++ = (double)maa / (double)mab;
            }
        }
        break;
        case TIFF_SRATIONAL:
        {
            uint32_t *ma;
            int32_t maa;
            uint32_t mab;
            double *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                maa = *(int32_t *)ma;
                ma++;
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                mab = *ma++;
                if (mab == 0)
                    *mb++ = 0.0;
                else
                    *mb++ = (double)maa / (double)mab;
            }
        }
        break;
        case TIFF_FLOAT:
        {
            float *ma;
            double *mb;
            uint32_t n;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong((uint32_t *)origdata, count);
            TIFFCvtIEEEFloatToNative(tif, count, (float *)origdata);
            ma = (float *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
                *mb++ = (double)(*ma++);
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryIfd8Array(TIFF *tif, TIFFDirEntry *direntry, uint64_t **value)
{
    enum TIFFReadDirEntryErr err;
    uint32_t count;
    void *origdata;
    uint64_t *data;
    switch (direntry->tdir_type)
    {
        case TIFF_LONG:
        case TIFF_LONG8:
        case TIFF_IFD:
        case TIFF_IFD8:
            break;
        default:
            return (TIFFReadDirEntryErrType);
    }
    err = TIFFReadDirEntryArray(tif, direntry, &count, 8, &origdata);
    if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
    {
        *value = 0;
        return (err);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_LONG8:
        case TIFF_IFD8:
            *value = (uint64_t *)origdata;
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabArrayOfLong8(*value, count);
            return (TIFFReadDirEntryErrOk);
    }
    data = (uint64_t *)_TIFFmallocExt(tif, count * 8);
    if (data == 0)
    {
        _TIFFfreeExt(tif, origdata);
        return (TIFFReadDirEntryErrAlloc);
    }
    switch (direntry->tdir_type)
    {
        case TIFF_LONG:
        case TIFF_IFD:
        {
            uint32_t *ma;
            uint64_t *mb;
            uint32_t n;
            ma = (uint32_t *)origdata;
            mb = data;
            for (n = 0; n < count; n++)
            {
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(ma);
                *mb++ = (uint64_t)(*ma++);
            }
        }
        break;
    }
    _TIFFfreeExt(tif, origdata);
    *value = data;
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryPersampleShort(TIFF *tif, TIFFDirEntry *direntry,
                               uint16_t *value)
{
    enum TIFFReadDirEntryErr err;
    uint16_t *m;
    uint16_t *na;
    uint16_t nb;
    if (direntry->tdir_count < (uint64_t)tif->tif_dir.td_samplesperpixel)
        return (TIFFReadDirEntryErrCount);
    err = TIFFReadDirEntryShortArray(tif, direntry, &m);
    if (err != TIFFReadDirEntryErrOk || m == NULL)
        return (err);
    na = m;
    nb = tif->tif_dir.td_samplesperpixel;
    *value = *na++;
    nb--;
    while (nb > 0)
    {
        if (*na++ != *value)
        {
            err = TIFFReadDirEntryErrPsdif;
            break;
        }
        nb--;
    }
    _TIFFfreeExt(tif, m);
    return (err);
}

static void TIFFReadDirEntryCheckedByte(TIFF *tif, TIFFDirEntry *direntry,
                                        uint8_t *value)
{
    (void)tif;
    *value = *(uint8_t *)(&direntry->tdir_offset);
}

static void TIFFReadDirEntryCheckedSbyte(TIFF *tif, TIFFDirEntry *direntry,
                                         int8_t *value)
{
    (void)tif;
    *value = *(int8_t *)(&direntry->tdir_offset);
}

static void TIFFReadDirEntryCheckedShort(TIFF *tif, TIFFDirEntry *direntry,
                                         uint16_t *value)
{
    *value = direntry->tdir_offset.toff_short;
    /* *value=*(uint16_t*)(&direntry->tdir_offset); */
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabShort(value);
}

static void TIFFReadDirEntryCheckedSshort(TIFF *tif, TIFFDirEntry *direntry,
                                          int16_t *value)
{
    *value = *(int16_t *)(&direntry->tdir_offset);
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabShort((uint16_t *)value);
}

static void TIFFReadDirEntryCheckedLong(TIFF *tif, TIFFDirEntry *direntry,
                                        uint32_t *value)
{
    *value = *(uint32_t *)(&direntry->tdir_offset);
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabLong(value);
}

static void TIFFReadDirEntryCheckedSlong(TIFF *tif, TIFFDirEntry *direntry,
                                         int32_t *value)
{
    *value = *(int32_t *)(&direntry->tdir_offset);
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabLong((uint32_t *)value);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedLong8(TIFF *tif, TIFFDirEntry *direntry, uint64_t *value)
{
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        enum TIFFReadDirEntryErr err;
        uint32_t offset = direntry->tdir_offset.toff_long;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&offset);
        err = TIFFReadDirEntryData(tif, offset, 8, value);
        if (err != TIFFReadDirEntryErrOk)
            return (err);
    }
    else
        *value = direntry->tdir_offset.toff_long8;
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabLong8(value);
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedSlong8(TIFF *tif, TIFFDirEntry *direntry, int64_t *value)
{
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        enum TIFFReadDirEntryErr err;
        uint32_t offset = direntry->tdir_offset.toff_long;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&offset);
        err = TIFFReadDirEntryData(tif, offset, 8, value);
        if (err != TIFFReadDirEntryErrOk)
            return (err);
    }
    else
        *value = *(int64_t *)(&direntry->tdir_offset);
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabLong8((uint64_t *)value);
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedRational(TIFF *tif, TIFFDirEntry *direntry,
                                double *value)
{
    UInt64Aligned_t m;

    assert(sizeof(double) == 8);
    assert(sizeof(uint64_t) == 8);
    assert(sizeof(uint32_t) == 4);
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        enum TIFFReadDirEntryErr err;
        uint32_t offset = direntry->tdir_offset.toff_long;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&offset);
        err = TIFFReadDirEntryData(tif, offset, 8, m.i);
        if (err != TIFFReadDirEntryErrOk)
            return (err);
    }
    else
        m.l = direntry->tdir_offset.toff_long8;
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong(m.i, 2);
    /* Not completely sure what we should do when m.i[1]==0, but some */
    /* sanitizers do not like division by 0.0: */
    /* http://bugzilla.maptools.org/show_bug.cgi?id=2644 */
    if (m.i[0] == 0 || m.i[1] == 0)
        *value = 0.0;
    else
        *value = (double)m.i[0] / (double)m.i[1];
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedSrational(TIFF *tif, TIFFDirEntry *direntry,
                                 double *value)
{
    UInt64Aligned_t m;
    assert(sizeof(double) == 8);
    assert(sizeof(uint64_t) == 8);
    assert(sizeof(int32_t) == 4);
    assert(sizeof(uint32_t) == 4);
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        enum TIFFReadDirEntryErr err;
        uint32_t offset = direntry->tdir_offset.toff_long;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&offset);
        err = TIFFReadDirEntryData(tif, offset, 8, m.i);
        if (err != TIFFReadDirEntryErrOk)
            return (err);
    }
    else
        m.l = direntry->tdir_offset.toff_long8;
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong(m.i, 2);
    /* Not completely sure what we should do when m.i[1]==0, but some */
    /* sanitizers do not like division by 0.0: */
    /* http://bugzilla.maptools.org/show_bug.cgi?id=2644 */
    if ((int32_t)m.i[0] == 0 || m.i[1] == 0)
        *value = 0.0;
    else
        *value = (double)((int32_t)m.i[0]) / (double)m.i[1];
    return (TIFFReadDirEntryErrOk);
}

#if 0
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedRationalDirect(TIFF *tif, TIFFDirEntry *direntry,
                                      TIFFRational_t *value)
{ /*--: SetGetRATIONAL_directly:_CustomTag: Read rational (and signed rationals)
     directly --*/
    UInt64Aligned_t m;

    assert(sizeof(double) == 8);
    assert(sizeof(uint64_t) == 8);
    assert(sizeof(uint32_t) == 4);

    if (direntry->tdir_count != 1)
        return (TIFFReadDirEntryErrCount);

    if (direntry->tdir_type != TIFF_RATIONAL &&
        direntry->tdir_type != TIFF_SRATIONAL)
        return (TIFFReadDirEntryErrType);

    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        enum TIFFReadDirEntryErr err;
        uint32_t offset = direntry->tdir_offset.toff_long;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&offset);
        err = TIFFReadDirEntryData(tif, offset, 8, m.i);
        if (err != TIFFReadDirEntryErrOk)
            return (err);
    }
    else
    {
        m.l = direntry->tdir_offset.toff_long8;
    }

    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabArrayOfLong(m.i, 2);

    value->uNum = m.i[0];
    value->uDenom = m.i[1];
    return (TIFFReadDirEntryErrOk);
} /*-- TIFFReadDirEntryCheckedRationalDirect() --*/
#endif

static void TIFFReadDirEntryCheckedFloat(TIFF *tif, TIFFDirEntry *direntry,
                                         float *value)
{
    union
    {
        float f;
        uint32_t i;
    } float_union;
    assert(sizeof(float) == 4);
    assert(sizeof(uint32_t) == 4);
    assert(sizeof(float_union) == 4);
    float_union.i = *(uint32_t *)(&direntry->tdir_offset);
    *value = float_union.f;
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabLong((uint32_t *)value);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckedDouble(TIFF *tif, TIFFDirEntry *direntry, double *value)
{
    assert(sizeof(double) == 8);
    assert(sizeof(uint64_t) == 8);
    assert(sizeof(UInt64Aligned_t) == 8);
    if (!(tif->tif_flags & TIFF_BIGTIFF))
    {
        enum TIFFReadDirEntryErr err;
        uint32_t offset = direntry->tdir_offset.toff_long;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&offset);
        err = TIFFReadDirEntryData(tif, offset, 8, value);
        if (err != TIFFReadDirEntryErrOk)
            return (err);
    }
    else
    {
        UInt64Aligned_t uint64_union;
        uint64_union.l = direntry->tdir_offset.toff_long8;
        *value = uint64_union.d;
    }
    if (tif->tif_flags & TIFF_SWAB)
        TIFFSwabLong8((uint64_t *)value);
    return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteSbyte(int8_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteShort(uint16_t value)
{
    if (value > 0xFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteSshort(int16_t value)
{
    if ((value < 0) || (value > 0xFF))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteLong(uint32_t value)
{
    if (value > 0xFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteSlong(int32_t value)
{
    if ((value < 0) || (value > 0xFF))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteLong8(uint64_t value)
{
    if (value > 0xFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeByteSlong8(int64_t value)
{
    if ((value < 0) || (value > 0xFF))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteByte(uint8_t value)
{
    if (value > 0x7F)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteShort(uint16_t value)
{
    if (value > 0x7F)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteSshort(int16_t value)
{
    if ((value < -0x80) || (value > 0x7F))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteLong(uint32_t value)
{
    if (value > 0x7F)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteSlong(int32_t value)
{
    if ((value < -0x80) || (value > 0x7F))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteLong8(uint64_t value)
{
    if (value > 0x7F)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSbyteSlong8(int64_t value)
{
    if ((value < -0x80) || (value > 0x7F))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortSbyte(int8_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortSshort(int16_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortLong(uint32_t value)
{
    if (value > 0xFFFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortSlong(int32_t value)
{
    if ((value < 0) || (value > 0xFFFF))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortLong8(uint64_t value)
{
    if (value > 0xFFFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeShortSlong8(int64_t value)
{
    if ((value < 0) || (value > 0xFFFF))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortShort(uint16_t value)
{
    if (value > 0x7FFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortLong(uint32_t value)
{
    if (value > 0x7FFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortSlong(int32_t value)
{
    if ((value < -0x8000) || (value > 0x7FFF))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortLong8(uint64_t value)
{
    if (value > 0x7FFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSshortSlong8(int64_t value)
{
    if ((value < -0x8000) || (value > 0x7FFF))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSbyte(int8_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSshort(int16_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSlong(int32_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongLong8(uint64_t value)
{
    if (value > UINT32_MAX)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSlong8(int64_t value)
{
    if ((value < 0) || (value > (int64_t)UINT32_MAX))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongLong(uint32_t value)
{
    if (value > 0x7FFFFFFFUL)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

/* Check that the 8-byte unsigned value can fit in a 4-byte unsigned range */
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongLong8(uint64_t value)
{
    if (value > 0x7FFFFFFF)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

/* Check that the 8-byte signed value can fit in a 4-byte signed range */
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongSlong8(int64_t value)
{
    if ((value < 0 - ((int64_t)0x7FFFFFFF + 1)) || (value > 0x7FFFFFFF))
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Sbyte(int8_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Sshort(int16_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Slong(int32_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Slong8(int64_t value)
{
    if (value < 0)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlong8Long8(uint64_t value)
{
    if (value > INT64_MAX)
        return (TIFFReadDirEntryErrRange);
    else
        return (TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryData(TIFF *tif, uint64_t offset,
                                                     tmsize_t size, void *dest)
{
    assert(size > 0);
    if (!isMapped(tif))
    {
        if (!SeekOK(tif, offset))
            return (TIFFReadDirEntryErrIo);
        if (!ReadOK(tif, dest, size))
            return (TIFFReadDirEntryErrIo);
    }
    else
    {
        size_t ma, mb;
        ma = (size_t)offset;
        if ((uint64_t)ma != offset || ma > (~(size_t)0) - (size_t)size)
        {
            return TIFFReadDirEntryErrIo;
        }
        mb = ma + size;
        if (mb > (uint64_t)tif->tif_size)
            return (TIFFReadDirEntryErrIo);
        _TIFFmemcpy(dest, tif->tif_base + ma, size);
    }
    return (TIFFReadDirEntryErrOk);
}

static void TIFFReadDirEntryOutputErr(TIFF *tif, enum TIFFReadDirEntryErr err,
                                      const char *module, const char *tagname,
                                      int recover)
{
    if (!recover)
    {
        switch (err)
        {
            case TIFFReadDirEntryErrCount:
                TIFFErrorExtR(tif, module, "Incorrect count for \"%s\"",
                              tagname);
                break;
            case TIFFReadDirEntryErrType:
                TIFFErrorExtR(tif, module, "Incompatible type for \"%s\"",
                              tagname);
                break;
            case TIFFReadDirEntryErrIo:
                TIFFErrorExtR(tif, module, "IO error during reading of \"%s\"",
                              tagname);
                break;
            case TIFFReadDirEntryErrRange:
                TIFFErrorExtR(tif, module, "Incorrect value for \"%s\"",
                              tagname);
                break;
            case TIFFReadDirEntryErrPsdif:
                TIFFErrorExtR(
                    tif, module,
                    "Cannot handle different values per sample for \"%s\"",
                    tagname);
                break;
            case TIFFReadDirEntryErrSizesan:
                TIFFErrorExtR(tif, module,
                              "Sanity check on size of \"%s\" value failed",
                              tagname);
                break;
            case TIFFReadDirEntryErrAlloc:
                TIFFErrorExtR(tif, module, "Out of memory reading of \"%s\"",
                              tagname);
                break;
            default:
                assert(0); /* we should never get here */
                break;
        }
    }
    else
    {
        switch (err)
        {
            case TIFFReadDirEntryErrCount:
                TIFFWarningExtR(tif, module,
                                "Incorrect count for \"%s\"; tag ignored",
                                tagname);
                break;
            case TIFFReadDirEntryErrType:
                TIFFWarningExtR(tif, module,
                                "Incompatible type for \"%s\"; tag ignored",
                                tagname);
                break;
            case TIFFReadDirEntryErrIo:
                TIFFWarningExtR(
                    tif, module,
                    "IO error during reading of \"%s\"; tag ignored", tagname);
                break;
            case TIFFReadDirEntryErrRange:
                TIFFWarningExtR(tif, module,
                                "Incorrect value for \"%s\"; tag ignored",
                                tagname);
                break;
            case TIFFReadDirEntryErrPsdif:
                TIFFWarningExtR(tif, module,
                                "Cannot handle different values per sample for "
                                "\"%s\"; tag ignored",
                                tagname);
                break;
            case TIFFReadDirEntryErrSizesan:
                TIFFWarningExtR(
                    tif, module,
                    "Sanity check on size of \"%s\" value failed; tag ignored",
                    tagname);
                break;
            case TIFFReadDirEntryErrAlloc:
                TIFFWarningExtR(tif, module,
                                "Out of memory reading of \"%s\"; tag ignored",
                                tagname);
                break;
            default:
                assert(0); /* we should never get here */
                break;
        }
    }
}

/*
 * Return the maximum number of color channels specified for a given photometric
 * type. 0 is returned if photometric type isn't supported or no default value
 * is defined by the specification.
 */
static int _TIFFGetMaxColorChannels(uint16_t photometric)
{
    switch (photometric)
    {
        case PHOTOMETRIC_PALETTE:
        case PHOTOMETRIC_MINISWHITE:
        case PHOTOMETRIC_MINISBLACK:
            return 1;
        case PHOTOMETRIC_YCBCR:
        case PHOTOMETRIC_RGB:
        case PHOTOMETRIC_CIELAB:
        case PHOTOMETRIC_LOGLUV:
        case PHOTOMETRIC_ITULAB:
        case PHOTOMETRIC_ICCLAB:
            return 3;
        case PHOTOMETRIC_SEPARATED:
        case PHOTOMETRIC_MASK:
            return 4;
        case PHOTOMETRIC_LOGL:
        case PHOTOMETRIC_CFA:
        default:
            return 0;
    }
}

static int ByteCountLooksBad(TIFF *tif)
{
    /*
     * Assume we have wrong StripByteCount value (in case
     * of single strip) in following cases:
     *   - it is equal to zero along with StripOffset;
     *   - it is larger than file itself (in case of uncompressed
     *     image);
     *   - it is smaller than the size of the bytes per row
     *     multiplied on the number of rows.  The last case should
     *     not be checked in the case of writing new image,
     *     because we may do not know the exact strip size
     *     until the whole image will be written and directory
     *     dumped out.
     */
    uint64_t bytecount = TIFFGetStrileByteCount(tif, 0);
    uint64_t offset = TIFFGetStrileOffset(tif, 0);
    uint64_t filesize;

    if (offset == 0)
        return 0;
    if (bytecount == 0)
        return 1;
    if (tif->tif_dir.td_compression != COMPRESSION_NONE)
        return 0;
    filesize = TIFFGetFileSize(tif);
    if (offset <= filesize && bytecount > filesize - offset)
        return 1;
    if (tif->tif_mode == O_RDONLY)
    {
        uint64_t scanlinesize = TIFFScanlineSize64(tif);
        if (tif->tif_dir.td_imagelength > 0 &&
            scanlinesize > UINT64_MAX / tif->tif_dir.td_imagelength)
        {
            return 1;
        }
        if (bytecount < scanlinesize * tif->tif_dir.td_imagelength)
            return 1;
    }
    return 0;
}

/*
 * Read the next TIFF directory from a file and convert it to the internal
 * format. We read directories sequentially.
 */
int TIFFReadDirectory(TIFF *tif)
{
    static const char module[] = "TIFFReadDirectory";
    TIFFDirEntry *dir;
    uint16_t dircount;
    TIFFDirEntry *dp;
    uint16_t di;
    const TIFFField *fip;
    uint32_t fii = FAILED_FII;
    toff_t nextdiroff;
    int bitspersample_read = FALSE;
    int color_channels;

    if (tif->tif_nextdiroff == 0)
    {
        /* In this special case, tif_diroff needs also to be set to 0.
         * This is behind the last IFD, thus no checking or reading necessary.
         */
        tif->tif_diroff = tif->tif_nextdiroff;
        return 0;
    }

    nextdiroff = tif->tif_nextdiroff;
    /* tif_curdir++ and tif_nextdiroff should only be updated after SUCCESSFUL
     * reading of the directory. Otherwise, invalid IFD offsets could corrupt
     * the IFD list. */
    if (!_TIFFCheckDirNumberAndOffset(tif,
                                      tif->tif_curdir ==
                                              TIFF_NON_EXISTENT_DIR_NUMBER
                                          ? 0
                                          : tif->tif_curdir + 1,
                                      nextdiroff))
    {
        return 0; /* bad offset (IFD looping or more than TIFF_MAX_DIR_COUNT
                     IFDs) */
    }
    dircount = TIFFFetchDirectory(tif, nextdiroff, &dir, &tif->tif_nextdiroff);
    if (!dircount)
    {
        TIFFErrorExtR(tif, module,
                      "Failed to read directory at offset %" PRIu64,
                      nextdiroff);
        return 0;
    }
    /* Set global values after a valid directory has been fetched.
     * tif_diroff is already set to nextdiroff in TIFFFetchDirectory() in the
     * beginning. */
    if (tif->tif_curdir == TIFF_NON_EXISTENT_DIR_NUMBER)
        tif->tif_curdir = 0;
    else
        tif->tif_curdir++;
    (*tif->tif_cleanup)(tif); /* cleanup any previous compression state */

    TIFFReadDirectoryCheckOrder(tif, dir, dircount);

    /*
     * Mark duplicates of any tag to be ignored (bugzilla 1994)
     * to avoid certain pathological problems.
     */
    {
        TIFFDirEntry *ma;
        uint16_t mb;
        for (ma = dir, mb = 0; mb < dircount; ma++, mb++)
        {
            TIFFDirEntry *na;
            uint16_t nb;
            for (na = ma + 1, nb = mb + 1; nb < dircount; na++, nb++)
            {
                if (ma->tdir_tag == na->tdir_tag)
                {
                    na->tdir_ignore = TRUE;
                }
            }
        }
    }

    tif->tif_flags &= ~TIFF_BEENWRITING; /* reset before new dir */
    tif->tif_flags &= ~TIFF_BUF4WRITE;   /* reset before new dir */
    tif->tif_flags &= ~TIFF_CHOPPEDUPARRAYS;

    /* free any old stuff and reinit */
    TIFFFreeDirectory(tif);
    TIFFDefaultDirectory(tif);
    /*
     * Electronic Arts writes gray-scale TIFF files
     * without a PlanarConfiguration directory entry.
     * Thus we setup a default value here, even though
     * the TIFF spec says there is no default value.
     * After PlanarConfiguration is preset in TIFFDefaultDirectory()
     * the following setting is not needed, but does not harm either.
     */
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    /*
     * Setup default value and then make a pass over
     * the fields to check type and tag information,
     * and to extract info required to size data
     * structures.  A second pass is made afterwards
     * to read in everything not taken in the first pass.
     * But we must process the Compression tag first
     * in order to merge in codec-private tag definitions (otherwise
     * we may get complaints about unknown tags).  However, the
     * Compression tag may be dependent on the SamplesPerPixel
     * tag value because older TIFF specs permitted Compression
     * to be written as a SamplesPerPixel-count tag entry.
     * Thus if we don't first figure out the correct SamplesPerPixel
     * tag value then we may end up ignoring the Compression tag
     * value because it has an incorrect count value (if the
     * true value of SamplesPerPixel is not 1).
     */
    dp =
        TIFFReadDirectoryFindEntry(tif, dir, dircount, TIFFTAG_SAMPLESPERPIXEL);
    if (dp)
    {
        if (!TIFFFetchNormalTag(tif, dp, 0))
            goto bad;
        dp->tdir_ignore = TRUE;
    }
    dp = TIFFReadDirectoryFindEntry(tif, dir, dircount, TIFFTAG_COMPRESSION);
    if (dp)
    {
        /*
         * The 5.0 spec says the Compression tag has one value, while
         * earlier specs say it has one value per sample.  Because of
         * this, we accept the tag if one value is supplied with either
         * count.
         */
        uint16_t value;
        enum TIFFReadDirEntryErr err;
        err = TIFFReadDirEntryShort(tif, dp, &value);
        if (err == TIFFReadDirEntryErrCount)
            err = TIFFReadDirEntryPersampleShort(tif, dp, &value);
        if (err != TIFFReadDirEntryErrOk)
        {
            TIFFReadDirEntryOutputErr(tif, err, module, "Compression", 0);
            goto bad;
        }
        if (!TIFFSetField(tif, TIFFTAG_COMPRESSION, value))
            goto bad;
        dp->tdir_ignore = TRUE;
    }
    else
    {
        if (!TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE))
            goto bad;
    }
    /*
     * First real pass over the directory.
     */
    for (di = 0, dp = dir; di < dircount; di++, dp++)
    {
        if (!dp->tdir_ignore)
        {
            TIFFReadDirectoryFindFieldInfo(tif, dp->tdir_tag, &fii);
            if (fii == FAILED_FII)
            {
                TIFFWarningExtR(tif, module,
                                "Unknown field with tag %" PRIu16 " (0x%" PRIx16
                                ") encountered",
                                dp->tdir_tag, dp->tdir_tag);
                /* the following knowingly leaks the
                   anonymous field structure */
                if (!_TIFFMergeFields(
                        tif,
                        _TIFFCreateAnonField(tif, dp->tdir_tag,
                                             (TIFFDataType)dp->tdir_type),
                        1))
                {
                    TIFFWarningExtR(
                        tif, module,
                        "Registering anonymous field with tag %" PRIu16
                        " (0x%" PRIx16 ") failed",
                        dp->tdir_tag, dp->tdir_tag);
                    dp->tdir_ignore = TRUE;
                }
                else
                {
                    TIFFReadDirectoryFindFieldInfo(tif, dp->tdir_tag, &fii);
                    assert(fii != FAILED_FII);
                }
            }
        }
        if (!dp->tdir_ignore)
        {
            fip = tif->tif_fields[fii];
            if (fip->field_bit == FIELD_IGNORE)
                dp->tdir_ignore = TRUE;
            else
            {
                switch (dp->tdir_tag)
                {
                    case TIFFTAG_STRIPOFFSETS:
                    case TIFFTAG_STRIPBYTECOUNTS:
                    case TIFFTAG_TILEOFFSETS:
                    case TIFFTAG_TILEBYTECOUNTS:
                        TIFFSetFieldBit(tif, fip->field_bit);
                        break;
                    case TIFFTAG_IMAGEWIDTH:
                    case TIFFTAG_IMAGELENGTH:
                    case TIFFTAG_IMAGEDEPTH:
                    case TIFFTAG_TILELENGTH:
                    case TIFFTAG_TILEWIDTH:
                    case TIFFTAG_TILEDEPTH:
                    case TIFFTAG_PLANARCONFIG:
                    case TIFFTAG_ROWSPERSTRIP:
                    case TIFFTAG_EXTRASAMPLES:
                        if (!TIFFFetchNormalTag(tif, dp, 0))
                            goto bad;
                        dp->tdir_ignore = TRUE;
                        break;
                    default:
                        if (!_TIFFCheckFieldIsValidForCodec(tif, dp->tdir_tag))
                            dp->tdir_ignore = TRUE;
                        break;
                }
            }
        }
    }
    /*
     * XXX: OJPEG hack.
     * If a) compression is OJPEG, b) planarconfig tag says it's separate,
     * c) strip offsets/bytecounts tag are both present and
     * d) both contain exactly one value, then we consistently find
     * that the buggy implementation of the buggy compression scheme
     * matches contig planarconfig best. So we 'fix-up' the tag here
     */
    if ((tif->tif_dir.td_compression == COMPRESSION_OJPEG) &&
        (tif->tif_dir.td_planarconfig == PLANARCONFIG_SEPARATE))
    {
        if (!_TIFFFillStriles(tif))
            goto bad;
        dp = TIFFReadDirectoryFindEntry(tif, dir, dircount,
                                        TIFFTAG_STRIPOFFSETS);
        if ((dp != 0) && (dp->tdir_count == 1))
        {
            dp = TIFFReadDirectoryFindEntry(tif, dir, dircount,
                                            TIFFTAG_STRIPBYTECOUNTS);
            if ((dp != 0) && (dp->tdir_count == 1))
            {
                tif->tif_dir.td_planarconfig = PLANARCONFIG_CONTIG;
                TIFFWarningExtR(tif, module,
                                "Planarconfig tag value assumed incorrect, "
                                "assuming data is contig instead of chunky");
            }
        }
    }
    /*
     * Allocate directory structure and setup defaults.
     */
    if (!TIFFFieldSet(tif, FIELD_IMAGEDIMENSIONS))
    {
        MissingRequired(tif, "ImageLength");
        goto bad;
    }

    /*
     * Second pass: extract other information.
     */
    for (di = 0, dp = dir; di < dircount; di++, dp++)
    {
        if (!dp->tdir_ignore)
        {
            switch (dp->tdir_tag)
            {
                case TIFFTAG_MINSAMPLEVALUE:
                case TIFFTAG_MAXSAMPLEVALUE:
                case TIFFTAG_BITSPERSAMPLE:
                case TIFFTAG_DATATYPE:
                case TIFFTAG_SAMPLEFORMAT:
                    /*
                     * The MinSampleValue, MaxSampleValue, BitsPerSample
                     * DataType and SampleFormat tags are supposed to be
                     * written as one value/sample, but some vendors
                     * incorrectly write one value only -- so we accept
                     * that as well (yuck). Other vendors write correct
                     * value for NumberOfSamples, but incorrect one for
                     * BitsPerSample and friends, and we will read this
                     * too.
                     */
                    {
                        uint16_t value;
                        enum TIFFReadDirEntryErr err;
                        err = TIFFReadDirEntryShort(tif, dp, &value);
                        if (err == TIFFReadDirEntryErrCount)
                            err =
                                TIFFReadDirEntryPersampleShort(tif, dp, &value);
                        if (err != TIFFReadDirEntryErrOk)
                        {
                            fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                            TIFFReadDirEntryOutputErr(
                                tif, err, module,
                                fip ? fip->field_name : "unknown tagname", 0);
                            goto bad;
                        }
                        if (!TIFFSetField(tif, dp->tdir_tag, value))
                            goto bad;
                        if (dp->tdir_tag == TIFFTAG_BITSPERSAMPLE)
                            bitspersample_read = TRUE;
                    }
                    break;
                case TIFFTAG_SMINSAMPLEVALUE:
                case TIFFTAG_SMAXSAMPLEVALUE:
                {

                    double *data = NULL;
                    enum TIFFReadDirEntryErr err;
                    uint32_t saved_flags;
                    int m;
                    if (dp->tdir_count !=
                        (uint64_t)tif->tif_dir.td_samplesperpixel)
                        err = TIFFReadDirEntryErrCount;
                    else
                        err = TIFFReadDirEntryDoubleArray(tif, dp, &data);
                    if (err != TIFFReadDirEntryErrOk)
                    {
                        fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                        TIFFReadDirEntryOutputErr(
                            tif, err, module,
                            fip ? fip->field_name : "unknown tagname", 0);
                        goto bad;
                    }
                    saved_flags = tif->tif_flags;
                    tif->tif_flags |= TIFF_PERSAMPLE;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    tif->tif_flags = saved_flags;
                    _TIFFfreeExt(tif, data);
                    if (!m)
                        goto bad;
                }
                break;
                case TIFFTAG_STRIPOFFSETS:
                case TIFFTAG_TILEOFFSETS:
                    switch (dp->tdir_type)
                    {
                        case TIFF_SHORT:
                        case TIFF_LONG:
                        case TIFF_LONG8:
                            break;
                        default:
                            /* Warn except if directory typically created with
                             * TIFFDeferStrileArrayWriting() */
                            if (!(tif->tif_mode == O_RDWR &&
                                  dp->tdir_count == 0 && dp->tdir_type == 0 &&
                                  dp->tdir_offset.toff_long8 == 0))
                            {
                                fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                                TIFFWarningExtR(
                                    tif, module, "Invalid data type for tag %s",
                                    fip ? fip->field_name : "unknown tagname");
                            }
                            break;
                    }
                    _TIFFmemcpy(&(tif->tif_dir.td_stripoffset_entry), dp,
                                sizeof(TIFFDirEntry));
                    break;
                case TIFFTAG_STRIPBYTECOUNTS:
                case TIFFTAG_TILEBYTECOUNTS:
                    switch (dp->tdir_type)
                    {
                        case TIFF_SHORT:
                        case TIFF_LONG:
                        case TIFF_LONG8:
                            break;
                        default:
                            /* Warn except if directory typically created with
                             * TIFFDeferStrileArrayWriting() */
                            if (!(tif->tif_mode == O_RDWR &&
                                  dp->tdir_count == 0 && dp->tdir_type == 0 &&
                                  dp->tdir_offset.toff_long8 == 0))
                            {
                                fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                                TIFFWarningExtR(
                                    tif, module, "Invalid data type for tag %s",
                                    fip ? fip->field_name : "unknown tagname");
                            }
                            break;
                    }
                    _TIFFmemcpy(&(tif->tif_dir.td_stripbytecount_entry), dp,
                                sizeof(TIFFDirEntry));
                    break;
                case TIFFTAG_COLORMAP:
                case TIFFTAG_TRANSFERFUNCTION:
                {
                    enum TIFFReadDirEntryErr err;
                    uint32_t countpersample;
                    uint32_t countrequired;
                    uint32_t incrementpersample;
                    uint16_t *value = NULL;
                    /* It would be dangerous to instantiate those tag values */
                    /* since if td_bitspersample has not yet been read (due to
                     */
                    /* unordered tags), it could be read afterwards with a */
                    /* values greater than the default one (1), which may cause
                     */
                    /* crashes in user code */
                    if (!bitspersample_read)
                    {
                        fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                        TIFFWarningExtR(
                            tif, module,
                            "Ignoring %s since BitsPerSample tag not found",
                            fip ? fip->field_name : "unknown tagname");
                        continue;
                    }
                    /* ColorMap or TransferFunction for high bit */
                    /* depths do not make much sense and could be */
                    /* used as a denial of service vector */
                    if (tif->tif_dir.td_bitspersample > 24)
                    {
                        fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                        TIFFWarningExtR(
                            tif, module,
                            "Ignoring %s because BitsPerSample=%" PRIu16 ">24",
                            fip ? fip->field_name : "unknown tagname",
                            tif->tif_dir.td_bitspersample);
                        continue;
                    }
                    countpersample = (1U << tif->tif_dir.td_bitspersample);
                    if ((dp->tdir_tag == TIFFTAG_TRANSFERFUNCTION) &&
                        (dp->tdir_count == (uint64_t)countpersample))
                    {
                        countrequired = countpersample;
                        incrementpersample = 0;
                    }
                    else
                    {
                        countrequired = 3 * countpersample;
                        incrementpersample = countpersample;
                    }
                    if (dp->tdir_count != (uint64_t)countrequired)
                        err = TIFFReadDirEntryErrCount;
                    else
                        err = TIFFReadDirEntryShortArray(tif, dp, &value);
                    if (err != TIFFReadDirEntryErrOk)
                    {
                        fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                        TIFFReadDirEntryOutputErr(
                            tif, err, module,
                            fip ? fip->field_name : "unknown tagname", 1);
                    }
                    else
                    {
                        TIFFSetField(tif, dp->tdir_tag, value,
                                     value + incrementpersample,
                                     value + 2 * incrementpersample);
                        _TIFFfreeExt(tif, value);
                    }
                }
                break;
                    /* BEGIN REV 4.0 COMPATIBILITY */
                case TIFFTAG_OSUBFILETYPE:
                {
                    uint16_t valueo;
                    uint32_t value;
                    if (TIFFReadDirEntryShort(tif, dp, &valueo) ==
                        TIFFReadDirEntryErrOk)
                    {
                        switch (valueo)
                        {
                            case OFILETYPE_REDUCEDIMAGE:
                                value = FILETYPE_REDUCEDIMAGE;
                                break;
                            case OFILETYPE_PAGE:
                                value = FILETYPE_PAGE;
                                break;
                            default:
                                value = 0;
                                break;
                        }
                        if (value != 0)
                            TIFFSetField(tif, TIFFTAG_SUBFILETYPE, value);
                    }
                }
                break;
                /* END REV 4.0 COMPATIBILITY */
#if 0
                case TIFFTAG_EP_BATTERYLEVEL:
                    /* TIFFTAG_EP_BATTERYLEVEL can be RATIONAL or ASCII.
                     * LibTiff defines it as ASCII and converts RATIONAL to an
                     * ASCII string. */
                    switch (dp->tdir_type)
                    {
                        case TIFF_RATIONAL:
                        {
                            /* Read rational and convert to ASCII*/
                            enum TIFFReadDirEntryErr err;
                            TIFFRational_t rValue;
                            err = TIFFReadDirEntryCheckedRationalDirect(
                                tif, dp, &rValue);
                            if (err != TIFFReadDirEntryErrOk)
                            {
                                fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                                TIFFReadDirEntryOutputErr(
                                    tif, err, module,
                                    fip ? fip->field_name : "unknown tagname",
                                    1);
                            }
                            else
                            {
                                char szAux[32];
                                snprintf(szAux, sizeof(szAux) - 1, "%d/%d",
                                         rValue.uNum, rValue.uDenom);
                                TIFFSetField(tif, dp->tdir_tag, szAux);
                            }
                        }
                        break;
                        case TIFF_ASCII:
                            (void)TIFFFetchNormalTag(tif, dp, TRUE);
                            break;
                        default:
                            fip = TIFFFieldWithTag(tif, dp->tdir_tag);
                            TIFFWarningExtR(tif, module,
                                            "Invalid data type for tag %s. "
                                            "ASCII or RATIONAL expected",
                                            fip ? fip->field_name
                                                : "unknown tagname");
                            break;
                    }
                    break;
#endif
                default:
                    (void)TIFFFetchNormalTag(tif, dp, TRUE);
                    break;
            }
        } /* -- if (!dp->tdir_ignore) */
    }     /* -- for-loop -- */

    /*
     * OJPEG hack:
     * - If a) compression is OJPEG, and b) photometric tag is missing,
     * then we consistently find that photometric should be YCbCr
     * - If a) compression is OJPEG, and b) photometric tag says it's RGB,
     * then we consistently find that the buggy implementation of the
     * buggy compression scheme matches photometric YCbCr instead.
     * - If a) compression is OJPEG, and b) bitspersample tag is missing,
     * then we consistently find bitspersample should be 8.
     * - If a) compression is OJPEG, b) samplesperpixel tag is missing,
     * and c) photometric is RGB or YCbCr, then we consistently find
     * samplesperpixel should be 3
     * - If a) compression is OJPEG, b) samplesperpixel tag is missing,
     * and c) photometric is MINISWHITE or MINISBLACK, then we consistently
     * find samplesperpixel should be 3
     */
    if (tif->tif_dir.td_compression == COMPRESSION_OJPEG)
    {
        if (!TIFFFieldSet(tif, FIELD_PHOTOMETRIC))
        {
            TIFFWarningExtR(
                tif, module,
                "Photometric tag is missing, assuming data is YCbCr");
            if (!TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_YCBCR))
                goto bad;
        }
        else if (tif->tif_dir.td_photometric == PHOTOMETRIC_RGB)
        {
            tif->tif_dir.td_photometric = PHOTOMETRIC_YCBCR;
            TIFFWarningExtR(tif, module,
                            "Photometric tag value assumed incorrect, "
                            "assuming data is YCbCr instead of RGB");
        }
        if (!TIFFFieldSet(tif, FIELD_BITSPERSAMPLE))
        {
            TIFFWarningExtR(
                tif, module,
                "BitsPerSample tag is missing, assuming 8 bits per sample");
            if (!TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8))
                goto bad;
        }
        if (!TIFFFieldSet(tif, FIELD_SAMPLESPERPIXEL))
        {
            if (tif->tif_dir.td_photometric == PHOTOMETRIC_RGB)
            {
                TIFFWarningExtR(tif, module,
                                "SamplesPerPixel tag is missing, "
                                "assuming correct SamplesPerPixel value is 3");
                if (!TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3))
                    goto bad;
            }
            if (tif->tif_dir.td_photometric == PHOTOMETRIC_YCBCR)
            {
                TIFFWarningExtR(tif, module,
                                "SamplesPerPixel tag is missing, "
                                "applying correct SamplesPerPixel value of 3");
                if (!TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3))
                    goto bad;
            }
            else if ((tif->tif_dir.td_photometric == PHOTOMETRIC_MINISWHITE) ||
                     (tif->tif_dir.td_photometric == PHOTOMETRIC_MINISBLACK))
            {
                /*
                 * SamplesPerPixel tag is missing, but is not required
                 * by spec.  Assume correct SamplesPerPixel value of 1.
                 */
                if (!TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1))
                    goto bad;
            }
        }
    }

    /*
     * Setup appropriate structures (by strip or by tile)
     * We do that only after the above OJPEG hack which alters SamplesPerPixel
     * and thus influences the number of strips in the separate planarconfig.
     */
    if (!TIFFFieldSet(tif, FIELD_TILEDIMENSIONS))
    {
        tif->tif_dir.td_nstrips = TIFFNumberOfStrips(tif);
        tif->tif_dir.td_tilewidth = tif->tif_dir.td_imagewidth;
        tif->tif_dir.td_tilelength = tif->tif_dir.td_rowsperstrip;
        tif->tif_dir.td_tiledepth = tif->tif_dir.td_imagedepth;
        tif->tif_flags &= ~TIFF_ISTILED;
    }
    else
    {
        tif->tif_dir.td_nstrips = TIFFNumberOfTiles(tif);
        tif->tif_flags |= TIFF_ISTILED;
    }
    if (!tif->tif_dir.td_nstrips)
    {
        TIFFErrorExtR(tif, module, "Cannot handle zero number of %s",
                      isTiled(tif) ? "tiles" : "strips");
        goto bad;
    }
    tif->tif_dir.td_stripsperimage = tif->tif_dir.td_nstrips;
    if (tif->tif_dir.td_planarconfig == PLANARCONFIG_SEPARATE)
        tif->tif_dir.td_stripsperimage /= tif->tif_dir.td_samplesperpixel;
    if (!TIFFFieldSet(tif, FIELD_STRIPOFFSETS))
    {
#ifdef OJPEG_SUPPORT
        if ((tif->tif_dir.td_compression == COMPRESSION_OJPEG) &&
            (isTiled(tif) == 0) && (tif->tif_dir.td_nstrips == 1))
        {
            /*
             * XXX: OJPEG hack.
             * If a) compression is OJPEG, b) it's not a tiled TIFF,
             * and c) the number of strips is 1,
             * then we tolerate the absence of stripoffsets tag,
             * because, presumably, all required data is in the
             * JpegInterchangeFormat stream.
             */
            TIFFSetFieldBit(tif, FIELD_STRIPOFFSETS);
        }
        else
#endif
        {
            MissingRequired(tif, isTiled(tif) ? "TileOffsets" : "StripOffsets");
            goto bad;
        }
    }

    if (tif->tif_mode == O_RDWR &&
        tif->tif_dir.td_stripoffset_entry.tdir_tag != 0 &&
        tif->tif_dir.td_stripoffset_entry.tdir_count == 0 &&
        tif->tif_dir.td_stripoffset_entry.tdir_type == 0 &&
        tif->tif_dir.td_stripoffset_entry.tdir_offset.toff_long8 == 0 &&
        tif->tif_dir.td_stripbytecount_entry.tdir_tag != 0 &&
        tif->tif_dir.td_stripbytecount_entry.tdir_count == 0 &&
        tif->tif_dir.td_stripbytecount_entry.tdir_type == 0 &&
        tif->tif_dir.td_stripbytecount_entry.tdir_offset.toff_long8 == 0)
    {
        /* Directory typically created with TIFFDeferStrileArrayWriting() */
        TIFFSetupStrips(tif);
    }
    else if (!(tif->tif_flags & TIFF_DEFERSTRILELOAD))
    {
        if (tif->tif_dir.td_stripoffset_entry.tdir_tag != 0)
        {
            if (!TIFFFetchStripThing(tif, &(tif->tif_dir.td_stripoffset_entry),
                                     tif->tif_dir.td_nstrips,
                                     &tif->tif_dir.td_stripoffset_p))
            {
                goto bad;
            }
        }
        if (tif->tif_dir.td_stripbytecount_entry.tdir_tag != 0)
        {
            if (!TIFFFetchStripThing(
                    tif, &(tif->tif_dir.td_stripbytecount_entry),
                    tif->tif_dir.td_nstrips, &tif->tif_dir.td_stripbytecount_p))
            {
                goto bad;
            }
        }
    }

    /*
     * Make sure all non-color channels are extrasamples.
     * If it's not the case, define them as such.
     */
    color_channels = _TIFFGetMaxColorChannels(tif->tif_dir.td_photometric);
    if (color_channels &&
        tif->tif_dir.td_samplesperpixel - tif->tif_dir.td_extrasamples >
            color_channels)
    {
        uint16_t old_extrasamples;
        uint16_t *new_sampleinfo;

        TIFFWarningExtR(
            tif, module,
            "Sum of Photometric type-related "
            "color channels and ExtraSamples doesn't match SamplesPerPixel. "
            "Defining non-color channels as ExtraSamples.");

        old_extrasamples = tif->tif_dir.td_extrasamples;
        tif->tif_dir.td_extrasamples =
            (uint16_t)(tif->tif_dir.td_samplesperpixel - color_channels);

        // sampleinfo should contain information relative to these new extra
        // samples
        new_sampleinfo = (uint16_t *)_TIFFcallocExt(
            tif, tif->tif_dir.td_extrasamples, sizeof(uint16_t));
        if (!new_sampleinfo)
        {
            TIFFErrorExtR(tif, module,
                          "Failed to allocate memory for "
                          "temporary new sampleinfo array "
                          "(%" PRIu16 " 16 bit elements)",
                          tif->tif_dir.td_extrasamples);
            goto bad;
        }

        if (old_extrasamples > 0)
            memcpy(new_sampleinfo, tif->tif_dir.td_sampleinfo,
                   old_extrasamples * sizeof(uint16_t));
        _TIFFsetShortArrayExt(tif, &tif->tif_dir.td_sampleinfo, new_sampleinfo,
                              tif->tif_dir.td_extrasamples);
        _TIFFfreeExt(tif, new_sampleinfo);
    }

    /*
     * Verify Palette image has a Colormap.
     */
    if (tif->tif_dir.td_photometric == PHOTOMETRIC_PALETTE &&
        !TIFFFieldSet(tif, FIELD_COLORMAP))
    {
        if (tif->tif_dir.td_bitspersample >= 8 &&
            tif->tif_dir.td_samplesperpixel == 3)
            tif->tif_dir.td_photometric = PHOTOMETRIC_RGB;
        else if (tif->tif_dir.td_bitspersample >= 8)
            tif->tif_dir.td_photometric = PHOTOMETRIC_MINISBLACK;
        else
        {
            MissingRequired(tif, "Colormap");
            goto bad;
        }
    }
    /*
     * OJPEG hack:
     * We do no further messing with strip/tile offsets/bytecounts in OJPEG
     * TIFFs
     */
    if (tif->tif_dir.td_compression != COMPRESSION_OJPEG)
    {
        /*
         * Attempt to deal with a missing StripByteCounts tag.
         */
        if (!TIFFFieldSet(tif, FIELD_STRIPBYTECOUNTS))
        {
            /*
             * Some manufacturers violate the spec by not giving
             * the size of the strips.  In this case, assume there
             * is one uncompressed strip of data.
             */
            if ((tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG &&
                 tif->tif_dir.td_nstrips > 1) ||
                (tif->tif_dir.td_planarconfig == PLANARCONFIG_SEPARATE &&
                 tif->tif_dir.td_nstrips !=
                     (uint32_t)tif->tif_dir.td_samplesperpixel))
            {
                MissingRequired(tif, "StripByteCounts");
                goto bad;
            }
            TIFFWarningExtR(
                tif, module,
                "TIFF directory is missing required "
                "\"StripByteCounts\" field, calculating from imagelength");
            if (EstimateStripByteCounts(tif, dir, dircount) < 0)
                goto bad;
        }
        else if (tif->tif_dir.td_nstrips == 1 &&
                 !(tif->tif_flags & TIFF_ISTILED) && ByteCountLooksBad(tif))
        {
            /*
             * XXX: Plexus (and others) sometimes give a value of
             * zero for a tag when they don't know what the
             * correct value is!  Try and handle the simple case
             * of estimating the size of a one strip image.
             */
            TIFFWarningExtR(tif, module,
                            "Bogus \"StripByteCounts\" field, ignoring and "
                            "calculating from imagelength");
            if (EstimateStripByteCounts(tif, dir, dircount) < 0)
                goto bad;
        }
        else if (!(tif->tif_flags & TIFF_DEFERSTRILELOAD) &&
                 tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG &&
                 tif->tif_dir.td_nstrips > 2 &&
                 tif->tif_dir.td_compression == COMPRESSION_NONE &&
                 TIFFGetStrileByteCount(tif, 0) !=
                     TIFFGetStrileByteCount(tif, 1) &&
                 TIFFGetStrileByteCount(tif, 0) != 0 &&
                 TIFFGetStrileByteCount(tif, 1) != 0)
        {
            /*
             * XXX: Some vendors fill StripByteCount array with
             * absolutely wrong values (it can be equal to
             * StripOffset array, for example). Catch this case
             * here.
             *
             * We avoid this check if deferring strile loading
             * as it would always force us to load the strip/tile
             * information.
             */
            TIFFWarningExtR(tif, module,
                            "Wrong \"StripByteCounts\" field, ignoring and "
                            "calculating from imagelength");
            if (EstimateStripByteCounts(tif, dir, dircount) < 0)
                goto bad;
        }
    }
    if (dir)
    {
        _TIFFfreeExt(tif, dir);
        dir = NULL;
    }
    if (!TIFFFieldSet(tif, FIELD_MAXSAMPLEVALUE))
    {
        if (tif->tif_dir.td_bitspersample >= 16)
            tif->tif_dir.td_maxsamplevalue = 0xFFFF;
        else
            tif->tif_dir.td_maxsamplevalue =
                (uint16_t)((1L << tif->tif_dir.td_bitspersample) - 1);
    }

#ifdef STRIPBYTECOUNTSORTED_UNUSED
    /*
     * XXX: We can optimize checking for the strip bounds using the sorted
     * bytecounts array. See also comments for TIFFAppendToStrip()
     * function in tif_write.c.
     */
    if (!(tif->tif_flags & TIFF_DEFERSTRILELOAD) && tif->tif_dir.td_nstrips > 1)
    {
        uint32_t strip;

        tif->tif_dir.td_stripbytecountsorted = 1;
        for (strip = 1; strip < tif->tif_dir.td_nstrips; strip++)
        {
            if (TIFFGetStrileOffset(tif, strip - 1) >
                TIFFGetStrileOffset(tif, strip))
            {
                tif->tif_dir.td_stripbytecountsorted = 0;
                break;
            }
        }
    }
#endif

    /*
     * An opportunity for compression mode dependent tag fixup
     */
    (*tif->tif_fixuptags)(tif);

    /*
     * Some manufacturers make life difficult by writing
     * large amounts of uncompressed data as a single strip.
     * This is contrary to the recommendations of the spec.
     * The following makes an attempt at breaking such images
     * into strips closer to the recommended 8k bytes.  A
     * side effect, however, is that the RowsPerStrip tag
     * value may be changed.
     */
    if ((tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG) &&
        (tif->tif_dir.td_nstrips == 1) &&
        (tif->tif_dir.td_compression == COMPRESSION_NONE) &&
        ((tif->tif_flags & (TIFF_STRIPCHOP | TIFF_ISTILED)) == TIFF_STRIPCHOP))
    {
        ChopUpSingleUncompressedStrip(tif);
    }

    /* There are also uncompressed striped files with strips larger than */
    /* 2 GB, which make them unfriendly with a lot of code. If possible, */
    /* try to expose smaller "virtual" strips. */
    if (tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG &&
        tif->tif_dir.td_compression == COMPRESSION_NONE &&
        (tif->tif_flags & (TIFF_STRIPCHOP | TIFF_ISTILED)) == TIFF_STRIPCHOP &&
        TIFFStripSize64(tif) > 0x7FFFFFFFUL)
    {
        TryChopUpUncompressedBigTiff(tif);
    }

    /*
     * Clear the dirty directory flag.
     */
    tif->tif_flags &= ~TIFF_DIRTYDIRECT;
    tif->tif_flags &= ~TIFF_DIRTYSTRIP;

    /*
     * Reinitialize i/o since we are starting on a new directory.
     */
    tif->tif_row = (uint32_t)-1;
    tif->tif_curstrip = (uint32_t)-1;
    tif->tif_col = (uint32_t)-1;
    tif->tif_curtile = (uint32_t)-1;
    tif->tif_tilesize = (tmsize_t)-1;

    tif->tif_scanlinesize = TIFFScanlineSize(tif);
    if (!tif->tif_scanlinesize)
    {
        TIFFErrorExtR(tif, module, "Cannot handle zero scanline size");
        return (0);
    }

    if (isTiled(tif))
    {
        tif->tif_tilesize = TIFFTileSize(tif);
        if (!tif->tif_tilesize)
        {
            TIFFErrorExtR(tif, module, "Cannot handle zero tile size");
            return (0);
        }
    }
    else
    {
        if (!TIFFStripSize(tif))
        {
            TIFFErrorExtR(tif, module, "Cannot handle zero strip size");
            return (0);
        }
    }
    return (1);
bad:
    if (dir)
        _TIFFfreeExt(tif, dir);
    return (0);
}

static void TIFFReadDirectoryCheckOrder(TIFF *tif, TIFFDirEntry *dir,
                                        uint16_t dircount)
{
    static const char module[] = "TIFFReadDirectoryCheckOrder";
    uint32_t m;
    uint16_t n;
    TIFFDirEntry *o;
    m = 0;
    for (n = 0, o = dir; n < dircount; n++, o++)
    {
        if (o->tdir_tag < m)
        {
            TIFFWarningExtR(tif, module,
                            "Invalid TIFF directory; tags are not sorted in "
                            "ascending order");
            break;
        }
        m = o->tdir_tag + 1;
    }
}

static TIFFDirEntry *TIFFReadDirectoryFindEntry(TIFF *tif, TIFFDirEntry *dir,
                                                uint16_t dircount,
                                                uint16_t tagid)
{
    TIFFDirEntry *m;
    uint16_t n;
    (void)tif;
    for (m = dir, n = 0; n < dircount; m++, n++)
    {
        if (m->tdir_tag == tagid)
            return (m);
    }
    return (0);
}

static void TIFFReadDirectoryFindFieldInfo(TIFF *tif, uint16_t tagid,
                                           uint32_t *fii)
{
    int32_t ma, mb, mc;
    ma = -1;
    mc = (int32_t)tif->tif_nfields;
    while (1)
    {
        if (ma + 1 == mc)
        {
            *fii = FAILED_FII;
            return;
        }
        mb = (ma + mc) / 2;
        if (tif->tif_fields[mb]->field_tag == (uint32_t)tagid)
            break;
        if (tif->tif_fields[mb]->field_tag < (uint32_t)tagid)
            ma = mb;
        else
            mc = mb;
    }
    while (1)
    {
        if (mb == 0)
            break;
        if (tif->tif_fields[mb - 1]->field_tag != (uint32_t)tagid)
            break;
        mb--;
    }
    *fii = mb;
}

/*
 * Read custom directory from the arbitrary offset.
 * The code is very similar to TIFFReadDirectory().
 */
int TIFFReadCustomDirectory(TIFF *tif, toff_t diroff,
                            const TIFFFieldArray *infoarray)
{
    static const char module[] = "TIFFReadCustomDirectory";
    TIFFDirEntry *dir;
    uint16_t dircount;
    TIFFDirEntry *dp;
    uint16_t di;
    const TIFFField *fip;
    uint32_t fii;
    (*tif->tif_cleanup)(tif); /* cleanup any previous compression state */
    _TIFFSetupFields(tif, infoarray);
    dircount = TIFFFetchDirectory(tif, diroff, &dir, NULL);
    if (!dircount)
    {
        TIFFErrorExtR(tif, module,
                      "Failed to read custom directory at offset %" PRIu64,
                      diroff);
        return 0;
    }
    TIFFFreeDirectory(tif);
    _TIFFmemset(&tif->tif_dir, 0, sizeof(TIFFDirectory));
    TIFFReadDirectoryCheckOrder(tif, dir, dircount);
    for (di = 0, dp = dir; di < dircount; di++, dp++)
    {
        TIFFReadDirectoryFindFieldInfo(tif, dp->tdir_tag, &fii);
        if (fii == FAILED_FII)
        {
            TIFFWarningExtR(tif, module,
                            "Unknown field with tag %" PRIu16 " (0x%" PRIx16
                            ") encountered",
                            dp->tdir_tag, dp->tdir_tag);
            if (!_TIFFMergeFields(
                    tif,
                    _TIFFCreateAnonField(tif, dp->tdir_tag,
                                         (TIFFDataType)dp->tdir_type),
                    1))
            {
                TIFFWarningExtR(tif, module,
                                "Registering anonymous field with tag %" PRIu16
                                " (0x%" PRIx16 ") failed",
                                dp->tdir_tag, dp->tdir_tag);
                dp->tdir_ignore = TRUE;
            }
            else
            {
                TIFFReadDirectoryFindFieldInfo(tif, dp->tdir_tag, &fii);
                assert(fii != FAILED_FII);
            }
        }
        if (!dp->tdir_ignore)
        {
            fip = tif->tif_fields[fii];
            if (fip->field_bit == FIELD_IGNORE)
                dp->tdir_ignore = TRUE;
            else
            {
                /* check data type */
                while ((fip->field_type != TIFF_ANY) &&
                       (fip->field_type != dp->tdir_type))
                {
                    fii++;
                    if ((fii == tif->tif_nfields) ||
                        (tif->tif_fields[fii]->field_tag !=
                         (uint32_t)dp->tdir_tag))
                    {
                        fii = 0xFFFF;
                        break;
                    }
                    fip = tif->tif_fields[fii];
                }
                if (fii == 0xFFFF)
                {
                    TIFFWarningExtR(tif, module,
                                    "Wrong data type %" PRIu16
                                    " for \"%s\"; tag ignored",
                                    dp->tdir_type, fip->field_name);
                    dp->tdir_ignore = TRUE;
                }
                else
                {
                    /* check count if known in advance */
                    if ((fip->field_readcount != TIFF_VARIABLE) &&
                        (fip->field_readcount != TIFF_VARIABLE2))
                    {
                        uint32_t expected;
                        if (fip->field_readcount == TIFF_SPP)
                            expected =
                                (uint32_t)tif->tif_dir.td_samplesperpixel;
                        else
                            expected = (uint32_t)fip->field_readcount;
                        if (!CheckDirCount(tif, dp, expected))
                            dp->tdir_ignore = TRUE;
                    }
                }
            }
            if (!dp->tdir_ignore)
            {
                switch (dp->tdir_tag)
                {
                    case EXIFTAG_SUBJECTDISTANCE:
                        if (!TIFFFieldIsAnonymous(fip))
                        {
                            /* should only be called on a Exif directory */
                            /* when exifFields[] is active */
                            (void)TIFFFetchSubjectDistance(tif, dp);
                        }
                        else
                        {
                            (void)TIFFFetchNormalTag(tif, dp, TRUE);
                        }
                        break;
                    default:
                        (void)TIFFFetchNormalTag(tif, dp, TRUE);
                        break;
                }
            } /*-- if (!dp->tdir_ignore) */
        }
    }
    /* To be able to return from SubIFD or custom-IFD to main-IFD */
    tif->tif_setdirectory_force_absolute = TRUE;
    if (dir)
        _TIFFfreeExt(tif, dir);
    return 1;
}

/*
 * EXIF is important special case of custom IFD, so we have a special
 * function to read it.
 */
int TIFFReadEXIFDirectory(TIFF *tif, toff_t diroff)
{
    const TIFFFieldArray *exifFieldArray;
    exifFieldArray = _TIFFGetExifFields();
    return TIFFReadCustomDirectory(tif, diroff, exifFieldArray);
}

/*
 *--: EXIF-GPS custom directory reading as another special case of custom IFD.
 */
int TIFFReadGPSDirectory(TIFF *tif, toff_t diroff)
{
    const TIFFFieldArray *gpsFieldArray;
    gpsFieldArray = _TIFFGetGpsFields();
    return TIFFReadCustomDirectory(tif, diroff, gpsFieldArray);
}

static int EstimateStripByteCounts(TIFF *tif, TIFFDirEntry *dir,
                                   uint16_t dircount)
{
    static const char module[] = "EstimateStripByteCounts";

    TIFFDirEntry *dp;
    TIFFDirectory *td = &tif->tif_dir;
    uint32_t strip;

    /* Do not try to load stripbytecount as we will compute it */
    if (!_TIFFFillStrilesInternal(tif, 0))
        return -1;

    if (td->td_stripbytecount_p)
        _TIFFfreeExt(tif, td->td_stripbytecount_p);
    td->td_stripbytecount_p = (uint64_t *)_TIFFCheckMalloc(
        tif, td->td_nstrips, sizeof(uint64_t), "for \"StripByteCounts\" array");
    if (td->td_stripbytecount_p == NULL)
        return -1;

    if (td->td_compression != COMPRESSION_NONE)
    {
        uint64_t space;
        uint64_t filesize;
        uint16_t n;
        filesize = TIFFGetFileSize(tif);
        if (!(tif->tif_flags & TIFF_BIGTIFF))
            space = sizeof(TIFFHeaderClassic) + 2 + dircount * 12 + 4;
        else
            space = sizeof(TIFFHeaderBig) + 8 + dircount * 20 + 8;
        /* calculate amount of space used by indirect values */
        for (dp = dir, n = dircount; n > 0; n--, dp++)
        {
            uint32_t typewidth;
            uint64_t datasize;
            typewidth = TIFFDataWidth((TIFFDataType)dp->tdir_type);
            if (typewidth == 0)
            {
                TIFFErrorExtR(
                    tif, module,
                    "Cannot determine size of unknown tag type %" PRIu16,
                    dp->tdir_type);
                return -1;
            }
            if (dp->tdir_count > UINT64_MAX / typewidth)
                return -1;
            datasize = (uint64_t)typewidth * dp->tdir_count;
            if (!(tif->tif_flags & TIFF_BIGTIFF))
            {
                if (datasize <= 4)
                    datasize = 0;
            }
            else
            {
                if (datasize <= 8)
                    datasize = 0;
            }
            if (space > UINT64_MAX - datasize)
                return -1;
            space += datasize;
        }
        if (filesize < space)
            /* we should perhaps return in error ? */
            space = filesize;
        else
            space = filesize - space;
        if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
            space /= td->td_samplesperpixel;
        for (strip = 0; strip < td->td_nstrips; strip++)
            td->td_stripbytecount_p[strip] = space;
        /*
         * This gross hack handles the case were the offset to
         * the last strip is past the place where we think the strip
         * should begin.  Since a strip of data must be contiguous,
         * it's safe to assume that we've overestimated the amount
         * of data in the strip and trim this number back accordingly.
         */
        strip--;
        if (td->td_stripoffset_p[strip] >
            UINT64_MAX - td->td_stripbytecount_p[strip])
            return -1;
        if (td->td_stripoffset_p[strip] + td->td_stripbytecount_p[strip] >
            filesize)
        {
            if (td->td_stripoffset_p[strip] >= filesize)
            {
                /* Not sure what we should in that case... */
                td->td_stripbytecount_p[strip] = 0;
            }
            else
            {
                td->td_stripbytecount_p[strip] =
                    filesize - td->td_stripoffset_p[strip];
            }
        }
    }
    else if (isTiled(tif))
    {
        uint64_t bytespertile = TIFFTileSize64(tif);

        for (strip = 0; strip < td->td_nstrips; strip++)
            td->td_stripbytecount_p[strip] = bytespertile;
    }
    else
    {
        uint64_t rowbytes = TIFFScanlineSize64(tif);
        uint32_t rowsperstrip = td->td_imagelength / td->td_stripsperimage;
        for (strip = 0; strip < td->td_nstrips; strip++)
        {
            if (rowbytes > 0 && rowsperstrip > UINT64_MAX / rowbytes)
                return -1;
            td->td_stripbytecount_p[strip] = rowbytes * rowsperstrip;
        }
    }
    TIFFSetFieldBit(tif, FIELD_STRIPBYTECOUNTS);
    if (!TIFFFieldSet(tif, FIELD_ROWSPERSTRIP))
        td->td_rowsperstrip = td->td_imagelength;
    return 1;
}

static void MissingRequired(TIFF *tif, const char *tagname)
{
    static const char module[] = "MissingRequired";

    TIFFErrorExtR(tif, module,
                  "TIFF directory is missing required \"%s\" field", tagname);
}

static unsigned long hashFuncOffsetToNumber(const void *elt)
{
    const TIFFOffsetAndDirNumber *offsetAndDirNumber =
        (const TIFFOffsetAndDirNumber *)elt;
    const uint32_t hash = (uint32_t)(offsetAndDirNumber->offset >> 32) ^
                          ((uint32_t)offsetAndDirNumber->offset & 0xFFFFFFFFU);
    return hash;
}

static bool equalFuncOffsetToNumber(const void *elt1, const void *elt2)
{
    const TIFFOffsetAndDirNumber *offsetAndDirNumber1 =
        (const TIFFOffsetAndDirNumber *)elt1;
    const TIFFOffsetAndDirNumber *offsetAndDirNumber2 =
        (const TIFFOffsetAndDirNumber *)elt2;
    return offsetAndDirNumber1->offset == offsetAndDirNumber2->offset;
}

static unsigned long hashFuncNumberToOffset(const void *elt)
{
    const TIFFOffsetAndDirNumber *offsetAndDirNumber =
        (const TIFFOffsetAndDirNumber *)elt;
    return offsetAndDirNumber->dirNumber;
}

static bool equalFuncNumberToOffset(const void *elt1, const void *elt2)
{
    const TIFFOffsetAndDirNumber *offsetAndDirNumber1 =
        (const TIFFOffsetAndDirNumber *)elt1;
    const TIFFOffsetAndDirNumber *offsetAndDirNumber2 =
        (const TIFFOffsetAndDirNumber *)elt2;
    return offsetAndDirNumber1->dirNumber == offsetAndDirNumber2->dirNumber;
}

/*
 * Check the directory number and offset against the list of already seen
 * directory numbers and offsets. This is a trick to prevent IFD looping.
 * The one can create TIFF file with looped directory pointers. We will
 * maintain a list of already seen directories and check every IFD offset
 * and its IFD number against that list. However, the offset of an IFD number
 * can change - e.g. when writing updates to file.
 * Returns 1 if all is ok; 0 if last directory or IFD loop is encountered,
 * or an error has occurred.
 */
int _TIFFCheckDirNumberAndOffset(TIFF *tif, tdir_t dirn, uint64_t diroff)
{
    if (diroff == 0) /* no more directories */
        return 0;

    if (tif->tif_map_dir_offset_to_number == NULL)
    {
        tif->tif_map_dir_offset_to_number = TIFFHashSetNew(
            hashFuncOffsetToNumber, equalFuncOffsetToNumber, free);
        if (tif->tif_map_dir_offset_to_number == NULL)
        {
            TIFFErrorExtR(tif, "_TIFFCheckDirNumberAndOffset",
                          "Not enough memory");
            return 1;
        }
    }

    if (tif->tif_map_dir_number_to_offset == NULL)
    {
        /* No free callback for this map, as it shares the same items as
         * tif->tif_map_dir_offset_to_number. */
        tif->tif_map_dir_number_to_offset = TIFFHashSetNew(
            hashFuncNumberToOffset, equalFuncNumberToOffset, NULL);
        if (tif->tif_map_dir_number_to_offset == NULL)
        {
            TIFFErrorExtR(tif, "_TIFFCheckDirNumberAndOffset",
                          "Not enough memory");
            return 1;
        }
    }

    /* Check if offset is already in the list:
     * - yes: check, if offset is at the same IFD number - if not, it is an IFD
     * loop
     * -  no: add to list or update offset at that IFD number
     */
    TIFFOffsetAndDirNumber entry;
    entry.offset = diroff;
    entry.dirNumber = dirn;

    TIFFOffsetAndDirNumber *foundEntry =
        (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
            tif->tif_map_dir_offset_to_number, &entry);
    if (foundEntry)
    {
        if (foundEntry->dirNumber == dirn)
        {
            return 1;
        }
        else
        {
            TIFFWarningExtR(tif, "_TIFFCheckDirNumberAndOffset",
                            "TIFF directory %d has IFD looping to directory %u "
                            "at offset 0x%" PRIx64 " (%" PRIu64 ")",
                            (int)dirn - 1, foundEntry->dirNumber, diroff,
                            diroff);
            return 0;
        }
    }

    /* Check if offset of an IFD has been changed and update offset of that IFD
     * number. */
    foundEntry = (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
        tif->tif_map_dir_number_to_offset, &entry);
    if (foundEntry)
    {
        if (foundEntry->offset != diroff)
        {
            TIFFOffsetAndDirNumber entryOld;
            entryOld.offset = foundEntry->offset;
            entryOld.dirNumber = dirn;
            /* We must remove first from tif_map_dir_number_to_offset as the */
            /* entry is owned (and thus freed) by */
            /* tif_map_dir_offset_to_number */
            TIFFOffsetAndDirNumber *foundEntryOld =
                (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
                    tif->tif_map_dir_number_to_offset, &entryOld);
            if (foundEntryOld)
            {
                TIFFHashSetRemove(tif->tif_map_dir_number_to_offset,
                                  foundEntryOld);
            }
            foundEntryOld = (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
                tif->tif_map_dir_offset_to_number, &entryOld);
            if (foundEntryOld)
            {
                TIFFHashSetRemove(tif->tif_map_dir_offset_to_number,
                                  foundEntryOld);
            }

            TIFFOffsetAndDirNumber *entryPtr = (TIFFOffsetAndDirNumber *)malloc(
                sizeof(TIFFOffsetAndDirNumber));
            if (entryPtr == NULL)
            {
                return 0;
            }

            /* Add IFD offset and dirn to IFD directory list */
            *entryPtr = entry;

            if (!TIFFHashSetInsert(tif->tif_map_dir_offset_to_number, entryPtr))
            {
                TIFFErrorExtR(
                    tif, "_TIFFCheckDirNumberAndOffset",
                    "Insertion in tif_map_dir_offset_to_number failed");
                return 0;
            }
            if (!TIFFHashSetInsert(tif->tif_map_dir_number_to_offset, entryPtr))
            {
                TIFFErrorExtR(
                    tif, "_TIFFCheckDirNumberAndOffset",
                    "Insertion in tif_map_dir_number_to_offset failed");
                return 0;
            }
        }
        return 1;
    }

    /* Arbitrary (hopefully big enough) limit */
    if (TIFFHashSetSize(tif->tif_map_dir_offset_to_number) >=
        TIFF_MAX_DIR_COUNT)
    {
        TIFFErrorExtR(tif, "_TIFFCheckDirNumberAndOffset",
                      "Cannot handle more than %u TIFF directories",
                      TIFF_MAX_DIR_COUNT);
        return 0;
    }

    TIFFOffsetAndDirNumber *entryPtr =
        (TIFFOffsetAndDirNumber *)malloc(sizeof(TIFFOffsetAndDirNumber));
    if (entryPtr == NULL)
    {
        TIFFErrorExtR(tif, "_TIFFCheckDirNumberAndOffset",
                      "malloc(sizeof(TIFFOffsetAndDirNumber)) failed");
        return 0;
    }

    /* Add IFD offset and dirn to IFD directory list */
    *entryPtr = entry;

    if (!TIFFHashSetInsert(tif->tif_map_dir_offset_to_number, entryPtr))
    {
        TIFFErrorExtR(tif, "_TIFFCheckDirNumberAndOffset",
                      "Insertion in tif_map_dir_offset_to_number failed");
        return 0;
    }
    if (!TIFFHashSetInsert(tif->tif_map_dir_number_to_offset, entryPtr))
    {
        TIFFErrorExtR(tif, "_TIFFCheckDirNumberAndOffset",
                      "Insertion in tif_map_dir_number_to_offset failed");
        return 0;
    }

    return 1;
} /* --- _TIFFCheckDirNumberAndOffset() ---*/

/*
 * Retrieve the matching IFD directory number of a given IFD offset
 * from the list of directories already seen.
 * Returns 1 if the offset was in the list and the directory number
 * can be returned.
 * Otherwise returns 0 or if an error occurred.
 */
int _TIFFGetDirNumberFromOffset(TIFF *tif, uint64_t diroff, tdir_t *dirn)
{
    if (diroff == 0) /* no more directories */
        return 0;

    /* Check if offset is already in the list and return matching directory
     * number. Otherwise update IFD list using TIFFNumberOfDirectories() and
     * search again in IFD list.
     */
    if (tif->tif_map_dir_offset_to_number == NULL)
        return 0;
    TIFFOffsetAndDirNumber entry;
    entry.offset = diroff;
    entry.dirNumber = 0; /* not used */

    TIFFOffsetAndDirNumber *foundEntry =
        (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
            tif->tif_map_dir_offset_to_number, &entry);
    if (foundEntry)
    {
        *dirn = foundEntry->dirNumber;
        return 1;
    }

    /* This updates the directory list for all main-IFDs in the file. */
    TIFFNumberOfDirectories(tif);

    foundEntry = (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
        tif->tif_map_dir_offset_to_number, &entry);
    if (foundEntry)
    {
        *dirn = foundEntry->dirNumber;
        return 1;
    }

    return 0;
} /*--- _TIFFGetDirNumberFromOffset() ---*/

/*
 * Retrieve the matching IFD directory offset of a given IFD number
 * from the list of directories already seen.
 * Returns 1 if the offset was in the list of already seen IFDs and the
 * directory offset can be returned. The directory list is not updated.
 * Otherwise returns 0 or if an error occurred.
 */
int _TIFFGetOffsetFromDirNumber(TIFF *tif, tdir_t dirn, uint64_t *diroff)
{

    if (tif->tif_map_dir_number_to_offset == NULL)
        return 0;
    TIFFOffsetAndDirNumber entry;
    entry.offset = 0; /* not used */
    entry.dirNumber = dirn;

    TIFFOffsetAndDirNumber *foundEntry =
        (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
            tif->tif_map_dir_number_to_offset, &entry);
    if (foundEntry)
    {
        *diroff = foundEntry->offset;
        return 1;
    }

    return 0;
} /*--- _TIFFGetOffsetFromDirNumber() ---*/

/*
 * Remove an entry from the directory list of already seen directories
 * by directory offset.
 * If an entry is to be removed from the list, it is also okay if the entry
 * is not in the list or the list does not exist.
 */
int _TIFFRemoveEntryFromDirectoryListByOffset(TIFF *tif, uint64_t diroff)
{
    if (tif->tif_map_dir_offset_to_number == NULL)
        return 1;

    TIFFOffsetAndDirNumber entryOld;
    entryOld.offset = diroff;
    entryOld.dirNumber = 0;
    /* We must remove first from tif_map_dir_number_to_offset as the
     * entry is owned (and thus freed) by tif_map_dir_offset_to_number.
     * However, we need firstly to find the directory number from offset. */

    TIFFOffsetAndDirNumber *foundEntryOldOff =
        (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
            tif->tif_map_dir_offset_to_number, &entryOld);
    if (foundEntryOldOff)
    {
        entryOld.dirNumber = foundEntryOldOff->dirNumber;
        if (tif->tif_map_dir_number_to_offset != NULL)
        {
            TIFFOffsetAndDirNumber *foundEntryOldDir =
                (TIFFOffsetAndDirNumber *)TIFFHashSetLookup(
                    tif->tif_map_dir_number_to_offset, &entryOld);
            if (foundEntryOldDir)
            {
                TIFFHashSetRemove(tif->tif_map_dir_number_to_offset,
                                  foundEntryOldDir);
                TIFFHashSetRemove(tif->tif_map_dir_offset_to_number,
                                  foundEntryOldOff);
                return 1;
            }
        }
        else
        {
            TIFFErrorExtR(tif, "_TIFFRemoveEntryFromDirectoryListByOffset",
                          "Unexpectedly tif_map_dir_number_to_offset is "
                          "missing but tif_map_dir_offset_to_number exists.");
            return 0;
        }
    }
    return 1;
} /*--- _TIFFRemoveEntryFromDirectoryListByOffset() ---*/

/*
 * Check the count field of a directory entry against a known value.  The
 * caller is expected to skip/ignore the tag if there is a mismatch.
 */
static int CheckDirCount(TIFF *tif, TIFFDirEntry *dir, uint32_t count)
{
    if ((uint64_t)count > dir->tdir_count)
    {
        const TIFFField *fip = TIFFFieldWithTag(tif, dir->tdir_tag);
        TIFFWarningExtR(tif, tif->tif_name,
                        "incorrect count for field \"%s\" (%" PRIu64
                        ", expecting %" PRIu32 "); tag ignored",
                        fip ? fip->field_name : "unknown tagname",
                        dir->tdir_count, count);
        return (0);
    }
    else if ((uint64_t)count < dir->tdir_count)
    {
        const TIFFField *fip = TIFFFieldWithTag(tif, dir->tdir_tag);
        TIFFWarningExtR(tif, tif->tif_name,
                        "incorrect count for field \"%s\" (%" PRIu64
                        ", expecting %" PRIu32 "); tag trimmed",
                        fip ? fip->field_name : "unknown tagname",
                        dir->tdir_count, count);
        dir->tdir_count = count;
        return (1);
    }
    return (1);
}

/*
 * Read IFD structure from the specified offset. If the pointer to
 * nextdiroff variable has been specified, read it too. Function returns a
 * number of fields in the directory or 0 if failed.
 */
static uint16_t TIFFFetchDirectory(TIFF *tif, uint64_t diroff,
                                   TIFFDirEntry **pdir, uint64_t *nextdiroff)
{
    static const char module[] = "TIFFFetchDirectory";

    void *origdir;
    uint16_t dircount16;
    uint32_t dirsize;
    TIFFDirEntry *dir;
    uint8_t *ma;
    TIFFDirEntry *mb;
    uint16_t n;

    assert(pdir);

    tif->tif_diroff = diroff;
    if (nextdiroff)
        *nextdiroff = 0;
    if (!isMapped(tif))
    {
        if (!SeekOK(tif, tif->tif_diroff))
        {
            TIFFErrorExtR(tif, module,
                          "%s: Seek error accessing TIFF directory",
                          tif->tif_name);
            return 0;
        }
        if (!(tif->tif_flags & TIFF_BIGTIFF))
        {
            if (!ReadOK(tif, &dircount16, sizeof(uint16_t)))
            {
                TIFFErrorExtR(tif, module,
                              "%s: Can not read TIFF directory count",
                              tif->tif_name);
                return 0;
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort(&dircount16);
            if (dircount16 > 4096)
            {
                TIFFErrorExtR(tif, module,
                              "Sanity check on directory count failed, this is "
                              "probably not a valid IFD offset");
                return 0;
            }
            dirsize = 12;
        }
        else
        {
            uint64_t dircount64;
            if (!ReadOK(tif, &dircount64, sizeof(uint64_t)))
            {
                TIFFErrorExtR(tif, module,
                              "%s: Can not read TIFF directory count",
                              tif->tif_name);
                return 0;
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&dircount64);
            if (dircount64 > 4096)
            {
                TIFFErrorExtR(tif, module,
                              "Sanity check on directory count failed, this is "
                              "probably not a valid IFD offset");
                return 0;
            }
            dircount16 = (uint16_t)dircount64;
            dirsize = 20;
        }
        origdir = _TIFFCheckMalloc(tif, dircount16, dirsize,
                                   "to read TIFF directory");
        if (origdir == NULL)
            return 0;
        if (!ReadOK(tif, origdir, (tmsize_t)(dircount16 * dirsize)))
        {
            TIFFErrorExtR(tif, module, "%.100s: Can not read TIFF directory",
                          tif->tif_name);
            _TIFFfreeExt(tif, origdir);
            return 0;
        }
        /*
         * Read offset to next directory for sequential scans if
         * needed.
         */
        if (nextdiroff)
        {
            if (!(tif->tif_flags & TIFF_BIGTIFF))
            {
                uint32_t nextdiroff32;
                if (!ReadOK(tif, &nextdiroff32, sizeof(uint32_t)))
                    nextdiroff32 = 0;
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(&nextdiroff32);
                *nextdiroff = nextdiroff32;
            }
            else
            {
                if (!ReadOK(tif, nextdiroff, sizeof(uint64_t)))
                    *nextdiroff = 0;
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(nextdiroff);
            }
        }
    }
    else
    {
        tmsize_t m;
        tmsize_t off;
        if (tif->tif_diroff > (uint64_t)INT64_MAX)
        {
            TIFFErrorExtR(tif, module, "Can not read TIFF directory count");
            return (0);
        }
        off = (tmsize_t)tif->tif_diroff;

        /*
         * Check for integer overflow when validating the dir_off,
         * otherwise a very high offset may cause an OOB read and
         * crash the client. Make two comparisons instead of
         *
         *  off + sizeof(uint16_t) > tif->tif_size
         *
         * to avoid overflow.
         */
        if (!(tif->tif_flags & TIFF_BIGTIFF))
        {
            m = off + sizeof(uint16_t);
            if ((m < off) || (m < (tmsize_t)sizeof(uint16_t)) ||
                (m > tif->tif_size))
            {
                TIFFErrorExtR(tif, module, "Can not read TIFF directory count");
                return 0;
            }
            else
            {
                _TIFFmemcpy(&dircount16, tif->tif_base + off, sizeof(uint16_t));
            }
            off += sizeof(uint16_t);
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort(&dircount16);
            if (dircount16 > 4096)
            {
                TIFFErrorExtR(tif, module,
                              "Sanity check on directory count failed, this is "
                              "probably not a valid IFD offset");
                return 0;
            }
            dirsize = 12;
        }
        else
        {
            uint64_t dircount64;
            m = off + sizeof(uint64_t);
            if ((m < off) || (m < (tmsize_t)sizeof(uint64_t)) ||
                (m > tif->tif_size))
            {
                TIFFErrorExtR(tif, module, "Can not read TIFF directory count");
                return 0;
            }
            else
            {
                _TIFFmemcpy(&dircount64, tif->tif_base + off, sizeof(uint64_t));
            }
            off += sizeof(uint64_t);
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&dircount64);
            if (dircount64 > 4096)
            {
                TIFFErrorExtR(tif, module,
                              "Sanity check on directory count failed, this is "
                              "probably not a valid IFD offset");
                return 0;
            }
            dircount16 = (uint16_t)dircount64;
            dirsize = 20;
        }
        if (dircount16 == 0)
        {
            TIFFErrorExtR(tif, module,
                          "Sanity check on directory count failed, zero tag "
                          "directories not supported");
            return 0;
        }
        origdir = _TIFFCheckMalloc(tif, dircount16, dirsize,
                                   "to read TIFF directory");
        if (origdir == NULL)
            return 0;
        m = off + dircount16 * dirsize;
        if ((m < off) || (m < (tmsize_t)(dircount16 * dirsize)) ||
            (m > tif->tif_size))
        {
            TIFFErrorExtR(tif, module, "Can not read TIFF directory");
            _TIFFfreeExt(tif, origdir);
            return 0;
        }
        else
        {
            _TIFFmemcpy(origdir, tif->tif_base + off, dircount16 * dirsize);
        }
        if (nextdiroff)
        {
            off += dircount16 * dirsize;
            if (!(tif->tif_flags & TIFF_BIGTIFF))
            {
                uint32_t nextdiroff32;
                m = off + sizeof(uint32_t);
                if ((m < off) || (m < (tmsize_t)sizeof(uint32_t)) ||
                    (m > tif->tif_size))
                    nextdiroff32 = 0;
                else
                    _TIFFmemcpy(&nextdiroff32, tif->tif_base + off,
                                sizeof(uint32_t));
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong(&nextdiroff32);
                *nextdiroff = nextdiroff32;
            }
            else
            {
                m = off + sizeof(uint64_t);
                if ((m < off) || (m < (tmsize_t)sizeof(uint64_t)) ||
                    (m > tif->tif_size))
                    *nextdiroff = 0;
                else
                    _TIFFmemcpy(nextdiroff, tif->tif_base + off,
                                sizeof(uint64_t));
                if (tif->tif_flags & TIFF_SWAB)
                    TIFFSwabLong8(nextdiroff);
            }
        }
    }
    dir = (TIFFDirEntry *)_TIFFCheckMalloc(
        tif, dircount16, sizeof(TIFFDirEntry), "to read TIFF directory");
    if (dir == 0)
    {
        _TIFFfreeExt(tif, origdir);
        return 0;
    }
    ma = (uint8_t *)origdir;
    mb = dir;
    for (n = 0; n < dircount16; n++)
    {
        mb->tdir_ignore = FALSE;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabShort((uint16_t *)ma);
        mb->tdir_tag = *(uint16_t *)ma;
        ma += sizeof(uint16_t);
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabShort((uint16_t *)ma);
        mb->tdir_type = *(uint16_t *)ma;
        ma += sizeof(uint16_t);
        if (!(tif->tif_flags & TIFF_BIGTIFF))
        {
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong((uint32_t *)ma);
            mb->tdir_count = (uint64_t)(*(uint32_t *)ma);
            ma += sizeof(uint32_t);
            mb->tdir_offset.toff_long8 = 0;
            *(uint32_t *)(&mb->tdir_offset) = *(uint32_t *)ma;
            ma += sizeof(uint32_t);
        }
        else
        {
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8((uint64_t *)ma);
            mb->tdir_count = TIFFReadUInt64(ma);
            ma += sizeof(uint64_t);
            mb->tdir_offset.toff_long8 = TIFFReadUInt64(ma);
            ma += sizeof(uint64_t);
        }
        mb++;
    }
    _TIFFfreeExt(tif, origdir);
    *pdir = dir;
    return dircount16;
}

/*
 * Fetch a tag that is not handled by special case code.
 */
static int TIFFFetchNormalTag(TIFF *tif, TIFFDirEntry *dp, int recover)
{
    static const char module[] = "TIFFFetchNormalTag";
    enum TIFFReadDirEntryErr err;
    uint32_t fii;
    const TIFFField *fip = NULL;
    TIFFReadDirectoryFindFieldInfo(tif, dp->tdir_tag, &fii);
    if (fii == FAILED_FII)
    {
        TIFFErrorExtR(tif, "TIFFFetchNormalTag",
                      "No definition found for tag %" PRIu16, dp->tdir_tag);
        return 0;
    }
    fip = tif->tif_fields[fii];
    assert(fip != NULL); /* should not happen */
    assert(fip->set_field_type !=
           TIFF_SETGET_OTHER); /* if so, we shouldn't arrive here but deal with
                                  this in specialized code */
    assert(fip->set_field_type !=
           TIFF_SETGET_INT); /* if so, we shouldn't arrive here as this is only
                                the case for pseudo-tags */
    err = TIFFReadDirEntryErrOk;
    switch (fip->set_field_type)
    {
        case TIFF_SETGET_UNDEFINED:
            TIFFErrorExtR(
                tif, "TIFFFetchNormalTag",
                "Defined set_field_type of custom tag %u (%s) is "
                "TIFF_SETGET_UNDEFINED and thus tag is not read from file",
                fip->field_tag, fip->field_name);
            break;
        case TIFF_SETGET_ASCII:
        {
            uint8_t *data;
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntryByteArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                size_t mb = 0;
                int n;
                if (data != NULL)
                {
                    if (dp->tdir_count > 0 && data[dp->tdir_count - 1] == 0)
                    {
                        /* optimization: if data is known to be 0 terminated, we
                         * can use strlen() */
                        mb = strlen((const char *)data);
                    }
                    else
                    {
                        /* general case. equivalent to non-portable */
                        /* mb = strnlen((const char*)data,
                         * (uint32_t)dp->tdir_count); */
                        uint8_t *ma = data;
                        while (mb < (uint32_t)dp->tdir_count)
                        {
                            if (*ma == 0)
                                break;
                            ma++;
                            mb++;
                        }
                    }
                }
                if (mb + 1 < (uint32_t)dp->tdir_count)
                    TIFFWarningExtR(
                        tif, module,
                        "ASCII value for tag \"%s\" contains null byte in "
                        "value; value incorrectly truncated during reading due "
                        "to implementation limitations",
                        fip->field_name);
                else if (mb + 1 > (uint32_t)dp->tdir_count)
                {
                    uint8_t *o;
                    TIFFWarningExtR(
                        tif, module,
                        "ASCII value for tag \"%s\" does not end in null byte",
                        fip->field_name);
                    /* TIFFReadDirEntryArrayWithLimit() ensures this can't be
                     * larger than MAX_SIZE_TAG_DATA */
                    assert((uint32_t)dp->tdir_count + 1 == dp->tdir_count + 1);
                    o = _TIFFmallocExt(tif, (uint32_t)dp->tdir_count + 1);
                    if (o == NULL)
                    {
                        if (data != NULL)
                            _TIFFfreeExt(tif, data);
                        return (0);
                    }
                    if (dp->tdir_count > 0)
                    {
                        _TIFFmemcpy(o, data, (uint32_t)dp->tdir_count);
                    }
                    o[(uint32_t)dp->tdir_count] = 0;
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    data = o;
                }
                n = TIFFSetField(tif, dp->tdir_tag, data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!n)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_UINT8:
        {
            uint8_t data = 0;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntryByte(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_SINT8:
        {
            int8_t data = 0;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntrySbyte(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_UINT16:
        {
            uint16_t data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntryShort(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_SINT16:
        {
            int16_t data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntrySshort(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_UINT32:
        {
            uint32_t data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntryLong(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_SINT32:
        {
            int32_t data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntrySlong(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_UINT64:
        {
            uint64_t data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntryLong8(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_SINT64:
        {
            int64_t data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntrySlong8(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_FLOAT:
        {
            float data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntryFloat(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_DOUBLE:
        {
            double data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntryDouble(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_IFD8:
        {
            uint64_t data;
            assert(fip->field_readcount == 1);
            assert(fip->field_passcount == 0);
            err = TIFFReadDirEntryIfd8(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                if (!TIFFSetField(tif, dp->tdir_tag, data))
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_UINT16_PAIR:
        {
            uint16_t *data;
            assert(fip->field_readcount == 2);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != 2)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected 2, "
                                "got %" PRIu64,
                                fip->field_name, dp->tdir_count);
                return (0);
            }
            err = TIFFReadDirEntryShortArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                assert(data); /* avoid CLang static Analyzer false positive */
                m = TIFFSetField(tif, dp->tdir_tag, data[0], data[1]);
                _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C0_UINT8:
        {
            uint8_t *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntryByteArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C0_SINT8:
        {
            int8_t *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntrySbyteArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C0_UINT16:
        {
            uint16_t *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntryShortArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C0_SINT16:
        {
            int16_t *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntrySshortArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C0_UINT32:
        {
            uint32_t *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntryLongArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C0_SINT32:
        {
            int32_t *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntrySlongArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C0_UINT64:
        {
            uint64_t *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntryLong8Array(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C0_SINT64:
        {
            int64_t *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntrySlong8Array(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C0_FLOAT:
        {
            float *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntryFloatArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        /*--: Rational2Double: Extend for Double Arrays and Rational-Arrays read
         * into Double-Arrays. */
        case TIFF_SETGET_C0_DOUBLE:
        {
            double *data;
            assert(fip->field_readcount >= 1);
            assert(fip->field_passcount == 0);
            if (dp->tdir_count != (uint64_t)fip->field_readcount)
            {
                TIFFWarningExtR(tif, module,
                                "incorrect count for field \"%s\", expected "
                                "%d, got %" PRIu64,
                                fip->field_name, (int)fip->field_readcount,
                                dp->tdir_count);
                return (0);
            }
            else
            {
                err = TIFFReadDirEntryDoubleArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag, data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_ASCII:
        {
            uint8_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntryByteArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    if (data != 0 && dp->tdir_count > 0 &&
                        data[dp->tdir_count - 1] != '\0')
                    {
                        TIFFWarningExtR(
                            tif, module,
                            "ASCII value for tag \"%s\" does not end in null "
                            "byte. Forcing it to be null",
                            fip->field_name);
                        data[dp->tdir_count - 1] = '\0';
                    }
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_UINT8:
        {
            uint8_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntryByteArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_SINT8:
        {
            int8_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntrySbyteArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_UINT16:
        {
            uint16_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntryShortArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_SINT16:
        {
            int16_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntrySshortArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_UINT32:
        {
            uint32_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntryLongArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_SINT32:
        {
            int32_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntrySlongArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_UINT64:
        {
            uint64_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntryLong8Array(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_SINT64:
        {
            int64_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntrySlong8Array(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_FLOAT:
        {
            float *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntryFloatArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_DOUBLE:
        {
            double *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntryDoubleArray(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C16_IFD8:
        {
            uint64_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE);
            assert(fip->field_passcount == 1);
            if (dp->tdir_count > 0xFFFF)
                err = TIFFReadDirEntryErrCount;
            else
            {
                err = TIFFReadDirEntryIfd8Array(tif, dp, &data);
                if (err == TIFFReadDirEntryErrOk)
                {
                    int m;
                    m = TIFFSetField(tif, dp->tdir_tag,
                                     (uint16_t)(dp->tdir_count), data);
                    if (data != 0)
                        _TIFFfreeExt(tif, data);
                    if (!m)
                        return (0);
                }
            }
        }
        break;
        case TIFF_SETGET_C32_ASCII:
        {
            uint8_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntryByteArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                if (data != 0 && dp->tdir_count > 0 &&
                    data[dp->tdir_count - 1] != '\0')
                {
                    TIFFWarningExtR(tif, module,
                                    "ASCII value for tag \"%s\" does not end "
                                    "in null byte. Forcing it to be null",
                                    fip->field_name);
                    data[dp->tdir_count - 1] = '\0';
                }
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_UINT8:
        {
            uint8_t *data;
            uint32_t count = 0;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            if (fip->field_tag == TIFFTAG_RICHTIFFIPTC &&
                dp->tdir_type == TIFF_LONG)
            {
                /* Adobe's software (wrongly) writes RichTIFFIPTC tag with
                 * data type LONG instead of UNDEFINED. Work around this
                 * frequently found issue */
                void *origdata;
                err = TIFFReadDirEntryArray(tif, dp, &count, 4, &origdata);
                if ((err != TIFFReadDirEntryErrOk) || (origdata == 0))
                {
                    data = NULL;
                }
                else
                {
                    if (tif->tif_flags & TIFF_SWAB)
                        TIFFSwabArrayOfLong((uint32_t *)origdata, count);
                    data = (uint8_t *)origdata;
                    count = (uint32_t)(count * 4);
                }
            }
            else
            {
                err = TIFFReadDirEntryByteArray(tif, dp, &data);
                count = (uint32_t)(dp->tdir_count);
            }
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, count, data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_SINT8:
        {
            int8_t *data = NULL;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntrySbyteArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_UINT16:
        {
            uint16_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntryShortArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_SINT16:
        {
            int16_t *data = NULL;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntrySshortArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_UINT32:
        {
            uint32_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntryLongArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_SINT32:
        {
            int32_t *data = NULL;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntrySlongArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_UINT64:
        {
            uint64_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntryLong8Array(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_SINT64:
        {
            int64_t *data = NULL;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntrySlong8Array(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_FLOAT:
        {
            float *data;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntryFloatArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_DOUBLE:
        {
            double *data;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntryDoubleArray(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        case TIFF_SETGET_C32_IFD8:
        {
            uint64_t *data;
            assert(fip->field_readcount == TIFF_VARIABLE2);
            assert(fip->field_passcount == 1);
            err = TIFFReadDirEntryIfd8Array(tif, dp, &data);
            if (err == TIFFReadDirEntryErrOk)
            {
                int m;
                m = TIFFSetField(tif, dp->tdir_tag, (uint32_t)(dp->tdir_count),
                                 data);
                if (data != 0)
                    _TIFFfreeExt(tif, data);
                if (!m)
                    return (0);
            }
        }
        break;
        default:
            assert(0); /* we should never get here */
            break;
    }
    if (err != TIFFReadDirEntryErrOk)
    {
        TIFFReadDirEntryOutputErr(tif, err, module, fip->field_name, recover);
        return (0);
    }
    return (1);
}

/*
 * Fetch a set of offsets or lengths.
 * While this routine says "strips", in fact it's also used for tiles.
 */
static int TIFFFetchStripThing(TIFF *tif, TIFFDirEntry *dir, uint32_t nstrips,
                               uint64_t **lpp)
{
    static const char module[] = "TIFFFetchStripThing";
    enum TIFFReadDirEntryErr err;
    uint64_t *data;
    err = TIFFReadDirEntryLong8ArrayWithLimit(tif, dir, &data, nstrips);
    if (err != TIFFReadDirEntryErrOk)
    {
        const TIFFField *fip = TIFFFieldWithTag(tif, dir->tdir_tag);
        TIFFReadDirEntryOutputErr(tif, err, module,
                                  fip ? fip->field_name : "unknown tagname", 0);
        return (0);
    }
    if (dir->tdir_count < (uint64_t)nstrips)
    {
        uint64_t *resizeddata;
        const TIFFField *fip = TIFFFieldWithTag(tif, dir->tdir_tag);
        const char *pszMax = getenv("LIBTIFF_STRILE_ARRAY_MAX_RESIZE_COUNT");
        uint32_t max_nstrips = 1000000;
        if (pszMax)
            max_nstrips = (uint32_t)atoi(pszMax);
        TIFFReadDirEntryOutputErr(tif, TIFFReadDirEntryErrCount, module,
                                  fip ? fip->field_name : "unknown tagname",
                                  (nstrips <= max_nstrips));

        if (nstrips > max_nstrips)
        {
            _TIFFfreeExt(tif, data);
            return (0);
        }

        resizeddata = (uint64_t *)_TIFFCheckMalloc(
            tif, nstrips, sizeof(uint64_t), "for strip array");
        if (resizeddata == 0)
        {
            _TIFFfreeExt(tif, data);
            return (0);
        }
        if (dir->tdir_count)
            _TIFFmemcpy(resizeddata, data,
                        (uint32_t)dir->tdir_count * sizeof(uint64_t));
        _TIFFmemset(resizeddata + (uint32_t)dir->tdir_count, 0,
                    (nstrips - (uint32_t)dir->tdir_count) * sizeof(uint64_t));
        _TIFFfreeExt(tif, data);
        data = resizeddata;
    }
    *lpp = data;
    return (1);
}

/*
 * Fetch and set the SubjectDistance EXIF tag.
 */
static int TIFFFetchSubjectDistance(TIFF *tif, TIFFDirEntry *dir)
{
    static const char module[] = "TIFFFetchSubjectDistance";
    enum TIFFReadDirEntryErr err;
    UInt64Aligned_t m;
    m.l = 0;
    assert(sizeof(double) == 8);
    assert(sizeof(uint64_t) == 8);
    assert(sizeof(uint32_t) == 4);
    if (dir->tdir_count != 1)
        err = TIFFReadDirEntryErrCount;
    else if (dir->tdir_type != TIFF_RATIONAL)
        err = TIFFReadDirEntryErrType;
    else
    {
        if (!(tif->tif_flags & TIFF_BIGTIFF))
        {
            uint32_t offset;
            offset = *(uint32_t *)(&dir->tdir_offset);
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong(&offset);
            err = TIFFReadDirEntryData(tif, offset, 8, m.i);
        }
        else
        {
            m.l = dir->tdir_offset.toff_long8;
            err = TIFFReadDirEntryErrOk;
        }
    }
    if (err == TIFFReadDirEntryErrOk)
    {
        double n;
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabArrayOfLong(m.i, 2);
        if (m.i[0] == 0)
            n = 0.0;
        else if (m.i[0] == 0xFFFFFFFF || m.i[1] == 0)
            /*
             * XXX: Numerator 0xFFFFFFFF means that we have infinite
             * distance. Indicate that with a negative floating point
             * SubjectDistance value.
             */
            n = -1.0;
        else
            n = (double)m.i[0] / (double)m.i[1];
        return (TIFFSetField(tif, dir->tdir_tag, n));
    }
    else
    {
        TIFFReadDirEntryOutputErr(tif, err, module, "SubjectDistance", TRUE);
        return (0);
    }
}

static void allocChoppedUpStripArrays(TIFF *tif, uint32_t nstrips,
                                      uint64_t stripbytes,
                                      uint32_t rowsperstrip)
{
    TIFFDirectory *td = &tif->tif_dir;
    uint64_t bytecount;
    uint64_t offset;
    uint64_t last_offset;
    uint64_t last_bytecount;
    uint32_t i;
    uint64_t *newcounts;
    uint64_t *newoffsets;

    offset = TIFFGetStrileOffset(tif, 0);
    last_offset = TIFFGetStrileOffset(tif, td->td_nstrips - 1);
    last_bytecount = TIFFGetStrileByteCount(tif, td->td_nstrips - 1);
    if (last_offset > UINT64_MAX - last_bytecount ||
        last_offset + last_bytecount < offset)
    {
        return;
    }
    bytecount = last_offset + last_bytecount - offset;

    newcounts =
        (uint64_t *)_TIFFCheckMalloc(tif, nstrips, sizeof(uint64_t),
                                     "for chopped \"StripByteCounts\" array");
    newoffsets = (uint64_t *)_TIFFCheckMalloc(
        tif, nstrips, sizeof(uint64_t), "for chopped \"StripOffsets\" array");
    if (newcounts == NULL || newoffsets == NULL)
    {
        /*
         * Unable to allocate new strip information, give up and use
         * the original one strip information.
         */
        if (newcounts != NULL)
            _TIFFfreeExt(tif, newcounts);
        if (newoffsets != NULL)
            _TIFFfreeExt(tif, newoffsets);
        return;
    }

    /*
     * Fill the strip information arrays with new bytecounts and offsets
     * that reflect the broken-up format.
     */
    for (i = 0; i < nstrips; i++)
    {
        if (stripbytes > bytecount)
            stripbytes = bytecount;
        newcounts[i] = stripbytes;
        newoffsets[i] = stripbytes ? offset : 0;
        offset += stripbytes;
        bytecount -= stripbytes;
    }

    /*
     * Replace old single strip info with multi-strip info.
     */
    td->td_stripsperimage = td->td_nstrips = nstrips;
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

    _TIFFfreeExt(tif, td->td_stripbytecount_p);
    _TIFFfreeExt(tif, td->td_stripoffset_p);
    td->td_stripbytecount_p = newcounts;
    td->td_stripoffset_p = newoffsets;
#ifdef STRIPBYTECOUNTSORTED_UNUSED
    td->td_stripbytecountsorted = 1;
#endif
    tif->tif_flags |= TIFF_CHOPPEDUPARRAYS;
}

/*
 * Replace a single strip (tile) of uncompressed data by multiple strips
 * (tiles), each approximately STRIP_SIZE_DEFAULT bytes. This is useful for
 * dealing with large images or for dealing with machines with a limited
 * amount memory.
 */
static void ChopUpSingleUncompressedStrip(TIFF *tif)
{
    register TIFFDirectory *td = &tif->tif_dir;
    uint64_t bytecount;
    uint64_t offset;
    uint32_t rowblock;
    uint64_t rowblockbytes;
    uint64_t stripbytes;
    uint32_t nstrips;
    uint32_t rowsperstrip;

    bytecount = TIFFGetStrileByteCount(tif, 0);
    /* On a newly created file, just re-opened to be filled, we */
    /* don't want strip chop to trigger as it is going to cause issues */
    /* later ( StripOffsets and StripByteCounts improperly filled) . */
    if (bytecount == 0 && tif->tif_mode != O_RDONLY)
        return;
    offset = TIFFGetStrileByteCount(tif, 0);
    assert(td->td_planarconfig == PLANARCONFIG_CONTIG);
    if ((td->td_photometric == PHOTOMETRIC_YCBCR) && (!isUpSampled(tif)))
        rowblock = td->td_ycbcrsubsampling[1];
    else
        rowblock = 1;
    rowblockbytes = TIFFVTileSize64(tif, rowblock);
    /*
     * Make the rows hold at least one scanline, but fill specified amount
     * of data if possible.
     */
    if (rowblockbytes > STRIP_SIZE_DEFAULT)
    {
        stripbytes = rowblockbytes;
        rowsperstrip = rowblock;
    }
    else if (rowblockbytes > 0)
    {
        uint32_t rowblocksperstrip;
        rowblocksperstrip = (uint32_t)(STRIP_SIZE_DEFAULT / rowblockbytes);
        rowsperstrip = rowblocksperstrip * rowblock;
        stripbytes = rowblocksperstrip * rowblockbytes;
    }
    else
        return;

    /*
     * never increase the number of rows per strip
     */
    if (rowsperstrip >= td->td_rowsperstrip)
        return;
    nstrips = TIFFhowmany_32(td->td_imagelength, rowsperstrip);
    if (nstrips == 0)
        return;

    /* If we are going to allocate a lot of memory, make sure that the */
    /* file is as big as needed */
    if (tif->tif_mode == O_RDONLY && nstrips > 1000000 &&
        (offset >= TIFFGetFileSize(tif) ||
         stripbytes > (TIFFGetFileSize(tif) - offset) / (nstrips - 1)))
    {
        return;
    }

    allocChoppedUpStripArrays(tif, nstrips, stripbytes, rowsperstrip);
}

/*
 * Replace a file with contiguous strips > 2 GB of uncompressed data by
 * multiple smaller strips. This is useful for
 * dealing with large images or for dealing with machines with a limited
 * amount memory.
 */
static void TryChopUpUncompressedBigTiff(TIFF *tif)
{
    TIFFDirectory *td = &tif->tif_dir;
    uint32_t rowblock;
    uint64_t rowblockbytes;
    uint32_t i;
    uint64_t stripsize;
    uint32_t rowblocksperstrip;
    uint32_t rowsperstrip;
    uint64_t stripbytes;
    uint32_t nstrips;

    stripsize = TIFFStripSize64(tif);

    assert(tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG);
    assert(tif->tif_dir.td_compression == COMPRESSION_NONE);
    assert((tif->tif_flags & (TIFF_STRIPCHOP | TIFF_ISTILED)) ==
           TIFF_STRIPCHOP);
    assert(stripsize > 0x7FFFFFFFUL);

    /* On a newly created file, just re-opened to be filled, we */
    /* don't want strip chop to trigger as it is going to cause issues */
    /* later ( StripOffsets and StripByteCounts improperly filled) . */
    if (TIFFGetStrileByteCount(tif, 0) == 0 && tif->tif_mode != O_RDONLY)
        return;

    if ((td->td_photometric == PHOTOMETRIC_YCBCR) && (!isUpSampled(tif)))
        rowblock = td->td_ycbcrsubsampling[1];
    else
        rowblock = 1;
    rowblockbytes = TIFFVStripSize64(tif, rowblock);
    if (rowblockbytes == 0 || rowblockbytes > 0x7FFFFFFFUL)
    {
        /* In case of file with gigantic width */
        return;
    }

    /* Check that the strips are contiguous and of the expected size */
    for (i = 0; i < td->td_nstrips; i++)
    {
        if (i == td->td_nstrips - 1)
        {
            if (TIFFGetStrileByteCount(tif, i) <
                TIFFVStripSize64(tif,
                                 td->td_imagelength - i * td->td_rowsperstrip))
            {
                return;
            }
        }
        else
        {
            if (TIFFGetStrileByteCount(tif, i) != stripsize)
            {
                return;
            }
            if (i > 0 && TIFFGetStrileOffset(tif, i) !=
                             TIFFGetStrileOffset(tif, i - 1) +
                                 TIFFGetStrileByteCount(tif, i - 1))
            {
                return;
            }
        }
    }

    /* Aim for 512 MB strips (that will still be manageable by 32 bit builds */
    rowblocksperstrip = (uint32_t)(512 * 1024 * 1024 / rowblockbytes);
    if (rowblocksperstrip == 0)
        rowblocksperstrip = 1;
    rowsperstrip = rowblocksperstrip * rowblock;
    stripbytes = rowblocksperstrip * rowblockbytes;
    assert(stripbytes <= 0x7FFFFFFFUL);

    nstrips = TIFFhowmany_32(td->td_imagelength, rowsperstrip);
    if (nstrips == 0)
        return;

    /* If we are going to allocate a lot of memory, make sure that the */
    /* file is as big as needed */
    if (tif->tif_mode == O_RDONLY && nstrips > 1000000)
    {
        uint64_t last_offset = TIFFGetStrileOffset(tif, td->td_nstrips - 1);
        uint64_t filesize = TIFFGetFileSize(tif);
        uint64_t last_bytecount =
            TIFFGetStrileByteCount(tif, td->td_nstrips - 1);
        if (last_offset > filesize || last_bytecount > filesize - last_offset)
        {
            return;
        }
    }

    allocChoppedUpStripArrays(tif, nstrips, stripbytes, rowsperstrip);
}

TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
static uint64_t _TIFFUnsanitizedAddUInt64AndInt(uint64_t a, int b)
{
    return a + b;
}

/* Read the value of [Strip|Tile]Offset or [Strip|Tile]ByteCount around
 * strip/tile of number strile. Also fetch the neighbouring values using a
 * 4096 byte page size.
 */
static int _TIFFPartialReadStripArray(TIFF *tif, TIFFDirEntry *dirent,
                                      int strile, uint64_t *panVals)
{
    static const char module[] = "_TIFFPartialReadStripArray";
#define IO_CACHE_PAGE_SIZE 4096

    size_t sizeofval;
    const int bSwab = (tif->tif_flags & TIFF_SWAB) != 0;
    int sizeofvalint;
    uint64_t nBaseOffset;
    uint64_t nOffset;
    uint64_t nOffsetStartPage;
    uint64_t nOffsetEndPage;
    tmsize_t nToRead;
    tmsize_t nRead;
    uint64_t nLastStripOffset;
    int iStartBefore;
    int i;
    const uint32_t arraySize = tif->tif_dir.td_stripoffsetbyteallocsize;
    unsigned char buffer[2 * IO_CACHE_PAGE_SIZE];

    assert(dirent->tdir_count > 4);

    if (dirent->tdir_type == TIFF_SHORT)
    {
        sizeofval = sizeof(uint16_t);
    }
    else if (dirent->tdir_type == TIFF_LONG)
    {
        sizeofval = sizeof(uint32_t);
    }
    else if (dirent->tdir_type == TIFF_LONG8)
    {
        sizeofval = sizeof(uint64_t);
    }
    else if (dirent->tdir_type == TIFF_SLONG8)
    {
        /* Non conformant but used by some images as in */
        /* https://github.com/OSGeo/gdal/issues/2165 */
        sizeofval = sizeof(int64_t);
    }
    else
    {
        TIFFErrorExtR(tif, module,
                      "Invalid type for [Strip|Tile][Offset/ByteCount] tag");
        panVals[strile] = 0;
        return 0;
    }
    sizeofvalint = (int)(sizeofval);

    if (tif->tif_flags & TIFF_BIGTIFF)
    {
        uint64_t offset = dirent->tdir_offset.toff_long8;
        if (bSwab)
            TIFFSwabLong8(&offset);
        nBaseOffset = offset;
    }
    else
    {
        uint32_t offset = dirent->tdir_offset.toff_long;
        if (bSwab)
            TIFFSwabLong(&offset);
        nBaseOffset = offset;
    }
    /* To avoid later unsigned integer overflows */
    if (nBaseOffset > (uint64_t)INT64_MAX)
    {
        TIFFErrorExtR(tif, module, "Cannot read offset/size for strile %d",
                      strile);
        panVals[strile] = 0;
        return 0;
    }
    nOffset = nBaseOffset + sizeofval * strile;
    nOffsetStartPage = (nOffset / IO_CACHE_PAGE_SIZE) * IO_CACHE_PAGE_SIZE;
    nOffsetEndPage = nOffsetStartPage + IO_CACHE_PAGE_SIZE;

    if (nOffset + sizeofval > nOffsetEndPage)
        nOffsetEndPage += IO_CACHE_PAGE_SIZE;
#undef IO_CACHE_PAGE_SIZE

    nLastStripOffset = nBaseOffset + arraySize * sizeofval;
    if (nLastStripOffset < nOffsetEndPage)
        nOffsetEndPage = nLastStripOffset;
    if (nOffsetStartPage >= nOffsetEndPage)
    {
        TIFFErrorExtR(tif, module, "Cannot read offset/size for strile %d",
                      strile);
        panVals[strile] = 0;
        return 0;
    }
    if (!SeekOK(tif, nOffsetStartPage))
    {
        panVals[strile] = 0;
        return 0;
    }

    nToRead = (tmsize_t)(nOffsetEndPage - nOffsetStartPage);
    nRead = TIFFReadFile(tif, buffer, nToRead);
    if (nRead < nToRead)
    {
        TIFFErrorExtR(tif, module,
                      "Cannot read offset/size for strile around ~%d", strile);
        return 0;
    }
    iStartBefore = -(int)((nOffset - nOffsetStartPage) / sizeofval);
    if (strile + iStartBefore < 0)
        iStartBefore = -strile;
    for (i = iStartBefore;
         (uint32_t)(strile + i) < arraySize &&
         _TIFFUnsanitizedAddUInt64AndInt(nOffset, (i + 1) * sizeofvalint) <=
             nOffsetEndPage;
         ++i)
    {
        if (dirent->tdir_type == TIFF_SHORT)
        {
            uint16_t val;
            memcpy(&val,
                   buffer + (nOffset - nOffsetStartPage) + i * sizeofvalint,
                   sizeof(val));
            if (bSwab)
                TIFFSwabShort(&val);
            panVals[strile + i] = val;
        }
        else if (dirent->tdir_type == TIFF_LONG)
        {
            uint32_t val;
            memcpy(&val,
                   buffer + (nOffset - nOffsetStartPage) + i * sizeofvalint,
                   sizeof(val));
            if (bSwab)
                TIFFSwabLong(&val);
            panVals[strile + i] = val;
        }
        else if (dirent->tdir_type == TIFF_LONG8)
        {
            uint64_t val;
            memcpy(&val,
                   buffer + (nOffset - nOffsetStartPage) + i * sizeofvalint,
                   sizeof(val));
            if (bSwab)
                TIFFSwabLong8(&val);
            panVals[strile + i] = val;
        }
        else /* if( dirent->tdir_type == TIFF_SLONG8 ) */
        {
            /* Non conformant data type */
            int64_t val;
            memcpy(&val,
                   buffer + (nOffset - nOffsetStartPage) + i * sizeofvalint,
                   sizeof(val));
            if (bSwab)
                TIFFSwabLong8((uint64_t *)&val);
            panVals[strile + i] = (uint64_t)val;
        }
    }
    return 1;
}

static int _TIFFFetchStrileValue(TIFF *tif, uint32_t strile,
                                 TIFFDirEntry *dirent, uint64_t **parray)
{
    static const char module[] = "_TIFFFetchStrileValue";
    TIFFDirectory *td = &tif->tif_dir;
    if (strile >= dirent->tdir_count)
    {
        return 0;
    }
    if (strile >= td->td_stripoffsetbyteallocsize)
    {
        uint32_t nStripArrayAllocBefore = td->td_stripoffsetbyteallocsize;
        uint32_t nStripArrayAllocNew;
        uint64_t nArraySize64;
        size_t nArraySize;
        uint64_t *offsetArray;
        uint64_t *bytecountArray;

        if (strile > 1000000)
        {
            uint64_t filesize = TIFFGetFileSize(tif);
            /* Avoid excessive memory allocation attempt */
            /* For such a big blockid we need at least a TIFF_LONG per strile */
            /* for the offset array. */
            if (strile > filesize / sizeof(uint32_t))
            {
                TIFFErrorExtR(tif, module, "File too short");
                return 0;
            }
        }

        if (td->td_stripoffsetbyteallocsize == 0 &&
            td->td_nstrips < 1024 * 1024)
        {
            nStripArrayAllocNew = td->td_nstrips;
        }
        else
        {
#define TIFF_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define TIFF_MIN(a, b) (((a) < (b)) ? (a) : (b))
            nStripArrayAllocNew = TIFF_MAX(strile + 1, 1024U * 512U);
            if (nStripArrayAllocNew < 0xFFFFFFFFU / 2)
                nStripArrayAllocNew *= 2;
            nStripArrayAllocNew = TIFF_MIN(nStripArrayAllocNew, td->td_nstrips);
        }
        assert(strile < nStripArrayAllocNew);
        nArraySize64 = (uint64_t)sizeof(uint64_t) * nStripArrayAllocNew;
        nArraySize = (size_t)(nArraySize64);
#if SIZEOF_SIZE_T == 4
        if (nArraySize != nArraySize64)
        {
            TIFFErrorExtR(tif, module,
                          "Cannot allocate strip offset and bytecount arrays");
            return 0;
        }
#endif
        offsetArray = (uint64_t *)(_TIFFreallocExt(tif, td->td_stripoffset_p,
                                                   nArraySize));
        bytecountArray = (uint64_t *)(_TIFFreallocExt(
            tif, td->td_stripbytecount_p, nArraySize));
        if (offsetArray)
            td->td_stripoffset_p = offsetArray;
        if (bytecountArray)
            td->td_stripbytecount_p = bytecountArray;
        if (offsetArray && bytecountArray)
        {
            td->td_stripoffsetbyteallocsize = nStripArrayAllocNew;
            /* Initialize new entries to ~0 / -1 */
            /* coverity[overrun-buffer-arg] */
            memset(td->td_stripoffset_p + nStripArrayAllocBefore, 0xFF,
                   (td->td_stripoffsetbyteallocsize - nStripArrayAllocBefore) *
                       sizeof(uint64_t));
            /* coverity[overrun-buffer-arg] */
            memset(td->td_stripbytecount_p + nStripArrayAllocBefore, 0xFF,
                   (td->td_stripoffsetbyteallocsize - nStripArrayAllocBefore) *
                       sizeof(uint64_t));
        }
        else
        {
            TIFFErrorExtR(tif, module,
                          "Cannot allocate strip offset and bytecount arrays");
            _TIFFfreeExt(tif, td->td_stripoffset_p);
            td->td_stripoffset_p = NULL;
            _TIFFfreeExt(tif, td->td_stripbytecount_p);
            td->td_stripbytecount_p = NULL;
            td->td_stripoffsetbyteallocsize = 0;
        }
    }
    if (*parray == NULL || strile >= td->td_stripoffsetbyteallocsize)
        return 0;

    if (~((*parray)[strile]) == 0)
    {
        if (!_TIFFPartialReadStripArray(tif, dirent, strile, *parray))
        {
            (*parray)[strile] = 0;
            return 0;
        }
    }

    return 1;
}

static uint64_t _TIFFGetStrileOffsetOrByteCountValue(TIFF *tif, uint32_t strile,
                                                     TIFFDirEntry *dirent,
                                                     uint64_t **parray,
                                                     int *pbErr)
{
    TIFFDirectory *td = &tif->tif_dir;
    if (pbErr)
        *pbErr = 0;
    if ((tif->tif_flags & TIFF_DEFERSTRILELOAD) &&
        !(tif->tif_flags & TIFF_CHOPPEDUPARRAYS))
    {
        if (!(tif->tif_flags & TIFF_LAZYSTRILELOAD) ||
            /* If the values may fit in the toff_long/toff_long8 member */
            /* then use _TIFFFillStriles to simplify _TIFFFetchStrileValue */
            dirent->tdir_count <= 4)
        {
            if (!_TIFFFillStriles(tif))
            {
                if (pbErr)
                    *pbErr = 1;
                /* Do not return, as we want this function to always */
                /* return the same value if called several times with */
                /* the same arguments */
            }
        }
        else
        {
            if (!_TIFFFetchStrileValue(tif, strile, dirent, parray))
            {
                if (pbErr)
                    *pbErr = 1;
                return 0;
            }
        }
    }
    if (*parray == NULL || strile >= td->td_nstrips)
    {
        if (pbErr)
            *pbErr = 1;
        return 0;
    }
    return (*parray)[strile];
}

/* Return the value of the TileOffsets/StripOffsets array for the specified
 * tile/strile */
uint64_t TIFFGetStrileOffset(TIFF *tif, uint32_t strile)
{
    return TIFFGetStrileOffsetWithErr(tif, strile, NULL);
}

/* Return the value of the TileOffsets/StripOffsets array for the specified
 * tile/strile */
uint64_t TIFFGetStrileOffsetWithErr(TIFF *tif, uint32_t strile, int *pbErr)
{
    TIFFDirectory *td = &tif->tif_dir;
    return _TIFFGetStrileOffsetOrByteCountValue(tif, strile,
                                                &(td->td_stripoffset_entry),
                                                &(td->td_stripoffset_p), pbErr);
}

/* Return the value of the TileByteCounts/StripByteCounts array for the
 * specified tile/strile */
uint64_t TIFFGetStrileByteCount(TIFF *tif, uint32_t strile)
{
    return TIFFGetStrileByteCountWithErr(tif, strile, NULL);
}

/* Return the value of the TileByteCounts/StripByteCounts array for the
 * specified tile/strile */
uint64_t TIFFGetStrileByteCountWithErr(TIFF *tif, uint32_t strile, int *pbErr)
{
    TIFFDirectory *td = &tif->tif_dir;
    return _TIFFGetStrileOffsetOrByteCountValue(
        tif, strile, &(td->td_stripbytecount_entry), &(td->td_stripbytecount_p),
        pbErr);
}

int _TIFFFillStriles(TIFF *tif) { return _TIFFFillStrilesInternal(tif, 1); }

static int _TIFFFillStrilesInternal(TIFF *tif, int loadStripByteCount)
{
    register TIFFDirectory *td = &tif->tif_dir;
    int return_value = 1;

    /* Do not do anything if TIFF_DEFERSTRILELOAD is not set */
    if (!(tif->tif_flags & TIFF_DEFERSTRILELOAD) ||
        (tif->tif_flags & TIFF_CHOPPEDUPARRAYS) != 0)
        return 1;

    if (tif->tif_flags & TIFF_LAZYSTRILELOAD)
    {
        /* In case of lazy loading, reload completely the arrays */
        _TIFFfreeExt(tif, td->td_stripoffset_p);
        _TIFFfreeExt(tif, td->td_stripbytecount_p);
        td->td_stripoffset_p = NULL;
        td->td_stripbytecount_p = NULL;
        td->td_stripoffsetbyteallocsize = 0;
        tif->tif_flags &= ~TIFF_LAZYSTRILELOAD;
    }

    /* If stripoffset array is already loaded, exit with success */
    if (td->td_stripoffset_p != NULL)
        return 1;

    /* If tdir_count was canceled, then we already got there, but in error */
    if (td->td_stripoffset_entry.tdir_count == 0)
        return 0;

    if (!TIFFFetchStripThing(tif, &(td->td_stripoffset_entry), td->td_nstrips,
                             &td->td_stripoffset_p))
    {
        return_value = 0;
    }

    if (loadStripByteCount &&
        !TIFFFetchStripThing(tif, &(td->td_stripbytecount_entry),
                             td->td_nstrips, &td->td_stripbytecount_p))
    {
        return_value = 0;
    }

    _TIFFmemset(&(td->td_stripoffset_entry), 0, sizeof(TIFFDirEntry));
    _TIFFmemset(&(td->td_stripbytecount_entry), 0, sizeof(TIFFDirEntry));

#ifdef STRIPBYTECOUNTSORTED_UNUSED
    if (tif->tif_dir.td_nstrips > 1 && return_value == 1)
    {
        uint32_t strip;

        tif->tif_dir.td_stripbytecountsorted = 1;
        for (strip = 1; strip < tif->tif_dir.td_nstrips; strip++)
        {
            if (tif->tif_dir.td_stripoffset_p[strip - 1] >
                tif->tif_dir.td_stripoffset_p[strip])
            {
                tif->tif_dir.td_stripbytecountsorted = 0;
                break;
            }
        }
    }
#endif

    return return_value;
}
