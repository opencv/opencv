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
 * - add a field 'ignore' to the TIFFDirEntry structure, to flag status,
 *   eliminating current use of the IGNORE value, and therefore eliminating
 *   current irrational behaviour on tags with tag id code 0
 * - add a field 'field_info' to the TIFFDirEntry structure, and set that with
 *   the pointer to the appropriate TIFFField structure early on in
 *   TIFFReadDirectory, so as to eliminate current possibly repetitive lookup.
 */

#include "tiffiop.h"
#include <float.h>
#include <stdlib.h>

#define IGNORE 0          /* tag placeholder used below */
#define FAILED_FII    ((uint32) -1)

#ifdef HAVE_IEEEFP
# define TIFFCvtIEEEFloatToNative(tif, n, fp)
# define TIFFCvtIEEEDoubleToNative(tif, n, dp)
#else
extern void TIFFCvtIEEEFloatToNative(TIFF*, uint32, float*);
extern void TIFFCvtIEEEDoubleToNative(TIFF*, uint32, double*);
#endif

enum TIFFReadDirEntryErr {
	TIFFReadDirEntryErrOk = 0,
	TIFFReadDirEntryErrCount = 1,
	TIFFReadDirEntryErrType = 2,
	TIFFReadDirEntryErrIo = 3,
	TIFFReadDirEntryErrRange = 4,
	TIFFReadDirEntryErrPsdif = 5,
	TIFFReadDirEntryErrSizesan = 6,
	TIFFReadDirEntryErrAlloc = 7,
};

static enum TIFFReadDirEntryErr TIFFReadDirEntryByte(TIFF* tif, TIFFDirEntry* direntry, uint8* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryShort(TIFF* tif, TIFFDirEntry* direntry, uint16* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryLong(TIFF* tif, TIFFDirEntry* direntry, uint32* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryLong8(TIFF* tif, TIFFDirEntry* direntry, uint64* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryFloat(TIFF* tif, TIFFDirEntry* direntry, float* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryDouble(TIFF* tif, TIFFDirEntry* direntry, double* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryIfd8(TIFF* tif, TIFFDirEntry* direntry, uint64* value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryArray(TIFF* tif, TIFFDirEntry* direntry, uint32* count, uint32 desttypesize, void** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryByteArray(TIFF* tif, TIFFDirEntry* direntry, uint8** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntrySbyteArray(TIFF* tif, TIFFDirEntry* direntry, int8** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryShortArray(TIFF* tif, TIFFDirEntry* direntry, uint16** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntrySshortArray(TIFF* tif, TIFFDirEntry* direntry, int16** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryLongArray(TIFF* tif, TIFFDirEntry* direntry, uint32** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntrySlongArray(TIFF* tif, TIFFDirEntry* direntry, int32** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryLong8Array(TIFF* tif, TIFFDirEntry* direntry, uint64** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntrySlong8Array(TIFF* tif, TIFFDirEntry* direntry, int64** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryFloatArray(TIFF* tif, TIFFDirEntry* direntry, float** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryDoubleArray(TIFF* tif, TIFFDirEntry* direntry, double** value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryIfd8Array(TIFF* tif, TIFFDirEntry* direntry, uint64** value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryPersampleShort(TIFF* tif, TIFFDirEntry* direntry, uint16* value);
#if 0
static enum TIFFReadDirEntryErr TIFFReadDirEntryPersampleDouble(TIFF* tif, TIFFDirEntry* direntry, double* value);
#endif

static void TIFFReadDirEntryCheckedByte(TIFF* tif, TIFFDirEntry* direntry, uint8* value);
static void TIFFReadDirEntryCheckedSbyte(TIFF* tif, TIFFDirEntry* direntry, int8* value);
static void TIFFReadDirEntryCheckedShort(TIFF* tif, TIFFDirEntry* direntry, uint16* value);
static void TIFFReadDirEntryCheckedSshort(TIFF* tif, TIFFDirEntry* direntry, int16* value);
static void TIFFReadDirEntryCheckedLong(TIFF* tif, TIFFDirEntry* direntry, uint32* value);
static void TIFFReadDirEntryCheckedSlong(TIFF* tif, TIFFDirEntry* direntry, int32* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedLong8(TIFF* tif, TIFFDirEntry* direntry, uint64* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedSlong8(TIFF* tif, TIFFDirEntry* direntry, int64* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedRational(TIFF* tif, TIFFDirEntry* direntry, double* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedSrational(TIFF* tif, TIFFDirEntry* direntry, double* value);
static void TIFFReadDirEntryCheckedFloat(TIFF* tif, TIFFDirEntry* direntry, float* value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedDouble(TIFF* tif, TIFFDirEntry* direntry, double* value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteSbyte(int8 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteShort(uint16 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteSshort(int16 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteLong(uint32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteSlong(int32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteLong8(uint64 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteSlong8(int64 value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteByte(uint8 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteShort(uint16 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteSshort(int16 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteLong(uint32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteSlong(int32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteLong8(uint64 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteSlong8(int64 value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortSbyte(int8 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortSshort(int16 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortLong(uint32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortSlong(int32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortLong8(uint64 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortSlong8(int64 value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortShort(uint16 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortLong(uint32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortSlong(int32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortLong8(uint64 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortSlong8(int64 value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLongSbyte(int8 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLongSshort(int16 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLongSlong(int32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLongLong8(uint64 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLongSlong8(int64 value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSlongLong(uint32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSlongLong8(uint64 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSlongSlong8(int64 value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLong8Sbyte(int8 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLong8Sshort(int16 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLong8Slong(int32 value);
static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLong8Slong8(int64 value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSlong8Long8(uint64 value);

static enum TIFFReadDirEntryErr TIFFReadDirEntryData(TIFF* tif, uint64 offset, tmsize_t size, void* dest);
static void TIFFReadDirEntryOutputErr(TIFF* tif, enum TIFFReadDirEntryErr err, const char* module, const char* tagname, int recover);

static void TIFFReadDirectoryCheckOrder(TIFF* tif, TIFFDirEntry* dir, uint16 dircount);
static TIFFDirEntry* TIFFReadDirectoryFindEntry(TIFF* tif, TIFFDirEntry* dir, uint16 dircount, uint16 tagid);
static void TIFFReadDirectoryFindFieldInfo(TIFF* tif, uint16 tagid, uint32* fii);

static int EstimateStripByteCounts(TIFF* tif, TIFFDirEntry* dir, uint16 dircount);
static void MissingRequired(TIFF*, const char*);
static int TIFFCheckDirOffset(TIFF* tif, uint64 diroff);
static int CheckDirCount(TIFF*, TIFFDirEntry*, uint32);
static uint16 TIFFFetchDirectory(TIFF* tif, uint64 diroff, TIFFDirEntry** pdir, uint64* nextdiroff);
static int TIFFFetchNormalTag(TIFF*, TIFFDirEntry*, int recover);
static int TIFFFetchStripThing(TIFF* tif, TIFFDirEntry* dir, uint32 nstrips, uint64** lpp);
static int TIFFFetchSubjectDistance(TIFF*, TIFFDirEntry*);
static void ChopUpSingleUncompressedStrip(TIFF*);
static void TryChopUpUncompressedBigTiff(TIFF*);
static uint64 TIFFReadUInt64(const uint8 *value);
static int _TIFFGetMaxColorChannels(uint16 photometric);

static int _TIFFFillStrilesInternal( TIFF *tif, int loadStripByteCount );

typedef union _UInt64Aligned_t
{
        double d;
	uint64 l;
	uint32 i[2];
	uint16 s[4];
	uint8  c[8];
} UInt64Aligned_t;

/*
  Unaligned safe copy of a uint64 value from an octet array.
*/
static uint64 TIFFReadUInt64(const uint8 *value)
{
	UInt64Aligned_t result;

	result.c[0]=value[0];
	result.c[1]=value[1];
	result.c[2]=value[2];
	result.c[3]=value[3];
	result.c[4]=value[4];
	result.c[5]=value[5];
	result.c[6]=value[6];
	result.c[7]=value[7];

	return result.l;
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryByte(TIFF* tif, TIFFDirEntry* direntry, uint8* value)
{
	enum TIFFReadDirEntryErr err;
	if (direntry->tdir_count!=1)
		return(TIFFReadDirEntryErrCount);
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			TIFFReadDirEntryCheckedByte(tif,direntry,value);
			return(TIFFReadDirEntryErrOk);
		case TIFF_SBYTE:
			{
				int8 m;
				TIFFReadDirEntryCheckedSbyte(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeByteSbyte(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint8)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SHORT:
			{
				uint16 m;
				TIFFReadDirEntryCheckedShort(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeByteShort(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint8)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SSHORT:
			{
				int16 m;
				TIFFReadDirEntryCheckedSshort(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeByteSshort(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint8)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG:
			{
				uint32 m;
				TIFFReadDirEntryCheckedLong(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeByteLong(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint8)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG:
			{
				int32 m;
				TIFFReadDirEntryCheckedSlong(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeByteSlong(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint8)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG8:
			{
				uint64 m;
				err=TIFFReadDirEntryCheckedLong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				err=TIFFReadDirEntryCheckRangeByteLong8(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint8)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG8:
			{
				int64 m;
				err=TIFFReadDirEntryCheckedSlong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				err=TIFFReadDirEntryCheckRangeByteSlong8(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint8)m;
				return(TIFFReadDirEntryErrOk);
			}
		default:
			return(TIFFReadDirEntryErrType);
	}
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryShort(TIFF* tif, TIFFDirEntry* direntry, uint16* value)
{
	enum TIFFReadDirEntryErr err;
	if (direntry->tdir_count!=1)
		return(TIFFReadDirEntryErrCount);
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8 m;
				TIFFReadDirEntryCheckedByte(tif,direntry,&m);
				*value=(uint16)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SBYTE:
			{
				int8 m;
				TIFFReadDirEntryCheckedSbyte(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeShortSbyte(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint16)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SHORT:
			TIFFReadDirEntryCheckedShort(tif,direntry,value);
			return(TIFFReadDirEntryErrOk);
		case TIFF_SSHORT:
			{
				int16 m;
				TIFFReadDirEntryCheckedSshort(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeShortSshort(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint16)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG:
			{
				uint32 m;
				TIFFReadDirEntryCheckedLong(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeShortLong(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint16)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG:
			{
				int32 m;
				TIFFReadDirEntryCheckedSlong(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeShortSlong(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint16)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG8:
			{
				uint64 m;
				err=TIFFReadDirEntryCheckedLong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				err=TIFFReadDirEntryCheckRangeShortLong8(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint16)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG8:
			{
				int64 m;
				err=TIFFReadDirEntryCheckedSlong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				err=TIFFReadDirEntryCheckRangeShortSlong8(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint16)m;
				return(TIFFReadDirEntryErrOk);
			}
		default:
			return(TIFFReadDirEntryErrType);
	}
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryLong(TIFF* tif, TIFFDirEntry* direntry, uint32* value)
{
	enum TIFFReadDirEntryErr err;
	if (direntry->tdir_count!=1)
		return(TIFFReadDirEntryErrCount);
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8 m;
				TIFFReadDirEntryCheckedByte(tif,direntry,&m);
				*value=(uint32)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SBYTE:
			{
				int8 m;
				TIFFReadDirEntryCheckedSbyte(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeLongSbyte(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint32)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SHORT:
			{
				uint16 m;
				TIFFReadDirEntryCheckedShort(tif,direntry,&m);
				*value=(uint32)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SSHORT:
			{
				int16 m;
				TIFFReadDirEntryCheckedSshort(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeLongSshort(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint32)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG:
			TIFFReadDirEntryCheckedLong(tif,direntry,value);
			return(TIFFReadDirEntryErrOk);
		case TIFF_SLONG:
			{
				int32 m;
				TIFFReadDirEntryCheckedSlong(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeLongSlong(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint32)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG8:
			{
				uint64 m;
				err=TIFFReadDirEntryCheckedLong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				err=TIFFReadDirEntryCheckRangeLongLong8(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint32)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG8:
			{
				int64 m;
				err=TIFFReadDirEntryCheckedSlong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				err=TIFFReadDirEntryCheckRangeLongSlong8(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint32)m;
				return(TIFFReadDirEntryErrOk);
			}
		default:
			return(TIFFReadDirEntryErrType);
	}
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryLong8(TIFF* tif, TIFFDirEntry* direntry, uint64* value)
{
	enum TIFFReadDirEntryErr err;
	if (direntry->tdir_count!=1)
		return(TIFFReadDirEntryErrCount);
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8 m;
				TIFFReadDirEntryCheckedByte(tif,direntry,&m);
				*value=(uint64)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SBYTE:
			{
				int8 m;
				TIFFReadDirEntryCheckedSbyte(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeLong8Sbyte(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint64)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SHORT:
			{
				uint16 m;
				TIFFReadDirEntryCheckedShort(tif,direntry,&m);
				*value=(uint64)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SSHORT:
			{
				int16 m;
				TIFFReadDirEntryCheckedSshort(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeLong8Sshort(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint64)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG:
			{
				uint32 m;
				TIFFReadDirEntryCheckedLong(tif,direntry,&m);
				*value=(uint64)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG:
			{
				int32 m;
				TIFFReadDirEntryCheckedSlong(tif,direntry,&m);
				err=TIFFReadDirEntryCheckRangeLong8Slong(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint64)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG8:
			err=TIFFReadDirEntryCheckedLong8(tif,direntry,value);
			return(err);
		case TIFF_SLONG8:
			{
				int64 m;
				err=TIFFReadDirEntryCheckedSlong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				err=TIFFReadDirEntryCheckRangeLong8Slong8(m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(uint64)m;
				return(TIFFReadDirEntryErrOk);
			}
		default:
			return(TIFFReadDirEntryErrType);
	}
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryFloat(TIFF* tif, TIFFDirEntry* direntry, float* value)
{
	enum TIFFReadDirEntryErr err;
	if (direntry->tdir_count!=1)
		return(TIFFReadDirEntryErrCount);
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8 m;
				TIFFReadDirEntryCheckedByte(tif,direntry,&m);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SBYTE:
			{
				int8 m;
				TIFFReadDirEntryCheckedSbyte(tif,direntry,&m);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SHORT:
			{
				uint16 m;
				TIFFReadDirEntryCheckedShort(tif,direntry,&m);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SSHORT:
			{
				int16 m;
				TIFFReadDirEntryCheckedSshort(tif,direntry,&m);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG:
			{
				uint32 m;
				TIFFReadDirEntryCheckedLong(tif,direntry,&m);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG:
			{
				int32 m;
				TIFFReadDirEntryCheckedSlong(tif,direntry,&m);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG8:
			{
				uint64 m;
				err=TIFFReadDirEntryCheckedLong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
#if defined(__WIN32__) && (_MSC_VER < 1500)
				/*
				 * XXX: MSVC 6.0 does not support conversion
				 * of 64-bit integers into floating point
				 * values.
				 */
				*value = _TIFFUInt64ToFloat(m);
#else
				*value=(float)m;
#endif
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG8:
			{
				int64 m;
				err=TIFFReadDirEntryCheckedSlong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_RATIONAL:
			{
				double m;
				err=TIFFReadDirEntryCheckedRational(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SRATIONAL:
			{
				double m;
				err=TIFFReadDirEntryCheckedSrational(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_FLOAT:
			TIFFReadDirEntryCheckedFloat(tif,direntry,value);
			return(TIFFReadDirEntryErrOk);
		case TIFF_DOUBLE:
			{
				double m;
				err=TIFFReadDirEntryCheckedDouble(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				if ((m > FLT_MAX) || (m < FLT_MIN))
					return(TIFFReadDirEntryErrRange);
				*value=(float)m;
				return(TIFFReadDirEntryErrOk);
			}
		default:
			return(TIFFReadDirEntryErrType);
	}
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryDouble(TIFF* tif, TIFFDirEntry* direntry, double* value)
{
	enum TIFFReadDirEntryErr err;
	if (direntry->tdir_count!=1)
		return(TIFFReadDirEntryErrCount);
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8 m;
				TIFFReadDirEntryCheckedByte(tif,direntry,&m);
				*value=(double)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SBYTE:
			{
				int8 m;
				TIFFReadDirEntryCheckedSbyte(tif,direntry,&m);
				*value=(double)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SHORT:
			{
				uint16 m;
				TIFFReadDirEntryCheckedShort(tif,direntry,&m);
				*value=(double)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SSHORT:
			{
				int16 m;
				TIFFReadDirEntryCheckedSshort(tif,direntry,&m);
				*value=(double)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG:
			{
				uint32 m;
				TIFFReadDirEntryCheckedLong(tif,direntry,&m);
				*value=(double)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG:
			{
				int32 m;
				TIFFReadDirEntryCheckedSlong(tif,direntry,&m);
				*value=(double)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG8:
			{
				uint64 m;
				err=TIFFReadDirEntryCheckedLong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
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
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG8:
			{
				int64 m;
				err=TIFFReadDirEntryCheckedSlong8(tif,direntry,&m);
				if (err!=TIFFReadDirEntryErrOk)
					return(err);
				*value=(double)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_RATIONAL:
			err=TIFFReadDirEntryCheckedRational(tif,direntry,value);
			return(err);
		case TIFF_SRATIONAL:
			err=TIFFReadDirEntryCheckedSrational(tif,direntry,value);
			return(err);
		case TIFF_FLOAT:
			{
				float m;
				TIFFReadDirEntryCheckedFloat(tif,direntry,&m);
				*value=(double)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_DOUBLE:
			err=TIFFReadDirEntryCheckedDouble(tif,direntry,value);
			return(err);
		default:
			return(TIFFReadDirEntryErrType);
	}
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryIfd8(TIFF* tif, TIFFDirEntry* direntry, uint64* value)
{
	enum TIFFReadDirEntryErr err;
	if (direntry->tdir_count!=1)
		return(TIFFReadDirEntryErrCount);
	switch (direntry->tdir_type)
	{
		case TIFF_LONG:
		case TIFF_IFD:
			{
				uint32 m;
				TIFFReadDirEntryCheckedLong(tif,direntry,&m);
				*value=(uint64)m;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_LONG8:
		case TIFF_IFD8:
			err=TIFFReadDirEntryCheckedLong8(tif,direntry,value);
			return(err);
		default:
			return(TIFFReadDirEntryErrType);
	}
}


#define INITIAL_THRESHOLD (1024 * 1024)
#define THRESHOLD_MULTIPLIER 10
#define MAX_THRESHOLD (THRESHOLD_MULTIPLIER * THRESHOLD_MULTIPLIER * THRESHOLD_MULTIPLIER * INITIAL_THRESHOLD)

static enum TIFFReadDirEntryErr TIFFReadDirEntryDataAndRealloc(
                    TIFF* tif, uint64 offset, tmsize_t size, void** pdest)
{
#if SIZEOF_SIZE_T == 8
        tmsize_t threshold = INITIAL_THRESHOLD;
#endif
        tmsize_t already_read = 0;

        assert( !isMapped(tif) );

        if (!SeekOK(tif,offset))
                return(TIFFReadDirEntryErrIo);

        /* On 64 bit processes, read first a maximum of 1 MB, then 10 MB, etc */
        /* so as to avoid allocating too much memory in case the file is too */
        /* short. We could ask for the file size, but this might be */
        /* expensive with some I/O layers (think of reading a gzipped file) */
        /* Restrict to 64 bit processes, so as to avoid reallocs() */
        /* on 32 bit processes where virtual memory is scarce.  */
        while( already_read < size )
        {
            void* new_dest;
            tmsize_t bytes_read;
            tmsize_t to_read = size - already_read;
#if SIZEOF_SIZE_T == 8
            if( to_read >= threshold && threshold < MAX_THRESHOLD )
            {
                to_read = threshold;
                threshold *= THRESHOLD_MULTIPLIER;
            }
#endif

            new_dest = (uint8*) _TIFFrealloc(
                            *pdest, already_read + to_read);
            if( new_dest == NULL )
            {
                TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
                            "Failed to allocate memory for %s "
                            "(%ld elements of %ld bytes each)",
                            "TIFFReadDirEntryArray",
                             (long) 1, (long) (already_read + to_read));
                return TIFFReadDirEntryErrAlloc;
            }
            *pdest = new_dest;

            bytes_read = TIFFReadFile(tif,
                (char*)*pdest + already_read, to_read);
            already_read += bytes_read;
            if (bytes_read != to_read) {
                return TIFFReadDirEntryErrIo;
            }
        }
        return TIFFReadDirEntryErrOk;
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryArrayWithLimit(
    TIFF* tif, TIFFDirEntry* direntry, uint32* count, uint32 desttypesize,
    void** value, uint64 maxcount)
{
	int typesize;
	uint32 datasize;
	void* data;
        uint64 target_count64;
	typesize=TIFFDataWidth(direntry->tdir_type);

        target_count64 = (direntry->tdir_count > maxcount) ?
                maxcount : direntry->tdir_count;

	if ((target_count64==0)||(typesize==0))
	{
		*value=0;
		return(TIFFReadDirEntryErrOk);
	}
        (void) desttypesize;

        /* 
         * As a sanity check, make sure we have no more than a 2GB tag array 
         * in either the current data type or the dest data type.  This also
         * avoids problems with overflow of tmsize_t on 32bit systems.
         */
	if ((uint64)(2147483647/typesize)<target_count64)
		return(TIFFReadDirEntryErrSizesan);
	if ((uint64)(2147483647/desttypesize)<target_count64)
		return(TIFFReadDirEntryErrSizesan);

	*count=(uint32)target_count64;
	datasize=(*count)*typesize;
	assert((tmsize_t)datasize>0);

	if( isMapped(tif) && datasize > (uint32)tif->tif_size )
		return TIFFReadDirEntryErrIo;

	if( !isMapped(tif) &&
		(((tif->tif_flags&TIFF_BIGTIFF) && datasize > 8) ||
		(!(tif->tif_flags&TIFF_BIGTIFF) && datasize > 4)) )
	{
		data = NULL;
	}
	else
	{
		data=_TIFFCheckMalloc(tif, *count, typesize, "ReadDirEntryArray");
		if (data==0)
			return(TIFFReadDirEntryErrAlloc);
	}
	if (!(tif->tif_flags&TIFF_BIGTIFF))
	{
		if (datasize<=4)
			_TIFFmemcpy(data,&direntry->tdir_offset,datasize);
		else
		{
			enum TIFFReadDirEntryErr err;
			uint32 offset = direntry->tdir_offset.toff_long;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabLong(&offset);
			if( isMapped(tif) )
				err=TIFFReadDirEntryData(tif,(uint64)offset,(tmsize_t)datasize,data);
			else
				err=TIFFReadDirEntryDataAndRealloc(tif,(uint64)offset,(tmsize_t)datasize,&data);
			if (err!=TIFFReadDirEntryErrOk)
			{
				_TIFFfree(data);
				return(err);
			}
		}
	}
	else
	{
		if (datasize<=8)
			_TIFFmemcpy(data,&direntry->tdir_offset,datasize);
		else
		{
			enum TIFFReadDirEntryErr err;
			uint64 offset = direntry->tdir_offset.toff_long8;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabLong8(&offset);
			if( isMapped(tif) )
				err=TIFFReadDirEntryData(tif,(uint64)offset,(tmsize_t)datasize,data);
			else
				err=TIFFReadDirEntryDataAndRealloc(tif,(uint64)offset,(tmsize_t)datasize,&data);
			if (err!=TIFFReadDirEntryErrOk)
			{
				_TIFFfree(data);
				return(err);
			}
		}
	}
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryArray(TIFF* tif, TIFFDirEntry* direntry, uint32* count, uint32 desttypesize, void** value)
{
    return TIFFReadDirEntryArrayWithLimit(tif, direntry, count,
                                          desttypesize, value, ~((uint64)0));
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryByteArray(TIFF* tif, TIFFDirEntry* direntry, uint8** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	uint8* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,1,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_ASCII:
		case TIFF_UNDEFINED:
		case TIFF_BYTE:
			*value=(uint8*)origdata;
			return(TIFFReadDirEntryErrOk);
		case TIFF_SBYTE:
			{
				int8* m;
				uint32 n;
				m=(int8*)origdata;
				for (n=0; n<count; n++)
				{
					err=TIFFReadDirEntryCheckRangeByteSbyte(*m);
					if (err!=TIFFReadDirEntryErrOk)
					{
						_TIFFfree(origdata);
						return(err);
					}
					m++;
				}
				*value=(uint8*)origdata;
				return(TIFFReadDirEntryErrOk);
			}
	}
	data=(uint8*)_TIFFmalloc(count);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_SHORT:
			{
				uint16* ma;
				uint8* mb;
				uint32 n;
				ma=(uint16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(ma);
					err=TIFFReadDirEntryCheckRangeByteShort(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint8)(*ma++);
				}
			}
			break;
		case TIFF_SSHORT:
			{
				int16* ma;
				uint8* mb;
				uint32 n;
				ma=(int16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)ma);
					err=TIFFReadDirEntryCheckRangeByteSshort(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint8)(*ma++);
				}
			}
			break;
		case TIFF_LONG:
			{
				uint32* ma;
				uint8* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					err=TIFFReadDirEntryCheckRangeByteLong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint8)(*ma++);
				}
			}
			break;
		case TIFF_SLONG:
			{
				int32* ma;
				uint8* mb;
				uint32 n;
				ma=(int32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)ma);
					err=TIFFReadDirEntryCheckRangeByteSlong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint8)(*ma++);
				}
			}
			break;
		case TIFF_LONG8:
			{
				uint64* ma;
				uint8* mb;
				uint32 n;
				ma=(uint64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8(ma);
					err=TIFFReadDirEntryCheckRangeByteLong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint8)(*ma++);
				}
			}
			break;
		case TIFF_SLONG8:
			{
				int64* ma;
				uint8* mb;
				uint32 n;
				ma=(int64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)ma);
					err=TIFFReadDirEntryCheckRangeByteSlong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint8)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	if (err!=TIFFReadDirEntryErrOk)
	{
		_TIFFfree(data);
		return(err);
	}
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntrySbyteArray(TIFF* tif, TIFFDirEntry* direntry, int8** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	int8* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,1,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_UNDEFINED:
		case TIFF_BYTE:
			{
				uint8* m;
				uint32 n;
				m=(uint8*)origdata;
				for (n=0; n<count; n++)
				{
					err=TIFFReadDirEntryCheckRangeSbyteByte(*m);
					if (err!=TIFFReadDirEntryErrOk)
					{
						_TIFFfree(origdata);
						return(err);
					}
					m++;
				}
				*value=(int8*)origdata;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SBYTE:
			*value=(int8*)origdata;
			return(TIFFReadDirEntryErrOk);
	}
	data=(int8*)_TIFFmalloc(count);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_SHORT:
			{
				uint16* ma;
				int8* mb;
				uint32 n;
				ma=(uint16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(ma);
					err=TIFFReadDirEntryCheckRangeSbyteShort(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int8)(*ma++);
				}
			}
			break;
		case TIFF_SSHORT:
			{
				int16* ma;
				int8* mb;
				uint32 n;
				ma=(int16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)ma);
					err=TIFFReadDirEntryCheckRangeSbyteSshort(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int8)(*ma++);
				}
			}
			break;
		case TIFF_LONG:
			{
				uint32* ma;
				int8* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					err=TIFFReadDirEntryCheckRangeSbyteLong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int8)(*ma++);
				}
			}
			break;
		case TIFF_SLONG:
			{
				int32* ma;
				int8* mb;
				uint32 n;
				ma=(int32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)ma);
					err=TIFFReadDirEntryCheckRangeSbyteSlong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int8)(*ma++);
				}
			}
			break;
		case TIFF_LONG8:
			{
				uint64* ma;
				int8* mb;
				uint32 n;
				ma=(uint64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8(ma);
					err=TIFFReadDirEntryCheckRangeSbyteLong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int8)(*ma++);
				}
			}
			break;
		case TIFF_SLONG8:
			{
				int64* ma;
				int8* mb;
				uint32 n;
				ma=(int64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)ma);
					err=TIFFReadDirEntryCheckRangeSbyteSlong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int8)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	if (err!=TIFFReadDirEntryErrOk)
	{
		_TIFFfree(data);
		return(err);
	}
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryShortArray(TIFF* tif, TIFFDirEntry* direntry, uint16** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	uint16* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,2,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_SHORT:
			*value=(uint16*)origdata;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfShort(*value,count);  
			return(TIFFReadDirEntryErrOk);
		case TIFF_SSHORT:
			{
				int16* m;
				uint32 n;
				m=(int16*)origdata;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)m);
					err=TIFFReadDirEntryCheckRangeShortSshort(*m);
					if (err!=TIFFReadDirEntryErrOk)
					{
						_TIFFfree(origdata);
						return(err);
					}
					m++;
				}
				*value=(uint16*)origdata;
				return(TIFFReadDirEntryErrOk);
			}
	}
	data=(uint16*)_TIFFmalloc(count*2);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8* ma;
				uint16* mb;
				uint32 n;
				ma=(uint8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(uint16)(*ma++);
			}
			break;
		case TIFF_SBYTE:
			{
				int8* ma;
				uint16* mb;
				uint32 n;
				ma=(int8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					err=TIFFReadDirEntryCheckRangeShortSbyte(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint16)(*ma++);
				}
			}
			break;
		case TIFF_LONG:
			{
				uint32* ma;
				uint16* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					err=TIFFReadDirEntryCheckRangeShortLong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint16)(*ma++);
				}
			}
			break;
		case TIFF_SLONG:
			{
				int32* ma;
				uint16* mb;
				uint32 n;
				ma=(int32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)ma);
					err=TIFFReadDirEntryCheckRangeShortSlong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint16)(*ma++);
				}
			}
			break;
		case TIFF_LONG8:
			{
				uint64* ma;
				uint16* mb;
				uint32 n;
				ma=(uint64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8(ma);
					err=TIFFReadDirEntryCheckRangeShortLong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint16)(*ma++);
				}
			}
			break;
		case TIFF_SLONG8:
			{
				int64* ma;
				uint16* mb;
				uint32 n;
				ma=(int64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)ma);
					err=TIFFReadDirEntryCheckRangeShortSlong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint16)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	if (err!=TIFFReadDirEntryErrOk)
	{
		_TIFFfree(data);
		return(err);
	}
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntrySshortArray(TIFF* tif, TIFFDirEntry* direntry, int16** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	int16* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,2,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_SHORT:
			{
				uint16* m;
				uint32 n;
				m=(uint16*)origdata;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(m);
					err=TIFFReadDirEntryCheckRangeSshortShort(*m);
					if (err!=TIFFReadDirEntryErrOk)
					{
						_TIFFfree(origdata);
						return(err);
					}
					m++;
				}
				*value=(int16*)origdata;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SSHORT:
			*value=(int16*)origdata;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfShort((uint16*)(*value),count);
			return(TIFFReadDirEntryErrOk);
	}
	data=(int16*)_TIFFmalloc(count*2);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8* ma;
				int16* mb;
				uint32 n;
				ma=(uint8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(int16)(*ma++);
			}
			break;
		case TIFF_SBYTE:
			{
				int8* ma;
				int16* mb;
				uint32 n;
				ma=(int8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(int16)(*ma++);
			}
			break;
		case TIFF_LONG:
			{
				uint32* ma;
				int16* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					err=TIFFReadDirEntryCheckRangeSshortLong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int16)(*ma++);
				}
			}
			break;
		case TIFF_SLONG:
			{
				int32* ma;
				int16* mb;
				uint32 n;
				ma=(int32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)ma);
					err=TIFFReadDirEntryCheckRangeSshortSlong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int16)(*ma++);
				}
			}
			break;
		case TIFF_LONG8:
			{
				uint64* ma;
				int16* mb;
				uint32 n;
				ma=(uint64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8(ma);
					err=TIFFReadDirEntryCheckRangeSshortLong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int16)(*ma++);
				}
			}
			break;
		case TIFF_SLONG8:
			{
				int64* ma;
				int16* mb;
				uint32 n;
				ma=(int64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)ma);
					err=TIFFReadDirEntryCheckRangeSshortSlong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int16)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	if (err!=TIFFReadDirEntryErrOk)
	{
		_TIFFfree(data);
		return(err);
	}
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryLongArray(TIFF* tif, TIFFDirEntry* direntry, uint32** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	uint32* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,4,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_LONG:
			*value=(uint32*)origdata;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfLong(*value,count);
			return(TIFFReadDirEntryErrOk);
		case TIFF_SLONG:
			{
				int32* m;
				uint32 n;
				m=(int32*)origdata;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)m);
					err=TIFFReadDirEntryCheckRangeLongSlong(*m);
					if (err!=TIFFReadDirEntryErrOk)
					{
						_TIFFfree(origdata);
						return(err);
					}
					m++;
				}
				*value=(uint32*)origdata;
				return(TIFFReadDirEntryErrOk);
			}
	}
	data=(uint32*)_TIFFmalloc(count*4);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8* ma;
				uint32* mb;
				uint32 n;
				ma=(uint8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(uint32)(*ma++);
			}
			break;
		case TIFF_SBYTE:
			{
				int8* ma;
				uint32* mb;
				uint32 n;
				ma=(int8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					err=TIFFReadDirEntryCheckRangeLongSbyte(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint32)(*ma++);
				}
			}
			break;
		case TIFF_SHORT:
			{
				uint16* ma;
				uint32* mb;
				uint32 n;
				ma=(uint16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(ma);
					*mb++=(uint32)(*ma++);
				}
			}
			break;
		case TIFF_SSHORT:
			{
				int16* ma;
				uint32* mb;
				uint32 n;
				ma=(int16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)ma);
					err=TIFFReadDirEntryCheckRangeLongSshort(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint32)(*ma++);
				}
			}
			break;
		case TIFF_LONG8:
			{
				uint64* ma;
				uint32* mb;
				uint32 n;
				ma=(uint64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8(ma);
					err=TIFFReadDirEntryCheckRangeLongLong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint32)(*ma++);
				}
			}
			break;
		case TIFF_SLONG8:
			{
				int64* ma;
				uint32* mb;
				uint32 n;
				ma=(int64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)ma);
					err=TIFFReadDirEntryCheckRangeLongSlong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint32)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	if (err!=TIFFReadDirEntryErrOk)
	{
		_TIFFfree(data);
		return(err);
	}
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntrySlongArray(TIFF* tif, TIFFDirEntry* direntry, int32** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	int32* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,4,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_LONG:
			{
				uint32* m;
				uint32 n;
				m=(uint32*)origdata;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)m);
					err=TIFFReadDirEntryCheckRangeSlongLong(*m);
					if (err!=TIFFReadDirEntryErrOk)
					{
						_TIFFfree(origdata);
						return(err);
					}
					m++;
				}
				*value=(int32*)origdata;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG:
			*value=(int32*)origdata;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfLong((uint32*)(*value),count);
			return(TIFFReadDirEntryErrOk);
	}
	data=(int32*)_TIFFmalloc(count*4);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8* ma;
				int32* mb;
				uint32 n;
				ma=(uint8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(int32)(*ma++);
			}
			break;
		case TIFF_SBYTE:
			{
				int8* ma;
				int32* mb;
				uint32 n;
				ma=(int8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(int32)(*ma++);
			}
			break;
		case TIFF_SHORT:
			{
				uint16* ma;
				int32* mb;
				uint32 n;
				ma=(uint16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(ma);
					*mb++=(int32)(*ma++);
				}
			}
			break;
		case TIFF_SSHORT:
			{
				int16* ma;
				int32* mb;
				uint32 n;
				ma=(int16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)ma);
					*mb++=(int32)(*ma++);
				}
			}
			break;
		case TIFF_LONG8:
			{
				uint64* ma;
				int32* mb;
				uint32 n;
				ma=(uint64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8(ma);
					err=TIFFReadDirEntryCheckRangeSlongLong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int32)(*ma++);
				}
			}
			break;
		case TIFF_SLONG8:
			{
				int64* ma;
				int32* mb;
				uint32 n;
				ma=(int64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)ma);
					err=TIFFReadDirEntryCheckRangeSlongSlong8(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(int32)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	if (err!=TIFFReadDirEntryErrOk)
	{
		_TIFFfree(data);
		return(err);
	}
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryLong8ArrayWithLimit(
        TIFF* tif, TIFFDirEntry* direntry, uint64** value, uint64 maxcount)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	uint64* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArrayWithLimit(tif,direntry,&count,8,&origdata,maxcount);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_LONG8:
			*value=(uint64*)origdata;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfLong8(*value,count);
			return(TIFFReadDirEntryErrOk);
		case TIFF_SLONG8:
			{
				int64* m;
				uint32 n;
				m=(int64*)origdata;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)m);
					err=TIFFReadDirEntryCheckRangeLong8Slong8(*m);
					if (err!=TIFFReadDirEntryErrOk)
					{
						_TIFFfree(origdata);
						return(err);
					}
					m++;
				}
				*value=(uint64*)origdata;
				return(TIFFReadDirEntryErrOk);
			}
	}
	data=(uint64*)_TIFFmalloc(count*8);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8* ma;
				uint64* mb;
				uint32 n;
				ma=(uint8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(uint64)(*ma++);
			}
			break;
		case TIFF_SBYTE:
			{
				int8* ma;
				uint64* mb;
				uint32 n;
				ma=(int8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					err=TIFFReadDirEntryCheckRangeLong8Sbyte(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint64)(*ma++);
				}
			}
			break;
		case TIFF_SHORT:
			{
				uint16* ma;
				uint64* mb;
				uint32 n;
				ma=(uint16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(ma);
					*mb++=(uint64)(*ma++);
				}
			}
			break;
		case TIFF_SSHORT:
			{
				int16* ma;
				uint64* mb;
				uint32 n;
				ma=(int16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)ma);
					err=TIFFReadDirEntryCheckRangeLong8Sshort(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint64)(*ma++);
				}
			}
			break;
		case TIFF_LONG:
			{
				uint32* ma;
				uint64* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					*mb++=(uint64)(*ma++);
				}
			}
			break;
		case TIFF_SLONG:
			{
				int32* ma;
				uint64* mb;
				uint32 n;
				ma=(int32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)ma);
					err=TIFFReadDirEntryCheckRangeLong8Slong(*ma);
					if (err!=TIFFReadDirEntryErrOk)
						break;
					*mb++=(uint64)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	if (err!=TIFFReadDirEntryErrOk)
	{
		_TIFFfree(data);
		return(err);
	}
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryLong8Array(TIFF* tif, TIFFDirEntry* direntry, uint64** value)
{
    return TIFFReadDirEntryLong8ArrayWithLimit(tif, direntry, value, ~((uint64)0));
}

static enum TIFFReadDirEntryErr TIFFReadDirEntrySlong8Array(TIFF* tif, TIFFDirEntry* direntry, int64** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	int64* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,8,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_LONG8:
			{
				uint64* m;
				uint32 n;
				m=(uint64*)origdata;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8(m);
					err=TIFFReadDirEntryCheckRangeSlong8Long8(*m);
					if (err!=TIFFReadDirEntryErrOk)
					{
						_TIFFfree(origdata);
						return(err);
					}
					m++;
				}
				*value=(int64*)origdata;
				return(TIFFReadDirEntryErrOk);
			}
		case TIFF_SLONG8:
			*value=(int64*)origdata;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfLong8((uint64*)(*value),count);
			return(TIFFReadDirEntryErrOk);
	}
	data=(int64*)_TIFFmalloc(count*8);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8* ma;
				int64* mb;
				uint32 n;
				ma=(uint8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(int64)(*ma++);
			}
			break;
		case TIFF_SBYTE:
			{
				int8* ma;
				int64* mb;
				uint32 n;
				ma=(int8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(int64)(*ma++);
			}
			break;
		case TIFF_SHORT:
			{
				uint16* ma;
				int64* mb;
				uint32 n;
				ma=(uint16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(ma);
					*mb++=(int64)(*ma++);
				}
			}
			break;
		case TIFF_SSHORT:
			{
				int16* ma;
				int64* mb;
				uint32 n;
				ma=(int16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)ma);
					*mb++=(int64)(*ma++);
				}
			}
			break;
		case TIFF_LONG:
			{
				uint32* ma;
				int64* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					*mb++=(int64)(*ma++);
				}
			}
			break;
		case TIFF_SLONG:
			{
				int32* ma;
				int64* mb;
				uint32 n;
				ma=(int32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)ma);
					*mb++=(int64)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryFloatArray(TIFF* tif, TIFFDirEntry* direntry, float** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	float* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,4,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_FLOAT:
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfLong((uint32*)origdata,count);  
			TIFFCvtIEEEDoubleToNative(tif,count,(float*)origdata);
			*value=(float*)origdata;
			return(TIFFReadDirEntryErrOk);
	}
	data=(float*)_TIFFmalloc(count*sizeof(float));
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8* ma;
				float* mb;
				uint32 n;
				ma=(uint8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(float)(*ma++);
			}
			break;
		case TIFF_SBYTE:
			{
				int8* ma;
				float* mb;
				uint32 n;
				ma=(int8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(float)(*ma++);
			}
			break;
		case TIFF_SHORT:
			{
				uint16* ma;
				float* mb;
				uint32 n;
				ma=(uint16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(ma);
					*mb++=(float)(*ma++);
				}
			}
			break;
		case TIFF_SSHORT:
			{
				int16* ma;
				float* mb;
				uint32 n;
				ma=(int16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)ma);
					*mb++=(float)(*ma++);
				}
			}
			break;
		case TIFF_LONG:
			{
				uint32* ma;
				float* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					*mb++=(float)(*ma++);
				}
			}
			break;
		case TIFF_SLONG:
			{
				int32* ma;
				float* mb;
				uint32 n;
				ma=(int32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)ma);
					*mb++=(float)(*ma++);
				}
			}
			break;
		case TIFF_LONG8:
			{
				uint64* ma;
				float* mb;
				uint32 n;
				ma=(uint64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
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
				int64* ma;
				float* mb;
				uint32 n;
				ma=(int64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)ma);
					*mb++=(float)(*ma++);
				}
			}
			break;
		case TIFF_RATIONAL:
			{
				uint32* ma;
				uint32 maa;
				uint32 mab;
				float* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					maa=*ma++;
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					mab=*ma++;
					if (mab==0)
						*mb++=0.0;
					else
						*mb++=(float)maa/(float)mab;
				}
			}
			break;
		case TIFF_SRATIONAL:
			{
				uint32* ma;
				int32 maa;
				uint32 mab;
				float* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					maa=*(int32*)ma;
					ma++;
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					mab=*ma++;
					if (mab==0)
						*mb++=0.0;
					else
						*mb++=(float)maa/(float)mab;
				}
			}
			break;
		case TIFF_DOUBLE:
			{
				double* ma;
				float* mb;
				uint32 n;
				if (tif->tif_flags&TIFF_SWAB)
					TIFFSwabArrayOfLong8((uint64*)origdata,count);
				TIFFCvtIEEEDoubleToNative(tif,count,(double*)origdata);
				ma=(double*)origdata;
				mb=data;
				for (n=0; n<count; n++)
                                {
                                    double val = *ma++;
                                    if( val > FLT_MAX )
                                        val = FLT_MAX;
                                    else if( val < -FLT_MAX )
                                        val = -FLT_MAX;
                                    *mb++=(float)val;
                                }
			}
			break;
	}
	_TIFFfree(origdata);
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryDoubleArray(TIFF* tif, TIFFDirEntry* direntry, double** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	double* data;
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
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,8,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_DOUBLE:
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfLong8((uint64*)origdata,count);
			TIFFCvtIEEEDoubleToNative(tif,count,(double*)origdata);
			*value=(double*)origdata;
			return(TIFFReadDirEntryErrOk);
	}
	data=(double*)_TIFFmalloc(count*sizeof(double));
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_BYTE:
			{
				uint8* ma;
				double* mb;
				uint32 n;
				ma=(uint8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(double)(*ma++);
			}
			break;
		case TIFF_SBYTE:
			{
				int8* ma;
				double* mb;
				uint32 n;
				ma=(int8*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(double)(*ma++);
			}
			break;
		case TIFF_SHORT:
			{
				uint16* ma;
				double* mb;
				uint32 n;
				ma=(uint16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort(ma);
					*mb++=(double)(*ma++);
				}
			}
			break;
		case TIFF_SSHORT:
			{
				int16* ma;
				double* mb;
				uint32 n;
				ma=(int16*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabShort((uint16*)ma);
					*mb++=(double)(*ma++);
				}
			}
			break;
		case TIFF_LONG:
			{
				uint32* ma;
				double* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					*mb++=(double)(*ma++);
				}
			}
			break;
		case TIFF_SLONG:
			{
				int32* ma;
				double* mb;
				uint32 n;
				ma=(int32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong((uint32*)ma);
					*mb++=(double)(*ma++);
				}
			}
			break;
		case TIFF_LONG8:
			{
				uint64* ma;
				double* mb;
				uint32 n;
				ma=(uint64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
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
				int64* ma;
				double* mb;
				uint32 n;
				ma=(int64*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong8((uint64*)ma);
					*mb++=(double)(*ma++);
				}
			}
			break;
		case TIFF_RATIONAL:
			{
				uint32* ma;
				uint32 maa;
				uint32 mab;
				double* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					maa=*ma++;
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					mab=*ma++;
					if (mab==0)
						*mb++=0.0;
					else
						*mb++=(double)maa/(double)mab;
				}
			}
			break;
		case TIFF_SRATIONAL:
			{
				uint32* ma;
				int32 maa;
				uint32 mab;
				double* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					maa=*(int32*)ma;
					ma++;
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					mab=*ma++;
					if (mab==0)
						*mb++=0.0;
					else
						*mb++=(double)maa/(double)mab;
				}
			}
			break;
		case TIFF_FLOAT:
			{
				float* ma;
				double* mb;
				uint32 n;
				if (tif->tif_flags&TIFF_SWAB)
					TIFFSwabArrayOfLong((uint32*)origdata,count);  
				TIFFCvtIEEEFloatToNative(tif,count,(float*)origdata);
				ma=(float*)origdata;
				mb=data;
				for (n=0; n<count; n++)
					*mb++=(double)(*ma++);
			}
			break;
	}
	_TIFFfree(origdata);
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryIfd8Array(TIFF* tif, TIFFDirEntry* direntry, uint64** value)
{
	enum TIFFReadDirEntryErr err;
	uint32 count;
	void* origdata;
	uint64* data;
	switch (direntry->tdir_type)
	{
		case TIFF_LONG:
		case TIFF_LONG8:
		case TIFF_IFD:
		case TIFF_IFD8:
			break;
		default:
			return(TIFFReadDirEntryErrType);
	}
	err=TIFFReadDirEntryArray(tif,direntry,&count,8,&origdata);
	if ((err!=TIFFReadDirEntryErrOk)||(origdata==0))
	{
		*value=0;
		return(err);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_LONG8:
		case TIFF_IFD8:
			*value=(uint64*)origdata;
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabArrayOfLong8(*value,count);
			return(TIFFReadDirEntryErrOk);
	}
	data=(uint64*)_TIFFmalloc(count*8);
	if (data==0)
	{
		_TIFFfree(origdata);
		return(TIFFReadDirEntryErrAlloc);
	}
	switch (direntry->tdir_type)
	{
		case TIFF_LONG:
		case TIFF_IFD:
			{
				uint32* ma;
				uint64* mb;
				uint32 n;
				ma=(uint32*)origdata;
				mb=data;
				for (n=0; n<count; n++)
				{
					if (tif->tif_flags&TIFF_SWAB)
						TIFFSwabLong(ma);
					*mb++=(uint64)(*ma++);
				}
			}
			break;
	}
	_TIFFfree(origdata);
	*value=data;
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryPersampleShort(TIFF* tif, TIFFDirEntry* direntry, uint16* value)
{
	enum TIFFReadDirEntryErr err;
	uint16* m;
	uint16* na;
	uint16 nb;
	if (direntry->tdir_count<(uint64)tif->tif_dir.td_samplesperpixel)
		return(TIFFReadDirEntryErrCount);
	err=TIFFReadDirEntryShortArray(tif,direntry,&m);
	if (err!=TIFFReadDirEntryErrOk || m == NULL)
		return(err);
	na=m;
	nb=tif->tif_dir.td_samplesperpixel;
	*value=*na++;
	nb--;
	while (nb>0)
	{
		if (*na++!=*value)
		{
			err=TIFFReadDirEntryErrPsdif;
			break;
		}
		nb--;
	}
	_TIFFfree(m);
	return(err);
}

#if 0
static enum TIFFReadDirEntryErr TIFFReadDirEntryPersampleDouble(TIFF* tif, TIFFDirEntry* direntry, double* value)
{
	enum TIFFReadDirEntryErr err;
	double* m;
	double* na;
	uint16 nb;
	if (direntry->tdir_count<(uint64)tif->tif_dir.td_samplesperpixel)
		return(TIFFReadDirEntryErrCount);
	err=TIFFReadDirEntryDoubleArray(tif,direntry,&m);
	if (err!=TIFFReadDirEntryErrOk)
		return(err);
	na=m;
	nb=tif->tif_dir.td_samplesperpixel;
	*value=*na++;
	nb--;
	while (nb>0)
	{
		if (*na++!=*value)
		{
			err=TIFFReadDirEntryErrPsdif;
			break;
		}
		nb--;
	}
	_TIFFfree(m);
	return(err);
}
#endif

static void TIFFReadDirEntryCheckedByte(TIFF* tif, TIFFDirEntry* direntry, uint8* value)
{
	(void) tif;
	*value=*(uint8*)(&direntry->tdir_offset);
}

static void TIFFReadDirEntryCheckedSbyte(TIFF* tif, TIFFDirEntry* direntry, int8* value)
{
	(void) tif;
	*value=*(int8*)(&direntry->tdir_offset);
}

static void TIFFReadDirEntryCheckedShort(TIFF* tif, TIFFDirEntry* direntry, uint16* value)
{
	*value = direntry->tdir_offset.toff_short;
	/* *value=*(uint16*)(&direntry->tdir_offset); */
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabShort(value);
}

static void TIFFReadDirEntryCheckedSshort(TIFF* tif, TIFFDirEntry* direntry, int16* value)
{
	*value=*(int16*)(&direntry->tdir_offset);
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabShort((uint16*)value);
}

static void TIFFReadDirEntryCheckedLong(TIFF* tif, TIFFDirEntry* direntry, uint32* value)
{
	*value=*(uint32*)(&direntry->tdir_offset);
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabLong(value);
}

static void TIFFReadDirEntryCheckedSlong(TIFF* tif, TIFFDirEntry* direntry, int32* value)
{
	*value=*(int32*)(&direntry->tdir_offset);
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabLong((uint32*)value);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedLong8(TIFF* tif, TIFFDirEntry* direntry, uint64* value)
{
	if (!(tif->tif_flags&TIFF_BIGTIFF))
	{
		enum TIFFReadDirEntryErr err;
		uint32 offset = direntry->tdir_offset.toff_long;
		if (tif->tif_flags&TIFF_SWAB)
			TIFFSwabLong(&offset);
		err=TIFFReadDirEntryData(tif,offset,8,value);
		if (err!=TIFFReadDirEntryErrOk)
			return(err);
	}
	else
		*value = direntry->tdir_offset.toff_long8;
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabLong8(value);
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedSlong8(TIFF* tif, TIFFDirEntry* direntry, int64* value)
{
	if (!(tif->tif_flags&TIFF_BIGTIFF))
	{
		enum TIFFReadDirEntryErr err;
		uint32 offset = direntry->tdir_offset.toff_long;
		if (tif->tif_flags&TIFF_SWAB)
			TIFFSwabLong(&offset);
		err=TIFFReadDirEntryData(tif,offset,8,value);
		if (err!=TIFFReadDirEntryErrOk)
			return(err);
	}
	else
		*value=*(int64*)(&direntry->tdir_offset);
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabLong8((uint64*)value);
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedRational(TIFF* tif, TIFFDirEntry* direntry, double* value)
{
	UInt64Aligned_t m;

	assert(sizeof(double)==8);
	assert(sizeof(uint64)==8);
	assert(sizeof(uint32)==4);
	if (!(tif->tif_flags&TIFF_BIGTIFF))
	{
		enum TIFFReadDirEntryErr err;
		uint32 offset = direntry->tdir_offset.toff_long;
		if (tif->tif_flags&TIFF_SWAB)
			TIFFSwabLong(&offset);
		err=TIFFReadDirEntryData(tif,offset,8,m.i);
		if (err!=TIFFReadDirEntryErrOk)
			return(err);
	}
	else
		m.l = direntry->tdir_offset.toff_long8;
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabArrayOfLong(m.i,2);
        /* Not completely sure what we should do when m.i[1]==0, but some */
        /* sanitizers do not like division by 0.0: */
        /* http://bugzilla.maptools.org/show_bug.cgi?id=2644 */
	if (m.i[0]==0 || m.i[1]==0)
		*value=0.0;
	else
		*value=(double)m.i[0]/(double)m.i[1];
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedSrational(TIFF* tif, TIFFDirEntry* direntry, double* value)
{
	UInt64Aligned_t m;
	assert(sizeof(double)==8);
	assert(sizeof(uint64)==8);
	assert(sizeof(int32)==4);
	assert(sizeof(uint32)==4);
	if (!(tif->tif_flags&TIFF_BIGTIFF))
	{
		enum TIFFReadDirEntryErr err;
		uint32 offset = direntry->tdir_offset.toff_long;
		if (tif->tif_flags&TIFF_SWAB)
			TIFFSwabLong(&offset);
		err=TIFFReadDirEntryData(tif,offset,8,m.i);
		if (err!=TIFFReadDirEntryErrOk)
			return(err);
	}
	else
		m.l=direntry->tdir_offset.toff_long8;
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabArrayOfLong(m.i,2);
        /* Not completely sure what we should do when m.i[1]==0, but some */
        /* sanitizers do not like division by 0.0: */
        /* http://bugzilla.maptools.org/show_bug.cgi?id=2644 */
	if ((int32)m.i[0]==0 || m.i[1]==0)
		*value=0.0;
	else
		*value=(double)((int32)m.i[0])/(double)m.i[1];
	return(TIFFReadDirEntryErrOk);
}

static void TIFFReadDirEntryCheckedFloat(TIFF* tif, TIFFDirEntry* direntry, float* value)
{
         union
	 {
	   float  f;
	   uint32 i;
	 } float_union;
	assert(sizeof(float)==4);
	assert(sizeof(uint32)==4);
	assert(sizeof(float_union)==4);
	float_union.i=*(uint32*)(&direntry->tdir_offset);
	*value=float_union.f;
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabLong((uint32*)value);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckedDouble(TIFF* tif, TIFFDirEntry* direntry, double* value)
{
	assert(sizeof(double)==8);
	assert(sizeof(uint64)==8);
	assert(sizeof(UInt64Aligned_t)==8);
	if (!(tif->tif_flags&TIFF_BIGTIFF))
	{
		enum TIFFReadDirEntryErr err;
		uint32 offset = direntry->tdir_offset.toff_long;
		if (tif->tif_flags&TIFF_SWAB)
			TIFFSwabLong(&offset);
		err=TIFFReadDirEntryData(tif,offset,8,value);
		if (err!=TIFFReadDirEntryErrOk)
			return(err);
	}
	else
	{
	       UInt64Aligned_t uint64_union;
	       uint64_union.l=direntry->tdir_offset.toff_long8;
	       *value=uint64_union.d;
	}
	if (tif->tif_flags&TIFF_SWAB)
		TIFFSwabLong8((uint64*)value);
	return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteSbyte(int8 value)
{
	if (value<0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteShort(uint16 value)
{
	if (value>0xFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteSshort(int16 value)
{
	if ((value<0)||(value>0xFF))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteLong(uint32 value)
{
	if (value>0xFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteSlong(int32 value)
{
	if ((value<0)||(value>0xFF))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteLong8(uint64 value)
{
	if (value>0xFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeByteSlong8(int64 value)
{
	if ((value<0)||(value>0xFF))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteByte(uint8 value)
{
	if (value>0x7F)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteShort(uint16 value)
{
	if (value>0x7F)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteSshort(int16 value)
{
	if ((value<-0x80)||(value>0x7F))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteLong(uint32 value)
{
	if (value>0x7F)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteSlong(int32 value)
{
	if ((value<-0x80)||(value>0x7F))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteLong8(uint64 value)
{
	if (value>0x7F)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSbyteSlong8(int64 value)
{
	if ((value<-0x80)||(value>0x7F))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortSbyte(int8 value)
{
	if (value<0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortSshort(int16 value)
{
	if (value<0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortLong(uint32 value)
{
	if (value>0xFFFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortSlong(int32 value)
{
	if ((value<0)||(value>0xFFFF))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortLong8(uint64 value)
{
	if (value>0xFFFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortSlong8(int64 value)
{
	if ((value<0)||(value>0xFFFF))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortShort(uint16 value)
{
	if (value>0x7FFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortLong(uint32 value)
{
	if (value>0x7FFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortSlong(int32 value)
{
	if ((value<-0x8000)||(value>0x7FFF))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortLong8(uint64 value)
{
	if (value>0x7FFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeSshortSlong8(int64 value)
{
	if ((value<-0x8000)||(value>0x7FFF))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLongSbyte(int8 value)
{
	if (value<0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLongSshort(int16 value)
{
	if (value<0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeLongSlong(int32 value)
{
	if (value<0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

/*
 * Largest 32-bit unsigned integer value.
 */
#define TIFF_UINT32_MAX 0xFFFFFFFFU

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongLong8(uint64 value)
{
	if (value > TIFF_UINT32_MAX)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLongSlong8(int64 value)
{
	if ((value < 0) || (value > (int64) TIFF_UINT32_MAX))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

#undef TIFF_UINT32_MAX

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongLong(uint32 value)
{
	if (value > 0x7FFFFFFFUL)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

/* Check that the 8-byte unsigned value can fit in a 4-byte unsigned range */
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongLong8(uint64 value)
{
	if (value > 0x7FFFFFFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

/* Check that the 8-byte signed value can fit in a 4-byte signed range */
static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlongSlong8(int64 value)
{
        if ((value < 0-((int64) 0x7FFFFFFF+1)) || (value > 0x7FFFFFFF))
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Sbyte(int8 value)
{
	if (value < 0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Sshort(int16 value)
{
	if (value < 0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Slong(int32 value)
{
	if (value < 0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeLong8Slong8(int64 value)
{
	if (value < 0)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

/*
 * Largest 64-bit signed integer value.
 */
#define TIFF_INT64_MAX ((int64)(((uint64) ~0) >> 1))

static enum TIFFReadDirEntryErr
TIFFReadDirEntryCheckRangeSlong8Long8(uint64 value)
{
	if (value > TIFF_INT64_MAX)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}

#undef TIFF_INT64_MAX

static enum TIFFReadDirEntryErr
TIFFReadDirEntryData(TIFF* tif, uint64 offset, tmsize_t size, void* dest)
{
	assert(size>0);
	if (!isMapped(tif)) {
		if (!SeekOK(tif,offset))
			return(TIFFReadDirEntryErrIo);
		if (!ReadOK(tif,dest,size))
			return(TIFFReadDirEntryErrIo);
	} else {
		size_t ma,mb;
		ma=(size_t)offset;
		mb=ma+size;
		if (((uint64)ma!=offset)
		    || (mb < ma)
		    || (mb - ma != (size_t) size)
		    || (mb < (size_t)size)
		    || (mb > (size_t)tif->tif_size)
		    )
			return(TIFFReadDirEntryErrIo);
		_TIFFmemcpy(dest,tif->tif_base+ma,size);
	}
	return(TIFFReadDirEntryErrOk);
}

static void TIFFReadDirEntryOutputErr(TIFF* tif, enum TIFFReadDirEntryErr err, const char* module, const char* tagname, int recover)
{
	if (!recover) {
		switch (err) {
			case TIFFReadDirEntryErrCount:
				TIFFErrorExt(tif->tif_clientdata, module,
					     "Incorrect count for \"%s\"",
					     tagname);
				break;
			case TIFFReadDirEntryErrType:
				TIFFErrorExt(tif->tif_clientdata, module,
					     "Incompatible type for \"%s\"",
					     tagname);
				break;
			case TIFFReadDirEntryErrIo:
				TIFFErrorExt(tif->tif_clientdata, module,
					     "IO error during reading of \"%s\"",
					     tagname);
				break;
			case TIFFReadDirEntryErrRange:
				TIFFErrorExt(tif->tif_clientdata, module,
					     "Incorrect value for \"%s\"",
					     tagname);
				break;
			case TIFFReadDirEntryErrPsdif:
				TIFFErrorExt(tif->tif_clientdata, module,
			"Cannot handle different values per sample for \"%s\"",
					     tagname);
				break;
			case TIFFReadDirEntryErrSizesan:
				TIFFErrorExt(tif->tif_clientdata, module,
				"Sanity check on size of \"%s\" value failed",
					     tagname);
				break;
			case TIFFReadDirEntryErrAlloc:
				TIFFErrorExt(tif->tif_clientdata, module,
					     "Out of memory reading of \"%s\"",
					     tagname);
				break;
			default:
				assert(0);   /* we should never get here */
				break;
		}
	} else {
		switch (err) {
			case TIFFReadDirEntryErrCount:
				TIFFWarningExt(tif->tif_clientdata, module,
				"Incorrect count for \"%s\"; tag ignored",
					     tagname);
				break;
			case TIFFReadDirEntryErrType:
				TIFFWarningExt(tif->tif_clientdata, module,
				"Incompatible type for \"%s\"; tag ignored",
					       tagname);
				break;
			case TIFFReadDirEntryErrIo:
				TIFFWarningExt(tif->tif_clientdata, module,
			"IO error during reading of \"%s\"; tag ignored",
					       tagname);
				break;
			case TIFFReadDirEntryErrRange:
				TIFFWarningExt(tif->tif_clientdata, module,
				"Incorrect value for \"%s\"; tag ignored",
					       tagname);
				break;
			case TIFFReadDirEntryErrPsdif:
				TIFFWarningExt(tif->tif_clientdata, module,
	"Cannot handle different values per sample for \"%s\"; tag ignored",
					       tagname);
				break;
			case TIFFReadDirEntryErrSizesan:
				TIFFWarningExt(tif->tif_clientdata, module,
		"Sanity check on size of \"%s\" value failed; tag ignored",
					       tagname);
				break;
			case TIFFReadDirEntryErrAlloc:
				TIFFWarningExt(tif->tif_clientdata, module,
				"Out of memory reading of \"%s\"; tag ignored",
					       tagname);
				break;
			default:
				assert(0);   /* we should never get here */
				break;
		}
	}
}

/*
 * Return the maximum number of color channels specified for a given photometric
 * type. 0 is returned if photometric type isn't supported or no default value
 * is defined by the specification.
 */
static int _TIFFGetMaxColorChannels( uint16 photometric )
{
    switch (photometric) {
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

/*
 * Read the next TIFF directory from a file and convert it to the internal
 * format. We read directories sequentially.
 */
int
TIFFReadDirectory(TIFF* tif)
{
	static const char module[] = "TIFFReadDirectory";
	TIFFDirEntry* dir;
	uint16 dircount;
	TIFFDirEntry* dp;
	uint16 di;
	const TIFFField* fip;
	uint32 fii=FAILED_FII;
        toff_t nextdiroff;
    int bitspersample_read = FALSE;
        int color_channels;

	tif->tif_diroff=tif->tif_nextdiroff;
	if (!TIFFCheckDirOffset(tif,tif->tif_nextdiroff))
		return 0;           /* last offset or bad offset (IFD looping) */
	(*tif->tif_cleanup)(tif);   /* cleanup any previous compression state */
	tif->tif_curdir++;
        nextdiroff = tif->tif_nextdiroff;
	dircount=TIFFFetchDirectory(tif,nextdiroff,&dir,&tif->tif_nextdiroff);
	if (!dircount)
	{
		TIFFErrorExt(tif->tif_clientdata,module,
		    "Failed to read directory at offset " TIFF_UINT64_FORMAT,nextdiroff);
		return 0;
	}
	TIFFReadDirectoryCheckOrder(tif,dir,dircount);

        /*
         * Mark duplicates of any tag to be ignored (bugzilla 1994)
         * to avoid certain pathological problems.
         */
	{
		TIFFDirEntry* ma;
		uint16 mb;
		for (ma=dir, mb=0; mb<dircount; ma++, mb++)
		{
			TIFFDirEntry* na;
			uint16 nb;
			for (na=ma+1, nb=mb+1; nb<dircount; na++, nb++)
			{
				if (ma->tdir_tag==na->tdir_tag)
					na->tdir_tag=IGNORE;
			}
		}
	}
        
	tif->tif_flags &= ~TIFF_BEENWRITING;    /* reset before new dir */
	tif->tif_flags &= ~TIFF_BUF4WRITE;      /* reset before new dir */
	/* free any old stuff and reinit */
	TIFFFreeDirectory(tif);
	TIFFDefaultDirectory(tif);
	/*
	 * Electronic Arts writes gray-scale TIFF files
	 * without a PlanarConfiguration directory entry.
	 * Thus we setup a default value here, even though
	 * the TIFF spec says there is no default value.
	 */
	TIFFSetField(tif,TIFFTAG_PLANARCONFIG,PLANARCONFIG_CONTIG);
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
	dp=TIFFReadDirectoryFindEntry(tif,dir,dircount,TIFFTAG_SAMPLESPERPIXEL);
	if (dp)
	{
		if (!TIFFFetchNormalTag(tif,dp,0))
			goto bad;
		dp->tdir_tag=IGNORE;
	}
	dp=TIFFReadDirectoryFindEntry(tif,dir,dircount,TIFFTAG_COMPRESSION);
	if (dp)
	{
		/*
		 * The 5.0 spec says the Compression tag has one value, while
		 * earlier specs say it has one value per sample.  Because of
		 * this, we accept the tag if one value is supplied with either
		 * count.
		 */
		uint16 value;
		enum TIFFReadDirEntryErr err;
		err=TIFFReadDirEntryShort(tif,dp,&value);
		if (err==TIFFReadDirEntryErrCount)
			err=TIFFReadDirEntryPersampleShort(tif,dp,&value);
		if (err!=TIFFReadDirEntryErrOk)
		{
			TIFFReadDirEntryOutputErr(tif,err,module,"Compression",0);
			goto bad;
		}
		if (!TIFFSetField(tif,TIFFTAG_COMPRESSION,value))
			goto bad;
		dp->tdir_tag=IGNORE;
	}
	else
	{
		if (!TIFFSetField(tif,TIFFTAG_COMPRESSION,COMPRESSION_NONE))
			goto bad;
	}
	/*
	 * First real pass over the directory.
	 */
	for (di=0, dp=dir; di<dircount; di++, dp++)
	{
		if (dp->tdir_tag!=IGNORE)
		{
			TIFFReadDirectoryFindFieldInfo(tif,dp->tdir_tag,&fii);
			if (fii == FAILED_FII)
			{
				TIFFWarningExt(tif->tif_clientdata, module,
				    "Unknown field with tag %d (0x%x) encountered",
				    dp->tdir_tag,dp->tdir_tag);
                                /* the following knowingly leaks the 
                                   anonymous field structure */
				if (!_TIFFMergeFields(tif,
					_TIFFCreateAnonField(tif,
						dp->tdir_tag,
						(TIFFDataType) dp->tdir_type),
					1)) {
					TIFFWarningExt(tif->tif_clientdata,
					    module,
					    "Registering anonymous field with tag %d (0x%x) failed",
					    dp->tdir_tag,
					    dp->tdir_tag);
					dp->tdir_tag=IGNORE;
				} else {
					TIFFReadDirectoryFindFieldInfo(tif,dp->tdir_tag,&fii);
					assert(fii != FAILED_FII);
				}
			}
		}
		if (dp->tdir_tag!=IGNORE)
		{
			fip=tif->tif_fields[fii];
			if (fip->field_bit==FIELD_IGNORE)
				dp->tdir_tag=IGNORE;
			else
			{
				switch (dp->tdir_tag)
				{
					case TIFFTAG_STRIPOFFSETS:
					case TIFFTAG_STRIPBYTECOUNTS:
					case TIFFTAG_TILEOFFSETS:
					case TIFFTAG_TILEBYTECOUNTS:
						TIFFSetFieldBit(tif,fip->field_bit);
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
						if (!TIFFFetchNormalTag(tif,dp,0))
							goto bad;
						dp->tdir_tag=IGNORE;
						break;
                                        default:
                                            if( !_TIFFCheckFieldIsValidForCodec(tif, dp->tdir_tag) )
                                                dp->tdir_tag=IGNORE;
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
	if ((tif->tif_dir.td_compression==COMPRESSION_OJPEG)&&
	    (tif->tif_dir.td_planarconfig==PLANARCONFIG_SEPARATE))
	{
        if (!_TIFFFillStriles(tif))
            goto bad;
		dp=TIFFReadDirectoryFindEntry(tif,dir,dircount,TIFFTAG_STRIPOFFSETS);
		if ((dp!=0)&&(dp->tdir_count==1))
		{
			dp=TIFFReadDirectoryFindEntry(tif,dir,dircount,
			    TIFFTAG_STRIPBYTECOUNTS);
			if ((dp!=0)&&(dp->tdir_count==1))
			{
				tif->tif_dir.td_planarconfig=PLANARCONFIG_CONTIG;
				TIFFWarningExt(tif->tif_clientdata,module,
				    "Planarconfig tag value assumed incorrect, "
				    "assuming data is contig instead of chunky");
			}
		}
	}
	/*
	 * Allocate directory structure and setup defaults.
	 */
	if (!TIFFFieldSet(tif,FIELD_IMAGEDIMENSIONS))
	{
		MissingRequired(tif,"ImageLength");
		goto bad;
	}
	/*
	 * Setup appropriate structures (by strip or by tile)
	 */
	if (!TIFFFieldSet(tif, FIELD_TILEDIMENSIONS)) {
		tif->tif_dir.td_nstrips = TIFFNumberOfStrips(tif);  
		tif->tif_dir.td_tilewidth = tif->tif_dir.td_imagewidth;
		tif->tif_dir.td_tilelength = tif->tif_dir.td_rowsperstrip;
		tif->tif_dir.td_tiledepth = tif->tif_dir.td_imagedepth;
		tif->tif_flags &= ~TIFF_ISTILED;
	} else {
		tif->tif_dir.td_nstrips = TIFFNumberOfTiles(tif);
		tif->tif_flags |= TIFF_ISTILED;
	}
	if (!tif->tif_dir.td_nstrips) {
		TIFFErrorExt(tif->tif_clientdata, module,
		    "Cannot handle zero number of %s",
		    isTiled(tif) ? "tiles" : "strips");
		goto bad;
	}
	tif->tif_dir.td_stripsperimage = tif->tif_dir.td_nstrips;
	if (tif->tif_dir.td_planarconfig == PLANARCONFIG_SEPARATE)
		tif->tif_dir.td_stripsperimage /= tif->tif_dir.td_samplesperpixel;
	if (!TIFFFieldSet(tif, FIELD_STRIPOFFSETS)) {
#ifdef OJPEG_SUPPORT
		if ((tif->tif_dir.td_compression==COMPRESSION_OJPEG) &&
		    (isTiled(tif)==0) &&
		    (tif->tif_dir.td_nstrips==1)) {
			/*
			 * XXX: OJPEG hack.
			 * If a) compression is OJPEG, b) it's not a tiled TIFF,
			 * and c) the number of strips is 1,
			 * then we tolerate the absence of stripoffsets tag,
			 * because, presumably, all required data is in the
			 * JpegInterchangeFormat stream.
			 */
			TIFFSetFieldBit(tif, FIELD_STRIPOFFSETS);
		} else
#endif
        {
			MissingRequired(tif,
				isTiled(tif) ? "TileOffsets" : "StripOffsets");
			goto bad;
		}
	}
	/*
	 * Second pass: extract other information.
	 */
	for (di=0, dp=dir; di<dircount; di++, dp++)
	{
		switch (dp->tdir_tag)
		{
			case IGNORE:
				break;
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
					uint16 value;
					enum TIFFReadDirEntryErr err;
					err=TIFFReadDirEntryShort(tif,dp,&value);
					if (err==TIFFReadDirEntryErrCount)
						err=TIFFReadDirEntryPersampleShort(tif,dp,&value);
					if (err!=TIFFReadDirEntryErrOk)
					{
						fip = TIFFFieldWithTag(tif,dp->tdir_tag);
						TIFFReadDirEntryOutputErr(tif,err,module,fip ? fip->field_name : "unknown tagname",0);
						goto bad;
					}
					if (!TIFFSetField(tif,dp->tdir_tag,value))
						goto bad;
                    if( dp->tdir_tag == TIFFTAG_BITSPERSAMPLE )
                        bitspersample_read = TRUE;
				}
				break;
			case TIFFTAG_SMINSAMPLEVALUE:
			case TIFFTAG_SMAXSAMPLEVALUE:
				{

					double *data = NULL;
					enum TIFFReadDirEntryErr err;
					uint32 saved_flags;
					int m;
					if (dp->tdir_count != (uint64)tif->tif_dir.td_samplesperpixel)
						err = TIFFReadDirEntryErrCount;
					else
						err = TIFFReadDirEntryDoubleArray(tif, dp, &data);
					if (err!=TIFFReadDirEntryErrOk)
					{
						fip = TIFFFieldWithTag(tif,dp->tdir_tag);
						TIFFReadDirEntryOutputErr(tif,err,module,fip ? fip->field_name : "unknown tagname",0);
						goto bad;
					}
					saved_flags = tif->tif_flags;
					tif->tif_flags |= TIFF_PERSAMPLE;
					m = TIFFSetField(tif,dp->tdir_tag,data);
					tif->tif_flags = saved_flags;
					_TIFFfree(data);
					if (!m)
						goto bad;
				}
				break;
			case TIFFTAG_STRIPOFFSETS:
			case TIFFTAG_TILEOFFSETS:
#if defined(DEFER_STRILE_LOAD)
                                _TIFFmemcpy( &(tif->tif_dir.td_stripoffset_entry),
                                             dp, sizeof(TIFFDirEntry) );
#else                          
                                if( tif->tif_dir.td_stripoffset != NULL )
                                {
                                    TIFFErrorExt(tif->tif_clientdata, module,
                                        "tif->tif_dir.td_stripoffset is "
                                        "already allocated. Likely duplicated "
                                        "StripOffsets/TileOffsets tag");
                                    goto bad;
                                }
				if (!TIFFFetchStripThing(tif,dp,tif->tif_dir.td_nstrips,&tif->tif_dir.td_stripoffset))  
					goto bad;
#endif                                
				break;
			case TIFFTAG_STRIPBYTECOUNTS:
			case TIFFTAG_TILEBYTECOUNTS:
#if defined(DEFER_STRILE_LOAD)
                                _TIFFmemcpy( &(tif->tif_dir.td_stripbytecount_entry),
                                             dp, sizeof(TIFFDirEntry) );
#else                          
                                if( tif->tif_dir.td_stripbytecount != NULL )
                                {
                                    TIFFErrorExt(tif->tif_clientdata, module,
                                        "tif->tif_dir.td_stripbytecount is "
                                        "already allocated. Likely duplicated "
                                        "StripByteCounts/TileByteCounts tag");
                                    goto bad;
                                }
                                if (!TIFFFetchStripThing(tif,dp,tif->tif_dir.td_nstrips,&tif->tif_dir.td_stripbytecount))  
					goto bad;
#endif                                
				break;
			case TIFFTAG_COLORMAP:
			case TIFFTAG_TRANSFERFUNCTION:
				{
					enum TIFFReadDirEntryErr err;
					uint32 countpersample;
					uint32 countrequired;
					uint32 incrementpersample;
					uint16* value=NULL;
                    /* It would be dangerous to instantiate those tag values */
                    /* since if td_bitspersample has not yet been read (due to */
                    /* unordered tags), it could be read afterwards with a */
                    /* values greater than the default one (1), which may cause */
                    /* crashes in user code */
                    if( !bitspersample_read )
                    {
                        fip = TIFFFieldWithTag(tif,dp->tdir_tag);
                        TIFFWarningExt(tif->tif_clientdata,module,
                                       "Ignoring %s since BitsPerSample tag not found",
                                       fip ? fip->field_name : "unknown tagname");
                        continue;
                    }
					/* ColorMap or TransferFunction for high bit */
					/* depths do not make much sense and could be */
					/* used as a denial of service vector */
					if (tif->tif_dir.td_bitspersample > 24)
					{
					    fip = TIFFFieldWithTag(tif,dp->tdir_tag);
					    TIFFWarningExt(tif->tif_clientdata,module,
						"Ignoring %s because BitsPerSample=%d>24",
						fip ? fip->field_name : "unknown tagname",
						tif->tif_dir.td_bitspersample);
					    continue;
					}
					countpersample=(1U<<tif->tif_dir.td_bitspersample);
					if ((dp->tdir_tag==TIFFTAG_TRANSFERFUNCTION)&&(dp->tdir_count==(uint64)countpersample))
					{
						countrequired=countpersample;
						incrementpersample=0;
					}
					else
					{
						countrequired=3*countpersample;
						incrementpersample=countpersample;
					}
					if (dp->tdir_count!=(uint64)countrequired)
						err=TIFFReadDirEntryErrCount;
					else
						err=TIFFReadDirEntryShortArray(tif,dp,&value);
					if (err!=TIFFReadDirEntryErrOk)
                    {
						fip = TIFFFieldWithTag(tif,dp->tdir_tag);
						TIFFReadDirEntryOutputErr(tif,err,module,fip ? fip->field_name : "unknown tagname",1);
                    }
					else
					{
						TIFFSetField(tif,dp->tdir_tag,value,value+incrementpersample,value+2*incrementpersample);
						_TIFFfree(value);
					}
				}
				break;
/* BEGIN REV 4.0 COMPATIBILITY */
			case TIFFTAG_OSUBFILETYPE:
				{
					uint16 valueo;
					uint32 value;
					if (TIFFReadDirEntryShort(tif,dp,&valueo)==TIFFReadDirEntryErrOk)
					{
						switch (valueo)
						{
							case OFILETYPE_REDUCEDIMAGE: value=FILETYPE_REDUCEDIMAGE; break;
							case OFILETYPE_PAGE: value=FILETYPE_PAGE; break;
							default: value=0; break;
						}
						if (value!=0)
							TIFFSetField(tif,TIFFTAG_SUBFILETYPE,value);
					}
				}
				break;
/* END REV 4.0 COMPATIBILITY */
			default:
				(void) TIFFFetchNormalTag(tif, dp, TRUE);
				break;
		}
	}
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
	if (tif->tif_dir.td_compression==COMPRESSION_OJPEG)
	{
		if (!TIFFFieldSet(tif,FIELD_PHOTOMETRIC))
		{
			TIFFWarningExt(tif->tif_clientdata, module,
			    "Photometric tag is missing, assuming data is YCbCr");
			if (!TIFFSetField(tif,TIFFTAG_PHOTOMETRIC,PHOTOMETRIC_YCBCR))
				goto bad;
		}
		else if (tif->tif_dir.td_photometric==PHOTOMETRIC_RGB)
		{
			tif->tif_dir.td_photometric=PHOTOMETRIC_YCBCR;
			TIFFWarningExt(tif->tif_clientdata, module,
			    "Photometric tag value assumed incorrect, "
			    "assuming data is YCbCr instead of RGB");
		}
		if (!TIFFFieldSet(tif,FIELD_BITSPERSAMPLE))
		{
			TIFFWarningExt(tif->tif_clientdata,module,
			    "BitsPerSample tag is missing, assuming 8 bits per sample");
			if (!TIFFSetField(tif,TIFFTAG_BITSPERSAMPLE,8))
				goto bad;
		}
		if (!TIFFFieldSet(tif,FIELD_SAMPLESPERPIXEL))
		{
			if (tif->tif_dir.td_photometric==PHOTOMETRIC_RGB)
			{
				TIFFWarningExt(tif->tif_clientdata,module,
				    "SamplesPerPixel tag is missing, "
				    "assuming correct SamplesPerPixel value is 3");
				if (!TIFFSetField(tif,TIFFTAG_SAMPLESPERPIXEL,3))
					goto bad;
			}
			if (tif->tif_dir.td_photometric==PHOTOMETRIC_YCBCR)
			{
				TIFFWarningExt(tif->tif_clientdata,module,
				    "SamplesPerPixel tag is missing, "
				    "applying correct SamplesPerPixel value of 3");
				if (!TIFFSetField(tif,TIFFTAG_SAMPLESPERPIXEL,3))
					goto bad;
			}
			else if ((tif->tif_dir.td_photometric==PHOTOMETRIC_MINISWHITE)
				 || (tif->tif_dir.td_photometric==PHOTOMETRIC_MINISBLACK))
			{
				/*
				 * SamplesPerPixel tag is missing, but is not required
				 * by spec.  Assume correct SamplesPerPixel value of 1.
				 */
				if (!TIFFSetField(tif,TIFFTAG_SAMPLESPERPIXEL,1))
					goto bad;
			}
		}
	}

	/*
	 * Make sure all non-color channels are extrasamples.
	 * If it's not the case, define them as such.
	 */
        color_channels = _TIFFGetMaxColorChannels(tif->tif_dir.td_photometric);
        if (color_channels && tif->tif_dir.td_samplesperpixel - tif->tif_dir.td_extrasamples > color_channels) {
                uint16 old_extrasamples;
                uint16 *new_sampleinfo;

                TIFFWarningExt(tif->tif_clientdata,module, "Sum of Photometric type-related "
                    "color channels and ExtraSamples doesn't match SamplesPerPixel. "
                    "Defining non-color channels as ExtraSamples.");

                old_extrasamples = tif->tif_dir.td_extrasamples;
                tif->tif_dir.td_extrasamples = (uint16) (tif->tif_dir.td_samplesperpixel - color_channels);

                // sampleinfo should contain information relative to these new extra samples
                new_sampleinfo = (uint16*) _TIFFcalloc(tif->tif_dir.td_extrasamples, sizeof(uint16));
                if (!new_sampleinfo) {
                    TIFFErrorExt(tif->tif_clientdata, module, "Failed to allocate memory for "
                                "temporary new sampleinfo array (%d 16 bit elements)",
                                tif->tif_dir.td_extrasamples);
                    goto bad;
                }

                memcpy(new_sampleinfo, tif->tif_dir.td_sampleinfo, old_extrasamples * sizeof(uint16));
                _TIFFsetShortArray(&tif->tif_dir.td_sampleinfo, new_sampleinfo, tif->tif_dir.td_extrasamples);
                _TIFFfree(new_sampleinfo);
        }

	/*
	 * Verify Palette image has a Colormap.
	 */
	if (tif->tif_dir.td_photometric == PHOTOMETRIC_PALETTE &&
	    !TIFFFieldSet(tif, FIELD_COLORMAP)) {
		if ( tif->tif_dir.td_bitspersample>=8 && tif->tif_dir.td_samplesperpixel==3)
			tif->tif_dir.td_photometric = PHOTOMETRIC_RGB;
		else if (tif->tif_dir.td_bitspersample>=8)
			tif->tif_dir.td_photometric = PHOTOMETRIC_MINISBLACK;
		else {
			MissingRequired(tif, "Colormap");
			goto bad;
		}
	}
	/*
	 * OJPEG hack:
	 * We do no further messing with strip/tile offsets/bytecounts in OJPEG
	 * TIFFs
	 */
	if (tif->tif_dir.td_compression!=COMPRESSION_OJPEG)
	{
		/*
		 * Attempt to deal with a missing StripByteCounts tag.
		 */
		if (!TIFFFieldSet(tif, FIELD_STRIPBYTECOUNTS)) {
			/*
			 * Some manufacturers violate the spec by not giving
			 * the size of the strips.  In this case, assume there
			 * is one uncompressed strip of data.
			 */
			if ((tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG &&
			    tif->tif_dir.td_nstrips > 1) ||
			    (tif->tif_dir.td_planarconfig == PLANARCONFIG_SEPARATE &&
			     tif->tif_dir.td_nstrips != (uint32)tif->tif_dir.td_samplesperpixel)) {
			    MissingRequired(tif, "StripByteCounts");
			    goto bad;
			}
			TIFFWarningExt(tif->tif_clientdata, module,
				"TIFF directory is missing required "
				"\"StripByteCounts\" field, calculating from imagelength");
			if (EstimateStripByteCounts(tif, dir, dircount) < 0)
			    goto bad;
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
		#define	BYTECOUNTLOOKSBAD \
		    ( (tif->tif_dir.td_stripbytecount[0] == 0 && tif->tif_dir.td_stripoffset[0] != 0) || \
		      (tif->tif_dir.td_compression == COMPRESSION_NONE && \
		       (tif->tif_dir.td_stripoffset[0] <= TIFFGetFileSize(tif) && \
		        tif->tif_dir.td_stripbytecount[0] > TIFFGetFileSize(tif) - tif->tif_dir.td_stripoffset[0])) || \
		      (tif->tif_mode == O_RDONLY && \
		       tif->tif_dir.td_compression == COMPRESSION_NONE && \
		       tif->tif_dir.td_stripbytecount[0] < TIFFScanlineSize64(tif) * tif->tif_dir.td_imagelength) )

		} else if (tif->tif_dir.td_nstrips == 1
                           && !(tif->tif_flags&TIFF_ISTILED)
                           && _TIFFFillStriles(tif)
			   && tif->tif_dir.td_stripoffset[0] != 0
			   && BYTECOUNTLOOKSBAD) {
			/*
			 * XXX: Plexus (and others) sometimes give a value of
			 * zero for a tag when they don't know what the
			 * correct value is!  Try and handle the simple case
			 * of estimating the size of a one strip image.
			 */
			TIFFWarningExt(tif->tif_clientdata, module,
			    "Bogus \"StripByteCounts\" field, ignoring and calculating from imagelength");
			if(EstimateStripByteCounts(tif, dir, dircount) < 0)
			    goto bad;

#if !defined(DEFER_STRILE_LOAD)
		} else if (tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG
			   && tif->tif_dir.td_nstrips > 2
			   && tif->tif_dir.td_compression == COMPRESSION_NONE
			   && tif->tif_dir.td_stripbytecount[0] != tif->tif_dir.td_stripbytecount[1]
			   && tif->tif_dir.td_stripbytecount[0] != 0
			   && tif->tif_dir.td_stripbytecount[1] != 0 ) {
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
			TIFFWarningExt(tif->tif_clientdata, module,
			    "Wrong \"StripByteCounts\" field, ignoring and calculating from imagelength");
			if (EstimateStripByteCounts(tif, dir, dircount) < 0)
			    goto bad;
#endif /* !defined(DEFER_STRILE_LOAD) */                        
		}
	}
	if (dir)
	{
		_TIFFfree(dir);
		dir=NULL;
	}
	if (!TIFFFieldSet(tif, FIELD_MAXSAMPLEVALUE))
	{
		if (tif->tif_dir.td_bitspersample>=16)
			tif->tif_dir.td_maxsamplevalue=0xFFFF;
		else
			tif->tif_dir.td_maxsamplevalue = (uint16)((1L<<tif->tif_dir.td_bitspersample)-1);
	}
	/*
	 * XXX: We can optimize checking for the strip bounds using the sorted
	 * bytecounts array. See also comments for TIFFAppendToStrip()
	 * function in tif_write.c.
	 */
#if !defined(DEFER_STRILE_LOAD)        
	if (tif->tif_dir.td_nstrips > 1) {
		uint32 strip;

		tif->tif_dir.td_stripbytecountsorted = 1;
		for (strip = 1; strip < tif->tif_dir.td_nstrips; strip++) {
			if (tif->tif_dir.td_stripoffset[strip - 1] >
			    tif->tif_dir.td_stripoffset[strip]) {
				tif->tif_dir.td_stripbytecountsorted = 0;
				break;
			}
		}
	}
#endif /* !defined(DEFER_STRILE_LOAD) */
        
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
	if ((tif->tif_dir.td_planarconfig==PLANARCONFIG_CONTIG)&&
	    (tif->tif_dir.td_nstrips==1)&&
	    (tif->tif_dir.td_compression==COMPRESSION_NONE)&&  
	    ((tif->tif_flags&(TIFF_STRIPCHOP|TIFF_ISTILED))==TIFF_STRIPCHOP))
    {
        if ( !_TIFFFillStriles(tif) || !tif->tif_dir.td_stripbytecount )
            return 0;
		ChopUpSingleUncompressedStrip(tif);
    }

        /* There are also uncompressed stripped files with strips larger than */
        /* 2 GB, which make them unfriendly with a lot of code. If possible, */
        /* try to expose smaller "virtual" strips. */
        if( tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG &&
            tif->tif_dir.td_compression == COMPRESSION_NONE &&
            (tif->tif_flags&(TIFF_STRIPCHOP|TIFF_ISTILED)) == TIFF_STRIPCHOP &&
            TIFFStripSize64(tif) > 0x7FFFFFFFUL )
        {
            if ( !_TIFFFillStriles(tif) || !tif->tif_dir.td_stripbytecount )
                return 0;
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
	tif->tif_row = (uint32) -1;
	tif->tif_curstrip = (uint32) -1;
	tif->tif_col = (uint32) -1;
	tif->tif_curtile = (uint32) -1;
	tif->tif_tilesize = (tmsize_t) -1;

	tif->tif_scanlinesize = TIFFScanlineSize(tif);
	if (!tif->tif_scanlinesize) {
		TIFFErrorExt(tif->tif_clientdata, module,
		    "Cannot handle zero scanline size");
		return (0);
	}

	if (isTiled(tif)) {
		tif->tif_tilesize = TIFFTileSize(tif);
		if (!tif->tif_tilesize) {
			TIFFErrorExt(tif->tif_clientdata, module,
			     "Cannot handle zero tile size");
			return (0);
		}
	} else {
		if (!TIFFStripSize(tif)) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "Cannot handle zero strip size");
			return (0);
		}
	}
	return (1);
bad:
	if (dir)
		_TIFFfree(dir);
	return (0);
}

static void
TIFFReadDirectoryCheckOrder(TIFF* tif, TIFFDirEntry* dir, uint16 dircount)
{
	static const char module[] = "TIFFReadDirectoryCheckOrder";
	uint16 m;
	uint16 n;
	TIFFDirEntry* o;
	m=0;
	for (n=0, o=dir; n<dircount; n++, o++)
	{
		if (o->tdir_tag<m)
		{
			TIFFWarningExt(tif->tif_clientdata,module,
			    "Invalid TIFF directory; tags are not sorted in ascending order");
			break;
		}
		m=o->tdir_tag+1;
	}
}

static TIFFDirEntry*
TIFFReadDirectoryFindEntry(TIFF* tif, TIFFDirEntry* dir, uint16 dircount, uint16 tagid)
{
	TIFFDirEntry* m;
	uint16 n;
	(void) tif;
	for (m=dir, n=0; n<dircount; m++, n++)
	{
		if (m->tdir_tag==tagid)
			return(m);
	}
	return(0);
}

static void
TIFFReadDirectoryFindFieldInfo(TIFF* tif, uint16 tagid, uint32* fii)
{
	int32 ma,mb,mc;
	ma=-1;
	mc=(int32)tif->tif_nfields;
	while (1)
	{
		if (ma+1==mc)
		{
			*fii = FAILED_FII;
			return;
		}
		mb=(ma+mc)/2;
		if (tif->tif_fields[mb]->field_tag==(uint32)tagid)
			break;
		if (tif->tif_fields[mb]->field_tag<(uint32)tagid)
			ma=mb;
		else
			mc=mb;
	}
	while (1)
	{
		if (mb==0)
			break;
		if (tif->tif_fields[mb-1]->field_tag!=(uint32)tagid)
			break;
		mb--;
	}
	*fii=mb;
}

/*
 * Read custom directory from the arbitrary offset.
 * The code is very similar to TIFFReadDirectory().
 */
int
TIFFReadCustomDirectory(TIFF* tif, toff_t diroff,
			const TIFFFieldArray* infoarray)
{
	static const char module[] = "TIFFReadCustomDirectory";
	TIFFDirEntry* dir;
	uint16 dircount;
	TIFFDirEntry* dp;
	uint16 di;
	const TIFFField* fip;
	uint32 fii;
	_TIFFSetupFields(tif, infoarray);
	dircount=TIFFFetchDirectory(tif,diroff,&dir,NULL);
	if (!dircount)
	{
		TIFFErrorExt(tif->tif_clientdata,module,
		    "Failed to read custom directory at offset " TIFF_UINT64_FORMAT,diroff);
		return 0;
	}
	TIFFFreeDirectory(tif);
	_TIFFmemset(&tif->tif_dir, 0, sizeof(TIFFDirectory));
	TIFFReadDirectoryCheckOrder(tif,dir,dircount);
	for (di=0, dp=dir; di<dircount; di++, dp++)
	{
		TIFFReadDirectoryFindFieldInfo(tif,dp->tdir_tag,&fii);
		if (fii == FAILED_FII)
		{
			TIFFWarningExt(tif->tif_clientdata, module,
			    "Unknown field with tag %d (0x%x) encountered",
			    dp->tdir_tag, dp->tdir_tag);
			if (!_TIFFMergeFields(tif, _TIFFCreateAnonField(tif,
						dp->tdir_tag,
						(TIFFDataType) dp->tdir_type),
					     1)) {
				TIFFWarningExt(tif->tif_clientdata, module,
				    "Registering anonymous field with tag %d (0x%x) failed",
				    dp->tdir_tag, dp->tdir_tag);
				dp->tdir_tag=IGNORE;
			} else {
				TIFFReadDirectoryFindFieldInfo(tif,dp->tdir_tag,&fii);
				assert( fii != FAILED_FII );
			}
		}
		if (dp->tdir_tag!=IGNORE)
		{
			fip=tif->tif_fields[fii];
			if (fip->field_bit==FIELD_IGNORE)
				dp->tdir_tag=IGNORE;
			else
			{
				/* check data type */
				while ((fip->field_type!=TIFF_ANY)&&(fip->field_type!=dp->tdir_type))
				{
					fii++;
					if ((fii==tif->tif_nfields)||
					    (tif->tif_fields[fii]->field_tag!=(uint32)dp->tdir_tag))
					{
						fii=0xFFFF;
						break;
					}
					fip=tif->tif_fields[fii];
				}
				if (fii==0xFFFF)
				{
					TIFFWarningExt(tif->tif_clientdata, module,
					    "Wrong data type %d for \"%s\"; tag ignored",
					    dp->tdir_type,fip->field_name);
					dp->tdir_tag=IGNORE;
				}
				else
				{
					/* check count if known in advance */
					if ((fip->field_readcount!=TIFF_VARIABLE)&&
					    (fip->field_readcount!=TIFF_VARIABLE2))
					{
						uint32 expected;
						if (fip->field_readcount==TIFF_SPP)
							expected=(uint32)tif->tif_dir.td_samplesperpixel;
						else
							expected=(uint32)fip->field_readcount;
						if (!CheckDirCount(tif,dp,expected))
							dp->tdir_tag=IGNORE;
					}
				}
			}
			switch (dp->tdir_tag)
			{
				case IGNORE:
					break;
				case EXIFTAG_SUBJECTDISTANCE:
					(void) TIFFFetchSubjectDistance(tif,dp);
					break;
				default:
					(void) TIFFFetchNormalTag(tif, dp, TRUE);
					break;
			}
		}
	}
	if (dir)
		_TIFFfree(dir);
	return 1;
}

/*
 * EXIF is important special case of custom IFD, so we have a special
 * function to read it.
 */
int
TIFFReadEXIFDirectory(TIFF* tif, toff_t diroff)
{
	const TIFFFieldArray* exifFieldArray;
	exifFieldArray = _TIFFGetExifFields();
	return TIFFReadCustomDirectory(tif, diroff, exifFieldArray);  
}

static int
EstimateStripByteCounts(TIFF* tif, TIFFDirEntry* dir, uint16 dircount)
{
	static const char module[] = "EstimateStripByteCounts";

	TIFFDirEntry *dp;
	TIFFDirectory *td = &tif->tif_dir;
	uint32 strip;

    /* Do not try to load stripbytecount as we will compute it */
        if( !_TIFFFillStrilesInternal( tif, 0 ) )
            return -1;

	if (td->td_stripbytecount)
		_TIFFfree(td->td_stripbytecount);
	td->td_stripbytecount = (uint64*)
	    _TIFFCheckMalloc(tif, td->td_nstrips, sizeof (uint64),
		"for \"StripByteCounts\" array");
        if( td->td_stripbytecount == NULL )
            return -1;

	if (td->td_compression != COMPRESSION_NONE) {
		uint64 space;
		uint64 filesize;
		uint16 n;
		filesize = TIFFGetFileSize(tif);
		if (!(tif->tif_flags&TIFF_BIGTIFF))
			space=sizeof(TIFFHeaderClassic)+2+dircount*12+4;
		else
			space=sizeof(TIFFHeaderBig)+8+dircount*20+8;
		/* calculate amount of space used by indirect values */
		for (dp = dir, n = dircount; n > 0; n--, dp++)
		{
			uint32 typewidth;
			uint64 datasize;
			typewidth = TIFFDataWidth((TIFFDataType) dp->tdir_type);
			if (typewidth == 0) {
				TIFFErrorExt(tif->tif_clientdata, module,
				    "Cannot determine size of unknown tag type %d",
				    dp->tdir_type);
				return -1;
			}
			datasize=(uint64)typewidth*dp->tdir_count;
			if (!(tif->tif_flags&TIFF_BIGTIFF))
			{
				if (datasize<=4)
					datasize=0;
			}
			else
			{
				if (datasize<=8)
					datasize=0;
			}
			space+=datasize;
		}
		if( filesize < space )
                    /* we should perhaps return in error ? */
                    space = filesize;
                else
                    space = filesize - space;
		if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
			space /= td->td_samplesperpixel;
		for (strip = 0; strip < td->td_nstrips; strip++)
			td->td_stripbytecount[strip] = space;
		/*
		 * This gross hack handles the case were the offset to
		 * the last strip is past the place where we think the strip
		 * should begin.  Since a strip of data must be contiguous,
		 * it's safe to assume that we've overestimated the amount
		 * of data in the strip and trim this number back accordingly.
		 */
		strip--;
		if (td->td_stripoffset[strip]+td->td_stripbytecount[strip] > filesize)
			td->td_stripbytecount[strip] = filesize - td->td_stripoffset[strip];
	} else if (isTiled(tif)) {
		uint64 bytespertile = TIFFTileSize64(tif);

		for (strip = 0; strip < td->td_nstrips; strip++)
		    td->td_stripbytecount[strip] = bytespertile;
	} else {
		uint64 rowbytes = TIFFScanlineSize64(tif);
		uint32 rowsperstrip = td->td_imagelength/td->td_stripsperimage;
		for (strip = 0; strip < td->td_nstrips; strip++)
			td->td_stripbytecount[strip] = rowbytes * rowsperstrip;
	}
	TIFFSetFieldBit(tif, FIELD_STRIPBYTECOUNTS);
	if (!TIFFFieldSet(tif, FIELD_ROWSPERSTRIP))
		td->td_rowsperstrip = td->td_imagelength;
	return 1;
}

static void
MissingRequired(TIFF* tif, const char* tagname)
{
	static const char module[] = "MissingRequired";

	TIFFErrorExt(tif->tif_clientdata, module,
	    "TIFF directory is missing required \"%s\" field",
	    tagname);
}

/*
 * Check the directory offset against the list of already seen directory
 * offsets. This is a trick to prevent IFD looping. The one can create TIFF
 * file with looped directory pointers. We will maintain a list of already
 * seen directories and check every IFD offset against that list.
 */
static int
TIFFCheckDirOffset(TIFF* tif, uint64 diroff)
{
	uint16 n;

	if (diroff == 0)			/* no more directories */
		return 0;
	if (tif->tif_dirnumber == 65535) {
	    TIFFErrorExt(tif->tif_clientdata, "TIFFCheckDirOffset",
			 "Cannot handle more than 65535 TIFF directories");
	    return 0;
	}

	for (n = 0; n < tif->tif_dirnumber && tif->tif_dirlist; n++) {
		if (tif->tif_dirlist[n] == diroff)
			return 0;
	}

	tif->tif_dirnumber++;

	if (tif->tif_dirlist == NULL || tif->tif_dirnumber > tif->tif_dirlistsize) {
		uint64* new_dirlist;

		/*
		 * XXX: Reduce memory allocation granularity of the dirlist
		 * array.
		 */
		new_dirlist = (uint64*)_TIFFCheckRealloc(tif, tif->tif_dirlist,
		    tif->tif_dirnumber, 2 * sizeof(uint64), "for IFD list");
		if (!new_dirlist)
			return 0;
		if( tif->tif_dirnumber >= 32768 )
		    tif->tif_dirlistsize = 65535;
		else
		    tif->tif_dirlistsize = 2 * tif->tif_dirnumber;
		tif->tif_dirlist = new_dirlist;
	}

	tif->tif_dirlist[tif->tif_dirnumber - 1] = diroff;

	return 1;
}

/*
 * Check the count field of a directory entry against a known value.  The
 * caller is expected to skip/ignore the tag if there is a mismatch.
 */
static int
CheckDirCount(TIFF* tif, TIFFDirEntry* dir, uint32 count)
{
	if ((uint64)count > dir->tdir_count) {
		const TIFFField* fip = TIFFFieldWithTag(tif, dir->tdir_tag);
		TIFFWarningExt(tif->tif_clientdata, tif->tif_name,
	"incorrect count for field \"%s\" (" TIFF_UINT64_FORMAT ", expecting %u); tag ignored",
		    fip ? fip->field_name : "unknown tagname",
		    dir->tdir_count, count);
		return (0);
	} else if ((uint64)count < dir->tdir_count) {
		const TIFFField* fip = TIFFFieldWithTag(tif, dir->tdir_tag);
		TIFFWarningExt(tif->tif_clientdata, tif->tif_name,
	"incorrect count for field \"%s\" (" TIFF_UINT64_FORMAT ", expecting %u); tag trimmed",
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
static uint16
TIFFFetchDirectory(TIFF* tif, uint64 diroff, TIFFDirEntry** pdir,
                   uint64 *nextdiroff)
{
	static const char module[] = "TIFFFetchDirectory";

	void* origdir;
	uint16 dircount16;
	uint32 dirsize;
	TIFFDirEntry* dir;
	uint8* ma;
	TIFFDirEntry* mb;
	uint16 n;

	assert(pdir);

	tif->tif_diroff = diroff;
	if (nextdiroff)
		*nextdiroff = 0;
	if (!isMapped(tif)) {
		if (!SeekOK(tif, tif->tif_diroff)) {
			TIFFErrorExt(tif->tif_clientdata, module,
				"%s: Seek error accessing TIFF directory",
				tif->tif_name);
			return 0;
		}
		if (!(tif->tif_flags&TIFF_BIGTIFF))
		{
			if (!ReadOK(tif, &dircount16, sizeof (uint16))) {
				TIFFErrorExt(tif->tif_clientdata, module,
				    "%s: Can not read TIFF directory count",
				    tif->tif_name);
				return 0;
			}
			if (tif->tif_flags & TIFF_SWAB)
				TIFFSwabShort(&dircount16);
			if (dircount16>4096)
			{
				TIFFErrorExt(tif->tif_clientdata, module,
				    "Sanity check on directory count failed, this is probably not a valid IFD offset");
				return 0;
			}
			dirsize = 12;
		} else {
			uint64 dircount64;
			if (!ReadOK(tif, &dircount64, sizeof (uint64))) {
				TIFFErrorExt(tif->tif_clientdata, module,
					"%s: Can not read TIFF directory count",
					tif->tif_name);
				return 0;
			}
			if (tif->tif_flags & TIFF_SWAB)
				TIFFSwabLong8(&dircount64);
			if (dircount64>4096)
			{
				TIFFErrorExt(tif->tif_clientdata, module,
				    "Sanity check on directory count failed, this is probably not a valid IFD offset");
				return 0;
			}
			dircount16 = (uint16)dircount64;
			dirsize = 20;
		}
		origdir = _TIFFCheckMalloc(tif, dircount16,
		    dirsize, "to read TIFF directory");
		if (origdir == NULL)
			return 0;
		if (!ReadOK(tif, origdir, (tmsize_t)(dircount16*dirsize))) {
			TIFFErrorExt(tif->tif_clientdata, module,
				"%.100s: Can not read TIFF directory",
				tif->tif_name);
			_TIFFfree(origdir);
			return 0;
		}
		/*
		 * Read offset to next directory for sequential scans if
		 * needed.
		 */
		if (nextdiroff)
		{
			if (!(tif->tif_flags&TIFF_BIGTIFF))
			{
				uint32 nextdiroff32;
				if (!ReadOK(tif, &nextdiroff32, sizeof(uint32)))
					nextdiroff32 = 0;
				if (tif->tif_flags&TIFF_SWAB)
					TIFFSwabLong(&nextdiroff32);
				*nextdiroff=nextdiroff32;
			} else {
				if (!ReadOK(tif, nextdiroff, sizeof(uint64)))
					*nextdiroff = 0;
				if (tif->tif_flags&TIFF_SWAB)
					TIFFSwabLong8(nextdiroff);
			}
		}
	} else {
		tmsize_t m;
		tmsize_t off = (tmsize_t) tif->tif_diroff;
		if ((uint64)off!=tif->tif_diroff)
		{
			TIFFErrorExt(tif->tif_clientdata,module,"Can not read TIFF directory count");
			return(0);
		}

		/*
		 * Check for integer overflow when validating the dir_off,
		 * otherwise a very high offset may cause an OOB read and
		 * crash the client. Make two comparisons instead of
		 *
		 *  off + sizeof(uint16) > tif->tif_size
		 *
		 * to avoid overflow.
		 */
		if (!(tif->tif_flags&TIFF_BIGTIFF))
		{
			m=off+sizeof(uint16);
			if ((m<off)||(m<(tmsize_t)sizeof(uint16))||(m>tif->tif_size)) {
				TIFFErrorExt(tif->tif_clientdata, module,
					"Can not read TIFF directory count");
				return 0;
			} else {
				_TIFFmemcpy(&dircount16, tif->tif_base + off,
					    sizeof(uint16));
			}
			off += sizeof (uint16);
			if (tif->tif_flags & TIFF_SWAB)
				TIFFSwabShort(&dircount16);
			if (dircount16>4096)
			{
				TIFFErrorExt(tif->tif_clientdata, module,
				    "Sanity check on directory count failed, this is probably not a valid IFD offset");
				return 0;
			}
			dirsize = 12;
		}
		else
		{
			uint64 dircount64;
			m=off+sizeof(uint64);
			if ((m<off)||(m<(tmsize_t)sizeof(uint64))||(m>tif->tif_size)) {
				TIFFErrorExt(tif->tif_clientdata, module,
					"Can not read TIFF directory count");
				return 0;
			} else {
				_TIFFmemcpy(&dircount64, tif->tif_base + off,
					    sizeof(uint64));
			}
			off += sizeof (uint64);
			if (tif->tif_flags & TIFF_SWAB)
				TIFFSwabLong8(&dircount64);
			if (dircount64>4096)
			{
				TIFFErrorExt(tif->tif_clientdata, module,
				    "Sanity check on directory count failed, this is probably not a valid IFD offset");
				return 0;
			}
			dircount16 = (uint16)dircount64;
			dirsize = 20;
		}
		if (dircount16 == 0 )
		{
			TIFFErrorExt(tif->tif_clientdata, module,
			             "Sanity check on directory count failed, zero tag directories not supported");
			return 0;
		}
		origdir = _TIFFCheckMalloc(tif, dircount16,
						dirsize,
						"to read TIFF directory");
		if (origdir == NULL)
			return 0;
		m=off+dircount16*dirsize;
		if ((m<off)||(m<(tmsize_t)(dircount16*dirsize))||(m>tif->tif_size)) {
			TIFFErrorExt(tif->tif_clientdata, module,
				     "Can not read TIFF directory");
			_TIFFfree(origdir);
			return 0;
		} else {
			_TIFFmemcpy(origdir, tif->tif_base + off,
				    dircount16 * dirsize);
		}
		if (nextdiroff) {
			off += dircount16 * dirsize;
			if (!(tif->tif_flags&TIFF_BIGTIFF))
			{
				uint32 nextdiroff32;
				m=off+sizeof(uint32);
				if ((m<off)||(m<(tmsize_t)sizeof(uint32))||(m>tif->tif_size))
					nextdiroff32 = 0;
				else
					_TIFFmemcpy(&nextdiroff32, tif->tif_base + off,
						    sizeof (uint32));
				if (tif->tif_flags&TIFF_SWAB)
					TIFFSwabLong(&nextdiroff32);
				*nextdiroff = nextdiroff32;
			}
			else
			{
				m=off+sizeof(uint64);
				if ((m<off)||(m<(tmsize_t)sizeof(uint64))||(m>tif->tif_size))
					*nextdiroff = 0;
				else
					_TIFFmemcpy(nextdiroff, tif->tif_base + off,
						    sizeof (uint64));
				if (tif->tif_flags&TIFF_SWAB)
					TIFFSwabLong8(nextdiroff);
			}
		}
	}
	dir = (TIFFDirEntry*)_TIFFCheckMalloc(tif, dircount16,
						sizeof(TIFFDirEntry),
						"to read TIFF directory");
	if (dir==0)
	{
		_TIFFfree(origdir);
		return 0;
	}
	ma=(uint8*)origdir;
	mb=dir;
	for (n=0; n<dircount16; n++)
	{
		if (tif->tif_flags&TIFF_SWAB)
			TIFFSwabShort((uint16*)ma);
		mb->tdir_tag=*(uint16*)ma;
		ma+=sizeof(uint16);
		if (tif->tif_flags&TIFF_SWAB)
			TIFFSwabShort((uint16*)ma);
		mb->tdir_type=*(uint16*)ma;
		ma+=sizeof(uint16);
		if (!(tif->tif_flags&TIFF_BIGTIFF))
		{
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabLong((uint32*)ma);
			mb->tdir_count=(uint64)(*(uint32*)ma);
			ma+=sizeof(uint32);
			*(uint32*)(&mb->tdir_offset)=*(uint32*)ma;
			ma+=sizeof(uint32);
		}
		else
		{
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabLong8((uint64*)ma);
                        mb->tdir_count=TIFFReadUInt64(ma);
			ma+=sizeof(uint64);
			mb->tdir_offset.toff_long8=TIFFReadUInt64(ma);
			ma+=sizeof(uint64);
		}
		mb++;
	}
	_TIFFfree(origdir);
	*pdir = dir;
	return dircount16;
}

/*
 * Fetch a tag that is not handled by special case code.
 */
static int
TIFFFetchNormalTag(TIFF* tif, TIFFDirEntry* dp, int recover)
{
	static const char module[] = "TIFFFetchNormalTag";
	enum TIFFReadDirEntryErr err;
	uint32 fii;
	const TIFFField* fip = NULL;
	TIFFReadDirectoryFindFieldInfo(tif,dp->tdir_tag,&fii);
        if( fii == FAILED_FII )
        {
            TIFFErrorExt(tif->tif_clientdata, "TIFFFetchNormalTag",
                         "No definition found for tag %d",
                         dp->tdir_tag);
            return 0;
        }
	fip=tif->tif_fields[fii];
	assert(fip != NULL); /* should not happen */
	assert(fip->set_field_type!=TIFF_SETGET_OTHER);  /* if so, we shouldn't arrive here but deal with this in specialized code */
	assert(fip->set_field_type!=TIFF_SETGET_INT);    /* if so, we shouldn't arrive here as this is only the case for pseudo-tags */
	err=TIFFReadDirEntryErrOk;
	switch (fip->set_field_type)
	{
		case TIFF_SETGET_UNDEFINED:
			break;
		case TIFF_SETGET_ASCII:
			{
				uint8* data;
				assert(fip->field_passcount==0);
				err=TIFFReadDirEntryByteArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					uint32 mb = 0;
					int n;
					if (data != NULL)
					{
					    uint8* ma = data;
					    while (mb<(uint32)dp->tdir_count)
					    {
					            if (*ma==0)
					                    break;
					            ma++;
					            mb++;
					    }
					}
					if (mb+1<(uint32)dp->tdir_count)
						TIFFWarningExt(tif->tif_clientdata,module,"ASCII value for tag \"%s\" contains null byte in value; value incorrectly truncated during reading due to implementation limitations",fip->field_name);
					else if (mb+1>(uint32)dp->tdir_count)
					{
						uint8* o;
						TIFFWarningExt(tif->tif_clientdata,module,"ASCII value for tag \"%s\" does not end in null byte",fip->field_name);
						if ((uint32)dp->tdir_count+1!=dp->tdir_count+1)
							o=NULL;
						else
							o=_TIFFmalloc((uint32)dp->tdir_count+1);
						if (o==NULL)
						{
							if (data!=NULL)
								_TIFFfree(data);
							return(0);
						}
						_TIFFmemcpy(o,data,(uint32)dp->tdir_count);
						o[(uint32)dp->tdir_count]=0;
						if (data!=0)
							_TIFFfree(data);
						data=o;
					}
					n=TIFFSetField(tif,dp->tdir_tag,data);
					if (data!=0)
						_TIFFfree(data);
					if (!n)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_UINT8:
			{
				uint8 data=0;
				assert(fip->field_readcount==1);
				assert(fip->field_passcount==0);
				err=TIFFReadDirEntryByte(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					if (!TIFFSetField(tif,dp->tdir_tag,data))
						return(0);
				}
			}
			break;
		case TIFF_SETGET_UINT16:
			{
				uint16 data;
				assert(fip->field_readcount==1);
				assert(fip->field_passcount==0);
				err=TIFFReadDirEntryShort(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					if (!TIFFSetField(tif,dp->tdir_tag,data))
						return(0);
				}
			}
			break;
		case TIFF_SETGET_UINT32:
			{
				uint32 data;
				assert(fip->field_readcount==1);
				assert(fip->field_passcount==0);
				err=TIFFReadDirEntryLong(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					if (!TIFFSetField(tif,dp->tdir_tag,data))
						return(0);
				}
			}
			break;
		case TIFF_SETGET_UINT64:
			{
				uint64 data;
				assert(fip->field_readcount==1);
				assert(fip->field_passcount==0);
				err=TIFFReadDirEntryLong8(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					if (!TIFFSetField(tif,dp->tdir_tag,data))
						return(0);
				}
			}
			break;
		case TIFF_SETGET_FLOAT:
			{
				float data;
				assert(fip->field_readcount==1);
				assert(fip->field_passcount==0);
				err=TIFFReadDirEntryFloat(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					if (!TIFFSetField(tif,dp->tdir_tag,data))
						return(0);
				}
			}
			break;
		case TIFF_SETGET_DOUBLE:
			{
				double data;
				assert(fip->field_readcount==1);
				assert(fip->field_passcount==0);
				err=TIFFReadDirEntryDouble(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					if (!TIFFSetField(tif,dp->tdir_tag,data))
						return(0);
				}
			}
			break;
		case TIFF_SETGET_IFD8:
			{
				uint64 data;
				assert(fip->field_readcount==1);
				assert(fip->field_passcount==0);
				err=TIFFReadDirEntryIfd8(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					if (!TIFFSetField(tif,dp->tdir_tag,data))
						return(0);
				}
			}
			break;
		case TIFF_SETGET_UINT16_PAIR:
			{
				uint16* data;
				assert(fip->field_readcount==2);
				assert(fip->field_passcount==0);
				if (dp->tdir_count!=2) {
					TIFFWarningExt(tif->tif_clientdata,module,
						       "incorrect count for field \"%s\", expected 2, got %d",
						       fip->field_name,(int)dp->tdir_count);
					return(0);
				}
				err=TIFFReadDirEntryShortArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,data[0],data[1]);
					_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C0_UINT8:
			{
				uint8* data;
				assert(fip->field_readcount>=1);
				assert(fip->field_passcount==0);
				if (dp->tdir_count!=(uint64)fip->field_readcount) {
					TIFFWarningExt(tif->tif_clientdata,module,
						       "incorrect count for field \"%s\", expected %d, got %d",
						       fip->field_name,(int) fip->field_readcount, (int)dp->tdir_count);
					return 0;
				}
				else
				{
					err=TIFFReadDirEntryByteArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C0_UINT16:
			{
				uint16* data;
				assert(fip->field_readcount>=1);
				assert(fip->field_passcount==0);
				if (dp->tdir_count!=(uint64)fip->field_readcount)
                                    /* corrupt file */;
				else
				{
					err=TIFFReadDirEntryShortArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C0_UINT32:
			{
				uint32* data;
				assert(fip->field_readcount>=1);
				assert(fip->field_passcount==0);
				if (dp->tdir_count!=(uint64)fip->field_readcount)
                                    /* corrupt file */;
				else
				{
					err=TIFFReadDirEntryLongArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C0_FLOAT:
			{
				float* data;
				assert(fip->field_readcount>=1);
				assert(fip->field_passcount==0);
				if (dp->tdir_count!=(uint64)fip->field_readcount)
                                    /* corrupt file */;
				else
				{
					err=TIFFReadDirEntryFloatArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C16_ASCII:
			{
				uint8* data;
				assert(fip->field_readcount==TIFF_VARIABLE);
				assert(fip->field_passcount==1);
				if (dp->tdir_count>0xFFFF)
					err=TIFFReadDirEntryErrCount;
				else
				{
					err=TIFFReadDirEntryByteArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						if( data != 0 && dp->tdir_count > 0 && data[dp->tdir_count-1] != '\0' )
						{
						    TIFFWarningExt(tif->tif_clientdata,module,"ASCII value for tag \"%s\" does not end in null byte. Forcing it to be null",fip->field_name);
						    data[dp->tdir_count-1] = '\0';
						}
						m=TIFFSetField(tif,dp->tdir_tag,(uint16)(dp->tdir_count),data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C16_UINT8:
			{
				uint8* data;
				assert(fip->field_readcount==TIFF_VARIABLE);
				assert(fip->field_passcount==1);
				if (dp->tdir_count>0xFFFF)
					err=TIFFReadDirEntryErrCount;
				else
				{
					err=TIFFReadDirEntryByteArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,(uint16)(dp->tdir_count),data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C16_UINT16:
			{
				uint16* data;
				assert(fip->field_readcount==TIFF_VARIABLE);
				assert(fip->field_passcount==1);
				if (dp->tdir_count>0xFFFF)
					err=TIFFReadDirEntryErrCount;
				else
				{
					err=TIFFReadDirEntryShortArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,(uint16)(dp->tdir_count),data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C16_UINT32:
			{
				uint32* data;
				assert(fip->field_readcount==TIFF_VARIABLE);
				assert(fip->field_passcount==1);
				if (dp->tdir_count>0xFFFF)
					err=TIFFReadDirEntryErrCount;
				else
				{
					err=TIFFReadDirEntryLongArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,(uint16)(dp->tdir_count),data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C16_UINT64:
			{
				uint64* data;
				assert(fip->field_readcount==TIFF_VARIABLE);
				assert(fip->field_passcount==1);
				if (dp->tdir_count>0xFFFF)
					err=TIFFReadDirEntryErrCount;
				else
				{
					err=TIFFReadDirEntryLong8Array(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,(uint16)(dp->tdir_count),data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C16_FLOAT:
			{
				float* data;
				assert(fip->field_readcount==TIFF_VARIABLE);
				assert(fip->field_passcount==1);
				if (dp->tdir_count>0xFFFF)
					err=TIFFReadDirEntryErrCount;
				else
				{
					err=TIFFReadDirEntryFloatArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,(uint16)(dp->tdir_count),data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C16_DOUBLE:
			{
				double* data;
				assert(fip->field_readcount==TIFF_VARIABLE);
				assert(fip->field_passcount==1);
				if (dp->tdir_count>0xFFFF)
					err=TIFFReadDirEntryErrCount;
				else
				{
					err=TIFFReadDirEntryDoubleArray(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,(uint16)(dp->tdir_count),data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C16_IFD8:
			{
				uint64* data;
				assert(fip->field_readcount==TIFF_VARIABLE);
				assert(fip->field_passcount==1);
				if (dp->tdir_count>0xFFFF)
					err=TIFFReadDirEntryErrCount;
				else
				{
					err=TIFFReadDirEntryIfd8Array(tif,dp,&data);
					if (err==TIFFReadDirEntryErrOk)
					{
						int m;
						m=TIFFSetField(tif,dp->tdir_tag,(uint16)(dp->tdir_count),data);
						if (data!=0)
							_TIFFfree(data);
						if (!m)
							return(0);
					}
				}
			}
			break;
		case TIFF_SETGET_C32_ASCII:
			{
				uint8* data;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntryByteArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					if( data != 0 && dp->tdir_count > 0 && data[dp->tdir_count-1] != '\0' )
					{
					    TIFFWarningExt(tif->tif_clientdata,module,"ASCII value for tag \"%s\" does not end in null byte. Forcing it to be null",fip->field_name);
                                            data[dp->tdir_count-1] = '\0';
					}
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_UINT8:
			{
				uint8* data;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntryByteArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_SINT8:
			{
				int8* data = NULL;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntrySbyteArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_UINT16:
			{
				uint16* data;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntryShortArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_SINT16:
			{
				int16* data = NULL;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntrySshortArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_UINT32:
			{
				uint32* data;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntryLongArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_SINT32:
			{
				int32* data = NULL;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntrySlongArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_UINT64:
			{
				uint64* data;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntryLong8Array(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_SINT64:
			{
				int64* data = NULL;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntrySlong8Array(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_FLOAT:
			{
				float* data;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntryFloatArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_DOUBLE:
			{
				double* data;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntryDoubleArray(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		case TIFF_SETGET_C32_IFD8:
			{
				uint64* data;
				assert(fip->field_readcount==TIFF_VARIABLE2);
				assert(fip->field_passcount==1);
				err=TIFFReadDirEntryIfd8Array(tif,dp,&data);
				if (err==TIFFReadDirEntryErrOk)
				{
					int m;
					m=TIFFSetField(tif,dp->tdir_tag,(uint32)(dp->tdir_count),data);
					if (data!=0)
						_TIFFfree(data);
					if (!m)
						return(0);
				}
			}
			break;
		default:
			assert(0);    /* we should never get here */
			break;
	}
	if (err!=TIFFReadDirEntryErrOk)
	{
		TIFFReadDirEntryOutputErr(tif,err,module,fip->field_name,recover);
		return(0);
	}
	return(1);
}

/*
 * Fetch a set of offsets or lengths.
 * While this routine says "strips", in fact it's also used for tiles.
 */
static int
TIFFFetchStripThing(TIFF* tif, TIFFDirEntry* dir, uint32 nstrips, uint64** lpp)
{
	static const char module[] = "TIFFFetchStripThing";
	enum TIFFReadDirEntryErr err;
	uint64* data;
	err=TIFFReadDirEntryLong8ArrayWithLimit(tif,dir,&data,nstrips);
	if (err!=TIFFReadDirEntryErrOk)
	{
		const TIFFField* fip = TIFFFieldWithTag(tif,dir->tdir_tag); 
		TIFFReadDirEntryOutputErr(tif,err,module,fip ? fip->field_name : "unknown tagname",0);
		return(0);
	}
	if (dir->tdir_count<(uint64)nstrips)
	{
		uint64* resizeddata;
		const TIFFField* fip = TIFFFieldWithTag(tif,dir->tdir_tag);
		const char* pszMax = getenv("LIBTIFF_STRILE_ARRAY_MAX_RESIZE_COUNT");
		uint32 max_nstrips = 1000000;
		if( pszMax )
			max_nstrips = (uint32) atoi(pszMax);
		TIFFReadDirEntryOutputErr(tif,TIFFReadDirEntryErrCount,
		            module,
		            fip ? fip->field_name : "unknown tagname",
		            ( nstrips <= max_nstrips ) );

		if( nstrips > max_nstrips )
		{
			_TIFFfree(data);
			return(0);
		}

		resizeddata=(uint64*)_TIFFCheckMalloc(tif,nstrips,sizeof(uint64),"for strip array");
		if (resizeddata==0) {
			_TIFFfree(data);
			return(0);
		}
                _TIFFmemcpy(resizeddata,data,(uint32)dir->tdir_count*sizeof(uint64));
                _TIFFmemset(resizeddata+(uint32)dir->tdir_count,0,(nstrips-(uint32)dir->tdir_count)*sizeof(uint64));
		_TIFFfree(data);
		data=resizeddata;
	}
	*lpp=data;
	return(1);
}

/*
 * Fetch and set the SubjectDistance EXIF tag.
 */
static int
TIFFFetchSubjectDistance(TIFF* tif, TIFFDirEntry* dir)
{
	static const char module[] = "TIFFFetchSubjectDistance";
	enum TIFFReadDirEntryErr err;
	UInt64Aligned_t m;
    m.l=0;
	assert(sizeof(double)==8);
	assert(sizeof(uint64)==8);
	assert(sizeof(uint32)==4);
	if (dir->tdir_count!=1)
		err=TIFFReadDirEntryErrCount;
	else if (dir->tdir_type!=TIFF_RATIONAL)
		err=TIFFReadDirEntryErrType;
	else
	{
		if (!(tif->tif_flags&TIFF_BIGTIFF))
		{
			uint32 offset;
			offset=*(uint32*)(&dir->tdir_offset);
			if (tif->tif_flags&TIFF_SWAB)
				TIFFSwabLong(&offset);
			err=TIFFReadDirEntryData(tif,offset,8,m.i);
		}
		else
		{
			m.l=dir->tdir_offset.toff_long8;
			err=TIFFReadDirEntryErrOk;
		}
	}
	if (err==TIFFReadDirEntryErrOk)
	{
		double n;
		if (tif->tif_flags&TIFF_SWAB)
			TIFFSwabArrayOfLong(m.i,2);
		if (m.i[0]==0)
			n=0.0;
		else if (m.i[0]==0xFFFFFFFF)
			/*
			 * XXX: Numerator 0xFFFFFFFF means that we have infinite
			 * distance. Indicate that with a negative floating point
			 * SubjectDistance value.
			 */
			n=-1.0;
		else
			n=(double)m.i[0]/(double)m.i[1];
		return(TIFFSetField(tif,dir->tdir_tag,n));
	}
	else
	{
		TIFFReadDirEntryOutputErr(tif,err,module,"SubjectDistance",TRUE);
		return(0);
	}
}

static void allocChoppedUpStripArrays(TIFF* tif, uint32 nstrips,
                                      uint64 stripbytes, uint32 rowsperstrip)
{
    TIFFDirectory *td = &tif->tif_dir;
    uint64 bytecount;
    uint64 offset;
    uint32 i;
    uint64 *newcounts;
    uint64 *newoffsets;

    newcounts = (uint64*) _TIFFCheckMalloc(tif, nstrips, sizeof (uint64),
                            "for chopped \"StripByteCounts\" array");
    newoffsets = (uint64*) _TIFFCheckMalloc(tif, nstrips, sizeof (uint64),
                            "for chopped \"StripOffsets\" array");
    if (newcounts == NULL || newoffsets == NULL) {
        /*
        * Unable to allocate new strip information, give up and use
        * the original one strip information.
        */
        if (newcounts != NULL)
            _TIFFfree(newcounts);
        if (newoffsets != NULL)
            _TIFFfree(newoffsets);
        return;
    }

    /*
     * Fill the strip information arrays with new bytecounts and offsets
     * that reflect the broken-up format.
     */
    offset = td->td_stripoffset[0];
    bytecount = td->td_stripoffset[td->td_nstrips-1] +
                td->td_stripbytecount[td->td_nstrips-1] - offset;
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

    _TIFFfree(td->td_stripbytecount);
    _TIFFfree(td->td_stripoffset);
    td->td_stripbytecount = newcounts;
    td->td_stripoffset = newoffsets;
    td->td_stripbytecountsorted = 1;
}


/*
 * Replace a single strip (tile) of uncompressed data by multiple strips
 * (tiles), each approximately STRIP_SIZE_DEFAULT bytes. This is useful for
 * dealing with large images or for dealing with machines with a limited
 * amount memory.
 */
static void
ChopUpSingleUncompressedStrip(TIFF* tif)
{
	register TIFFDirectory *td = &tif->tif_dir;
	uint64 bytecount;
	uint64 offset;
	uint32 rowblock;
	uint64 rowblockbytes;
	uint64 stripbytes;
	uint32 nstrips;
	uint32 rowsperstrip;

	bytecount = td->td_stripbytecount[0];
        /* On a newly created file, just re-opened to be filled, we */
        /* don't want strip chop to trigger as it is going to cause issues */
        /* later ( StripOffsets and StripByteCounts improperly filled) . */
        if( bytecount == 0 && tif->tif_mode != O_RDONLY )
            return;
	offset = td->td_stripoffset[0];
	assert(td->td_planarconfig == PLANARCONFIG_CONTIG);
	if ((td->td_photometric == PHOTOMETRIC_YCBCR)&&
	    (!isUpSampled(tif)))
		rowblock = td->td_ycbcrsubsampling[1];
	else
		rowblock = 1;
	rowblockbytes = TIFFVTileSize64(tif, rowblock);
	/*
	 * Make the rows hold at least one scanline, but fill specified amount
	 * of data if possible.
	 */
	if (rowblockbytes > STRIP_SIZE_DEFAULT) {
		stripbytes = rowblockbytes;
		rowsperstrip = rowblock;
	} else if (rowblockbytes > 0 ) {
		uint32 rowblocksperstrip;
		rowblocksperstrip = (uint32) (STRIP_SIZE_DEFAULT / rowblockbytes);
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
        if( nstrips == 0 )
            return;

        /* If we are going to allocate a lot of memory, make sure that the */
        /* file is as big as needed */
        if( tif->tif_mode == O_RDONLY &&
            nstrips > 1000000 &&
            (offset >= TIFFGetFileSize(tif) ||
             stripbytes > (TIFFGetFileSize(tif) - offset) / (nstrips - 1)) )
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
static void TryChopUpUncompressedBigTiff( TIFF* tif )
{
    TIFFDirectory *td = &tif->tif_dir;
    uint32 rowblock;
    uint64 rowblockbytes;
    uint32 i;
    uint64 stripsize;
    uint32 rowblocksperstrip;
    uint32 rowsperstrip;
    uint64 stripbytes;
    uint32 nstrips;

    stripsize = TIFFStripSize64(tif);

    assert( tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG );
    assert( tif->tif_dir.td_compression == COMPRESSION_NONE );
    assert( (tif->tif_flags&(TIFF_STRIPCHOP|TIFF_ISTILED)) == TIFF_STRIPCHOP );
    assert( stripsize > 0x7FFFFFFFUL );

    /* On a newly created file, just re-opened to be filled, we */
    /* don't want strip chop to trigger as it is going to cause issues */
    /* later ( StripOffsets and StripByteCounts improperly filled) . */
    if( td->td_stripbytecount[0] == 0 && tif->tif_mode != O_RDONLY )
        return;

    if ((td->td_photometric == PHOTOMETRIC_YCBCR)&&
        (!isUpSampled(tif)))
        rowblock = td->td_ycbcrsubsampling[1];
    else
        rowblock = 1;
    rowblockbytes = TIFFVStripSize64(tif, rowblock);
    if( rowblockbytes == 0 || rowblockbytes > 0x7FFFFFFFUL )
    {
        /* In case of file with gigantic width */
        return;
    }

    /* Check that the strips are contiguous and of the expected size */
    for( i = 0; i < td->td_nstrips; i++ )
    {
        if( i == td->td_nstrips - 1 )
        {
            if( td->td_stripbytecount[i] < TIFFVStripSize64(
                    tif, td->td_imagelength - i * td->td_rowsperstrip ) )
            {
                return;
            }
        }
        else
        {
            if( td->td_stripbytecount[i] != stripsize )
            {
                return;
            }
            if( i > 0 && td->td_stripoffset[i] !=
                    td->td_stripoffset[i-1] + td->td_stripbytecount[i - 1] )
            {
                return;
            }
        }
    }

    /* Aim for 512 MB strips (that will still be manageable by 32 bit builds */
    rowblocksperstrip = (uint32) (512 * 1024 * 1024 / rowblockbytes);
    if( rowblocksperstrip == 0 )
        rowblocksperstrip = 1;
    rowsperstrip = rowblocksperstrip * rowblock;
    stripbytes = rowblocksperstrip * rowblockbytes;
    assert( stripbytes <= 0x7FFFFFFFUL );

    nstrips = TIFFhowmany_32(td->td_imagelength, rowsperstrip);
    if( nstrips == 0 )
        return;

    /* If we are going to allocate a lot of memory, make sure that the */
    /* file is as big as needed */
    if( tif->tif_mode == O_RDONLY &&
        nstrips > 1000000 &&
        (td->td_stripoffset[td->td_nstrips-1] > TIFFGetFileSize(tif) ||
         td->td_stripoffset[td->td_nstrips-1] +
         td->td_stripbytecount[td->td_nstrips-1] > TIFFGetFileSize(tif)) )
    {
        return;
    }

    allocChoppedUpStripArrays(tif, nstrips, stripbytes, rowsperstrip);
}



int _TIFFFillStriles( TIFF *tif )
{
    return _TIFFFillStrilesInternal( tif, 1 );
}

static int _TIFFFillStrilesInternal( TIFF *tif, int loadStripByteCount )
{
#if defined(DEFER_STRILE_LOAD)
        register TIFFDirectory *td = &tif->tif_dir;
        int return_value = 1;

        if( td->td_stripoffset != NULL )
                return 1;

        if( td->td_stripoffset_entry.tdir_count == 0 )
                return 0;

        if (!TIFFFetchStripThing(tif,&(td->td_stripoffset_entry),
                                 td->td_nstrips,&td->td_stripoffset))
        {
                return_value = 0;
        }

        if (loadStripByteCount &&
            !TIFFFetchStripThing(tif,&(td->td_stripbytecount_entry),
                                 td->td_nstrips,&td->td_stripbytecount))
        {
                return_value = 0;
        }

        _TIFFmemset( &(td->td_stripoffset_entry), 0, sizeof(TIFFDirEntry));
        _TIFFmemset( &(td->td_stripbytecount_entry), 0, sizeof(TIFFDirEntry));

	if (tif->tif_dir.td_nstrips > 1 && return_value == 1 ) {
		uint32 strip;

		tif->tif_dir.td_stripbytecountsorted = 1;
		for (strip = 1; strip < tif->tif_dir.td_nstrips; strip++) {
			if (tif->tif_dir.td_stripoffset[strip - 1] >
			    tif->tif_dir.td_stripoffset[strip]) {
				tif->tif_dir.td_stripbytecountsorted = 0;
				break;
			}
		}
	}

        return return_value;
#else /* !defined(DEFER_STRILE_LOAD) */
        (void) tif;
        (void) loadStripByteCount;
        return 1;
#endif 
}


/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
