/* $Header: /home/vp/work/opencv-cvsbackup/opencv/3rdparty/libtiff/tif_apple.c,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

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
 * TIFF Library Macintosh-specific routines.
 *
 * These routines use only Toolbox and high-level File Manager traps.
 * They make no calls to the THINK C "unix" compatibility library.  Also,
 * malloc is not used directly but it is still referenced internally by
 * the ANSI library in rare cases.  Heap fragmentation by the malloc ring
 * buffer is therefore minimized.
 *
 * O_RDONLY and O_RDWR are treated identically here.  The tif_mode flag is
 * checked in TIFFWriteCheck().
 *
 * Create below fills in a blank creator signature and sets the file type
 * to 'TIFF'.  It is much better for the application to do this by Create'ing
 * the file first and TIFFOpen'ing it later.
 * ---------
 * This code has been "Carbonized", and may not work with older MacOS versions.
 * If so, grab the tif_apple.c out of an older libtiff distribution, like
 * 3.5.5 from www.libtiff.org.
 */

#include "tiffiop.h"
#include <Errors.h>
#include <Files.h>
#include <Memory.h>
#include <Script.h>

#if defined(__PPCC__) || defined(__SC__) || defined(__MRC__) || defined(applec)
#define	CtoPstr	c2pstr
#endif

static tsize_t
_tiffReadProc(thandle_t fd, tdata_t buf, tsize_t size)
{
	return (FSRead((short) fd, (long*) &size, (char*) buf) == noErr ?
	    size : (tsize_t) -1);
}

static tsize_t
_tiffWriteProc(thandle_t fd, tdata_t buf, tsize_t size)
{
	return (FSWrite((short) fd, (long*) &size, (char*) buf) == noErr ?
	    size : (tsize_t) -1);
}

static toff_t
_tiffSeekProc(thandle_t fd, toff_t off, int whence)
{
	long fpos, size;

	if (GetEOF((short) fd, &size) != noErr)
		return EOF;
	(void) GetFPos((short) fd, &fpos);

	switch (whence) {
	case SEEK_CUR:
		if (off + fpos > size)
			SetEOF((short) fd, off + fpos);
		if (SetFPos((short) fd, fsFromMark, off) != noErr)
			return EOF;
		break;
	case SEEK_END:
		if (off > 0)
			SetEOF((short) fd, off + size);
		if (SetFPos((short) fd, fsFromStart, off + size) != noErr)
			return EOF;
		break;
	case SEEK_SET:
		if (off > size)
			SetEOF((short) fd, off);
		if (SetFPos((short) fd, fsFromStart, off) != noErr)
			return EOF;
		break;
	}

	return (toff_t)(GetFPos((short) fd, &fpos) == noErr ? fpos : EOF);
}

static int
_tiffMapProc(thandle_t fd, tdata_t* pbase, toff_t* psize)
{
	return (0);
}

static void
_tiffUnmapProc(thandle_t fd, tdata_t base, toff_t size)
{
}

static int
_tiffCloseProc(thandle_t fd)
{
	return (FSClose((short) fd));
}

static toff_t
_tiffSizeProc(thandle_t fd)
{
	long size;

	if (GetEOF((short) fd, &size) != noErr) {
		TIFFError("_tiffSizeProc", "%s: Cannot get file size");
		return (-1L);
	}
	return ((toff_t) size);
}

/*
 * Open a TIFF file descriptor for read/writing.
 */
TIFF*
TIFFFdOpen(int fd, const char* name, const char* mode)
{
	TIFF* tif;

	tif = TIFFClientOpen(name, mode, (thandle_t) fd,
	    _tiffReadProc, _tiffWriteProc, _tiffSeekProc, _tiffCloseProc,
	    _tiffSizeProc, _tiffMapProc, _tiffUnmapProc);
	if (tif)
		tif->tif_fd = fd;
	return (tif);
}

static void ourc2pstr( char* inString )
{
	int	sLen = strlen( inString );
	BlockMoveData( inString, &inString[1], sLen );
	inString[0] = sLen;
}

/*
 * Open a TIFF file for read/writing.
 */
TIFF*
TIFFOpen(const char* name, const char* mode)
{
	static const char module[] = "TIFFOpen";
	Str255 pname;
	FInfo finfo;
	short fref;
	OSErr err;
	FSSpec	fSpec;

	strcpy((char*) pname, name);
	ourc2pstr((char*) pname);
	
	err = FSMakeFSSpec( 0, 0, pname, &fSpec );

	switch (_TIFFgetMode(mode, module)) {
	default:
		return ((TIFF*) 0);
	case O_RDWR | O_CREAT | O_TRUNC:
		if (FSpGetFInfo(&fSpec, &finfo) == noErr)
			FSpDelete(&fSpec);
		/* fall through */
	case O_RDWR | O_CREAT:
		if ((err = FSpGetFInfo(&fSpec, &finfo)) == fnfErr) {
			if (FSpCreate(&fSpec, '    ', 'TIFF', smSystemScript) != noErr)
				goto badCreate;
			if (FSpOpenDF(&fSpec, fsRdWrPerm, &fref) != noErr)
				goto badOpen;
		} else if (err == noErr) {
			if (FSpOpenDF(&fSpec, fsRdWrPerm, &fref) != noErr)
				goto badOpen;
		} else
			goto badOpen;
		break;
	case O_RDONLY:
		if (FSpOpenDF(&fSpec, fsRdPerm, &fref) != noErr)
			goto badOpen;
		break;
	case O_RDWR:
		if (FSpOpenDF(&fSpec, fsRdWrPerm, &fref) != noErr)
			goto badOpen;
		break;
	}
	return (TIFFFdOpen((int) fref, name, mode));
badCreate:
	TIFFError(module, "%s: Cannot create", name);
	return ((TIFF*) 0);
badOpen:
	TIFFError(module, "%s: Cannot open", name);
	return ((TIFF*) 0);
}

void
_TIFFmemset(tdata_t p, int v, tsize_t c)
{
	memset(p, v, (size_t) c);
}

void
_TIFFmemcpy(tdata_t d, const tdata_t s, tsize_t c)
{
	memcpy(d, s, (size_t) c);
}

int
_TIFFmemcmp(const tdata_t p1, const tdata_t p2, tsize_t c)
{
	return (memcmp(p1, p2, (size_t) c));
}

tdata_t
_TIFFmalloc(tsize_t s)
{
	return (NewPtr((size_t) s));
}

void
_TIFFfree(tdata_t p)
{
	DisposePtr(p);
}

tdata_t
_TIFFrealloc(tdata_t p, tsize_t s)
{
	Ptr n = p;

	SetPtrSize(p, (size_t) s);
	if (MemError() && (n = NewPtr((size_t) s)) != NULL) {
		BlockMove(p, n, GetPtrSize(p));
		DisposePtr(p);
	}
	return ((tdata_t) n);
}

static void
appleWarningHandler(const char* module, const char* fmt, va_list ap)
{
	if (module != NULL)
		fprintf(stderr, "%s: ", module);
	fprintf(stderr, "Warning, ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, ".\n");
}
TIFFErrorHandler _TIFFwarningHandler = appleWarningHandler;

static void
appleErrorHandler(const char* module, const char* fmt, va_list ap)
{
	if (module != NULL)
		fprintf(stderr, "%s: ", module);
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, ".\n");
}
TIFFErrorHandler _TIFFerrorHandler = appleErrorHandler;
