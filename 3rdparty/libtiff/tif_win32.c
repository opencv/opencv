/* $Id: tif_win32.c,v 1.21.2.1 2010-06-08 18:50:43 bfriesen Exp $ */

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
 * TIFF Library Win32-specific Routines.  Adapted from tif_unix.c 4/5/95 by
 * Scott Wagner (wagner@itek.com), Itek Graphix, Rochester, NY USA
 */
#include "tiffiop.h"

#include <windows.h>

static tsize_t
_tiffReadProc(thandle_t fd, tdata_t buf, tsize_t size)
{
	DWORD dwSizeRead;
	if (!ReadFile(fd, buf, size, &dwSizeRead, NULL))
		return(0);
	return ((tsize_t) dwSizeRead);
}

static tsize_t
_tiffWriteProc(thandle_t fd, tdata_t buf, tsize_t size)
{
	DWORD dwSizeWritten;
	if (!WriteFile(fd, buf, size, &dwSizeWritten, NULL))
		return(0);
	return ((tsize_t) dwSizeWritten);
}

static toff_t
_tiffSeekProc(thandle_t fd, toff_t off, int whence)
{
        ULARGE_INTEGER li;
	DWORD dwMoveMethod;

	li.QuadPart = off;
        
	switch(whence)
	{
	case SEEK_SET:
		dwMoveMethod = FILE_BEGIN;
		break;
	case SEEK_CUR:
		dwMoveMethod = FILE_CURRENT;
		break;
	case SEEK_END:
		dwMoveMethod = FILE_END;
		break;
	default:
		dwMoveMethod = FILE_BEGIN;
		break;
	}
	return ((toff_t)SetFilePointer(fd, (LONG) li.LowPart,
				       (PLONG)&li.HighPart, dwMoveMethod));
}

static int
_tiffCloseProc(thandle_t fd)
{
	return (CloseHandle(fd) ? 0 : -1);
}

static toff_t
_tiffSizeProc(thandle_t fd)
{
	return ((toff_t)GetFileSize(fd, NULL));
}

static int
_tiffDummyMapProc(thandle_t fd, tdata_t* pbase, toff_t* psize)
{
	(void) fd;
	(void) pbase;
	(void) psize;
	return (0);
}

/*
 * From "Hermann Josef Hill" <lhill@rhein-zeitung.de>:
 *
 * Windows uses both a handle and a pointer for file mapping,
 * but according to the SDK documentation and Richter's book
 * "Advanced Windows Programming" it is safe to free the handle
 * after obtaining the file mapping pointer
 *
 * This removes a nasty OS dependency and cures a problem
 * with Visual C++ 5.0
 */
static int
_tiffMapProc(thandle_t fd, tdata_t* pbase, toff_t* psize)
{
	toff_t size;
	HANDLE hMapFile;

	if ((size = _tiffSizeProc(fd)) == 0xFFFFFFFF)
		return (0);
	hMapFile = CreateFileMapping(fd, NULL, PAGE_READONLY, 0, size, NULL);
	if (hMapFile == NULL)
		return (0);
	*pbase = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
	CloseHandle(hMapFile);
	if (*pbase == NULL)
		return (0);
	*psize = size;
	return(1);
}

static void
_tiffDummyUnmapProc(thandle_t fd, tdata_t base, toff_t size)
{
	(void) fd;
	(void) base;
	(void) size;
}

static void
_tiffUnmapProc(thandle_t fd, tdata_t base, toff_t size)
{
	UnmapViewOfFile(base);
}

/*
 * Open a TIFF file descriptor for read/writing.
 * Note that TIFFFdOpen and TIFFOpen recognise the character 'u' in the mode
 * string, which forces the file to be opened unmapped.
 */
TIFF*
TIFFFdOpen(int ifd, const char* name, const char* mode)
{
	TIFF* tif;
	BOOL fSuppressMap = (mode[1] == 'u' || (mode[1]!=0 && mode[2] == 'u'));

	tif = TIFFClientOpen(name, mode, (thandle_t)ifd,
			_tiffReadProc, _tiffWriteProc,
			_tiffSeekProc, _tiffCloseProc, _tiffSizeProc,
			fSuppressMap ? _tiffDummyMapProc : _tiffMapProc,
			fSuppressMap ? _tiffDummyUnmapProc : _tiffUnmapProc);
	if (tif)
		tif->tif_fd = ifd;
	return (tif);
}

#ifndef _WIN32_WCE

/*
 * Open a TIFF file for read/writing.
 */
TIFF*
TIFFOpen(const char* name, const char* mode)
{
	static const char module[] = "TIFFOpen";
	thandle_t fd;
	int m;
	DWORD dwMode;
	TIFF* tif;

	m = _TIFFgetMode(mode, module);

	switch(m)
	{
	case O_RDONLY:
		dwMode = OPEN_EXISTING;
		break;
	case O_RDWR:
		dwMode = OPEN_ALWAYS;
		break;
	case O_RDWR|O_CREAT:
		dwMode = OPEN_ALWAYS;
		break;
	case O_RDWR|O_TRUNC:
		dwMode = CREATE_ALWAYS;
		break;
	case O_RDWR|O_CREAT|O_TRUNC:
		dwMode = CREATE_ALWAYS;
		break;
	default:
		return ((TIFF*)0);
	}
	fd = (thandle_t)CreateFileA(name,
		(m == O_RDONLY)?GENERIC_READ:(GENERIC_READ | GENERIC_WRITE),
		FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, dwMode,
		(m == O_RDONLY)?FILE_ATTRIBUTE_READONLY:FILE_ATTRIBUTE_NORMAL,
		NULL);
	if (fd == INVALID_HANDLE_VALUE) {
		TIFFErrorExt(0, module, "%s: Cannot open", name);
		return ((TIFF *)0);
	}

	tif = TIFFFdOpen((int)fd, name, mode);
	if(!tif)
		CloseHandle(fd);
	return tif;
}

/*
 * Open a TIFF file with a Unicode filename, for read/writing.
 */
TIFF*
TIFFOpenW(const wchar_t* name, const char* mode)
{
	static const char module[] = "TIFFOpenW";
	thandle_t fd;
	int m;
	DWORD dwMode;
	int mbsize;
	char *mbname;
	TIFF *tif;

	m = _TIFFgetMode(mode, module);

	switch(m) {
		case O_RDONLY:			dwMode = OPEN_EXISTING; break;
		case O_RDWR:			dwMode = OPEN_ALWAYS;   break;
		case O_RDWR|O_CREAT:		dwMode = OPEN_ALWAYS;   break;
		case O_RDWR|O_TRUNC:		dwMode = CREATE_ALWAYS; break;
		case O_RDWR|O_CREAT|O_TRUNC:	dwMode = CREATE_ALWAYS; break;
		default:			return ((TIFF*)0);
	}

	fd = (thandle_t)CreateFileW(name,
		(m == O_RDONLY)?GENERIC_READ:(GENERIC_READ|GENERIC_WRITE),
		FILE_SHARE_READ, NULL, dwMode,
		(m == O_RDONLY)?FILE_ATTRIBUTE_READONLY:FILE_ATTRIBUTE_NORMAL,
		NULL);
	if (fd == INVALID_HANDLE_VALUE) {
		TIFFErrorExt(0, module, "%S: Cannot open", name);
		return ((TIFF *)0);
	}

	mbname = NULL;
	mbsize = WideCharToMultiByte(CP_ACP, 0, name, -1, NULL, 0, NULL, NULL);
	if (mbsize > 0) {
		mbname = (char *)_TIFFmalloc(mbsize);
		if (!mbname) {
			TIFFErrorExt(0, module,
			"Can't allocate space for filename conversion buffer");
			return ((TIFF*)0);
		}

		WideCharToMultiByte(CP_ACP, 0, name, -1, mbname, mbsize,
				    NULL, NULL);
	}

	tif = TIFFFdOpen((int)fd,
			 (mbname != NULL) ? mbname : "<unknown>", mode);
	if(!tif)
		CloseHandle(fd);

	_TIFFfree(mbname);

	return tif;
}

#endif /* ndef _WIN32_WCE */


tdata_t
_TIFFmalloc(tsize_t s)
{
	return ((tdata_t)GlobalAlloc(GMEM_FIXED, s));
}

void
_TIFFfree(tdata_t p)
{
	GlobalFree(p);
	return;
}

tdata_t
_TIFFrealloc(tdata_t p, tsize_t s)
{
	void* pvTmp;
	tsize_t old;

	if(p == NULL)
		return ((tdata_t)GlobalAlloc(GMEM_FIXED, s));

	old = GlobalSize(p);

	if (old>=s) {
		if ((pvTmp = GlobalAlloc(GMEM_FIXED, s)) != NULL) {
			CopyMemory(pvTmp, p, s);
			GlobalFree(p);
		}
	} else {
		if ((pvTmp = GlobalAlloc(GMEM_FIXED, s)) != NULL) {
			CopyMemory(pvTmp, p, old);
			GlobalFree(p);
		}
	}
	return ((tdata_t)pvTmp);
}

void
_TIFFmemset(void* p, int v, tsize_t c)
{
	FillMemory(p, c, (BYTE)v);
}

void
_TIFFmemcpy(void* d, const tdata_t s, tsize_t c)
{
	CopyMemory(d, s, c);
}

int
_TIFFmemcmp(const tdata_t p1, const tdata_t p2, tsize_t c)
{
	register const BYTE *pb1 = (const BYTE *) p1;
	register const BYTE *pb2 = (const BYTE *) p2;
	register DWORD dwTmp = c;
	register int iTmp;
	for (iTmp = 0; dwTmp-- && !iTmp; iTmp = (int)*pb1++ - (int)*pb2++)
		;
	return (iTmp);
}

#ifndef _WIN32_WCE

static void
Win32WarningHandler(const char* module, const char* fmt, va_list ap)
{
#ifndef TIF_PLATFORM_CONSOLE
	LPTSTR szTitle;
	LPTSTR szTmp;
	LPCTSTR szTitleText = "%s Warning";
	LPCTSTR szDefaultModule = "LIBTIFF";
	LPCTSTR szTmpModule = (module == NULL) ? szDefaultModule : module;
	if ((szTitle = (LPTSTR)LocalAlloc(LMEM_FIXED, (strlen(szTmpModule) +
		strlen(szTitleText) + strlen(fmt) + 128)*sizeof(char))) == NULL)
		return;
	sprintf(szTitle, szTitleText, szTmpModule);
	szTmp = szTitle + (strlen(szTitle)+2)*sizeof(char);
	vsprintf(szTmp, fmt, ap);
	MessageBoxA(GetFocus(), szTmp, szTitle, MB_OK | MB_ICONINFORMATION);
	LocalFree(szTitle);
	return;
#else
	if (module != NULL)
		fprintf(stderr, "%s: ", module);
	fprintf(stderr, "Warning, ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, ".\n");
#endif        
}
TIFFErrorHandler _TIFFwarningHandler = Win32WarningHandler;

static void
Win32ErrorHandler(const char* module, const char* fmt, va_list ap)
{
#ifndef TIF_PLATFORM_CONSOLE
	LPTSTR szTitle;
	LPTSTR szTmp;
	LPCTSTR szTitleText = "%s Error";
	LPCTSTR szDefaultModule = "LIBTIFF";
	LPCTSTR szTmpModule = (module == NULL) ? szDefaultModule : module;
	if ((szTitle = (LPTSTR)LocalAlloc(LMEM_FIXED, (strlen(szTmpModule) +
		strlen(szTitleText) + strlen(fmt) + 128)*sizeof(char))) == NULL)
		return;
	sprintf(szTitle, szTitleText, szTmpModule);
	szTmp = szTitle + (strlen(szTitle)+2)*sizeof(char);
	vsprintf(szTmp, fmt, ap);
	MessageBoxA(GetFocus(), szTmp, szTitle, MB_OK | MB_ICONEXCLAMATION);
	LocalFree(szTitle);
	return;
#else
	if (module != NULL)
		fprintf(stderr, "%s: ", module);
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, ".\n");
#endif        
}
TIFFErrorHandler _TIFFerrorHandler = Win32ErrorHandler;

#endif /* ndef _WIN32_WCE */

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
