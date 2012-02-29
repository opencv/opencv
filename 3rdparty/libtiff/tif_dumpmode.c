/* $Header: /cvs/maptools/cvsroot/libtiff/libtiff/tif_dumpmode.c,v 1.14 2011-04-02 20:54:09 bfriesen Exp $ */

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
 * "Null" Compression Algorithm Support.
 */
#include "tiffiop.h"

static int
DumpFixupTags(TIFF* tif)
{
	(void) tif;
	return (1);
}

/*
 * Encode a hunk of pixels.
 */
static int
DumpModeEncode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s)
{
	(void) s;
	while (cc > 0) {
		tmsize_t n;

		n = cc;
		if (tif->tif_rawcc + n > tif->tif_rawdatasize)
			n = tif->tif_rawdatasize - tif->tif_rawcc;

		assert( n > 0 );

		/*
		 * Avoid copy if client has setup raw
		 * data buffer to avoid extra copy.
		 */
		if (tif->tif_rawcp != pp)
			_TIFFmemcpy(tif->tif_rawcp, pp, n);
		tif->tif_rawcp += n;
		tif->tif_rawcc += n;
		pp += n;
		cc -= n;
		if (tif->tif_rawcc >= tif->tif_rawdatasize &&
		    !TIFFFlushData1(tif))
			return (-1);
	}
	return (1);
}

/*
 * Decode a hunk of pixels.
 */
static int
DumpModeDecode(TIFF* tif, uint8* buf, tmsize_t cc, uint16 s)
{
	static const char module[] = "DumpModeDecode";
	(void) s;
	if (tif->tif_rawcc < cc) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
		TIFFErrorExt(tif->tif_clientdata, module,
"Not enough data for scanline %lu, expected a request for at most %I64d bytes, got a request for %I64d bytes",
		             (unsigned long) tif->tif_row,
		             (signed __int64) tif->tif_rawcc,
		             (signed __int64) cc);
#else
		TIFFErrorExt(tif->tif_clientdata, module,
"Not enough data for scanline %lu, expected a request for at most %lld bytes, got a request for %lld bytes",
		             (unsigned long) tif->tif_row,
		             (signed long long) tif->tif_rawcc,
		             (signed long long) cc);
#endif
		return (0);
	}
	/*
	 * Avoid copy if client has setup raw
	 * data buffer to avoid extra copy.
	 */
	if (tif->tif_rawcp != buf)
		_TIFFmemcpy(buf, tif->tif_rawcp, cc);
	tif->tif_rawcp += cc;
	tif->tif_rawcc -= cc;  
	return (1);
}

/*
 * Seek forwards nrows in the current strip.
 */
static int
DumpModeSeek(TIFF* tif, uint32 nrows)
{
	tif->tif_rawcp += nrows * tif->tif_scanlinesize;
	tif->tif_rawcc -= nrows * tif->tif_scanlinesize;
	return (1);
}

/*
 * Initialize dump mode.
 */
int
TIFFInitDumpMode(TIFF* tif, int scheme)
{
	(void) scheme;
	tif->tif_fixuptags = DumpFixupTags;  
	tif->tif_decoderow = DumpModeDecode;
	tif->tif_decodestrip = DumpModeDecode;
	tif->tif_decodetile = DumpModeDecode;
	tif->tif_encoderow = DumpModeEncode;
	tif->tif_encodestrip = DumpModeEncode;
	tif->tif_encodetile = DumpModeEncode; 
	tif->tif_seek = DumpModeSeek;
	return (1);
}
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
