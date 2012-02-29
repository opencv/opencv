/* $Id: tif_packbits.c,v 1.20 2010-03-10 18:56:49 bfriesen Exp $ */

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

#include "tiffiop.h"
#ifdef PACKBITS_SUPPORT
/*
 * TIFF Library.
 *
 * PackBits Compression Algorithm Support
 */
#include <stdio.h>

static int
PackBitsPreEncode(TIFF* tif, uint16 s)
{
	(void) s;

	if (!(tif->tif_data = (uint8*)_TIFFmalloc(sizeof(tmsize_t))))
		return (0);
	/*
	 * Calculate the scanline/tile-width size in bytes.
	 */
	if (isTiled(tif))
		*(tmsize_t*)tif->tif_data = TIFFTileRowSize(tif);
	else
		*(tmsize_t*)tif->tif_data = TIFFScanlineSize(tif);
	return (1);
}

static int
PackBitsPostEncode(TIFF* tif)
{
        if (tif->tif_data)
            _TIFFfree(tif->tif_data);
	return (1);
}

/*
 * Encode a run of pixels.
 */
static int
PackBitsEncode(TIFF* tif, uint8* buf, tmsize_t cc, uint16 s)
{
	unsigned char* bp = (unsigned char*) buf;
	uint8* op;
	uint8* ep;
	uint8* lastliteral;
	long n, slop;
	int b;
	enum { BASE, LITERAL, RUN, LITERAL_RUN } state;

	(void) s;
	op = tif->tif_rawcp;
	ep = tif->tif_rawdata + tif->tif_rawdatasize;
	state = BASE;
	lastliteral = 0;
	while (cc > 0) {
		/*
		 * Find the longest string of identical bytes.
		 */
		b = *bp++, cc--, n = 1;
		for (; cc > 0 && b == *bp; cc--, bp++)
			n++;
	again:
		if (op + 2 >= ep) {		/* insure space for new data */
			/*
			 * Be careful about writing the last
			 * literal.  Must write up to that point
			 * and then copy the remainder to the
			 * front of the buffer.
			 */
			if (state == LITERAL || state == LITERAL_RUN) {
				slop = (long)(op - lastliteral);
				tif->tif_rawcc += (tmsize_t)(lastliteral - tif->tif_rawcp);
				if (!TIFFFlushData1(tif))
					return (-1);
				op = tif->tif_rawcp;
				while (slop-- > 0)
					*op++ = *lastliteral++;
				lastliteral = tif->tif_rawcp;
			} else {
				tif->tif_rawcc += (tmsize_t)(op - tif->tif_rawcp);
				if (!TIFFFlushData1(tif))
					return (-1);
				op = tif->tif_rawcp;
			}
		}
		switch (state) {
		case BASE:		/* initial state, set run/literal */
			if (n > 1) {
				state = RUN;
				if (n > 128) {
					*op++ = (uint8) -127;
					*op++ = (uint8) b;
					n -= 128;
					goto again;
				}
				*op++ = (uint8)(-(n-1));
				*op++ = (uint8) b;
			} else {
				lastliteral = op;
				*op++ = 0;
				*op++ = (uint8) b;
				state = LITERAL;
			}
			break;
		case LITERAL:		/* last object was literal string */
			if (n > 1) {
				state = LITERAL_RUN;
				if (n > 128) {
					*op++ = (uint8) -127;
					*op++ = (uint8) b;
					n -= 128;
					goto again;
				}
				*op++ = (uint8)(-(n-1));	/* encode run */
				*op++ = (uint8) b;
			} else {			/* extend literal */
				if (++(*lastliteral) == 127)
					state = BASE;
				*op++ = (uint8) b;
			}
			break;
		case RUN:		/* last object was run */
			if (n > 1) {
				if (n > 128) {
					*op++ = (uint8) -127;
					*op++ = (uint8) b;
					n -= 128;
					goto again;
				}
				*op++ = (uint8)(-(n-1));
				*op++ = (uint8) b;
			} else {
				lastliteral = op;
				*op++ = 0;
				*op++ = (uint8) b;
				state = LITERAL;
			}
			break;
		case LITERAL_RUN:	/* literal followed by a run */
			/*
			 * Check to see if previous run should
			 * be converted to a literal, in which
			 * case we convert literal-run-literal
			 * to a single literal.
			 */
			if (n == 1 && op[-2] == (uint8) -1 &&
			    *lastliteral < 126) {
				state = (((*lastliteral) += 2) == 127 ?
				    BASE : LITERAL);
				op[-2] = op[-1];	/* replicate */
			} else
				state = RUN;
			goto again;
		}
	}
	tif->tif_rawcc += (tmsize_t)(op - tif->tif_rawcp);
	tif->tif_rawcp = op;
	return (1);
}

/*
 * Encode a rectangular chunk of pixels.  We break it up
 * into row-sized pieces to insure that encoded runs do
 * not span rows.  Otherwise, there can be problems with
 * the decoder if data is read, for example, by scanlines
 * when it was encoded by strips.
 */
static int
PackBitsEncodeChunk(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s)
{
	tmsize_t rowsize = *(tmsize_t*)tif->tif_data;

	while (cc > 0) {
		tmsize_t chunk = rowsize;
		
		if( cc < chunk )
		    chunk = cc;

		if (PackBitsEncode(tif, bp, chunk, s) < 0)
		    return (-1);
		bp += chunk;
		cc -= chunk;
	}
	return (1);
}

static int
PackBitsDecode(TIFF* tif, uint8* op, tmsize_t occ, uint16 s)
{
	static const char module[] = "PackBitsDecode";
	char *bp;
	tmsize_t cc;
	long n;
	int b;

	(void) s;
	bp = (char*) tif->tif_rawcp;
	cc = tif->tif_rawcc;
	while (cc > 0 && occ > 0) {
		n = (long) *bp++, cc--;
		/*
		 * Watch out for compilers that
		 * don't sign extend chars...
		 */
		if (n >= 128)
			n -= 256;
		if (n < 0) {		/* replicate next byte -n+1 times */
			if (n == -128)	/* nop */
				continue;
			n = -n + 1;
			if( occ < (tmsize_t)n )
			{
				TIFFWarningExt(tif->tif_clientdata, module,
				    "Discarding %lu bytes to avoid buffer overrun",
				    (unsigned long) ((tmsize_t)n - occ));
				n = (long)occ;
			}
			occ -= n;
			b = *bp++, cc--;      /* TODO: may be reading past input buffer here when input data is corrupt or ends prematurely */
			while (n-- > 0)
				*op++ = (uint8) b;
		} else {		/* copy next n+1 bytes literally */
			if (occ < (tmsize_t)(n + 1))
			{
				TIFFWarningExt(tif->tif_clientdata, module,
				    "Discarding %lu bytes to avoid buffer overrun",
				    (unsigned long) ((tmsize_t)n - occ + 1));
				n = (long)occ - 1;
			}
			_TIFFmemcpy(op, bp, ++n);  /* TODO: may be reading past input buffer here when input data is corrupt or ends prematurely */
			op += n; occ -= n;
			bp += n; cc -= n;
		}
	}
	tif->tif_rawcp = (uint8*) bp;
	tif->tif_rawcc = cc;
	if (occ > 0) {
		TIFFErrorExt(tif->tif_clientdata, module,
		    "Not enough data for scanline %lu",
		    (unsigned long) tif->tif_row);
		return (0);
	}
	return (1);
}

int
TIFFInitPackBits(TIFF* tif, int scheme)
{
	(void) scheme;
	tif->tif_decoderow = PackBitsDecode;
	tif->tif_decodestrip = PackBitsDecode;
	tif->tif_decodetile = PackBitsDecode;
	tif->tif_preencode = PackBitsPreEncode;
	tif->tif_postencode = PackBitsPostEncode;
	tif->tif_encoderow = PackBitsEncode;
	tif->tif_encodestrip = PackBitsEncodeChunk;
	tif->tif_encodetile = PackBitsEncodeChunk;
	return (1);
}
#endif /* PACKBITS_SUPPORT */

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
