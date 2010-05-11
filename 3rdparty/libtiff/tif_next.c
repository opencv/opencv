/* $Id: tif_next.c,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

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
#ifdef NEXT_SUPPORT
/*
 * TIFF Library.
 *
 * NeXT 2-bit Grey Scale Compression Algorithm Support
 */

#define SETPIXEL(op, v) {			\
	switch (npixels++ & 3) {		\
	case 0:	op[0]  = (unsigned char) ((v) << 6); break;	\
	case 1:	op[0] |= (v) << 4; break;	\
	case 2:	op[0] |= (v) << 2; break;	\
	case 3:	*op++ |= (v);	   break;	\
	}					\
}

#define LITERALROW	0x00
#define LITERALSPAN	0x40
#define WHITE   	((1<<2)-1)

static int
NeXTDecode(TIFF* tif, tidata_t buf, tsize_t occ, tsample_t s)
{
	register unsigned char *bp, *op;
	register tsize_t cc;
	register int n;
	tidata_t row;
	tsize_t scanline;

	(void) s;
	/*
	 * Each scanline is assumed to start off as all
	 * white (we assume a PhotometricInterpretation
	 * of ``min-is-black'').
	 */
	for (op = buf, cc = occ; cc-- > 0;)
		*op++ = 0xff;

	bp = (unsigned char *)tif->tif_rawcp;
	cc = tif->tif_rawcc;
	scanline = tif->tif_scanlinesize;
	for (row = buf; (long)occ > 0; occ -= scanline, row += scanline) {
		n = *bp++, cc--;
		switch (n) {
		case LITERALROW:
			/*
			 * The entire scanline is given as literal values.
			 */
			if (cc < scanline)
				goto bad;
			_TIFFmemcpy(row, bp, scanline);
			bp += scanline;
			cc -= scanline;
			break;
		case LITERALSPAN: {
			int off;
			/*
			 * The scanline has a literal span
			 * that begins at some offset.
			 */
			off = (bp[0] * 256) + bp[1];
			n = (bp[2] * 256) + bp[3];
			if (cc < 4+n || off+n > scanline)
				goto bad;
			_TIFFmemcpy(row+off, bp+4, n);
			bp += 4+n;
			cc -= 4+n;
			break;
		}
		default: {
			register int npixels = 0, grey;
			unsigned long imagewidth = tif->tif_dir.td_imagewidth;

			/*
			 * The scanline is composed of a sequence
			 * of constant color ``runs''.  We shift
			 * into ``run mode'' and interpret bytes
			 * as codes of the form <color><npixels>
			 * until we've filled the scanline.
			 */
			op = row;
			for (;;) {
				grey = (n>>6) & 0x3;
				n &= 0x3f;
				while (n-- > 0)
					SETPIXEL(op, grey);
				if (npixels >= (int) imagewidth)
					break;
				if (cc == 0)
					goto bad;
				n = *bp++, cc--;
			}
			break;
		}
		}
	}
	tif->tif_rawcp = (tidata_t) bp;
	tif->tif_rawcc = cc;
	return (1);
bad:
	TIFFError(tif->tif_name, "NeXTDecode: Not enough data for scanline %ld",
	    (long) tif->tif_row);
	return (0);
}

int
TIFFInitNeXT(TIFF* tif, int scheme)
{
	(void) scheme;
	tif->tif_decoderow = NeXTDecode;
	tif->tif_decodestrip = NeXTDecode;
	tif->tif_decodetile = NeXTDecode;
	return (1);
}
#endif /* NEXT_SUPPORT */

/* vim: set ts=8 sts=8 sw=8 noet: */
