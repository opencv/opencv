/* $Id: tif_predict.h,v 1.8 2010-03-10 18:56:49 bfriesen Exp $ */

/*
 * Copyright (c) 1995-1997 Sam Leffler
 * Copyright (c) 1995-1997 Silicon Graphics, Inc.
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

#ifndef _TIFFPREDICT_
#define	_TIFFPREDICT_
/*
 * ``Library-private'' Support for the Predictor Tag
 */

/*
 * Codecs that want to support the Predictor tag must place
 * this structure first in their private state block so that
 * the predictor code can cast tif_data to find its state.
 */
typedef struct {
	int             predictor;	/* predictor tag value */
	tmsize_t        stride;		/* sample stride over data */
	tmsize_t        rowsize;	/* tile/strip row size */

	TIFFCodeMethod  encoderow;	/* parent codec encode/decode row */
	TIFFCodeMethod  encodestrip;	/* parent codec encode/decode strip */
	TIFFCodeMethod  encodetile;	/* parent codec encode/decode tile */ 
	TIFFPostMethod  encodepfunc;	/* horizontal differencer */

	TIFFCodeMethod  decoderow;	/* parent codec encode/decode row */
	TIFFCodeMethod  decodestrip;	/* parent codec encode/decode strip */
	TIFFCodeMethod  decodetile;	/* parent codec encode/decode tile */ 
	TIFFPostMethod  decodepfunc;	/* horizontal accumulator */

	TIFFVGetMethod  vgetparent;	/* super-class method */
	TIFFVSetMethod  vsetparent;	/* super-class method */
	TIFFPrintMethod printdir;	/* super-class method */
	TIFFBoolMethod  setupdecode;	/* super-class method */
	TIFFBoolMethod  setupencode;	/* super-class method */
} TIFFPredictorState;

#if defined(__cplusplus)
extern "C" {
#endif
extern int TIFFPredictorInit(TIFF*);
extern int TIFFPredictorCleanup(TIFF*);
#if defined(__cplusplus)
}
#endif
#endif /* _TIFFPREDICT_ */

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
