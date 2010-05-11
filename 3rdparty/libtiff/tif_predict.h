/* $Header: /home/vp/work/opencv-cvsbackup/opencv/3rdparty/libtiff/tif_predict.h,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

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
	int	predictor;		/* predictor tag value */
	int	stride;			/* sample stride over data */
	tsize_t	rowsize;		/* tile/strip row size */

	TIFFPostMethod	pfunc;		/* horizontal differencer/accumulator */
	TIFFCodeMethod	coderow;	/* parent codec encode/decode row */
	TIFFCodeMethod	codestrip;	/* parent codec encode/decode strip */
	TIFFCodeMethod	codetile;	/* parent codec encode/decode tile */
	TIFFVGetMethod	vgetparent;	/* super-class method */
	TIFFVSetMethod	vsetparent;	/* super-class method */
	TIFFPrintMethod	printdir;	/* super-class method */
	TIFFBoolMethod	setupdecode;	/* super-class method */
	TIFFBoolMethod	setupencode;	/* super-class method */
} TIFFPredictorState;

#if defined(__cplusplus)
extern "C" {
#endif
extern	int TIFFPredictorInit(TIFF*);
#if defined(__cplusplus)
}
#endif
#endif /* _TIFFPREDICT_ */
