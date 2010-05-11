/* $Id: tif_tile.c,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

/*
 * Copyright (c) 1991-1997 Sam Leffler
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
 * Tiled Image Support Routines.
 */
#include "tiffiop.h"

static uint32
summarize(TIFF* tif, size_t summand1, size_t summand2, const char* where)
{
	/*
	 * XXX: We are using casting to uint32 here, bacause sizeof(size_t)
	 * may be larger than sizeof(uint32) on 64-bit architectures.
	 */
	uint32	bytes = summand1 + summand2;

	if (bytes - summand1 != summand2) {
		TIFFError(tif->tif_name, "Integer overflow in %s", where);
		bytes = 0;
	}

	return (bytes);
}

static uint32
multiply(TIFF* tif, size_t nmemb, size_t elem_size, const char* where)
{
	uint32	bytes = nmemb * elem_size;

	if (elem_size && bytes / elem_size != nmemb) {
		TIFFError(tif->tif_name, "Integer overflow in %s", where);
		bytes = 0;
	}

	return (bytes);
}

/*
 * Compute which tile an (x,y,z,s) value is in.
 */
ttile_t
TIFFComputeTile(TIFF* tif, uint32 x, uint32 y, uint32 z, tsample_t s)
{
	TIFFDirectory *td = &tif->tif_dir;
	uint32 dx = td->td_tilewidth;
	uint32 dy = td->td_tilelength;
	uint32 dz = td->td_tiledepth;
	ttile_t tile = 1;

	if (td->td_imagedepth == 1)
		z = 0;
	if (dx == (uint32) -1)
		dx = td->td_imagewidth;
	if (dy == (uint32) -1)
		dy = td->td_imagelength;
	if (dz == (uint32) -1)
		dz = td->td_imagedepth;
	if (dx != 0 && dy != 0 && dz != 0) {
		uint32 xpt = TIFFhowmany(td->td_imagewidth, dx); 
		uint32 ypt = TIFFhowmany(td->td_imagelength, dy); 
		uint32 zpt = TIFFhowmany(td->td_imagedepth, dz); 

		if (td->td_planarconfig == PLANARCONFIG_SEPARATE) 
			tile = (xpt*ypt*zpt)*s +
			     (xpt*ypt)*(z/dz) +
			     xpt*(y/dy) +
			     x/dx;
		else
			tile = (xpt*ypt)*(z/dz) + xpt*(y/dy) + x/dx;
	}
	return (tile);
}

/*
 * Check an (x,y,z,s) coordinate
 * against the image bounds.
 */
int
TIFFCheckTile(TIFF* tif, uint32 x, uint32 y, uint32 z, tsample_t s)
{
	TIFFDirectory *td = &tif->tif_dir;

	if (x >= td->td_imagewidth) {
		TIFFError(tif->tif_name, "%lu: Col out of range, max %lu",
		    (unsigned long) x, (unsigned long) td->td_imagewidth);
		return (0);
	}
	if (y >= td->td_imagelength) {
		TIFFError(tif->tif_name, "%lu: Row out of range, max %lu",
		    (unsigned long) y, (unsigned long) td->td_imagelength);
		return (0);
	}
	if (z >= td->td_imagedepth) {
		TIFFError(tif->tif_name, "%lu: Depth out of range, max %lu",
		    (unsigned long) z, (unsigned long) td->td_imagedepth);
		return (0);
	}
	if (td->td_planarconfig == PLANARCONFIG_SEPARATE &&
	    s >= td->td_samplesperpixel) {
		TIFFError(tif->tif_name, "%lu: Sample out of range, max %lu",
		    (unsigned long) s, (unsigned long) td->td_samplesperpixel);
		return (0);
	}
	return (1);
}

/*
 * Compute how many tiles are in an image.
 */
ttile_t
TIFFNumberOfTiles(TIFF* tif)
{
	TIFFDirectory *td = &tif->tif_dir;
	uint32 dx = td->td_tilewidth;
	uint32 dy = td->td_tilelength;
	uint32 dz = td->td_tiledepth;
	ttile_t ntiles;

	if (dx == (uint32) -1)
		dx = td->td_imagewidth;
	if (dy == (uint32) -1)
		dy = td->td_imagelength;
	if (dz == (uint32) -1)
		dz = td->td_imagedepth;
	ntiles = (dx == 0 || dy == 0 || dz == 0) ? 0 :
	    multiply(tif, multiply(tif, TIFFhowmany(td->td_imagewidth, dx),
				   TIFFhowmany(td->td_imagelength, dy),
				   "TIFFNumberOfTiles"),
		     TIFFhowmany(td->td_imagedepth, dz), "TIFFNumberOfTiles");
	if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
		ntiles = multiply(tif, ntiles, td->td_samplesperpixel,
				  "TIFFNumberOfTiles");
	return (ntiles);
}

/*
 * Compute the # bytes in each row of a tile.
 */
tsize_t
TIFFTileRowSize(TIFF* tif)
{
	TIFFDirectory *td = &tif->tif_dir;
	tsize_t rowsize;
	
	if (td->td_tilelength == 0 || td->td_tilewidth == 0)
		return ((tsize_t) 0);
	rowsize = multiply(tif, td->td_bitspersample, td->td_tilewidth,
			   "TIFFTileRowSize");
	if (td->td_planarconfig == PLANARCONFIG_CONTIG)
		rowsize = multiply(tif, rowsize, td->td_samplesperpixel,
				   "TIFFTileRowSize");
	return ((tsize_t) TIFFhowmany8(rowsize));
}

/*
 * Compute the # bytes in a variable length, row-aligned tile.
 */
tsize_t
TIFFVTileSize(TIFF* tif, uint32 nrows)
{
	TIFFDirectory *td = &tif->tif_dir;
	tsize_t tilesize;

	if (td->td_tilelength == 0 || td->td_tilewidth == 0 ||
	    td->td_tiledepth == 0)
		return ((tsize_t) 0);
	if (td->td_planarconfig == PLANARCONFIG_CONTIG &&
	    td->td_photometric == PHOTOMETRIC_YCBCR &&
	    !isUpSampled(tif)) {
		/*
		 * Packed YCbCr data contain one Cb+Cr for every
		 * HorizontalSampling*VerticalSampling Y values.
		 * Must also roundup width and height when calculating
		 * since images that are not a multiple of the
		 * horizontal/vertical subsampling area include
		 * YCbCr data for the extended image.
		 */
		tsize_t w =
		    TIFFroundup(td->td_tilewidth, td->td_ycbcrsubsampling[0]);
		tsize_t rowsize =
		    TIFFhowmany8(multiply(tif, w, td->td_bitspersample,
					  "TIFFVTileSize"));
		tsize_t samplingarea =
		    td->td_ycbcrsubsampling[0]*td->td_ycbcrsubsampling[1];
		if (samplingarea == 0) {
			TIFFError(tif->tif_name, "Invalid YCbCr subsampling");
			return 0;
		}
		nrows = TIFFroundup(nrows, td->td_ycbcrsubsampling[1]);
		/* NB: don't need TIFFhowmany here 'cuz everything is rounded */
		tilesize = multiply(tif, nrows, rowsize, "TIFFVTileSize");
		tilesize = summarize(tif, tilesize,
				     multiply(tif, 2, tilesize / samplingarea,
					      "TIFFVTileSize"),
				     "TIFFVTileSize");
	} else
		tilesize = multiply(tif, nrows, TIFFTileRowSize(tif),
				    "TIFFVTileSize");
	return ((tsize_t)
	    multiply(tif, tilesize, td->td_tiledepth, "TIFFVTileSize"));
}

/*
 * Compute the # bytes in a row-aligned tile.
 */
tsize_t
TIFFTileSize(TIFF* tif)
{
	return (TIFFVTileSize(tif, tif->tif_dir.td_tilelength));
}

/*
 * Compute a default tile size based on the image
 * characteristics and a requested value.  If a
 * request is <1 then we choose a size according
 * to certain heuristics.
 */
void
TIFFDefaultTileSize(TIFF* tif, uint32* tw, uint32* th)
{
	(*tif->tif_deftilesize)(tif, tw, th);
}

void
_TIFFDefaultTileSize(TIFF* tif, uint32* tw, uint32* th)
{
	(void) tif;
	if (*(int32*) tw < 1)
		*tw = 256;
	if (*(int32*) th < 1)
		*th = 256;
	/* roundup to a multiple of 16 per the spec */
	if (*tw & 0xf)
		*tw = TIFFroundup(*tw, 16);
	if (*th & 0xf)
		*th = TIFFroundup(*th, 16);
}

/* vim: set ts=8 sts=8 sw=8 noet: */
