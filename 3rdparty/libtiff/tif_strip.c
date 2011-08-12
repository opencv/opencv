/* $Id: tif_strip.c,v 1.19.2.3 2010-12-15 00:50:30 faxguy Exp $ */

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
 * Strip-organized Image Support Routines.
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
		TIFFErrorExt(tif->tif_clientdata, tif->tif_name, "Integer overflow in %s", where);
		bytes = 0;
	}

	return (bytes);
}

static uint32
multiply(TIFF* tif, size_t nmemb, size_t elem_size, const char* where)
{
	uint32	bytes = nmemb * elem_size;

	if (elem_size && bytes / elem_size != nmemb) {
		TIFFErrorExt(tif->tif_clientdata, tif->tif_name, "Integer overflow in %s", where);
		bytes = 0;
	}

	return (bytes);
}

/*
 * Compute which strip a (row,sample) value is in.
 */
tstrip_t
TIFFComputeStrip(TIFF* tif, uint32 row, tsample_t sample)
{
	TIFFDirectory *td = &tif->tif_dir;
	tstrip_t strip;

	strip = row / td->td_rowsperstrip;
	if (td->td_planarconfig == PLANARCONFIG_SEPARATE) {
		if (sample >= td->td_samplesperpixel) {
			TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
			    "%lu: Sample out of range, max %lu",
			    (unsigned long) sample, (unsigned long) td->td_samplesperpixel);
			return ((tstrip_t) 0);
		}
		strip += sample*td->td_stripsperimage;
	}
	return (strip);
}

/*
 * Compute how many strips are in an image.
 */
tstrip_t
TIFFNumberOfStrips(TIFF* tif)
{
	TIFFDirectory *td = &tif->tif_dir;
	tstrip_t nstrips;

	nstrips = (td->td_rowsperstrip == (uint32) -1 ? 1 :
	     TIFFhowmany(td->td_imagelength, td->td_rowsperstrip));
	if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
		nstrips = multiply(tif, nstrips, td->td_samplesperpixel,
				   "TIFFNumberOfStrips");
	return (nstrips);
}

/*
 * Compute the # bytes in a variable height, row-aligned strip.
 */
tsize_t
TIFFVStripSize(TIFF* tif, uint32 nrows)
{
	TIFFDirectory *td = &tif->tif_dir;

	if (nrows == (uint32) -1)
		nrows = td->td_imagelength;
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
		uint16 ycbcrsubsampling[2];
		tsize_t w, scanline, samplingarea;

		TIFFGetFieldDefaulted(tif, TIFFTAG_YCBCRSUBSAMPLING,
				      ycbcrsubsampling + 0,
				      ycbcrsubsampling + 1);

		samplingarea = ycbcrsubsampling[0]*ycbcrsubsampling[1];
		if (samplingarea == 0) {
			TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
				     "Invalid YCbCr subsampling");
			return 0;
		}

		w = TIFFroundup(td->td_imagewidth, ycbcrsubsampling[0]);
		scanline = TIFFhowmany8(multiply(tif, w, td->td_bitspersample,
						 "TIFFVStripSize"));
		nrows = TIFFroundup(nrows, ycbcrsubsampling[1]);
		/* NB: don't need TIFFhowmany here 'cuz everything is rounded */
		scanline = multiply(tif, nrows, scanline, "TIFFVStripSize");
		return ((tsize_t)
		    summarize(tif, scanline,
			      multiply(tif, 2, scanline / samplingarea,
				       "TIFFVStripSize"), "TIFFVStripSize"));
	} else
		return ((tsize_t) multiply(tif, nrows, TIFFScanlineSize(tif),
					   "TIFFVStripSize"));
}


/*
 * Compute the # bytes in a raw strip.
 */
tsize_t
TIFFRawStripSize(TIFF* tif, tstrip_t strip)
{
	TIFFDirectory* td = &tif->tif_dir;
	tsize_t bytecount = td->td_stripbytecount[strip];

	if (bytecount <= 0) {
		TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
			  "%lu: Invalid strip byte count, strip %lu",
			  (unsigned long) bytecount, (unsigned long) strip);
		bytecount = (tsize_t) -1;
	}

	return bytecount;
}

/*
 * Compute the # bytes in a (row-aligned) strip.
 *
 * Note that if RowsPerStrip is larger than the
 * recorded ImageLength, then the strip size is
 * truncated to reflect the actual space required
 * to hold the strip.
 */
tsize_t
TIFFStripSize(TIFF* tif)
{
	TIFFDirectory* td = &tif->tif_dir;
	uint32 rps = td->td_rowsperstrip;
	if (rps > td->td_imagelength)
		rps = td->td_imagelength;
	return (TIFFVStripSize(tif, rps));
}

/*
 * Compute a default strip size based on the image
 * characteristics and a requested value.  If the
 * request is <1 then we choose a strip size according
 * to certain heuristics.
 */
uint32
TIFFDefaultStripSize(TIFF* tif, uint32 request)
{
	return (*tif->tif_defstripsize)(tif, request);
}

uint32
_TIFFDefaultStripSize(TIFF* tif, uint32 s)
{
	if ((int32) s < 1) {
		/*
		 * If RowsPerStrip is unspecified, try to break the
		 * image up into strips that are approximately
		 * STRIP_SIZE_DEFAULT bytes long.
		 */
		tsize_t scanline = TIFFScanlineSize(tif);
		s = (uint32)STRIP_SIZE_DEFAULT / (scanline == 0 ? 1 : scanline);
		if (s == 0)		/* very wide images */
			s = 1;
	}
	return (s);
}

/*
 * Return the number of bytes to read/write in a call to
 * one of the scanline-oriented i/o routines.  Note that
 * this number may be 1/samples-per-pixel if data is
 * stored as separate planes.
 */
tsize_t
TIFFScanlineSize(TIFF* tif)
{
	TIFFDirectory *td = &tif->tif_dir;
	tsize_t scanline;

	if (td->td_planarconfig == PLANARCONFIG_CONTIG) {
		if (td->td_photometric == PHOTOMETRIC_YCBCR
		    && !isUpSampled(tif)) {
			uint16 ycbcrsubsampling[2];

			TIFFGetFieldDefaulted(tif, TIFFTAG_YCBCRSUBSAMPLING,
					      ycbcrsubsampling + 0,
					      ycbcrsubsampling + 1);

			if (ycbcrsubsampling[0]*ycbcrsubsampling[1] == 0) {
				TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
					     "Invalid YCbCr subsampling");
				return 0;
			}

			/* number of sample clumps per line */
			scanline = TIFFhowmany(td->td_imagewidth,
					       ycbcrsubsampling[0]);
			/* number of samples per line */
			scanline = multiply(tif, scanline,
					    ycbcrsubsampling[0]*ycbcrsubsampling[1] + 2,
					    "TIFFScanlineSize");
		} else {
			scanline = multiply(tif, td->td_imagewidth,
					    td->td_samplesperpixel,
					    "TIFFScanlineSize");
		}
	} else
		scanline = td->td_imagewidth;
	return ((tsize_t) TIFFhowmany8(multiply(tif, scanline,
						td->td_bitspersample,
						"TIFFScanlineSize")));
}

/*
 * Some stuff depends on this older version of TIFFScanlineSize
 * TODO: resolve this
 */
tsize_t
TIFFOldScanlineSize(TIFF* tif)
{
	TIFFDirectory *td = &tif->tif_dir;
	tsize_t scanline;

	scanline = multiply (tif, td->td_bitspersample, td->td_imagewidth,
			     "TIFFScanlineSize");
	if (td->td_planarconfig == PLANARCONFIG_CONTIG)
		scanline = multiply (tif, scanline, td->td_samplesperpixel,
				     "TIFFScanlineSize");
	return ((tsize_t) TIFFhowmany8(scanline));
}

/*
 * Return the number of bytes to read/write in a call to
 * one of the scanline-oriented i/o routines.  Note that
 * this number may be 1/samples-per-pixel if data is
 * stored as separate planes.
 * The ScanlineSize in case of YCbCrSubsampling is defined as the
 * strip size divided by the strip height, i.e. the size of a pack of vertical
 * subsampling lines divided by vertical subsampling. It should thus make
 * sense when multiplied by a multiple of vertical subsampling.
 * Some stuff depends on this newer version of TIFFScanlineSize
 * TODO: resolve this
 */
tsize_t
TIFFNewScanlineSize(TIFF* tif)
{
	TIFFDirectory *td = &tif->tif_dir;
	tsize_t scanline;

	if (td->td_planarconfig == PLANARCONFIG_CONTIG) {
		if (td->td_photometric == PHOTOMETRIC_YCBCR
		    && !isUpSampled(tif)) {
			uint16 ycbcrsubsampling[2];

			TIFFGetFieldDefaulted(tif, TIFFTAG_YCBCRSUBSAMPLING,
					      ycbcrsubsampling + 0,
					      ycbcrsubsampling + 1);

			if (ycbcrsubsampling[0]*ycbcrsubsampling[1] == 0) {
				TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
					     "Invalid YCbCr subsampling");
				return 0;
			}

			return((tsize_t) ((((td->td_imagewidth+ycbcrsubsampling[0]-1)
					    /ycbcrsubsampling[0])
					   *(ycbcrsubsampling[0]*ycbcrsubsampling[1]+2)
					   *td->td_bitspersample+7)
					  /8)/ycbcrsubsampling[1]);

		} else {
			scanline = multiply(tif, td->td_imagewidth,
					    td->td_samplesperpixel,
					    "TIFFScanlineSize");
		}
	} else
		scanline = td->td_imagewidth;
	return ((tsize_t) TIFFhowmany8(multiply(tif, scanline,
						td->td_bitspersample,
						"TIFFScanlineSize")));
}

/*
 * Return the number of bytes required to store a complete
 * decoded and packed raster scanline (as opposed to the
 * I/O size returned by TIFFScanlineSize which may be less
 * if data is store as separate planes).
 */
tsize_t
TIFFRasterScanlineSize(TIFF* tif)
{
	TIFFDirectory *td = &tif->tif_dir;
	tsize_t scanline;
	
	scanline = multiply (tif, td->td_bitspersample, td->td_imagewidth,
			     "TIFFRasterScanlineSize");
	if (td->td_planarconfig == PLANARCONFIG_CONTIG) {
		scanline = multiply (tif, scanline, td->td_samplesperpixel,
				     "TIFFRasterScanlineSize");
		return ((tsize_t) TIFFhowmany8(scanline));
	} else
		return ((tsize_t) multiply (tif, TIFFhowmany8(scanline),
					    td->td_samplesperpixel,
					    "TIFFRasterScanlineSize"));
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
