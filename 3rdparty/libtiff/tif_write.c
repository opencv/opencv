/* $Id: tif_write.c,v 1.36 2011-02-18 20:53:04 fwarmerdam Exp $ */

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
 * Scanline-oriented Write Support
 */
#include "tiffiop.h"
#include <stdio.h>

#define STRIPINCR	20		/* expansion factor on strip array */

#define WRITECHECKSTRIPS(tif, module)				\
	(((tif)->tif_flags&TIFF_BEENWRITING) || TIFFWriteCheck((tif),0,module))
#define WRITECHECKTILES(tif, module)				\
	(((tif)->tif_flags&TIFF_BEENWRITING) || TIFFWriteCheck((tif),1,module))
#define BUFFERCHECK(tif)					\
	((((tif)->tif_flags & TIFF_BUFFERSETUP) && tif->tif_rawdata) ||	\
	    TIFFWriteBufferSetup((tif), NULL, (tmsize_t) -1))

static int TIFFGrowStrips(TIFF* tif, uint32 delta, const char* module);
static int TIFFAppendToStrip(TIFF* tif, uint32 strip, uint8* data, tmsize_t cc);

int
TIFFWriteScanline(TIFF* tif, void* buf, uint32 row, uint16 sample)
{
	static const char module[] = "TIFFWriteScanline";
	register TIFFDirectory *td;
	int status, imagegrew = 0;
	uint32 strip;

	if (!WRITECHECKSTRIPS(tif, module))
		return (-1);
	/*
	 * Handle delayed allocation of data buffer.  This
	 * permits it to be sized more intelligently (using
	 * directory information).
	 */
	if (!BUFFERCHECK(tif))
		return (-1);
        tif->tif_flags |= TIFF_BUF4WRITE; /* not strictly sure this is right*/

	td = &tif->tif_dir;
	/*
	 * Extend image length if needed
	 * (but only for PlanarConfig=1).
	 */
	if (row >= td->td_imagelength) {	/* extend image */
		if (td->td_planarconfig == PLANARCONFIG_SEPARATE) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "Can not change \"ImageLength\" when using separate planes");
			return (-1);
		}
		td->td_imagelength = row+1;
		imagegrew = 1;
	}
	/*
	 * Calculate strip and check for crossings.
	 */
	if (td->td_planarconfig == PLANARCONFIG_SEPARATE) {
		if (sample >= td->td_samplesperpixel) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "%lu: Sample out of range, max %lu",
			    (unsigned long) sample, (unsigned long) td->td_samplesperpixel);
			return (-1);
		}
		strip = sample*td->td_stripsperimage + row/td->td_rowsperstrip;
	} else
		strip = row / td->td_rowsperstrip;
	/*
	 * Check strip array to make sure there's space. We don't support
	 * dynamically growing files that have data organized in separate
	 * bitplanes because it's too painful.  In that case we require that
	 * the imagelength be set properly before the first write (so that the
	 * strips array will be fully allocated above).
	 */
	if (strip >= td->td_nstrips && !TIFFGrowStrips(tif, 1, module))
		return (-1);
	if (strip != tif->tif_curstrip) {
		/*
		 * Changing strips -- flush any data present.
		 */
		if (!TIFFFlushData(tif))
			return (-1);
		tif->tif_curstrip = strip;
		/*
		 * Watch out for a growing image.  The value of strips/image
		 * will initially be 1 (since it can't be deduced until the
		 * imagelength is known).
		 */
		if (strip >= td->td_stripsperimage && imagegrew)
			td->td_stripsperimage =
			    TIFFhowmany_32(td->td_imagelength,td->td_rowsperstrip);
		tif->tif_row =
		    (strip % td->td_stripsperimage) * td->td_rowsperstrip;
		if ((tif->tif_flags & TIFF_CODERSETUP) == 0) {
			if (!(*tif->tif_setupencode)(tif))
				return (-1);
			tif->tif_flags |= TIFF_CODERSETUP;
		}
        
		tif->tif_rawcc = 0;
		tif->tif_rawcp = tif->tif_rawdata;

		if( td->td_stripbytecount[strip] > 0 )
		{
			/* if we are writing over existing tiles, zero length */
			td->td_stripbytecount[strip] = 0;

			/* this forces TIFFAppendToStrip() to do a seek */
			tif->tif_curoff = 0;
		}

		if (!(*tif->tif_preencode)(tif, sample))
			return (-1);
		tif->tif_flags |= TIFF_POSTENCODE;
	}
	/*
	 * Ensure the write is either sequential or at the
	 * beginning of a strip (or that we can randomly
	 * access the data -- i.e. no encoding).
	 */
	if (row != tif->tif_row) {
		if (row < tif->tif_row) {
			/*
			 * Moving backwards within the same strip:
			 * backup to the start and then decode
			 * forward (below).
			 */
			tif->tif_row = (strip % td->td_stripsperimage) *
			    td->td_rowsperstrip;
			tif->tif_rawcp = tif->tif_rawdata;
		}
		/*
		 * Seek forward to the desired row.
		 */
		if (!(*tif->tif_seek)(tif, row - tif->tif_row))
			return (-1);
		tif->tif_row = row;
	}

	/* swab if needed - note that source buffer will be altered */
	tif->tif_postdecode( tif, (uint8*) buf, tif->tif_scanlinesize );

	status = (*tif->tif_encoderow)(tif, (uint8*) buf,
	    tif->tif_scanlinesize, sample);

        /* we are now poised at the beginning of the next row */
	tif->tif_row = row + 1;
	return (status);
}

/*
 * Encode the supplied data and write it to the
 * specified strip.
 *
 * NB: Image length must be setup before writing.
 */
tmsize_t
TIFFWriteEncodedStrip(TIFF* tif, uint32 strip, void* data, tmsize_t cc)
{
	static const char module[] = "TIFFWriteEncodedStrip";
	TIFFDirectory *td = &tif->tif_dir;
	uint16 sample;

	if (!WRITECHECKSTRIPS(tif, module))
		return ((tmsize_t) -1);
	/*
	 * Check strip array to make sure there's space.
	 * We don't support dynamically growing files that
	 * have data organized in separate bitplanes because
	 * it's too painful.  In that case we require that
	 * the imagelength be set properly before the first
	 * write (so that the strips array will be fully
	 * allocated above).
	 */
	if (strip >= td->td_nstrips) {
		if (td->td_planarconfig == PLANARCONFIG_SEPARATE) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "Can not grow image by strips when using separate planes");
			return ((tmsize_t) -1);
		}
		if (!TIFFGrowStrips(tif, 1, module))
			return ((tmsize_t) -1);
		td->td_stripsperimage =
		    TIFFhowmany_32(td->td_imagelength, td->td_rowsperstrip);  
	}
	/*
	 * Handle delayed allocation of data buffer.  This
	 * permits it to be sized according to the directory
	 * info.
	 */
	if (!BUFFERCHECK(tif))
		return ((tmsize_t) -1);

        tif->tif_flags |= TIFF_BUF4WRITE;
	tif->tif_curstrip = strip;

	tif->tif_row = (strip % td->td_stripsperimage) * td->td_rowsperstrip;
	if ((tif->tif_flags & TIFF_CODERSETUP) == 0) {
		if (!(*tif->tif_setupencode)(tif))
			return ((tmsize_t) -1);
		tif->tif_flags |= TIFF_CODERSETUP;
	}
        
	tif->tif_rawcc = 0;
	tif->tif_rawcp = tif->tif_rawdata;  

	if( td->td_stripbytecount[strip] > 0 )
        {
	    /* Force TIFFAppendToStrip() to consider placing data at end
               of file. */
            tif->tif_curoff = 0;
        }
        
	tif->tif_flags &= ~TIFF_POSTENCODE;
	sample = (uint16)(strip / td->td_stripsperimage);
	if (!(*tif->tif_preencode)(tif, sample))
		return ((tmsize_t) -1);

        /* swab if needed - note that source buffer will be altered */
	tif->tif_postdecode( tif, (uint8*) data, cc );

	if (!(*tif->tif_encodestrip)(tif, (uint8*) data, cc, sample))
		return (0);
	if (!(*tif->tif_postencode)(tif))
		return ((tmsize_t) -1);
	if (!isFillOrder(tif, td->td_fillorder) &&
	    (tif->tif_flags & TIFF_NOBITREV) == 0)
		TIFFReverseBits(tif->tif_rawdata, tif->tif_rawcc);
	if (tif->tif_rawcc > 0 &&
	    !TIFFAppendToStrip(tif, strip, tif->tif_rawdata, tif->tif_rawcc))
		return ((tmsize_t) -1);
	tif->tif_rawcc = 0;
	tif->tif_rawcp = tif->tif_rawdata;
	return (cc);
}

/*
 * Write the supplied data to the specified strip.
 *
 * NB: Image length must be setup before writing.
 */
tmsize_t
TIFFWriteRawStrip(TIFF* tif, uint32 strip, void* data, tmsize_t cc)
{
	static const char module[] = "TIFFWriteRawStrip";
	TIFFDirectory *td = &tif->tif_dir;

	if (!WRITECHECKSTRIPS(tif, module))
		return ((tmsize_t) -1);
	/*
	 * Check strip array to make sure there's space.
	 * We don't support dynamically growing files that
	 * have data organized in separate bitplanes because
	 * it's too painful.  In that case we require that
	 * the imagelength be set properly before the first
	 * write (so that the strips array will be fully
	 * allocated above).
	 */
	if (strip >= td->td_nstrips) {
		if (td->td_planarconfig == PLANARCONFIG_SEPARATE) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "Can not grow image by strips when using separate planes");
			return ((tmsize_t) -1);
		}
		/*
		 * Watch out for a growing image.  The value of
		 * strips/image will initially be 1 (since it
		 * can't be deduced until the imagelength is known).
		 */
		if (strip >= td->td_stripsperimage)
			td->td_stripsperimage =
			    TIFFhowmany_32(td->td_imagelength,td->td_rowsperstrip);
		if (!TIFFGrowStrips(tif, 1, module))
			return ((tmsize_t) -1);
	}
	tif->tif_curstrip = strip;
	tif->tif_row = (strip % td->td_stripsperimage) * td->td_rowsperstrip;
	return (TIFFAppendToStrip(tif, strip, (uint8*) data, cc) ?
	    cc : (tmsize_t) -1);
}

/*
 * Write and compress a tile of data.  The
 * tile is selected by the (x,y,z,s) coordinates.
 */
tmsize_t
TIFFWriteTile(TIFF* tif, void* buf, uint32 x, uint32 y, uint32 z, uint16 s)
{
	if (!TIFFCheckTile(tif, x, y, z, s))
		return ((tmsize_t)(-1));
	/*
	 * NB: A tile size of -1 is used instead of tif_tilesize knowing
	 *     that TIFFWriteEncodedTile will clamp this to the tile size.
	 *     This is done because the tile size may not be defined until
	 *     after the output buffer is setup in TIFFWriteBufferSetup.
	 */
	return (TIFFWriteEncodedTile(tif,
	    TIFFComputeTile(tif, x, y, z, s), buf, (tmsize_t)(-1)));
}

/*
 * Encode the supplied data and write it to the
 * specified tile.  There must be space for the
 * data.  The function clamps individual writes
 * to a tile to the tile size, but does not (and
 * can not) check that multiple writes to the same
 * tile do not write more than tile size data.
 *
 * NB: Image length must be setup before writing; this
 *     interface does not support automatically growing
 *     the image on each write (as TIFFWriteScanline does).
 */
tmsize_t
TIFFWriteEncodedTile(TIFF* tif, uint32 tile, void* data, tmsize_t cc)
{
	static const char module[] = "TIFFWriteEncodedTile";
	TIFFDirectory *td;
	uint16 sample;

	if (!WRITECHECKTILES(tif, module))
		return ((tmsize_t)(-1));
	td = &tif->tif_dir;
	if (tile >= td->td_nstrips) {
		TIFFErrorExt(tif->tif_clientdata, module, "Tile %lu out of range, max %lu",
		    (unsigned long) tile, (unsigned long) td->td_nstrips);
		return ((tmsize_t)(-1));
	}
	/*
	 * Handle delayed allocation of data buffer.  This
	 * permits it to be sized more intelligently (using
	 * directory information).
	 */
	if (!BUFFERCHECK(tif))
		return ((tmsize_t)(-1));

        tif->tif_flags |= TIFF_BUF4WRITE;
	tif->tif_curtile = tile;

	tif->tif_rawcc = 0;
	tif->tif_rawcp = tif->tif_rawdata;

	if( td->td_stripbytecount[tile] > 0 )
        {
	    /* Force TIFFAppendToStrip() to consider placing data at end
               of file. */
            tif->tif_curoff = 0;
        }
        
	/* 
	 * Compute tiles per row & per column to compute
	 * current row and column
	 */
	tif->tif_row = (tile % TIFFhowmany_32(td->td_imagelength, td->td_tilelength))
		* td->td_tilelength;
	tif->tif_col = (tile % TIFFhowmany_32(td->td_imagewidth, td->td_tilewidth))
		* td->td_tilewidth;

	if ((tif->tif_flags & TIFF_CODERSETUP) == 0) {
		if (!(*tif->tif_setupencode)(tif))
			return ((tmsize_t)(-1));
		tif->tif_flags |= TIFF_CODERSETUP;
	}
	tif->tif_flags &= ~TIFF_POSTENCODE;
	sample = (uint16)(tile/td->td_stripsperimage);
	if (!(*tif->tif_preencode)(tif, sample))
		return ((tmsize_t)(-1));
	/*
	 * Clamp write amount to the tile size.  This is mostly
	 * done so that callers can pass in some large number
	 * (e.g. -1) and have the tile size used instead.
	 */
	if ( cc < 1 || cc > tif->tif_tilesize)
		cc = tif->tif_tilesize;

        /* swab if needed - note that source buffer will be altered */
	tif->tif_postdecode( tif, (uint8*) data, cc );

	if (!(*tif->tif_encodetile)(tif, (uint8*) data, cc, sample))
		return (0);
	if (!(*tif->tif_postencode)(tif))
		return ((tmsize_t)(-1));
	if (!isFillOrder(tif, td->td_fillorder) &&
	    (tif->tif_flags & TIFF_NOBITREV) == 0)
		TIFFReverseBits((uint8*)tif->tif_rawdata, tif->tif_rawcc);
	if (tif->tif_rawcc > 0 && !TIFFAppendToStrip(tif, tile,
	    tif->tif_rawdata, tif->tif_rawcc))
		return ((tmsize_t)(-1));
	tif->tif_rawcc = 0;
	tif->tif_rawcp = tif->tif_rawdata;
	return (cc);
}

/*
 * Write the supplied data to the specified strip.
 * There must be space for the data; we don't check
 * if strips overlap!
 *
 * NB: Image length must be setup before writing; this
 *     interface does not support automatically growing
 *     the image on each write (as TIFFWriteScanline does).
 */
tmsize_t
TIFFWriteRawTile(TIFF* tif, uint32 tile, void* data, tmsize_t cc)
{
	static const char module[] = "TIFFWriteRawTile";

	if (!WRITECHECKTILES(tif, module))
		return ((tmsize_t)(-1));
	if (tile >= tif->tif_dir.td_nstrips) {
		TIFFErrorExt(tif->tif_clientdata, module, "Tile %lu out of range, max %lu",
		    (unsigned long) tile,
		    (unsigned long) tif->tif_dir.td_nstrips);
		return ((tmsize_t)(-1));
	}
	return (TIFFAppendToStrip(tif, tile, (uint8*) data, cc) ?
	    cc : (tmsize_t)(-1));
}

#define	isUnspecified(tif, f) \
    (TIFFFieldSet(tif,f) && (tif)->tif_dir.td_imagelength == 0)

int
TIFFSetupStrips(TIFF* tif)
{
	TIFFDirectory* td = &tif->tif_dir;

	if (isTiled(tif))
		td->td_stripsperimage =
		    isUnspecified(tif, FIELD_TILEDIMENSIONS) ?
			td->td_samplesperpixel : TIFFNumberOfTiles(tif);
	else
		td->td_stripsperimage =
		    isUnspecified(tif, FIELD_ROWSPERSTRIP) ?
			td->td_samplesperpixel : TIFFNumberOfStrips(tif);
	td->td_nstrips = td->td_stripsperimage;
	if (td->td_planarconfig == PLANARCONFIG_SEPARATE)
		td->td_stripsperimage /= td->td_samplesperpixel;
	td->td_stripoffset = (uint64 *)
	    _TIFFmalloc(td->td_nstrips * sizeof (uint64));
	td->td_stripbytecount = (uint64 *)
	    _TIFFmalloc(td->td_nstrips * sizeof (uint64));
	if (td->td_stripoffset == NULL || td->td_stripbytecount == NULL)
		return (0);
	/*
	 * Place data at the end-of-file
	 * (by setting offsets to zero).
	 */
	_TIFFmemset(td->td_stripoffset, 0, td->td_nstrips*sizeof (uint64));
	_TIFFmemset(td->td_stripbytecount, 0, td->td_nstrips*sizeof (uint64));
	TIFFSetFieldBit(tif, FIELD_STRIPOFFSETS);
	TIFFSetFieldBit(tif, FIELD_STRIPBYTECOUNTS);
	return (1);
}
#undef isUnspecified

/*
 * Verify file is writable and that the directory
 * information is setup properly.  In doing the latter
 * we also "freeze" the state of the directory so
 * that important information is not changed.
 */
int
TIFFWriteCheck(TIFF* tif, int tiles, const char* module)
{
	if (tif->tif_mode == O_RDONLY) {
		TIFFErrorExt(tif->tif_clientdata, module, "File not open for writing");
		return (0);
	}
	if (tiles ^ isTiled(tif)) {
		TIFFErrorExt(tif->tif_clientdata, module, tiles ?
		    "Can not write tiles to a stripped image" :
		    "Can not write scanlines to a tiled image");
		return (0);
	}

        _TIFFFillStriles( tif );
        
	/*
	 * On the first write verify all the required information
	 * has been setup and initialize any data structures that
	 * had to wait until directory information was set.
	 * Note that a lot of our work is assumed to remain valid
	 * because we disallow any of the important parameters
	 * from changing after we start writing (i.e. once
	 * TIFF_BEENWRITING is set, TIFFSetField will only allow
	 * the image's length to be changed).
	 */
	if (!TIFFFieldSet(tif, FIELD_IMAGEDIMENSIONS)) {
		TIFFErrorExt(tif->tif_clientdata, module,
		    "Must set \"ImageWidth\" before writing data");
		return (0);
	}
	if (tif->tif_dir.td_samplesperpixel == 1) {
		/* 
		 * Planarconfiguration is irrelevant in case of single band
		 * images and need not be included. We will set it anyway,
		 * because this field is used in other parts of library even
		 * in the single band case.
		 */
		if (!TIFFFieldSet(tif, FIELD_PLANARCONFIG))
                    tif->tif_dir.td_planarconfig = PLANARCONFIG_CONTIG;
	} else {
		if (!TIFFFieldSet(tif, FIELD_PLANARCONFIG)) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "Must set \"PlanarConfiguration\" before writing data");
			return (0);
		}
	}
	if (tif->tif_dir.td_stripoffset == NULL && !TIFFSetupStrips(tif)) {
		tif->tif_dir.td_nstrips = 0;
		TIFFErrorExt(tif->tif_clientdata, module, "No space for %s arrays",
		    isTiled(tif) ? "tile" : "strip");
		return (0);
	}
	if (isTiled(tif))
	{
		tif->tif_tilesize = TIFFTileSize(tif);
		if (tif->tif_tilesize == 0)
			return (0);
	}
	else
		tif->tif_tilesize = (tmsize_t)(-1);
	tif->tif_scanlinesize = TIFFScanlineSize(tif);
	if (tif->tif_scanlinesize == 0)
		return (0);
	tif->tif_flags |= TIFF_BEENWRITING;
	return (1);
}

/*
 * Setup the raw data buffer used for encoding.
 */
int
TIFFWriteBufferSetup(TIFF* tif, void* bp, tmsize_t size)
{
	static const char module[] = "TIFFWriteBufferSetup";

	if (tif->tif_rawdata) {
		if (tif->tif_flags & TIFF_MYBUFFER) {
			_TIFFfree(tif->tif_rawdata);
			tif->tif_flags &= ~TIFF_MYBUFFER;
		}
		tif->tif_rawdata = NULL;
	}
	if (size == (tmsize_t)(-1)) {
		size = (isTiled(tif) ?
		    tif->tif_tilesize : TIFFStripSize(tif));
		/*
		 * Make raw data buffer at least 8K
		 */
		if (size < 8*1024)
			size = 8*1024;
		bp = NULL;			/* NB: force malloc */
	}
	if (bp == NULL) {
		bp = _TIFFmalloc(size);
		if (bp == NULL) {
			TIFFErrorExt(tif->tif_clientdata, module, "No space for output buffer");
			return (0);
		}
		tif->tif_flags |= TIFF_MYBUFFER;
	} else
		tif->tif_flags &= ~TIFF_MYBUFFER;
	tif->tif_rawdata = (uint8*) bp;
	tif->tif_rawdatasize = size;
	tif->tif_rawcc = 0;
	tif->tif_rawcp = tif->tif_rawdata;
	tif->tif_flags |= TIFF_BUFFERSETUP;
	return (1);
}

/*
 * Grow the strip data structures by delta strips.
 */
static int
TIFFGrowStrips(TIFF* tif, uint32 delta, const char* module)
{
	TIFFDirectory *td = &tif->tif_dir;
	uint64* new_stripoffset;
	uint64* new_stripbytecount;

	assert(td->td_planarconfig == PLANARCONFIG_CONTIG);
	new_stripoffset = (uint64*)_TIFFrealloc(td->td_stripoffset,
		(td->td_nstrips + delta) * sizeof (uint64));
	new_stripbytecount = (uint64*)_TIFFrealloc(td->td_stripbytecount,
		(td->td_nstrips + delta) * sizeof (uint64));
	if (new_stripoffset == NULL || new_stripbytecount == NULL) {
		if (new_stripoffset)
			_TIFFfree(new_stripoffset);
		if (new_stripbytecount)
			_TIFFfree(new_stripbytecount);
		td->td_nstrips = 0;
		TIFFErrorExt(tif->tif_clientdata, module, "No space to expand strip arrays");
		return (0);
	}
	td->td_stripoffset = new_stripoffset;
	td->td_stripbytecount = new_stripbytecount;
	_TIFFmemset(td->td_stripoffset + td->td_nstrips,
		    0, delta*sizeof (uint64));
	_TIFFmemset(td->td_stripbytecount + td->td_nstrips,
		    0, delta*sizeof (uint64));
	td->td_nstrips += delta;
        tif->tif_flags |= TIFF_DIRTYDIRECT;

	return (1);
}

/*
 * Append the data to the specified strip.
 */
static int
TIFFAppendToStrip(TIFF* tif, uint32 strip, uint8* data, tmsize_t cc)
{
	static const char module[] = "TIFFAppendToStrip";
	TIFFDirectory *td = &tif->tif_dir;
	uint64 m;
        int64 old_byte_count = -1;

	if (td->td_stripoffset[strip] == 0 || tif->tif_curoff == 0) {
            assert(td->td_nstrips > 0);

            if( td->td_stripbytecount[strip] != 0 
                && td->td_stripoffset[strip] != 0 
                && td->td_stripbytecount[strip] >= (uint64) cc )
            {
                /* 
                 * There is already tile data on disk, and the new tile
                 * data we have will fit in the same space.  The only 
                 * aspect of this that is risky is that there could be
                 * more data to append to this strip before we are done
                 * depending on how we are getting called.
                 */
                if (!SeekOK(tif, td->td_stripoffset[strip])) {
                    TIFFErrorExt(tif->tif_clientdata, module,
                                 "Seek error at scanline %lu",
                                 (unsigned long)tif->tif_row);
                    return (0);
                }
            }
            else
            {
                /* 
                 * Seek to end of file, and set that as our location to 
                 * write this strip.
                 */
                td->td_stripoffset[strip] = TIFFSeekFile(tif, 0, SEEK_END);
                tif->tif_flags |= TIFF_DIRTYSTRIP;
            }

            tif->tif_curoff = td->td_stripoffset[strip];

            /*
             * We are starting a fresh strip/tile, so set the size to zero.
             */
            old_byte_count = td->td_stripbytecount[strip];
            td->td_stripbytecount[strip] = 0;
	}

	m = tif->tif_curoff+cc;
	if (!(tif->tif_flags&TIFF_BIGTIFF))
		m = (uint32)m;
	if ((m<tif->tif_curoff)||(m<(uint64)cc))
	{
		TIFFErrorExt(tif->tif_clientdata, module, "Maximum TIFF file size exceeded");
		return (0);
	}
	if (!WriteOK(tif, data, cc)) {
		TIFFErrorExt(tif->tif_clientdata, module, "Write error at scanline %lu",
		    (unsigned long) tif->tif_row);
		    return (0);
	}
	tif->tif_curoff = m;
	td->td_stripbytecount[strip] += cc;

        if( (int64) td->td_stripbytecount[strip] != old_byte_count )
            tif->tif_flags |= TIFF_DIRTYSTRIP;
            
	return (1);
}

/*
 * Internal version of TIFFFlushData that can be
 * called by ``encodestrip routines'' w/o concern
 * for infinite recursion.
 */
int
TIFFFlushData1(TIFF* tif)
{
	if (tif->tif_rawcc > 0 && tif->tif_flags & TIFF_BUF4WRITE ) {
		if (!isFillOrder(tif, tif->tif_dir.td_fillorder) &&
		    (tif->tif_flags & TIFF_NOBITREV) == 0)
			TIFFReverseBits((uint8*)tif->tif_rawdata,
			    tif->tif_rawcc);
		if (!TIFFAppendToStrip(tif,
		    isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip,
		    tif->tif_rawdata, tif->tif_rawcc))
			return (0);
		tif->tif_rawcc = 0;
		tif->tif_rawcp = tif->tif_rawdata;
	}
	return (1);
}

/*
 * Set the current write offset.  This should only be
 * used to set the offset to a known previous location
 * (very carefully), or to 0 so that the next write gets
 * appended to the end of the file.
 */
void
TIFFSetWriteOffset(TIFF* tif, toff_t off)
{
	tif->tif_curoff = off;
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
