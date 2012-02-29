/* $Id: tif_read.c,v 1.38 2011-12-09 03:29:10 fwarmerdam Exp $ */

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
 * Scanline-oriented Read Support
 */
#include "tiffiop.h"
#include <stdio.h>

int TIFFFillStrip(TIFF* tif, uint32 strip);
int TIFFFillTile(TIFF* tif, uint32 tile);
static int TIFFStartStrip(TIFF* tif, uint32 strip);
static int TIFFStartTile(TIFF* tif, uint32 tile);
static int TIFFCheckRead(TIFF*, int);
static tmsize_t
TIFFReadRawStrip1(TIFF* tif, uint32 strip, void* buf, tmsize_t size,const char* module);

#define NOSTRIP ((uint32)(-1))       /* undefined state */
#define NOTILE ((uint32)(-1))         /* undefined state */

static int
TIFFFillStripPartial( TIFF *tif, int strip, tmsize_t read_ahead, int restart )
{
	static const char module[] = "TIFFFillStripPartial";
	register TIFFDirectory *td = &tif->tif_dir;
        uint64 unused_data;
        uint64 read_offset;
        tmsize_t cc, to_read;
        tmsize_t bytecountm;
        
        if (!_TIFFFillStriles( tif ) || !tif->tif_dir.td_stripbytecount)
            return 0;
        
        /*
         * Expand raw data buffer, if needed, to hold data
         * strip coming from file (perhaps should set upper
         * bound on the size of a buffer we'll use?).
         */

        bytecountm=(tmsize_t) td->td_stripbytecount[strip];
        if (read_ahead*2 > tif->tif_rawdatasize) {
                assert( restart );
                
                tif->tif_curstrip = NOSTRIP;
                if ((tif->tif_flags & TIFF_MYBUFFER) == 0) {
                        TIFFErrorExt(tif->tif_clientdata, module,
                                     "Data buffer too small to hold part of strip %lu",
                                     (unsigned long) strip);
                        return (0);
                }
                if (!TIFFReadBufferSetup(tif, 0, read_ahead*2))
                        return (0);
        }

        if( restart )
        {
                tif->tif_rawdataloaded = 0;
                tif->tif_rawdataoff = 0;
        }

        /*
        ** If we are reading more data, move any unused data to the
        ** start of the buffer.
        */
        if( tif->tif_rawdataloaded > 0 )
                unused_data = tif->tif_rawdataloaded - (tif->tif_rawcp - tif->tif_rawdata);
        else
                unused_data = 0;
        
        if( unused_data > 0 )
        {
                memmove( tif->tif_rawdata, tif->tif_rawcp, unused_data );
        }

        /*
        ** Seek to the point in the file where more data should be read.
        */
        read_offset = td->td_stripoffset[strip]
                + tif->tif_rawdataoff + tif->tif_rawdataloaded;

        if (!SeekOK(tif, read_offset)) {
                TIFFErrorExt(tif->tif_clientdata, module,
                             "Seek error at scanline %lu, strip %lu",
                             (unsigned long) tif->tif_row, (unsigned long) strip);
                return 0;
        }

        /*
        ** How much do we want to read?
        */
        to_read = tif->tif_rawdatasize - unused_data;
        if( (uint64) to_read > td->td_stripbytecount[strip] 
            - tif->tif_rawdataoff - tif->tif_rawdataloaded )
        {
                to_read = td->td_stripbytecount[strip]
                        - tif->tif_rawdataoff - tif->tif_rawdataloaded;
        }

        cc = TIFFReadFile(tif, tif->tif_rawdata + unused_data, to_read);

        if (cc != to_read) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
                TIFFErrorExt(tif->tif_clientdata, module,
                             "Read error at scanline %lu; got %I64u bytes, expected %I64u",
                             (unsigned long) tif->tif_row,
                             (unsigned __int64) cc,
                             (unsigned __int64) to_read);
#else
                TIFFErrorExt(tif->tif_clientdata, module,
                             "Read error at scanline %lu; got %llu bytes, expected %llu",
                             (unsigned long) tif->tif_row,
                             (unsigned long long) cc,
                             (unsigned long long) to_read);
#endif
                return 0;
        }
        
        tif->tif_rawdataoff = tif->tif_rawdataoff + tif->tif_rawdataloaded - unused_data ;
        tif->tif_rawdataloaded = unused_data + to_read;

        tif->tif_rawcp = tif->tif_rawdata;
                        
        if (!isFillOrder(tif, td->td_fillorder) &&
            (tif->tif_flags & TIFF_NOBITREV) == 0)
                TIFFReverseBits(tif->tif_rawdata + unused_data, to_read );

        /*
        ** When starting a strip from the beginning we need to
        ** restart the decoder.
        */
        if( restart )
                return TIFFStartStrip(tif, strip);
        else
                return 1;
}

/*
 * Seek to a random row+sample in a file.
 *
 * Only used by TIFFReadScanline, and is only used on
 * strip organized files.  We do some tricky stuff to try
 * and avoid reading the whole compressed raw data for big
 * strips.
 */
static int
TIFFSeek(TIFF* tif, uint32 row, uint16 sample )
{
	register TIFFDirectory *td = &tif->tif_dir;
	uint32 strip;
        int    whole_strip;
	tmsize_t read_ahead = 0;

        /*
        ** Establish what strip we are working from.
        */
	if (row >= td->td_imagelength) {	/* out of range */
		TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
		    "%lu: Row out of range, max %lu",
		    (unsigned long) row,
		    (unsigned long) td->td_imagelength);
		return (0);
	}
	if (td->td_planarconfig == PLANARCONFIG_SEPARATE) {
		if (sample >= td->td_samplesperpixel) {
			TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
			    "%lu: Sample out of range, max %lu",
			    (unsigned long) sample, (unsigned long) td->td_samplesperpixel);
			return (0);
		}
		strip = (uint32)sample*td->td_stripsperimage + row/td->td_rowsperstrip;
	} else
		strip = row / td->td_rowsperstrip;

        /*
         * Do we want to treat this strip as one whole chunk or
         * read it a few lines at a time?
         */
#if defined(CHUNKY_STRIP_READ_SUPPORT)
        if (!_TIFFFillStriles( tif ) || !tif->tif_dir.td_stripbytecount)
            return 0;
        whole_strip = tif->tif_dir.td_stripbytecount[strip] < 10
                || isMapped(tif);
#else
        whole_strip = 1;
#endif
        
        if( !whole_strip )
        {
                read_ahead = tif->tif_scanlinesize * 16 + 5000;
        }

        /*
         * If we haven't loaded this strip, do so now, possibly
         * only reading the first part.
         */
	if (strip != tif->tif_curstrip) {	/* different strip, refill */
                
                if( whole_strip )
                {
                        if (!TIFFFillStrip(tif, strip))
                                return (0);
                }
                else
                {
                        if( !TIFFFillStripPartial(tif,strip,read_ahead,1) )
                                return 0;
                }
	}

        /*
        ** If we already have some data loaded, do we need to read some more?
        */
        else if( !whole_strip )
        {
                if( ((tif->tif_rawdata + tif->tif_rawdataloaded) - tif->tif_rawcp) < read_ahead 
                    && (uint64) tif->tif_rawdataoff+tif->tif_rawdataloaded < td->td_stripbytecount[strip] )
                {
                        if( !TIFFFillStripPartial(tif,strip,read_ahead,0) )
                                return 0;
                }
        }

        if (row < tif->tif_row) {
		/*
		 * Moving backwards within the same strip: backup
		 * to the start and then decode forward (below).
		 *
		 * NB: If you're planning on lots of random access within a
		 * strip, it's better to just read and decode the entire
		 * strip, and then access the decoded data in a random fashion.
		 */

                if( tif->tif_rawdataoff != 0 )
                {
                        if( !TIFFFillStripPartial(tif,strip,read_ahead,1) )
                                return 0;
                }
                else
                {
                        if (!TIFFStartStrip(tif, strip))
                                return (0);
                }
	}
        
	if (row != tif->tif_row) {
		/*
		 * Seek forward to the desired row.
		 */

                /* TODO: Will this really work with partial buffers? */
                
		if (!(*tif->tif_seek)(tif, row - tif->tif_row))
			return (0);
		tif->tif_row = row;
	}

	return (1);
}

int
TIFFReadScanline(TIFF* tif, void* buf, uint32 row, uint16 sample)
{
	int e;

	if (!TIFFCheckRead(tif, 0))
		return (-1);
	if( (e = TIFFSeek(tif, row, sample)) != 0) {
		/*
		 * Decompress desired row into user buffer.
		 */
		e = (*tif->tif_decoderow)
		    (tif, (uint8*) buf, tif->tif_scanlinesize, sample);  

		/* we are now poised at the beginning of the next row */
		tif->tif_row = row + 1;

		if (e)
			(*tif->tif_postdecode)(tif, (uint8*) buf,
			    tif->tif_scanlinesize);  
	}
	return (e > 0 ? 1 : -1);
}

/*
 * Read a strip of data and decompress the specified
 * amount into the user-supplied buffer.
 */
tmsize_t
TIFFReadEncodedStrip(TIFF* tif, uint32 strip, void* buf, tmsize_t size)
{
	static const char module[] = "TIFFReadEncodedStrip";
	TIFFDirectory *td = &tif->tif_dir;
	uint32 rowsperstrip;
	uint32 stripsperplane;
	uint32 stripinplane;
	uint16 plane;
	uint32 rows;
	tmsize_t stripsize;
	if (!TIFFCheckRead(tif,0))
		return((tmsize_t)(-1));
	if (strip>=td->td_nstrips)
	{
		TIFFErrorExt(tif->tif_clientdata,module,
		    "%lu: Strip out of range, max %lu",(unsigned long)strip,
		    (unsigned long)td->td_nstrips);
		return((tmsize_t)(-1));
	}
	/*
	 * Calculate the strip size according to the number of
	 * rows in the strip (check for truncated last strip on any
	 * of the separations).
	 */
	rowsperstrip=td->td_rowsperstrip;
	if (rowsperstrip>td->td_imagelength)
		rowsperstrip=td->td_imagelength;
	stripsperplane=((td->td_imagelength+rowsperstrip-1)/rowsperstrip);
	stripinplane=(strip%stripsperplane);
	plane=(strip/stripsperplane);
	rows=td->td_imagelength-stripinplane*rowsperstrip;
	if (rows>rowsperstrip)
		rows=rowsperstrip;
	stripsize=TIFFVStripSize(tif,rows);
	if (stripsize==0)
		return((tmsize_t)(-1));
	if ((size!=(tmsize_t)(-1))&&(size<stripsize))
		stripsize=size;
	if (!TIFFFillStrip(tif,strip))
		return((tmsize_t)(-1));
	if ((*tif->tif_decodestrip)(tif,buf,stripsize,plane)<=0)
		return((tmsize_t)(-1));
	(*tif->tif_postdecode)(tif,buf,stripsize);
	return(stripsize);
}

static tmsize_t
TIFFReadRawStrip1(TIFF* tif, uint32 strip, void* buf, tmsize_t size,
    const char* module)
{
	TIFFDirectory *td = &tif->tif_dir;

    if (!_TIFFFillStriles( tif ))
        return ((tmsize_t)(-1));
        
	assert((tif->tif_flags&TIFF_NOREADRAW)==0);
	if (!isMapped(tif)) {
		tmsize_t cc;

		if (!SeekOK(tif, td->td_stripoffset[strip])) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "Seek error at scanline %lu, strip %lu",
			    (unsigned long) tif->tif_row, (unsigned long) strip);
			return ((tmsize_t)(-1));
		}
		cc = TIFFReadFile(tif, buf, size);
		if (cc != size) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
			TIFFErrorExt(tif->tif_clientdata, module,
		"Read error at scanline %lu; got %I64u bytes, expected %I64u",
				     (unsigned long) tif->tif_row,
				     (unsigned __int64) cc,
				     (unsigned __int64) size);
#else
			TIFFErrorExt(tif->tif_clientdata, module,
		"Read error at scanline %lu; got %llu bytes, expected %llu",
				     (unsigned long) tif->tif_row,
				     (unsigned long long) cc,
				     (unsigned long long) size);
#endif
			return ((tmsize_t)(-1));
		}
	} else {
		tmsize_t ma,mb;
		tmsize_t n;
		ma=(tmsize_t)td->td_stripoffset[strip];
		mb=ma+size;
		if (((uint64)ma!=td->td_stripoffset[strip])||(ma>tif->tif_size))
			n=0;
		else if ((mb<ma)||(mb<size)||(mb>tif->tif_size))
			n=tif->tif_size-ma;
		else
			n=size;
		if (n!=size) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
			TIFFErrorExt(tif->tif_clientdata, module,
	"Read error at scanline %lu, strip %lu; got %I64u bytes, expected %I64u",
				     (unsigned long) tif->tif_row,
				     (unsigned long) strip,
				     (unsigned __int64) n,
				     (unsigned __int64) size);
#else
			TIFFErrorExt(tif->tif_clientdata, module,
	"Read error at scanline %lu, strip %lu; got %llu bytes, expected %llu",
				     (unsigned long) tif->tif_row,
				     (unsigned long) strip,
				     (unsigned long long) n,
				     (unsigned long long) size);
#endif
			return ((tmsize_t)(-1));
		}
		_TIFFmemcpy(buf, tif->tif_base + ma,
			    size);
	}
	return (size);
}

/*
 * Read a strip of data from the file.
 */
tmsize_t
TIFFReadRawStrip(TIFF* tif, uint32 strip, void* buf, tmsize_t size)
{
	static const char module[] = "TIFFReadRawStrip";
	TIFFDirectory *td = &tif->tif_dir;
	uint64 bytecount;
	tmsize_t bytecountm;

	if (!TIFFCheckRead(tif, 0))
		return ((tmsize_t)(-1));
	if (strip >= td->td_nstrips) {
		TIFFErrorExt(tif->tif_clientdata, module,
		     "%lu: Strip out of range, max %lu",
		     (unsigned long) strip,
		     (unsigned long) td->td_nstrips);
		return ((tmsize_t)(-1));
	}
	if (tif->tif_flags&TIFF_NOREADRAW)
	{
		TIFFErrorExt(tif->tif_clientdata, module,
		    "Compression scheme does not support access to raw uncompressed data");
		return ((tmsize_t)(-1));
	}
	bytecount = td->td_stripbytecount[strip];
	if (bytecount <= 0) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
		TIFFErrorExt(tif->tif_clientdata, module,
			     "%I64u: Invalid strip byte count, strip %lu",
			     (unsigned __int64) bytecount,
			     (unsigned long) strip);
#else
		TIFFErrorExt(tif->tif_clientdata, module,
			     "%llu: Invalid strip byte count, strip %lu",
			     (unsigned long long) bytecount,
			     (unsigned long) strip);
#endif
		return ((tmsize_t)(-1));
	}
	bytecountm = (tmsize_t)bytecount;
	if ((uint64)bytecountm!=bytecount) {
		TIFFErrorExt(tif->tif_clientdata, module, "Integer overflow");
		return ((tmsize_t)(-1));
	}
	if (size != (tmsize_t)(-1) && size < bytecountm)
		bytecountm = size;
	return (TIFFReadRawStrip1(tif, strip, buf, bytecountm, module));
}

/*
 * Read the specified strip and setup for decoding. The data buffer is
 * expanded, as necessary, to hold the strip's data.
 */
int
TIFFFillStrip(TIFF* tif, uint32 strip)
{
	static const char module[] = "TIFFFillStrip";
	TIFFDirectory *td = &tif->tif_dir;

    if (!_TIFFFillStriles( tif ) || !tif->tif_dir.td_stripbytecount)
        return 0;
        
	if ((tif->tif_flags&TIFF_NOREADRAW)==0)
	{
		uint64 bytecount = td->td_stripbytecount[strip];
		if (bytecount <= 0) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
			TIFFErrorExt(tif->tif_clientdata, module,
				"Invalid strip byte count %I64u, strip %lu",
				     (unsigned __int64) bytecount,
				     (unsigned long) strip);
#else
			TIFFErrorExt(tif->tif_clientdata, module,
				"Invalid strip byte count %llu, strip %lu",
				     (unsigned long long) bytecount,
				     (unsigned long) strip);
#endif
			return (0);
		}
		if (isMapped(tif) &&
		    (isFillOrder(tif, td->td_fillorder)
		    || (tif->tif_flags & TIFF_NOBITREV))) {
			/*
			 * The image is mapped into memory and we either don't
			 * need to flip bits or the compression routine is
			 * going to handle this operation itself.  In this
			 * case, avoid copying the raw data and instead just
			 * reference the data from the memory mapped file
			 * image.  This assumes that the decompression
			 * routines do not modify the contents of the raw data
			 * buffer (if they try to, the application will get a
			 * fault since the file is mapped read-only).
			 */
			if ((tif->tif_flags & TIFF_MYBUFFER) && tif->tif_rawdata)
				_TIFFfree(tif->tif_rawdata);
			tif->tif_flags &= ~TIFF_MYBUFFER;
			/*
			 * We must check for overflow, potentially causing
			 * an OOB read. Instead of simple
			 *
			 *  td->td_stripoffset[strip]+bytecount > tif->tif_size
			 *
			 * comparison (which can overflow) we do the following
			 * two comparisons:
			 */
			if (bytecount > (uint64)tif->tif_size ||
			    td->td_stripoffset[strip] > (uint64)tif->tif_size - bytecount) {
				/*
				 * This error message might seem strange, but
				 * it's what would happen if a read were done
				 * instead.
				 */
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
				TIFFErrorExt(tif->tif_clientdata, module,

					"Read error on strip %lu; "
					"got %I64u bytes, expected %I64u",
					(unsigned long) strip,
					(unsigned __int64) tif->tif_size - td->td_stripoffset[strip],
					(unsigned __int64) bytecount);
#else
				TIFFErrorExt(tif->tif_clientdata, module,

					"Read error on strip %lu; "
					"got %llu bytes, expected %llu",
					(unsigned long) strip,
					(unsigned long long) tif->tif_size - td->td_stripoffset[strip],
					(unsigned long long) bytecount);
#endif
				tif->tif_curstrip = NOSTRIP;
				return (0);
			}
			tif->tif_rawdatasize = (tmsize_t)bytecount;
			tif->tif_rawdata = tif->tif_base + (tmsize_t)td->td_stripoffset[strip];
                        tif->tif_rawdataoff = 0;
                        tif->tif_rawdataloaded = (tmsize_t) bytecount;
		} else {
			/*
			 * Expand raw data buffer, if needed, to hold data
			 * strip coming from file (perhaps should set upper
			 * bound on the size of a buffer we'll use?).
			 */
			tmsize_t bytecountm;
			bytecountm=(tmsize_t)bytecount;
			if ((uint64)bytecountm!=bytecount)
			{
				TIFFErrorExt(tif->tif_clientdata,module,"Integer overflow");
				return(0);
			}
			if (bytecountm > tif->tif_rawdatasize) {
				tif->tif_curstrip = NOSTRIP;
				if ((tif->tif_flags & TIFF_MYBUFFER) == 0) {
					TIFFErrorExt(tif->tif_clientdata, module,
					    "Data buffer too small to hold strip %lu",
					    (unsigned long) strip);
					return (0);
				}
				if (!TIFFReadBufferSetup(tif, 0, bytecountm))
					return (0);
			}
			if (TIFFReadRawStrip1(tif, strip, tif->tif_rawdata,
				bytecountm, module) != bytecountm)
				return (0);

                        tif->tif_rawdataoff = 0;
                        tif->tif_rawdataloaded = bytecountm;
                        
			if (!isFillOrder(tif, td->td_fillorder) &&
			    (tif->tif_flags & TIFF_NOBITREV) == 0)
				TIFFReverseBits(tif->tif_rawdata, bytecountm);
                }
	}
	return (TIFFStartStrip(tif, strip));
}

/*
 * Tile-oriented Read Support
 * Contributed by Nancy Cam (Silicon Graphics).
 */

/*
 * Read and decompress a tile of data.  The
 * tile is selected by the (x,y,z,s) coordinates.
 */
tmsize_t
TIFFReadTile(TIFF* tif, void* buf, uint32 x, uint32 y, uint32 z, uint16 s)
{
	if (!TIFFCheckRead(tif, 1) || !TIFFCheckTile(tif, x, y, z, s))
		return ((tmsize_t)(-1));
	return (TIFFReadEncodedTile(tif,
	    TIFFComputeTile(tif, x, y, z, s), buf, (tmsize_t)(-1)));
}

/*
 * Read a tile of data and decompress the specified
 * amount into the user-supplied buffer.
 */
tmsize_t
TIFFReadEncodedTile(TIFF* tif, uint32 tile, void* buf, tmsize_t size)
{
	static const char module[] = "TIFFReadEncodedTile";
	TIFFDirectory *td = &tif->tif_dir;
	tmsize_t tilesize = tif->tif_tilesize;

	if (!TIFFCheckRead(tif, 1))
		return ((tmsize_t)(-1));
	if (tile >= td->td_nstrips) {
		TIFFErrorExt(tif->tif_clientdata, module,
		    "%lu: Tile out of range, max %lu",
		    (unsigned long) tile, (unsigned long) td->td_nstrips);
		return ((tmsize_t)(-1));
	}
	if (size == (tmsize_t)(-1))
		size = tilesize;
	else if (size > tilesize)
		size = tilesize;
	if (TIFFFillTile(tif, tile) && (*tif->tif_decodetile)(tif,
	    (uint8*) buf, size, (uint16)(tile/td->td_stripsperimage))) {
		(*tif->tif_postdecode)(tif, (uint8*) buf, size);
		return (size);
	} else
		return ((tmsize_t)(-1));
}

static tmsize_t
TIFFReadRawTile1(TIFF* tif, uint32 tile, void* buf, tmsize_t size, const char* module)
{
	TIFFDirectory *td = &tif->tif_dir;

    if (!_TIFFFillStriles( tif ))
        return ((tmsize_t)(-1));

	assert((tif->tif_flags&TIFF_NOREADRAW)==0);
	if (!isMapped(tif)) {
		tmsize_t cc;

		if (!SeekOK(tif, td->td_stripoffset[tile])) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "Seek error at row %lu, col %lu, tile %lu",
			    (unsigned long) tif->tif_row,
			    (unsigned long) tif->tif_col,
			    (unsigned long) tile);
			return ((tmsize_t)(-1));
		}
		cc = TIFFReadFile(tif, buf, size);
		if (cc != size) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
			TIFFErrorExt(tif->tif_clientdata, module,
	"Read error at row %lu, col %lu; got %I64u bytes, expected %I64u",
				     (unsigned long) tif->tif_row,
				     (unsigned long) tif->tif_col,
				     (unsigned __int64) cc,
				     (unsigned __int64) size);
#else
			TIFFErrorExt(tif->tif_clientdata, module,
	"Read error at row %lu, col %lu; got %llu bytes, expected %llu",
				     (unsigned long) tif->tif_row,
				     (unsigned long) tif->tif_col,
				     (unsigned long long) cc,
				     (unsigned long long) size);
#endif
			return ((tmsize_t)(-1));
		}
	} else {
		tmsize_t ma,mb;
		tmsize_t n;
		ma=(tmsize_t)td->td_stripoffset[tile];
		mb=ma+size;
		if (((uint64)ma!=td->td_stripoffset[tile])||(ma>tif->tif_size))
			n=0;
		else if ((mb<ma)||(mb<size)||(mb>tif->tif_size))
			n=tif->tif_size-ma;
		else
			n=size;
		if (n!=size) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
			TIFFErrorExt(tif->tif_clientdata, module,
"Read error at row %lu, col %lu, tile %lu; got %I64u bytes, expected %I64u",
				     (unsigned long) tif->tif_row,
				     (unsigned long) tif->tif_col,
				     (unsigned long) tile,
				     (unsigned __int64) n,
				     (unsigned __int64) size);
#else
			TIFFErrorExt(tif->tif_clientdata, module,
"Read error at row %lu, col %lu, tile %lu; got %llu bytes, expected %llu",
				     (unsigned long) tif->tif_row,
				     (unsigned long) tif->tif_col,
				     (unsigned long) tile,
				     (unsigned long long) n,
				     (unsigned long long) size);
#endif
			return ((tmsize_t)(-1));
		}
		_TIFFmemcpy(buf, tif->tif_base + ma, size);
	}
	return (size);
}

/*
 * Read a tile of data from the file.
 */
tmsize_t
TIFFReadRawTile(TIFF* tif, uint32 tile, void* buf, tmsize_t size)
{
	static const char module[] = "TIFFReadRawTile";
	TIFFDirectory *td = &tif->tif_dir;
	uint64 bytecount64;
	tmsize_t bytecountm;

	if (!TIFFCheckRead(tif, 1))
		return ((tmsize_t)(-1));
	if (tile >= td->td_nstrips) {
		TIFFErrorExt(tif->tif_clientdata, module,
		    "%lu: Tile out of range, max %lu",
		    (unsigned long) tile, (unsigned long) td->td_nstrips);
		return ((tmsize_t)(-1));
	}
	if (tif->tif_flags&TIFF_NOREADRAW)
	{
		TIFFErrorExt(tif->tif_clientdata, module,
		"Compression scheme does not support access to raw uncompressed data");
		return ((tmsize_t)(-1));
	}
	bytecount64 = td->td_stripbytecount[tile];
	if (size != (tmsize_t)(-1) && (uint64)size < bytecount64)
		bytecount64 = (uint64)size;
	bytecountm = (tmsize_t)bytecount64;
	if ((uint64)bytecountm!=bytecount64)
	{
		TIFFErrorExt(tif->tif_clientdata,module,"Integer overflow");
		return ((tmsize_t)(-1));
	}
	return (TIFFReadRawTile1(tif, tile, buf, bytecountm, module));
}

/*
 * Read the specified tile and setup for decoding. The data buffer is
 * expanded, as necessary, to hold the tile's data.
 */
int
TIFFFillTile(TIFF* tif, uint32 tile)
{
	static const char module[] = "TIFFFillTile";
	TIFFDirectory *td = &tif->tif_dir;

    if (!_TIFFFillStriles( tif ) || !tif->tif_dir.td_stripbytecount)
        return 0;
        
	if ((tif->tif_flags&TIFF_NOREADRAW)==0)
	{
		uint64 bytecount = td->td_stripbytecount[tile];
		if (bytecount <= 0) {
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
			TIFFErrorExt(tif->tif_clientdata, module,
				"%I64u: Invalid tile byte count, tile %lu",
				     (unsigned __int64) bytecount,
				     (unsigned long) tile);
#else
			TIFFErrorExt(tif->tif_clientdata, module,
				"%llu: Invalid tile byte count, tile %lu",
				     (unsigned long long) bytecount,
				     (unsigned long) tile);
#endif
			return (0);
		}
		if (isMapped(tif) &&
		    (isFillOrder(tif, td->td_fillorder)
		     || (tif->tif_flags & TIFF_NOBITREV))) {
			/*
			 * The image is mapped into memory and we either don't
			 * need to flip bits or the compression routine is
			 * going to handle this operation itself.  In this
			 * case, avoid copying the raw data and instead just
			 * reference the data from the memory mapped file
			 * image.  This assumes that the decompression
			 * routines do not modify the contents of the raw data
			 * buffer (if they try to, the application will get a
			 * fault since the file is mapped read-only).
			 */
			if ((tif->tif_flags & TIFF_MYBUFFER) && tif->tif_rawdata)
				_TIFFfree(tif->tif_rawdata);
			tif->tif_flags &= ~TIFF_MYBUFFER;
			/*
			 * We must check for overflow, potentially causing
			 * an OOB read. Instead of simple
			 *
			 *  td->td_stripoffset[tile]+bytecount > tif->tif_size
			 *
			 * comparison (which can overflow) we do the following
			 * two comparisons:
			 */
			if (bytecount > (uint64)tif->tif_size ||
			    td->td_stripoffset[tile] > (uint64)tif->tif_size - bytecount) {
				tif->tif_curtile = NOTILE;
				return (0);
			}
			tif->tif_rawdatasize = (tmsize_t)bytecount;
			tif->tif_rawdata =
				tif->tif_base + (tmsize_t)td->td_stripoffset[tile];
                        tif->tif_rawdataoff = 0;
                        tif->tif_rawdataloaded = (tmsize_t) bytecount;
		} else {
			/*
			 * Expand raw data buffer, if needed, to hold data
			 * tile coming from file (perhaps should set upper
			 * bound on the size of a buffer we'll use?).
			 */
			tmsize_t bytecountm;
			bytecountm=(tmsize_t)bytecount;
			if ((uint64)bytecountm!=bytecount)
			{
				TIFFErrorExt(tif->tif_clientdata,module,"Integer overflow");
				return(0);
			}
			if (bytecountm > tif->tif_rawdatasize) {
				tif->tif_curtile = NOTILE;
				if ((tif->tif_flags & TIFF_MYBUFFER) == 0) {
					TIFFErrorExt(tif->tif_clientdata, module,
					    "Data buffer too small to hold tile %lu",
					    (unsigned long) tile);
					return (0);
				}
				if (!TIFFReadBufferSetup(tif, 0, bytecountm))
					return (0);
			}
			if (TIFFReadRawTile1(tif, tile, tif->tif_rawdata,
			    bytecountm, module) != bytecountm)
				return (0);

                        tif->tif_rawdataoff = 0;
                        tif->tif_rawdataloaded = bytecountm;
                        
			if (!isFillOrder(tif, td->td_fillorder) &&
			    (tif->tif_flags & TIFF_NOBITREV) == 0)
				TIFFReverseBits(tif->tif_rawdata,
                                                tif->tif_rawdataloaded);
		}
	}
	return (TIFFStartTile(tif, tile));
}

/*
 * Setup the raw data buffer in preparation for
 * reading a strip of raw data.  If the buffer
 * is specified as zero, then a buffer of appropriate
 * size is allocated by the library.  Otherwise,
 * the client must guarantee that the buffer is
 * large enough to hold any individual strip of
 * raw data.
 */
int
TIFFReadBufferSetup(TIFF* tif, void* bp, tmsize_t size)
{
	static const char module[] = "TIFFReadBufferSetup";

	assert((tif->tif_flags&TIFF_NOREADRAW)==0);
	if (tif->tif_rawdata) {
		if (tif->tif_flags & TIFF_MYBUFFER)
			_TIFFfree(tif->tif_rawdata);
		tif->tif_rawdata = NULL;
	}
	if (bp) {
		tif->tif_rawdatasize = size;
		tif->tif_rawdata = (uint8*) bp;
		tif->tif_flags &= ~TIFF_MYBUFFER;
	} else {
		tif->tif_rawdatasize = (tmsize_t)TIFFroundup_64((uint64)size, 1024);
		if (tif->tif_rawdatasize==0)
			tif->tif_rawdatasize=(tmsize_t)(-1);
		tif->tif_rawdata = (uint8*) _TIFFmalloc(tif->tif_rawdatasize);
		tif->tif_flags |= TIFF_MYBUFFER;
	}
	if (tif->tif_rawdata == NULL) {
		TIFFErrorExt(tif->tif_clientdata, module,
		    "No space for data buffer at scanline %lu",
		    (unsigned long) tif->tif_row);
		tif->tif_rawdatasize = 0;
		return (0);
	}
	return (1);
}

/*
 * Set state to appear as if a
 * strip has just been read in.
 */
static int
TIFFStartStrip(TIFF* tif, uint32 strip)
{
	TIFFDirectory *td = &tif->tif_dir;

    if (!_TIFFFillStriles( tif ) || !tif->tif_dir.td_stripbytecount)
        return 0;

	if ((tif->tif_flags & TIFF_CODERSETUP) == 0) {
		if (!(*tif->tif_setupdecode)(tif))
			return (0);
		tif->tif_flags |= TIFF_CODERSETUP;
	}
	tif->tif_curstrip = strip;
	tif->tif_row = (strip % td->td_stripsperimage) * td->td_rowsperstrip;
        tif->tif_flags &= ~TIFF_BUF4WRITE;

	if (tif->tif_flags&TIFF_NOREADRAW)
	{
		tif->tif_rawcp = NULL;
		tif->tif_rawcc = 0;  
	}
	else
	{
		tif->tif_rawcp = tif->tif_rawdata;
		tif->tif_rawcc = (tmsize_t)td->td_stripbytecount[strip];
	}
	return ((*tif->tif_predecode)(tif,
			(uint16)(strip / td->td_stripsperimage)));
}

/*
 * Set state to appear as if a
 * tile has just been read in.
 */
static int
TIFFStartTile(TIFF* tif, uint32 tile)
{
	TIFFDirectory *td = &tif->tif_dir;

    if (!_TIFFFillStriles( tif ) || !tif->tif_dir.td_stripbytecount)
        return 0;

	if ((tif->tif_flags & TIFF_CODERSETUP) == 0) {
		if (!(*tif->tif_setupdecode)(tif))
			return (0);
		tif->tif_flags |= TIFF_CODERSETUP;
	}
	tif->tif_curtile = tile;
	tif->tif_row =
	    (tile % TIFFhowmany_32(td->td_imagewidth, td->td_tilewidth)) *
		td->td_tilelength;
	tif->tif_col =
	    (tile % TIFFhowmany_32(td->td_imagelength, td->td_tilelength)) *
		td->td_tilewidth;
        tif->tif_flags &= ~TIFF_BUF4WRITE;
	if (tif->tif_flags&TIFF_NOREADRAW)
	{
		tif->tif_rawcp = NULL;
		tif->tif_rawcc = 0;
	}
	else
	{
		tif->tif_rawcp = tif->tif_rawdata;
		tif->tif_rawcc = (tmsize_t)td->td_stripbytecount[tile];
	}
	return ((*tif->tif_predecode)(tif,
			(uint16)(tile/td->td_stripsperimage)));
}

static int
TIFFCheckRead(TIFF* tif, int tiles)
{
	if (tif->tif_mode == O_WRONLY) {
		TIFFErrorExt(tif->tif_clientdata, tif->tif_name, "File not open for reading");
		return (0);
	}
	if (tiles ^ isTiled(tif)) {
		TIFFErrorExt(tif->tif_clientdata, tif->tif_name, tiles ?
		    "Can not read tiles from a stripped image" :
		    "Can not read scanlines from a tiled image");
		return (0);
	}
	return (1);
}

void
_TIFFNoPostDecode(TIFF* tif, uint8* buf, tmsize_t cc)
{
    (void) tif; (void) buf; (void) cc;
}

void
_TIFFSwab16BitData(TIFF* tif, uint8* buf, tmsize_t cc)
{
    (void) tif;
    assert((cc & 1) == 0);
    TIFFSwabArrayOfShort((uint16*) buf, cc/2);
}

void
_TIFFSwab24BitData(TIFF* tif, uint8* buf, tmsize_t cc)
{
    (void) tif;
    assert((cc % 3) == 0);
    TIFFSwabArrayOfTriples((uint8*) buf, cc/3);
}

void
_TIFFSwab32BitData(TIFF* tif, uint8* buf, tmsize_t cc)
{
    (void) tif;
    assert((cc & 3) == 0);
    TIFFSwabArrayOfLong((uint32*) buf, cc/4);
}

void
_TIFFSwab64BitData(TIFF* tif, uint8* buf, tmsize_t cc)
{
    (void) tif;
    assert((cc & 7) == 0);
    TIFFSwabArrayOfDouble((double*) buf, cc/8);
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
