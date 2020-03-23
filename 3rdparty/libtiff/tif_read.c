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

#define TIFF_SIZE_T_MAX ((size_t) ~ ((size_t)0))
#define TIFF_TMSIZE_T_MAX (tmsize_t)(TIFF_SIZE_T_MAX >> 1)

int TIFFFillStrip(TIFF* tif, uint32 strip);
int TIFFFillTile(TIFF* tif, uint32 tile);
static int TIFFStartStrip(TIFF* tif, uint32 strip);
static int TIFFStartTile(TIFF* tif, uint32 tile);
static int TIFFCheckRead(TIFF*, int);
static tmsize_t
TIFFReadRawStrip1(TIFF* tif, uint32 strip, void* buf, tmsize_t size,const char* module);
static tmsize_t
TIFFReadRawTile1(TIFF* tif, uint32 tile, void* buf, tmsize_t size, const char* module);

#define NOSTRIP ((uint32)(-1))       /* undefined state */
#define NOTILE ((uint32)(-1))         /* undefined state */

#define INITIAL_THRESHOLD (1024 * 1024)
#define THRESHOLD_MULTIPLIER 10
#define MAX_THRESHOLD (THRESHOLD_MULTIPLIER * THRESHOLD_MULTIPLIER * THRESHOLD_MULTIPLIER * INITIAL_THRESHOLD)

/* Read 'size' bytes in tif_rawdata buffer starting at offset 'rawdata_offset'
 * Returns 1 in case of success, 0 otherwise. */
static int TIFFReadAndRealloc( TIFF* tif, tmsize_t size,
                               tmsize_t rawdata_offset,
                               int is_strip, uint32 strip_or_tile,
                               const char* module )
{
#if SIZEOF_SIZE_T == 8
        tmsize_t threshold = INITIAL_THRESHOLD;
#endif
        tmsize_t already_read = 0;

        /* On 64 bit processes, read first a maximum of 1 MB, then 10 MB, etc */
        /* so as to avoid allocating too much memory in case the file is too */
        /* short. We could ask for the file size, but this might be */
        /* expensive with some I/O layers (think of reading a gzipped file) */
        /* Restrict to 64 bit processes, so as to avoid reallocs() */
        /* on 32 bit processes where virtual memory is scarce.  */
        while( already_read < size )
        {
            tmsize_t bytes_read;
            tmsize_t to_read = size - already_read;
#if SIZEOF_SIZE_T == 8
            if( to_read >= threshold && threshold < MAX_THRESHOLD &&
                already_read + to_read + rawdata_offset > tif->tif_rawdatasize )
            {
                to_read = threshold;
                threshold *= THRESHOLD_MULTIPLIER;
            }
#endif
            if (already_read + to_read + rawdata_offset > tif->tif_rawdatasize) {
                uint8* new_rawdata;
                assert((tif->tif_flags & TIFF_MYBUFFER) != 0);
                tif->tif_rawdatasize = (tmsize_t)TIFFroundup_64(
                        (uint64)already_read + to_read + rawdata_offset, 1024);
                if (tif->tif_rawdatasize==0) {
                    TIFFErrorExt(tif->tif_clientdata, module,
                                "Invalid buffer size");
                    return 0;
                }
                new_rawdata = (uint8*) _TIFFrealloc(
                                tif->tif_rawdata, tif->tif_rawdatasize);
                if( new_rawdata == 0 )
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                        "No space for data buffer at scanline %lu",
                        (unsigned long) tif->tif_row);
                    _TIFFfree(tif->tif_rawdata);
                    tif->tif_rawdata = 0;
                    tif->tif_rawdatasize = 0;
                    return 0;
                }
                tif->tif_rawdata = new_rawdata;
            }
            if( tif->tif_rawdata == NULL )
            {
                /* should not happen in practice but helps CoverityScan */
                return 0;
            }

            bytes_read = TIFFReadFile(tif,
                tif->tif_rawdata + rawdata_offset + already_read, to_read);
            already_read += bytes_read;
            if (bytes_read != to_read) {
                memset( tif->tif_rawdata + rawdata_offset + already_read, 0,
                        tif->tif_rawdatasize - rawdata_offset - already_read );
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
                if( is_strip )
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                        "Read error at scanline %lu; got %I64u bytes, "
                        "expected %I64u",
                                        (unsigned long) tif->tif_row,
                                        (unsigned __int64) already_read,
                                        (unsigned __int64) size);
                }
                else
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                        "Read error at row %lu, col %lu, tile %lu; "
                        "got %I64u bytes, expected %I64u",
                                        (unsigned long) tif->tif_row,
                                        (unsigned long) tif->tif_col,
                                        (unsigned long) strip_or_tile,
                                        (unsigned __int64) already_read,
                                        (unsigned __int64) size);
                }
#else
                if( is_strip )
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                        "Read error at scanline %lu; got %llu bytes, "
                        "expected %llu",
                                        (unsigned long) tif->tif_row,
                                        (unsigned long long) already_read,
                                        (unsigned long long) size);
                }
                else
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                        "Read error at row %lu, col %lu, tile %lu; "
                        "got %llu bytes, expected %llu",
                                        (unsigned long) tif->tif_row,
                                        (unsigned long) tif->tif_col,
                                        (unsigned long) strip_or_tile,
                                        (unsigned long long) already_read,
                                        (unsigned long long) size);
                }
#endif
                return 0;
            }
        }
        return 1;
}


static int
TIFFFillStripPartial( TIFF *tif, int strip, tmsize_t read_ahead, int restart )
{
	static const char module[] = "TIFFFillStripPartial";
	register TIFFDirectory *td = &tif->tif_dir;
        tmsize_t unused_data;
        uint64 read_offset;
        tmsize_t to_read;
        tmsize_t read_ahead_mod;
        /* tmsize_t bytecountm; */
        
        if (!_TIFFFillStriles( tif ) || !tif->tif_dir.td_stripbytecount)
            return 0;
        
        /*
         * Expand raw data buffer, if needed, to hold data
         * strip coming from file (perhaps should set upper
         * bound on the size of a buffer we'll use?).
         */

        /* bytecountm=(tmsize_t) td->td_stripbytecount[strip]; */

        /* Not completely sure where the * 2 comes from, but probably for */
        /* an exponentional growth strategy of tif_rawdatasize */
        if( read_ahead < TIFF_TMSIZE_T_MAX / 2 )
                read_ahead_mod = read_ahead * 2;
        else
                read_ahead_mod = read_ahead;
        if (read_ahead_mod > tif->tif_rawdatasize) {
                assert( restart );
                
                tif->tif_curstrip = NOSTRIP;
                if ((tif->tif_flags & TIFF_MYBUFFER) == 0) {
                        TIFFErrorExt(tif->tif_clientdata, module,
                                     "Data buffer too small to hold part of strip %lu",
                                     (unsigned long) strip);
                        return (0);
                }
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
		assert((tif->tif_flags&TIFF_BUFFERMMAP)==0);
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
        if( read_ahead_mod > tif->tif_rawdatasize )
                to_read = read_ahead_mod - unused_data;
        else
                to_read = tif->tif_rawdatasize - unused_data;
        if( (uint64) to_read > td->td_stripbytecount[strip] 
            - tif->tif_rawdataoff - tif->tif_rawdataloaded )
        {
                to_read = (tmsize_t) td->td_stripbytecount[strip]
                        - tif->tif_rawdataoff - tif->tif_rawdataloaded;
        }

	assert((tif->tif_flags&TIFF_BUFFERMMAP)==0);
        if( !TIFFReadAndRealloc( tif, to_read, unused_data,
                                 1, /* is_strip */
                                 0, /* strip_or_tile */
                                 module) )
        {
                return 0;
        }

        tif->tif_rawdataoff = tif->tif_rawdataoff + tif->tif_rawdataloaded - unused_data ;
        tif->tif_rawdataloaded = unused_data + to_read;

        tif->tif_rawcc = tif->tif_rawdataloaded;
        tif->tif_rawcp = tif->tif_rawdata;
                        
        if (!isFillOrder(tif, td->td_fillorder) &&
            (tif->tif_flags & TIFF_NOBITREV) == 0) {
		assert((tif->tif_flags&TIFF_BUFFERMMAP)==0);
                TIFFReverseBits(tif->tif_rawdata + unused_data, to_read );
	}

        /*
        ** When starting a strip from the beginning we need to
        ** restart the decoder.
        */
        if( restart )
        {

#ifdef JPEG_SUPPORT
            /* A bit messy since breaks the codec abstraction. Ultimately */
            /* there should be a function pointer for that, but it seems */
            /* only JPEG is affected. */
            /* For JPEG, if there are multiple scans (can generally be known */
            /* with the  read_ahead used), we need to read the whole strip */
            if( tif->tif_dir.td_compression==COMPRESSION_JPEG &&
                (uint64)tif->tif_rawcc < td->td_stripbytecount[strip] )
            {
                if( TIFFJPEGIsFullStripRequired(tif) )
                {
                    return TIFFFillStrip(tif, strip);
                }
            }
#endif

            return TIFFStartStrip(tif, strip);
        }
        else
        {
                return 1;
        }
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
        if( td->td_compression == COMPRESSION_LERC ||
            td->td_compression == COMPRESSION_JBIG )
        {
            /* Ideally plugins should have a way to declare they don't support
             * chunk strip */
            whole_strip = 1;
        }
#else
        whole_strip = 1;
#endif
        
        if( !whole_strip )
        {
                /* 16 is for YCbCr mode where we may need to read 16 */
                /* lines at a time to get a decompressed line, and 5000 */
                /* is some constant value, for example for JPEG tables */
                if( tif->tif_scanlinesize < TIFF_TMSIZE_T_MAX / 16 &&
                    tif->tif_scanlinesize * 16 < TIFF_TMSIZE_T_MAX - 5000 )
                {
                        read_ahead = tif->tif_scanlinesize * 16 + 5000;
                }
                else
                {
                        read_ahead = tif->tif_scanlinesize;
                }
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
 * Calculate the strip size according to the number of
 * rows in the strip (check for truncated last strip on any
 * of the separations).
 */
static tmsize_t TIFFReadEncodedStripGetStripSize(TIFF* tif, uint32 strip, uint16* pplane)
{
	static const char module[] = "TIFFReadEncodedStrip";
	TIFFDirectory *td = &tif->tif_dir;
	uint32 rowsperstrip;
	uint32 stripsperplane;
	uint32 stripinplane;
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

	rowsperstrip=td->td_rowsperstrip;
	if (rowsperstrip>td->td_imagelength)
		rowsperstrip=td->td_imagelength;
	stripsperplane= TIFFhowmany_32_maxuint_compat(td->td_imagelength, rowsperstrip);
	stripinplane=(strip%stripsperplane);
	if( pplane ) *pplane=(uint16)(strip/stripsperplane);
	rows=td->td_imagelength-stripinplane*rowsperstrip;
	if (rows>rowsperstrip)
		rows=rowsperstrip;
	stripsize=TIFFVStripSize(tif,rows);
	if (stripsize==0)
		return((tmsize_t)(-1));
	return stripsize;
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
	tmsize_t stripsize;
	uint16 plane;

	stripsize=TIFFReadEncodedStripGetStripSize(tif, strip, &plane);
	if (stripsize==((tmsize_t)(-1)))
		return((tmsize_t)(-1));

    /* shortcut to avoid an extra memcpy() */
    if( td->td_compression == COMPRESSION_NONE &&
        size!=(tmsize_t)(-1) && size >= stripsize &&
        !isMapped(tif) &&
        ((tif->tif_flags&TIFF_NOREADRAW)==0) )
    {
        if (TIFFReadRawStrip1(tif, strip, buf, stripsize, module) != stripsize)
            return ((tmsize_t)(-1));

        if (!isFillOrder(tif, td->td_fillorder) &&
            (tif->tif_flags & TIFF_NOBITREV) == 0)
            TIFFReverseBits(buf,stripsize);

        (*tif->tif_postdecode)(tif,buf,stripsize);
        return (stripsize);
    }

	if ((size!=(tmsize_t)(-1))&&(size<stripsize))
		stripsize=size;
	if (!TIFFFillStrip(tif,strip))
		return((tmsize_t)(-1));
	if ((*tif->tif_decodestrip)(tif,buf,stripsize,plane)<=0)
		return((tmsize_t)(-1));
	(*tif->tif_postdecode)(tif,buf,stripsize);
	return(stripsize);
}

/* Variant of TIFFReadEncodedStrip() that does 
 * * if *buf == NULL, *buf = _TIFFmalloc(bufsizetoalloc) only after TIFFFillStrip() has
 *   succeeded. This avoid excessive memory allocation in case of truncated
 *   file.
 * * calls regular TIFFReadEncodedStrip() if *buf != NULL
 */
tmsize_t
_TIFFReadEncodedStripAndAllocBuffer(TIFF* tif, uint32 strip,
                                    void **buf, tmsize_t bufsizetoalloc,
                                    tmsize_t size_to_read)
{
    tmsize_t this_stripsize;
    uint16 plane;

    if( *buf != NULL )
    {
        return TIFFReadEncodedStrip(tif, strip, *buf, size_to_read);
    }

    this_stripsize=TIFFReadEncodedStripGetStripSize(tif, strip, &plane);
    if (this_stripsize==((tmsize_t)(-1)))
            return((tmsize_t)(-1));

    if ((size_to_read!=(tmsize_t)(-1))&&(size_to_read<this_stripsize))
            this_stripsize=size_to_read;
    if (!TIFFFillStrip(tif,strip))
            return((tmsize_t)(-1));

    *buf = _TIFFmalloc(bufsizetoalloc);
    if (*buf == NULL) {
            TIFFErrorExt(tif->tif_clientdata, TIFFFileName(tif), "No space for strip buffer");
            return((tmsize_t)(-1));
    }
    _TIFFmemset(*buf, 0, bufsizetoalloc);

    if ((*tif->tif_decodestrip)(tif,*buf,this_stripsize,plane)<=0)
            return((tmsize_t)(-1));
    (*tif->tif_postdecode)(tif,*buf,this_stripsize);
    return(this_stripsize);


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
		tmsize_t ma = 0;
		tmsize_t n;
		if ((td->td_stripoffset[strip] > (uint64)TIFF_TMSIZE_T_MAX)||
                    ((ma=(tmsize_t)td->td_stripoffset[strip])>tif->tif_size))
                {
                    n=0;
                }
                else if( ma > TIFF_TMSIZE_T_MAX - size )
                {
                    n=0;
                }
                else
                {
                    tmsize_t mb=ma+size;
                    if (mb>tif->tif_size)
                            n=tif->tif_size-ma;
                    else
                            n=size;
                }
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

static tmsize_t
TIFFReadRawStripOrTile2(TIFF* tif, uint32 strip_or_tile, int is_strip,
                        tmsize_t size, const char* module)
{
        TIFFDirectory *td = &tif->tif_dir;

        assert( !isMapped(tif) );
        assert((tif->tif_flags&TIFF_NOREADRAW)==0);

        if (!SeekOK(tif, td->td_stripoffset[strip_or_tile])) {
            if( is_strip )
            {
                TIFFErrorExt(tif->tif_clientdata, module,
                    "Seek error at scanline %lu, strip %lu",
                    (unsigned long) tif->tif_row,
                    (unsigned long) strip_or_tile);
            }
            else
            {
                TIFFErrorExt(tif->tif_clientdata, module,
                    "Seek error at row %lu, col %lu, tile %lu",
                    (unsigned long) tif->tif_row,
                    (unsigned long) tif->tif_col,
                    (unsigned long) strip_or_tile);
            }
            return ((tmsize_t)(-1));
        }

        if( !TIFFReadAndRealloc( tif, size, 0, is_strip,
                                 strip_or_tile, module ) )
        {
            return ((tmsize_t)(-1));
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
	if ((int64)bytecount <= 0) {
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
		if ((int64)bytecount <= 0) {
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

		/* To avoid excessive memory allocations: */
		/* Byte count should normally not be larger than a number of */
		/* times the uncompressed size plus some margin */
                if( bytecount > 1024 * 1024 )
                {
			/* 10 and 4096 are just values that could be adjusted. */
			/* Hopefully they are safe enough for all codecs */
			tmsize_t stripsize = TIFFStripSize(tif);
			if( stripsize != 0 &&
			    (bytecount - 4096) / 10 > (uint64)stripsize  )
			{
				uint64 newbytecount = (uint64)stripsize * 10 + 4096;
				if( (int64)newbytecount >= 0 )
				{
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
					TIFFWarningExt(tif->tif_clientdata, module,
					  "Too large strip byte count %I64u, strip %lu. Limiting to %I64u",
					     (unsigned __int64) bytecount,
					     (unsigned long) strip,
					     (unsigned __int64) newbytecount);
#else
					TIFFErrorExt(tif->tif_clientdata, module,
					  "Too large strip byte count %llu, strip %lu. Limiting to %llu",
					     (unsigned long long) bytecount,
					     (unsigned long) strip,
					     (unsigned long long) newbytecount);
#endif
					bytecount = newbytecount;
				}
			}
		}

		if (isMapped(tif)) {
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
			if ((tif->tif_flags & TIFF_MYBUFFER) && tif->tif_rawdata) {
				_TIFFfree(tif->tif_rawdata);
				tif->tif_rawdata = NULL;
				tif->tif_rawdatasize = 0;
			}
			tif->tif_flags &= ~TIFF_MYBUFFER;
			tif->tif_rawdatasize = (tmsize_t)bytecount;
			tif->tif_rawdata = tif->tif_base + (tmsize_t)td->td_stripoffset[strip];
                        tif->tif_rawdataoff = 0;
                        tif->tif_rawdataloaded = (tmsize_t) bytecount;

			/* 
			 * When we have tif_rawdata reference directly into the memory mapped file
			 * we need to be pretty careful about how we use the rawdata.  It is not
			 * a general purpose working buffer as it normally otherwise is.  So we
			 * keep track of this fact to avoid using it improperly.
			 */
			tif->tif_flags |= TIFF_BUFFERMMAP;
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
			}
			if (tif->tif_flags&TIFF_BUFFERMMAP) {
				tif->tif_curstrip = NOSTRIP;
				tif->tif_rawdata = NULL;
				tif->tif_rawdatasize = 0;
				tif->tif_flags &= ~TIFF_BUFFERMMAP;
			}

			if( isMapped(tif) )
			{
				if (bytecountm > tif->tif_rawdatasize &&
				    !TIFFReadBufferSetup(tif, 0, bytecountm))
				{
					return (0);
				}
				if (TIFFReadRawStrip1(tif, strip, tif->tif_rawdata,
				    bytecountm, module) != bytecountm)
				{
					return (0);
				}
			}
			else
			{
				if (TIFFReadRawStripOrTile2(tif, strip, 1,
				    bytecountm, module) != bytecountm)
				{
					return (0);
				}
			}


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

    /* shortcut to avoid an extra memcpy() */
    if( td->td_compression == COMPRESSION_NONE &&
        size!=(tmsize_t)(-1) && size >= tilesize &&
        !isMapped(tif) &&
        ((tif->tif_flags&TIFF_NOREADRAW)==0) )
    {
        if (TIFFReadRawTile1(tif, tile, buf, tilesize, module) != tilesize)
            return ((tmsize_t)(-1));

        if (!isFillOrder(tif, td->td_fillorder) &&
            (tif->tif_flags & TIFF_NOBITREV) == 0)
            TIFFReverseBits(buf,tilesize);

        (*tif->tif_postdecode)(tif,buf,tilesize);
        return (tilesize);
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

/* Variant of TIFFReadTile() that does 
 * * if *buf == NULL, *buf = _TIFFmalloc(bufsizetoalloc) only after TIFFFillTile() has
 *   succeeded. This avoid excessive memory allocation in case of truncated
 *   file.
 * * calls regular TIFFReadEncodedTile() if *buf != NULL
 */
tmsize_t
_TIFFReadTileAndAllocBuffer(TIFF* tif,
                            void **buf, tmsize_t bufsizetoalloc,
                            uint32 x, uint32 y, uint32 z, uint16 s)
{
    if (!TIFFCheckRead(tif, 1) || !TIFFCheckTile(tif, x, y, z, s))
            return ((tmsize_t)(-1));
    return (_TIFFReadEncodedTileAndAllocBuffer(tif,
                                               TIFFComputeTile(tif, x, y, z, s),
                                               buf, bufsizetoalloc,
                                               (tmsize_t)(-1)));
}

/* Variant of TIFFReadEncodedTile() that does 
 * * if *buf == NULL, *buf = _TIFFmalloc(bufsizetoalloc) only after TIFFFillTile() has
 *   succeeded. This avoid excessive memory allocation in case of truncated
 *   file.
 * * calls regular TIFFReadEncodedTile() if *buf != NULL
 */
tmsize_t
_TIFFReadEncodedTileAndAllocBuffer(TIFF* tif, uint32 tile,
                                    void **buf, tmsize_t bufsizetoalloc,
                                    tmsize_t size_to_read)
{
    static const char module[] = "_TIFFReadEncodedTileAndAllocBuffer";
    TIFFDirectory *td = &tif->tif_dir;
    tmsize_t tilesize = tif->tif_tilesize;

    if( *buf != NULL )
    {
        return TIFFReadEncodedTile(tif, tile, *buf, size_to_read);
    }

    if (!TIFFCheckRead(tif, 1))
            return ((tmsize_t)(-1));
    if (tile >= td->td_nstrips) {
            TIFFErrorExt(tif->tif_clientdata, module,
                "%lu: Tile out of range, max %lu",
                (unsigned long) tile, (unsigned long) td->td_nstrips);
            return ((tmsize_t)(-1));
    }

    if (!TIFFFillTile(tif,tile))
            return((tmsize_t)(-1));

    *buf = _TIFFmalloc(bufsizetoalloc);
    if (*buf == NULL) {
            TIFFErrorExt(tif->tif_clientdata, TIFFFileName(tif),
                         "No space for tile buffer");
            return((tmsize_t)(-1));
    }
    _TIFFmemset(*buf, 0, bufsizetoalloc);

    if (size_to_read == (tmsize_t)(-1))
        size_to_read = tilesize;
    else if (size_to_read > tilesize)
        size_to_read = tilesize;
    if( (*tif->tif_decodetile)(tif,
        (uint8*) *buf, size_to_read, (uint16)(tile/td->td_stripsperimage))) {
        (*tif->tif_postdecode)(tif, (uint8*) *buf, size_to_read);
        return (size_to_read);
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
		if ((td->td_stripoffset[tile] > (uint64)TIFF_TMSIZE_T_MAX)||(ma>tif->tif_size))
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
		if ((int64)bytecount <= 0) {
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

		/* To avoid excessive memory allocations: */
		/* Byte count should normally not be larger than a number of */
		/* times the uncompressed size plus some margin */
                if( bytecount > 1024 * 1024 )
                {
			/* 10 and 4096 are just values that could be adjusted. */
			/* Hopefully they are safe enough for all codecs */
			tmsize_t stripsize = TIFFTileSize(tif);
			if( stripsize != 0 &&
			    (bytecount - 4096) / 10 > (uint64)stripsize  )
			{
				uint64 newbytecount = (uint64)stripsize * 10 + 4096;
				if( (int64)newbytecount >= 0 )
				{
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
					TIFFWarningExt(tif->tif_clientdata, module,
					  "Too large tile byte count %I64u, tile %lu. Limiting to %I64u",
					     (unsigned __int64) bytecount,
					     (unsigned long) tile,
					     (unsigned __int64) newbytecount);
#else
					TIFFErrorExt(tif->tif_clientdata, module,
					  "Too large tile byte count %llu, tile %lu. Limiting to %llu",
					     (unsigned long long) bytecount,
					     (unsigned long) tile,
					     (unsigned long long) newbytecount);
#endif
					bytecount = newbytecount;
				}
			}
		}

		if (isMapped(tif)) {
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
			if ((tif->tif_flags & TIFF_MYBUFFER) && tif->tif_rawdata) {
				_TIFFfree(tif->tif_rawdata);
				tif->tif_rawdata = NULL;
				tif->tif_rawdatasize = 0;
			}
			tif->tif_flags &= ~TIFF_MYBUFFER;

			tif->tif_rawdatasize = (tmsize_t)bytecount;
			tif->tif_rawdata =
				tif->tif_base + (tmsize_t)td->td_stripoffset[tile];
                        tif->tif_rawdataoff = 0;
                        tif->tif_rawdataloaded = (tmsize_t) bytecount;
			tif->tif_flags |= TIFF_BUFFERMMAP;
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
			}
			if (tif->tif_flags&TIFF_BUFFERMMAP) {
				tif->tif_curtile = NOTILE;
				tif->tif_rawdata = NULL;
				tif->tif_rawdatasize = 0;
				tif->tif_flags &= ~TIFF_BUFFERMMAP;
			}

			if( isMapped(tif) )
			{
				if (bytecountm > tif->tif_rawdatasize &&
				    !TIFFReadBufferSetup(tif, 0, bytecountm))
				{
					return (0);
				}
				if (TIFFReadRawTile1(tif, tile, tif->tif_rawdata,
				    bytecountm, module) != bytecountm)
				{
					return (0);
				}
			}
			else
			{
				if (TIFFReadRawStripOrTile2(tif, tile, 0,
				    bytecountm, module) != bytecountm)
				{
					return (0);
				}
			}


                        tif->tif_rawdataoff = 0;
                        tif->tif_rawdataloaded = bytecountm;
                        
			if (tif->tif_rawdata != NULL &&
                            !isFillOrder(tif, td->td_fillorder) &&
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
	tif->tif_flags &= ~TIFF_BUFFERMMAP;

	if (tif->tif_rawdata) {
		if (tif->tif_flags & TIFF_MYBUFFER)
			_TIFFfree(tif->tif_rawdata);
		tif->tif_rawdata = NULL;
		tif->tif_rawdatasize = 0;
	}
	if (bp) {
		tif->tif_rawdatasize = size;
		tif->tif_rawdata = (uint8*) bp;
		tif->tif_flags &= ~TIFF_MYBUFFER;
	} else {
		tif->tif_rawdatasize = (tmsize_t)TIFFroundup_64((uint64)size, 1024);
		if (tif->tif_rawdatasize==0) {
		    TIFFErrorExt(tif->tif_clientdata, module,
				 "Invalid buffer size");
		    return (0);
		}
		/* Initialize to zero to avoid uninitialized buffers in case of */
                /* short reads (http://bugzilla.maptools.org/show_bug.cgi?id=2651) */
		tif->tif_rawdata = (uint8*) _TIFFcalloc(1, tif->tif_rawdatasize);
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
		if( tif->tif_rawdataloaded > 0 )
			tif->tif_rawcc = tif->tif_rawdataloaded;
		else
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
        static const char module[] = "TIFFStartTile";
	TIFFDirectory *td = &tif->tif_dir;
        uint32 howmany32;

        if (!_TIFFFillStriles( tif ) || !tif->tif_dir.td_stripbytecount)
                return 0;

	if ((tif->tif_flags & TIFF_CODERSETUP) == 0) {
		if (!(*tif->tif_setupdecode)(tif))
			return (0);
		tif->tif_flags |= TIFF_CODERSETUP;
	}
	tif->tif_curtile = tile;
        howmany32=TIFFhowmany_32(td->td_imagewidth, td->td_tilewidth);
        if (howmany32 == 0) {
                 TIFFErrorExt(tif->tif_clientdata,module,"Zero tiles");
                return 0;
        }
	tif->tif_row = (tile % howmany32) * td->td_tilelength;
        howmany32=TIFFhowmany_32(td->td_imagelength, td->td_tilelength);
        if (howmany32 == 0) {
                TIFFErrorExt(tif->tif_clientdata,module,"Zero tiles");
                return 0;
        }
	tif->tif_col = (tile % howmany32) * td->td_tilewidth;
        tif->tif_flags &= ~TIFF_BUF4WRITE;
	if (tif->tif_flags&TIFF_NOREADRAW)
	{
		tif->tif_rawcp = NULL;
		tif->tif_rawcc = 0;
	}
	else
	{
		tif->tif_rawcp = tif->tif_rawdata;
		if( tif->tif_rawdataloaded > 0 )
			tif->tif_rawcc = tif->tif_rawdataloaded;
		else
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
