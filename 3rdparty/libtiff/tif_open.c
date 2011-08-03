/* $Id: tif_open.c,v 1.33.2.2 2010-12-06 16:54:22 faxguy Exp $ */

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
 */
#include "tiffiop.h"

static const long typemask[13] = {
	(long)0L,		/* TIFF_NOTYPE */
	(long)0x000000ffL,	/* TIFF_BYTE */
	(long)0xffffffffL,	/* TIFF_ASCII */
	(long)0x0000ffffL,	/* TIFF_SHORT */
	(long)0xffffffffL,	/* TIFF_LONG */
	(long)0xffffffffL,	/* TIFF_RATIONAL */
	(long)0x000000ffL,	/* TIFF_SBYTE */
	(long)0x000000ffL,	/* TIFF_UNDEFINED */
	(long)0x0000ffffL,	/* TIFF_SSHORT */
	(long)0xffffffffL,	/* TIFF_SLONG */
	(long)0xffffffffL,	/* TIFF_SRATIONAL */
	(long)0xffffffffL,	/* TIFF_FLOAT */
	(long)0xffffffffL,	/* TIFF_DOUBLE */
};
static const int bigTypeshift[13] = {
	0,		/* TIFF_NOTYPE */
	24,		/* TIFF_BYTE */
	0,		/* TIFF_ASCII */
	16,		/* TIFF_SHORT */
	0,		/* TIFF_LONG */
	0,		/* TIFF_RATIONAL */
	24,		/* TIFF_SBYTE */
	24,		/* TIFF_UNDEFINED */
	16,		/* TIFF_SSHORT */
	0,		/* TIFF_SLONG */
	0,		/* TIFF_SRATIONAL */
	0,		/* TIFF_FLOAT */
	0,		/* TIFF_DOUBLE */
};
static const int litTypeshift[13] = {
	0,		/* TIFF_NOTYPE */
	0,		/* TIFF_BYTE */
	0,		/* TIFF_ASCII */
	0,		/* TIFF_SHORT */
	0,		/* TIFF_LONG */
	0,		/* TIFF_RATIONAL */
	0,		/* TIFF_SBYTE */
	0,		/* TIFF_UNDEFINED */
	0,		/* TIFF_SSHORT */
	0,		/* TIFF_SLONG */
	0,		/* TIFF_SRATIONAL */
	0,		/* TIFF_FLOAT */
	0,		/* TIFF_DOUBLE */
};

/*
 * Dummy functions to fill the omitted client procedures.
 */
static int
_tiffDummyMapProc(thandle_t fd, tdata_t* pbase, toff_t* psize)
{
	(void) fd; (void) pbase; (void) psize;
	return (0);
}

static void
_tiffDummyUnmapProc(thandle_t fd, tdata_t base, toff_t size)
{
	(void) fd; (void) base; (void) size;
}

/*
 * Initialize the shift & mask tables, and the
 * byte swapping state according to the file
 * contents and the machine architecture.
 */
static void
TIFFInitOrder(TIFF* tif, int magic)
{
	tif->tif_typemask = typemask;
	if (magic == TIFF_BIGENDIAN) {
		tif->tif_typeshift = bigTypeshift;
#ifndef WORDS_BIGENDIAN
		tif->tif_flags |= TIFF_SWAB;
#endif
	} else {
		tif->tif_typeshift = litTypeshift;
#ifdef WORDS_BIGENDIAN
		tif->tif_flags |= TIFF_SWAB;
#endif
	}
}

int
_TIFFgetMode(const char* mode, const char* module)
{
	int m = -1;

	switch (mode[0]) {
	case 'r':
		m = O_RDONLY;
		if (mode[1] == '+')
			m = O_RDWR;
		break;
	case 'w':
	case 'a':
		m = O_RDWR|O_CREAT;
		if (mode[0] == 'w')
			m |= O_TRUNC;
		break;
	default:
		TIFFErrorExt(0, module, "\"%s\": Bad mode", mode);
		break;
	}
	return (m);
}

TIFF*
TIFFClientOpen(
	const char* name, const char* mode,
	thandle_t clientdata,
	TIFFReadWriteProc readproc,
	TIFFReadWriteProc writeproc,
	TIFFSeekProc seekproc,
	TIFFCloseProc closeproc,
	TIFFSizeProc sizeproc,
	TIFFMapFileProc mapproc,
	TIFFUnmapFileProc unmapproc
)
{
	static const char module[] = "TIFFClientOpen";
	TIFF *tif;
	int m;
	const char* cp;

	m = _TIFFgetMode(mode, module);
	if (m == -1)
		goto bad2;
	tif = (TIFF *)_TIFFmalloc(sizeof (TIFF) + strlen(name) + 1);
	if (tif == NULL) {
		TIFFErrorExt(clientdata, module, "%s: Out of memory (TIFF structure)", name);
		goto bad2;
	}
	_TIFFmemset(tif, 0, sizeof (*tif));
	tif->tif_name = (char *)tif + sizeof (TIFF);
	strcpy(tif->tif_name, name);
	tif->tif_mode = m &~ (O_CREAT|O_TRUNC);
	tif->tif_curdir = (tdir_t) -1;		/* non-existent directory */
	tif->tif_curoff = 0;
	tif->tif_curstrip = (tstrip_t) -1;	/* invalid strip */
	tif->tif_row = (uint32) -1;		/* read/write pre-increment */
	tif->tif_clientdata = clientdata;
	if (!readproc || !writeproc || !seekproc || !closeproc || !sizeproc) {
		TIFFErrorExt(clientdata, module,
			  "One of the client procedures is NULL pointer.");
		goto bad2;
	}
	tif->tif_readproc = readproc;
	tif->tif_writeproc = writeproc;
	tif->tif_seekproc = seekproc;
	tif->tif_closeproc = closeproc;
	tif->tif_sizeproc = sizeproc;
        if (mapproc)
		tif->tif_mapproc = mapproc;
	else
		tif->tif_mapproc = _tiffDummyMapProc;
	if (unmapproc)
		tif->tif_unmapproc = unmapproc;
	else
		tif->tif_unmapproc = _tiffDummyUnmapProc;
	_TIFFSetDefaultCompressionState(tif);	/* setup default state */
	/*
	 * Default is to return data MSB2LSB and enable the
	 * use of memory-mapped files and strip chopping when
	 * a file is opened read-only.
	 */
	tif->tif_flags = FILLORDER_MSB2LSB;
	if (m == O_RDONLY )
		tif->tif_flags |= TIFF_MAPPED;

#ifdef STRIPCHOP_DEFAULT
	if (m == O_RDONLY || m == O_RDWR)
		tif->tif_flags |= STRIPCHOP_DEFAULT;
#endif

	/*
	 * Process library-specific flags in the open mode string.
	 * The following flags may be used to control intrinsic library
	 * behaviour that may or may not be desirable (usually for
	 * compatibility with some application that claims to support
	 * TIFF but only supports some braindead idea of what the
	 * vendor thinks TIFF is):
	 *
	 * 'l'		use little-endian byte order for creating a file
	 * 'b'		use big-endian byte order for creating a file
	 * 'L'		read/write information using LSB2MSB bit order
	 * 'B'		read/write information using MSB2LSB bit order
	 * 'H'		read/write information using host bit order
	 * 'M'		enable use of memory-mapped files when supported
	 * 'm'		disable use of memory-mapped files
	 * 'C'		enable strip chopping support when reading
	 * 'c'		disable strip chopping support
	 * 'h'		read TIFF header only, do not load the first IFD
	 *
	 * The use of the 'l' and 'b' flags is strongly discouraged.
	 * These flags are provided solely because numerous vendors,
	 * typically on the PC, do not correctly support TIFF; they
	 * only support the Intel little-endian byte order.  This
	 * support is not configured by default because it supports
	 * the violation of the TIFF spec that says that readers *MUST*
	 * support both byte orders.  It is strongly recommended that
	 * you not use this feature except to deal with busted apps
	 * that write invalid TIFF.  And even in those cases you should
	 * bang on the vendors to fix their software.
	 *
	 * The 'L', 'B', and 'H' flags are intended for applications
	 * that can optimize operations on data by using a particular
	 * bit order.  By default the library returns data in MSB2LSB
	 * bit order for compatibiltiy with older versions of this
	 * library.  Returning data in the bit order of the native cpu
	 * makes the most sense but also requires applications to check
	 * the value of the FillOrder tag; something they probably do
	 * not do right now.
	 *
	 * The 'M' and 'm' flags are provided because some virtual memory
	 * systems exhibit poor behaviour when large images are mapped.
	 * These options permit clients to control the use of memory-mapped
	 * files on a per-file basis.
	 *
	 * The 'C' and 'c' flags are provided because the library support
	 * for chopping up large strips into multiple smaller strips is not
	 * application-transparent and as such can cause problems.  The 'c'
	 * option permits applications that only want to look at the tags,
	 * for example, to get the unadulterated TIFF tag information.
	 */
	for (cp = mode; *cp; cp++)
		switch (*cp) {
		case 'b':
#ifndef WORDS_BIGENDIAN
		    if (m&O_CREAT)
				tif->tif_flags |= TIFF_SWAB;
#endif
			break;
		case 'l':
#ifdef WORDS_BIGENDIAN
			if ((m&O_CREAT))
				tif->tif_flags |= TIFF_SWAB;
#endif
			break;
		case 'B':
			tif->tif_flags = (tif->tif_flags &~ TIFF_FILLORDER) |
			    FILLORDER_MSB2LSB;
			break;
		case 'L':
			tif->tif_flags = (tif->tif_flags &~ TIFF_FILLORDER) |
			    FILLORDER_LSB2MSB;
			break;
		case 'H':
			tif->tif_flags = (tif->tif_flags &~ TIFF_FILLORDER) |
			    HOST_FILLORDER;
			break;
		case 'M':
			if (m == O_RDONLY)
				tif->tif_flags |= TIFF_MAPPED;
			break;
		case 'm':
			if (m == O_RDONLY)
				tif->tif_flags &= ~TIFF_MAPPED;
			break;
		case 'C':
			if (m == O_RDONLY)
				tif->tif_flags |= TIFF_STRIPCHOP;
			break;
		case 'c':
			if (m == O_RDONLY)
				tif->tif_flags &= ~TIFF_STRIPCHOP;
			break;
		case 'h':
			tif->tif_flags |= TIFF_HEADERONLY;
			break;
		}
	/*
	 * Read in TIFF header.
	 */
	if ((m & O_TRUNC) ||
	    !ReadOK(tif, &tif->tif_header, sizeof (TIFFHeader))) {
		if (tif->tif_mode == O_RDONLY) {
			TIFFErrorExt(tif->tif_clientdata, name,
				     "Cannot read TIFF header");
			goto bad;
		}
		/*
		 * Setup header and write.
		 */
#ifdef WORDS_BIGENDIAN
		tif->tif_header.tiff_magic = tif->tif_flags & TIFF_SWAB
		    ? TIFF_LITTLEENDIAN : TIFF_BIGENDIAN;
#else
		tif->tif_header.tiff_magic = tif->tif_flags & TIFF_SWAB
		    ? TIFF_BIGENDIAN : TIFF_LITTLEENDIAN;
#endif
		tif->tif_header.tiff_version = TIFF_VERSION;
		if (tif->tif_flags & TIFF_SWAB)
			TIFFSwabShort(&tif->tif_header.tiff_version);
		tif->tif_header.tiff_diroff = 0;	/* filled in later */


                /*
                 * The doc for "fopen" for some STD_C_LIBs says that if you 
                 * open a file for modify ("+"), then you must fseek (or 
                 * fflush?) between any freads and fwrites.  This is not
                 * necessary on most systems, but has been shown to be needed
                 * on Solaris. 
                 */
                TIFFSeekFile( tif, 0, SEEK_SET );
               
		if (!WriteOK(tif, &tif->tif_header, sizeof (TIFFHeader))) {
			TIFFErrorExt(tif->tif_clientdata, name,
				     "Error writing TIFF header");
			goto bad;
		}
		/*
		 * Setup the byte order handling.
		 */
		TIFFInitOrder(tif, tif->tif_header.tiff_magic);
		/*
		 * Setup default directory.
		 */
		if (!TIFFDefaultDirectory(tif))
			goto bad;
		tif->tif_diroff = 0;
		tif->tif_dirlist = NULL;
		tif->tif_dirlistsize = 0;
		tif->tif_dirnumber = 0;
		return (tif);
	}
	/*
	 * Setup the byte order handling.
	 */
	if (tif->tif_header.tiff_magic != TIFF_BIGENDIAN &&
	    tif->tif_header.tiff_magic != TIFF_LITTLEENDIAN
#if MDI_SUPPORT
	    &&
#if HOST_BIGENDIAN
	    tif->tif_header.tiff_magic != MDI_BIGENDIAN
#else
	    tif->tif_header.tiff_magic != MDI_LITTLEENDIAN
#endif
	    ) {
		TIFFErrorExt(tif->tif_clientdata, name,
			"Not a TIFF or MDI file, bad magic number %d (0x%x)",
#else
	    ) {
		TIFFErrorExt(tif->tif_clientdata, name,
			     "Not a TIFF file, bad magic number %d (0x%x)",
#endif
		    tif->tif_header.tiff_magic,
		    tif->tif_header.tiff_magic);
		goto bad;
	}
	TIFFInitOrder(tif, tif->tif_header.tiff_magic);
	/*
	 * Swap header if required.
	 */
	if (tif->tif_flags & TIFF_SWAB) {
		TIFFSwabShort(&tif->tif_header.tiff_version);
		TIFFSwabLong(&tif->tif_header.tiff_diroff);
	}
	/*
	 * Now check version (if needed, it's been byte-swapped).
	 * Note that this isn't actually a version number, it's a
	 * magic number that doesn't change (stupid).
	 */
	if (tif->tif_header.tiff_version == TIFF_BIGTIFF_VERSION) {
		TIFFErrorExt(tif->tif_clientdata, name,
                          "This is a BigTIFF file.  This format not supported\n"
                          "by this version of libtiff." );
		goto bad;
	}
	if (tif->tif_header.tiff_version != TIFF_VERSION) {
		TIFFErrorExt(tif->tif_clientdata, name,
		    "Not a TIFF file, bad version number %d (0x%x)",
		    tif->tif_header.tiff_version,
		    tif->tif_header.tiff_version);
		goto bad;
	}
	tif->tif_flags |= TIFF_MYBUFFER;
	tif->tif_rawcp = tif->tif_rawdata = 0;
	tif->tif_rawdatasize = 0;

	/*
	 * Sometimes we do not want to read the first directory (for example,
	 * it may be broken) and want to proceed to other directories. I this
	 * case we use the TIFF_HEADERONLY flag to open file and return
	 * immediately after reading TIFF header.
	 */
	if (tif->tif_flags & TIFF_HEADERONLY)
		return (tif);

	/*
	 * Setup initial directory.
	 */
	switch (mode[0]) {
	case 'r':
		tif->tif_nextdiroff = tif->tif_header.tiff_diroff;
		/*
		 * Try to use a memory-mapped file if the client
		 * has not explicitly suppressed usage with the
		 * 'm' flag in the open mode (see above).
		 */
		if ((tif->tif_flags & TIFF_MAPPED) &&
	!TIFFMapFileContents(tif, (tdata_t*) &tif->tif_base, &tif->tif_size))
			tif->tif_flags &= ~TIFF_MAPPED;
		if (TIFFReadDirectory(tif)) {
			tif->tif_rawcc = -1;
			tif->tif_flags |= TIFF_BUFFERSETUP;
			return (tif);
		}
		break;
	case 'a':
		/*
		 * New directories are automatically append
		 * to the end of the directory chain when they
		 * are written out (see TIFFWriteDirectory).
		 */
		if (!TIFFDefaultDirectory(tif))
			goto bad;
		return (tif);
	}
bad:
	tif->tif_mode = O_RDONLY;	/* XXX avoid flush */
        TIFFCleanup(tif);
bad2:
	return ((TIFF*)0);
}

/*
 * Query functions to access private data.
 */

/*
 * Return open file's name.
 */
const char *
TIFFFileName(TIFF* tif)
{
	return (tif->tif_name);
}

/*
 * Set the file name.
 */
const char *
TIFFSetFileName(TIFF* tif, const char *name)
{
	const char* old_name = tif->tif_name;
	tif->tif_name = (char *)name;
	return (old_name);
}

/*
 * Return open file's I/O descriptor.
 */
int
TIFFFileno(TIFF* tif)
{
	return (tif->tif_fd);
}

/*
 * Set open file's I/O descriptor, and return previous value.
 */
int
TIFFSetFileno(TIFF* tif, int fd)
{
        int old_fd = tif->tif_fd;
	tif->tif_fd = fd;
	return old_fd;
}

/*
 * Return open file's clientdata.
 */
thandle_t
TIFFClientdata(TIFF* tif)
{
	return (tif->tif_clientdata);
}

/*
 * Set open file's clientdata, and return previous value.
 */
thandle_t
TIFFSetClientdata(TIFF* tif, thandle_t newvalue)
{
	thandle_t m = tif->tif_clientdata;
	tif->tif_clientdata = newvalue;
	return m;
}

/*
 * Return read/write mode.
 */
int
TIFFGetMode(TIFF* tif)
{
	return (tif->tif_mode);
}

/*
 * Return read/write mode.
 */
int
TIFFSetMode(TIFF* tif, int mode)
{
	int old_mode = tif->tif_mode;
	tif->tif_mode = mode;
	return (old_mode);
}

/*
 * Return nonzero if file is organized in
 * tiles; zero if organized as strips.
 */
int
TIFFIsTiled(TIFF* tif)
{
	return (isTiled(tif));
}

/*
 * Return current row being read/written.
 */
uint32
TIFFCurrentRow(TIFF* tif)
{
	return (tif->tif_row);
}

/*
 * Return index of the current directory.
 */
tdir_t
TIFFCurrentDirectory(TIFF* tif)
{
	return (tif->tif_curdir);
}

/*
 * Return current strip.
 */
tstrip_t
TIFFCurrentStrip(TIFF* tif)
{
	return (tif->tif_curstrip);
}

/*
 * Return current tile.
 */
ttile_t
TIFFCurrentTile(TIFF* tif)
{
	return (tif->tif_curtile);
}

/*
 * Return nonzero if the file has byte-swapped data.
 */
int
TIFFIsByteSwapped(TIFF* tif)
{
	return ((tif->tif_flags & TIFF_SWAB) != 0);
}

/*
 * Return nonzero if the data is returned up-sampled.
 */
int
TIFFIsUpSampled(TIFF* tif)
{
	return (isUpSampled(tif));
}

/*
 * Return nonzero if the data is returned in MSB-to-LSB bit order.
 */
int
TIFFIsMSB2LSB(TIFF* tif)
{
	return (isFillOrder(tif, FILLORDER_MSB2LSB));
}

/*
 * Return nonzero if given file was written in big-endian order.
 */
int
TIFFIsBigEndian(TIFF* tif)
{
	return (tif->tif_header.tiff_magic == TIFF_BIGENDIAN);
}

/*
 * Return pointer to file read method.
 */
TIFFReadWriteProc
TIFFGetReadProc(TIFF* tif)
{
	return (tif->tif_readproc);
}

/*
 * Return pointer to file write method.
 */
TIFFReadWriteProc
TIFFGetWriteProc(TIFF* tif)
{
	return (tif->tif_writeproc);
}

/*
 * Return pointer to file seek method.
 */
TIFFSeekProc
TIFFGetSeekProc(TIFF* tif)
{
	return (tif->tif_seekproc);
}

/*
 * Return pointer to file close method.
 */
TIFFCloseProc
TIFFGetCloseProc(TIFF* tif)
{
	return (tif->tif_closeproc);
}

/*
 * Return pointer to file size requesting method.
 */
TIFFSizeProc
TIFFGetSizeProc(TIFF* tif)
{
	return (tif->tif_sizeproc);
}

/*
 * Return pointer to memory mapping method.
 */
TIFFMapFileProc
TIFFGetMapFileProc(TIFF* tif)
{
	return (tif->tif_mapproc);
}

/*
 * Return pointer to memory unmapping method.
 */
TIFFUnmapFileProc
TIFFGetUnmapFileProc(TIFF* tif)
{
	return (tif->tif_unmapproc);
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
