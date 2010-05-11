/* $Id: tif_zip.c,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

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

#include "tiffiop.h"
#ifdef ZIP_SUPPORT
/*
 * TIFF Library.
 *
 * ZIP (aka Deflate) Compression Support
 *
 * This file is simply an interface to the zlib library written by
 * Jean-loup Gailly and Mark Adler.  You must use version 1.0 or later
 * of the library: this code assumes the 1.0 API and also depends on
 * the ability to write the zlib header multiple times (one per strip)
 * which was not possible with versions prior to 0.95.  Note also that
 * older versions of this codec avoided this bug by supressing the header
 * entirely.  This means that files written with the old library cannot
 * be read; they should be converted to a different compression scheme
 * and then reconverted.
 *
 * The data format used by the zlib library is described in the files
 * zlib-3.1.doc, deflate-1.1.doc and gzip-4.1.doc, available in the
 * directory ftp://ftp.uu.net/pub/archiving/zip/doc.  The library was
 * last found at ftp://ftp.uu.net/pub/archiving/zip/zlib/zlib-0.99.tar.gz.
 */
#include "tif_predict.h"
#include "zlib.h"

#include <stdio.h>

/*
 * Sigh, ZLIB_VERSION is defined as a string so there's no
 * way to do a proper check here.  Instead we guess based
 * on the presence of #defines that were added between the
 * 0.95 and 1.0 distributions.
 */
#if !defined(Z_NO_COMPRESSION) || !defined(Z_DEFLATED)
#error "Antiquated ZLIB software; you must use version 1.0 or later"
#endif

/*
 * State block for each open TIFF
 * file using ZIP compression/decompression.
 */
typedef	struct {
	TIFFPredictorState predict;
	z_stream	stream;
	int		zipquality;		/* compression level */
	int		state;			/* state flags */
#define	ZSTATE_INIT	0x1		/* zlib setup successfully */

	TIFFVGetMethod	vgetparent;		/* super-class method */
	TIFFVSetMethod	vsetparent;		/* super-class method */
} ZIPState;

#define	ZState(tif)		((ZIPState*) (tif)->tif_data)
#define	DecoderState(tif)	ZState(tif)
#define	EncoderState(tif)	ZState(tif)

static	int ZIPEncode(TIFF*, tidata_t, tsize_t, tsample_t);
static	int ZIPDecode(TIFF*, tidata_t, tsize_t, tsample_t);

static int
ZIPSetupDecode(TIFF* tif)
{
	ZIPState* sp = DecoderState(tif);
	static const char module[] = "ZIPSetupDecode";

	assert(sp != NULL);
	if (inflateInit(&sp->stream) != Z_OK) {
		TIFFError(module, "%s: %s", tif->tif_name, sp->stream.msg);
		return (0);
	} else {
		sp->state |= ZSTATE_INIT;
		return (1);
	}
}

/*
 * Setup state for decoding a strip.
 */
static int
ZIPPreDecode(TIFF* tif, tsample_t s)
{
	ZIPState* sp = DecoderState(tif);

	(void) s;
	assert(sp != NULL);
	sp->stream.next_in = tif->tif_rawdata;
	sp->stream.avail_in = tif->tif_rawcc;
	return (inflateReset(&sp->stream) == Z_OK);
}

static int
ZIPDecode(TIFF* tif, tidata_t op, tsize_t occ, tsample_t s)
{
	ZIPState* sp = DecoderState(tif);
	static const char module[] = "ZIPDecode";

	(void) s;
	assert(sp != NULL);
	sp->stream.next_out = op;
	sp->stream.avail_out = occ;
	do {
		int state = inflate(&sp->stream, Z_PARTIAL_FLUSH);
		if (state == Z_STREAM_END)
			break;
		if (state == Z_DATA_ERROR) {
			TIFFError(module,
			    "%s: Decoding error at scanline %d, %s",
			    tif->tif_name, tif->tif_row, sp->stream.msg);
			if (inflateSync(&sp->stream) != Z_OK)
				return (0);
			continue;
		}
		if (state != Z_OK) {
			TIFFError(module, "%s: zlib error: %s",
			    tif->tif_name, sp->stream.msg);
			return (0);
		}
	} while (sp->stream.avail_out > 0);
	if (sp->stream.avail_out != 0) {
		TIFFError(module,
		    "%s: Not enough data at scanline %d (short %d bytes)",
		    tif->tif_name, tif->tif_row, sp->stream.avail_out);
		return (0);
	}
	return (1);
}

static int
ZIPSetupEncode(TIFF* tif)
{
	ZIPState* sp = EncoderState(tif);
	static const char module[] = "ZIPSetupEncode";

	assert(sp != NULL);
	if (deflateInit(&sp->stream, sp->zipquality) != Z_OK) {
		TIFFError(module, "%s: %s", tif->tif_name, sp->stream.msg);
		return (0);
	} else {
		sp->state |= ZSTATE_INIT;
		return (1);
	}
}

/*
 * Reset encoding state at the start of a strip.
 */
static int
ZIPPreEncode(TIFF* tif, tsample_t s)
{
	ZIPState *sp = EncoderState(tif);

	(void) s;
	assert(sp != NULL);
	sp->stream.next_out = tif->tif_rawdata;
	sp->stream.avail_out = tif->tif_rawdatasize;
	return (deflateReset(&sp->stream) == Z_OK);
}

/*
 * Encode a chunk of pixels.
 */
static int
ZIPEncode(TIFF* tif, tidata_t bp, tsize_t cc, tsample_t s)
{
	ZIPState *sp = EncoderState(tif);
	static const char module[] = "ZIPEncode";

	(void) s;
	sp->stream.next_in = bp;
	sp->stream.avail_in = cc;
	do {
		if (deflate(&sp->stream, Z_NO_FLUSH) != Z_OK) {
			TIFFError(module, "%s: Encoder error: %s",
			    tif->tif_name, sp->stream.msg);
			return (0);
		}
		if (sp->stream.avail_out == 0) {
			tif->tif_rawcc = tif->tif_rawdatasize;
			TIFFFlushData1(tif);
			sp->stream.next_out = tif->tif_rawdata;
			sp->stream.avail_out = tif->tif_rawdatasize;
		}
	} while (sp->stream.avail_in > 0);
	return (1);
}

/*
 * Finish off an encoded strip by flushing the last
 * string and tacking on an End Of Information code.
 */
static int
ZIPPostEncode(TIFF* tif)
{
	ZIPState *sp = EncoderState(tif);
	static const char module[] = "ZIPPostEncode";
	int state;

	sp->stream.avail_in = 0;
	do {
		state = deflate(&sp->stream, Z_FINISH);
		switch (state) {
		case Z_STREAM_END:
		case Z_OK:
		    if ((int)sp->stream.avail_out != (int)tif->tif_rawdatasize)
                    {
			    tif->tif_rawcc =
				tif->tif_rawdatasize - sp->stream.avail_out;
			    TIFFFlushData1(tif);
			    sp->stream.next_out = tif->tif_rawdata;
			    sp->stream.avail_out = tif->tif_rawdatasize;
		    }
		    break;
		default:
		    TIFFError(module, "%s: zlib error: %s",
			tif->tif_name, sp->stream.msg);
		    return (0);
		}
	} while (state != Z_STREAM_END);
	return (1);
}

static void
ZIPCleanup(TIFF* tif)
{
	ZIPState* sp = ZState(tif);
	if (sp) {
		if (sp->state&ZSTATE_INIT) {
			/* NB: avoid problems in the library */
			if (tif->tif_mode == O_RDONLY)
				inflateEnd(&sp->stream);
			else
				deflateEnd(&sp->stream);
		}
		_TIFFfree(sp);
		tif->tif_data = NULL;
	}
}

static int
ZIPVSetField(TIFF* tif, ttag_t tag, va_list ap)
{
	ZIPState* sp = ZState(tif);
	static const char module[] = "ZIPVSetField";

	switch (tag) {
	case TIFFTAG_ZIPQUALITY:
		sp->zipquality = va_arg(ap, int);
		if (tif->tif_mode != O_RDONLY && (sp->state&ZSTATE_INIT)) {
			if (deflateParams(&sp->stream,
			    sp->zipquality, Z_DEFAULT_STRATEGY) != Z_OK) {
				TIFFError(module, "%s: zlib error: %s",
				    tif->tif_name, sp->stream.msg);
				return (0);
			}
		}
		return (1);
	default:
		return (*sp->vsetparent)(tif, tag, ap);
	}
	/*NOTREACHED*/
}

static int
ZIPVGetField(TIFF* tif, ttag_t tag, va_list ap)
{
	ZIPState* sp = ZState(tif);

	switch (tag) {
	case TIFFTAG_ZIPQUALITY:
		*va_arg(ap, int*) = sp->zipquality;
		break;
	default:
		return (*sp->vgetparent)(tif, tag, ap);
	}
	return (1);
}

static const TIFFFieldInfo zipFieldInfo[] = {
    { TIFFTAG_ZIPQUALITY,	 0, 0,	TIFF_ANY,	FIELD_PSEUDO,
      TRUE,	FALSE,	"" },
};
#define	N(a)	(sizeof (a) / sizeof (a[0]))

int
TIFFInitZIP(TIFF* tif, int scheme)
{
	ZIPState* sp;

	assert( (scheme == COMPRESSION_DEFLATE) || (scheme == COMPRESSION_ADOBE_DEFLATE));

	/*
	 * Allocate state block so tag methods have storage to record values.
	 */
	tif->tif_data = (tidata_t) _TIFFmalloc(sizeof (ZIPState));
	if (tif->tif_data == NULL)
		goto bad;
	sp = ZState(tif);
	sp->stream.zalloc = NULL;
	sp->stream.zfree = NULL;
	sp->stream.opaque = NULL;
	sp->stream.data_type = Z_BINARY;

	/*
	 * Merge codec-specific tag information and
	 * override parent get/set field methods.
	 */
	_TIFFMergeFieldInfo(tif, zipFieldInfo, N(zipFieldInfo));
	sp->vgetparent = tif->tif_tagmethods.vgetfield;
	tif->tif_tagmethods.vgetfield = ZIPVGetField;	/* hook for codec tags */
	sp->vsetparent = tif->tif_tagmethods.vsetfield;
	tif->tif_tagmethods.vsetfield = ZIPVSetField;	/* hook for codec tags */

	/* Default values for codec-specific fields */
	sp->zipquality = Z_DEFAULT_COMPRESSION;	/* default comp. level */
	sp->state = 0;

	/*
	 * Install codec methods.
	 */
	tif->tif_setupdecode = ZIPSetupDecode;
	tif->tif_predecode = ZIPPreDecode;
	tif->tif_decoderow = ZIPDecode;
	tif->tif_decodestrip = ZIPDecode;
	tif->tif_decodetile = ZIPDecode;
	tif->tif_setupencode = ZIPSetupEncode;
	tif->tif_preencode = ZIPPreEncode;
	tif->tif_postencode = ZIPPostEncode;
	tif->tif_encoderow = ZIPEncode;
	tif->tif_encodestrip = ZIPEncode;
	tif->tif_encodetile = ZIPEncode;
	tif->tif_cleanup = ZIPCleanup;
	/*
	 * Setup predictor setup.
	 */
	(void) TIFFPredictorInit(tif);
	return (1);
bad:
	TIFFError("TIFFInitZIP", "No space for ZIP state block");
	return (0);
}
#endif /* ZIP_SUPORT */

/* vim: set ts=8 sts=8 sw=8 noet: */
