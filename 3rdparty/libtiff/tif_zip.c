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
 * This file is an interface to the zlib library written by
 * Jean-loup Gailly and Mark Adler.  You must use version 1.0 or later
 * of the library.
 *
 * Optionally, libdeflate (https://github.com/ebiggers/libdeflate) may be used
 * to do the compression and decompression, but only for whole strips and tiles.
 * For scanline access, zlib will be sued as a fallback.
 */
#include "tif_predict.h"
#include "zlib.h"

#if LIBDEFLATE_SUPPORT
#include "libdeflate.h"
#endif
#define LIBDEFLATE_MAX_COMPRESSION_LEVEL 12

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

#define SAFE_MSG(sp)   ((sp)->stream.msg == NULL ? "" : (sp)->stream.msg)

/*
 * State block for each open TIFF
 * file using ZIP compression/decompression.
 */
typedef struct {
	TIFFPredictorState predict;
        z_stream        stream;
	int             zipquality;            /* compression level */
	int             state;                 /* state flags */
	int             subcodec;              /* DEFLATE_SUBCODEC_ZLIB or DEFLATE_SUBCODEC_LIBDEFLATE */
#if LIBDEFLATE_SUPPORT
	int             libdeflate_state;       /* -1 = until first time ZIPEncode() / ZIPDecode() is called, 0 = use zlib, 1 = use libdeflate */
	struct libdeflate_decompressor* libdeflate_dec;
	struct libdeflate_compressor*   libdeflate_enc;
#endif
#define ZSTATE_INIT_DECODE 0x01
#define ZSTATE_INIT_ENCODE 0x02

	TIFFVGetMethod  vgetparent;            /* super-class method */
	TIFFVSetMethod  vsetparent;            /* super-class method */
} ZIPState;

#define ZState(tif)             ((ZIPState*) (tif)->tif_data)
#define DecoderState(tif)       ZState(tif)
#define EncoderState(tif)       ZState(tif)

static int ZIPEncode(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s);
static int ZIPDecode(TIFF* tif, uint8* op, tmsize_t occ, uint16 s);

static int
ZIPFixupTags(TIFF* tif)
{
	(void) tif;
	return (1);
}

static int
ZIPSetupDecode(TIFF* tif)
{
	static const char module[] = "ZIPSetupDecode";
	ZIPState* sp = DecoderState(tif);

	assert(sp != NULL);
        
        /* if we were last encoding, terminate this mode */
	if (sp->state & ZSTATE_INIT_ENCODE) {
	    deflateEnd(&sp->stream);
	    sp->state = 0;
	}

	/* This function can possibly be called several times by */
	/* PredictorSetupDecode() if this function succeeds but */
	/* PredictorSetup() fails */
	if ((sp->state & ZSTATE_INIT_DECODE) == 0 &&
	    inflateInit(&sp->stream) != Z_OK) {
		TIFFErrorExt(tif->tif_clientdata, module, "%s", SAFE_MSG(sp));
		return (0);
	} else {
		sp->state |= ZSTATE_INIT_DECODE;
		return (1);
	}
}

/*
 * Setup state for decoding a strip.
 */
static int
ZIPPreDecode(TIFF* tif, uint16 s)
{
	ZIPState* sp = DecoderState(tif);

	(void) s;
	assert(sp != NULL);

	if( (sp->state & ZSTATE_INIT_DECODE) == 0 )
            tif->tif_setupdecode( tif );

#if LIBDEFLATE_SUPPORT
        sp->libdeflate_state = -1;
#endif
	sp->stream.next_in = tif->tif_rawdata;
	assert(sizeof(sp->stream.avail_in)==4);  /* if this assert gets raised,
	    we need to simplify this code to reflect a ZLib that is likely updated
	    to deal with 8byte memory sizes, though this code will respond
	    appropriately even before we simplify it */
	sp->stream.avail_in = (uint64)tif->tif_rawcc < 0xFFFFFFFFU ? (uInt) tif->tif_rawcc : 0xFFFFFFFFU;
	return (inflateReset(&sp->stream) == Z_OK);
}

static int
ZIPDecode(TIFF* tif, uint8* op, tmsize_t occ, uint16 s)
{
	static const char module[] = "ZIPDecode";
	ZIPState* sp = DecoderState(tif);

	(void) s;
	assert(sp != NULL);
	assert(sp->state == ZSTATE_INIT_DECODE);

#if LIBDEFLATE_SUPPORT
        if( sp->libdeflate_state == 1 )
            return 0;

        /* If we have libdeflate support and we are asked to read a whole */
        /* strip/tile, then go for using it */
        do {
            TIFFDirectory *td = &tif->tif_dir;

            if( sp->libdeflate_state == 0 )
                break;
            if( sp->subcodec == DEFLATE_SUBCODEC_ZLIB )
                break;

            /* Check if we are in the situation where we can use libdeflate */
            if (isTiled(tif)) {
                if( TIFFTileSize64(tif) != (uint64)occ )
                    break;
            } else {
                uint32 strip_height = td->td_imagelength - tif->tif_row;
                if (strip_height > td->td_rowsperstrip)
                    strip_height = td->td_rowsperstrip;
                if( TIFFVStripSize64(tif, strip_height) != (uint64)occ )
                    break;
            }

            /* Check for overflow */
            if( (size_t)tif->tif_rawcc != (uint64)tif->tif_rawcc )
                break;
            if( (size_t)occ != (uint64)occ )
                break;

            /* Go for decompression using libdeflate */
            {
                enum libdeflate_result res;
                if( sp->libdeflate_dec == NULL )
                {
                    sp->libdeflate_dec = libdeflate_alloc_decompressor();
                    if( sp->libdeflate_dec == NULL )
                    {
                        break;
                    }
                }

                sp->libdeflate_state = 1;

                res = libdeflate_zlib_decompress(
                    sp->libdeflate_dec, tif->tif_rawcp, (size_t)tif->tif_rawcc, op, (size_t)occ, NULL);

                tif->tif_rawcp += tif->tif_rawcc;
                tif->tif_rawcc = 0;

                /* We accept LIBDEFLATE_INSUFFICIENT_SPACE has a return */
                /* There are odd files in the wild where the last strip, when */
                /* it is smaller in height than td_rowsperstrip, actually contains */
                /* data for td_rowsperstrip lines. Just ignore that silently. */
                if( res != LIBDEFLATE_SUCCESS &&
                    res != LIBDEFLATE_INSUFFICIENT_SPACE )
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                                 "Decoding error at scanline %lu",
                                 (unsigned long) tif->tif_row);
                    return 0;
                }

                return 1;
            }
        } while(0);
        sp->libdeflate_state = 0;
#endif /* LIBDEFLATE_SUPPORT */

        sp->stream.next_in = tif->tif_rawcp;
        
	sp->stream.next_out = op;
	assert(sizeof(sp->stream.avail_out)==4);  /* if this assert gets raised,
	    we need to simplify this code to reflect a ZLib that is likely updated
	    to deal with 8byte memory sizes, though this code will respond
	    appropriately even before we simplify it */
	do {
                int state;
                uInt avail_in_before = (uint64)tif->tif_rawcc <= 0xFFFFFFFFU ? (uInt)tif->tif_rawcc : 0xFFFFFFFFU;
                uInt avail_out_before = (uint64)occ < 0xFFFFFFFFU ? (uInt) occ : 0xFFFFFFFFU;
                sp->stream.avail_in = avail_in_before;
                sp->stream.avail_out = avail_out_before;
		state = inflate(&sp->stream, Z_PARTIAL_FLUSH);
		tif->tif_rawcc -= (avail_in_before - sp->stream.avail_in);
                occ -= (avail_out_before - sp->stream.avail_out);
		if (state == Z_STREAM_END)
			break;
		if (state == Z_DATA_ERROR) {
			TIFFErrorExt(tif->tif_clientdata, module,
			    "Decoding error at scanline %lu, %s",
			     (unsigned long) tif->tif_row, SAFE_MSG(sp));
			return (0);
		}
		if (state != Z_OK) {
			TIFFErrorExt(tif->tif_clientdata, module, 
				     "ZLib error: %s", SAFE_MSG(sp));
			return (0);
		}
	} while (occ > 0);
	if (occ != 0) {
		TIFFErrorExt(tif->tif_clientdata, module,
		    "Not enough data at scanline %lu (short " TIFF_UINT64_FORMAT " bytes)",
		    (unsigned long) tif->tif_row, (TIFF_UINT64_T) occ);
		return (0);
	}

        tif->tif_rawcp = sp->stream.next_in;

	return (1);
}

static int
ZIPSetupEncode(TIFF* tif)
{
	static const char module[] = "ZIPSetupEncode";
	ZIPState* sp = EncoderState(tif);
        int cappedQuality;

	assert(sp != NULL);
	if (sp->state & ZSTATE_INIT_DECODE) {
		inflateEnd(&sp->stream);
		sp->state = 0;
	}

        cappedQuality = sp->zipquality;
        if( cappedQuality > Z_BEST_COMPRESSION )
            cappedQuality = Z_BEST_COMPRESSION;

	if (deflateInit(&sp->stream, cappedQuality) != Z_OK) {
		TIFFErrorExt(tif->tif_clientdata, module, "%s", SAFE_MSG(sp));
		return (0);
	} else {
		sp->state |= ZSTATE_INIT_ENCODE;
		return (1);
	}
}

/*
 * Reset encoding state at the start of a strip.
 */
static int
ZIPPreEncode(TIFF* tif, uint16 s)
{
	ZIPState *sp = EncoderState(tif);

	(void) s;
	assert(sp != NULL);
	if( sp->state != ZSTATE_INIT_ENCODE )
            tif->tif_setupencode( tif );

#if LIBDEFLATE_SUPPORT
        sp->libdeflate_state = -1;
#endif
	sp->stream.next_out = tif->tif_rawdata;
	assert(sizeof(sp->stream.avail_out)==4);  /* if this assert gets raised,
	    we need to simplify this code to reflect a ZLib that is likely updated
	    to deal with 8byte memory sizes, though this code will respond
	    appropriately even before we simplify it */
	sp->stream.avail_out = (uint64)tif->tif_rawdatasize <= 0xFFFFFFFFU ? (uInt)tif->tif_rawdatasize : 0xFFFFFFFFU;
	return (deflateReset(&sp->stream) == Z_OK);
}

/*
 * Encode a chunk of pixels.
 */
static int
ZIPEncode(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s)
{
	static const char module[] = "ZIPEncode";
	ZIPState *sp = EncoderState(tif);

	assert(sp != NULL);
	assert(sp->state == ZSTATE_INIT_ENCODE);

	(void) s;

#if LIBDEFLATE_SUPPORT
        if( sp->libdeflate_state == 1 )
            return 0;

        /* If we have libdeflate support and we are asked to write a whole */
        /* strip/tile, then go for using it */
        do {
            TIFFDirectory *td = &tif->tif_dir;

            if( sp->libdeflate_state == 0 )
                break;
            if( sp->subcodec == DEFLATE_SUBCODEC_ZLIB )
                break;

            /* Libdeflate does not support the 0-compression level */
            if( sp->zipquality == Z_NO_COMPRESSION )
                break;

            /* Check if we are in the situation where we can use libdeflate */
            if (isTiled(tif)) {
                if( TIFFTileSize64(tif) != (uint64)cc )
                    break;
            } else {
                uint32 strip_height = td->td_imagelength - tif->tif_row;
                if (strip_height > td->td_rowsperstrip)
                    strip_height = td->td_rowsperstrip;
                if( TIFFVStripSize64(tif, strip_height) != (uint64)cc )
                    break;
            }

            /* Check for overflow */
            if( (size_t)tif->tif_rawdatasize != (uint64)tif->tif_rawdatasize )
                break;
            if( (size_t)cc != (uint64)cc )
                break;

            /* Go for compression using libdeflate */
            {
                size_t nCompressedBytes;
                if( sp->libdeflate_enc == NULL )
                {
                    /* To get results as good as zlib, we asked for an extra */
                    /* level of compression */
                    sp->libdeflate_enc = libdeflate_alloc_compressor(
                        sp->zipquality == Z_DEFAULT_COMPRESSION ? 7 :
                        sp->zipquality >= 6 && sp->zipquality <= 9 ? sp->zipquality + 1 :
                        sp->zipquality);
                    if( sp->libdeflate_enc == NULL )
                    {
                        TIFFErrorExt(tif->tif_clientdata, module,
                                    "Cannot allocate compressor");
                        break;
                    }
                }

                /* Make sure the output buffer is large enough for the worse case. */
                /* In TIFFWriteBufferSetup(), when libtiff allocates the buffer */
                /* we've taken a 10% margin over the uncompressed size, which should */
                /* be large enough even for the the worse case scenario. */
                if( libdeflate_zlib_compress_bound(sp->libdeflate_enc, (size_t)cc) >
                        (size_t)tif->tif_rawdatasize)
                {
                    break;
                }

                sp->libdeflate_state = 1;
                nCompressedBytes = libdeflate_zlib_compress(
                    sp->libdeflate_enc, bp, (size_t)cc, tif->tif_rawdata, (size_t)tif->tif_rawdatasize);

                if( nCompressedBytes == 0 )
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                                 "Encoder error at scanline %lu",
                                 (unsigned long) tif->tif_row);
                    return 0;
                }

                tif->tif_rawcc = nCompressedBytes;

                if( !TIFFFlushData1(tif) )
                    return 0;

                return 1;
            }
        } while(0);
        sp->libdeflate_state = 0;
#endif /* LIBDEFLATE_SUPPORT */

	sp->stream.next_in = bp;
	assert(sizeof(sp->stream.avail_in)==4);  /* if this assert gets raised,
	    we need to simplify this code to reflect a ZLib that is likely updated
	    to deal with 8byte memory sizes, though this code will respond
	    appropriately even before we simplify it */
	do {
                uInt avail_in_before = (uint64)cc <= 0xFFFFFFFFU ? (uInt)cc : 0xFFFFFFFFU;
                sp->stream.avail_in = avail_in_before;
		if (deflate(&sp->stream, Z_NO_FLUSH) != Z_OK) {
			TIFFErrorExt(tif->tif_clientdata, module, 
				     "Encoder error: %s",
				     SAFE_MSG(sp));
			return (0);
		}
		if (sp->stream.avail_out == 0) {
			tif->tif_rawcc = tif->tif_rawdatasize;
			if (!TIFFFlushData1(tif))
				return 0;
			sp->stream.next_out = tif->tif_rawdata;
			sp->stream.avail_out = (uint64)tif->tif_rawdatasize <= 0xFFFFFFFFU ? (uInt)tif->tif_rawdatasize : 0xFFFFFFFFU;
		}
		cc -= (avail_in_before - sp->stream.avail_in);
	} while (cc > 0);
	return (1);
}

/*
 * Finish off an encoded strip by flushing the last
 * string and tacking on an End Of Information code.
 */
static int
ZIPPostEncode(TIFF* tif)
{
	static const char module[] = "ZIPPostEncode";
	ZIPState *sp = EncoderState(tif);
	int state;

#if LIBDEFLATE_SUPPORT
        if( sp->libdeflate_state == 1 )
            return 1;
#endif

	sp->stream.avail_in = 0;
	do {
		state = deflate(&sp->stream, Z_FINISH);
		switch (state) {
		case Z_STREAM_END:
		case Z_OK:
			if ((tmsize_t)sp->stream.avail_out != tif->tif_rawdatasize)
			{
				tif->tif_rawcc =  tif->tif_rawdatasize - sp->stream.avail_out;
				if (!TIFFFlushData1(tif))
					return 0;
				sp->stream.next_out = tif->tif_rawdata;
				sp->stream.avail_out = (uint64)tif->tif_rawdatasize <= 0xFFFFFFFFU ? (uInt)tif->tif_rawdatasize : 0xFFFFFFFFU;
			}
			break;
		default:
			TIFFErrorExt(tif->tif_clientdata, module, 
				     "ZLib error: %s", SAFE_MSG(sp));
			return (0);
		}
	} while (state != Z_STREAM_END);
	return (1);
}

static void
ZIPCleanup(TIFF* tif)
{
	ZIPState* sp = ZState(tif);

	assert(sp != 0);

	(void)TIFFPredictorCleanup(tif);

	tif->tif_tagmethods.vgetfield = sp->vgetparent;
	tif->tif_tagmethods.vsetfield = sp->vsetparent;

	if (sp->state & ZSTATE_INIT_ENCODE) {
		deflateEnd(&sp->stream);
		sp->state = 0;
	} else if( sp->state & ZSTATE_INIT_DECODE) {
		inflateEnd(&sp->stream);
		sp->state = 0;
	}

#if LIBDEFLATE_SUPPORT
        if( sp->libdeflate_dec )
            libdeflate_free_decompressor(sp->libdeflate_dec);
        if( sp->libdeflate_enc )
            libdeflate_free_compressor(sp->libdeflate_enc);
#endif

	_TIFFfree(sp);
	tif->tif_data = NULL;

	_TIFFSetDefaultCompressionState(tif);
}

static int
ZIPVSetField(TIFF* tif, uint32 tag, va_list ap)
{
	static const char module[] = "ZIPVSetField";
	ZIPState* sp = ZState(tif);

	switch (tag) {
	case TIFFTAG_ZIPQUALITY:
		sp->zipquality = (int) va_arg(ap, int);
                if( sp->zipquality < Z_DEFAULT_COMPRESSION ||
                    sp->zipquality > LIBDEFLATE_MAX_COMPRESSION_LEVEL ) {
                    TIFFErrorExt(tif->tif_clientdata, module,
                                 "Invalid ZipQuality value. Should be in [-1,%d] range",
                                 LIBDEFLATE_MAX_COMPRESSION_LEVEL);
                    return 0;
                }

                if ( sp->state&ZSTATE_INIT_ENCODE ) {
                        int cappedQuality = sp->zipquality;
                        if( cappedQuality > Z_BEST_COMPRESSION )
                            cappedQuality = Z_BEST_COMPRESSION;
			if (deflateParams(&sp->stream,
			    cappedQuality, Z_DEFAULT_STRATEGY) != Z_OK) {
				TIFFErrorExt(tif->tif_clientdata, module, "ZLib error: %s",
					     SAFE_MSG(sp));
				return (0);
			}
		}

#if LIBDEFLATE_SUPPORT
                if( sp->libdeflate_enc )
                {
                    libdeflate_free_compressor(sp->libdeflate_enc);
                    sp->libdeflate_enc = NULL;
                }
#endif

		return (1);

        case TIFFTAG_DEFLATE_SUBCODEC:
                sp->subcodec = (int) va_arg(ap, int);
                if( sp->subcodec != DEFLATE_SUBCODEC_ZLIB &&
                    sp->subcodec != DEFLATE_SUBCODEC_LIBDEFLATE )
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                                 "Invalid DeflateCodec value.");
                    return 0;
                }
#if !LIBDEFLATE_SUPPORT
                if( sp->subcodec == DEFLATE_SUBCODEC_LIBDEFLATE )
                {
                    TIFFErrorExt(tif->tif_clientdata, module,
                                 "DeflateCodec = DEFLATE_SUBCODEC_LIBDEFLATE unsupported in this build");
                    return 0;
                }
#endif
                return 1;

	default:
		return (*sp->vsetparent)(tif, tag, ap);
	}
	/*NOTREACHED*/
}

static int
ZIPVGetField(TIFF* tif, uint32 tag, va_list ap)
{
	ZIPState* sp = ZState(tif);

	switch (tag) {
	case TIFFTAG_ZIPQUALITY:
		*va_arg(ap, int*) = sp->zipquality;
		break;

        case TIFFTAG_DEFLATE_SUBCODEC:
		*va_arg(ap, int*) = sp->subcodec;
		break;

	default:
		return (*sp->vgetparent)(tif, tag, ap);
	}
	return (1);
}

static const TIFFField zipFields[] = {
    { TIFFTAG_ZIPQUALITY, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT, TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE, "", NULL },
    { TIFFTAG_DEFLATE_SUBCODEC, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT, TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE, "", NULL },
};

int
TIFFInitZIP(TIFF* tif, int scheme)
{
	static const char module[] = "TIFFInitZIP";
	ZIPState* sp;

	assert( (scheme == COMPRESSION_DEFLATE)
		|| (scheme == COMPRESSION_ADOBE_DEFLATE));
#ifdef NDEBUG
	(void)scheme;
#endif

	/*
	 * Merge codec-specific tag information.
	 */
	if (!_TIFFMergeFields(tif, zipFields, TIFFArrayCount(zipFields))) {
		TIFFErrorExt(tif->tif_clientdata, module,
			     "Merging Deflate codec-specific tags failed");
		return 0;
	}

	/*
	 * Allocate state block so tag methods have storage to record values.
	 */
	tif->tif_data = (uint8*) _TIFFcalloc(sizeof (ZIPState), 1);
	if (tif->tif_data == NULL)
		goto bad;
	sp = ZState(tif);
	sp->stream.zalloc = NULL;
	sp->stream.zfree = NULL;
	sp->stream.opaque = NULL;
	sp->stream.data_type = Z_BINARY;

	/*
	 * Override parent get/set field methods.
	 */
	sp->vgetparent = tif->tif_tagmethods.vgetfield;
	tif->tif_tagmethods.vgetfield = ZIPVGetField; /* hook for codec tags */
	sp->vsetparent = tif->tif_tagmethods.vsetfield;
	tif->tif_tagmethods.vsetfield = ZIPVSetField; /* hook for codec tags */

	/* Default values for codec-specific fields */
	sp->zipquality = Z_DEFAULT_COMPRESSION;	/* default comp. level */
	sp->state = 0;
#if LIBDEFLATE_SUPPORT
        sp->subcodec = DEFLATE_SUBCODEC_LIBDEFLATE;
#else
        sp->subcodec = DEFLATE_SUBCODEC_ZLIB;
#endif

	/*
	 * Install codec methods.
	 */
	tif->tif_fixuptags = ZIPFixupTags; 
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
	TIFFErrorExt(tif->tif_clientdata, module,
		     "No space for ZIP state block");
	return (0);
}
#endif /* ZIP_SUPPORT */

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
