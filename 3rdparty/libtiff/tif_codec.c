/* $Id: tif_codec.c,v 1.15 2010-12-14 12:53:00 dron Exp $ */

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
 * TIFF Library
 *
 * Builtin Compression Scheme Configuration Support.
 */
#include "tiffiop.h"

static int NotConfigured(TIFF*, int);

#ifndef LZW_SUPPORT
#define TIFFInitLZW NotConfigured
#endif
#ifndef PACKBITS_SUPPORT
#define TIFFInitPackBits NotConfigured
#endif
#ifndef THUNDER_SUPPORT
#define TIFFInitThunderScan NotConfigured
#endif
#ifndef NEXT_SUPPORT
#define TIFFInitNeXT NotConfigured
#endif
#ifndef JPEG_SUPPORT
#define TIFFInitJPEG NotConfigured
#endif
#ifndef OJPEG_SUPPORT
#define TIFFInitOJPEG NotConfigured
#endif
#ifndef CCITT_SUPPORT
#define TIFFInitCCITTRLE NotConfigured
#define TIFFInitCCITTRLEW NotConfigured
#define TIFFInitCCITTFax3 NotConfigured
#define TIFFInitCCITTFax4 NotConfigured
#endif
#ifndef JBIG_SUPPORT
#define TIFFInitJBIG NotConfigured
#endif
#ifndef ZIP_SUPPORT
#define TIFFInitZIP NotConfigured
#endif
#ifndef PIXARLOG_SUPPORT
#define TIFFInitPixarLog NotConfigured
#endif
#ifndef LOGLUV_SUPPORT
#define TIFFInitSGILog NotConfigured
#endif
#ifndef LZMA_SUPPORT
#define TIFFInitLZMA NotConfigured
#endif

/*
 * Compression schemes statically built into the library.
 */
#ifdef VMS
const TIFFCodec _TIFFBuiltinCODECS[] = {
#else
TIFFCodec _TIFFBuiltinCODECS[] = {
#endif
    { "None",		COMPRESSION_NONE,	TIFFInitDumpMode },
    { "LZW",		COMPRESSION_LZW,	TIFFInitLZW },
    { "PackBits",	COMPRESSION_PACKBITS,	TIFFInitPackBits },
    { "ThunderScan",	COMPRESSION_THUNDERSCAN,TIFFInitThunderScan },
    { "NeXT",		COMPRESSION_NEXT,	TIFFInitNeXT },
    { "JPEG",		COMPRESSION_JPEG,	TIFFInitJPEG },
    { "Old-style JPEG",	COMPRESSION_OJPEG,	TIFFInitOJPEG },
    { "CCITT RLE",	COMPRESSION_CCITTRLE,	TIFFInitCCITTRLE },
    { "CCITT RLE/W",	COMPRESSION_CCITTRLEW,	TIFFInitCCITTRLEW },
    { "CCITT Group 3",	COMPRESSION_CCITTFAX3,	TIFFInitCCITTFax3 },
    { "CCITT Group 4",	COMPRESSION_CCITTFAX4,	TIFFInitCCITTFax4 },
    { "ISO JBIG",	COMPRESSION_JBIG,	TIFFInitJBIG },
    { "Deflate",	COMPRESSION_DEFLATE,	TIFFInitZIP },
    { "AdobeDeflate",   COMPRESSION_ADOBE_DEFLATE , TIFFInitZIP },
    { "PixarLog",	COMPRESSION_PIXARLOG,	TIFFInitPixarLog },
    { "SGILog",		COMPRESSION_SGILOG,	TIFFInitSGILog },
    { "SGILog24",	COMPRESSION_SGILOG24,	TIFFInitSGILog },
    { "LZMA",		COMPRESSION_LZMA,	TIFFInitLZMA },
    { NULL,             0,                      NULL }
};

static int
_notConfigured(TIFF* tif)
{
    const TIFFCodec* c = TIFFFindCODEC(tif->tif_dir.td_compression);
        char compression_code[20];

        sprintf( compression_code, "%d", tif->tif_dir.td_compression );
    TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
                     "%s compression support is not configured",
                     c ? c->name : compression_code );
    return (0);
}

static int
NotConfigured(TIFF* tif, int scheme)
{
    (void) scheme;

    tif->tif_fixuptags = _notConfigured;
    tif->tif_decodestatus = FALSE;
    tif->tif_setupdecode = _notConfigured;
    tif->tif_encodestatus = FALSE;
    tif->tif_setupencode = _notConfigured;
    return (1);
}

/************************************************************************/
/*                       TIFFIsCODECConfigured()                        */
/************************************************************************/

/**
 * Check whether we have working codec for the specific coding scheme.
 *
 * @return returns 1 if the codec is configured and working. Otherwise
 * 0 will be returned.
 */

int
TIFFIsCODECConfigured(uint16 scheme)
{
    const TIFFCodec* codec = TIFFFindCODEC(scheme);

    if(codec == NULL) {
        return 0;
    }
    if(codec->init == NULL) {
        return 0;
    }
    if(codec->init != NotConfigured){
        return 1;
    }
    return 0;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
