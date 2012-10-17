/* $Id: tif_compress.c,v 1.22 2010-03-10 18:56:48 bfriesen Exp $ */

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
 * Compression Scheme Configuration Support.
 */
#include "tiffiop.h"

static int
TIFFNoEncode(TIFF* tif, const char* method)
{
    const TIFFCodec* c = TIFFFindCODEC(tif->tif_dir.td_compression);

    if (c) {
        TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
                 "%s %s encoding is not implemented",
                 c->name, method);
    } else {
        TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
            "Compression scheme %u %s encoding is not implemented",
                 tif->tif_dir.td_compression, method);
    }
    return (-1);
}

int
_TIFFNoRowEncode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s)
{
    (void) pp; (void) cc; (void) s;
    return (TIFFNoEncode(tif, "scanline"));
}

int
_TIFFNoStripEncode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s)
{
    (void) pp; (void) cc; (void) s;
    return (TIFFNoEncode(tif, "strip"));
}

int
_TIFFNoTileEncode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s)
{
    (void) pp; (void) cc; (void) s;
    return (TIFFNoEncode(tif, "tile"));
}

static int
TIFFNoDecode(TIFF* tif, const char* method)
{
    const TIFFCodec* c = TIFFFindCODEC(tif->tif_dir.td_compression);

    if (c)
        TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
                 "%s %s decoding is not implemented",
                 c->name, method);
    else
        TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
                 "Compression scheme %u %s decoding is not implemented",
                 tif->tif_dir.td_compression, method);
    return (-1);
}

int
_TIFFNoFixupTags(TIFF* tif)
{
    (void) tif;
    return (1);
}

int
_TIFFNoRowDecode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s)
{
    (void) pp; (void) cc; (void) s;
    return (TIFFNoDecode(tif, "scanline"));
}

int
_TIFFNoStripDecode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s)
{
    (void) pp; (void) cc; (void) s;
    return (TIFFNoDecode(tif, "strip"));
}

int
_TIFFNoTileDecode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s)
{
    (void) pp; (void) cc; (void) s;
    return (TIFFNoDecode(tif, "tile"));
}

int
_TIFFNoSeek(TIFF* tif, uint32 off)
{
    (void) off;
    TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
             "Compression algorithm does not support random access");
    return (0);
}

int
_TIFFNoPreCode(TIFF* tif, uint16 s)
{
    (void) tif; (void) s;
    return (1);
}

static int _TIFFtrue(TIFF* tif) { (void) tif; return (1); }
static void _TIFFvoid(TIFF* tif) { (void) tif; }

void
_TIFFSetDefaultCompressionState(TIFF* tif)
{
    tif->tif_fixuptags = _TIFFNoFixupTags;
    tif->tif_decodestatus = TRUE;
    tif->tif_setupdecode = _TIFFtrue;
    tif->tif_predecode = _TIFFNoPreCode;
    tif->tif_decoderow = _TIFFNoRowDecode;
    tif->tif_decodestrip = _TIFFNoStripDecode;
    tif->tif_decodetile = _TIFFNoTileDecode;
    tif->tif_encodestatus = TRUE;
    tif->tif_setupencode = _TIFFtrue;
    tif->tif_preencode = _TIFFNoPreCode;
    tif->tif_postencode = _TIFFtrue;
    tif->tif_encoderow = _TIFFNoRowEncode;
    tif->tif_encodestrip = _TIFFNoStripEncode;
    tif->tif_encodetile = _TIFFNoTileEncode;
    tif->tif_close = _TIFFvoid;
    tif->tif_seek = _TIFFNoSeek;
    tif->tif_cleanup = _TIFFvoid;
    tif->tif_defstripsize = _TIFFDefaultStripSize;
    tif->tif_deftilesize = _TIFFDefaultTileSize;
    tif->tif_flags &= ~(TIFF_NOBITREV|TIFF_NOREADRAW);
}

int
TIFFSetCompressionScheme(TIFF* tif, int scheme)
{
    const TIFFCodec *c = TIFFFindCODEC((uint16) scheme);

    _TIFFSetDefaultCompressionState(tif);
    /*
     * Don't treat an unknown compression scheme as an error.
     * This permits applications to open files with data that
     * the library does not have builtin support for, but which
     * may still be meaningful.
     */
    return (c ? (*c->init)(tif, scheme) : 1);
}

/*
 * Other compression schemes may be registered.  Registered
 * schemes can also override the builtin versions provided
 * by this library.
 */
typedef struct _codec {
    struct _codec* next;
    TIFFCodec* info;
} codec_t;
static codec_t* registeredCODECS = NULL;

const TIFFCodec*
TIFFFindCODEC(uint16 scheme)
{
    const TIFFCodec* c;
    codec_t* cd;

    for (cd = registeredCODECS; cd; cd = cd->next)
        if (cd->info->scheme == scheme)
            return ((const TIFFCodec*) cd->info);
    for (c = _TIFFBuiltinCODECS; c->name; c++)
        if (c->scheme == scheme)
            return (c);
    return ((const TIFFCodec*) 0);
}

TIFFCodec*
TIFFRegisterCODEC(uint16 scheme, const char* name, TIFFInitMethod init)
{
    codec_t* cd = (codec_t*)
        _TIFFmalloc((tmsize_t)(sizeof (codec_t) + sizeof (TIFFCodec) + strlen(name)+1));

    if (cd != NULL) {
        cd->info = (TIFFCodec*) ((uint8*) cd + sizeof (codec_t));
        cd->info->name = (char*)
            ((uint8*) cd->info + sizeof (TIFFCodec));
        strcpy(cd->info->name, name);
        cd->info->scheme = scheme;
        cd->info->init = init;
        cd->next = registeredCODECS;
        registeredCODECS = cd;
    } else {
        TIFFErrorExt(0, "TIFFRegisterCODEC",
            "No space to register compression scheme %s", name);
        return NULL;
    }
    return (cd->info);
}

void
TIFFUnRegisterCODEC(TIFFCodec* c)
{
    codec_t* cd;
    codec_t** pcd;

    for (pcd = &registeredCODECS; (cd = *pcd); pcd = &cd->next)
        if (cd->info == c) {
            *pcd = cd->next;
            _TIFFfree(cd);
            return;
        }
    TIFFErrorExt(0, "TIFFUnRegisterCODEC",
        "Cannot remove compression scheme %s; not registered", c->name);
}

/************************************************************************/
/*                       TIFFGetConfisuredCODECs()                      */
/************************************************************************/

/**
 * Get list of configured codecs, both built-in and registered by user.
 * Caller is responsible to free this structure.
 *
 * @return returns array of TIFFCodec records (the last record should be NULL)
 * or NULL if function failed.
 */

TIFFCodec*
TIFFGetConfiguredCODECs()
{
    int i = 1;
    codec_t *cd;
    const TIFFCodec* c;
    TIFFCodec* codecs = NULL;
    TIFFCodec* new_codecs;

    for (cd = registeredCODECS; cd; cd = cd->next) {
        new_codecs = (TIFFCodec *)
            _TIFFrealloc(codecs, i * sizeof(TIFFCodec));
        if (!new_codecs) {
            _TIFFfree (codecs);
            return NULL;
        }
        codecs = new_codecs;
        _TIFFmemcpy(codecs + i - 1, cd, sizeof(TIFFCodec));
        i++;
    }
    for (c = _TIFFBuiltinCODECS; c->name; c++) {
        if (TIFFIsCODECConfigured(c->scheme)) {
            new_codecs = (TIFFCodec *)
                _TIFFrealloc(codecs, i * sizeof(TIFFCodec));
            if (!new_codecs) {
                _TIFFfree (codecs);
                return NULL;
            }
            codecs = new_codecs;
            _TIFFmemcpy(codecs + i - 1, (const void*)c, sizeof(TIFFCodec));
            i++;
        }
    }

    new_codecs = (TIFFCodec *) _TIFFrealloc(codecs, i * sizeof(TIFFCodec));
    if (!new_codecs) {
        _TIFFfree (codecs);
        return NULL;
    }
    codecs = new_codecs;
    _TIFFmemset(codecs + i - 1, 0, sizeof(TIFFCodec));

    return codecs;
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
