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
 * JBIG Compression Algorithm Support.
 * Contributed by Lee Howard <faxguy@deanox.com>
 *
 */

#include "tiffiop.h"

#ifdef JBIG_SUPPORT
#include "jbig.h"

static int JBIGSetupDecode(TIFF *tif)
{
    if (TIFFNumberOfStrips(tif) != 1)
    {
        TIFFErrorExtR(tif, "JBIG",
                      "Multistrip images not supported in decoder");
        return 0;
    }

    return 1;
}

static int JBIGDecode(TIFF *tif, uint8_t *buffer, tmsize_t size, uint16_t s)
{
    struct jbg_dec_state decoder;
    int decodeStatus = 0;
    unsigned char *pImage = NULL;
    unsigned long decodedSize;
    (void)s;

    if (isFillOrder(tif, tif->tif_dir.td_fillorder))
    {
        TIFFReverseBits(tif->tif_rawcp, tif->tif_rawcc);
    }

    jbg_dec_init(&decoder);

#if defined(HAVE_JBG_NEWLEN)
    jbg_newlen(tif->tif_rawcp, (size_t)tif->tif_rawcc);
    /*
     * I do not check the return status of jbg_newlen because even if this
     * function fails it does not necessarily mean that decoding the image
     * will fail.  It is generally only needed for received fax images
     * that do not contain the actual length of the image in the BIE
     * header.  I do not log when an error occurs because that will cause
     * problems when converting JBIG encoded TIFF's to
     * PostScript.  As long as the actual image length is contained in the
     * BIE header jbg_dec_in should succeed.
     */
#endif /* HAVE_JBG_NEWLEN */

    decodeStatus = jbg_dec_in(&decoder, (unsigned char *)tif->tif_rawcp,
                              (size_t)tif->tif_rawcc, NULL);
    if (JBG_EOK != decodeStatus)
    {
        /*
         * XXX: JBG_EN constant was defined in pre-2.0 releases of the
         * JBIG-KIT. Since the 2.0 the error reporting functions were
         * changed. We will handle both cases here.
         */
        TIFFErrorExtR(tif, "JBIG", "Error (%d) decoding: %s", decodeStatus,
#if defined(JBG_EN)
                      jbg_strerror(decodeStatus, JBG_EN)
#else
                      jbg_strerror(decodeStatus)
#endif
        );
        jbg_dec_free(&decoder);
        return 0;
    }

    decodedSize = jbg_dec_getsize(&decoder);
    if ((tmsize_t)decodedSize < size)
    {
        TIFFWarningExtR(tif, "JBIG",
                        "Only decoded %lu bytes, whereas %" TIFF_SSIZE_FORMAT
                        " requested",
                        decodedSize, size);
    }
    else if ((tmsize_t)decodedSize > size)
    {
        TIFFErrorExtR(tif, "JBIG",
                      "Decoded %lu bytes, whereas %" TIFF_SSIZE_FORMAT
                      " were requested",
                      decodedSize, size);
        jbg_dec_free(&decoder);
        return 0;
    }
    pImage = jbg_dec_getimage(&decoder, 0);
    _TIFFmemcpy(buffer, pImage, decodedSize);
    jbg_dec_free(&decoder);

    tif->tif_rawcp += tif->tif_rawcc;
    tif->tif_rawcc = 0;

    return 1;
}

static int JBIGSetupEncode(TIFF *tif)
{
    if (TIFFNumberOfStrips(tif) != 1)
    {
        TIFFErrorExtR(tif, "JBIG",
                      "Multistrip images not supported in encoder");
        return 0;
    }

    return 1;
}

static int JBIGCopyEncodedData(TIFF *tif, unsigned char *pp, size_t cc,
                               uint16_t s)
{
    (void)s;
    while (cc > 0)
    {
        tmsize_t n = (tmsize_t)cc;

        if (tif->tif_rawcc + n > tif->tif_rawdatasize)
        {
            n = tif->tif_rawdatasize - tif->tif_rawcc;
        }

        assert(n > 0);
        _TIFFmemcpy(tif->tif_rawcp, pp, n);
        tif->tif_rawcp += n;
        tif->tif_rawcc += n;
        pp += n;
        cc -= (size_t)n;
        if (tif->tif_rawcc >= tif->tif_rawdatasize && !TIFFFlushData1(tif))
        {
            return (-1);
        }
    }

    return (1);
}

static void JBIGOutputBie(unsigned char *buffer, size_t len, void *userData)
{
    TIFF *tif = (TIFF *)userData;

    if (isFillOrder(tif, tif->tif_dir.td_fillorder))
    {
        TIFFReverseBits(buffer, (tmsize_t)len);
    }

    JBIGCopyEncodedData(tif, buffer, len, 0);
}

static int JBIGEncode(TIFF *tif, uint8_t *buffer, tmsize_t size, uint16_t s)
{
    TIFFDirectory *dir = &tif->tif_dir;
    struct jbg_enc_state encoder;

    (void)size, (void)s;

    jbg_enc_init(&encoder, dir->td_imagewidth, dir->td_imagelength, 1, &buffer,
                 JBIGOutputBie, tif);
    /*
     * jbg_enc_out does the "real" encoding.  As data is encoded,
     * JBIGOutputBie is called, which writes the data to the directory.
     */
    jbg_enc_out(&encoder);
    jbg_enc_free(&encoder);

    return 1;
}

int TIFFInitJBIG(TIFF *tif, int scheme)
{
    (void)scheme;
    assert(scheme == COMPRESSION_JBIG);

    /*
     * These flags are set so the JBIG Codec can control when to reverse
     * bits and when not to and to allow the jbig decoder and bit reverser
     * to write to memory when necessary.
     */
    tif->tif_flags |= TIFF_NOBITREV;
    tif->tif_flags &= ~TIFF_MAPPED;
    /* We may have read from a previous IFD and thus set TIFF_BUFFERMMAP and
     * cleared TIFF_MYBUFFER. It is necessary to restore them to their initial
     * value to be consistent with the state of a non-memory mapped file.
     */
    if (tif->tif_flags & TIFF_BUFFERMMAP)
    {
        tif->tif_rawdata = NULL;
        tif->tif_rawdatasize = 0;
        tif->tif_flags &= ~TIFF_BUFFERMMAP;
        tif->tif_flags |= TIFF_MYBUFFER;
    }

    /* Setup the function pointers for encode, decode, and cleanup. */
    tif->tif_setupdecode = JBIGSetupDecode;
    tif->tif_decodestrip = JBIGDecode;

    tif->tif_setupencode = JBIGSetupEncode;
    tif->tif_encodestrip = JBIGEncode;

    return 1;
}

#endif /* JBIG_SUPPORT */
