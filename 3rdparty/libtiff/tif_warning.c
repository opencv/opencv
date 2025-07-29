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

TIFFErrorHandlerExt _TIFFwarningHandlerExt = NULL;

TIFFErrorHandler TIFFSetWarningHandler(TIFFErrorHandler handler)
{
    TIFFErrorHandler prev = _TIFFwarningHandler;
    _TIFFwarningHandler = handler;
    return (prev);
}

TIFFErrorHandlerExt TIFFSetWarningHandlerExt(TIFFErrorHandlerExt handler)
{
    TIFFErrorHandlerExt prev = _TIFFwarningHandlerExt;
    _TIFFwarningHandlerExt = handler;
    return (prev);
}

void TIFFWarning(const char *module, const char *fmt, ...)
{
    va_list ap;
    if (_TIFFwarningHandler)
    {
        va_start(ap, fmt);
        (*_TIFFwarningHandler)(module, fmt, ap);
        va_end(ap);
    }
    if (_TIFFwarningHandlerExt)
    {
        va_start(ap, fmt);
        (*_TIFFwarningHandlerExt)(0, module, fmt, ap);
        va_end(ap);
    }
}

void TIFFWarningExt(thandle_t fd, const char *module, const char *fmt, ...)
{
    va_list ap;
    if (_TIFFwarningHandler)
    {
        va_start(ap, fmt);
        (*_TIFFwarningHandler)(module, fmt, ap);
        va_end(ap);
    }
    if (_TIFFwarningHandlerExt)
    {
        va_start(ap, fmt);
        (*_TIFFwarningHandlerExt)(fd, module, fmt, ap);
        va_end(ap);
    }
}

void TIFFWarningExtR(TIFF *tif, const char *module, const char *fmt, ...)
{
    va_list ap;
    if (tif && tif->tif_warnhandler)
    {
        va_start(ap, fmt);
        int stop = (*tif->tif_warnhandler)(tif, tif->tif_warnhandler_user_data,
                                           module, fmt, ap);
        va_end(ap);
        if (stop)
            return;
    }
    if (_TIFFwarningHandler)
    {
        va_start(ap, fmt);
        (*_TIFFwarningHandler)(module, fmt, ap);
        va_end(ap);
    }
    if (_TIFFwarningHandlerExt)
    {
        va_start(ap, fmt);
        (*_TIFFwarningHandlerExt)(tif ? tif->tif_clientdata : 0, module, fmt,
                                  ap);
        va_end(ap);
    }
}
