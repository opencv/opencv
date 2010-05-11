/* $Header: /home/vp/work/opencv-cvsbackup/opencv/3rdparty/libtiff/tif_extension.c,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

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
 * Various routines support external extension of the tag set, and other
 * application extension capabilities. 
 */

#include "tiffiop.h"

int TIFFGetTagListCount( TIFF *tif )

{
    TIFFDirectory* td = &tif->tif_dir;
    
    return td->td_customValueCount;
}

ttag_t TIFFGetTagListEntry( TIFF *tif, int tag_index )

{
    TIFFDirectory* td = &tif->tif_dir;

    if( tag_index < 0 || tag_index >= td->td_customValueCount )
        return (ttag_t) -1;
    else
        return td->td_customValues[tag_index].info->field_tag;
}

/*
** This provides read/write access to the TIFFTagMethods within the TIFF
** structure to application code without giving access to the private
** TIFF structure.
*/
TIFFTagMethods *TIFFAccessTagMethods( TIFF *tif )

{
    return &(tif->tif_tagmethods);
}

void *TIFFGetClientInfo( TIFF *tif, const char *name )

{
    TIFFClientInfoLink *link = tif->tif_clientinfo;

    while( link != NULL && strcmp(link->name,name) != 0 )
        link = link->next;

    if( link != NULL )
        return link->data;
    else
        return NULL;
}

void TIFFSetClientInfo( TIFF *tif, void *data, const char *name )

{
    TIFFClientInfoLink *link = tif->tif_clientinfo;

    /*
    ** Do we have an existing link with this name?  If so, just
    ** set it.
    */
    while( link != NULL && strcmp(link->name,name) != 0 )
        link = link->next;

    if( link != NULL )
    {
        link->data = data;
        return;
    }

    /*
    ** Create a new link.
    */

    link = (TIFFClientInfoLink *) _TIFFmalloc(sizeof(TIFFClientInfoLink));
    assert (link != NULL);
    link->next = tif->tif_clientinfo;
    link->name = (char *) _TIFFmalloc(strlen(name)+1);
    assert (link->name != NULL);
    strcpy(link->name, name);
    link->data = data;

    tif->tif_clientinfo = link;
}
