//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_IMAGE_IO_H
#define INCLUDED_IMF_IMAGE_IO_H

//----------------------------------------------------------------------------
//
//      Functions to load flat or deep images from OpenEXR files
//      and to save flat or deep images in OpenEXR files.
//
//----------------------------------------------------------------------------

#include "ImfUtilExport.h"

#include "ImfImage.h"
#include "ImfImageDataWindow.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// saveImage (n, h, i, d) or
// saveImage (n, i)
//
//      Saves image i in an OpenEXR file with name n.  The file will be
//      tiled if the image has more than one level, or if a header, h, is
//      given and contains a tile description attribute; otherwise the
//      file will be scan-line based.  The file will be deep if the image
//      is deep; otherwise the file will be flat.
//
//      If header h is given, then the channel list in h is replaced with
//      the channel list in i, and the levelMode and the levelRounding mode
//      fields of the tile description are replaced with the level mode
//      and the levelRounding mode of i.  In addition, if the data window
//      source flag, d, is set to USE_IMAGE_DATA_WINDOW, then the data
//      window in the image is copied into the header; if d is set to
//      USE_HEADER_DATA_WINDOW, then the data window in the header is
//      replaced with the intersection of the original data window in the
//      header and the data window in the image.  The modified header then
//      becomes the header of the image file.
//
//      Note: USE_HEADER_DATA_WINDOW can only be used for images with
//      level mode ONE_LEVEL.
//

IMFUTIL_EXPORT
void saveImage (
    const std::string& fileName,
    const Header&      hdr,
    const Image&       img,
    DataWindowSource   dws = USE_IMAGE_DATA_WINDOW);

IMFUTIL_EXPORT
void saveImage (const std::string& fileName, const Image& img);

//
// loadImage (n, h) or
// loadImage (n)
//
//      Loads deep an image from the OpenEXR file with name n, and returns
//      a pointer to the image.  The caller owns the image and is responsible
//      for deleting it.
//
//      If header h is given, then the header of the file is copied into h.
//

IMFUTIL_EXPORT
Image* loadImage (const std::string& fileName, Header& hdr);

IMFUTIL_EXPORT
Image* loadImage (const std::string& fileName);

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
