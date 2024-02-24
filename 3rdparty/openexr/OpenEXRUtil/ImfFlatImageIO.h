//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_FLAT_IMAGE_IO_H
#define INCLUDED_IMF_FLAT_IMAGE_IO_H

//----------------------------------------------------------------------------
//
//      Functions to load flat images from OpenEXR files
//      and to save flat images in OpenEXR files.
//
//----------------------------------------------------------------------------

#include "ImfFlatImage.h"
#include "ImfImageDataWindow.h"
#include "ImfUtilExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// saveFlatImage (n, h, i, d) or
// saveFlatImage (n, i)
//
//      Saves image i in an OpenEXR file with name n.  The file will be
//      tiled if the image has more than one level, or if a header, h, is
//      given and contains a tile description attribute; otherwise the
//      file will be scan-line based.
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

IMFUTIL_EXPORT
void saveFlatImage (
    const std::string& fileName,
    const Header&      hdr,
    const FlatImage&   img,
    DataWindowSource   dws = USE_IMAGE_DATA_WINDOW);

IMFUTIL_EXPORT
void saveFlatImage (const std::string& fileName, const FlatImage& img);

//
// loadFlatImage (n, h, i) or
// loadFlatImage (n, i)
//
//      Loads flat image i from the OpenEXR file with name n.
//
//      If header h is given, then the header of the file is copied into h.
//

IMFUTIL_EXPORT
void loadFlatImage (const std::string& fileName, Header& hdr, FlatImage& img);

IMFUTIL_EXPORT
void loadFlatImage (const std::string& fileName, FlatImage& img);

//
// saveFlatScanLineImage (n, h, i, d) or
// saveFlatScanLineImage (n, i)
//
//      Saves image i in a scan-line based flat OpenEXR file with file name n.
//
//      If header h is given, then the channel list in h is replaced with
//      the channel list in i.  In addition, if the data window source flag, d,
//      is set to USE_IMAGE_DATA_WINDOW, then the data window in the image is
//      copied into the header; if d is set to USE_HEADER_DATA_WINDOW, then
//      the data window in the header is replaced with the intersection of
//      the original data window in the header and the data window in the
//      image.  The modified header then becomes the header of the image file.
//

IMFUTIL_EXPORT
void saveFlatScanLineImage (
    const std::string& fileName,
    const Header&      hdr,
    const FlatImage&   img,
    DataWindowSource   dws = USE_IMAGE_DATA_WINDOW);

IMFUTIL_EXPORT
void saveFlatScanLineImage (const std::string& fileName, const FlatImage& img);

//
// loadFlatScanLineImage (n, h, i) or
// loadFlatScanLineImage (n, i)
//
//      Loads image i from a scan-line based flat OpenEXR file with file name n.
//      If header h is given, then the header of the file is copied into h.
//

IMFUTIL_EXPORT
void loadFlatScanLineImage (
    const std::string& fileName, Header& hdr, FlatImage& img);

IMFUTIL_EXPORT
void loadFlatScanLineImage (const std::string& fileName, FlatImage& img);

//
// saveFlatTiledImage (n, h, i, d) or
// saveFlatTiledImage (n, i)
//
//      Saves image i in a tiled flat OpenEXR file with file name n.
//
//      If header h is given, then the channel list in h is replaced with
//      the channel list i, and the levelMode and the levelRounding mode
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
void saveFlatTiledImage (
    const std::string& fileName,
    const Header&      hdr,
    const FlatImage&   img,
    DataWindowSource   dws = USE_IMAGE_DATA_WINDOW);

IMFUTIL_EXPORT
void saveFlatTiledImage (const std::string& fileName, const FlatImage& img);

//
// loadFlatTiledImage (n, h, i) or
// loadFlatTiledImage (n, i)
//
//      Loads image i from a tiled flat OpenEXR file with file name n.
//      If header h is given, then the header of the file is copied into h.
//

IMFUTIL_EXPORT
void
loadFlatTiledImage (const std::string& fileName, Header& hdr, FlatImage& img);

IMFUTIL_EXPORT
void loadFlatTiledImage (const std::string& fileName, FlatImage& img);

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
