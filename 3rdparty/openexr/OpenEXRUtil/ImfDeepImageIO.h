//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_DEEP_IMAGE_IO_H
#define INCLUDED_IMF_DEEP_IMAGE_IO_H

//----------------------------------------------------------------------------
//
//      Functions to load deep images from OpenEXR files
//      and to save deep images in OpenEXR files.
//
//----------------------------------------------------------------------------

#include "ImfNamespace.h"
#include "ImfUtilExport.h"

#include "ImfDeepImage.h"
#include "ImfImageDataWindow.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// saveDeepImage (n, h, i,d) or
// saveDeepImage (n, i)
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
void saveDeepImage (
    const std::string& fileName,
    const Header&      hdr,
    const DeepImage&   img,
    DataWindowSource   dws = USE_IMAGE_DATA_WINDOW);

IMFUTIL_EXPORT
void saveDeepImage (const std::string& fileName, const DeepImage& img);

//
// loadDeepImage (n, h, i) or
// loadDeepImage (n, i)
//
//      Loads deep image i from the OpenEXR file with name n.
//
//      If header h is given, then the header of the file is copied into h.
//

IMFUTIL_EXPORT
void loadDeepImage (const std::string& fileName, Header& hdr, DeepImage& img);

IMFUTIL_EXPORT
void loadDeepImage (const std::string& fileName, DeepImage& img);

//
// saveDeepScanLineImage (n, h, i, d) or
// saveDeepScanLineImage (n, i)
//
//      Saves image i in a scan-line based deep OpenEXR file with file name n.
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
void saveDeepScanLineImage (
    const std::string& fileName,
    const Header&      hdr,
    const DeepImage&   img,
    DataWindowSource   dws = USE_IMAGE_DATA_WINDOW);

IMFUTIL_EXPORT
void saveDeepScanLineImage (const std::string& fileName, const DeepImage& img);

//
// loadDeepScanLineImage (n, h, i) or
// loadDeepScanLineImage (n, i)
//
//      Loads image i from a scan-line based deep OpenEXR file with file name n.
//      If header h is given, then the header of the file is copied into h.
//

IMFUTIL_EXPORT
void loadDeepScanLineImage (
    const std::string& fileName, Header& hdr, DeepImage& img);

IMFUTIL_EXPORT
void loadDeepScanLineImage (const std::string& fileName, DeepImage& img);

//
// saveDeepTiledImage (n, h, i, d) or
// saveDeepTiledImage (n, i)
//
//      Saves image i in a tiled deep OpenEXR file with file name n.
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
void saveDeepTiledImage (
    const std::string& fileName,
    const Header&      hdr,
    const DeepImage&   img,
    DataWindowSource   dws = USE_IMAGE_DATA_WINDOW);

IMFUTIL_EXPORT
void saveDeepTiledImage (const std::string& fileName, const DeepImage& img);

//
// loadDeepTiledImage (n, h, i) or
// loadDeepTiledImage (n, i)
//
//      Loads image i from a tiled deep OpenEXR file with file name n.
//      If header h is given, then the header of the file is copied into h.
//

IMFUTIL_EXPORT
void
loadDeepTiledImage (const std::string& fileName, Header& hdr, DeepImage& img);

IMFUTIL_EXPORT
void loadDeepTiledImage (const std::string& fileName, DeepImage& img);

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
