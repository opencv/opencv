//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_IMAGE_LEVEL_H
#define INCLUDED_IMF_IMAGE_LEVEL_H

//----------------------------------------------------------------------------
//
//      class ImageLevel
//
//      For an explanation of images, levels and channels,
//      see the comments in header file Image.h.
//
//----------------------------------------------------------------------------

#include "ImfImageChannel.h"
#include "ImfImageChannelRenaming.h"
#include "ImfUtilExport.h"
#include <ImathBox.h>
#include <string>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class Image;

class IMFUTIL_EXPORT_TYPE ImageLevel
{
public:
    //
    // Access to the image to which the level belongs.
    //

    Image&       image () { return _image; }
    const Image& image () const { return _image; }

    //
    // Access to the level number and the data window of this level.
    //

    int xLevelNumber () const { return _xLevelNumber; }
    int yLevelNumber () const { return _yLevelNumber; }

    const IMATH_NAMESPACE::Box2i& dataWindow () const { return _dataWindow; }

protected:
    friend class Image;

    IMFUTIL_EXPORT
    ImageLevel (Image& image, int xLevelNumber, int yLevelNumber);

    IMFUTIL_EXPORT
    virtual ~ImageLevel ();

    IMFUTIL_EXPORT
    virtual void resize (const IMATH_NAMESPACE::Box2i& dataWindow);

    IMFUTIL_EXPORT
    virtual void shiftPixels (int dx, int dy);

    virtual void insertChannel (
        const std::string& name,
        PixelType          type,
        int                xSampling,
        int                ySampling,
        bool               pLinear) = 0;

    virtual void eraseChannel (const std::string& name) = 0;

    virtual void clearChannels () = 0;

    virtual void
    renameChannel (const std::string& oldName, const std::string& newName) = 0;

    virtual void renameChannels (const RenamingMap& oldToNewNames) = 0;

    IMFUTIL_EXPORT
    void throwChannelExists (const std::string& name) const;
    IMFUTIL_EXPORT
    void throwBadChannelName (const std::string& name) const;
    IMFUTIL_EXPORT
    void throwBadChannelNameOrType (const std::string& name) const;

private:
    ImageLevel (const ImageLevel&);            // not implemented
    ImageLevel& operator= (const ImageLevel&); // not implemented

    Image&                 _image;
    int                    _xLevelNumber;
    int                    _yLevelNumber;
    IMATH_NAMESPACE::Box2i _dataWindow;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
