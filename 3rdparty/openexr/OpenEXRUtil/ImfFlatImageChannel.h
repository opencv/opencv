//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_FLAT_IMAGE_CHANNEL_H
#define INCLUDED_IMF_FLAT_IMAGE_CHANNEL_H

//----------------------------------------------------------------------------
//
//      class FlatImageChannel,
//      template class TypedFlatImageChannel<T>
//
//      For an explanation of images, levels and channels,
//      see the comments in header file Image.h.
//
//----------------------------------------------------------------------------

#include "ImfImageChannel.h"
#include "ImfImageLevel.h"
#include "ImfUtilExport.h"

#include "ImfFrameBuffer.h"
#include "ImfPixelType.h"
#include <ImathBox.h>
#include <half.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class FlatImageLevel;

//
// Image channels:
//
// A TypedFlatImageChannel<T> holds the pixel data for a single channel
// of one level of a flat image.  The pixels in the channel are of type T,
// where T is either half, float or unsigned int.  Storage is allocated
// only for pixels within the data window of the level.
//

class IMFUTIL_EXPORT_TYPE FlatImageChannel : public ImageChannel
{
public:
    //
    // Construct an OpenEXR frame buffer slice for this channel.
    // This function is needed reading an image from an OpenEXR
    // file and for saving an image in an OpenEXR file.
    //

    virtual Slice slice () const = 0;

    //
    // Access to the flat image level to which this channel belongs.
    //

    IMFUTIL_EXPORT FlatImageLevel&       flatLevel ();
    IMFUTIL_EXPORT const FlatImageLevel& flatLevel () const;

protected:
    friend class FlatImageLevel;

    IMFUTIL_EXPORT
    FlatImageChannel (
        FlatImageLevel& level, int xSampling, int ySampling, bool pLinear);

    IMFUTIL_EXPORT virtual ~FlatImageChannel ();

    FlatImageChannel (const FlatImageChannel& other) = delete;
    FlatImageChannel& operator= (const FlatImageChannel& other) = delete;
    FlatImageChannel (FlatImageChannel&& other)                 = delete;
    FlatImageChannel& operator= (FlatImageChannel&& other) = delete;

    IMFUTIL_EXPORT
    virtual void resize ();

    virtual void resetBasePointer () = 0;
};

template <class T>
class IMFUTIL_EXPORT_TEMPLATE_TYPE TypedFlatImageChannel
    : public FlatImageChannel
{
public:
    //
    // The OpenEXR pixel type of this channel (HALF, FLOAT or UINT).
    //

    virtual PixelType pixelType () const;

    //
    // Construct an OpenEXR frame buffer slice for this channel.
    //

    virtual Slice slice () const;

    //
    // Access to the pixel at pixel space location (x, y), without
    // bounds checking.  Accessing a location outside the data window
    // of the image level results in undefined behavior.
    //

    T&       operator() (int x, int y);
    const T& operator() (int x, int y) const;

    //
    // Access to the pixel at pixel space location (x, y), with bounds
    // checking.  Accessing a location outside the data window of the
    // image level throws an Iex::ArgExc exception.
    //

    T&       at (int x, int y);
    const T& at (int x, int y) const;

    //
    // Faster access to all pixels in a single horizontal row of the
    // channel.  Rows are numbered from 0 to pixelsPerColumn()-1, and
    // each row contains pixelsPerRow() values.
    // Access is not bounds checked; accessing out of bounds rows or
    // pixels results in undefined behavior.
    //

    T*       row (int r);
    const T* row (int r) const;

private:
    friend class FlatImageLevel;

    //
    // The constructor and destructor are not public because flat
    // image channels exist only as parts of a flat image level.
    //

    IMFUTIL_HIDDEN
    TypedFlatImageChannel (
        FlatImageLevel& level, int xSampling, int ySampling, bool pLinear);

    IMFUTIL_HIDDEN
    virtual ~TypedFlatImageChannel ();

    TypedFlatImageChannel (const TypedFlatImageChannel& other) = delete;
    TypedFlatImageChannel&
    operator= (const TypedFlatImageChannel& other)        = delete;
    TypedFlatImageChannel (TypedFlatImageChannel&& other) = delete;
    TypedFlatImageChannel& operator= (TypedFlatImageChannel&& other) = delete;

    IMFUTIL_HIDDEN
    virtual void resize ();

    IMFUTIL_HIDDEN
    virtual void resetBasePointer ();

    T* _pixels; // Pointer to allocated storage
    T* _base;   // Base pointer for faster pixel access
};

//
// Channel typedefs for the pixel data types supported by OpenEXR.
//

typedef TypedFlatImageChannel<half>         FlatHalfChannel;
typedef TypedFlatImageChannel<float>        FlatFloatChannel;
typedef TypedFlatImageChannel<unsigned int> FlatUIntChannel;

//-----------------------------------------------------------------------------
// Implementation of templates and inline functions
//-----------------------------------------------------------------------------

template <class T>
inline T&
TypedFlatImageChannel<T>::operator() (int x, int y)
{
    return _base[(y / ySampling ()) * pixelsPerRow () + (x / xSampling ())];
}

template <class T>
inline const T&
TypedFlatImageChannel<T>::operator() (int x, int y) const
{
    return _base[(y / ySampling ()) * pixelsPerRow () + (x / xSampling ())];
}

template <class T>
inline T&
TypedFlatImageChannel<T>::at (int x, int y)
{
    boundsCheck (x, y);
    return _base[(y / ySampling ()) * pixelsPerRow () + (x / xSampling ())];
}

template <class T>
inline const T&
TypedFlatImageChannel<T>::at (int x, int y) const
{
    boundsCheck (x, y);
    return _base[(y / ySampling ()) * pixelsPerRow () + (x / xSampling ())];
}

template <class T>
inline T*
TypedFlatImageChannel<T>::row (int r)
{
    return _base + r * pixelsPerRow ();
}

template <class T>
inline const T*
TypedFlatImageChannel<T>::row (int n) const
{
    return _base + n * pixelsPerRow ();
}

#ifndef COMPILING_IMF_FLAT_IMAGE_CHANNEL
extern template class IMFUTIL_EXPORT_EXTERN_TEMPLATE
    TypedFlatImageChannel<half>;
extern template class IMFUTIL_EXPORT_EXTERN_TEMPLATE
    TypedFlatImageChannel<float>;
extern template class IMFUTIL_EXPORT_EXTERN_TEMPLATE
    TypedFlatImageChannel<unsigned int>;
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
