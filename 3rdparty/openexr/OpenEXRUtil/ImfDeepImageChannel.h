//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_DEEP_IMAGE_CHANNEL_H
#define INCLUDED_IMF_DEEP_IMAGE_CHANNEL_H

//----------------------------------------------------------------------------
//
//      class DeepImageChannel,
//      template class TypedDeepImageChannel<T>
//
//      For an explanation of images, levels and channels,
//      see the comments in header file Image.h.
//
//----------------------------------------------------------------------------

#include "ImfNamespace.h"
#include "ImfUtilExport.h"

#include "ImfImageChannel.h"
#include "ImfImageLevel.h"
#include "ImfSampleCountChannel.h"

#include "ImfDeepFrameBuffer.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class DeepImageLevel;
class SampleCountChannel;

//
// Image channels:
//
// A TypedDeepImageChannel<T> holds the pixel data for a single channel
// of one level of a deep image.  Each pixel in the channel contains an
// array of n samples of type T, where T is either half, float or
// unsigned int, and n is stored in a separate sample count channel.
// Sample storage is allocated only for pixels within the data window
// of the level.
//

class IMFUTIL_EXPORT_TYPE DeepImageChannel : public ImageChannel
{
public:
    //
    // Construct an OpenEXR frame buffer slice for this channel.
    // This function is needed reading an image from an OpenEXR
    // file and for saving an image in an OpenEXR file.
    //

    virtual DeepSlice slice () const = 0;

    //
    // Access to the image level to which this channel belongs.
    //

    IMFUTIL_EXPORT DeepImageLevel&       deepLevel ();
    IMFUTIL_EXPORT const DeepImageLevel& deepLevel () const;

    //
    // Access to the sample count channel for this deep channel.
    //

    IMFUTIL_EXPORT SampleCountChannel&       sampleCounts ();
    IMFUTIL_EXPORT const SampleCountChannel& sampleCounts () const;

protected:
    friend class DeepImageLevel;

    IMFUTIL_EXPORT DeepImageChannel (DeepImageLevel& level, bool pLinear);
    IMFUTIL_EXPORT virtual ~DeepImageChannel ();

    DeepImageChannel (const DeepImageChannel& other) = delete;
    DeepImageChannel& operator= (const DeepImageChannel& other) = delete;
    DeepImageChannel (DeepImageChannel&& other)                 = delete;
    DeepImageChannel& operator= (DeepImageChannel&& other) = delete;

    virtual void setSamplesToZero (
        size_t i, unsigned int oldNumSamples, unsigned int newNumSamples) = 0;

    virtual void moveSampleList (
        size_t       i,
        unsigned int oldNumSamples,
        unsigned int newNumSamples,
        size_t       newSampleListPosition) = 0;

    virtual void moveSamplesToNewBuffer (
        const unsigned int* oldNumSamples,
        const unsigned int* newNumSamples,
        const size_t*       newSampleListPositions) = 0;

    virtual void initializeSampleLists () = 0;

    IMFUTIL_EXPORT virtual void resize ();

    virtual void resetBasePointer () = 0;
};

template <class T>
class IMFUTIL_EXPORT_TEMPLATE_TYPE TypedDeepImageChannel
    : public DeepImageChannel
{
public:
    //
    // The OpenEXR pixel type of this channel (HALF, FLOAT or UINT).
    //

    virtual PixelType pixelType () const;

    //
    // Construct an OpenEXR frame buffer slice for this channel.
    // This function is needed reading an image from an OpenEXR
    // file and for saving an image in an OpenEXR file.
    //

    virtual DeepSlice slice () const;

    //
    // Access to the pixel at pixel space location (x, y), without bounds
    // checking.  Accessing a location outside the data window of the image
    // level results in undefined behavior.
    //
    // The pixel contains a pointer to an array of samples to type T.  The
    // number of samples in this array is sampleCounts().at(x,y).
    //

    T*       operator() (int x, int y);
    const T* operator() (int x, int y) const;

    //
    // Access to the pixel at pixel space location (x, y), with bounds
    // checking.  Accessing a location outside the data window of the
    // image level throws an Iex::ArgExc exception.
    //

    T*       at (int x, int y);
    const T* at (int x, int y) const;

    //
    // Faster access to all pixels in a single horizontal row of the
    // channel.  Access is not bounds checked; accessing out of bounds
    // rows or pixels results in undefined behavior.
    //
    // Rows are numbered from 0 to pixelsPerColumn()-1, and each row
    // contains pixelsPerRow() values.  The number of samples in
    // row(r)[i] is sampleCounts().row(r)[i].
    //

    T* const*       row (int r);
    const T* const* row (int r) const;

private:
    friend class DeepImageLevel;

    IMFUTIL_HIDDEN
    TypedDeepImageChannel (DeepImageLevel& level, bool pLinear);
    IMFUTIL_HIDDEN
    virtual ~TypedDeepImageChannel ();

    TypedDeepImageChannel (const TypedDeepImageChannel& other) = delete;
    TypedDeepImageChannel&
    operator= (const TypedDeepImageChannel& other)        = delete;
    TypedDeepImageChannel (TypedDeepImageChannel&& other) = delete;
    TypedDeepImageChannel& operator= (TypedDeepImageChannel&& other) = delete;

    IMFUTIL_HIDDEN
    virtual void setSamplesToZero (
        size_t i, unsigned int oldNumSamples, unsigned int newNumSamples);

    IMFUTIL_HIDDEN
    virtual void moveSampleList (
        size_t       i,
        unsigned int oldNumSamples,
        unsigned int newNumSamples,
        size_t       newSampleListPosition);

    IMFUTIL_HIDDEN
    virtual void moveSamplesToNewBuffer (
        const unsigned int* oldNumSamples,
        const unsigned int* newNumSamples,
        const size_t*       newSampleListPositions);

    IMFUTIL_HIDDEN
    virtual void initializeSampleLists ();

    IMFUTIL_HIDDEN
    virtual void resize ();

    IMFUTIL_HIDDEN
    virtual void resetBasePointer ();

    T** _sampleListPointers; // Array of pointers to per-pixel
                             //sample lists

    T** _base; // Base pointer for faster access
               // to entries in _sampleListPointers

    T* _sampleBuffer; // Contiguous memory block that
                      // contains all sample lists for
                      // this channel
};

//
// Channel typedefs for the pixel data types supported by OpenEXR.
//

typedef TypedDeepImageChannel<half>         DeepHalfChannel;
typedef TypedDeepImageChannel<float>        DeepFloatChannel;
typedef TypedDeepImageChannel<unsigned int> DeepUIntChannel;

//-----------------------------------------------------------------------------
// Implementation of templates and inline functions
//-----------------------------------------------------------------------------

template <class T>
inline T*
TypedDeepImageChannel<T>::operator() (int x, int y)
{
    return _base[y * pixelsPerRow () + x];
}

template <class T>
inline const T*
TypedDeepImageChannel<T>::operator() (int x, int y) const
{
    return _base[y * pixelsPerRow () + x];
}

template <class T>
inline T*
TypedDeepImageChannel<T>::at (int x, int y)
{
    boundsCheck (x, y);
    return _base[y * pixelsPerRow () + x];
}

template <class T>
inline const T*
TypedDeepImageChannel<T>::at (int x, int y) const
{
    boundsCheck (x, y);
    return _base[y * pixelsPerRow () + x];
}

template <class T>
inline T* const*
TypedDeepImageChannel<T>::row (int r)
{
    return _base + r * pixelsPerRow ();
}

template <class T>
inline const T* const*
TypedDeepImageChannel<T>::row (int r) const
{
    return _base + r * pixelsPerRow ();
}

#ifndef COMPILING_IMF_DEEP_IMAGE_CHANNEL
extern template class IMFUTIL_EXPORT_EXTERN_TEMPLATE
    TypedDeepImageChannel<half>;
extern template class IMFUTIL_EXPORT_EXTERN_TEMPLATE
    TypedDeepImageChannel<float>;
extern template class IMFUTIL_EXPORT_EXTERN_TEMPLATE
    TypedDeepImageChannel<unsigned int>;
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
