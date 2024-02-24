//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class FlatImageChannel
//
//----------------------------------------------------------------------------
#include "ImfUtilExport.h"
#include <ImathExport.h>
#include <ImathNamespace.h>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER
class IMFUTIL_EXPORT_TYPE half;
IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#define COMPILING_IMF_FLAT_IMAGE_CHANNEL

#include "ImfFlatImageChannel.h"
#include "ImfFlatImageLevel.h"
#include <Iex.h>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

FlatImageChannel::FlatImageChannel (
    FlatImageLevel& level, int xSampling, int ySampling, bool pLinear)
    : ImageChannel (level, xSampling, ySampling, pLinear)
{
    // empty
}

FlatImageChannel::~FlatImageChannel ()
{
    // empty
}

FlatImageLevel&
FlatImageChannel::flatLevel ()
{
    return static_cast<FlatImageLevel&> (level ());
}

const FlatImageLevel&
FlatImageChannel::flatLevel () const
{
    return static_cast<const FlatImageLevel&> (level ());
}

void
FlatImageChannel::resize ()
{
    ImageChannel::resize ();
}

//-----------------------------------------------------------------------------

template <class T>
TypedFlatImageChannel<T>::TypedFlatImageChannel (
    FlatImageLevel& level, int xSampling, int ySampling, bool pLinear)
    : FlatImageChannel (level, xSampling, ySampling, pLinear)
    , _pixels (0)
    , _base (0)
{
    resize ();
}

template <class T> TypedFlatImageChannel<T>::~TypedFlatImageChannel ()
{
    delete[] _pixels;
}

template <>
inline PixelType
FlatHalfChannel::pixelType () const
{
    return HALF;
}

template <>
inline PixelType
FlatFloatChannel::pixelType () const
{
    return FLOAT;
}

template <>
inline PixelType
FlatUIntChannel::pixelType () const
{
    return UINT;
}

template <class T>
Slice
TypedFlatImageChannel<T>::slice () const
{
    return Slice (
        pixelType (),                 // type
        (char*) _base,                // base
        sizeof (T),                   // xStride
        pixelsPerRow () * sizeof (T), // yStride
        xSampling (),
        ySampling ());
}

template <class T>
void
TypedFlatImageChannel<T>::resize ()
{
    delete[] _pixels;
    _pixels = 0;

    FlatImageChannel::resize (); // may throw an exception

    _pixels = new T[numPixels ()];

    for (size_t i = 0; i < numPixels (); ++i)
        _pixels[i] = T (0);

    resetBasePointer ();
}

template <class T>
void
TypedFlatImageChannel<T>::resetBasePointer ()
{
    _base = _pixels -
            (level ().dataWindow ().min.y / ySampling ()) * pixelsPerRow () -
            (level ().dataWindow ().min.x / xSampling ());
}

template class IMFUTIL_EXPORT_TEMPLATE_INSTANCE TypedFlatImageChannel<half>;
template class IMFUTIL_EXPORT_TEMPLATE_INSTANCE TypedFlatImageChannel<float>;
template class IMFUTIL_EXPORT_TEMPLATE_INSTANCE
    TypedFlatImageChannel<unsigned int>;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
