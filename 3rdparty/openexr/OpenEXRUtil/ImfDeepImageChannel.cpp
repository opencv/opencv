//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class DeepImageChannel
//
//----------------------------------------------------------------------------
#include "ImfUtilExport.h"
#include <ImathExport.h>
#include <ImathNamespace.h>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER
class IMFUTIL_EXPORT_TYPE half;
IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#define COMPILING_IMF_DEEP_IMAGE_CHANNEL

#include "ImfDeepImageChannel.h"
#include "ImfDeepImageLevel.h"
#include <Iex.h>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

DeepImageChannel::DeepImageChannel (DeepImageLevel& level, bool pLinear)
    : ImageChannel (level, 1, 1, pLinear)
{
    // empty
}

DeepImageChannel::~DeepImageChannel ()
{
    // empty
}

DeepImageLevel&
DeepImageChannel::deepLevel ()
{
    return static_cast<DeepImageLevel&> (level ());
}

const DeepImageLevel&
DeepImageChannel::deepLevel () const
{
    return static_cast<const DeepImageLevel&> (level ());
}

SampleCountChannel&
DeepImageChannel::sampleCounts ()
{
    return deepLevel ().sampleCounts ();
}

const SampleCountChannel&
DeepImageChannel::sampleCounts () const
{
    return deepLevel ().sampleCounts ();
}

void
DeepImageChannel::resize ()
{
    ImageChannel::resize ();
}

//-----------------------------------------------------------------------------

template <class T>
TypedDeepImageChannel<T>::TypedDeepImageChannel (
    DeepImageLevel& level, bool pLinear)
    : DeepImageChannel (level, pLinear)
    , _sampleListPointers (0)
    , _base (0)
    , _sampleBuffer (0)
{
    resize ();
}

template <class T> TypedDeepImageChannel<T>::~TypedDeepImageChannel ()
{
    delete[] _sampleListPointers;
    delete[] _sampleBuffer;
}

template <class T>
DeepSlice
TypedDeepImageChannel<T>::slice () const
{
    return DeepSlice (
        pixelType (),                  // type
        (char*) _base,                 // base
        sizeof (T*),                   // xStride
        pixelsPerRow () * sizeof (T*), // yStride
        sizeof (T),                    // sampleStride
        xSampling (),
        ySampling ());
}
template <class T>
void
TypedDeepImageChannel<T>::setSamplesToZero (
    size_t i, unsigned int oldNumSamples, unsigned int newNumSamples)
{
    //
    // Expand the size of a sample list for a single pixel and
    // set the new samples in the list to 0.
    //
    // i                The position of the affected pixel in
    //                  the channel's _sampleListPointers.
    //
    // oldNumSamples    Original number of samples in the sample list.
    //
    // newNumSamples    New number of samples in the sample list.
    //

    for (unsigned int j = oldNumSamples; j < newNumSamples; ++j)
        _sampleListPointers[i][j] = 0;
}

template <class T>
void
TypedDeepImageChannel<T>::moveSampleList (
    size_t       i,
    unsigned int oldNumSamples,
    unsigned int newNumSamples,
    size_t       newSampleListPosition)
{
    //
    // Resize the sample list for a single pixel and move it to a new
    // position in the sample buffer for this channel.
    //
    // i                        The position of the affected pixel in
    //                          the channel's _sampleListPointers.
    //
    // oldNumSamples            Original number of samples in sample list.
    //
    // newNumSamples            New number of samples in the sample list.
    //                          If the new number of samples is larger than
    //                          the old number of samples for a given sample
    //                          list, then the end of the new sample list
    //                          is filled with zeroes.  If the new number of
    //                          samples is smaller than the old one, then
    //                          samples at the end of the old sample list
    //                          are discarded.
    //
    // newSampleListPosition    The new position of the sample list in the
    //                          sample buffer.
    //

    T* oldSampleList = _sampleListPointers[i];
    T* newSampleList = _sampleBuffer + newSampleListPosition;

    if (oldNumSamples > newNumSamples)
    {
        for (unsigned int j = 0; j < newNumSamples; ++j)
            newSampleList[j] = oldSampleList[j];
    }
    else
    {
        for (unsigned int j = 0; j < oldNumSamples; ++j)
            newSampleList[j] = oldSampleList[j];

        for (unsigned int j = oldNumSamples; j < newNumSamples; ++j)
            newSampleList[j] = 0;
    }

    _sampleListPointers[i] = newSampleList;
}

template <class T>
void
TypedDeepImageChannel<T>::moveSamplesToNewBuffer (
    const unsigned int* oldNumSamples,
    const unsigned int* newNumSamples,
    const size_t*       newSampleListPositions)
{
    //
    // Allocate a new sample buffer for this channel.
    // Copy the sample lists for all pixels into the new buffer.
    // Then delete the old sample buffer.
    //
    // oldNumSamples            Number of samples in each sample list in the
    //                          old sample buffer.
    //
    // newNumSamples            Number of samples in each sample list in
    //                          the new sample buffer.  If the new number
    //                          of samples is larger than the old number of
    //                          samples for a given sample list, then the
    //                          end of the new sample list is filled with
    //                          zeroes.  If the new number of samples is
    //                          smaller than the old one, then samples at
    //                          the end of the old sample list are discarded.
    //
    // newSampleListPositions   The positions of the new sample lists in the
    //                          new sample buffer.
    //

    T* oldSampleBuffer = _sampleBuffer;
    _sampleBuffer      = new T[sampleCounts ().sampleBufferSize ()];

    for (size_t i = 0; i < numPixels (); ++i)
    {
        T* oldSampleList = _sampleListPointers[i];
        T* newSampleList = _sampleBuffer + newSampleListPositions[i];

        if (oldNumSamples[i] > newNumSamples[i])
        {
            for (unsigned int j = 0; j < newNumSamples[i]; ++j)
                newSampleList[j] = oldSampleList[j];
        }
        else
        {
            for (unsigned int j = 0; j < oldNumSamples[i]; ++j)
                newSampleList[j] = oldSampleList[j];

            for (unsigned int j = oldNumSamples[i]; j < newNumSamples[i]; ++j)
                newSampleList[j] = 0;
        }

        _sampleListPointers[i] = newSampleList;
    }

    delete[] oldSampleBuffer;
}

template <class T>
void
TypedDeepImageChannel<T>::initializeSampleLists ()
{
    //
    // Allocate a new set of sample lists for this channel, and
    // construct zero-filled sample lists for the pixels.
    //

    delete[] _sampleBuffer;

    _sampleBuffer = 0; // set to 0 to prevent double deletion
                       // in case of an exception

    const unsigned int* numSamples    = sampleCounts ().numSamples ();
    const size_t* sampleListPositions = sampleCounts ().sampleListPositions ();

    _sampleBuffer = new T[sampleCounts ().sampleBufferSize ()];

    resetBasePointer ();

    for (size_t i = 0; i < numPixels (); ++i)
    {
        _sampleListPointers[i] = _sampleBuffer + sampleListPositions[i];

        for (unsigned int j = 0; j < numSamples[i]; ++j)
            _sampleListPointers[i][j] = T (0);
    }
}

template <class T>
void
TypedDeepImageChannel<T>::resize ()
{
    DeepImageChannel::resize ();

    delete[] _sampleListPointers;
    _sampleListPointers = 0;
    _sampleListPointers = new T*[numPixels ()];
    initializeSampleLists ();
}

template <class T>
void
TypedDeepImageChannel<T>::resetBasePointer ()
{
    _base = _sampleListPointers -
            level ().dataWindow ().min.y * pixelsPerRow () -
            level ().dataWindow ().min.x;
}

template <>
PixelType
TypedDeepImageChannel<half>::pixelType () const
{
    return HALF;
}

template <>
PixelType
TypedDeepImageChannel<float>::pixelType () const
{
    return FLOAT;
}

template <>
PixelType
TypedDeepImageChannel<unsigned int>::pixelType () const
{
    return UINT;
}

template class IMFUTIL_EXPORT_TEMPLATE_INSTANCE TypedDeepImageChannel<half>;
template class IMFUTIL_EXPORT_TEMPLATE_INSTANCE TypedDeepImageChannel<float>;
template class IMFUTIL_EXPORT_TEMPLATE_INSTANCE
    TypedDeepImageChannel<unsigned int>;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
