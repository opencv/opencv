//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class FlatImageChannel
//
//----------------------------------------------------------------------------

#include "ImfSampleCountChannel.h"
#include "ImfDeepImageLevel.h"
#include "ImfImage.h"
#include <Iex.h>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER
namespace
{

unsigned int
roundListSizeUp (unsigned int n)
{
    //
    // Calculate the number of samples to be allocated for a sample list
    // with n entries by rounding n up to the next power of two.
    //

    if (n == 0) return 0;

    unsigned int s = 1;

    while (s < n)
        s <<= 1;

    return s;
}

size_t
roundBufferSizeUp (size_t n)
{
    //
    // Calculate the number of samples to be allocated for n samples.
    // Leave some extra space for new samples that may be added later.
    //

    return n + n / 2;
}

} // namespace

SampleCountChannel::SampleCountChannel (DeepImageLevel& level)
    : ImageChannel (level, 1, 1, false)
    , _numSamples (0)
    , _base (0)
    , _sampleListSizes (0)
    , _sampleListPositions (0)
    , _totalNumSamples (0)
    , _totalSamplesOccupied (0)
    , _sampleBufferSize (0)
{
    resize ();
}

SampleCountChannel::~SampleCountChannel ()
{
    delete[] _numSamples;
    delete[] _sampleListSizes;
    delete[] _sampleListPositions;
}

PixelType
SampleCountChannel::pixelType () const
{
    return UINT;
}

Slice
SampleCountChannel::slice () const
{
    return Slice (
        UINT,                                    // type
        (char*) _base,                           // base
        sizeof (unsigned int),                   // xStride
        pixelsPerRow () * sizeof (unsigned int), // yStride
        xSampling (),
        ySampling ());
}

DeepImageLevel&
SampleCountChannel::deepLevel ()
{
    return static_cast<DeepImageLevel&> (level ());
}

const DeepImageLevel&
SampleCountChannel::deepLevel () const
{
    return static_cast<const DeepImageLevel&> (level ());
}

void
SampleCountChannel::set (int x, int y, unsigned int newNumSamples)
{
    //
    // Set the number of samples for pixel (x,y) to newNumSamples.
    // Compute the position of the pixel in the various
    // arrays that describe it.
    //

    size_t i = (_base + y * pixelsPerRow () + x) - _numSamples;

    if (newNumSamples <= _numSamples[i])
    {
        //
        // The number of samples for the pixel becomes smaller.
        // Save the new number of samples.
        //

        _totalNumSamples -= _numSamples[i] - newNumSamples;
        _numSamples[i] = newNumSamples;
        return;
    }

    if (newNumSamples <= _sampleListSizes[i])
    {
        //
        // The number of samples for the pixel becomes larger, but the new
        // number of samples still fits into the space that has been allocated
        // for the sample list.  Set the new samples at the end of the list to
        // zero.
        //

        deepLevel ().setSamplesToZero (i, _numSamples[i], newNumSamples);

        _totalNumSamples += newNumSamples - _numSamples[i];
        _numSamples[i] = newNumSamples;
        return;
    }

    int newSampleListSize = roundListSizeUp (newNumSamples);

    if (_totalSamplesOccupied + newSampleListSize <= _sampleBufferSize)
    {
        //
        // The number of samples in the pixel no longer fits into the
        // space that has been allocated for the sample list, but there
        // is space available at the end of the sample buffer.  Allocate
        // space for a new list at the end of the sample buffer, and move
        // the sample list from its old location to its new, larger place.
        //

        deepLevel ().moveSampleList (
            i, _numSamples[i], newNumSamples, _totalSamplesOccupied);

        _sampleListPositions[i] = _totalSamplesOccupied;
        _totalSamplesOccupied += newSampleListSize;
        _totalNumSamples += newNumSamples - _numSamples[i];
        _numSamples[i] = newNumSamples;
        return;
    }

    //
    // The new number of samples no longer fits into the space that has
    // been allocated for the sample list, and there is not enough room
    // at the end of the sample buffer for a new, larger sample list.
    // Allocate an entirely new sample buffer, and move all existing
    // sample lists into it.
    //

    unsigned int* oldNumSamples          = 0;
    size_t*       oldSampleListPositions = 0;

    try
    {
        _totalNumSamples += newNumSamples - _numSamples[i];

        oldNumSamples = _numSamples;
        _numSamples   = new unsigned int[numPixels ()];

        resetBasePointer ();

        oldSampleListPositions = _sampleListPositions;
        _sampleListPositions   = new size_t[numPixels ()];

        _totalSamplesOccupied = 0;

        for (size_t j = 0; j < numPixels (); ++j)
        {
            if (j == i)
                _numSamples[j] = newNumSamples;
            else
                _numSamples[j] = oldNumSamples[j];

            _sampleListPositions[j] = _totalSamplesOccupied;
            _sampleListSizes[j]     = roundListSizeUp (_numSamples[j]);
            _totalSamplesOccupied += _sampleListSizes[j];
        }

        _sampleBufferSize = roundBufferSizeUp (_totalSamplesOccupied);

        deepLevel ().moveSamplesToNewBuffer (
            oldNumSamples, _numSamples, _sampleListPositions);

        delete[] oldNumSamples;
        delete[] oldSampleListPositions;
    }
    catch (...)
    {
        delete[] oldNumSamples;
        delete[] oldSampleListPositions;

        level ().image ().resize (Box2i (V2i (0, 0), V2i (-1, -1)));
        throw;
    }
}

void
SampleCountChannel::set (int r, unsigned int newNumSamples[])
{
    int x = level ().dataWindow ().min.x;
    int y = r + level ().dataWindow ().min.x;

    for (int i = 0; i < pixelsPerRow (); ++i, ++x)
        set (x, y, newNumSamples[i]);
}

void
SampleCountChannel::clear ()
{
    try
    {
        for (size_t i = 0; i < numPixels (); ++i)
        {
            _numSamples[i]          = 0;
            _sampleListSizes[i]     = 0;
            _sampleListPositions[i] = 0;
        }

        _totalNumSamples      = 0;
        _totalSamplesOccupied = 0;
        _sampleBufferSize     = roundBufferSizeUp (_totalSamplesOccupied);

        deepLevel ().initializeSampleLists ();
    }
    catch (...)
    {
        level ().image ().resize (Box2i (V2i (0, 0), V2i (-1, -1)));
        throw;
    }
}

unsigned int*
SampleCountChannel::beginEdit ()
{
    return _numSamples;
}

void
SampleCountChannel::endEdit ()
{
    try
    {
        _totalNumSamples      = 0;
        _totalSamplesOccupied = 0;

        for (size_t i = 0; i < numPixels (); ++i)
        {
            _sampleListSizes[i]     = roundListSizeUp (_numSamples[i]);
            _sampleListPositions[i] = _totalSamplesOccupied;
            _totalNumSamples += _numSamples[i];
            _totalSamplesOccupied += _sampleListSizes[i];
        }

        _sampleBufferSize = roundBufferSizeUp (_totalSamplesOccupied);

        deepLevel ().initializeSampleLists ();
    }
    catch (...)
    {
        level ().image ().resize (Box2i (V2i (0, 0), V2i (-1, -1)));
        throw;
    }
}

void
SampleCountChannel::resize ()
{
    ImageChannel::resize ();

    delete[] _numSamples;
    delete[] _sampleListSizes;
    delete[] _sampleListPositions;

    _numSamples          = 0; // set to 0 to prevent double
    _sampleListSizes     = 0; // deletion in case of an exception
    _sampleListPositions = 0;

    _numSamples          = new unsigned int[numPixels ()];
    _sampleListSizes     = new unsigned int[numPixels ()];
    _sampleListPositions = new size_t[numPixels ()];

    resetBasePointer ();

    for (size_t i = 0; i < numPixels (); ++i)
    {
        _numSamples[i]          = 0;
        _sampleListSizes[i]     = 0;
        _sampleListPositions[i] = 0;
    }

    _totalNumSamples      = 0;
    _totalSamplesOccupied = 0;

    _sampleBufferSize = roundBufferSizeUp (_totalSamplesOccupied);
}

void
SampleCountChannel::resetBasePointer ()
{
    _base = _numSamples - level ().dataWindow ().min.y * pixelsPerRow () -
            level ().dataWindow ().min.x;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
