//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_SAMPLE_COUNT_CHANNEL_H
#define INCLUDED_IMF_SAMPLE_COUNT_CHANNEL_H

//----------------------------------------------------------------------------
//
//      class SampleCountChannel
//
//      For an explanation of images, levels and channels,
//      see the comments in header file Image.h.
//
//----------------------------------------------------------------------------

#include "ImfImageChannel.h"
#include "ImfUtilExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class DeepImageLevel;

//
// Sample count channel for a deep image level:
//
// Each deep image level has a number of samples channel.  For each
// pixel location (x,y) within the data window of the level, the sample
// count channel stores a single integer, n(x,y).  A deep channel, c,
// in the level as the sample count channel stores n(x,y) samples at
// location (x,y) if
//
//          x % c.xSampling() == 0 and y % c.ySampling() == 0.
//
// The deep channel stores no samples at location (x,y) if
//
//          x % c.xSampling() != 0 or y % c.ySampling() != 0,
//

class IMFUTIL_EXPORT_TYPE SampleCountChannel : public ImageChannel
{
public:
    //
    // The OpenEXR pixel type of this channel (HALF, FLOAT or UINT).
    //

    IMFUTIL_EXPORT
    virtual PixelType pixelType () const;

    //
    // Construct an OpenEXR frame buffer slice for this channel.
    // This function is needed reading an image from an OpenEXR
    // file and for saving an image in an OpenEXR file.
    //

    IMFUTIL_EXPORT
    Slice slice () const;

    //
    // Access to the image level to which this channel belongs.
    //

    IMFUTIL_EXPORT
    DeepImageLevel& deepLevel ();
    IMFUTIL_EXPORT
    const DeepImageLevel& deepLevel () const;

    //
    // Access to n(x,y), without bounds checking.  Accessing a location
    // outside the data window of the image level results in undefined
    // behavior.
    //

    IMFUTIL_EXPORT
    const unsigned int& operator() (int x, int y) const;

    //
    // Access to n(x,y), with bounds checking.  Accessing a location outside
    // the data window of the image level throws an Iex::ArgExc exception.
    //

    IMFUTIL_EXPORT
    const unsigned int& at (int x, int y) const;

    //
    // Faster access to n(x,y) for all pixels in a single horizontal row of
    // the channel.  Rows are numbered from 0 to pixelsPerColumn()-1, and
    // each row contains pixelsPerRow() values.
    // Access is not bounds checked; accessing out of bounds rows or pixels
    // results in undefined behavior.
    //

    IMFUTIL_EXPORT
    const unsigned int* row (int r) const;

    //
    // Change the sample counts in one or more pixels:
    //
    // set(x,y,m)   sets n(x,y) to m.
    //
    // set(r,m)     sets n(x,y) for all pixels in row r according to the
    //              values in array m.  The array  must contain pixelsPerRow()
    //              entries, and the row number must be in the range from 0
    //              to pixelsPerColumn()-1.
    //
    // clear()      sets n(x,y) to 0 for all pixels within the data window
    //              of the level.
    //
    // If the sample count for a pixel is increased, then new samples are
    // appended at the end of the sample list of each deep channel.  The
    // new samples are initialized to zero.  If the sample count in a pixel
    // is decreased, then sample list of each deep channel is truncated by
    // discarding samples at the end of the list.
    //
    // Access is bounds-checked; attempting to set the number of samples of
    // a pixel outside the data window throws an Iex::ArgExc exception.
    //
    // Memory allocation for the sample lists is not particularly clever;
    // repeatedly increasing and decreasing the number of samples in the
    // pixels of a level is likely to result in serious memory fragmentation.
    //
    // Setting the number of samples for one or more pixels may cause the
    // program to run out of memory.  If this happens, the image is resized
    // to zero by zero pixels and an exception is thrown.  Note that the
    // resizing operation deletes this sample count channel and the image
    // level to which it belongs.
    //

    IMFUTIL_EXPORT
    void set (int x, int y, unsigned int newNumSamples);
    IMFUTIL_EXPORT
    void set (int r, unsigned int newNumSamples[]);
    IMFUTIL_EXPORT
    void clear ();

    //
    // OpenEXR file reading support / make sample counts editable:
    //
    //  beginEdit()     frees all memory that has been allocated for samples
    //                  in the deep channels, and returns a pointer to an
    //                  array of pixelsPerRow() by pixelsPerColumn() sample
    //                  counts in row-major order.
    //
    //                  After beginEdit() returns, application code is
    //                  free to change the values in the sample count array.
    //                  In particular, the application can fill the array by
    //                  reading the sample counts from an OpenEXR file.
    //
    //                  However, since memory for the samples in the deep
    //                  channels has been freed, attempting to access any
    //                  sample in a deep channel results in undefined
    //                  behavior, most likely a program crash.
    //
    //  endEdit()       allocates new memory for all samples in the deep
    //                  channels of the layer, according to the current
    //                  sample counts, and sets the samples to zero.
    //
    // Application code must take make sure that each call to beginEdit()
    // is followed by a corresponding endEdit() call, even if an
    // exception occurs while the sample counts are accessed.  In order to
    // do that, application code may want to create a temporary Edit
    //  object instead of calling beginEdit() and endEdit() directly.
    //
    // Setting the number of samples for all pixels in the image may
    // cause the program to run out of memory.  If this happens, the image
    // is resized to zero by zero pixels and an exception is thrown.
    // Note that the resizing operation deletes this sample count channel
    // and the image level to which it belongs.
    //

    IMFUTIL_EXPORT
    unsigned int* beginEdit ();
    IMFUTIL_EXPORT
    void endEdit ();

    class Edit
    {
    public:
        //
        // Constructor calls level->beginEdit(),
        // destructor calls level->endEdit().
        //

        IMFUTIL_EXPORT
        Edit (SampleCountChannel& level);
        IMFUTIL_EXPORT
        ~Edit ();

        Edit (const Edit& other) = delete;
        Edit& operator= (const Edit& other) = delete;
        Edit (Edit&& other)                 = delete;
        Edit& operator= (Edit&& other) = delete;

        //
        // Access to the writable sample count array.
        //

        IMFUTIL_EXPORT
        unsigned int* sampleCounts () const;

    private:
        SampleCountChannel& _channel;
        unsigned int*       _sampleCounts;
    };

    //
    // Functions that support the implementation of deep image channels.
    //

    IMFUTIL_EXPORT
    const unsigned int* numSamples () const;
    IMFUTIL_EXPORT
    const unsigned int* sampleListSizes () const;
    IMFUTIL_EXPORT
    const size_t* sampleListPositions () const;
    IMFUTIL_EXPORT
    size_t sampleBufferSize () const;

private:
    friend class DeepImageLevel;

    //
    // The constructor and destructor are not public because
    // image channels exist only as parts of a deep image level.
    //

    SampleCountChannel (DeepImageLevel& level);
    virtual ~SampleCountChannel ();

    virtual void resize ();

    void resetBasePointer ();

    unsigned int* _numSamples; // Array of per-pixel sample counts

    unsigned int* _base; // Base pointer for faster access
                         // to entries in _numSamples

    unsigned int* _sampleListSizes; // Array of allocated sizes of
                                    // per-pixel sample lists

    size_t* _sampleListPositions; // Array of positions of per-pixel
                                  // sample lists within sample list
                                  // buffer

    size_t _totalNumSamples; // Sum of all entries in the
                             // _numSamples array

    size_t _totalSamplesOccupied; // Total number of samples within
                                  // sample list buffer that have
                                  // either been allocated for sample
                                  // lists or lost to fragmentation

    size_t _sampleBufferSize; // Size of the sample list buffer.
};

//-----------------------------------------------------------------------------
// Implementation of templates and inline functions
//-----------------------------------------------------------------------------

inline SampleCountChannel::Edit::Edit (SampleCountChannel& channel)
    : _channel (channel), _sampleCounts (channel.beginEdit ())
{
    // empty
}

inline SampleCountChannel::Edit::~Edit ()
{
    _channel.endEdit ();
}

inline unsigned int*
SampleCountChannel::Edit::sampleCounts () const
{
    return _sampleCounts;
}

inline const unsigned int*
SampleCountChannel::numSamples () const
{
    return _numSamples;
}

inline const unsigned int*
SampleCountChannel::sampleListSizes () const
{
    return _sampleListSizes;
}

inline const size_t*
SampleCountChannel::sampleListPositions () const
{
    return _sampleListPositions;
}

inline size_t
SampleCountChannel::sampleBufferSize () const
{
    return _sampleBufferSize;
}

inline const unsigned int&
SampleCountChannel::operator() (int x, int y) const
{
    return _base[y * pixelsPerRow () + x];
}

inline const unsigned int&
SampleCountChannel::at (int x, int y) const
{
    boundsCheck (x, y);
    return _base[y * pixelsPerRow () + x];
}

inline const unsigned int*
SampleCountChannel::row (int n) const
{
    return _base + n * pixelsPerRow ();
}

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
