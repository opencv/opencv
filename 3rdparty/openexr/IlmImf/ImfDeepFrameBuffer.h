///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2011, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#ifndef IMFDEEPFRAMEBUFFER_H_
#define IMFDEEPFRAMEBUFFER_H_

#include "ImfFrameBuffer.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//--------------------------------------------------------
// Description of a single deep slice of the frame buffer:
//--------------------------------------------------------

struct DeepSlice : public Slice
{
    //---------------------------------------------------------------------
    // The stride for each sample in this slice.
    //
    // Memory layout:  The address of sample i in pixel (x, y) is
    //
    //  base + (xp / xSampling) * xStride + (yp / ySampling) * yStride
    //       + i * sampleStride
    //
    // where xp and yp are computed as follows:
    //
    //  * If we are reading or writing a scanline-based file:
    //
    //      xp = x
    //      yp = y
    //
    //  * If we are reading a tile whose upper left coorner is at (xt, yt):
    //
    //      if xTileCoords is true then xp = x - xt, else xp = x
    //      if yTileCoords is true then yp = y - yt, else yp = y
    //
    //---------------------------------------------------------------------

    int sampleStride;

    //------------
    // Constructor
    //------------
    IMF_EXPORT
    DeepSlice (PixelType type = HALF,
               char * base = 0,
               size_t xStride = 0,
               size_t yStride = 0,
               size_t sampleStride = 0,
               int xSampling = 1,
               int ySampling = 1,
               double fillValue = 0.0,
               bool xTileCoords = false,
               bool yTileCoords = false);
};

//-----------------
// DeepFrameBuffer.
//-----------------

class DeepFrameBuffer
{
  public:


    //------------
    // Add a slice
    //------------

    IMF_EXPORT
    void                        insert (const char name[],
                                        const DeepSlice &slice);

    IMF_EXPORT
    void                        insert (const std::string &name,
                                        const DeepSlice &slice);

    //----------------------------------------------------------------
    // Access to existing slices:
    //
    // [n]              Returns a reference to the slice with name n.
    //                  If no slice with name n exists, an IEX_NAMESPACE::ArgExc
    //                  is thrown.
    //
    // findSlice(n)     Returns a pointer to the slice with name n,
    //                  or 0 if no slice with name n exists.
    //
    //----------------------------------------------------------------

    IMF_EXPORT
    DeepSlice &                 operator [] (const char name[]);
    IMF_EXPORT
    const DeepSlice &           operator [] (const char name[]) const;

    IMF_EXPORT
    DeepSlice &                 operator [] (const std::string &name);
    IMF_EXPORT
    const DeepSlice &           operator [] (const std::string &name) const;

    IMF_EXPORT
    DeepSlice *                 findSlice (const char name[]);
    IMF_EXPORT
    const DeepSlice *           findSlice (const char name[]) const;

    IMF_EXPORT
    DeepSlice *                 findSlice (const std::string &name);
    IMF_EXPORT
    const DeepSlice *           findSlice (const std::string &name) const;


    //-----------------------------------------
    // Iterator-style access to existing slices
    //-----------------------------------------

    typedef std::map <Name, DeepSlice> SliceMap;

    class Iterator;
    class ConstIterator;

    IMF_EXPORT
    Iterator                    begin ();
    IMF_EXPORT
    ConstIterator               begin () const;

    IMF_EXPORT
    Iterator                    end ();
    IMF_EXPORT
    ConstIterator               end () const;

    IMF_EXPORT
    Iterator                    find (const char name[]);
    IMF_EXPORT
    ConstIterator               find (const char name[]) const;

    IMF_EXPORT
    Iterator                    find (const std::string &name);
    IMF_EXPORT
    ConstIterator               find (const std::string &name) const;

    //----------------------------------------------------
    // Public function for accessing a sample count slice.
    //----------------------------------------------------

    IMF_EXPORT
    void                        insertSampleCountSlice(const Slice & slice);
    IMF_EXPORT
    const Slice &               getSampleCountSlice() const;

  private:

    SliceMap                    _map;
    Slice                       _sampleCounts;
};

//----------
// Iterators
//----------

class DeepFrameBuffer::Iterator
{
  public:

    IMF_EXPORT
    Iterator ();
    IMF_EXPORT
    Iterator (const DeepFrameBuffer::SliceMap::iterator &i);

    IMF_EXPORT
    Iterator &                  operator ++ ();
    IMF_EXPORT
    Iterator                    operator ++ (int);

    IMF_EXPORT
    const char *                name () const;
    IMF_EXPORT
    DeepSlice &                 slice () const;

  private:

    friend class DeepFrameBuffer::ConstIterator;

    DeepFrameBuffer::SliceMap::iterator _i;
};


class DeepFrameBuffer::ConstIterator
{
  public:

    IMF_EXPORT
    ConstIterator ();
    IMF_EXPORT
    ConstIterator (const DeepFrameBuffer::SliceMap::const_iterator &i);
    IMF_EXPORT
    ConstIterator (const DeepFrameBuffer::Iterator &other);

    IMF_EXPORT
    ConstIterator &             operator ++ ();
    IMF_EXPORT
    ConstIterator               operator ++ (int);

    IMF_EXPORT
    const char *                name () const;
    IMF_EXPORT
    const DeepSlice &           slice () const;

  private:

    friend bool operator == (const ConstIterator &, const ConstIterator &);
    friend bool operator != (const ConstIterator &, const ConstIterator &);

    DeepFrameBuffer::SliceMap::const_iterator _i;
};


//-----------------
// Inline Functions
//-----------------

inline
DeepFrameBuffer::Iterator::Iterator (): _i()
{
    // empty
}


inline
DeepFrameBuffer::Iterator::Iterator (const DeepFrameBuffer::SliceMap::iterator &i):
    _i (i)
{
    // empty
}


inline DeepFrameBuffer::Iterator &
DeepFrameBuffer::Iterator::operator ++ ()
{
    ++_i;
    return *this;
}


inline DeepFrameBuffer::Iterator
DeepFrameBuffer::Iterator::operator ++ (int)
{
    Iterator tmp = *this;
    ++_i;
    return tmp;
}


inline const char *
DeepFrameBuffer::Iterator::name () const
{
    return *_i->first;
}


inline DeepSlice &
DeepFrameBuffer::Iterator::slice () const
{
    return _i->second;
}


inline
DeepFrameBuffer::ConstIterator::ConstIterator (): _i()
{
    // empty
}

inline
DeepFrameBuffer::ConstIterator::ConstIterator
    (const DeepFrameBuffer::SliceMap::const_iterator &i): _i (i)
{
    // empty
}


inline
DeepFrameBuffer::ConstIterator::ConstIterator (const DeepFrameBuffer::Iterator &other):
    _i (other._i)
{
    // empty
}

inline DeepFrameBuffer::ConstIterator &
DeepFrameBuffer::ConstIterator::operator ++ ()
{
    ++_i;
    return *this;
}


inline DeepFrameBuffer::ConstIterator
DeepFrameBuffer::ConstIterator::operator ++ (int)
{
    ConstIterator tmp = *this;
    ++_i;
    return tmp;
}


inline const char *
DeepFrameBuffer::ConstIterator::name () const
{
    return *_i->first;
}

inline const DeepSlice &
DeepFrameBuffer::ConstIterator::slice () const
{
    return _i->second;
}


inline bool
operator == (const DeepFrameBuffer::ConstIterator &x,
             const DeepFrameBuffer::ConstIterator &y)
{
    return x._i == y._i;
}


inline bool
operator != (const DeepFrameBuffer::ConstIterator &x,
             const DeepFrameBuffer::ConstIterator &y)
{
    return !(x == y);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT






#endif /* IMFDEEPFRAMEBUFFER_H_ */
