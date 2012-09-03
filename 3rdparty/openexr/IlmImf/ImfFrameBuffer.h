///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMF_FRAME_BUFFER_H
#define INCLUDED_IMF_FRAME_BUFFER_H

//-----------------------------------------------------------------------------
//
//	class Slice
//	class FrameBuffer
//
//-----------------------------------------------------------------------------

#include <ImfName.h>
#include <ImfPixelType.h>
#include <map>
#include <string>


namespace Imf {


//-------------------------------------------------------
// Description of a single slice of the frame buffer:
//
// Note -- terminology: as part of a file, a component of
// an image (e.g. red, green, blue, depth etc.) is called
// a "channel".  As part of a frame buffer, an image
// component is called a "slice".
//-------------------------------------------------------

struct Slice
{
    //------------------------------
    // Data type; see ImfPixelType.h
    //------------------------------

    PixelType		type;


    //---------------------------------------------------------------------
    // Memory layout:  The address of pixel (x, y) is
    //
    //	base + (xp / xSampling) * xStride + (yp / ySampling) * yStride
    //
    // where xp and yp are computed as follows:
    //
    //	* If we are reading or writing a scanline-based file:
    //
    //	    xp = x
    //	    yp = y
    //
    //  * If we are reading a tile whose upper left coorner is at (xt, yt):
    //
    //	    if xTileCoords is true then xp = x - xt, else xp = x
    //	    if yTileCoords is true then yp = y - yt, else yp = y
    //
    //---------------------------------------------------------------------

    char *		base;
    size_t		xStride;
    size_t		yStride;


    //--------------------------------------------
    // Subsampling: pixel (x, y) is present in the
    // slice only if 
    //
    //  x % xSampling == 0 && y % ySampling == 0
    //
    //--------------------------------------------

    int			xSampling;
    int			ySampling;


    //----------------------------------------------------------
    // Default value, used to fill the slice when a file without
    // a channel that corresponds to this slice is read.
    //----------------------------------------------------------

    double		fillValue;
    

    //-------------------------------------------------------
    // For tiled files, the xTileCoords and yTileCoords flags
    // determine whether pixel addressing is performed using
    // absolute coordinates or coordinates relative to a
    // tile's upper left corner.  (See the comment on base,
    // xStride and yStride, above.)
    //
    // For scanline-based files these flags have no effect;
    // pixel addressing is always done using absolute
    // coordinates.
    //-------------------------------------------------------

    bool                xTileCoords;
    bool                yTileCoords;


    //------------
    // Constructor
    //------------

    Slice (PixelType type = HALF,
	   char * base = 0,
	   size_t xStride = 0,
	   size_t yStride = 0,
	   int xSampling = 1,
	   int ySampling = 1,
	   double fillValue = 0.0,
           bool xTileCoords = false,
           bool yTileCoords = false);
};


class FrameBuffer
{
  public:

    //------------
    // Add a slice
    //------------

    void			insert (const char name[],
					const Slice &slice);

    void			insert (const std::string &name,
					const Slice &slice);

    //----------------------------------------------------------------
    // Access to existing slices:
    //
    // [n]		Returns a reference to the slice with name n.
    //			If no slice with name n exists, an Iex::ArgExc
    //			is thrown.
    //
    // findSlice(n)	Returns a pointer to the slice with name n,
    //			or 0 if no slice with name n exists.
    //
    //----------------------------------------------------------------

    Slice &			operator [] (const char name[]);
    const Slice &		operator [] (const char name[]) const;

    Slice &			operator [] (const std::string &name);
    const Slice &		operator [] (const std::string &name) const;

    Slice *			findSlice (const char name[]);
    const Slice *		findSlice (const char name[]) const;

    Slice *			findSlice (const std::string &name);
    const Slice *		findSlice (const std::string &name) const;


    //-----------------------------------------
    // Iterator-style access to existing slices
    //-----------------------------------------

    typedef std::map <Name, Slice> SliceMap;

    class Iterator;
    class ConstIterator;

    Iterator			begin ();
    ConstIterator		begin () const;

    Iterator			end ();
    ConstIterator		end () const;

    Iterator			find (const char name[]);
    ConstIterator		find (const char name[]) const;

    Iterator			find (const std::string &name);
    ConstIterator		find (const std::string &name) const;

  private:

    SliceMap			_map;
};


//----------
// Iterators
//----------

class FrameBuffer::Iterator
{
  public:

    Iterator ();
    Iterator (const FrameBuffer::SliceMap::iterator &i);

    Iterator &			operator ++ ();
    Iterator 			operator ++ (int);

    const char *		name () const;
    Slice &			slice () const;

  private:

    friend class FrameBuffer::ConstIterator;

    FrameBuffer::SliceMap::iterator _i;
};


class FrameBuffer::ConstIterator
{
  public:

    ConstIterator ();
    ConstIterator (const FrameBuffer::SliceMap::const_iterator &i);
    ConstIterator (const FrameBuffer::Iterator &other);

    ConstIterator &		operator ++ ();
    ConstIterator 		operator ++ (int);

    const char *		name () const;
    const Slice &		slice () const;

  private:

    friend bool operator == (const ConstIterator &, const ConstIterator &);
    friend bool operator != (const ConstIterator &, const ConstIterator &);

    FrameBuffer::SliceMap::const_iterator _i;
};


//-----------------
// Inline Functions
//-----------------

inline
FrameBuffer::Iterator::Iterator (): _i()
{
    // empty
}


inline
FrameBuffer::Iterator::Iterator (const FrameBuffer::SliceMap::iterator &i):
    _i (i)
{
    // empty
}


inline FrameBuffer::Iterator &		
FrameBuffer::Iterator::operator ++ ()
{
    ++_i;
    return *this;
}


inline FrameBuffer::Iterator 	
FrameBuffer::Iterator::operator ++ (int)
{
    Iterator tmp = *this;
    ++_i;
    return tmp;
}


inline const char *
FrameBuffer::Iterator::name () const
{
    return *_i->first;
}


inline Slice &	
FrameBuffer::Iterator::slice () const
{
    return _i->second;
}


inline
FrameBuffer::ConstIterator::ConstIterator (): _i()
{
    // empty
}

inline
FrameBuffer::ConstIterator::ConstIterator
    (const FrameBuffer::SliceMap::const_iterator &i): _i (i)
{
    // empty
}


inline
FrameBuffer::ConstIterator::ConstIterator (const FrameBuffer::Iterator &other):
    _i (other._i)
{
    // empty
}

inline FrameBuffer::ConstIterator &
FrameBuffer::ConstIterator::operator ++ ()
{
    ++_i;
    return *this;
}


inline FrameBuffer::ConstIterator 		
FrameBuffer::ConstIterator::operator ++ (int)
{
    ConstIterator tmp = *this;
    ++_i;
    return tmp;
}


inline const char *
FrameBuffer::ConstIterator::name () const
{
    return *_i->first;
}

inline const Slice &	
FrameBuffer::ConstIterator::slice () const
{
    return _i->second;
}


inline bool
operator == (const FrameBuffer::ConstIterator &x,
	     const FrameBuffer::ConstIterator &y)
{
    return x._i == y._i;
}


inline bool
operator != (const FrameBuffer::ConstIterator &x,
	     const FrameBuffer::ConstIterator &y)
{
    return !(x == y);
}


} // namespace Imf

#endif
