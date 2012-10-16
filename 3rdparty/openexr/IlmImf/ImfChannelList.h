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



#ifndef INCLUDED_IMF_CHANNEL_LIST_H
#define INCLUDED_IMF_CHANNEL_LIST_H

//-----------------------------------------------------------------------------
//
//	class Channel
//	class ChannelList
//
//-----------------------------------------------------------------------------

#include <ImfName.h>
#include <ImfPixelType.h>
#include <map>
#include <set>
#include <string>


namespace Imf {


struct Channel
{
    //------------------------------
    // Data type; see ImfPixelType.h
    //------------------------------

    PixelType		type;


    //--------------------------------------------
    // Subsampling: pixel (x, y) is present in the
    // channel only if 
    //
    //  x % xSampling == 0 && y % ySampling == 0
    //
    //--------------------------------------------

    int			xSampling;
    int			ySampling;


    //--------------------------------------------------------------
    // Hint to lossy compression methods that indicates whether
    // human perception of the quantity represented by this channel
    // is closer to linear or closer to logarithmic.  Compression
    // methods may optimize image quality by adjusting pixel data
    // quantization acording to this hint.
    // For example, perception of red, green, blue and luminance is
    // approximately logarithmic; the difference between 0.1 and 0.2
    // is perceived to be roughly the same as the difference between
    // 1.0 and 2.0.  Perception of chroma coordinates tends to be
    // closer to linear than logarithmic; the difference between 0.1
    // and 0.2 is perceived to be roughly the same as the difference
    // between 1.0 and 1.1.
    //--------------------------------------------------------------

    bool		pLinear;


    //------------
    // Constructor
    //------------
    
    Channel (PixelType type = HALF,
	     int xSampling = 1,
	     int ySampling = 1,
	     bool pLinear = false);


    //------------
    // Operator ==
    //------------

    bool		operator == (const Channel &other) const;
};


class ChannelList
{
  public:

    //--------------
    // Add a channel
    //--------------

    void			insert (const char name[],
					const Channel &channel);

    void			insert (const std::string &name,
					const Channel &channel);

    //------------------------------------------------------------------
    // Access to existing channels:
    //
    // [n]		Returns a reference to the channel with name n.
    //			If no channel with name n exists, an Iex::ArgExc
    //			is thrown.
    //
    // findChannel(n)	Returns a pointer to the channel with name n,
    //			or 0 if no channel with name n exists.
    //
    //------------------------------------------------------------------

    Channel &			operator [] (const char name[]);
    const Channel &		operator [] (const char name[]) const;

    Channel &			operator [] (const std::string &name);
    const Channel &		operator [] (const std::string &name) const;

    Channel *			findChannel (const char name[]);
    const Channel *		findChannel (const char name[]) const;

    Channel *			findChannel (const std::string &name);
    const Channel *		findChannel (const std::string &name) const;


    //-------------------------------------------
    // Iterator-style access to existing channels
    //-------------------------------------------

    typedef std::map <Name, Channel> ChannelMap;

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

    
    //-----------------------------------------------------------------
    // Support for image layers:
    //
    // In an image file with many channels it is sometimes useful to
    // group the channels into "layers", that is, into sets of channels
    // that logically belong together.  Grouping channels into layers
    // is done using a naming convention:  channel C in layer L is
    // called "L.C".
    //
    // For example, a computer graphic image may contain separate
    // R, G and B channels for light that originated at each of
    // several different virtual light sources.  The channels in
    // this image might be called "light1.R", "light1.G", "light1.B",
    // "light2.R", "light2.G", "light2.B", etc.
    // 
    // Note that this naming convention allows layers to be nested;
    // for example, "light1.specular.R" identifies the "R" channel
    // in the "specular" sub-layer of layer "light1".
    //
    // Channel names that don't contain a "." or that contain a
    // "." only at the beginning or at the end are not considered
    // to be part of any layer.
    //
    // layers(lns)		sorts the channels in this ChannelList
    //				into layers and stores the names of
    //				all layers, sorted alphabetically,
    //				into string set lns.
    //
    // channelsInLayer(ln,f,l)	stores a pair of iterators in f and l
    // 				such that the loop
    //
    // 				for (ConstIterator i = f; i != l; ++i)
    // 				   ...
    //
    //				iterates over all channels in layer ln.
    //				channelsInLayer (ln, l, p) calls
    //				channelsWithPrefix (ln + ".", l, p).
    //
    //-----------------------------------------------------------------

    void		layers (std::set <std::string> &layerNames) const;

    void		channelsInLayer (const std::string &layerName,
	    				 Iterator &first,
					 Iterator &last);

    void		channelsInLayer (const std::string &layerName,
	    				 ConstIterator &first,
					 ConstIterator &last) const;


    //-------------------------------------------------------------------
    // Find all channels whose name begins with a given prefix:
    //
    // channelsWithPrefix(p,f,l) stores a pair of iterators in f and l
    // such that the following loop iterates over all channels whose name
    // begins with string p:
    //
    //		for (ConstIterator i = f; i != l; ++i)
    //		    ...
    //
    //-------------------------------------------------------------------

    void			channelsWithPrefix (const char prefix[],
						    Iterator &first,
						    Iterator &last);

    void			channelsWithPrefix (const char prefix[],
						    ConstIterator &first,
						    ConstIterator &last) const;

    void			channelsWithPrefix (const std::string &prefix,
						    Iterator &first,
						    Iterator &last);

    void			channelsWithPrefix (const std::string &prefix,
						    ConstIterator &first,
						    ConstIterator &last) const;

    //------------
    // Operator ==
    //------------

    bool			operator == (const ChannelList &other) const;

  private:

    ChannelMap			_map;
};


//----------
// Iterators
//----------

class ChannelList::Iterator
{
  public:

    Iterator ();
    Iterator (const ChannelList::ChannelMap::iterator &i);

    Iterator &			operator ++ ();
    Iterator 			operator ++ (int);

    const char *		name () const;
    Channel &			channel () const;

  private:

    friend class ChannelList::ConstIterator;

    ChannelList::ChannelMap::iterator _i;
};


class ChannelList::ConstIterator
{
  public:

    ConstIterator ();
    ConstIterator (const ChannelList::ChannelMap::const_iterator &i);
    ConstIterator (const ChannelList::Iterator &other);

    ConstIterator &		operator ++ ();
    ConstIterator 		operator ++ (int);

    const char *		name () const;
    const Channel &		channel () const;

  private:

    friend bool operator == (const ConstIterator &, const ConstIterator &);
    friend bool operator != (const ConstIterator &, const ConstIterator &);

    ChannelList::ChannelMap::const_iterator _i;
};


//-----------------
// Inline Functions
//-----------------

inline
ChannelList::Iterator::Iterator (): _i()
{
    // empty
}


inline
ChannelList::Iterator::Iterator (const ChannelList::ChannelMap::iterator &i):
    _i (i)
{
    // empty
}


inline ChannelList::Iterator &		
ChannelList::Iterator::operator ++ ()
{
    ++_i;
    return *this;
}


inline ChannelList::Iterator 	
ChannelList::Iterator::operator ++ (int)
{
    Iterator tmp = *this;
    ++_i;
    return tmp;
}


inline const char *
ChannelList::Iterator::name () const
{
    return *_i->first;
}


inline Channel &	
ChannelList::Iterator::channel () const
{
    return _i->second;
}


inline
ChannelList::ConstIterator::ConstIterator (): _i()
{
    // empty
}

inline
ChannelList::ConstIterator::ConstIterator
    (const ChannelList::ChannelMap::const_iterator &i): _i (i)
{
    // empty
}


inline
ChannelList::ConstIterator::ConstIterator (const ChannelList::Iterator &other):
    _i (other._i)
{
    // empty
}

inline ChannelList::ConstIterator &
ChannelList::ConstIterator::operator ++ ()
{
    ++_i;
    return *this;
}


inline ChannelList::ConstIterator 		
ChannelList::ConstIterator::operator ++ (int)
{
    ConstIterator tmp = *this;
    ++_i;
    return tmp;
}


inline const char *
ChannelList::ConstIterator::name () const
{
    return *_i->first;
}

inline const Channel &	
ChannelList::ConstIterator::channel () const
{
    return _i->second;
}


inline bool
operator == (const ChannelList::ConstIterator &x,
	     const ChannelList::ConstIterator &y)
{
    return x._i == y._i;
}


inline bool
operator != (const ChannelList::ConstIterator &x,
	     const ChannelList::ConstIterator &y)
{
    return !(x == y);
}


} // namespace Imf

#endif
