///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMF_HEADER_H
#define INCLUDED_IMF_HEADER_H

//-----------------------------------------------------------------------------
//
//	class Header
//
//-----------------------------------------------------------------------------

#include "ImfLineOrder.h"
#include "ImfCompression.h"
#include "ImfName.h"
#include "ImfTileDescription.h"
#include "ImfInt64.h"
#include "ImathVec.h"
#include "ImathBox.h"
#include "IexBaseExc.h"

#include "ImfForward.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

#include <map>
#include <iosfwd>
#include <string>


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

using std::string;


class Header
{
  public:
    
    //----------------------------------------------------------------
    // Default constructor -- the display window and the data window
    // are both set to Box2i (V2i (0, 0), V2i (width-1, height-1).
    //----------------------------------------------------------------

    IMF_EXPORT
    Header (int width = 64,
	    int height = 64,
	    float pixelAspectRatio = 1,
	    const IMATH_NAMESPACE::V2f &screenWindowCenter = IMATH_NAMESPACE::V2f (0, 0),
	    float screenWindowWidth = 1,
	    LineOrder lineOrder = INCREASING_Y,
	    Compression = ZIP_COMPRESSION);


    //--------------------------------------------------------------------
    // Constructor -- the data window is specified explicitly; the display
    // window is set to Box2i (V2i (0, 0), V2i (width-1, height-1).
    //--------------------------------------------------------------------

    IMF_EXPORT
    Header (int width,
	    int height,
	    const IMATH_NAMESPACE::Box2i &dataWindow,
	    float pixelAspectRatio = 1,
	    const IMATH_NAMESPACE::V2f &screenWindowCenter = IMATH_NAMESPACE::V2f (0, 0),
	    float screenWindowWidth = 1,
	    LineOrder lineOrder = INCREASING_Y,
	    Compression = ZIP_COMPRESSION);


    //----------------------------------------------------------
    // Constructor -- the display window and the data window are
    // both specified explicitly.
    //----------------------------------------------------------

    IMF_EXPORT
    Header (const IMATH_NAMESPACE::Box2i &displayWindow,
	    const IMATH_NAMESPACE::Box2i &dataWindow,
	    float pixelAspectRatio = 1,
	    const IMATH_NAMESPACE::V2f &screenWindowCenter = IMATH_NAMESPACE::V2f (0, 0),
	    float screenWindowWidth = 1,
	    LineOrder lineOrder = INCREASING_Y,
	    Compression = ZIP_COMPRESSION);


    //-----------------
    // Copy constructor
    //-----------------

    IMF_EXPORT
    Header (const Header &other);


    //-----------
    // Destructor
    //-----------

    IMF_EXPORT
    ~Header ();


    //-----------
    // Assignment
    //-----------

    IMF_EXPORT
    Header &			operator = (const Header &other);


    //---------------------------------------------------------------
    // Add an attribute:
    //
    // insert(n,attr)	If no attribute with name n exists, a new
    //			attribute with name n, and the same type as
    //			attr, is added, and the value of attr is
    //			copied into the new attribute.
    //
    //			If an attribute with name n exists, and its
    //			type is the same as attr, the value of attr
    //			is copied into this attribute.
    //
    //			If an attribute with name n exists, and its
    //			type is different from attr, an IEX_NAMESPACE::TypeExc
    //			is thrown.
    //
    //---------------------------------------------------------------

    IMF_EXPORT
    void			insert (const char name[],
				        const Attribute &attribute);

    IMF_EXPORT
    void			insert (const std::string &name,
				        const Attribute &attribute);

    //---------------------------------------------------------------
    // Remove an attribute:
    //
    // remove(n)       If an attribute with name n exists, then it
    //                 is removed from the map of present attributes.
    //
    //                 If no attribute with name n exists, then this
    //                 functions becomes a 'no-op'
    //
    //---------------------------------------------------------------

    IMF_EXPORT
    void                        erase (const char name[]);
    IMF_EXPORT
    void                        erase (const std::string &name);

    
    
    //------------------------------------------------------------------
    // Access to existing attributes:
    //
    // [n]			Returns a reference to the attribute
    //				with name n.  If no attribute with
    //				name n exists, an IEX_NAMESPACE::ArgExc is thrown.
    //
    // typedAttribute<T>(n)	Returns a reference to the attribute
    //				with name n and type T.  If no attribute
    //				with name n exists, an IEX_NAMESPACE::ArgExc is
    //				thrown.  If an attribute with name n
    //				exists, but its type is not T, an
    //				IEX_NAMESPACE::TypeExc is thrown.
    //
    // findTypedAttribute<T>(n)	Returns a pointer to the attribute with
    //				name n and type T, or 0 if no attribute
    //				with name n and type T exists.
    //
    //------------------------------------------------------------------

    IMF_EXPORT
    Attribute &			operator [] (const char name[]);
    IMF_EXPORT
    const Attribute &		operator [] (const char name[]) const;

    IMF_EXPORT
    Attribute &			operator [] (const std::string &name);
    IMF_EXPORT
    const Attribute &		operator [] (const std::string &name) const;

    template <class T> T&	typedAttribute (const char name[]);
    template <class T> const T&	typedAttribute (const char name[]) const;

    template <class T> T&	typedAttribute (const std::string &name);
    template <class T> const T&	typedAttribute (const std::string &name) const;

    template <class T> T*	findTypedAttribute (const char name[]);
    template <class T> const T*	findTypedAttribute (const char name[]) const;

    template <class T> T*	findTypedAttribute (const std::string &name);
    template <class T> const T*	findTypedAttribute (const std::string &name)
								       const;

    //---------------------------------------------
    // Iterator-style access to existing attributes
    //---------------------------------------------

    typedef std::map <Name, Attribute *> AttributeMap;

    class Iterator;
    class ConstIterator;

    IMF_EXPORT
    Iterator			begin ();
    IMF_EXPORT
    ConstIterator		begin () const;

    IMF_EXPORT
    Iterator			end ();
    IMF_EXPORT
    ConstIterator		end () const;

    IMF_EXPORT
    Iterator			find (const char name[]);
    IMF_EXPORT
    ConstIterator		find (const char name[]) const;

    IMF_EXPORT
    Iterator			find (const std::string &name);
    IMF_EXPORT
    ConstIterator		find (const std::string &name) const;


    //--------------------------------
    // Access to predefined attributes
    //--------------------------------

    IMF_EXPORT
    IMATH_NAMESPACE::Box2i &		displayWindow ();
    IMF_EXPORT
    const IMATH_NAMESPACE::Box2i &	displayWindow () const;

    IMF_EXPORT
    IMATH_NAMESPACE::Box2i &		dataWindow ();
    IMF_EXPORT
    const IMATH_NAMESPACE::Box2i &	dataWindow () const;

    IMF_EXPORT
    float &			pixelAspectRatio ();
    IMF_EXPORT
    const float &		pixelAspectRatio () const;

    IMF_EXPORT
    IMATH_NAMESPACE::V2f &		screenWindowCenter ();
    IMF_EXPORT
    const IMATH_NAMESPACE::V2f &		screenWindowCenter () const;

    IMF_EXPORT
    float &			screenWindowWidth ();
    IMF_EXPORT
    const float &		screenWindowWidth () const;

    IMF_EXPORT
    ChannelList &		channels ();
    IMF_EXPORT
    const ChannelList &		channels () const;

    IMF_EXPORT
    LineOrder &			lineOrder ();
    IMF_EXPORT
    const LineOrder &		lineOrder () const;

    IMF_EXPORT
    Compression &		compression ();
    IMF_EXPORT
    const Compression &		compression () const;


    //-----------------------------------------------------
    // Access to required attributes for multipart files
    // They are optional to non-multipart files and mandatory
    // for multipart files.
    //-----------------------------------------------------
    IMF_EXPORT
    void                        setName (const string& name);

    IMF_EXPORT
    string&                     name();
    IMF_EXPORT
    const string&               name() const;

    IMF_EXPORT
    bool                        hasName() const;

    IMF_EXPORT
    void                        setType (const string& Type);

    IMF_EXPORT
    string&                     type();
    IMF_EXPORT
    const string&               type() const;

    IMF_EXPORT
    bool                        hasType() const;

    IMF_EXPORT
    void                        setVersion (const int version);

    IMF_EXPORT
    int&                        version();
    IMF_EXPORT
    const int&                  version() const;

    IMF_EXPORT
    bool                        hasVersion() const;

    //
    // the chunkCount attribute is set automatically when a file is written.
    // There is no need to set it manually
    //
    IMF_EXPORT
    void                        setChunkCount(int chunks);
    IMF_EXPORT
    bool                        hasChunkCount() const;
    IMF_EXPORT
    const int &                 chunkCount() const;
    IMF_EXPORT
    int &                       chunkCount();

    
    //
    // for multipart files, return whether the file has a view string attribute
    // (for the deprecated single part multiview format EXR, see ImfMultiView.h)
    //
    IMF_EXPORT
    void                       setView(const string & view);
    IMF_EXPORT
    bool                       hasView() const;
    IMF_EXPORT
    string &                   view();
    IMF_EXPORT
    const string &             view() const;
    

    //----------------------------------------------------------------------
    // Tile Description:
    //
    // The tile description is a TileDescriptionAttribute whose name
    // is "tiles".  The "tiles" attribute must be present in any tiled
    // image file. When present, it describes various properties of the
    // tiles that make up the file.
    //
    // Convenience functions:
    //
    // setTileDescription(td)
    //     calls insert ("tiles", TileDescriptionAttribute (td))
    //
    // tileDescription()
    //     returns typedAttribute<TileDescriptionAttribute>("tiles").value()
    //
    // hasTileDescription()
    //     return findTypedAttribute<TileDescriptionAttribute>("tiles") != 0
    //
    //----------------------------------------------------------------------

    IMF_EXPORT
    void			setTileDescription (const TileDescription & td);

    IMF_EXPORT
    TileDescription &		tileDescription ();
    IMF_EXPORT
    const TileDescription &	tileDescription () const;

    IMF_EXPORT
    bool			hasTileDescription() const;


    //----------------------------------------------------------------------
    // Preview image:
    //
    // The preview image is a PreviewImageAttribute whose name is "preview".
    // This attribute is special -- while an image file is being written,
    // the pixels of the preview image can be changed repeatedly by calling
    // OutputFile::updatePreviewImage().
    //
    // Convenience functions:
    //
    // setPreviewImage(p)
    //     calls insert ("preview", PreviewImageAttribute (p))
    //
    // previewImage()
    //     returns typedAttribute<PreviewImageAttribute>("preview").value()
    //
    // hasPreviewImage()
    //     return findTypedAttribute<PreviewImageAttribute>("preview") != 0
    //
    //----------------------------------------------------------------------

    IMF_EXPORT
    void			setPreviewImage (const PreviewImage &p);

    IMF_EXPORT
    PreviewImage &		previewImage ();
    IMF_EXPORT
    const PreviewImage &	previewImage () const;

    IMF_EXPORT
    bool			hasPreviewImage () const;


    //-------------------------------------------------------------
    // Sanity check -- examines the header, and throws an exception
    // if it finds something wrong (empty display window, negative
    // pixel aspect ratio, unknown compression sceme etc.)
    //
    // set isTiled to true if you are checking a tiled/multi-res
    // header
    //-------------------------------------------------------------

    IMF_EXPORT
    void			sanityCheck (bool isTiled = false,
        			             bool isMultipartFile = false) const;


    //----------------------------------------------------------------
    // Maximum image size and maximim tile size:
    //
    // sanityCheck() will throw an exception if the width or height of
    // the data window exceeds the maximum image width or height, or
    // if the size of a tile exceeds the maximum tile width or height.
    // 
    // At program startup the maximum image and tile width and height
    // are set to zero, meaning that width and height are unlimited.
    //
    // Limiting image and tile width and height limits how much memory
    // will be allocated when a file is opened.  This can help protect
    // applications from running out of memory while trying to read
    // a damaged image file.
    //----------------------------------------------------------------

    IMF_EXPORT
    static void			setMaxImageSize (int maxWidth, int maxHeight);
    IMF_EXPORT
    static void			setMaxTileSize (int maxWidth, int maxHeight);

    //
    // Check if the header reads nothing.
    //
    IMF_EXPORT
    bool                        readsNothing();


    //------------------------------------------------------------------
    // Input and output:
    //
    // If the header contains a preview image attribute, then writeTo()
    // returns the position of that attribute in the output stream; this
    // information is used by OutputFile::updatePreviewImage().
    // If the header contains no preview image attribute, then writeTo()
    // returns 0.
    //------------------------------------------------------------------


    IMF_EXPORT
    Int64			writeTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
					 bool isTiled = false) const;

    IMF_EXPORT
    void			readFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
        			          int &version);
    

  private:

    AttributeMap		_map;

    bool                        _readsNothing;
};


//----------
// Iterators
//----------

class Header::Iterator
{
  public:

    IMF_EXPORT
    Iterator ();
    IMF_EXPORT
    Iterator (const Header::AttributeMap::iterator &i);

    IMF_EXPORT
    Iterator &			operator ++ ();
    IMF_EXPORT
    Iterator 			operator ++ (int);

    IMF_EXPORT
    const char *		name () const;
    IMF_EXPORT
    Attribute &			attribute () const;

  private:

    friend class Header::ConstIterator;

    Header::AttributeMap::iterator _i;
};


class Header::ConstIterator
{
  public:

    IMF_EXPORT
    ConstIterator ();
    IMF_EXPORT
    ConstIterator (const Header::AttributeMap::const_iterator &i);
    IMF_EXPORT
    ConstIterator (const Header::Iterator &other);

    IMF_EXPORT
    ConstIterator &		operator ++ ();
    IMF_EXPORT
    ConstIterator 		operator ++ (int);

    IMF_EXPORT
    const char *		name () const;
    IMF_EXPORT
    const Attribute &		attribute () const;

  private:

    friend bool operator == (const ConstIterator &, const ConstIterator &);
    friend bool operator != (const ConstIterator &, const ConstIterator &);

    Header::AttributeMap::const_iterator _i;
};


//------------------------------------------------------------------------
// Library initialization:
//
// In a multithreaded program, staticInitialize() must be called once
// during startup, before the program accesses any other functions or
// classes in the IlmImf library.  Calling staticInitialize() in this
// way avoids races during initialization of the library's global
// variables.
//
// Single-threaded programs are not required to call staticInitialize();
// initialization of the library's global variables happens automatically.
//
//------------------------------------------------------------------------

void IMF_EXPORT staticInitialize ();


//-----------------
// Inline Functions
//-----------------


inline
Header::Iterator::Iterator (): _i()
{
    // empty
}


inline
Header::Iterator::Iterator (const Header::AttributeMap::iterator &i): _i (i)
{
    // empty
}


inline Header::Iterator &		
Header::Iterator::operator ++ ()
{
    ++_i;
    return *this;
}


inline Header::Iterator 	
Header::Iterator::operator ++ (int)
{
    Iterator tmp = *this;
    ++_i;
    return tmp;
}


inline const char *
Header::Iterator::name () const
{
    return *_i->first;
}


inline Attribute &	
Header::Iterator::attribute () const
{
    return *_i->second;
}


inline
Header::ConstIterator::ConstIterator (): _i()
{
    // empty
}

inline
Header::ConstIterator::ConstIterator
    (const Header::AttributeMap::const_iterator &i): _i (i)
{
    // empty
}


inline
Header::ConstIterator::ConstIterator (const Header::Iterator &other):
    _i (other._i)
{
    // empty
}

inline Header::ConstIterator &
Header::ConstIterator::operator ++ ()
{
    ++_i;
    return *this;
}


inline Header::ConstIterator 		
Header::ConstIterator::operator ++ (int)
{
    ConstIterator tmp = *this;
    ++_i;
    return tmp;
}


inline const char *
Header::ConstIterator::name () const
{
    return *_i->first;
}


inline const Attribute &	
Header::ConstIterator::attribute () const
{
    return *_i->second;
}


inline bool
operator == (const Header::ConstIterator &x, const Header::ConstIterator &y)
{
    return x._i == y._i;
}


inline bool
operator != (const Header::ConstIterator &x, const Header::ConstIterator &y)
{
    return !(x == y);
}


//---------------------
// Template definitions
//---------------------

template <class T>
T &
Header::typedAttribute (const char name[])
{
    Attribute *attr = &(*this)[name];
    T *tattr = dynamic_cast <T*> (attr);

    if (tattr == 0)
	throw IEX_NAMESPACE::TypeExc ("Unexpected attribute type.");

    return *tattr;
}


template <class T>
const T &
Header::typedAttribute (const char name[]) const
{
    const Attribute *attr = &(*this)[name];
    const T *tattr = dynamic_cast <const T*> (attr);

    if (tattr == 0)
	throw IEX_NAMESPACE::TypeExc ("Unexpected attribute type.");

    return *tattr;
}


template <class T>
T &
Header::typedAttribute (const std::string &name)
{
    return typedAttribute<T> (name.c_str());
}


template <class T>
const T &
Header::typedAttribute (const std::string &name) const
{
    return typedAttribute<T> (name.c_str());
}


template <class T>
T *
Header::findTypedAttribute (const char name[])
{
    AttributeMap::iterator i = _map.find (name);
    return (i == _map.end())? 0: dynamic_cast <T*> (i->second);
}


template <class T>
const T *
Header::findTypedAttribute (const char name[]) const
{
    AttributeMap::const_iterator i = _map.find (name);
    return (i == _map.end())? 0: dynamic_cast <const T*> (i->second);
}


template <class T>
T *
Header::findTypedAttribute (const std::string &name)
{
    return findTypedAttribute<T> (name.c_str());
}


template <class T>
const T *
Header::findTypedAttribute (const std::string &name) const
{
    return findTypedAttribute<T> (name.c_str());
}


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
