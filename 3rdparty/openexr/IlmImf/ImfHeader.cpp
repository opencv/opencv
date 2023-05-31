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



//-----------------------------------------------------------------------------
//
//	class Header
//
//-----------------------------------------------------------------------------

#include <ImfHeader.h>
#include <ImfStdIO.h>
#include <ImfVersion.h>
#include <ImfCompressor.h>
#include <ImfMisc.h>
#include <ImfBoxAttribute.h>
#include <ImfChannelListAttribute.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfCompressionAttribute.h>
#include <ImfDeepImageStateAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfDwaCompressor.h>
#include <ImfEnvmapAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfFloatVectorAttribute.h>
#include <ImfIntAttribute.h>
#include <ImfKeyCodeAttribute.h>
#include <ImfLineOrderAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOpaqueAttribute.h>
#include <ImfPreviewImageAttribute.h>
#include <ImfRationalAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfStringVectorAttribute.h>
#include <ImfTileDescriptionAttribute.h>
#include <ImfTimeCodeAttribute.h>
#include <ImfVecAttribute.h>
#include <ImfPartType.h>
#include "IlmThreadMutex.h"
#include "Iex.h"
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace std;
using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::V2i;
using IMATH_NAMESPACE::V2f;
using ILMTHREAD_NAMESPACE::Mutex;
using ILMTHREAD_NAMESPACE::Lock;


namespace {

int maxImageWidth = 0;
int maxImageHeight = 0;
int maxTileWidth = 0;
int maxTileHeight = 0;


void
initialize (Header &header,
	    const Box2i &displayWindow,
	    const Box2i &dataWindow,
	    float pixelAspectRatio,
	    const V2f &screenWindowCenter,
	    float screenWindowWidth,
	    LineOrder lineOrder,
	    Compression compression)
{
    header.insert ("displayWindow", Box2iAttribute (displayWindow));
    header.insert ("dataWindow", Box2iAttribute (dataWindow));
    header.insert ("pixelAspectRatio", FloatAttribute (pixelAspectRatio));
    header.insert ("screenWindowCenter", V2fAttribute (screenWindowCenter));
    header.insert ("screenWindowWidth", FloatAttribute (screenWindowWidth));
    header.insert ("lineOrder", LineOrderAttribute (lineOrder));
    header.insert ("compression", CompressionAttribute (compression));
    header.insert ("channels", ChannelListAttribute ());
}

template <size_t N>
void checkIsNullTerminated (const char (&str)[N], const char *what)
{
	for (size_t i = 0; i < N; ++i) {
		if (str[i] == '\0')
			return;
	}
	std::stringstream s;
	s << "Invalid " << what << ": it is more than " << (N - 1) 
		<< " characters long.";
	throw IEX_NAMESPACE::InputExc(s);
}

} // namespace


Header::Header (int width,
		int height,
		float pixelAspectRatio,
		const V2f &screenWindowCenter,
		float screenWindowWidth,
		LineOrder lineOrder,
		Compression compression)
:
    _map()
{
    staticInitialize();

    Box2i displayWindow (V2i (0, 0), V2i (width - 1, height - 1));

    initialize (*this,
		displayWindow,
		displayWindow,
		pixelAspectRatio,
		screenWindowCenter,
		screenWindowWidth,
		lineOrder,
		compression);
}


Header::Header (int width,
		int height,
		const Box2i &dataWindow,
		float pixelAspectRatio,
		const V2f &screenWindowCenter,
		float screenWindowWidth,
		LineOrder lineOrder,
		Compression compression)
:
    _map()
{
    staticInitialize();

    Box2i displayWindow (V2i (0, 0), V2i (width - 1, height - 1));

    initialize (*this,
		displayWindow,
		dataWindow,
		pixelAspectRatio,
		screenWindowCenter,
		screenWindowWidth,
		lineOrder,
		compression);
}


Header::Header (const Box2i &displayWindow,
		const Box2i &dataWindow,
		float pixelAspectRatio,
		const V2f &screenWindowCenter,
		float screenWindowWidth,
		LineOrder lineOrder,
		Compression compression)
:
    _map()
{
    staticInitialize();

    initialize (*this,
		displayWindow,
		dataWindow,
		pixelAspectRatio,
		screenWindowCenter,
		screenWindowWidth,
		lineOrder,
		compression);
}


Header::Header (const Header &other): _map()
{
    for (AttributeMap::const_iterator i = other._map.begin();
	 i != other._map.end();
	 ++i)
    {
	insert (*i->first, *i->second);
    }
}


Header::~Header ()
{
    for (AttributeMap::iterator i = _map.begin();
	 i != _map.end();
	 ++i)
    {
	 delete i->second;
    }
}


Header &		
Header::operator = (const Header &other)
{
    if (this != &other)
    {
	for (AttributeMap::iterator i = _map.begin();
	     i != _map.end();
	     ++i)
	{
	     delete i->second;
	}

	_map.erase (_map.begin(), _map.end());

	for (AttributeMap::const_iterator i = other._map.begin();
	     i != other._map.end();
	     ++i)
	{
	    insert (*i->first, *i->second);
	}
    }

    return *this;
}


void
Header::erase (const char name[])
{
    if (name[0] == 0)
        THROW (IEX_NAMESPACE::ArgExc, "Image attribute name cannot be an empty string.");
    
    
    AttributeMap::iterator i = _map.find (name);
    if (i != _map.end())
        _map.erase (i);

}


void
Header::erase (const string &name)
{
    erase (name.c_str());
}
    
    
void
Header::insert (const char name[], const Attribute &attribute)
{
    if (name[0] == 0)
	THROW (IEX_NAMESPACE::ArgExc, "Image attribute name cannot be an empty string.");

    AttributeMap::iterator i = _map.find (name);

    if (i == _map.end())
    {
	Attribute *tmp = attribute.copy();

	try
	{
	    _map[name] = tmp;
	}
	catch (...)
	{
	    delete tmp;
	    throw;
	}
    }
    else
    {
	if (strcmp (i->second->typeName(), attribute.typeName()))
	    THROW (IEX_NAMESPACE::TypeExc, "Cannot assign a value of "
				 "type \"" << attribute.typeName() << "\" "
				 "to image attribute \"" << name << "\" of "
				 "type \"" << i->second->typeName() << "\".");

	Attribute *tmp = attribute.copy();
	delete i->second;
	i->second = tmp;
    }
}


void
Header::insert (const string &name, const Attribute &attribute)
{
    insert (name.c_str(), attribute);
}


Attribute &		
Header::operator [] (const char name[])
{
    AttributeMap::iterator i = _map.find (name);

    if (i == _map.end())
	THROW (IEX_NAMESPACE::ArgExc, "Cannot find image attribute \"" << name << "\".");

    return *i->second;
}


const Attribute &	
Header::operator [] (const char name[]) const
{
    AttributeMap::const_iterator i = _map.find (name);

    if (i == _map.end())
	THROW (IEX_NAMESPACE::ArgExc, "Cannot find image attribute \"" << name << "\".");

    return *i->second;
}


Attribute &		
Header::operator [] (const string &name)
{
    return this->operator[] (name.c_str());
}


const Attribute &	
Header::operator [] (const string &name) const
{
    return this->operator[] (name.c_str());
}


Header::Iterator
Header::begin ()
{
    return _map.begin();
}


Header::ConstIterator
Header::begin () const
{
    return _map.begin();
}


Header::Iterator
Header::end ()
{
    return _map.end();
}


Header::ConstIterator
Header::end () const
{
    return _map.end();
}


Header::Iterator
Header::find (const char name[])
{
    return _map.find (name);
}


Header::ConstIterator
Header::find (const char name[]) const
{
    return _map.find (name);
}


Header::Iterator
Header::find (const string &name)
{
    return find (name.c_str());
}


Header::ConstIterator
Header::find (const string &name) const
{
    return find (name.c_str());
}


IMATH_NAMESPACE::Box2i &	
Header::displayWindow ()
{
    return static_cast <Box2iAttribute &>
	((*this)["displayWindow"]).value();
}


const IMATH_NAMESPACE::Box2i &
Header::displayWindow () const
{
    return static_cast <const Box2iAttribute &>
	((*this)["displayWindow"]).value();
}


IMATH_NAMESPACE::Box2i &	
Header::dataWindow ()
{
    return static_cast <Box2iAttribute &>
	((*this)["dataWindow"]).value();
}


const IMATH_NAMESPACE::Box2i &
Header::dataWindow () const
{
    return static_cast <const Box2iAttribute &>
	((*this)["dataWindow"]).value();
}


float &		
Header::pixelAspectRatio ()
{
    return static_cast <FloatAttribute &>
	((*this)["pixelAspectRatio"]).value();
}


const float &	
Header::pixelAspectRatio () const
{
    return static_cast <const FloatAttribute &>
	((*this)["pixelAspectRatio"]).value();
}


IMATH_NAMESPACE::V2f &	
Header::screenWindowCenter ()
{
    return static_cast <V2fAttribute &>
	((*this)["screenWindowCenter"]).value();
}


const IMATH_NAMESPACE::V2f &	
Header::screenWindowCenter () const
{
    return static_cast <const V2fAttribute &>
	((*this)["screenWindowCenter"]).value();
}


float &		
Header::screenWindowWidth ()
{
    return static_cast <FloatAttribute &>
	((*this)["screenWindowWidth"]).value();
}


const float &	
Header::screenWindowWidth () const
{
    return static_cast <const FloatAttribute &>
	((*this)["screenWindowWidth"]).value();
}


ChannelList &	
Header::channels ()
{
    return static_cast <ChannelListAttribute &>
	((*this)["channels"]).value();
}


const ChannelList &	
Header::channels () const
{
    return static_cast <const ChannelListAttribute &>
	((*this)["channels"]).value();
}


LineOrder &
Header::lineOrder ()
{
    return static_cast <LineOrderAttribute &>
	((*this)["lineOrder"]).value();
}


const LineOrder &
Header::lineOrder () const
{
    return static_cast <const LineOrderAttribute &>
	((*this)["lineOrder"]).value();
}


Compression &
Header::compression ()
{
    return static_cast <CompressionAttribute &>
	((*this)["compression"]).value();
}


const Compression &
Header::compression () const
{
    return static_cast <const CompressionAttribute &>
	((*this)["compression"]).value();
}


void
Header::setName(const string& name)
{
    insert ("name", StringAttribute (name));
}


bool
Header::hasName() const
{
    return findTypedAttribute <StringAttribute> ("name") != 0;
}


string &
Header::name()
{
    return typedAttribute <StringAttribute> ("name").value();
}


const string &
Header::name() const
{
    return typedAttribute <StringAttribute> ("name").value();
}


void
Header::setType(const string& type)
{
    if (isSupportedType(type) == false)
    {
        throw IEX_NAMESPACE::ArgExc (type + "is not a supported image type." +
                           "The following are supported: " +
                           SCANLINEIMAGE + ", " +
                           TILEDIMAGE + ", " +
                           DEEPSCANLINE + " or " +
                           DEEPTILE + ".");
    }

    insert ("type", StringAttribute (type));

    // (TODO) Should we do it here?
    if (isDeepData(type) && hasVersion() == false)
    {
        setVersion(1);
    }
}


bool
Header::hasType() const
{
    return findTypedAttribute <StringAttribute> ("type") != 0;
}


string &
Header::type()
{
    return typedAttribute <StringAttribute> ("type").value();
}


const string &
Header::type() const
{
    return typedAttribute <StringAttribute> ("type").value();
}


void
Header::setView(const string& view)
{
    insert ("view", StringAttribute (view));
}


bool
Header::hasView() const
{
    return findTypedAttribute <StringAttribute> ("view") != 0;
}


string &
Header::view()
{
    return typedAttribute <StringAttribute> ("view").value();
}


const string &
Header::view() const
{
    return typedAttribute <StringAttribute> ("view").value();
}


void
Header::setVersion(const int version)
{
    if (version != 1)
    {
        throw IEX_NAMESPACE::ArgExc ("We can only process version 1");
    }

    insert ("version", IntAttribute (version));
}


bool
Header::hasVersion() const
{
    return findTypedAttribute <IntAttribute> ("version") != 0;
}


int &
Header::version()
{
    return typedAttribute <IntAttribute> ("version").value();
}


const int &
Header::version() const
{
    return typedAttribute <IntAttribute> ("version").value();
}

void 
Header::setChunkCount(int chunks)
{
    insert("chunkCount",IntAttribute(chunks));
}

bool 
Header::hasChunkCount() const
{
   return findTypedAttribute<IntAttribute>("chunkCount") != 0;
}

int& 
Header::chunkCount()
{
    return typedAttribute <IntAttribute> ("chunkCount").value();
}

const int& 
Header::chunkCount() const
{
    return typedAttribute <IntAttribute> ("chunkCount").value();
}

void
Header::setTileDescription(const TileDescription& td)
{
    insert ("tiles", TileDescriptionAttribute (td));
}


bool
Header::hasTileDescription() const
{
    return findTypedAttribute <TileDescriptionAttribute> ("tiles") != 0;
}


TileDescription &
Header::tileDescription ()
{
    return typedAttribute <TileDescriptionAttribute> ("tiles").value();
}


const TileDescription &
Header::tileDescription () const
{
    return typedAttribute <TileDescriptionAttribute> ("tiles").value();
}

void		
Header::setPreviewImage (const PreviewImage &pi)
{
    insert ("preview", PreviewImageAttribute (pi));
}


PreviewImage &
Header::previewImage ()
{
    return typedAttribute <PreviewImageAttribute> ("preview").value();
}


const PreviewImage &
Header::previewImage () const
{
    return typedAttribute <PreviewImageAttribute> ("preview").value();
}


bool		
Header::hasPreviewImage () const
{
    return findTypedAttribute <PreviewImageAttribute> ("preview") != 0;
}


void		
Header::sanityCheck (bool isTiled, bool isMultipartFile) const
{
    //
    // The display window and the data window must each
    // contain at least one pixel.  In addition, the
    // coordinates of the window corners must be small
    // enough to keep expressions like max-min+1 or
    // max+min from overflowing.
    //

    const Box2i &displayWindow = this->displayWindow();

    if (displayWindow.min.x > displayWindow.max.x ||
	displayWindow.min.y > displayWindow.max.y ||
	displayWindow.min.x <= -(INT_MAX / 2) ||
	displayWindow.min.y <= -(INT_MAX / 2) ||
	displayWindow.max.x >=  (INT_MAX / 2) ||
	displayWindow.max.y >=  (INT_MAX / 2))
    {
	throw IEX_NAMESPACE::ArgExc ("Invalid display window in image header.");
    }

    const Box2i &dataWindow = this->dataWindow();

    if (dataWindow.min.x > dataWindow.max.x ||
	dataWindow.min.y > dataWindow.max.y ||
	dataWindow.min.x <= -(INT_MAX / 2) ||
	dataWindow.min.y <= -(INT_MAX / 2) ||
	dataWindow.max.x >=  (INT_MAX / 2) ||
	dataWindow.max.y >=  (INT_MAX / 2))
    {
	throw IEX_NAMESPACE::ArgExc ("Invalid data window in image header.");
    }

    if (maxImageWidth > 0 &&
        maxImageWidth < (dataWindow.max.x - dataWindow.min.x + 1))
    {
	THROW (IEX_NAMESPACE::ArgExc, "The width of the data window exceeds the "
			    "maximum width of " << maxImageWidth << "pixels.");
    }

    if (maxImageHeight > 0 &&
	maxImageHeight < dataWindow.max.y - dataWindow.min.y + 1)
    {
	THROW (IEX_NAMESPACE::ArgExc, "The width of the data window exceeds the "
			    "maximum width of " << maxImageHeight << "pixels.");
    }

   // chunk table must be smaller than the maximum image area
   // (only reachable for unknown types or damaged files: will have thrown earlier
   //  for regular image types)
   if( maxImageHeight>0 && maxImageWidth>0 && 
       hasChunkCount() && chunkCount()>Int64(maxImageWidth)*Int64(maxImageHeight))
   {
       THROW (IEX_NAMESPACE::ArgExc, "chunkCount exceeds maximum area of "
       << Int64(maxImageWidth)*Int64(maxImageHeight) << " pixels." );
       
   }


    //
    // The pixel aspect ratio must be greater than 0.
    // In applications, numbers like the the display or
    // data window dimensions are likely to be multiplied
    // or divided by the pixel aspect ratio; to avoid
    // arithmetic exceptions, we limit the pixel aspect
    // ratio to a range that is smaller than theoretically
    // possible (real aspect ratios are likely to be close
    // to 1.0 anyway).
    //

    float pixelAspectRatio = this->pixelAspectRatio();

    const float MIN_PIXEL_ASPECT_RATIO = 1e-6f;
    const float MAX_PIXEL_ASPECT_RATIO = 1e+6f;

    if (pixelAspectRatio < MIN_PIXEL_ASPECT_RATIO ||
	pixelAspectRatio > MAX_PIXEL_ASPECT_RATIO)
    {
	throw IEX_NAMESPACE::ArgExc ("Invalid pixel aspect ratio in image header.");
    }

    //
    // The screen window width must not be less than 0.
    // The size of the screen window can vary over a wide
    // range (fish-eye lens to astronomical telescope),
    // so we can't limit the screen window width to a
    // small range.
    //

    float screenWindowWidth = this->screenWindowWidth();

    if (screenWindowWidth < 0)
	throw IEX_NAMESPACE::ArgExc ("Invalid screen window width in image header.");

    //
    // If the file has multiple parts, verify that each header has attribute
    // name and type.
    // (TODO) We may want to check more stuff here.
    //

    if (isMultipartFile)
    {
        if (!hasName())
        {
            throw IEX_NAMESPACE::ArgExc ("Headers in a multipart file should"
                               " have name attribute.");
        }

        if (!hasType())
        {
            throw IEX_NAMESPACE::ArgExc ("Headers in a multipart file should"
                               " have type attribute.");
        }

    }
    
    const std::string & part_type=hasType() ? type() : "";
    
    if(part_type!="" && !isSupportedType(part_type))
    {
        //
        // skip remaining sanity checks with unsupported types - they may not hold
        //
        return;
    }
    
   
    //
    // If the file is tiled, verify that the tile description has reasonable
    // values and check to see if the lineOrder is one of the predefined 3.
    // If the file is not tiled, then the lineOrder can only be INCREASING_Y
    // or DECREASING_Y.
    //

    LineOrder lineOrder = this->lineOrder();

    if (isTiled)
    {
	if (!hasTileDescription())
	{
	    throw IEX_NAMESPACE::ArgExc ("Tiled image has no tile "
			       "description attribute.");
	}

	const TileDescription &tileDesc = tileDescription();

	if (tileDesc.xSize <= 0 || tileDesc.ySize <= 0)
	    throw IEX_NAMESPACE::ArgExc ("Invalid tile size in image header.");

	if (maxTileWidth > 0 &&
	    maxTileWidth < int(tileDesc.xSize))
	{
	    THROW (IEX_NAMESPACE::ArgExc, "The width of the tiles exceeds the maximum "
				"width of " << maxTileWidth << "pixels.");
	}

	if (maxTileHeight > 0 &&
	    maxTileHeight < int(tileDesc.ySize))
	{
	    THROW (IEX_NAMESPACE::ArgExc, "The width of the tiles exceeds the maximum "
				"width of " << maxTileHeight << "pixels.");
	}

	if (tileDesc.mode != ONE_LEVEL &&
	    tileDesc.mode != MIPMAP_LEVELS &&
	    tileDesc.mode != RIPMAP_LEVELS)
	    throw IEX_NAMESPACE::ArgExc ("Invalid level mode in image header.");

	if (tileDesc.roundingMode != ROUND_UP &&
	    tileDesc.roundingMode != ROUND_DOWN)
	    throw IEX_NAMESPACE::ArgExc ("Invalid level rounding mode in image header.");

	if (lineOrder != INCREASING_Y &&
	    lineOrder != DECREASING_Y &&
	    lineOrder != RANDOM_Y)
	    throw IEX_NAMESPACE::ArgExc ("Invalid line order in image header.");
    }
    else
    {
        if (lineOrder != INCREASING_Y &&
            lineOrder != DECREASING_Y)
            throw IEX_NAMESPACE::ArgExc ("Invalid line order in image header.");
        
        
    }

    //
    // The compression method must be one of the predefined values.
    //

    if (!isValidCompression (this->compression()))
  	throw IEX_NAMESPACE::ArgExc ("Unknown compression type in image header.");
    
    if(isDeepData(part_type))
    {
        if (!isValidDeepCompression (this->compression()))
            throw IEX_NAMESPACE::ArgExc ("Compression type in header not valid for deep data");
    }

    //
    // Check the channel list:
    //
    // If the file is tiled then for each channel, the type must be one of the
    // predefined values, and the x and y sampling must both be 1.
    //
    // If the file is not tiled then for each channel, the type must be one
    // of the predefined values, the x and y coordinates of the data window's
    // upper left corner must be divisible by the x and y subsampling factors,
    // and the width and height of the data window must be divisible by the
    // x and y subsampling factors.
    //

    const ChannelList &channels = this->channels();
    
    if (isTiled)
    {
	for (ChannelList::ConstIterator i = channels.begin();
	     i != channels.end();
	     ++i)
	{
	    if (i.channel().type != OPENEXR_IMF_INTERNAL_NAMESPACE::UINT &&
		    i.channel().type != OPENEXR_IMF_INTERNAL_NAMESPACE::HALF &&
		    i.channel().type != OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "Pixel type of \"" << i.name() << "\" "
			            "image channel is invalid.");
	    }

	    if (i.channel().xSampling != 1)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "The x subsampling factor for the "
				    "\"" << i.name() << "\" channel "
				    "is not 1.");
	    }	

	    if (i.channel().ySampling != 1)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "The y subsampling factor for the "
				    "\"" << i.name() << "\" channel "
				    "is not 1.");
	    }	
	}
    }
    else
    {
	for (ChannelList::ConstIterator i = channels.begin();
	     i != channels.end();
	     ++i)
	{
	    if (i.channel().type != OPENEXR_IMF_INTERNAL_NAMESPACE::UINT &&
		    i.channel().type != OPENEXR_IMF_INTERNAL_NAMESPACE::HALF &&
		    i.channel().type != OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "Pixel type of \"" << i.name() << "\" "
			            "image channel is invalid.");
	    }

	    if (i.channel().xSampling < 1)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "The x subsampling factor for the "
				    "\"" << i.name() << "\" channel "
				    "is invalid.");
	    }

	    if (i.channel().ySampling < 1)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "The y subsampling factor for the "
				    "\"" << i.name() << "\" channel "
				    "is invalid.");
	    }

	    if (dataWindow.min.x % i.channel().xSampling)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "The minimum x coordinate of the "
				    "image's data window is not a multiple "
				    "of the x subsampling factor of "
				    "the \"" << i.name() << "\" channel.");
	    }

	    if (dataWindow.min.y % i.channel().ySampling)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "The minimum y coordinate of the "
				    "image's data window is not a multiple "
				    "of the y subsampling factor of "
				    "the \"" << i.name() << "\" channel.");
	    }

	    if ((dataWindow.max.x - dataWindow.min.x + 1) %
		    i.channel().xSampling)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "Number of pixels per row in the "
				    "image's data window is not a multiple "
				    "of the x subsampling factor of "
				    "the \"" << i.name() << "\" channel.");
	    }

	    if ((dataWindow.max.y - dataWindow.min.y + 1) %
		    i.channel().ySampling)
	    {
		THROW (IEX_NAMESPACE::ArgExc, "Number of pixels per column in the "
				    "image's data window is not a multiple "
				    "of the y subsampling factor of "
				    "the \"" << i.name() << "\" channel.");
	    }
	}
    }
}


void		
Header::setMaxImageSize (int maxWidth, int maxHeight)
{
    maxImageWidth = maxWidth;
    maxImageHeight = maxHeight;
}


void		
Header::setMaxTileSize (int maxWidth, int maxHeight)
{
    maxTileWidth = maxWidth;
    maxTileHeight = maxHeight;
}


bool
Header::readsNothing()
{
    return _readsNothing;
}


Int64
Header::writeTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, bool isTiled) const
{
    //
    // Write a "magic number" to identify the file as an image file.
    // Write the current file format version number.
    //

    int version = EXR_VERSION;

    //
    // Write all attributes.  If we have a preview image attribute,
    // keep track of its position in the file.
    //

    Int64 previewPosition = 0;

    const Attribute *preview =
	    findTypedAttribute <PreviewImageAttribute> ("preview");

    for (ConstIterator i = begin(); i != end(); ++i)
    {
	//
	// Write the attribute's name and type.
	//

	OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, i.name());
	OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, i.attribute().typeName());

	//
	// Write the size of the attribute value,
	// and the value itself.
	//

	StdOSStream oss;
	i.attribute().writeValueTo (oss, version);

	std::string s = oss.str();
	OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, (int) s.length());

	if (&i.attribute() == preview)
	    previewPosition = os.tellp();

	os.write (s.data(), int(s.length()));
    }

    //
    // Write zero-length attribute name to mark the end of the header.
    //

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, "");

    return previewPosition;
}


void
Header::readFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int &version)
{
    //
    // Read all attributes.
    //

    int attrCount = 0;

    while (true)
    {
	//
	// Read the name of the attribute.
	// A zero-length attribute name indicates the end of the header.
	//

	char name[Name::SIZE];
	OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, Name::MAX_LENGTH, name);

	if (name[0] == 0)
	{
	    if (attrCount == 0) _readsNothing = true;
	    else                _readsNothing = false;
	    break;
	}

	attrCount++;

	checkIsNullTerminated (name, "attribute name");

	//
	// Read the attribute type and the size of the attribute value.
	//

	char typeName[Name::SIZE];
	int size;

	OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, Name::MAX_LENGTH, typeName);
	checkIsNullTerminated (typeName, "attribute type name");
	OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, size);

	AttributeMap::iterator i = _map.find (name);

	if (i != _map.end())
	{
	    //
	    // The attribute already exists (for example,
	    // because it is a predefined attribute).
	    // Read the attribute's new value from the file.
	    //

	    if (strncmp (i->second->typeName(), typeName, sizeof (typeName)))
		THROW (IEX_NAMESPACE::InputExc, "Unexpected type for image attribute "
				      "\"" << name << "\".");

	    i->second->readValueFrom (is, size, version);
	}
	else
	{
	    //
	    // The new attribute does not exist yet.
	    // If the attribute type is of a known type,
	    // read the attribute value.  If the attribute
	    // is of an unknown type, read its value and
	    // store it as an OpaqueAttribute.
	    //

	    Attribute *attr;

	    if (Attribute::knownType (typeName))
		attr = Attribute::newAttribute (typeName);
	    else
		attr = new OpaqueAttribute (typeName);

	    try
	    {
		attr->readValueFrom (is, size, version);
		_map[name] = attr;
	    }
	    catch (...)
	    {
		delete attr;
		throw;
	    }
	}
    }
}


void
staticInitialize ()
{
    static Mutex criticalSection;
    Lock lock (criticalSection);

    static bool initialized = false;

    if (!initialized)
    {
	//
	// One-time initialization -- register
	// some predefined attribute types.
	//
	
	Box2fAttribute::registerAttributeType();
	Box2iAttribute::registerAttributeType();
	ChannelListAttribute::registerAttributeType();
	CompressionAttribute::registerAttributeType();
	ChromaticitiesAttribute::registerAttributeType();
	DeepImageStateAttribute::registerAttributeType();
	DoubleAttribute::registerAttributeType();
	EnvmapAttribute::registerAttributeType();
	FloatAttribute::registerAttributeType();
	FloatVectorAttribute::registerAttributeType();
	IntAttribute::registerAttributeType();
	KeyCodeAttribute::registerAttributeType();
	LineOrderAttribute::registerAttributeType();
	M33dAttribute::registerAttributeType();
	M33fAttribute::registerAttributeType();
	M44dAttribute::registerAttributeType();
	M44fAttribute::registerAttributeType();
	PreviewImageAttribute::registerAttributeType();
	RationalAttribute::registerAttributeType();
	StringAttribute::registerAttributeType();
        StringVectorAttribute::registerAttributeType();
	TileDescriptionAttribute::registerAttributeType();
	TimeCodeAttribute::registerAttributeType();
	V2dAttribute::registerAttributeType();
	V2fAttribute::registerAttributeType();
	V2iAttribute::registerAttributeType();
	V3dAttribute::registerAttributeType();
	V3fAttribute::registerAttributeType();
	V3iAttribute::registerAttributeType();
	DwaCompressor::initializeFuncs();

	initialized = true;
    }
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
