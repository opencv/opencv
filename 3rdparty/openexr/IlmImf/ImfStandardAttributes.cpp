///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2003, Industrial Light & Magic, a division of Lucas
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
//	Optional Standard Attributes
//
//-----------------------------------------------------------------------------

#include <ImfStandardAttributes.h>


#define IMF_STRING(name) #name

#define IMF_STD_ATTRIBUTE_IMP(name,suffix,type)				 \
									 \
    void								 \
    add##suffix (Header &header, const type &value)			 \
    {									 \
	header.insert (IMF_STRING (name), TypedAttribute<type> (value)); \
    }									 \
									 \
    bool								 \
    has##suffix (const Header &header)					 \
    {									 \
	return header.findTypedAttribute <TypedAttribute <type> >	 \
		(IMF_STRING (name)) != 0;				 \
    }									 \
									 \
    const TypedAttribute<type> &					 \
    name##Attribute (const Header &header)				 \
    {									 \
	return header.typedAttribute <TypedAttribute <type> >		 \
		(IMF_STRING (name));					 \
    }									 \
									 \
    TypedAttribute<type> &						 \
    name##Attribute (Header &header)					 \
    {									 \
	return header.typedAttribute <TypedAttribute <type> >		 \
		(IMF_STRING (name));					 \
    }									 \
									 \
    const type &							 \
    name (const Header &header)						 \
    {									 \
	return name##Attribute(header).value();				 \
    }									 \
									 \
    type &								 \
    name (Header &header)						 \
    {									 \
	return name##Attribute(header).value();				 \
    }


namespace Imf {

   
IMF_STD_ATTRIBUTE_IMP (chromaticities, Chromaticities, Chromaticities)
IMF_STD_ATTRIBUTE_IMP (whiteLuminance, WhiteLuminance, float)
IMF_STD_ATTRIBUTE_IMP (adoptedNeutral, AdoptedNeutral, Imath::V2f)
IMF_STD_ATTRIBUTE_IMP (renderingTransform, RenderingTransform, std::string)
IMF_STD_ATTRIBUTE_IMP (lookModTransform, LookModTransform, std::string)
IMF_STD_ATTRIBUTE_IMP (xDensity, XDensity, float)
IMF_STD_ATTRIBUTE_IMP (owner, Owner, std::string)
IMF_STD_ATTRIBUTE_IMP (comments, Comments, std::string)
IMF_STD_ATTRIBUTE_IMP (capDate, CapDate, std::string)
IMF_STD_ATTRIBUTE_IMP (utcOffset, UtcOffset, float)
IMF_STD_ATTRIBUTE_IMP (longitude, Longitude, float)
IMF_STD_ATTRIBUTE_IMP (latitude, Latitude, float)
IMF_STD_ATTRIBUTE_IMP (altitude, Altitude, float)
IMF_STD_ATTRIBUTE_IMP (focus, Focus, float)
IMF_STD_ATTRIBUTE_IMP (expTime, ExpTime, float)
IMF_STD_ATTRIBUTE_IMP (aperture, Aperture, float)
IMF_STD_ATTRIBUTE_IMP (isoSpeed, IsoSpeed, float)
IMF_STD_ATTRIBUTE_IMP (envmap, Envmap, Envmap)
IMF_STD_ATTRIBUTE_IMP (keyCode, KeyCode, KeyCode)
IMF_STD_ATTRIBUTE_IMP (timeCode, TimeCode, TimeCode)
IMF_STD_ATTRIBUTE_IMP (wrapmodes, Wrapmodes, std::string)
IMF_STD_ATTRIBUTE_IMP (framesPerSecond, FramesPerSecond, Rational)
IMF_STD_ATTRIBUTE_IMP (multiView, MultiView, StringVector)
IMF_STD_ATTRIBUTE_IMP (worldToCamera, WorldToCamera, Imath::M44f)
IMF_STD_ATTRIBUTE_IMP (worldToNDC, WorldToNDC, Imath::M44f)

} // namespace Imf
