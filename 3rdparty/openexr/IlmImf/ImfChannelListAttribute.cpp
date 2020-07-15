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



//-----------------------------------------------------------------------------
//
//	class ChannelListAttribute
//
//-----------------------------------------------------------------------------

#include <ImfChannelListAttribute.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace {

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


template <>
const char *
ChannelListAttribute::staticTypeName ()
{
    return "chlist";
}

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
void
ChannelListAttribute::writeValueTo (OStream &os, int version) const
{
    for (ChannelList::ConstIterator i = _value.begin();
	 i != _value.end();
	 ++i)
    {
	//
	// Write name
	//

	Xdr::write <StreamIO> (os, i.name());

	//
	// Write Channel struct
	//

	Xdr::write <StreamIO> (os, int (i.channel().type));
	Xdr::write <StreamIO> (os, i.channel().pLinear);
	Xdr::pad   <StreamIO> (os, 3);
	Xdr::write <StreamIO> (os, i.channel().xSampling);
	Xdr::write <StreamIO> (os, i.channel().ySampling);
    }

    //
    // Write end of list marker
    //

    Xdr::write <StreamIO> (os, "");
}


template <>
void
ChannelListAttribute::readValueFrom (IStream &is,
                                     int size,
                                     int version)
{
    while (true)
    {
	//
	// Read name; zero length name means end of channel list
	//

	char name[Name::SIZE];
	Xdr::read <StreamIO> (is,Name::MAX_LENGTH,name);

	if (name[0] == 0)
	    break;

	checkIsNullTerminated (name, "channel name");

	//
	// Read Channel struct
	//

	int type;
	bool pLinear;
	int xSampling;
	int ySampling;

	Xdr::read <StreamIO> (is, type);
	Xdr::read <StreamIO> (is, pLinear);
	Xdr::skip <StreamIO> (is, 3);
	Xdr::read <StreamIO> (is, xSampling);
	Xdr::read <StreamIO> (is, ySampling);

	_value.insert (name, Channel (PixelType (type),
	                              xSampling,
	                              ySampling,
	                              pLinear));
    }
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
