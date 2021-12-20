//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class ChannelListAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_CHANNEL_LIST_ATTRIBUTE

#include "ImfChannelListAttribute.h"

#include "IexBaseExc.h"

#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

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
IMF_EXPORT const char *
ChannelListAttribute::staticTypeName ()
{
    return "chlist";
}

template <>
IMF_EXPORT void
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
IMF_EXPORT void
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

    //
    // prevent invalid values being written to PixelType enum
    // by forcing all unknown types to NUM_PIXELTYPES which is also an invalid
    // pixel type, but can be used as a PixelType enum value
    // (Header::sanityCheck will throw an exception when files with invalid PixelTypes are read)
    //
      if (type != OPENEXR_IMF_INTERNAL_NAMESPACE::UINT &&
          type != OPENEXR_IMF_INTERNAL_NAMESPACE::HALF &&
         type != OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT)
      {
          type = OPENEXR_IMF_INTERNAL_NAMESPACE::NUM_PIXELTYPES;
      }

	_value.insert (name, Channel (PixelType (type),
	                              xSampling,
	                              ySampling,
	                              pLinear));
    }
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::ChannelList>;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
