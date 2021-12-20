//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class TileDescriptionAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_STRING_VECTOR_ATTRIBUTE
#include "ImfTileDescriptionAttribute.h"


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char *
TileDescriptionAttribute::staticTypeName ()
{
    return "tiledesc";
}


template <>
IMF_EXPORT void
TileDescriptionAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.xSize);
    Xdr::write <StreamIO> (os, _value.ySize);

    unsigned char tmp = _value.mode | (_value.roundingMode << 4);
    Xdr::write <StreamIO> (os, tmp);
}


template <>
IMF_EXPORT void
TileDescriptionAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
					 int size,
					 int version)
{
    Xdr::read <StreamIO> (is, _value.xSize);
    Xdr::read <StreamIO> (is, _value.ySize);

    unsigned char tmp;
    Xdr::read <StreamIO> (is, tmp);

    //
    // four bits are allocated for 'mode' for future use (16 possible values)
    // but only values 0,1,2 are currently valid. '3' is a special valid enum value
    // that indicates bad values have been used
    //
    // roundingMode can only be 0 or 1, and 2 is a special enum value for 'bad enum'
    //
    unsigned char levelMode = tmp & 0x0f;
    if(levelMode > 3)
    {
        levelMode = 3;
    }

    _value.mode = LevelMode(levelMode);

    unsigned char levelRoundingMode = (tmp >> 4) & 0x0f;
    if(levelRoundingMode > 2)
    {
        levelRoundingMode = 2;
    }

    _value.roundingMode = LevelRoundingMode (levelRoundingMode);
    
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::TileDescription>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
