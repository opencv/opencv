//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class TimeCodeAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_TIMECODE_ATTRIBUTE

#include "ImfTimeCodeAttribute.h"


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char *
TimeCodeAttribute::staticTypeName ()
{
    return "timecode";
}


template <>
IMF_EXPORT void
TimeCodeAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.timeAndFlags());
    Xdr::write <StreamIO> (os, _value.userData());
}


template <>
IMF_EXPORT void
TimeCodeAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    unsigned int tmp;

    Xdr::read <StreamIO> (is, tmp);
    _value.setTimeAndFlags (tmp);

    Xdr::read <StreamIO> (is, tmp);
    _value.setUserData (tmp);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::TimeCode>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
