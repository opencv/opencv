//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class StringAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_STRING_ATTRIBUTE

#include "ImfStringAttribute.h"


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

//#if defined(__MINGW32__)
//template <>
//IMF_EXPORT
//TypedAttribute<std::string>::~TypedAttribute ()
//{
//}
//#endif

template <>
IMF_EXPORT const char *
TypedAttribute<std::string>::staticTypeName ()
{
    return "string";
}


template <>
IMF_EXPORT void
TypedAttribute<std::string>::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    int size = _value.size();

    for (int i = 0; i < size; i++)
	Xdr::write <StreamIO> (os, _value[i]);
}


template <>
IMF_EXPORT void
TypedAttribute<std::string>::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    _value.resize (size);

    for (int i = 0; i < size; i++)
	Xdr::read <StreamIO> (is, _value[i]);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<std::string>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
