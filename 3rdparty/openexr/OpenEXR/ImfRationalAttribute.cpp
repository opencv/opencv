//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class RationalAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_RATIONAL_ATTRIBUTE

#include "ImfRationalAttribute.h"


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char *
RationalAttribute::staticTypeName ()
{
    return "rational";
}


template <>
IMF_EXPORT void
RationalAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.n);
    Xdr::write <StreamIO> (os, _value.d);
}


template <>
IMF_EXPORT void
RationalAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.n);
    Xdr::read <StreamIO> (is, _value.d);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::Rational>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
