//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class DeepImageStateAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_DEEP_IMAGE_STATE_ATTRIBUTE

#include "ImfDeepImageStateAttribute.h"

#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char *
DeepImageStateAttribute::staticTypeName ()
{
    return "deepImageState";
}


template <>
IMF_EXPORT void
DeepImageStateAttribute::writeValueTo
    (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    unsigned char tmp = _value;
    Xdr::write <StreamIO> (os, tmp);
}


template <>
IMF_EXPORT void
DeepImageStateAttribute::readValueFrom
    (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    unsigned char tmp;
    Xdr::read <StreamIO> (is, tmp);
    _value = DeepImageState (tmp);
}

template <>
IMF_EXPORT void
DeepImageStateAttribute::copyValueFrom (const OPENEXR_IMF_INTERNAL_NAMESPACE::Attribute &other)
{
    _value = cast(other).value();

}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::DeepImageState>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
