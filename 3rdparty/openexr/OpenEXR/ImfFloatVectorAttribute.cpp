//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class FloatVectorAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_FLOAT_VECTOR_ATTRIBUTE
#include <ImfFloatVectorAttribute.h>

#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;


template <>
IMF_EXPORT const char *
FloatVectorAttribute::staticTypeName ()
{
    return "floatvector";
}


template <>
IMF_EXPORT void
FloatVectorAttribute::writeValueTo
    (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    int n = _value.size();

    for (int i = 0; i < n; ++i)
        Xdr::write <StreamIO> (os, _value[i]);
}


template <>
IMF_EXPORT void
FloatVectorAttribute::readValueFrom
    (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    int n = size / Xdr::size<float>();
    _value.resize (n);

    for (int i = 0; i < n; ++i)
       Xdr::read <StreamIO> (is, _value[i]);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::FloatVector>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
