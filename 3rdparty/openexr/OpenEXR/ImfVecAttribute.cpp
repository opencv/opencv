//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class V2iAttribute
//	class V2fAttribute
//	class V2dAttribute
//	class V3iAttribute
//	class V3fAttribute
//	class V3dAttribute
//
//-----------------------------------------------------------------------------
#include "ImfExport.h"
#include <ImathExport.h>
#include <ImathNamespace.h>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER
template <class V> class IMF_EXPORT_TEMPLATE_TYPE Vec2;
template <class V> class IMF_EXPORT_TEMPLATE_TYPE Vec3;
IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#define COMPILING_IMF_VECTOR_ATTRIBUTE
#include "ImfVecAttribute.h"


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char *
V2iAttribute::staticTypeName ()
{
    return "v2i";
}


template <>
IMF_EXPORT void
V2iAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
}


template <>
IMF_EXPORT void
V2iAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
}


template <>
IMF_EXPORT const char *
V2fAttribute::staticTypeName ()
{
    return "v2f";
}


template <>
IMF_EXPORT void
V2fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
}


template <>
IMF_EXPORT void
V2fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
}


template <>
IMF_EXPORT const char *
V2dAttribute::staticTypeName ()
{
    return "v2d";
}


template <>
IMF_EXPORT void
V2dAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
}


template <>
IMF_EXPORT void
V2dAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
}


template <>
IMF_EXPORT const char *
V3iAttribute::staticTypeName ()
{
    return "v3i";
}


template <>
IMF_EXPORT void
V3iAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
    Xdr::write <StreamIO> (os, _value.z);
}


template <>
IMF_EXPORT void
V3iAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
    Xdr::read <StreamIO> (is, _value.z);
}


template <>
IMF_EXPORT const char *
V3fAttribute::staticTypeName ()
{
    return "v3f";
}


template <>
IMF_EXPORT void
V3fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
    Xdr::write <StreamIO> (os, _value.z);
}


template <>
IMF_EXPORT void
V3fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
    Xdr::read <StreamIO> (is, _value.z);
}


template <>
IMF_EXPORT const char *
V3dAttribute::staticTypeName ()
{
    return "v3d";
}


template <>
IMF_EXPORT void
V3dAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.x);
    Xdr::write <StreamIO> (os, _value.y);
    Xdr::write <StreamIO> (os, _value.z);
}


template <>
IMF_EXPORT void
V3dAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.x);
    Xdr::read <StreamIO> (is, _value.y);
    Xdr::read <StreamIO> (is, _value.z);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::V2i>;
template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::V2f>;
template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::V2d>;
template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::V3i>;
template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::V3f>;
template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::V3d>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
