//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class Box2iAttribute
//	class Box2fAttribute
//
//-----------------------------------------------------------------------------
#include "ImfExport.h"
#include <ImathExport.h>
#include <ImathNamespace.h>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER
template <class V> class IMF_EXPORT_TEMPLATE_TYPE Vec2;
template <class V> class IMF_EXPORT_TEMPLATE_TYPE Vec3;
template <class V> class IMF_EXPORT_TEMPLATE_TYPE Box;
IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#define COMPILING_IMF_BOX_ATTRIBUTE

#include <ImfBoxAttribute.h>

#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif



OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char *
Box2iAttribute::staticTypeName ()
{
    return "box2i";
}


template <>
IMF_EXPORT void
Box2iAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.min.x);
    Xdr::write <StreamIO> (os, _value.min.y);
    Xdr::write <StreamIO> (os, _value.max.x);
    Xdr::write <StreamIO> (os, _value.max.y);
}


template <>
IMF_EXPORT void
Box2iAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.min.x);
    Xdr::read <StreamIO> (is, _value.min.y);
    Xdr::read <StreamIO> (is, _value.max.x);
    Xdr::read <StreamIO> (is, _value.max.y);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::Box2i>;


template <>
IMF_EXPORT const char *
Box2fAttribute::staticTypeName ()
{
    return "box2f";
}


template <>
IMF_EXPORT void
Box2fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.min.x);
    Xdr::write <StreamIO> (os, _value.min.y);
    Xdr::write <StreamIO> (os, _value.max.x);
    Xdr::write <StreamIO> (os, _value.max.y);
}


template <>
IMF_EXPORT void
Box2fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value.min.x);
    Xdr::read <StreamIO> (is, _value.min.y);
    Xdr::read <StreamIO> (is, _value.max.x);
    Xdr::read <StreamIO> (is, _value.max.y);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::Box2f>;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
