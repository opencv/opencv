//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class M33fAttribute
//	class M33dAttribute
//	class M44fAttribute
//	class M44dAttribute
//
//-----------------------------------------------------------------------------
#include "ImfExport.h"
#include <ImathExport.h>
#include <ImathNamespace.h>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER
template <class V> class IMF_EXPORT_TEMPLATE_TYPE Matrix33;
template <class V> class IMF_EXPORT_TEMPLATE_TYPE Matrix44;
IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#define COMPILING_IMF_MATRIX_ATTRIBUTE

#include "ImfMatrixAttribute.h"


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;


template <>
IMF_EXPORT const char *
M33fAttribute::staticTypeName ()
{
    return "m33f";
}


template <>
IMF_EXPORT void
M33fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value[0][0]);
    Xdr::write <StreamIO> (os, _value[0][1]);
    Xdr::write <StreamIO> (os, _value[0][2]);

    Xdr::write <StreamIO> (os, _value[1][0]);
    Xdr::write <StreamIO> (os, _value[1][1]);
    Xdr::write <StreamIO> (os, _value[1][2]);

    Xdr::write <StreamIO> (os, _value[2][0]);
    Xdr::write <StreamIO> (os, _value[2][1]);
    Xdr::write <StreamIO> (os, _value[2][2]);
}


template <>
IMF_EXPORT void
M33fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value[0][0]);
    Xdr::read <StreamIO> (is, _value[0][1]);
    Xdr::read <StreamIO> (is, _value[0][2]);

    Xdr::read <StreamIO> (is, _value[1][0]);
    Xdr::read <StreamIO> (is, _value[1][1]);
    Xdr::read <StreamIO> (is, _value[1][2]);

    Xdr::read <StreamIO> (is, _value[2][0]);
    Xdr::read <StreamIO> (is, _value[2][1]);
    Xdr::read <StreamIO> (is, _value[2][2]);
}


template <>
IMF_EXPORT const char *
M33dAttribute::staticTypeName ()
{
    return "m33d";
}


template <>
IMF_EXPORT void
M33dAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value[0][0]);
    Xdr::write <StreamIO> (os, _value[0][1]);
    Xdr::write <StreamIO> (os, _value[0][2]);

    Xdr::write <StreamIO> (os, _value[1][0]);
    Xdr::write <StreamIO> (os, _value[1][1]);
    Xdr::write <StreamIO> (os, _value[1][2]);

    Xdr::write <StreamIO> (os, _value[2][0]);
    Xdr::write <StreamIO> (os, _value[2][1]);
    Xdr::write <StreamIO> (os, _value[2][2]);
}


template <>
IMF_EXPORT void
M33dAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value[0][0]);
    Xdr::read <StreamIO> (is, _value[0][1]);
    Xdr::read <StreamIO> (is, _value[0][2]);

    Xdr::read <StreamIO> (is, _value[1][0]);
    Xdr::read <StreamIO> (is, _value[1][1]);
    Xdr::read <StreamIO> (is, _value[1][2]);

    Xdr::read <StreamIO> (is, _value[2][0]);
    Xdr::read <StreamIO> (is, _value[2][1]);
    Xdr::read <StreamIO> (is, _value[2][2]);
}


template <>
IMF_EXPORT const char *
M44fAttribute::staticTypeName ()
{
    return "m44f";
}


template <>
IMF_EXPORT void
M44fAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value[0][0]);
    Xdr::write <StreamIO> (os, _value[0][1]);
    Xdr::write <StreamIO> (os, _value[0][2]);
    Xdr::write <StreamIO> (os, _value[0][3]);

    Xdr::write <StreamIO> (os, _value[1][0]);
    Xdr::write <StreamIO> (os, _value[1][1]);
    Xdr::write <StreamIO> (os, _value[1][2]);
    Xdr::write <StreamIO> (os, _value[1][3]);

    Xdr::write <StreamIO> (os, _value[2][0]);
    Xdr::write <StreamIO> (os, _value[2][1]);
    Xdr::write <StreamIO> (os, _value[2][2]);
    Xdr::write <StreamIO> (os, _value[2][3]);

    Xdr::write <StreamIO> (os, _value[3][0]);
    Xdr::write <StreamIO> (os, _value[3][1]);
    Xdr::write <StreamIO> (os, _value[3][2]);
    Xdr::write <StreamIO> (os, _value[3][3]);
}


template <>
IMF_EXPORT void
M44fAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value[0][0]);
    Xdr::read <StreamIO> (is, _value[0][1]);
    Xdr::read <StreamIO> (is, _value[0][2]);
    Xdr::read <StreamIO> (is, _value[0][3]);

    Xdr::read <StreamIO> (is, _value[1][0]);
    Xdr::read <StreamIO> (is, _value[1][1]);
    Xdr::read <StreamIO> (is, _value[1][2]);
    Xdr::read <StreamIO> (is, _value[1][3]);

    Xdr::read <StreamIO> (is, _value[2][0]);
    Xdr::read <StreamIO> (is, _value[2][1]);
    Xdr::read <StreamIO> (is, _value[2][2]);
    Xdr::read <StreamIO> (is, _value[2][3]);

    Xdr::read <StreamIO> (is, _value[3][0]);
    Xdr::read <StreamIO> (is, _value[3][1]);
    Xdr::read <StreamIO> (is, _value[3][2]);
    Xdr::read <StreamIO> (is, _value[3][3]);
}


template <>
IMF_EXPORT const char *
M44dAttribute::staticTypeName ()
{
    return "m44d";
}


template <>
IMF_EXPORT void
M44dAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value[0][0]);
    Xdr::write <StreamIO> (os, _value[0][1]);
    Xdr::write <StreamIO> (os, _value[0][2]);
    Xdr::write <StreamIO> (os, _value[0][3]);

    Xdr::write <StreamIO> (os, _value[1][0]);
    Xdr::write <StreamIO> (os, _value[1][1]);
    Xdr::write <StreamIO> (os, _value[1][2]);
    Xdr::write <StreamIO> (os, _value[1][3]);

    Xdr::write <StreamIO> (os, _value[2][0]);
    Xdr::write <StreamIO> (os, _value[2][1]);
    Xdr::write <StreamIO> (os, _value[2][2]);
    Xdr::write <StreamIO> (os, _value[2][3]);

    Xdr::write <StreamIO> (os, _value[3][0]);
    Xdr::write <StreamIO> (os, _value[3][1]);
    Xdr::write <StreamIO> (os, _value[3][2]);
    Xdr::write <StreamIO> (os, _value[3][3]);
}


template <>
IMF_EXPORT void
M44dAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    Xdr::read <StreamIO> (is, _value[0][0]);
    Xdr::read <StreamIO> (is, _value[0][1]);
    Xdr::read <StreamIO> (is, _value[0][2]);
    Xdr::read <StreamIO> (is, _value[0][3]);

    Xdr::read <StreamIO> (is, _value[1][0]);
    Xdr::read <StreamIO> (is, _value[1][1]);
    Xdr::read <StreamIO> (is, _value[1][2]);
    Xdr::read <StreamIO> (is, _value[1][3]);

    Xdr::read <StreamIO> (is, _value[2][0]);
    Xdr::read <StreamIO> (is, _value[2][1]);
    Xdr::read <StreamIO> (is, _value[2][2]);
    Xdr::read <StreamIO> (is, _value[2][3]);

    Xdr::read <StreamIO> (is, _value[3][0]);
    Xdr::read <StreamIO> (is, _value[3][1]);
    Xdr::read <StreamIO> (is, _value[3][2]);
    Xdr::read <StreamIO> (is, _value[3][3]);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::M33f>;
template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::M33d>;
template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::M44f>;
template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<IMATH_NAMESPACE::M44d>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
