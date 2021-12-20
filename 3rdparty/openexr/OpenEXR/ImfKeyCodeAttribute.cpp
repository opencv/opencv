//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class KeyCodeAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_KEYCODE_ATTRIBUTE

#include <ImfKeyCodeAttribute.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char *
KeyCodeAttribute::staticTypeName ()
{
    return "keycode";
}


template <>
IMF_EXPORT void
KeyCodeAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.filmMfcCode());
    Xdr::write <StreamIO> (os, _value.filmType());
    Xdr::write <StreamIO> (os, _value.prefix());
    Xdr::write <StreamIO> (os, _value.count());
    Xdr::write <StreamIO> (os, _value.perfOffset());
    Xdr::write <StreamIO> (os, _value.perfsPerFrame());
    Xdr::write <StreamIO> (os, _value.perfsPerCount());
}


template <>
IMF_EXPORT void
KeyCodeAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    int tmp;

    Xdr::read <StreamIO> (is, tmp);
    _value.setFilmMfcCode (tmp);

    Xdr::read <StreamIO> (is, tmp);
    _value.setFilmType (tmp);

    Xdr::read <StreamIO> (is, tmp);
    _value.setPrefix (tmp);

    Xdr::read <StreamIO> (is, tmp);
    _value.setCount (tmp);

    Xdr::read <StreamIO> (is, tmp);
    _value.setPerfOffset (tmp);

    Xdr::read <StreamIO> (is, tmp);
    _value.setPerfsPerFrame (tmp);

    Xdr::read <StreamIO> (is, tmp);
    _value.setPerfsPerCount (tmp);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::KeyCode>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
