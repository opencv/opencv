//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class PreviewImageAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_PREVIEW_IMAGE_ATTRIBUTE
#include "ImfPreviewImageAttribute.h"


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char *
PreviewImageAttribute::staticTypeName ()
{
    return "preview";
}


template <>
IMF_EXPORT void
PreviewImageAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _value.width());
    Xdr::write <StreamIO> (os, _value.height());

    int numPixels = _value.width() * _value.height();
    const PreviewRgba *pixels = _value.pixels();

    for (int i = 0; i < numPixels; ++i)
    {
	Xdr::write <StreamIO> (os, pixels[i].r);
	Xdr::write <StreamIO> (os, pixels[i].g);
	Xdr::write <StreamIO> (os, pixels[i].b);
	Xdr::write <StreamIO> (os, pixels[i].a);
    }
}


template <>
IMF_EXPORT void
PreviewImageAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    int width, height;

    Xdr::read <StreamIO> (is, width);
    Xdr::read <StreamIO> (is, height);

    if (width < 0 || height < 0)
    {
        throw IEX_NAMESPACE::InputExc("Invalid dimensions in Preview Image Attribute");
    }

    // total attribute size should be four bytes per pixel + 8 bytes for width and height dimensions
    if (static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4l + 8l != static_cast<uint64_t>(size) )
    {
        throw IEX_NAMESPACE::InputExc("Mismatch between Preview Image Attribute size and dimensions");
    }

    PreviewImage p (width, height);

    int numPixels = p.width() * p.height();
    PreviewRgba *pixels = p.pixels();

    for (int i = 0; i < numPixels; ++i)
    {
	Xdr::read <StreamIO> (is, pixels[i].r);
	Xdr::read <StreamIO> (is, pixels[i].g);
	Xdr::read <StreamIO> (is, pixels[i].b);
	Xdr::read <StreamIO> (is, pixels[i].a);
    }

    _value = p;
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::PreviewImage>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
