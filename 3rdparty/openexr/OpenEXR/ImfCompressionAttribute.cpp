//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class CompressionAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_COMPRESSION_ATTRIBUTE

#include "ImfCompressionAttribute.h"

#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4275)
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;


template <>
IMF_EXPORT const char *
CompressionAttribute::staticTypeName ()
{
    return "compression";
}


template <>
IMF_EXPORT void
CompressionAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    unsigned char tmp = _value;
    Xdr::write <StreamIO> (os, tmp);
}


template <>
IMF_EXPORT void
CompressionAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    unsigned char tmp;
    Xdr::read <StreamIO> (is, tmp);

    //
    // prevent invalid values being written to Compressin enum
    // by forcing all unknown types to NUM_COMPRESSION_METHODS which is also an invalid
    // pixel type, but can be used as a PixelType enum value
    // (Header::sanityCheck will throw an exception when files with invalid Compression types are read)
    //

    if (tmp!= NO_COMPRESSION &&
      tmp != RLE_COMPRESSION &&
      tmp != ZIPS_COMPRESSION &&
      tmp != ZIP_COMPRESSION &&
      tmp != PIZ_COMPRESSION &&
      tmp != PXR24_COMPRESSION &&
      tmp != B44_COMPRESSION &&
      tmp != B44A_COMPRESSION &&
      tmp != DWAA_COMPRESSION &&
      tmp != DWAB_COMPRESSION)
    {
        tmp = NUM_COMPRESSION_METHODS;
    }

    _value = Compression (tmp);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::Compression>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
