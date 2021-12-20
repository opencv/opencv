// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.

#define COMPILING_IMF_IDMANIFEST_ATTRIBUTE
#include "ImfIDManifestAttribute.h"

#include <stdlib.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;

template <>
IMF_EXPORT const char*
IDManifestAttribute::staticTypeName()
{
   return "idmanifest";
}


template <>
IMF_EXPORT void
IDManifestAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    uint64_t uncompressedDataSize = _value._uncompressedDataSize;
    Xdr::write<StreamIO>(os,uncompressedDataSize);
    const char* output = (const char*) _value._data;
    Xdr::write <StreamIO> (os, output,_value._compressedDataSize);

}


template <>
IMF_EXPORT void
IDManifestAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{

    if (size<4)
    {
          throw IEX_NAMESPACE::InputExc("Invalid size field reading idmanifest attribute");
    }
    _value._compressedDataSize = size-4;

    if (_value._data)
    {
        // if attribute is reallocated , free up previous memory
        free( static_cast<void*>(_value._data) );
        _value._data = nullptr;
    }

    uint64_t uncompressedDataSize;
    //
    // first eight bytes: data size once data is uncompressed
    //
    Xdr::read<StreamIO>(is,uncompressedDataSize);

    _value._uncompressedDataSize = uncompressedDataSize;

    //
    // allocate memory for compressed storage and read data
    //
    _value._data = static_cast<unsigned char*>( malloc(size-4) );
    char* input = (char*) _value._data;
    Xdr::read<StreamIO>(is,input,_value._compressedDataSize);
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::CompressedIDManifest>;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
