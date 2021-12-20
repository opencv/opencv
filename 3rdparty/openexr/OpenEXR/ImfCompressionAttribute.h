//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_COMPRESSION_ATTRIBUTE_H
#define INCLUDED_IMF_COMPRESSION_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class CompressionAttribute
//
//-----------------------------------------------------------------------------

#include "ImfAttribute.h"
#include "ImfCompression.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::Compression> CompressionAttribute;

#ifndef COMPILING_IMF_COMPRESSION_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::Compression>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
