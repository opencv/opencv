// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.

#ifndef INCLUDED_IMF_IDMANIFEST_ATTRIBUTE_H
#define INCLUDED_IMF_IDMANIFEST_ATTRIBUTE_H

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include "ImfIDManifest.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::CompressedIDManifest>
    IDManifestAttribute;

#ifndef COMPILING_IMF_IDMANIFEST_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::CompressedIDManifest>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
