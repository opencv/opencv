//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_PREVIEW_IMAGE_ATTRIBUTE_H
#define INCLUDED_IMF_PREVIEW_IMAGE_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class PreviewImageAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include "ImfPreviewImage.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::PreviewImage> PreviewImageAttribute;

#ifndef COMPILING_IMF_PREVIEW_IMAGE_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::PreviewImage>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
