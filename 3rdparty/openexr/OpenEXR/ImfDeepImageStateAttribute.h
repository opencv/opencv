//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_DEEPIMAGESTATE_ATTRIBUTE_H
#define INCLUDED_IMF_DEEPIMAGESTATE_ATTRIBUTE_H


//-----------------------------------------------------------------------------
//
//	class DeepImageStateAttribute
//
//-----------------------------------------------------------------------------

#include "ImfAttribute.h"
#include "ImfDeepImageState.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::DeepImageState>
    DeepImageStateAttribute;

#ifndef COMPILING_IMF_DEEP_IMAGE_STATE_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::DeepImageState>;
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
