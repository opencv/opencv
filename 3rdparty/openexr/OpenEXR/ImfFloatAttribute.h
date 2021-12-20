//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_FLOAT_ATTRIBUTE_H
#define INCLUDED_IMF_FLOAT_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class FloatAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<float> FloatAttribute;

#ifndef COMPILING_IMF_FLOAT_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<float>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
