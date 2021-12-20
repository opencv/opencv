//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_INT_ATTRIBUTE_H
#define INCLUDED_IMF_INT_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class IntAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<int> IntAttribute;

#ifndef COMPILING_IMF_INT_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<int>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
