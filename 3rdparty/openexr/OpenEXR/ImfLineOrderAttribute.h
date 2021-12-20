//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_LINE_ORDER_ATTRIBUTE_H
#define INCLUDED_IMF_LINE_ORDER_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class LineOrderAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include "ImfLineOrder.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

using LineOrderAttribute = TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::LineOrder>;

#ifndef COMPILING_IMF_LINE_ORDER_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::LineOrder>;
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
