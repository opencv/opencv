//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_DOUBLE_ATTRIBUTE_H
#define INCLUDED_IMF_DOUBLE_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class DoubleAttribute
//
//-----------------------------------------------------------------------------

#include "ImfAttribute.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<double> DoubleAttribute;

#ifndef COMPILING_IMF_DOUBLE_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<double>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
