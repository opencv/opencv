//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_BOX_ATTRIBUTE_H
#define INCLUDED_IMF_BOX_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class Box2iAttribute
//	class Box2fAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include <ImathBox.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

using Box2iAttribute = TypedAttribute<IMATH_NAMESPACE::Box2i>;
using Box2fAttribute = TypedAttribute<IMATH_NAMESPACE::Box2f>;

#ifndef COMPILING_IMF_BOX_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::Box2i>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::Box2f>;
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
