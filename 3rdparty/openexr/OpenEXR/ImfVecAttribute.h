//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_VEC_ATTRIBUTE_H
#define INCLUDED_IMF_VEC_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class V2iAttribute
//	class V2fAttribute
//	class V2dAttribute
//	class V3iAttribute
//	class V3fAttribute
//	class V3dAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include <ImathVec.h>


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (push)
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

typedef TypedAttribute<IMATH_NAMESPACE::V2i> V2iAttribute;
typedef TypedAttribute<IMATH_NAMESPACE::V2f> V2fAttribute;
typedef TypedAttribute<IMATH_NAMESPACE::V2d> V2dAttribute;
typedef TypedAttribute<IMATH_NAMESPACE::V3i> V3iAttribute;
typedef TypedAttribute<IMATH_NAMESPACE::V3f> V3fAttribute;
typedef TypedAttribute<IMATH_NAMESPACE::V3d> V3dAttribute;

#ifndef COMPILING_IMF_VECTOR_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::V2i>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::V2f>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::V2d>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::V3i>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::V3f>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::V3d>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (pop)
#endif

#endif
