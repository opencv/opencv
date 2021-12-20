//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_MATRIX_ATTRIBUTE_H
#define INCLUDED_IMF_MATRIX_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class M33fAttribute
//	class M33dAttribute
//	class M44fAttribute
//	class M44dAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include <ImathMatrix.h>


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (push)
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<IMATH_NAMESPACE::M33f> M33fAttribute;
typedef TypedAttribute<IMATH_NAMESPACE::M33d> M33dAttribute;
typedef TypedAttribute<IMATH_NAMESPACE::M44f> M44fAttribute;
typedef TypedAttribute<IMATH_NAMESPACE::M44d> M44dAttribute;

#ifndef COMPILING_IMF_MATRIX_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::M33f>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::M33d>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::M44f>;
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<IMATH_NAMESPACE::M44d>;
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif
