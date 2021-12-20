//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_RATIONAL_ATTRIBUTE_H
#define INCLUDED_IMF_RATIONAL_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class RationalAttribute
//
//-----------------------------------------------------------------------------

#include "ImfAttribute.h"
#include "ImfRational.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::Rational> RationalAttribute;

#ifndef COMPILING_IMF_RATIONAL_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::Rational>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
