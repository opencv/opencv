//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_CHROMATICITIES_ATTRIBUTE_H
#define INCLUDED_IMF_CHROMATICITIES_ATTRIBUTE_H


//-----------------------------------------------------------------------------
//
//	class ChromaticitiesAttribute
//
//-----------------------------------------------------------------------------

#include "ImfAttribute.h"
#include "ImfChromaticities.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::Chromaticities> ChromaticitiesAttribute;

#ifndef COMPILING_IMF_CHROMATICITIES_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::Chromaticities>;
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
