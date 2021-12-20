//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_TILE_DESCRIPTION_ATTRIBUTE_H
#define INCLUDED_IMF_TILE_DESCRIPTION_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class TileDescriptionAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include "ImfTileDescription.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::TileDescription> TileDescriptionAttribute;

#ifndef COMPILING_IMF_STRING_VECTOR_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::TileDescription>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
