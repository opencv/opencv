//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_KEY_CODE_ATTRIBUTE_H
#define INCLUDED_IMF_KEY_CODE_ATTRIBUTE_H


//-----------------------------------------------------------------------------
//
//	class KeyCodeAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"

#include "ImfKeyCode.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::KeyCode> KeyCodeAttribute;

#ifndef COMPILING_IMF_KEYCODE_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::KeyCode>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
