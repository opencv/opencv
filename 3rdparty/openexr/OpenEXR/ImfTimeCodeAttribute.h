//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_TIME_CODE_ATTRIBUTE_H
#define INCLUDED_IMF_TIME_CODE_ATTRIBUTE_H


//-----------------------------------------------------------------------------
//
//	class TimeCodeAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"
#include "ImfTimeCode.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::TimeCode> TimeCodeAttribute;

#ifndef COMPILING_IMF_TIMECODE_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::TimeCode>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT




#endif
