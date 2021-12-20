//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_INT64_H
#define INCLUDED_IMF_INT64_H

//----------------------------------------------------------------------------
//
//	Deprecated Int64/SInt64 unsigned 64-bit integer type.
//      Use int64_t and uint64_t instead.
//
//----------------------------------------------------------------------------

#include "ImathInt64.h"
#include "ImfNamespace.h"
#include <stdint.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

IMATH_DEPRECATED("use uint64_t")
typedef IMATH_NAMESPACE::Int64 Int64;

IMATH_DEPRECATED("use int64_t")
typedef IMATH_NAMESPACE::SInt64 SInt64;

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT




#endif // INCLUDED_IMF_INT64_H
