//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// 64-bit integer types
//
// Deprecated, use int64_t/uint64_t instead.
//

#ifndef INCLUDED_IMATH_INT64_H
#define INCLUDED_IMATH_INT64_H

#include "ImathNamespace.h"
#include <limits.h>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

#if (defined _WIN32 || defined _WIN64) && _MSC_VER >= 1300
/// Int64 - unsigned 64-bit integer
IMATH_DEPRECATED("use uint64_t")
typedef unsigned __int64 Int64;
/// SInt64 - signed 64-bit integer
IMATH_DEPRECATED("use sint64_t")
typedef __int64 SInt64;
#elif ULONG_MAX == 18446744073709551615LU
/// Int64 - unsigned 64-bit integer
IMATH_DEPRECATED("use uint64_t")
typedef long unsigned int Int64;
/// SInt64 - signed 64-bit integer
IMATH_DEPRECATED("use sint64_t")
typedef long int SInt64;
#else
/// Int64 - unsigned 64-bit integer
IMATH_DEPRECATED("use uint64_t")
typedef long long unsigned int Int64;
/// SInt64 - signed 64-bit integer
IMATH_DEPRECATED("use sint64_t")
typedef long long int SInt64;
#endif

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATH_INT64_H
