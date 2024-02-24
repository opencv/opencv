//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMATHEXPORT_H
#define INCLUDED_IMATHEXPORT_H

#include "ImathConfig.h"

/// \defgroup ExportMacros Macros to manage symbol visibility
///
/// There is more information about the motivation for these macros
/// documented in the OpenEXR source tree
/// (https://github.com/AcademySoftwareFoundation/openexr) under
/// docs/SymbolVisibility.md
///
/// Imath only needs a couple of the possible macros outlined in the
/// above document, and due to it largely being inline only, does not
/// have much to do.
/// 
/// @{
#if defined(IMATH_DLL)

// when building Imath as a DLL for Windows, we have to control the
// typical DLL export / import things. Luckily, the typeinfo is all
// automatic there, so only have to deal with symbols, except Windows
// has some weirdness with DLLs and extern const, so we have to
// provide a macro to handle that.

#  if defined(IMATH_EXPORTS)
#    define IMATH_EXPORT __declspec(dllexport)
#    define IMATH_EXPORT_CONST extern __declspec(dllexport)
#  else
#    define IMATH_EXPORT __declspec(dllimport)
#    define IMATH_EXPORT_CONST extern __declspec(dllimport)
#  endif

// DLLs don't support these types of visibility controls, just leave them as empty
#  define IMATH_EXPORT_TYPE
#  define IMATH_EXPORT_ENUM
#  define IMATH_EXPORT_TEMPLATE_TYPE

#else

#  ifdef IMATH_PUBLIC_SYMBOL_ATTRIBUTE
#    define IMATH_EXPORT IMATH_PUBLIC_SYMBOL_ATTRIBUTE
#    define IMATH_EXPORT_CONST extern const IMATH_PUBLIC_SYMBOL_ATTRIBUTE
#  else
#    define IMATH_EXPORT
#    define IMATH_EXPORT_CONST extern const
#  endif

#  ifdef IMATH_PUBLIC_TYPE_VISIBILITY_ATTRIBUTE
#    define IMATH_EXPORT_ENUM IMATH_PUBLIC_TYPE_VISIBILITY_ATTRIBUTE
#    define IMATH_EXPORT_TEMPLATE_TYPE IMATH_PUBLIC_TYPE_VISIBILITY_ATTRIBUTE
#    define IMATH_EXPORT_TYPE IMATH_PUBLIC_TYPE_VISIBILITY_ATTRIBUTE
#  else
#    define IMATH_EXPORT_ENUM
#    define IMATH_EXPORT_TEMPLATE_TYPE IMATH_EXPORT
#    define IMATH_EXPORT_TYPE IMATH_EXPORT
#  endif

#endif // IMATH_DLL

/// @}

#endif // INCLUDED_IMATHEXPORT_H
