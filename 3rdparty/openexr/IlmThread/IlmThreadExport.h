//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_ILMTHREADEXPORT_H
#define INCLUDED_ILMTHREADEXPORT_H

#include "OpenEXRConfig.h"

// See docs/SymbolVisibility.md for more discussion

/// \addtogroup ExportMacros
/// @{

// are we making a DLL under windows (might be msvc or mingw or others)
#if defined(OPENEXR_DLL)

// when building as a DLL for windows, typical dllexport / import case
// where we need to switch depending on whether we are compiling
// internally or not
#  if defined(ILMTHREAD_EXPORTS)
#    define ILMTHREAD_EXPORT __declspec(dllexport)
#  else
#    define ILMTHREAD_EXPORT __declspec(dllimport)
#  endif

// DLLs don't support these types of visibility controls, just leave them as empty
#  define ILMTHREAD_EXPORT_TYPE
#  define ILMTHREAD_HIDDEN

#else // OPENEXR_DLL

// just pass these through from the top level config
#  define ILMTHREAD_EXPORT OPENEXR_EXPORT
#  define ILMTHREAD_HIDDEN OPENEXR_HIDDEN
#  define ILMTHREAD_EXPORT_TYPE OPENEXR_EXPORT_TYPE

#endif // OPENEXR_DLL

/// @}

#endif // INCLUDED_ILMTHREADEXPORT_H
