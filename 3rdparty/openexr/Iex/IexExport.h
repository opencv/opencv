//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IEXEXPORT_H
#define INCLUDED_IEXEXPORT_H

#include "OpenEXRConfig.h"

#if defined(OPENEXR_DLL)

// when building as a DLL for windows, typical dllexport / import case
// where we need to switch depending on whether we are compiling
// internally or not

#  if defined(IEX_EXPORTS)
#    define IEX_EXPORT __declspec(dllexport)
#  else
#    define IEX_EXPORT __declspec(dllimport)
#  endif

// DLLs don't support these types of visibility controls, just leave them as empty
#  define IEX_EXPORT_TYPE
#  define IEX_EXPORT_ENUM

#else // OPENEXR_DLL

// just pass these through from the top level config
#  define IEX_EXPORT OPENEXR_EXPORT
#  define IEX_EXPORT_TYPE OPENEXR_EXPORT_TYPE
#  define IEX_EXPORT_ENUM OPENEXR_EXPORT_ENUM

#endif // OPENEXR_DLL

#endif // #ifndef INCLUDED_IEXEXPORT_H

