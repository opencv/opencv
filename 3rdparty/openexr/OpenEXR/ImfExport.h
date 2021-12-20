//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMFEXPORT_H
#define INCLUDED_IMFEXPORT_H

#include "OpenEXRConfig.h"

/// \addtogroup ExportMacros
/// @{

// are we making a DLL under windows (might be msvc or mingw or others)
#if defined(OPENEXR_DLL)

// when building as a DLL for windows, typical dllexport / import case
// where we need to switch depending on whether we are compiling
// internally or not
#  if defined(OPENEXR_EXPORTS)
#    define IMF_EXPORT __declspec(dllexport)

     // mingw needs the export when the extern is defined
#    if defined(__MINGW32__)
#      define IMF_EXPORT_EXTERN_TEMPLATE IMF_EXPORT
#      define IMF_EXPORT_TEMPLATE_INSTANCE
       // for mingw windows, we need to cause this to export the
       // typeinfo tables (but you don't need to have the
       // complementary import, because might be a local template too!)
#      define IMF_EXPORT_TEMPLATE_TYPE IMF_EXPORT
#    else
       // for normal msvc, need to export the actual instantiation in
       // the cpp code, and none of the others
#      define IMF_EXPORT_EXTERN_TEMPLATE
#      define IMF_EXPORT_TEMPLATE_INSTANCE IMF_EXPORT
#      define IMF_EXPORT_TEMPLATE_TYPE
#    endif

#  else // OPENEXR_EXPORTS
#    define IMF_EXPORT __declspec(dllimport)
#    define IMF_EXPORT_EXTERN_TEMPLATE IMF_EXPORT
#    define IMF_EXPORT_TEMPLATE_INSTANCE
#    define IMF_EXPORT_TEMPLATE_TYPE
#  endif

// DLLs don't support these types of visibility controls, just leave them as empty
#  define IMF_EXPORT_TYPE
#  define IMF_EXPORT_ENUM
#  define IMF_HIDDEN

#else // not an OPENEXR_DLL

// just pass these through from the top level config
#  define IMF_EXPORT OPENEXR_EXPORT
#  define IMF_HIDDEN OPENEXR_HIDDEN
#  define IMF_EXPORT_ENUM OPENEXR_EXPORT_ENUM
#  define IMF_EXPORT_TYPE OPENEXR_EXPORT_TYPE
#  define IMF_EXPORT_TEMPLATE_TYPE OPENEXR_EXPORT_TEMPLATE_TYPE
#  define IMF_EXPORT_EXTERN_TEMPLATE OPENEXR_EXPORT_EXTERN_TEMPLATE
#  define IMF_EXPORT_TEMPLATE_INSTANCE OPENEXR_EXPORT_TEMPLATE_INSTANCE

#endif // OPENEXR_DLL

/// @}

#endif // INCLUDED_IMFEXPORT_H
