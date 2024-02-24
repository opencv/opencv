//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMFUTILEXPORT_H
#define INCLUDED_IMFUTILEXPORT_H

#include "OpenEXRConfig.h"

/// \addtogroup ExportMacros
/// @{

// are we making a DLL under windows (might be msvc or mingw or others)
#if defined(OPENEXR_DLL)

#    if defined(OPENEXRUTIL_EXPORTS)
#        define IMFUTIL_EXPORT __declspec(dllexport)

// mingw needs the export when the extern is defined
#        if defined(__MINGW32__)
#            define IMFUTIL_EXPORT_EXTERN_TEMPLATE IMFUTIL_EXPORT
#            define IMFUTIL_EXPORT_TEMPLATE_INSTANCE
// for mingw windows, we need to cause this to export the
// typeinfo tables (but you don't need to have the
// complementary import, because might be a local template too!)
#            define IMFUTIL_EXPORT_TEMPLATE_TYPE IMFUTIL_EXPORT
#        else
// for normal msvc, need to export the actual instantiation in
// the cpp code, and none of the others
#            define IMFUTIL_EXPORT_EXTERN_TEMPLATE
#            define IMFUTIL_EXPORT_TEMPLATE_INSTANCE IMFUTIL_EXPORT
#            define IMFUTIL_EXPORT_TEMPLATE_TYPE
#        endif

#    else // OPENEXRUTIL_EXPORTS
#        define IMFUTIL_EXPORT __declspec(dllimport)
#        define IMFUTIL_EXPORT_EXTERN_TEMPLATE IMFUTIL_EXPORT
#        define IMFUTIL_EXPORT_TEMPLATE_INSTANCE
#        define IMFUTIL_EXPORT_TEMPLATE_TYPE
#    endif

// DLLs don't support these types of visibility controls, just leave them as empty
#    define IMFUTIL_EXPORT_TYPE
#    define IMFUTIL_EXPORT_ENUM
#    define IMFUTIL_HIDDEN

#else // not an OPENEXR_DLL

// just pass these through from the top level config
#    define IMFUTIL_EXPORT OPENEXR_EXPORT
#    define IMFUTIL_HIDDEN OPENEXR_HIDDEN
#    define IMFUTIL_EXPORT_ENUM OPENEXR_EXPORT_ENUM
#    define IMFUTIL_EXPORT_TYPE OPENEXR_EXPORT_TYPE
#    define IMFUTIL_EXPORT_TEMPLATE_TYPE OPENEXR_EXPORT_TEMPLATE_TYPE
#    define IMFUTIL_EXPORT_EXTERN_TEMPLATE OPENEXR_EXPORT_EXTERN_TEMPLATE
#    define IMFUTIL_EXPORT_TEMPLATE_INSTANCE OPENEXR_EXPORT_TEMPLATE_INSTANCE

#endif // OPENEXR_DLL

/// @}

#endif // INCLUDED_IMFUTILEXPORT_H
