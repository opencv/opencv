/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMLEGACYMACRO_H
#define GDCMLEGACYMACRO_H

#if !defined(GDCMTYPES_H) && !defined(SWIG)
#error you need to include gdcmTypes.h instead
#endif

#include "gdcmException.h"

//----------------------------------------------------------------------------
// Setup legacy code policy.

// Define GDCM_LEGACY macro to mark legacy methods where they are
// declared in their class.  Example usage:
//
//   // @deprecated Replaced by MyOtherMethod() as of GDCM 2.0.
//   GDCM_LEGACY(void MyMethod());
#if defined(GDCM_LEGACY_REMOVE)
# define GDCM_LEGACY(method)
#elif defined(GDCM_LEGACY_SILENT) || defined(SWIG)
  // Provide legacy methods with no warnings.
# define GDCM_LEGACY(method) method;
#else
  // Setup compile-time warnings for uses of deprecated methods if
  // possible on this compiler.
# if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#  define GDCM_LEGACY(method) method __attribute__((deprecated));
# elif defined(_MSC_VER) && _MSC_VER >= 1300
#  define GDCM_LEGACY(method) __declspec(deprecated) method;
# else
#  define GDCM_LEGACY(method) method;
# endif
#endif

// Macros to create runtime deprecation warning messages in function
// bodies.  Example usage:
//
//   #if !defined(GDCM_LEGACY_REMOVE)
//   void gdcm::MyClass::MyOldMethod()
//   {
//     GDCM_LEGACY_BODY(gdcm::MyClass::MyOldMethod, "GDCM 2.0");
//   }
//   #endif
//
//   #if !defined(GDCM_LEGACY_REMOVE)
//   void gdcm::MyClass::MyMethod()
//   {
//     GDCM_LEGACY_REPLACED_BODY(gdcm::MyClass::MyMethod, "GDCM 2.0",
//                               gdcm::MyClass::MyOtherMethod);
//   }
//   #endif
#if defined(GDCM_LEGACY_REMOVE) || defined(GDCM_LEGACY_SILENT)
# define GDCM_LEGACY_BODY(method, version)
# define GDCM_LEGACY_REPLACED_BODY(method, version, replace)
#else
# define GDCM_LEGACY_BODY(method, version) \
  gdcmWarningMacro(#method " was deprecated for " version " and will be removed in a future version.")
# define GDCM_LEGACY_REPLACED_BODY(method, version, replace) \
  gdcmWarningMacro(#method " was deprecated for " version " and will be removed in a future version.  Use " #replace " instead.")
#endif

#include "gdcmTrace.h"

#endif // GDCMLEGACYMACRO_H
