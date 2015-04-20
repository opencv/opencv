/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMWIN32_H
#define GDCMWIN32_H

#if !defined(GDCMTYPES_H)
#error you need to include gdcmTypes.h instead
#endif
//-----------------------------------------------------------------------------
// http://gcc.gnu.org/wiki/Visibility
#if defined(WIN32) && defined(GDCM_BUILD_SHARED_LIBS)
  #if (defined(gdcmCommon_EXPORTS) || defined(gdcmDICT_EXPORTS) || defined(gdcmDSED_EXPORTS) || defined(gdcmIOD_EXPORTS) || defined(gdcmMSFF_EXPORTS) || defined(gdcmMEXD_EXPORTS)|| defined(_gdcmswig_EXPORTS)) || defined(vtkgdcm_EXPORTS)
    #define GDCM_EXPORT __declspec( dllexport )
  #else
    #define GDCM_EXPORT __declspec( dllimport )
  #endif
#else
  #if __GNUC__ >= 4
    #define GDCM_EXPORT __attribute__ ((visibility ("default")))
    #define GDCM_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define GDCM_EXPORT
  #endif
#endif

// In VTK 4.2 vtkWrapPython does not like anything other than VTK_*EXPORT
// [ 86%] Generating vtkGDCMImageReaderPython.cxx
// syntax error
// *** SYNTAX ERROR found in parsing the header file /usr/local/src/gdcm2/tags/gdcm-2-0-11/Utilities/VTK/vtkGDCMImageReader.h before line 128***
// make[2]: *** [Utilities/VTK/vtkGDCMImageReaderPython.cxx] Error 1
// make[1]: *** [Utilities/VTK/CMakeFiles/vtkgdcmPythonD.dir/all] Error 2
// make: *** [all] Error 2

#if defined(VTK_MAJOR_VERSION) && ( VTK_MAJOR_VERSION == 4 )
#undef VTK_EXPORT
#define VTK_EXPORT GDCM_EXPORT
#endif

//-----------------------------------------------------------------------------
//This is needed when compiling in debug mode
#ifdef _MSC_VER
// to allow construct such as: std::numeric_limits<int>::max() we need the following:
// warning C4003: not enough actual parameters for macro 'max'
#ifndef NOMINMAX
  #define NOMINMAX
#endif
# pragma warning ( default : 4263 ) /* no override, call convention differs */
// 'identifier' : class 'type' needs to have dll-interface to be used by
// clients of class 'type2'
#pragma warning ( disable : 4251 )
// non dll-interface class 'type' used as base for dll-interface class 'type2'
#pragma warning ( disable : 4275 )
// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#pragma warning ( disable : 4786 )
//'identifier' : decorated name length exceeded, name was truncated
#pragma warning ( disable : 4503 )
// C++ exception specification ignored except to indicate a
// function is not __declspec(nothrow)
//#pragma warning ( disable : 4290 )
// signed/unsigned mismatch
#pragma warning ( disable : 4018 )
// return type for 'identifier' is '' (ie; not a UDT or reference to UDT. Will
// produce errors if applied using infix notation
//#pragma warning ( disable : 4284 )
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
// //#pragma warning ( disable : 4800 )
#endif //_MSC_VER

//-----------------------------------------------------------------------------
#endif //GDCMWIN32_H
