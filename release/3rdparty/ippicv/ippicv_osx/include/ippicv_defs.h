/* 
// Copyright 1999 2015 Intel Corporation All Rights Reserved.
// 
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title
// to such Material remains with Intel Corporation or its suppliers or
// licensors. The Material contains proprietary information of Intel
// or its suppliers and licensors. The Material is protected by worldwide
// copyright laws and treaty provisions. No part of the Material may be used,
// copied, reproduced, modified, published, uploaded, posted, transmitted,
// distributed or disclosed in any way without Intel's prior express written
// permission. No license under any patent, copyright or other intellectual
// property rights in the Material is granted to or conferred upon you,
// either expressly, by implication, inducement, estoppel or otherwise.
// Any license under such intellectual property rights must be express and
// approved by Intel in writing.
// 
// Unless otherwise agreed by Intel in writing,
// you may not remove or alter this notice or any other notice embedded in
// Materials by Intel or Intel's suppliers or licensors in any way.
// 
*/

/* 
//              Intel(R) Integrated Performance Primitives
//              Common Types and Macro Definitions
// 
// 
*/


#ifndef __IPPICV_DEFS_H__
#define __IPPICV_DEFS_H__

#ifdef __cplusplus
extern "C" {
#endif

#if defined( __IPPDEFS_H__ )
#error "You try to include ippicv_defs.h and ippdefs.h together. You can't use IPP and ICV libraries together, please use only one and please use header files only for one library"
#endif


#if defined( _IPP_PARALLEL_STATIC ) || defined( _IPP_PARALLEL_DYNAMIC )
  #pragma message("Threaded versions of IPP libraries are deprecated and will be removed in one of the future IPP releases. Use the following link for details: https://software.intel.com/sites/products/ipp-deprecated-features-feedback/")
#endif

#if defined (_WIN64)
#define _INTEL_PLATFORM "intel64/"
#elif defined (_WIN32)
#define _INTEL_PLATFORM "ia32/"
#endif

#if !defined( IPPAPI )

  #if defined( IPP_W32DLL ) && (defined( _WIN32 ) || defined( _WIN64 ))
    #if defined( _MSC_VER ) || defined( __ICL )
      #define IPPAPI( type,name,arg ) \
                     __declspec(dllimport)   type __STDCALL name arg;
    #else
      #define IPPAPI( type,name,arg )        type __STDCALL name arg;
    #endif
  #else
    #define   IPPAPI( type,name,arg )        type __STDCALL name arg;
  #endif

#endif

#if (defined( __ICL ) || defined( __ECL ) || defined(_MSC_VER)) && !defined( _PCS ) && !defined( _PCS_GENSTUBS )
  #if( __INTEL_COMPILER >= 1100 ) /* icl 11.0 supports additional comment */
    #if( _MSC_VER >= 1400 )
      #define IPP_DEPRECATED( comment ) __declspec( deprecated ( comment ))
    #else
      #pragma message ("your icl version supports additional comment for deprecated functions but it can't be displayed")
      #pragma message ("because internal _MSC_VER macro variable setting requires compatibility with MSVC7.1")
      #pragma message ("use -Qvc8 switch for icl command line to see these additional comments")
      #define IPP_DEPRECATED( comment ) __declspec( deprecated )
    #endif
  #elif( _MSC_FULL_VER >= 140050727 )&&( !defined( __INTEL_COMPILER )) /* VS2005 supports additional comment */
    #define IPP_DEPRECATED( comment ) __declspec( deprecated ( comment ))
  #elif( _MSC_VER <= 1200 )&&( !defined( __INTEL_COMPILER )) /* VS 6 doesn't support deprecation */
    #define IPP_DEPRECATED( comment )
  #else
    #define IPP_DEPRECATED( comment ) __declspec( deprecated )
  #endif
#elif (defined(__ICC) || defined(__ECC) || defined( __GNUC__ )) && !defined( _PCS ) && !defined( _PCS_GENSTUBS )
  #if defined( __GNUC__ )
    #if __GNUC__ >= 4 && __GNUC_MINOR__ >= 5
      #define IPP_DEPRECATED( message ) __attribute__(( deprecated( message )))
    #else
      #define IPP_DEPRECATED( message ) __attribute__(( deprecated ))
    #endif
  #else
    #define IPP_DEPRECATED( comment ) __attribute__(( deprecated ))
  #endif
#else
  #define IPP_DEPRECATED( comment )
#endif

#include "ippicv_base.h"
#include "ippicv_types.h"

extern const IppiRect ippRectInfinite;

#ifdef __cplusplus
}
#endif

#endif /* __IPPICV_DEFS_H__ */
