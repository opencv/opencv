/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTYPES_H
#define GDCMTYPES_H

#include "gdcmConfigure.h"
#include "gdcmWin32.h"
#include "gdcmLegacyMacro.h"

//-----------------------------------------------------------------------------
#ifdef GDCM_HAVE_STDINT_H
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif // __STDC_LIMIT_MACROS
#include <stdint.h>
//#undef __STDC_LIMIT_MACROS
#else
#ifdef GDCM_HAVE_INTTYPES_H
// Old system only have this
#include <inttypes.h>   // For uint8_t uint16_t and uint32_t
#else
// Broken plateforms do not respect C99 and do not provide those typedef
// Special case for recent Borland compiler, comes with stdint.h
#if defined(__BORLANDC__) && (__BORLANDC__ < 0x0560) || defined(__MINGW32__)
typedef  signed char         int8_t;
typedef  signed short        int16_t;
typedef  signed int          int32_t;
typedef  unsigned char       uint8_t;
typedef  unsigned short      uint16_t;
typedef  unsigned int        uint32_t;
typedef  unsigned __int64    uint64_t;
#elif defined(_MSC_VER)
#include "stdint.h"
#else
#error "Sorry, your platform is not supported"
#endif // defined(_MSC_VER) || defined(__BORLANDC__) && (__BORLANDC__ < 0x0560)  || defined(__MINGW32__)
#endif // GDCM_HAVE_INTTYPES_H
#endif // GDCM_HAVE_STDINT_H

// Basically for VS6 and bcc 5.5.1:
#ifndef UINT32_MAX
#define UINT32_MAX    (4294967295U)
#endif

//-----------------------------------------------------------------------------
#endif //GDCMTYPES_H
