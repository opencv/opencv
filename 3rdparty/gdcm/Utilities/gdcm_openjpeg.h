/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCM_OPENJPEG_H
#define GDCM_OPENJPEG_H

/* Use the openjpeg library configured for gdcm.  */
#include "gdcmTypes.h"
#ifdef GDCM_USE_SYSTEM_OPENJPEG
#include <openjpeg.h>
// MM:
// See openjpeg issue #3:
// http://code.google.com/p/openjpeg/issues/detail?id=3
//#include <j2k.h>
//#include <jp2.h>

// Instead duplicate header (I know this is bad)
extern "C" {
#include "gdcm_j2k.h"
#include "gdcm_jp2.h"
}

#else
extern "C" {
#include <gdcmopenjpeg-v1/libopenjpeg/openjpeg.h>
#include <gdcmopenjpeg-v1/libopenjpeg/j2k.h>
#include <gdcmopenjpeg-v1/libopenjpeg/jp2.h>
}
#endif

#endif
