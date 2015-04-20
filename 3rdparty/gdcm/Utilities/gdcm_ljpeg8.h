/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCM_LJPEG8_H
#define GDCM_LJPEG8_H

/* Use the ljpeg library configured for gdcm.  */
#include "gdcmTypes.h"

#ifdef GDCM_USE_SYSTEM_LJPEG
extern "C" {
# include <ljpeg-62/8/jinclude.h>
# include <ljpeg-62/8/jpeglib.h>
# include <ljpeg-62/8/jerror.h>
}
#else
extern "C" {
#include "gdcmjpeg/8/jinclude.h"
#include "gdcmjpeg/8/jpeglib.h"
#include "gdcmjpeg/8/jerror.h"
}
#endif

#endif
