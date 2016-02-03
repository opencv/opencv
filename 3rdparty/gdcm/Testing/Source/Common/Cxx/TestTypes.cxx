/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTypes.h"

int TestTypes(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  if( sizeof(int8_t)   != 1 ) return 1;
  if( sizeof(int16_t)  != 2 ) return 1;
  if( sizeof(int32_t)  != 4 ) return 1;
  if( sizeof(uint8_t)  != 1 ) return 1;
  if( sizeof(uint16_t) != 2 ) return 1;
  if( sizeof(uint32_t) != 4 ) return 1;
  if( sizeof(uint64_t) != 8 ) return 1;

  if( sizeof(float)    != 4 ) return 1;
  if( sizeof(double)   != 8 ) return 1;

  return 0;
}
