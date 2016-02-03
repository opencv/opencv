/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmType.h"
#include <string.h>

namespace gdcm
{

static const char *TypeStrings[] = {
   "1",
   "1C",
   "2",
   "2C",
   "3",
   "UNKNOWN",
   0
};

const char *Type::GetTypeString(TypeType type)
{
  return TypeStrings[type];
}

Type::TypeType Type::GetTypeType(const char *type)
{
  int i = 0;
  while(TypeStrings[i] != 0)
    {
    if( strcmp(type, TypeStrings[i]) == 0 )
      return (TypeType)i;
    ++i;
    }
  return UNKNOWN;
}

} // end namespace gdcm
