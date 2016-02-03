/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmUsage.h"
#include <string.h>

namespace gdcm
{

static const char *UsageStrings[] = {
  "Mandatory", // (see A.1.3.1) , abbreviated M
  "Conditional", // (see A.1.3.2) , abbreviated C
  "UserOption", // (see A.1.3.3) , abbreviated U
  NULL
};

const char *Usage::GetUsageString(UsageType type)
{
  return UsageStrings[type];
}

Usage::UsageType Usage::GetUsageType(const char *type)
{
  int i = 0;
  while(UsageStrings[i] != 0)
    {
    if( strcmp(type, UsageStrings[i]) == 0 )
      return (UsageType)i;
    ++i;
    }
  return Invalid;
}

} // end namespace gdcm
