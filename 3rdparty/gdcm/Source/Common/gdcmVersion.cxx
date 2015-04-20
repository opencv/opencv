/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmVersion.h"

namespace gdcm
{

void Version::Print(std::ostream &os) const
{
  os << Version::GetVersion();
}

const char * Version::GetVersion()
{
  return GDCM_VERSION;
}

int Version::GetMajorVersion()
{
  return GDCM_MAJOR_VERSION;
}

int Version::GetMinorVersion()
{
  return GDCM_MINOR_VERSION;
}

int Version::GetBuildVersion()
{
  return GDCM_BUILD_VERSION;
}

}
