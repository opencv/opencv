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

int TestVersion(int, char *[])
{
  // The following statements just test whether those functions are callable:
  const char *version = gdcm::Version::GetVersion();
  (void)version;
  const int majorVersion = gdcm::Version::GetMajorVersion();
  (void)majorVersion;
  const int minorVersion = gdcm::Version::GetMinorVersion();
  (void)minorVersion;
  const int buildVersion = gdcm::Version::GetBuildVersion();
  (void)buildVersion;

  gdcm::Version v;
  v.Print( std::cout );

  return 0;
}
