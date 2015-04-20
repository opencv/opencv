/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSOPClassUIDToIOD.h"

#include <iostream>

int TestSOPClassUIDToIOD(int, char *[])
{

  gdcm::SOPClassUIDToIOD::SOPClassUIDToIODType& s = gdcm::SOPClassUIDToIOD::GetSOPClassUIDToIOD(0);
  std::cout << s[0] << std::endl;
  if( std::string(s[0] ) != "1.2.840.10008.1.3.10" ) return 1;
  std::cout << s[1] << std::endl;
  if( std::string(s[1] ) != "Basic Directory IOD Modules" ) return 1;

  gdcm::SOPClassUIDToIOD::SOPClassUIDToIODType& s2 = gdcm::SOPClassUIDToIOD::GetSOPClassUIDToIOD(100);
  std::cout << ( s2[0] == 0 ) << std::endl;
  if( !(s2[0] == 0) ) return 1;
  std::cout << ( s2[1] == 0 ) << std::endl;
  if( !(s2[1] == 0) ) return 1;

  const char *sopclassuid = gdcm::SOPClassUIDToIOD::GetSOPClassUIDFromIOD( s[1] );
  const char *iod = gdcm::SOPClassUIDToIOD::GetIODFromSOPClassUID( s[0] );
  std::cout << sopclassuid << std::endl;
  std::cout << iod << std::endl;
  if( std::string(sopclassuid) != s[0] ) return 1;
  if( std::string(iod) != s[1] ) return 1;

  return 0;
}
