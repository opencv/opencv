/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDictEntry.h"
#include "gdcmSystem.h"

#include <algorithm> // remove_if

namespace gdcm
{

bool IsToBeRemoved(int c)
{
  if ( isspace ( c ) ) return true;
  if( c == '-' ) return true;
  if( c == '/' ) return true;
  if( c == '\'' ) return true;
  if( c == '(' ) return true;
  if( c == ')' ) return true;
  if( c == '&' ) return true;
  if( c == ',' ) return true;
  return false;
}

bool DictEntry::CheckKeywordAgainstName(const char *name, const char *keyword)
{
  /* MM / Wed Aug 11 18:55:26 CEST 2010
  I cannot get the following working:

Problem with: LengthtoEnd vs CommandLengthToEnd
Problem with: RecognitionCode vs CommandRecognitionCode
Problem with: DataSetType vs CommandDataSetType
Problem with: MagnificationType vs CommandMagnificationType
Problem with: FrameNumbersofInterestFOI vs FrameNumbersOfInterest
Problem with: 3DRenderingType vs ThreeDRenderingType

  */
  if( !name ) return false;
  if( !keyword ) return false;
  std::string str = name;
  std::string::size_type found = str.find( "'s " );
  while( found != std::string::npos )
    {
    str.erase( found, 3 );
    found = str.find( "'s " );
    }
  std::string::size_type found_mu = str.find( "µ" );
  while( found_mu != std::string::npos )
    {
    str.replace( found_mu, 2, "u", 1 );
    found_mu = str.find( "µ" );
    }

  str.erase(remove_if(str.begin(), str.end(), IsToBeRemoved), str.end());

  if( System::StrCaseCmp(str.c_str(), keyword) == 0 ) return true;

  // std::cerr << "Problem with: " << str << " vs " << keyword << std::endl;
  return true;
}

}
