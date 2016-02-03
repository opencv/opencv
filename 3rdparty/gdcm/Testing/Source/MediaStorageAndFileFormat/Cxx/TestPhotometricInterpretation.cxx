/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPhotometricInterpretation.h"

#include <string.h> // strlen

int TestPhotometricInterpretation(int, char *[])
{
  gdcm::PhotometricInterpretation pi;
  int end = gdcm::PhotometricInterpretation::PI_END;
  for( int i = 0; i < end; ++i)
    {
    const char *pistr = gdcm::PhotometricInterpretation::GetPIString( (gdcm::PhotometricInterpretation::PIType)i );
    if( strlen( pistr ) % 2 )
      {
      std::cerr << pistr << std::endl;
      return 1;
      }
    }

  pi = gdcm::PhotometricInterpretation::RGB;

  pi = gdcm::PhotometricInterpretation::GetPIType( "MONOCHROME2" );
  if( pi != gdcm::PhotometricInterpretation::MONOCHROME2 )
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  pi = gdcm::PhotometricInterpretation::GetPIType( "MONOCHROME2 " );
  if( pi != gdcm::PhotometricInterpretation::MONOCHROME2 )
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  pi = gdcm::PhotometricInterpretation::GetPIType( " MONOCHROME2 " );
  if( pi != gdcm::PhotometricInterpretation::MONOCHROME2 )
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  pi = gdcm::PhotometricInterpretation::GetPIType( " MONOCHROME2  " );
  if( pi != gdcm::PhotometricInterpretation::MONOCHROME2 )
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  pi = gdcm::PhotometricInterpretation::GetPIType( "  MONOCHROME2  " );
  if( pi != gdcm::PhotometricInterpretation::MONOCHROME2 )
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  pi = gdcm::PhotometricInterpretation::GetPIType( "MONOCHROME" );
  if( pi != gdcm::PhotometricInterpretation::MONOCHROME1 )
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  pi = gdcm::PhotometricInterpretation::GetPIType( "YBR_PARTIAL_42 " );
  if( pi != gdcm::PhotometricInterpretation::YBR_PARTIAL_422)
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  pi = gdcm::PhotometricInterpretation::GetPIType( "PALETTE" );
  if( pi != gdcm::PhotometricInterpretation::PALETTE_COLOR )
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  pi = gdcm::PhotometricInterpretation::GetPIType( "YBR_FULL_4" );
  if( pi != gdcm::PhotometricInterpretation::YBR_FULL_422)
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }
  // FIXME ?
  pi = gdcm::PhotometricInterpretation::GetPIType( "YBR_FUL" );
  if( pi != gdcm::PhotometricInterpretation::YBR_FULL )
    {
    std::cerr << "PhotometricInterpretation: " << pi << std::endl;
    return 1;
    }

  return 0;
}
