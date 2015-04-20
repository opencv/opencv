/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTag.h"
#include "gdcmTrace.h"

#include <stdio.h> // sscanf

namespace gdcm
{
  bool Tag::ReadFromCommaSeparatedString(const char *str)
    {
    unsigned int group = 0, element = 0;
    if( !str || sscanf(str, "%04x,%04x", &group , &element) != 2 )
      {
      gdcmDebugMacro( "Problem reading Tag: " << str );
      return false;
      }
    SetGroup( (uint16_t)group );
    SetElement( (uint16_t)element );
    return true;
    }
  bool Tag::ReadFromContinuousString(const char *str)
    {
    unsigned int group = 0, element = 0;
    if( !str || sscanf(str, "%04x%04x", &group , &element) != 2 )
      {
      gdcmDebugMacro( "Problem reading Tag: " << str );
      return false;
      }
    SetGroup( (uint16_t)group );
    SetElement( (uint16_t)element );
    return true;
    }  
  bool Tag::ReadFromPipeSeparatedString(const char *str)
    {
    unsigned int group = 0, element = 0;
    if( !str || sscanf(str, "%04x|%04x", &group , &element) != 2 )
      {
      gdcmDebugMacro( "Problem reading Tag: " << str );
      return false;
      }
    SetGroup( (uint16_t)group );
    SetElement( (uint16_t)element );
    return true;
    }

  std::string Tag::PrintAsContinuousString() const
    {
    std::ostringstream os;
    const Tag &val = *this;
    os.setf( std::ios::right);
    os << std::hex << std::setw( 4 ) << std::setfill( '0' )
      << val[0] << std::setw( 4 ) << std::setfill( '0' )
      << val[1] << std::setfill( ' ' ) << std::dec;
    return os.str();
    }

  std::string Tag::PrintAsContinuousUpperCaseString() const
    {
    std::ostringstream os;
    const Tag &val = *this;
    os.setf( std::ios::right);
    os << std::uppercase << std::hex << std::setw( 4 ) << std::setfill( '0' )
      << val[0] << std::setw( 4 ) << std::setfill( '0' )
      << val[1] << std::setfill( ' ' ) << std::dec;
    return os.str();
    }

  std::string Tag::PrintAsPipeSeparatedString() const
    {
    std::ostringstream _os;
    const Tag &_val = *this;
    _os.setf( std::ios::right);
    _os << std::hex << std::setw( 4 ) << std::setfill( '0' )
      << _val[0] << '|' << std::setw( 4 ) << std::setfill( '0' )
      << _val[1] << std::setfill( ' ' ) << std::dec;
    return _os.str();
    }

} // end namespace gdcm
