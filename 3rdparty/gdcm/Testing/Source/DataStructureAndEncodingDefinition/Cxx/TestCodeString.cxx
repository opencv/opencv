/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCodeString.h"
#include "gdcmAttribute.h"

#include <iostream>

int TestCodeString(int , char *[])
{
    {
    gdcm::Attribute<0x0008,0x0008> at3;
    static const gdcm::CSComp values[] = {"DERIVED","SECONDARY"};
    at3.SetValues( values, 2, true );
    if( at3.GetNumberOfValues() != 2 ) return 1;
    }

  const char fn1[] = "IMG01";

  gdcm::Attribute< 0x0004, 0x1500 > at;
  at.SetNumberOfValues( 1 );
  at.SetValue( fn1 );

  unsigned int n = at.GetNumberOfValues();
  if( n != 1 ) return 1;

    {
    const char fn2[] = "SUBDIR\\IMG01";
    at.SetNumberOfValues( 2 );
    at.SetValue( fn2 );
    n = at.GetNumberOfValues();
    if( n != 2 ) return 1;
    }

  const char fn3[] = "SUBDIR1\\SUBDIR2\\IMG01 ";
    {
    gdcm::DataElement de( at.GetTag() );
    de.SetByteValue( fn3, (uint32_t)strlen(fn3) );

    at.SetFromDataElement( de );
    n = at.GetNumberOfValues();
    //std::cout << n << std::endl;
    if( n != 3 ) return 1;

    for( unsigned int i = 0; i < n; ++i)
      {
      gdcm::CodeString cs = at.GetValue( i );
      if( !cs.IsValid() )
        {
        std::cerr << "Invalid CS: " << cs << std::endl;
        return 1;
        }
      }
    }

  const char fn4[] = "SUBDIR1\\SUBDIR2\\IMG01";
    {
    std::string copy = fn4;
    if( copy.size() % 2 )
      {
      copy.push_back( ' ' );
      }
    gdcm::DataElement de( at.GetTag() );
    de.SetByteValue( copy.c_str(), (uint32_t)copy.size() );

    at.SetFromDataElement( de );
    n = at.GetNumberOfValues();
    //std::cout << n << std::endl;
    if( n != 3 ) return 1;

    for( unsigned int i = 0; i < n; ++i)
      {
      gdcm::CodeString cs = at.GetValue( i );
      if( !cs.IsValid() )
        {
        std::cerr << "Invalid CS: " << cs << std::endl;
        return 1;
        }
      }
    }

  const char fn5[] = "SUBDIR1\\SUBDIR2\\LONGSUBDIR\\IMG01";
    {
    std::string copy = fn5;
    if( copy.size() % 2 )
      {
      copy.push_back( ' ' );
      }

    gdcm::DataElement de( at.GetTag() );
    de.SetByteValue( copy.c_str(), (uint32_t)copy.size() );

    at.SetFromDataElement( de );
    n = at.GetNumberOfValues();
    //std::cout << n << std::endl;
    if( n != 4 ) return 1;

    for( unsigned int i = 0; i < n; ++i)
      {
      gdcm::CodeString cs = at.GetValue( i );
      if( !cs.IsValid() )
        {
        std::cerr << "Invalid CS: " << cs << std::endl;
        return 1;
        }
      }

    if( strlen(at.GetValue(2) ) < 8 )
      {
      return 1;
      }
    }

    {
    gdcm::CodeString cs0 = " SUB\\DIR ";
    if(  cs0.IsValid() ) return 1;

    gdcm::CodeString cs1 = " SUBDIR ";
    if( !cs1.IsValid() ) return 1;

    gdcm::CodeString cs2 = " SUBDIR_0123456789 ";
    // len == 19 => invalid
    if( cs2.IsValid() ) return 1;

    gdcm::CodeString cs3 = " IMG_0123456789 ";
    if( !cs3.IsValid() ) return 1;

    // cstor should trim on the fly:
    gdcm::CodeString cs4 = "  IMG_0123456789  ";
    if( !cs4.IsValid() ) return 1;

    if( !(cs3 == cs4) ) return 1;

    if( cs3 != cs4 ) return 1;

    gdcm::CodeString cs5 = "IMG";
    if( cs5 != "IMG ")
      return 1;


    // Begin ugly internals.

    }

  return 0;
}
