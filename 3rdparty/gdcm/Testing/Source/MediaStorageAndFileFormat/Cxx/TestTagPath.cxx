/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTagPath.h"
#include "gdcmTag.h"

int TestTagPath(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  gdcm::TagPath tp;

  const char path[] = "/0010,0010";
  if( !gdcm::TagPath::IsValid( path ) )
    {
    return 1;
    }
  if( !tp.ConstructFromString( path ) )
    {
    return 1;
    }

  tp.Print( std::cout );

  const char path2[] = "/0010,0011/*/1234,5678";
  if( !gdcm::TagPath::IsValid( path2 ) )
    {
    return 1;
    }
  if( !tp.ConstructFromString( path2 ) )
    {
    return 1;
    }

  std::cout << "FromString:" << std::endl;
  tp.Print( std::cout );

  const unsigned int n = 10;
  gdcm::Tag list[n];
  for(unsigned i = 0; i < n; ++i)
    {
    list[i].SetGroup( 0x1234 );
    list[i].SetElement( (uint16_t)i );
    }

  if( !tp.ConstructFromTagList( list, n ) )
    {
    return 1;
    }
  std::cout << "TagList:" << std::endl;
  tp.Print( std::cout );


  return 0;
}
