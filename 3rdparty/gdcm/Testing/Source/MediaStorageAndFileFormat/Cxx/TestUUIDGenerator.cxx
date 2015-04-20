/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmUUIDGenerator.h"

#include <iostream>

int TestUUIDGenerator(int , char *[])
{
  gdcm::UUIDGenerator uid;

  const char *suid = uid.Generate();
  if( !gdcm::UUIDGenerator::IsValid( suid ) )
    {
    std::cerr << "Invalid: " << suid << std::endl;
    return 1;
    }

  const char *valids[] = {
    "00000000-0000-0000-0000-000000000000",
    "ba209999-0c6c-11d2-97cf-00c04f8eea45",
    "67C8770B-44F1-410A-AB9A-F9B5446F13EE"
  };
  const size_t nv = sizeof( valids ) / sizeof( *valids );
  for( size_t i = 0; i < nv; ++i )
    {
    const char *valid = valids[i];
    if( !gdcm::UUIDGenerator::IsValid( valid ) )
      {
      std::cerr << "Invalid: " << valid << std::endl;
      return 1;
      }
    }

  return 0;
}
