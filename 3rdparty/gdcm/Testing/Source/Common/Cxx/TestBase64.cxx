/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmBase64.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"

#include <string.h>

int TestBase64(int , char *[])
{
  const char str[] = "GDCM Test Base64 Encoding";
  //const char str64[] = "R0RDTSBUZXN0IEJhc2U2NCBFbmNvZGluZwA="; (contains trailing \0 )
  const char str64[] = "R0RDTSBUZXN0IEJhc2U2NCBFbmNvZGluZw==";

  //std::cout << "sizeof:" << sizeof(str) << std::endl;
  //std::cout << "strlen:" << strlen(str) << std::endl;
  const size_t l1 = gdcm::Base64::GetEncodeLength( str, strlen(str) );
  if( l1 != 36 )
    {
    std::cerr << "Fail 1: " << l1 << std::endl;
    return 1;
    }

  char buffer[256] = {};
  if( l1 > sizeof(buffer) )
    {
    std::cerr << "Fail 2" << std::endl;
    return 1;
    }

  size_t l2 = gdcm::Base64::Encode( buffer, sizeof(buffer), str, strlen(str) );
  if( l2 == 0 )
    {
    std::cerr << "Fail 3: " << l2 << std::endl;
    return 1;
    }

  if( strcmp( buffer, str64 ) != 0 )
    {
    std::cerr << "Found: " << buffer << " instead of " << str64 << std::endl;
    return 1;
    }

  size_t lbuffer = strlen(buffer);
  if( lbuffer != l1 )
    {
    std::cerr << "Fail 4" << std::endl;
    return 1;
    }

  const size_t l3 = gdcm::Base64::GetDecodeLength( buffer, l1 );
  if( l3 != 25 )
    {
    std::cerr << "Fail 5: " << l3 << std::endl;
    return 1;
    }

  if( l3 != sizeof(str) - 1 )
    {
    std::cerr << "Fail 6" << std::endl;
    return 1;
    }

  char buffer2[256];
  if( l3 > sizeof(buffer2) )
    {
    std::cerr << "Fail 7" << std::endl;
    return 1;
    }
  const size_t l4 = gdcm::Base64::Decode( buffer2, sizeof(buffer2), buffer, l1);
  if( l4 == 0 )
    {
    std::cerr << "Fail 8" << std::endl;
    return 1;
    }

  if( strncmp( str, buffer2, strlen(str) ) != 0 )
    {
    std::cerr << "Fail 9: " << str << " vs " << buffer2 << std::endl;
    return 1;
    }

  const unsigned char bin[] = { 0x00, 0x00, 0xc8, 0x43 };
  const char bin64[] = "AADIQw==";

  const size_t l5 = gdcm::Base64::Decode( buffer2, sizeof(buffer2), bin64, strlen(bin64) );
  if( l5 == 0 )
    {
    std::cerr << "Fail 10" << std::endl;
    return 1;
    }

  if( memcmp( bin, buffer2, sizeof(bin) ) != 0 )
    {
    std::cerr << "Fail 11"  << std::endl;
    return 1;
    }
  

  return 0;
}
