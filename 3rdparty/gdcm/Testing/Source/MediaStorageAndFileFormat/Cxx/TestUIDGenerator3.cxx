/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmUIDGenerator.h"
#include "gdcmSystem.h"

#include <bitset>
#include <iostream>

int TestUIDGenerator3(int argc, char *argv[])
{
  char randbytesbuf[200];
  gdcm::UIDGenerator uid;
  unsigned char data[16];
  for(unsigned int i = 0; i < 100; ++i)
    {
    uid.GenerateUUID( data );

    size_t len = gdcm::System::EncodeBytes(randbytesbuf, data, sizeof(data));
    //std::cout << randbytesbuf << std::endl;

    std::bitset<8> x;
    //x.reset();
    //std::cout << x << std::endl;
    //x.flip();
    //std::cout << x << std::endl;
    //std::cout << sizeof(x) << std::endl;
    //std::cout << sizeof(data) << std::endl;
    x = data[0];
    //std::cout << x << std::endl;
    //std::cout << (int)data[0] << std::endl;
    //x = data[5];
    //std::cout << x << std::endl;
    x[2+0] = 0;
    x[2+1] = 0;
    x[2+2] = 0;
    x[2+3] = 0;
    x[2+4] = 0;
    x[2+5] = 0;
    data[0] = x.to_ulong();
    //std::cout << x << std::endl;
    //std::cout << (int)data[0] << std::endl;

    len = gdcm::System::EncodeBytes(randbytesbuf, data, sizeof(data));
    std::cout << randbytesbuf << std::endl;
    if( len > 37 )
      {
      return 1;
      }
    // Can't use the following, this would declare x2 as a function
    //std::bitset<128> x2( std::string(randbytesbuf) );
    // instead split it out :
    //std::string s(randbytesbuf);
    //std::bitset<128> x2( s );
    //std::cout << x2 << std::endl;

    }
  return 0;
}
