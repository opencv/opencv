/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmUnpacker12Bits.h"

#include <iostream>
#include <set>
#include <vector>

#include <stdlib.h>

int TestUnpacker12Bits(int, char *[])
{
  int res = 0;
{
  const size_t len = (256 / 3 ) * 3;
  char *values = new char[len];
  for( size_t i = 0; i < len; ++i)
    {
    values[i] = (char)i;
    }
  //const char values[] = {0, 1, 2};
  //const size_t len = sizeof(values) / sizeof(*values);
  const size_t outlen = 16 * len / 12;
  char * output = new char[outlen];
  bool b = gdcm::Unpacker12Bits::Unpack(output, values, len);
  if( b )
    {
    std::set<short> out;
    short * output_s = (short*)output;
    for( size_t i = 0; i < outlen / 2; ++i )
      {
      const short &v = output_s[i];
      // There is no way we can have values greater than a 12bits integer:
      if( v >= 4096 || v < 0 )
        {
        std::cerr << "Too big:" << v << std::endl;
        res = 1;
        }
      // no duplicate possible
      std::pair< std::set<short>::iterator, bool> p = out.insert( v );
      if( p.second == false )
        {
        std::cerr << "duplicate:" << v << std::endl;
        res = 1;
        }
      //std::cout << v << std::endl;
      }
    }
  else
    {
    res = 1;
    }
  delete[] values;
  delete[] output;
}

{
  const unsigned char values[] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab };
  const size_t len = sizeof(values) / sizeof(*values);
  const size_t outlen = 16 * len / 12;
  char * output = new char[outlen];
  bool b = gdcm::Unpacker12Bits::Unpack(output, (char*)values, len);
  if (!b) res = 1;
  if( b )
    {
    unsigned short * output_s = (unsigned short*)output;
    const unsigned short outputvalues[] = { 0x301, 0x452, 0x967, 0xab8 };
    const size_t outputlen = sizeof(outputvalues) / sizeof(*outputvalues);
    assert( outlen / 2 == outputlen );
    for(size_t i = 0; i < outputlen; ++i)
      {
      if( outputvalues[i] != output_s[i] )
        {
        ++res;
        }
      }
    }
  delete[] output;
}

{
  const unsigned short input[] = { 0x301, 0x452, 0x967, 0xab8 };
  unsigned char values[6] = {};
  const unsigned char ref[] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab };
  bool b = gdcm::Unpacker12Bits::Pack((char*)values, (char*)input, 8); // 4 * sizeof(us) == 8
  if(!b)
    {
    return 1;
    }
  for(size_t i = 0; i < 6; ++i)
    {
    if( values[i] != ref[i] )
      {
      assert(0);
      ++res;
      }
    }
}
{
  //struct uint12_t { unsigned short v:12; };
  std::vector<unsigned short> v;
  for(uint16_t val = 0; val < 4096; ++val)
    {
    v.push_back( val );
    }
  assert( v.size() == 4096 );
  assert( v[0] == 0 );
  const size_t outsize = 4096 / 2 * 3;
  unsigned char outvalues[outsize] = {};
  gdcm::Unpacker12Bits::Pack( (char*)outvalues, (char*)&v[0], 4096 * sizeof(unsigned short) );
  unsigned short outvalues2[4096] = {};
  gdcm::Unpacker12Bits::Unpack( (char*)outvalues2, (char*)outvalues, outsize);

  for(uint16_t val = 0; val < 4096; ++val)
    {
    if( v[val] != outvalues2[val] )
      {
      ++res;
      }
    }

}

  return res;
}
