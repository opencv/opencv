/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPixelFormat.h"

int TestPixelFormat(int , char *[])
{
  using gdcm::PixelFormat;
  gdcm::PixelFormat pf;
  pf.SetScalarType( gdcm::PixelFormat::UNKNOWN );
  if( pf.GetScalarType() != gdcm::PixelFormat::UNKNOWN )
    {
    return 1;
    }
  pf.SetScalarType( gdcm::PixelFormat::UINT32 );
  //pf.SetScalarType( gdcm::PixelFormat::UINT64 );
  static const int64_t values[][2] = {
     { 0LL,255LL },
     { 0LL,4095LL },
     { 0LL,65535LL },
     { 0LL,4294967295LL },
     { -128LL,127LL },
     { -2048LL,2047LL },
     { -32768LL,32767LL },
     { -2147483648LL,2147483647LL },
//     { -2147483648LL,2147483647LL }
  };
  static const size_t n = sizeof( values ) / sizeof( *values );
  size_t c = 0;
  pf.SetBitsStored( 8 );
  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
    {
    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
    return 1;
    }
  ++c;
  pf.SetBitsStored( 12 );
  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
    {
    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
    return 1;
    }
  ++c;
  pf.SetBitsStored( 16 );
  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
    {
    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
    return 1;
    }
  ++c;
  pf.SetBitsStored( 32 );
  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
    {
    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
    return 1;
    }
//  ++c;
//  pf.SetBitsStored( 64 );
//  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
//    {
//    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
//    return 1;
//    }

  pf.SetPixelRepresentation( 1 );

  ++c;
  pf.SetBitsStored( 8 );
  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
    {
    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
    return 1;
    }
  ++c;
  pf.SetBitsStored( 12 );
  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
    {
    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
    return 1;
    }
  ++c;
  pf.SetBitsStored( 16 );
  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
    {
    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
    return 1;
    }
  ++c;
  pf.SetBitsStored( 32 );
  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
    {
    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
    return 1;
    }
//  ++c;
//  pf.SetBitsStored( 64 );
//  if( pf.GetMin() != values[c][0] || pf.GetMax() != values[c][1] )
//    {
//    std::cerr << pf.GetMin() << "," << pf.GetMax() << std::endl;
//    return 1;
//    }
  ++c;
  if ( c != n ) return 1;

  for(unsigned int i = 0; i < PixelFormat::UNKNOWN; ++i)
    {
    PixelFormat::ScalarType st = (PixelFormat::ScalarType)i;
    pf.SetScalarType( st );
    gdcm::PixelFormat pf2 = st;
    std::cout << pf << std::endl;
    std::cout << pf.GetPixelRepresentation() << std::endl;
    std::cout << pf.GetScalarTypeAsString() << std::endl;
    if( pf2 != pf ) return 1;
    }

  // make to avoid user mistakes:
  gdcm::PixelFormat pf3 = PixelFormat::UINT8;
  if( pf3.GetBitsStored() != 8 ) return 1;
  pf3.SetBitsStored( 32 );
  // previous call should not execute
  if( pf3.GetBitsStored() != 8 ) return 1;
  pf3.SetHighBit( 8 );
  if( pf3.GetHighBit() != 7 ) return 1;

  return 0;
}

