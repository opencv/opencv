/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmByteValue.h"

#include "gdcmSwapper.h"

int TestByteValue(int, char *[])
{
  const char array[] = "GDCM";
  const size_t len = strlen(array);
  gdcm::ByteValue bv1( array, len );
  std::cout << bv1 << std::endl;
  if( memcmp(bv1.GetPointer(), array, len ) != 0 )
    {
    return 1;
    }
  std::stringstream ss( array );
  gdcm::ByteValue bv2;
  bv2.SetLength( len );
  bv2.Read<gdcm::SwapperNoOp>( ss );
  std::cout << bv2 << std::endl;
  if( memcmp(bv2.GetPointer(), array, len ) != 0 )
    {
    return 1;
    }
  if( !(bv1 == bv2) )
    {
    return 1;
    }

  gdcm::ByteValue bv3(bv2);
  if( memcmp(bv3.GetPointer(), array, len ) != 0 )
    {
    return 1;
    }
  if( !(bv3 == bv1) )
    {
    return 1;
    }
  gdcm::ByteValue bv4 = bv3;
  if( memcmp(bv4.GetPointer(), array, len ) != 0 )
    {
    return 1;
    }
  if( !(bv4 == bv1) )
    {
    return 1;
    }
  gdcm::ByteValue bv5;
  if( bv5 == bv1 )
    {
    return 1;
    }

  return 0;
}
