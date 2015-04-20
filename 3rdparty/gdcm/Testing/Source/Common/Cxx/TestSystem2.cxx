/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSystem.h"
#include "gdcmTesting.h"

#define _FILE_OFFSET_BITS   64
#include <sys/stat.h>

static bool mybool;
static size_t actualde;

static bool check( const int64_t inslen )
{
  std::cerr << "check:" << inslen << std::endl;
  if( inslen < 0 ) return true;
  return false;
}

static bool append( size_t len )
{
  off_t newlen = len;
#if 1
  newlen -= actualde;
  return check( newlen );
#else
  return check( newlen - actualde );
#endif
}

int TestSystem2(int, char *[])
{
  const int soff = sizeof( off_t );
  std::cerr << soff << std::endl;
  mybool = true;
  actualde = 26;

  off_t o0 = -1;
  off_t o1 = 0;
  //off_t o2 = 1;

    std::cerr << "t:" << o0 << std::endl;
  if( o0 > o1 )
    {
    std::cerr << "Not a long value" << std::endl;
    return 1;
    }
  int val1 = 5;
  int val2 = 10;
  size_t size = 2;
  const off_t o = size;
  
  if( !check( o + val1 - val2 ) )
    {
    return 1;
    }
  if( !append( 2 ) )
    {
    return 1;
    }

  return 0;
}
