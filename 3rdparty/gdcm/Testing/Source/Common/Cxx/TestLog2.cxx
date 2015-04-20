/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <iostream>

static inline int log2( int n )
{
  int bits = 0;
  while (n > 0)
    {
    bits++;
    n >>= 1;
    }
  return bits;
}

int TestLog2(int argc, char *argv[])
{
  int v;
  v = log2(255);
  if( v != 8 ) return 1;
  v = log2(256);
  if( v != 9 ) return 1;
  v = log2(4095);
  if( v != 12 ) return 1;
  v = log2(4096);
  if( v != 13 ) return 1;

  return 0;
}
