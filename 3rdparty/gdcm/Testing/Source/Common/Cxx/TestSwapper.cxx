/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSwapper.h"


int TestSwapper(int argc, char *argv[])
{
  (void)argv; (void)argc;
  int res = 0;

  typedef union {
    uint64_t v64;
    uint32_t v32[2];
    uint16_t v16[4];
    uint8_t  v8[8];
  } testswapper;
  testswapper t;
  for(uint_fast8_t i = 0; i < 8; ++i) t.v8[i] = i;

  testswapper val;
  val.v64 = gdcm::SwapperDoOp::Swap(t.v64);
  //for(int i = 0; i < 8; ++i) std::cout << (int)val.v8[i] << std::endl;
  for(int i = 0; i < 8; ++i)
    {
    if( val.v8[i] != 8 - i - 1)
      {
      ++res;
      }
    }


  return res;
}
