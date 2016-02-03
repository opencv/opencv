/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSwapCode.h"
#include <assert.h>

namespace gdcm
{

static const char *SwapCodeStrings[] = {
  "Unknown",
  "LittleEndian",
  "BigEndian",
  "BadLittleEndian",
  "BadBigEndian",
   0
};

int SwapCode::GetIndex(SwapCode const & sc)
{
  int idx = 0;
  switch(sc)
    {
  case Unknown:
    idx = 0;
    break;
  case LittleEndian:
    idx = 1;
    break;
  case BigEndian:
    idx = 2;
    break;
  case BadLittleEndian:
    idx = 3;
    break;
  case BadBigEndian:
    idx = 4;
    break;
  default:
    assert(0 && "Should not happen" );
    }
  assert( idx < 5 );
  return idx;
}

const char* SwapCode::GetSwapCodeString(SwapCode const & sc)
{
  int idx = GetIndex(sc);
  return SwapCodeStrings[idx];
}

/*
std::ostream& operator<<(std::ostream& os, const SwapCode& sc)
{
  os << SwapCode::GetSwapCodeString(sc);
  return os;
}
*/
} // namespace gdcm
