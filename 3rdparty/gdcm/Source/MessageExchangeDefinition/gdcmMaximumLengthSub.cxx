/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmMaximumLengthSub.h"
#include "gdcmSwapper.h"

namespace gdcm
{
namespace network
{
const uint8_t MaximumLengthSub::ItemType = 0x51;
const uint8_t MaximumLengthSub::Reserved2 = 0x00;

MaximumLengthSub::MaximumLengthSub()
{
  ItemLength = 0x4;
  MaximumLength = 0x4000;
  assert( (size_t)ItemLength + 4 == Size() );
}

std::istream &MaximumLengthSub::Read(std::istream &is)
{
  //uint8_t itemtype = 0x0;
  //is.read( (char*)&itemtype, sizeof(ItemType) );
  //assert( itemtype == ItemType );
  uint8_t reserved2;
  is.read( (char*)&reserved2, sizeof(Reserved2) );
  uint16_t itemlength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;

  uint32_t maximumlength;
  is.read( (char*)&maximumlength, sizeof(MaximumLength) );
  SwapperDoOp::SwapArray(&maximumlength,1);
  MaximumLength = maximumlength; // 16384 == max possible (0x4000)

  return is;
}

const std::ostream &MaximumLengthSub::Write(std::ostream &os) const
{
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  //os.write( (char*)&ItemLength, sizeof(ItemLength) );
  {
  uint16_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );
  }

  //os.write( (char*)&MaximumLength, sizeof(MaximumLength) );
  {
  uint32_t copy = MaximumLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(MaximumLength) );
  }

  return os;
}

size_t MaximumLengthSub::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += sizeof(MaximumLength);

  return ret;
}

void MaximumLengthSub::Print(std::ostream &os) const
{
  os << "MaximumLength: " << MaximumLength << std::endl;
}

void MaximumLengthSub::SetMaximumLength(uint32_t maximumlength)
{
  MaximumLength = maximumlength;
  MaximumLength -= (maximumlength % 2);
}

} // end namespace network
} // end namespace gdcm
