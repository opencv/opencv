/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAReleaseRPPDU.h"
#include "gdcmSwapper.h"

namespace gdcm
{
namespace network
{
const uint8_t AReleaseRPPDU::ItemType = 0x6; // PDUType ?
const uint8_t AReleaseRPPDU::Reserved2 = 0x0;
const uint32_t AReleaseRPPDU::Reserved7_10 = 0x0;

AReleaseRPPDU::AReleaseRPPDU()
{
  ItemLength = (uint32_t)(Size() - 6); // PDU Length
  assert( ItemLength + 6 == Size() );
}

std::istream &AReleaseRPPDU::Read(std::istream &is)
{
  //uint8_t itemtype = 0;
  //is.read( (char*)&itemtype, sizeof(ItemType) );
  //assert( itemtype == ItemType );
  uint8_t reserved2 = 0;
  is.read( (char*)&reserved2, sizeof(Reserved2) );
  uint32_t itemlength = ItemLength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;
  uint32_t reserved7_10;
  is.read( (char*)&reserved7_10, sizeof(Reserved7_10) );

  assert( ItemLength + 6 == Size() );
  return is;
}

const std::ostream &AReleaseRPPDU::Write(std::ostream &os) const
{
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  uint32_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );
  os.write( (char*)&Reserved7_10, sizeof(Reserved7_10) );

  assert( ItemLength + 6 == Size() );

  return os;
}

size_t AReleaseRPPDU::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength); // len of
  ret += sizeof(Reserved7_10);

  return ret;
}

void AReleaseRPPDU::Print(std::ostream &os) const
{
  os << "PDULength: " << ItemLength << std::endl;
}

} // end namespace network
} // end namespace gdcm
