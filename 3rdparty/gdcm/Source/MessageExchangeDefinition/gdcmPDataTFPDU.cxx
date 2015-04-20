/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPDataTFPDU.h"
#include "gdcmSwapper.h"

namespace gdcm
{
namespace network
{
const uint8_t PDataTFPDU::ItemType = 0x04; // PDUType ?
const uint8_t PDataTFPDU::Reserved2 = 0x00;

PDataTFPDU::PDataTFPDU()
{
  assert(Size() < std::numeric_limits<uint32_t>::max());
  ItemLength = (uint32_t)Size() - 6;
  assert( (ItemLength + 4 + 1 + 1) == Size() );
}

std::istream &PDataTFPDU::Read(std::istream &is)
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

  size_t curlen = 0;
  while( curlen < ItemLength )
    {
    PresentationDataValue pdv;
    pdv.Read( is );
    V.push_back( pdv );
    curlen += pdv.Size();
    }
  assert( curlen == ItemLength );
  assert( (ItemLength + 4 + 1 + 1) == Size() );

  return is;
}

std::istream &PDataTFPDU::ReadInto(std::istream &is, std::ostream &os)
{
  uint8_t itemtype = 0;
  is.read( (char*)&itemtype, sizeof(ItemType) );
  assert( itemtype == ItemType );
  uint8_t reserved2 = 0;
  is.read( (char*)&reserved2, sizeof(Reserved2) );
  uint32_t itemlength = ItemLength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;

  size_t curlen = 0;
  while( curlen < ItemLength )
    {
    PresentationDataValue pdv;
    pdv.ReadInto( is, os );
    V.push_back( pdv );
    curlen += pdv.Size();
    }
  assert( curlen == ItemLength );
  assert( (ItemLength + 4 + 1 + 1) == Size() );

  return is;
}

const std::ostream &PDataTFPDU::Write(std::ostream &os) const
{
  assert( (ItemLength + 4 + 1 + 1) == Size() );
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  //os.write( (char*)&ItemLength, sizeof(ItemLength) );
  uint32_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );
  std::vector<PresentationDataValue>::const_iterator it = V.begin();
  for( ; it != V.end(); ++it )
    {
    it->Write( os );
    }

  return os;
}

size_t PDataTFPDU::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  std::vector<PresentationDataValue>::const_iterator it = V.begin();
  for( ; it != V.end(); ++it )
    {
    ret += it->Size( );
    }

  return ret;
}

void PDataTFPDU::Print(std::ostream &os) const
{
  //static const uint8_t ItemType; // PDUType ?
  //static const uint8_t Reserved2;
  os << "ItemLength: " << ItemLength << std::endl; // PDU Length ?
  os << "PresentationDataValue: " << std::endl;
  std::vector<PresentationDataValue>::const_iterator it = V.begin();
  for( ; it != V.end(); ++it )
    {
    it->Print( os );
    }
  os << std::endl;
}


bool PDataTFPDU::IsLastFragment() const
{
  if (V.empty()) return true;
  return V[V.size()-1].GetIsLastFragment();
}

} // end namespace network
} // end namespace gdcm
