/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPresentationContextAC.h"
#include "gdcmSwapper.h"

#include <limits>

namespace gdcm
{
namespace network
{
const uint8_t PresentationContextAC::ItemType = 0x21;
const uint8_t PresentationContextAC::Reserved2 = 0x00;
const uint8_t PresentationContextAC::Reserved6 = 0x00;
const uint8_t PresentationContextAC::Reserved8 = 0x00;

PresentationContextAC::PresentationContextAC()
{
  ID = 1; // odd [1-255]
  Result = 0;
  ItemLength = 8;
  assert( (size_t)ItemLength + 4 == Size() );
}

std::istream &PresentationContextAC::Read(std::istream &is)
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
  uint8_t id;
  is.read( (char*)&id, sizeof(ID) );
  ID = id;
  uint8_t reserved6;
  is.read( (char*)&reserved6, sizeof(Reserved6) );
  uint8_t result;
  is.read( (char*)&result, sizeof(Result) );
  Result = result; // 0-4
  uint8_t reserved8;
  is.read( (char*)&reserved8, sizeof(Reserved6) );
  SubItems.Read( is );

  assert( (size_t)ItemLength + 4 == Size() );
  return is;
}

const std::ostream &PresentationContextAC::Write(std::ostream &os) const
{
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  //os.write( (char*)&ItemLength, sizeof(ItemLength) );
  uint16_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );
  os.write( (char*)&ID, sizeof(ID) );
  os.write( (char*)&Reserved6, sizeof(Reserved6) );
  os.write( (char*)&Result, sizeof(Result) );
  os.write( (char*)&Reserved8, sizeof(Reserved8) );
  SubItems.Write(os);

  assert( (size_t)ItemLength + 4 == Size() );
  return os;
}

size_t PresentationContextAC::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += sizeof(ID);
  ret += sizeof(Reserved6);
  ret += sizeof(Result);
  ret += sizeof(Reserved8);
  ret += SubItems.Size();

  assert(ret <= (size_t)std::numeric_limits<uint16_t>::max);
  assert(ret >= 4);
  return ret;
}

void PresentationContextAC::SetTransferSyntax( TransferSyntaxSub const &ts )
{
  SubItems = ts;
  ItemLength = (uint16_t)(Size() - 4);
  assert( (size_t)ItemLength + 4 == Size() );
}

void PresentationContextAC::Print(std::ostream &os) const
{
  os << "ID: " << (int)ID << std::endl;
  os << "Result: " << (int)Result << std::endl;
  os << "TransferSyntax: ";
  SubItems.Print( os );
}

void PresentationContextAC::SetPresentationContextID( uint8_t id )
{
  ID = id;
}

} // end namespace network
} // end namespace gdcm
