/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPresentationContextRQ.h"

#include "gdcmPresentationContext.h"
#include "gdcmUIDs.h"
#include "gdcmSwapper.h"
#include "gdcmAttribute.h"
#include "gdcmGlobal.h"
#include "gdcmMediaStorage.h"

#include <limits>

namespace gdcm
{
namespace network
{
const uint8_t PresentationContextRQ::ItemType = 0x20;
const uint8_t PresentationContextRQ::Reserved2 = 0x00;
const uint8_t PresentationContextRQ::Reserved6 = 0x00;
const uint8_t PresentationContextRQ::Reserved7 = 0x00;
const uint8_t PresentationContextRQ::Reserved8 = 0x00;

PresentationContextRQ::PresentationContextRQ()
{
  ID = 0x01;
  ItemLength = 8;
  assert( (size_t)ItemLength + 4 == Size() );
}

PresentationContextRQ::PresentationContextRQ( UIDs::TSName asname, UIDs::TSName tsname )
{
  ID = 0x01;
  AbstractSyntax as;
  as.SetNameFromUID( asname );
  SetAbstractSyntax( as );

  TransferSyntaxSub ts;
  ts.SetNameFromUID( tsname );
  assert( TransferSyntaxes.empty() );
  AddTransferSyntax( ts );
}

std::istream &PresentationContextRQ::Read(std::istream &is)
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
  uint8_t reserved7;
  is.read( (char*)&reserved7, sizeof(Reserved7) );
//  assert( reserved7 == 0 );
  //no need for this assert--'This reserved field shall be sent with a value 00H but not tested to this value when received.'
  uint8_t reserved8;
  is.read( (char*)&reserved8, sizeof(Reserved6) );
  SubItems.Read( is );

  size_t curlen = 0;
  size_t offset = SubItems.Size() + 4;
  while( curlen + offset < ItemLength )
    {
    TransferSyntaxSub ts;
    ts.Read( is );
    TransferSyntaxes.push_back( ts );
    curlen += ts.Size();
    }
  assert( curlen + offset == ItemLength );

  assert( (size_t)ItemLength + 4 == Size() );
  return is;
}

const std::ostream &PresentationContextRQ::Write(std::ostream &os) const
{
  assert( (size_t)ItemLength + 4 == Size() );
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  uint16_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );

  os.write( (char*)&ID, sizeof(ID) );
  os.write( (char*)&Reserved6, sizeof(Reserved6) );
  os.write( (char*)&Reserved7, sizeof(Reserved7) );
  os.write( (char*)&Reserved8, sizeof(Reserved8) );
  SubItems.Write(os);
  std::vector<TransferSyntaxSub>::const_iterator it = TransferSyntaxes.begin();
  for( ; it != TransferSyntaxes.end(); ++it )
    {
    it->Write( os );
    }

  return os;
}

size_t PresentationContextRQ::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += sizeof(ID);
  ret += sizeof(Reserved6);
  ret += sizeof(Reserved7);
  ret += sizeof(Reserved8);
  ret += SubItems.Size();
  std::vector<TransferSyntaxSub>::const_iterator it = TransferSyntaxes.begin();
  for( ; it != TransferSyntaxes.end(); ++it )
    {
    ret += it->Size();
    }

  assert(ret <= (size_t)std::numeric_limits<uint16_t>::max);
  assert(ret >= 4);
  return ret;
}

void PresentationContextRQ::SetAbstractSyntax( AbstractSyntax const & as )
{
  SubItems = as;
  ItemLength = (uint16_t)(Size() - 4);
  assert( (size_t)ItemLength + 4 == Size() );
}

void PresentationContextRQ::AddTransferSyntax( TransferSyntaxSub const &ts )
{
  TransferSyntaxes.push_back( ts );
  ItemLength = (uint16_t)(Size() - 4);
  assert( (size_t)ItemLength + 4 == Size() );
}

void PresentationContextRQ::SetPresentationContextID( uint8_t id )
{
  assert( id );
  ID = id;
}

uint8_t PresentationContextRQ::GetPresentationContextID() const
{
  return ID;
}

void PresentationContextRQ::Print(std::ostream &os) const
{
  //static const uint8_t ItemType;
  //static const uint8_t Reserved2;
  os << "ItemLength: " << ItemLength << std::endl; // len of last transfer syntax
  os << "PresentationContext ID: " << (int)ID << std::endl;
  //static const uint8_t Reserved6;
  //static const uint8_t Reserved7;
  //static const uint8_t Reserved8;
  SubItems.Print( os );
  std::vector<TransferSyntaxSub>::const_iterator it = TransferSyntaxes.begin();
  for( ; it != TransferSyntaxes.end(); ++it )
    {
    it->Print( os );
    }
}

PresentationContextRQ::PresentationContextRQ(const PresentationContext & in)
{
  AbstractSyntax as;
  as.SetName( in.GetAbstractSyntax() );
  SetAbstractSyntax( as );
  size_t n = in.GetNumberOfTransferSyntaxes();
  TransferSyntaxes.clear();
  for( size_t j = 0; j < n; ++j )
    {
    TransferSyntaxSub ts;
    ts.SetName( in.GetTransferSyntax(j) );
    AddTransferSyntax( ts );
    }
  SetPresentationContextID( in.GetPresentationContextID() );
  assert( GetNumberOfTransferSyntaxes() == in.GetNumberOfTransferSyntaxes() );
}

} // end namespace network
} // end namespace gdcm
