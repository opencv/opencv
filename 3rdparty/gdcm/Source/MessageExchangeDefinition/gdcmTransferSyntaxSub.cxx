/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTransferSyntaxSub.h"
#include "gdcmSwapper.h"

#include <limits>

namespace gdcm
{
namespace network
{
const uint8_t TransferSyntaxSub::ItemType = 0x40;
const uint8_t TransferSyntaxSub::Reserved2 = 0x00;

TransferSyntaxSub::TransferSyntaxSub()
{
  //UpdateName( "1.2.840.10008.1.1" );
  ItemLength = 0;
}

void TransferSyntaxSub::SetName( const char *name )
{
  if( name )
    {
    Name = name;
    assert( Name.size() <= std::numeric_limits<uint16_t>::max() );
    ItemLength = (uint16_t)Name.size();
    }
}

std::istream &TransferSyntaxSub::Read(std::istream &is)
{
  uint8_t itemtype = 0xf;
  is.read( (char*)&itemtype, sizeof(ItemType) );
  assert( itemtype == ItemType );
  uint8_t reserved2;
  is.read( (char*)&reserved2, sizeof(Reserved2) );
  uint16_t itemlength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;

  char name[256];
  assert( itemlength < 256 );
  is.read( name, itemlength );
  Name = std::string(name,itemlength);

  return is;
}

const std::ostream &TransferSyntaxSub::Write(std::ostream &os) const
{
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  //os.write( (char*)&ItemLength, sizeof(ItemLength) );
  uint16_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );

  assert( Name.size() < 256 );
  os.write( Name.c_str(), Name.size() );
  return os;
}

size_t TransferSyntaxSub::Size() const
{
  size_t ret = 0;
  assert( Name.size() == ItemLength );
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += ItemLength;
  return ret;
}

void TransferSyntaxSub::UpdateName( const char *name )
{
  if( name )
    {
    UIDs uids;
    bool b = uids.SetFromUID( name );
    if( b )
      {
      Name = name;
      ItemLength = (uint16_t)Name.size();
      assert( (size_t)ItemLength + 4 == Size() );
      return;
      }
    }

  gdcmErrorMacro( "Invalid Name: " << name );
  throw "Invalid Name";
}

void TransferSyntaxSub::SetNameFromUID( UIDs::TSName tsname )
{
  const char *name = UIDs::GetUIDString( tsname );
  UpdateName( name );
}

void TransferSyntaxSub::Print(std::ostream &os) const
{
  os << "Name: " << Name;
  UIDs uids;
  if( uids.SetFromUID( Name.c_str() ) )
    {
    os << " (" << uids.GetName() << ")" << std::endl;
    }
  os << std::endl;
}

} // end namespace network
} // end namespace gdcm
