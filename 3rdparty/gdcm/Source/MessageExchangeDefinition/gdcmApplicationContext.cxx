/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmApplicationContext.h"
#include "gdcmSwapper.h"

#include <limits>

namespace gdcm
{
namespace network
{
const uint8_t ApplicationContext::ItemType = 0x10;
const uint8_t ApplicationContext::Reserved2 = 0x00;

// PS 3.7 - 2011
// A.2.1 DICOM Registered Application Context Names
static const char DICOMApplicationContextName[] = "1.2.840.10008.3.1.1.1";

ApplicationContext::ApplicationContext()
{
  UpdateName( DICOMApplicationContextName );
}

std::istream &ApplicationContext::Read(std::istream &is)
{
  //uint8_t itemtype = 0x0;
  //is.read( (char*)&itemtype, sizeof(ItemType) );
  //assert( itemtype == ItemType );
  uint8_t reserved2 = 0x0;
  is.read( (char*)&reserved2, sizeof(Reserved2) );
  uint16_t itemlength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;

  char name[256];
  assert( itemlength < 256 );
  is.read( name, ItemLength );
  Name = std::string(name,itemlength);
  assert( Name == DICOMApplicationContextName );

  return is;
}

const std::ostream &ApplicationContext::Write(std::ostream &os) const
{
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  uint16_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );

  assert( Name == DICOMApplicationContextName );
  os.write( Name.c_str(), Name.size() );
  return os;
}

size_t ApplicationContext::Size() const
{
  size_t ret = 0;
  assert( Name.size() == ItemLength );
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += ItemLength;
  return ret;
}

void ApplicationContext::UpdateName( const char *name )
{
  if( name )
    {
    Name = name;
    assert( Name.size() < std::numeric_limits<uint16_t>::max() );
    ItemLength = (uint16_t)Name.size();
    assert( (size_t)ItemLength + 4 == Size() );
    }
}

void ApplicationContext::Print(std::ostream &os) const
{
  os << Name << std::endl;
}

} // end namespace network
} // end namespace gdcm
