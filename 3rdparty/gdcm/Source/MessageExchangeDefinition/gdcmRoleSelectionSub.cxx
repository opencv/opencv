/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmRoleSelectionSub.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmSwapper.h"

namespace gdcm
{
namespace network
{
const uint8_t RoleSelectionSub::ItemType = 0x54;
const uint8_t RoleSelectionSub::Reserved2 = 0x00;

RoleSelectionSub::RoleSelectionSub()
{
  ItemLength = 0;
  UIDLength = 0;
  SCURole = 0;
  SCPRole = 0;

  ItemLength = (uint16_t)(Size() - 4);
  assert( (size_t)ItemLength + 4 == Size() );
}

std::istream &RoleSelectionSub::Read(std::istream &is)
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

  uint16_t uidlength;
  is.read( (char*)&uidlength, sizeof(UIDLength) );
  SwapperDoOp::SwapArray(&uidlength,1);
  UIDLength = uidlength;

  char name[256];
  assert( uidlength < 256 );
  is.read( name, uidlength );
  Name = std::string(name,uidlength);

  uint8_t scurole;
  is.read( (char*)&scurole, sizeof(SCURole) );
  SCURole = scurole;

  uint8_t scprole;
  is.read( (char*)&scprole, sizeof(SCPRole) );
  SCPRole = scprole;

  assert( (size_t)ItemLength + 4 == Size() );

  return is;
}

const std::ostream &RoleSelectionSub::Write(std::ostream &os) const
{
  assert( (size_t)ItemLength + 4 == Size() );

  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  //os.write( (char*)&ItemLength, sizeof(ItemLength) );
  uint16_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );

  assert( ItemLength > UIDLength );
  uint16_t uidlength = UIDLength;
  SwapperDoOp::SwapArray(&uidlength,1);
  os.write( (char*)&uidlength, sizeof(UIDLength) );

  assert( (size_t)UIDLength == Name.size() );
  os.write( Name.c_str(), Name.size() );

  uint8_t scurole = SCURole;
  assert( scurole == 0 || scurole == 1 );
  os.write( (char*)&scurole, sizeof(SCURole) );

  uint8_t scprole = SCPRole;
  assert( scprole == 0 || scprole == 1 );
  os.write( (char*)&scprole, sizeof(SCPRole) );

  return os;
}

size_t RoleSelectionSub::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += sizeof(UIDLength);
  assert( Name.size() == UIDLength );
  ret += UIDLength;
  ret += sizeof(SCURole);
  ret += sizeof(SCPRole);

  return ret;
}

/*
SCU-role This byte field shall contain the SCU-role as defined for the
         Association-requester in Section D.3.3.4. It shall be encoded
         as an unsigned binary and shall use one of the following
         values:
         0 - non support of the SCU role
         1 - support of the SCU role

SCP-role This byte field shall contain the SCP-role as defined for the
         Association-requester in Section D.3.3.4. It shall be encoded
         as an unsigned binary and shall use one of the following
         values:
         0 - non support of the SCP role
         1 - support of the SCP role.
*/
void RoleSelectionSub::SetTuple(const char *uid, uint8_t scurole, uint8_t scprole)
{
  if( uid )
    {
    Name = uid;
    UIDLength = (uint16_t)strlen( uid );
    assert( (size_t)UIDLength == Name.size() );
    SCURole = scurole % 2;
    SCPRole = scprole % 2;
    ItemLength = (uint16_t)(Size() - 4);
    }
  // post condition
  assert( (size_t)ItemLength + 4 == Size() );
}

void RoleSelectionSub::Print(std::ostream &os) const
{
  os << "SOP-class-uid" << Name << std::endl;
  os << "SCURole: " << (int)SCURole << std::endl;
  os << "SCPRole: " << (int)SCPRole << std::endl;
}

} // end namespace network
} // end namespace gdcm
