/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmUserInformation.h"
#include "gdcmSwapper.h"
#include "gdcmAsynchronousOperationsWindowSub.h"
#include "gdcmRoleSelectionSub.h"
#include "gdcmSOPClassExtendedNegociationSub.h"

#include <vector>

namespace gdcm
{
namespace network
{

const uint8_t UserInformation::ItemType = 0x50;
const uint8_t UserInformation::Reserved2 = 0x00;

struct RoleSelectionSubItems
{
  void Print(std::ostream &os) const
    {
    std::vector<RoleSelectionSub>::const_iterator it = RSSArray.begin();
    for( ; it != RSSArray.end(); ++it )
      {
      it->Print(os);
      }
    }
  const std::ostream &Write(std::ostream &os) const
    {
    std::vector<RoleSelectionSub>::const_iterator it = RSSArray.begin();
    for( ; it != RSSArray.end(); ++it )
      {
      it->Write(os);
      }
    return os;
    }
  void AddTuple(const char *uid, uint8_t scurole, uint8_t scprole)
    {
    RoleSelectionSub rss;
    rss.SetTuple( uid, scurole, scprole );
    RSSArray.push_back( rss );
    }
  bool Empty() const
    {
    return RSSArray.empty();
    }
  size_t Size() const
    {
    size_t s = 0;
    std::vector<RoleSelectionSub>::const_iterator it = RSSArray.begin();
    for( ; it != RSSArray.end(); ++it )
      {
      s += it->Size();
      }
    return s;
    }
  std::vector<RoleSelectionSub> RSSArray;
};

struct SOPClassExtendedNegociationSubItems
{
  void Print(std::ostream &os) const
    {
    std::vector<SOPClassExtendedNegociationSub>::const_iterator it = SOPCENSArray.begin();
    for( ; it != SOPCENSArray.end(); ++it )
      {
      it->Print(os);
      }
    }
  const std::ostream &Write(std::ostream &os) const
    {
    std::vector<SOPClassExtendedNegociationSub>::const_iterator it = SOPCENSArray.begin();
    for( ; it != SOPCENSArray.end(); ++it )
      {
      it->Write(os);
      }
    return os;
    }
  void AddDefault(const char *uid)
    {
    SOPClassExtendedNegociationSub sub;
    sub.SetTuple( uid );
    SOPCENSArray.push_back( sub );
    }
  bool Empty() const
    {
    return SOPCENSArray.empty();
    }
  size_t Size() const
    {
    size_t s = 0;
    std::vector<SOPClassExtendedNegociationSub>::const_iterator it = SOPCENSArray.begin();
    for( ; it != SOPCENSArray.end(); ++it )
      {
      s += it->Size();
      }
    return s;
    }
  std::vector<SOPClassExtendedNegociationSub> SOPCENSArray;
};

UserInformation::UserInformation()
{
  AOWS = NULL;
  RSSI = new RoleSelectionSubItems;
  SOPCENSI = new SOPClassExtendedNegociationSubItems;
#if 0
  RSSI->AddTuple("1.2.840.10008.5.1.4.1.1.2", 1, 1); // DEBUG
  RSSI->AddTuple("1.2.840.10008.5.1.4.1.1.4", 1, 1); // DEBUG
  RSSI->AddTuple("1.2.840.10008.5.1.4.1.1.7", 1, 1); // DEBUG
  SOPCENSI->AddDefault("1.2.840.10008.5.1.4.1.1.2"); // DEBUG
  SOPCENSI->AddDefault("1.2.840.10008.5.1.4.1.1.4"); // DEBUG
  SOPCENSI->AddDefault("1.2.840.10008.5.1.4.1.1.7"); // DEBUG
#endif
  size_t t0 = MLS.Size();
  size_t t1 = ICUID.Size();
  size_t t2 = 0; //AOWS.Size();
  size_t t3 = IVNS.Size();
  ItemLength = (uint16_t)(t0 + t1 + t2 + t3);
#if 0
  if( !RSSI->Empty() ) ItemLength += RSSI->Size();
  if( !SOPCENSI->Empty() ) ItemLength += SOPCENSI->Size();
#endif
  assert( (size_t)ItemLength + 4 == Size() );
}

UserInformation::~UserInformation()
{
  delete AOWS;
  delete SOPCENSI;
  delete RSSI;
}

std::istream &UserInformation::Read(std::istream &is)
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

  uint8_t itemtype2 = 0x0;
  size_t curlen = 0;
#if 0
  RSSI->RSSArray.clear(); // DEBUG
  SOPCENSI->SOPCENSArray.clear(); // DEBUG
#endif
  while( curlen < ItemLength )
    {
    is.read( (char*)&itemtype2, sizeof(ItemType) );
    switch ( itemtype2 )
      {
    case 0x51: // MaximumLengthSub
      MLS.Read( is );
      curlen += MLS.Size();
      break;
    case 0x52: // ImplementationClassUIDSub
      ICUID.Read(is);
      curlen += ICUID.Size();
      break;
    case 0x53: // AsynchronousOperationsWindowSub
      assert( !AOWS );
      AOWS = new AsynchronousOperationsWindowSub;
      AOWS->Read( is );
      curlen += AOWS->Size();
      break;
    case 0x54: // RoleSelectionSub
      assert( RSSI );
        {
        RoleSelectionSub rss;
        rss.Read( is );
        curlen += rss.Size();
        RSSI->RSSArray.push_back( rss );
        }
      break;
    case 0x55: // ImplementationVersionNameSub
      IVNS.Read( is );
      curlen += IVNS.Size();
      break;
    case 0x56: // SOPClassExtendedNegociationSub
      assert( SOPCENSI );
        {
        SOPClassExtendedNegociationSub sopcens;
        sopcens.Read( is );
        curlen += sopcens.Size();
        SOPCENSI->SOPCENSArray.push_back( sopcens );
        }
      break;
    default:
      gdcmErrorMacro( "Unknown ItemType: " << std::hex << (int) itemtype2 );
      curlen = ItemLength; // make sure to exit
      assert(0);
      break;
      }
    }
  assert( curlen == ItemLength );

  assert( (size_t)ItemLength + 4 == Size() );
  return is;
}

const std::ostream &UserInformation::Write(std::ostream &os) const
{
  assert( (size_t)ItemLength + 4 == Size() );
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  uint16_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );

  MLS.Write(os);
  ICUID.Write(os);
  if( AOWS )
    {
    AOWS->Write(os);
    }
  if( !RSSI->Empty() )
    {
    RSSI->Write(os);
    }
  IVNS.Write(os);
  if( !SOPCENSI->Empty() )
    {
    SOPCENSI->Write(os);
    }

  assert( (size_t)ItemLength + 4 == Size() );

  return os;
}

size_t UserInformation::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength); // len of
  ret += MLS.Size();
  ret += ICUID.Size();
  if( AOWS )
    ret += AOWS->Size();
  if( !RSSI->Empty() )
    ret += RSSI->Size();
  ret += IVNS.Size();
  if( !SOPCENSI->Empty() )
    ret += SOPCENSI->Size();

  return ret;
}

void UserInformation::Print(std::ostream &os) const
{
  os << "MaximumLengthSub: ";
  MLS.Print( os );
  os << "ImplementationClassUIDSub: ";
  ICUID.Print( os );
  if( AOWS )
    {
    os << "AsynchronousOperationsWindowSub: ";
    AOWS->Print( os );
    }
  if( !RSSI->Empty() )
    {
    os << "RoleSelectionSub: ";
    RSSI->Print( os );
    }
  os << "ImplementationVersionNameSub: ";
  IVNS.Print( os );
  if( !SOPCENSI->Empty() )
    {
    os << "SOPClassExtendedNegociationSub: ";
    SOPCENSI->Print( os );
    }
  os << std::endl;
}

UserInformation &UserInformation::operator=(const UserInformation& ui)
{
  ItemLength = ui.ItemLength;
  MLS = ui.MLS;
  ICUID = ui.ICUID;
  if( ui.AOWS )
    {
    delete AOWS;
    AOWS = new AsynchronousOperationsWindowSub;
    *AOWS = *ui.AOWS;
    }
  *RSSI = *ui.RSSI;
  *SOPCENSI = *ui.SOPCENSI;
  IVNS = ui.IVNS;

  assert( (size_t)ItemLength + 4 == Size() );

  return *this;
}

void UserInformation::AddRoleSelectionSub( RoleSelectionSub const & rss )
{
  RSSI->RSSArray.push_back( rss );
  ItemLength = (uint16_t)(Size() - 4);
  assert( (size_t)ItemLength + 4 == Size() );
}

void UserInformation::AddSOPClassExtendedNegociationSub( SOPClassExtendedNegociationSub const & sopcens )
{
  SOPCENSI->SOPCENSArray.push_back( sopcens );
  ItemLength = (uint16_t)(Size() - 4);
  assert( (size_t)ItemLength + 4 == Size() );
}

} // end namespace network
} // end namespace gdcm
