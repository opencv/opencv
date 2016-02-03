/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAAssociateRQPDU.h"
#include "gdcmSwapper.h"

#include "gdcmAAssociateACPDU.h"
#include <string>
#include <locale>

namespace gdcm
{
/*
9.3.2 A-ASSOCIATE-RQ PDU STRUCTURE
An A-ASSOCIATE-RQ PDU shall be made of a sequence of mandatory fields followed by a variable
length field. Table 9-11 shows the sequence of the mandatory fields.
The variable field shall consist of one Application Context Item, one or more Presentation Context Items,
and one User Information Item. Sub-Items shall exist for the Presentation Context and User Information
Items.
*/
namespace network
{
const uint8_t AAssociateRQPDU::ItemType = 0x1; // PDUType ?
const uint8_t AAssociateRQPDU::Reserved2 = 0x0;
const uint16_t AAssociateRQPDU::ProtocolVersion = 0x1; // big - endian ?
const uint16_t AAssociateRQPDU::Reserved9_10 = 0x0;
//const uint8_t AAssociateRQPDU::Reserved43_74[32] = {};

AAssociateRQPDU::AAssociateRQPDU()
{
  memset(CalledAETitle, ' ', sizeof(CalledAETitle));
  //const char called[] = "ANY-SCP";
  //strncpy(CalledAETitle, called, strlen(called) );
  memset(CallingAETitle, ' ', sizeof(CallingAETitle));
  //const char calling[] = "ECHOSCU";
  //strncpy(CallingAETitle, calling, strlen(calling) );
  memset(Reserved43_74, 0x0, sizeof(Reserved43_74));

  //SetCallingAETitle( "MOVESCU" );

  ItemLength = (uint32_t)Size() - 6;
  assert( (ItemLength + 4 + 1 + 1) == Size() );
}

std::istream &AAssociateRQPDU::Read(std::istream &is)
{
  //uint8_t itemtype = 0;
  //is.read( (char*)&itemtype, sizeof(ItemType) );
  //assert( itemtype == ItemType );
  uint8_t reserved2;
  is >> reserved2;
  uint32_t itemlength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;
  uint16_t protocolversion;
  is.read( (char*)&protocolversion, sizeof(ProtocolVersion) );
  SwapperDoOp::SwapArray(&protocolversion,1);
  if( protocolversion != ProtocolVersion )
    {
    gdcmWarningMacro( "ProtocolVersion is: " << protocolversion );
    }
  uint16_t reserved9_10;
  is.read( (char*)&reserved9_10, sizeof(Reserved9_10) );
  SwapperDoOp::SwapArray(&reserved9_10,1);
  //char calledaetitle[16];
  is.read( (char*)&CalledAETitle, sizeof(CalledAETitle) ); // called
  //char callingaetitle[16];
  is.read( (char*)&CallingAETitle, sizeof(CallingAETitle) ); // calling
  uint8_t reserved43_74[32] = {  };
  is.read( (char*)&reserved43_74, sizeof(Reserved43_74) ); // 0 (32 times)
  memcpy( Reserved43_74, reserved43_74, sizeof(Reserved43_74) );

  uint8_t itemtype2 = 0x0;
  size_t curlen = 0;
  while( curlen + 68 < ItemLength )
    {
    is.read( (char*)&itemtype2, sizeof(ItemType) );
    switch ( itemtype2 )
      {
    case 0x10: // ApplicationContext ItemType
      AppContext.Read( is );
      curlen += AppContext.Size();
      break;
    case 0x20: // PresentationContextRQ ItemType
        {
        PresentationContextRQ pc;
        pc.Read( is );
        PresContext.push_back( pc );
        curlen += pc.Size();
        }
      break;
    case 0x50: // UserInformation ItemType
      UserInfo.Read( is );
      curlen += UserInfo.Size();
      break;
    default:
      gdcmErrorMacro( "Unknown ItemType: " << std::hex << (int) itemtype2 );
      curlen = ItemLength; // make sure to exit
      break;
      }
    // WARNING: I cannot simply call Size() since UserInfo is initialized with GDCM
    // own parameter, this will bias the computation. Instead compute relative
    // length of remaining bytes to read.
    //curlen = Size();
    }
  assert( curlen + 68 == ItemLength );

  assert( ItemLength + 4 + 1 + 1 == Size() );

  return is;
}

const std::ostream &AAssociateRQPDU::Write(std::ostream &os) const
{
  assert( ItemLength + 4 + 1 + 1 == Size() );
#if 0
  // Need to check all context Id are ordered ? and odd number ?
  std::vector<PresentationContextRQ>::const_iterator it = PresContext.begin();
  for( ; it != PresContext.end(); ++it)
    {
    it->Write(os);
    }
#endif
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  uint32_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );
  uint16_t protocolversion = ProtocolVersion;
  SwapperDoOp::SwapArray(&protocolversion,1);
  os.write( (char*)&protocolversion, sizeof(ProtocolVersion) );
  os.write( (char*)&Reserved9_10, sizeof(Reserved9_10) );
  assert( AAssociateRQPDU::IsAETitleValid(CalledAETitle) );
  os.write( CalledAETitle, 16 );
  assert( AAssociateRQPDU::IsAETitleValid(CallingAETitle) );
  os.write( CallingAETitle, 16 );
  os.write( (char*)&Reserved43_74, sizeof(Reserved43_74) );
  AppContext.Write(os);
  std::vector<PresentationContextRQ>::const_iterator it = PresContext.begin();
  for( ; it != PresContext.end(); ++it)
    {
    it->Write(os);
    }
  UserInfo.Write(os);

  return os;
}

size_t AAssociateRQPDU::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += sizeof(ProtocolVersion);
  ret += sizeof(Reserved9_10);
  ret += sizeof(CalledAETitle);
  ret += sizeof(CallingAETitle);
  ret += sizeof(Reserved43_74);
  ret += AppContext.Size();
  std::vector<PresentationContextRQ>::const_iterator it = PresContext.begin();
  for( ; it != PresContext.end(); ++it)
    {
    ret += it->Size();
    }
  ret += UserInfo.Size();

  return ret;
}

bool AAssociateRQPDU::IsAETitleValid(const char title[16])
{
  if(!title) return false;
#if 0
  std::string s ( title, 16 );
  // check no \0 :
  //size_t len = strlen( s.c_str() );

  // FIXME:
//  if( len != 16 ) return false;
  std::locale loc;
  std::string str = s;
  for (size_t i=0; i < str.size(); ++i)
    {
    str[i] = std::toupper(str[i],loc);
    }
  if( str != s ) return false;
#else
  const size_t reallen = strlen( title );
  std::string s ( title, std::min(reallen, (size_t)16) );
  // check no \0 :
  size_t len = strlen( s.c_str() );

  char OnlySpaces[16];
  memset(OnlySpaces, ' ', sizeof(OnlySpaces));
  if( strncmp( title, OnlySpaces, len ) == 0 )
    {
    return false;
    }
#endif
  return true;
}

void AAssociateRQPDU::AddPresentationContext( PresentationContextRQ const &pc )
{
  PresContext.push_back( pc );
  ItemLength = (uint32_t)Size() - 6;
  assert( (ItemLength + 4 + 1 + 1) == Size() );
}

void AAssociateRQPDU::SetCalledAETitle(const char calledaetitle[16])
{
  assert( AAssociateRQPDU::IsAETitleValid(calledaetitle) );
  size_t len = strlen( calledaetitle );
  if( len <= 16 )
    {
    memset(CalledAETitle, ' ', sizeof(CalledAETitle));
    strncpy(CalledAETitle, calledaetitle, len );
    }
  // FIXME Need to check upper case
  // FIXME cannot set to only whitespaces
}

void AAssociateRQPDU::SetCallingAETitle(const char callingaetitle[16])
{
  assert( AAssociateRQPDU::IsAETitleValid(callingaetitle) );
  size_t len = strlen( callingaetitle );
  if( len <= 16 )
    {
    memset(CallingAETitle, ' ', sizeof(CallingAETitle));
    strncpy(CallingAETitle, callingaetitle, len );
    }
  // FIXME Need to check upper case
  // FIXME cannot set to only whitespaces
}

std::string AAssociateRQPDU::GetReserved43_74() const
{
  return std::string(Reserved43_74,32);
}

void AAssociateRQPDU::Print(std::ostream &os) const
{
  //static const uint8_t ItemType; // PDUType ?
  //static const uint8_t Reserved2;
  //uint32_t ItemLength; // PDU Length
  //static const uint16_t ProtocolVersion;
  //static const uint16_t Reserved9_10;
  os << "CalledAETitle: ";
  os << GetCalledAETitle() << std::endl;
  os << "CallingAETitle: ";
  os << GetCallingAETitle() << std::endl;
  //static const uint8_t Reserved43_74[32]; // { 0 }
  os << "ApplicationContext: ";
  AppContext.Print( os );
  os << std::endl;
  //std::vector<PresentationContextRQ> PresContext;
  os << "PresentationContext(s): ";
  std::vector<PresentationContextRQ>::const_iterator it = PresContext.begin();
  for( ; it != PresContext.end(); ++it)
    {
    it->Print( os << std::endl );
    }
  os << "UserInformation: ";
  UserInfo.Print( os );
  os << std::endl;
}

const PresentationContextRQ *AAssociateRQPDU::GetPresentationContextByID(uint8_t id) const
{
  std::vector<PresentationContextRQ>::const_iterator it = PresContext.begin();
  for( ; it != PresContext.end(); ++it)
    {
    if( it->GetPresentationContextID() == id )
      {
      return &*it;
      }
    }
  return NULL;
}

const PresentationContextRQ *AAssociateRQPDU::GetPresentationContextByAbstractSyntax(AbstractSyntax const & as ) const
{
  std::vector<PresentationContextRQ>::const_iterator it = PresContext.begin();
  for( ; it != PresContext.end(); ++it)
    {
    if( it->GetAbstractSyntax() == as )
      {
      return &*it;
      }
    }
  return NULL;
}

void AAssociateRQPDU::SetUserInformation( UserInformation const & ui )
{
  UserInfo = ui;
  ItemLength = (uint32_t)Size() - 6;
  assert( (ItemLength + 4 + 1 + 1) == Size() );
}

} // end namespace network
} // end namespace gdcm
