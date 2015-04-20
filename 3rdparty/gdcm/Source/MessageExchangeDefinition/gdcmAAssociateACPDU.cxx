/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAAssociateACPDU.h"
#include "gdcmSwapper.h"
#include "gdcmAAssociateRQPDU.h"

namespace gdcm
{
namespace network
{
const uint8_t AAssociateACPDU::ItemType = 0x02; // PDUType ?
const uint8_t AAssociateACPDU::Reserved2 = 0x00;
const uint16_t AAssociateACPDU::ProtocolVersion = 0x01; // big endian
const uint16_t AAssociateACPDU::Reserved9_10 = 0x0000;
//const uint8_t AAssociateACPDU::Reserved11_26[16] = {  };
//const uint8_t AAssociateACPDU::Reserved27_42[16] = {  };
//const uint8_t AAssociateACPDU::Reserved43_74[32] = {  };

AAssociateACPDU::AAssociateACPDU()
{
  PDULength = 0; // len of
  memset(Reserved11_26, ' ', sizeof(Reserved11_26));
  memset(Reserved27_42, ' ', sizeof(Reserved27_42));
  memset(Reserved43_74, ' ', sizeof(Reserved43_74));

  PDULength = (uint32_t)(Size() - 6);
}

void AAssociateACPDU::SetCalledAETitle(const char calledaetitle[16])
{
  //size_t len = strlen( calledaetitle );
  //assert( len <= 16 ); // since forwared from AA-RQ no reason to be invalid
  strncpy(Reserved11_26, calledaetitle, 16 );
}

void AAssociateACPDU::SetCallingAETitle(const char callingaetitle[16])
{
  //size_t len = strlen( callingaetitle );
  //assert( len <= 16 ); // since forwared from AA-RQ no reason to be invalid
  strncpy(Reserved27_42, callingaetitle, 16 );
}

std::istream &AAssociateACPDU::Read(std::istream &is)
{
  //uint8_t itemtype = 0;
  //is.read( (char*)&itemtype, sizeof(ItemType) );
  //assert( itemtype == ItemType );
  assert( is.good() );
  uint8_t reserved2;
  is.read( (char*)&reserved2, sizeof(Reserved2) );
  uint32_t pdulength = 0;
  is.read( (char*)&pdulength, sizeof(PDULength) );
  SwapperDoOp::SwapArray(&pdulength,1);
  PDULength = pdulength;
  uint16_t protocolversion;
  is.read( (char*)&protocolversion, sizeof(ProtocolVersion) );
  SwapperDoOp::SwapArray(&protocolversion,1);
  if( protocolversion != ProtocolVersion )
    {
    gdcmErrorMacro( "Improper Protocol Version: " << protocolversion );
    }
  uint16_t reserved9_10;
  is.read( (char*)&reserved9_10, sizeof(Reserved9_10) );
  SwapperDoOp::SwapArray(&reserved9_10,1);
  char reserved11_26[16];
  memset( reserved11_26, 0, sizeof(reserved11_26));
  is.read( (char*)&reserved11_26, sizeof(Reserved11_26) ); // called
  memcpy( Reserved11_26, reserved11_26, sizeof(Reserved11_26) );
  char reserved27_42[16];
  memset( reserved27_42, 0, sizeof(reserved27_42));
  is.read( (char*)&reserved27_42, sizeof(Reserved27_42) ); // calling
  memcpy( Reserved27_42, reserved27_42, sizeof(Reserved27_42) );
  uint8_t reserved43_74[32];
  memset( reserved43_74, 0, sizeof(reserved43_74));
  is.read( (char*)&reserved43_74, sizeof(Reserved43_74) ); // 0 (32 times)
  memcpy( Reserved43_74, reserved43_74, sizeof(Reserved43_74) );

  uint8_t itemtype2 = 0x0;
  size_t curlen = 0;
  while( curlen + 68 < PDULength )
    {
    is.read( (char*)&itemtype2, sizeof(ItemType) );
    switch ( itemtype2 )
      {
    case 0x10: // ApplicationContext ItemType
      AppContext.Read( is );
      curlen += AppContext.Size();
      break;
    case 0x21: // PresentationContextAC ItemType
        {
        PresentationContextAC pcac;
        pcac.Read( is );
        PresContextAC.push_back( pcac );
        curlen += pcac.Size();
        }
      break;
    case 0x50: // UserInformation ItemType
      UserInfo.Read( is );
      curlen += UserInfo.Size();
      break;
    default:
      gdcmErrorMacro( "Unknown ItemType: " << std::hex << (int) itemtype2 );
      curlen = PDULength; // make sure to exit
      break;
      }
    // WARNING: I cannot simply call Size() since UserInfo is initialized with GDCM
    // own parameter, this will bias the computation. Instead compute relative
    // length of remaining bytes to read.
    //curlen = Size();
    }
  assert( curlen + 68 == PDULength );
  assert( PDULength + 4 + 1 + 1 == Size() );

  return is;
}

const std::ostream &AAssociateACPDU::Write(std::ostream &os) const
{
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  //os.write( (char*)&PDULength, sizeof(PDULength) );
  uint32_t copy = PDULength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(PDULength) );
  uint16_t protocolversion = ProtocolVersion;
  SwapperDoOp::SwapArray(&protocolversion,1);
  os.write( (char*)&protocolversion, sizeof(ProtocolVersion) );
  os.write( (char*)&Reserved9_10, sizeof(Reserved9_10) );
  os.write( (char*)&Reserved11_26, sizeof(Reserved11_26) );
  //const char calling[] = "ANY-SCP         ";
  //os.write( calling, 16 );

  os.write( (char*)&Reserved27_42, sizeof(Reserved27_42) );
  //const char called[] = "STORESCU        ";
  //const char called[] = "ECHOSCU        ";
  //os.write( called, 16 );
  os.write( (char*)&Reserved43_74, sizeof(Reserved43_74) );
  AppContext.Write( os );
  gdcmAssertAlwaysMacro( PresContextAC.size() );
  std::vector<PresentationContextAC>::const_iterator it = PresContextAC.begin();
  for( ; it != PresContextAC.end(); ++it )
    {
    it->Write( os );
    }
  UserInfo.Write( os );

  assert( PDULength + 4 + 1 + 1 == Size() );

  return os;
}

size_t AAssociateACPDU::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(PDULength);
  ret += sizeof(ProtocolVersion);
  ret += sizeof(Reserved9_10);
  ret += sizeof(Reserved11_26);
  ret += sizeof(Reserved27_42);
  ret += sizeof(Reserved43_74);
  ret += AppContext.Size();
  std::vector<PresentationContextAC>::const_iterator it = PresContextAC.begin();
  for( ; it != PresContextAC.end(); ++it )
    {
    ret += it->Size();
    }
  ret += UserInfo.Size();
  return ret;
}

void AAssociateACPDU::AddPresentationContextAC( PresentationContextAC const &pcac )
{
  PresContextAC.push_back( pcac );
  PDULength = (uint32_t)(Size() - 6);
  assert( PDULength + 4 + 1 + 1 == Size() );
}

void AAssociateACPDU::Print(std::ostream &os) const
{
  os << "ProtocolVersion: " << std::hex << ProtocolVersion << std::dec << std::endl;
  os << "Reserved9_10: " << std::hex << Reserved9_10 << std::dec << std::endl;
  os << "Reserved11_26: [" << std::string(Reserved11_26,sizeof(Reserved11_26)) << "]" << std::endl;
  os << "Reserved27_42: [" << std::string(Reserved27_42,sizeof(Reserved27_42)) << "]" << std::endl;
  /*os << "Reserved43_74: [" << std::string(Reserved43_74,sizeof(Reserved43_74)) << "]" << std::endl;*/
  os << "Application Context Name: ";
  AppContext.Print( os );
  os << "List of PresentationContextAC: " << std::endl;
  std::vector<PresentationContextAC>::const_iterator it = PresContextAC.begin();
  for( ; it != PresContextAC.end(); ++it )
    {
    it->Print(os);
    }
  os << "User Information: ";
  UserInfo.Print( os );
}

void AAssociateACPDU::InitFromRQ( AAssociateRQPDU const & rqpdu )
{
  // Table 9-17 ASSOCIATE-AC PDU fields
  // This reserved field shall be sent with a value identical to the value
  // received in the same field of the A-ASSOCIATE-RQ PDU
  const std::string called = rqpdu.GetCalledAETitle();
  SetCalledAETitle( rqpdu.GetCalledAETitle().c_str() );
  const std::string calling = rqpdu.GetCallingAETitle();
  SetCallingAETitle( rqpdu.GetCallingAETitle().c_str() );
  const std::string reserved = rqpdu.GetReserved43_74();
  memcpy( Reserved43_74, reserved.c_str(), sizeof(Reserved43_74) );

  assert( ProtocolVersion == 0x01 );
  assert( Reserved9_10 == 0x0 );
  assert( memcmp( Reserved11_26, called.c_str(), sizeof( Reserved11_26) ) == 0 );
  assert( memcmp( Reserved27_42, calling.c_str(), sizeof(Reserved27_42) ) == 0 );
  assert( memcmp( Reserved43_74, reserved.c_str(), sizeof(Reserved43_74) ) == 0 );
}


void AAssociateACPDU::InitSimple( AAssociateRQPDU const & rqpdu )
{
  TransferSyntaxSub ts1;
  ts1.SetNameFromUID( UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM );

  assert( rqpdu.GetNumberOfPresentationContext() );
  for( unsigned int index = 0; index < rqpdu.GetNumberOfPresentationContext(); index++ )
    {
    // FIXME / HARDCODED We only ever accept Little Endian
    // FIXME we should check :
    // rqpdu.GetAbstractSyntax() contains LittleEndian
    PresentationContextAC pcac1;
    PresentationContextRQ const &pc = rqpdu.GetPresentationContext(index);
    uint8_t id = pc.GetPresentationContextID();

    pcac1.SetPresentationContextID( id ); // DCMTK MR
    pcac1.SetTransferSyntax( ts1 );
    AddPresentationContextAC( pcac1 );
    }

}

} // end namespace network
} // end namespace gdcm
