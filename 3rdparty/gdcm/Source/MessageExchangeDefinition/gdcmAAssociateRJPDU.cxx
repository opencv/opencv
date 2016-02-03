/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAAssociateRJPDU.h"
#include "gdcmSwapper.h"

namespace gdcm
{
namespace network
{
const uint8_t AAssociateRJPDU::ItemType = 0x03; // PDUType ?
const uint8_t AAssociateRJPDU::Reserved2 = 0x00;
const uint8_t AAssociateRJPDU::Reserved8 = 0x00;

AAssociateRJPDU::AAssociateRJPDU()
{
  ItemLength = 0;
  Result = 0;
  Source = 0;
  Reason = 0; // diag ?
}

std::istream &AAssociateRJPDU::Read(std::istream &is)
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
  uint8_t reserved8;
  is >> reserved8;
  uint8_t result;
  is >> result;
  Result = result;
  uint8_t source;
  is >> source;
  Source = source;
  uint8_t reason;
  is >> reason;
  Reason = reason;

  //assert( ItemLength + 4 + 1 + 1 == Size() );

  return is;
}

const std::ostream &AAssociateRJPDU::Write(std::ostream &os) const
{
  return os;
}

namespace {
static const char *PrintResultAsString( uint8_t result )
{
  switch( result )
    {
  case 0x1:
    return "rejected-permanent";
  case 0x2:
    return "rejected-transient";
    }
  assert( 0 );
  return NULL;
}

static const char *PrintSourceAsString( uint8_t source )
{
  switch( source )
    {
  case 0x0:
    return "DICOM UL service-user";
  case 0x1:
    return "DICOM UL service-provider (ACSE related function)";
  case 0x2:
    return "DICOM UL service-provider (Presentation related function)";
    }
  assert( 0 );
  return NULL;
}

static const char *PrintReasonAsString( uint8_t source, uint8_t reason )
{
  switch( source )
    {
  case 0x1:
    switch( reason )
      {
    case 0x1:
      return "1 - no-reason-given";
    case 0x2:
      return "2 - application-context-name-not-supported";
    case 0x3:
      return "3 - calling-AE-title-not-recognized";
    case 0x4:
    case 0x5:
    case 0x6:
      return "4-6 - reserved";
    case 0x7:
      return "7 - called-AE-title-not-recognized";
    case 0x8:
    case 0x9:
    case 0xa:
      return "8-10 - reserved";
      }
  case 0x2:
    switch( reason )
      {
    case 0x1:
      return "no-reason-given";
    case 0x2:
      return "protocol-version-not-supported";
      }
  case 0x3:
    switch( reason )
      {
    case 0x0:
      return "0 - reserved";
    case 0x1:
      return "1 - temporary-congestion";
    case 0x2:
      return "2 - local-limit-exceeded";
    case 0x3:
    case 0x4:
    case 0x5:
    case 0x6:
    case 0x7:
      return "3-7 - reserved";
      }
    }
  assert( 0 );
  return NULL;
}

}


void AAssociateRJPDU::Print(std::ostream &os) const
{
  os << "PDULength: " << ItemLength << std::endl;
  os << "Result: " << PrintResultAsString( Result ) << std::endl;
  os << "Source: " << PrintSourceAsString( Source ) << std::endl;
  os << "Reason: " << PrintReasonAsString( Source, Reason ) << std::endl;
}


size_t AAssociateRJPDU::Size() const{
  return sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint32_t)+
    sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t);
}

} // end namespace network
} // end namespace gdcm
