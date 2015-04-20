/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAAbortPDU.h"
#include "gdcmSwapper.h"

namespace gdcm
{
namespace network
{
const uint8_t AAbortPDU::ItemType = 0x7; // PDUType ?
const uint8_t AAbortPDU::Reserved2 = 0x0;
const uint8_t AAbortPDU::Reserved7 = 0x0;
const uint8_t AAbortPDU::Reserved8 = 0x0;

/*
This Source field shall contain an integer value encoded as an
unsigned binary number. One of the following values shall be used:
0 - DICOM UL service-user (initiated abort)
1 - reserved
2 - DICOM UL service-provider (initiated abort)
*/
/*
This field shall contain an integer value encoded as an unsigned
binary number. If the Source field has the value (2) “DICOM UL
service-provider,” it shall take one of the following:
0 - reason-not-specified
1 - unrecognized-PDU
2 - unexpected-PDU
3 - reserved
4 - unrecognized-PDU parameter
5 - unexpected-PDU parameter
6 - invalid-PDU-parameter value
If the Source field has the value (0) “DICOM UL service-user,” this
reason field shall not be significant. It shall be sent with a value 00H
but not tested to this value when received.
Note: The reserved fields are used to preserve symmetry with OSI
ACSE/Presentation Services and Protocol.
*/
AAbortPDU::AAbortPDU()
{
  ItemLength = 0;
  Source = 0;
  Reason = 0;

  ItemLength = (uint32_t)Size() - 6;
  assert( (ItemLength + 4 + 1 + 1) == Size() );
}

std::istream &AAbortPDU::Read(std::istream &is)
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
  uint8_t reserved7 = 0;
  is.read( (char*)&reserved7, sizeof(Reserved7) );
  uint8_t reserved8 = 0;
  is.read( (char*)&reserved8, sizeof(Reserved8) );
  uint8_t source = 0;
  is.read( (char*)&source, sizeof(Source) );
  Source = source;
  uint8_t reason = 0;
  is.read( (char*)&reason, sizeof(Reason) );
  Reason = reason;

  assert( (ItemLength + 4 + 1 + 1) == Size() );
  return is;
}

const std::ostream &AAbortPDU::Write(std::ostream &os) const
{
  return os;
}

size_t AAbortPDU::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += sizeof(Reserved7);
  ret += sizeof(Reserved8);
  ret += sizeof(Source);
  ret += sizeof(Reason);

  return ret;
}

namespace {
static const char *PrintSourceAsString( uint8_t source )
{
  // See PS 3.8-2011 Table 9-26 A-ABORT PDU FIELDS
  switch( source )
    {
  case 0x0:
    return "DICOM UL service-user (initiated abort)";
  case 0x1:
    return "reserved";
  case 0x2:
    return "DICOM UL service-provider (initiated abort)";
    }
  // Conquest DICOM 1.14.17c, return '3' as source value:
  return "BOGUS SCP IMPLEMENTATION, REPORT UPSTREAM";
}

static const char *PrintReasonAsString( uint8_t reason )
{
  switch( reason )
    {
  case 0x0:
    return "reason-not-specified";
  case 0x1:
    return "unrecognized-PDU";
  case 0x2:
    return "unexpected-PDU";
  case 0x3:
    return "reserved";
  case 0x4:
    return "unrecognized-PDU parameter";
  case 0x5:
    return "unexpected-PDU parameter";
  case 0x6:
    return "invalid-PDU-parameter value";
    }
  assert( 0 );
  return NULL;
}
}

void AAbortPDU::Print(std::ostream &os) const
{
  os << "PDULength: " << ItemLength << std::endl;
  os << "Source: " << PrintSourceAsString( Source ) << std::endl;
  os << "Reason: " << PrintReasonAsString( Reason ) << std::endl;
}

void AAbortPDU::SetSource(const uint8_t s)
{
  Source = s;
}

void AAbortPDU::SetReason(const uint8_t r)
{
  Reason = r;
}

} // end namespace network
} // end namespace gdcm
