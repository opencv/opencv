/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPixelFormat.h"
#include "gdcmTrace.h"
#include "gdcmTransferSyntax.h"

#include <stdlib.h>

namespace gdcm
{
static const char *ScalarTypeStrings[] = {
  "UINT8",
  "INT8",
  "UINT12",
  "INT12",
  "UINT16",
  "INT16",
  "UINT32",
  "INT32",
  "UINT64",
  "INT64",
  "FLOAT16",
  "FLOAT32",
  "FLOAT64",
  "SINGLEBIT",
  "UNKNOWN",
  NULL,
};

PixelFormat::PixelFormat(ScalarType st)
{
  SamplesPerPixel = 1;
  SetScalarType( st );
}

unsigned short PixelFormat::GetSamplesPerPixel() const
{
  // \postcondition
  assert( SamplesPerPixel == 1 || SamplesPerPixel == 3 || SamplesPerPixel == 4 );
  return SamplesPerPixel;
}

void PixelFormat::SetScalarType(ScalarType st)
{
  SamplesPerPixel = 1;
  switch(st)
    {
  case PixelFormat::UINT8:
    BitsAllocated = 8;
    PixelRepresentation = 0;
    break;
  case PixelFormat::INT8:
    BitsAllocated = 8;
    PixelRepresentation = 1;
    break;
  case PixelFormat::UINT12:
    BitsAllocated = 12;
    PixelRepresentation = 0;
    break;
  case PixelFormat::INT12:
    BitsAllocated = 12;
    PixelRepresentation = 1;
    break;
  case PixelFormat::UINT16:
    BitsAllocated = 16;
    PixelRepresentation = 0;
    break;
  case PixelFormat::INT16:
    BitsAllocated = 16;
    PixelRepresentation = 1;
    break;
  case PixelFormat::UINT32:
    BitsAllocated = 32;
    PixelRepresentation = 0;
    break;
  case PixelFormat::INT32:
    BitsAllocated = 32;
    PixelRepresentation = 1;
    break;
  case PixelFormat::UINT64:
    BitsAllocated = 64;
    PixelRepresentation = 0;
    break;
  case PixelFormat::INT64:
    BitsAllocated = 64;
    PixelRepresentation = 1;
    break;
  case PixelFormat::FLOAT16:
    BitsAllocated = 16;
    // secret code:
    PixelRepresentation = 2;
    break;
  case PixelFormat::FLOAT32:
    BitsAllocated = 32;
    // secret code:
    PixelRepresentation = 3;
    break;
  case PixelFormat::FLOAT64:
    BitsAllocated = 64;
    // secret code:
    PixelRepresentation = 4;
    break;
  case PixelFormat::SINGLEBIT:
    BitsAllocated = 1;
    PixelRepresentation = 0;
    break;
  case PixelFormat::UNKNOWN:
    BitsAllocated = 0;
    PixelRepresentation = 0;
    break;
  default:
    assert(0);
    break;
    }
  BitsStored = BitsAllocated;
  HighBit = (uint16_t)(BitsStored - 1);
}

PixelFormat::ScalarType PixelFormat::GetScalarType() const
{
  ScalarType type = PixelFormat::UNKNOWN;
  switch( BitsAllocated )
    {
  case 0:
    type = PixelFormat::UNKNOWN;
    break;
  case 1:
    type = PixelFormat::SINGLEBIT;
    break;
  case 8:
    type = PixelFormat::UINT8;
    break;
  case 12:
    type = PixelFormat::UINT12;
    break;
  case 16:
    type = PixelFormat::UINT16;
    break;
  case 32:
    type = PixelFormat::UINT32;
    break;
  case 64:
    type = PixelFormat::UINT64;
    break;
  case 24:
    gdcmDebugMacro( "This is illegal in DICOM, assuming a RGB image" );
    type = PixelFormat::UINT8;
    break;
  default:
    gdcmErrorMacro( "I have never seen this before BitsAllocated "
      << BitsAllocated );
    type = PixelFormat::UNKNOWN;
    }
  if( type != PixelFormat::UNKNOWN )
    {
    if( PixelRepresentation == 0 )
      {
      // all set !
      }
    else if( PixelRepresentation == 1 )
      {
      assert( type <= INT64 );
      // That's why you need to order properly type in ScalarType
      type = ScalarType(int(type)+1);
      }
    else if( PixelRepresentation == 2 )
      {
      assert( BitsAllocated == 16 );
      return FLOAT16;
      }
    else if( PixelRepresentation == 3 )
      {
      assert( BitsAllocated == 32 );
      return FLOAT32;
      }
    else if( PixelRepresentation == 4 )
      {
      assert( BitsAllocated == 64 );
      return FLOAT64;
      }
    else
      {
      assert(0);
      }
    }
  return type;
}

const char *PixelFormat::GetScalarTypeAsString() const
{
  return ScalarTypeStrings[GetScalarType()];
}

uint8_t PixelFormat::GetPixelSize() const
{
  uint8_t pixelsize = (uint8_t)(BitsAllocated / 8);
  if( BitsAllocated == 12 )
    {
    pixelsize = 2; // fake a short value
    }
  else
    {
    assert( !(BitsAllocated % 8) );
    }
  pixelsize *= SamplesPerPixel;

  return pixelsize;
}

int64_t PixelFormat::GetMin() const
{
  assert( BitsAllocated ); // cannot be unknown
  if( BitsStored <= 32 )
    {
    if( PixelRepresentation == 1 )
      {
      return (int64_t)(~(((1ull << BitsStored) - 1) >> 1));
      }
    else if( PixelRepresentation == 0 )
      {
      return 0;
      }
    }
  // else
  throw "PixelFormat bad representation";
  return 0;
}

int64_t PixelFormat::GetMax() const
{
  assert( BitsAllocated ); // cannot be unknown
  if( BitsStored <= 32 )
    {
    if( PixelRepresentation == 1 )
      {
      return (int64_t)(((1ull << BitsStored) - 1) >> 1);
      }
    else if( PixelRepresentation == 0 )
      {
      return (int64_t)((1ull << BitsStored) - 1);
      }
    }
  // else
  throw "PixelFormat bad representation";
  return 0;
}

bool PixelFormat::IsValid() const
{
  if( PixelRepresentation != 0 && PixelRepresentation != 1 )
    {
    return false;
    }
  if( BitsAllocated < BitsStored ) return false;
  if( BitsAllocated < HighBit ) return false;
  if( BitsStored > 32 ) return false;
  return true;
}

bool PixelFormat::Validate()
{
  if( !IsValid() ) return false;
  //assert( BitsStored    >= HighBit ); // DigitexAlpha_no_7FE0.dcm
  assert( PixelRepresentation == 0 || PixelRepresentation == 1 );
  assert( SamplesPerPixel == 1 || SamplesPerPixel == 3 || SamplesPerPixel == 4 );
  if ( BitsStored == 0 )
    {
    gdcmDebugMacro( "Bits Stored is 0. Setting is to max value" );
    BitsStored = BitsAllocated;
    }
  if ( BitsAllocated == 24 )
    {
    gdcmDebugMacro( "ACR-NEMA way of storing RGB data. Updating" );
    if( BitsStored == 24 && HighBit == 23 && SamplesPerPixel == 1 )
      {
      BitsAllocated = 8;
      BitsStored = 8;
      HighBit = 7;
      SamplesPerPixel = 3;
      return true;
      }
    // all other case, simply give up
    return false;
    }
  return true;
}

void PixelFormat::Print(std::ostream &os) const
{
  os << "SamplesPerPixel    :" << SamplesPerPixel     << "\n";
  os << "BitsAllocated      :" << BitsAllocated       << "\n";
  os << "BitsStored         :" << BitsStored          << "\n";
  os << "HighBit            :" << HighBit             << "\n";
  os << "PixelRepresentation:" << PixelRepresentation << "\n";
  os << "ScalarType found   :" << GetScalarTypeAsString() << "\n";
}

bool PixelFormat::IsCompatible(const TransferSyntax & ts ) const
{
  if( ts == TransferSyntax::JPEGBaselineProcess1 && BitsAllocated != 8 ) return false;
  // FIXME are we missing any ?
  return true;
}

} // end namespace gdcm
