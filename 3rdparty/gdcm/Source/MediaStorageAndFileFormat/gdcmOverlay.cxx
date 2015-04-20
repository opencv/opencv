/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmOverlay.h"
#include "gdcmTag.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmAttribute.h"

#include <vector>

namespace gdcm
{

class OverlayInternal
{
public:
  OverlayInternal():
  InPixelData(false),
  Group(0), // invalid default
  Rows(0),
  Columns(0),
  NumberOfFrames(0),
  Description(),
  Type(),
  //Origin[2],
  FrameOrigin(0),
  BitsAllocated(0),
  BitPosition(0),
  Data() { Origin[0] = Origin[1] = 0; }
  /*
  (6000,0010) US 484                                      #   2, 1 OverlayRows
  (6000,0011) US 484                                      #   2, 1 OverlayColumns
  (6000,0015) IS [1]                                      #   2, 1 NumberOfFramesInOverlay
  (6000,0022) LO [Siemens MedCom Object Graphics]         #  30, 1 OverlayDescription
  (6000,0040) CS [G]                                      #   2, 1 OverlayType
  (6000,0050) SS 1\1                                      #   4, 2 OverlayOrigin
  (6000,0051) US 1                                        #   2, 1 ImageFrameOrigin
  (6000,0100) US 1                                        #   2, 1 OverlayBitsAllocated
  (6000,0102) US 0                                        #   2, 1 OverlayBitPosition
  (6000,3000) OW 0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000... # 29282, 1 OverlayData
  */

  bool InPixelData;
// Identifier need to be in the [6000,60FF] range (no odd number):
  unsigned short Group;
// Descriptor:
  unsigned short Rows;           // (6000,0010) US 484                                      #   2, 1 OverlayRows
  unsigned short Columns;        // (6000,0011) US 484                                      #   2, 1 OverlayColumns
  unsigned int   NumberOfFrames; // (6000,0015) IS [1]                                      #   2, 1 NumberOfFramesInOverlay
  std::string    Description;    // (6000,0022) LO [Siemens MedCom Object Graphics]         #  30, 1 OverlayDescription
  std::string    Type;           // (6000,0040) CS [G]                                      #   2, 1 OverlayType
  signed short   Origin[2];      // (6000,0050) SS 1\1                                      #   4, 2 OverlayOrigin
  unsigned short FrameOrigin;    // (6000,0051) US 1                                        #   2, 1 ImageFrameOrigin
  unsigned short BitsAllocated;  // (6000,0100) US 1                                        #   2, 1 OverlayBitsAllocated
  unsigned short BitPosition;    // (6000,0102) US 0                                        #   2, 1 OverlayBitPosition
  //std::vector<bool> Data;
  std::vector<char> Data; // hold the Overlay data, but not the trailing DICOM padding (\0)
  void Print(std::ostream &os) const {
    os << "Group           0x" <<  std::hex << Group << std::dec << std::endl;
    os << "Rows            " <<  Rows << std::endl;
    os << "Columns         " <<  Columns << std::endl;
    os << "NumberOfFrames  " <<  NumberOfFrames << std::endl;
    os << "Description     " <<  Description << std::endl;
    os << "Type            " <<  Type << std::endl;
    os << "Origin[2]       " <<  Origin[0] << "," << Origin[1] << std::endl;
    os << "FrameOrigin     " <<  FrameOrigin << std::endl;
    os << "BitsAllocated   " <<  BitsAllocated << std::endl;
    os << "BitPosition     " <<  BitPosition << std::endl;
    //std::vector<bool> Data;
  }
};

Overlay::Overlay()
{
  Internal = new OverlayInternal;
}

Overlay::~Overlay()
{
  delete Internal;
}

Overlay::Overlay(Overlay const &ov):Object(ov)
{
  //delete Internal;
  Internal = new OverlayInternal;
  // TODO: copy OverlayInternal into other...
  *Internal = *ov.Internal;
}

void Overlay::Update(const DataElement & de)
{
/*
  8.1.2 Overlay data encoding of related data elements
    Encoded Overlay Planes always have a bit depth of 1, and are encoded
    separately from the Pixel Data in Overlay Data (60xx,3000). The following two
    Data Elements shall define the Overlay Plane structure:
    - Overlay Bits Allocated (60xx,0100)
    - Overlay Bit Position (60xx,0102)
    Notes: 1. There is no Data Element analogous to Bits Stored (0028,0101)
    since Overlay Planes always have a bit depth of 1.
    2. Restrictions on the allowed values for these Data Elements are defined
    in PS 3.3. Formerly overlay data stored in unused bits of Pixel Data
    (7FE0,0010) was described, and these attributes had meaningful values but this
    usage has been retired. See PS 3.5 2004. For overlays encoded in Overlay Data
    Element (60xx,3000), Overlay Bits Allocated (60xx,0100) is always 1 and Overlay
    Bit Position (60xx,0102) is always 0.
*/

  assert( de.GetTag().IsPublic() );
  const ByteValue* bv = de.GetByteValue();
  if( !bv ) return; // Discard any empty element (will default to another value)
  assert( bv->GetPointer() && bv->GetLength() );
  std::string s( bv->GetPointer(), bv->GetLength() );
  // What if a \0 can be found before the end of string...
  //assert( strlen( s.c_str() ) == s.size() );

  // First thing check consistency:
  if( !GetGroup() )
    {
    SetGroup( de.GetTag().GetGroup() );
    }
  else // check consistency
    {
    assert( GetGroup() == de.GetTag().GetGroup() ); // programmer error
    }

  //std::cerr << "Tag: " << de.GetTag() << std::endl;
  if( de.GetTag().GetElement() == 0x0000 ) // OverlayGroupLength
    {
    // ??
    }
  else if( de.GetTag().GetElement() == 0x0010 ) // OverlayRows
    {
    Attribute<0x6000,0x0010> at;
    at.SetFromDataElement( de );
    SetRows( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0011 ) // OverlayColumns
    {
    Attribute<0x6000,0x0011> at;
    at.SetFromDataElement( de );
    SetColumns( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0015 ) // NumberOfFramesInOverlay
    {
    Attribute<0x6000,0x0015> at;
    at.SetFromDataElement( de );
    SetNumberOfFrames( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0022 ) // OverlayDescription
    {
    SetDescription( s.c_str() );
    }
  else if( de.GetTag().GetElement() == 0x0040 ) // OverlayType
    {
    SetType( s.c_str() );
    }
  else if( de.GetTag().GetElement() == 0x0045 ) // OverlaySubtype
    {
    gdcmWarningMacro( "FIXME" );
    }
  else if( de.GetTag().GetElement() == 0x0050 ) // OverlayOrigin
    {
    Attribute<0x6000,0x0050> at;
    at.SetFromDataElement( de );
    SetOrigin( at.GetValues() );
    }
  else if( de.GetTag().GetElement() == 0x0051 ) // ImageFrameOrigin
    {
    Attribute<0x6000,0x0051> at;
    at.SetFromDataElement( de );
    SetFrameOrigin( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0060 ) // OverlayCompressionCode (RET)
    {
    assert( s == "NONE" ); // FIXME ??
    }
  else if( de.GetTag().GetElement() == 0x0100 ) // OverlayBitsAllocated
    {
    Attribute<0x6000,0x0100> at;
    at.SetFromDataElement( de );
    // if OverlayBitsAllocated is 1 it imply OverlayData element
    // if OverlayBitsAllocated is 16 it imply Overlay in unused pixel bits
    if( at.GetValue() != 1 )
      {
      gdcmWarningMacro( "Unsuported OverlayBitsAllocated: " << at.GetValue() );
      }
    SetBitsAllocated( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0102 ) // OverlayBitPosition
    {
    Attribute<0x6000,0x0102> at;
    at.SetFromDataElement( de );
    if( at.GetValue() != 0 ) // For old ACR when using unused bits...
      {
      gdcmDebugMacro( "Unsuported OverlayBitPosition: " << at.GetValue() );
      }
    SetBitPosition( at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x0110 ) // OverlayFormat (RET)
    {
    assert( s == "RECT" );
    }
  else if( de.GetTag().GetElement() == 0x0200 ) // OverlayLocation (RET)
    {
    Attribute<0x6000,0x0200> at;
    at.SetFromDataElement( de );
    gdcmWarningMacro( "FIXME: " << at.GetValue() );
    }
  else if( de.GetTag().GetElement() == 0x1301 ) // ROIArea
    {
    gdcmWarningMacro( "FIXME" );
    }
  else if( de.GetTag().GetElement() == 0x1302 ) // ROIMean
    {
    gdcmWarningMacro( "FIXME" );
    }
  else if( de.GetTag().GetElement() == 0x1303 ) // ROIStandardDeviation
    {
    gdcmWarningMacro( "FIXME" );
    }
  else if( de.GetTag().GetElement() == 0x1500 ) // OverlayLabel
    {
    gdcmWarningMacro( "FIXME" );
    }
  else if( de.GetTag().GetElement() == 0x3000 ) // OverlayData
    {
    SetOverlay(bv->GetPointer(), bv->GetLength());
    }
  else
    {
    gdcmErrorMacro( "Tag is not supported: " << de.GetTag() << std::endl );
    assert(0);
    }
}

bool Overlay::GrabOverlayFromPixelData(DataSet const &ds)
{
  const unsigned int ovlength = Internal->Rows * Internal->Columns / 8;
  Internal->Data.resize( ovlength ); // set to 0
  if( Internal->BitsAllocated == 16 )
    {
    //assert( Internal->BitPosition >= 12 );
    if( ds.FindDataElement( Tag(0x7fe0,0x0010) ) )
      {
      gdcmErrorMacro("Could not find Pixel Data. Cannot extract Overlay." );
      return false;
      }
    const DataElement &pixeldata = ds.GetDataElement( Tag(0x7fe0,0x0010) );
    const ByteValue *bv = pixeldata.GetByteValue();
    if( !bv )
      {
      // XA_GE_JPEG_02_with_Overlays.dcm
      return false;
      }
    assert( bv );
    const char *array = bv->GetPointer();
    // SIEMENS_GBS_III-16-ACR_NEMA_1.acr is pain to support,
    // I cannot simply use the bv->GetLength I have to use the image dim:
    const unsigned int length = ovlength * 8 * 2; //bv->GetLength();
    const uint16_t *p = (uint16_t*)array;
    const uint16_t *end = (uint16_t*)(array + length);
    //const unsigned int ovlength = length / (8*2);
    assert( 8 * ovlength == (unsigned int)Internal->Rows * Internal->Columns );
    if( Internal->Data.empty() )
      {
      return false;
      }
    unsigned char * overlay = (unsigned char*)&Internal->Data[0];
    int c = 0;
    uint16_t pmask = (uint16_t)(1 << Internal->BitPosition);
    assert( length / 2 == ovlength * 8 );
    while( p != end )
      {
      const uint16_t val = *p & pmask;
      assert( val == 0x0 || val == pmask );
      // 128 -> 0x80
      if( val )
        {
        overlay[ c / 8 ] |= (unsigned char)(0x1 << c%8);
        }
      else
        {
        // else overlay[ c / 8 ] is already 0
        }
      ++p;
      ++c;
      }
    assert( (unsigned)c / 8 == ovlength );
    }
  else
    {
    gdcmErrorMacro( "Could not grab Overlay from image. Please report." );
    return false;
    }
  return true;
}

void Overlay::SetGroup(unsigned short group) { Internal->Group = group; }
unsigned short Overlay::GetGroup() const { return Internal->Group; }

void Overlay::SetRows(unsigned short rows) { Internal->Rows = rows; }
unsigned short Overlay::GetRows() const { return Internal->Rows; }
void Overlay::SetColumns(unsigned short columns) { Internal->Columns = columns; }
unsigned short Overlay::GetColumns() const { return Internal->Columns; }
void Overlay::SetNumberOfFrames(unsigned int numberofframes) { Internal->NumberOfFrames = numberofframes; }
void Overlay::SetDescription(const char* description) { if( description ) Internal->Description = description; }
const char *Overlay::GetDescription() const { return Internal->Description.c_str(); }
void Overlay::SetType(const char* type) { if( type ) Internal->Type = type; }
const char *Overlay::GetType() const { return Internal->Type.c_str(); }
static const char *OverlayTypeStrings[] = {
  "INVALID",
  "G ",
  "R ",
};
const char *Overlay::GetOverlayTypeAsString(OverlayType ot)
{
  return OverlayTypeStrings[ (int) ot ];
}
Overlay::OverlayType Overlay::GetOverlayTypeFromString(const char *s)
{
  static const int n = sizeof( OverlayTypeStrings ) / sizeof ( *OverlayTypeStrings );
  if( s )
    {
    for( int i = 0; i < n; ++i )
      {
      if( strcmp(s, OverlayTypeStrings[i] ) == 0 )
        {
        return (OverlayType)i;
        }
      }
    }
  // could not find the proper type, maybe padded with \0 ?
  if( strlen(s) == 1 )
    {
    for( int i = 0; i < n; ++i )
      {
      if( strncmp(s, OverlayTypeStrings[i], 1 ) == 0 )
        {
        gdcmDebugMacro( "Invalid Padding for OVerlay Type" );
        return (OverlayType)i;
        }
      }
    }
  return Overlay::Invalid;
}
Overlay::OverlayType Overlay::GetTypeAsEnum() const
{
  return GetOverlayTypeFromString( GetType() );
}

void Overlay::SetOrigin(const signed short origin[2])
{
  if( origin )
    {
    Internal->Origin[0] = origin[0];
    Internal->Origin[1] = origin[1];
    }
}
const signed short * Overlay::GetOrigin() const
{
  return &Internal->Origin[0];
}
void Overlay::SetFrameOrigin(unsigned short frameorigin) { Internal->FrameOrigin = frameorigin; }
void Overlay::SetBitsAllocated(unsigned short bitsallocated) { Internal->BitsAllocated = bitsallocated; }
unsigned short Overlay::GetBitsAllocated() const { return Internal->BitsAllocated; }
void Overlay::SetBitPosition(unsigned short bitposition) { Internal->BitPosition = bitposition; }
unsigned short Overlay::GetBitPosition() const { return Internal->BitPosition; }

bool Overlay::IsEmpty() const
{
  return Internal->Data.empty();
}
bool Overlay::IsZero() const
{
  if( IsEmpty() ) return false;

  std::vector<char>::const_iterator it = Internal->Data.begin();
  for(; it != Internal->Data.end(); ++it )
    {
    if( *it ) return true;
    }
  return false;
}
bool Overlay::IsInPixelData() const { return Internal->InPixelData; }
void Overlay::IsInPixelData(bool b) { Internal->InPixelData = b; }

/*
 * row,col = 400,400   => 20000
 * row,col = 1665,1453 => 302406
 * row,col = 20,198    => 495 words + 1 dicom \0 padding
 */
inline unsigned int compute_bit_and_dicom_padding(unsigned short rows, unsigned short columns)
{
  unsigned int word_padding = ( rows * columns + 7 ) / 8; // need to send full word (8bits at a time)
  return word_padding + word_padding%2; // Cannot have odd length
}

void Overlay::SetOverlay(const char *array, size_t length)
{
  if( !array || !length ) return;
  const size_t computed_length = (Internal->Rows * Internal->Columns + 7) / 8;
  Internal->Data.resize( computed_length ); // filled with 0 if length < computed_length
  if( length < computed_length )
    {
    gdcmWarningMacro( "Not enough data found in Overlay. Proceed with caution" );
    std::copy(array, array+length, Internal->Data.begin());
    }
  else
    {
    // do not try to copy more than allocated:
    // technically we may be missing the trailing DICOM padding (\0), but we have all the data needed anyway:
    std::copy(array, array+computed_length, Internal->Data.begin());
    }
  /* warning need to take into account padding to the next word (8bits) */
  //assert( length == compute_bit_and_dicom_padding(Internal->Rows, Internal->Columns) );
  assert( Internal->Data.size() == computed_length );
}

const ByteValue &Overlay::GetOverlayData() const
{
  static ByteValue bv;
  bv = ByteValue( Internal->Data );
  return bv;
}

size_t Overlay::GetUnpackBufferLength() const
{
  const size_t unpacklen = Internal->Rows * Internal->Columns;
  return unpacklen;
}

bool Overlay::GetUnpackBuffer(char *buffer, size_t len) const
{
  const size_t unpacklen = GetUnpackBufferLength();
  if( len < unpacklen ) return false;
  unsigned char *unpackedbytes = (unsigned char*)buffer;
  const unsigned char *begin = unpackedbytes;
  for( std::vector<char>::const_iterator it = Internal->Data.begin(); it != Internal->Data.end(); ++it )
    {
    assert( unpackedbytes <= begin + len ); // We never store more than actually required
    // const unsigned char &packedbytes = *it;
    // weird bug with gcc 3.3 (prerelease on SuSE) apparently:
    unsigned char packedbytes = static_cast<unsigned char>(*it);
    unsigned char mask = 1;
    for (unsigned int i = 0; i < 8 && unpackedbytes < begin + len; ++i)
      {
      if ( (packedbytes & mask) == 0)
        {
        *unpackedbytes = 0;
        }
      else
        {
        *unpackedbytes = 255;
        }
      ++unpackedbytes;
      mask <<= 1;
      }
    }
  assert(unpackedbytes <= begin + len);
  return true;
}

void Overlay::Decompress(std::ostream &os) const
{
  const size_t unpacklen = GetUnpackBufferLength();
  unsigned char unpackedbytes[8];
  //std::vector<char>::const_iterator beg = Internal->Data.begin();
  size_t curlen = 0;
  for( std::vector<char>::const_iterator it = Internal->Data.begin(); it != Internal->Data.end(); ++it )
    {
    unsigned char packedbytes = *it;
    unsigned char mask = 1;
    unsigned int i = 0;
    for (; i < 8 && curlen < unpacklen; ++i)
      {
      if ( (packedbytes & mask) == 0)
        {
        unpackedbytes[i] = 0;
        }
      else
        {
        unpackedbytes[i] = 255;
        }
      mask <<= 1;
      ++curlen;
      }
    os.write(reinterpret_cast<char*>(unpackedbytes), i);
    }
}

void Overlay::Print(std::ostream &os) const
{
  Internal->Print( os );
}

} // end namespace gdcm
