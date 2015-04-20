/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIconImage.h"

namespace gdcm
{
} // end namespace gdcm

#if 0
#include "gdcmRAWCodec.h"
#include "gdcmSequenceOfFragments.h"

namespace gdcm
{
/*
 * PICKER-16-MONO2-Nested_icon.dcm:
(0088,0200) SQ (Sequence with undefined length #=1)     # u/l, 1 IconImageSequence
  (fffe,e000) na (Item with undefined length #=10)        # u/l, 1 Item
    (0028,0002) US 1                                        #   2, 1 SamplesPerPixel
    (0028,0004) CS [MONOCHROME2]                            #  12, 1 PhotometricInterpretation
    (0028,0010) US 64                                       #   2, 1 Rows
    (0028,0011) US 64                                       #   2, 1 Columns
    (0028,0034) IS [1\1]                                    #   4, 2 PixelAspectRatio
    (0028,0100) US 8                                        #   2, 1 BitsAllocated
    (0028,0101) US 8                                        #   2, 1 BitsStored
    (0028,0102) US 7                                        #   2, 1 HighBit
    (0028,0103) US 0                                        #   2, 1 PixelRepresentation
    (7fe0,0010) OW 0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000... # 4096, 1 PixelData
  (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
(fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
*/

IconImage::IconImage():
  TS(),
  PF(),
  PI(),
  Dimensions(),
  Spacing(),
  PixelData() {}

IconImage::~IconImage() {}

void IconImage::SetDimension(unsigned int idx, unsigned int dim)
{
  assert( idx < NumberOfDimensions );
  Dimensions.resize( NumberOfDimensions );
  // Can dim be 0 ??
  Dimensions[idx] = dim;
}

void IconImage::Clear()
{
  Dimensions.clear();
}

const PhotometricInterpretation &IconImage::GetPhotometricInterpretation() const
{
  return PI;
}

void IconImage::SetPhotometricInterpretation(
  PhotometricInterpretation const &pi)
{
  PI = pi;
}

bool IconImage::GetBuffer(char *buffer) const
{
  if( IsEmpty() )
    {
    buffer = 0;
    return false;
    }

  const ByteValue *bv = PixelData.GetByteValue();
  if( !bv )
    {
    // KODAK_CompressedIcon.dcm
    // contains a compressed Icon Sequence, one has to guess this is lossless jpeg...
#ifdef MDEBUG
    const SequenceOfFragments *sqf = PixelData.GetSequenceOfFragments();
    std::ofstream os( "/tmp/kodak.ljpeg", std::ios::binary );
    sqf->WriteBuffer( os );
#endif
    gdcmWarningMacro( "Compressed Icon are not support for now" );
    buffer = 0;
    return false;
    }
  assert( bv );
  RAWCodec codec;
  //assert( GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME2 );
  //codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
  if( GetPhotometricInterpretation() != PhotometricInterpretation::MONOCHROME2 )
    {
    gdcmWarningMacro( "PhotometricInterpretation: " << GetPhotometricInterpretation() << " not handled for now" );
    }
  codec.SetPhotometricInterpretation( PhotometricInterpretation::MONOCHROME2 );
  codec.SetPixelFormat( GetPixelFormat() );
  codec.SetPlanarConfiguration( 0 );
  DataElement out;
  bool r = codec.Decode(PixelData, out);
  assert( r );
  const ByteValue *outbv = out.GetByteValue();
  assert( outbv );
  //unsigned long check = outbv->GetLength();  // FIXME
  memcpy(buffer, outbv->GetPointer(), outbv->GetLength() );  // FIXME
  return r;
}


}
#endif
