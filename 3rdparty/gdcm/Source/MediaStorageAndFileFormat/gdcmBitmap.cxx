/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmBitmap.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmRAWCodec.h"
#include "gdcmJPEGCodec.h"
#include "gdcmPVRGCodec.h"
#include "gdcmKAKADUCodec.h"
#include "gdcmJPEGLSCodec.h"
#if defined(GDCM_USE_OPENJPEG)
#include "gdcmJPEG2000Codec.h"
#endif
#include "gdcmRLECodec.h"

#include <cstring>

namespace gdcm
{
/*
 * PICKER-16-MONO2-Nested_icon.dcm:
(0088,0200) SQ (Sequence with undefined length #=1)     # u/l, 1 BitmapSequence
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

Bitmap::Bitmap():
  PlanarConfiguration(0),
  NumberOfDimensions(2),
  TS(),
  PF(),
  PI(),
  Dimensions(),
  PixelData(),
  LUT(new LookupTable),
  NeedByteSwap(false),
  LossyFlag(false)
{}

Bitmap::~Bitmap() {}

/*
 * Internal implementation everything assume that NumberOfDimensions was set
 */
unsigned int Bitmap::GetNumberOfDimensions() const
{
  assert( NumberOfDimensions );
  return NumberOfDimensions;
}

void Bitmap::SetNumberOfDimensions(unsigned int dim)
{
  NumberOfDimensions = dim;
  assert( NumberOfDimensions );
  Dimensions.resize( 3 /*NumberOfDimensions*/ ); // fill with 0
  assert( NumberOfDimensions == 2 || NumberOfDimensions == 3 );
  if( NumberOfDimensions == 2 )
    {
    Dimensions[2] = 1;
    //Spacing[2] = 1;
    }
}

const unsigned int *Bitmap::GetDimensions() const
{
  assert( NumberOfDimensions );
  return &Dimensions[0];
}

unsigned int Bitmap::GetDimension(unsigned int idx) const
{
  assert( NumberOfDimensions );
  return Dimensions[idx];
}

void Bitmap::SetDimensions(const unsigned int *dims)
{
  assert( NumberOfDimensions );
  //assert( Dimensions.empty() );
  Dimensions = std::vector<unsigned int>(dims,
    dims+NumberOfDimensions);
}

void Bitmap::SetDimension(unsigned int idx, unsigned int dim)
{
  //assert( dim );
  assert( NumberOfDimensions );
  assert( idx < NumberOfDimensions );
  Dimensions.resize( 3 /*NumberOfDimensions*/ );
  // Can dim be 0 ??
  // -> no !
  //assert( dim ); // PhilipsLosslessRice.dcm
  Dimensions[idx] = dim;
  if( NumberOfDimensions == 2 )
    {
    Dimensions[2] = 1;
    }
}

// TODO does it make sense to PlanarConfiguration in Bitmap
// and SamplesPerPixel in PixelFormat when those two are linked...
unsigned int Bitmap::GetPlanarConfiguration() const
{
  if( PlanarConfiguration && PF.GetSamplesPerPixel() != 3 )
    {
    assert(0);
    // LEADTOOLS_FLOWERS-8-PAL-RLE.dcm
    // User specify PlanarConfiguration whereas SamplesPerPixel != 3
    gdcmWarningMacro(
      "Can't set PlanarConfiguration if SamplesPerPixel is not 3" );
    // Let's assume it's this way...
    return 0;
    }
  return PlanarConfiguration;
}

void Bitmap::SetPlanarConfiguration(unsigned int pc)
{
  // precondition
  assert( pc == 0 || pc == 1 );
  PlanarConfiguration = pc;
  if( pc )
    {
    // LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm
    if( PF.GetSamplesPerPixel() != 3 ) // Please set PixelFormat first
      {
      gdcmWarningMacro( "Cant have Planar Configuration in non RGB input. Discarding" );
      PlanarConfiguration = 0;
      }
    const TransferSyntax &ts = GetTransferSyntax();
    if(  ts == TransferSyntax::JPEGBaselineProcess1
      || ts == TransferSyntax::JPEGExtendedProcess2_4
      || ts == TransferSyntax::JPEGExtendedProcess3_5
      || ts == TransferSyntax::JPEGSpectralSelectionProcess6_8
      || ts == TransferSyntax::JPEGFullProgressionProcess10_12
      || ts == TransferSyntax::JPEGLosslessProcess14
      || ts == TransferSyntax::JPEGLosslessProcess14_1
      || ts == TransferSyntax::JPEGLSLossless
      || ts == TransferSyntax::JPEGLSNearLossless
      || ts == TransferSyntax::JPEG2000Lossless
      || ts == TransferSyntax::JPEG2000
      || ts == TransferSyntax::JPIPReferenced
    )
      {
      // PS 3.6 - 2011 8.2.4 JPEG 2000 IMAGE COMPRESSION
      // The value of Planar Configuration (0028,0006) is irrelevant since the
      // manner of encoding components is specified in the JPEG 2000 standard,
      // hence it shall be set to 0.
      // By extension, this behavior has been applied also to JPEG and JPEG-LS
      gdcmWarningMacro( "Cant have Planar Configuration in JPEG/JPEG-LS/JPEG 2000. Discarding" );
      PlanarConfiguration = 0;
      }
    }
  // \postcondition
  assert( PlanarConfiguration == 0 || PlanarConfiguration == 1 );
}



void Bitmap::Clear()
{
  Dimensions.clear();
}

const PhotometricInterpretation &Bitmap::GetPhotometricInterpretation() const
{
  return PI;
}

void Bitmap::SetPhotometricInterpretation(
  PhotometricInterpretation const &pi)
{
  PI = pi;
}

#if 0
bool Bitmap::GetBuffer(char *buffer) const
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
    std::ofstream os( "/tmp/kodak.ljpeg", std::ios::binary);
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
#endif

unsigned long Bitmap::GetBufferLength() const
{
  //assert( !IsEncapsulated() );
  if( PF == PixelFormat::UNKNOWN ) return 0;

  assert( NumberOfDimensions );
  //assert( NumberOfDimensions == Dimensions.size() );
  if( NumberOfDimensions != Dimensions.size() )
    {
    assert( Dimensions[2] == 1 );
    }
  unsigned long len = 0;
  unsigned int mul = 1;
  // First multiply the dimensions:
  std::vector<unsigned int>::const_iterator it = Dimensions.begin();
  for(; it != Dimensions.end(); ++it)
    {
    assert( *it );
    mul *= *it;
    }
  // Multiply by the pixel size:
  // Special handling of packed format:
  if( PF == PixelFormat::UINT12 || PF == PixelFormat::INT12 )
    {
#if 1
    mul *= PF.GetPixelSize();
#else
    assert( PF.GetSamplesPerPixel() == 1 );
    unsigned int save = mul;
    save *= 12;
    save /= 8;
    assert( save * 8 / 12 == mul );
    mul = save;
#endif
    }
  else if( PF == PixelFormat::SINGLEBIT )
    {
    assert( PF.GetSamplesPerPixel() == 1 );
    unsigned int save = mul;
    save /= 8;
    assert( save * 8 == mul );
    mul = save;
    }
  else if( PF.GetBitsAllocated() % 8 != 0 )
    {
    // gdcmDataExtra/gdcmSampleData/images_of_interest/USBitsAllocated14.dcm
    // BitsAllocated      :14
    // BitsStored         :14
    // HighBit            :13
    assert( PF.GetSamplesPerPixel() == 1 );
    const ByteValue *bv = PixelData.GetByteValue();
    assert( bv );
    unsigned int ref = bv->GetLength() / mul;
    if( !GetTransferSyntax().IsEncapsulated() )
      assert( bv->GetLength() % mul == 0 );
    mul *= ref;
    }
  else
    {
    mul *= PF.GetPixelSize();
    }
  len = mul;

  assert( len != 0 );
  return len;
}

bool Bitmap::TryRAWCodec(char *buffer, bool &lossyflag) const
{
  RAWCodec codec;
  const TransferSyntax &ts = GetTransferSyntax();
  if(!buffer)
    {
    if( codec.CanDecode( ts ) ) // short path
      {
      lossyflag = false;
      return true;
      }
    return false;
    }

  const ByteValue *bv = PixelData.GetByteValue();
  if( bv )
    {
    unsigned long len = GetBufferLength();
    if( !codec.CanDecode( ts ) ) return false;
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetLUT( GetLUT() );
    codec.SetPixelFormat( GetPixelFormat() );
    codec.SetNeedByteSwap( GetNeedByteSwap() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    DataElement out;
    //bool r = codec.Decode(PixelData, out);
    bool r = codec.DecodeBytes(bv->GetPointer(), bv->GetLength(),
      buffer, len);
    if( !r ) return false;
    //const ByteValue *outbv = out.GetByteValue();
    //assert( outbv );
    if( len != bv->GetLength() )
      {
      // SIEMENS_GBS_III-16-ACR_NEMA_1.acr
      // This is also handling the famous DermaColorLossLess.dcm issue
      // where RGB image is odd length (GetBufferLength()) but
      // ByteValue::GetLength is rounded up to the next even byte length
    //  gdcmDebugMacro( "Pixel Length " << bv->GetLength() <<
    //    " is different from computed value " << len );
    //  ((ByteValue*)outbv)->SetLength( len );
      }
#if 0
    if ( GetPixelFormat() != codec.GetPixelFormat() )
      {
      Bitmap *i = (Bitmap*)this;
      i->SetPixelFormat( codec.GetPixelFormat() );
      }
#endif

    unsigned long check; // = outbv->GetLength();  // FIXME
    check = len;
    // DermaColorLossLess.dcm
    assert( check == len || check == len + 1 );
    (void)check;// removing warning
    //if(buffer) memcpy(buffer, outbv->GetPointer(), outbv->GetLength() );  // FIXME
    return r;
    }
  return false;
}

bool Bitmap::TryJPEGCodec(char *buffer, bool &lossyflag) const
{
  JPEGCodec codec;
  const TransferSyntax &ts = GetTransferSyntax();
  if(!buffer)
    {
    if( codec.CanDecode( ts ) ) // short path
      {
      TransferSyntax ts2;
      const SequenceOfFragments *sf = PixelData.GetSequenceOfFragments();
      if( !sf ) return false;
      const Fragment &frag = sf->GetFragment(0);
      const ByteValue &bv2 = dynamic_cast<const ByteValue&>(frag.GetValue());
      PixelFormat pf = GetPixelFormat(); // PixelFormat::UINT8;
      codec.SetPixelFormat( pf );

      std::stringstream ss;
      ss.write( bv2.GetPointer(), bv2.GetLength() );
      bool b = codec.GetHeaderInfo( ss, ts2 );
      //bool b = codec.GetHeaderInfo( bv2.GetPointer(), bv2.GetLength() , ts2 );
      if(!b) return false;
      assert( b );
      lossyflag = codec.IsLossy();
      // we need to know the actual pixeltype after ::Read
#if 0
      if( codec.GetPixelFormat() != GetPixelFormat() )
        {
        Bitmap *i = (Bitmap*)this;
        i->SetPixelFormat( codec.GetPixelFormat() );
        }
#endif

      return true;
      }
    return false;
    }

  if( codec.CanDecode( ts ) )
    {
    unsigned long len = GetBufferLength();
    codec.SetNumberOfDimensions( GetNumberOfDimensions() );
    codec.SetDimensions( GetDimensions() );
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetPixelFormat( GetPixelFormat() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    DataElement out;
    bool r = codec.Decode(PixelData, out);
    // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
    if( !r )
      {
      return false;
      }
    // FIXME ! This should be done all the time for all codec:
    // Did PI change or not ?
    if ( GetPlanarConfiguration() != codec.GetPlanarConfiguration() )
      {
      Bitmap *i = (Bitmap*)this; (void)i;
      //i->SetPlanarConfiguration( codec.GetPlanarConfiguration() );
      }
    // I cannot re-activate the following since I would loose the palette color information
    // (this is not stored in the JPEG header).
    //if ( GetPhotometricInterpretation() != codec.GetPhotometricInterpretation() )
    //  {
    //  // HACK
    //  // YBRisGray.dcm
    //  Bitmap *i = (Bitmap*)this;
    //  i->SetPhotometricInterpretation( codec.GetPhotometricInterpretation() );
    //  }
#if 1
    if ( GetPixelFormat() != codec.GetPixelFormat() )
      {
      // gdcmData/DCMTK_JPEGExt_12Bits.dcm
      assert( GetPixelFormat().GetPixelRepresentation() ==
        codec.GetPixelFormat().GetPixelRepresentation() );
//      assert( GetPixelFormat().GetBitsStored() ==
//        codec.GetPixelFormat().GetBitsStored() );
      assert( GetPixelFormat().GetBitsAllocated() == 12 );
      Bitmap *i = (Bitmap*)this;
      i->SetPixelFormat( codec.GetPixelFormat() );
      }
#endif
    //if ( GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL_422
    //|| GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL )
    //  {
    //  Bitmap *i = (Bitmap*)this;
    //  i->SetPhotometricInterpretation( PhotometricInterpretation::RGB );
    //  }
    const ByteValue *outbv = out.GetByteValue();
    assert( outbv );
    unsigned long check = outbv->GetLength();  // FIXME
    (void)check;
    // DermaColorLossLess.dcm has a len of 63531, but DICOM will give us: 63532 ...
    if( len > outbv->GetLength() )
      {
      gdcmErrorMacro( "Impossible length: " << len << " should be (max): " << outbv->GetLength() );
      return false;
      }
    assert( len <= outbv->GetLength() );
    if(buffer) memcpy(buffer, outbv->GetPointer(), len /*outbv->GetLength()*/ );  // FIXME

    lossyflag = codec.IsLossy();
    //assert( codec.IsLossy() == ts.IsLossy() );

    return true;
    }
  return false;
}

bool Bitmap::TryJPEGCodec2(std::ostream &os) const
{
  unsigned long len = GetBufferLength();
  const TransferSyntax &ts = GetTransferSyntax();

  JPEGCodec codec;
  if( codec.CanCode( ts ) )
    {
    codec.SetDimensions( GetDimensions() );
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetPixelFormat( GetPixelFormat() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    DataElement out;
    bool r = codec.Code(PixelData, out);
    // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
    if( !r )
      {
      return false;
      }
    // FIXME ! This should be done all the time for all codec:
    // Did PI change or not ?
    if ( GetPhotometricInterpretation() != codec.GetPhotometricInterpretation() )
      {
      // HACK
      //Bitmap *i = (Bitmap*)this;
      //i->SetPhotometricInterpretation( codec.GetPhotometricInterpretation() );
      }
    const ByteValue *outbv = out.GetByteValue();
    assert( outbv );
    unsigned long check = outbv->GetLength();  // FIXME
    (void)check;
    // DermaColorLossLess.dcm has a len of 63531, but DICOM will give us: 63532 ...
    assert( outbv->GetLength() < len ); (void)len;
    //memcpy(buffer, outbv->GetPointer(), outbv->GetLength() );
    os.write( outbv->GetPointer(), outbv->GetLength() );

    return true;
    }
  return false;
}

bool Bitmap::TryPVRGCodec(char *buffer, bool &lossyflag) const
{
  unsigned long len = GetBufferLength();
  const TransferSyntax &ts = GetTransferSyntax();

  PVRGCodec codec;
  if( codec.CanDecode( ts ) )
    {
    codec.SetPixelFormat( GetPixelFormat() );
    //codec.SetBufferLength( len );
    //codec.SetNumberOfDimensions( GetNumberOfDimensions() );
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    codec.SetDimensions( GetDimensions() );
    DataElement out;
    bool r = codec.Decode(PixelData, out);
    if(!r) return false;
    codec.SetLossyFlag( ts.IsLossy() );
    assert( r );
    if ( GetPlanarConfiguration() != codec.GetPlanarConfiguration() )
      {
      Bitmap *i = (Bitmap*)this;
      i->PlanarConfiguration = codec.GetPlanarConfiguration();
      }
    const ByteValue *outbv = out.GetByteValue();
    assert( outbv );
    unsigned long check = outbv->GetLength();  // FIXME
    (void)check;
    assert( len <= outbv->GetLength() );
    if(buffer) memcpy(buffer, outbv->GetPointer(), len /*outbv->GetLength()*/ );  // FIXME

    lossyflag = codec.IsLossy();
    //assert( codec.IsLossy() == ts.IsLossy() );

    return r;
    }
  return false;
}

bool Bitmap::TryKAKADUCodec(char *buffer, bool &lossyflag) const
{
  unsigned long len = GetBufferLength();
  const TransferSyntax &ts = GetTransferSyntax();

  KAKADUCodec codec;
  if( codec.CanDecode( ts ) )
    {
    codec.SetPixelFormat( GetPixelFormat() );
    //codec.SetBufferLength( len );
    codec.SetNumberOfDimensions( GetNumberOfDimensions() );
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    codec.SetDimensions( GetDimensions() );
    DataElement out;
    bool r = codec.Decode(PixelData, out);
    if( !r ) return false;
    const ByteValue *outbv = out.GetByteValue();
    assert( outbv );
    unsigned long check = outbv->GetLength();  // FIXME
    (void)check;
    assert( len <= outbv->GetLength() );
    // DermaColorLossLess.dcm has a len of 63531, but DICOM will give us: 63532 ...
    assert( len <= outbv->GetLength() );
    if(buffer) memcpy(buffer, outbv->GetPointer(), len /*outbv->GetLength()*/ );  // FIXME

    //assert( codec.IsLossy() == ts.IsLossy() );
    lossyflag = codec.IsLossy();
    if( codec.IsLossy() != ts.IsLossy() )
      {
      gdcmErrorMacro( "EVIL file, it is declared as lossless but is in fact lossy." );
      }

    return r;
    }
  return false;
}

bool Bitmap::TryJPEGLSCodec(char *buffer, bool &lossyflag) const
{
  JPEGLSCodec codec;
  const TransferSyntax &ts = GetTransferSyntax();
  if(!buffer)
    {
    if( codec.CanDecode( ts ) ) // short path
      {
      TransferSyntax ts2;
      const SequenceOfFragments *sf = PixelData.GetSequenceOfFragments();
      if( !sf ) return false;
      const Fragment &frag = sf->GetFragment(0);
      const ByteValue &bv2 = dynamic_cast<const ByteValue&>(frag.GetValue());

      std::stringstream ss;
      ss.write( bv2.GetPointer(), bv2.GetLength() );
      bool b = codec.GetHeaderInfo( ss, ts2 );
      //bool b = codec.GetHeaderInfo( bv2.GetPointer(), bv2.GetLength() , ts2 );
      if( !b ) return false;
      lossyflag = codec.IsLossy();
      // we need to know the actual pixeltype after ::Read
#if 0
// This is actually very dangerous to change the pixel format right here. What if
// user stored a 16/10/9 signed image using JPEG-LS, JPEG-LS would be required to use
// the full spectrum of the unsigned short 16 bits range to store that image and would
// therefore -rightfully- declared as 16 bits...
      if( codec.GetPixelFormat() != GetPixelFormat() )
        {
        Bitmap *i = (Bitmap*)this;
        i->SetPixelFormat( codec.GetPixelFormat() );
        }
#endif

      return true;
      }
    return false;
    }


  if( codec.CanDecode( ts ) )
    {
    unsigned long len = GetBufferLength();
    codec.SetPixelFormat( GetPixelFormat() );
    codec.SetBufferLength( len );
    codec.SetNumberOfDimensions( GetNumberOfDimensions() );
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    codec.SetDimensions( GetDimensions() );
    DataElement out;
    bool r = codec.Decode(PixelData, out);
    if( !r ) return false;
    const ByteValue *outbv = out.GetByteValue();
    assert( outbv );
    unsigned long check = outbv->GetLength();  // FIXME
    (void)check;
    assert( len <= outbv->GetLength() );
    // DermaColorLossLess.dcm has a len of 63531, but DICOM will give us: 63532 ...
    assert( len <= outbv->GetLength() );
    if(buffer) memcpy(buffer, outbv->GetPointer(), len /*outbv->GetLength()*/ );  // FIXME

    //assert( codec.IsLossy() == ts.IsLossy() );
    lossyflag = codec.IsLossy();
    if( codec.IsLossy() != ts.IsLossy() )
      {
      gdcmErrorMacro( "EVIL file, it is declared as lossless but is in fact lossy." );
      }

    return r;
    }
  return false;
}

bool Bitmap::IsLossy() const
{
  // FIXME each call is expensive...
  //bool lossyflag;
  //if( this->GetBufferInternal(0, lossyflag) )
  //  {
  //  return lossyflag;
  //  }
  //return false;
  return LossyFlag;
}

bool Bitmap::ComputeLossyFlag()
{
  bool lossyflag;
  if( this->GetBufferInternal(0, lossyflag) )
    {
    LossyFlag = lossyflag;
    return true;
    }
  LossyFlag = false;
  return false;
}

#if defined(GDCM_USE_OPENJPEG)
bool Bitmap::TryJPEG2000Codec(char *buffer, bool &lossyflag) const
{
  JPEG2000Codec codec;
  const TransferSyntax &ts = GetTransferSyntax();
  if(!buffer)
    {
    if( codec.CanDecode( ts ) ) // short path
      {
      TransferSyntax ts2;
      const SequenceOfFragments *sf = PixelData.GetSequenceOfFragments();
      if( !sf ) return false;
      const Fragment &frag = sf->GetFragment(0);
      const ByteValue &bv2 = dynamic_cast<const ByteValue&>(frag.GetValue());

      bool b = codec.GetHeaderInfo( bv2.GetPointer(), bv2.GetLength() , ts2 );
      if( !b ) return false;
      lossyflag = codec.IsLossy();
      // we need to know the actual pixeltype after ::Read
#if 0
      if( codec.GetPixelFormat() != GetPixelFormat() )
        {
        // Because J2K support the full spectrum I do not see any issue
        // with the following:
        Bitmap *i = (Bitmap*)this;
        i->SetPixelFormat( codec.GetPixelFormat() );
        }
#else
      // lets only check the only issue we have:
      // OsirixFake16BitsStoredFakeSpacing.dcm
      const PixelFormat & cpf = codec.GetPixelFormat();
      const PixelFormat & pf = GetPixelFormat();
      if( cpf.GetBitsAllocated() == pf.GetBitsAllocated() )
        {
        if( cpf.GetPixelRepresentation() == pf.GetPixelRepresentation() )
          {
          if( cpf.GetSamplesPerPixel() == pf.GetSamplesPerPixel() )
            {
            if( cpf.GetBitsStored() < pf.GetBitsStored() )
              {
              Bitmap *i = (Bitmap*)this;
              gdcmWarningMacro( "Encapsulated stream has fewer bits actually stored on disk. correcting." );
              i->GetPixelFormat().SetBitsStored( cpf.GetBitsStored() );
              }
            }
          }
        }
      else
        {
        // SC16BitsAllocated_8BitsStoredJ2K.dcm
        gdcmWarningMacro( "Bits Allocated are different. This is pretty bad using info from codestream" );
        Bitmap *i = (Bitmap*)this;
        i->SetPixelFormat( codec.GetPixelFormat() );
        }
#endif

      return true;
      }
    return false;
    }

  if( codec.CanDecode( ts ) )
    {
    unsigned long len = GetBufferLength();
    codec.SetPixelFormat( GetPixelFormat() );
    codec.SetNumberOfDimensions( GetNumberOfDimensions() );
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    codec.SetDimensions( GetDimensions() );
    DataElement out;
    bool r = codec.Decode(PixelData, out);
    if(!r) return false;
    assert( r );
    const ByteValue *outbv = out.GetByteValue();
    assert( outbv );
    unsigned long check = outbv->GetLength();  // FIXME
    (void)check;
    assert( len <= outbv->GetLength() );
    if(buffer) memcpy(buffer, outbv->GetPointer(), len /*outbv->GetLength()*/ );  // FIXME

    lossyflag = codec.IsLossy();
    if( codec.IsLossy() && !ts.IsLossy() )
      {
      assert( codec.IsLossy() );
      assert( !ts.IsLossy() );
      gdcmErrorMacro( "EVIL file, it is declared as lossless but is in fact lossy." );
      }
#if 0
    if( codec.GetPixelFormat() != GetPixelFormat() )
      {
      Bitmap *i = (Bitmap*)this;
      i->SetPixelFormat( codec.GetPixelFormat() );
      }
#else
      // lets only check the only issue we have:
      // OsirixFake16BitsStoredFakeSpacing.dcm
      const PixelFormat & cpf = codec.GetPixelFormat();
      const PixelFormat & pf = GetPixelFormat();
      if( cpf.GetBitsAllocated() == pf.GetBitsAllocated() )
        {
        if( cpf.GetPixelRepresentation() == pf.GetPixelRepresentation() )
          {
          if( cpf.GetSamplesPerPixel() == pf.GetSamplesPerPixel() )
            {
            if( cpf.GetBitsStored() < pf.GetBitsStored() )
              {
              Bitmap *i = (Bitmap*)this;
              gdcmWarningMacro( "Encapsulated stream has fewer bits actually stored on disk. correcting." );
              i->GetPixelFormat().SetBitsStored( cpf.GetBitsStored() );
              }
            }
          }
        }

#endif
    return r;
    }
  return false;
}
#endif

#if defined(GDCM_USE_OPENJPEG)
bool Bitmap::TryJPEG2000Codec2(std::ostream &os) const
{
  unsigned long len = GetBufferLength();
  (void)len;
  const TransferSyntax &ts = GetTransferSyntax();

  JPEG2000Codec codec;
  if( codec.CanCode( ts ) )
    {
    codec.SetDimensions( GetDimensions() );
    codec.SetPixelFormat( GetPixelFormat() );
    codec.SetNumberOfDimensions( GetNumberOfDimensions() );
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    DataElement out;
    bool r = codec.Code(PixelData, out);
    assert( r );
    const ByteValue *outbv = out.GetByteValue();
    assert( outbv );
    unsigned long check = outbv->GetLength();  // FIXME
    (void)check;
    //memcpy(buffer, outbv->GetPointer(), outbv->GetLength() );  // FIXME
    os.write(outbv->GetPointer(), outbv->GetLength() );
    return r;
    }
  return false;
}
#endif

bool Bitmap::TryRLECodec(char *buffer, bool &lossyflag ) const
{
  unsigned long len = GetBufferLength();
  const TransferSyntax &ts = GetTransferSyntax();

  RLECodec codec;
  if( codec.CanDecode( ts ) )
    {
    //assert( sf->GetNumberOfFragments() == 1 );
    //assert( sf->GetNumberOfFragments() == GetDimensions(2) );
    codec.SetDimensions( GetDimensions() );
    codec.SetNumberOfDimensions( GetNumberOfDimensions() );
    codec.SetPlanarConfiguration( GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( GetPhotometricInterpretation() );
    codec.SetPixelFormat( GetPixelFormat() );
    codec.SetLUT( GetLUT() );
    codec.SetNeedOverlayCleanup( AreOverlaysInPixelData() );
    codec.SetBufferLength( len );
    DataElement out;
    bool r = codec.Decode(PixelData, out);
    if( !r ) return false;
    const ByteValue *outbv = out.GetByteValue();
    //unsigned long check = outbv->GetLength();  // FIXME
    // DermaColorLossLess.dcm has a len of 63531, but DICOM will give us: 63532 ...
    assert( len <= outbv->GetLength() );
    if(buffer) memcpy(buffer, outbv->GetPointer(), len /*outbv->GetLength()*/ );  // FIXME
    lossyflag = false;
    return true;
    }
  return false;
}

// Acces the raw data
bool Bitmap::GetBuffer(char *buffer) const
{
  bool dummy;
  return GetBufferInternal(buffer, dummy);
}

bool Bitmap::GetBufferInternal(char *buffer, bool &lossyflag) const
{
  bool success = false;
  if( !success ) success = TryRAWCodec(buffer, lossyflag);
  if( !success ) success = TryJPEGCodec(buffer, lossyflag);
  if( !success ) success = TryPVRGCodec(buffer, lossyflag); // AFTER IJG trial !
  //if( !success ) success = TryKAKADUCodec(buffer, lossyflag);
#if defined(GDCM_USE_OPENJPEG)
  if( !success ) success = TryJPEG2000Codec(buffer, lossyflag);
#endif
  if( !success ) success = TryJPEGLSCodec(buffer, lossyflag);
  if( !success ) success = TryRLECodec(buffer, lossyflag);
  //if( !success ) success = TryDeltaEncodingCodec(buffer);
  if( !success )
    {
    buffer = 0;
    //throw Exception( "No codec found for this image");
    }

  return success;
}

// Compress the raw data
bool Bitmap::GetBuffer2(std::ostream &os) const
{
  bool success = false;
  //if( !success ) success = TryRAWCodec2(buffer);
  if( !success ) success = TryJPEGCodec2(os);
  //if( !success ) success = TryJPEG2000Codec2(os);
  //if( !success ) success = TryRLECodec2(buffer);
  if( !success )
    {
    //buffer = 0;
    throw Exception( "No codec found for this image");
    }

  return success;
}

bool Bitmap::IsTransferSyntaxCompatible( TransferSyntax const & ts ) const
{
  if( GetTransferSyntax() == ts ) return true;
  // Special cases:
  if( GetTransferSyntax() == TransferSyntax::JPEGExtendedProcess2_4 )
    {
    if( GetPixelFormat().GetBitsAllocated() == 8 )
      {
      if( ts == TransferSyntax::JPEGBaselineProcess1 ) return true;
      }
    }
  // default:
  return false;
}

void Bitmap::Print(std::ostream &os) const
{
  Object::Print(os);
  //assert( NumberOfDimensions );
  if( !IsEmpty() )
    {
    os << "NumberOfDimensions: " << NumberOfDimensions << "\n";
    assert( Dimensions.size() );
    os << "Dimensions: (";
    std::vector<unsigned int>::const_iterator it = Dimensions.begin();
    os << *it;
    for(++it; it != Dimensions.end(); ++it)
      {
      os << "," << *it;
      }
    os << ")\n";
    PF.Print(os);
    os << "PhotometricInterpretation: " << PI << "\n";
    os << "PlanarConfiguration: " << PlanarConfiguration << "\n";
    os << "TransferSyntax: " << TS << "\n";
    }
}

}
