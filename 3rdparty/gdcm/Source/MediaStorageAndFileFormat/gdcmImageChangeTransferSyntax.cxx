/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageChangeTransferSyntax.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmFragment.h"
#include "gdcmPixmap.h"
#include "gdcmBitmap.h"
#include "gdcmRAWCodec.h"
#include "gdcmJPEGCodec.h"
#include "gdcmJPEGLSCodec.h"
#include "gdcmJPEG2000Codec.h"
#include "gdcmRLECodec.h"

namespace gdcm
{

/*
bool ImageChangeTransferSyntax::TryRAWCodecIcon(const DataElement &pixelde)
{
  unsigned long len = Input->GetIconImage().GetBufferLength();
  //assert( len == pixelde.GetByteValue()->GetLength() );
  const TransferSyntax &ts = GetTransferSyntax();

  RAWCodec codec;
  if( codec.CanCode( ts ) )
    {
    codec.SetDimensions( Input->GetIconImage().GetDimensions() );
    codec.SetPlanarConfiguration( Input->GetIconImage().GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( Input->GetIconImage().GetPhotometricInterpretation() );
    codec.SetPixelFormat( Input->GetIconImage().GetPixelFormat() );
    codec.SetNeedOverlayCleanup( Input->GetIconImage().AreOverlaysInPixelData() );
    DataElement out;
    //bool r = codec.Code(Input->GetDataElement(), out);
    bool r = codec.Code(pixelde, out);

    DataElement &de = Output->GetIconImage().GetDataElement();
    de.SetValue( out.GetValue() );
    if( !r )
      {
      return false;
      }
    return true;
    }
  return false;
}
*/

void UpdatePhotometricInterpretation( Bitmap const &input, Bitmap &output )
{
  // when decompressing J2K, need to revert to proper photo inter in uncompressed TS:
  if( input.GetPhotometricInterpretation() == PhotometricInterpretation::YBR_RCT
    || input.GetPhotometricInterpretation() == PhotometricInterpretation::YBR_ICT )
    {
    output.SetPhotometricInterpretation( PhotometricInterpretation::RGB );
    }
  // when decompressing loss jpeg, need to revert to proper photo inter in uncompressed TS:
  if( input.GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL_422 )
    {
    output.SetPhotometricInterpretation( PhotometricInterpretation::YBR_FULL );
    }
  assert( output.GetPhotometricInterpretation() == PhotometricInterpretation::RGB
    || output.GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL
    || output.GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME1
    || output.GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME2
    || output.GetPhotometricInterpretation() == PhotometricInterpretation::ARGB
    || output.GetPhotometricInterpretation() == PhotometricInterpretation::PALETTE_COLOR ); // programmer error
}

bool ImageChangeTransferSyntax::TryRAWCodec(const DataElement &pixelde, Bitmap const &input, Bitmap &output)
{
  unsigned long len = input.GetBufferLength(); (void)len;
  //assert( len == pixelde.GetByteValue()->GetLength() );
  const TransferSyntax &ts = GetTransferSyntax();

  RAWCodec codec;
  if( codec.CanCode( ts ) )
    {
    codec.SetDimensions( input.GetDimensions() );
    codec.SetPlanarConfiguration( input.GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( input.GetPhotometricInterpretation() );
    codec.SetPixelFormat( input.GetPixelFormat() );
    codec.SetNeedOverlayCleanup( input.AreOverlaysInPixelData() );
    DataElement out;
    //bool r = codec.Code(input.GetDataElement(), out);
    bool r = codec.Code(pixelde, out);

    if( !r )
      {
      return false;
      }
    DataElement &de = output.GetDataElement();
    de.SetValue( out.GetValue() );
    UpdatePhotometricInterpretation( input, output );
    return true;
    }
  return false;
}

bool ImageChangeTransferSyntax::TryRLECodec(const DataElement &pixelde, Bitmap const &input, Bitmap &output)
{
  unsigned long len = input.GetBufferLength(); (void)len;
  //assert( len == pixelde.GetByteValue()->GetLength() );
  const TransferSyntax &ts = GetTransferSyntax();

  RLECodec codec;
  if( codec.CanCode( ts ) )
    {
    codec.SetDimensions( input.GetDimensions() );
    codec.SetPlanarConfiguration( input.GetPlanarConfiguration() );
    codec.SetPhotometricInterpretation( input.GetPhotometricInterpretation() );
    codec.SetPixelFormat( input.GetPixelFormat() );
    codec.SetNeedOverlayCleanup( input.AreOverlaysInPixelData() );
    DataElement out;
    //bool r = codec.Code(input.GetDataElement(), out);
    bool r = codec.Code(pixelde, out);

    if( !r )
      {
      return false;
      }
    DataElement &de = output.GetDataElement();
    de.SetValue( out.GetValue() );
    UpdatePhotometricInterpretation( input, output );
    return true;
    }
  return false;
}

bool ImageChangeTransferSyntax::TryJPEGCodec(const DataElement &pixelde, Bitmap const &input, Bitmap &output)
{
  unsigned long len = input.GetBufferLength(); (void)len;
  //assert( len == pixelde.GetByteValue()->GetLength() );
  const TransferSyntax &ts = GetTransferSyntax();

  JPEGCodec jpgcodec;
  // pass lossy/lossless flag:
  // JPEGCodec are easier to deal with since there is no dual transfer syntax
  // that can be both lossy and lossless:
  if( ts.IsLossy() )
    {
    //assert( !ts.IsLossless() ); // I cannot do since since Try* functions are called with all TS, I could be receiving a JPEGLS TS...
    jpgcodec.SetLossless( false );
    }

  ImageCodec *codec = &jpgcodec;
  JPEGCodec *usercodec = dynamic_cast<JPEGCodec*>(UserCodec);
  if( usercodec && usercodec->CanCode( ts ) )
    {
    codec = usercodec;
    }

  if( codec->CanCode( ts ) )
    {
    codec->SetDimensions( input.GetDimensions() );
    // FIXME: GDCM always apply the planar configuration to 0...
    //if( input.GetPlanarConfiguration() )
    //  {
    //  output.SetPlanarConfiguration( 0 );
    //  }
    codec->SetPlanarConfiguration( input.GetPlanarConfiguration() );
    codec->SetPhotometricInterpretation( input.GetPhotometricInterpretation() );
    codec->SetPixelFormat( input.GetPixelFormat() );
    codec->SetNeedOverlayCleanup( input.AreOverlaysInPixelData() );
    // let's check we are not trying to compress 16bits with JPEG/Lossy/8bits
    if( !input.GetPixelFormat().IsCompatible( ts ) )
      {
      gdcmErrorMacro("Pixel Format incompatible with TS" );
      return false;
      }
    DataElement out;
    //bool r = codec.Code(input.GetDataElement(), out);
    bool r = codec->Code(pixelde, out);
    // FIXME: this is not the best place to change the Output image internal type,
    // but since I know IJG is always applying the Planar Configuration, it does make
    // any sense to EVER produce a JPEG image where the Planar Configuration would be one
    // so let's be nice and actually sync JPEG configuration with DICOM Planar Conf.
    output.SetPlanarConfiguration( 0 );
    //output.SetPhotometricInterpretation( PhotometricInterpretation::RGB );

    // Indeed one cannot produce a true lossless RGB image according to DICOM standard
    // when doing lossless jpeg:
    if( output.GetPhotometricInterpretation() == PhotometricInterpretation::RGB )
      {
      gdcmWarningMacro( "Technically this is not defined in the standard. \n"
        "Some validator may complains this image is invalid, but would be wrong.");
      }

    // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
    if( !r )
      {
      return false;
      }
    DataElement &de = output.GetDataElement();
    de.SetValue( out.GetValue() );
    UpdatePhotometricInterpretation( input, output );
    // When compressing with JPEG I think planar should always be:
    //output.SetPlanarConfiguration(0);
    // FIXME ! This should be done all the time for all codec:
    // Did PI change or not ?
    if ( !output.GetPhotometricInterpretation().IsSameColorSpace( codec->GetPhotometricInterpretation() ) )
      {
      // HACK
      //Image *i = (Image*)this;
      //i->SetPhotometricInterpretation( codec.GetPhotometricInterpretation() );
      assert(0);
      }
    return true;
    }
  return false;
}

bool ImageChangeTransferSyntax::TryJPEGLSCodec(const DataElement &pixelde, Bitmap const &input, Bitmap &output)
{
  unsigned long len = input.GetBufferLength(); (void)len;
  //assert( len == pixelde.GetByteValue()->GetLength() );
  const TransferSyntax &ts = GetTransferSyntax();

  JPEGLSCodec jlscodec;
  ImageCodec *codec = &jlscodec;
  JPEGLSCodec *usercodec = dynamic_cast<JPEGLSCodec*>(UserCodec);
  if( usercodec && usercodec->CanCode( ts ) )
    {
    codec = usercodec;
    }

  if( codec->CanCode( ts ) )
    {
    codec->SetDimensions( input.GetDimensions() );
    codec->SetPixelFormat( input.GetPixelFormat() );
    //codec.SetNumberOfDimensions( input.GetNumberOfDimensions() );
    codec->SetPlanarConfiguration( input.GetPlanarConfiguration() );
    codec->SetPhotometricInterpretation( input.GetPhotometricInterpretation() );
    codec->SetNeedOverlayCleanup( input.AreOverlaysInPixelData() );
    DataElement out;
    //bool r = codec.Code(input.GetDataElement(), out);
    bool r = codec->Code(pixelde, out);
    if(!r) return false;
    output.SetPlanarConfiguration( 0 );

    DataElement &de = output.GetDataElement();
    de.SetValue( out.GetValue() );
    UpdatePhotometricInterpretation( input, output );
    return r;
    }
  return false;
}

bool ImageChangeTransferSyntax::TryJPEG2000Codec(const DataElement &pixelde, Bitmap const &input, Bitmap &output)
{
  unsigned long len = input.GetBufferLength(); (void)len;
  //assert( len == pixelde.GetByteValue()->GetLength() );
  const TransferSyntax &ts = GetTransferSyntax();

  JPEG2000Codec j2kcodec;
  ImageCodec *codec = &j2kcodec;
  JPEG2000Codec *usercodec = dynamic_cast<JPEG2000Codec*>(UserCodec);
  if( usercodec && usercodec->CanCode( ts ) )
    {
    codec = usercodec;
    }

  if( codec->CanCode( ts ) )
    {
    codec->SetDimensions( input.GetDimensions() );
    codec->SetPixelFormat( input.GetPixelFormat() );
    codec->SetNumberOfDimensions( input.GetNumberOfDimensions() );
    codec->SetPlanarConfiguration( input.GetPlanarConfiguration() );
    codec->SetPhotometricInterpretation( input.GetPhotometricInterpretation() );
    codec->SetNeedOverlayCleanup( input.AreOverlaysInPixelData() );
    DataElement out;
    //bool r = codec.Code(input.GetDataElement(), out);
    bool r = codec->Code(pixelde, out);

    // The value of Planar Configuration (0028,0006) is irrelevant since the
    // manner of encoding components is specified in the JPEG 2000 standard,
    // hence it shall be set to 0.
    output.SetPlanarConfiguration( 0 );

    if( input.GetPixelFormat().GetSamplesPerPixel() == 3 )
      {
      if( input.GetPhotometricInterpretation().IsSameColorSpace( PhotometricInterpretation::RGB ) )
        {
        if( ts == TransferSyntax::JPEG2000Lossless )
          {
          output.SetPhotometricInterpretation( PhotometricInterpretation::YBR_RCT );
          }
        else
          {
          assert( ts == TransferSyntax::JPEG2000 );
          output.SetPhotometricInterpretation( PhotometricInterpretation::YBR_ICT );
          }
        }
      else
        {
        assert( input.GetPhotometricInterpretation().IsSameColorSpace( PhotometricInterpretation::YBR_FULL ) );
        if( ts == TransferSyntax::JPEG2000Lossless )
          {
          output.SetPhotometricInterpretation( PhotometricInterpretation::YBR_FULL );
          // Indeed one cannot produce a true lossless RGB image according to DICOM standard
          gdcmWarningMacro( "Technically this is not defined in the standard. \n"
            "Some validator may complains this image is invalid, but would be wrong.");
          }
        else
          {
          assert( ts == TransferSyntax::JPEG2000 );
          //output.SetPhotometricInterpretation( PhotometricInterpretation::YBR_ICT );
          // FIXME: technically when doing lossy we could be standard compliant and first convert to
          // RGB THEN compress to YBR_ICT. For now produce improper j2k image
          output.SetPhotometricInterpretation( PhotometricInterpretation::YBR_FULL );
          }
        }
      }
    else
      {
      assert( input.GetPixelFormat().GetSamplesPerPixel() == 1 );
      }

    if( !r ) return false;
    DataElement &de = output.GetDataElement();
    de.SetValue( out.GetValue() );
    UpdatePhotometricInterpretation( input, output );
    return r;
    }
  return false;
}

bool ImageChangeTransferSyntax::Change()
{
  if( TS == TransferSyntax::TS_END )
    {
    if( !Force ) return false;
    // When force option is set but no specific TransferSyntax has been set, only inspect the
    // encapsulated stream...
    // See ImageReader::Read
    if( Input->GetTransferSyntax().IsEncapsulated() && Input->GetTransferSyntax() != TransferSyntax::RLELossless )
      {
      Output = Input;
      return true;
      }
    return false;
    }
  // let's get rid of some easy case:
  if( Input->GetPhotometricInterpretation() == PhotometricInterpretation::PALETTE_COLOR &&
    TS.IsLossy() )
    {
    gdcmErrorMacro( "PALETTE_COLOR and Lossy compression are impossible. Convert to RGB first." );
    return false;
    }

  Output = Input;
//  if( TS.IsLossy() && !TS.IsLossless() )
//    Output->SetLossyFlag( true );

  // Fast path
  if( Input->GetTransferSyntax() == TS && !Force ) return true;

  // FIXME
  // For now only support raw input, otherwise we would need to first decompress them
  if( (Input->GetTransferSyntax() != TransferSyntax::ImplicitVRLittleEndian
    && Input->GetTransferSyntax() != TransferSyntax::ExplicitVRLittleEndian
    && Input->GetTransferSyntax() != TransferSyntax::ExplicitVRBigEndian)
    || Force )
    {
    // In memory decompression:
    DataElement pixeldata( Tag(0x7fe0,0x0010) );
    ByteValue *bv0 = new ByteValue();
    uint32_t len0 = (uint32_t)Input->GetBufferLength();
    bv0->SetLength( len0 );
    bool b = Input->GetBuffer( (char*)bv0->GetPointer() );
    if( !b )
      {
      gdcmErrorMacro( "Error in getting buffer from input image." );
      return false;
      }
    pixeldata.SetValue( *bv0 );

    bool success = false;
    if( !success ) success = TryRAWCodec(pixeldata, *Input, *Output);
    if( !success ) success = TryJPEGCodec(pixeldata, *Input, *Output);
    if( !success ) success = TryJPEGLSCodec(pixeldata, *Input, *Output);
    if( !success ) success = TryJPEG2000Codec(pixeldata, *Input, *Output);
    if( !success ) success = TryRLECodec(pixeldata, *Input, *Output);
    Output->SetTransferSyntax( TS );
    if( !success )
      {
      //assert(0);
      return false;
      }

    // same goes for icon
    DataElement iconpixeldata( Tag(0x7fe0,0x0010) );
    Bitmap &bitmap = *Input;
    if( Pixmap *pixmap = dynamic_cast<Pixmap*>( &bitmap ) )
      {
      Bitmap &outbitmap = *Output;
      Pixmap *outpixmap = dynamic_cast<Pixmap*>( &outbitmap );
      assert( outpixmap != NULL );
      if( !pixmap->GetIconImage().IsEmpty() )
        {
        // same goes for icon
        ByteValue *bv = new ByteValue();
        uint32_t len = (uint32_t)pixmap->GetIconImage().GetBufferLength();
        bv->SetLength( len );
        bool bb = pixmap->GetIconImage().GetBuffer( (char*)bv->GetPointer() );
        if( !bb )
          {
          return false;
          }
        iconpixeldata.SetValue( *bv );

        success = false;
        if( !success ) success = TryRAWCodec(iconpixeldata, pixmap->GetIconImage(), outpixmap->GetIconImage());
        if( !success ) success = TryJPEGCodec(iconpixeldata, pixmap->GetIconImage(), outpixmap->GetIconImage());
        if( !success ) success = TryJPEGLSCodec(iconpixeldata, pixmap->GetIconImage(), outpixmap->GetIconImage());
        if( !success ) success = TryJPEG2000Codec(iconpixeldata, pixmap->GetIconImage(), outpixmap->GetIconImage());
        if( !success ) success = TryRLECodec(iconpixeldata, pixmap->GetIconImage(), outpixmap->GetIconImage());
        outpixmap->GetIconImage().SetTransferSyntax( TS );
        if( !success )
          {
          //assert(0);
          return false;
          }
        assert( outpixmap->GetIconImage().GetTransferSyntax() == TS );
        }
      }

    //Output->ComputeLossyFlag();
    assert( Output->GetTransferSyntax() == TS );
    //if( TS.IsLossy() ) assert( Output->IsLossy() );
    return success;
    }

  // too bad we actually have to do some work...
  bool success = false;
  if( !success ) success = TryRAWCodec(Input->GetDataElement(), *Input, *Output);
  if( !success ) success = TryJPEGCodec(Input->GetDataElement(), *Input, *Output);
  if( !success ) success = TryJPEG2000Codec(Input->GetDataElement(), *Input, *Output);
  if( !success ) success = TryJPEGLSCodec(Input->GetDataElement(), *Input, *Output);
  if( !success ) success = TryRLECodec(Input->GetDataElement(), *Input, *Output);
  Output->SetTransferSyntax( TS );
  if( !success )
    {
    //assert(0);
    return false;
    }

  Bitmap &bitmap = *Input;
  if( Pixmap *pixmap = dynamic_cast<Pixmap*>( &bitmap ) )
    {
    if( !pixmap->GetIconImage().IsEmpty() && CompressIconImage )
      {
      Bitmap &outbitmap = *Output;
      Pixmap *outpixmap = dynamic_cast<Pixmap*>( &outbitmap );

      // same goes for icon
      success = false;
      if( !success ) success = TryRAWCodec(pixmap->GetIconImage().GetDataElement(), pixmap->GetIconImage(), outpixmap->GetIconImage());
      if( !success ) success = TryJPEGCodec(pixmap->GetIconImage().GetDataElement(), pixmap->GetIconImage(), outpixmap->GetIconImage());
      if( !success ) success = TryJPEGLSCodec(pixmap->GetIconImage().GetDataElement(), pixmap->GetIconImage(), outpixmap->GetIconImage());
      if( !success ) success = TryJPEG2000Codec(pixmap->GetIconImage().GetDataElement(), pixmap->GetIconImage(), outpixmap->GetIconImage());
      if( !success ) success = TryRLECodec(pixmap->GetIconImage().GetDataElement(), pixmap->GetIconImage(), outpixmap->GetIconImage());
      outpixmap->GetIconImage().SetTransferSyntax( TS );
      if( !success )
        {
        //assert(0);
        return false;
        }
      assert( outpixmap->GetIconImage().GetTransferSyntax() == TS );
      }
    }

  //Output->ComputeLossyFlag();

  assert( Output->GetTransferSyntax() == TS );
  return success;
}


} // end namespace gdcm
