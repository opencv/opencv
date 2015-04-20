/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmJPEGCodec.h"
#include "gdcmTransferSyntax.h"
#include "gdcmTrace.h"
#include "gdcmDataElement.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSwapper.h"
#include "gdcmJPEG8Codec.h"
#include "gdcmJPEG12Codec.h"
#include "gdcmJPEG16Codec.h"

#include <numeric>
#include <string.h>

namespace gdcm
{

JPEGCodec::JPEGCodec():BitSample(0)/*,Lossless(true)*/,Quality(100)
{
  Internal = NULL;
}

JPEGCodec::~JPEGCodec()
{
  delete Internal;
}

void JPEGCodec::SetQuality(double q)
{
  Quality = (int)q;//not sure why a double is passed and stored in an int.
  //the casting will happen here anyway, so making it explicit removes a warning.
}

double JPEGCodec::GetQuality() const
{
  return Quality;
}

void JPEGCodec::SetLossless(bool l)
{
  LossyFlag = !l;
}

bool JPEGCodec::GetLossless() const
{
  return !LossyFlag;
}

bool JPEGCodec::CanDecode(TransferSyntax const &ts) const
{
  return ts == TransferSyntax::JPEGBaselineProcess1
      || ts == TransferSyntax::JPEGExtendedProcess2_4
      || ts == TransferSyntax::JPEGExtendedProcess3_5
      || ts == TransferSyntax::JPEGSpectralSelectionProcess6_8
      || ts == TransferSyntax::JPEGFullProgressionProcess10_12
      || ts == TransferSyntax::JPEGLosslessProcess14
      || ts == TransferSyntax::JPEGLosslessProcess14_1;
}

bool JPEGCodec::CanCode(TransferSyntax const &ts) const
{
  return ts == TransferSyntax::JPEGBaselineProcess1
      || ts == TransferSyntax::JPEGExtendedProcess2_4
      || ts == TransferSyntax::JPEGExtendedProcess3_5
      || ts == TransferSyntax::JPEGSpectralSelectionProcess6_8
      || ts == TransferSyntax::JPEGFullProgressionProcess10_12
      || ts == TransferSyntax::JPEGLosslessProcess14
      || ts == TransferSyntax::JPEGLosslessProcess14_1;
}

void JPEGCodec::SetPixelFormat(PixelFormat const &pt)
{
  ImageCodec::SetPixelFormat(pt);
  // Here is the deal: D_CLUNIE_RG3_JPLY.dcm is a 12Bits Stored / 16 Bits Allocated image
  // the jpeg encapsulated is: a 12 Sample Precision
  // so far so good.
  // So what if we are dealing with image such as: SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm
  // which is also a 12Bits Stored / 16 Bits Allocated image
  // however the jpeg encapsulated is now a 16 Sample Precision

  // We have the choice to decide to use Bits Stored or Bits Allocated, however in the case of
  // such an image as: gdcmData/MR16BitsAllocated_8BitsStored.dcm we are required to use
  // bits allocated to deal with the logic to decide withe the encoder
  SetBitSample( pt.GetBitsAllocated() );
  //SetBitSample( pt.GetBitsStored() );
}

void JPEGCodec::SetupJPEGBitCodec(int bit)
{
  BitSample = bit;
  delete Internal; Internal = NULL;
  assert( Internal == NULL );
  // what should I do with those single bit images ?
  if ( BitSample <= 8 )
    {
    gdcmDebugMacro( "Using JPEG8" );
    Internal = new JPEG8Codec;
    }
  else if ( /*BitSample > 8 &&*/ BitSample <= 12 )
    {
    gdcmDebugMacro( "Using JPEG12" );
    Internal = new JPEG12Codec;
    }
  else if ( /*BitSample > 12 &&*/ BitSample <= 16 )
    {
    gdcmDebugMacro( "Using JPEG16" );
    Internal = new JPEG16Codec;
    }
  else
    {
    // gdcmNonImageData/RT/RTDOSE.dcm
    gdcmWarningMacro( "Cannot instantiate JPEG codec for bit sample: " << bit );
    // Clearly make sure Internal will not be used
    delete Internal;
    Internal = NULL;
    }
}

void JPEGCodec::SetBitSample(int bit)
{
  SetupJPEGBitCodec(bit);
  if( Internal )
    {
    Internal->SetDimensions( this->GetDimensions() );
    Internal->SetPlanarConfiguration( this->GetPlanarConfiguration() );
    Internal->SetPhotometricInterpretation( this->GetPhotometricInterpretation() );
    Internal->SetLossless( this->GetLossless() );
    Internal->SetQuality( this->GetQuality() );
    Internal->ImageCodec::SetPixelFormat( this->ImageCodec::GetPixelFormat() );
    //Internal->SetNeedOverlayCleanup( this->AreOverlaysInPixelData() );
    }
}

/*
A.4.1 JPEG image compression
For all images, including all frames of a multi-frame image, the JPEG
Interchange Format shall be used (the table specification shall be included).
*/
bool JPEGCodec::Decode(DataElement const &in, DataElement &out)
{
  assert( Internal );
  out = in;
  // Fragments...
  const SequenceOfFragments *sf0 = in.GetSequenceOfFragments();
  const ByteValue *jpegbv = in.GetByteValue();
  if( !sf0 && !jpegbv ) return false;
  std::stringstream os;
  if( sf0 )
    {
    for(unsigned int i = 0; i < sf0->GetNumberOfFragments(); ++i)
      {
      std::stringstream is;
      const Fragment &frag = sf0->GetFragment(i);
      if( frag.IsEmpty() ) return false;
      const ByteValue &bv = dynamic_cast<const ByteValue&>(frag.GetValue());
      size_t bv_len = bv.GetLength();
      char *mybuffer = new char[bv_len];
      bool b = bv.GetBuffer(mybuffer, bv.GetLength());
      assert( b ); (void)b;
      is.write(mybuffer, bv.GetLength());
      delete[] mybuffer;
      bool r = DecodeByStreams(is, os);
      // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
      if( !r )
        {
        gdcmDebugMacro( "Failed to decompress Frag #" << i );
        const bool suspended = Internal->IsStateSuspension();
        const size_t nfrags = sf0->GetNumberOfFragments();
        // In case of chunked-jpeg, this is always an error
        if( suspended )
          return false;
        // Ok so we are decoding a multiple frame jpeg DICOM file:
        // if we are lucky, we might be trying to decode some sort of broken multi-frame
        // DICOM file. In this case check that we have read all Fragment properly:
        if( i >= this->GetDimensions()[2] )
          {
          // JPEGInvalidSecondFrag.dcm
          assert( nfrags == this->GetNumberOfDimensions() ); (void)nfrags; // sentinel
          gdcmWarningMacro( "Invalid JPEG Fragment found at pos #" << i + 1 << ". Skipping it" );
          }
        else
          return false;
        }
      }
    }
  else if ( jpegbv )
    {
    // GEIIS Icon:
    std::stringstream is0;
    size_t jpegbv_len = jpegbv->GetLength();
    char *mybuffer0 = new char[jpegbv_len];
    bool b0 = jpegbv->GetBuffer(mybuffer0, jpegbv->GetLength());
    assert( b0 ); (void)b0;
    is0.write(mybuffer0, jpegbv->GetLength());
    delete[] mybuffer0;
    bool r = DecodeByStreams(is0, os);
    if( !r )
      {
      // let's try another time:
      // JPEGDefinedLengthSequenceOfFragments.dcm
      is0.seekg(0);
      SequenceOfFragments sf_bug;
      try {
        sf_bug.Read<SwapperNoOp>(is0,true);
      } catch ( ... ) {
        return false;
      }

      const SequenceOfFragments *sf = &sf_bug;
      for(unsigned int i = 0; i < sf->GetNumberOfFragments(); ++i)
        {
        std::stringstream is;
        const Fragment &frag = sf->GetFragment(i);
        if( frag.IsEmpty() ) return false;
        const ByteValue &bv = dynamic_cast<const ByteValue&>(frag.GetValue());
        size_t bv_len = bv.GetLength();
        char *mybuffer = new char[bv_len];
        bool b = bv.GetBuffer(mybuffer, bv.GetLength());
        assert( b ); (void)b;
        is.write(mybuffer, bv.GetLength());
        delete[] mybuffer;
        bool r2 = DecodeByStreams(is, os);
        if( !r2 )
          {
          return false;
          }
        }

      }
    }
  //assert( pos == len );
  const size_t sizeOfOs = os.tellp();
  os.seekp( 0, std::ios::beg );
  ByteValue * bv = new ByteValue;
  bv->SetLength( (uint32_t)sizeOfOs );
  bv->Read<SwapperNoOp>( os );
  out.SetValue( *bv );

  return true;
}

void JPEGCodec::ComputeOffsetTable(bool b)
{
  (void)b;
  // Not implemented
  assert(0);
}

bool JPEGCodec::GetHeaderInfo( std::istream & is, TransferSyntax &ts )
{
  assert( Internal );
  if ( !Internal->GetHeaderInfo(is, ts) )
    {
    // let's check if this is one of those buggy lossless JPEG
    if( this->BitSample != Internal->BitSample )
      {
      // MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm
      // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
      gdcmWarningMacro( "DICOM header said it was " << this->BitSample <<
        " but JPEG header says it's: " << Internal->BitSample );
      if( this->BitSample < Internal->BitSample )
        {
        //assert(0); // Outside buffer will be too small
        }
      is.seekg(0, std::ios::beg);
      SetupJPEGBitCodec( Internal->BitSample );
      if( Internal && Internal->GetHeaderInfo(is, ts) )
        {
        // Foward everything back to meta jpeg codec:
        this->SetLossyFlag( Internal->GetLossyFlag() );
        this->SetDimensions( Internal->GetDimensions() );
        this->SetPhotometricInterpretation( Internal->GetPhotometricInterpretation() );
        int prep = this->GetPixelFormat().GetPixelRepresentation();
        this->PF = Internal->GetPixelFormat(); // DO NOT CALL SetPixelFormat
        this->PF.SetPixelRepresentation( (uint16_t)prep );
        return true;
        }
      else
        {
        //assert(0); // FATAL ERROR
        gdcmErrorMacro( "Do not support this JPEG Type" );
        return false;
        }
      }
    return false;
    }
  // else
  // Foward everything back to meta jpeg codec:
  this->SetLossyFlag( Internal->GetLossyFlag() );
  this->SetDimensions( Internal->GetDimensions() );
  this->SetPhotometricInterpretation( Internal->GetPhotometricInterpretation() );
  this->PF = Internal->GetPixelFormat(); // DO NOT CALL SetPixelFormat

  if( this->PI != Internal->PI )
    {
    gdcmWarningMacro( "PhotometricInterpretation issue" );
    this->PI = Internal->PI;
    }

  return true;
}

bool JPEGCodec::Code(DataElement const &in, DataElement &out)
{
  out = in;

  // Create a Sequence Of Fragments:
  SmartPointer<SequenceOfFragments> sq = new SequenceOfFragments;
  const Tag itemStart(0xfffe, 0xe000);
  //sq->GetTable().SetTag( itemStart );
  //const char dummy[4] = {};
  //sq->GetTable().SetByteValue( dummy, sizeof(dummy) );

  const ByteValue *bv = in.GetByteValue();
  const unsigned int *dims = this->GetDimensions();
  const char *input = bv->GetPointer();
  unsigned long len = bv->GetLength();
  unsigned long image_len = len / dims[2];
  if(!Internal) return false;

  // forward parameter to low level bits implementation (8/12/16)
  Internal->SetLossless( this->GetLossless() );
  Internal->SetQuality( this->GetQuality() );

  for(unsigned int dim = 0; dim < dims[2]; ++dim)
    {
    std::stringstream os;
    const char *p = input + dim * image_len;
    bool r = Internal->InternalCode(p, image_len, os);
    if( !r )
      {
      return false;
      }

    std::string str = os.str();
    assert( str.size() );
    Fragment frag;
    //frag.SetTag( itemStart );
    VL::Type strSize = (VL::Type)str.size();
    frag.SetByteValue( &str[0], strSize );
    sq->AddFragment( frag );

    }
  //unsigned int n = sq->GetNumberOfFragments();
  assert( sq->GetNumberOfFragments() == dims[2] );
  out.SetValue( *sq );

  return true;
}


bool JPEGCodec::DecodeByStreams(std::istream &is, std::ostream &os)
{
  std::stringstream tmpos;
  if ( !Internal->DecodeByStreams(is,tmpos) )
    {
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    // let's check if this is one of those buggy lossless JPEG
    if( this->BitSample != Internal->BitSample )
      {
      // MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm
      // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
      gdcmWarningMacro( "DICOM header said it was " << this->BitSample <<
        " but JPEG header says it's: " << Internal->BitSample );
      if( this->BitSample < Internal->BitSample )
        {
        //assert(0); // Outside buffer will be too small
        }
      is.seekg(0, std::ios::beg);
      SetupJPEGBitCodec( Internal->BitSample );
      if( Internal )
        {
        //Internal->SetPixelFormat( this->GetPixelFormat() ); // FIXME
        Internal->SetDimensions( this->GetDimensions() );
        Internal->SetPlanarConfiguration( this->GetPlanarConfiguration() ); // meaningless ?
        Internal->SetPhotometricInterpretation( this->GetPhotometricInterpretation() );
        if( Internal->DecodeByStreams(is,tmpos) )
          {
          return ImageCodec::DecodeByStreams(tmpos,os);
          }
        else
          {
          gdcmErrorMacro( "Could not succeed after 2 tries" );
          }
        }
      }
#endif
    return false;
    }
  if( this->PlanarConfiguration != Internal->PlanarConfiguration )
    {
    gdcmWarningMacro( "PlanarConfiguration issue" );
    this->PlanarConfiguration = Internal->PlanarConfiguration;
    //this->RequestPlanarConfiguration = true;
    }
  if( this->PI != Internal->PI )
    {
    gdcmWarningMacro( "PhotometricInterpretation issue" );
    this->PI = Internal->PI;
    }
  if( this->PF == PixelFormat::UINT12
   || this->PF == PixelFormat::INT12 )
    {
    this->PF.SetBitsAllocated( 16 );
    }

  return ImageCodec::DecodeByStreams(tmpos,os);
}

bool JPEGCodec::IsValid(PhotometricInterpretation const &pi)
{
  bool ret = false;
  switch( pi )
    {
    // JPEGCodec can produce the following PhotometricInterpretation as output:
    case PhotometricInterpretation::MONOCHROME1:
    case PhotometricInterpretation::MONOCHROME2:
    case PhotometricInterpretation::PALETTE_COLOR:
    case PhotometricInterpretation::RGB:
    case PhotometricInterpretation::YBR_FULL:
    case PhotometricInterpretation::YBR_FULL_422:
    case PhotometricInterpretation::YBR_PARTIAL_422:
    case PhotometricInterpretation::YBR_PARTIAL_420:
      ret = true;
      break;
    default:
      ;
//    case HSV:
//    case ARGB: // retired
//    case CMYK:
//    case YBR_RCT:
//    case YBR_ICT:
//      ret = false;
    }
  return ret;
}

bool JPEGCodec::DecodeExtent(
    char *buffer,
    unsigned int xmin, unsigned int xmax,
    unsigned int ymin, unsigned int ymax,
    unsigned int zmin, unsigned int zmax,
    std::istream & is
  )
{
  BasicOffsetTable bot;
  bot.Read<SwapperNoOp>( is );

  const unsigned int * dimensions = this->GetDimensions();
  const PixelFormat & pf = this->GetPixelFormat();
  //assert( pf.GetBitsAllocated() % 8 == 0 );
  assert( pf != PixelFormat::SINGLEBIT );
  //assert( pf != PixelFormat::UINT12 && pf != PixelFormat::INT12 );

  if( NumberOfDimensions == 2 )
    {
    //char *dummy_buffer = NULL;
    std::vector<char> vdummybuffer;
    size_t buf_size = 0;

    const Tag seqDelItem(0xfffe,0xe0dd);
    Fragment frag;
    unsigned int nfrags = 0;
    try
      {
      while( frag.ReadPreValue<SwapperNoOp>(is) && frag.GetTag() != seqDelItem )
        {
        ++nfrags;
        size_t fraglen = frag.GetVL();
        size_t oldlen = vdummybuffer.size();
        // update
        buf_size = fraglen + oldlen;
        vdummybuffer.resize( buf_size );
        //dummy_buffer = &vdummybuffer[0];
        // read J2K
        is.read( &vdummybuffer[oldlen], fraglen );
        }
    assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
      }
    catch(Exception &ex)
      {
      (void)ex;
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
      // that's ok ! In all cases the whole file was read, because
      // Fragment::Read only fail on eof() reached 1.
      // SIEMENS-JPEG-CorruptFrag.dcm is more difficult to deal with, we have a
      // partial fragment, read we decide to add it anyway to the stack of
      // fragments (eof was reached so we need to clear error bit)
      if( frag.GetTag() == Tag(0xfffe,0xe000)  )
        {
        gdcmWarningMacro( "Pixel Data Fragment could be corrupted. Use file at own risk" );
        //Fragments.push_back( frag );
        is.clear(); // clear the error bit
        }
      // 2. GENESIS_SIGNA-JPEG-CorruptFrag.dcm
      else if ( frag.GetTag() == Tag(0xddff,0x00e0) )
        {
        assert( nfrags == 1 );
        const ByteValue *bv = frag.GetByteValue();
        assert( (unsigned char)bv->GetPointer()[ bv->GetLength() - 1 ] == 0xfe );
        // Yes this is an extra copy, this is a bug anyway, go fix YOUR code
        frag.SetByteValue( bv->GetPointer(), bv->GetLength() - 1 );
        assert( 0 );
        gdcmWarningMacro( "JPEG Fragment length was declared with an extra byte"
          " at the end: stripped !" );
        is.clear(); // clear the error bit
        }
      else
        {
        // 3. gdcm-JPEG-LossLess3a.dcm: easy case, an extra tag was found
        // instead of terminator (eof is the next char)
        gdcmWarningMacro( "Reading failed at Tag:" << frag.GetTag() <<
          ". Use file at own risk." << ex.what() );
        }
#endif /* GDCM_SUPPORT_BROKEN_IMPLEMENTATION */
      }

    assert( zmin == zmax );
    assert( zmin == 0 );

    std::stringstream iis;
    iis.write( &vdummybuffer[0], vdummybuffer.size() );
    std::stringstream os;
    bool b = DecodeByStreams(iis,os);
    if(!b) return false;
    assert( b );

    const unsigned int rowsize = xmax - xmin + 1;
    const unsigned int colsize = ymax - ymin + 1;
    const unsigned int bytesPerPixel = pf.GetPixelSize();
    os.seekg(0, std::ios::beg );
    assert( os.good() );
    std::istream *theStream = &os;
    std::vector<char> buffer1;
    buffer1.resize( rowsize*bytesPerPixel );
    char *tmpBuffer1 = &buffer1[0];
    unsigned int y, z;
    std::streamoff theOffset;
    for (z = zmin; z <= zmax; ++z)
      {
      for (y = ymin; y <= ymax; ++y)
        {
        theStream->seekg(std::ios::beg);
        theOffset = 0 + (z*dimensions[1]*dimensions[0] + y*dimensions[0] + xmin)*bytesPerPixel;
        theStream->seekg(theOffset);
        theStream->read(tmpBuffer1, rowsize*bytesPerPixel);
        memcpy(&(buffer[((z-zmin)*rowsize*colsize +
              (y-ymin)*rowsize)*bytesPerPixel]),
          tmpBuffer1, rowsize*bytesPerPixel);
        }
      }
    }
  else if ( NumberOfDimensions == 3 )
    {
    const Tag seqDelItem(0xfffe,0xe0dd);
    Fragment frag;
    std::streamoff thestart = is.tellg();
    unsigned int numfrags = 0;
    std::vector< size_t > offsets;
    while( frag.ReadPreValue<SwapperNoOp>(is) && frag.GetTag() != seqDelItem )
      {
      //std::streamoff relstart = is.tellg();
      //assert( relstart - thestart == 8 );
      std::streamoff off = frag.GetVL();
      offsets.push_back( off );
      is.seekg( off, std::ios::cur );
      ++numfrags;
      }
    assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
    assert( numfrags == offsets.size() );
    if( numfrags != Dimensions[2] )
      {
      gdcmErrorMacro( "Not handled" );
      return false;
      }

    for( unsigned int z = zmin; z <= zmax; ++z )
      {
      size_t curoffset = std::accumulate( offsets.begin(), offsets.begin() + z, 0 );
      is.seekg( thestart + curoffset + 8 * z, std::ios::beg );
      is.seekg( 8, std::ios::cur );

      //const size_t buf_size = offsets[z];
      //char *dummy_buffer = new char[ buf_size ];
      //is.read( dummy_buffer, buf_size );

      std::stringstream os;
      const bool b = DecodeByStreams(is, os); (void)b;
      assert( b );
      /* free the memory containing the code-stream */
      //delete[] dummy_buffer;

      os.seekg(0, std::ios::beg );
      assert( os.good() );
      std::istream *theStream = &os;

      unsigned int rowsize = xmax - xmin + 1;
      unsigned int colsize = ymax - ymin + 1;
      unsigned int bytesPerPixel = pf.GetPixelSize();

      std::vector<char> buffer1;
      buffer1.resize( rowsize*bytesPerPixel );
      char *tmpBuffer1 = &buffer1[0];
      unsigned int y;
      std::streamoff theOffset;
      for (y = ymin; y <= ymax; ++y)
        {
        theStream->seekg(std::ios::beg);
        theOffset = 0 + (0*dimensions[1]*dimensions[0] + y*dimensions[0] + xmin)*bytesPerPixel;
        theStream->seekg(theOffset);
        theStream->read(tmpBuffer1, rowsize*bytesPerPixel);
        memcpy(&(buffer[((z-zmin)*rowsize*colsize +
              (y-ymin)*rowsize)*bytesPerPixel]),
          tmpBuffer1, rowsize*bytesPerPixel);
        }
      }
    }
  return true;
}

bool JPEGCodec::IsStateSuspension() const
{
  assert( 0 );
  return false;
}

ImageCodec * JPEGCodec::Clone() const
{
  JPEGCodec *copy = new JPEGCodec;
  ImageCodec &ic = *copy;
  ic = *this;
  assert( copy->PF == PF );
  //copy->SetupJPEGBitCodec( BitSample );
  copy->SetPixelFormat( GetPixelFormat() );
  assert( copy->BitSample == BitSample || BitSample == 0 );
  //copy->Lossless = Lossless;
  copy->Quality = Quality;

  return copy;
}

bool JPEGCodec::EncodeBuffer( std::ostream & out,
    const char *inbuffer, size_t inlen)
{
  assert( Internal );
  return Internal->EncodeBuffer(out, inbuffer, inlen);
}

bool JPEGCodec::StartEncode( std::ostream & )
{
  return true;
}
bool JPEGCodec::IsRowEncoder()
{
  return true;
}
bool JPEGCodec::IsFrameEncoder()
{
  assert(0);
  return false;
}
bool JPEGCodec::AppendRowEncode( std::ostream & os, const char * data, size_t datalen)
{
  return EncodeBuffer(os, data, datalen );
}
// TODO: technically the frame encoder could use the row encoder when present
// this could reduce code duplication
bool JPEGCodec::AppendFrameEncode( std::ostream & , const char * , size_t )
{
  assert(0);
  return false;
}
bool JPEGCodec::StopEncode( std::ostream & )
{
  return true;
}

} // end namespace gdcm
