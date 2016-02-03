/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmJPEGLSCodec.h"
#include "gdcmTransferSyntax.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmDataElement.h"
#include "gdcmSwapper.h"

#include <numeric>

// CharLS includes
#include "gdcm_charls.h"

namespace gdcm
{

JPEGLSCodec::JPEGLSCodec():BufferLength(0)/*,Lossless(true)*/,LossyError(0)
{
}

JPEGLSCodec::~JPEGLSCodec()
{
}

void JPEGLSCodec::SetLossless(bool l)
{
  LossyFlag = !l;
}

bool JPEGLSCodec::GetLossless() const
{
  return !LossyFlag;
}

bool JPEGLSCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
{
#ifndef GDCM_USE_JPEGLS
  return false;
#else
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  //assert(buf_size < INT_MAX);
  char *dummy_buffer = new char[(unsigned int)buf_size];
  is.seekg(0, std::ios::beg);
  is.read( dummy_buffer, buf_size);

  JlsParameters metadata = {};
  //assert(buf_size < INT_MAX);
  if (JpegLsReadHeader(dummy_buffer, (unsigned int)buf_size, &metadata) != OK)
    {
    return false;
    }
  delete[] dummy_buffer;

  // $1 = {width = 512, height = 512, bitspersample = 8, components = 1, allowedlossyerror = 0, ilv = ILV_NONE, colorTransform = 0, custom = {MAXVAL = 0, T1 = 0, T2 = 0, T3 = 0, RESET = 0}}

  this->Dimensions[0] = metadata.width;
  this->Dimensions[1] = metadata.height;
  if( metadata.bitspersample <= 8 )
    {
    this->PF = PixelFormat( PixelFormat::UINT8 );
    }
  else if( metadata.bitspersample <= 16 )
    {
    assert( metadata.bitspersample > 8 );
    this->PF = PixelFormat( PixelFormat::UINT16 );
    }
  else
    {
    assert(0);
    }
  this->PF.SetBitsStored( (uint16_t)metadata.bitspersample );
  assert( this->PF.IsValid() );
//  switch( metadata.bitspersample )
//    {
//  case 8:
//    this->PF = PixelFormat( PixelFormat::UINT8 );
//    break;
//  case 12:
//    this->PF = PixelFormat( PixelFormat::UINT16 );
//    this->PF.SetBitsStored( 12 );
//    break;
//  case 16:
//    this->PF = PixelFormat( PixelFormat::UINT16 );
//    break;
//  default:
//    assert(0);
//    }
  if( metadata.components == 1 )
    {
    PI = PhotometricInterpretation::MONOCHROME2;
    this->PF.SetSamplesPerPixel( 1 );
    }
  else if( metadata.components == 3 )
    {
    PI = PhotometricInterpretation::RGB;
    this->PF.SetSamplesPerPixel( 3 );
    }
  else assert(0);

  // allowedlossyerror == 0 => Lossless
  LossyFlag = metadata.allowedlossyerror != 0;

  if( metadata.allowedlossyerror == 0 )
    {
    ts = TransferSyntax::JPEGLSLossless;
    }
  else
    {
    ts = TransferSyntax::JPEGLSNearLossless;
    }

  return true;
#endif
}

bool JPEGLSCodec::CanDecode(TransferSyntax const &ts) const
{
#ifndef GDCM_USE_JPEGLS
  return false;
#else
  return ts == TransferSyntax::JPEGLSLossless
      || ts == TransferSyntax::JPEGLSNearLossless;
#endif
}

bool JPEGLSCodec::CanCode(TransferSyntax const &ts) const
{
#ifndef GDCM_USE_JPEGLS
  return false;
#else
  return ts == TransferSyntax::JPEGLSLossless
      || ts == TransferSyntax::JPEGLSNearLossless;
#endif
}

bool JPEGLSCodec::DecodeByStreamsCommon(char *buffer, size_t totalLen, std::vector<unsigned char> &rgbyteOut)
{
  const BYTE* pbyteCompressed = (const BYTE*)buffer;
  size_t cbyteCompressed = totalLen;

  JlsParameters params = {};
  if(JpegLsReadHeader(pbyteCompressed, cbyteCompressed, &params) != OK )
    {
    gdcmDebugMacro( "Could not parse JPEG-LS header" );
    return false;
    }

  // allowedlossyerror == 0 => Lossless
  LossyFlag = params.allowedlossyerror!= 0;

  rgbyteOut.resize(params.height *params.width * ((params.bitspersample + 7) / 8) * params.components);

  JLS_ERROR result = JpegLsDecode(&rgbyteOut[0], rgbyteOut.size(), pbyteCompressed, cbyteCompressed, &params);

  if (result != OK)
    {
    return false;
    }

  return true;
}

bool JPEGLSCodec::Decode(DataElement const &in, DataElement &out)
{
#ifndef GDCM_USE_JPEGLS
  return false;
#else
  if( NumberOfDimensions == 2 )
    {
    const SequenceOfFragments *sf = in.GetSequenceOfFragments();
    assert( sf );
    unsigned long totalLen = sf->ComputeByteLength();
    char *buffer = new char[totalLen];
    sf->GetBuffer(buffer, totalLen);

    std::vector<BYTE> rgbyteOut;
    bool b = DecodeByStreamsCommon(buffer, totalLen, rgbyteOut);
    if( !b ) return false;
    delete[] buffer;

    out = in;

    out.SetByteValue( (char*)&rgbyteOut[0], (uint32_t)rgbyteOut.size() );
    return true;
    }
  else if( NumberOfDimensions == 3 )
    {
    const SequenceOfFragments *sf = in.GetSequenceOfFragments();
    assert( sf );
    gdcmAssertAlwaysMacro( sf->GetNumberOfFragments() == Dimensions[2] );
    std::stringstream os;
    for(unsigned int i = 0; i < sf->GetNumberOfFragments(); ++i)
      {
      const Fragment &frag = sf->GetFragment(i);
      if( frag.IsEmpty() ) return false;
      const ByteValue *bv = frag.GetByteValue();
      assert( bv );
      size_t totalLen = bv->GetLength();
      char *mybuffer = new char[totalLen];

      bv->GetBuffer(mybuffer, bv->GetLength());

      const BYTE* pbyteCompressed = (const BYTE*)mybuffer;
      while( totalLen > 0 && pbyteCompressed[totalLen-1] != 0xd9 )
        {
        totalLen--;
        }
      // what if 0xd9 is never found ?
      assert( totalLen > 0 && pbyteCompressed[totalLen-1] == 0xd9 );

      size_t cbyteCompressed = totalLen;

      JlsParameters params = {};
      if( JpegLsReadHeader(pbyteCompressed, cbyteCompressed, &params) != OK )
        {
        gdcmDebugMacro( "Could not parse JPEG-LS header" );
        return false;
        }

      // allowedlossyerror == 0 => Lossless
      LossyFlag = params.allowedlossyerror!= 0;

      std::vector<BYTE> rgbyteOut;
      rgbyteOut.resize(params.height *params.width * ((params.bitspersample + 7) / 8) * params.components);

      JLS_ERROR result = JpegLsDecode(&rgbyteOut[0], rgbyteOut.size(), pbyteCompressed, cbyteCompressed, &params);
      bool r = true;

      delete[] mybuffer;
      if (result != OK)
        {
        return false;
        }
      os.write( (char*)&rgbyteOut[0], rgbyteOut.size() );

      if(!r) return false;
      assert( r == true );
      }
    std::string str = os.str();
    assert( str.size() );
    out.SetByteValue( &str[0], (uint32_t)str.size() );

    return true;
    }
  return false;

#endif
}

bool JPEGLSCodec::CodeFrameIntoBuffer(char * outdata, size_t outlen, size_t & complen, const char * indata, size_t inlen )
{
#ifndef GDCM_USE_JPEGLS
  return false;
#else
  const unsigned int *dims = this->GetDimensions();
  int image_width = dims[0];
  int image_height = dims[1];

  const PixelFormat &pf = this->GetPixelFormat();
  int sample_pixel = pf.GetSamplesPerPixel();
  int bitsallocated = pf.GetBitsAllocated();
  int bitsstored = pf.GetBitsStored();

  JlsParameters params = {};
  /*
  The fields in JlsCustomParameters do not control lossy/lossless. They
  provide the possiblity to tune the JPEG-LS internals for better compression
  ratios. Expect a lot of work and testing to achieve small improvements.

  Lossy/lossless is controlled by the field allowedlossyerror. If you put in
  0, encoding is lossless. If it is non-zero, then encoding is lossy. The
  value of 3 is often suggested as a default.

  The nice part about JPEG-LS encoding is that in lossy encoding, there is a
  guarenteed maximum error for each pixel. So a pixel that has value 12,
  encoded with a maximum lossy error of 3, may be decoded as a value between 9
  and 15, but never anything else. In medical imaging this could be a useful
  guarantee.

  The not so nice part is that you may see striping artifacts when decoding
  "non-natural" images. I haven't seen the effects myself on medical images,
  but I suspect screenshots may suffer the most. Also, the bandwidth saving is
  not as big as with other lossy schemes.

  As for 12 bit, I am about to commit a unit test (with the sample you gave
  me) that does a successful round trip encoding of 12 bit color. I did notice
  that for 12 bit, the encoder fails if the unused bits are non-zero, but the
  sample dit not suffer from that.
   */
  params.allowedlossyerror = !LossyFlag ? 0 : LossyError;
  params.components = sample_pixel;
  // D_CLUNIE_RG3_JPLY.dcm. The famous 16bits allocated / 10 bits stored with the pixel value = 1024
  // CharLS properly encode 1024 considering it as 10bits data, so the output
  // Using bitsstored for the encoder gives a slightly better compression ratio, and is indeed the
  // right way of doing it.

  // gdcmData/PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm
  if( true || pf.GetPixelRepresentation() )
    {
    // gdcmData/CT_16b_signed-UsedBits13.dcm
    params.bitspersample = bitsallocated;
    }
  else
    {
    params.bitspersample = bitsstored;
    }
  params.height = image_height;
  params.width = image_width;

  if (sample_pixel == 4)
    {
    params.ilv = ILV_LINE;
    }
  else if (sample_pixel == 3)
    {
    params.ilv = ILV_LINE;
    params.colorTransform = COLORXFORM_HP1;
    }


  JLS_ERROR error = JpegLsEncode(outdata, outlen, &complen, indata, inlen, &params);
  if( error != OK )
    {
    gdcmErrorMacro( "Error compressing: " << (int)error );
    return false;
    }

  assert( complen < outlen );

  return true;
#endif
}

// Compress into JPEG
bool JPEGLSCodec::Code(DataElement const &in, DataElement &out)
{
#ifndef GDCM_USE_JPEGLS
  return false;
#else
  out = in;
  //
  // Create a Sequence Of Fragments:
  SmartPointer<SequenceOfFragments> sq = new SequenceOfFragments;

  const unsigned int *dims = this->GetDimensions();
  int image_width = dims[0];
  int image_height = dims[1];

  const ByteValue *bv = in.GetByteValue();
  const char *input = bv->GetPointer();
  unsigned long len = bv->GetLength();
  unsigned long image_len = len / dims[2];
  size_t inputlength = image_len;

  for(unsigned int dim = 0; dim < dims[2]; ++dim)
    {
    const char *inputdata = input + dim * image_len;

    std::vector<BYTE> rgbyteCompressed;
    rgbyteCompressed.resize(image_width * image_height * 4);

    size_t cbyteCompressed;
    const bool b = this->CodeFrameIntoBuffer((char*)&rgbyteCompressed[0], rgbyteCompressed.size(), cbyteCompressed, inputdata, inputlength );
    if( !b ) return false;

    Fragment frag;
    frag.SetByteValue( (char*)&rgbyteCompressed[0], (uint32_t)cbyteCompressed );
    sq->AddFragment( frag );
    }

  assert( sq->GetNumberOfFragments() == dims[2] );
  out.SetValue( *sq );

  return true;

#endif
}

void JPEGLSCodec::SetLossyError(int error)
{
  LossyError = error;
}

bool JPEGLSCodec::Decode(DataElement const &, char* , size_t,
              uint32_t , uint32_t , uint32_t ,
              uint32_t , uint32_t , uint32_t )
{
 return false;
}

bool JPEGLSCodec::DecodeExtent(
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
  assert( pf.GetBitsAllocated() % 8 == 0 );
  assert( pf != PixelFormat::SINGLEBIT );
  assert( pf != PixelFormat::UINT12 && pf != PixelFormat::INT12 );

  if( NumberOfDimensions == 2 )
    {
    char *dummy_buffer = NULL;
    std::vector<char> vdummybuffer;
    size_t buf_size = 0;

    const Tag seqDelItem(0xfffe,0xe0dd);
    Fragment frag;
    while( frag.ReadPreValue<SwapperNoOp>(is) && frag.GetTag() != seqDelItem )
      {
      size_t fraglen = frag.GetVL();
      size_t oldlen = vdummybuffer.size();
      // update
      buf_size = fraglen + oldlen;
      vdummybuffer.resize( buf_size );
      dummy_buffer = &vdummybuffer[0];
      // read J2K
      is.read( &vdummybuffer[oldlen], fraglen );
      }
    assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
    assert( zmin == zmax );
    assert( zmin == 0 );

    std::vector <unsigned char> outv;
    bool b = DecodeByStreamsCommon(dummy_buffer, buf_size, outv);
    if( !b ) return false;

    unsigned char *raw = &outv[0];
    const unsigned int rowsize = xmax - xmin + 1;
    const unsigned int colsize = ymax - ymin + 1;
    const unsigned int bytesPerPixel = pf.GetPixelSize();

    const unsigned char *tmpBuffer1 = raw;
    unsigned int z = 0;
    for (unsigned int y = ymin; y <= ymax; ++y)
      {
      size_t theOffset = 0 + (z*dimensions[1]*dimensions[0] + y*dimensions[0] + xmin)*bytesPerPixel;
      tmpBuffer1 = raw + theOffset;
      memcpy(&(buffer[((z-zmin)*rowsize*colsize +
            (y-ymin)*rowsize)*bytesPerPixel]),
        tmpBuffer1, rowsize*bytesPerPixel);
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

      const size_t buf_size = offsets[z];
      char *dummy_buffer = new char[ buf_size ];
      is.read( dummy_buffer, buf_size );

      std::vector <unsigned char> outv;
      bool b = DecodeByStreamsCommon(dummy_buffer, buf_size, outv);
      delete[] dummy_buffer;

      if( !b ) return false;

      unsigned char *raw = &outv[0];
      const unsigned int rowsize = xmax - xmin + 1;
      const unsigned int colsize = ymax - ymin + 1;
      const unsigned int bytesPerPixel = pf.GetPixelSize();

      const unsigned char *tmpBuffer1 = raw;
      for (unsigned int y = ymin; y <= ymax; ++y)
        {
        size_t theOffset = 0 + (0*dimensions[1]*dimensions[0] + y*dimensions[0] + xmin)*bytesPerPixel;
        tmpBuffer1 = raw + theOffset;
        memcpy(&(buffer[((z-zmin)*rowsize*colsize +
              (y-ymin)*rowsize)*bytesPerPixel]),
          tmpBuffer1, rowsize*bytesPerPixel);
        }
      }
    }
  return true;
}

ImageCodec * JPEGLSCodec::Clone() const
{
  JPEGLSCodec * copy = new JPEGLSCodec;
  return copy;
}

bool JPEGLSCodec::StartEncode( std::ostream & )
{
  return true;
}
bool JPEGLSCodec::IsRowEncoder()
{
  return false;
}

bool JPEGLSCodec::IsFrameEncoder()
{
  return true;
}

bool JPEGLSCodec::AppendRowEncode( std::ostream & , const char * , size_t )
{
  return false;
}

bool JPEGLSCodec::AppendFrameEncode( std::ostream & out, const char * data, size_t datalen )
{
  const unsigned int * dimensions = this->GetDimensions();
  const PixelFormat & pf = this->GetPixelFormat();
  assert( datalen == dimensions[0] * dimensions[1] * pf.GetPixelSize() );

  std::vector<BYTE> rgbyteCompressed;
  rgbyteCompressed.resize(dimensions[0] * dimensions[1] * 4);

  size_t cbyteCompressed;
  const bool b = this->CodeFrameIntoBuffer((char*)&rgbyteCompressed[0], rgbyteCompressed.size(), cbyteCompressed, data, datalen );
  if( !b ) return false;

  out.write( (char*)&rgbyteCompressed[0], cbyteCompressed );

  return true;
}

bool JPEGLSCodec::StopEncode( std::ostream & )
{
  return true;
}


} // end namespace gdcm
