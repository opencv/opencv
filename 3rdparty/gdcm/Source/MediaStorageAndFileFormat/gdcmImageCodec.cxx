/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageCodec.h"
#include "gdcmJPEGCodec.h"
#include "gdcmByteSwap.txx"
#include "gdcmTrace.h"

#include <iostream>
#include <iomanip>
#include <iterator>

#include <cstring>
#include <limits.h>

namespace gdcm
{

class ImageInternals
{
public:
};

ImageCodec::ImageCodec()
{
  PlanarConfiguration = 0;
  RequestPlanarConfiguration = false;
  RequestPaddedCompositePixelCode = false;
  PI = PhotometricInterpretation::UNKNOW;
  //LUT = LookupTable(LookupTable::UNKNOWN);
  LUT = new LookupTable;
  NeedByteSwap = false;
  NeedOverlayCleanup = false;
  Dimensions[0] = Dimensions[1] = Dimensions[2] = 0;
  NumberOfDimensions = 0;
  LossyFlag = false;
}

ImageCodec::~ImageCodec()
{
}

bool ImageCodec::GetHeaderInfo(std::istream &, TransferSyntax &)
{
  // This function should really be virtual pure.
  assert( 0 );
  return false;
}

void ImageCodec::SetLossyFlag(bool l)
{
  LossyFlag = l;
}

bool ImageCodec::GetLossyFlag() const
{
  return LossyFlag;
}

bool ImageCodec::IsLossy() const
{
  return LossyFlag;
}

void ImageCodec::SetNumberOfDimensions(unsigned int dim)
{
  NumberOfDimensions = dim;
}

unsigned int ImageCodec::GetNumberOfDimensions() const
{
  return NumberOfDimensions;
}


const PhotometricInterpretation &ImageCodec::GetPhotometricInterpretation() const
{
  return PI;
}

void ImageCodec::SetPhotometricInterpretation(
  PhotometricInterpretation const &pi)
{
  PI = pi;
}

bool ImageCodec::DoByteSwap(std::istream &is, std::ostream &os)
{
  // FIXME: Do some stupid work:
  std::streampos start = is.tellg();
  assert( 0 - start == 0 );
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  //assert(buf_size < INT_MAX);
  char *dummy_buffer = new char[(unsigned int)buf_size];
  is.seekg(start, std::ios::beg);
  is.read( dummy_buffer, buf_size);
  is.seekg(start, std::ios::beg); // reset
  //SwapCode sc = is.GetSwapCode();

  assert( !(buf_size % 2) );
#ifdef GDCM_WORDS_BIGENDIAN
  if( PF.GetBitsAllocated() == 16 )
    {
    ByteSwap<uint16_t>::SwapRangeFromSwapCodeIntoSystem((uint16_t*)
      dummy_buffer, SwapCode::LittleEndian, buf_size/2);
    }
#else
  // GE_DLX-8-MONO2-PrivateSyntax.dcm is 8bits
  //  assert( PF.GetBitsAllocated() == 16 );
  if ( PF.GetBitsAllocated() == 16 )
    {
    ByteSwap<uint16_t>::SwapRangeFromSwapCodeIntoSystem((uint16_t*)
      dummy_buffer, SwapCode::BigEndian, buf_size/2);
    }
#endif
  os.write(dummy_buffer, buf_size);
  delete[] dummy_buffer;
  return true;
}

bool ImageCodec::DoYBR(std::istream &is, std::ostream &os)
{
  // FIXME: Do some stupid work:
  std::streampos start = is.tellg();
  assert( 0 - start == 0 );
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  //assert(buf_size < INT_MAX);
  char *dummy_buffer = new char[(unsigned int)buf_size];
  is.seekg(start, std::ios::beg);
  is.read( dummy_buffer, buf_size);
  is.seekg(start, std::ios::beg); // reset
  //SwapCode sc = is.GetSwapCode();

  // Code is coming from:
  // http://lestourtereaux.free.fr/papers/data/yuvrgb.pdf
  assert( !(buf_size % 3) );
  unsigned long size = (unsigned long)buf_size/3;
  //assert(buf_size < INT_MAX);
  unsigned char *copy = new unsigned char[ (unsigned int)buf_size ];
  memmove( copy, dummy_buffer, (size_t)buf_size);
assert(0); // Do not use this code !
  // FIXME FIXME FIXME
  // The following is bogus: we are doing two operation at once:
  // Planar configuration AND YBR... doh !
  const unsigned char *a = copy + 0;
  const unsigned char *b = copy + size;
  const unsigned char *c = copy + size + size;
  int R, G, B;

  unsigned char *p = (unsigned char*)dummy_buffer;
  for (unsigned long j = 0; j < size; ++j)
    {
    R = 38142 *(*a-16) + 52298 *(*c -128);
    G = 38142 *(*a-16) - 26640 *(*c -128) - 12845 *(*b -128);
    B = 38142 *(*a-16) + 66093 *(*b -128);

    R = (R+16384)>>15;
    G = (G+16384)>>15;
    B = (B+16384)>>15;

    if (R < 0)   R = 0;
    if (G < 0)   G = 0;
    if (B < 0)   B = 0;
    if (R > 255) R = 255;
    if (G > 255) G = 255;
    if (B > 255) B = 255;

    *(p++) = (unsigned char)R;
    *(p++) = (unsigned char)G;
    *(p++) = (unsigned char)B;
    a++;
    b++;
    c++;
    }
  delete[] copy;

  os.write(dummy_buffer, buf_size);
  delete[] dummy_buffer;
  return true;
}

bool ImageCodec::DoPlanarConfiguration(std::istream &is, std::ostream &os)
{
  // FIXME: Do some stupid work:
  std::streampos start = is.tellg();
  assert( 0 - start == 0 );
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  //assert(buf_size < INT_MAX);
  char *dummy_buffer = new char[(unsigned int)buf_size];
  is.seekg(start, std::ios::beg);
  is.read( dummy_buffer, buf_size);
  is.seekg(start, std::ios::beg); // reset
  //SwapCode sc = is.GetSwapCode();

  // US-RGB-8-epicard.dcm
  //assert( image.GetNumberOfDimensions() == 3 );
  assert( buf_size % 3 == 0 );
  unsigned long size = (unsigned long)buf_size/3;
  char *copy = new char[ (unsigned int)buf_size ];
  //memmove( copy, dummy_buffer, buf_size);

  const char *r = dummy_buffer /*copy*/;
  const char *g = dummy_buffer /*copy*/ + size;
  const char *b = dummy_buffer /*copy*/ + size + size;

  char *p = copy /*dummy_buffer*/;
  for (unsigned long j = 0; j < size; ++j)
    {
    *(p++) = *(r++);
    *(p++) = *(g++);
    *(p++) = *(b++);
    }
  delete[] dummy_buffer /*copy*/;

  os.write(copy /*dummy_buffer*/, buf_size);
  delete[] copy;
  return true;
}

bool ImageCodec::DoSimpleCopy(std::istream &is, std::ostream &os)
{
#if 1
  std::streampos start = is.tellg();
  assert( 0 - start == 0 );
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  //assert(buf_size < INT_MAX);
  char *dummy_buffer = new char[(unsigned int)buf_size];
  is.seekg(start, std::ios::beg);
  is.read( dummy_buffer, buf_size);
  is.seekg(start, std::ios::beg); // reset
  os.write( dummy_buffer, buf_size);
  delete[] dummy_buffer ;
#else
  // This code is ideal but is failing on an RLE image...need to figure out
  // what is wrong to reactivate this code.
  os.rdbuf( is.rdbuf() );
#endif

  return true;
}

bool ImageCodec::DoPaddedCompositePixelCode(std::istream &is, std::ostream &os)
{
  // FIXME: Do some stupid work:
  std::streampos start = is.tellg();
  assert( 0 - start == 0 );
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  //assert(buf_size < INT_MAX);
  char *dummy_buffer = new char[(unsigned int)buf_size];
  is.seekg(start, std::ios::beg);
  is.read( dummy_buffer, buf_size);
  is.seekg(start, std::ios::beg); // reset
  //SwapCode sc = is.GetSwapCode();

  assert( !(buf_size % 2) );
  if( GetPixelFormat().GetBitsAllocated() == 16 )
    {
    for(size_t i = 0; i < buf_size/2; ++i)
      {
#ifdef GDCM_WORDS_BIGENDIAN
      os.write( dummy_buffer+i, 1 );
      os.write( dummy_buffer+i+buf_size/2, 1 );
#else
      os.write( dummy_buffer+i+buf_size/2, 1 );
      os.write( dummy_buffer+i, 1 );
#endif
      }
    }
  else if( GetPixelFormat().GetBitsAllocated() == 32 )
    {
  assert( !(buf_size % 4) );
    for(size_t i = 0; i < buf_size/4; ++i)
      {
#ifdef GDCM_WORDS_BIGENDIAN
      os.write( dummy_buffer+i, 1 );
      os.write( dummy_buffer+i+1*buf_size/4, 1 );
      os.write( dummy_buffer+i+2*buf_size/4, 1 );
      os.write( dummy_buffer+i+3*buf_size/4, 1 );
#else
      os.write( dummy_buffer+i+3*buf_size/4, 1 );
      os.write( dummy_buffer+i+2*buf_size/4, 1 );
      os.write( dummy_buffer+i+1*buf_size/4, 1 );
      os.write( dummy_buffer+i, 1 );
#endif
      }
    }
  else
    {
    return false;
    }
  delete[] dummy_buffer;
  return true;
}

bool ImageCodec::DoInvertMonochrome(std::istream &is, std::ostream &os)
{
  if ( PF.GetPixelRepresentation() )
    {
    if ( PF.GetBitsAllocated() == 8 )
      {
      uint8_t c;
      while( is.read((char*)&c,1) )
        {
        c = (uint8_t)(255 - c);
        os.write((char*)&c, 1 );
        }
      }
    else if ( PF.GetBitsAllocated() == 16 )
      {
      assert( PF.GetBitsStored() != 12 );
      uint16_t smask16 = 65535;
      uint16_t c;
      while( is.read((char*)&c,2) )
        {
        c = (uint16_t)(smask16 - c);
        os.write((char*)&c, 2);
        }
      }
    }
  else
    {
    if ( PF.GetBitsAllocated() == 8 )
      {
      uint8_t c;
      while( is.read((char*)&c,1) )
        {
        c = (uint8_t)(255 - c);
        os.write((char*)&c, 1);
        }
      }
    else if ( PF.GetBitsAllocated() == 16 )
      {
      uint16_t mask = 1;
      for (int j=0; j<PF.GetBitsStored()-1; ++j)
        {
        mask = (uint16_t)((mask << 1) + 1); // will be 0x0fff when BitsStored = 12
        }

      uint16_t c;
      while( is.read((char*)&c,2) )
        {
        if( c > mask )
          {
          // IMAGES/JPLY/RG3_JPLY aka CompressedSamples^RG3/1.3.6.1.4.1.5962.1.1.11.1.5.20040826185059.5457
          // gdcmData/D_CLUNIE_RG3_JPLY.dcm
          // stores a 12bits JPEG stream with scalar value [0,1024], however
          // the DICOM header says the data are stored on 10bits [0,1023], thus this HACK:
          gdcmWarningMacro( "Bogus max value: "<< c << " max should be at most: " << mask
            << " results will be truncated. Use at own risk");
          c = mask;
          }
        assert( c <= mask );
        c = (uint16_t)(mask - c);
        assert( c <= mask );
        os.write((char*)&c, 2);
        }
      }

    }
  return true;
}

struct ApplyMask
{
  uint16_t operator()(uint16_t c) const {
    return (uint16_t)((c >> (BitsStored - HighBit - 1)) & pmask);
  }
  unsigned short BitsStored;
  unsigned short HighBit;
  uint16_t pmask;
};

// Cleanup the unused bits
bool ImageCodec::DoOverlayCleanup(std::istream &is, std::ostream &os)
{
  assert( PF.GetBitsAllocated() > 8 );
  if( PF.GetBitsAllocated() == 16 )
    {
    // pmask : to mask the 'unused bits' (may contain overlays)
    uint16_t pmask = 0xffff;
    pmask = (uint16_t)(pmask >> ( PF.GetBitsAllocated() - PF.GetBitsStored() ));

    if( PF.GetPixelRepresentation() )
      {
      // smask : to check the 'sign' when BitsStored != BitsAllocated
      uint16_t smask = 0x0001;
      smask = (uint16_t)(
        smask << ( 16 - (PF.GetBitsAllocated() - PF.GetBitsStored() + 1) ));
      // nmask : to propagate sign bit on negative values
      int16_t nmask = (int16_t)0x8000;
      nmask = (int16_t)(nmask >> ( PF.GetBitsAllocated() - PF.GetBitsStored() - 1 ));

      uint16_t c;
      while( is.read((char*)&c,2) )
        {
        c = (uint16_t)(c >> (PF.GetBitsStored() - PF.GetHighBit() - 1));
        if ( c & smask )
          {
          c = (uint16_t)(c | nmask);
          }
        else
          {
          c = c & pmask;
          }
        os.write((char*)&c, 2 );
        }
      }
    else // Pixel are unsigned
      {
#if 1
      uint16_t c;
      while( is.read((char*)&c,2) )
        {
        c = (uint16_t)(
          (c >> (PF.GetBitsStored() - PF.GetHighBit() - 1)) & pmask);
        os.write((char*)&c, 2 );
        }
      //os.rdbuf( is.rdbuf() );
#else
      //std::ostreambuf_iterator<char> end_of_stream_iterator;
      //std::ostreambuf_iterator<char> out_iter(os.rdbuf());
      //while( out_iter != end_of_stream_iterator )
      //  {
      //  *out_iter =
      //    (*out_iter >> (PF.GetBitsStored() - PF.GetHighBit() - 1)) & pmask;
      //  }
      std::istreambuf_iterator<int> it_in(is);
      std::istreambuf_iterator<int> eos;
      std::ostreambuf_iterator<int> it_out(os);
      ApplyMask am;
      am.BitsStored = PF.GetBitsStored();
      am.HighBit = PF.GetHighBit();
      am.pmask = pmask;

      std::transform(it_in, eos, it_out, am);
#endif
      }
    }
  else
    {
    assert(0); // TODO
    }
  return true;
}

bool ImageCodec::Decode(DataElement const &, DataElement &)
{
  return true;
}

bool ImageCodec::DecodeByStreams(std::istream &is, std::ostream &os)
{
  assert( PlanarConfiguration == 0 || PlanarConfiguration == 1);
  assert( PI != PhotometricInterpretation::UNKNOW );
  std::stringstream bs_os; // ByteSwap
  std::stringstream pcpc_os; // Padded Composite Pixel Code
  std::stringstream pi_os; // PhotometricInterpretation
  std::stringstream pl_os; // PlanarConf
  std::istream *cur_is = &is;

  // First thing do the byte swap:
  if( NeedByteSwap )
    {
    // MR_GE_with_Private_Compressed_Icon_0009_1110.dcm
    DoByteSwap(*cur_is,bs_os);
    cur_is = &bs_os;
    }
  if ( RequestPaddedCompositePixelCode )
    {
    // D_CLUNIE_CT2_RLE.dcm
    DoPaddedCompositePixelCode(*cur_is,pcpc_os);
    cur_is = &pcpc_os;
    }

  // Second thing do palette color.
  // This way PALETTE COLOR will be applied before we do
  // Planar Configuration
  switch(PI)
    {
  case PhotometricInterpretation::MONOCHROME2:
  case PhotometricInterpretation::RGB:
  case PhotometricInterpretation::ARGB:
    break;
  case PhotometricInterpretation::MONOCHROME1:
    // CR-MONO1-10-chest.dcm
    //DoInvertMonochrome(*cur_is, pi_os);
    //cur_is = &pi_os;
    break;
  case PhotometricInterpretation::YBR_FULL:
    //DoYBR(*cur_is,pi_os);
    //cur_is = &pi_os;
    {
      const JPEGCodec *c = dynamic_cast<const JPEGCodec*>(this);
      if( c )
        {
        // The following is required for very special case of color space conversion
        // dcmdrle ACUSON-24-YBR_FULL-RLE.dcm bla.dcm
        // dcmcjpeg bla.dcm foo.dcm
        // foo.dcm would be not displayed correctly
        //this->SetPhotometricInterpretation( PhotometricInterpretation::RGB );
        }
    }
    break;
  case PhotometricInterpretation::PALETTE_COLOR:
    //assert( LUT );
    // Nothing needs to be done
    break;
  case PhotometricInterpretation::YBR_FULL_422:
      {
      // US-GE-4AICL142.dcm
      // Hopefully it has been done by the JPEG decoder itself...
      const JPEGCodec *c = dynamic_cast<const JPEGCodec*>(this);
      if( !c )
        {
        gdcmErrorMacro( "YBR_FULL_422 is not implemented in GDCM. Image will be displayed incorrectly" );
        //this->SetPhotometricInterpretation( PhotometricInterpretation::RGB );
        }
      }
    break;
  case PhotometricInterpretation::YBR_ICT:
    break;
  case PhotometricInterpretation::YBR_RCT:
    break;
  default:
    gdcmErrorMacro( "Unhandled PhotometricInterpretation: " << PI );
    return false;
    assert(0);
    }

  if( /*PlanarConfiguration ||*/ RequestPlanarConfiguration )
    {
    DoPlanarConfiguration(*cur_is,pl_os);
    cur_is = &pl_os;
    }

  // Do the overlay cleanup (cleanup the unused bits)
  // must be the last operation (duh!)
  if ( PF.GetBitsAllocated() != PF.GetBitsStored()
    && PF.GetBitsAllocated() != 8 )
    {
    // Technically we should only run this operation if the image declares it has overlay AND
    // there is no (0x60xx,0x3000) element, for example:
    // - XA_GE_JPEG_02_with_Overlays.dcm
    // - SIEMENS_GBS_III-16-ACR_NEMA_1.acr
    // Sigh, I finally found someone not declaring that unused bits where not zero:
    // gdcmConformanceTests/dcm4chee_unusedbits_not_zero.dcm
    if( NeedOverlayCleanup )
      DoOverlayCleanup(*cur_is,os);
    else
      {
      // Once the issue with IMAGES/JPLY/RG3_JPLY aka gdcmData/D_CLUNIE_RG3_JPLY.dcm is solved the previous
      // code will be replace with a simple call to:
      DoSimpleCopy(*cur_is,os);
      }
    }
  else
    {
    assert( PF.GetBitsAllocated() == PF.GetBitsStored() );
    DoSimpleCopy(*cur_is,os);
    }

  return true;
}

bool ImageCodec::IsValid(PhotometricInterpretation const &)
{
  return false;
}

void ImageCodec::SetDimensions(const unsigned int d[3])
{
  Dimensions[0] = d[0];
  Dimensions[1] = d[1];
  Dimensions[2] = d[2];
}

void ImageCodec::SetDimensions(const std::vector<unsigned int> & d)
{
  size_t theSize = d.size();
  assert(theSize<= 3);
  for (size_t i = 0; i < 3; i++)
    {
    if (i < theSize)
      Dimensions[i] = d[i];
    else
      Dimensions[i] = 1;
    }
}

bool ImageCodec::StartEncode( std::ostream & )
{
  assert(0);
  return false;
}
bool ImageCodec::IsRowEncoder()
{
  return false;
}
bool ImageCodec::IsFrameEncoder()
{
  return false;
}
bool ImageCodec::AppendRowEncode( std::ostream & , const char * , size_t )
{
  assert(0);
  return false;
}
// TODO: technically the frame encoder could use the row encoder when present
// this could reduce code duplication
bool ImageCodec::AppendFrameEncode( std::ostream & , const char * , size_t )
{
  assert(0);
  return false;
}
bool ImageCodec::StopEncode( std::ostream & )
{
  assert(0);
  return false;
}

} // end namespace gdcm
