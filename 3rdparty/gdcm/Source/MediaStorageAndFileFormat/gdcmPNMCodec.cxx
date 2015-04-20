/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPNMCodec.h"
#include "gdcmTransferSyntax.h"
#include "gdcmSystem.h"
#include "gdcmDataElement.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSwapper.h"

namespace gdcm
{

PNMCodec::PNMCodec():BufferLength(0)
{
}

PNMCodec::~PNMCodec()
{
}

bool PNMCodec::CanDecode(TransferSyntax const &) const
{
  return false;
}

bool PNMCodec::CanCode(TransferSyntax const &) const
{
  return false;
}



bool PNMCodec::Write(const char *filename, const DataElement &out) const
{
  std::ofstream os(filename, std::ios::binary);
  const unsigned int *dims = this->GetDimensions();
  const PhotometricInterpretation &pi = this->GetPhotometricInterpretation();
  if( pi == PhotometricInterpretation::MONOCHROME2
    || pi == PhotometricInterpretation::MONOCHROME1 ) // warning viz will be surprising
    {
    os << "P5\n";
    }
  else if( pi == PhotometricInterpretation::RGB
    || pi == PhotometricInterpretation::PALETTE_COLOR )
    {
    os << "P6\n";
    }
  else
    {
    gdcmErrorMacro( "PhotometricInterpretation unhandled: " << pi );
    return false;
    }
  os << dims[0] << " " << dims[1] << "\n";

  const unsigned int pc = this->GetPlanarConfiguration();
  if( pc )
    {
    gdcmErrorMacro( "PlanarConfiguration unhandled: " << pc );
    return false;
    }

  const PixelFormat& pf = GetPixelFormat();
  switch(pf)
    {
  case PixelFormat::UINT8:
  //case PixelFormat::INT8:
    os << "255";
    break;
  case PixelFormat::UINT16:
  //case PixelFormat::INT16:
    os << "65535";
    break;
  default:
    gdcmErrorMacro( "Unhandled PF: " << pf );
    return false;
    }
  os << "\n";

  const ByteValue *bv = out.GetByteValue();
  // FIXME: PNM Codec cannot handle encapsulated syntax... sigh
  if(!bv)
    {
    gdcmErrorMacro( "PNM Codec does not handle compress syntax. You need to decompress first." );
    return false;
    }
  assert(bv);

  if( pi == PhotometricInterpretation::PALETTE_COLOR )
    {
    std::stringstream is;
    is.write( bv->GetPointer(), bv->GetLength() );

    const LookupTable &lut = this->GetLUT();
    lut.Decode(is, os);
    }
  else
    {
    if( pf.GetBitsAllocated() == 16 )
      {
      bv->Write<SwapperDoOp, uint16_t>( os );
      }
    else
      {
      //bv->Write<SwapperDoOp, uint8_t>( os );
      bv->WriteBuffer( os );
      }
    }

  os.close();


  return true;
}

bool PNMCodec::Read(const char *filename, DataElement &out) const
{
  size_t len = System::FileSize(filename);
  std::ifstream is(filename, std::ios::binary);
  std::string type, str;
  std::getline(is,type);
  PhotometricInterpretation pi;
  if( type == "P5" )
    pi = PhotometricInterpretation::MONOCHROME2;
  else if( type == "P6" )
    pi = PhotometricInterpretation::RGB;
  else
    {
    std::cerr << "Unhandled PGM type: " << type << std::endl;
    return false;
    }

  // skip comments:
  while( is.peek() == '#' )
    {
    std::getline(is, str);
    //std::cout << str << std::endl;
    }
  unsigned int dims[3] = {};
  is >> dims[0]; is >> dims[1];
  unsigned int maxval;
  is >> maxval;
  // some kind of empty line...
  if( is.peek() == '\n' )
    {
    is.get();
    }
  std::streampos pos = is.tellg();
  //assert(pos < INT_MAX);
  size_t m = (len - (size_t)pos ) / ( dims[0]*dims[1] );
  if( m * dims[0] * dims[1] != len - pos )
    {
    std::cerr << "Problem computing length" << std::endl;
    return false;
    }
  PixelFormat pf;
  switch(maxval)
    {
  case 255:
    pf = PixelFormat::UINT8;
    break;
  case 1023:
    pf = PixelFormat::UINT16;
    pf.SetBitsStored( 10 );
    break;
  case 4095:
    pf = PixelFormat::UINT16;
    pf.SetBitsStored( 12 );
    break;
  case 32767:
    pf = PixelFormat::UINT16;
    pf.SetBitsStored( 15 );
    break;
  case 65535:
    pf = PixelFormat::UINT16;
    break;
  default:
    std::cerr << "Unhandled max val: " << maxval << std::endl;
    return false;
    }
  if( pi == PhotometricInterpretation::RGB )
    {
    pf.SetSamplesPerPixel( 3 );
    }
  //if ( maxval * 8 != bpp ) return 1;

  size_t pdlen = GetBufferLength();
  assert( pdlen );
  char * buf = new char[pdlen];
  // is should be at right offset, just read!
  is.read(buf, len);
  if( !is.eof() ) return false;

  out.SetTag( Tag(0x7fe0,0x0010) );
  VL::Type pdLenSize = (VL::Type)pdlen;
  out.SetByteValue( buf, pdLenSize );
  delete[] buf;

  is.close();

  return true;
}

static inline int log2( int n )
{
  int bits = 0;
  while (n > 0)
    {
    bits++;
    n >>= 1;
    }
  return bits;
}

bool PNMCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
{
  is.seekg( 0, std::ios::end );
  std::streampos len = is.tellg();
  //assert(len < INT_MAX);
  is.seekg( 0, std::ios::beg );

  std::string type, str;
  std::getline(is,type);
  PhotometricInterpretation pi;
  if( type == "P5" )
    pi = PhotometricInterpretation::MONOCHROME2;
  else if( type == "P6" ) // P3 => ASCII
    pi = PhotometricInterpretation::RGB;
  else
    {
    std::cerr << "Unhandled PGM type: " << type << std::endl;
    return false;
    }

  // skip comments:
  while( is.peek() == '#' )
    {
    std::getline(is, str);
    //std::cout << str << std::endl;
    }
  unsigned int dims[3] = {};
  is >> dims[0]; is >> dims[1];
  unsigned int maxval;
  is >> maxval;
  // http://netpbm.sourceforge.net/doc/pgm.html
  // some kind of empty line...
  if( is.peek() == '\n' )
    {
    is.get();
    }
  std::streamoff pos = is.tellg();
  //assert(len < INT_MAX);
  //assert(pos < INT_MAX);
  size_t m = ((size_t)len - (size_t)pos ) / ( dims[0]*dims[1] );
  if( m * dims[0] * dims[1] != (size_t)len - pos )
    {
    std::cerr << "Problem computing length" << std::endl;
    std::cerr << "Pos: " << len - pos << std::endl;
    std::cerr << "expected: " << m * dims[0] * dims[1] << std::endl;
    return false;
    }
  PixelFormat pf = GetPixelFormat();
#if 0
  switch(maxval)
    {
  case 255:
    pf = PixelFormat::UINT8;
    break;
  case 1023:
    pf = PixelFormat::UINT16;
    pf.SetBitsStored( 10 );
    break;
  case 4095:
    pf = PixelFormat::UINT16;
    pf.SetBitsStored( 12 );
    break;
  case 32767:
    pf = PixelFormat::UINT16;
    pf.SetBitsStored( 15 );
    break;
  case 65535:
    pf = PixelFormat::UINT16;
    break;
  default:
    std::cerr << "Unhandled max val: " << maxval << std::endl;
    return false;
    }
#else
  const int nbits = log2( maxval );
  // handle case where nbits = 0 also:
  if( nbits > 0 && nbits <= 8 )
    {
    pf.SetBitsAllocated( 8 );
    pf.SetBitsStored( (unsigned short)nbits );
    }
  else if( nbits > 8 && nbits <= 16 )
    {
    pf.SetBitsAllocated( 16 );
    pf.SetBitsStored( (unsigned short)nbits );
    }
  else if( nbits > 16 && nbits <= 32 )
    {
    pf.SetBitsAllocated( 32 );
    pf.SetBitsStored( (unsigned short)nbits );
    }
  else
    {
    std::cerr << "Unhandled max val: " << maxval << std::endl;
    return false;
    }
#endif
  if( pi == PhotometricInterpretation::RGB )
    {
    pf.SetSamplesPerPixel( 3 );
    }
  //if ( maxval * 8 != bpp ) return 1;

  //image.SetTransferSyntax( TransferSyntax::ExplicitVRBigEndian ); // PGM are big endian
  //image.SetTransferSyntax( TransferSyntax::ExplicitVRLittleEndian ); // PGM are big endian
  //image.SetTransferSyntax( TransferSyntax::ImplicitVRBigEndianPrivateGE ); // PGM are big endian
  if( pf.GetBitsAllocated() == 8 )
    //ts = TransferSyntax::ImplicitVRLittleEndian; // nicer to handle than private GE
    ts = TransferSyntax::ExplicitVRLittleEndian; // nicer to handle than private GE
  else
    ts = TransferSyntax::ImplicitVRBigEndianPrivateGE;

  SetPhotometricInterpretation( pi );
  SetPixelFormat( pf );
  SetDimensions( dims );

  return true;
}

ImageCodec * PNMCodec::Clone() const
{
  return NULL;
}

} // end namespace gdcm
