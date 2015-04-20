/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmLookupTable.h"
#include <vector>
#include <set>

#include <string.h>

namespace gdcm
{

class LookupTableInternal
{
public:
  LookupTableInternal():RGB()
  {
    Length[0] = Length[1] = Length[2] = 0;
    Subscript[0] = Subscript[1] = Subscript[2] = 0;
    BitSize[0] = BitSize[1] = BitSize[2] = 0;
  }
  unsigned int Length[3]; // In DICOM the length is specified on a short
  // but 65536 is expressed as 0 ...
  unsigned short Subscript[3];
  unsigned short BitSize[3];
  std::vector<unsigned char> RGB;
};

LookupTable::LookupTable()
{
  Internal = new LookupTableInternal;
  BitSample = 0;
  IncompleteLUT = false;
}

LookupTable::~LookupTable()
{
  delete Internal;
}

bool LookupTable::Initialized() const
{
  bool b1 = BitSample != 0;
  bool b2 =
    Internal->BitSize[0] != 0 &&
    Internal->BitSize[1] != 0 &&
    Internal->BitSize[2] != 0;
  return b1 && b2;
}

void LookupTable::Clear()
{
  BitSample = 0;
  IncompleteLUT = false;
  delete Internal;
  Internal = new LookupTableInternal;
}

void LookupTable::Allocate( unsigned short bitsample )
{
  if( bitsample == 8 )
    {
    Internal->RGB.resize( 256 * 3 );
    }
  else if ( bitsample == 16 )
    {
    Internal->RGB.resize( 65536 * 2 * 3 );
    }
  else
    {
    gdcmAssertAlwaysMacro(0);
    }
  BitSample = bitsample;
}

void LookupTable::InitializeLUT(LookupTableType type, unsigned short length,
  unsigned short subscript, unsigned short bitsize)
{
  if( bitsize != 8 && bitsize != 16 )
    {
    return;
    }
  assert( type >= RED && type <= BLUE );
  assert( subscript == 0 );
  assert( bitsize == 8 || bitsize == 16 );
  if( length == 0 )
    {
    Internal->Length[type] = 65536;
    }
  else
    {
    if( length != 256 )
      {
      IncompleteLUT = true;
      }
    Internal->Length[type] = length;
    }
  Internal->Subscript[type] = subscript;
  Internal->BitSize[type] = bitsize;
}

unsigned int LookupTable::GetLUTLength(LookupTableType type) const
{
  return Internal->Length[type];
}

void LookupTable::SetLUT(LookupTableType type, const unsigned char *array,
  unsigned int length)
{
  (void)length;
  //if( !Initialized() ) return;
  if( !Internal->Length[type] )
    {
    gdcmDebugMacro( "Need to set length first" );
    return;
    }

  if( !IncompleteLUT )
    {
    assert( Internal->RGB.size() == 3*Internal->Length[type]*(BitSample/8) );
    }
  // Too funny: 05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm
  // There is pseudo PALETTE_COLOR LUT in the Icon, if one look carefully the LUT values
  // goes like this: 0, 1, 2, 3, 4, 5, 6, 7 ...
  if( BitSample == 8 )
    {
    const unsigned int mult = Internal->BitSize[type]/8;
    assert( Internal->Length[type]*mult == length
      || Internal->Length[type]*mult + 1 == length );
    unsigned int offset = 0;
    if( mult == 2 )
      {
      offset = 1;
      }
    for( unsigned int i = 0; i < Internal->Length[type]; ++i)
      {
      assert( i*mult+offset < length );
      assert( 3*i+type < Internal->RGB.size() );
      Internal->RGB[3*i+type] = array[i*mult+offset];
      }
    }
  else if( BitSample == 16 )
    {
    assert( Internal->Length[type]*(BitSample/8) == length );
    uint16_t *uchar16 = (uint16_t*)&Internal->RGB[0];
    const uint16_t *array16 = (uint16_t*)array;
    for( unsigned int i = 0; i < Internal->Length[type]; ++i)
      {
      assert( 2*i < length );
      assert( 2*(3*i+type) < Internal->RGB.size() );
      uchar16[3*i+type] = array16[i];
      }
    }
}

void LookupTable::GetLUT(LookupTableType type, unsigned char *array, unsigned int &length) const
{
  if( BitSample == 8 )
    {
    const unsigned int mult = Internal->BitSize[type]/8;
    length = Internal->Length[type]*mult;
    unsigned int offset = 0;
    if( mult == 2 )
      {
      offset = 1;
      }
    for( unsigned int i = 0; i < Internal->Length[type]; ++i)
      {
      assert( i*mult+offset < length );
      assert( 3*i+type < Internal->RGB.size() );
      array[i*mult+offset] = Internal->RGB[3*i+type];
      }
    }
  else if( BitSample == 16 )
    {
    length = Internal->Length[type]*(BitSample/8);
    uint16_t *uchar16 = (uint16_t*)&Internal->RGB[0];
    uint16_t *array16 = (uint16_t*)array;
    for( unsigned int i = 0; i < Internal->Length[type]; ++i)
      {
      assert( 2*i < length );
      assert( 2*(3*i+type) < Internal->RGB.size() );
      array16[i] = uchar16[3*i+type];
      }
    }
}

void LookupTable::GetLUTDescriptor(LookupTableType type, unsigned short &length,
  unsigned short &subscript, unsigned short &bitsize) const
{
  assert( type >= RED && type <= BLUE );
  if( Internal->Length[type] == 65536 )
    {
    length = 0;
    }
  else
    {
    length = (unsigned short)Internal->Length[type];
    }
  subscript = Internal->Subscript[type];
  bitsize = Internal->BitSize[type];

  // postcondition
  assert( subscript == 0 );
  assert( bitsize == 8 || bitsize == 16 );
}

void LookupTable::InitializeRedLUT(unsigned short length,
unsigned short subscript,
unsigned short bitsize)
  {
  InitializeLUT(RED, length, subscript, bitsize);
  }
void LookupTable::InitializeGreenLUT(unsigned short length,
  unsigned short subscript,
  unsigned short bitsize)
{
  InitializeLUT(GREEN, length, subscript, bitsize);
}
void LookupTable::InitializeBlueLUT(unsigned short length,
  unsigned short subscript,
  unsigned short bitsize)
{
  InitializeLUT(BLUE, length, subscript, bitsize);
}

void LookupTable::SetRedLUT(const unsigned char *red, unsigned int length)
{
  SetLUT(RED, red, length);
}

void LookupTable::SetGreenLUT(const unsigned char *green, unsigned int length)
{
  SetLUT(GREEN, green, length);
}

void LookupTable::SetBlueLUT(const unsigned char *blue, unsigned int length)
{
  SetLUT(BLUE, blue, length);
}

namespace {
  typedef union {
    uint8_t rgb[4]; // 3rd value = 0
    uint32_t I;
  } U8;

  typedef union {
    uint16_t rgb[4]; // 3rd value = 0
    uint64_t I;
  } U16;

  struct ltstr8
    {
    bool operator()(U8 u1, U8 u2) const
      {
      return u1.I < u2.I;
      }
    };
  struct ltstr16
    {
    bool operator()(U16 u1, U16 u2) const
      {
      return u1.I < u2.I;
      }
    };
} // end namespace

inline void printrgb( const unsigned char *rgb )
{
  std::cout << int(rgb[0]) << "," << int(rgb[1]) << "," << int(rgb[2]);
}

void LookupTable::Encode(std::istream &is, std::ostream &os)
{
  if ( BitSample == 8 )
    {
#if 0
    // FIXME:
    // There is a very subbtle issue here. We are trying to compress a 8bits RGB image
    // into an 8bits allocated indexed Pixel Data with 8bits LUT... this is just not
    // possible in the general case
    typedef std::set< U8, ltstr8 > RGBColorIndexer;
    RGBColorIndexer s;

    int count = 0;
    while( !is.eof() )
      {
      U8 u;
      u.rgb[3] = 0;
      is.read( (char*)u.rgb, 3);
      assert( u.rgb[3] == 0 );
      //assert( u.rgb[0] == u.rgb[1] && u.rgb[1] == u.rgb[2] );
      std::pair<RGBColorIndexer::iterator,bool> it = s.insert( u );
      if( it.second ) ++count;
      int d = std::distance(s.begin(), it.first);
      //std::cout << count << " Index: " << d << " -> "; printrgb( u.rgb ); std::cout << "\n";
      //assert( s.size() < 256 );
      assert( d < s.size() );
      //assert( d < 256 && d >= 0 );
      }

    // now generate output image
    // this has two been done in two passes as std::set always re-balance
    is.clear();
    is.seekg( 0 );
    while( !is.eof() )
      {
      U8 u;
      u.rgb[3] = 0;
      is.read( (char*)u.rgb, 3);
      assert( u.rgb[3] == 0 );
      //assert( u.rgb[0] == u.rgb[1] && u.rgb[1] == u.rgb[2] );
      std::pair<RGBColorIndexer::iterator,bool> it = s.insert( u );
      int d = std::distance(s.begin(), it.first);
      //std::cout << "Index: " << d << " -> "; printrgb( u.rgb ); std::cout << "\n";
      assert( d < s.size() );
      //assert( d < 256 && d >= 0 );
      os.put( d );
      }

    // now setup the LUT itself:
    unsigned short ncolor = s.size();
    // FIXME: shop'off any RGB that is not at the beginning.
    if( ncolor > 256 ) ncolor = 256;
    InitializeRedLUT(ncolor, 0, 8);
    InitializeGreenLUT(ncolor, 0, 8);
    InitializeBlueLUT(ncolor, 0, 8);
    //int i = Internal->RGB.size();
    //assert( Internal->RGB.size() == 5 );
    int idx = 0;
    for( RGBColorIndexer::const_iterator it = s.begin(); it != s.end() && idx < 256; ++it, ++idx )
      {
      assert( idx == std::distance( s.begin(), it ) );
      //std::cout << "Index: " << idx << " -> "; printrgb( it->rgb ); std::cout << "\n";
      Internal->RGB[3*idx+RED]   = it->rgb[RED];
      Internal->RGB[3*idx+GREEN] = it->rgb[GREEN];
      Internal->RGB[3*idx+BLUE]  = it->rgb[BLUE];
      }
#else
    while( !is.eof() )
      {
      U8 u;
      u.rgb[3] = 0;
      is.read( (char*)u.rgb, 3);
      assert( u.rgb[3] == 0 );
      int d = 0;
      assert( d < 256 && d >= 0 );
      os.put( (char)d );
      }
#endif
    }
  else if ( BitSample == 16 )
    {
#if 0
    typedef std::set< U16, ltstr16 > RGBColorIndexer;
    RGBColorIndexer s;

    while( !is.eof() )
      {
      U16 u;
      u.rgb[3] = 0;
      is.read( (char*)u.rgb, 3*2);
      assert( u.rgb[3] == 0 );
      //assert( u.rgb[0] == u.rgb[1] && u.rgb[1] == u.rgb[2] );
      std::pair<RGBColorIndexer::iterator,bool> it = s.insert( u );
      int d = std::distance(s.begin(), it.first);
      //std::cout << "Index: " << d << " -> "; printrgb( u.rgb ); std::cout << "\n";
      assert( d < s.size() );
      assert( d < 65536 && d >= 0 );
      }

    // now generate output image
    // this has two been done in two passes as std::set always re-balance
    is.clear();
    is.seekg( 0 );
    while( !is.eof() )
      {
      U16 u;
      u.rgb[3] = 0;
      is.read( (char*)u.rgb, 3*2);
      assert( u.rgb[3] == 0 );
      //assert( u.rgb[0] == u.rgb[1] && u.rgb[1] == u.rgb[2] );
      std::pair<RGBColorIndexer::iterator,bool> it = s.insert( u );
      unsigned short d = std::distance(s.begin(), it.first);
      //std::cout << "Index: " << d << " -> "; printrgb( u.rgb ); std::cout << "\n";
      assert( d < s.size() );
      assert( d < 65536 && d >= 0 );
      os.write( (char*)&d , 2 );
      }

    // now setup the LUT itself:
    unsigned short ncolor = s.size();
    InitializeRedLUT(ncolor, 0, 16);
    InitializeGreenLUT(ncolor, 0, 16);
    InitializeBlueLUT(ncolor, 0, 16);
    //int i = Internal->RGB.size();
    //assert( Internal->RGB.size() == 5 );
    int idx = 0;
    uint16_t *rgb16 = (uint16_t*)&Internal->RGB[0];
    for( RGBColorIndexer::const_iterator it = s.begin(); it != s.end(); ++it, ++idx )
      {
      assert( idx == std::distance( s.begin(), it ) );
      //std::cout << "Index: " << idx << " -> "; printrgb( it->rgb ); std::cout << "\n";
      rgb16[3*idx+RED]   = it->rgb[RED];
      rgb16[3*idx+GREEN] = it->rgb[GREEN];
      rgb16[3*idx+BLUE]  = it->rgb[BLUE];
      }
#else
    while( !is.eof() )
      {
      U16 u;
      u.rgb[3] = 0;
      is.read( (char*)u.rgb, 3*2);
      assert( u.rgb[3] == 0 );
      int d = 0;
      assert( d < 65536 && d >= 0 );
      os.write( (char*)&d , 2 );
      }
#endif
  }
}

void LookupTable::Decode(std::istream &is, std::ostream &os) const
{
  assert( Initialized() );
  if ( BitSample == 8 )
    {
    unsigned char idx;
    unsigned char rgb[3];
    while( !is.eof() )
      {
      is.read( (char*)(&idx), 1);
      if( is.eof() || !is.good() ) break;
      if( IncompleteLUT )
        {
        assert( idx < Internal->Length[RED] );
        assert( idx < Internal->Length[GREEN] );
        assert( idx < Internal->Length[BLUE] );
        }
      rgb[RED]   = Internal->RGB[3*idx+RED];
      rgb[GREEN] = Internal->RGB[3*idx+GREEN];
      rgb[BLUE]  = Internal->RGB[3*idx+BLUE];
      os.write((char*)rgb, 3 );
      }
    }
  else if ( BitSample == 16 )
    {
    // gdcmData/NM-PAL-16-PixRep1.dcm
    const uint16_t *rgb16 = (uint16_t*)&Internal->RGB[0];
    while( !is.eof() )
      {
      unsigned short idx;
      unsigned short rgb[3];
      is.read( (char*)(&idx), 2);
      if( is.eof() || !is.good() ) break;
      if( IncompleteLUT )
        {
        assert( idx < Internal->Length[RED] );
        assert( idx < Internal->Length[GREEN] );
        assert( idx < Internal->Length[BLUE] );
        }
      rgb[RED]   = rgb16[3*idx+RED];
      rgb[GREEN] = rgb16[3*idx+GREEN];
      rgb[BLUE]  = rgb16[3*idx+BLUE];
      os.write((char*)rgb, 3*2);
      }
    }
}

bool LookupTable::Decode(char *output, size_t outlen, const char *input, size_t inlen ) const
{
  bool success = false;
  if( outlen < 3 * inlen )
    {
    gdcmDebugMacro( "Out buffer too small" );
    return false;
    }
  if( !Initialized() )
    {
    gdcmDebugMacro( "Not Initialized" );
    return false;
    }
  if ( BitSample == 8 )
    {
    const unsigned char * end = (unsigned char*)input + inlen;
    unsigned char * rgb = (unsigned char*)output;
    for( unsigned char * idx = (unsigned char*)input; idx != end; ++idx )
      {
      if( IncompleteLUT )
        {
        assert( *idx < Internal->Length[RED] );
        assert( *idx < Internal->Length[GREEN] );
        assert( *idx < Internal->Length[BLUE] );
        }
      rgb[RED]   = Internal->RGB[3 * *idx+RED];
      rgb[GREEN] = Internal->RGB[3 * *idx+GREEN];
      rgb[BLUE]  = Internal->RGB[3 * *idx+BLUE];
      rgb += 3;
      }
    success = true;
    }
  else if ( BitSample == 16 )
    {
    const uint16_t *rgb16 = (uint16_t*)&Internal->RGB[0];
    assert( inlen % 2 == 0 );
    const uint16_t * end = (uint16_t*)(input + inlen);
    uint16_t * rgb = (uint16_t*)output;
    for( uint16_t * idx = (uint16_t*)input; idx != end; ++idx )
      {
      if( IncompleteLUT )
        {
        assert( *idx < Internal->Length[RED] );
        assert( *idx < Internal->Length[GREEN] );
        assert( *idx < Internal->Length[BLUE] );
        }
      rgb[RED]   = rgb16[3 * *idx+RED];
      rgb[GREEN] = rgb16[3 * *idx+GREEN];
      rgb[BLUE]  = rgb16[3 * *idx+BLUE];
      rgb += 3;
      }
    success = true;
    }
  return success;
}

const unsigned char *LookupTable::GetPointer() const
{
  if ( BitSample == 8 )
    {
    return &Internal->RGB[0];
    }
  return 0;
}

bool LookupTable::GetBufferAsRGBA(unsigned char *rgba) const
{
  bool ret = false;
  if ( BitSample == 8 )
    {
    std::vector<unsigned char>::const_iterator it = Internal->RGB.begin();
    for(; it != Internal->RGB.end() ;)
      {
      // RED
      *rgba++ = *it++;
      // GREEN
      *rgba++ = *it++;
      // BLUE
      *rgba++ = *it++;
      // ALPHA
      *rgba++ = 255;
      }
    ret = true;
    }
  else if ( BitSample == 16 )
    {
/*
    assert( Internal->Length[type]*(BitSample/8) == length );
    uint16_t *uchar16 = (uint16_t*)&Internal->RGB[0];
    const uint16_t *array16 = (uint16_t*)array;
    for( unsigned int i = 0; i < Internal->Length[type]; ++i)
      {
      assert( 2*i < length );
      assert( 2*(3*i+type) < Internal->RGB.size() );
      uchar16[3*i+type] = array16[i];
      std::cout << i << " -> " << array16[i] << "\n";
      }

    ret = true;
*/
    //std::vector<unsigned char>::const_iterator it = Internal->RGB.begin();
    uint16_t *uchar16 = (uint16_t*)&Internal->RGB[0];
    uint16_t *rgba16 = (uint16_t*)rgba;
    size_t s = Internal->RGB.size();
    s /= 2;
    s /= 3;
    memset(rgba,0,Internal->RGB.size() * 4 / 3); // FIXME
    for(size_t i = 0; i < s; ++i)
      {
      // RED
      *rgba16++ = *uchar16++;
      // GREEN
      *rgba16++ = *uchar16++;
      // BLUE
      *rgba16++ = *uchar16++;
      // ALPHA
      *rgba16++ = 255*255;
      }

    ret = true;
    }
  return ret;
}

bool LookupTable::WriteBufferAsRGBA(const unsigned char *rgba)
{
  bool ret = false;
  if ( BitSample == 8 )
    {
    std::vector<unsigned char>::iterator it = Internal->RGB.begin();
    for(; it != Internal->RGB.end() ;)
      {
      // RED
      *it++ = *rgba++;
      // GREEN
      *it++ = *rgba++;
      // BLUE
      *it++ = *rgba++;
      // ALPHA
      rgba++; // = 255;
      }
    ret = true;
    }
  else if ( BitSample == 16 )
    {
    //assert( Internal->Length[type]*(BitSample/8) == length );
    uint16_t *uchar16 = (uint16_t*)&Internal->RGB[0];
    const uint16_t *rgba16 = (uint16_t*)rgba;
    size_t s = Internal->RGB.size();
    s /= 2;
    s /= 3;
    assert( s == 65536 );

    for( unsigned int i = 0; i < s /*i < Internal->Length[type]*/; ++i)
      {
      //assert( 2*i < length );
      //assert( 2*(3*i+type) < Internal->RGB.size() );
      //uchar16[3*i+type] = array16[i];
      //std::cout << i << " -> " << array16[i] << "\n";
      // RED
      *uchar16++ = *rgba16++;
      // GREEN
      *uchar16++ = *rgba16++;
      // BLUE
      *uchar16++ = *rgba16++;
      //
      rgba16++; // = *rgba16++;
      }

    ret = true;
    //ret = false;
    }
  return ret;
}

} // end namespace gdcm
