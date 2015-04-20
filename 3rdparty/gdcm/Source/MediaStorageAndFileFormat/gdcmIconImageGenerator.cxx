/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIconImageGenerator.h"
#include "gdcmIconImage.h"
#include "gdcmAttribute.h"
#include "gdcmPrivateTag.h"
#include "gdcmImage.h"
#include "gdcmJPEGCodec.h"
#include "gdcmRescaler.h"

#include <list>
#include <limits>
#include <queue>
#include <algorithm>

namespace gdcm
{
class IconImageGeneratorInternals
{
public:
  IconImageGeneratorInternals()
    {
    dims[0] = dims[1] = 0;
    Min = 0;
    Max = 0;
    UseMinMax = false;
    AutoMinMax = false;
    ConvertRGBToPaletteColor = true;
    UseOutsideValuePixel = false;
    OutsideValuePixel = 0;
    }
  unsigned int dims[2];
  double Min;
  double Max;
  bool UseMinMax;
  bool AutoMinMax;
  bool ConvertRGBToPaletteColor;
  bool UseOutsideValuePixel;
  double OutsideValuePixel;
};

IconImageGenerator::IconImageGenerator():P(new Pixmap),I(new IconImage),Internals(new IconImageGeneratorInternals)
{
}

IconImageGenerator::~IconImageGenerator()
{
  delete Internals;
}

// Implementation detail:
// This function was required at some point in time since the implementation
// RGB -> PALETTE is extremely slow
void IconImageGenerator::ConvertRGBToPaletteColor(bool b)
{
  Internals->ConvertRGBToPaletteColor = b;
}

void IconImageGenerator::SetOutputDimensions(const unsigned int dims[2])
{
  Internals->dims[0] = dims[0];
  Internals->dims[1] = dims[1];
}

namespace quantization
{
// retrieved from:
// http://en.literateprograms.org/Special:Downloadcode/Median_cut_algorithm_(C_Plus_Plus)

/* Copyright (c) 2011 the authors listed at the following URL, and/or
the authors of referenced articles or incorporated external code:
http://en.literateprograms.org/Median_cut_algorithm_(C_Plus_Plus)?action=history&offset=20080309133934

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Retrieved from: http://en.literateprograms.org/Median_cut_algorithm_(C_Plus_Plus)?oldid=12754
*/

  const int NUM_DIMENSIONS = 3;

  struct Point
    {
    unsigned char x[NUM_DIMENSIONS];
    };

  class Block
    {
    Point minCorner, maxCorner;
    Point* points;
    int pointsLength;
  public:
    Block(Point* points, int pointsLength);
    Point * getPoints();
    int numPoints() const;
    int longestSideIndex() const;
    int longestSideLength() const;
    bool operator<(const Block& rhs) const;
    void shrink();
    };

  template <int index>
    class CoordinatePointComparator
      {
    public:
      bool operator()(Point left, Point right)
        {
        return left.x[index] < right.x[index];
        }
      };

  //std::list<Point> medianCut(Point* image, int numPoints, unsigned int desiredSize);

  Block::Block(Point* pts, int ptslen)
    {
    assert( ptslen > 0 );
    this->points = pts;
    this->pointsLength = ptslen;
    for(int i=0; i < NUM_DIMENSIONS; i++)
      {
      minCorner.x[i] = std::numeric_limits<unsigned char>::min();
      maxCorner.x[i] = std::numeric_limits<unsigned char>::max();
      }
    }

  Point * Block::getPoints()
    {
    return points;
    }

  int Block::numPoints() const
    {
    return pointsLength;
    }

  int Block::longestSideIndex() const
    {
    int m = maxCorner.x[0] - minCorner.x[0];
    int maxIndex = 0;
    for(int i=1; i < NUM_DIMENSIONS; i++)
      {
      int diff = maxCorner.x[i] - minCorner.x[i];
      if (diff > m)
        {
        m = diff;
        maxIndex = i;
        }
      }
    return maxIndex;
    }

  int Block::longestSideLength() const
    {
    int i = longestSideIndex();
    return maxCorner.x[i] - minCorner.x[i];
    }

  bool Block::operator<(const Block& rhs) const
    {
    return this->longestSideLength() < rhs.longestSideLength();
    }

  void Block::shrink()
    {
    int i,j;
    for(j=0; j<NUM_DIMENSIONS; j++)
      {
      minCorner.x[j] = maxCorner.x[j] = points[0].x[j];
      }
    for(i=1; i < pointsLength; i++)
      {
      for(j=0; j<NUM_DIMENSIONS; j++)
        {
        minCorner.x[j] = std::min(minCorner.x[j], points[i].x[j]);
        maxCorner.x[j] = std::max(maxCorner.x[j], points[i].x[j]);
        }
      }
    }

  std::list<Point> medianCut(DataElement const &PixelData, int numPoints, unsigned int desiredSize,
    std::vector<unsigned char> & outputimage )
    {
    assert( numPoints > 0 );
    //Point* Points = (Point*)malloc(sizeof(Point) * numPoints);
    Point* Points = new Point[numPoints];
    assert( Points );
    const ByteValue *bv = PixelData.GetByteValue();
    assert( bv );
    const unsigned char *inbuffer = (unsigned char*)bv->GetPointer();
    assert( inbuffer );
    size_t bvlen = bv->GetLength(); (void)bvlen;
    assert( bvlen == (size_t) numPoints * 3 ); // only 8bits RGB please
    for(int i = 0; i < numPoints; i++)
      {
#if 0
      memcpy(&Points[i], inbuffer + 3 * i, 3);
#else
      Points[i].x[0] = inbuffer[ 3 * i + 0 ];
      Points[i].x[1] = inbuffer[ 3 * i + 1 ];
      Points[i].x[2] = inbuffer[ 3 * i + 2 ];
#endif
      }
    Point* image = Points;

    std::priority_queue<Block> blockQueue;

    Block initialBlock(image, numPoints);
    initialBlock.shrink();

    blockQueue.push(initialBlock);
    while (blockQueue.size() < desiredSize /*&& blockQueue.top().numPoints() > 1*/ )
      {
      Block longestBlock = blockQueue.top();

      blockQueue.pop();
      Point * begin  = longestBlock.getPoints();
      Point * median = longestBlock.getPoints() + (longestBlock.numPoints()+1)/2;
      Point * end    = longestBlock.getPoints() + longestBlock.numPoints();
      switch(longestBlock.longestSideIndex())
        {
      case 0: std::nth_element(begin, median, end, CoordinatePointComparator<0>()); break;
      case 1: std::nth_element(begin, median, end, CoordinatePointComparator<1>()); break;
      case 2: std::nth_element(begin, median, end, CoordinatePointComparator<2>()); break;
        }

      Block block1(begin, median-begin), block2(median, end-median);
      block1.shrink();
      block2.shrink();

      blockQueue.push(block1);
      blockQueue.push(block2);
      }

    std::list<Point> result;
    //int s = blockQueue.size();
    outputimage.resize( numPoints );
    //const ByteValue *bv = PixelData.GetByteValue();
    //const char *inbuffer = bv->GetPointer();

    while(!blockQueue.empty())
      {
      Block block = blockQueue.top();
      blockQueue.pop();
      Point * points = block.getPoints();

      int sum[NUM_DIMENSIONS] = {0,0,0};
      for(int i=0; i < block.numPoints(); i++)
        {
        for(int j=0; j < NUM_DIMENSIONS; j++)
          {
          sum[j] += points[i].x[j];
          }
        }

      Point averagePoint;
      for(int j=0; j < NUM_DIMENSIONS; j++)
        {
        averagePoint.x[j] = sum[j] / block.numPoints();
        }

      result.push_back(averagePoint);

      //int index = std::distance(s.begin(), it.first);
      size_t index = result.size();
      assert( index <= 256 );

      for(int i = 0; i < numPoints; i++)
        {
        const unsigned char *currentcolor = inbuffer + 3 * i;
        for(size_t j = 0; j < block.numPoints(); j++)
          {
          assert( currentcolor < inbuffer + bvlen );
          assert( currentcolor + 3 <= inbuffer + bvlen );
          if( std::equal( currentcolor, currentcolor + 3, points[j].x ) )
            {
            //assert( outputimage[i] == 0 );
            assert( index > 0 );
            outputimage[i] = (unsigned char)(index - 1);
            }
          }
        }
      }

    delete[] Points;
    return result;
    }

} // end namespace quantization

// Create LUT with a maximum number of color equal to \param maxcolor
void IconImageGenerator::BuildLUT( Bitmap & bitmap, unsigned int maxcolor )
{
  assert( Internals->ConvertRGBToPaletteColor );
  using namespace quantization;
  const unsigned int *dims = bitmap.GetDimensions();
  unsigned int numPoints = dims[0]*dims[1];

  std::vector<unsigned char> indeximage;
  std::list<Point> palette =
    medianCut(bitmap.GetDataElement(), numPoints, maxcolor, indeximage);

  size_t ncolors = palette.size();
  LookupTable & lut = bitmap.GetLUT();
  lut.Clear();
  lut.Allocate( 8 );
  std::vector< unsigned char > buffer[3];
  for( int i = 0; i < 3; ++i )
    buffer[i].reserve( ncolors );

  std::list<Point>::const_iterator it = palette.begin();
  for( ; it != palette.end(); ++it )
    {
    Point const & p = *it;
    for( int i = 0; i < 3; ++i )
      buffer[i].push_back( p.x[i] );
    }

  for( int i = 0; i < 3; ++i )
    {
    lut.InitializeLUT( LookupTable::LookupTableType(i), (unsigned short)ncolors, 0, 8 );
    lut.SetLUT( LookupTable::LookupTableType(i), &buffer[i][0], (unsigned short)ncolors );
    }

  bitmap.GetDataElement().SetByteValue( (char*)&indeximage[0], (uint32_t)indeximage.size() );
  assert( lut.Initialized() );
}

void IconImageGenerator::SetOutsideValuePixel(double v)
{
  if( Internals->AutoMinMax )
    {
    Internals->UseOutsideValuePixel = true;
    Internals->OutsideValuePixel = v;
    }
}

void IconImageGenerator::AutoPixelMinMax(bool b)
{
  if( b )
    {
    Internals->UseMinMax = false;
    Internals->AutoMinMax = true;
    }
}

void IconImageGenerator::SetPixelMinMax(double min, double max)
{
  Internals->Min = min;
  Internals->Max = max;
  Internals->UseMinMax = true;
  Internals->AutoMinMax = false;
}

template <typename TPixelType>
void ComputeMinMax( const TPixelType *p, size_t npixels , double & min, double &max, double discardvalue)
{
  assert( npixels );
  const TPixelType discard = (TPixelType)discardvalue;
  assert( (double)discard == discardvalue );
  TPixelType lmin = std::numeric_limits< TPixelType>::max();
  TPixelType lmax = std::numeric_limits< TPixelType>::min();
  for( size_t i = 0; i < npixels; ++i )
    {
    if( p[i] < lmin && p[i] != discard )
      {
      lmin = p[i];
      }
    else if( p[i] > lmax /* && p[i] != discard */ )
      {
      lmax = p[i];
      }
    }
  //assert( lmin != std::numeric_limits< TPixelType>::max() );
  //assert( lmax != std::numeric_limits< TPixelType>::min() );

  // what if lmin == lmax == 0 for example:
  // let's fake a slightly different min/max found:
  if( lmin == lmax )
    {
    if( lmax == std::numeric_limits<TPixelType>::max() )
      {
      lmin--;
      assert( lmin + 1 > lmin );
      }
    else
      {
      lmax++;
      }
    }
  min = lmin;
  max = lmax;
//  std::cout << min << " " << max << std::endl;
}

template <typename TPixelType>
void ComputeMinMax( const TPixelType *p, size_t npixels , double & min, double &max)
{
  assert( npixels );
  TPixelType lmin = std::numeric_limits< TPixelType>::max();
  TPixelType lmax = std::numeric_limits< TPixelType>::min();
  for( size_t i = 0; i < npixels; ++i )
    {
    if( p[i] < lmin )
      {
      lmin = p[i];
      }
    else if( p[i] > lmax )
      {
      lmax = p[i];
      }
    }
  //assert( lmin != std::numeric_limits< TPixelType>::max() );
  //assert( lmax != std::numeric_limits< TPixelType>::min() );

  // what if lmin == lmax == 0 for example:
  // let's fake a slightly different min/max found:
  if( lmin == lmax )
    {
    if( lmax == std::numeric_limits<TPixelType>::max() )
      {
      lmin--;
      assert( lmin + 1 > lmin );
      }
    else
      {
      lmax++;
      }
    }
  min = lmin;
  max = lmax;
//  std::cout << min << " " << max << std::endl;
}

bool IconImageGenerator::Generate()
{
/*
PS 3.3-2009
F.7 ICON IMAGE KEY DEFINITION
a. Only monochrome and palette color images shall be used. Samples per Pixel
(0028,0002) shall have a Value of 1, Photometric Interpretation (0028,0004) shall
have a Value of either MONOCHROME 1, MONOCHROME 2 or PALETTE COLOR,
Planar Configuration (0028,0006) shall not be present
Note: True color icon images are not supported. This is due to the fact that the reduced size of the
Icon Image makes the quality of a palette color image (with 256 colors) sufficient in most cases.
This simplifies the handling of Icon Images by File-set Readers and File-set Updaters.
b. If an FSR/FSU supports Icons (i.e. does not ignore them) then it shall support at least
a maximum size of 64 by 64 Icons. An FSC may write Icons of any size. Icons larger
than 64 by 64 may be ignored by FSRs and FSUs unless specialized by Application
Profiles
c. Pixel samples have a Value of either 1 or 8 for Bits Allocated (0028,0100) and Bits
Stored (0028,0101). High Bit (0028,0102) shall have a Value of one less than the
Value used in Bit Stored
d. Pixel Representation (0028,0103) shall used an unsigned integer representation
(Value 0000H)
e. Pixel Aspect Ratio (0028,0034) shall have a Value of 1:1
f. If a Palette Color lookup Table is used, an 8 Bit Allocated (0028,0100) shall be used
*/
  I->Clear();
  I->SetNumberOfDimensions(2);
  I->SetDimension(0, Internals->dims[0] );
  I->SetDimension(1, Internals->dims[1] );

  PixelFormat oripf = P->GetPixelFormat();

  if( P->GetPhotometricInterpretation() != PhotometricInterpretation::MONOCHROME1
    && P->GetPhotometricInterpretation() != PhotometricInterpretation::MONOCHROME2
    && P->GetPhotometricInterpretation() != PhotometricInterpretation::PALETTE_COLOR
    && P->GetPhotometricInterpretation() != PhotometricInterpretation::RGB
    && P->GetPhotometricInterpretation() != PhotometricInterpretation::YBR_FULL
    && P->GetPhotometricInterpretation() != PhotometricInterpretation::YBR_FULL_422
    && P->GetPhotometricInterpretation() != PhotometricInterpretation::YBR_RCT
    && P->GetPhotometricInterpretation() != PhotometricInterpretation::YBR_ICT )
    {
    gdcmErrorMacro( "PhotometricInterpretation is not supported: "
      << P->GetPhotometricInterpretation() );
    return false;
    }

  if( P->GetPhotometricInterpretation() == PhotometricInterpretation::RGB
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL_422
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_RCT
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_ICT )
    {
    if( Internals->ConvertRGBToPaletteColor )
      {
      I->SetPhotometricInterpretation( PhotometricInterpretation::PALETTE_COLOR );
      }
    else
      {
      I->SetPhotometricInterpretation( PhotometricInterpretation::RGB );
      }
    }
  else
    {
    I->SetPhotometricInterpretation( P->GetPhotometricInterpretation() );
    assert( I->GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME1
      || I->GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME2
      || I->GetPhotometricInterpretation() == PhotometricInterpretation::PALETTE_COLOR );
    if( !Internals->ConvertRGBToPaletteColor
      && P->GetPhotometricInterpretation() == PhotometricInterpretation::PALETTE_COLOR )
      {
      I->SetPhotometricInterpretation( PhotometricInterpretation::RGB );
      }
    }

  assert( I->GetPlanarConfiguration() == 0 );

  // FIXME we should not retrieve the whole image, ideally we only need a
  // single 2D frame
  std::vector< char > vbuffer;
  size_t framelen = P->GetBufferLength();
  if( P->GetNumberOfDimensions() == 3 )
    {
    const unsigned int *dims = P->GetDimensions();
    assert( framelen % dims[2] == 0 );
    framelen /= dims[2];
    }
  vbuffer.resize( P->GetBufferLength() );
  char *buffer = &vbuffer[0];
  bool boolean = P->GetBuffer(buffer);
  if( !boolean ) return false;

  // truncate to the size of a single frame:
  vbuffer.resize( framelen );

  // Important: After call to GetBuffer() in case we have a 12bits stored image
  I->SetPixelFormat( P->GetPixelFormat() );

  DataElement& pixeldata = I->GetDataElement();
  std::vector< char > vbuffer2;
  vbuffer2.resize( I->GetBufferLength() );

  uint8_t ps = I->GetPixelFormat().GetPixelSize();

  char *iconb = &vbuffer2[0];
  char *imgb = &vbuffer[0];

  const unsigned int *imgdims = P->GetDimensions();
  const unsigned int stepi = imgdims[0] / Internals->dims[0];
  const unsigned int stepj = imgdims[1] / Internals->dims[1];
  // Let's cherry-pick pixel from the input image. The nice thing about this approach
  // is that this also works for palletized image.
  // In the future it would be nice to also support averaging a group of pixel, instead
  // of always picking the top-left pixel from the block.
  for(unsigned int i = 0; i < Internals->dims[1]; ++i )
    for(unsigned int j = 0; j < Internals->dims[0]; ++j )
      {
      assert( (i * Internals->dims[0] + j) * ps < I->GetBufferLength() );
      assert( (i * imgdims[0] * stepj + j * stepi) * ps < framelen /*P->GetBufferLength()*/ );
      memcpy(iconb + (i * Internals->dims[0] + j) * ps,
        imgb + (i * imgdims[0] * stepj + j * stepi) * ps, ps );
      }
  // Apply LUT
  if( P->GetPhotometricInterpretation() == PhotometricInterpretation::PALETTE_COLOR )
    {
    std::string tempvbuf(&vbuffer2[0], vbuffer2.size());
    std::istringstream is( tempvbuf );
    std::stringstream ss;
    P->GetLUT().Decode( is, ss );

    if( I->GetPixelFormat().GetBitsAllocated() == 16 )
      {
      //assert( I->GetPixelFormat().GetPixelRepresentation() == 0 );
      std::string s = ss.str();
      Rescaler r;
      r.SetPixelFormat( I->GetPixelFormat() );
      //r.SetPixelFormat( PixelFormat::UINT16 );

      // FIXME: This is not accurate. We should either:
      // - Read the value from window/level to get better min,max value
      // - iterate over all possible value to find the min,max as we are looping
      // over all values anyway
      const double min = 0; // oripf.GetMin();
      const double max = 65536 - 1; //oripf.GetMax();
      r.SetSlope( 255. / (max - min + 0) ); // UINT8_MAX
      const double step = min * r.GetSlope();
      r.SetIntercept( 0 - step );

      // paranoid self check:
      assert( r.GetIntercept() + r.GetSlope() * min == 0. );
      assert( r.GetIntercept() + r.GetSlope() * max == 255. );

      r.SetTargetPixelType( PixelFormat::UINT8 );
      r.SetUseTargetPixelType(true);

      std::vector<char> v8;
      v8.resize( Internals->dims[0] * Internals->dims[1] * 3 );
      if( !r.Rescale(&v8[0],&s[0],s.size()) )
        {
        assert( 0 ); // should not happen in real life
        gdcmErrorMacro( "Problem in the rescaler" );
        return false;
        }
      if( Internals->ConvertRGBToPaletteColor )
        {
        LookupTable &lut = I->GetLUT();
        lut.Allocate();

        // re-encode:
        std::stringstream ss2;
        ss2.str( std::string( &v8[0], v8.size() ) );

        std::string s2 = ss2.str();
        // As per standard, we only support 8bits icon
        I->SetPixelFormat( PixelFormat::UINT8 );
        pixeldata.SetByteValue( &s2[0], (uint32_t)s2.size() );

        BuildLUT( *I, 256 );
        }
      else
        {
        I->SetPixelFormat( PixelFormat::UINT8 );
        I->GetPixelFormat().SetSamplesPerPixel( 3 );
        pixeldata.SetByteValue( &v8[0], (uint32_t)v8.size() );
        }
      }
    else
      {
      assert( I->GetPixelFormat() == PixelFormat::UINT8 );
      std::string s = ss.str();
      if( Internals->ConvertRGBToPaletteColor )
        {
        LookupTable &lut = I->GetLUT();
        lut.Allocate();

        // As per standard, we only support 8bits icon
        I->SetPixelFormat( PixelFormat::UINT8 );
        pixeldata.SetByteValue( &s[0], (uint32_t)s.size() );

        BuildLUT(*I, 256 );
        }
      else
        {
        I->SetPixelFormat( PixelFormat::UINT8 );
        I->GetPixelFormat().SetSamplesPerPixel( 3 );
        pixeldata.SetByteValue( &s[0], (uint32_t)s.size() );
        }
      }
    }
  else if( P->GetPhotometricInterpretation() == PhotometricInterpretation::RGB
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL_422
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_ICT
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_RCT )
    {
    std::string tempvbuf( &vbuffer2[0], vbuffer2.size() );
    if( P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL
    || P->GetPhotometricInterpretation() == PhotometricInterpretation::YBR_FULL_422 )
      {
      assert( I->GetPixelFormat() == PixelFormat::UINT8 );
      if( P->GetPlanarConfiguration() == 0 )
        {
        unsigned char *ybr = (unsigned char*)&tempvbuf[0];
        unsigned char *ybr_out = ybr;
        unsigned char *ybr_end = ybr + vbuffer2.size();
        int R, G, B;
        for( ; ybr != ybr_end; )
          {
          unsigned char a = (unsigned char)(*ybr); ++ybr;
          unsigned char b = (unsigned char)(*ybr); ++ybr;
          unsigned char c = (unsigned char)(*ybr); ++ybr;

          R = 38142 *(a-16) + 52298 *(c -128);
          G = 38142 *(a-16) - 26640 *(c -128) - 12845 *(b -128);
          B = 38142 *(a-16) + 66093 *(b -128);

          R = (R+16384)>>15;
          G = (G+16384)>>15;
          B = (B+16384)>>15;

          if (R < 0)   R = 0;
          if (G < 0)   G = 0;
          if (B < 0)   B = 0;
          if (R > 255) R = 255;
          if (G > 255) G = 255;
          if (B > 255) B = 255;

          *ybr_out = (unsigned char)R; ++ybr_out;
          *ybr_out = (unsigned char)G; ++ybr_out;
          *ybr_out = (unsigned char)B; ++ybr_out;
          }
#if 0
    std::ofstream d( "/tmp/d.rgb", std::ios::binary );
    d.write( &tempvbuf[0], tempvbuf.size() );
    d.close();
#endif
        assert( ybr_out == ybr_end );
        }
      else // ( P->GetPlanarConfiguration() == 1 )
        {
        std::string tempvbufybr = tempvbuf;

        unsigned char *ybr = (unsigned char*)&tempvbufybr[0];
        unsigned char *ybr_end = ybr + vbuffer2.size();
        assert( vbuffer2.size() % 3 == 0 );
        size_t ybrl = vbuffer2.size() / 3;
        unsigned char *ybra = ybr + 0 * ybrl;
        unsigned char *ybrb = ybr + 1 * ybrl;
        unsigned char *ybrc = ybr + 2 * ybrl;

        unsigned char *ybr_out = (unsigned char*)&tempvbuf[0];
        unsigned char *ybr_out_end = ybr_out + vbuffer2.size();
        int R, G, B;
        for( ; ybr_out != ybr_out_end; )
          {
          unsigned char a = (unsigned char)(*ybra); ++ybra;
          unsigned char b = (unsigned char)(*ybrb); ++ybrb;
          unsigned char c = (unsigned char)(*ybrc); ++ybrc;

          R = 38142 *(a-16) + 52298 *(c -128);
          G = 38142 *(a-16) - 26640 *(c -128) - 12845 *(b -128);
          B = 38142 *(a-16) + 66093 *(b -128);

          R = (R+16384)>>15;
          G = (G+16384)>>15;
          B = (B+16384)>>15;

          if (R < 0)   R = 0;
          if (G < 0)   G = 0;
          if (B < 0)   B = 0;
          if (R > 255) R = 255;
          if (G > 255) G = 255;
          if (B > 255) B = 255;

          *ybr_out = (unsigned char)R; ++ybr_out;
          *ybr_out = (unsigned char)G; ++ybr_out;
          *ybr_out = (unsigned char)B; ++ybr_out;
          }
        assert( ybra + 2 * ybrl == ybr_end ); (void)ybr_end;
        assert( ybrb + 1 * ybrl == ybr_end );
        assert( ybrc + 0 * ybrl == ybr_end );
        }
      }
    else
      {
      if( P->GetPlanarConfiguration() == 1 )
        {
        assert( I->GetPixelFormat() == PixelFormat::UINT8 );
        std::string tempvbufrgb = tempvbuf;

        unsigned char *rgb = (unsigned char*)&tempvbufrgb[0];
        unsigned char *rgb_end = rgb + vbuffer2.size();
        assert( vbuffer2.size() % 3 == 0 );
        size_t rgbl = vbuffer2.size() / 3;
        unsigned char *rgba = rgb + 0 * rgbl;
        unsigned char *rgbb = rgb + 1 * rgbl;
        unsigned char *rgbc = rgb + 2 * rgbl;

        unsigned char *rgb_out = (unsigned char*)&tempvbuf[0];
        unsigned char *rgb_out_end = rgb_out + vbuffer2.size();
        for( ; rgb_out != rgb_out_end; )
          {
          unsigned char a = (unsigned char)(*rgba); ++rgba;
          unsigned char b = (unsigned char)(*rgbb); ++rgbb;
          unsigned char c = (unsigned char)(*rgbc); ++rgbc;

          *rgb_out = a; ++rgb_out;
          *rgb_out = b; ++rgb_out;
          *rgb_out = c; ++rgb_out;
          }
        assert( rgba + 2 * rgbl == rgb_end ); (void)rgb_end;
        assert( rgbb + 1 * rgbl == rgb_end );
        assert( rgbc + 0 * rgbl == rgb_end );
        }
      }

    std::istringstream is( tempvbuf );
    if( I->GetPixelFormat() == PixelFormat::UINT8 )
      {
      std::string s = is.str();
      if( Internals->ConvertRGBToPaletteColor )
        {
        // As per standard, we only support 8bits icon
        I->SetPixelFormat( PixelFormat::UINT8 );
        pixeldata.SetByteValue( &s[0], (uint32_t)s.size() );

        BuildLUT(*I, 256 );
        }
      else
        {
        I->SetPixelFormat( PixelFormat::UINT8 );
        I->GetPixelFormat().SetSamplesPerPixel( 3 );
        pixeldata.SetByteValue( &s[0], (uint32_t)s.size() );
        }
      }
    else
      {
      assert( I->GetPixelFormat() == PixelFormat::UINT16 );
      assert( I->GetPixelFormat().GetPixelRepresentation() == 0 );
      std::string s = is.str();
      Rescaler r;
      r.SetPixelFormat( I->GetPixelFormat() );
      //r.SetPixelFormat( PixelFormat::UINT16 );

      // FIXME: This is not accurate. We should either:
      // - Read the value from window/level to get better min,max value
      // - iterate over all possible value to find the min,max as we are looping
      // over all values anyway
      const double min = 0; // oripf.GetMin();
      const double max = 65536 - 1; //oripf.GetMax();
      r.SetSlope( 255. / (max - min + 0) ); // UINT8_MAX
      const double step = min * r.GetSlope();
      r.SetIntercept( 0 - step );

      // paranoid self check:
      assert( r.GetIntercept() + r.GetSlope() * min == 0. );
      assert( r.GetIntercept() + r.GetSlope() * max == 255. );

      r.SetTargetPixelType( PixelFormat::UINT8 );
      r.SetUseTargetPixelType(true);

      std::vector<char> v8;
      v8.resize( Internals->dims[0] * Internals->dims[1] * 3 );
      if( !r.Rescale(&v8[0],&s[0],s.size()) )
        {
        assert( 0 ); // should not happen in real life
        gdcmErrorMacro( "Problem in the rescaler" );
        return false;
        }

      if( Internals->ConvertRGBToPaletteColor )
        {
        LookupTable &lut = I->GetLUT();
        lut.Allocate();

        I->SetPixelFormat( PixelFormat::UINT8 );
        pixeldata.SetByteValue( &v8[0], (uint32_t)v8.size() );

        BuildLUT(*I, 256 );
        }
      else
        {
        I->SetPixelFormat( PixelFormat::UINT8 );
        I->GetPixelFormat().SetSamplesPerPixel( 3 );
        pixeldata.SetByteValue( &v8[0], (uint32_t)v8.size() );
        }
      }
    }
  else
    {
    // MONOCHROME1 / MONOCHROME2 ...
    char *buffer2 = &vbuffer2[0];
    pixeldata.SetByteValue( buffer2, (uint32_t)vbuffer2.size() );

    Rescaler r;
    r.SetPixelFormat( I->GetPixelFormat() );

    // FIXME: This is not accurate. We should either:
    // - Read the value from window/level to get better min,max value
    // - iterate over all possible value to find the min,max as we are looping
    // over all values anyway
    double min = (double)oripf.GetMin();
    double max = (double)oripf.GetMax();
    if( Internals->UseMinMax )
      {
      min = Internals->Min;
      max = Internals->Max;
      }
    if( Internals->AutoMinMax )
      {
      const char *p = &vbuffer2[0];
      size_t len = vbuffer2.size();
      const PixelFormat &pf = I->GetPixelFormat();
      assert( pf.GetSamplesPerPixel() == 1 );
      if( Internals->UseOutsideValuePixel )
        {
        const double d = Internals->OutsideValuePixel;
        switch ( pf.GetScalarType() )
          {
        case PixelFormat::UINT8:
          ComputeMinMax<uint8_t>( (uint8_t*)p, len / sizeof( uint8_t ), min, max, d);
          break;
        case PixelFormat::INT8:
          ComputeMinMax<int8_t>( (int8_t*)p, len / sizeof( int8_t ), min, max, d);
          break;
        case PixelFormat::UINT16:
          ComputeMinMax<uint16_t>( (uint16_t*)p, len / sizeof( uint16_t ), min, max, d);
          break;
        case PixelFormat::INT16:
          ComputeMinMax<int16_t>( (int16_t*)p, len / sizeof( int16_t ), min, max, d);
          break;
        default:
          assert( 0 ); // should not happen
          break;
          }
        // ok we have found the min value, we should now be able to replace all value 'd' with this min now:
        switch ( pf.GetScalarType() )
          {
        case PixelFormat::UINT8:
          std::replace( (uint8_t*)p, (uint8_t*)p + len / sizeof( uint8_t ), (uint8_t)d, (uint8_t)min);
          break;
        case PixelFormat::INT8:
          std::replace( (int8_t*)p, (int8_t*)p + len / sizeof( int8_t ), (int8_t)d, (int8_t)min);
          break;
        case PixelFormat::UINT16:
          std::replace( (uint16_t*)p, (uint16_t*)p + len / sizeof( uint16_t ), (uint16_t)d, (uint16_t)min);
          break;
        case PixelFormat::INT16:
          std::replace( (int16_t*)p, (int16_t*)p + len / sizeof( int16_t ), (int16_t)d, (int16_t)min);
          break;
        default:
          assert( 0 ); // should not happen
          break;
          }
        }
      switch ( pf.GetScalarType() )
        {
      case PixelFormat::UINT8:
        ComputeMinMax<uint8_t>( (uint8_t*)p, len / sizeof( uint8_t ), min, max);
        break;
      case PixelFormat::INT8:
        ComputeMinMax<int8_t>( (int8_t*)p, len / sizeof( int8_t ), min, max);
        break;
      case PixelFormat::UINT16:
        ComputeMinMax<uint16_t>( (uint16_t*)p, len / sizeof( uint16_t ), min, max);
        break;
      case PixelFormat::INT16:
        ComputeMinMax<int16_t>( (int16_t*)p, len / sizeof( int16_t ), min, max);
        break;
      default:
        assert( 0 ); // should not happen
        break;
        }
      }
    r.SetSlope( 255. / (max - min + 0) ); // UINT8_MAX
    const double step = min * r.GetSlope();
    r.SetIntercept( 0 - step );

    // paranoid self check:
    assert( (int)(0.5 + r.GetIntercept() + r.GetSlope() * min) == 0 );
    assert( (int)(0.5 + r.GetIntercept() + r.GetSlope() * max) == 255 );

    r.SetTargetPixelType( PixelFormat::UINT8 );
    r.SetUseTargetPixelType(true);

    std::vector<char> v8;
    v8.resize( Internals->dims[0] * Internals->dims[1] );
    if( !r.Rescale(&v8[0],&vbuffer2[0],vbuffer2.size()) )
      {
      assert( 0 ); // should not happen in real life
      gdcmErrorMacro( "Problem in the rescaler" );
      return false;
      }

    // As per standard, we only support 8bits icon
    I->SetPixelFormat( PixelFormat::UINT8 );
    pixeldata.SetByteValue( &v8[0], (uint32_t)v8.size() );
    }

  // \postcondition
  if( !Internals->ConvertRGBToPaletteColor
    && I->GetPhotometricInterpretation() == PhotometricInterpretation::RGB )
    {
    assert( I->GetPixelFormat().GetSamplesPerPixel() == 3 );
    }
  else
    {
    assert( I->GetPixelFormat().GetSamplesPerPixel() == 1 );
    }
  assert( I->GetPixelFormat().GetBitsAllocated() == 8 );
  assert( I->GetPixelFormat().GetBitsStored() == 8 );
  assert( I->GetPixelFormat().GetHighBit() == 7 );
  assert( I->GetPixelFormat().GetPixelRepresentation() == 0 );

  return true;
}

} // end namespace gdcm
