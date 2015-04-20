/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmRescaler.h"

#include <vector>
#include <limits>

template < typename T >
struct TypeToPixelFormat;

template<> struct TypeToPixelFormat<double> { enum { Type = gdcm::PixelFormat::FLOAT64 }; };
template<> struct TypeToPixelFormat<float> { enum { Type = gdcm::PixelFormat::FLOAT32 }; };
template<> struct TypeToPixelFormat<short> { enum { Type = gdcm::PixelFormat::INT16 }; };
template<> struct TypeToPixelFormat<unsigned short> { enum { Type = gdcm::PixelFormat::UINT16 }; };
template<> struct TypeToPixelFormat<unsigned char> { enum { Type = gdcm::PixelFormat::UINT8 }; };
template<> struct TypeToPixelFormat<char> { enum { Type = gdcm::PixelFormat::INT8 }; };
template<> struct TypeToPixelFormat<int> { enum { Type = gdcm::PixelFormat::INT32 }; };

template <typename input_pixel, typename output_pixel>
bool TestRescaler2Func(
  const double intercept,
  const double slope,
  const unsigned short bitsallocated,
  const unsigned short bitsstored,
  const unsigned short pixelrepresentation,
  const bool best_fit
)
{
  const gdcm::PixelFormat pixeltype(1,bitsallocated,bitsstored,
    (unsigned short)(bitsstored-1),pixelrepresentation);

  uint8_t ps = pixeltype.GetPixelSize();
  // programmer error:
  if( ps != sizeof( input_pixel ) ) return false;
  if( pixelrepresentation ) // signed
    {
    if( std::numeric_limits<input_pixel>::min() == 0 ) return false;
    }
  else
    {
    if( std::numeric_limits<input_pixel>::min() != 0 ) return false;
    }

  gdcm::Rescaler r;
  r.SetIntercept( intercept );
  r.SetSlope( slope );
  r.SetPixelFormat( pixeltype );

  const int64_t min = pixeltype.GetMin();
  const int64_t max = pixeltype.GetMax();

  gdcm::PixelFormat::ScalarType outputpt;
  outputpt = r.ComputeInterceptSlopePixelType();

  gdcm::PixelFormat::ScalarType targetpixeltype =
    (gdcm::PixelFormat::ScalarType)TypeToPixelFormat<output_pixel>::Type;
  if( best_fit )
    {
    if( targetpixeltype != outputpt )
      {
      return false;
      }
    }
  else
    {
    if( targetpixeltype == outputpt )
      {
      return false;
      }
    }
  r.SetTargetPixelType( targetpixeltype );
  r.SetUseTargetPixelType(true);

  std::vector<input_pixel> values;
  const int nvalues = 1 << bitsstored;
  values.reserve( nvalues );
  for( int i = 0; i < nvalues; ++i )
    {
    //values.push_back( (input_pixel)(std::numeric_limits<input_pixel>::min() + i) );
    values.push_back( (input_pixel)(min + i) );
    }

  std::vector<output_pixel> output;
  output.resize( nvalues );
  if( !r.Rescale((char*)&output[0],(char*)&values[0],nvalues * sizeof( values[0] ) ) )
    return false;

  // get the min/max:
  double min2 = (double)output[0];
  double max2 = (double)output[nvalues-1];
  if( min2 > max2 ) return false;

  if( intercept == 0 && slope == 1 )
    {
    if( min != min2 ) return false;
    if( max != max2 ) return false;
    }

  gdcm::Rescaler ir;
  ir.SetIntercept( intercept );
  ir.SetSlope( slope );
  ir.SetPixelFormat( targetpixeltype );
  ir.SetMinMaxForPixelType( min2, max2 );
  const gdcm::PixelFormat pf2 = ir.ComputePixelTypeFromMinMax();

  if( pf2 != pixeltype ) return false;

  std::vector<input_pixel> check;
  check.resize( nvalues );

  if( check == values ) // dummy check
    return false;

  if( !ir.InverseRescale((char*)&check[0],(char*)&output[0], nvalues * sizeof( output[0] ) ) )
    return false;

  if( check != values )
    return false;

  return true;
}

int TestRescaler2(int, char *[])
{
    {
    double intercept = 0.000061;
    double slope     = 3.774114;
    // Case 1.
    // gdcmData/MR-MONO2-12-shoulder.dcm
    // (0028,0100) US 16         # 2,1 Bits Allocated
    // (0028,0101) US 12         # 2,1 Bits Stored
    // (0028,0102) US 11         # 2,1 High Bit
    // (0028,0103) US 0          # 2,1 Pixel Representation
    // [...]
    // (0028,1052) DS [0.000061] # 8,1 Rescale Intercept
    // (0028,1053) DS [3.774114] # 8,1 Rescale Slope
    if( !TestRescaler2Func<unsigned short,double>(intercept, slope,16,12,0,true) ) return 1;

    // Case 2.
    // use float instead of default double
    if( !TestRescaler2Func<unsigned short,float>(intercept, slope,16,12,0,false) ) return 1;
    }

    {
    double intercept = 0;
    double slope     = 1;

    // Case 3.
    if( !TestRescaler2Func<unsigned short,double>(intercept, slope,16,12,0,false) ) return 1;
    // Case 4.
    if( !TestRescaler2Func<unsigned short,float>(intercept, slope,16,12,0,false) ) return 1;
    // Case 5. best fit
    if( !TestRescaler2Func<unsigned short,unsigned short>(intercept, slope,16,12,0,true) ) return 1;
    // Case 6. unsigned char:
    if( !TestRescaler2Func<unsigned char,unsigned char>(intercept,slope,8,8,0,true) ) return 1;
    // Case 7. char
    if( !TestRescaler2Func<unsigned char,char>(intercept,slope,8,7,0,false) ) return 1;
    }

    {
    double intercept = 0;
    double slope     = 1;

    // Case 8.
    if( !TestRescaler2Func<short,double>(intercept,slope,16,12,1,false) ) return 1;
    // Case 9.
    if( !TestRescaler2Func<short,float>(intercept, slope,16,12,1,false) ) return 1;
    // Case 10. best fit
    if( !TestRescaler2Func<short,short>(intercept, slope,16,12,1,true) ) return 1;
    // Case 11. unsigned char:
    if( !TestRescaler2Func<char,char>(intercept,slope,8,8,1,true) ) return 1;
    // Case 12. char
    if( !TestRescaler2Func<char,char>(intercept,slope,8,7,1,true) ) return 1;
    }

    {
    double intercept = -1024;
    double slope     = 1;

    // Case 13.
    if( !TestRescaler2Func<unsigned short,short>(intercept, slope,16,12,0,true) ) return 1;
    // Case 14.
    if( !TestRescaler2Func<unsigned short,int>(intercept, slope,16,12,0,false) ) return 1;
    // Case 15.
    if( !TestRescaler2Func<unsigned char,short>(intercept, slope,8,8,0,true) ) return 1;
    // Case 16.
    if( !TestRescaler2Func<unsigned char,int>(intercept, slope,8,8,0,false) ) return 1;
    }

  return 0;
}
