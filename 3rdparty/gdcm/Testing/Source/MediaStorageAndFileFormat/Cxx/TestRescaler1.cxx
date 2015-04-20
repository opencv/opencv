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
#include <limits>

#include <stdlib.h> // atof

static bool check_roundtrip(const gdcm::PixelFormat & pf )
{
  gdcm::Rescaler r;
  r.SetIntercept( 0. );
  r.SetSlope( 1. );
  r.SetPixelFormat( pf );
  r.SetMinMaxForPixelType((double)pf.GetMin(),(double)pf.GetMax());
  const gdcm::PixelFormat outputpt = r.ComputePixelTypeFromMinMax();
  if( outputpt != pf ) return false;
  return true;
}

int TestRescaler1(int, char *[])
{
  gdcm::Rescaler ir;

  /*
gdcmData/MR-MONO2-12-shoulder.dcm
(gdb) p intercept
$1 = 6.0999999999999999e-05
(gdb) p slope
$2 = 3.774114
(gdb) p in[i]
$3 = 3.77417493
...
p (in[i] - intercept)/slope
$7 = 0.99999998109891775

$10 = {Intercept = 6.0999999999999999e-05, Slope = 3.774114, PF = {SamplesPerPixel = 1, BitsAllocated = 32, BitsStored = 32, HighBit = 31, PixelRepresentation = 3}, ScalarRangeMin = 6.0999998822808266e-05,
  ScalarRangeMax = 247336.561051}

*/

  // (0028,1052) DS [0.000061]                               #   8, 1 RescaleIntercept
  // (0028,1053) DS [3.774114]                               #   8, 1 RescaleSlope

  const double intercept = atof( "0.000061" );
  const double slope     = atof( "3.774114" );
  ir.SetIntercept( intercept );
  ir.SetSlope( slope );
  ir.SetPixelFormat( gdcm::PixelFormat::FLOAT64 );
  const double smin = 6.0999998822808266e-05;
  const double smax = 247336.561051;
  ir.SetMinMaxForPixelType( smin, smax );

  double outref[] = { 0 };
    {
    char *copy = (char*)outref;
    const uint16_t in[] = { 1 };
    const char *tempimage = (char*)in;
    size_t vtklen = sizeof(in);
    ir.SetPixelFormat( gdcm::PixelFormat::UINT16 );
    bool b = ir.Rescale(copy,tempimage,vtklen);
    if( !b ) return 1;

    std::cout << outref[0] << std::endl;
    }

  ir.SetPixelFormat( gdcm::PixelFormat::FLOAT64 );
  uint16_t out[] = { 0 };
  char *copy = (char*)out;
  //const double in[] = { 3.77417493 };
  const double in[] = { 3.774175 };
  if( outref[0] != in[0] )
    {
    std::cerr << "Wrong input/output:" << std::endl;
    std::cerr << outref[0] << " vs " << in[0] << std::endl;
    std::cerr << (outref[0] - in[0]) << std::endl;
    return 1;
    }
  const char *tempimage = (char*)in;
  size_t vtklen = sizeof(in);
  ir.InverseRescale(copy,tempimage,vtklen);

  std::cout << out[0] << std::endl;
  if( out[0] != 1 )
    {
    return 1;
    }

  // Let's make sure that rescaler works in the simpliest case
  // it should be idempotent:
{
  gdcm::PixelFormat pixeltype = gdcm::PixelFormat::INT16;
  gdcm::Rescaler r;
  r.SetIntercept( 0.0 );
  r.SetSlope( 1.0 );
  r.SetPixelFormat( pixeltype );
  gdcm::PixelFormat::ScalarType outputpt;
  outputpt = r.ComputeInterceptSlopePixelType();

  if( outputpt != pixeltype )
    {
    return 1;
    }
  if( ! (outputpt == pixeltype) )
    {
    return 1;
    }
}

{
  gdcm::PixelFormat::ScalarType outputpt ;
  double shift = -1024;
  double scale = 1;
  // gdcmData/CT-MONO2-16-ort.dcm
  gdcm::PixelFormat pixeltype( 1, 16, 16, 15, 1 );
  gdcm::Rescaler r;
  r.SetIntercept( shift );
  r.SetSlope( scale );
  r.SetPixelFormat( pixeltype );
  outputpt = r.ComputeInterceptSlopePixelType();
  // min,max = [-33792, 31743]
  // we need at least int32 to store that
  if( outputpt != gdcm::PixelFormat::INT32 )
    {
    return  1;
    }
  // let's pretend image is really the full range:
  // FIXME: I think it is ok to compute this way since shift is double anyway:
  r.SetMinMaxForPixelType(std::numeric_limits<int16_t>::min() + shift,std::numeric_limits<int16_t>::max() + shift );

  gdcm::PixelFormat pf2 = r.ComputePixelTypeFromMinMax();
  if( pf2 != pixeltype )
    {
    return 1;
    }
}

// ComputePixelTypeFromMinMax()
{
  if( !check_roundtrip(gdcm::PixelFormat(1,16,12,11,0) ) ) return 1;
  if( !check_roundtrip(gdcm::PixelFormat(1,16,12,11,1) ) ) return 1;
  if( !check_roundtrip(gdcm::PixelFormat(1,8,8,7,0) ) ) return 1;
  if( !check_roundtrip(gdcm::PixelFormat(1,8,8,7,1) ) ) return 1;
}


  return 0;
}
