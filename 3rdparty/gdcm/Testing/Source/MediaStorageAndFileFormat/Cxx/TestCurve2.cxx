/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCurve.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmSystem.h"
#include "gdcmImageReader.h"
#include "gdcmFilename.h"
#include "gdcmByteSwap.h"
#include "gdcmTrace.h"
#include "gdcmTesting.h"

struct curveinfo
{
  unsigned short dimensions;
  unsigned short numpts;
  const char *typeofdata;
  unsigned short datavaluerepresentation;
  const char *datamd5;
};

static const curveinfo numptsarray1[] = { { 1, 1126, "PHYSIO", 0, "0fee912671ae158390efc7b49fe39f9b" } };
static const curveinfo numptsarray2[] = { { 1, 969, "PHYSIO", 0, "b46d2c6eed2944f1e16c46125022d2d4" } };
static const curveinfo numptsarray3[] = { { 2, 1864, "ECG ", 0, "22aadb1260fcb53a487a2e9e9ee445b3" } };
static const curveinfo numptsarray4[] = { { 2, 1590, "PRESSURE", 0, "6b4125d8a00c65feddb8c7c6f157e6ce" } , { 2, 1588, "ECG ", 0, "7a8e1cf198e74dd37738dbce5262c49d" } };

struct curveel
{
  const char *name;
  size_t numcurves;
  const curveinfo *info;
};

static const curveel arraycurve[] = {
// gdcmData
{ "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm", 1, numptsarray1 },
{ "GE_DLX-8-MONO2-Multiframe.dcm", 1, numptsarray2 },
// gdcmDataExtra
{ "xa_integris.dcm", 1, numptsarray3 },
// random stuff:
{ "XA.1.2.826.0.1.3680043.3.29.1.3230389164.20272.1340974735.2.3.0.000001.dcm", 2, numptsarray4 },
};

static const curveel *getcurveelfromname(const char *filename)
{
  static const size_t nel = sizeof( arraycurve ) / sizeof( *arraycurve );
  for( size_t i = 0; i < nel; ++i )
    {
    const curveel &c = arraycurve[i];
    if( strcmp( filename, c.name) == 0 )
      {
      return &c;
      }
    }
  return NULL;
}

static int TestCurve2Read(const char* filename, bool verbose = false)
{
  if( verbose )
    std::cerr << "Reading: " << filename << std::endl;
  gdcm::ImageReader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    }
  int res = 0;
  const gdcm::Image &img = reader.GetImage();
  size_t numcurves = img.GetNumberOfCurves();
  gdcm::Filename fn( filename );
  if( numcurves )
    {
    const curveel *c = getcurveelfromname( fn.GetName() );
    if( c == NULL )
      {
      std::cerr << "Cant find: " << filename << std::endl;
      return 1;
      }
    if( c->numcurves != numcurves )
      {
      std::cerr << "Should be: " << numcurves << " while " << c->numcurves << std::endl;
      return 1;
      }
    const curveinfo *info = c->info;
    for( size_t idx = 0; idx < numcurves; ++idx )
      {
      const gdcm::Curve &curve = img.GetCurve(idx);
      //curve.Print( std::cout );
      unsigned short dim = curve.GetDimensions();
      if( info[idx].dimensions != dim )
        {
        std::cerr << "Should be: " << dim << " while " << info[idx].dimensions << " for idx: " << idx << std::endl;
        return 1;
        }
      unsigned short npts = curve.GetNumberOfPoints();
      if( info[idx].numpts != npts )
        {
        std::cerr << "Should be: " << npts << " while " << info[idx].numpts << " for idx: " << idx << std::endl;
        return 1;
        }
      const char *tofdata = curve.GetTypeOfData();
      if( strcmp(info[idx].typeofdata, tofdata ) != 0 )
        {
        std::cerr << "Should be: [" << tofdata << "] while [" << info[idx].typeofdata << "] for idx: " << idx << std::endl;
        return 1;
        }
      unsigned short dvr = curve.GetDataValueRepresentation();
      if( info[idx].datavaluerepresentation != dvr )
        {
        std::cerr << "Should be: " << dvr << " while " << info[idx].datavaluerepresentation << " for idx: " << idx << std::endl;
        return 1;
        }
      std::vector<float> points;
      points.resize( 3 * npts );
      curve.GetAsPoints( (float*)&points[0] );
#if 0
      for( size_t i = 0; i < npts; i += 3 )
        {
        std::cout << points[i + 0] << ","
          << points[i + 1] << ","
          << points[i + 2] << "\n";
        }
#endif
      char digest[33];
      const char *buffer = (char*)&points[0];
      size_t len = sizeof(float) * 3 * npts;
      const char *ref = info[idx].datamd5;
      gdcm::Testing::ComputeMD5(buffer, len, digest);
      if( verbose )
        {
        std::cout << "ref=" << ref << std::endl;
        std::cout << "md5=" << digest << std::endl;
        }
      if( !ref )
        {
        // new regression image needs a md5 sum
        std::cout << "Missing md5 " << digest << " for: " << filename <<  std::endl;
        //assert(0);
        res = 1;
        }
      else if( strcmp(digest, ref) )
        {
        std::cerr << "Problem reading image from: " << filename << std::endl;
        std::cerr << "Found " << digest << " instead of " << ref << std::endl;
        res = 1;
        }


      }
    }

  return res;
}

int TestCurve2(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestCurve2Read(filename, true);
    }

  // else
  // First of get rid of warning/debug message
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestCurve2Read(filename);
    ++i;
    }

  return r;
}
