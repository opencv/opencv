/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageRegionReader.h"
#include "gdcmImageHelper.h"
#include "gdcmFilename.h"
#include "gdcmBoxRegion.h"

#include "gdcmTesting.h"

static int TestImageRegionRead(const char* filename, bool verbose = false)
{
  if( verbose )
    std::cerr << "Reading: " << filename << std::endl;
  gdcm::ImageRegionReader reader;

  gdcm::Filename fn( filename );
  // DMCPACS_ExplicitImplicit_BogusIOP.dcm is very difficult to handle since
  // we need to read 3 attribute to detect the "well known" bug. However the third
  // attribute is (b500,b700) which make ReadUpToTag(7fe0,0010) fails...
  if( strcmp(fn.GetName(), "DMCPACS_ExplicitImplicit_BogusIOP.dcm" ) == 0
    || strcmp(fn.GetName(), "SC16BitsAllocated_8BitsStoredJ2K.dcm" ) == 0 // mismatch pixel format in JPEG 200 vs DICOM
    || strcmp(fn.GetName(), "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" ) == 0 ) // bogus JPEG cannot be streamed
    {
    std::cerr << "Skipping impossible file: " << filename << std::endl;
    return 0;
    }

  reader.SetFileName( filename );
  bool canReadInformation = reader.ReadInformation();
  if (!canReadInformation)
    {
    //std::cerr << "Cannot ReadInformation: " << filename << std::endl;
    return 0; //unable to read tags as expected.
    }

  int res = 0;

  //std::cout << reader.GetFile().GetDataSet() << std::endl;
  std::vector<unsigned int> dims =
    gdcm::ImageHelper::GetDimensionsValue(reader.GetFile());
  gdcm::BoxRegion box;
  box.SetDomain(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1);
  reader.SetRegion( box );
  size_t len = reader.ComputeBufferLength();
  if( !len )
    {
    std::cerr << "No length for: " << filename << std::endl;
    return 1;
    }
  std::vector<char> vbuffer;
  vbuffer.resize( len );
  char* buffer = &vbuffer[0];
  bool b = reader.ReadIntoBuffer(buffer, len);
  if( !b )
    {
    std::cerr << "Could not ReadIntoBuffer: " << filename << std::endl;
    return 1;
    }
#if 0
  std::ofstream of( "/tmp/dd.raw", std::ios::binary );
  of.write( buffer, len );
  of.close();
#endif

  const char *ref = gdcm::Testing::GetMD5FromFile(filename);

  char digest[33];
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
    res = 1;
    }
  else if( strcmp(digest, ref) )
    {
    std::cerr << "Problem reading image from: " << filename << std::endl;
    std::cerr << "Found " << digest << " instead of " << ref << std::endl;
    res = 1;
    }

  return res;
}


int TestImageRegionReader1(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestImageRegionRead(filename, true);
    }

  // else
  // First of get rid of warning/debug message
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestImageRegionRead(filename);
    ++i;
    }

  return r;
}
