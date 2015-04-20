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
#include "gdcmSystem.h"

static int TestImageRegionRead(const char* filename, bool verbose = false)
{
  using gdcm::System;
  using gdcm::Testing;
  if( verbose )
    std::cerr << "Reading: " << filename << std::endl;
  gdcm::ImageRegionReader reader;

  reader.SetFileName( filename );
  bool canReadInformation = reader.ReadInformation();
  if (!canReadInformation)
    {
    std::cerr << "Cannot ReadInformation: " << filename << std::endl;
    return 1;
    }

  // Create directory first:
  const char subdir[] = "TestImageRegionReader4";
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );
  outfilename += ".raw";

  std::ofstream of( outfilename.c_str(), std::ios::binary );

  int res = 0;

  std::vector<unsigned int> dims =
    gdcm::ImageHelper::GetDimensionsValue(reader.GetFile());
  std::vector<char> vbuffer;
  gdcm::BoxRegion box;
  for( unsigned int z = 0; z < dims[2]; ++z )
    {
    box.SetDomain(0, dims[0] - 1, 0, dims[1] - 1, z, z);
    reader.SetRegion( box );
    size_t len = reader.ComputeBufferLength();
    if( !len )
      {
      std::cerr << "No length for: " << filename << std::endl;
      return 1;
      }
    vbuffer.resize( len );
    char* buffer = &vbuffer[0];
    bool b = reader.ReadIntoBuffer(buffer, len);
    if( !b )
      {
      std::cerr << "Could not ReadIntoBuffer" << std::endl;
      return 1;
      }
    of.write( buffer, len );
    }
  of.close();

  char digest[33];
  bool b = gdcm::Testing::ComputeFileMD5(outfilename.c_str(), digest);
  if( !b )
    {
    std::cerr << "Could not ComputeFileMD5: " << outfilename << std::endl;
    return 1;
    }

  const char *ref = gdcm::Testing::GetMD5FromFile(filename);
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
    std::cerr << "Into: " << outfilename << std::endl;
    std::cerr << "Found " << digest << " instead of " << ref << std::endl;
    res = 1;
    }

  return res;
}


int TestImageRegionReader4(int argc, char *argv[])
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
  const char *extradataroot = gdcm::Testing::GetDataExtraRoot();
  static const char *names[] = {
    "/gdcmSampleData/images_of_interest/US_512x512x2496_JPEG_BaseLine_Process_1.dcm",
    "/gdcmSampleData/images_of_interest/PHILIPS_Integris_V-10-MONO2-Multiframe.dcm",
    "/gdcmSampleData/ForSeriesTesting/MultiFramesSingleSerieXR/1.3.46.670589.7.5.10.80002138018.20001204.181556.9.1.1.dcm",
    "/gdcmSampleData/images_of_interest/i32.XADC.7.215MegaBytes.dcm",
    NULL
  };
  const char *filename;
  while( (filename = names[i]) )
    {
    std::string fn = extradataroot;
    fn += filename;
    r += TestImageRegionRead(fn.c_str());
    ++i;
    }

  return r;
}
