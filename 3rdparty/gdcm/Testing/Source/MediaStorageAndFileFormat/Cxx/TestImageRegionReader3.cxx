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
  using gdcm::System;
  using gdcm::Testing;

  if( verbose )
    std::cerr << "Reading: " << filename << std::endl;

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

  gdcm::ImageRegionReader reader;
  reader.SetFileName( filename );
  bool canReadInformation = reader.ReadInformation();
  if (!canReadInformation)
    {
    std::cerr << "Cannot ReadInformation: " << filename << std::endl;
    return 0; //unable to read tags as expected.
    }

  // Create directory first:
  const char subdir[] = "TestImageRegionReader3";
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
    const unsigned int xdelta = dims[0] / 1;
    const unsigned int ydelta = dims[1] / 4;
    size_t zlen;
    bool b;

    for( unsigned int y = 0; y < 4; ++y )
      {
      unsigned int maxy = ydelta + y * ydelta;
      if( y == 3 ) maxy = dims[1];
      box.SetDomain(0, xdelta - 1, 0 + y * ydelta, (unsigned int)(maxy - 1), z, z);
      reader.SetRegion(box);
      zlen = reader.ComputeBufferLength();
      assert( zlen );
      vbuffer.resize( zlen );
      char* buffer = &vbuffer[0];
      b = reader.ReadIntoBuffer(buffer, zlen);
      if( !b ) return 1;
      assert( zlen );
      of.write( buffer, zlen );
      }
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
    std::cerr << "Found " << digest << " instead of " << ref << std::endl;
    res = 1;
    }

  return res;
}


int TestImageRegionReader3(int argc, char *argv[])
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
