/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageApplyLookupTable.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#include "gdcmImage.h"
#include "gdcmFilename.h"

static const char * const lutarray[][2] = {
    { "d613050ca0f9c924fb5282d140281fcc", "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
    { "d613050ca0f9c924fb5282d140281fcc", "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
    { "7b8d795eaf99f1fff176c43f9cf76bfb", "NM-PAL-16-PixRep1.dcm" },
    { "47715f0a5d5089268bbef6f83251a8ad", "OT-PAL-8-face.dcm" },
    { "c70309b66045140b8e08c11aa319c0ab", "US-PAL-8-10x-echo.dcm" },
    { "c370ca934dc910eb4b629a2fa8650b67", "gdcm-US-ALOKA-16.dcm" },
    { "49ca8ad45fa7f24b0406a5a03ba8aff6", "rle16loo.dcm" },
    { "964ea27345a7004325896d34b257f289", "rle16sti.dcm" },

    // sentinel
    { 0, 0 }
};

int TestImageApplyLookupTableFunc(const char *filename, bool verbose = false)
{
  gdcm::ImageReader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    const gdcm::FileMetaInformation &header = reader.GetFile().GetHeader();
    gdcm::MediaStorage ms = header.GetMediaStorage();
    bool isImage = gdcm::MediaStorage::IsImage( ms );
    bool pixeldata = reader.GetFile().GetDataSet().FindDataElement( gdcm::Tag(0x7fe0,0x0010) );
    if( isImage && pixeldata )
      {
      std::cout << "Could not read: " << filename << std::endl;
      return 1;
      }
    return 0;
    }
  const gdcm::Image &image = reader.GetImage();

  const gdcm::PhotometricInterpretation &pi = image.GetPhotometricInterpretation();
  if( pi != gdcm::PhotometricInterpretation::PALETTE_COLOR )
    {
    // yeah well not much I can do here...
    if( verbose )
      {
      std::cout << "PhotometricInterpretation is: " << pi << " cannot apply LUT then..." << std::endl;
      }
    return 0;
    }

  gdcm::ImageApplyLookupTable lutfilt;
  lutfilt.SetInput( image );
  bool b = lutfilt.Apply();
  if( !b )
    {
    std::cerr << "Could not apply lut: " << filename << std::endl;
    return 1;
    }

  // Create directory first:
  const char subdir[] = "TestImageApplyLookupTable";
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = gdcm::Testing::GetTempFilename( filename, subdir );

  gdcm::ImageWriter writer;
  writer.SetFileName( outfilename.c_str() );
  //writer.SetFile( reader.GetFile() ); // increase test goal
  writer.SetImage( lutfilt.GetOutput() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }

  // Let's read that file back in !
  gdcm::ImageReader reader2;
  reader2.SetFileName( outfilename.c_str() );
  if ( !reader2.Read() )
    {
    std::cerr << "Failed to read back: " << outfilename << std::endl;
    return 1;
    }

  const gdcm::Image &img = reader2.GetImage();
  unsigned long len = img.GetBufferLength();
  char* buffer = new char[len];
  img.GetBuffer(buffer);

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();
  unsigned int i = 0;
  const char *p = lutarray[i][1];
  while( p != 0 )
    {
    if( strcmp( name, p ) == 0 )
      {
      break;
      }
    ++i;
    p = lutarray[i][1];
    }
  const char *ref = lutarray[i][0];

  char digest[33] = {};
  gdcm::Testing::ComputeMD5(buffer, len, digest);
  int res = 0;
  if( !ref )
    {
    std::cerr << "Missing LUT-applied MD5 for image from: " << filename << std::endl;
    res = 1;
    }
  else if( strcmp(digest, ref) )
    {
    std::cerr << "Problem reading image from: " << filename << std::endl;
    std::cerr << "Found " << digest << " instead of " << ref << std::endl;
    res = 1;
    }
  delete[] buffer;

  return res;
}

int TestImageApplyLookupTable(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestImageApplyLookupTableFunc(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestImageApplyLookupTableFunc( filename );
    ++i;
    }

  return r;
}
