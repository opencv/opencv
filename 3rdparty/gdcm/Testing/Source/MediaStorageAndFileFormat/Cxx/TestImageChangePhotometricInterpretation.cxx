/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageChangePhotometricInterpretation.h"
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"

namespace gdcm
{
PhotometricInterpretation InvertPI(PhotometricInterpretation pi)
{
  assert( pi == PhotometricInterpretation::MONOCHROME1 || pi == PhotometricInterpretation::MONOCHROME2 );
  if( pi == PhotometricInterpretation::MONOCHROME1 )
    {
    return PhotometricInterpretation::MONOCHROME2;
    }
  return PhotometricInterpretation::MONOCHROME1;
}
}

int TestImageChangePhotometricInterpretationFunc(const char *filename, bool verbose = false)
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

  //unsigned int pc = image.GetPlanarConfiguration();
  gdcm::PhotometricInterpretation pi = image.GetPhotometricInterpretation();
  if( pi != gdcm::PhotometricInterpretation::MONOCHROME1 && pi != gdcm::PhotometricInterpretation::MONOCHROME2 )
    {
    // nothing to do:
    return 0;
    }
  gdcm::PhotometricInterpretation invert_pi = gdcm::InvertPI(pi);

  gdcm::ImageChangePhotometricInterpretation pcfilt;
  pcfilt.SetInput( image );
  pcfilt.SetPhotometricInterpretation( invert_pi );
  bool b = pcfilt.Change();
  if( !b )
    {
    std::cerr << "Could not apply pcfilt: " << filename << std::endl;
    return 1;
    }

  // Create directory first:
  const char subdir[] = "TestImageChangePhotometricInterpretation";
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
  writer.SetImage( pcfilt.GetOutput() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }

  return 0;
}

int TestImageChangePhotometricInterpretation(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestImageChangePhotometricInterpretationFunc(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestImageChangePhotometricInterpretationFunc( filename );
    ++i;
    }

  return r;
}
