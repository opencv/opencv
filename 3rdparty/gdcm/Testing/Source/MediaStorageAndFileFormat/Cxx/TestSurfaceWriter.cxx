/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTesting.h"
#include "gdcmSurfaceWriter.h"
#include "gdcmSurfaceReader.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmAttribute.h"

namespace gdcm
{

int TestSurfaceWriter(const char *subdir, const char* filename)
{
  SurfaceReader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    int ret = 1;
    gdcm::MediaStorage ms;
    if( ms.SetFromFile( reader.GetFile() ) )
      {
      if( ms == gdcm::MediaStorage::SurfaceSegmentationStorage )
        {
        ret = 0;
        }
      }
    if( !ret )
      {
      std::cerr << "Failed to read: " << filename << std::endl;
      std::cerr << "MediaStorage is: " << ms.GetString() << std::endl;
      }
    return !ret;
    }

  // Get content of filename
  const File &                FReader   = reader.GetFile();
  const FileMetaInformation & fmiReader = FReader.GetHeader();
  const DataSet             & dsReader  = FReader.GetDataSet();

  // Create directory first:
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );

  // Write file from content reader
  SurfaceWriter writer;
  writer.SetFileName( outfilename.c_str() );
  SegmentReader::SegmentVector segments = reader.GetSegments();
  writer.SetSegments( segments );
  writer.SetNumberOfSurfaces( reader.GetNumberOfSurfaces() );

  // Set content from filename
  File & FWriter = writer.GetFile();
  FWriter.SetHeader(fmiReader);
  FWriter.SetDataSet(dsReader);

  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }

  // reuse the filename, since outfilename is simply the new representation
  // of the old filename
  const char * ref        = Testing::GetMD5FromFile(filename);
  char         digest[33] = {};
  Testing::ComputeFileMD5(outfilename.c_str(), digest);
  if( ref == 0 )
    {
    // new regression file needs a md5 sum
    std::cout << "Missing md5 " << digest << " for: " << outfilename <<  std::endl;
    return 1;
    }
  else if( strcmp(digest, ref) != 0 )
    {
    std::cerr << "Found " << digest << " instead of " << ref << std::endl;
    return 1;
    }

  return 0;
}

}

int TestSurfaceWriter(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestSurfaceWriter(argv[0],filename);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += gdcm::TestSurfaceWriter(argv[0], filename );
    ++i;
    }

  return r;
}
