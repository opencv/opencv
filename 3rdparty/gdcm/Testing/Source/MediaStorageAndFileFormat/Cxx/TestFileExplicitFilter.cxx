/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileExplicitFilter.h"
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"

int TestFileExplicitFilt(const char *subdir, const char *filename, bool verbose = false)
{
  if( verbose )
    std::cerr << "Reading: " << filename << std::endl;
  gdcm::Reader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    std::cerr << "could not read: " << filename << std::endl;
    return 1;
    }
  //const gdcm::FileMetaInformation &h = reader.GetFile().GetHeader();
  //const gdcm::DataSet &ds = reader.GetFile().GetDataSet();

  gdcm::FileExplicitFilter im2ex;
  im2ex.SetFile( reader.GetFile() );
  if( !im2ex.Change() )
    {
    std::cerr << "Could not im2ex change: " << filename << std::endl;
    return 1;
    }

  // Create directory first:
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = gdcm::Testing::GetTempFilename( filename, subdir );
  gdcm::Writer writer;
  writer.SetFileName( outfilename.c_str() );
  writer.SetFile( reader.GetFile() );
  writer.SetCheckFileMetaInformation( false );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }

  if(verbose)
    std::cerr << "write out: " << outfilename << std::endl;

  char digest1[33] = {};
  char digest2[33] = {};
  bool b1 = gdcm::Testing::ComputeFileMD5(filename, digest1);
  if( !b1 )
    {
    std::cerr << "Could not compute md5:" << filename << std::endl;
    return 1;
    }
  bool b2 = gdcm::Testing::ComputeFileMD5(outfilename.c_str(), digest2);
  if( !b2 )
    {
    std::cerr << "Could not compute md5:" << outfilename << std::endl;
    return 1;
    }
  if( strcmp(digest1, digest2 ) == 0 )
    {
    // Easy case input file was explicit
    return 0;
    }
  else
    {
    if(verbose)
      {
      std::cerr << "input file contained wrong VR: " << filename << std::endl;
      std::cerr << "see: " << outfilename << std::endl;
      }
    }

  return 0;
}

int TestFileExplicitFilter(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestFileExplicitFilt(argv[0], filename, true);
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
    r += TestFileExplicitFilt( argv[0], filename );
    ++i;
    }

  return r;
}
