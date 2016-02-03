/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmReader.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmFile.h"
#include "gdcmTesting.h"
#include "gdcmMediaStorage.h"
#include "gdcmSystem.h"
#include "gdcmDirectory.h"

static int TestReadUpToTag(const char* filename, bool verbose = false)
{
  if( verbose )
  std::cout << "TestRead: " << filename << std::endl;

  std::ifstream is( filename, std::ios::binary );
  gdcm::Reader reader;
  reader.SetStream( is );
  // Let's read up to Pixel Data el...
  gdcm::Tag pixeldata (0x7fe0,0x0010);
  std::set<gdcm::Tag> skiptags;
  // ... but do not read it (to skip mem allocation)
  skiptags.insert( pixeldata );
  if ( !reader.ReadUpToTag( pixeldata, skiptags) )
    {
    if( verbose )
      std::cerr << "TestReadError: Failed to read: " << filename << std::endl;
    return 1;
    }
  is.clear();
  std::streamoff outStreamOffset = is.tellg();

#if 0
  const gdcm::FileMetaInformation &h = reader.GetFile().GetHeader();
  const gdcm::DataSet &ds = reader.GetFile().GetDataSet();
  std::cout << ds << std::endl;
#endif

  if(verbose)
    std::cout << "{ \"" << filename << "\"," << outStreamOffset << " }," << std::endl;
  std::streamoff refoffset = gdcm::Testing::GetStreamOffsetFromFile(filename);
  if( refoffset != outStreamOffset && (refoffset || verbose) )
    {
    std::cerr << filename << ": " << outStreamOffset << " should be " << refoffset << std::endl;
    return 1;
    }
  is.close();

  return 0;
}

static int TestReadUpToTagExtra()
{
  const char *extradataroot = gdcm::Testing::GetDataExtraRoot();
  if( !extradataroot )
    {
    return 1;
    }
  if( !gdcm::System::FileIsDirectory(extradataroot) )
    {
    std::cerr << "No such directory: " << extradataroot <<  std::endl;
    return 1;
    }

  gdcm::Directory d;
  unsigned int nfiles = d.Load( extradataroot, true ); // no recursion
  std::cout << "done retrieving file list. " << nfiles << " files found." <<  std::endl;

  gdcm::Directory::FilenamesType const & fns = d.GetFilenames();
  int r = 0;
  for( gdcm::Directory::FilenamesType::const_iterator it = fns.begin();
    it != fns.end(); ++it )
    {
    const char *filename = it->c_str();
    r += TestReadUpToTag( filename );
    }

  return r;
}

int TestReaderUpToTag1(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestReadUpToTag(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestReadUpToTag( filename );
    ++i;
    }

  // puposely discard gdcmDataExtra test, this is just an 'extra' test...
  int b2 = TestReadUpToTagExtra(); (void)b2;

  return r;
}
