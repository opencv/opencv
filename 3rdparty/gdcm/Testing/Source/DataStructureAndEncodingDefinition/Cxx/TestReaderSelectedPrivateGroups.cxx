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

int TestReadSelectedPrivateGroups(const char* filename, bool verbose = false)
{
  if( verbose )
    std::cout << "TestRead: " << filename << std::endl;

  std::ifstream is( filename, std::ios::binary );
  gdcm::Reader reader;
  reader.SetStream( is );
  gdcm::PrivateTag group9(0x9,0x10,"GEMS_IDEN_01");
  std::set<gdcm::PrivateTag> selectedtags;
  selectedtags.insert ( group9 );
  if ( !reader.ReadSelectedPrivateTags( selectedtags ) )
    {
    std::cerr << "TestReadSelectedGroups: Failed to read: " << filename << std::endl;
    return 1;
    }

  std::streamoff outStreamOffset = is.tellg();

  gdcm::File & file = reader.GetFile();
  gdcm::DataSet & ds = file.GetDataSet();
  if( verbose )
    std::cout << ds << std::endl;
  const bool found = ds.FindDataElement(group9);

  if(verbose)
    std::cout << "{ \"" << filename << "\"," << outStreamOffset << " }," << std::endl;
  std::streamoff refoffset = gdcm::Testing::GetSelectedPrivateGroupOffsetFromFile(filename);
  if( refoffset != outStreamOffset )
    {
    std::cerr << filename << ": " << outStreamOffset << " should be " << refoffset << " found: " << found << std::endl;
    return 1;
    }
  is.close();

  return 0;
}


int TestReaderSelectedPrivateGroups(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestReadSelectedPrivateGroups(filename, true);
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
    r += TestReadSelectedPrivateGroups( filename );
    ++i;
    }

  return r;
}
