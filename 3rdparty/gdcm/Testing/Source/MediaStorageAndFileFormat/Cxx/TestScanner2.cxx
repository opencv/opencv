/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmScanner.h"
#include "gdcmDirectory.h"
#include "gdcmSystem.h"
#include "gdcmTesting.h"
#include "gdcmTrace.h"

int TestScanner2(int argc, char *argv[])
{
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();

  const char *directory = gdcm::Testing::GetDataRoot();
  if( argc == 2 )
    {
    directory = argv[1];
    }

  if( !gdcm::System::FileIsDirectory(directory) )
    {
    std::cerr << "No such directory: " << directory <<  std::endl;
    return 1;
    }

  gdcm::Directory d;
  unsigned int nfiles = d.Load( directory ); // no recursion
  std::cout << "done retrieving file list. " << nfiles << " files found." <<  std::endl;

  gdcm::Scanner s;
  const gdcm::Tag t2(0x0020,0x000e); // Series Instance UID
  s.AddTag( t2 );
  bool b = s.Scan( d.GetFilenames() );
  if( !b )
    {
    std::cerr << "Scanner failed" << std::endl;
    return 1;
    }

  gdcm::Directory::FilenamesType const & files = s.GetFilenames();
  if( files != d.GetFilenames() )
    {
    return 1;
    }

  const char str1[] = "1.3.12.2.1107.5.2.4.7630.20010301125744000008";
  gdcm::Directory::FilenamesType fns = s.GetAllFilenamesFromTagToValue(t2, str1);

  // all SIEMENS_MAGNETOM-12-MONO2-FileSeq*.dcm:
  if( fns.size() != 4 ) return 1;

  const char str2[] = "1.3.12.2.1107.5.2.4.7630.2001030112574400000";
  fns = s.GetAllFilenamesFromTagToValue(t2, str2);

  if( !fns.empty() ) return 1;

  return 0;
}
