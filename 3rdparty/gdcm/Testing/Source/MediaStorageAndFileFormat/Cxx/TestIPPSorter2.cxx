/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIPPSorter.h"
#include "gdcmDirectory.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmAttribute.h"

// Sort image using Instance Number:
bool mysort(gdcm::DataSet const & ds1, gdcm::DataSet const & ds2 )
{
  gdcm::Attribute<0x0020,0x0013> at1; // Instance Number
  at1.Set( ds1 );
  gdcm::Attribute<0x0020,0x0013> at2;
  at2.Set( ds2 );
  return at1 < at2;
}


int TestIPPSorter2(int argc, char *argv[])
{
  const char *directory = gdcm::Testing::GetDataRoot();
  std::vector<std::string> filenames;
  if( argc == 2 )
    {
    gdcm::Trace::DebugOn();
    directory = argv[1];
    if( gdcm::System::FileIsDirectory( directory ) )
      {
      gdcm::Directory d;
      unsigned int nfiles = d.Load( directory ); // no recursion
      d.Print( std::cout );
      std::cout << "done retrieving file list. " << nfiles << " files found." <<  std::endl;
      filenames = d.GetFilenames();
      }
    else
      {
      std::cerr << "file:" << directory << " is not a directory" << std::endl;
      return 1;
      }
    }
  else
    {
    // default execution (nightly test)
    // let's take 4 files that can be sorted:
    std::string file0 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm";
    std::string file1 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm";
    std::string file2 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm";
    std::string file3 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm";
    // let's push them in random order (oh my god how are we going to succeed ??)
    filenames.push_back( file1 );
    filenames.push_back( file3 );
    filenames.push_back( file2 );
    filenames.push_back( file0 );
    }

  gdcm::IPPSorter s;
  s.SetComputeZSpacing( true );
  s.SetZSpacingTolerance( 1e-10 );
  bool b = s.Sort( filenames );
  if( !b )
    {
    std::cerr << "Failed to sort:" << directory << std::endl;
    return 1;
    }

  std::cout << "Sorting succeeded:" << std::endl;
  s.Print( std::cout );

  double zspacing = s.GetZSpacing();
  if(!zspacing)
    {
    std::cerr << "computation of ZSpacing failed." << std::endl;
    return 1;
    }
  std::cout << "Found z-spacing:" << std::endl;
  std::cout << s.GetZSpacing() << std::endl;

  // Now apply a StableSort on them:
  s.SetSortFunction( mysort );
  s.StableSort( s.GetFilenames() );

  s.Print( std::cout );

  return 0;
}
