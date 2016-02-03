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
#include "gdcmReader.h"
#include "gdcmWriter.h"

int TestIPPSorter3(int , char *[])
{
  const char *directory = gdcm::Testing::GetDataRoot();
  std::vector<std::string> filenames;
  // let's take 4 files that can be sorted:
  std::string file0 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm";
  std::string file1 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm";
  std::string file2 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm";
  std::string file3 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm";

  // Make a fake copy:
  const char * reference = file0.c_str();
  gdcm::Reader reader;
  reader.SetFileName( reference );
  if( !reader.Read() ) return 1;

  // Create directory first:
  const char subdir[] = "TestIPPSorter3";
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }

  std::string outfilename = gdcm::Testing::GetTempFilename( reference, subdir );

  // Tweak the orientation just a little:
  // [001.000000E+00\-0.000000E+00\-0.000000E+00\00.000000E+00\01.000000E+00\-0.000000E+00]
  gdcm::Writer writer;
  writer.SetFileName( outfilename.c_str() );

  //const char iop_orig[] = "1\\-0\\-0\\0\\1\\-0";
  const char iop[] = "1\\-0\\-0\\0\\0.99999\\0.00001";
  gdcm::DataElement de( gdcm::Tag(0x0020,0x0037) );
  de.SetByteValue( iop, (uint32_t)strlen( iop ) );
  reader.GetFile().GetDataSet().Replace( de );

  writer.SetFile( reader.GetFile() );
  if( !writer.Write() )
    {
    return 1;
    }

  // let's push them in random order (oh my god how are we going to succeed ??)
  filenames.push_back( file1 );
  filenames.push_back( file3 );
  filenames.push_back( outfilename );
  filenames.push_back( file2 );
  //filenames.push_back( file0 );

  gdcm::IPPSorter s;
  s.SetComputeZSpacing( true );
  s.SetZSpacingTolerance( 1e-10 );
  s.SetDirectionCosinesTolerance( 1e-6 );
  bool b = s.Sort( filenames );
  if( b )
    {
    std::cerr << "Success to sort (we should have failed): " << directory << std::endl;
    return 1;
    }

  // Lower the threshold:
  s.SetDirectionCosinesTolerance( 1e-5 );
  b = s.Sort( filenames );
  if( !b )
    {
    std::cerr << "Failed to sort: " << directory << std::endl;
    return 1;
    }

//  std::cout << "Sorting succeeded:" << std::endl;
//  s.Print( std::cout );

  double zspacing = s.GetZSpacing();
  if(!zspacing)
    {
    std::cerr << "computation of ZSpacing failed." << std::endl;
    return 1;
    }
  std::cout << "Found z-spacing:" << std::endl;
  std::cout << s.GetZSpacing() << std::endl;

  return 0;
}
