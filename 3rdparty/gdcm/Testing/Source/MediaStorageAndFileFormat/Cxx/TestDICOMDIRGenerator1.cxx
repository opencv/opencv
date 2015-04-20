/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDICOMDIRGenerator.h"
#include "gdcmDirectory.h"
#include "gdcmWriter.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmFilenameGenerator.h"

int TestDICOMDIRGenerator1(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  gdcm::Directory::FilenamesType filenames;
  gdcm::Directory::FilenamesType outfilenames;

  const char outsubdir[] = "TestDICOMDIRGenerator1";
  std::string outtmpdir = gdcm::Testing::GetTempDirectory( outsubdir );
  if( !gdcm::System::FileIsDirectory( outtmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( outtmpdir.c_str() );
    }

  const char subdir[] = "TestImageChangeTransferSyntax4";
  std::string directory = gdcm::Testing::GetTempDirectory( subdir );

  std::string file0 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm";
  std::string file1 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm";
  std::string file2 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm";
  std::string file3 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm";
  filenames.push_back( file1 );
  filenames.push_back( file3 );
  filenames.push_back( file2 );
  filenames.push_back( file0 );
  size_t nfiles = filenames.size();

  gdcm::FilenameGenerator fg;
  const char pattern[] = "FILE%03d";
  fg.SetPattern( pattern );
  fg.SetNumberOfFilenames( nfiles );
  if( !fg.Generate() )
    {
    std::cerr << "Could not generate" << std::endl;
    return 1;
    }
  for( unsigned int i = 0; i < nfiles; ++i )
    {
    std::string copy = outtmpdir;
    copy += "/";
    copy += fg.GetFilename( i );
    std::cerr << filenames[i] << " -> " << copy << std::endl;
    std::ifstream f1(filenames[i].c_str(), std::fstream::binary);
    std::ofstream f2(copy.c_str(), std::fstream::binary);
    f2 << f1.rdbuf();
    outfilenames.push_back( copy );
    }

  gdcm::DICOMDIRGenerator gen;
  gen.SetFilenames( outfilenames );
  gen.SetRootDirectory( outtmpdir );
  gen.SetDescriptor( "MYDESCRIPTOR" );
  if( !gen.Generate() )
    {
    return 1;
    }

  gdcm::Writer writer;
  writer.SetFile( gen.GetFile() );
  std::string outfilename = outtmpdir;
  outfilename += "/DICOMDIR";
  writer.SetFileName( outfilename.c_str() );
  if( !writer.Write() )
    {
    return 1;
    }

  return 0;
}
