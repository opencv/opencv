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

int TestDICOMDIRGenerator2(int argc, char *argv[])
{
  (void)argc;
  const char *directory = gdcm::Testing::GetDataRoot();
  (void)argv;

  gdcm::Directory::FilenamesType filenames;
  gdcm::Directory::FilenamesType outfilenames;
  gdcm::Directory dir;
  int recursive = 0;
  unsigned int nfiles = 1;

  const char subdir[] = "TestImageChangeTransferSyntax4";
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    std::cerr << "Need to run TestImageChangeTransferSyntax4 before" << std::endl;
    return 1;
    }
  directory = tmpdir.c_str();

  const char outsubdir[] = "TestDICOMDIRGenerator2";
  std::string outtmpdir = gdcm::Testing::GetTempDirectory( outsubdir );
  if( !gdcm::System::FileIsDirectory( outtmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( outtmpdir.c_str() );
    }

  nfiles = dir.Load(directory, (recursive > 0 ? true : false));

  gdcm::FilenameGenerator fg;
  const char pattern[] = "FILE%03d";
  fg.SetPattern( pattern );
  fg.SetNumberOfFilenames( nfiles );
  if( !fg.Generate() )
    {
    std::cerr << "Could not generate" << std::endl;
    return 1;
    }
  filenames = dir.GetFilenames();
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
