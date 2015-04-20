/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileStreamer.h"

#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmReader.h"
#include "gdcmDataSet.h"
#include "gdcmPrivateTag.h"
#include "gdcmFilename.h"

int TestFileStream5(const char *filename, bool verbose = false)
{
  using namespace gdcm;
  if( verbose )
    std::cout << "Processing: " << filename << std::endl;

  // Create directory first:
  const char subdir[] = "TestFileStreamer5";
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );
  if( verbose )
    std::cout << "Generating: " << outfilename << std::endl;

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();
  // Special handling:
  bool checktemplate = false;
  if( strcmp(name, "DMCPACS_ExplicitImplicit_BogusIOP.dcm" ) == 0
    || strcmp(name, "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm") == 0
    || strcmp(name, "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm") == 0 
  )
    {
    checktemplate = true;
    }

  gdcm::FileStreamer fs;
  fs.SetTemplateFileName( filename );
  fs.CheckTemplateFileName( checktemplate );
  fs.SetOutputFileName( outfilename.c_str() );

  std::vector<char> vbuffer;
  vbuffer.resize( 8192 );
  const char *buffer = &vbuffer[0];
  const size_t len = vbuffer.size();
  PrivateTag pt( Tag(0x9,0x10), "MYTEST" );
  fs.StartGroupDataElement( pt, 10 );
  if( fs.AppendToGroupDataElement( pt, buffer, len ) )
    {
    std::cerr << "We should not succeed, we should fail this test" << std::endl;
    return 1;
    }
  fs.StopGroupDataElement( pt );

  return 0;
}

int TestFileStreamer5(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestFileStream5(filename, true);
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
    r += TestFileStream5( filename );
    ++i;
    }

  return r;
}
