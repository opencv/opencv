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
#include "gdcmFilename.h"

int TestFileStream3(const char *filename, bool verbose = false)
{
  using namespace gdcm;
  if( verbose )
    std::cout << "Processing: " << filename << std::endl;

  // Create directory first:
  const char subdir[] = "TestFileStreamer3";
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

  const gdcm::Tag t1(0x0008,0x0010);
  const gdcm::Tag t2(0x0010,0x0010);

  // Try a small buffer to find a case where existing element is larger
  bool b;
  const char buffer[] = "  ";
  const size_t len = strlen( buffer );
  // Recognition Code
  b = fs.StartDataElement( t1 );
  if( !b )
    {
    std::cerr << "Could not StartDataElement" << std::endl;
    return 1;
    }
  b = fs.AppendToDataElement( t1, buffer, len );
  if( !b )
    {
    std::cerr << "Could not AppendToDataElement (t1)" << std::endl;
    return 1;
    }
  b = fs.StopDataElement( t1 );
  if( !b )
    {
    std::cerr << "Could not StopDataElement" << std::endl;
    return 1;
    }
  // Patient's Name
  b = fs.StartDataElement( t2 );
  if( !b )
    {
    std::cerr << "Could not StartDataElement" << std::endl;
    return 1;
    }
  b = fs.AppendToDataElement( t2, buffer, len );
  if( !b )
    {
    std::cerr << "Could not AppendToDataElement" << std::endl;
    return 1;
    }
  b = fs.StopDataElement( t2 );
  if( !b )
    {
    std::cerr << "Could not StopDataElement" << std::endl;
    return 1;
    }

  // Read back and check:
  gdcm::Reader r;
  r.SetFileName( outfilename.c_str() );
  if( !r.Read() )
    {
    std::cerr << "Failed to read: " << outfilename << std::endl;
    return 1;
    }

  gdcm::File & f = r.GetFile();
  gdcm::DataSet & ds = f.GetDataSet();

  if( !ds.FindDataElement( t1 ) )
    {
    std::cerr << "Could not find tag: " << t1 << std::endl;
    return 1;
    }

{
  const DataElement & de = ds.GetDataElement( t1 );
  const ByteValue * bv = de.GetByteValue();
  if( !bv ) return 1;
  if( bv->GetLength() != 2 )
    {
    std::cerr << "Wrong length: " << bv->GetLength() << std::endl;
    return 1;
    }
  if( memcmp( bv->GetPointer(), buffer, 2 ) )
    {
    std::cerr << "Wrong content" << std::endl;
    return 1;
    }
}

{
  const DataElement & de = ds.GetDataElement( t2 );
  const ByteValue * bv = de.GetByteValue();
  if( !bv ) return 1;
  if( bv->GetLength() != 2 ) return 1;
  if( memcmp( bv->GetPointer(), buffer, 2 ) )
    {
    std::cerr << "Wrong content" << std::endl;
    return 1;
    }
}

  return 0;
}

int TestFileStreamer3(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestFileStream3(filename, true);
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
    r += TestFileStream3( filename );
    ++i;
    }

  return r;
}
