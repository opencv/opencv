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

int TestFileStream2(const char *filename, bool verbose = false)
{
  using namespace gdcm;
  if( verbose )
    std::cout << "Processing: " << filename << std::endl;

  // Create directory first:
  const char subdir[] = "TestFileStreamer2";
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );
  if( verbose )
    std::cout << "Generating: " << outfilename << std::endl;

  const gdcm::Tag t1(0x0042,0x0011);

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

  const char buffer[10] = { 0, 1, 2 , 3, 4, 5, 6, 7, 8, 9 };
  const size_t len = sizeof( buffer );
  //fs.ReserveDataElement( 36 );
  fs.StartDataElement( t1 );
  fs.AppendToDataElement( t1, buffer, len );
  fs.AppendToDataElement( t1, buffer, len );
  fs.AppendToDataElement( t1, buffer, len );
  fs.AppendToDataElement( t1, buffer, len / 2 + 0 );
  fs.StopDataElement( t1 );

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
  const DataElement & de = ds.GetDataElement( t1 );
  const int vl = de.GetVL();

  if( vl != 36 ) // 35 + padding
    {
    return 1;
    }
  const ByteValue *bv = de.GetByteValue();
  const char *ptr = bv->GetPointer();
  const int b1 = memcmp( ptr + 0 * len, buffer, len );
  const int b2 = memcmp( ptr + 1 * len, buffer, len );
  const int b3 = memcmp( ptr + 2 * len, buffer, len );
  const int b4 = memcmp( ptr + 3 * len, buffer, len / 2 );
  if( b1 || b2 || b3 || b4 )
    {
    std::cerr << "Problem:" << b1 << " "
      << b2 << " "
      << b3 << " "
      << b4 << std::endl;
    return 1;
    }
  // NULL padding
  if( ptr[35] != 0x0 )
    {
    std::cerr << "Error in padding" << std::endl;
    return 1;
    }

  return 0;
}

int TestFileStreamer2(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestFileStream2(filename, true);
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
    r += TestFileStream2( filename );
    ++i;
    }

  return r;
}
