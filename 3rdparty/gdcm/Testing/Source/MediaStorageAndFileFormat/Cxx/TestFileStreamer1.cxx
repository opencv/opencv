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

int TestFileStream1(const char *filename, bool verbose = false)
{
  using namespace gdcm;
  if( verbose )
    std::cout << "Processing: " << filename << std::endl;

  // Create directory first:
  const char subdir[] = "TestFileStreamer1";
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
  if( !fs.ReserveGroupDataElement( 20 ) )
    {
    return 1;
    }
  const uint8_t startoffset = 0x13; // why not ?
  fs.StartGroupDataElement( pt, 1000, startoffset );
  fs.AppendToGroupDataElement( pt, buffer, len );
  fs.AppendToGroupDataElement( pt, buffer, len );
  fs.StopGroupDataElement( pt );

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

  const DataElement private_creator = pt.GetAsDataElement();
  if( !ds.FindDataElement( private_creator.GetTag() ) )
    {
    std::cerr << "Could not find priv creator: " << outfilename << std::endl;
    return 1;
    }
  // Check all the group:
  const size_t nels = (2 * vbuffer.size() + 999) / 1000;
  if( nels != 17 ) return 1;
  PrivateTag check = pt;
  for( size_t i = startoffset; i < startoffset + nels; ++i )
    {
#if 1
    check.SetElement( (uint16_t)i );
    check.SetPrivateCreator( pt );
#else
    check.SetElement( check.GetElement() + 1 );
#endif
    if( !ds.FindDataElement( check ) )
      {
      std::cerr << "Could not find priv tag: " << check << " " << outfilename << std::endl;
      return 1;
      }
    const DataElement & de = ds.GetDataElement( check );
    const int vl = de.GetVL();
    int reflen = 0;
    if( i == (startoffset + nels - 1) )
      {
      reflen = 384;
      }
    else
      {
      reflen = 1000;
      }
    if( vl != reflen )
      {
      std::cerr << "Wrong length for: " << check << ":" << vl << " should be :" << reflen << std::endl;
      return 1;
      }
    }

  return 0;
}

int TestFileStreamer1(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestFileStream1(filename, true);
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
    r += TestFileStream1( filename );
    ++i;
    }

  return r;
}
