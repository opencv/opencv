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
#include "gdcmWriter.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmTesting.h"

namespace gdcm
{
/*
 * we are only testing that we can convert an implicit dataset to explicit all the time...
 */
int TestWrite2(const char *subdir, const char* filename, bool )
{
  Reader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  // Create directory first:
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );

  // Invert Transfer Syntax just for fun:
  const TransferSyntax &ts = reader.GetFile().GetHeader().GetDataSetTransferSyntax();
  if( ts.IsExplicit() && ts == TransferSyntax::ExplicitVRLittleEndian )
    {
    reader.GetFile().GetHeader().SetDataSetTransferSyntax( gdcm::TransferSyntax::ImplicitVRLittleEndian );
    }
  else if( ts.IsImplicit() )
    {
    gdcm::FileMetaInformation &fmi = reader.GetFile().GetHeader();
    gdcm::TransferSyntax ts2 = gdcm::TransferSyntax::ImplicitVRLittleEndian;
    ts2 = gdcm::TransferSyntax::ExplicitVRLittleEndian;

    const char *tsuid = gdcm::TransferSyntax::GetTSString( ts2 );
    gdcm::DataElement de( gdcm::Tag(0x0002,0x0010) );
    de.SetByteValue( tsuid, (uint32_t)strlen(tsuid) );
    de.SetVR( VR::UI ); //gdcm::Attribute<0x0002, 0x0010>::GetVR() );
    fmi.Replace( de );

    reader.GetFile().GetHeader().SetDataSetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );
    }
  else
    {
    // nothing to test
    return 0;
    }

  const char str[] = "1.2.3.4.5.6.8.9.0";
  DataElement xde;
  xde.SetByteValue(str, (uint32_t)strlen(str));
  xde.SetVR( VR::UI );
  xde.SetTag( Tag(0x0008,0x0018) );
  reader.GetFile().GetDataSet().Insert( xde );


  Writer writer;
  writer.SetFileName( outfilename.c_str() );
  writer.SetFile( reader.GetFile() );
  //writer.SetCheckFileMetaInformation( true );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }

  Reader reader2;
  reader2.SetFileName( outfilename.c_str() );
  if ( !reader2.Read() )
    {
    std::cerr << "Failed to re-read: " << outfilename << std::endl;
    return 1;
    }

  return 0;

}
}

int TestWriter2(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestWrite2(argv[0], filename, false );
    }

  // else
  int r = 0, i = 0;
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += gdcm::TestWrite2(argv[0], filename, false );
    ++i;
    }

  return r;
}
