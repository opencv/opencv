/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileDerivation.h"
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmUIDGenerator.h"

int TestFileDerive(const char *subdir, const char* filename)
{
  using namespace gdcm;

  Reader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }
  FileDerivation fd;
  fd.SetFile( reader.GetFile() );
  // Setup some actions:

  // TODO we should reference the original image
  File &file = reader.GetFile();
  DataSet &ds = file.GetDataSet();

  if( !ds.FindDataElement( Tag(0x0008,0x0016) )
    || ds.GetDataElement( Tag(0x0008,0x0016) ).IsEmpty() )
    {
    std::cerr << "Missing sop class giving up: " << filename << std::endl;
    return 0;
    }
  if( !ds.FindDataElement( Tag(0x0008,0x0018) )
    || ds.GetDataElement( Tag(0x0008,0x0018) ).IsEmpty() )
    {
    std::cerr << "Missing sop instance giving up: " << filename << std::endl;
    return 0;
    }

  const DataElement &sopclassuid = ds.GetDataElement( Tag(0x0008,0x0016) );
  const DataElement &sopinstanceuid = ds.GetDataElement( Tag(0x0008,0x0018) );
  // Make sure that const char* pointer will be properly padded with \0 char:
  std::string sopclassuid_str( sopclassuid.GetByteValue()->GetPointer(), sopclassuid.GetByteValue()->GetLength() );
  std::string sopinstanceuid_str( sopinstanceuid.GetByteValue()->GetPointer(), sopinstanceuid.GetByteValue()->GetLength() );

  fd.AddReference( sopclassuid_str.c_str(), sopinstanceuid_str.c_str() );

  // CID 7202 Source Image Purposes of Reference
  // {"DCM",121320,"Uncompressed predecessor"},
  fd.SetPurposeOfReferenceCodeSequenceCodeValue( 121320 );

  // CID 7203 Image Derivation
  // { "DCM",113040,"Lossy Compression" },
  fd.SetDerivationCodeSequenceCodeValue( 113040 );
  fd.SetDerivationDescription( "lossy conversion" );

  if( !fd.Derive() )
    {
    std::cerr << "Failed to derive: " << filename << std::endl;
    if( ds.FindDataElement( Tag(0x8,0x2112) ) )
      {
      return 0;
      }
    return 1;
    }

  // Create directory first:
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );

  Writer writer;
  writer.SetFileName( outfilename.c_str() );
  writer.SetFile( reader.GetFile() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }

  // now let's try to read it back in:
  Reader reader2;
  reader2.SetFileName( outfilename.c_str() );
  if ( !reader2.Read() )
    {
    std::cerr << "Could not reread written file: " << outfilename << std::endl;
    return 1;
    }

  return 0;
}

int TestFileDerivation( int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestFileDerive( argv[0], filename);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestFileDerive( argv[0], filename );
    ++i;
    }

  return r;
}
