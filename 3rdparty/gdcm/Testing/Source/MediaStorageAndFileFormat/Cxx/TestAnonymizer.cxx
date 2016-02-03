/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAnonymizer.h"
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmUIDGenerator.h"

namespace gdcm
{
int TestAnonymize(const char *subdir, const char* filename)
{
  Reader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    return 1;
    }

  Anonymizer anonymizer;
  anonymizer.SetFile( reader.GetFile() );
  // Setup some actions:
  const char patname[] = "test^anonymize";
  const Tag pattag = Tag(0x0010,0x0010);
  anonymizer.Replace( pattag , patname );
  anonymizer.Remove( Tag(0x0008,0x2112) );
  anonymizer.Empty( Tag(0x0008,0x0070) );
  UIDGenerator uid;
  // Those two are very special, since (0008,0016) needs to be propagated to (0002,0002) and
  // (0008,0018) needs to be propagated to (0002,0003)
  std::string newuid = uid.Generate();
  anonymizer.Replace( Tag(0x0008,0x0018), newuid.c_str() );
  anonymizer.Replace( Tag(0x0008,0x0016), "1.2.840.10008.5.1.4.1.1.1" ); // Make it a CT
  if( !anonymizer.RemovePrivateTags() )
    {
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
  writer.SetCheckFileMetaInformation( false );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }
  std::cout << "Success to write: " << outfilename << std::endl;

  // now let's try to read it back in:
  Reader reader2;
  reader2.SetFileName( outfilename.c_str() );
  if ( !reader2.Read() )
    {
    std::cerr << "Could not reread written file: " << outfilename << std::endl;
    return 1;
    }

  const DataSet & ds = reader.GetFile().GetDataSet();
  //std::cout << ds << std::endl;

  const ByteValue *bv = ds.GetDataElement( pattag ).GetByteValue();
  if( !bv )
    {
    return 1;
    }
  if (strncmp( bv->GetPointer(), patname, strlen(patname) ) != 0 )
    {
    return 1;
    }
  if( bv->GetLength() != strlen(patname) )
    {
    return 1;
    }

  return 0;
}
}

int TestAnonymizer(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestAnonymize(argv[0], filename);
    }

  // else
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += gdcm::TestAnonymize( argv[0], filename );
    ++i;
    }

  return r;
}
