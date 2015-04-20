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

//bool IsImpossibleToRewrite(const char *filename)
//{
//  const char *impossible;
//  int i = 0;
//  while( (impossible= gdcmBlackListWriterDataImages[i]) )
//    {
//    if( strcmp( impossible, filename ) == 0 )
//      {
//      return true;
//      }
//    ++i;
//    }
//  return false;
//}

namespace gdcm
{


int TestWrite(const char *subdir, const char* filename, bool recursing, bool verbose = false)
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

  // Ok we have now two files let's compare their md5 sum:
  char digest[33], outdigest[33];
  Testing::ComputeFileMD5(filename, digest);
  Testing::ComputeFileMD5(outfilename.c_str(), outdigest);
  if( strcmp(digest, outdigest) )
    {
    if (recursing)
      return 1;
    // too bad the file is not identical, so let's be paranoid and
    // try to reread-rewrite this just-writen file:
    // TODO: Copy file System::CopyFile( );
    std::string subsubdir = subdir;
    subsubdir += "/";
    subsubdir += subdir;
    if( TestWrite(subsubdir.c_str(), outfilename.c_str(), true ) )
      {
      std::cerr << filename << " and "
        << outfilename << " are different\n";
      return 1;
      }
   const char * ref = Testing::GetMD5FromBrokenFile(filename);
   if( ref )
     {
     if( strcmp(ref, outdigest) == 0 )
       {
       // ok this situation was already analyzed and the writen file is
       // readable by dcmtk and such
       //size_t size1 = System::FileSize( filename );
       //size_t size2 = System::FileSize( outfilename.c_str() );
       //assert( size1 == size2 ); // cannot deal with implicit VR meta data header
       return 0;
       }
      std::cerr << "incompatible ref: " << ref << " vs " << outdigest << " for file: " << filename << " & " << outfilename << std::endl;
      return 1; // ref exist but does not match, how is that possible ?
     }
   //if( !ref )
   //  {
   //  return 1;
   //  }
    // In theory I need to compare the two documents to check they
    // are identical... TODO
    std::cerr << filename << " and "
      << outfilename << " are different, output can be read though. Need manual intervention\n";
    return 1;
    }
  else
    {
    size_t size1 = System::FileSize( filename );
    size_t size2 = System::FileSize( outfilename.c_str() );
    if( size1 != size2 ) return 1;
    if(verbose)
    std::cerr << filename << " and " << outfilename << " are identical\n";
    return 0;
    }
}
}

int TestWriter(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestWrite(argv[0], filename, false, true);
    }

  // else
  int r = 0, i = 0;
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += gdcm::TestWrite(argv[0], filename, false );
    ++i;
    }

  return r;
}
