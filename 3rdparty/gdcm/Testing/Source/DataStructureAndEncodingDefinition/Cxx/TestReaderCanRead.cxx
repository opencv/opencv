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
#include "gdcmTesting.h"

#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmWriter.h"
#include "gdcmDirectory.h"

int TestReadCanRead(const char *subdir, const char* filename, bool verbose = false)
{
  if( verbose )
  std::cout << "TestReadCanRead: " << filename << std::endl;

  bool b1, b2;

  gdcm::Reader reader;
  reader.SetFileName( filename );
  b1 = reader.CanRead();
  b2 = reader.Read();

  gdcm::Filename fn( filename );
  const char *name = fn.GetName();
  int ret = 1;
  const char dicomdir_bad[] =
    "gdcmSampleData/ForSeriesTesting/Perfusion/DICOMDIR";
  if( fn.EndWith( dicomdir_bad ) )
    {
    ret = 0;
    }
  // This is not an error if you are in the black list:
  if( strcmp(name, "BuggedDicomWorksImage_Hopeless.dcm" ) != 0
   || strcmp(name, "Philips_Wrong_Terminator_XX_0277" ) != 0
   || strcmp(name, "DICOMDIR_noPATIENT_noSTUDY" ) != 0
   || strcmp(name, "JpegLosslessMultiframePapyrus.pap" ) != 0
   || strcmp(name, "Fish-1.img" ) != 0
   || strcmp(name, "NM_FromJulius_Caesar.dcm" ) != 0
   || strcmp(name, "Siemens-leonardo-bugged.dcm" ) != 0
   || strcmp(name, "illegal_UN_stands_for_OB.dcm" ) != 0
   || strcmp(name, "MR_PHILIPS_LargePreamble.dcm" ) != 0
   || strcmp(name, "illegal_UN_stands_for_OB.dcm.secu" ) != 0 )
    {
    ret = 0;
    }
  if ( (b1 && !b2) || (!b1 && b2)  )
    {
    std::cerr << "TestReadCanRead: incompatible result " << filename << std::endl;
    if( b1 )
    std::cerr << "TestReadCanRead: CanRead was true: " << filename << std::endl;
    //assert( !ret );
    return ret;
    }

  // Create directory first:
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = gdcm::Testing::GetTempFilename( filename, subdir );

  if( b1 && b2 )
    {

    gdcm::Writer writer;
    writer.SetFileName( outfilename.c_str() );
    writer.GetFile().GetHeader().SetDataSetTransferSyntax(
      reader.GetFile().GetHeader().GetDataSetTransferSyntax() );
    writer.GetFile().SetDataSet( reader.GetFile().GetDataSet() );
    writer.SetCheckFileMetaInformation( false );
    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      return 1;
      }

    gdcm::Reader reader2;
    reader2.SetFileName( outfilename.c_str() );
    b1 = reader2.CanRead( );
    //reader2.GetFile().GetHeader().GetPreamble().Remove();
    //assert( reader2.GetFile().GetHeader().GetPreamble().IsEmpty() );
    b2 = reader2.Read();
    if ( (b1 && !b2) || (!b1 && b2)  )
      {
      std::cerr << "TestReadCanRead|Write: incompatible result " << outfilename << std::endl;
      if( b1 )
        std::cerr << "TestReadCanRead|Write: CanRead was true: " << outfilename << std::endl;
      return 1;
      }
    }

  return 0;
}

int TestReadCanReadExtra()
{
  const char *extradataroot = gdcm::Testing::GetDataExtraRoot();
  if( !extradataroot )
    {
    return 0;
    }
  if( !gdcm::System::FileIsDirectory(extradataroot) )
    {
    std::cerr << "No such directory: " << extradataroot <<  std::endl;
    return 0;
    }

  gdcm::Directory d;
  unsigned int nfiles = d.Load( extradataroot, true ); // no recursion
  std::cout << "done retrieving file list. " << nfiles << " files found." <<  std::endl;

  gdcm::Directory::FilenamesType const & fns = d.GetFilenames();
  int r = 0;
  for( gdcm::Directory::FilenamesType::const_iterator it = fns.begin();
    it != fns.end(); ++it )
    {
    const char *filename = it->c_str();
    r += TestReadCanRead( "TestReaderCanReadExtra", filename );
    }

  return r;
}

int TestReaderCanRead(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestReadCanRead(argv[0], filename, true);
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
    r += TestReadCanRead( argv[0], filename );
    ++i;
    }

  //int r2 =
  TestReadCanReadExtra();
  //r += r2;

  return r;
}
