/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageReader.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmSystem.h"
#include "gdcmWriter.h"
#include "gdcmFilename.h"
#include "gdcmAnonymizer.h"
#include "gdcmByteSwap.h"
#include "gdcmTrace.h"
#include "gdcmTesting.h"

#include <sstream>

int TestImageReaderRandomEmptyFunc(const char *subdir, const char* filename, bool verbose = false, bool lossydump = false)
{
  (void)lossydump;
  if( verbose )
    std::cerr << "Reading: " << filename << std::endl;
  gdcm::ImageReader reader;

  reader.SetFileName( filename );
  if( !reader.Read() )
    {
    return 0;//this is NOT a test of the ImageReader read function
    //this is a test of the anonymizer.  As such, if the reader can't read this file,
    //that should be handled in the TestImageReader test, NOT here.
    //There is a lot of logic involved in testing non-standard images that should not be
    //duplicated here.
    }

  const gdcm::File &file = reader.GetFile();
  const gdcm::DataSet &ds = file.GetDataSet();
  const gdcm::FileMetaInformation &fmi = file.GetHeader();
  gdcm::DataSet::ConstIterator it = ds.Begin();

  // Create directory first:
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = gdcm::Testing::GetTempFilename( filename, subdir );

  int ret = 0;
  int i = 0;
  for( ; it != ds.End(); ++it, ++i )
    {
    //std::cout << "Processing Tag: " << *it << std::endl;
    gdcm::Writer writer;
    gdcm::File &filecopy = writer.GetFile();
    filecopy.SetDataSet( ds );
    filecopy.SetHeader( fmi );

    gdcm::Anonymizer ano;
    ano.SetFile( filecopy );
    ano.Empty( it->GetTag() );

    //std::ostringstream os;
    //os << "/tmp/ddd";
    //os << i;
    //os << ".dcm";
    //std::string outfn = os.str();
    std::string outfn = outfilename;
    writer.SetFile( ano.GetFile() );
    writer.CheckFileMetaInformationOff(); // FIXME ?
    writer.SetFileName( outfn.c_str() );
    if( !writer.Write() )
      {
      std::cerr << "Could not write: " << outfn << std::endl;
      ret++;
      }

    gdcm::ImageReader readercopy;
    readercopy.SetFileName( outfn.c_str() );
    readercopy.Read();
    }

  return ret;
}

int TestImageReaderRandomEmpty(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestImageReaderRandomEmptyFunc(argv[0], filename, true);
    }

  // else
  // First of get rid of warning/debug message
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestImageReaderRandomEmptyFunc( argv[0], filename);
    ++i;
    }

  return r;
}
