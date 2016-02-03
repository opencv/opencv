/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageFragmentSplitter.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#include "gdcmTransferSyntax.h"
#include "gdcmImage.h"
#include "gdcmFilename.h"

int TestImageFragmentSplitterFunc(const char *filename, bool verbose = false)
{
  gdcm::ImageReader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    const gdcm::FileMetaInformation &header = reader.GetFile().GetHeader();
    gdcm::MediaStorage ms = header.GetMediaStorage();
    bool isImage = gdcm::MediaStorage::IsImage( ms );
    if( isImage )
      {
      if( reader.GetFile().GetDataSet().FindDataElement( gdcm::Tag(0x7fe0,0x0010) ) )
        {
        std::cerr << "Could not read: " << filename << std::endl;
        return 1;
        }
      }
    return 0;
    }
  const gdcm::Image &image = reader.GetImage();
  const unsigned int *dims = image.GetDimensions();
  if( dims[2] != 1 )
    {
    return 0; // nothing to do
    }
  const gdcm::File &file = reader.GetFile();
  const gdcm::FileMetaInformation &header = file.GetHeader();
  //gdcm::MediaStorage ms = header.GetMediaStorage();
  if(  header.GetDataSetTransferSyntax() == gdcm::TransferSyntax::ImplicitVRLittleEndian
    || header.GetDataSetTransferSyntax() == gdcm::TransferSyntax::ImplicitVRBigEndianPrivateGE
    || header.GetDataSetTransferSyntax() == gdcm::TransferSyntax::ExplicitVRLittleEndian
    || header.GetDataSetTransferSyntax() == gdcm::TransferSyntax::DeflatedExplicitVRLittleEndian
    || header.GetDataSetTransferSyntax() == gdcm::TransferSyntax::ExplicitVRBigEndian
  )
    {
    return 0; // nothing to do
    }

  gdcm::ImageFragmentSplitter splitter;
  splitter.SetInput( image );
  splitter.SetFragmentSizeMax( 65536 );
  bool b = splitter.Split();
  if( !b )
    {
    const gdcm::DataElement &pixeldata = file.GetDataSet().GetDataElement( gdcm::Tag(0x7fe0,0x0010) );
    const gdcm::SequenceOfFragments* sqf = pixeldata.GetSequenceOfFragments();
    if( sqf && dims[2] == 1 )
      {
      return 0;
      }
    gdcm::Filename fn( filename );
    if( fn.GetName() == std::string("JPEGDefinedLengthSequenceOfFragments.dcm" ) )
      {
      // JPEG Fragments are packed in a VR:OB Attribute
      return 0;
      }
    std::cerr << "Could not apply splitter: " << filename << std::endl;
    return 1;
    }

  // Create directory first:
  const char subdir[] = "TestImageFragmentSplitter";
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = gdcm::Testing::GetTempFilename( filename, subdir );

  gdcm::ImageWriter writer;
  writer.SetFileName( outfilename.c_str() );
  //writer.SetFile( reader.GetFile() ); // increase test goal
  writer.SetImage( splitter.GetOutput() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }
  if( verbose )
    std::cout << "Success to write: " << outfilename << std::endl;

  return 0;
}

int TestImageFragmentSplitter(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestImageFragmentSplitterFunc(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestImageFragmentSplitterFunc( filename );
    ++i;
    }

  return r;
}
