/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageChangeTransferSyntax.h"
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmTesting.h"
#include "gdcmByteSwap.h"

namespace gdcm
{

int TestImageChangeTransferSyntaxRAWBE(const char *filename, bool verbose = false)
{
  ImageReader reader;
  reader.SetFileName( filename );
  if ( !reader.Read() )
    {
    const FileMetaInformation &header = reader.GetFile().GetHeader();
    MediaStorage ms = header.GetMediaStorage();
    bool isImage = MediaStorage::IsImage( ms );
    bool pixeldata = reader.GetFile().GetDataSet().FindDataElement( Tag(0x7fe0,0x0010) );
    if( isImage && pixeldata )
      {
      std::cerr << "Failed to read: " << filename << std::endl;
      return 1;
      }
    else
      {
      // not an image give up...
      std::cerr << "Problem with: " << filename << " but that's ok" << std::endl;
      return 0;
      }
    }

  const gdcm::Image &image = reader.GetImage();
  //const TransferSyntax &ts = reader.GetFile().GetHeader().GetDataSetTransferSyntax();

  gdcm::ImageChangeTransferSyntax change;
  change.SetTransferSyntax( gdcm::TransferSyntax::ExplicitVRBigEndian );
  change.SetInput( image );
  bool b = change.Change();
  if( !b )
    {
    std::cerr << "Could not change the Transfer Syntax: " << filename << std::endl;
    return 1;
    }

  // Create directory first:
  const char subdir[] = "TestImageChangeTransferSyntax6";
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );

  ImageWriter writer;
  writer.SetFileName( outfilename.c_str() );
  writer.SetFile( reader.GetFile() ); // increase test goal
  writer.SetImage( change.GetOutput() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }

  // Let's read that file back in !
  ImageReader reader2;

  reader2.SetFileName( outfilename.c_str() );
  if ( !reader2.Read() )
    {
    std::cerr << "Could not even reread our generated file : " << outfilename << std::endl;
    return 1;
    }
  // Check that after decompression we still find the same thing:
  int res = 0;
  const gdcm::Image &img = reader2.GetImage();
  //std::cerr << "Success to read image from file: " << filename << std::endl;
  unsigned long len = img.GetBufferLength();
  char* buffer = new char[len];
  bool res2 = img.GetBuffer(buffer);
  if( !res2 )
    {
    std::cerr << "could not get buffer: " << outfilename << std::endl;
    return 1;
    }
  // On big Endian system we have byteswapped the buffer (duh!)
  // Since the md5sum is byte based there is now way it would detect
  // that the file is written in big endian word, so comparing against
  // a md5sum computed on LittleEndian would fail. Thus we need to
  // byteswap (again!) here:
#ifdef GDCM_WORDS_BIGENDIAN
  if( img.GetPixelFormat().GetBitsAllocated() == 16 )
    {
    assert( !(len % 2) );
    assert( img.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME1
      || img.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME2 );
    gdcm::ByteSwap<unsigned short>::SwapRangeFromSwapCodeIntoSystem(
      (unsigned short*)buffer, gdcm::SwapCode::LittleEndian, len/2);
    }
#endif
  const char *ref = gdcm::Testing::GetMD5FromFile(filename);

  char digest[33];
  gdcm::Testing::ComputeMD5(buffer, len, digest);
  if( !ref )
    {
    // new regression image needs a md5 sum
    std::cerr << "Missing md5 " << digest << " for: " << filename <<  std::endl;
    //assert(0);
    res = 1;
    }
  else if( strcmp(digest, ref) )
    {
    std::cerr << "Problem reading image from: " << filename << std::endl;
    std::cerr << "Found " << digest << " instead of " << ref << std::endl;
    res = 1;
    }
  if(res)
    {
    std::cerr << "problem with: " << outfilename << std::endl;
    }
  if( verbose )
    {
    std::cout << "file was written in: " << outfilename << std::endl;
    }

  delete[] buffer;
  return res;
}

} // end namespace gdcm

int TestImageChangeTransferSyntax6(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestImageChangeTransferSyntaxRAWBE(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += gdcm::TestImageChangeTransferSyntaxRAWBE( filename );
    ++i;
    }

  return r;
}
