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
#include "gdcmImageWriter.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmTesting.h"
#include "gdcmByteSwap.h"

namespace gdcm
{
int TestImageWrite(const char *subdir, const char* filename)
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

  // Create directory first:
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );

  ImageWriter writer;
  writer.SetFileName( outfilename.c_str() );
  writer.SetImage( reader.GetImage() );
  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }
  std::cout << "success: " << outfilename << std::endl;

  // Let's read that file back in !
  ImageReader reader2;

  reader2.SetFileName( outfilename.c_str() );
  if ( reader2.Read() )
    {
    int res = 0;
    const Image &img = reader2.GetImage();
    //std::cerr << "Success to read image from file: " << filename << std::endl;
    unsigned long len = img.GetBufferLength();
    char* buffer = new char[len];
    img.GetBuffer(buffer);
    // On big Endian system we have byteswapped the buffer (duh!)
    // Since the md5sum is byte based there is now way it would detect
    // that the file is written in big endian word, so comparing against
    // a md5sum computed on LittleEndian would fail. Thus we need to
    // byteswap (again!) here:
#ifdef GDCM_WORDS_BIGENDIAN
    if( img.GetPixelFormat().GetBitsAllocated() == 16 )
      {
      assert( !(len % 2) );
      assert( img.GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME1
        || img.GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME2 );
      ByteSwap<unsigned short>::SwapRangeFromSwapCodeIntoSystem(
        (unsigned short*)buffer, SwapCode::LittleEndian, len/2);
      }
#endif
    // reuse the filename, since outfilename is simply the new representation of the old filename
    const char *ref = Testing::GetMD5FromFile(filename);

    char digest[33] = {};
    Testing::ComputeMD5(buffer, len, digest);
    if( !ref )
      {
      // new regression image needs a md5 sum
      std::cout << "Missing md5 " << digest << " for: " << outfilename <<  std::endl;
      //assert(0);
      res = 1;
      }
    else if( strcmp(digest, ref) )
      {
      std::cerr << "Problem reading image from: " << outfilename << std::endl;
      std::cerr << "Found " << digest << " instead of " << ref << std::endl;
      res = 1;
#if 0
      std::ofstream debug("/tmp/dump.gray", std::ios::binary);
      debug.write(buffer, len);
      debug.close();
      //assert(0);
#endif
      }
    delete[] buffer;
    return res;
    }

#if 0
  // Ok we have now two files let's compare their md5 sum:
  char digest[33], outdigest[33];
  System::ComputeFileMD5(filename, digest);
  System::ComputeFileMD5(outfilename.c_str(), outdigest);
  if( strcmp(digest, outdigest) )
    {
    // too bad the file is not identical, so let's be paranoid and
    // try to reread-rewrite this just-writen file:
    // TODO: Copy file System::CopyFile( );
    if( TestImageWrite( outfilename.c_str() ) )
      {
      std::cerr << filename << " and "
        << outfilename << " are different\n";
      return 1;
      }
    // In theory I need to compare the two documents to check they
    // are identical... TODO
    std::cerr << filename << " and "
      << outfilename << " should be compatible\n";
    return 0;
    }
  else
    {
    std::cerr << filename << " and "
      << outfilename << " are identical\n";
    return 0;
    }
#endif
  return 0;
}
}

int TestImageWriter(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return gdcm::TestImageWrite(argv[0],filename);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += gdcm::TestImageWrite(argv[0], filename );
    ++i;
    }

  return r;
}
