/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmStreamImageReader.h"
#include "gdcmStreamImageWriter.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmSystem.h"
#include "gdcmFilename.h"
#include "gdcmByteSwap.h"
#include "gdcmTrace.h"
#include "gdcmTesting.h"
#include "gdcmImageHelper.h"
#include "gdcmImageReader.h"
#include "gdcmImage.h"

int TestStreamImageWrite(const char *subdir, const char* filename, bool verbose = false, bool lossydump = false)
{
  (void)lossydump;
  if( verbose )
    std::cerr << "Reading and writing: " << filename << std::endl;
  gdcm::ImageReader theImageReaderOriginal;
  gdcm::ImageReader theImageReader;
  gdcm::StreamImageWriter theStreamWriter;

  theImageReaderOriginal.SetFileName( filename );

  // Create directory first:
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
      gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = gdcm::Testing::GetTempFilename( filename, subdir );


  if ( theImageReaderOriginal.Read() )
    {
    //int res = 0;

    //to test the writer out, we have to read an image and then
    //write it out one line at a time.
    //a _real_ test would include both the streamimagereader and writer.
    //so, this test is:
    //1) read in the image via the stream reader
    //2) write it out line by line
    //3) read it in by a reader
    //4) compare the image from step 1 with step 3
    //5) go to step2, replace step 3 with a regular image reader.
    //for now, we'll do 1-4

    //pull image information prior to messing with the file
    gdcm::Image theOriginalImage = theImageReaderOriginal.GetImage();
    char* theOriginalBuffer = new char[theOriginalImage.GetBufferLength()];
    if (!theOriginalImage.GetBuffer(theOriginalBuffer)){
      std::cerr << "Unable to get original image buffer, stopping." << std::endl;
      delete [] theOriginalBuffer;
      return 1;
    }

    //first, check that the image information can be written
    //theStreamReader.GetFile().GetDataSet().Print( std::cout );

    theStreamWriter.SetFile(theImageReaderOriginal.GetFile());
#if 0
    theStreamWriter.SetFileName(outfilename.c_str());
#else
    std::ofstream of;
    of.open( outfilename.c_str(), std::ios::out | std::ios::binary );
    theStreamWriter.SetStream(of);
#endif
    if (!theStreamWriter.CanWriteFile()){
      delete [] theOriginalBuffer;
      return 0;//this means that the file was unwritable, period.
      //very similar to a ReadImageInformation failure
    }
    if (!theStreamWriter.WriteImageInformation()){
      std::cerr << "unable to write image information" << std::endl;
      delete [] theOriginalBuffer;
      return 1; //the CanWrite function should prevent getting here, else,
      //that's a test failure
    }
    std::vector<unsigned int> extent =
      gdcm::ImageHelper::GetDimensionsValue(theImageReaderOriginal.GetFile());

    unsigned short xmax = (unsigned short)extent[0];
    unsigned short ymax = (unsigned short)extent[1];
    unsigned short theChunkSize = 4;
    unsigned short ychunk = (unsigned short)(extent[1]/theChunkSize); //go in chunk sizes of theChunkSize
    unsigned short zmax = (unsigned short)extent[2];

    if (xmax == 0 || ymax == 0)
      {
      std::cerr << "Image has no size, unable to write zero-sized image." << std::endl;
      return 0;
      }

    int z, y, nexty;
    unsigned long prevLen = 0; //when going through the char buffer, make sure to grab
    //the bytes sequentially.  So, store how far you got in the buffer with each iteration.
    for (z = 0; z < zmax; ++z){
      for (y = 0; y < ymax; y += ychunk){
        nexty = y + ychunk;
        if (nexty > ymax) nexty = ymax;
        theStreamWriter.DefinePixelExtent(0, (uint16_t)xmax, (uint16_t)y, (uint16_t)nexty, (uint16_t)z, (uint16_t)(z+1));
        unsigned long len = theStreamWriter.DefineProperBufferLength();
        char* finalBuffer = new char[len];
        memcpy(finalBuffer, &(theOriginalBuffer[prevLen]), len);

        if (!theStreamWriter.Write(finalBuffer, len)){
          std::cerr << "writing failure:" << outfilename << " at y = " << y << " and z= " << z << std::endl;
          delete [] theOriginalBuffer;
          delete [] finalBuffer;
          return 1;
        }
        delete [] finalBuffer;
        prevLen += len;
      }
    }
    delete [] theOriginalBuffer;
    theImageReader.SetFileName(outfilename.c_str());
    if (!theImageReader.Read()){
      std::cerr << "unable to read in the written test file: " << outfilename << std::endl;
      return 1;
    } else {
      int res = 0;
      const gdcm::Image &img = theImageReader.GetImage();
      //std::cerr << "Success to read image from file: " << filename << std::endl;
      unsigned long len = img.GetBufferLength();
      char* buffer = new char[len];
      img.GetBuffer(buffer);
      // On big Endian system we have byteswapped the buffer (duh!)
      // Since the md5sum is byte based there is now way it would detect
      // that the file is written in big endian word, so comparing against
      // a md5sum computed on LittleEndian would fail. Thus we need to
      // byteswap (again!) here:
      const char *ref = gdcm::Testing::GetMD5FromFile(filename);
      const char *correct_ref = gdcm::Testing::GetMD5FromBrokenFile(filename);

      char digest[33];
      gdcm::Testing::ComputeMD5(buffer, len, digest);
      if( verbose )
        {
          std::cout << "ref=" << ref << std::endl;
          std::cout << "md5=" << digest << std::endl;
        }
      if( !ref )
        {
          // new regression image needs a md5 sum
          std::cout << "Missing md5 " << digest << " for: " << filename <<  std::endl;
          //assert(0);
          res = 1;
        }
      else if( strcmp(digest, ref) )
        {
          
          // let's be nice for now and only truly fails when file is proper DICOM
          if( correct_ref  && !strcmp(correct_ref, ref))
            {
            std::cerr << "Problem reading image from: " << filename << std::endl;
            std::cerr << "Found " << digest << " instead of " << ref << std::endl;
            res = 1;
            }
        }
      delete[] buffer;
      return res;
    }
  }
  else
    {
    //std::cerr << "Unable to read test file: " << filename << std::endl;
    //return 1;
      return 0; //this is NOT a test of the reader, but a test of streaming writing
    }

#if 0
  const gdcm::FileMetaInformation &header = reader.GetFile().GetHeader();
  gdcm::MediaStorage ms = header.GetMediaStorage();
  bool isImage = gdcm::MediaStorage::IsImage( ms );
  if( isImage )
    {
    if( reader.GetFile().GetDataSet().FindDataElement( gdcm::Tag(0x7fe0,0x0010) ) )
      {
      std::cerr << "Failed to read image from file: " << filename << std::endl;
      return 1;
      }
    else
      {
      std::cerr << "no Pixel Data Element found in the file:" << filename << std::endl;
      return 0;
      }
    }
  // else
  // well this is not an image, so thankfully we fail to read it
  std::cerr << "Could not read image(" << filename << "), since file is a: " << ms << std::endl;
  //assert( ms != gdcm::MediaStorage::MS_END );
#endif
  return 0;
}

int TestStreamImageWriter(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestStreamImageWrite(argv[0], filename, true);
    }

  // else
  // First of get rid of warning/debug message
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestStreamImageWrite( argv[0], filename);
    ++i;
    }

  return r;
}
