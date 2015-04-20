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
#include "gdcmFileMetaInformation.h"
#include "gdcmSystem.h"
#include "gdcmFilename.h"
#include "gdcmByteSwap.h"
#include "gdcmTrace.h"
#include "gdcmTesting.h"
#include "gdcmImageHelper.h"
#include "gdcmImageReader.h"

#include "gdcmImage.h"
#include "gdcmMediaStorage.h"

#include "gdcmRAWCodec.h"
#include "gdcmJPEGLSCodec.h"

int TestStreamImageRead(const char* filename, bool verbose = false, bool lossydump = false)
{
  (void)lossydump;
  if( verbose )
    std::cerr << "Reading: " << filename << std::endl;
  gdcm::StreamImageReader reader;

  reader.SetFileName( filename );
  bool canReadImageInformation = reader.ReadImageInformation();
  if (!canReadImageInformation)
    {
    return 0; //unable to read tags as expected.
    }
  else
    {
    int res = 0;

    //let's be tricky; each image will be read in portions, first the top half, then the bottom
    //that way, we can test how the stream handles fragmentation of the data
    //we could also loop this to get various different size combinations, but I'm not sure
    //that's useful, yet.
    std::vector<unsigned int> extent =
      gdcm::ImageHelper::GetDimensionsValue(reader.GetFile());

    //at this point, these values aren't used, but may be in the future
    //unsigned short xmin = 0;
    //unsigned short xmax = extent[0];
    //unsigned short ymin = 0;
    //unsigned short ymax = extent[1];
    //unsigned short zmin = 0;
    //unsigned short zmax = extent[2];
      
    reader.DefinePixelExtent(0, (uint16_t)extent[0], 0, (uint16_t)extent[1], 0, (uint16_t)extent[2]);
    unsigned long len = reader.DefineProperBufferLength();
    char* finalBuffer = new char[len];
    memset(finalBuffer, 0, sizeof(char)*len);
    if (reader.CanReadImage()){
      bool result = reader.Read(finalBuffer, len);
      if( !result ){
        std::cerr << "res2 failure:" << filename << std::endl;
        delete [] finalBuffer;
        return 1;
      }
    } else {
      delete [] finalBuffer;
      return 0; //essentially, we're going to skip this file since it can't be read by the streamer
    }
/*
    //now, read in smaller buffer extents
    reader.DefinePixelExtent(xmin, xmax, ymin, ymax);
    len = reader.DefineProperBufferLength();

    char* buffer = new char[len];
    bool res2 = reader.Read(buffer, len);
    if( !res2 ){
      std::cerr << "res2 failure:" << filename << std::endl;
      return 1;
    }
    //copy the result into finalBuffer
    memcpy(finalBuffer, buffer, len);

    //now read the next half of the image
    ymin = ymax;
    ymax = extent[1];

    reader.DefinePixelExtent(xmin, xmax, ymin, ymax);

    //std::cerr << "Success to read image from file: " << filename << std::endl;
    unsigned long len2 = reader.DefineProperBufferLength();

    char* buffer2 = new char[len2];
    bool res3 = reader.Read(buffer2, len2);
    if( !res3 ){
      std::cerr << "res3 failure:" << filename << std::endl;
      return 1;
    }
    //copy the result into finalBuffer
    memcpy(&(finalBuffer[len]), buffer2, len2);

    delete [] buffer;
    delete [] buffer2;
*/
    // On big Endian system we have byteswapped the buffer (duh!)
    // Since the md5sum is byte based there is now way it would detect
    // that the file is written in big endian word, so comparing against
    // a md5sum computed on LittleEndian would fail. Thus we need to
    // byteswap (again!) here:
    const char *ref = gdcm::Testing::GetMD5FromFile(filename);
    const char *correct_ref = gdcm::Testing::GetMD5FromBrokenFile(filename);

    char digest[33];
    gdcm::Testing::ComputeMD5(finalBuffer, len, digest);
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
      if( correct_ref && !strcmp(correct_ref, ref) )
        {
        std::cerr << "Problem reading image from: " << filename << std::endl;
        std::cerr << "Found " << digest << " instead of " << ref << std::endl;
        /*
        //uncomment this code to explicitly read and check the md5 hash
        gdcm::ImageReader reader;
        
        reader.SetFileName( filename );
        if (reader.Read())
          {
          const gdcm::Image &img = reader.GetImage();
          //std::cerr << "Success to read image from file: " << filename << std::endl;
          unsigned long len2 = img.GetBufferLength();
          
          char* buffer2 = new char[len];
          bool res3 = img.GetBuffer(buffer2);
          if (res3)
            {
            char digest2[33];
            gdcm::Testing::ComputeMD5(buffer2, len2, digest2);
            if (strcmp(digest2, digest))
              {
              std::cerr << "Second check read " << digest2 << " not " << digest << std::endl;
              }
            }
          delete [] buffer2;
          }
        */
        res = 1;
        } 
#if 0
      std::ofstream debug("/tmp/dump.gray",std::ios::binary);
      debug.write(finalBuffer, len);
      debug.close();
#endif
      }
    delete[] finalBuffer;
    if (res == 0) 
    {
      //uncomment this line to determine if a file was correctly read
      //std::cerr << "Correctly read " << filename << std::endl;
    }
    return res;
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
  return 0;
#endif
}


int TestStreamImageReader(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestStreamImageRead(filename, true);
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
    r += TestStreamImageRead( filename);
    ++i;
    }

  return r;
}
