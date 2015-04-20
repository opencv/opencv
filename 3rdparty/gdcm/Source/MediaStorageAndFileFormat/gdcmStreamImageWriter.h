/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef GDCMSTREAMIMAGEWRITER_H
#define GDCMSTREAMIMAGEWRITER_H

#include "gdcmWriter.h"
#include <iostream>
#include "gdcmDataSet.h"

namespace gdcm
{

class MediaStorage;
class RAWCodec;
/**
 * \brief StreamImageReader
 * \note its role is to convert the DICOM DataSet into a Image
 * representation via an ITK streaming (ie, multithreaded) interface
 * Image is different from Pixmap has it has a position and a direction in
 * Space.
 * Currently, this class is threadsafe in that it can read a single extent
 * in a single thread.  Multiple versions can be used for multiple extents/threads.
 *
 * \see Image
 */
class GDCM_EXPORT StreamImageWriter
{

public:
  StreamImageWriter();
  virtual ~StreamImageWriter();


  /// One of either SetFileName or SetStream must be called prior
  /// to any other functions.  These initialize an internal Reader class
  /// to be able to get non-pixel image information.
  void SetFileName(const char* inFileName);
  void SetStream(std::ostream& inStream);

  /// Defines an image extent for the Read function.
  /// DICOM states that an image can have no more than 2^16 pixels per edge (as of 2009)
  /// In this case, the pixel extents ignore the direction cosines entirely, and
  /// assumes that the origin of the image is at location 0,0 (regardless of the definition
  /// in space per the tags).  So, if the first 100 pixels of the first row are to be read in,
  /// this function should be called with DefinePixelExtent(0, 100, 0, 1), regardless
  /// of pixel size or orientation.
  /// 15 nov 2010: added z dimension, defaults to being 1 plane large
  void DefinePixelExtent(uint16_t inXMin, uint16_t inXMax,
    uint16_t inYMin, uint16_t inYMax, uint16_t inZMin = 0, uint16_t inZMax = 1);


  /// Paying attention to the pixel format and so forth, define the proper buffer length for the user.
  /// The return amount is in bytes.
  /// If the return is 0, then that means that the pixel extent was not defined prior
  /// this return is for RAW inputs which are then encoded by the writer, but are used
  /// to ensure that the writer gets the proper buffer size
  uint32_t DefineProperBufferLength();

  /// Read the DICOM image. There are three reasons for failure:
  /// 1. The extent is not set
  /// 2. the conversion from void* to std::ostream (internally) fails
  /// 3. the given buffer isn't large enough to accomodate the desired pixel extent.
  /// This method has been implemented to look similar to the metaimageio in itk
  /// MUST have an extent defined, or else Read will return false.
  /// If no particular extent is required, use ImageReader instead.
  bool Write(void* inWriteBuffer, const std::size_t& inBufferLength);

  /// Write the header information to disk, and a bunch of zeros for the actual pixel information
  /// Of course, if we're doing a non-compressed format, that works
  /// but if it's compressed, we have to force the ordering of chunks that are written.
  virtual bool WriteImageInformation();
  
  /// This function determines if a file can even be written using the streaming writer
  /// unlike the reader, can be called before WriteImageInformation, but must be called
  /// after SetFile.
  bool CanWriteFile() const;
  

  /// Set the image information to be written to disk that is everything but
  /// the pixel information: (7fe0,0010) PixelData
  void SetFile(const File& inFile);

protected:

  //contains the PrepareWrite function, which will get the given dataset ready
  //for writing to disk by manufacturing the header information.
  //note that if there is a pixel element in the given dataset, that will be removed
  //during the copy, so that the imagewriter can write everything else out
  Writer mWriter;

  //is the offset necessary if we always append?
  //std::streamoff mFileOffset; //the fileoffset for getting header information
  SmartPointer<File> mspFile; //all the non-pixel information

  //for thread safety, these should not be stored here, but should be used
  //for every read subregion operation.
  uint16_t mXMin, mYMin, mXMax, mYMax, mZMin, mZMax;

  /// Using the min, max, etc set by DefinePixelExtent, this will fill the given buffer
  ///  Make sure to call DefinePixelExtent and to initialize the buffer with the
  /// amount given by DefineProperBufferLength prior to calling this.
  /// reads by the RAW codec; other codecs are added once implemented
  //virtual bool ReadImageSubregionRAW(std::ostream& os);
  virtual bool WriteImageSubregionRAW(char* inWriteBuffer, const std::size_t& inBufferLength);

  /// when writing a raw file, we know the full extent, and can just write the first
  /// 12 bytes out (the tag, the VR, and the size)
  /// when we do compressed files, we'll do it in chunks, as described in
  /// 2009-3, part 5, Annex A, section 4.
  /// Pass the raw codec so that in the rare case of a bigendian explicit raw,
  /// the first 12 bytes written out should still be kosher.
  /// returns -1 if there's any failure, or the complete offset (12 bytes)
  /// if it works.  Those 12 bytes are then added to the position in order to determine
  /// where to write.
  int WriteRawHeader(RAWCodec* inCodec, std::ostream* inStream);

  /// The result of WriteRawHeader (or another header, when that's implemented)
  /// This result is saved so that the first N bytes aren't constantly being
  /// rewritten for each chunk that's passed in.
  /// For compressed data, the offset table will require rewrites of data.
  int mElementOffsets;
  int mElementOffsets1;

};


} // end namespace gdcm

#endif //GDCMSTREAMIMAGEWRITER_H
