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
#ifndef GDCMSTREAMIMAGEREADER_H
#define GDCMSTREAMIMAGEREADER_H

#include "gdcmReader.h"

namespace gdcm
{

class MediaStorage;
/**
 * \brief StreamImageReader
 * \note its role is to convert the DICOM DataSet into a Image
 * representation via an ITK streaming (ie, multithreaded) interface
 * Image is different from Pixmap has it has a position and a direction in
 * Space.
 * Currently, this class is thread safe in that it can read a single extent
 * in a single thread.  Multiple versions can be used for multiple extents/threads.
 *
 * \see Image
 */
class GDCM_EXPORT StreamImageReader
{

public:
  StreamImageReader();
  virtual ~StreamImageReader();

  /// One of either SetFileName or SetStream must be called prior
  /// to any other functions.  These initialize an internal Reader class
  /// to be able to get non-pixel image information.
  void SetFileName(const char* inFileName);
  void SetStream(std::istream& inStream);

  std::vector<unsigned int> GetDimensionsValueForResolution( unsigned int  );

  /// Defines an image extent for the Read function.
  /// DICOM states that an image can have no more than 2^16 pixels per edge (as of 2009)
  /// In this case, the pixel extents ignore the direction cosines entirely, and
  /// assumes that the origin of the image is at location 0,0 (regardless of the definition
  /// in space per the tags).  So, if the first 100 pixels of the first row are to be read in,
  /// this function should be called with DefinePixelExtent(0, 100, 0, 1), regardless
  /// of pixel size or orientation.
  void DefinePixelExtent(uint16_t inXMin, uint16_t inXMax,
    uint16_t inYMin, uint16_t inYMax, uint16_t inZMin = 0, uint16_t inZMax = 1);

  /// Paying attention to the pixel format and so forth, define the proper buffer length for the user.
  /// The return amount is in bytes.  Call this function to determine the size of the char* buffer
  /// that will need to be passed in to ReadImageSubregion().
  /// If the return is 0, then that means that the pixel extent was not defined prior
  uint32_t DefineProperBufferLength() const;

  /// Read the DICOM image. There are three reasons for failure:
  /// 1. The extent is not set
  /// 2. the conversion from char* to std::ostream (internally) fails
  /// 3. the given buffer isn't large enough to accommodate the desired pixel extent.
  /// This method has been implemented to look similar to the metaimageio in itk
  /// MUST have an extent defined, or else Read will return false.
  /// If no particular extent is required, use ImageReader instead.
  bool Read(char* inReadBuffer, const std::size_t& inBufferLength);

  /// Only RAW images are currently readable by the stream reader.  As more
  /// streaming codecs are added, then this function will be updated to reflect
  /// those changes.  Calling this function prior to reading will ensure that 
  /// only streamable files are streamed.  Make sure to call ReadImageInformation
  /// prior to calling this function.
  bool CanReadImage() const;

  /// Set the spacing and dimension information for the set filename.
  /// returns false if the file is not initialized or not an image,
  /// with the pixel (7fe0,0010) tag.
  virtual bool ReadImageInformation();

  /// Returns the dataset read by ReadImageInformation
  /// Couple this with the ImageHelper to get statistics about the image,
  /// like pixel extent, to be able to initialize buffers for reading
  File const & GetFile() const;

protected:
private:
  //contains a reader for being able to ReadUpToTag
  //however, we don't want the user to be able to call Read
  //either directly or via a parent class call, so we hide the reader in here.
  Reader mReader;

  std::streamoff mFileOffset; //the file offset for getting header information
#if 0
  std::streamoff mFileOffset1;
#endif
  DataSet mHeaderInformation; //all the non-pixel information

  //for thread safety, these should not be stored here, but should be used
  //for every read subregion operation.
  uint16_t mXMin, mYMin, mXMax, mYMax, mZMin, mZMax;

  /// Using the min, max, etc set by DefinePixelExtent, this will fill the given buffer
  ///  Make sure to call DefinePixelExtent and to initialize the buffer with the
  /// amount given by DefineProperBufferLength prior to calling this.
  /// reads by the RAW codec; other codecs are added once implemented
  bool ReadImageSubregionRAW(char* inReadBuffer, const std::size_t& inBufferLength);

  /// Reads the file via JpegLS.  The JpegLS codec, as of this writing, requires that the
  /// entire file be read in in order to decode a subregion, so that's what's done here.
  bool ReadImageSubregionJpegLS(char* inReadBuffer, const std::size_t& inBufferLength);
};

} // end namespace gdcm

#endif //GDCMSTREAMIMAGEREADER_H

