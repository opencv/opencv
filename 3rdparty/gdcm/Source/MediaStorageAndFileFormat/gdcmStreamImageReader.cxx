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
#include "gdcmStreamImageReader.h"

#include "gdcmImage.h"
#include "gdcmMediaStorage.h"
#include "gdcmImageHelper.h"
#include "gdcmRAWCodec.h"
#include "gdcmJPEGLSCodec.h"

#include <algorithm>

namespace gdcm
{

//see http://stackoverflow.com/questions/1448467/initializing-a-c-stdistringstream-from-an-in-memory-buffer/1449527#1449527
struct OneShotReadBuf : public std::streambuf
{
  OneShotReadBuf(void* s, std::size_t n){
    char* cast = (char*)s;
    setg(cast, cast, cast+n);
  }
};

StreamImageReader::StreamImageReader()
{
  //set these values to be the opposite ends of possible,
  //so that if the extent is not defined, read can fail properly.
  mXMin = mYMin = mZMin = std::numeric_limits<uint16_t>::max();
  mXMax = mYMax = mZMax = std::numeric_limits<uint16_t>::min();
}

StreamImageReader::~StreamImageReader()
{
}

/// One of either SetFileName or SetStream must be called prior
/// to any other functions.
void StreamImageReader::SetFileName(const char* inFileName)
{
  mReader.SetFileName(inFileName);
}

void StreamImageReader::SetStream(std::istream& inStream)
{
  mReader.SetStream(inStream);
}

std::vector<unsigned int> StreamImageReader::GetDimensionsValueForResolution( unsigned int res )
{
  std::vector<unsigned int> extent(3);
  File &file_t = mReader.GetFile();
  DataSet &ds_t = file_t.GetDataSet();

  const DataElement &seq = ds_t.GetDataElement( Tag(0x0048,0x0200) );
  SmartPointer<SequenceOfItems> sqi = seq.GetValueAsSQ();

  Item &itemL = sqi->GetItem(res);
  DataSet &subds_L = itemL.GetNestedDataSet();

  const DataElement &brrL = subds_L.GetDataElement( Tag(0x0048,0x0202) );
  Element<VR::US,VM::VM2> elL1;
  elL1.SetFromDataElement( brrL );
  extent[0] = elL1.GetValue(0);
  extent[1] = elL1.GetValue(1);
  extent[2] = res;
  return extent;
}

/// Defines an image extent for the Read function.
/// DICOM states that an image can have no more than 2^16 pixels per edge (as of 2009)
/// In this case, the pixel extents ignore the direction cosines entirely, and
/// assumes that the origin of the image is at location 0,0 (regardless of the definition
/// in space per the tags).  So, if the first 100 pixels of the first row are to be read in,
/// this function should be called with DefinePixelExtent(0, 100, 0, 1), regardless
/// of pixel size or orientation.
void StreamImageReader::DefinePixelExtent(uint16_t inXMin, uint16_t inXMax,
                                          uint16_t inYMin, uint16_t inYMax,
                                          uint16_t inZMin, uint16_t inZMax){
  mXMin = inXMin;
  mYMin = inYMin;
  mXMax = inXMax;
  mYMax = inYMax;
  mZMin = inZMin;
  mZMax = inZMax;
}
/// Paying attention to the pixel format and so forth, define the proper buffer length for the user.
/// The return amount is in bytes.
/// If the return is 0, then that means that the pixel extent was not defined prior
uint32_t StreamImageReader::DefineProperBufferLength() const
{
  if (mXMax < mXMin || mYMax < mYMin || mZMax < mZMin) return 0;
  PixelFormat pixelInfo = ImageHelper::GetPixelFormatValue(mReader.GetFile());
  //unsigned short samplesPerPixel = pixelInfo.GetSamplesPerPixel();
  int bytesPerPixel = pixelInfo.GetPixelSize();
  return (mYMax - mYMin)*(mXMax - mXMin)*(mZMax - mZMin)*bytesPerPixel;
}

/// Read the DICOM image. There are two reason for failure:
/// 1. The extent is not set
/// 2. The output buffer is not set
/// This method has been implemented to look similar to the metaimageio in itk
bool StreamImageReader::Read(char* inReadBuffer, const std::size_t& inBufferLength)
{
  //need to have some kind of extent defined.
  if (mXMin > mXMax || mYMin > mYMax || mZMin > mZMax)
    {
    return false; //for now
    }

//  OneShotReadBuf osrb(inReadBuffer, inBufferLength);
//  std::ostream ostr(&osrb);

  //put the codec interpretation here

//  return ReadImageSubregionRAW(ostr);
  //just do memcpys instead of doing this stream shenanigans
  return ReadImageSubregionRAW(inReadBuffer, inBufferLength);
}

/** Read a particular subregion, using the stored mFileOffset as the beginning of the stream.
    This class reads uncompressed data; other subclasses will reimplement this function for compression.
    Assumes that the given buffer is the size in bytes returned from DefineProperBufferLength.
    */
bool StreamImageReader::ReadImageSubregionRAW(char* inReadBuffer, const std::size_t& inBufferLength)
{
  //assumes that the file is organized in row-major format, with each row rastering across
  assert( mFileOffset != -1 );
  (void)inBufferLength;
  int y, z;
  std::streamoff theOffset;

  //need to get the pixel size information
  //should that come from the header?
  //most likely  that's a tag in the header
  std::vector<unsigned int> extent = ImageHelper::GetDimensionsValue(mReader.GetFile());
  PixelFormat pixelInfo = ImageHelper::GetPixelFormatValue(mReader.GetFile());
  //unsigned short samplesPerPixel = pixelInfo.GetSamplesPerPixel();
  int bytesPerPixel = pixelInfo.GetPixelSize();
  int SubRowSize = mXMax - mXMin;
  int SubColSize = mYMax - mYMin;

  //set up the codec prior to resetting the file, just in case that affects the way that
  //files are handled by the ImageHelper

  const FileMetaInformation &header = mReader.GetFile().GetHeader();
  const TransferSyntax &ts = header.GetDataSetTransferSyntax();
  bool needbyteswap = (ts == TransferSyntax::ImplicitVRBigEndianPrivateGE);

  RAWCodec theCodec;
  if( !theCodec.CanDecode(ts) )
    {
    JPEGLSCodec theJpegLSCodec;
    if (!theJpegLSCodec.CanDecode(ts))
      {
      gdcmDebugMacro( "Raw Codec cannot decode this file by streaming." );
      return false;
      } 
    else 
      {
      //read in the entire file for jpegls
      //right now, it needs to be read entirely off of disk first
      //kind of a shame, but it's the way it is now
      mReader.Read();
      }
    }

  theCodec.SetNeedByteSwap( needbyteswap );
  theCodec.SetDimensions(ImageHelper::GetDimensionsValue(mReader.GetFile()));
  theCodec.SetPlanarConfiguration(
  ImageHelper::GetPlanarConfigurationValue(mReader.GetFile()));
  theCodec.SetPhotometricInterpretation(
    ImageHelper::GetPhotometricInterpretationValue(mReader.GetFile()));
  //how do I handle byte swapping here?  where is it set?

  //have to reset the stream to the proper position
  //first, reopen the stream,then the loop should set the right position
  //mReader.SetFileName(mReader.GetFileName().c_str());
  std::istream* theStream = mReader.GetStreamPtr();//probably going to need a copy of this

  //to ensure thread safety; if the stream ptr handler gets used simultaneously by different threads,
  //that would be BAD
  //tmpBuffer is for a single raster

  char* tmpBuffer = new char[SubRowSize*bytesPerPixel];
  char* tmpBuffer2 = new char[SubRowSize*bytesPerPixel];
  try
    {
    for (z = mZMin; z < mZMax; ++z)
      {
#if 0
      theStream->seekg(std::ios::beg);

      if(mFileOffset1 == 0)
        {
        mFileOffset1 = mFileOffset;
        for(int j = 1; j<=z; j++)
          {
          std::vector<unsigned int> extent = GetDimensionsValueForResolution(j);
          mFileOffset1 =  mFileOffset1 + (int)((extent[1])*(extent[0])*bytesPerPixel);
          mFileOffset1 = mFileOffset1 + 4*sizeof(uint16_t);
          }
        }
      else
        {
        mFileOffset1 = mFileOffset;
        }

      std::vector<unsigned int> extent = GetDimensionsValueForResolution(z+1);
#endif

      for (y = mYMin; y < mYMax; ++y)
        {
#if 0
        theOffset = mFileOffset1 + (y*(int)(extent[0]) + mXMin)*bytesPerPixel;
#else
        theStream->seekg(std::ios::beg);
        theOffset = mFileOffset + (z * (int)(extent[1]*extent[0]) + y*(int)extent[0] + mXMin)*bytesPerPixel;
#endif
        theStream->seekg(theOffset);
        theStream->read(tmpBuffer, SubRowSize*bytesPerPixel);
        //now, convert that buffer.
        if (!theCodec.DecodeBytes(tmpBuffer, SubRowSize*bytesPerPixel,
            tmpBuffer2, SubRowSize*bytesPerPixel))
          {
          delete [] tmpBuffer;
          delete [] tmpBuffer2;
          return false;
          }
        //this next line may require a bit of finagling...
        //std::copy(tmpBuffer2, &(tmpBuffer2[SubRowSize*bytesPerPixel]), std::ostream_iterator<char>(os));
        //make sure to have a test that will test different x, y, and z mins and maxes
        memcpy(&(inReadBuffer[((z-mZMin)*SubRowSize*SubColSize +
              (y-mYMin)*SubRowSize)// + mXMin)//shouldn't need mXMin
          *bytesPerPixel]), tmpBuffer2, SubRowSize*bytesPerPixel);
        }
#if 0
      if((mYMax == extent[1]) && (mXMax == extent[0]))
        {
        mFileOffset1 = mFileOffset1 + (int)((extent[1])*(extent[0])*bytesPerPixel) + 4*sizeof(uint16_t);
        }
#endif
      }
#if 0
    mFileOffset = mFileOffset1;
#endif
    }

  catch (std::exception & ex)
    {
    (void)ex;
    gdcmWarningMacro( "Failed to read with ex:" << ex.what() );
    delete [] tmpBuffer;
    delete [] tmpBuffer2;
    return false;
    } 
  catch (...)
    {
    gdcmWarningMacro( "Failed to read with unknown error." );
    delete [] tmpBuffer;
    delete [] tmpBuffer2;
    return false;
    }

  delete [] tmpBuffer;
  delete [] tmpBuffer2;
  return true;
}

/** This class reads via the jpegls codec.
Due to limitations in that codec, the entire file must be read into memory before a subregion
can be decoded.
 */
bool StreamImageReader::ReadImageSubregionJpegLS(char* inReadBuffer, const std::size_t& inBufferLength) {
  //assumes that the file is organized in row-major format, with each row rastering across
  //don't need to get all the other stuff (ie, the file offset) since we have to read it all in anyway
  
  //set up the codec prior to resetting the file, just in case that affects the way that
  //files are handled by the ImageHelper
  
  const FileMetaInformation &header = mReader.GetFile().GetHeader();
  const TransferSyntax &ts = header.GetDataSetTransferSyntax();
  bool needbyteswap = (ts == TransferSyntax::ImplicitVRBigEndianPrivateGE);
  
  JPEGLSCodec theCodec;
  if (!theCodec.CanDecode(ts))
  {
    gdcmDebugMacro( "JpegLS cannot read this." );
    return false;
  }
  //read in the entire file for jpegls
  //right now, it needs to be read entirely off of disk first
  //kind of a shame, but it's the way it is now
  mReader.Read();
  
  theCodec.SetNeedByteSwap( needbyteswap );
  theCodec.SetDimensions(ImageHelper::GetDimensionsValue(mReader.GetFile()));
  theCodec.SetPlanarConfiguration(ImageHelper::GetPlanarConfigurationValue(mReader.GetFile()));
  theCodec.SetPhotometricInterpretation(ImageHelper::GetPhotometricInterpretationValue(mReader.GetFile()));
  
  try {
    DataSet ds = mReader.GetFile().GetDataSet();
    Tag thePixelDataTag(0x7fe0, 0x0010);
    DataElement de = ds.GetDataElement(thePixelDataTag);
    theCodec.Decode(de, inReadBuffer, inBufferLength, mXMin, mXMax, mYMin, mYMax, mZMin, mZMax);
  }
  catch (std::exception & ex){
    (void)ex;
    gdcmWarningMacro( "Failed to read with ex:" << ex.what() );
    return false;
  } 
  catch (...){
    gdcmWarningMacro( "Failed to read with unknown error." );
    return false;
  }
  return true;
}

/// Set the spacing and dimension information for the set filename.
/// returns false if the file is not initialized or not an image,
/// with the pixel 0x7fe0, 0x0010 tag.
bool StreamImageReader::ReadImageInformation()
{
  //read up to the point in the stream where the pixel information tag is
  //store that location and keep the rest of the data as the header information dataset
  std::set<Tag> theSkipTags;
  Tag thePixelDataTag(0x7fe0, 0x0010);//must be LESS than the pixel information tag, 0x7fe0,0x0010
  //otherwise, it'll read that tag as well.
  //make a reader object in readimageinformation
  //call read up to tag
  //then create data structures from that dataset that's been read-up-to
  theSkipTags.insert(thePixelDataTag);

  try
    {
    //ok, need to read up until I know what kind of endianness i'm dealing with?
    if (!mReader.ReadUpToTag(thePixelDataTag, theSkipTags))
      {
      gdcmWarningMacro("Failed to read tags in the gdcm stream image reader.");
      return false;
      }
#if 0
    std::streampos shift = 4*sizeof(uint16_t)+2*sizeof(uint32_t);
    mFileOffset = mReader.GetStreamPtr()->tellg()+ shift;
    mFileOffset1 = 0;
#else
    mFileOffset = mReader.GetStreamPtr()->tellg();
#endif
    }
  catch(std::exception & ex)
    {
    (void)ex;
    gdcmWarningMacro( "Failed to read with ex:" << ex.what() );
    return false;
    }
  catch(...)
    {
    gdcmWarningMacro( "Failed to read with unknown error" );
    return false;
    }

  // eg. ELSCINT1_PMSCT_RLE1.dcm
  if( mFileOffset == -1 ) return false;

  // postcondition
  assert( mFileOffset != -1 );

  const File &file_t = mReader.GetFile();
  const DataSet &ds_t = file_t.GetDataSet();

  MediaStorage ms;
  ms.SetFromFile(file_t);

  if( ms == MediaStorage::VLWholeSlideMicroscopyImageStorage )
    {
    if( !ds_t.FindDataElement( Tag(0x0048,0x0200) ) )
      {
      gdcmWarningMacro( "error occured in WSI File read" );
      return false;
      }

    DataElement seq = ds_t.GetDataElement( Tag(0x0048,0x0200) );
    SmartPointer<SequenceOfItems> sqi = seq.GetValueAsSQ();
    SequenceOfItems::SizeType s = sqi->GetNumberOfItems();

    Item itemL = sqi->GetItem(1);
    DataSet &subds_L = itemL.GetNestedDataSet();

    if( !subds_L.FindDataElement( Tag(0x0008,0x1160) ) )
      {
      gdcmWarningMacro( "Error occured during WSI File Read" );
      return false;
      }

    DataElement rfnL = subds_L.GetDataElement( Tag(0x0008,0x1160) );
    Element<VR::IS,VM::VM1> elL;
    elL.SetFromDataElement( rfnL );

    if( !subds_L.FindDataElement( Tag(0x0048,0x0202) ) )
      {
      gdcmWarningMacro( "Error During WSI File Read" );
      return false;
      }

    DataElement brrL = subds_L.GetDataElement( Tag(0x0048,0x0202) );
    Element<VR::US,VM::VM2> elL1;
    elL1.SetFromDataElement( brrL );

    Item itemH = sqi->GetItem(s);
    DataSet &subds_H = itemH.GetNestedDataSet();

    if( !subds_H.FindDataElement( Tag(0x0008,0x1160) ) )
      {
      gdcmWarningMacro( "Error occured during WSI File Read" );
      return false;
      }

    DataElement rfnH = subds_H.GetDataElement( Tag(0x0008,0x1160) );
    Element<VR::IS,VM::VM1> elH;
    elH.SetFromDataElement( rfnH );

    if( !subds_H.FindDataElement( Tag(0x0048,0x0202) ) )
      {
      gdcmWarningMacro( "Error During WSI File Read" );
      return false;
      }

    DataElement brrH = subds_H.GetDataElement( Tag(0x0048,0x0202) );
    Element<VR::US,VM::VM2> elH1;
    elH1.SetFromDataElement( brrH );
    }

  return true;
}

bool StreamImageReader::CanReadImage() const
{
  //this is the check to ensure that ReadImageInformation was read in properly
  if (mFileOffset == -1)
    {
    return false;
    }
  
  const FileMetaInformation &header = mReader.GetFile().GetHeader();
  const TransferSyntax &ts = header.GetDataSetTransferSyntax();
  //bool needbyteswap = (ts == TransferSyntax::ImplicitVRBigEndianPrivateGE);
  
  RAWCodec theCodec;
  bool canDecodeRaw = theCodec.CanDecode(ts);
  if (!canDecodeRaw) return false;
  
  std::vector<unsigned int> extent = ImageHelper::GetDimensionsValue(mReader.GetFile());
  if (extent.empty()) return false; //should not happen with current GetDimensionsValue implementation
  //but just in case...
  
  if (extent[0] == 0 || extent[1] == 0)
    return false;
    
  return true;
}

  /// Returns the dataset read by ReadImageInformation
  /// Couple this with the ImageHelper to get statistics about the image,
  /// like pixel extent, to be able to initialize buffers for reading
File const &StreamImageReader::GetFile() const
{
  if (mFileOffset > 0)
    {
    /// mFileOffset1 = 0;
    return mReader.GetFile();
    }
  else
    {
    assert(0);
    return mReader.GetFile();
    }
}

} // end namespace gdcm
