/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSplitMosaicFilter.h"
#include "gdcmCSAHeader.h"
#include "gdcmAttribute.h"
#include "gdcmImageHelper.h"

#include <math.h>

namespace gdcm
{
SplitMosaicFilter::SplitMosaicFilter():F(new File),I(new Image) {}
SplitMosaicFilter::~SplitMosaicFilter() {}

namespace details {
/*
 *  gdcmDataExtra/gdcmSampleData/images_of_interest/MR-sonata-3D-as-Tile.dcm
 */
static bool reorganize_mosaic(const unsigned short *input, const unsigned int *inputdims,
  unsigned int square, const unsigned int *outputdims, unsigned short *output )
{
  for(unsigned int x = 0; x < outputdims[0]; ++x)
    {
    for(unsigned int y = 0; y < outputdims[1]; ++y)
      {
      for(unsigned int z = 0; z < outputdims[2]; ++z)
        {
        const size_t outputidx = x + y*outputdims[0] + z*outputdims[0]*outputdims[1];
        const size_t inputidx = (x + (z%square)*outputdims[0]) +
          (y + (z/square)*outputdims[1])*inputdims[0];
        output[ outputidx ] = input[ inputidx ];
        }
      }
    }
  return true;
}
}

void SplitMosaicFilter::SetImage(const Image& image)
{
  I = image;
}

bool SplitMosaicFilter::ComputeMOSAICDimensions( unsigned int dims[3] )
{
  CSAHeader csa;
  DataSet& ds = GetFile().GetDataSet();

  const PrivateTag &t1 = csa.GetCSAImageHeaderInfoTag();
  if( !csa.LoadFromDataElement( ds.GetDataElement( t1 ) ) )
    {
    return false;
    }

  std::vector<unsigned int> colrow =
    ImageHelper::GetDimensionsValue( GetFile() );
  dims[0] = colrow[0];
  dims[1] = colrow[1];

  // SliceThickness ??
  int numberOfImagesInMosaic = 0;
  if( csa.FindCSAElementByName( "NumberOfImagesInMosaic" ) )
    {
    const CSAElement &csael4 = csa.GetCSAElementByName( "NumberOfImagesInMosaic" );
    if( !csael4.IsEmpty() )
      {
        Element<VR::IS, VM::VM1> el4 = {{ 0 }};
      el4.Set( csael4.GetValue() );
      numberOfImagesInMosaic = el4.GetValue();
      }
    }

  if( !numberOfImagesInMosaic ) return false;

  unsigned int div = (unsigned int )ceil(sqrt( (double)numberOfImagesInMosaic ) );
  dims[0] /= div;
  dims[1] /= div;
  dims[2] = numberOfImagesInMosaic;
  return true;
}

bool SplitMosaicFilter::Split()
{
  bool success = true;
  DataSet& ds = GetFile().GetDataSet();

  unsigned int dims[3] = {0,0,0};
  if( ! ComputeMOSAICDimensions( dims ) )
    {
    return false;
    }
  unsigned int div = (unsigned int )ceil(sqrt( (double)dims[2]) );

  const Image &inputimage = GetImage();
  if( inputimage.GetPixelFormat() != PixelFormat::UINT16 )
    {
    gdcmDebugMacro( "Expecting UINT16 PixelFormat" );
    return false;
    }
  unsigned long l = inputimage.GetBufferLength();
  std::vector<char> buf;
  buf.resize(l);
  inputimage.GetBuffer( &buf[0] );
  DataElement pixeldata( Tag(0x7fe0,0x0010) );

  std::vector<char> outbuf;
  outbuf.resize(l);

  bool b = details::reorganize_mosaic(
    (unsigned short*)&buf[0], inputimage.GetDimensions(), div, dims,
    (unsigned short*)&outbuf[0] );
  if( !b ) return false;

  VL::Type outbufSize = (VL::Type)outbuf.size();
  pixeldata.SetByteValue( &outbuf[0], outbufSize );

  Image &image = GetImage();

  image.SetNumberOfDimensions( 3 );
  image.SetDimension(0, dims[0] );
  image.SetDimension(1, dims[1] );
  image.SetDimension(2, dims[2] );

  PhotometricInterpretation pi;
  pi = PhotometricInterpretation::MONOCHROME2;

  image.SetDataElement( pixeldata );

  // Second part need to fix the Media Storage, now that this is not a single slice anymore
  MediaStorage ms = MediaStorage::SecondaryCaptureImageStorage;
  ms.SetFromFile( GetFile() );

  if( ms == MediaStorage::MRImageStorage )
    {
    // Ok make it a MediaStorage::EnhancedMRImageStorage
//    ms = MediaStorage::EnhancedMRImageStorage;
//
//    // Remove old MRImageStorage attribute then:
//    ds.Remove( Tag(0x0020,0x0032) ); // Image Position (Patient)
//    ds.Remove( Tag(0x0020,0x0037) ); // Image Orientation (Patient)
//    ds.Remove( Tag(0x0028,0x1052) ); // Rescale Intercept
//    ds.Remove( Tag(0x0028,0x1053) ); // Rescale Slope
//    ds.Remove( Tag(0x0028,0x1054) ); // Rescale Type
    }
  else
    {
    gdcmDebugMacro( "Expecting MRImageStorage" );
    return false;
    }
  DataElement de( Tag(0x0008, 0x0016) );
  const char* msstr = MediaStorage::GetMSString(ms);
  VL::Type strlenMsstr = (VL::Type)strlen(msstr);
  de.SetByteValue( msstr, strlenMsstr );
  de.SetVR( Attribute<0x0008, 0x0016>::GetVR() );
  ds.Replace( de );

  return success;
}


} // end namespace gdcm
