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
#include "gdcmExplicitDataElement.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmValue.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmElement.h"
#include "gdcmPhotometricInterpretation.h"
#include "gdcmTransferSyntax.h"
#include "gdcmAttribute.h"
#include "gdcmImageHelper.h"
#include "gdcmPrivateTag.h"
#include "gdcmJPEGCodec.h"

namespace gdcm
{
ImageReader::ImageReader()
{
  PixelData = new Image;
}

ImageReader::~ImageReader()
{
}

const Image& ImageReader::GetImage() const
{
  return dynamic_cast<const Image&>(*PixelData);
}
Image& ImageReader::GetImage()
{
  return dynamic_cast<Image&>(*PixelData);
}

//void ImageReader::SetImage(Image const &img)
//{
//  PixelData = img;
//}

bool ImageReader::Read()
{
  return PixmapReader::Read();
}

bool ImageReader::ReadImage(MediaStorage const &ms)
{
  if( !PixmapReader::ReadImage(ms) )
    {
    return false;
    }
  //const DataSet &ds = F->GetDataSet();
  Image& pixeldata = GetImage();

  // 4 1/2 Let's do Pixel Spacing
  std::vector<double> spacing = ImageHelper::GetSpacingValue(*F);
  // FIXME: Only SC is allowed not to have spacing:
  if( !spacing.empty() )
    {
    assert( spacing.size() >= pixeldata.GetNumberOfDimensions() ); // In MR, you can have a Z spacing, but store a 2D image
    pixeldata.SetSpacing( &spacing[0] );
    if( spacing.size() > pixeldata.GetNumberOfDimensions() ) // FIXME HACK
      {
      pixeldata.SetSpacing(pixeldata.GetNumberOfDimensions(), spacing[pixeldata.GetNumberOfDimensions()] );
      }
    }
  // 4 2/3 Let's do Origin
  std::vector<double> origin = ImageHelper::GetOriginValue(*F);
  if( !origin.empty() )
    {
    pixeldata.SetOrigin( &origin[0] );
    if( origin.size() > pixeldata.GetNumberOfDimensions() ) // FIXME HACK
      {
      pixeldata.SetOrigin(pixeldata.GetNumberOfDimensions(), origin[pixeldata.GetNumberOfDimensions()] );
      }
    }

  std::vector<double> dircos = ImageHelper::GetDirectionCosinesValue(*F);
  if( !dircos.empty() )
    {
    pixeldata.SetDirectionCosines( &dircos[0] );
    }

  // Do the Rescale Intercept & Slope
  std::vector<double> is = ImageHelper::GetRescaleInterceptSlopeValue(*F);
  pixeldata.SetIntercept( is[0] );
  pixeldata.SetSlope( is[1] );

  return true;
}

bool ImageReader::ReadACRNEMAImage()
{
  if( !PixmapReader::ReadACRNEMAImage() )
    {
    return false;
    }
  const DataSet &ds = F->GetDataSet();
  Image& pixeldata = GetImage();

  // 4 1/2 Let's do Pixel Spacing
  const Tag tpixelspacing(0x0028, 0x0030);
  if( ds.FindDataElement( tpixelspacing ) )
    {
    const DataElement& de = ds.GetDataElement( tpixelspacing );
    Attribute<0x0028,0x0030> at;
    at.SetFromDataElement( de );
    pixeldata.SetSpacing( 0, at.GetValue(0));
    pixeldata.SetSpacing( 1, at.GetValue(1));
    }
  // 4 2/3 Let's do Origin
  const Tag timageposition(0x0020, 0x0030);
  if( ds.FindDataElement( timageposition) )
    {
    const DataElement& de = ds.GetDataElement( timageposition);
    Attribute<0x0020,0x0030> at = {{}};
    at.SetFromDataElement( de );
    pixeldata.SetOrigin( at.GetValues() );
    if( at.GetNumberOfValues() > pixeldata.GetNumberOfDimensions() ) // FIXME HACK
      {
      pixeldata.SetOrigin(pixeldata.GetNumberOfDimensions(), at.GetValue(pixeldata.GetNumberOfDimensions()) );
      }
    }
  const Tag timageorientation(0x0020, 0x0035);
  if( ds.FindDataElement( timageorientation) )
    {
    const DataElement& de = ds.GetDataElement( timageorientation);
    Attribute<0x0020,0x0035> at = {{1,0,0,0,1,0}};//to get rid of brackets warnings in linux, lots of {}
    at.SetFromDataElement( de );
    pixeldata.SetDirectionCosines( at.GetValues() );
    }

  // Do the Rescale Intercept & Slope
  std::vector<double> is = ImageHelper::GetRescaleInterceptSlopeValue(*F);
  pixeldata.SetIntercept( is[0] );
  pixeldata.SetSlope( is[1] );

  return true;
}


} // end namespace gdcm
