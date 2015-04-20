/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageConverter.h"
#include "gdcmImage.h"

namespace gdcm
{

ImageConverter::ImageConverter()
{
  Output = new Image;
}

ImageConverter::~ImageConverter()
{
  delete Output;
}

void ImageConverter::SetInput(Image const &input)
{
  Input = const_cast<Image*>(&input);
}

const Image& ImageConverter::GetOuput() const
{
  return *Output;
}

void ImageConverter::Convert()
{
}

} // end namespace gdcm
