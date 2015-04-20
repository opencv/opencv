/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmBitmapToBitmapFilter.h"

#include "gdcmImage.h"

#include <limits>
#include <stdlib.h> // abort
#include <string.h> // memcpy

namespace gdcm
{

BitmapToBitmapFilter::BitmapToBitmapFilter()
{
}

void BitmapToBitmapFilter::SetInput(const Bitmap& image)
{
  Input = image;
  const Bitmap *p = &image;
  if( dynamic_cast<const Image*>(p) )
    {
    Output = new Image;
    }
  else if( dynamic_cast<const Pixmap*>(p) )
    {
    Output = new Pixmap;
    }
  else if( dynamic_cast<const Bitmap*>(p) )
    {
    Output = new Bitmap;
    }
  else
    {
    Output = NULL;
    assert( 0 );
    }
}

const Bitmap &BitmapToBitmapFilter::GetOutputAsBitmap() const
{
  return *Output;
}


} // end namespace gdcm
