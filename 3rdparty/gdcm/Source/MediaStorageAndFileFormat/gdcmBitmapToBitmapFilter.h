/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMBITMAPTOBITMAPFILTER_H
#define GDCMBITMAPTOBITMAPFILTER_H

#include "gdcmBitmap.h"

namespace gdcm
{

/**
 * \brief BitmapToBitmapFilter class
 * Super class for all filter taking an image and producing an output image
 */
class GDCM_EXPORT BitmapToBitmapFilter
{
public:
  BitmapToBitmapFilter();
  ~BitmapToBitmapFilter() {}

  /// Set input image
  void SetInput(const Bitmap& image);

  /// Get Output image
  const Bitmap &GetOutput() const { return *Output; }

  // SWIG/Java hack:
  const Bitmap &GetOutputAsBitmap() const;

protected:
  SmartPointer<Bitmap> Input;
  SmartPointer<Bitmap> Output;
};

} // end namespace gdcm

#endif //GDCMBITMAPTOBITMAPFILTER_H
