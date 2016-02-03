/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPIXMAPTOPIXMAPFILTER_H
#define GDCMPIXMAPTOPIXMAPFILTER_H

#include "gdcmBitmapToBitmapFilter.h"

namespace gdcm
{

class Pixmap;
/**
 * \brief PixmapToPixmapFilter class
 * Super class for all filter taking an image and producing an output image
 */
class GDCM_EXPORT PixmapToPixmapFilter : public BitmapToBitmapFilter
{
public:
  PixmapToPixmapFilter();
  ~PixmapToPixmapFilter() {}

  Pixmap &GetInput();

  /// Get Output image
  const Pixmap &GetOutput() const;

  // SWIG/Java hack:
  const Pixmap &GetOutputAsPixmap() const;
};

} // end namespace gdcm

#endif //GDCMPIXMAPTOPIXMAPFILTER_H
