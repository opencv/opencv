/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMIMAGEAPPLYLOOKUPTABLE_H
#define GDCMIMAGEAPPLYLOOKUPTABLE_H

#include "gdcmImageToImageFilter.h"

namespace gdcm
{

class DataElement;
/**
 * \brief ImageApplyLookupTable class
 * It applies the LUT the PixelData (only PALETTE_COLOR images)
 * Output will be a PhotometricInterpretation=RGB image
 */
class GDCM_EXPORT ImageApplyLookupTable : public ImageToImageFilter
{
public:
  ImageApplyLookupTable() {}
  ~ImageApplyLookupTable() {}

  /// Apply
  bool Apply();

protected:

private:
};

} // end namespace gdcm

#endif //GDCMIMAGEAPPLYLOOKUPTABLE_H
