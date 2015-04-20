/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMIMAGEREADER_H
#define GDCMIMAGEREADER_H

#include "gdcmPixmapReader.h"
#include "gdcmImage.h"

namespace gdcm
{

class MediaStorage;
/**
 * \brief ImageReader
 * \note its role is to convert the DICOM DataSet into a Image
 * representation
 * Image is different from Pixmap has it has a position and a direction in
 * Space.
 *
 * \see Image
 */
class GDCM_EXPORT ImageReader : public PixmapReader
{
public:
  ImageReader();
  virtual ~ImageReader();//needs to be virtual to ensure lack of memory leaks

  /// Read the DICOM image. There are two reason for failure:
  /// 1. The input filename is not DICOM
  /// 2. The input DICOM file does not contains an Image.

  virtual bool Read();

  // Following methods are valid only after a call to 'Read'

  /// Return the read image
  const Image& GetImage() const;
  Image& GetImage();
  //void SetImage(Image const &img);

protected:
  bool ReadImage(MediaStorage const &ms);
  bool ReadACRNEMAImage();
};

} // end namespace gdcm

#endif //GDCMIMAGEREADER_H
