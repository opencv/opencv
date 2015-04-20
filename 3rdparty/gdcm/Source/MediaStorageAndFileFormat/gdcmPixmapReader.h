/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPIXMAPREADER_H
#define GDCMPIXMAPREADER_H

#include "gdcmReader.h"
#include "gdcmPixmap.h"

namespace gdcm
{

class ByteValue;
class MediaStorage;
/**
 * \brief PixmapReader
 * \note its role is to convert the DICOM DataSet into a Pixmap
 * representation
 * By default it is also loading the lookup table and overlay when found as
 * they impact the rendering or the image
 *
 * See PS 3.3-2008, Table C.7-11b IMAGE PIXEL MACRO ATTRIBUTES for the list of
 * attribute that belong to what gdcm calls a 'Pixmap'
 *
 * \warning the API ReadUpToTag and ReadSelectedTag
 *
 * \see Pixmap
 */
class GDCM_EXPORT PixmapReader : public Reader
{
public:
  PixmapReader();
  virtual ~PixmapReader(); //needs to be virtual to ensure lack of memory leaks

  /// Read the DICOM image. There are two reason for failure:
  /// 1. The input filename is not DICOM
  /// 2. The input DICOM file does not contains an Pixmap.

  virtual bool Read();

  // Following methods are valid only after a call to 'Read'

  /// Return the read image (need to call Read() first)
  const Pixmap& GetPixmap() const;
  Pixmap& GetPixmap();
  //void SetPixamp(Pixmap const &pix);

protected:
  bool ReadImageInternal(MediaStorage const &ms, bool handlepixeldata = true);
  virtual bool ReadImage(MediaStorage const &ms);
  virtual bool ReadACRNEMAImage();

  SmartPointer<Pixmap> PixelData;
};

/**
 * \example StandardizeFiles.cs
 * This is a C++ example on how to use PixmapReader
 */

} // end namespace gdcm

#endif //GDCMPIXMAPREADER_H
