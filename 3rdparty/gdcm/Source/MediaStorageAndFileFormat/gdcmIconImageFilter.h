/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMICONIMAGEFILTER_H
#define GDCMICONIMAGEFILTER_H

#include "gdcmFile.h"
#include "gdcmIconImage.h"

namespace gdcm
{
class IconImageFilterInternals;

/**
 * \brief IconImageFilter
 * This filter will extract icons from a File
 * This filter will loop over all known sequence (public and private) that may
 * contains an IconImage and retrieve them. The filter will fails with a value
 * of false if no icon can be found
 * Since it handle both public and private icon type, one should not assume the
 * icon is in uncompress form, some private vendor store private icon in
 * JPEG8/JPEG12
 *
 * Implementation details:
 * This filter supports the following Icons:
 * - (0088,0200) Icon Image Sequence
 * - (0009,10,GEIIS) GE IIS Thumbnail Sequence
 * - (6003,10,GEMS_Ultrasound_ImageGroup_001) GEMS Image Thumbnail Sequence
 * - (0055,30,VEPRO VIF 3.0 DATA) Icon Data
 * - (0055,30,VEPRO VIM 5.0 DATA) ICONDATA2
 *
 * \warning the icon stored in those private attribute do not conform to
 * definition of Icon Image Sequence (do not simply copy/paste). For example
 * some private icon can be expressed as 12bits pixel, while the DICOM standard
 * only allow 8bits icons.
 *
 * \see ImageReader
 */
class GDCM_EXPORT IconImageFilter
{
public:
  IconImageFilter();
  ~IconImageFilter();

  /// Set/Get File
  void SetFile(const File& f) { F = f; }
  File &GetFile() { return *F; }
  const File &GetFile() const { return *F; }

  /// Extract all Icon found in File
  bool Extract();

  /// Retrieve extract IconImage (need to call Extract first)
  unsigned int GetNumberOfIconImages() const;
  IconImage& GetIconImage( unsigned int i ) const;

protected:
  void ExtractIconImages();
  void ExtractVeproIconImages();

private:
  SmartPointer<File> F;
  IconImageFilterInternals *Internals;
};

} // end namespace gdcm

#endif //GDCMICONIMAGEFILTER_H
