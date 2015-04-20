/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMICONIMAGEGENERATOR_H
#define GDCMICONIMAGEGENERATOR_H

#include "gdcmPixmap.h"
#include "gdcmIconImage.h"

namespace gdcm
{
class IconImageGeneratorInternals;
/**
 * \brief IconImageGenerator
 * This filter will generate a valid Icon from the Pixel Data element (an
 * instance of Pixmap).
 * To generate a valid Icon, one is only allowed the following Photometric
 * Interpretation:
 * - MONOCHROME1
 * - MONOCHROME2
 * - PALETTE_COLOR
 *
 * The Pixel Bits Allocated is restricted to 8bits, therefore 16 bits image
 * needs to be rescaled. By default the filter will use the full scalar range
 * of 16bits image to rescale to unsigned 8bits.
 * This may not be ideal for some situation, in which case the API
 * SetPixelMinMax can be used to overwrite the default min,max interval used.
 *
 * \see ImageReader
 */
class GDCM_EXPORT IconImageGenerator
{
public:
  IconImageGenerator();
  ~IconImageGenerator();

  /// Set/Get File
  void SetPixmap(const Pixmap& p) { P = p; }
  Pixmap &GetPixmap() { return *P; }
  const Pixmap &GetPixmap() const { return *P; }

  /// Set Target dimension of output Icon
  void SetOutputDimensions(const unsigned int dims[2]);

  /// Override default min/max to compute best rescale for 16bits -> 8bits
  /// downscale. Typically those value can be read from the SmallestImagePixelValue
  /// LargestImagePixelValue DICOM attribute.
  void SetPixelMinMax(double min, double max);

  /// Instead of explicitely specifying the min/max value for the rescale
  /// operation, let the internal mechanism compute the min/max of icon and
  /// rescale to best appropriate.
  void AutoPixelMinMax(bool b);

  /// Converting from RGB to PALETTE_COLOR can be a slow operation. However DICOM
  /// standard requires that color icon be described as palette. Set this boolean
  /// to false only if you understand the consequences.
  /// default value is true, false generates invalid Icon Image Sequence
  void ConvertRGBToPaletteColor(bool b);

  /// Set a pixel value that should be discarded. This happen typically for CT image, where
  /// a pixel has been used to pad outside the image (see Pixel Padding Value).
  /// Requires AutoPixelMinMax(true)
  void SetOutsideValuePixel(double v);

  /// Generate Icon
  bool Generate();

  /// Retrieve generated Icon
  const IconImage& GetIconImage() const { return *I; }

protected:

private:
  void BuildLUT( Bitmap & bitmap, unsigned int maxcolor );

  SmartPointer<Pixmap> P;
  SmartPointer<IconImage> I;
  IconImageGeneratorInternals *Internals;
};

} // end namespace gdcm

#endif //GDCMICONIMAGEGENERATOR_H
