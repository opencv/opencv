/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMIMAGECHANGEPLANARCONFIGURATION_H
#define GDCMIMAGECHANGEPLANARCONFIGURATION_H

#include "gdcmImageToImageFilter.h"

namespace gdcm
{

class DataElement;
/**
 * \brief ImageChangePlanarConfiguration class
 * Class to change the Planar configuration of an input DICOM
 * By default it will change into the more usual reprensentation: PlanarConfiguration = 0
 */
class GDCM_EXPORT ImageChangePlanarConfiguration : public ImageToImageFilter
{
public:
  ImageChangePlanarConfiguration():PlanarConfiguration(0) {}
  ~ImageChangePlanarConfiguration() {}

  /// Set/Get requested PlanarConfigation
  void SetPlanarConfiguration(unsigned int pc) { PlanarConfiguration = pc; }
  unsigned int GetPlanarConfiguration() const { return PlanarConfiguration; }

  /// s is the size of one plane (r,g or b). Thus the output buffer needs to be at least 3*s bytes long
  /// s can be seen as the number of RGB pixels in the output
  template <typename T>
  static size_t RGBPlanesToRGBPixels(T *out, const T *r, const T *g, const T *b, size_t s);

  /// Convert a regular RGB pixel image (R,G,B,R,G,B...) into a planar R,G,B image (R,R..,G,G...B,B)
  /// \warning this works on a frame basis, you need to loop over all frames in multiple frames
  /// image to apply this function
  template <typename T>
  static size_t RGBPixelsToRGBPlanes(T *r, T *g, T *b, const T* rgb, size_t s);

  /// Change
  bool Change();

protected:

private:
  unsigned int PlanarConfiguration;
};

template <typename T>
size_t ImageChangePlanarConfiguration::RGBPlanesToRGBPixels(T *out, const T *r, const T *g, const T *b, size_t s)
{
  T *pout = out;
  for(size_t i = 0; i < s; ++i )
    {
    *pout++ = *r++;
    *pout++ = *g++;
    *pout++ = *b++;
    }

  assert( (size_t)(pout - out) == 3 * s * sizeof(T) );
  return pout - out;
}

template <typename T>
size_t ImageChangePlanarConfiguration::RGBPixelsToRGBPlanes(T *r, T *g, T *b, const T *rgb, size_t s)
{
  const T *prgb = rgb;
  for(size_t i = 0; i < s; ++i )
    {
    *r++ = *prgb++;
    *g++ = *prgb++;
    *b++ = *prgb++;
    }
  assert( (size_t)(prgb - rgb) == 3 * s * sizeof(T) );
  return prgb - rgb;
}


} // end namespace gdcm

#endif //GDCMIMAGECHANGEPLANARCONFIGURATION_H
