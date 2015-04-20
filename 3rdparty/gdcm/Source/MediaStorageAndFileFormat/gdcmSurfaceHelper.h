/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSURFACEHELPER_H
#define GDCMSURFACEHELPER_H

#include "gdcmTypes.h"  // for GDCM_EXPORT

#include <vector>
#include <iostream>

namespace gdcm
{

/**
 * \brief SurfaceHelper
 * Helper class for Surface object
 */
class GDCM_EXPORT SurfaceHelper
{
public:

  typedef std::vector< unsigned short > ColorArray;

  /**
    * \brief  Convert a RGB color into DICOM grayscale (ready to write).
    *
    * \see    PS 3.3 C.27.1 tag(0062,000C)
    *
    * \param  RGB RGB array.
    * \param  rangeMax  Max value of the RGB range.
    *
    * \tparam T Type of RGB components.
    * \tparam U Type of rangeMax value.
    */
  template <typename T, typename U>
  static unsigned short RGBToRecommendedDisplayGrayscale(const std::vector<T> & RGB,
                                                         const U rangeMax = 255);
  /**
    * \brief  Convert a RGB color into DICOM CIE-Lab (ready to write).
    *
    * \see    PS 3.3 C.10.7.1.1
    *
    * \param  RGB RGB array.
    * \param  rangeMax  Max value of the RGB range.
    *
    * \tparam T Type of RGB components.
    * \tparam U Type of rangeMax value.
    */
  template <typename T, typename U>
  static ColorArray RGBToRecommendedDisplayCIELab(const std::vector<T> & RGB,
                                                  const U rangeMax = 255);
  /**
    * \brief  Convert a DICOM CIE-Lab (after reading) color into RGB.
    *
    * \see    PS 3.3 C.10.7.1.1
    *
    * \param  CIELab DICOM CIE-Lab array.
    * \param  rangeMax  Max value of the RGB range.
    *
    * \tparam T Type of CIELab components.
    * \tparam U Type of rangeMax value.
    */
  template <typename T, typename U>
  static std::vector<T> RecommendedDisplayCIELabToRGB(const ColorArray & CIELab,
                                                      const U rangeMax = 255);
  /**
    * \brief  Convert a DICOM CIE-Lab (after reading) color into RGB.
    *
    * \see    PS 3.3 C.10.7.1.1
    *
    * \param  CIELab DICOM CIE-Lab array.
    * \param  rangeMax  Max value of the RGB range.
    *
    * \tparam U Type of rangeMax value.
    */
  template <typename U>
  static std::vector<float> RecommendedDisplayCIELabToRGB(const ColorArray & CIELab,
                                                      const U rangeMax = 255);

private:

  static std::vector< float > RGBToXYZ(const std::vector<float> & RGB);

  static std::vector< float > XYZToRGB(const std::vector<float> & XYZ);

  static std::vector< float > XYZToCIELab(const std::vector<float> & XYZ);

  static std::vector< float > CIELabToXYZ(const std::vector<float> & CIELab);
};

template <typename T, typename U>
unsigned short SurfaceHelper::RGBToRecommendedDisplayGrayscale(const std::vector<T> & RGB,
                                                               const U rangeMax/* = 255*/)
{
  assert(RGB.size() > 2);

  unsigned short Grayscale = 0;

  const float inverseRangeMax = 1. / (float) rangeMax;

  // 0xFFFF "=" 255 "=" white
  Grayscale = (unsigned short) ((0.2989 * RGB[0] + 0.5870 * RGB[1] + 0.1140 * RGB[2])
                                * inverseRangeMax // Convert to range 0-1
                                * 0xFFFF);        // Convert to range 0x0000-0xFFFF

  return Grayscale;
}

template <typename T, typename U>
SurfaceHelper::ColorArray SurfaceHelper::RGBToRecommendedDisplayCIELab(const std::vector<T> & RGB,
                                                                       const U rangeMax/* = 255*/)
{
  assert(RGB.size() > 2);

  ColorArray CIELab(3);
  std::vector<float> tmp(3);

  // Convert to range 0-1
  const float inverseRangeMax = 1. / (float) rangeMax;
  tmp[0] = (float) (RGB[0] * inverseRangeMax);
  tmp[1] = (float) (RGB[1] * inverseRangeMax);
  tmp[2] = (float) (RGB[2] * inverseRangeMax);

  tmp = SurfaceHelper::XYZToCIELab( SurfaceHelper::RGBToXYZ( tmp ) );

  // Convert to range 0x0000-0xFFFF
  // 0xFFFF "=" 127, 0x8080 "=" 0, 0x0000 "=" -128
    CIELab[0] = (unsigned short) (          0xFFFF           * (tmp[0]*0.01));
    if(tmp[1] >= -128 && tmp[1] <= 0)
    {
        CIELab[1] = (unsigned short)(((float)(0x8080)/128.0)*tmp[1] + ((float)0x8080));
    }
    else if(tmp[1] <= 127 && tmp[1] > 0)
    {
        CIELab[1] = (unsigned short)(((float)(0xFFFF - 0x8080)/127.0)*tmp[1] + (float)(0x8080));
    }
    if(tmp[2] >= -128 && tmp[2] <= 0)
    {
        CIELab[2] = (unsigned short)(((float)0x8080/128.0)*tmp[2] + ((float)0x8080));
    }
    else if(tmp[2] <= 127 && tmp[2] > 0)
    {
        CIELab[2] = (unsigned short)(((float)(0xFFFF - 0x8080)/127.0)*tmp[2] + (float)(0x8080));
    }

  return CIELab;
}

template <typename T, typename U>
std::vector<T> SurfaceHelper::RecommendedDisplayCIELabToRGB(const ColorArray & CIELab,
                                                            const U rangeMax/* = 255*/)
{
  assert(CIELab.size() > 2);

  std::vector<T> RGB(3);
  std::vector<float> tmp(3);

  // Convert to range 0-1

    tmp[0] = 100.0*CIELab[0] /(float)(0xFFFF);
    if(CIELab[1] >= 0x0000 && CIELab[1] <= 0x8080)
    {
        tmp[1] = (float)(((CIELab[1] - 0x8080) * 128.0)/(float)0x8080);
    }
    else if(CIELab[1] <= 0xFFFF && CIELab[1] > 0x8080)
    {
        tmp[1] = (float)((CIELab[1]-0x8080)*127.0 / (float)(0xFFFF - 0x8080));
    }
    if(CIELab[2] >= 0x0000 && CIELab[2] <= 0x8080)
    {
        tmp[2] = (float)(((CIELab[2] - 0x8080) * 128.0)/(float)0x8080);
    }
    else if(CIELab[2] <= 0xFFFF && CIELab[2] > 0x8080)
    {
        tmp[2] = (float)((CIELab[2]-0x8080)*127.0 / (float)(0XFFFF - 0x8080));
    }

  tmp = SurfaceHelper::XYZToRGB( SurfaceHelper::CIELabToXYZ( tmp ) );

  // Convert to range 0-rangeMax
  RGB[0] = (T) (tmp[0] * rangeMax);
  RGB[1] = (T) (tmp[1] * rangeMax);
  RGB[2] = (T) (tmp[2] * rangeMax);

  return RGB;
}

template <typename U>
std::vector<float> SurfaceHelper::RecommendedDisplayCIELabToRGB(const ColorArray & CIELab,
                                                            const U rangeMax/* = 255*/)
{
  return RecommendedDisplayCIELabToRGB<float>(CIELab, rangeMax);
}

} // end namespace gdcm

#endif // GDCMSURFACEHELPER_H
