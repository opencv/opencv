/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMIMAGECHANGEPHOTOMETRICINTERPRETATION_H
#define GDCMIMAGECHANGEPHOTOMETRICINTERPRETATION_H

#include "gdcmImageToImageFilter.h"
#include "gdcmPhotometricInterpretation.h"

namespace gdcm
{

class DataElement;
/**
 * \brief ImageChangePhotometricInterpretation class
 * Class to change the Photometric Interpetation of an input DICOM
 */
class GDCM_EXPORT ImageChangePhotometricInterpretation : public ImageToImageFilter
{
public:
  ImageChangePhotometricInterpretation():PI() {}
  ~ImageChangePhotometricInterpretation() {}

  /// Set/Get requested PhotometricInterpretation
  void SetPhotometricInterpretation(PhotometricInterpretation const &pi) { PI = pi; }
  const PhotometricInterpretation &GetPhotometricInterpretation() const { return PI; }

  /// Change
  bool Change();

  /// colorspace converstion (based on CCIR Recommendation 601-2)
  template <typename T>
  static void RGB2YBR(T ybr[3], const T rgb[3]);
  template <typename T>
  static void YBR2RGB(T rgb[3], const T ybr[3]);

protected:
  bool ChangeMonochrome();

private:
  PhotometricInterpretation PI;
};


// http://en.wikipedia.org/wiki/YCbCr
template <typename T>
void ImageChangePhotometricInterpretation::RGB2YBR(T ybr[3], const T rgb[3])
{
#if 1
  ybr[0] =   65.738 * rgb[0] +    129.057 * rgb[1] +    25.064 * rgb[2] + 16;
  ybr[1] =  -37.945 * rgb[0] +    -74.494 * rgb[1] +   112.439 * rgb[2] + 128;
  ybr[2] =  112.439 * rgb[0] +    -94.154 * rgb[1] +   -18.285 * rgb[2] + 128;
#else

  const double R = rgb[0];
  const double G = rgb[1];
  const double B = rgb[2];
  const double Y  =  .2990 * R + .5870 * G + .1140 * B;
  const double CB = -.168736 * R - .331264 * G + .5000 * B + 128;
  const double CR =  .5000 * R - .418688 * G - .081312 * B + 128;
  //assert( Y >= 0  && Y <= 255 );
  //assert( CB >= 0 && CB <= 255 );
  //assert( CR >= 0 && CR <= 255 );
  ybr[0] = Y  /*+ 0.5*/;
  ybr[1] = CB /*+ 0.5*/;
  ybr[2] = CR /*+ 0.5*/;
#endif
}

template <typename T>
void ImageChangePhotometricInterpretation::YBR2RGB(T rgb[3], const T ybr[3])
{

#if 1
 rgb[0] = 298.082 * ((int)ybr[0]-16) +     0.    * ((int)ybr[1]-128) +   408.583 * ((int)ybr[2]-128) - 1. / 256;
 rgb[1] = 298.082 * ((int)ybr[0]-16) +  -100.291 * ((int)ybr[1]-128) +  -208.12  * ((int)ybr[2]-128) - 1. / 256;
 rgb[2] = 298.082 * ((int)ybr[0]-16) +   516.411 * ((int)ybr[1]-128) +     0.    * ((int)ybr[2]-128) - 1. / 256;

#else
  const double Y  = ybr[0];
  const double Cb = ybr[1];
  const double Cr = ybr[2];
  //const double R =  1.0000e+00 * Y - 3.6820e-05 * CB + 1.4020e+00 * CR;
  //const double G =  1.0000e+00 * Y - 3.4411e-01 * CB - 7.1410e-01 * CR;
  //const double B =  1.0000e+00 * Y + 1.7720e+00 * CB - 1.3458e-04 * CR;
  const double r = Y                    + 1.402   * (Cr-128);
  const double g = Y - 0.344136 * (Cb-128) - 0.714136 * (Cr-128);
  const double b = Y + 1.772   * (Cb-128);
  double R = r < 0 ? 0 : r;
  R = R > 255 ? 255 : R;
  double G = g < 0 ? 0 : g;
  G = G > 255 ? 255 : G;
  double B = b < 0 ? 0 : b;
  B = B > 255 ? 255 : B;
  assert( R >= 0 && R <= 255 );
  assert( G >= 0 && G <= 255 );
  assert( B >= 0 && B <= 255 );
  rgb[0] = ((R < 0 ? 0 : R) > 255 ? 255 : R);
  rgb[1] = G;
  rgb[2] = B;
#endif

}

} // end namespace gdcm

#endif //GDCMIMAGECHANGEPHOTOMETRICINTERPRETATION_H
