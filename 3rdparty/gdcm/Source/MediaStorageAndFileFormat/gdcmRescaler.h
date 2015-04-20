/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMRESCALER_H
#define GDCMRESCALER_H

#include "gdcmTypes.h"
#include "gdcmPixelFormat.h"

namespace gdcm
{

/**
 * \brief Rescale class
 * This class is meant to apply the linear transform of Stored Pixel Value to
 * Real World Value.
 * This is mostly found in CT or PET dataset, where the value are stored using
 * one type, but need to be converted to another scale using a linear
 * transform.
 * There are basically two cases:
 * In CT: the linear transform is generally integer based. E.g. the Stored
 * Pixel Type is unsigned short 12bits, but to get Hounsfield unit, one need to
 * apply the linear transform:
 *  \f[
 *   RWV = 1. * SV - 1024
 *  \f]
 * So the best scalar to store the Real World Value will be 16 bits signed
 * type.
 *
 * In PET: the linear transform is generally floating point based.
 * Since the dynamic range can be quite high, the Rescale Slope / Rescale
 * Intercept can be changing throughout the Series. So it is important to read
 * all linear transform and deduce the best Pixel Type only at the end (when
 * all the images to be read have been parsed).
 *
 * \warning Internally any time a floating point value is found either in the
 * Rescale Slope or the Rescale Intercept it is assumed that the best matching
 * output pixel type is FLOAT64 (in previous implementation it was FLOAT32).
 * Because VR:DS is closer to a 64bits floating point type FLOAT64 is thus a
 * best matching pixel type for the floating point transformation.
 *
 * Example: Let say input is FLOAT64, and we want UINT16 as ouput, we would do:
 *
 *\code
 *  Rescaler ir;
 *  ir.SetIntercept( 0 );
 *  ir.SetSlope( 5.6789 );
 *  ir.SetPixelFormat( FLOAT64 );
 *  ir.SetMinMaxForPixelType( ((PixelFormat)UINT16).GetMin(), ((PixelFormat)UINT16).GetMax() );
 *  ir.InverseRescale(output,input,numberofbytes );
 *\endcode
 *
 * \note handle floating point transformation back and forth to integer
 * properly (no loss)
 *
 * \see Unpacker12Bits
 */
class GDCM_EXPORT Rescaler
{
public:
  Rescaler():Intercept(0),Slope(1),PF(PixelFormat::UNKNOWN),TargetScalarType(PixelFormat::UNKNOWN), ScalarRangeMin(0), ScalarRangeMax(0), UseTargetPixelType(false) {}
  ~Rescaler() {}

  /// Direct transform
  bool Rescale(char *out, const char *in, size_t n);

  /// Inverse transform
  bool InverseRescale(char *out, const char *in, size_t n);

  /// Set Intercept: used for both direct&inverse transformation
  void SetIntercept(double i) { Intercept = i; }
  double GetIntercept() const { return Intercept; }

  /// Set Slope: user for both direct&inverse transformation
  void SetSlope(double s) { Slope = s; }
  double GetSlope() const { return Slope; }

  /// By default (when UseTargetPixelType is false), a best
  /// matching Target Pixel Type is computed. However user can override
  /// this auto selection by switching UseTargetPixelType:true and
  /// also specifying the specifix Target Pixel Type
  void SetTargetPixelType( PixelFormat const & targetst );

  /// Override default behavior of Rescale
  void SetUseTargetPixelType(bool b);

  /// Set Pixel Format of input data
  void SetPixelFormat(PixelFormat const & pf) { PF = pf; }

  /// Compute the Pixel Format of the output data
  /// Used for direct transformation
  PixelFormat::ScalarType ComputeInterceptSlopePixelType();

  /// Set target interval for output data. A best match will be computed (if possible)
  /// Used for inverse transformation
  void SetMinMaxForPixelType(double min, double max)
    {
    ScalarRangeMin = min;
    ScalarRangeMax = max;
    }

  /// Compute the Pixel Format of the output data
  /// Used for inverse transformation
  PixelFormat ComputePixelTypeFromMinMax();

protected:
  template <typename TIn>
    void RescaleFunctionIntoBestFit(char *out, const TIn *in, size_t n);
  template <typename TIn>
    void InverseRescaleFunctionIntoBestFit(char *out, const TIn *in, size_t n);

private:
  double Intercept; // 0028,1052
  double Slope;     // 0028,1053
  PixelFormat PF;
  PixelFormat::ScalarType TargetScalarType;
  double ScalarRangeMin;
  double ScalarRangeMax;
  bool UseTargetPixelType;
};

} // end namespace gdcm

#endif //GDCMRESCALER_H
