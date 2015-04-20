/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMORIENTATION_H
#define GDCMORIENTATION_H

#include "gdcmTypes.h"

namespace gdcm
{

/**
 * \brief class to handle Orientation
 */
class GDCM_EXPORT Orientation
{
  friend std::ostream& operator<<(std::ostream &_os, const Orientation &o);
public:
  Orientation();
  ~Orientation();

  /// Print
  void Print(std::ostream &) const;

  typedef enum {
    UNKNOWN,
    AXIAL,
    CORONAL,
    SAGITTAL,
    OBLIQUE
  } OrientationType;

  /// Return the type of orientation from a direction cosines
  /// Input is an array of 6 double
  static OrientationType GetType(const double dircos[6]);

  /// ObliquityThresholdCosineValue stuff
  static void SetObliquityThresholdCosineValue(double val);
  static double GetObliquityThresholdCosineValue();

  /// Return the label of an Orientation
  static const char *GetLabel(OrientationType type);

protected:
  static char GetMajorAxisFromPatientRelativeDirectionCosine(double x, double y, double z);

private:
  static double ObliquityThresholdCosineValue;
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const Orientation &o)
{
  o.Print( os );
  return os;
}

} // end namespace gdcm

#endif //GDCMORIENTATION_H
