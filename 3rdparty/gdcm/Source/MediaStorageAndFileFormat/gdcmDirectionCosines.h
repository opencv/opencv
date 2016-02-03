/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDIRECTIONCOSINES_H
#define GDCMDIRECTIONCOSINES_H

#include "gdcmTypes.h"

namespace gdcm
{

/**
 * \brief class to handle DirectionCosines
 */
class GDCM_EXPORT DirectionCosines
{
public:
  DirectionCosines();
  DirectionCosines(const double dircos[6]);
  // Cannot get the following signature to be wrapped with swig...
  //DirectionCosines(const double *dircos = 0 );
  ~DirectionCosines();

  /// Print
  void Print(std::ostream &) const;

  /// Compute Cross product
  void Cross(double z[3]) const;

  /// Compute Dot
  double Dot() const;

  /// Normalize in-place
  void Normalize();

  /// Make the class behave like a const double *
  operator const double* () const { return Values; }

  /// Return whether or not this is a valid direction cosines
  bool IsValid() const;

  /// Initialize from string str. It requires 6 floating point separated by a
  /// backslash character.
  bool SetFromString(const char *str);

  /// Compute the Dot product of the two cross vector of both DirectionCosines object
  double CrossDot(DirectionCosines const &dc) const;

  /// Compute the distance along the normal
  double ComputeDistAlongNormal(const double ipp[3]) const;

private:
  double Values[6];
};

} // end namespace gdcm

#endif //GDCMDIRECTIONCOSINES_H
