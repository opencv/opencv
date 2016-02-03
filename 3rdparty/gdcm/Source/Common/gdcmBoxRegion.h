/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMBOXREGION_H
#define GDCMBOXREGION_H

#include "gdcmRegion.h"

namespace gdcm
{
class BoxRegionInternals;
/**
 * \brief Class for manipulation box region
 * This is a very simple implementation of the Region class.
 * It only support 3D box type region.
 * It assumes the 3D Box does not have a tilt
 * Origin is as (0,0,0)
 */
//-----------------------------------------------------------------------------
class GDCM_EXPORT BoxRegion : public Region
{
public :
  BoxRegion();
  ~BoxRegion();

  /// Set domain
  void SetDomain(unsigned int xmin, unsigned int xmax,
    unsigned int ymin, unsigned int ymax,
    unsigned int zmin, unsigned int zmax);

  /// Get domain
  unsigned int GetXMin() const;
  unsigned int GetXMax() const;
  unsigned int GetYMin() const;
  unsigned int GetYMax() const;
  unsigned int GetZMin() const;
  unsigned int GetZMax() const;

  // Satisfy pure virtual parent class
  Region *Clone() const;
  bool Empty() const;
  bool IsValid() const;
  size_t Area() const;
  BoxRegion ComputeBoundingBox();

  void Print(std::ostream &os = std::cout) const;

  /// Helper class to compute the bounding box of two BoxRegion
  static BoxRegion BoundingBox(BoxRegion const & b1, BoxRegion const & b2 );

  /// copy/cstor and al.
  BoxRegion(const BoxRegion&);
  void operator=(const BoxRegion&);
private:
  BoxRegionInternals *Internals;
};

} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMREGION_H
