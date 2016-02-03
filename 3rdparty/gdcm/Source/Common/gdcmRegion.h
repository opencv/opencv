/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMREGION_H
#define GDCMREGION_H

#include "gdcmTypes.h"
#include <vector>
#include <iostream>

namespace gdcm
{
class BoxRegion;
/**
 * \brief Class for manipulation region
 */
//-----------------------------------------------------------------------------
class GDCM_EXPORT Region
{
public :
  Region();
  virtual ~Region();

  /// Print
  virtual void Print(std::ostream &os = std::cout) const;

  /// return whether this domain is empty:
  virtual bool Empty() const = 0;

  /// return whether this is valid domain
  virtual bool IsValid() const = 0;

  /// compute the area
  virtual size_t Area() const = 0;

  // implementation detail of heterogenous container in C++
  virtual Region *Clone() const = 0;

  /// Return the Axis-Aligned minimum bounding box for all regions
  virtual BoxRegion ComputeBoundingBox() = 0;
private:
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const Region&r)
{
  r.Print( os );
  return os;
}

} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMREGION_H
