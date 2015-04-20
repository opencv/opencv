/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmBoxRegion.h"

#include <limits>
#include <algorithm> // req C++11

namespace gdcm
{
class BoxRegionInternals
{
public:
  BoxRegionInternals()
    {
    XMin = YMin = ZMin = std::numeric_limits<unsigned int>::max();
    XMax = YMax = ZMax = std::numeric_limits<unsigned int>::min();
    }
  unsigned int XMin, XMax;
  unsigned int YMin, YMax;
  unsigned int ZMin, ZMax;
  void Print(std::ostream &os) const
    {
    os << "Min:" << XMin << "," << YMin << "," << ZMin << "\n";
    os << "Max:" << XMax << "," << YMax << "," << ZMax;
    }

};

BoxRegion::BoxRegion()
{
  Internals = new BoxRegionInternals;
}

BoxRegion::~BoxRegion()
{
  delete Internals;
}

void BoxRegion::SetDomain(unsigned int xmin, unsigned int xmax,
    unsigned int ymin, unsigned int ymax,
    unsigned int zmin, unsigned int zmax)
{
  Internals->XMin = xmin;
  Internals->YMin = ymin;
  Internals->ZMin = zmin;
  Internals->XMax = xmax;
  Internals->YMax = ymax;
  Internals->ZMax = zmax;
}

Region *BoxRegion::Clone() const
{
  BoxRegion *br = new BoxRegion( *this );
  return br;
}

bool BoxRegion::Empty() const
{
  assert( 0 );
  return false;
}

bool BoxRegion::IsValid() const
{
  if (Internals->XMax < Internals->XMin ||
    Internals->YMax < Internals->YMin ||
    Internals->ZMax < Internals->ZMin )
    {
    return false;
    }
  return true;
}

size_t BoxRegion::Area() const
{
  return (Internals->YMax - Internals->YMin + 1)*
         (Internals->XMax - Internals->XMin + 1)*
         (Internals->ZMax - Internals->ZMin + 1);
}

unsigned int BoxRegion::GetXMin() const
{
  return Internals->XMin;
}
unsigned int BoxRegion::GetXMax() const
{
  return Internals->XMax;
}
unsigned int BoxRegion::GetYMin() const
{
  return Internals->YMin;
}
unsigned int BoxRegion::GetYMax() const
{
  return Internals->YMax;
}
unsigned int BoxRegion::GetZMin() const
{
  return Internals->ZMin;
}
unsigned int BoxRegion::GetZMax() const
{
  return Internals->ZMax;
}

BoxRegion BoxRegion::BoundingBox(BoxRegion const & b1, BoxRegion const & b2 )
{
  BoxRegion r;
  unsigned int xmin = std::min( b1.GetXMin(), b2.GetXMin() );
  unsigned int xmax = std::min( b1.GetXMax(), b2.GetXMax() );
  unsigned int ymin = std::min( b1.GetYMin(), b2.GetYMin() );
  unsigned int ymax = std::min( b1.GetYMax(), b2.GetYMax() );
  unsigned int zmin = std::min( b1.GetZMin(), b2.GetZMin() );
  unsigned int zmax = std::min( b1.GetZMax(), b2.GetZMax() );

  r.SetDomain(xmin, xmax, ymin, ymax, zmin, zmax);
  return r;
}

BoxRegion::BoxRegion(const BoxRegion& b)
{
  assert( b.Internals );
  Internals = new BoxRegionInternals;
  *Internals = *b.Internals;
}

void BoxRegion::operator=(const BoxRegion& b)
{
  assert( b.Internals );
  *Internals = *b.Internals;
}

BoxRegion BoxRegion::ComputeBoundingBox()
{
  return *this;
}

void BoxRegion::Print(std::ostream &os) const
{
  Region::Print( os );
  os << "Domain:\n";
  this->Internals->Print( os );
}

} // end namespace gdcm
