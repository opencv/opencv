/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImage.h"
#include "gdcmTrace.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmByteValue.h"
#include "gdcmDataSet.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmFragment.h"

#include <iostream>

namespace gdcm
{

const double *Image::GetSpacing() const
{
  assert( NumberOfDimensions );
  return &Spacing[0];
}

double Image::GetSpacing(unsigned int idx) const
{
  assert( NumberOfDimensions );
  //if( idx < Spacing.size() )
    {
    return Spacing[idx];
    }
  //assert( 0 && "Should not happen" );
  //return 1; // FIXME ???
}

void Image::SetSpacing(const double *spacing)
{
  assert( NumberOfDimensions );
  Spacing = std::vector<double>(spacing,
    spacing+NumberOfDimensions);
}

void Image::SetSpacing(unsigned int idx, double spacing)
{
  //assert( spacing > 1.e3 );
  Spacing.resize( 3 /*idx + 1*/ );
  Spacing[idx] = spacing;
}

const double *Image::GetOrigin() const
{
  assert( NumberOfDimensions );
  if( !Origin.empty() )
    return &Origin[0];
  return 0;
}

double Image::GetOrigin(unsigned int idx) const
{
  assert( NumberOfDimensions );
  if( idx < Origin.size() )
    {
    return Origin[idx];
    }
  return 0; // FIXME ???
}

void Image::SetOrigin(const float *ori)
{
  assert( NumberOfDimensions );
  Origin.resize( NumberOfDimensions );
  for(unsigned int i = 0; i < NumberOfDimensions; ++i)
    {
    Origin[i] = ori[i];
    }
}

void Image::SetOrigin(const double *ori)
{
  assert( NumberOfDimensions );
  Origin = std::vector<double>(ori,
    ori+NumberOfDimensions);
}

void Image::SetOrigin(unsigned int idx, double ori)
{
  Origin.resize( idx + 1 );
  Origin[idx] = ori;
}

const double *Image::GetDirectionCosines() const
{
  assert( NumberOfDimensions );
  if( !DirectionCosines.empty() )
    return &DirectionCosines[0];
  return 0;
}
double Image::GetDirectionCosines(unsigned int idx) const
{
  assert( NumberOfDimensions );
  if( idx < DirectionCosines.size() )
    {
    return DirectionCosines[idx];
    }
  return 0; // FIXME !!
}

void Image::SetDirectionCosines(const float *dircos)
{
  assert( NumberOfDimensions );
  DirectionCosines.resize( 6 );
  for(int i = 0; i < 6; ++i)
    {
    DirectionCosines[i] = dircos[i];
    }
}

void Image::SetDirectionCosines(const double *dircos)
{
  assert( NumberOfDimensions );
  DirectionCosines = std::vector<double>(dircos,
    dircos+6);
}

void Image::SetDirectionCosines(unsigned int idx, double dircos)
{
  DirectionCosines.resize( idx + 1 );
  DirectionCosines[idx] = dircos;
}

void Image::Print(std::ostream &os) const
{
  Pixmap::Print(os);
  if( NumberOfDimensions )
    {
      {
      os << "Origin: (";
      if( !Origin.empty() )
        {
        std::vector<double>::const_iterator it = Origin.begin();
        os << *it;
        for(++it; it != Origin.end(); ++it)
          {
          os << "," << *it;
          }
        }
      os << ")\n";
      }
      {
      os << "Spacing: (";
      std::vector<double>::const_iterator it = Spacing.begin();
      os << *it;
      for(++it; it != Spacing.end(); ++it)
        {
        os << "," << *it;
        }
      os << ")\n";
      }
      {
      os << "DirectionCosines: (";
      if( !DirectionCosines.empty() )
        {
        std::vector<double>::const_iterator it = DirectionCosines.begin();
        os << *it;
        for(++it; it != DirectionCosines.end(); ++it)
          {
          os << "," << *it;
          }
        }
      os << ")\n";
      }
      {
      os << "Rescale Intercept/Slope: (" << Intercept << "," << Slope << ")\n";
      }
    //std::vector<double> Spacing;
    //std::vector<double> Origin;

    }
}

} // end namespace gdcm
