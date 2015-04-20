/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDirectionCosines.h"

#include <limits>
#include <math.h> // fabs
#include <stdio.h> // sscanf

namespace gdcm
{

DirectionCosines::DirectionCosines()
{
    Values[0] = 1;
    Values[1] = 0;
    Values[2] = 0;
    Values[3] = 0;
    Values[4] = 1;
    Values[5] = 0;
}

DirectionCosines::DirectionCosines(const double dircos[6])
{
    Values[0] = dircos[0];
    Values[1] = dircos[1];
    Values[2] = dircos[2];
    Values[3] = dircos[3];
    Values[4] = dircos[4];
    Values[5] = dircos[5];
}

DirectionCosines::~DirectionCosines() {}

void DirectionCosines::Print(std::ostream &os) const
{
  os << Values[0] << ",";
  os << Values[1] << ",";
  os << Values[2] << ",";
  os << Values[3] << ",";
  os << Values[4] << ",";
  os << Values[5];
}

bool DirectionCosines::IsValid() const
{
  // gdcmData/gdcm-MR-SIEMENS-16-2.acr
  // => {-1, -0, 0, -0, 0.05233599990606308, 0.99862951040267944}
  const double epsilon = 1e-3; //std::numeric_limits<double>::epsilon();

  double norm_v1 = Values[0] * Values[0] + Values[1]*Values[1] + Values[2]*Values[2];
  double norm_v2 = Values[3] * Values[3] + Values[4]*Values[4] + Values[5]*Values[5];
  double dot = Dot();

  bool ret = false;
  if( fabs(norm_v1 - 1) < epsilon && fabs(norm_v2 - 1) < epsilon )
    {
    if( fabs(dot) < epsilon )
      {
      ret = true;
      }
    }
  return ret;
}

void DirectionCosines::Cross(double z[3]) const
{
  //assert( Dot() == 0 );
  const double *x = Values;
  const double *y = x+3;
  double Zx = x[1]*y[2] - x[2]*y[1];
  double Zy = x[2]*y[0] - x[0]*y[2];
  double Zz = x[0]*y[1] - x[1]*y[0];
  z[0] = Zx; z[1] = Zy; z[2] = Zz;
}

double DirectionCosines::Dot() const
{
  const double *x = Values;
  const double *y = x+3;
  return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}

// static function is within gdcm:: namespace, so should not pollute too much on UNIX
static double Norm(const double x[3])
{
  return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}

void DirectionCosines::Normalize()
{
  double *x = Values;
  double den;
  if ( (den = Norm(x)) != 0.0 )
    {
    for (int i=0; i < 3; i++)
      {
      x[i] /= den;
      }
    }
  x = Values+3;
  if ( (den = Norm(x)) != 0.0 )
    {
    for (int i=0; i < 3; i++)
      {
      x[i] /= den;
      }
    }
}

bool DirectionCosines::SetFromString(const char *str)
{
  if( !str ) return false;
  int n = sscanf( str, "%lf\\%lf\\%lf\\%lf\\%lf\\%lf", Values, Values+1, Values+2, Values+3, Values+4, Values+5 );
  if( n == 6 )
    {
    return true;
    }
  // else
  Values[0] = 1;
  Values[1] = 0;
  Values[2] = 0;
  Values[3] = 0;
  Values[4] = 1;
  Values[5] = 0;
  return false;
}

double DirectionCosines::CrossDot(DirectionCosines const &dc) const
{
  double z1[3];
  Cross(z1);
  double z2[3];
  dc.Cross(z2);

  const double *x = z1;
  const double *y = z2;
  return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}

double DirectionCosines::ComputeDistAlongNormal(const double ipp[3]) const
{
  double normal[3];
  Cross(normal);
  double dist = 0.;
  for (int i = 0; i < 3; ++i) dist += normal[i]*ipp[i];
  return dist;
}


}
