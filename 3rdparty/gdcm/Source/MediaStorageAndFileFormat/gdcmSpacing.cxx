/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSpacing.h"

namespace gdcm
{

Spacing::Spacing() {}

Spacing::~Spacing() {}

/*
 * http://www.ics.uci.edu/~eppstein/numth/frap.c
 * http://stackoverflow.com/questions/95727/how-to-convert-floats-to-human-readable-fractions
 * http://www.cs.umd.edu/class/sum2003/cmsc311/Notes/Data/fracToBaseK.html
 * http://docs.sun.com/source/806-3568/ncg_goldberg.html
 */
/*
** find rational approximation to given real number
** David Eppstein / UC Irvine / 8 Aug 1993
**
** With corrections from Arno Formella, May 2008
**
** usage: a.out r d
**   r is real number to approx
**   d is the maximum denominator allowed
**
** based on the theory of continued fractions
** if x = a1 + 1/(a2 + 1/(a3 + 1/(a4 + ...)))
** then best approximation is found by truncating this series
** (with some adjustments in the last term).
**
** Note the fraction can be recovered as the first column of the matrix
**  ( a1 1 ) ( a2 1 ) ( a3 1 ) ...
**  ( 1  0 ) ( 1  0 ) ( 1  0 )
** Instead of keeping the sequence of continued fraction terms,
** we just keep the last partial product of these matrices.
*/

// return error
double frap(double frac[2], double startx, double maxden = 10 )
{
  long m[2][2];
  double x;
  long ai;

  x = startx;

  /* initialize matrix */
  m[0][0] = m[1][1] = 1;
  m[0][1] = m[1][0] = 0;

  /* loop finding terms until denom gets too big */
  while (m[1][0] *  ( ai = (long)x ) + m[1][1] <= maxden)
    {
    long t;
    t = m[0][0] * ai + m[0][1];
    m[0][1] = m[0][0];
    m[0][0] = t;
    t = m[1][0] * ai + m[1][1];
    m[1][1] = m[1][0];
    m[1][0] = t;
    if(x==(double)ai) break;     // AF: division by zero
    x = 1/(x - (double) ai);
    if(x>(double)0x7FFFFFFF) break;  // AF: representation failure
    }

  /* now remaining x is between 0 and 1/ai */
  /* approx as either 0 or 1/m where m is max that will fit in maxden */
  /* first try zero */
  //printf("%ld/%ld, error = %e\n", m[0][0], m[1][0],
  //  startx - ((double) m[0][0] / (double) m[1][0]));
  frac[0] = (double)m[0][0];
  frac[1] = (double)m[1][0];
  const double error = startx - ((double) m[0][0] / (double) m[1][0]);

  /* now try other possibility */
  ai = ((long)maxden - m[1][1]) / m[1][0];
  m[0][0] = m[0][0] * ai + m[0][1];
  m[1][0] = m[1][0] * ai + m[1][1];
  //printf("%ld/%ld, error = %e\n", m[0][0], m[1][0],
  //  startx - ((double) m[0][0] / (double) m[1][0]));
  const double error2 = startx - ((double) m[0][0] / (double) m[1][0]);
  assert( fabs(error) < fabs(error2) ); (void)error2;

  return error;
}

Attribute<0x28,0x34> Spacing::ComputePixelAspectRatioFromPixelSpacing(const Attribute<0x28,0x30>& pixelspacing)
{
  Attribute<0x28,0x34> pixelaspectratio;
  const double ratio = pixelspacing[0] / pixelspacing[1];
//  double value = 2.5;
//  double integral_part;
//  double frac_part = modf(value, &integral_part);
  double frac[2];
  double error = frap(frac, ratio);
  (void)error;
  pixelaspectratio[0] = static_cast<int>(frac[0]);
  pixelaspectratio[1] = static_cast<int>(frac[1]);

  return pixelaspectratio;
}


} // end namespace gdcm
