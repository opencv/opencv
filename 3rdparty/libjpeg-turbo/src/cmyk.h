/*
 * cmyk.h
 *
 * Copyright (C) 2017-2018, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains convenience functions for performing quick & dirty
 * CMYK<->RGB conversion.  This algorithm is suitable for testing purposes
 * only.  Properly converting between CMYK and RGB requires a color management
 * system.
 */

#ifndef CMYK_H
#define CMYK_H

#include <jinclude.h>
#define JPEG_INTERNALS
#include <jpeglib.h>
#include "jconfigint.h"


/* Fully reversible */

INLINE
LOCAL(void)
rgb_to_cmyk(JSAMPLE r, JSAMPLE g, JSAMPLE b, JSAMPLE *c, JSAMPLE *m,
            JSAMPLE *y, JSAMPLE *k)
{
  double ctmp = 1.0 - ((double)r / 255.0);
  double mtmp = 1.0 - ((double)g / 255.0);
  double ytmp = 1.0 - ((double)b / 255.0);
  double ktmp = MIN(MIN(ctmp, mtmp), ytmp);

  if (ktmp == 1.0) ctmp = mtmp = ytmp = 0.0;
  else {
    ctmp = (ctmp - ktmp) / (1.0 - ktmp);
    mtmp = (mtmp - ktmp) / (1.0 - ktmp);
    ytmp = (ytmp - ktmp) / (1.0 - ktmp);
  }
  *c = (JSAMPLE)(255.0 - ctmp * 255.0 + 0.5);
  *m = (JSAMPLE)(255.0 - mtmp * 255.0 + 0.5);
  *y = (JSAMPLE)(255.0 - ytmp * 255.0 + 0.5);
  *k = (JSAMPLE)(255.0 - ktmp * 255.0 + 0.5);
}


/* Fully reversible only for C/M/Y/K values generated with rgb_to_cmyk() */

INLINE
LOCAL(void)
cmyk_to_rgb(JSAMPLE c, JSAMPLE m, JSAMPLE y, JSAMPLE k, JSAMPLE *r, JSAMPLE *g,
            JSAMPLE *b)
{
  *r = (JSAMPLE)((double)c * (double)k / 255.0 + 0.5);
  *g = (JSAMPLE)((double)m * (double)k / 255.0 + 0.5);
  *b = (JSAMPLE)((double)y * (double)k / 255.0 + 0.5);
}


#endif /* CMYK_H */
