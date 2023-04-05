/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *
 LibRaw uses code from dcraw.c -- Dave Coffin's raw photo decoder,
 dcraw.c is copyright 1997-2018 by Dave Coffin, dcoffin a cybercom o net.
 LibRaw do not use RESTRICTED code from dcraw.c

 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/dcraw_defs.h"

void LibRaw::fuji_rotate()
{
  int i, row, col;
  double step;
  float r, c, fr, fc;
  unsigned ur, uc;
  ushort wide, high, (*img)[4], (*pix)[4];

  if (!fuji_width)
    return;
  fuji_width = (fuji_width - 1 + shrink) >> shrink;
  step = sqrt(0.5);
  wide = fuji_width / step;
  high = (height - fuji_width) / step;

  // All real fuji/rotated images are small, so check against max_raw_memory_mb here is safe
  if (INT64(wide) * INT64(high) * INT64(sizeof(*img))    >
      INT64(imgdata.rawparams.max_raw_memory_mb) * INT64(1024 * 1024))
    throw LIBRAW_EXCEPTION_TOOBIG;

  img = (ushort(*)[4])calloc(high, wide * sizeof *img);

  RUN_CALLBACK(LIBRAW_PROGRESS_FUJI_ROTATE, 0, 2);

  for (row = 0; row < high; row++)
    for (col = 0; col < wide; col++)
    {
      ur = r = fuji_width + (row - col) * step;
      uc = c = (row + col) * step;
      if (ur > (unsigned)height - 2 || uc > (unsigned)width - 2)
        continue;
      fr = r - ur;
      fc = c - uc;
      pix = image + ur * width + uc;
      for (i = 0; i < colors; i++)
        img[row * wide + col][i] =
            (pix[0][i] * (1 - fc) + pix[1][i] * fc) * (1 - fr) +
            (pix[width][i] * (1 - fc) + pix[width + 1][i] * fc) * fr;
    }

  free(image);
  width = wide;
  height = high;
  image = img;
  fuji_width = 0;
  RUN_CALLBACK(LIBRAW_PROGRESS_FUJI_ROTATE, 1, 2);
}

void LibRaw::stretch()
{
  ushort newdim, (*img)[4], *pix0, *pix1;
  int row, col, c;
  double rc, frac;

  if (pixel_aspect == 1)
    return;
  RUN_CALLBACK(LIBRAW_PROGRESS_STRETCH, 0, 2);
  if (pixel_aspect < 1)
  {
    newdim = height / pixel_aspect + 0.5;
    img = (ushort(*)[4])calloc(width, newdim * sizeof *img);
    for (rc = row = 0; row < newdim; row++, rc += pixel_aspect)
    {
      frac = rc - (c = rc);
      pix0 = pix1 = image[c * width];
      if (c + 1 < height)
        pix1 += width * 4;
      for (col = 0; col < width; col++, pix0 += 4, pix1 += 4)
        FORCC img[row * width + col][c] =
            pix0[c] * (1 - frac) + pix1[c] * frac + 0.5;
    }
    height = newdim;
  }
  else
  {
    newdim = width * pixel_aspect + 0.5;
    img = (ushort(*)[4])calloc(height, newdim * sizeof *img);
    for (rc = col = 0; col < newdim; col++, rc += 1 / pixel_aspect)
    {
      frac = rc - (c = rc);
      pix0 = pix1 = image[c];
      if (c + 1 < width)
        pix1 += 4;
      for (row = 0; row < height; row++, pix0 += width * 4, pix1 += width * 4)
        FORCC img[row * newdim + col][c] =
            pix0[c] * (1 - frac) + pix1[c] * frac + 0.5;
    }
    width = newdim;
  }
  free(image);
  image = img;
  RUN_CALLBACK(LIBRAW_PROGRESS_STRETCH, 1, 2);
}
