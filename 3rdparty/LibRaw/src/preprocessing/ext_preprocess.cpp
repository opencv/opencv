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

#include "../../internal/dcraw_fileio_defs.h"

/*
   Search from the current directory up to the root looking for
   a ".badpixels" file, and fix those pixels now.
 */
void LibRaw::bad_pixels(const char *cfname)
{
  FILE *fp = NULL;
  char *cp, line[128];
  int time, row, col, r, c, rad, tot, n;

  if (!filters)
    return;
  RUN_CALLBACK(LIBRAW_PROGRESS_BAD_PIXELS, 0, 2);
  if (cfname)
    fp = fopen(cfname, "r");
  if (!fp)
  {
    imgdata.process_warnings |= LIBRAW_WARN_NO_BADPIXELMAP;
    return;
  }
  while (fgets(line, 128, fp))
  {
    cp = strchr(line, '#');
    if (cp)
      *cp = 0;
    if (sscanf(line, "%d %d %d", &col, &row, &time) != 3)
      continue;
    if ((unsigned)col >= width || (unsigned)row >= height)
      continue;
    if (time > timestamp)
      continue;
    for (tot = n = 0, rad = 1; rad < 3 && n == 0; rad++)
      for (r = row - rad; r <= row + rad; r++)
        for (c = col - rad; c <= col + rad; c++)
          if ((unsigned)r < height && (unsigned)c < width &&
              (r != row || c != col) && fcol(r, c) == fcol(row, col))
          {
            tot += BAYER2(r, c);
            n++;
          }
    if (n > 0)
      BAYER2(row, col) = tot / n;
  }
  fclose(fp);
  RUN_CALLBACK(LIBRAW_PROGRESS_BAD_PIXELS, 1, 2);
}

void LibRaw::subtract(const char *fname)
{
  FILE *fp;
  int dim[3] = {0, 0, 0}, comment = 0, number = 0, error = 0, nd = 0, c, row,
      col;
  RUN_CALLBACK(LIBRAW_PROGRESS_DARK_FRAME, 0, 2);

  if (!(fp = fopen(fname, "rb")))
  {
    imgdata.process_warnings |= LIBRAW_WARN_BAD_DARKFRAME_FILE;
    return;
  }
  if (fgetc(fp) != 'P' || fgetc(fp) != '5')
    error = 1;
  while (!error && nd < 3 && (c = fgetc(fp)) != EOF)
  {
    if (c == '#')
      comment = 1;
    if (c == '\n')
      comment = 0;
    if (comment)
      continue;
    if (isdigit(c))
      number = 1;
    if (number)
    {
      if (isdigit(c))
        dim[nd] = dim[nd] * 10 + c - '0';
      else if (isspace(c))
      {
        number = 0;
        nd++;
      }
      else
        error = 1;
    }
  }
  if (error || nd < 3)
  {
    fclose(fp);
    return;
  }
  else if (dim[0] != width || dim[1] != height || dim[2] != 65535)
  {
    imgdata.process_warnings |= LIBRAW_WARN_BAD_DARKFRAME_DIM;
    fclose(fp);
    return;
  }
  std::vector<ushort> pixel(width, 0);
  for (row = 0; row < height; row++)
  {
    fread(pixel.data(), 2, width, fp);
    for (col = 0; col < width; col++)
      BAYER(row, col) = MAX(BAYER(row, col) - ntohs(pixel[col]), 0);
  }
  fclose(fp);
  memset(cblack, 0, sizeof cblack);
  black = 0;
  RUN_CALLBACK(LIBRAW_PROGRESS_DARK_FRAME, 1, 2);
}
