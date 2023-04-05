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

void LibRaw::pre_interpolate()
{
  ushort(*img)[4];
  int row, col, c;
  RUN_CALLBACK(LIBRAW_PROGRESS_PRE_INTERPOLATE, 0, 2);
  if (shrink)
  {
    if (half_size)
    {
      height = iheight;
      width = iwidth;
      if (filters == 9)
      {
        for (row = 0; row < 3; row++)
          for (col = 1; col < 4; col++)
            if (!(image[row * width + col][0] | image[row * width + col][2]))
              goto break2;
      break2:
        for (; row < height; row += 3)
          for (col = (col - 1) % 3 + 1; col < width - 1; col += 3)
          {
            img = image + row * width + col;
            for (c = 0; c < 3; c += 2)
              img[0][c] = (img[-1][c] + img[1][c]) >> 1;
          }
      }
    }
    else
    {
      img = (ushort(*)[4])calloc(height, width * sizeof *img);
      for (row = 0; row < height; row++)
        for (col = 0; col < width; col++)
        {
          c = fcol(row, col);
          img[row * width + col][c] =
              image[(row >> 1) * iwidth + (col >> 1)][c];
        }
      free(image);
      image = img;
      shrink = 0;
    }
  }
  if (filters > 1000 && colors == 3)
  {
    mix_green = four_color_rgb ^ half_size;
    if (four_color_rgb | half_size)
      colors++;
    else
    {
      for (row = FC(1, 0) >> 1; row < height; row += 2)
        for (col = FC(row, 1) & 1; col < width; col += 2)
          image[row * width + col][1] = image[row * width + col][3];
      filters &= ~((filters & 0x55555555U) << 1);
    }
  }
  if (half_size)
    filters = 0;
  RUN_CALLBACK(LIBRAW_PROGRESS_PRE_INTERPOLATE, 1, 2);
}

void LibRaw::border_interpolate(int border)
{
  unsigned row, col, y, x, f, c, sum[8];

  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
    {
      if (col == (unsigned)border && row >= (unsigned)border && row < (unsigned)(height - border))
        col = width - border;
      memset(sum, 0, sizeof sum);
      for (y = row - 1; y != row + 2; y++)
        for (x = col - 1; x != col + 2; x++)
          if (y < height && x < width)
          {
            f = fcol(y, x);
            sum[f] += image[y * width + x][f];
            sum[f + 4]++;
          }
      f = fcol(row, col);
      FORC(unsigned(colors)) if (c != f && sum[c + 4]) image[row * width + col][c] =
          sum[c] / sum[c + 4];
    }
}

void LibRaw::lin_interpolate_loop(int *code, int size)
{
  int row;
  for (row = 1; row < height - 1; row++)
  {
    int col, *ip;
    ushort *pix;
    for (col = 1; col < width - 1; col++)
    {
      int i;
      int sum[4];
      pix = image[row * width + col];
      ip = code + ((((row % size) * 16) + (col % size)) * 32);
      memset(sum, 0, sizeof sum);
      for (i = *ip++; i--; ip += 3)
        sum[ip[2]] += pix[ip[0]] << ip[1];
      for (i = colors; --i; ip += 2)
        pix[ip[0]] = sum[ip[0]] * ip[1] >> 8;
    }
  }
}

void LibRaw::lin_interpolate()
{
  std::vector<int> code_buffer(16 * 16 * 32);
  int* code = &code_buffer[0], size = 16, *ip, sum[4];
  int f, c, x, y, row, col, shift, color;

  RUN_CALLBACK(LIBRAW_PROGRESS_INTERPOLATE, 0, 3);

  if (filters == 9)
    size = 6;
  border_interpolate(1);
  for (row = 0; row < size; row++)
    for (col = 0; col < size; col++)
    {
      ip = code + (((row * 16) + col) * 32) + 1;
      f = fcol(row, col);
      memset(sum, 0, sizeof sum);
      for (y = -1; y <= 1; y++)
        for (x = -1; x <= 1; x++)
        {
          shift = (y == 0) + (x == 0);
          color = fcol(row + y + 48, col + x + 48);
          if (color == f)
            continue;
          *ip++ = (width * y + x) * 4 + color;
          *ip++ = shift;
          *ip++ = color;
          sum[color] += 1 << shift;
        }
      code[(row * 16 + col) * 32] = (ip - (code + ((row * 16) + col) * 32)) / 3;
      FORCC
      if (c != f)
      {
        *ip++ = c;
        *ip++ = sum[c] > 0 ? 256 / sum[c] : 0;
      }
    }
  RUN_CALLBACK(LIBRAW_PROGRESS_INTERPOLATE, 1, 3);
  lin_interpolate_loop(code, size);
  RUN_CALLBACK(LIBRAW_PROGRESS_INTERPOLATE, 2, 3);
}

/*
   This algorithm is officially called:

   "Interpolation using a Threshold-based variable number of gradients"

   described in
   http://scien.stanford.edu/pages/labsite/1999/psych221/projects/99/tingchen/algodep/vargra.html

   I've extended the basic idea to work with non-Bayer filter arrays.
   Gradients are numbered clockwise from NW=0 to W=7.
 */
void LibRaw::vng_interpolate()
{
  static const signed char *cp,
      terms[] =
          {-2, -2, +0,   -1, 0,  0x01, -2, -2, +0,   +0, 1,  0x01, -2, -1, -1,
           +0, 0,  0x01, -2, -1, +0,   -1, 0,  0x02, -2, -1, +0,   +0, 0,  0x03,
           -2, -1, +0,   +1, 1,  0x01, -2, +0, +0,   -1, 0,  0x06, -2, +0, +0,
           +0, 1,  0x02, -2, +0, +0,   +1, 0,  0x03, -2, +1, -1,   +0, 0,  0x04,
           -2, +1, +0,   -1, 1,  0x04, -2, +1, +0,   +0, 0,  0x06, -2, +1, +0,
           +1, 0,  0x02, -2, +2, +0,   +0, 1,  0x04, -2, +2, +0,   +1, 0,  0x04,
           -1, -2, -1,   +0, 0,  -128, -1, -2, +0,   -1, 0,  0x01, -1, -2, +1,
           -1, 0,  0x01, -1, -2, +1,   +0, 1,  0x01, -1, -1, -1,   +1, 0,  -120,
           -1, -1, +1,   -2, 0,  0x40, -1, -1, +1,   -1, 0,  0x22, -1, -1, +1,
           +0, 0,  0x33, -1, -1, +1,   +1, 1,  0x11, -1, +0, -1,   +2, 0,  0x08,
           -1, +0, +0,   -1, 0,  0x44, -1, +0, +0,   +1, 0,  0x11, -1, +0, +1,
           -2, 1,  0x40, -1, +0, +1,   -1, 0,  0x66, -1, +0, +1,   +0, 1,  0x22,
           -1, +0, +1,   +1, 0,  0x33, -1, +0, +1,   +2, 1,  0x10, -1, +1, +1,
           -1, 1,  0x44, -1, +1, +1,   +0, 0,  0x66, -1, +1, +1,   +1, 0,  0x22,
           -1, +1, +1,   +2, 0,  0x10, -1, +2, +0,   +1, 0,  0x04, -1, +2, +1,
           +0, 1,  0x04, -1, +2, +1,   +1, 0,  0x04, +0, -2, +0,   +0, 1,  -128,
           +0, -1, +0,   +1, 1,  -120, +0, -1, +1,   -2, 0,  0x40, +0, -1, +1,
           +0, 0,  0x11, +0, -1, +2,   -2, 0,  0x40, +0, -1, +2,   -1, 0,  0x20,
           +0, -1, +2,   +0, 0,  0x30, +0, -1, +2,   +1, 1,  0x10, +0, +0, +0,
           +2, 1,  0x08, +0, +0, +2,   -2, 1,  0x40, +0, +0, +2,   -1, 0,  0x60,
           +0, +0, +2,   +0, 1,  0x20, +0, +0, +2,   +1, 0,  0x30, +0, +0, +2,
           +2, 1,  0x10, +0, +1, +1,   +0, 0,  0x44, +0, +1, +1,   +2, 0,  0x10,
           +0, +1, +2,   -1, 1,  0x40, +0, +1, +2,   +0, 0,  0x60, +0, +1, +2,
           +1, 0,  0x20, +0, +1, +2,   +2, 0,  0x10, +1, -2, +1,   +0, 0,  -128,
           +1, -1, +1,   +1, 0,  -120, +1, +0, +1,   +2, 0,  0x08, +1, +0, +2,
           -1, 0,  0x40, +1, +0, +2,   +1, 0,  0x10},
      chood[] = {-1, -1, -1, 0, -1, +1, 0, +1, +1, +1, +1, 0, +1, -1, 0, -1};
  ushort(*brow[5])[4], *pix;
  int prow = 8, pcol = 2, *ip, *code[16][16], gval[8], gmin, gmax, sum[4];
  int row, col, x, y, x1, x2, y1, y2, t, weight, grads, color, diag;
  int g, diff, thold, num, c;

  lin_interpolate();

  if (filters == 1)
    prow = pcol = 16;
  if (filters == 9)
    prow = pcol = 6;
  ip = (int *)calloc(prow * pcol, 1280);
  for (row = 0; row < prow; row++) /* Precalculate for VNG */
    for (col = 0; col < pcol; col++)
    {
      code[row][col] = ip;
      for (cp = terms, t = 0; t < 64; t++)
      {
        y1 = *cp++;
        x1 = *cp++;
        y2 = *cp++;
        x2 = *cp++;
        weight = *cp++;
        grads = *cp++;
        color = fcol(row + y1 + 144, col + x1 + 144);
        if (fcol(row + y2 + 144, col + x2 + 144) != color)
          continue;
        diag = (fcol(row, col + 1) == color && fcol(row + 1, col) == color) ? 2
                                                                            : 1;
        if (abs(y1 - y2) == diag && abs(x1 - x2) == diag)
          continue;
        *ip++ = (y1 * width + x1) * 4 + color;
        *ip++ = (y2 * width + x2) * 4 + color;
        *ip++ = weight;
        for (g = 0; g < 8; g++)
          if (grads & 1 << g)
            *ip++ = g;
        *ip++ = -1;
      }
      *ip++ = INT_MAX;
      for (cp = chood, g = 0; g < 8; g++)
      {
        y = *cp++;
        x = *cp++;
        *ip++ = (y * width + x) * 4;
        color = fcol(row, col);
        if (fcol(row + y + 144, col + x + 144) != color &&
            fcol(row + y * 2 + 144, col + x * 2 + 144) == color)
          *ip++ = (y * width + x) * 8 + color;
        else
          *ip++ = 0;
      }
    }
  brow[4] = (ushort(*)[4])calloc(width * 3, sizeof **brow);
  for (row = 0; row < 3; row++)
    brow[row] = brow[4] + row * width;
  for (row = 2; row < height - 2; row++)
  { /* Do VNG interpolation */
    if (!((row - 2) % 256))
      RUN_CALLBACK(LIBRAW_PROGRESS_INTERPOLATE, (row - 2) / 256 + 1,
                   ((height - 3) / 256) + 1);
    for (col = 2; col < width - 2; col++)
    {
      pix = image[row * width + col];
      ip = code[row % prow][col % pcol];
      memset(gval, 0, sizeof gval);
      while ((g = ip[0]) != INT_MAX)
      { /* Calculate gradients */
        diff = ABS(pix[g] - pix[ip[1]]) << ip[2];
        gval[ip[3]] += diff;
        ip += 5;
        if ((g = ip[-1]) == -1)
          continue;
        gval[g] += diff;
        while ((g = *ip++) != -1)
          gval[g] += diff;
      }
      ip++;
      gmin = gmax = gval[0]; /* Choose a threshold */
      for (g = 1; g < 8; g++)
      {
        if (gmin > gval[g])
          gmin = gval[g];
        if (gmax < gval[g])
          gmax = gval[g];
      }
      if (gmax == 0)
      {
        memcpy(brow[2][col], pix, sizeof *image);
        continue;
      }
      thold = gmin + (gmax >> 1);
      memset(sum, 0, sizeof sum);
      color = fcol(row, col);
      for (num = g = 0; g < 8; g++, ip += 2)
      { /* Average the neighbors */
        if (gval[g] <= thold)
        {
          FORCC
          if (c == color && ip[1])
            sum[c] += (pix[c] + pix[ip[1]]) >> 1;
          else
            sum[c] += pix[ip[0] + c];
          num++;
        }
      }
      FORCC
      { /* Save to buffer */
        t = pix[color];
        if (c != color)
          t += (sum[c] - sum[color]) / num;
        brow[2][col][c] = CLIP(t);
      }
    }
    if (row > 3) /* Write buffer to image */
      memcpy(image[(row - 2) * width + 2], brow[0] + 2,
             (width - 4) * sizeof *image);
    for (g = 0; g < 4; g++)
      brow[(g - 1) & 3] = brow[g];
  }
  memcpy(image[(row - 2) * width + 2], brow[0] + 2,
         (width - 4) * sizeof *image);
  memcpy(image[(row - 1) * width + 2], brow[1] + 2,
         (width - 4) * sizeof *image);
  free(brow[4]);
  free(code[0][0]);
}

/*
   Patterned Pixel Grouping Interpolation by Alain Desbiolles
*/
void LibRaw::ppg_interpolate()
{
  int dir[5] = {1, width, -1, -width, 1};
  int row, col, diff[2], guess[2], c, d, i;
  ushort(*pix)[4];

  border_interpolate(3);

  /*  Fill in the green layer with gradients and pattern recognition: */
  RUN_CALLBACK(LIBRAW_PROGRESS_INTERPOLATE, 0, 3);
#ifdef LIBRAW_USE_OPENMP
#pragma omp parallel for default(shared) private(guess, diff, row, col, d, c,  \
                                                 i, pix) schedule(static)
#endif
  for (row = 3; row < height - 3; row++)
    for (col = 3 + (FC(row, 3) & 1), c = FC(row, col); col < width - 3;
         col += 2)
    {
      pix = image + row * width + col;
      for (i = 0; i < 2; i++)
      {
        d = dir[i];
        guess[i] = (pix[-d][1] + pix[0][c] + pix[d][1]) * 2 - pix[-2 * d][c] -
                   pix[2 * d][c];
        diff[i] =
            (ABS(pix[-2 * d][c] - pix[0][c]) + ABS(pix[2 * d][c] - pix[0][c]) +
             ABS(pix[-d][1] - pix[d][1])) *
                3 +
            (ABS(pix[3 * d][1] - pix[d][1]) +
             ABS(pix[-3 * d][1] - pix[-d][1])) *
                2;
      }
      d = dir[i = diff[0] > diff[1]];
      pix[0][1] = ULIM(guess[i] >> 2, pix[d][1], pix[-d][1]);
    }
  /*  Calculate red and blue for each green pixel:		*/
  RUN_CALLBACK(LIBRAW_PROGRESS_INTERPOLATE, 1, 3);
#ifdef LIBRAW_USE_OPENMP
#pragma omp parallel for default(shared) private(guess, diff, row, col, d, c,  \
                                                 i, pix) schedule(static)
#endif
  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 2) & 1), c = FC(row, col + 1); col < width - 1;
         col += 2)
    {
      pix = image + row * width + col;
      for (i = 0; i < 2; c = 2 - c, i++)
      {
        d = dir[i];
        pix[0][c] = CLIP(
            (pix[-d][c] + pix[d][c] + 2 * pix[0][1] - pix[-d][1] - pix[d][1]) >>
            1);
      }
    }
  /*  Calculate blue for red pixels and vice versa:		*/
  RUN_CALLBACK(LIBRAW_PROGRESS_INTERPOLATE, 2, 3);
#ifdef LIBRAW_USE_OPENMP
#pragma omp parallel for default(shared) private(guess, diff, row, col, d, c,  \
                                                 i, pix) schedule(static)
#endif
  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 1) & 1), c = 2 - FC(row, col); col < width - 1;
         col += 2)
    {
      pix = image + row * width + col;
      for (i = 0; i < 2; i++)
      {
        d = dir[i] + dir[i+1];
        diff[i] = ABS(pix[-d][c] - pix[d][c]) + ABS(pix[-d][1] - pix[0][1]) +
                  ABS(pix[d][1] - pix[0][1]);
        guess[i] =
            pix[-d][c] + pix[d][c] + 2 * pix[0][1] - pix[-d][1] - pix[d][1];
      }
      if (diff[0] != diff[1])
        pix[0][c] = CLIP(guess[diff[0] > diff[1]] >> 1);
      else
        pix[0][c] = CLIP((guess[0] + guess[1]) >> 2);
    }
}
