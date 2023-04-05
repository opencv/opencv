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

void LibRaw::hat_transform(float *temp, float *base, int st, int size, int sc)
{
  int i;
  for (i = 0; i < sc; i++)
    temp[i] = 2 * base[st * i] + base[st * (sc - i)] + base[st * (i + sc)];
  for (; i + sc < size; i++)
    temp[i] = 2 * base[st * i] + base[st * (i - sc)] + base[st * (i + sc)];
  for (; i < size; i++)
    temp[i] = 2 * base[st * i] + base[st * (i - sc)] +
              base[st * (2 * size - 2 - (i + sc))];
}

#if !defined(LIBRAW_USE_OPENMP)
void LibRaw::wavelet_denoise()
{
  float *fimg = 0, *temp, thold, mul[2], avg, diff;
  int scale = 1, size, lev, hpass, lpass, row, col, nc, c, i, wlast, blk[2];
  ushort *window[4];
  static const float noise[] = {0.8002f, 0.2735f, 0.1202f, 0.0585f,
                                0.0291f, 0.0152f, 0.0080f, 0.0044f};


  while (maximum << scale < 0x10000)
    scale++;
  maximum <<= --scale;
  black <<= scale;
  FORC4 cblack[c] <<= scale;
  if ((size = iheight * iwidth) < 0x15550000)
    fimg = (float *)malloc((size * 3 + iheight + iwidth + 128) * sizeof *fimg);
  temp = fimg + size * 3;
  if ((nc = colors) == 3 && filters)
    nc++;
  FORC(nc)
  { /* denoise R,G1,B,G3 individually */
    for (i = 0; i < size; i++)
      fimg[i] = 256 * sqrt((double)(image[i][c] << scale));
    for (hpass = lev = 0; lev < 5; lev++)
    {
      lpass = size * ((lev & 1) + 1);
      for (row = 0; row < iheight; row++)
      {
        hat_transform(temp, fimg + hpass + row * iwidth, 1, iwidth, 1 << lev);
        for (col = 0; col < iwidth; col++)
          fimg[lpass + row * iwidth + col] = temp[col] * 0.25;
      }
      for (col = 0; col < iwidth; col++)
      {
        hat_transform(temp, fimg + lpass + col, iwidth, iheight, 1 << lev);
        for (row = 0; row < iheight; row++)
          fimg[lpass + row * iwidth + col] = temp[row] * 0.25;
      }
      thold = threshold * noise[lev];
      for (i = 0; i < size; i++)
      {
        fimg[hpass + i] -= fimg[lpass + i];
        if (fimg[hpass + i] < -thold)
          fimg[hpass + i] += thold;
        else if (fimg[hpass + i] > thold)
          fimg[hpass + i] -= thold;
        else
          fimg[hpass + i] = 0;
        if (hpass)
          fimg[i] += fimg[hpass + i];
      }
      hpass = lpass;
    }
    for (i = 0; i < size; i++)
      image[i][c] = CLIP(SQR(fimg[i] + fimg[lpass + i]) / 0x10000);
  }
  if (filters && colors == 3)
  { /* pull G1 and G3 closer together */
    for (row = 0; row < 2; row++)
    {
      mul[row] = 0.125 * pre_mul[FC(row + 1, 0) | 1] / pre_mul[FC(row, 0) | 1];
      blk[row] = cblack[FC(row, 0) | 1];
    }
    for (i = 0; i < 4; i++)
      window[i] = (ushort *)fimg + width * i;
    for (wlast = -1, row = 1; row < height - 1; row++)
    {
      while (wlast < row + 1)
      {
        for (wlast++, i = 0; i < 4; i++)
          window[(i + 3) & 3] = window[i];
        for (col = FC(wlast, 1) & 1; col < width; col += 2)
          window[2][col] = BAYER(wlast, col);
      }
      thold = threshold / 512;
      for (col = (FC(row, 0) & 1) + 1; col < width - 1; col += 2)
      {
        avg = (window[0][col - 1] + window[0][col + 1] + window[2][col - 1] +
               window[2][col + 1] - blk[~row & 1] * 4) *
                  mul[row & 1] +
              (window[1][col] + blk[row & 1]) * 0.5;
        avg = avg < 0 ? 0 : sqrt(avg);
        diff = sqrt((double)BAYER(row, col)) - avg;
        if (diff < -thold)
          diff += thold;
        else if (diff > thold)
          diff -= thold;
        else
          diff = 0;
        BAYER(row, col) = CLIP(SQR(avg + diff) + 0.5);
      }
    }
  }
  free(fimg);
}
#else /* LIBRAW_USE_OPENMP */
void LibRaw::wavelet_denoise()
{
  float *fimg = 0, *temp, thold, mul[2], avg, diff;
  int scale = 1, size, lev, hpass, lpass, row, col, nc, c, i, wlast, blk[2];
  ushort *window[4];
  static const float noise[] = {0.8002, 0.2735, 0.1202, 0.0585,
                                0.0291, 0.0152, 0.0080, 0.0044};

  while (maximum << scale < 0x10000)
    scale++;
  maximum <<= --scale;
  black <<= scale;
  FORC4 cblack[c] <<= scale;
  if ((size = iheight * iwidth) < 0x15550000)
    fimg = (float *)malloc((size * 3 + iheight + iwidth) * sizeof *fimg);
  temp = fimg + size * 3;
  if ((nc = colors) == 3 && filters)
    nc++;
#pragma omp parallel default(shared) private(                                  \
    i, col, row, thold, lev, lpass, hpass, temp, c) firstprivate(scale, size)
  {
    temp = (float *)malloc((iheight + iwidth) * sizeof *fimg);
    FORC(nc)
    { /* denoise R,G1,B,G3 individually */
#pragma omp for
      for (i = 0; i < size; i++)
        fimg[i] = 256 * sqrt((double)(image[i][c] << scale));
      for (hpass = lev = 0; lev < 5; lev++)
      {
        lpass = size * ((lev & 1) + 1);
#pragma omp for
        for (row = 0; row < iheight; row++)
        {
          hat_transform(temp, fimg + hpass + row * iwidth, 1, iwidth, 1 << lev);
          for (col = 0; col < iwidth; col++)
            fimg[lpass + row * iwidth + col] = temp[col] * 0.25;
        }
#pragma omp for
        for (col = 0; col < iwidth; col++)
        {
          hat_transform(temp, fimg + lpass + col, iwidth, iheight, 1 << lev);
          for (row = 0; row < iheight; row++)
            fimg[lpass + row * iwidth + col] = temp[row] * 0.25;
        }
        thold = threshold * noise[lev];
#pragma omp for
        for (i = 0; i < size; i++)
        {
          fimg[hpass + i] -= fimg[lpass + i];
          if (fimg[hpass + i] < -thold)
            fimg[hpass + i] += thold;
          else if (fimg[hpass + i] > thold)
            fimg[hpass + i] -= thold;
          else
            fimg[hpass + i] = 0;
          if (hpass)
            fimg[i] += fimg[hpass + i];
        }
        hpass = lpass;
      }
#pragma omp for
      for (i = 0; i < size; i++)
        image[i][c] = CLIP(SQR(fimg[i] + fimg[lpass + i]) / 0x10000);
    }
    free(temp);
  } /* end omp parallel */
  /* the following loops are hard to parallelize, no idea yes,
   * problem is wlast which is carrying dependency
   * second part should be easier, but did not yet get it right.
   */
  if (filters && colors == 3)
  { /* pull G1 and G3 closer together */
    for (row = 0; row < 2; row++)
    {
      mul[row] = 0.125 * pre_mul[FC(row + 1, 0) | 1] / pre_mul[FC(row, 0) | 1];
      blk[row] = cblack[FC(row, 0) | 1];
    }
    for (i = 0; i < 4; i++)
      window[i] = (ushort *)fimg + width * i;
    for (wlast = -1, row = 1; row < height - 1; row++)
    {
      while (wlast < row + 1)
      {
        for (wlast++, i = 0; i < 4; i++)
          window[(i + 3) & 3] = window[i];
        for (col = FC(wlast, 1) & 1; col < width; col += 2)
          window[2][col] = BAYER(wlast, col);
      }
      thold = threshold / 512;
      for (col = (FC(row, 0) & 1) + 1; col < width - 1; col += 2)
      {
        avg = (window[0][col - 1] + window[0][col + 1] + window[2][col - 1] +
               window[2][col + 1] - blk[~row & 1] * 4) *
                  mul[row & 1] +
              (window[1][col] + blk[row & 1]) * 0.5;
        avg = avg < 0 ? 0 : sqrt(avg);
        diff = sqrt((double)BAYER(row, col)) - avg;
        if (diff < -thold)
          diff += thold;
        else if (diff > thold)
          diff -= thold;
        else
          diff = 0;
        BAYER(row, col) = CLIP(SQR(avg + diff) + 0.5);
      }
    }
  }
  free(fimg);
}

#endif
void LibRaw::median_filter()
{
  ushort(*pix)[4];
  int pass, c, i, j, k, med[9];
  static const uchar opt[] = /* Optimal 9-element median search */
      {1, 2, 4, 5, 7, 8, 0, 1, 3, 4, 6, 7, 1, 2, 4, 5, 7, 8, 0,
       3, 5, 8, 4, 7, 3, 6, 1, 4, 2, 5, 4, 7, 4, 2, 6, 4, 4, 2};

  for (pass = 1; pass <= med_passes; pass++)
  {
    RUN_CALLBACK(LIBRAW_PROGRESS_MEDIAN_FILTER, pass - 1, med_passes);
    for (c = 0; c < 3; c += 2)
    {
      for (pix = image; pix < image + width * height; pix++)
        pix[0][3] = pix[0][c];
      for (pix = image + width; pix < image + width * (height - 1); pix++)
      {
        if ((pix - image + 1) % width < 2)
          continue;
        for (k = 0, i = -width; i <= width; i += width)
          for (j = i - 1; j <= i + 1; j++)
            med[k++] = pix[j][3] - pix[j][1];
        for (i = 0; i < int(sizeof opt); i += 2)
          if (med[opt[i]] > med[opt[i + 1]])
            SWAP(med[opt[i]], med[opt[i + 1]]);
        pix[0][c] = CLIP(med[4] + pix[0][1]);
      }
    }
  }
}

void LibRaw::blend_highlights()
{
  int clip = INT_MAX, row, col, c, i, j;
  static const float trans[2][4][4] = {
      {{1, 1, 1}, {1.7320508f, -1.7320508f, 0}, {-1, -1, 2}},
      {{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}}};
  static const float itrans[2][4][4] = {
      {{1, 0.8660254f, -0.5}, {1, -0.8660254f, -0.5}, {1, 0, 1}},
      {{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}}};
  float cam[2][4], lab[2][4], sum[2], chratio;

  if ((unsigned)(colors - 3) > 1)
    return;
  RUN_CALLBACK(LIBRAW_PROGRESS_HIGHLIGHTS, 0, 2);
  FORCC if (clip > (i = 65535 * pre_mul[c])) clip = i;
  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
    {
      FORCC if (image[row * width + col][c] > clip) break;
      if (c == colors)
        continue;
      FORCC
      {
        cam[0][c] = image[row * width + col][c];
        cam[1][c] = MIN(cam[0][c], clip);
      }
      for (i = 0; i < 2; i++)
      {
        FORCC for (lab[i][c] = j = 0; j < colors; j++) lab[i][c] +=
            trans[colors - 3][c][j] * cam[i][j];
        for (sum[i] = 0, c = 1; c < colors; c++)
          sum[i] += SQR(lab[i][c]);
      }
      chratio = sqrt(sum[1] / sum[0]);
      for (c = 1; c < colors; c++)
        lab[0][c] *= chratio;
      FORCC for (cam[0][c] = j = 0; j < colors; j++) cam[0][c] +=
          itrans[colors - 3][c][j] * lab[0][j];
      FORCC image[row * width + col][c] = cam[0][c] / colors;
    }
  RUN_CALLBACK(LIBRAW_PROGRESS_HIGHLIGHTS, 1, 2);
}

#define SCALE (4 >> shrink)
void LibRaw::recover_highlights()
{
  float *map, sum, wgt, grow;
  int hsat[4], count, spread, change, val, i;
  unsigned high, wide, mrow, mcol, row, col, kc, c, d, y, x;
  ushort *pixel;
  static const signed char dir[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1},
                                        {1, 1},   {1, 0},  {1, -1}, {0, -1}};

  grow = pow(2.0, 4 - highlight);
  FORC(unsigned(colors)) hsat[c] = 32000 * pre_mul[c];
  for (kc = 0, c = 1; c < (unsigned)colors; c++)
    if (pre_mul[kc] < pre_mul[c])
      kc = c;
  high = height / SCALE;
  wide = width / SCALE;
  map = (float *)calloc(high, wide * sizeof *map);
  FORC(unsigned(colors)) if (c != kc)
  {
    RUN_CALLBACK(LIBRAW_PROGRESS_HIGHLIGHTS, c - 1, colors - 1);
    memset(map, 0, high * wide * sizeof *map);
    for (mrow = 0; mrow < high; mrow++)
      for (mcol = 0; mcol < wide; mcol++)
      {
        sum = wgt = count = 0;
        for (row = mrow * SCALE; row < (mrow + 1) * SCALE; row++)
          for (col = mcol * SCALE; col < (mcol + 1) * SCALE; col++)
          {
            pixel = image[row * width + col];
            if (pixel[c] / hsat[c] == 1 && pixel[kc] > 24000)
            {
              sum += pixel[c];
              wgt += pixel[kc];
              count++;
            }
          }
        if (count == SCALE * SCALE)
          map[mrow * wide + mcol] = sum / wgt;
      }
    for (spread = 32 / grow; spread--;)
    {
      for (mrow = 0; mrow < high; mrow++)
        for (mcol = 0; mcol < wide; mcol++)
        {
          if (map[mrow * wide + mcol])
            continue;
          sum = count = 0;
          for (d = 0; d < 8; d++)
          {
            y = mrow + dir[d][0];
            x = mcol + dir[d][1];
            if (y < high && x < wide && map[y * wide + x] > 0)
            {
              sum += (1 + (d & 1)) * map[y * wide + x];
              count += 1 + (d & 1);
            }
          }
          if (count > 3)
            map[mrow * wide + mcol] = -(sum + grow) / (count + grow);
        }
      for (change = i = 0; i < int(high * wide); i++)
        if (map[i] < 0)
        {
          map[i] = -map[i];
          change = 1;
        }
      if (!change)
        break;
    }
    for (i = 0; i < int(high * wide); i++)
      if (map[i] == 0)
        map[i] = 1;
    for (mrow = 0; mrow < high; mrow++)
      for (mcol = 0; mcol < wide; mcol++)
      {
        for (row = mrow * SCALE; row < (mrow + 1) * SCALE; row++)
          for (col = mcol * SCALE; col < (mcol + 1) * SCALE; col++)
          {
            pixel = image[row * width + col];
            if (pixel[c] / hsat[c] > 1)
            {
              val = pixel[kc] * map[mrow * wide + mcol];
              if (pixel[c] < val)
                pixel[c] = CLIP(val);
            }
          }
      }
  }
  free(map);
}
#undef SCALE
