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

/*
   Adaptive Homogeneity-Directed interpolation is based on
   the work of Keigo Hirakawa, Thomas Parks, and Paul Lee.
 */

void LibRaw::cielab(ushort rgb[3], short lab[3])
{
  int c, i, j, k;
  float r, xyz[3];
#ifdef LIBRAW_NOTHREADS
  static float cbrt[0x10000], xyz_cam[3][4];
#else
#define cbrt tls->ahd_data.cbrt
#define xyz_cam tls->ahd_data.xyz_cam
#endif

  if (!rgb)
  {
#ifndef LIBRAW_NOTHREADS
    if (cbrt[0] < -1.0f)
#endif
      for (i = 0; i < 0x10000; i++)
      {
        r = i / 65535.0;
        cbrt[i] =
            r > 0.008856 ? pow(r, 1.f / 3.0f) : 7.787f * r + 16.f / 116.0f;
      }
    for (i = 0; i < 3; i++)
      for (j = 0; j < colors; j++)
        for (xyz_cam[i][j] = k = 0; k < 3; k++)
          xyz_cam[i][j] += LibRaw_constants::xyz_rgb[i][k] * rgb_cam[k][j] /
                           LibRaw_constants::d65_white[i];
    return;
  }
  xyz[0] = xyz[1] = xyz[2] = 0.5;
  FORCC
  {
    xyz[0] += xyz_cam[0][c] * rgb[c];
    xyz[1] += xyz_cam[1][c] * rgb[c];
    xyz[2] += xyz_cam[2][c] * rgb[c];
  }
  xyz[0] = cbrt[CLIP((int)xyz[0])];
  xyz[1] = cbrt[CLIP((int)xyz[1])];
  xyz[2] = cbrt[CLIP((int)xyz[2])];
  lab[0] = 64 * (116 * xyz[1] - 16);
  lab[1] = 64 * 500 * (xyz[0] - xyz[1]);
  lab[2] = 64 * 200 * (xyz[1] - xyz[2]);
#ifndef LIBRAW_NOTHREADS
#undef cbrt
#undef xyz_cam
#endif
}

void LibRaw::ahd_interpolate_green_h_and_v(
    int top, int left, ushort (*out_rgb)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3])
{
  int row, col;
  int c, val;
  ushort(*pix)[4];
  const int rowlimit = MIN(top + LIBRAW_AHD_TILE, height - 2);
  const int collimit = MIN(left + LIBRAW_AHD_TILE, width - 2);

  for (row = top; row < rowlimit; row++)
  {
    col = left + (FC(row, left) & 1);
    for (c = FC(row, col); col < collimit; col += 2)
    {
      pix = image + row * width + col;
      val =
          ((pix[-1][1] + pix[0][c] + pix[1][1]) * 2 - pix[-2][c] - pix[2][c]) >>
          2;
      out_rgb[0][row - top][col - left][1] = ULIM(val, pix[-1][1], pix[1][1]);
      val = ((pix[-width][1] + pix[0][c] + pix[width][1]) * 2 -
             pix[-2 * width][c] - pix[2 * width][c]) >>
            2;
      out_rgb[1][row - top][col - left][1] =
          ULIM(val, pix[-width][1], pix[width][1]);
    }
  }
}
void LibRaw::ahd_interpolate_r_and_b_in_rgb_and_convert_to_cielab(
    int top, int left, ushort (*inout_rgb)[LIBRAW_AHD_TILE][3],
    short (*out_lab)[LIBRAW_AHD_TILE][3])
{
  unsigned row, col;
  int c, val;
  ushort(*pix)[4];
  ushort(*rix)[3];
  short(*lix)[3];
  const unsigned num_pix_per_row = 4 * width;
  const unsigned rowlimit = MIN(top + LIBRAW_AHD_TILE - 1, height - 3);
  const unsigned collimit = MIN(left + LIBRAW_AHD_TILE - 1, width - 3);
  ushort *pix_above;
  ushort *pix_below;
  int t1, t2;

  for (row = top + 1; row < rowlimit; row++)
  {
    pix = image + row * width + left;
    rix = &inout_rgb[row - top][0];
    lix = &out_lab[row - top][0];

    for (col = left + 1; col < collimit; col++)
    {
      pix++;
      pix_above = &pix[0][0] - num_pix_per_row;
      pix_below = &pix[0][0] + num_pix_per_row;
      rix++;
      lix++;

      c = 2 - FC(row, col);

      if (c == 1)
      {
        c = FC(row + 1, col);
        t1 = 2 - c;
        val = pix[0][1] +
              ((pix[-1][t1] + pix[1][t1] - rix[-1][1] - rix[1][1]) >> 1);
        rix[0][t1] = CLIP(val);
        val =
            pix[0][1] + ((pix_above[c] + pix_below[c] -
                          rix[-LIBRAW_AHD_TILE][1] - rix[LIBRAW_AHD_TILE][1]) >>
                         1);
      }
      else
      {
        t1 = -4 + c; /* -4+c: pixel of color c to the left */
        t2 = 4 + c;  /* 4+c: pixel of color c to the right */
        val = rix[0][1] +
              ((pix_above[t1] + pix_above[t2] + pix_below[t1] + pix_below[t2] -
                rix[-LIBRAW_AHD_TILE - 1][1] - rix[-LIBRAW_AHD_TILE + 1][1] -
                rix[+LIBRAW_AHD_TILE - 1][1] - rix[+LIBRAW_AHD_TILE + 1][1] +
                1) >>
               2);
      }
      rix[0][c] = CLIP(val);
      c = FC(row, col);
      rix[0][c] = pix[0][c];
      cielab(rix[0], lix[0]);
    }
  }
}
void LibRaw::ahd_interpolate_r_and_b_and_convert_to_cielab(
    int top, int left, ushort (*inout_rgb)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3],
    short (*out_lab)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3])
{
  int direction;
  for (direction = 0; direction < 2; direction++)
  {
    ahd_interpolate_r_and_b_in_rgb_and_convert_to_cielab(
        top, left, inout_rgb[direction], out_lab[direction]);
  }
}

void LibRaw::ahd_interpolate_build_homogeneity_map(
    int top, int left, short (*lab)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3],
    char (*out_homogeneity_map)[LIBRAW_AHD_TILE][2])
{
  int row, col;
  int tr;
  int direction;
  int i;
  short(*lix)[3];
  short(*lixs[2])[3];
  short *adjacent_lix;
  unsigned ldiff[2][4], abdiff[2][4], leps, abeps;
  static const int dir[4] = {-1, 1, -LIBRAW_AHD_TILE, LIBRAW_AHD_TILE};
  const int rowlimit = MIN(top + LIBRAW_AHD_TILE - 2, height - 4);
  const int collimit = MIN(left + LIBRAW_AHD_TILE - 2, width - 4);
  int homogeneity;
  char(*homogeneity_map_p)[2];

  memset(out_homogeneity_map, 0, 2 * LIBRAW_AHD_TILE * LIBRAW_AHD_TILE);

  for (row = top + 2; row < rowlimit; row++)
  {
    tr = row - top;
    homogeneity_map_p = &out_homogeneity_map[tr][1];
    for (direction = 0; direction < 2; direction++)
    {
      lixs[direction] = &lab[direction][tr][1];
    }

    for (col = left + 2; col < collimit; col++)
    {
      homogeneity_map_p++;

      for (direction = 0; direction < 2; direction++)
      {
        lix = ++lixs[direction];
        for (i = 0; i < 4; i++)
        {
          adjacent_lix = lix[dir[i]];
          ldiff[direction][i] = ABS(lix[0][0] - adjacent_lix[0]);
          abdiff[direction][i] = SQR(lix[0][1] - adjacent_lix[1]) +
                                 SQR(lix[0][2] - adjacent_lix[2]);
        }
      }
      leps = MIN(MAX(ldiff[0][0], ldiff[0][1]), MAX(ldiff[1][2], ldiff[1][3]));
      abeps =
          MIN(MAX(abdiff[0][0], abdiff[0][1]), MAX(abdiff[1][2], abdiff[1][3]));
      for (direction = 0; direction < 2; direction++)
      {
        homogeneity = 0;
        for (i = 0; i < 4; i++)
        {
          if (ldiff[direction][i] <= leps && abdiff[direction][i] <= abeps)
          {
            homogeneity++;
          }
        }
        homogeneity_map_p[0][direction] = homogeneity;
      }
    }
  }
}
void LibRaw::ahd_interpolate_combine_homogeneous_pixels(
    int top, int left, ushort (*rgb)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3],
    char (*homogeneity_map)[LIBRAW_AHD_TILE][2])
{
  int row, col;
  int tr, tc;
  int i, j;
  int direction;
  int hm[2];
  int c;
  const int rowlimit = MIN(top + LIBRAW_AHD_TILE - 3, height - 5);
  const int collimit = MIN(left + LIBRAW_AHD_TILE - 3, width - 5);

  ushort(*pix)[4];
  ushort(*rix[2])[3];

  for (row = top + 3; row < rowlimit; row++)
  {
    tr = row - top;
    pix = &image[row * width + left + 2];
    for (direction = 0; direction < 2; direction++)
    {
      rix[direction] = &rgb[direction][tr][2];
    }

    for (col = left + 3; col < collimit; col++)
    {
      tc = col - left;
      pix++;
      for (direction = 0; direction < 2; direction++)
      {
        rix[direction]++;
      }

      for (direction = 0; direction < 2; direction++)
      {
        hm[direction] = 0;
        for (i = tr - 1; i <= tr + 1; i++)
        {
          for (j = tc - 1; j <= tc + 1; j++)
          {
            hm[direction] += homogeneity_map[i][j][direction];
          }
        }
      }
      if (hm[0] != hm[1])
      {
        memcpy(pix[0], rix[hm[1] > hm[0]][0], 3 * sizeof(ushort));
      }
      else
      {
        FORC3 { pix[0][c] = (rix[0][0][c] + rix[1][0][c]) >> 1; }
      }
    }
  }
}
void LibRaw::ahd_interpolate()
{
    int terminate_flag = 0;
    cielab(0, 0);
    border_interpolate(5);

#ifdef LIBRAW_USE_OPENMP
    int buffer_count = omp_get_max_threads();
#else
    int buffer_count = 1;
#endif

    size_t buffer_size = 26 * LIBRAW_AHD_TILE * LIBRAW_AHD_TILE; /* 1664 kB */
    char** buffers = malloc_omp_buffers(buffer_count, buffer_size);

#ifdef LIBRAW_USE_OPENMP
#pragma omp parallel for schedule(dynamic) default(none) shared(terminate_flag) firstprivate(buffers)
#endif
    for (int top = 2; top < height - 5; top += LIBRAW_AHD_TILE - 6)
    {
#ifdef LIBRAW_USE_OPENMP
        if (0 == omp_get_thread_num())
#endif
            if (callbacks.progress_cb)
            {
                int rr = (*callbacks.progress_cb)(callbacks.progresscb_data,
                    LIBRAW_PROGRESS_INTERPOLATE,
                    top - 2, height - 7);
                if (rr)
                    terminate_flag = 1;
            }

#if defined(LIBRAW_USE_OPENMP)
        char* buffer = buffers[omp_get_thread_num()];
#else
        char* buffer = buffers[0];
#endif

        ushort(*rgb)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3];
        short(*lab)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3];
        char(*homo)[LIBRAW_AHD_TILE][2];

        rgb = (ushort(*)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3])buffer;
        lab = (short(*)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3])(
            buffer + 12 * LIBRAW_AHD_TILE * LIBRAW_AHD_TILE);
        homo = (char(*)[LIBRAW_AHD_TILE][2])(buffer + 24 * LIBRAW_AHD_TILE *
            LIBRAW_AHD_TILE);

        for (int left = 2; !terminate_flag && (left < width - 5);
            left += LIBRAW_AHD_TILE - 6)
        {
            ahd_interpolate_green_h_and_v(top, left, rgb);
            ahd_interpolate_r_and_b_and_convert_to_cielab(top, left, rgb, lab);
            ahd_interpolate_build_homogeneity_map(top, left, lab, homo);
            ahd_interpolate_combine_homogeneous_pixels(top, left, rgb, homo);
        }
    }

    free_omp_buffers(buffers, buffer_count);

    if (terminate_flag)
        throw LIBRAW_EXCEPTION_CANCELLED_BY_CALLBACK;
}
