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

void LibRaw::cubic_spline(const int *x_, const int *y_, const int len)
{
  float **A, *b, *c, *d, *x, *y;
  int i, j;

  A = (float **)calloc(((2 * len + 4) * sizeof **A + sizeof *A), 2 * len);
  if (!A)
    return;
  A[0] = (float *)(A + 2 * len);
  for (i = 1; i < 2 * len; i++)
    A[i] = A[0] + 2 * len * i;
  y = len + (x = i + (d = i + (c = i + (b = A[0] + i * i))));
  for (i = 0; i < len; i++)
  {
    x[i] = x_[i] / 65535.0;
    y[i] = y_[i] / 65535.0;
  }
  for (i = len - 1; i > 0; i--)
  {
    b[i] = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    d[i - 1] = x[i] - x[i - 1];
  }
  for (i = 1; i < len - 1; i++)
  {
    A[i][i] = 2 * (d[i - 1] + d[i]);
    if (i > 1)
    {
      A[i][i - 1] = d[i - 1];
      A[i - 1][i] = d[i - 1];
    }
    A[i][len - 1] = 6 * (b[i + 1] - b[i]);
  }
  for (i = 1; i < len - 2; i++)
  {
    float v = A[i + 1][i] / A[i][i];
    for (j = 1; j <= len - 1; j++)
      A[i + 1][j] -= v * A[i][j];
  }
  for (i = len - 2; i > 0; i--)
  {
    float acc = 0;
    for (j = i; j <= len - 2; j++)
      acc += A[i][j] * c[j];
    c[i] = (A[i][len - 1] - acc) / A[i][i];
  }
  for (i = 0; i < 0x10000; i++)
  {
    float x_out = (float)(i / 65535.0);
    float y_out = 0;
    for (j = 0; j < len - 1; j++)
    {
      if (x[j] <= x_out && x_out <= x[j + 1])
      {
        float v = x_out - x[j];
        y_out = y[j] +
                ((y[j + 1] - y[j]) / d[j] -
                 (2 * d[j] * c[j] + c[j + 1] * d[j]) / 6) *
                    v +
                (c[j] * 0.5) * v * v +
                ((c[j + 1] - c[j]) / (6 * d[j])) * v * v * v;
      }
    }
    curve[i] = y_out < 0.0
                   ? 0
                   : (y_out >= 1.0 ? 65535 : (ushort)(y_out * 65535.0 + 0.5));
  }
  free(A);
}
void LibRaw::gamma_curve(double pwr, double ts, int mode, int imax)
{
  int i;
  double g[6], bnd[2] = {0, 0}, r;

  g[0] = pwr;
  g[1] = ts;
  g[2] = g[3] = g[4] = 0;
  bnd[g[1] >= 1] = 1;
  if (g[1] && (g[1] - 1) * (g[0] - 1) <= 0)
  {
    for (i = 0; i < 48; i++)
    {
      g[2] = (bnd[0] + bnd[1]) / 2;
      if (g[0])
        bnd[(pow(g[2] / g[1], -g[0]) - 1) / g[0] - 1 / g[2] > -1] = g[2];
      else
        bnd[g[2] / exp(1 - 1 / g[2]) < g[1]] = g[2];
    }
    g[3] = g[2] / g[1];
    if (g[0])
      g[4] = g[2] * (1 / g[0] - 1);
  }
  if (g[0])
    g[5] = 1 / (g[1] * SQR(g[3]) / 2 - g[4] * (1 - g[3]) +
                (1 - pow(g[3], 1 + g[0])) * (1 + g[4]) / (1 + g[0])) -
           1;
  else
    g[5] = 1 / (g[1] * SQR(g[3]) / 2 + 1 - g[2] - g[3] -
                g[2] * g[3] * (log(g[3]) - 1)) -
           1;
  if (!mode--)
  {
    memcpy(gamm, g, sizeof gamm);
    return;
  }
  for (i = 0; i < 0x10000; i++)
  {
    curve[i] = 0xffff;
    if ((r = (double)i / imax) < 1)
      curve[i] =
          0x10000 *
          (mode ? (r < g[3] ? r * g[1]
                            : (g[0] ? pow(r, g[0]) * (1 + g[4]) - g[4]
                                    : log(r) * g[2] + 1))
                : (r < g[2] ? r / g[1]
                            : (g[0] ? pow((r + g[4]) / (1 + g[4]), 1 / g[0])
                                    : exp((r - 1) / g[2]))));
  }
}

void LibRaw::linear_table(unsigned len)
{
  int i;
  if (len > 0x10000)
    len = 0x10000;
  else if (len < 1)
    return;
  read_shorts(curve, len);
  for (i = len; i < 0x10000; i++)
    curve[i] = curve[i - 1];
  maximum = curve[len < 0x1000 ? 0xfff : len - 1];
}
