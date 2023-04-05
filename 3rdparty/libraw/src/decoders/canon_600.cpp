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

void LibRaw::canon_600_fixed_wb(int temp)
{
  static const short mul[4][5] = {{667, 358, 397, 565, 452},
                                  {731, 390, 367, 499, 517},
                                  {1119, 396, 348, 448, 537},
                                  {1399, 485, 431, 508, 688}};
  int lo, hi, i;
  float frac = 0;

  for (lo = 4; --lo;)
    if (*mul[lo] <= temp)
      break;
  for (hi = 0; hi < 3; hi++)
    if (*mul[hi] >= temp)
      break;
  if (lo != hi)
    frac = (float)(temp - *mul[lo]) / (*mul[hi] - *mul[lo]);
  for (i = 1; i < 5; i++)
    pre_mul[i - 1] = 1 / (frac * mul[hi][i] + (1 - frac) * mul[lo][i]);
}

/* Return values:  0 = white  1 = near white  2 = not white */
int LibRaw::canon_600_color(int ratio[2], int mar)
{
  int clipped = 0, target, miss;

  if (flash_used)
  {
    if (ratio[1] < -104)
    {
      ratio[1] = -104;
      clipped = 1;
    }
    if (ratio[1] > 12)
    {
      ratio[1] = 12;
      clipped = 1;
    }
  }
  else
  {
    if (ratio[1] < -264 || ratio[1] > 461)
      return 2;
    if (ratio[1] < -50)
    {
      ratio[1] = -50;
      clipped = 1;
    }
    if (ratio[1] > 307)
    {
      ratio[1] = 307;
      clipped = 1;
    }
  }
  target = flash_used || ratio[1] < 197 ? -38 - (398 * ratio[1] >> 10)
                                        : -123 + (48 * ratio[1] >> 10);
  if (target - mar <= ratio[0] && target + 20 >= ratio[0] && !clipped)
    return 0;
  miss = target - ratio[0];
  if (abs(miss) >= mar * 4)
    return 2;
  if (miss < -20)
    miss = -20;
  if (miss > mar)
    miss = mar;
  ratio[0] = target - miss;
  return 1;
}

void LibRaw::canon_600_auto_wb()
{
  int mar, row, col, i, j, st, count[] = {0, 0};
  int test[8], total[2][8], ratio[2][2], stat[2];

  memset(&total, 0, sizeof total);
  i = int(canon_ev + 0.5);
  if (i < 10)
    mar = 150;
  else if (i > 12)
    mar = 20;
  else
    mar = 280 - 20 * i;
  if (flash_used)
    mar = 80;
  for (row = 14; row < height - 14; row += 4)
    for (col = 10; col < width; col += 2)
    {
      for (i = 0; i < 8; i++)
        test[(i & 4) + FC(row + (i >> 1), col + (i & 1))] =
            BAYER(row + (i >> 1), col + (i & 1));
      for (i = 0; i < 8; i++)
        if (test[i] < 150 || test[i] > 1500)
          goto next;
      for (i = 0; i < 4; i++)
        if (abs(test[i] - test[i + 4]) > 50)
          goto next;
      for (i = 0; i < 2; i++)
      {
        for (j = 0; j < 4; j += 2)
          ratio[i][j >> 1] =
              ((test[i * 4 + j + 1] - test[i * 4 + j]) << 10) / test[i * 4 + j];
        stat[i] = canon_600_color(ratio[i], mar);
      }
      if ((st = stat[0] | stat[1]) > 1)
        goto next;
      for (i = 0; i < 2; i++)
        if (stat[i])
          for (j = 0; j < 2; j++)
            test[i * 4 + j * 2 + 1] =
                test[i * 4 + j * 2] * (0x400 + ratio[i][j]) >> 10;
      for (i = 0; i < 8; i++)
        total[st][i] += test[i];
      count[st]++;
    next:;
    }
  if (count[0] | count[1])
  {
    st = count[0] * 200 < count[1];
    for (i = 0; i < 4; i++)
		if (total[st][i] + total[st][i + 4])
			pre_mul[i] = 1.0f / (total[st][i] + total[st][i + 4]);
  }
}

void LibRaw::canon_600_coeff()
{
  static const short table[6][12] = {
      {-190, 702, -1878, 2390, 1861, -1349, 905, -393, -432, 944, 2617, -2105},
      {-1203, 1715, -1136, 1648, 1388, -876, 267, 245, -1641, 2153, 3921,
       -3409},
      {-615, 1127, -1563, 2075, 1437, -925, 509, 3, -756, 1268, 2519, -2007},
      {-190, 702, -1886, 2398, 2153, -1641, 763, -251, -452, 964, 3040, -2528},
      {-190, 702, -1878, 2390, 1861, -1349, 905, -393, -432, 944, 2617, -2105},
      {-807, 1319, -1785, 2297, 1388, -876, 769, -257, -230, 742, 2067, -1555}};
  int t = 0, i, c;
  float mc, yc;

  mc = pre_mul[1] / pre_mul[2];
  yc = pre_mul[3] / pre_mul[2];
  if (mc > 1 && mc <= 1.28 && yc < 0.8789)
    t = 1;
  if (mc > 1.28 && mc <= 2)
  {
    if (yc < 0.8789)
      t = 3;
    else if (yc <= 2)
      t = 4;
  }
  if (flash_used)
    t = 5;
  for (raw_color = i = 0; i < 3; i++)
    FORCC rgb_cam[i][c] = float(table[t][i * 4 + c]) / 1024.f;
}

void LibRaw::canon_600_load_raw()
{
  uchar data[1120], *dp;
  ushort *pix;
  int irow, row;

  for (irow = row = 0; irow < height; irow++)
  {
    checkCancel();
    if (fread(data, 1, 1120, ifp) < 1120)
      derror();
    pix = raw_image + row * raw_width;
    for (dp = data; dp < data + 1120; dp += 10, pix += 8)
    {
      pix[0] = (dp[0] << 2) + (dp[1] >> 6);
      pix[1] = (dp[2] << 2) + (dp[1] >> 4 & 3);
      pix[2] = (dp[3] << 2) + (dp[1] >> 2 & 3);
      pix[3] = (dp[4] << 2) + (dp[1] & 3);
      pix[4] = (dp[5] << 2) + (dp[9] & 3);
      pix[5] = (dp[6] << 2) + (dp[9] >> 2 & 3);
      pix[6] = (dp[7] << 2) + (dp[9] >> 4 & 3);
      pix[7] = (dp[8] << 2) + (dp[9] >> 6);
    }
    if ((row += 2) > height)
      row = 1;
  }
}

void LibRaw::canon_600_correct()
{
  int row, col, val;
  static const short mul[4][2] = {
      {1141, 1145}, {1128, 1109}, {1178, 1149}, {1128, 1109}};

  for (row = 0; row < height; row++)
  {
    checkCancel();
    for (col = 0; col < width; col++)
    {
      if ((val = BAYER(row, col) - black) < 0)
        val = 0;
      val = val * mul[row & 3][col & 1] >> 9;
      BAYER(row, col) = val;
    }
  }
  canon_600_fixed_wb(1311);
  canon_600_auto_wb();
  canon_600_coeff();
  maximum = (0x3ff - black) * 1109 >> 9;
  black = 0;
}
