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

#define HOLE(row) ((holes >> (((row)-raw_height) & 7)) & 1)

/* Kudos to Rich Taylor for figuring out SMaL's compression algorithm. */
void LibRaw::smal_decode_segment(unsigned seg[2][2], int holes)
{
  uchar hist[3][13] = {{7, 7, 0, 0, 63, 55, 47, 39, 31, 23, 15, 7, 0},
                       {7, 7, 0, 0, 63, 55, 47, 39, 31, 23, 15, 7, 0},
                       {3, 3, 0, 0, 63, 47, 31, 15, 0}};
  int low, high = 0xff, carry = 0, nbits = 8;
  int s, count, bin, next, i, sym[3];
  unsigned pix;
  uchar diff, pred[] = {0, 0};
  ushort data = 0, range = 0;

  fseek(ifp, seg[0][1] + 1, SEEK_SET);
  getbits(-1);
  if (seg[1][0] > unsigned(raw_width * raw_height))
    seg[1][0] = raw_width * raw_height;
  for (pix = seg[0][0]; pix < seg[1][0]; pix++)
  {
    for (s = 0; s < 3; s++)
    {
      data = data << nbits | getbits(nbits);
      if (carry < 0)
        carry = (nbits += carry + 1) < 1 ? nbits - 1 : 0;
      while (--nbits >= 0)
        if ((data >> nbits & 0xff) == 0xff)
          break;
      if (nbits > 0)
        data =
            ((data & ((1 << (nbits - 1)) - 1)) << 1) |
            ((data + (((data & (1 << (nbits - 1)))) << 1)) & ((~0u) << nbits));
      if (nbits >= 0)
      {
        data += getbits(1);
        carry = nbits - 8;
      }
      count = ((((data - range + 1) & 0xffff) << 2) - 1) / (high >> 4);
      for (bin = 0; hist[s][bin + 5] > count; bin++)
        ;
      low = hist[s][bin + 5] * (high >> 4) >> 2;
      if (bin)
        high = hist[s][bin + 4] * (high >> 4) >> 2;
      high -= low;
      for (nbits = 0; high << nbits < 128; nbits++)
        ;
      range = (range + low) << nbits;
      high <<= nbits;
      next = hist[s][1];
      if (++hist[s][2] > hist[s][3])
      {
        next = (next + 1) & hist[s][0];
        hist[s][3] = (hist[s][next + 4] - hist[s][next + 5]) >> 2;
        hist[s][2] = 1;
      }
      if (hist[s][hist[s][1] + 4] - hist[s][hist[s][1] + 5] > 1)
      {
        if (bin < hist[s][1])
          for (i = bin; i < hist[s][1]; i++)
            hist[s][i + 5]--;
        else if (next <= bin)
          for (i = hist[s][1]; i < bin; i++)
            hist[s][i + 5]++;
      }
      hist[s][1] = next;
      sym[s] = bin;
    }
    diff = sym[2] << 5 | sym[1] << 2 | (sym[0] & 3);
    if (sym[0] & 4)
      diff = diff ? -diff : 0x80;
    if (ftell(ifp) + 12 >= seg[1][1])
      diff = 0;
    if (pix >= unsigned(raw_width * raw_height))
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    raw_image[pix] = pred[pix & 1] += diff;
    if (!(pix & 1) && HOLE(pix / raw_width))
      pix += 2;
  }
  maximum = 0xff;
}

void LibRaw::smal_v6_load_raw()
{
  unsigned seg[2][2];

  fseek(ifp, 16, SEEK_SET);
  seg[0][0] = 0;
  seg[0][1] = get2();
  seg[1][0] = raw_width * raw_height;
  seg[1][1] = INT_MAX;
  smal_decode_segment(seg, 0);
}

int LibRaw::median4(int *p)
{
  int min, max, sum, i;

  min = max = sum = p[0];
  for (i = 1; i < 4; i++)
  {
    sum += p[i];
    if (min > p[i])
      min = p[i];
    if (max < p[i])
      max = p[i];
  }
  return (sum - min - max) >> 1;
}

void LibRaw::fill_holes(int holes)
{
  int row, col, val[4];

  for (row = 2; row < height - 2; row++)
  {
    if (!HOLE(row))
      continue;
    for (col = 1; col < width - 1; col += 4)
    {
      val[0] = RAW(row - 1, col - 1);
      val[1] = RAW(row - 1, col + 1);
      val[2] = RAW(row + 1, col - 1);
      val[3] = RAW(row + 1, col + 1);
      RAW(row, col) = median4(val);
    }
    for (col = 2; col < width - 2; col += 4)
      if (HOLE(row - 2) || HOLE(row + 2))
        RAW(row, col) = (RAW(row, col - 2) + RAW(row, col + 2)) >> 1;
      else
      {
        val[0] = RAW(row, col - 2);
        val[1] = RAW(row, col + 2);
        val[2] = RAW(row - 2, col);
        val[3] = RAW(row + 2, col);
        RAW(row, col) = median4(val);
      }
  }
}

void LibRaw::smal_v9_load_raw()
{
  unsigned seg[256][2], offset, nseg, holes, i;

  fseek(ifp, 67, SEEK_SET);
  offset = get4();
  nseg = (uchar)fgetc(ifp);
  fseek(ifp, offset, SEEK_SET);
  for (i = 0; i < nseg * 2; i++)
    ((unsigned *)seg)[i] = get4() + data_offset * (i & 1);
  fseek(ifp, 78, SEEK_SET);
  holes = fgetc(ifp);
  fseek(ifp, 88, SEEK_SET);
  seg[nseg][0] = raw_height * raw_width;
  seg[nseg][1] = get4() + data_offset;
  for (i = 0; i < nseg; i++)
    smal_decode_segment(seg + i, holes);
  if (holes)
    fill_holes(holes);
}

#undef HOLE
