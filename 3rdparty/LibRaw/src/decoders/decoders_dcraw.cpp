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
#include "../../internal/libraw_cameraids.h"

unsigned LibRaw::getbithuff(int nbits, ushort *huff)
{
#ifdef LIBRAW_NOTHREADS
  static unsigned bitbuf = 0;
  static int vbits = 0, reset = 0;
#else
#define bitbuf tls->getbits.bitbuf
#define vbits tls->getbits.vbits
#define reset tls->getbits.reset
#endif
  unsigned c;

  if (nbits > 25)
    return 0;
  if (nbits < 0)
    return bitbuf = vbits = reset = 0;
  if (nbits == 0 || vbits < 0)
    return 0;
  while (!reset && vbits < nbits && (c = fgetc(ifp)) != (unsigned)EOF &&
         !(reset = zero_after_ff && c == 0xff && fgetc(ifp)))
  {
    bitbuf = (bitbuf << 8) + (uchar)c;
    vbits += 8;
  }
  c = vbits == 0 ? 0 : bitbuf << (32 - vbits) >> (32 - nbits);
  if (huff)
  {
    vbits -= huff[c] >> 8;
    c = (uchar)huff[c];
  }
  else
    vbits -= nbits;
  if (vbits < 0)
    derror();
  return c;
#ifndef LIBRAW_NOTHREADS
#undef bitbuf
#undef vbits
#undef reset
#endif
}

/*
   Construct a decode tree according the specification in *source.
   The first 16 bytes specify how many codes should be 1-bit, 2-bit
   3-bit, etc.  Bytes after that are the leaf values.

   For example, if the source is

    { 0,1,4,2,3,1,2,0,0,0,0,0,0,0,0,0,
      0x04,0x03,0x05,0x06,0x02,0x07,0x01,0x08,0x09,0x00,0x0a,0x0b,0xff  },

   then the code is

        00		0x04
        010		0x03
        011		0x05
        100		0x06
        101		0x02
        1100		0x07
        1101		0x01
        11100		0x08
        11101		0x09
        11110		0x00
        111110		0x0a
        1111110		0x0b
        1111111		0xff
 */
ushort *LibRaw::make_decoder_ref(const uchar **source)
{
  int max, len, h, i, j;
  const uchar *count;
  ushort *huff;

  count = (*source += 16) - 17;
  for (max = 16; max && !count[max]; max--)
    ;
  huff = (ushort *)calloc(1 + (1 << max), sizeof *huff);
  huff[0] = max;
  for (h = len = 1; len <= max; len++)
    for (i = 0; i < count[len]; i++, ++*source)
      for (j = 0; j < 1 << (max - len); j++)
        if (h <= 1 << max)
          huff[h++] = len << 8 | **source;
  return huff;
}

ushort *LibRaw::make_decoder(const uchar *source)
{
  return make_decoder_ref(&source);
}

void LibRaw::crw_init_tables(unsigned table, ushort *huff[2])
{
  static const uchar first_tree[3][29] = {
      {0,    1,    4,    2,    3,    1,    2,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0x04, 0x03, 0x05, 0x06,
       0x02, 0x07, 0x01, 0x08, 0x09, 0x00, 0x0a, 0x0b, 0xff},
      {0,    2,    2,    3,    1,    1,    1,    1,    2,    0,
       0,    0,    0,    0,    0,    0,    0x03, 0x02, 0x04, 0x01,
       0x05, 0x00, 0x06, 0x07, 0x09, 0x08, 0x0a, 0x0b, 0xff},
      {0,    0,    6,    3,    1,    1,    2,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0x06, 0x05, 0x07, 0x04,
       0x08, 0x03, 0x09, 0x02, 0x00, 0x0a, 0x01, 0x0b, 0xff},
  };
  static const uchar second_tree[3][180] = {
      {0,    2,    2,    2,    1,    4,    2,    1,    2,    5,    1,    1,
       0,    0,    0,    139,  0x03, 0x04, 0x02, 0x05, 0x01, 0x06, 0x07, 0x08,
       0x12, 0x13, 0x11, 0x14, 0x09, 0x15, 0x22, 0x00, 0x21, 0x16, 0x0a, 0xf0,
       0x23, 0x17, 0x24, 0x31, 0x32, 0x18, 0x19, 0x33, 0x25, 0x41, 0x34, 0x42,
       0x35, 0x51, 0x36, 0x37, 0x38, 0x29, 0x79, 0x26, 0x1a, 0x39, 0x56, 0x57,
       0x28, 0x27, 0x52, 0x55, 0x58, 0x43, 0x76, 0x59, 0x77, 0x54, 0x61, 0xf9,
       0x71, 0x78, 0x75, 0x96, 0x97, 0x49, 0xb7, 0x53, 0xd7, 0x74, 0xb6, 0x98,
       0x47, 0x48, 0x95, 0x69, 0x99, 0x91, 0xfa, 0xb8, 0x68, 0xb5, 0xb9, 0xd6,
       0xf7, 0xd8, 0x67, 0x46, 0x45, 0x94, 0x89, 0xf8, 0x81, 0xd5, 0xf6, 0xb4,
       0x88, 0xb1, 0x2a, 0x44, 0x72, 0xd9, 0x87, 0x66, 0xd4, 0xf5, 0x3a, 0xa7,
       0x73, 0xa9, 0xa8, 0x86, 0x62, 0xc7, 0x65, 0xc8, 0xc9, 0xa1, 0xf4, 0xd1,
       0xe9, 0x5a, 0x92, 0x85, 0xa6, 0xe7, 0x93, 0xe8, 0xc1, 0xc6, 0x7a, 0x64,
       0xe1, 0x4a, 0x6a, 0xe6, 0xb3, 0xf1, 0xd3, 0xa5, 0x8a, 0xb2, 0x9a, 0xba,
       0x84, 0xa4, 0x63, 0xe5, 0xc5, 0xf3, 0xd2, 0xc4, 0x82, 0xaa, 0xda, 0xe4,
       0xf2, 0xca, 0x83, 0xa3, 0xa2, 0xc3, 0xea, 0xc2, 0xe2, 0xe3, 0xff, 0xff},
      {0,    2,    2,    1,    4,    1,    4,    1,    3,    3,    1,    0,
       0,    0,    0,    140,  0x02, 0x03, 0x01, 0x04, 0x05, 0x12, 0x11, 0x06,
       0x13, 0x07, 0x08, 0x14, 0x22, 0x09, 0x21, 0x00, 0x23, 0x15, 0x31, 0x32,
       0x0a, 0x16, 0xf0, 0x24, 0x33, 0x41, 0x42, 0x19, 0x17, 0x25, 0x18, 0x51,
       0x34, 0x43, 0x52, 0x29, 0x35, 0x61, 0x39, 0x71, 0x62, 0x36, 0x53, 0x26,
       0x38, 0x1a, 0x37, 0x81, 0x27, 0x91, 0x79, 0x55, 0x45, 0x28, 0x72, 0x59,
       0xa1, 0xb1, 0x44, 0x69, 0x54, 0x58, 0xd1, 0xfa, 0x57, 0xe1, 0xf1, 0xb9,
       0x49, 0x47, 0x63, 0x6a, 0xf9, 0x56, 0x46, 0xa8, 0x2a, 0x4a, 0x78, 0x99,
       0x3a, 0x75, 0x74, 0x86, 0x65, 0xc1, 0x76, 0xb6, 0x96, 0xd6, 0x89, 0x85,
       0xc9, 0xf5, 0x95, 0xb4, 0xc7, 0xf7, 0x8a, 0x97, 0xb8, 0x73, 0xb7, 0xd8,
       0xd9, 0x87, 0xa7, 0x7a, 0x48, 0x82, 0x84, 0xea, 0xf4, 0xa6, 0xc5, 0x5a,
       0x94, 0xa4, 0xc6, 0x92, 0xc3, 0x68, 0xb5, 0xc8, 0xe4, 0xe5, 0xe6, 0xe9,
       0xa2, 0xa3, 0xe3, 0xc2, 0x66, 0x67, 0x93, 0xaa, 0xd4, 0xd5, 0xe7, 0xf8,
       0x88, 0x9a, 0xd7, 0x77, 0xc4, 0x64, 0xe2, 0x98, 0xa5, 0xca, 0xda, 0xe8,
       0xf3, 0xf6, 0xa9, 0xb2, 0xb3, 0xf2, 0xd2, 0x83, 0xba, 0xd3, 0xff, 0xff},
      {0,    0,    6,    2,    1,    3,    3,    2,    5,    1,    2,    2,
       8,    10,   0,    117,  0x04, 0x05, 0x03, 0x06, 0x02, 0x07, 0x01, 0x08,
       0x09, 0x12, 0x13, 0x14, 0x11, 0x15, 0x0a, 0x16, 0x17, 0xf0, 0x00, 0x22,
       0x21, 0x18, 0x23, 0x19, 0x24, 0x32, 0x31, 0x25, 0x33, 0x38, 0x37, 0x34,
       0x35, 0x36, 0x39, 0x79, 0x57, 0x58, 0x59, 0x28, 0x56, 0x78, 0x27, 0x41,
       0x29, 0x77, 0x26, 0x42, 0x76, 0x99, 0x1a, 0x55, 0x98, 0x97, 0xf9, 0x48,
       0x54, 0x96, 0x89, 0x47, 0xb7, 0x49, 0xfa, 0x75, 0x68, 0xb6, 0x67, 0x69,
       0xb9, 0xb8, 0xd8, 0x52, 0xd7, 0x88, 0xb5, 0x74, 0x51, 0x46, 0xd9, 0xf8,
       0x3a, 0xd6, 0x87, 0x45, 0x7a, 0x95, 0xd5, 0xf6, 0x86, 0xb4, 0xa9, 0x94,
       0x53, 0x2a, 0xa8, 0x43, 0xf5, 0xf7, 0xd4, 0x66, 0xa7, 0x5a, 0x44, 0x8a,
       0xc9, 0xe8, 0xc8, 0xe7, 0x9a, 0x6a, 0x73, 0x4a, 0x61, 0xc7, 0xf4, 0xc6,
       0x65, 0xe9, 0x72, 0xe6, 0x71, 0x91, 0x93, 0xa6, 0xda, 0x92, 0x85, 0x62,
       0xf3, 0xc5, 0xb2, 0xa4, 0x84, 0xba, 0x64, 0xa5, 0xb3, 0xd2, 0x81, 0xe5,
       0xd3, 0xaa, 0xc4, 0xca, 0xf2, 0xb1, 0xe4, 0xd1, 0x83, 0x63, 0xea, 0xc3,
       0xe2, 0x82, 0xf1, 0xa3, 0xc2, 0xa1, 0xc1, 0xe3, 0xa2, 0xe1, 0xff, 0xff}};
  if (table > 2)
    table = 2;
  huff[0] = make_decoder(first_tree[table]);
  huff[1] = make_decoder(second_tree[table]);
}

/*
   Return 0 if the image starts with compressed data,
   1 if it starts with uncompressed low-order bits.

   In Canon compressed data, 0xff is always followed by 0x00.
 */
int LibRaw::canon_has_lowbits()
{
  uchar test[0x4000];
  int ret = 1, i;

  fseek(ifp, 0, SEEK_SET);
  fread(test, 1, sizeof test, ifp);
  for (i = 540; i < int(sizeof test - 1); i++)
    if (test[i] == 0xff)
    {
      if (test[i + 1])
        return 1;
      ret = 0;
    }
  return ret;
}

void LibRaw::canon_load_raw()
{
  ushort *pixel, *prow, *huff[2];
  int nblocks, lowbits, i, c, row, r, val;
  INT64 save;
  int block, diffbuf[64], leaf, len, diff, carry = 0, pnum = 0, base[2];

  crw_init_tables(tiff_compress, huff);
  lowbits = canon_has_lowbits();
  if (!lowbits)
    maximum = 0x3ff;
  fseek(ifp, 540 + lowbits * raw_height * raw_width / 4, SEEK_SET);
  zero_after_ff = 1;
  getbits(-1);
  try
  {
    for (row = 0; row < raw_height; row += 8)
    {
      checkCancel();
      pixel = raw_image + row * raw_width;
      nblocks = MIN(8, raw_height - row) * raw_width >> 6;
      for (block = 0; block < nblocks; block++)
      {
        memset(diffbuf, 0, sizeof diffbuf);
        for (i = 0; i < 64; i++)
        {
          leaf = gethuff(huff[i > 0]);
          if (leaf == 0 && i)
            break;
          if (leaf == 0xff)
            continue;
          i += leaf >> 4;
          len = leaf & 15;
          if (len == 0)
            continue;
          diff = getbits(len);
          if ((diff & (1 << (len - 1))) == 0)
            diff -= (1 << len) - 1;
          if (i < 64)
            diffbuf[i] = diff;
        }
        diffbuf[0] += carry;
        carry = diffbuf[0];
        for (i = 0; i < 64; i++)
        {
          if (pnum++ % raw_width == 0)
            base[0] = base[1] = 512;
          if ((pixel[(block << 6) + i] = base[i & 1] += diffbuf[i]) >> 10)
            derror();
        }
      }
      if (lowbits)
      {
        save = ftell(ifp);
        fseek(ifp, 26 + row * raw_width / 4, SEEK_SET);
        for (prow = pixel, i = 0; i < raw_width * 2; i++)
        {
          c = fgetc(ifp);
          for (r = 0; r < 8; r += 2, prow++)
          {
            val = (*prow << 2) + ((c >> r) & 3);
            if (raw_width == 2672 && val < 512)
              val += 2;
            *prow = val;
          }
        }
        fseek(ifp, save, SEEK_SET);
      }
    }
  }
  catch (...)
  {
    FORC(2) free(huff[c]);
    throw;
  }
  FORC(2) free(huff[c]);
}

int LibRaw::ljpeg_start(struct jhead *jh, int info_only)
{
  ushort c, tag, len;
  int cnt = 0;
  std::vector<uchar> data_buffer(0x10000);
  uchar* data = &data_buffer[0];
  const uchar *dp;

  memset(jh, 0, sizeof *jh);
  jh->restart = INT_MAX;
  if (fread(data, 2, 1, ifp) != 1 || data[1] != 0xd8)
    return 0;
  do
  {
    if (feof(ifp))
      return 0;
    if (cnt++ > 1024)
      return 0; // 1024 tags limit
    if (fread(data, 2, 2, ifp) != 2)
      return 0;
    tag = data[0] << 8 | data[1];
    len = (data[2] << 8 | data[3]) - 2;
    if (tag <= 0xff00)
      return 0;
    if (fread(data, 1, len, ifp) != len)
      return 0;
    switch (tag)
    {
    case 0xffc3: // start of frame; lossless, Huffman
      jh->sraw = ((data[7] >> 4) * (data[7] & 15) - 1) & 3;
    case 0xffc1:
    case 0xffc0:
      jh->algo = tag & 0xff;
      jh->bits = data[0];
      jh->high = data[1] << 8 | data[2];
      jh->wide = data[3] << 8 | data[4];
      jh->clrs = data[5] + jh->sraw;
      if (len == 9 && !dng_version)
        getc(ifp);
      break;
    case 0xffc4: // define Huffman tables
      if (info_only)
        break;
      for (dp = data; dp < data + len && !((c = *dp++) & -20);)
        jh->free[c] = jh->huff[c] = make_decoder_ref(&dp);
      break;
    case 0xffda: // start of scan
      jh->psv = data[1 + data[0] * 2];
      jh->bits -= data[3 + data[0] * 2] & 15;
      break;
    case 0xffdb:
      FORC(64) jh->quant[c] = data[c * 2 + 1] << 8 | data[c * 2 + 2];
      break;
    case 0xffdd:
      jh->restart = data[0] << 8 | data[1];
    }
  } while (tag != 0xffda);
  if (jh->bits > 16 || jh->clrs > 6 || !jh->bits || !jh->high || !jh->wide ||
      !jh->clrs)
    return 0;
  if (info_only)
    return 1;
  if (!jh->huff[0])
    return 0;
  FORC(19) if (!jh->huff[c + 1]) jh->huff[c + 1] = jh->huff[c];
  if (jh->sraw)
  {
    FORC(4) jh->huff[2 + c] = jh->huff[1];
    FORC(jh->sraw) jh->huff[1 + c] = jh->huff[0];
  }
  jh->row = (ushort *)calloc(jh->wide * jh->clrs, 16);
  return zero_after_ff = 1;
}

void LibRaw::ljpeg_end(struct jhead *jh)
{
  int c;
  FORC4 if (jh->free[c]) free(jh->free[c]);
  free(jh->row);
}

int LibRaw::ljpeg_diff(ushort *huff)
{
  int len, diff;
  if (!huff)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  len = gethuff(huff);
  if (len == 16 && (!dng_version || dng_version >= 0x1010000))
    return -32768;
  diff = getbits(len);


  if ((diff & (1 << (len - 1))) == 0)
    diff -= (1 << len) - 1;
  return diff;
}

ushort *LibRaw::ljpeg_row(int jrow, struct jhead *jh)
{
  int col, c, diff, pred, spred = 0;
  ushort mark = 0, *row[3];

  // Use the optimized, unrolled version if possible.
  if (!jh->sraw)
    return ljpeg_row_unrolled(jrow, jh);

  if (jh->restart != 0 && jrow * jh->wide % jh->restart == 0)
  {
    FORC(6) jh->vpred[c] = 1 << (jh->bits - 1);
    if (jrow)
    {
      fseek(ifp, -2, SEEK_CUR);
      do
        mark = (mark << 8) + (c = fgetc(ifp));
      while (c != EOF && mark >> 4 != 0xffd);
    }
    getbits(-1);
  }
  FORC3 row[c] = jh->row + jh->wide * jh->clrs * ((jrow + c) & 1);
  for (col = 0; col < jh->wide; col++)
    FORC(jh->clrs)
    {
      diff = ljpeg_diff(jh->huff[c]);
      if (jh->sraw && c <= jh->sraw && (col | c))
        pred = spred;
      else if (col)
        pred = row[0][-jh->clrs];
      else
        pred = (jh->vpred[c] += diff) - diff;
      if (jrow && col)
        switch (jh->psv)
        {
        case 1:
          break;
        case 2:
          pred = row[1][0];
          break;
        case 3:
          pred = row[1][-jh->clrs];
          break;
        case 4:
          pred = pred + row[1][0] - row[1][-jh->clrs];
          break;
        case 5:
          pred = pred + ((row[1][0] - row[1][-jh->clrs]) >> 1);
          break;
        case 6:
          pred = row[1][0] + ((pred - row[1][-jh->clrs]) >> 1);
          break;
        case 7:
          pred = (pred + row[1][0]) >> 1;
          break;
        default:
          pred = 0;
        }
      if ((**row = pred + diff) >> jh->bits)
		  if(!(load_flags & 512))
			derror();
      if (c <= jh->sraw)
        spred = **row;
      row[0]++;
      row[1]++;
    }
  return row[2];
}

ushort *LibRaw::ljpeg_row_unrolled(int jrow, struct jhead *jh)
{
  int col, c, diff, pred;
  ushort mark = 0, *row[3];

  if (jh->restart != 0 && jrow * jh->wide % jh->restart == 0)
  {
    FORC(6) jh->vpred[c] = 1 << (jh->bits - 1);
    if (jrow)
    {
      fseek(ifp, -2, SEEK_CUR);
      do
        mark = (mark << 8) + (c = fgetc(ifp));
      while (c != EOF && mark >> 4 != 0xffd);
    }
    getbits(-1);
  }
  FORC3 row[c] = jh->row + jh->wide * jh->clrs * ((jrow + c) & 1);

  // The first column uses one particular predictor.
  FORC(jh->clrs)
  {
    diff = ljpeg_diff(jh->huff[c]);
    pred = (jh->vpred[c] += diff) - diff;
    if ((**row = pred + diff) >> jh->bits)
      derror();
    row[0]++;
    row[1]++;
  }

  if (!jrow)
  {
    for (col = 1; col < jh->wide; col++)
      FORC(jh->clrs)
      {
        diff = ljpeg_diff(jh->huff[c]);
        pred = row[0][-jh->clrs];
        if ((**row = pred + diff) >> jh->bits)
          derror();
        row[0]++;
        row[1]++;
      }
  }
  else if (jh->psv == 1)
  {
    for (col = 1; col < jh->wide; col++)
      FORC(jh->clrs)
      {
        diff = ljpeg_diff(jh->huff[c]);
        pred = row[0][-jh->clrs];
        if ((**row = pred + diff) >> jh->bits)
          derror();
        row[0]++;
      }
  }
  else
  {
    for (col = 1; col < jh->wide; col++)
      FORC(jh->clrs)
      {
        diff = ljpeg_diff(jh->huff[c]);
        pred = row[0][-jh->clrs];
        switch (jh->psv)
        {
        case 1:
          break;
        case 2:
          pred = row[1][0];
          break;
        case 3:
          pred = row[1][-jh->clrs];
          break;
        case 4:
          pred = pred + row[1][0] - row[1][-jh->clrs];
          break;
        case 5:
          pred = pred + ((row[1][0] - row[1][-jh->clrs]) >> 1);
          break;
        case 6:
          pred = row[1][0] + ((pred - row[1][-jh->clrs]) >> 1);
          break;
        case 7:
          pred = (pred + row[1][0]) >> 1;
          break;
        default:
          pred = 0;
        }
        if ((**row = pred + diff) >> jh->bits)
          derror();
        row[0]++;
        row[1]++;
      }
  }
  return row[2];
}

void LibRaw::lossless_jpeg_load_raw()
{
  int jwide, jhigh, jrow, jcol, val, jidx, i, j, row = 0, col = 0;
  struct jhead jh;
  ushort *rp;

  if (!ljpeg_start(&jh, 0))
    return;

  if (jh.wide < 1 || jh.high < 1 || jh.clrs < 1 || jh.bits < 1)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  if(cr2_slice[0] && !cr2_slice[1])
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  jwide = jh.wide * jh.clrs;
  jhigh = jh.high;
  if (jh.clrs == 4 && jwide >= raw_width * 2)
    jhigh *= 2;

  try
  {
    for (jrow = 0; jrow < jh.high; jrow++)
    {
      checkCancel();
      rp = ljpeg_row(jrow, &jh);
      if (load_flags & 1)
        row = jrow & 1 ? height - 1 - jrow / 2 : jrow / 2;
      for (jcol = 0; jcol < jwide; jcol++)
      {
        val = curve[*rp++];
        if (cr2_slice[0])
        {
          jidx = jrow * jwide + jcol;
          i = jidx / (cr2_slice[1] * raw_height);
          if ((j = i >= cr2_slice[0]))
            i = cr2_slice[0];
          jidx -= i * (cr2_slice[1] * raw_height);
          row = jidx / cr2_slice[1 + j];
          col = jidx % cr2_slice[1 + j] + i * cr2_slice[1];
        }
        if (raw_width == 3984 && (col -= 2) < 0)
          col += (row--, raw_width);
        if (row > raw_height)
          throw LIBRAW_EXCEPTION_IO_CORRUPT;
        if ((unsigned)row < raw_height)
          RAW(row, col) = val;
        if (++col >= raw_width)
          col = (row++, 0);
      }
    }
  }
  catch (...)
  {
    ljpeg_end(&jh);
    throw;
  }
  ljpeg_end(&jh);
}

void LibRaw::canon_sraw_load_raw()
{
  struct jhead jh;
  short *rp = 0, (*ip)[4];
  int jwide, slice, scol, ecol, row, col, jrow = 0, jcol = 0, pix[3], c;
  int v[3] = {0, 0, 0}, ver, hue;
  int saved_w = width, saved_h = height;
  char *cp;

  if(!image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  if (!ljpeg_start(&jh, 0) || jh.clrs < 4)
    return;
  jwide = (jh.wide >>= 1) * jh.clrs;

  if (load_flags & 256)
  {
    width = raw_width;
    height = raw_height;
  }

  try
  {
    for (ecol = slice = 0; slice <= cr2_slice[0]; slice++)
    {
      scol = ecol;
      ecol += cr2_slice[1] * 2 / jh.clrs;
      if (!cr2_slice[0] || ecol > raw_width - 1)
        ecol = raw_width & -2;
      for (row = 0; row < height; row += (jh.clrs >> 1) - 1)
      {
        checkCancel();
        ip = (short(*)[4])image + row * width;
        for (col = scol; col < ecol; col += 2, jcol += jh.clrs)
        {
          if ((jcol %= jwide) == 0)
            rp = (short *)ljpeg_row(jrow++, &jh);
          if (col >= width)
            continue;
          if (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SRAW_NO_INTERPOLATE)
          {
            FORC(jh.clrs - 2)
            {
              ip[col + (c >> 1) * width + (c & 1)][0] = rp[jcol + c];
              ip[col + (c >> 1) * width + (c & 1)][1] =
                  ip[col + (c >> 1) * width + (c & 1)][2] = 8192;
            }
            ip[col][1] = rp[jcol + jh.clrs - 2] - 8192;
            ip[col][2] = rp[jcol + jh.clrs - 1] - 8192;
          }
          else if (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SRAW_NO_RGB)
          {
            FORC(jh.clrs - 2)
            ip[col + (c >> 1) * width + (c & 1)][0] = rp[jcol + c];
            ip[col][1] = rp[jcol + jh.clrs - 2] - 8192;
            ip[col][2] = rp[jcol + jh.clrs - 1] - 8192;
          }
          else
          {
            FORC(jh.clrs - 2)
            ip[col + (c >> 1) * width + (c & 1)][0] = rp[jcol + c];
            ip[col][1] = rp[jcol + jh.clrs - 2] - 16384;
            ip[col][2] = rp[jcol + jh.clrs - 1] - 16384;
          }
        }
      }
    }
  }
  catch (...)
  {
    ljpeg_end(&jh);
    throw;
  }

  if (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SRAW_NO_INTERPOLATE)
  {
    ljpeg_end(&jh);
    maximum = 0x3fff;
    height = saved_h;
    width = saved_w;
    return;
  }

  try
  {
    for (cp = model2; *cp && !isdigit(*cp); cp++)
      ;
    sscanf(cp, "%d.%d.%d", v, v + 1, v + 2);
    ver = (v[0] * 1000 + v[1]) * 1000 + v[2];
    hue = (jh.sraw + 1) << 2;
    if (unique_id >= 0x80000281ULL ||
        (unique_id == 0x80000218ULL && ver > 1000006))
      hue = jh.sraw << 1;
    ip = (short(*)[4])image;
    rp = ip[0];
    for (row = 0; row < height; row++, ip += width)
    {
      checkCancel();
      if (row & (jh.sraw >> 1))
      {
        for (col = 0; col < width; col += 2)
          for (c = 1; c < 3; c++)
            if (row == height - 1)
            {
              ip[col][c] = ip[col - width][c];
            }
            else
            {
              ip[col][c] = (ip[col - width][c] + ip[col + width][c] + 1) >> 1;
            }
      }
      for (col = 1; col < width; col += 2)
        for (c = 1; c < 3; c++)
          if (col == width - 1)
            ip[col][c] = ip[col - 1][c];
          else
            ip[col][c] = (ip[col - 1][c] + ip[col + 1][c] + 1) >> 1;
    }
    if (!(imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SRAW_NO_RGB))
      for (; rp < ip[0]; rp += 4)
      {
        checkCancel();
        if ((unique_id == CanonID_EOS_5D_Mark_II) ||
            (unique_id == CanonID_EOS_7D)         ||
            (unique_id == CanonID_EOS_50D)        ||
            (unique_id == CanonID_EOS_1D_Mark_IV) ||
            (unique_id == CanonID_EOS_60D))
        {
          rp[1] = (rp[1] << 2) + hue;
          rp[2] = (rp[2] << 2) + hue;
          pix[0] = rp[0] + ((50 * rp[1] + 22929 * rp[2]) >> 14);
          pix[1] = rp[0] + ((-5640 * rp[1] - 11751 * rp[2]) >> 14);
          pix[2] = rp[0] + ((29040 * rp[1] - 101 * rp[2]) >> 14);
        }
        else
        {
          if (unique_id < CanonID_EOS_5D_Mark_II)
            rp[0] -= 512;
          pix[0] = rp[0] + rp[2];
          pix[2] = rp[0] + rp[1];
          pix[1] = rp[0] + ((-778 * rp[1] - (rp[2] << 11)) >> 12);
        }
        FORC3 rp[c] = CLIP15(pix[c] * sraw_mul[c] >> 10);
      }
  }
  catch (...)
  {
    ljpeg_end(&jh);
    throw;
  }
  height = saved_h;
  width = saved_w;
  ljpeg_end(&jh);
  maximum = 0x3fff;
}

void LibRaw::ljpeg_idct(struct jhead *jh)
{
  int c, i, j, len, skip, coef;
  float work[3][8][8];
  static float cs[106] = {0};
  static const uchar zigzag[80] = {
      0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
      12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
      35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
      58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
      63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

  if (!cs[0])
    FORC(106) cs[c] = cos((c & 31) * M_PI / 16) / 2;
  memset(work, 0, sizeof work);
  work[0][0][0] = jh->vpred[0] += ljpeg_diff(jh->huff[0]) * jh->quant[0];
  for (i = 1; i < 64; i++)
  {
    len = gethuff(jh->huff[16]);
    i += skip = len >> 4;
    if (!(len &= 15) && skip < 15)
      break;
    coef = getbits(len);
    if ((coef & (1 << (len - 1))) == 0)
      coef -= (1 << len) - 1;
    ((float *)work)[zigzag[i]] = coef * jh->quant[i];
  }
  FORC(8) work[0][0][c] *= float(M_SQRT1_2);
  FORC(8) work[0][c][0] *= float(M_SQRT1_2);
  for (i = 0; i < 8; i++)
    for (j = 0; j < 8; j++)
      FORC(8) work[1][i][j] += work[0][i][c] * cs[(j * 2 + 1) * c];
  for (i = 0; i < 8; i++)
    for (j = 0; j < 8; j++)
      FORC(8) work[2][i][j] += work[1][c][j] * cs[(i * 2 + 1) * c];

  FORC(64) jh->idct[c] = CLIP(((float *)work[2])[c] + 0.5);
}

void LibRaw::pentax_load_raw()
{
  ushort bit[2][15], huff[4097];
  int dep, row, col, diff, c, i;
  ushort vpred[2][2] = {{0, 0}, {0, 0}}, hpred[2];

  fseek(ifp, meta_offset, SEEK_SET);
  dep = (get2() + 12) & 15;
  fseek(ifp, 12, SEEK_CUR);
  FORC(dep) bit[0][c] = get2();
  FORC(dep) bit[1][c] = fgetc(ifp);
  FORC(dep)
  for (i = bit[0][c]; i <= ((bit[0][c] + (4096 >> bit[1][c]) - 1) & 4095);)
    huff[++i] = bit[1][c] << 8 | c;
  huff[0] = 12;
  fseek(ifp, data_offset, SEEK_SET);
  getbits(-1);
  for (row = 0; row < raw_height; row++)
  {
    checkCancel();
    for (col = 0; col < raw_width; col++)
    {
      diff = ljpeg_diff(huff);
      if (col < 2)
        hpred[col] = vpred[row & 1][col] += diff;
      else
        hpred[col & 1] += diff;
      RAW(row, col) = hpred[col & 1];
      if (hpred[col & 1] >> tiff_bps)
        derror();
    }
  }
}
void LibRaw::nikon_read_curve()
{
  ushort ver0, ver1, vpred[2][2], csize;
  int i, step, max;

  fseek(ifp, meta_offset, SEEK_SET);
  ver0 = fgetc(ifp);
  ver1 = fgetc(ifp);
  if (ver0 == 0x49 || ver1 == 0x58)
    fseek(ifp, 2110, SEEK_CUR);
  read_shorts(vpred[0], 4);
  step = max = 1 << tiff_bps & 0x7fff;
  if ((csize = get2()) > 1)
    step = max / (csize - 1);
  if (ver0 == 0x44 && (ver1 == 0x20 || (ver1 == 0x40 && step > 3)) && step > 0)
  {
    if (ver1 == 0x40)
    {
      step /= 4;
      max /= 4;
    }
    for (i = 0; i < csize; i++)
      curve[i * step] = get2();
    for (i = 0; i < max; i++)
      curve[i] = (curve[i - i % step] * (step - i % step) +
                  curve[i - i % step + step] * (i % step)) /
                 step;
  }
  else if (ver0 != 0x46 && csize <= 0x4001)
    read_shorts(curve, max = csize);
}

void LibRaw::nikon_load_raw()
{
  static const uchar nikon_tree[][32] = {
      {0, 1, 5, 1, 1, 1, 1, 1, 1, 2, 0,  0,  0, 0, 0, 0, /* 12-bit lossy */
       5, 4, 3, 6, 2, 7, 1, 0, 8, 9, 11, 10, 12},
      {0,    1,    5,    1,    1,    1, 1, 1, 1, 2, 0, 0,  0,  0,
       0,    0, /* 12-bit lossy after split */
       0x39, 0x5a, 0x38, 0x27, 0x16, 5, 4, 3, 2, 1, 0, 11, 12, 12},

      {0, 1, 4, 2, 3, 1, 2, 0, 0, 0, 0,  0,  0, 0, 0, 0, /* 12-bit lossless */
       5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10, 11, 12},
      {0, 1, 4, 3, 1, 1, 1, 1, 1, 2, 0,  0,  0,  0,  0, 0, /* 14-bit lossy */
       5, 6, 4, 7, 8, 3, 9, 2, 1, 0, 10, 11, 12, 13, 14},
      {0, 1,    5,    1,    1,    1, 1, 1, 1, 1, 2, 0, 0, 0,  0,
       0, /* 14-bit lossy after split */
       8, 0x5c, 0x4b, 0x3a, 0x29, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14},
      {0, 1, 4, 2, 2, 3, 1,  2, 0,  0,  0, 0, 0, 0,  0, 0, /* 14-bit lossless */
       7, 6, 8, 5, 9, 4, 10, 3, 11, 12, 2, 0, 1, 13, 14}};
  ushort *huff, ver0, ver1, vpred[2][2], hpred[2];
  int i, min, max, tree = 0, split = 0, row, col, len, shl, diff;

  fseek(ifp, meta_offset, SEEK_SET);
  ver0 = fgetc(ifp);
  ver1 = fgetc(ifp);
  if (ver0 == 0x49 || ver1 == 0x58)
    fseek(ifp, 2110, SEEK_CUR);
  if (ver0 == 0x46)
    tree = 2;
  if (tiff_bps == 14)
    tree += 3;
  read_shorts(vpred[0], 4);
  max = 1 << tiff_bps & 0x7fff;
  if (ver0 == 0x44 && (ver1 == 0x20 || ver1 == 0x40))
  {
    if (ver1 == 0x40)
      max /= 4;
    fseek(ifp, meta_offset + 562, SEEK_SET);
    split = get2();
  }

  while (max > 2 && (curve[max - 2] == curve[max - 1]))
    max--;
  huff = make_decoder(nikon_tree[tree]);
  fseek(ifp, data_offset, SEEK_SET);
  getbits(-1);
  try
  {
    for (min = row = 0; row < height; row++)
    {
      checkCancel();
      if (split && row == split)
      {
        free(huff);
        huff = make_decoder(nikon_tree[tree + 1]);
        max += (min = 16) << 1;
      }
      for (col = 0; col < raw_width; col++)
      {
        i = gethuff(huff);
        len = i & 15;
        shl = i >> 4;
        diff = ((getbits(len - shl) << 1) + 1) << shl >> 1;
        if (len > 0 && (diff & (1 << (len - 1))) == 0)
          diff -= (1 << len) - !shl;
        if (col < 2)
          hpred[col] = vpred[row & 1][col] += diff;
        else
          hpred[col & 1] += diff;
        if ((ushort)(hpred[col & 1] + min) >= max)
          derror();
        RAW(row, col) = curve[LIM((short)hpred[col & 1], 0, 0x3fff)];
      }
    }
  }
  catch (...)
  {
    free(huff);
    throw;
  }
  free(huff);
}

void LibRaw::nikon_yuv_load_raw()
{
  if (!image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  int row, col, yuv[4]={0,0,0,0}, rgb[3], b, c;
  UINT64 bitbuf = 0;
  float cmul[4];
  FORC4 { cmul[c] = cam_mul[c] > 0.001f ? cam_mul[c] : 1.f; }
  for (row = 0; row < raw_height; row++)
  {
    checkCancel();

    for (col = 0; col < raw_width; col++)
    {
      if (!(b = col & 1))
      {
        bitbuf = 0;
        FORC(6) bitbuf |= (UINT64)fgetc(ifp) << c * 8;
        FORC(4) yuv[c] = (bitbuf >> c * 12 & 0xfff) - (c >> 1 << 11);
      }
      rgb[0] = yuv[b] + 1.370705 * yuv[3];
      rgb[1] = yuv[b] - 0.337633 * yuv[2] - 0.698001 * yuv[3];
      rgb[2] = yuv[b] + 1.732446 * yuv[2];
      FORC3 image[row * width + col][c] =
          curve[LIM(rgb[c], 0, 0xfff)] / cmul[c];
    }
  }
}

void LibRaw::rollei_load_raw()
{
  uchar pixel[10];
  unsigned iten = 0, isix, i, buffer = 0, todo[16];
  if (raw_width > 32767 || raw_height > 32767)
    throw LIBRAW_EXCEPTION_IO_BADFILE;
  unsigned maxpixel = raw_width * (raw_height + 7);

  isix = raw_width * raw_height * 5 / 8;
  while (fread(pixel, 1, 10, ifp) == 10)
  {
    checkCancel();
    for (i = 0; i < 10; i += 2)
    {
      todo[i] = iten++;
      todo[i + 1] = pixel[i] << 8 | pixel[i + 1];
      buffer = pixel[i] >> 2 | buffer << 6;
    }
    for (; i < 16; i += 2)
    {
      todo[i] = isix++;
      todo[i + 1] = buffer >> (14 - i) * 5;
    }
    for (i = 0; i < 16; i += 2)
      if (todo[i] < maxpixel)
        raw_image[todo[i]] = (todo[i + 1] & 0x3ff);
      else
        derror();
  }
  maximum = 0x3ff;
}

void LibRaw::nokia_load_raw()
{
  uchar *dp;
  int rev, dwide, row, col, c;
  double sum[] = {0, 0};

  rev = 3 * (order == 0x4949);
  dwide = (raw_width * 5 + 1) / 4;
#ifdef USE_6BY9RPI
  if (raw_stride)
	  dwide = raw_stride;
#endif
  std::vector<uchar> data(dwide * 2 + 4);
  for (row = 0; row < raw_height; row++)
  {
      checkCancel();
      if (fread(data.data() + dwide, 1, dwide, ifp) < dwide)
          derror();
      FORC(dwide) data[c] = data[dwide + (c ^ rev)];
      for (dp = data.data(), col = 0; col < raw_width; dp += 5, col += 4)
          FORC4 RAW(row, col + c) = (dp[c] << 2) | (dp[4] >> (c << 1) & 3);
  }
  maximum = 0x3ff;
#ifdef USE_6BY9RPI
  if (!strcmp(make, "OmniVision") ||
	  !strcmp(make, "Sony") ||
	  !strcmp(make, "RaspberryPi")) return;
#else
  if (strncmp(make, "OmniVision", 10))
	  return;
#endif
  row = raw_height / 2;
  FORC(width - 1)
  {
    sum[c & 1] += SQR(RAW(row, c) - RAW(row + 1, c + 1));
    sum[~c & 1] += SQR(RAW(row + 1, c) - RAW(row, c + 1));
  }
  if (sum[1] > sum[0])
    filters = 0x4b4b4b4b;
}

#ifdef LIBRAW_OLD_VIDEO_SUPPORT

void LibRaw::canon_rmf_load_raw()
{
  int row, col, bits, orow, ocol, c;

  int *words = (int *)malloc(sizeof(int) * (raw_width / 3 + 1));
  for (row = 0; row < raw_height; row++)
  {
    checkCancel();
    fread(words, sizeof(int), raw_width / 3, ifp);
    for (col = 0; col < raw_width - 2; col += 3)
    {
      bits = words[col / 3];
      FORC3
      {
        orow = row;
        if ((ocol = col + c - 4) < 0)
        {
          ocol += raw_width;
          if ((orow -= 2) < 0)
            orow += raw_height;
        }
        RAW(orow, ocol) = curve[bits >> (10 * c + 2) & 0x3ff];
      }
    }
  }
  free(words);
  maximum = curve[0x3ff];
}
#endif

unsigned LibRaw::pana_data(int nb, unsigned *bytes)
{
#ifndef LIBRAW_NOTHREADS
#define vpos tls->pana_data.vpos
#define buf tls->pana_data.buf
#else
  static uchar buf[0x4002];
  static int vpos;
#endif
  int byte;

  if (!nb && !bytes)
    return vpos = 0;

  if (!vpos)
  {
    fread(buf + load_flags, 1, 0x4000 - load_flags, ifp);
    fread(buf, 1, load_flags, ifp);
  }

  if (pana_encoding == 5)
  {
    for (byte = 0; byte < 16; byte++)
    {
      bytes[byte] = buf[vpos++];
      vpos &= 0x3FFF;
    }
  }
  else
  {
    vpos = (vpos - nb) & 0x1ffff;
    byte = vpos >> 3 ^ 0x3ff0;
    return (buf[byte] | buf[byte + 1] << 8) >> (vpos & 7) & ~((~0u) << nb);
  }
  return 0;
#ifndef LIBRAW_NOTHREADS
#undef vpos
#undef buf
#endif
}

void LibRaw::panasonic_load_raw()
{
  int row, col, i, j, sh = 0, pred[2], nonz[2];
  unsigned bytes[16];
  memset(bytes,0,sizeof(bytes)); // make gcc11 happy
  ushort *raw_block_data;

  pana_data(0, 0);

  int enc_blck_size = pana_bpp == 12 ? 10 : 9;
  if (pana_encoding == 5)
  {
    for (row = 0; row < raw_height; row++)
    {
      raw_block_data = raw_image + row * raw_width;
      checkCancel();
      for (col = 0; col < raw_width; col += enc_blck_size)
      {
        pana_data(0, bytes);

        if (pana_bpp == 12)
        {
          raw_block_data[col] = ((bytes[1] & 0xF) << 8) + bytes[0];
          raw_block_data[col + 1] = 16 * bytes[2] + (bytes[1] >> 4);
          raw_block_data[col + 2] = ((bytes[4] & 0xF) << 8) + bytes[3];
          raw_block_data[col + 3] = 16 * bytes[5] + (bytes[4] >> 4);
          raw_block_data[col + 4] = ((bytes[7] & 0xF) << 8) + bytes[6];
          raw_block_data[col + 5] = 16 * bytes[8] + (bytes[7] >> 4);
          raw_block_data[col + 6] = ((bytes[10] & 0xF) << 8) + bytes[9];
          raw_block_data[col + 7] = 16 * bytes[11] + (bytes[10] >> 4);
          raw_block_data[col + 8] = ((bytes[13] & 0xF) << 8) + bytes[12];
          raw_block_data[col + 9] = 16 * bytes[14] + (bytes[13] >> 4);
        }
        else if (pana_bpp == 14)
        {
          raw_block_data[col] = bytes[0] + ((bytes[1] & 0x3F) << 8);
          raw_block_data[col + 1] =
              (bytes[1] >> 6) + 4 * (bytes[2]) + ((bytes[3] & 0xF) << 10);
          raw_block_data[col + 2] =
              (bytes[3] >> 4) + 16 * (bytes[4]) + ((bytes[5] & 3) << 12);
          raw_block_data[col + 3] = ((bytes[5] & 0xFC) >> 2) + (bytes[6] << 6);
          raw_block_data[col + 4] = bytes[7] + ((bytes[8] & 0x3F) << 8);
          raw_block_data[col + 5] =
              (bytes[8] >> 6) + 4 * bytes[9] + ((bytes[10] & 0xF) << 10);
          raw_block_data[col + 6] =
              (bytes[10] >> 4) + 16 * bytes[11] + ((bytes[12] & 3) << 12);
          raw_block_data[col + 7] =
              ((bytes[12] & 0xFC) >> 2) + (bytes[13] << 6);
          raw_block_data[col + 8] = bytes[14] + ((bytes[15] & 0x3F) << 8);
        }
      }
    }
  }
  else
  {
    for (row = 0; row < raw_height; row++)
    {
      checkCancel();
      for (col = 0; col < raw_width; col++)
      {
        if ((i = col % 14) == 0)
          pred[0] = pred[1] = nonz[0] = nonz[1] = 0;
        if (i % 3 == 2)
          sh = 4 >> (3 - pana_data(2, 0));
        if (nonz[i & 1])
        {
          if ((j = pana_data(8, 0)))
          {
            if ((pred[i & 1] -= 0x80 << sh) < 0 || sh == 4)
              pred[i & 1] &= ~((~0u) << sh);
            pred[i & 1] += j << sh;
          }
        }
        else if ((nonz[i & 1] = pana_data(8, 0)) || i > 11)
          pred[i & 1] = nonz[i & 1] << 4 | pana_data(4, 0);
        if ((RAW(row, col) = pred[col & 1]) > 4098 && col < width &&
            row < height)
          derror();
      }
    }
  }
}

void LibRaw::olympus_load_raw()
{
  ushort huff[4096];
  int row, col, nbits, sign, low, high, i, c, w, n, nw;
  int acarry[2][3], *carry, pred, diff;

  huff[n = 0] = 0xc0c;
  for (i = 12; i--;)
    FORC(2048 >> i) huff[++n] = (i + 1) << 8 | i;
  fseek(ifp, 7, SEEK_CUR);
  getbits(-1);
  for (row = 0; row < height; row++)
  {
    checkCancel();
    memset(acarry, 0, sizeof acarry);
    for (col = 0; col < raw_width; col++)
    {
      carry = acarry[col & 1];
      i = 2 * (carry[2] < 3);
      for (nbits = 2 + i; (ushort)carry[0] >> (nbits + i); nbits++)
        ;
      low = (sign = getbits(3)) & 3;
      sign = sign << 29 >> 31;
      if ((high = getbithuff(12, huff)) == 12)
        high = getbits(16 - nbits) >> 1;
      carry[0] = (high << nbits) | getbits(nbits);
      diff = (carry[0] ^ sign) + carry[1];
      carry[1] = (diff * 3 + carry[1]) >> 5;
      carry[2] = carry[0] > 16 ? 0 : carry[2] + 1;
      if (col >= width)
        continue;
      if (row < 2 && col < 2)
        pred = 0;
      else if (row < 2)
        pred = RAW(row, col - 2);
      else if (col < 2)
        pred = RAW(row - 2, col);
      else
      {
        w = RAW(row, col - 2);
        n = RAW(row - 2, col);
        nw = RAW(row - 2, col - 2);
        if ((w < nw && nw < n) || (n < nw && nw < w))
        {
          if (ABS(w - nw) > 32 || ABS(n - nw) > 32)
            pred = w + n - nw;
          else
            pred = (w + n) >> 1;
        }
        else
          pred = ABS(w - nw) > ABS(n - nw) ? w : n;
      }
      if ((RAW(row, col) = pred + ((diff << 2) | low)) >> 12)
        derror();
    }
  }
}

void LibRaw::minolta_rd175_load_raw()
{
  uchar pixel[768];
  unsigned irow, box, row, col;

  for (irow = 0; irow < 1481; irow++)
  {
    checkCancel();
    if (fread(pixel, 1, 768, ifp) < 768)
      derror();
    box = irow / 82;
    row = irow % 82 * 12 + ((box < 12) ? box | 1 : (box - 12) * 2);
    switch (irow)
    {
    case 1477:
    case 1479:
      continue;
    case 1476:
      row = 984;
      break;
    case 1480:
      row = 985;
      break;
    case 1478:
      row = 985;
      box = 1;
    }
    if ((box < 12) && (box & 1))
    {
      for (col = 0; col < 1533; col++, row ^= 1)
        if (col != 1)
          RAW(row, col) = (col + 1) & 2
                              ? pixel[col / 2 - 1] + pixel[col / 2 + 1]
                              : pixel[col / 2] << 1;
      RAW(row, 1) = pixel[1] << 1;
      RAW(row, 1533) = pixel[765] << 1;
    }
    else
      for (col = row & 1; col < 1534; col += 2)
        RAW(row, col) = pixel[col / 2] << 1;
  }
  maximum = 0xff << 1;
}

void LibRaw::quicktake_100_load_raw()
{
  std::vector<uchar> pixel_buffer(484 * 644, 0x80);
  uchar* pixel = &pixel_buffer[0];
  static const short gstep[16] = {-89, -60, -44, -32, -22, -15, -8, -2,
                                  2,   8,   15,  22,  32,  44,  60, 89};
  static const short rstep[6][4] = {{-3, -1, 1, 3},   {-5, -1, 1, 5},
                                    {-8, -2, 2, 8},   {-13, -3, 3, 13},
                                    {-19, -4, 4, 19}, {-28, -6, 6, 28}};
  static const short t_curve[256] = {
      0,   1,    2,    3,   4,   5,   6,   7,   8,   9,   11,  12,  13,  14,
      15,  16,   17,   18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,
      29,  30,   32,   33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
      44,  45,   46,   47,  48,  49,  50,  51,  53,  54,  55,  56,  57,  58,
      59,  60,   61,   62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
      74,  75,   76,   77,  78,  79,  80,  81,  82,  83,  84,  86,  88,  90,
      92,  94,   97,   99,  101, 103, 105, 107, 110, 112, 114, 116, 118, 120,
      123, 125,  127,  129, 131, 134, 136, 138, 140, 142, 144, 147, 149, 151,
      153, 155,  158,  160, 162, 164, 166, 168, 171, 173, 175, 177, 179, 181,
      184, 186,  188,  190, 192, 195, 197, 199, 201, 203, 205, 208, 210, 212,
      214, 216,  218,  221, 223, 226, 230, 235, 239, 244, 248, 252, 257, 261,
      265, 270,  274,  278, 283, 287, 291, 296, 300, 305, 309, 313, 318, 322,
      326, 331,  335,  339, 344, 348, 352, 357, 361, 365, 370, 374, 379, 383,
      387, 392,  396,  400, 405, 409, 413, 418, 422, 426, 431, 435, 440, 444,
      448, 453,  457,  461, 466, 470, 474, 479, 483, 487, 492, 496, 500, 508,
      519, 531,  542,  553, 564, 575, 587, 598, 609, 620, 631, 643, 654, 665,
      676, 687,  698,  710, 721, 732, 743, 754, 766, 777, 788, 799, 810, 822,
      833, 844,  855,  866, 878, 889, 900, 911, 922, 933, 945, 956, 967, 978,
      989, 1001, 1012, 1023};
  int rb, row, col, sharp, val = 0;

  if (width > 640 || height > 480)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  getbits(-1);
  for (row = 2; row < height + 2; row++)
  {
    checkCancel();
    for (col = 2 + (row & 1); col < width + 2; col += 2)
    {
      val = ((pixel[(row - 1) * 644 + col - 1] + 2 * pixel[(row - 1) * 644 + col + 1] + pixel[row * 644 + col - 2]) >> 2) + gstep[getbits(4)];
      pixel[row * 644 + col] = val = LIM(val, 0, 255);
      if (col < 4)
        pixel[row * 644 + col - 2] = pixel[(row + 1) * 644 + (~row & 1)] = val;
      if (row == 2)
        pixel[(row - 1) * 644 + col + 1] = pixel[(row - 1) * 644 + col + 3] = val;
    }
    pixel[row * 644 + col] = val;
  }
  for (rb = 0; rb < 2; rb++)
    for (row = 2 + rb; row < height + 2; row += 2)
    {
      checkCancel();
      for (col = 3 - (row & 1); col < width + 2; col += 2)
      {
        if (row < 4 || col < 4)
          sharp = 2;
        else
        {
          val = ABS(pixel[(row - 2) * 644 + col] - pixel[row * 644 + col - 2]) + ABS(pixel[(row - 2) * 644 + col] - pixel[(row - 2) * 644 + col - 2]) +
                ABS(pixel[row * 644 + col - 2] - pixel[(row - 2) * 644 + col - 2]);
          sharp = val < 4
                      ? 0
                      : val < 8
                            ? 1
                            : val < 16 ? 2 : val < 32 ? 3 : val < 48 ? 4 : 5;
        }
        val = ((pixel[(row - 2) * 644 + col] + pixel[row * 644 + col - 2]) >> 1) + rstep[sharp][getbits(2)];
        pixel[row * 644 + col] = val = LIM(val, 0, 255);
        if (row < 4)
          pixel[(row - 2) * 644 + col + 2] = val;
        if (col < 4)
          pixel[(row + 2) * 644 + col - 2] = val;
      }
    }
  for (row = 2; row < height + 2; row++)
  {
    checkCancel();
    for (col = 3 - (row & 1); col < width + 2; col += 2)
    {
      val = ((pixel[row * 644 + col - 1] + (pixel[row * 644 + col] << 2) + pixel[row * 644 + col + 1]) >> 1) - 0x100;
      pixel[row * 644 + col] = LIM(val, 0, 255);
    }
  }
  for (row = 0; row < height; row++)
  {
    checkCancel();
    for (col = 0; col < width; col++)
      RAW(row, col) = t_curve[pixel[(row + 2) * 644 + col + 2]];
  }
  maximum = 0x3ff;
}

void LibRaw::sony_load_raw()
{
  uchar head[40];
  ushort *pixel;
  unsigned i, key, row, col;

  fseek(ifp, 200896, SEEK_SET);
  fseek(ifp, (unsigned)fgetc(ifp) * 4 - 1, SEEK_CUR);
  order = 0x4d4d;
  key = get4();

  fseek(ifp, 164600, SEEK_SET);
  fread(head, 1, 40, ifp);
  sony_decrypt((unsigned *)head, 10, 1, key);
  for (i = 26; i-- > 22;)
    key = key << 8 | head[i];

  fseek(ifp, data_offset, SEEK_SET);
  for (row = 0; row < raw_height; row++)
  {
    checkCancel();
    pixel = raw_image + row * raw_width;
    if (fread(pixel, 2, raw_width, ifp) < raw_width)
      derror();
    sony_decrypt((unsigned *)pixel, raw_width / 2, !row, key);
    for (col = 0; col < raw_width; col++)
      if ((pixel[col] = ntohs(pixel[col])) >> 14)
        derror();
  }
  maximum = 0x3ff0;
}

void LibRaw::sony_arw_load_raw()
{
  std::vector<ushort> huff_buffer(32770);
  ushort* huff = &huff_buffer[0];
  static const ushort tab[18] = {0xf11, 0xf10, 0xe0f, 0xd0e, 0xc0d, 0xb0c,
                                 0xa0b, 0x90a, 0x809, 0x708, 0x607, 0x506,
                                 0x405, 0x304, 0x303, 0x300, 0x202, 0x201};
  int i, c, n, col, row, sum = 0;

  huff[0] = 15;
  for (n = i = 0; i < 18; i++)
    FORC(32768 >> (tab[i] >> 8)) huff[++n] = tab[i];
  getbits(-1);
  for (col = raw_width; col--;)
  {
    checkCancel();
    for (row = 0; row < raw_height + 1; row += 2)
    {
      if (row == raw_height)
        row = 1;
      if ((sum += ljpeg_diff(huff)) >> 12)
        derror();
      if (row < height)
        RAW(row, col) = sum;
    }
  }
}

void LibRaw::sony_arw2_load_raw()
{
  uchar *data, *dp;
  ushort pix[16];
  int row, col, val, max, min, imax, imin, sh, bit, i;

  data = (uchar *)malloc(raw_width + 1);
  try
  {
    for (row = 0; row < height; row++)
    {
      checkCancel();
      fread(data, 1, raw_width, ifp);
      for (dp = data, col = 0; col < raw_width - 30; dp += 16)
      {
        max = 0x7ff & (val = sget4(dp));
        min = 0x7ff & val >> 11;
        imax = 0x0f & val >> 22;
        imin = 0x0f & val >> 26;
        for (sh = 0; sh < 4 && 0x80 << sh <= max - min; sh++)
          ;
        /* flag checks if outside of loop */
        if (!(imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SONYARW2_ALLFLAGS) // no flag set
            || (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SONYARW2_DELTATOVALUE))
        {
          for (bit = 30, i = 0; i < 16; i++)
            if (i == imax)
              pix[i] = max;
            else if (i == imin)
              pix[i] = min;
            else
            {
              pix[i] =
                  ((sget2(dp + (bit >> 3)) >> (bit & 7) & 0x7f) << sh) + min;
              if (pix[i] > 0x7ff)
                pix[i] = 0x7ff;
              bit += 7;
            }
        }
        else if (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SONYARW2_BASEONLY)
        {
          for (bit = 30, i = 0; i < 16; i++)
            if (i == imax)
              pix[i] = max;
            else if (i == imin)
              pix[i] = min;
            else
              pix[i] = 0;
        }
        else if (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SONYARW2_DELTAONLY)
        {
          for (bit = 30, i = 0; i < 16; i++)
            if (i == imax)
              pix[i] = 0;
            else if (i == imin)
              pix[i] = 0;
            else
            {
              pix[i] =
                  ((sget2(dp + (bit >> 3)) >> (bit & 7) & 0x7f) << sh) + min;
              if (pix[i] > 0x7ff)
                pix[i] = 0x7ff;
              bit += 7;
            }
        }
        else if (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SONYARW2_DELTAZEROBASE)
        {
          for (bit = 30, i = 0; i < 16; i++)
            if (i == imax)
              pix[i] = 0;
            else if (i == imin)
              pix[i] = 0;
            else
            {
              pix[i] = ((sget2(dp + (bit >> 3)) >> (bit & 7) & 0x7f) << sh);
              if (pix[i] > 0x7ff)
                pix[i] = 0x7ff;
              bit += 7;
            }
        }

        if (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SONYARW2_DELTATOVALUE)
        {
          for (i = 0; i < 16; i++, col += 2)
          {
            unsigned slope =
                pix[i] < 1001 ? 2
                              : curve[pix[i] << 1] - curve[(pix[i] << 1) - 2];
            unsigned step = 1 << sh;
            RAW(row, col) =
                curve[pix[i] << 1] >
                        black + imgdata.rawparams.sony_arw2_posterization_thr
                    ? LIM(((slope * step * 1000) /
                           (curve[pix[i] << 1] - black)),
                          0, 10000)
                    : 0;
          }
        }
        else
          for (i = 0; i < 16; i++, col += 2)
            RAW(row, col) = curve[pix[i] << 1];
        col -= col & 1 ? 1 : 31;
      }
    }
  }
  catch (...)
  {
    free(data);
    throw;
  }
  if (imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_SONYARW2_DELTATOVALUE)
    maximum = 10000;
  free(data);
}

void LibRaw::samsung_load_raw()
{
  int row, col, c, i, dir, op[4], len[4];
  if (raw_width > 32768 ||
      raw_height > 32768) // definitely too much for old samsung
    throw LIBRAW_EXCEPTION_IO_BADFILE;
  unsigned maxpixels = raw_width * (raw_height + 7);

  order = 0x4949;
  for (row = 0; row < raw_height; row++)
  {
    checkCancel();
    fseek(ifp, strip_offset + row * 4, SEEK_SET);
    fseek(ifp, data_offset + get4(), SEEK_SET);
    ph1_bits(-1);
    FORC4 len[c] = row < 2 ? 7 : 4;
    for (col = 0; col < raw_width; col += 16)
    {
      dir = ph1_bits(1);
      FORC4 op[c] = ph1_bits(2);
      FORC4 switch (op[c])
      {
      case 3:
        len[c] = ph1_bits(4);
        break;
      case 2:
        len[c]--;
        break;
      case 1:
        len[c]++;
      }
      for (c = 0; c < 16; c += 2)
      {
        i = len[((c & 1) << 1) | (c >> 3)];
        unsigned idest = RAWINDEX(row, col + c);
        unsigned isrc = (dir ? RAWINDEX(row + (~c | -2), col + c)
                             : col ? RAWINDEX(row, col + (c | -2)) : 0);
        if (idest < maxpixels &&
            isrc <
                maxpixels) // less than zero is handled by unsigned conversion
          RAW(row, col + c) = (i > 0 ? ((signed)ph1_bits(i) << (32 - i) >> (32 - i)) : 0) +
            (dir ? RAW(row + (~c | -2), col + c) : col ? RAW(row, col + (c | -2)) : 128);
        else
          derror();
        if (c == 14)
          c = -1;
      }
    }
  }
  for (row = 0; row < raw_height - 1; row += 2)
    for (col = 0; col < raw_width - 1; col += 2)
      SWAP(RAW(row, col + 1), RAW(row + 1, col));
}

void LibRaw::samsung2_load_raw()
{
  static const ushort tab[14] = {0x304, 0x307, 0x206, 0x205, 0x403,
                                 0x600, 0x709, 0x80a, 0x90b, 0xa0c,
                                 0xa0d, 0x501, 0x408, 0x402};
  ushort huff[1026], vpred[2][2] = {{0, 0}, {0, 0}}, hpred[2];
  int i, c, n, row, col, diff;

  huff[0] = 10;
  for (n = i = 0; i < 14; i++)
    FORC(1024 >> (tab[i] >> 8)) huff[++n] = tab[i];
  getbits(-1);
  for (row = 0; row < raw_height; row++)
  {
    checkCancel();
    for (col = 0; col < raw_width; col++)
    {
      diff = ljpeg_diff(huff);
      if (col < 2)
        hpred[col] = vpred[row & 1][col] += diff;
      else
        hpred[col & 1] += diff;
      RAW(row, col) = hpred[col & 1];
      if (hpred[col & 1] >> tiff_bps)
        derror();
    }
  }
}

void LibRaw::samsung3_load_raw()
{
  int opt, init, mag, pmode, row, tab, col, pred, diff, i, c;
  ushort lent[3][2], len[4], *prow[2];
  order = 0x4949;
  fseek(ifp, 9, SEEK_CUR);
  opt = fgetc(ifp);
  init = (get2(), get2());
  for (row = 0; row < raw_height; row++)
  {
    checkCancel();
    fseek(ifp, (data_offset - ftell(ifp)) & 15, SEEK_CUR);
    ph1_bits(-1);
    mag = 0;
    pmode = 7;
    FORC(6)((ushort *)lent)[c] = row < 2 ? 7 : 4;
    prow[row & 1] = &RAW(row - 1, 1 - ((row & 1) << 1)); // green
    prow[~row & 1] = &RAW(row - 2, 0);                   // red and blue
    for (tab = 0; tab + 15 < raw_width; tab += 16)
    {
      if (~opt & 4 && !(tab & 63))
      {
        i = ph1_bits(2);
        mag = i < 3 ? mag - '2' + "204"[i] : ph1_bits(12);
      }
      if (opt & 2)
        pmode = 7 - 4 * ph1_bits(1);
      else if (!ph1_bits(1))
        pmode = ph1_bits(3);
      if (opt & 1 || !ph1_bits(1))
      {
        FORC4 len[c] = ph1_bits(2);
        FORC4
        {
          i = ((row & 1) << 1 | (c & 1)) % 3;
          if (i < 0)
            throw LIBRAW_EXCEPTION_IO_CORRUPT;
          len[c] = len[c] < 3 ? lent[i][0] - '1' + "120"[len[c]] : ph1_bits(4);
          lent[i][0] = lent[i][1];
          lent[i][1] = len[c];
        }
      }
      FORC(16)
      {
        col = tab + (((c & 7) << 1) ^ (c >> 3) ^ (row & 1));
        if (col < 0)
          throw LIBRAW_EXCEPTION_IO_CORRUPT;
        if (pmode < 0)
          throw LIBRAW_EXCEPTION_IO_CORRUPT;
        if (pmode != 7 && row >= 2 && (col - '4' + "0224468"[pmode]) < 0)
          throw LIBRAW_EXCEPTION_IO_CORRUPT;
        pred = (pmode == 7 || row < 2)
                   ? (tab ? RAW(row, tab - 2 + (col & 1)) : init)
                   : (prow[col & 1][col - '4' + "0224468"[pmode]] +
                      prow[col & 1][col - '4' + "0244668"[pmode]] + 1) >>
                         1;
        diff = ph1_bits(i = len[c >> 2]);
        if (i > 0 && diff >> (i - 1))
          diff -= 1 << i;
        diff = diff * (mag * 2 + 1) + mag;
        RAW(row, col) = pred + diff;
      }
    }
  }
}

#ifdef LIBRAW_OLD_VIDEO_SUPPORT
void LibRaw::redcine_load_raw()
{
#ifndef NO_JASPER
  int c, row, col;
  jas_stream_t *in;
  jas_image_t *jimg;
  jas_matrix_t *jmat;
  jas_seqent_t *data;
  ushort *img, *pix;

  jas_init();
  in = (jas_stream_t *)ifp->make_jas_stream();
  if (!in)
    throw LIBRAW_EXCEPTION_DECODE_JPEG2000;
  jas_stream_seek(in, data_offset + 20, SEEK_SET);
  jimg = jas_image_decode(in, -1, 0);
  if (!jimg)
  {
    jas_stream_close(in);
    throw LIBRAW_EXCEPTION_DECODE_JPEG2000;
  }
  jmat = jas_matrix_create(height / 2, width / 2);
  img = (ushort *)calloc((height + 2), (width + 2) * 2);
  bool fastexitflag = false;
  try
  {
    FORC4
    {
      checkCancel();
      jas_image_readcmpt(jimg, c, 0, 0, width / 2, height / 2, jmat);
      data = jas_matrix_getref(jmat, 0, 0);
      for (row = c >> 1; row < height; row += 2)
        for (col = c & 1; col < width; col += 2)
          img[(row + 1) * (width + 2) + col + 1] =
              data[(row / 2) * (width / 2) + col / 2];
    }
    for (col = 1; col <= width; col++)
    {
      img[col] = img[2 * (width + 2) + col];
      img[(height + 1) * (width + 2) + col] =
          img[(height - 1) * (width + 2) + col];
    }
    for (row = 0; row < height + 2; row++)
    {
      img[row * (width + 2)] = img[row * (width + 2) + 2];
      img[(row + 1) * (width + 2) - 1] = img[(row + 1) * (width + 2) - 3];
    }
    for (row = 1; row <= height; row++)
    {
      checkCancel();
      pix = img + row * (width + 2) + (col = 1 + (FC(row, 1) & 1));
      for (; col <= width; col += 2, pix += 2)
      {
        c = (((pix[0] - 0x800) << 3) + pix[-(width + 2)] + pix[width + 2] +
             pix[-1] + pix[1]) >>
            2;
        pix[0] = LIM(c, 0, 4095);
      }
    }
    for (row = 0; row < height; row++)
    {
      checkCancel();
      for (col = 0; col < width; col++)
        RAW(row, col) = curve[img[(row + 1) * (width + 2) + col + 1]];
    }
  }
  catch (...)
  {
    fastexitflag = true;
  }
  free(img);
  jas_matrix_destroy(jmat);
  jas_image_destroy(jimg);
  jas_stream_close(in);
  if (fastexitflag)
    throw LIBRAW_EXCEPTION_CANCELLED_BY_CALLBACK;
#endif
}
#endif
