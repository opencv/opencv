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

#define radc_token(tree) ((signed char)getbithuff(8, huff + (tree) * 256))

#define FORYX                                                                  \
  for (y = 1; y < 3; y++)                                                      \
    for (x = col + 1; x >= col; x--)

#define PREDICTOR                                                              \
  (c ? (buf[c][y - 1][x] + buf[c][y][x + 1]) / 2                               \
     : (buf[c][y - 1][x + 1] + 2 * buf[c][y - 1][x] + buf[c][y][x + 1]) / 4)

#ifdef __GNUC__
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
#pragma GCC optimize("no-aggressive-loop-optimizations")
#endif
#endif

void LibRaw::kodak_radc_load_raw()
{
  // All kodak radc images are 768x512
  if (width > 768 || raw_width > 768 || height > 512 || raw_height > 512)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  static const signed char src[] = {
      1, 1,   2, 3,   3, 4,   4, 2,   5, 7,   6, 5,   7, 6,   7, 8,   1, 0,
      2, 1,   3, 3,   4, 4,   5, 2,   6, 7,   7, 6,   8, 5,   8, 8,   2, 1,
      2, 3,   3, 0,   3, 2,   3, 4,   4, 6,   5, 5,   6, 7,   6, 8,   2, 0,
      2, 1,   2, 3,   3, 2,   4, 4,   5, 6,   6, 7,   7, 5,   7, 8,   2, 1,
      2, 4,   3, 0,   3, 2,   3, 3,   4, 7,   5, 5,   6, 6,   6, 8,   2, 3,
      3, 1,   3, 2,   3, 4,   3, 5,   3, 6,   4, 7,   5, 0,   5, 8,   2, 3,
      2, 6,   3, 0,   3, 1,   4, 4,   4, 5,   4, 7,   5, 2,   5, 8,   2, 4,
      2, 7,   3, 3,   3, 6,   4, 1,   4, 2,   4, 5,   5, 0,   5, 8,   2, 6,
      3, 1,   3, 3,   3, 5,   3, 7,   3, 8,   4, 0,   5, 2,   5, 4,   2, 0,
      2, 1,   3, 2,   3, 3,   4, 4,   4, 5,   5, 6,   5, 7,   4, 8,   1, 0,
      2, 2,   2, -2,  1, -3,  1, 3,   2, -17, 2, -5,  2, 5,   2, 17,  2, -7,
      2, 2,   2, 9,   2, 18,  2, -18, 2, -9,  2, -2,  2, 7,   2, -28, 2, 28,
      3, -49, 3, -9,  3, 9,   4, 49,  5, -79, 5, 79,  2, -1,  2, 13,  2, 26,
      3, 39,  4, -16, 5, 55,  6, -37, 6, 76,  2, -26, 2, -13, 2, 1,   3, -39,
      4, 16,  5, -55, 6, -76, 6, 37};
  std::vector<ushort> huff_buffer(19 * 256);
  ushort* huff = &huff_buffer[0];
  int row, col, tree, nreps, rep, step, i, c, s, r, x, y, val;
  short last[3] = {16, 16, 16}, mul[3], buf[3][3][386];
  static const ushort pt[] = {0,    0,    1280, 1344,  2320,  3616,
                              3328, 8000, 4095, 16383, 65535, 16383};

  for (i = 2; i < 12; i += 2)
    for (c = pt[i - 2]; c <= pt[i]; c++)
      curve[c] = (float)(c - pt[i - 2]) / (pt[i] - pt[i - 2]) *
                     (pt[i + 1] - pt[i - 1]) +
                 pt[i - 1] + 0.5;
  for (s = i = 0; i < int(sizeof src); i += 2)
    FORC(256 >> src[i])
  ((ushort *)huff)[s++] = src[i] << 8 | (uchar)src[i + 1];
  s = kodak_cbpp == 243 ? 2 : 3;
  FORC(256) huff[18 * 256 + c] = (8 - s) << 8 | c >> s << s | 1 << (s - 1);
  getbits(-1);
  for (i = 0; i < int(sizeof(buf) / sizeof(short)); i++)
    ((short *)buf)[i] = 2048;
  for (row = 0; row < height; row += 4)
  {
    checkCancel();
    FORC3 mul[c] = getbits(6);
    if (!mul[0] || !mul[1] || !mul[2])
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    FORC3
    {
      val = ((0x1000000 / last[c] + 0x7ff) >> 12) * mul[c];
      s = val > 65564 ? 10 : 12;
      x = ~((~0u) << (s - 1));
      val <<= 12 - s;
      for (i = 0; i < int(sizeof(buf[0]) / sizeof(short)); i++)
        ((short *)buf[c])[i] = MIN(0x7FFFFFFF, (((short *)buf[c])[i] * static_cast<long long>(val) + x)) >> s;
      last[c] = mul[c];
      for (r = 0; r <= int(!c); r++)
      {
        buf[c][1][width / 2] = buf[c][2][width / 2] = mul[c] << 7;
        for (tree = 1, col = width / 2; col > 0;)
        {
          if ((tree = radc_token(tree)))
          {
            col -= 2;
            if (col >= 0)
            {
              if (tree == 8)
                FORYX buf[c][y][x] = (uchar)radc_token(18) * mul[c];
              else
                FORYX buf[c][y][x] = radc_token(tree + 10) * 16 + PREDICTOR;
            }
          }
          else
            do
            {
              nreps = (col > 2) ? radc_token(9) + 1 : 1;
              for (rep = 0; rep < 8 && rep < nreps && col > 0; rep++)
              {
                col -= 2;
                if (col >= 0)
                  FORYX buf[c][y][x] = PREDICTOR;
                if (rep & 1)
                {
                  step = radc_token(10) << 4;
                  FORYX buf[c][y][x] += step;
                }
              }
            } while (nreps == 9);
        }
        for (y = 0; y < 2; y++)
          for (x = 0; x < width / 2; x++)
          {
            val = (buf[c][y + 1][x] << 4) / mul[c];
            if (val < 0)
              val = 0;
            if (c)
              RAW(row + y * 2 + c - 1, x * 2 + 2 - c) = val;
            else
              RAW(row + r * 2 + y, x * 2 + y) = val;
          }
        memcpy(buf[c][0] + !c, buf[c][2], sizeof buf[c][0] - 2 * !c);
      }
    }
    for (y = row; y < row + 4; y++)
      for (x = 0; x < width; x++)
        if ((x + y) & 1)
        {
          r = x ? x - 1 : x + 1;
          s = x + 1 < width ? x + 1 : x - 1;
          val = (RAW(y, x) - 2048) * 2 + (RAW(y, r) + RAW(y, s)) / 2;
          if (val < 0)
            val = 0;
          RAW(y, x) = val;
        }
  }
  for (i = 0; i < height * width; i++)
    raw_image[i] = curve[raw_image[i]];
  maximum = 0x3fff;
}

#undef FORYX
#undef PREDICTOR

#ifdef NO_JPEG
void LibRaw::kodak_jpeg_load_raw() {}
#else
static void jpegErrorExit_k(j_common_ptr /*cinfo*/)
{
  throw LIBRAW_EXCEPTION_DECODE_JPEG;
}

// LibRaw's Kodak_jpeg_load_raw
void LibRaw::kodak_jpeg_load_raw()
{
  if (data_size < 1)
    throw LIBRAW_EXCEPTION_DECODE_JPEG;

  int row, col;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr pub;
  cinfo.err = jpeg_std_error(&pub);
  pub.error_exit = jpegErrorExit_k;

  if (INT64(data_size) >
          INT64(imgdata.rawparams.max_raw_memory_mb) * INT64(1024 * 1024))
	  throw LIBRAW_EXCEPTION_TOOBIG;

  unsigned char *jpg_buf = (unsigned char *)malloc(data_size);
  std::vector<uchar> pixel_buf(width * 3);
  jpeg_create_decompress(&cinfo);

  fread(jpg_buf, data_size, 1, ifp);
  libraw_swab(jpg_buf, data_size);
  try
  {
    jpeg_mem_src(&cinfo, jpg_buf, data_size);
    int rc = jpeg_read_header(&cinfo, TRUE);
    if (rc != 1)
      throw LIBRAW_EXCEPTION_DECODE_JPEG;

    jpeg_start_decompress(&cinfo);
    if ((cinfo.output_width != width) || (cinfo.output_height * 2 != height) ||
        (cinfo.output_components != 3))
    {
      throw LIBRAW_EXCEPTION_DECODE_JPEG;
    }

    unsigned char *buf[1];
    buf[0] = pixel_buf.data();

    while (cinfo.output_scanline < cinfo.output_height)
    {
      checkCancel();
      row = cinfo.output_scanline * 2;
      jpeg_read_scanlines(&cinfo, buf, 1);
      unsigned char(*pixel)[3] = (unsigned char(*)[3])buf[0];
      for (col = 0; col < width; col += 2)
      {
        RAW(row + 0, col + 0) = pixel[col + 0][1] << 1;
        RAW(row + 1, col + 1) = pixel[col + 1][1] << 1;
        RAW(row + 0, col + 1) = pixel[col][0] + pixel[col + 1][0];
        RAW(row + 1, col + 0) = pixel[col][2] + pixel[col + 1][2];
      }
    }
  }
  catch (...)
  {
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    free(jpg_buf);
    throw;
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  free(jpg_buf);
  maximum = 0xff << 1;
}
#endif

void LibRaw::kodak_dc120_load_raw()
{
  static const int mul[4] = {162, 192, 187, 92};
  static const int add[4] = {0, 636, 424, 212};
  uchar pixel[848];
  int row, shift, col;

  for (row = 0; row < height; row++)
  {
    checkCancel();
    if (fread(pixel, 1, 848, ifp) < 848)
      derror();
    shift = row * mul[row & 3] + add[row & 3];
    for (col = 0; col < width; col++)
      RAW(row, col) = (ushort)pixel[(col + shift) % 848];
  }
  maximum = 0xff;
}
void LibRaw::kodak_c330_load_raw()
{
  if (!image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  int row, col, y, cb, cr, rgb[3], c;

  std::vector<uchar> pixel(raw_width*2 + 4);

  for (row = 0; row < height; row++)
  {
      checkCancel();
      if (fread(pixel.data(), raw_width, 2, ifp) < 2)
          derror();
      if (load_flags && (row & 31) == 31)
          fseek(ifp, raw_width * 32, SEEK_CUR);
      for (col = 0; col < width; col++)
      {
          y = pixel[col * 2];
          cb = pixel[(col * 2 & -4) | 1] - 128;
          cr = pixel[(col * 2 & -4) | 3] - 128;
          rgb[1] = y - ((cb + cr + 2) >> 2);
          rgb[2] = rgb[1] + cb;
          rgb[0] = rgb[1] + cr;
          FORC3 image[row * width + col][c] = curve[LIM(rgb[c], 0, 255)];
      }
  }
  maximum = curve[0xff];
}

void LibRaw::kodak_c603_load_raw()
{
  if (!image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  int row, col, y, cb, cr, rgb[3], c;

  std::vector<uchar> pixel(raw_width * 3);
  for (row = 0; row < height; row++)
  {
      checkCancel();
      if (~row & 1)
          if (fread(pixel.data(), raw_width, 3, ifp) < 3)
              derror();
      for (col = 0; col < width; col++)
      {
          y = pixel[width * 2 * (row & 1) + col];
          cb = pixel[width + (col & -2)] - 128;
          cr = pixel[width + (col & -2) + 1] - 128;
          rgb[1] = y - ((cb + cr + 2) >> 2);
          rgb[2] = rgb[1] + cb;
          rgb[0] = rgb[1] + cr;
          FORC3 image[row * width + col][c] = curve[LIM(rgb[c], 0, 255)];
      }
  }
  maximum = curve[0xff];
}

void LibRaw::kodak_262_load_raw()
{
  static const uchar kodak_tree[2][26] = {
      {0, 1, 5, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {0, 3, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
  ushort *huff[2];
  int *strip, ns, c, row, col, chess, pi = 0, pi1, pi2, pred, val;

  FORC(2) huff[c] = make_decoder(kodak_tree[c]);
  ns = (raw_height + 63) >> 5;
  std::vector<uchar> pixel(raw_width * 32 + ns * 4);
  strip = (int *)(pixel.data() + raw_width * 32);
  order = 0x4d4d;
  FORC(ns) strip[c] = get4();
  try
  {
    for (row = 0; row < raw_height; row++)
    {
      checkCancel();
      if ((row & 31) == 0)
      {
        fseek(ifp, strip[row >> 5], SEEK_SET);
        getbits(-1);
        pi = 0;
      }
      for (col = 0; col < raw_width; col++)
      {
        chess = (row + col) & 1;
        pi1 = chess ? pi - 2 : pi - raw_width - 1;
        pi2 = chess ? pi - 2 * raw_width : pi - raw_width + 1;
        if (col <= chess)
          pi1 = -1;
        if (pi1 < 0)
          pi1 = pi2;
        if (pi2 < 0)
          pi2 = pi1;
        if (pi1 < 0 && col > 1)
          pi1 = pi2 = pi - 2;
        pred = (pi1 < 0) ? 0 : (pixel[pi1] + pixel[pi2]) >> 1;
        pixel[pi] = val = pred + ljpeg_diff(huff[chess]);
        if (val >> 8)
          derror();
        val = curve[pixel[pi++]];
        RAW(row, col) = val;
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

int LibRaw::kodak_65000_decode(short *out, int bsize)
{
  uchar c, blen[768];
  ushort raw[6];
  INT64 bitbuf = 0;
  int save, bits = 0, i, j, len, diff;

  save = ftell(ifp);
  bsize = (bsize + 3) & -4;
  for (i = 0; i < bsize; i += 2)
  {
    c = fgetc(ifp);
    if ((blen[i] = c & 15) > 12 || (blen[i + 1] = c >> 4) > 12)
    {
      fseek(ifp, save, SEEK_SET);
      for (i = 0; i < bsize; i += 8)
      {
        read_shorts(raw, 6);
        out[i] = raw[0] >> 12 << 8 | raw[2] >> 12 << 4 | raw[4] >> 12;
        out[i + 1] = raw[1] >> 12 << 8 | raw[3] >> 12 << 4 | raw[5] >> 12;
        for (j = 0; j < 6; j++)
          out[i + 2 + j] = raw[j] & 0xfff;
      }
      return 1;
    }
  }
  if ((bsize & 7) == 4)
  {
    bitbuf = fgetc(ifp) << 8;
    bitbuf += fgetc(ifp);
    bits = 16;
  }
  for (i = 0; i < bsize; i++)
  {
    len = blen[i];
    if (bits < len)
    {
      for (j = 0; j < 32; j += 8)
        bitbuf += (INT64)fgetc(ifp) << (bits + (j ^ 8));
      bits += 32;
    }
    diff = bitbuf & (0xffff >> (16 - len));
    bitbuf >>= len;
    bits -= len;
    if (len > 0 && (diff & (1 << (len - 1))) == 0)
      diff -= (1 << len) - 1;
    out[i] = diff;
  }
  return 0;
}

void LibRaw::kodak_65000_load_raw()
{
  short buf[272]; /* 264 looks enough */
  int row, col, len, pred[2], ret, i;

  for (row = 0; row < height; row++)
  {
    checkCancel();
    for (col = 0; col < width; col += 256)
    {
      pred[0] = pred[1] = 0;
      len = MIN(256, width - col);
      ret = kodak_65000_decode(buf, len);
      for (i = 0; i < len; i++)
      {
        int idx = ret ? buf[i] : (pred[i & 1] += buf[i]);
        if (idx >= 0 && idx < 0xffff)
        {
          if ((RAW(row, col + i) = curve[idx]) >> 12)
            derror();
        }
        else
          derror();
      }
    }
  }
}

void LibRaw::kodak_ycbcr_load_raw()
{
  if (!image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  short buf[384], *bp;
  int row, col, len, c, i, j, k, y[2][2], cb, cr, rgb[3];
  ushort *ip;

  unsigned int bits =
      (load_flags && load_flags > 9 && load_flags < 17) ? load_flags : 10;
  const int pixels = int(width)*int(height);
  for (row = 0; row < height; row += 2)
  {
    checkCancel();
    for (col = 0; col < width; col += 128)
    {
      len = MIN(128, width - col);
      kodak_65000_decode(buf, len * 3);
      y[0][1] = y[1][1] = cb = cr = 0;
      for (bp = buf, i = 0; i < len; i += 2, bp += 2)
      {
        cb += bp[4];
        cr += bp[5];
        rgb[1] = -((cb + cr + 2) >> 2);
        rgb[2] = rgb[1] + cb;
        rgb[0] = rgb[1] + cr;
        for (j = 0; j < 2; j++)
          for (k = 0; k < 2; k++)
          {
            if ((y[j][k] = y[j][k ^ 1] + *bp++) >> bits)
              derror();
            int indx = (row + j) * width + col + i + k;
            if(indx>=0 && indx < pixels)
            {
                ip = image[indx];
                FORC3 ip[c] = curve[LIM(y[j][k] + rgb[c], 0, 0xfff)];
            }
          }
      }
    }
  }
}

void LibRaw::kodak_rgb_load_raw()
{
  if (!image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  short buf[768], *bp;
  int row, col, len, c, i, rgb[3], ret;
  ushort *ip = image[0];

  for (row = 0; row < height; row++)
  {
    checkCancel();
    for (col = 0; col < width; col += 256)
    {
      len = MIN(256, width - col);
      ret = kodak_65000_decode(buf, len * 3);
      memset(rgb, 0, sizeof rgb);
      for (bp = buf, i = 0; i < len; i++, ip += 4)
        if (load_flags == 12)
          FORC3 ip[c] = ret ? (*bp++) : (rgb[c] += *bp++);
        else
          FORC3 if ((ip[c] = ret ? (*bp++) : (rgb[c] += *bp++)) >> 12) derror();
    }
  }
}

void LibRaw::kodak_thumb_load_raw()
{
  if (!image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  int row, col;
  colors = thumb_misc >> 5;
  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
      read_shorts(image[row * width + col], colors);
  maximum = (1 << (thumb_misc & 31)) - 1;
}
