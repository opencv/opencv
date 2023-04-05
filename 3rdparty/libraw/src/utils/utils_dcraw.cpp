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

int LibRaw::fcol(int row, int col)
{
  static const char filter[16][16] = {
      {2, 1, 1, 3, 2, 3, 2, 0, 3, 2, 3, 0, 1, 2, 1, 0},
      {0, 3, 0, 2, 0, 1, 3, 1, 0, 1, 1, 2, 0, 3, 3, 2},
      {2, 3, 3, 2, 3, 1, 1, 3, 3, 1, 2, 1, 2, 0, 0, 3},
      {0, 1, 0, 1, 0, 2, 0, 2, 2, 0, 3, 0, 1, 3, 2, 1},
      {3, 1, 1, 2, 0, 1, 0, 2, 1, 3, 1, 3, 0, 1, 3, 0},
      {2, 0, 0, 3, 3, 2, 3, 1, 2, 0, 2, 0, 3, 2, 2, 1},
      {2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 1, 2, 3, 0, 0, 1},
      {1, 0, 0, 2, 3, 0, 0, 3, 0, 3, 0, 3, 2, 1, 2, 3},
      {2, 3, 3, 1, 1, 2, 1, 0, 3, 2, 3, 0, 2, 3, 1, 3},
      {1, 0, 2, 0, 3, 0, 3, 2, 0, 1, 1, 2, 0, 1, 0, 2},
      {0, 1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 0, 2, 1, 3, 2},
      {2, 3, 2, 0, 0, 1, 3, 0, 2, 0, 1, 2, 3, 0, 1, 0},
      {1, 3, 1, 2, 3, 2, 3, 2, 0, 2, 0, 1, 1, 0, 3, 0},
      {0, 2, 0, 3, 1, 0, 0, 1, 1, 3, 3, 2, 3, 2, 2, 1},
      {2, 1, 3, 2, 3, 1, 2, 1, 0, 3, 0, 2, 0, 2, 0, 2},
      {0, 3, 1, 0, 0, 2, 0, 3, 2, 1, 3, 1, 1, 3, 1, 3}};

  if (filters == 1)
    return filter[(row + top_margin) & 15][(col + left_margin) & 15];
  if (filters == 9)
    return xtrans[(row + 6) % 6][(col + 6) % 6];
  return FC(row, col);
}

size_t LibRaw::strnlen(const char *s, size_t n)
{
#if !defined(__FreeBSD__) && !defined(__OpenBSD__)
  const char *p = (const char *)memchr(s, 0, n);
  return (p ? p - s : n);
#else
  return ::strnlen(s, n);
#endif
}

void *LibRaw::memmem(char *haystack, size_t haystacklen, char *needle,
                     size_t needlelen)
{
#if !defined(__GLIBC__) && !defined(__FreeBSD__) && !defined(__OpenBSD__)
  char *c;
  for (c = haystack; c <= haystack + haystacklen - needlelen; c++)
    if (!memcmp(c, needle, needlelen))
      return c;
  return 0;
#else
  return ::memmem(haystack, haystacklen, needle, needlelen);
#endif
}

char *LibRaw::strcasestr(char *haystack, const char *needle)
{
  char *c;
  for (c = haystack; *c; c++)
    if (!strncasecmp(c, needle, strlen(needle)))
      return c;
  return 0;
}

void LibRaw::initdata()
{
  tiff_flip = flip = filters = UINT_MAX; /* unknown */
  raw_height = raw_width = fuji_width = fuji_layout = cr2_slice[0] = 0;
  maximum = height = width = top_margin = left_margin = 0;
  cdesc[0] = desc[0] = artist[0] = make[0] = model[0] = model2[0] = 0;
  iso_speed = shutter = aperture = focal_len = 0;
  unique_id = 0ULL;
  tiff_nifds = 0;
  memset(tiff_ifd, 0, sizeof tiff_ifd);
  for (int i = 0; i < LIBRAW_IFD_MAXCOUNT; i++)
  {
    tiff_ifd[i].dng_color[0].illuminant = tiff_ifd[i].dng_color[1].illuminant =
        0xffff;
    for (int c = 0; c < 4; c++)
      tiff_ifd[i].dng_levels.analogbalance[c] = 1.0f;
  }
  for (int i = 0; i < 0x10000; i++)
    curve[i] = i;
  memset(gpsdata, 0, sizeof gpsdata);
  memset(cblack, 0, sizeof cblack);
  memset(white, 0, sizeof white);
  memset(mask, 0, sizeof mask);
  thumb_offset = thumb_length = thumb_width = thumb_height = 0;
  load_raw = 0;
  thumb_format = LIBRAW_INTERNAL_THUMBNAIL_JPEG; // default to JPEG
  data_offset = meta_offset = meta_length = tiff_bps = tiff_compress = 0;
  kodak_cbpp = zero_after_ff = dng_version = load_flags = 0;
  timestamp = shot_order = tiff_samples = black = is_foveon = 0;
  mix_green = profile_length = data_error = zero_is_bad = 0;
  pixel_aspect = is_raw = raw_color = 1;
  tile_width = tile_length = 0;
  metadata_blocks = 0;
  is_NikonTransfer = 0;
  is_Olympus = 0;
  OlympusDNG_SubDirOffsetValid = 0;
  is_Sony = 0;
  is_pana_raw = 0;
  maker_index = LIBRAW_CAMERAMAKER_Unknown;
  FujiCropMode = 0;
  is_PentaxRicohMakernotes = 0;
  normalized_model[0] = 0;
  normalized_make[0] = 0;
  CM_found = 0;
}

void LibRaw::aRGB_coeff(double aRGB_cam[3][3])
{
  static const double rgb_aRGB[3][3] = {
      {1.39828313770000, -0.3982830047, 9.64980900741708E-8},
      {6.09219200572997E-8, 0.9999999809, 1.33230799934103E-8},
      {2.17237099975343E-8, -0.0429383201, 1.04293828050000}};

  double cmatrix_tmp[3][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  int i, j, k;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
    {
      for (k = 0; k < 3; k++)
        cmatrix_tmp[i][j] += rgb_aRGB[i][k] * aRGB_cam[k][j];
      cmatrix[i][j] = (float)cmatrix_tmp[i][j];
    }
}

void LibRaw::romm_coeff(float romm_cam[3][3])
{
  static const float rgb_romm[3][3] = /* ROMM == Kodak ProPhoto */
      {{2.034193f, -0.727420f, -0.306766f},
       {-0.228811f, 1.231729f, -0.002922f},
       {-0.008565f, -0.153273f, 1.161839f}};
  int i, j, k;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (cmatrix[i][j] = k = 0; k < 3; k++)
        cmatrix[i][j] += rgb_romm[i][k] * romm_cam[k][j];
}

void LibRaw::remove_zeroes()
{
  unsigned row, col, tot, n;
  int r, c;

  RUN_CALLBACK(LIBRAW_PROGRESS_REMOVE_ZEROES, 0, 2);

  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
      if (BAYER(row, col) == 0)
      {
        tot = n = 0;
        for (r = (int)row - 2; r <= (int)row + 2; r++)
          for (c = (int)col - 2; c <= (int)col + 2; c++)
            if (r >= 0 && r < height && c >= 0 && c < width &&
                FC(r, c) == FC(row, col) && BAYER(r, c))
              tot += (n++, BAYER(r, c));
        if (n)
          BAYER(row, col) = tot / n;
      }
  RUN_CALLBACK(LIBRAW_PROGRESS_REMOVE_ZEROES, 1, 2);
}
void LibRaw::crop_masked_pixels()
{
  int row, col;
  unsigned c, m, zero, val;
#define mblack imgdata.color.black_stat

  if (mask[0][3] > 0)
    goto mask_set;
  if (load_raw == &LibRaw::canon_load_raw ||
      load_raw == &LibRaw::lossless_jpeg_load_raw ||
      load_raw == &LibRaw::crxLoadRaw)
  {
    mask[0][1] = mask[1][1] += 2;
    mask[0][3] -= 2;
    goto sides;
  }
  if (load_raw == &LibRaw::canon_600_load_raw ||
      load_raw == &LibRaw::sony_load_raw ||
      (load_raw == &LibRaw::eight_bit_load_raw && strncmp(model, "DC2", 3)) ||
      load_raw == &LibRaw::kodak_262_load_raw ||
      (load_raw == &LibRaw::packed_load_raw && (load_flags & 32)))
  {
  sides:
    mask[0][0] = mask[1][0] = top_margin;
    mask[0][2] = mask[1][2] = top_margin + height;
    mask[0][3] += left_margin;
    mask[1][1] += left_margin + width;
    mask[1][3] += raw_width;
  }
  if (load_raw == &LibRaw::nokia_load_raw)
  {
    mask[0][2] = top_margin;
    mask[0][3] = width;
  }
  if (load_raw == &LibRaw::broadcom_load_raw)
  {
    mask[0][2] = top_margin;
    mask[0][3] = width;
  }
mask_set:
  memset(mblack, 0, sizeof mblack);
  for (zero = m = 0; m < 8; m++)
    for (row = MAX(mask[m][0], 0); row < MIN(mask[m][2], raw_height); row++)
      for (col = MAX(mask[m][1], 0); col < MIN(mask[m][3], raw_width); col++)
      {
        /* No need to subtract margins because full area and active area filters are the same */
        c = FC(row, col);
        mblack[c] += val = raw_image[(row)*raw_pitch / 2 + (col)];
        mblack[4 + c]++;
        zero += !val;
      }
  if (load_raw == &LibRaw::canon_600_load_raw && width < raw_width)
  {
    black = (mblack[0] + mblack[1] + mblack[2] + mblack[3]) /
                MAX(1, (mblack[4] + mblack[5] + mblack[6] + mblack[7])) -
            4;
  }
  else if (zero < mblack[4] && mblack[5] && mblack[6] && mblack[7])
  {
    FORC4 cblack[c] = mblack[c] / MAX(1, mblack[4 + c]);
    black = cblack[4] = cblack[5] = cblack[6] = 0;
  }
}
#undef mblack

void LibRaw::pseudoinverse(double (*in)[3], double (*out)[3], int size)
{
  double work[3][6], num;
  int i, j, k;

  for (i = 0; i < 3; i++)
  {
    for (j = 0; j < 6; j++)
      work[i][j] = j == i + 3;
    for (j = 0; j < 3; j++)
      for (k = 0; k < size && k < 4; k++)
        work[i][j] += in[k][i] * in[k][j];
  }
  for (i = 0; i < 3; i++)
  {
    num = work[i][i];
    for (j = 0; j < 6; j++)
      if (fabs(num) > 0.00001f)
        work[i][j] /= num;
    for (k = 0; k < 3; k++)
    {
      if (k == i)
        continue;
      num = work[k][i];
      for (j = 0; j < 6; j++)
        work[k][j] -= work[i][j] * num;
    }
  }
  for (i = 0; i < size && i < 4; i++)
    for (j = 0; j < 3; j++)
      for (out[i][j] = k = 0; k < 3; k++)
        out[i][j] += work[j][k + 3] * in[i][k];
}

void LibRaw::cam_xyz_coeff(float _rgb_cam[3][4], double cam_xyz[4][3])
{
  double cam_rgb[4][3], inverse[4][3], num;
  int i, j, k;

  for (i = 0; i < colors && i < 4; i++) /* Multiply out XYZ colorspace */
    for (j = 0; j < 3; j++)
      for (cam_rgb[i][j] = k = 0; k < 3; k++)
        cam_rgb[i][j] += cam_xyz[i][k] * LibRaw_constants::xyz_rgb[k][j];

  for (i = 0; i < colors && i < 4; i++)
  {                               /* Normalize cam_rgb so that */
    for (num = j = 0; j < 3; j++) /* cam_rgb * (1,1,1) is (1,1,1,1) */
      num += cam_rgb[i][j];
    if (num > 0.00001)
    {
      for (j = 0; j < 3; j++)
        cam_rgb[i][j] /= num;
      pre_mul[i] = 1 / num;
    }
    else
    {
      for (j = 0; j < 3; j++)
        cam_rgb[i][j] = 0.0;
      pre_mul[i] = 1.0;
    }
  }
  pseudoinverse(cam_rgb, inverse, colors);
  for (i = 0; i < 3; i++)
    for (j = 0; j < colors && j < 4; j++)
      _rgb_cam[i][j] = inverse[j][i];
}

void LibRaw::tiff_get(unsigned base, unsigned *tag, unsigned *type,
                      unsigned *len, unsigned *save)
{
#ifdef LIBRAW_IOSPACE_CHECK
  INT64 pos = ftell(ifp);
  INT64 fsize = ifp->size();
  if (fsize < 12 || (fsize - pos) < 12)
    throw LIBRAW_EXCEPTION_IO_EOF;
#endif
  *tag = get2();
  *type = get2();
  *len = get4();
  *save = ftell(ifp) + 4;
  if (*len * tagtype_dataunit_bytes[(*type <= LIBRAW_EXIFTAG_TYPE_IFD8) ? *type : 0] > 4)
    fseek(ifp, get4() + base, SEEK_SET);
}
