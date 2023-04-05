/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *
 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/dcraw_defs.h"

void LibRaw::packed_tiled_dng_load_raw()
{
  ushort *rp;
  unsigned row, col;

  int ss = shot_select;
  shot_select = libraw_internal_data.unpacker_data.dng_frames[LIM(ss, 0, (LIBRAW_IFD_MAXCOUNT * 2 - 1))] & 0xff;
  std::vector<ushort> pixel;

  try
  {
    int ntiles = 1 + (raw_width) / tile_width;
    if ((unsigned)ntiles * tile_width > raw_width * 2u) throw LIBRAW_EXCEPTION_ALLOC;
    pixel.resize(tile_width * ntiles * tiff_samples);
  }
  catch (...)
  {
    throw LIBRAW_EXCEPTION_ALLOC; // rethrow
  }
  try
  {
      unsigned trow = 0, tcol = 0;
      INT64 save;
      while (trow < raw_height)
      {
        checkCancel();
        save = ftell(ifp);
        if (tile_length < INT_MAX)
          fseek(ifp, get4(), SEEK_SET);

        for (row = 0; row < tile_length && (row + trow) < raw_height; row++)
        {
          if (tiff_bps == 16)
            read_shorts(pixel.data(), tile_width * tiff_samples);
          else
          {
            getbits(-1);
            for (col = 0; col < tile_width * tiff_samples; col++)
              pixel[col] = getbits(tiff_bps);
          }
          for (rp = pixel.data(), col = 0; col < tile_width; col++)
            adobe_copy_pixel(trow+row, tcol+col, &rp);
        }
        fseek(ifp, save + 4, SEEK_SET);
        if ((tcol += tile_width) >= raw_width)
          trow += tile_length + (tcol = 0);
      }
  }
  catch (...)
  {
    shot_select = ss;
    throw;
  }
  shot_select = ss;
}



void LibRaw::sony_ljpeg_load_raw()
{
  unsigned trow = 0, tcol = 0, jrow, jcol, row, col;
  INT64 save;
  struct jhead jh;

  while (trow < raw_height)
  {
    checkCancel();
    save = ftell(ifp); // We're at
    if (tile_length < INT_MAX)
      fseek(ifp, get4(), SEEK_SET);
    if (!ljpeg_start(&jh, 0))
      break;
    try
    {
      for (row = jrow = 0; jrow < (unsigned)jh.high; jrow++, row += 2)
      {
        checkCancel();
        ushort(*rowp)[4] = (ushort(*)[4])ljpeg_row(jrow, &jh);
        for (col = jcol = 0; jcol < (unsigned)jh.wide; jcol++, col += 2)
        {
          RAW(trow + row, tcol + col) = rowp[jcol][0];
          RAW(trow + row, tcol + col + 1) = rowp[jcol][1];
          RAW(trow + row + 1, tcol + col) = rowp[jcol][2];
          RAW(trow + row + 1, tcol + col + 1) = rowp[jcol][3];
        }
      }
    }
    catch (...)
    {
      ljpeg_end(&jh);
      throw;
    }
    fseek(ifp, save + 4, SEEK_SET);
    if ((tcol += tile_width) >= raw_width)
      trow += tile_length + (tcol = 0);
    ljpeg_end(&jh);
  }
}

void LibRaw::nikon_he_load_raw_placeholder()
{
    throw LIBRAW_EXCEPTION_UNSUPPORTED_FORMAT;
}

void LibRaw::nikon_coolscan_load_raw()
{
  int clrs = colors == 3 ? 3 : 1;

  if (clrs == 3 && !image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  if(clrs == 1 && !raw_image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  int bypp = tiff_bps <= 8 ? 1 : 2;
  int bufsize = width * clrs * bypp;
  unsigned char *buf = (unsigned char *)malloc(bufsize);
  unsigned short *ubuf = (unsigned short *)buf;

  if (tiff_bps <= 8)
    gamma_curve(1.0 / imgdata.rawparams.coolscan_nef_gamma, 0., 1, 255);
  else
    gamma_curve(1.0 / imgdata.rawparams.coolscan_nef_gamma, 0., 1, 65535);
  fseek(ifp, data_offset, SEEK_SET);
  for (int row = 0; row < raw_height; row++)
  {
      if(tiff_bps <=8)
        fread(buf, 1, bufsize, ifp);
      else
          read_shorts(ubuf,width*clrs);

    unsigned short(*ip)[4] = (unsigned short(*)[4])image + row * width;
    unsigned short *rp =  raw_image + row * raw_width;

    if (is_NikonTransfer == 2)
    { // it is also (tiff_bps == 8)
        if (clrs == 3)
        {
          for (int col = 0; col < width; col++)
          {
            ip[col][0] = ((float)curve[buf[col * 3]]) / 255.0f;
            ip[col][1] = ((float)curve[buf[col * 3 + 1]]) / 255.0f;
            ip[col][2] = ((float)curve[buf[col * 3 + 2]]) / 255.0f;
            ip[col][3] = 0;
          }
        }
        else
        {
          for (int col = 0; col < width; col++)
            rp[col] = ((float)curve[buf[col]]) / 255.0f;
        }
    }
    else if (tiff_bps <= 8)
    {
        if (clrs == 3)
        {
          for (int col = 0; col < width; col++)
          {
            ip[col][0] = curve[buf[col * 3]];
            ip[col][1] = curve[buf[col * 3 + 1]];
            ip[col][2] = curve[buf[col * 3 + 2]];
            ip[col][3] = 0;
          }
        }
        else
        {
          for (int col = 0; col < width; col++)
            rp[col] = curve[buf[col]];
        }
    }
    else
    {
        if (clrs == 3)
        {
          for (int col = 0; col < width; col++)
          {
            ip[col][0] = curve[ubuf[col * 3]];
            ip[col][1] = curve[ubuf[col * 3 + 1]];
            ip[col][2] = curve[ubuf[col * 3 + 2]];
            ip[col][3] = 0;
          }
        }
        else
        {
          for (int col = 0; col < width; col++)
            rp[col] = curve[ubuf[col]];
        }
    }
  }
  free(buf);
}

void LibRaw::broadcom_load_raw()
{
  uchar *dp;
  int rev, row, col, c;
  rev = 3 * (order == 0x4949);
  std::vector<uchar> data(raw_stride * 2);

  for (row = 0; row < raw_height; row++)
  {
    if (fread(data.data() + raw_stride, 1, raw_stride, ifp) < raw_stride)
      derror();
    FORC(raw_stride) data[c] = data[raw_stride + (c ^ rev)];
    for (dp = data.data(), col = 0; col < raw_width; dp += 5, col += 4)
      FORC4 RAW(row, col + c) = (dp[c] << 2) | (dp[4] >> (c << 1) & 3);
  }
}

void LibRaw::android_tight_load_raw()
{
  uchar *data, *dp;
  int bwide, row, col, c;

  bwide = -(-5 * raw_width >> 5) << 3;
  data = (uchar *)malloc(bwide);
  for (row = 0; row < raw_height; row++)
  {
    if (fread(data, 1, bwide, ifp) < bwide)
      derror();
    for (dp = data, col = 0; col < raw_width; dp += 5, col += 4)
      FORC4 RAW(row, col + c) = (dp[c] << 2) | (dp[4] >> (c << 1) & 3);
  }
  free(data);
}

void LibRaw::android_loose_load_raw()
{
  uchar *data, *dp;
  int bwide, row, col, c;
  UINT64 bitbuf = 0;

  bwide = (raw_width + 5) / 6 << 3;
  data = (uchar *)malloc(bwide);
  for (row = 0; row < raw_height; row++)
  {
    if (fread(data, 1, bwide, ifp) < bwide)
      derror();
    for (dp = data, col = 0; col < raw_width; dp += 8, col += 6)
    {
      FORC(8) bitbuf = (bitbuf << 8) | dp[c ^ 7];
      FORC(6) RAW(row, col + c) = (bitbuf >> c * 10) & 0x3ff;
    }
  }
  free(data);
}

void LibRaw::unpacked_load_raw_reversed()
{
  int row, col, bits = 0;
  while (1 << ++bits < (int)maximum)
    ;
  for (row = raw_height - 1; row >= 0; row--)
  {
    checkCancel();
    read_shorts(&raw_image[row * raw_width], raw_width);
    for (col = 0; col < raw_width; col++)
      if ((RAW(row, col) >>= load_flags) >> bits &&
          (unsigned)(row - top_margin) < height &&
          (unsigned)(col - left_margin) < width)
        derror();
  }
}

#ifdef USE_6BY9RPI

void LibRaw::rpi_load_raw8()
{
	uchar  *data, *dp;
	int rev, dwide, row, col, c;
	double sum[] = { 0,0 };
	rev = 3 * (order == 0x4949);
	if (raw_stride == 0)
		dwide = raw_width;
	else
		dwide = raw_stride;
	data = (uchar *)malloc(dwide * 2);
	for (row = 0; row < raw_height; row++) {
		if (fread(data + dwide, 1, dwide, ifp) < dwide) derror();
		FORC(dwide) data[c] = data[dwide + (c ^ rev)];
		for (dp = data, col = 0; col < raw_width; dp++, col++)
			RAW(row, col + c) = dp[c];
	}
	free(data);
	maximum = 0xff;
	if (!strcmp(make, "OmniVision") ||
		!strcmp(make, "Sony") ||
		!strcmp(make, "RaspberryPi")) return;

	row = raw_height / 2;
	FORC(width - 1) {
		sum[c & 1] += SQR(RAW(row, c) - RAW(row + 1, c + 1));
		sum[~c & 1] += SQR(RAW(row + 1, c) - RAW(row, c + 1));
	}
	if (sum[1] > sum[0]) filters = 0x4b4b4b4b;
}

void LibRaw::rpi_load_raw12()
{
	uchar  *data, *dp;
	int rev, dwide, row, col, c;
	double sum[] = { 0,0 };
	rev = 3 * (order == 0x4949);
	if (raw_stride == 0)
		dwide = (raw_width * 3 + 1) / 2;
	else
		dwide = raw_stride;
	data = (uchar *)malloc(dwide * 2);
	for (row = 0; row < raw_height; row++) {
		if (fread(data + dwide, 1, dwide, ifp) < dwide) derror();
		FORC(dwide) data[c] = data[dwide + (c ^ rev)];
		for (dp = data, col = 0; col < raw_width; dp += 3, col += 2)
			FORC(2) RAW(row, col + c) = (dp[c] << 4) | (dp[2] >> (c << 2) & 0xF);
	}
	free(data);
	maximum = 0xfff;
	if (!strcmp(make, "OmniVision") ||
		!strcmp(make, "Sony") ||
		!strcmp(make, "RaspberryPi")) return;

	row = raw_height / 2;
	FORC(width - 1) {
		sum[c & 1] += SQR(RAW(row, c) - RAW(row + 1, c + 1));
		sum[~c & 1] += SQR(RAW(row + 1, c) - RAW(row, c + 1));
	}
	if (sum[1] > sum[0]) filters = 0x4b4b4b4b;
}

void LibRaw::rpi_load_raw14()
{
	uchar  *data, *dp;
	int rev, dwide, row, col, c;
	double sum[] = { 0,0 };
	rev = 3 * (order == 0x4949);
	if (raw_stride == 0)
		dwide = ((raw_width * 7) + 3) >> 2;
	else
		dwide = raw_stride;
	data = (uchar *)malloc(dwide * 2);
	for (row = 0; row < raw_height; row++) {
		if (fread(data + dwide, 1, dwide, ifp) < dwide) derror();
		FORC(dwide) data[c] = data[dwide + (c ^ rev)];
		for (dp = data, col = 0; col < raw_width; dp += 7, col += 4) {
			RAW(row, col + 0) = (dp[0] << 6) | (dp[4] >> 2);
			RAW(row, col + 1) = (dp[1] << 6) | ((dp[4] & 0x3) << 4) | ((dp[5] & 0xf0) >> 4);
			RAW(row, col + 2) = (dp[2] << 6) | ((dp[5] & 0xf) << 2) | ((dp[6] & 0xc0) >> 6);
			RAW(row, col + 3) = (dp[3] << 6) | ((dp[6] & 0x3f) << 2);
		}
	}
	free(data);
	maximum = 0x3fff;
	if (!strcmp(make, "OmniVision") ||
		!strcmp(make, "Sony") ||
		!strcmp(make, "RaspberryPi")) return;

	row = raw_height / 2;
	FORC(width - 1) {
		sum[c & 1] += SQR(RAW(row, c) - RAW(row + 1, c + 1));
		sum[~c & 1] += SQR(RAW(row + 1, c) - RAW(row, c + 1));
	}
	if (sum[1] > sum[0]) filters = 0x4b4b4b4b;
}

void LibRaw::rpi_load_raw16()
{
	uchar  *data, *dp;
	int rev, dwide, row, col, c;
	double sum[] = { 0,0 };
	rev = 3 * (order == 0x4949);
	if (raw_stride == 0)
		dwide = (raw_width * 2);
	else
		dwide = raw_stride;
	data = (uchar *)malloc(dwide * 2);
	for (row = 0; row < raw_height; row++) {
		if (fread(data + dwide, 1, dwide, ifp) < dwide) derror();
		FORC(dwide) data[c] = data[dwide + (c ^ rev)];
		for (dp = data, col = 0; col < raw_width; dp += 2, col++)
			RAW(row, col + c) = (dp[1] << 8) | dp[0];
	}
	free(data);
	maximum = 0xffff;
	if (!strcmp(make, "OmniVision") ||
		!strcmp(make, "Sony") ||
		!strcmp(make, "RaspberryPi")) return;

	row = raw_height / 2;
	FORC(width - 1) {
		sum[c & 1] += SQR(RAW(row, c) - RAW(row + 1, c + 1));
		sum[~c & 1] += SQR(RAW(row + 1, c) - RAW(row, c + 1));
	}
	if (sum[1] > sum[0]) filters = 0x4b4b4b4b;
}

#endif
