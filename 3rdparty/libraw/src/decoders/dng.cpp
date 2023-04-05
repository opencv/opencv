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

void LibRaw::vc5_dng_load_raw_placeholder()
{
    // placeholder only, real decoding implemented in GPR SDK
    throw LIBRAW_EXCEPTION_DECODE_RAW;
}

void LibRaw::adobe_copy_pixel(unsigned row, unsigned col, ushort **rp)
{
  int c;

  if (tiff_samples == 2 && shot_select)
    (*rp)++;
  if (raw_image)
  {
    if (row < raw_height && col < raw_width)
      RAW(row, col) = curve[**rp];
    *rp += tiff_samples;
  }
  else
  {
    if (row < raw_height && col < raw_width)
      FORC(int(tiff_samples))
    image[row * raw_width + col][c] = curve[(*rp)[c]];
    *rp += tiff_samples;
  }
  if (tiff_samples == 2 && shot_select)
    (*rp)--;
}
void LibRaw::lossless_dng_load_raw()
{
  unsigned trow = 0, tcol = 0, jwide, jrow, jcol, row, col, i, j;
  INT64 save;
  struct jhead jh;
  ushort *rp;

  int ss = shot_select;
  shot_select = libraw_internal_data.unpacker_data.dng_frames[LIM(ss,0,(LIBRAW_IFD_MAXCOUNT*2-1))] & 0xff;

  while (trow < raw_height)
  {
    checkCancel();
    save = ftell(ifp);
    if (tile_length < INT_MAX)
      fseek(ifp, get4(), SEEK_SET);
    if (!ljpeg_start(&jh, 0))
      break;
    jwide = jh.wide;
    if (filters)
      jwide *= jh.clrs;

    if(filters && (tiff_samples == 2)) // Fuji Super CCD
        jwide /= 2;
    try
    {
      switch (jh.algo)
      {
      case 0xc1:
        jh.vpred[0] = 16384;
        getbits(-1);
        for (jrow = 0; jrow + 7 < (unsigned)jh.high; jrow += 8)
        {
          checkCancel();
          for (jcol = 0; jcol + 7 < (unsigned)jh.wide; jcol += 8)
          {
            ljpeg_idct(&jh);
            rp = jh.idct;
            row = trow + jcol / tile_width + jrow * 2;
            col = tcol + jcol % tile_width;
            for (i = 0; i < 16; i += 2)
              for (j = 0; j < 8; j++)
                adobe_copy_pixel(row + i, col + j, &rp);
          }
        }
        break;
      case 0xc3:
        for (row = col = jrow = 0; jrow < (unsigned)jh.high; jrow++)
        {
          checkCancel();
          rp = ljpeg_row(jrow, &jh);
          if (tiff_samples == 1 && jh.clrs > 1 && jh.clrs * jwide == raw_width)
            for (jcol = 0; jcol < jwide * jh.clrs; jcol++)
            {
              adobe_copy_pixel(trow + row, tcol + col, &rp);
              if (++col >= tile_width || col >= raw_width)
                row += 1 + (col = 0);
            }
          else
            for (jcol = 0; jcol < jwide; jcol++)
            {
              adobe_copy_pixel(trow + row, tcol + col, &rp);
              if (++col >= tile_width || col >= raw_width)
                row += 1 + (col = 0);
            }
        }
      }
    }
    catch (...)
    {
      ljpeg_end(&jh);
      shot_select = ss;
      throw;
    }
    fseek(ifp, save + 4, SEEK_SET);
    if ((tcol += tile_width) >= raw_width)
      trow += tile_length + (tcol = 0);
    ljpeg_end(&jh);
  }
  shot_select = ss;
}

void LibRaw::packed_dng_load_raw()
{
  ushort *pixel, *rp;
  unsigned row, col;

  if (tile_length < INT_MAX)
  {
      packed_tiled_dng_load_raw();
      return;
  }

  int ss = shot_select;
  shot_select = libraw_internal_data.unpacker_data.dng_frames[LIM(ss,0,(LIBRAW_IFD_MAXCOUNT*2-1))] & 0xff;

  pixel = (ushort *)calloc(raw_width, tiff_samples * sizeof *pixel);
  try
  {
    for (row = 0; row < raw_height; row++)
    {
      checkCancel();
      if (tiff_bps == 16)
        read_shorts(pixel, raw_width * tiff_samples);
      else
      {
        getbits(-1);
        for (col = 0; col < raw_width * tiff_samples; col++)
          pixel[col] = getbits(tiff_bps);
      }
      for (rp = pixel, col = 0; col < raw_width; col++)
        adobe_copy_pixel(row, col, &rp);
    }
  }
  catch (...)
  {
    free(pixel);
    shot_select = ss;
    throw;
  }
  free(pixel);
  shot_select = ss;
}
#ifdef NO_JPEG
void LibRaw::lossy_dng_load_raw() {}
#else

static void jpegErrorExit_d(j_common_ptr /*cinfo*/)
{
  throw LIBRAW_EXCEPTION_DECODE_JPEG;
}

void LibRaw::lossy_dng_load_raw()
{
  if (!image)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  struct jpeg_decompress_struct cinfo;
  JSAMPARRAY buf;
  JSAMPLE(*pixel)[3];
  unsigned sorder = order, ntags, opcode, deg, i, j, c;
  unsigned trow = 0, tcol = 0, row, col;
  INT64 save = data_offset - 4;
  ushort cur[3][256];
  double coeff[9], tot;

  if (meta_offset)
  {
    fseek(ifp, meta_offset, SEEK_SET);
    order = 0x4d4d;
    ntags = get4();
    while (ntags--)
    {
      opcode = get4();
      get4();
      get4();
      if (opcode != 8)
      {
        fseek(ifp, get4(), SEEK_CUR);
        continue;
      }
      fseek(ifp, 20, SEEK_CUR);
      if ((c = get4()) > 2)
        break;
      fseek(ifp, 12, SEEK_CUR);
      if ((deg = get4()) > 8)
        break;
      for (i = 0; i <= deg && i < 9; i++)
        coeff[i] = getreal(LIBRAW_EXIFTAG_TYPE_DOUBLE);
      for (i = 0; i < 256; i++)
      {
        for (tot = j = 0; j <= deg; j++)
          tot += coeff[j] * pow(i / 255.0, (int)j);
        cur[c][i] = (ushort)(tot * 0xffff);
      }
    }
    order = sorder;
  }
  else
  {
    gamma_curve(1 / 2.4, 12.92, 1, 255);
    FORC3 memcpy(cur[c], curve, sizeof cur[0]);
  }

  struct jpeg_error_mgr pub;
  cinfo.err = jpeg_std_error(&pub);
  pub.error_exit = jpegErrorExit_d;

  jpeg_create_decompress(&cinfo);

  while (trow < raw_height)
  {
    fseek(ifp, save += 4, SEEK_SET);
    if (tile_length < INT_MAX)
      fseek(ifp, get4(), SEEK_SET);
    if (libraw_internal_data.internal_data.input->jpeg_src(&cinfo) == -1)
    {
      jpeg_destroy_decompress(&cinfo);
      throw LIBRAW_EXCEPTION_DECODE_JPEG;
    }
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    buf = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                     cinfo.output_width * 3, 1);
    try
    {
      while (cinfo.output_scanline < cinfo.output_height &&
             (row = trow + cinfo.output_scanline) < height)
      {
        checkCancel();
        jpeg_read_scanlines(&cinfo, buf, 1);
        pixel = (JSAMPLE(*)[3])buf[0];
        for (col = 0; col < cinfo.output_width && tcol + col < width; col++)
        {
          FORC3 image[row * width + tcol + col][c] = cur[c][pixel[col][c]];
        }
      }
    }
    catch (...)
    {
      jpeg_destroy_decompress(&cinfo);
      throw;
    }
    jpeg_abort_decompress(&cinfo);
    if ((tcol += tile_width) >= raw_width)
      trow += tile_length + (tcol = 0);
  }
  jpeg_destroy_decompress(&cinfo);
  maximum = 0xffff;
}
#endif
