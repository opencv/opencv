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

#include "../../internal/libraw_cxx_defs.h"

void LibRaw::kodak_thumb_loader()
{
  INT64 est_datasize =
      T.theight * T.twidth / 3; // is 0.3 bytes per pixel good estimate?
  if (ID.toffset < 0)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  if (ID.toffset + est_datasize > ID.input->size() + THUMB_READ_BEYOND)
    throw LIBRAW_EXCEPTION_IO_EOF;

  if(INT64(T.theight) * INT64(T.twidth) > 1024ULL * 1024ULL * LIBRAW_MAX_THUMBNAIL_MB)
      throw LIBRAW_EXCEPTION_IO_CORRUPT;

  if (INT64(T.theight) * INT64(T.twidth) < 64ULL)
      throw LIBRAW_EXCEPTION_IO_CORRUPT;

  if(T.twidth < 16 || T.twidth > 8192 || T.theight < 16 || T.theight > 8192)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  // some kodak cameras
  ushort s_height = S.height, s_width = S.width, s_iwidth = S.iwidth,
         s_iheight = S.iheight;
  ushort s_flags = libraw_internal_data.unpacker_data.load_flags;
  libraw_internal_data.unpacker_data.load_flags = 12;
  int s_colors = P1.colors;
  unsigned s_filters = P1.filters;
  ushort(*s_image)[4] = imgdata.image;

  S.height = T.theight;
  S.width = T.twidth;
  P1.filters = 0;

#define Tformat libraw_internal_data.unpacker_data.thumb_format


  if (Tformat == LIBRAW_INTERNAL_THUMBNAIL_KODAK_YCBCR)
  {
    S.height += S.height & 1;
    S.width += S.width & 1;
  }

  S.iheight = S.height;
  S.iwidth = S.width;

  imgdata.image =
      (ushort(*)[4])calloc(S.iheight * S.iwidth, sizeof(*imgdata.image));

  ID.input->seek(ID.toffset, SEEK_SET);
  // read kodak thumbnail into T.image[]
  try
  {
      if (Tformat == LIBRAW_INTERNAL_THUMBNAIL_KODAK_YCBCR)
          kodak_ycbcr_load_raw();
      else if(Tformat == LIBRAW_INTERNAL_THUMBNAIL_KODAK_RGB)
        kodak_rgb_load_raw();
      else if (Tformat == LIBRAW_INTERNAL_THUMBNAIL_KODAK_THUMB)
        kodak_thumb_load_raw();
  }
  catch (...)
  {
    free(imgdata.image);
    imgdata.image = s_image;

    T.twidth = 0;
    S.width = s_width;

    S.iwidth = s_iwidth;
    S.iheight = s_iheight;

    T.theight = 0;
    S.height = s_height;

    T.tcolors = 0;
    P1.colors = s_colors;

    P1.filters = s_filters;
    T.tlength = 0;
    libraw_internal_data.unpacker_data.load_flags = s_flags;
    return;
  }

  // from scale_colors
  {
    double dmax;
    float scale_mul[4];
    int c, val;
    for (dmax = DBL_MAX, c = 0; c < 3; c++)
      if (dmax > C.pre_mul[c])
        dmax = C.pre_mul[c];

    for (c = 0; c < 3; c++)
      scale_mul[c] = (C.pre_mul[c] / dmax) * 65535.0 / C.maximum;
    scale_mul[3] = scale_mul[1];

    size_t size = S.height * S.width;
    for (unsigned i = 0; i < size * 4; i++)
    {
      val = imgdata.image[0][i];
      if (!val)
        continue;
      val *= scale_mul[i & 3];
      imgdata.image[0][i] = CLIP(val);
    }
  }

  // from convert_to_rgb
  ushort *img;
  int row, col;

  int(*t_hist)[LIBRAW_HISTOGRAM_SIZE] =
      (int(*)[LIBRAW_HISTOGRAM_SIZE])calloc(sizeof(*t_hist), 4);

  float out[3], out_cam[3][4] = {{2.81761312f, -1.98369181f, 0.166078627f, 0},
                                 {-0.111855984f, 1.73688626f, -0.625030339f, 0},
                                 {-0.0379119813f, -0.891268849f, 1.92918086f, 0}};

  for (img = imgdata.image[0], row = 0; row < S.height; row++)
    for (col = 0; col < S.width; col++, img += 4)
    {
      out[0] = out[1] = out[2] = 0;
      int c;
      for (c = 0; c < 3; c++)
      {
        out[0] += out_cam[0][c] * img[c];
        out[1] += out_cam[1][c] * img[c];
        out[2] += out_cam[2][c] * img[c];
      }
      for (c = 0; c < 3; c++)
        img[c] = CLIP((int)out[c]);
      for (c = 0; c < P1.colors; c++)
        t_hist[c][img[c] >> 3]++;
    }

  // from gamma_lut
  int(*save_hist)[LIBRAW_HISTOGRAM_SIZE] =
      libraw_internal_data.output_data.histogram;
  libraw_internal_data.output_data.histogram = t_hist;

  // make curve output curve!
  ushort *t_curve = (ushort *)calloc(sizeof(C.curve), 1);
  memmove(t_curve, C.curve, sizeof(C.curve));
  memset(C.curve, 0, sizeof(C.curve));
  {
    int perc, val, total, t_white = 0x2000, c;

    perc = S.width * S.height * 0.01; /* 99th percentile white level */
    if (IO.fuji_width)
      perc /= 2;
    if (!((O.highlight & ~2) || O.no_auto_bright))
      for (t_white = c = 0; c < P1.colors; c++)
      {
        for (val = 0x2000, total = 0; --val > 32;)
          if ((total += libraw_internal_data.output_data.histogram[c][val]) >
              perc)
            break;
        if (t_white < val)
          t_white = val;
      }
    gamma_curve(O.gamm[0], O.gamm[1], 2, (t_white << 3) / O.bright);
  }

  libraw_internal_data.output_data.histogram = save_hist;
  free(t_hist);

  // from write_ppm_tiff - copy pixels into bitmap

  int s_flip = imgdata.sizes.flip;
  if (imgdata.rawparams.options & LIBRAW_RAWOPTIONS_NO_ROTATE_FOR_KODAK_THUMBNAILS)
    imgdata.sizes.flip = 0;

  S.iheight = S.height;
  S.iwidth = S.width;
  if (S.flip & 4)
    SWAP(S.height, S.width);

  if (T.thumb)
    free(T.thumb);
  T.thumb = (char *)calloc(S.width * S.height, P1.colors);
  T.tlength = S.width * S.height * P1.colors;

  // from write_tiff_ppm
  {
    int soff = flip_index(0, 0);
    int cstep = flip_index(0, 1) - soff;
    int rstep = flip_index(1, 0) - flip_index(0, S.width);

    for (int rr = 0; rr < S.height; rr++, soff += rstep)
    {
      char *ppm = T.thumb + rr * S.width * P1.colors;
      for (int cc = 0; cc < S.width; cc++, soff += cstep)
        for (int c = 0; c < P1.colors; c++)
          ppm[cc * P1.colors + c] =
              imgdata.color.curve[imgdata.image[soff][c]] >> 8;
    }
  }

  memmove(C.curve, t_curve, sizeof(C.curve));
  free(t_curve);

  // restore variables
  free(imgdata.image);
  imgdata.image = s_image;

  if (imgdata.rawparams.options & LIBRAW_RAWOPTIONS_NO_ROTATE_FOR_KODAK_THUMBNAILS)
    imgdata.sizes.flip = s_flip;

  T.twidth = S.width;
  S.width = s_width;

  S.iwidth = s_iwidth;
  S.iheight = s_iheight;

  T.theight = S.height;
  S.height = s_height;

  T.tcolors = P1.colors;
  P1.colors = s_colors;

  P1.filters = s_filters;
  libraw_internal_data.unpacker_data.load_flags = s_flags;
}

// ������� thumbnail �� �����, ������ thumb_format � ������������ � ��������

int LibRaw::thumbOK(INT64 maxsz)
{
  if (!ID.input)
    return 0;
  if (!ID.toffset && !(imgdata.thumbnail.tlength > 0 &&
                       load_raw == &LibRaw::broadcom_load_raw) // RPi
#ifdef USE_6BY9RPI
      && !(imgdata.thumbnail.tlength > 0 && libraw_internal_data.unpacker_data.load_flags & 0x4000 &&
           (load_raw == &LibRaw::rpi_load_raw8 || load_raw == &LibRaw::nokia_load_raw ||
            load_raw == &LibRaw::rpi_load_raw12 || load_raw == &LibRaw::rpi_load_raw14))
#endif
  )
    return 0;
  INT64 fsize = ID.input->size();
  if (fsize > 0xffffffffU)
    return 0; // No thumb for raw > 4Gb-1
  int tsize = 0;
  int tcol = (T.tcolors > 0 && T.tcolors < 4) ? T.tcolors : 3;
  if (Tformat == LIBRAW_INTERNAL_THUMBNAIL_JPEG)
    tsize = T.tlength;
  else if (Tformat == LIBRAW_INTERNAL_THUMBNAIL_PPM)
    tsize = tcol * T.twidth * T.theight;
  else if (Tformat == LIBRAW_INTERNAL_THUMBNAIL_PPM16)
    tsize = tcol * T.twidth * T.theight *
            ((imgdata.rawparams.options & LIBRAW_RAWOPTIONS_USE_PPM16_THUMBS) ? 2 : 1);
#ifdef USE_X3FTOOLS
  else if (Tformat == LIBRAW_INTERNAL_THUMBNAIL_X3F)
  {
    tsize = x3f_thumb_size();
  }
#endif
  else // Kodak => no check
    tsize = 1;
  if (tsize < 0)
    return 0;
  if (maxsz > 0 && tsize > maxsz)
    return 0;
  return (tsize + ID.toffset <= fsize) ? 1 : 0;
}

int LibRaw::dcraw_thumb_writer(const char *fname)
{
  //    CHECK_ORDER_LOW(LIBRAW_PROGRESS_THUMB_LOAD);

  if (!fname)
    return ENOENT;

  FILE *tfp = fopen(fname, "wb");

  if (!tfp)
    return errno;

  if (!T.thumb)
  {
    fclose(tfp);
    return LIBRAW_OUT_OF_ORDER_CALL;
  }

  try
  {
    switch (T.tformat)
    {
    case LIBRAW_THUMBNAIL_JPEG:
      jpeg_thumb_writer(tfp, T.thumb, T.tlength);
      break;
    case LIBRAW_THUMBNAIL_BITMAP:
      fprintf(tfp, "P%d\n%d %d\n255\n", T.tcolors == 1 ? 5 : 6,  T.twidth, T.theight);
      fwrite(T.thumb, 1, T.tlength, tfp);
      break;
    default:
      fclose(tfp);
      return LIBRAW_UNSUPPORTED_THUMBNAIL;
    }
    fclose(tfp);
    return 0;
  }
  catch (const std::bad_alloc&)
  {
      fclose(tfp);
      EXCEPTION_HANDLER(LIBRAW_EXCEPTION_ALLOC);
  }
  catch (const LibRaw_exceptions& err)
  {
    fclose(tfp);
    EXCEPTION_HANDLER(err);
  }
}
