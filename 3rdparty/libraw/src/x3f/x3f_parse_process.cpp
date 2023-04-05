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

/* Library for accessing X3F Files
----------------------------------------------------------------
BSD-style License
----------------------------------------------------------------

* Copyright (c) 2010, Roland Karlsson (roland@proxel.se)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY ROLAND KARLSSON ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL ROLAND KARLSSON BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifdef USE_X3FTOOLS

#include "../../internal/libraw_cxx_defs.h"

#if defined __sun && defined DS
#undef DS
#endif
#ifdef ID
#undef ID /* used in x3f utils */
#endif

#include "../../internal/x3f_tools.h"

#define Sigma_X3F 22

void x3f_clear(void *p) { x3f_delete((x3f_t *)p); }

static void utf2char(utf16_t *str, char *buffer, unsigned bufsz)
{
  if (bufsz < 1)
    return;
  buffer[bufsz - 1] = 0;
  char *b = buffer;

  while (*str != 0x00 && --bufsz > 0)
  {
    char *chr = (char *)str;
    *b++ = *chr;
    str++;
  }
  *b = 0;
}

static void *lr_memmem(const void *l, size_t l_len, const void *s, size_t s_len)
{
  char *cur, *last;
  const char *cl = (const char *)l;
  const char *cs = (const char *)s;

  /* we need something to compare */
  if (l_len == 0 || s_len == 0)
    return NULL;

  /* "s" must be smaller or equal to "l" */
  if (l_len < s_len)
    return NULL;

  /* special case where s_len == 1 */
  if (s_len == 1)
    return (void *)memchr(l, (int)*cs, l_len);

  /* the last position where its possible to find "s" in "l" */
  last = (char *)cl + l_len - s_len;

  for (cur = (char *)cl; cur <= last; cur++)
    if (cur[0] == cs[0] && memcmp(cur, cs, s_len) == 0)
      return cur;
  return NULL;
}

void LibRaw::parse_x3f()
{
  x3f_t *x3f = x3f_new_from_file(libraw_internal_data.internal_data.input);
  if (!x3f)
    return;
  _x3f_data = x3f;

  x3f_header_t *H = NULL;

  H = &x3f->header;
  // Parse RAW size from RAW section
  x3f_directory_entry_t *DE = x3f_get_raw(x3f);
  if (!DE)
    return;
  imgdata.sizes.flip = H->rotation;
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  imgdata.sizes.raw_width = ID->columns;
  imgdata.sizes.raw_height = ID->rows;
  // Parse other params from property section

  DE = x3f_get_prop(x3f);
  if ((x3f_load_data(x3f, DE) == X3F_OK))
  {
    // Parse property list
    DEH = &DE->header;
    x3f_property_list_t *PL = &DEH->data_subsection.property_list;
    utf16_t *datap = (utf16_t *)PL->data;
    uint32_t maxitems = PL->data_size / sizeof(utf16_t);
    if (PL->property_table.size != 0)
    {
      int i;
      x3f_property_t *P = PL->property_table.element;
      for (i = 0; i < (int)PL->num_properties; i++)
      {
        char name[100], value[100];
        int noffset = (P[i].name - datap);
        int voffset = (P[i].value - datap);
        if (noffset < 0 || noffset > (int)maxitems || voffset < 0 ||
            voffset > (int)maxitems)
          throw LIBRAW_EXCEPTION_IO_CORRUPT;
        int maxnsize = maxitems - (P[i].name - datap);
        int maxvsize = maxitems - (P[i].value - datap);
        utf2char(P[i].name, name, MIN(maxnsize, ((int)sizeof(name))));
        utf2char(P[i].value, value, MIN(maxvsize, ((int)sizeof(value))));
        if (!strcmp(name, "ISO"))
          imgdata.other.iso_speed = atoi(value);
        if (!strcmp(name, "CAMMANUF"))
          strcpy(imgdata.idata.make, value);
        if (!strcmp(name, "CAMMODEL"))
          strcpy(imgdata.idata.model, value);
        if (!strcmp(name, "CAMSERIAL"))
          strcpy(imgdata.shootinginfo.BodySerial, value);
        if (!strcmp(name, "WB_DESC"))
          strcpy(imgdata.color.model2, value);
        if (!strcmp(name, "TIME"))
          imgdata.other.timestamp = atoi(value);
        if (!strcmp(name, "SHUTTER"))
          imgdata.other.shutter = atof(value);
        if (!strcmp(name, "APERTURE"))
          imgdata.other.aperture = atof(value);
        if (!strcmp(name, "FLENGTH"))
          imgdata.other.focal_len = atof(value);
        if (!strcmp(name, "FLEQ35MM"))
          imgdata.lens.makernotes.FocalLengthIn35mmFormat = atof(value);
        if (!strcmp(name, "IMAGERTEMP"))
          MN.common.SensorTemperature = atof(value);
        if (!strcmp(name, "LENSARANGE"))
        {
          char *sp;
          imgdata.lens.makernotes.MaxAp4CurFocal =
              imgdata.lens.makernotes.MinAp4CurFocal = atof(value);
          sp = strrchr(value, ' ');
          if (sp)
          {
            imgdata.lens.makernotes.MinAp4CurFocal = atof(sp);
            if (imgdata.lens.makernotes.MaxAp4CurFocal >
                imgdata.lens.makernotes.MinAp4CurFocal)
              my_swap(float, imgdata.lens.makernotes.MaxAp4CurFocal,
                      imgdata.lens.makernotes.MinAp4CurFocal);
          }
        }
        if (!strcmp(name, "LENSFRANGE"))
        {
          char *sp;
          imgdata.lens.makernotes.MinFocal = imgdata.lens.makernotes.MaxFocal =
              atof(value);
          sp = strrchr(value, ' ');
          if (sp)
          {
            imgdata.lens.makernotes.MaxFocal = atof(sp);
            if ((imgdata.lens.makernotes.MaxFocal + 0.17f) <
                imgdata.lens.makernotes.MinFocal)
              my_swap(float, imgdata.lens.makernotes.MaxFocal,
                      imgdata.lens.makernotes.MinFocal);
          }
        }
        if (!strcmp(name, "LENSMODEL"))
        {
          char *sp;
          imgdata.lens.makernotes.LensID =
              strtol(value, &sp, 16); // atoi(value);
          if (imgdata.lens.makernotes.LensID)
            imgdata.lens.makernotes.LensMount = Sigma_X3F;
        }
      }
      imgdata.idata.raw_count = 1;
      load_raw = &LibRaw::x3f_load_raw;
      imgdata.sizes.raw_pitch = imgdata.sizes.raw_width * 6;
      imgdata.idata.is_foveon = 1;
      libraw_internal_data.internal_output_params.raw_color =
          1;                          // Force adobe coeff
      imgdata.color.maximum = 0x3fff; // To be reset by color table
      libraw_internal_data.unpacker_data.order = 0x4949;
    }
  }
  else
  {
    // No property list
    if (imgdata.sizes.raw_width == 5888 || imgdata.sizes.raw_width == 2944 ||
        imgdata.sizes.raw_width == 6656 || imgdata.sizes.raw_width == 3328 ||
        imgdata.sizes.raw_width == 5504 ||
        imgdata.sizes.raw_width == 2752) // Quattro
    {
      imgdata.idata.raw_count = 1;
      load_raw = &LibRaw::x3f_load_raw;
      imgdata.sizes.raw_pitch = imgdata.sizes.raw_width * 6;
      imgdata.idata.is_foveon = 1;
      libraw_internal_data.internal_output_params.raw_color =
          1; // Force adobe coeff
      libraw_internal_data.unpacker_data.order = 0x4949;
      strcpy(imgdata.idata.make, "SIGMA");
#if 1
      // Try to find model number in first 2048 bytes;
      int pos = libraw_internal_data.internal_data.input->tell();
      libraw_internal_data.internal_data.input->seek(0, SEEK_SET);
      unsigned char buf[2048];
      libraw_internal_data.internal_data.input->read(buf, 2048, 1);
      libraw_internal_data.internal_data.input->seek(pos, SEEK_SET);
      unsigned char *fnd = (unsigned char *)lr_memmem(buf, 2048, "SIGMA dp", 8);
      unsigned char *fndsd =
          (unsigned char *)lr_memmem(buf, 2048, "sd Quatt", 8);
      if (fnd)
      {
        unsigned char *nm = fnd + 8;
        snprintf(imgdata.idata.model, 64, "dp%c Quattro",
                 *nm <= '9' && *nm >= '0' ? *nm : '2');
      }
      else if (fndsd)
      {
        snprintf(imgdata.idata.model, 64, "%s", fndsd);
      }
      else
#endif
          if (imgdata.sizes.raw_width == 6656 ||
              imgdata.sizes.raw_width == 3328)
        strcpy(imgdata.idata.model, "sd Quattro H");
      else
        strcpy(imgdata.idata.model, "dp2 Quattro");
    }
    // else
  }
  // Try to get thumbnail data
  LibRaw_thumbnail_formats format = LIBRAW_THUMBNAIL_UNKNOWN;
  if ((DE = x3f_get_thumb_jpeg(x3f)))
  {
    format = LIBRAW_THUMBNAIL_JPEG;
  }
  else if ((DE = x3f_get_thumb_plain(x3f)))
  {
    format = LIBRAW_THUMBNAIL_BITMAP;
  }
  if (DE)
  {
    x3f_directory_entry_header_t *_DEH = &DE->header;
    x3f_image_data_t *_ID = &_DEH->data_subsection.image_data;
    imgdata.thumbnail.twidth = _ID->columns;
    imgdata.thumbnail.theight = _ID->rows;
    imgdata.thumbnail.tcolors = 3;
    imgdata.thumbnail.tformat = format;
    libraw_internal_data.internal_data.toffset = DE->input.offset;
    libraw_internal_data.unpacker_data.thumb_format = LIBRAW_INTERNAL_THUMBNAIL_X3F;
  }
  DE = x3f_get_camf(x3f);
  if (DE && DE->input.size > 28)
  {
    libraw_internal_data.unpacker_data.meta_offset = DE->input.offset + 8;
    libraw_internal_data.unpacker_data.meta_length = DE->input.size - 28;
  }
}

INT64 LibRaw::x3f_thumb_size()
{
  try
  {
    x3f_t *x3f = (x3f_t *)_x3f_data;
    if (!x3f)
      return -1; // No data pointer set
    x3f_directory_entry_t *DE = x3f_get_thumb_jpeg(x3f);
    if (!DE)
      DE = x3f_get_thumb_plain(x3f);
    if (!DE)
      return -1;
    int64_t p = x3f_load_data_size(x3f, DE);
    if (p < 0 || p > 0xffffffff)
      return -1;
    return p;
  }
  catch (...)
  {
    return -1;
  }
}

void LibRaw::x3f_thumb_loader()
{
  try
  {
    x3f_t *x3f = (x3f_t *)_x3f_data;
    if (!x3f)
      return; // No data pointer set
    x3f_directory_entry_t *DE = x3f_get_thumb_jpeg(x3f);
    if (!DE)
      DE = x3f_get_thumb_plain(x3f);
    if (!DE)
      return;
    if (X3F_OK != x3f_load_data(x3f, DE))
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    x3f_directory_entry_header_t *DEH = &DE->header;
    x3f_image_data_t *ID = &DEH->data_subsection.image_data;
    imgdata.thumbnail.twidth = ID->columns;
    imgdata.thumbnail.theight = ID->rows;
    imgdata.thumbnail.tcolors = 3;
    if (imgdata.thumbnail.tformat == LIBRAW_THUMBNAIL_JPEG)
    {
      imgdata.thumbnail.thumb = (char *)malloc(ID->data_size);
      memmove(imgdata.thumbnail.thumb, ID->data, ID->data_size);
      imgdata.thumbnail.tlength = ID->data_size;
    }
    else if (imgdata.thumbnail.tformat == LIBRAW_THUMBNAIL_BITMAP)
    {
      imgdata.thumbnail.tlength = ID->columns * ID->rows * 3;
      imgdata.thumbnail.thumb = (char *)malloc(ID->columns * ID->rows * 3);
      char *src0 = (char *)ID->data;
      for (int row = 0; row < (int)ID->rows; row++)
      {
        int offset = row * ID->row_stride;
        if (offset + ID->columns * 3 > ID->data_size)
          break;
        char *dest = &imgdata.thumbnail.thumb[row * ID->columns * 3];
        char *src = &src0[offset];
        memmove(dest, src, ID->columns * 3);
      }
    }
  }
  catch (...)
  {
    // do nothing
  }
}

void LibRaw::x3f_dpq_interpolate_rg()
{
  int w = imgdata.sizes.raw_width / 2;
  int h = imgdata.sizes.raw_height / 2;
  unsigned short *image = (ushort *)imgdata.rawdata.color3_image;

  for (int color = 0; color < 2; color++)
  {
    for (int y = 2; y < (h - 2); y++)
    {
      uint16_t *row0 =
          &image[imgdata.sizes.raw_width * 3 * (y * 2) + color]; // dst[1]
      uint16_t *row1 =
          &image[imgdata.sizes.raw_width * 3 * (y * 2 + 1) + color]; // dst1[1]
      for (int x = 2; x < (w - 2); x++)
      {
        row1[0] = row1[3] = row0[3] = row0[0];
        row0 += 6;
        row1 += 6;
      }
    }
  }
}

#ifdef _ABS
#undef _ABS
#endif
#define _ABS(a) ((a) < 0 ? -(a) : (a))

#undef CLIP
#define CLIP(value, high) ((value) > (high) ? (high) : (value))

void LibRaw::x3f_dpq_interpolate_af(int xstep, int ystep, int scale)
{
  unsigned short *image = (ushort *)imgdata.rawdata.color3_image;
  for (int y = 0;
       y < imgdata.rawdata.sizes.height + imgdata.rawdata.sizes.top_margin;
       y += ystep)
  {
    if (y < imgdata.rawdata.sizes.top_margin)
      continue;
    if (y < scale)
      continue;
    if (y > imgdata.rawdata.sizes.raw_height - scale)
      break;
    uint16_t *row0 = &image[imgdata.sizes.raw_width * 3 * y]; // Наша строка
    uint16_t *row_minus =
        &image[imgdata.sizes.raw_width * 3 * (y - scale)]; // Строка выше
    uint16_t *row_plus =
        &image[imgdata.sizes.raw_width * 3 * (y + scale)]; // Строка ниже
    for (int x = 0;
         x < imgdata.rawdata.sizes.width + imgdata.rawdata.sizes.left_margin;
         x += xstep)
    {
      if (x < imgdata.rawdata.sizes.left_margin)
        continue;
      if (x < scale)
        continue;
      if (x > imgdata.rawdata.sizes.raw_width - scale)
        break;
      uint16_t *pixel0 = &row0[x * 3];
      uint16_t *pixel_top = &row_minus[x * 3];
      uint16_t *pixel_bottom = &row_plus[x * 3];
      uint16_t *pixel_left = &row0[(x - scale) * 3];
      uint16_t *pixel_right = &row0[(x + scale) * 3];
      uint16_t *pixf = pixel_top;
      if (_ABS(pixf[2] - pixel0[2]) > _ABS(pixel_bottom[2] - pixel0[2]))
        pixf = pixel_bottom;
      if (_ABS(pixf[2] - pixel0[2]) > _ABS(pixel_left[2] - pixel0[2]))
        pixf = pixel_left;
      if (_ABS(pixf[2] - pixel0[2]) > _ABS(pixel_right[2] - pixel0[2]))
        pixf = pixel_right;
      int blocal = pixel0[2], bnear = pixf[2];
      if (blocal < (int)imgdata.color.black + 16 || bnear < (int)imgdata.color.black + 16)
      {
        if (pixel0[0] < imgdata.color.black)
          pixel0[0] = imgdata.color.black;
        if (pixel0[1] < imgdata.color.black)
          pixel0[1] = imgdata.color.black;
        pixel0[0] = CLIP(
            (pixel0[0] - imgdata.color.black) * 4 + imgdata.color.black, 16383);
        pixel0[1] = CLIP(
            (pixel0[1] - imgdata.color.black) * 4 + imgdata.color.black, 16383);
      }
      else
      {
        float multip = float(bnear - imgdata.color.black) /
                       float(blocal - imgdata.color.black);
        if (pixel0[0] < imgdata.color.black)
          pixel0[0] = imgdata.color.black;
        if (pixel0[1] < imgdata.color.black)
          pixel0[1] = imgdata.color.black;
        float pixf0 = pixf[0];
        if (pixf0 < imgdata.color.black)
          pixf0 = imgdata.color.black;
        float pixf1 = pixf[1];
        if (pixf1 < imgdata.color.black)
          pixf1 = imgdata.color.black;

        pixel0[0] = CLIP(
            ((float(pixf0 - imgdata.color.black) * multip +
              imgdata.color.black) +
             ((pixel0[0] - imgdata.color.black) * 3.75 + imgdata.color.black)) /
                2,
            16383);
        pixel0[1] = CLIP(
            ((float(pixf1 - imgdata.color.black) * multip +
              imgdata.color.black) +
             ((pixel0[1] - imgdata.color.black) * 3.75 + imgdata.color.black)) /
                2,
            16383);
        // pixel0[1] = float(pixf[1]-imgdata.color.black)*multip +
        // imgdata.color.black;
      }
    }
  }
}

void LibRaw::x3f_dpq_interpolate_af_sd(int xstart, int ystart, int xend,
                                       int yend, int xstep, int ystep,
                                       int scale)
{
  unsigned short *image = (ushort *)imgdata.rawdata.color3_image;
  for (int y = ystart; y <= yend && y < imgdata.rawdata.sizes.height +
                                           imgdata.rawdata.sizes.top_margin;
       y += ystep)
  {
    uint16_t *row0 = &image[imgdata.sizes.raw_width * 3 * y]; // Наша строка
    uint16_t *row1 =
        &image[imgdata.sizes.raw_width * 3 * (y + 1)]; // Следующая строка
    uint16_t *row_minus =
        &image[imgdata.sizes.raw_width * 3 * (y - scale)]; // Строка выше
    uint16_t *row_plus =
        &image[imgdata.sizes.raw_width * 3 *
               (y + scale)]; // Строка ниже AF-point (scale=2 -> ниже row1
    uint16_t *row_minus1 = &image[imgdata.sizes.raw_width * 3 * (y - 1)];
    for (int x = xstart; x < xend && x < imgdata.rawdata.sizes.width +
                                             imgdata.rawdata.sizes.left_margin;
         x += xstep)
    {
      uint16_t *pixel00 = &row0[x * 3]; // Current pixel
      float sumR = 0.f, sumG = 0.f;
      float cnt = 0.f;
      for (int xx = -scale; xx <= scale; xx += scale)
      {
        sumR += row_minus[(x + xx) * 3];
        sumR += row_plus[(x + xx) * 3];
        sumG += row_minus[(x + xx) * 3 + 1];
        sumG += row_plus[(x + xx) * 3 + 1];
        cnt += 1.f;
        if (xx)
        {
          cnt += 1.f;
          sumR += row0[(x + xx) * 3];
          sumG += row0[(x + xx) * 3 + 1];
        }
      }
      pixel00[0] = sumR / 8.f;
      pixel00[1] = sumG / 8.f;

      if (scale == 2)
      {
        uint16_t *pixel0B = &row0[x * 3 + 3]; // right pixel
        uint16_t *pixel1B = &row1[x * 3 + 3]; // right pixel
        float sumG0 = 0, sumG1 = 0.f;
        float _cnt = 0.f;
        for (int xx = -scale; xx <= scale; xx += scale)
        {
          sumG0 += row_minus1[(x + xx) * 3 + 2];
          sumG1 += row_plus[(x + xx) * 3 + 2];
          _cnt += 1.f;
          if (xx)
          {
            sumG0 += row0[(x + xx) * 3 + 2];
            sumG1 += row1[(x + xx) * 3 + 2];
            _cnt += 1.f;
          }
        }
        if (_cnt > 1.0)
        {
          pixel0B[2] = sumG0 / _cnt;
          pixel1B[2] = sumG1 / _cnt;
        }
      }

      //			uint16_t* pixel10 = &row1[x*3]; // Pixel below current
      //			uint16_t* pixel_bottom = &row_plus[x*3];
    }
  }
}

void LibRaw::x3f_load_raw()
{
  // already in try/catch
  int raise_error = 0;
  x3f_t *x3f = (x3f_t *)_x3f_data;
  if (!x3f)
    return; // No data pointer set
  if (X3F_OK == x3f_load_data(x3f, x3f_get_raw(x3f)))
  {
    x3f_directory_entry_t *DE = x3f_get_raw(x3f);
    x3f_directory_entry_header_t *DEH = &DE->header;
    x3f_image_data_t *ID = &DEH->data_subsection.image_data;
    if (!ID)
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    x3f_quattro_t *Q = ID->quattro;
    x3f_huffman_t *HUF = ID->huffman;
    x3f_true_t *TRU = ID->tru;
    uint16_t *data = NULL;
    if (ID->rows != S.raw_height || ID->columns != S.raw_width)
    {
      raise_error = 1;
      goto end;
    }
    if (HUF != NULL)
      data = HUF->x3rgb16.data;
    if (TRU != NULL)
      data = TRU->x3rgb16.data;
    if (data == NULL)
    {
      raise_error = 1;
      goto end;
    }

    size_t datasize = S.raw_height * S.raw_width * 3 * sizeof(unsigned short);
    S.raw_pitch = S.raw_width * 3 * sizeof(unsigned short);
    if (!(imgdata.rawdata.raw_alloc = malloc(datasize)))
      throw LIBRAW_EXCEPTION_ALLOC;

    imgdata.rawdata.color3_image = (ushort(*)[3])imgdata.rawdata.raw_alloc;
    // swap R/B channels for known old cameras
    if (!strcasecmp(P1.make, "Polaroid") && !strcasecmp(P1.model, "x530"))
    {
      ushort(*src)[3] = (ushort(*)[3])data;
      for (int p = 0; p < S.raw_height * S.raw_width; p++)
      {
        imgdata.rawdata.color3_image[p][0] = src[p][2];
        imgdata.rawdata.color3_image[p][1] = src[p][1];
        imgdata.rawdata.color3_image[p][2] = src[p][0];
      }
    }
    else if (HUF)
      memmove(imgdata.rawdata.raw_alloc, data, datasize);
    else if (TRU && (!Q || !Q->quattro_layout))
      memmove(imgdata.rawdata.raw_alloc, data, datasize);
    else if (TRU && Q)
    {
      // Move quattro data in place
      // R/B plane
      for (int prow = 0; prow < (int)TRU->x3rgb16.rows && prow < S.raw_height / 2;
           prow++)
      {
        ushort(*destrow)[3] =
            (unsigned short(*)[3]) &
            imgdata.rawdata
                .color3_image[prow * 2 * S.raw_pitch / 3 / sizeof(ushort)][0];
        ushort(*srcrow)[3] =
            (unsigned short(*)[3]) & data[prow * TRU->x3rgb16.row_stride];
        for (int pcol = 0;
             pcol < (int)TRU->x3rgb16.columns && pcol < S.raw_width / 2; pcol++)
        {
          destrow[pcol * 2][0] = srcrow[pcol][0];
          destrow[pcol * 2][1] = srcrow[pcol][1];
        }
      }
      for (int row = 0; row < (int)Q->top16.rows && row < S.raw_height; row++)
      {
        ushort(*destrow)[3] =
            (unsigned short(*)[3]) &
            imgdata.rawdata
                .color3_image[row * S.raw_pitch / 3 / sizeof(ushort)][0];
        ushort *srcrow =
            (unsigned short *)&Q->top16.data[row * Q->top16.columns];
        for (int col = 0; col < (int)Q->top16.columns && col < S.raw_width; col++)
          destrow[col][2] = srcrow[col];
      }
    }

#if 1
    if (TRU && Q &&
        !(imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_NODP2Q_INTERPOLATEAF))
    {
      if (imgdata.sizes.raw_width == 5888 &&
          imgdata.sizes.raw_height == 3672) // dpN Quattro normal
      {
        x3f_dpq_interpolate_af(32, 8, 2);
      }
      else if (imgdata.sizes.raw_width == 5888 &&
               imgdata.sizes.raw_height == 3776) // sd Quattro normal raw
      {
        x3f_dpq_interpolate_af_sd(216, 464, imgdata.sizes.raw_width - 1, 3312,
                                  16, 32, 2);
      }
      else if (imgdata.sizes.raw_width == 6656 &&
               imgdata.sizes.raw_height == 4480) // sd Quattro H normal raw
      {
        x3f_dpq_interpolate_af_sd(232, 592, imgdata.sizes.raw_width - 1, 3920,
                                  16, 32, 2);
      }
      else if (imgdata.sizes.raw_width == 3328 &&
               imgdata.sizes.raw_height == 2240) // sd Quattro H half size
      {
        x3f_dpq_interpolate_af_sd(116, 296, imgdata.sizes.raw_width - 1, 2200,
                                  8, 16, 1);
      }
      else if (imgdata.sizes.raw_width == 5504 &&
               imgdata.sizes.raw_height == 3680) // sd Quattro H APS-C raw
      {
        x3f_dpq_interpolate_af_sd(8, 192, imgdata.sizes.raw_width - 1, 3185, 16,
                                  32, 2);
      }
      else if (imgdata.sizes.raw_width == 2752 &&
               imgdata.sizes.raw_height == 1840) // sd Quattro H APS-C half size
      {
        x3f_dpq_interpolate_af_sd(4, 96, imgdata.sizes.raw_width - 1, 1800, 8,
                                  16, 1);
      }
      else if (imgdata.sizes.raw_width == 2944 &&
               imgdata.sizes.raw_height == 1836) // dpN Quattro small raw
      {
        x3f_dpq_interpolate_af(16, 4, 1);
      }
      else if (imgdata.sizes.raw_width == 2944 &&
               imgdata.sizes.raw_height == 1888) // sd Quattro small
      {
        x3f_dpq_interpolate_af_sd(108, 232, imgdata.sizes.raw_width - 1, 1656,
                                  8, 16, 1);
      }
    }
#endif
    if (TRU && Q && Q->quattro_layout &&
        !(imgdata.rawparams.specials & LIBRAW_RAWSPECIAL_NODP2Q_INTERPOLATERG))
      x3f_dpq_interpolate_rg();
  }
  else
    raise_error = 1;
end:
  if (raise_error)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
}
#endif
