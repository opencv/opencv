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

void LibRaw::raw2image_start()
{
  // restore color,sizes and internal data into raw_image fields
  memmove(&imgdata.color, &imgdata.rawdata.color, sizeof(imgdata.color));
  memmove(&imgdata.sizes, &imgdata.rawdata.sizes, sizeof(imgdata.sizes));
  memmove(&imgdata.idata, &imgdata.rawdata.iparams, sizeof(imgdata.idata));
  memmove(&libraw_internal_data.internal_output_params,
          &imgdata.rawdata.ioparams,
          sizeof(libraw_internal_data.internal_output_params));

  if (O.user_flip >= 0)
    S.flip = O.user_flip;

  switch ((S.flip + 3600) % 360)
  {
  case 270:
    S.flip = 5;
    break;
  case 180:
    S.flip = 3;
    break;
  case 90:
    S.flip = 6;
    break;
  }

  // adjust for half mode!
  IO.shrink =
	  !imgdata.rawdata.color4_image && !imgdata.rawdata.color3_image &&
	  !imgdata.rawdata.float4_image && !imgdata.rawdata.float3_image &&
      P1.filters &&
      (O.half_size || ((O.threshold || O.aber[0] != 1 || O.aber[2] != 1)));

  S.iheight = (S.height + IO.shrink) >> IO.shrink;
  S.iwidth = (S.width + IO.shrink) >> IO.shrink;
}

int LibRaw::raw2image(void)
{

  CHECK_ORDER_LOW(LIBRAW_PROGRESS_LOAD_RAW);

  try
  {
    raw2image_start();

    if (is_phaseone_compressed() && imgdata.rawdata.raw_alloc)
    {
      phase_one_allocate_tempbuffer();
      int rc = phase_one_subtract_black((ushort *)imgdata.rawdata.raw_alloc,
                                        imgdata.rawdata.raw_image);
      if (rc == 0)
        rc = phase_one_correct();
      if (rc != 0)
      {
        phase_one_free_tempbuffer();
        return rc;
      }
    }

    // free and re-allocate image bitmap
    if (imgdata.image)
    {
      imgdata.image = (ushort(*)[4])realloc(
          imgdata.image, S.iheight * S.iwidth * sizeof(*imgdata.image));
      memset(imgdata.image, 0, S.iheight * S.iwidth * sizeof(*imgdata.image));
    }
    else
      imgdata.image =
          (ushort(*)[4])calloc(S.iheight * S.iwidth, sizeof(*imgdata.image));


    libraw_decoder_info_t decoder_info;
    get_decoder_info(&decoder_info);

    // Copy area size
    int copyheight = MAX(0, MIN(int(S.height), int(S.raw_height) - int(S.top_margin)));
    int copywidth = MAX(0, MIN(int(S.width), int(S.raw_width) - int(S.left_margin)));

    // Move saved bitmap to imgdata.image
    if ((imgdata.idata.filters || P1.colors == 1) && imgdata.rawdata.raw_image)
    {
      if (IO.fuji_width)
      {
        unsigned r, c;
        int row, col;
        for (row = 0; row < S.raw_height - S.top_margin * 2; row++)
        {
          for (col = 0;
               col < IO.fuji_width
                         << int(!libraw_internal_data.unpacker_data.fuji_layout);
               col++)
          {
            if (libraw_internal_data.unpacker_data.fuji_layout)
            {
              r = IO.fuji_width - 1 - col + (row >> 1);
              c = col + ((row + 1) >> 1);
            }
            else
            {
              r = IO.fuji_width - 1 + row - (col >> 1);
              c = row + ((col + 1) >> 1);
            }
            if (r < S.height && c < S.width && col + int(S.left_margin) < int(S.raw_width))
              imgdata.image[((r) >> IO.shrink) * S.iwidth + ((c) >> IO.shrink)]
                           [FC(r, c)] =
                  imgdata.rawdata
                      .raw_image[(row + S.top_margin) * S.raw_pitch / 2 +
                                 (col + S.left_margin)];
          }
        }
      }
      else
      {
        int row, col;
        for (row = 0; row < copyheight; row++)
          for (col = 0; col < copywidth; col++)
            imgdata.image[((row) >> IO.shrink) * S.iwidth +
                          ((col) >> IO.shrink)][fcol(row, col)] =
                imgdata.rawdata
                    .raw_image[(row + S.top_margin) * S.raw_pitch / 2 +
                               (col + S.left_margin)];
      }
    }
    else // if(decoder_info.decoder_flags & LIBRAW_DECODER_LEGACY)
    {
      if (imgdata.rawdata.color4_image)
      {
        if (S.width * 8u == S.raw_pitch && S.height == S.raw_height)
          memmove(imgdata.image, imgdata.rawdata.color4_image,
                  S.width * S.height * sizeof(*imgdata.image));
        else
        {
            for (int row = 0; row < copyheight; row++)
            memmove(&imgdata.image[row * S.width],
                    &imgdata.rawdata
                         .color4_image[(row + S.top_margin) * S.raw_pitch / 8 +
                                       S.left_margin],
                    copywidth * sizeof(*imgdata.image));
        }
      }
      else if (imgdata.rawdata.color3_image)
      {
        unsigned char *c3image = (unsigned char *)imgdata.rawdata.color3_image;
        for (int row = 0; row < copyheight; row++)
        {
          ushort(*srcrow)[3] =
              (ushort(*)[3]) & c3image[(row + S.top_margin) * S.raw_pitch];
          ushort(*dstrow)[4] = (ushort(*)[4]) & imgdata.image[row * S.width];
          for (int col = 0; col < copywidth; col++)
          {
            for (int c = 0; c < 3; c++)
              dstrow[col][c] = srcrow[S.left_margin + col][c];
            dstrow[col][3] = 0;
          }
        }
      }
      else
      {
        // legacy decoder, but no data?
        throw LIBRAW_EXCEPTION_DECODE_RAW;
      }
    }

    // Free PhaseOne separate copy allocated at function start
    if (is_phaseone_compressed())
    {
      phase_one_free_tempbuffer();
    }
    // hack - clear later flags!

    if (load_raw == &LibRaw::canon_600_load_raw && S.width < S.raw_width)
    {
      canon_600_correct();
    }

    imgdata.progress_flags =
        LIBRAW_PROGRESS_START | LIBRAW_PROGRESS_OPEN |
        LIBRAW_PROGRESS_RAW2_IMAGE | LIBRAW_PROGRESS_IDENTIFY |
        LIBRAW_PROGRESS_SIZE_ADJUST | LIBRAW_PROGRESS_LOAD_RAW;
    return 0;
  }
  catch (const std::bad_alloc&)
  {
      EXCEPTION_HANDLER(LIBRAW_EXCEPTION_ALLOC);
  }
  catch (const LibRaw_exceptions& err)
  {
    EXCEPTION_HANDLER(err);
  }
}

void LibRaw::copy_fuji_uncropped(unsigned short cblack[4],
                                 unsigned short *dmaxp)
{
#if defined(LIBRAW_USE_OPENMP)
#pragma omp parallel for schedule(dynamic) default(none) firstprivate(cblack) shared(dmaxp)
#endif
  for (int row = 0; row < int(S.raw_height) - int(S.top_margin) * 2; row++)
  {
    int col;
    unsigned short ldmax = 0;
    for (col = 0;
         col < IO.fuji_width << int(!libraw_internal_data.unpacker_data.fuji_layout)
         && col + int(S.left_margin) < int(S.raw_width);
         col++)
    {
      unsigned r, c;
      if (libraw_internal_data.unpacker_data.fuji_layout)
      {
        r = IO.fuji_width - 1 - col + (row >> 1);
        c = col + ((row + 1) >> 1);
      }
      else
      {
        r = IO.fuji_width - 1 + row - (col >> 1);
        c = row + ((col + 1) >> 1);
      }
      if (r < S.height && c < S.width)
      {
        unsigned short val =
            imgdata.rawdata.raw_image[(row + S.top_margin) * S.raw_pitch / 2 +
                                      (col + S.left_margin)];
        int cc = FC(r, c);
        if (val > cblack[cc])
        {
          val -= cblack[cc];
          if (val > ldmax)
            ldmax = val;
        }
        else
          val = 0;
        imgdata.image[((r) >> IO.shrink) * S.iwidth + ((c) >> IO.shrink)][cc] =
            val;
      }
    }
#if defined(LIBRAW_USE_OPENMP)
#pragma omp critical(dataupdate)
#endif
    {
      if (*dmaxp < ldmax)
        *dmaxp = ldmax;
    }
  }
}

void LibRaw::copy_bayer(unsigned short cblack[4], unsigned short *dmaxp)
{
  // Both cropped and uncropped
  int maxHeight = MIN(int(S.height),int(S.raw_height)-int(S.top_margin));
#if defined(LIBRAW_USE_OPENMP)
#pragma omp parallel for schedule(dynamic) default(none) shared(dmaxp) firstprivate(cblack, maxHeight)
#endif
  for (int row = 0; row < maxHeight ; row++)
  {
    int col;
    unsigned short ldmax = 0;
    for (col = 0; col < S.width && col + S.left_margin < S.raw_width; col++)
    {
      unsigned short val =
          imgdata.rawdata.raw_image[(row + S.top_margin) * S.raw_pitch / 2 +
                                    (col + S.left_margin)];
      int cc = fcol(row, col);
      if (val > cblack[cc])
      {
        val -= cblack[cc];
        if (val > ldmax)
          ldmax = val;
      }
      else
        val = 0;
      imgdata.image[((row) >> IO.shrink) * S.iwidth + ((col) >> IO.shrink)][cc] = val;
    }
#if defined(LIBRAW_USE_OPENMP)
#pragma omp critical(dataupdate)
#endif
    {
      if (*dmaxp < ldmax)
        *dmaxp = ldmax;
    }
  }
}

int LibRaw::raw2image_ex(int do_subtract_black)
{

  CHECK_ORDER_LOW(LIBRAW_PROGRESS_LOAD_RAW);

  try
  {
    raw2image_start();

    // Compressed P1 files with bl data!
    if (is_phaseone_compressed() && imgdata.rawdata.raw_alloc)
    {
      phase_one_allocate_tempbuffer();
      int rc = phase_one_subtract_black((ushort *)imgdata.rawdata.raw_alloc,
                                        imgdata.rawdata.raw_image);
      if (rc == 0)
        rc = phase_one_correct();
      if (rc != 0)
      {
        phase_one_free_tempbuffer();
        return rc;
      }
    }

    // process cropping
    int do_crop = 0;
    if (~O.cropbox[2] && ~O.cropbox[3])
    {
      int crop[4], c, filt;
      for (int q = 0; q < 4; q++)
      {
        crop[q] = O.cropbox[q];
        if (crop[q] < 0)
          crop[q] = 0;
      }

      if (IO.fuji_width && imgdata.idata.filters >= 1000)
      {
        crop[0] = (crop[0] / 4) * 4;
        crop[1] = (crop[1] / 4) * 4;
        if (!libraw_internal_data.unpacker_data.fuji_layout)
        {
          crop[2] *= sqrt(2.0);
          crop[3] /= sqrt(2.0);
        }
        crop[2] = (crop[2] / 4 + 1) * 4;
        crop[3] = (crop[3] / 4 + 1) * 4;
      }
      else if (imgdata.idata.filters == 1)
      {
        crop[0] = (crop[0] / 16) * 16;
        crop[1] = (crop[1] / 16) * 16;
      }
      else if (imgdata.idata.filters == LIBRAW_XTRANS)
      {
        crop[0] = (crop[0] / 6) * 6;
        crop[1] = (crop[1] / 6) * 6;
      }
      do_crop = 1;

      crop[2] = MIN(crop[2], (signed)S.width - crop[0]);
      crop[3] = MIN(crop[3], (signed)S.height - crop[1]);
      if (crop[2] <= 0 || crop[3] <= 0)
        throw LIBRAW_EXCEPTION_BAD_CROP;

      // adjust sizes!
      S.left_margin += crop[0];
      S.top_margin += crop[1];
      S.width = crop[2];
      S.height = crop[3];

      S.iheight = (S.height + IO.shrink) >> IO.shrink;
      S.iwidth = (S.width + IO.shrink) >> IO.shrink;
      if (!IO.fuji_width && imgdata.idata.filters &&
          imgdata.idata.filters >= 1000)
      {
        for (filt = c = 0; c < 16; c++)
          filt |= FC((c >> 1) + (crop[1]), (c & 1) + (crop[0])) << c * 2;
        imgdata.idata.filters = filt;
      }
    }

    int alloc_width = S.iwidth;
    int alloc_height = S.iheight;

    if (IO.fuji_width && do_crop)
    {
      int IO_fw = S.width >> int(!libraw_internal_data.unpacker_data.fuji_layout);
      int t_alloc_width =
          (S.height >> libraw_internal_data.unpacker_data.fuji_layout) + IO_fw;
      int t_alloc_height = t_alloc_width - 1;
      alloc_height = (t_alloc_height + IO.shrink) >> IO.shrink;
      alloc_width = (t_alloc_width + IO.shrink) >> IO.shrink;
    }
    int alloc_sz = alloc_width * alloc_height;

    if (imgdata.image)
    {
      imgdata.image = (ushort(*)[4])realloc(imgdata.image,
                                            alloc_sz * sizeof(*imgdata.image));
      memset(imgdata.image, 0, alloc_sz * sizeof(*imgdata.image));
    }
    else
      imgdata.image = (ushort(*)[4])calloc(alloc_sz, sizeof(*imgdata.image));

    libraw_decoder_info_t decoder_info;
    get_decoder_info(&decoder_info);

    // Adjust black levels
    unsigned short cblack[4] = {0, 0, 0, 0};
    unsigned short dmax = 0;
    if (do_subtract_black)
    {
      adjust_bl();
      for (int i = 0; i < 4; i++)
        cblack[i] = (unsigned short)C.cblack[i];
    }

    // Max area size to definitely not overrun in/out buffers
    int copyheight = MAX(0, MIN(int(S.height), int(S.raw_height) - int(S.top_margin)));
    int copywidth = MAX(0, MIN(int(S.width), int(S.raw_width) - int(S.left_margin)));

    // Move saved bitmap to imgdata.image
    if ((imgdata.idata.filters || P1.colors == 1) && imgdata.rawdata.raw_image)
    {
      if (IO.fuji_width)
      {
        if (do_crop)
        {
          IO.fuji_width =
              S.width >> int(!libraw_internal_data.unpacker_data.fuji_layout);
          int IO_fwidth =
              (S.height >> int(libraw_internal_data.unpacker_data.fuji_layout)) +
              IO.fuji_width;
          int IO_fheight = IO_fwidth - 1;

          int row, col;
          for (row = 0; row < S.height; row++)
          {
            for (col = 0; col < S.width; col++)
            {
              int r, c;
              if (libraw_internal_data.unpacker_data.fuji_layout)
              {
                r = IO.fuji_width - 1 - col + (row >> 1);
                c = col + ((row + 1) >> 1);
              }
              else
              {
                r = IO.fuji_width - 1 + row - (col >> 1);
                c = row + ((col + 1) >> 1);
              }

              unsigned short val =
                  imgdata.rawdata
                      .raw_image[(row + S.top_margin) * S.raw_pitch / 2 +
                                 (col + S.left_margin)];
              int cc = FCF(row, col);
              if (val > cblack[cc])
              {
                val -= cblack[cc];
                if (dmax < val)
                  dmax = val;
              }
              else
                val = 0;
              imgdata.image[((r) >> IO.shrink) * alloc_width +
                            ((c) >> IO.shrink)][cc] = val;
            }
          }
          S.height = IO_fheight;
          S.width = IO_fwidth;
          S.iheight = (S.height + IO.shrink) >> IO.shrink;
          S.iwidth = (S.width + IO.shrink) >> IO.shrink;
          S.raw_height -= 2 * S.top_margin;
        }
        else
        {
          copy_fuji_uncropped(cblack, &dmax);
        }
      } // end Fuji
      else
      {
        copy_bayer(cblack, &dmax);
      }
    }
    else // if(decoder_info.decoder_flags & LIBRAW_DECODER_LEGACY)
    {
      if (imgdata.rawdata.color4_image)
      {
          if (S.raw_pitch != S.width * 8u || S.height != S.raw_height)
          {
              for (int row = 0; row < copyheight; row++)
                  memmove(&imgdata.image[row * S.width],
                      &imgdata.rawdata
                      .color4_image[(row + S.top_margin) * S.raw_pitch / 8 +
                      S.left_margin],
                      copywidth * sizeof(*imgdata.image));
          }
          else
          {
              // legacy is always 4channel and not shrinked!
              memmove(imgdata.image, imgdata.rawdata.color4_image,
                  S.width*copyheight * sizeof(*imgdata.image));
          }
      }
      else if (imgdata.rawdata.color3_image)
      {
          unsigned char *c3image = (unsigned char *)imgdata.rawdata.color3_image;
          for (int row = 0; row < copyheight; row++)
        {
          ushort(*srcrow)[3] =
              (ushort(*)[3]) & c3image[(row + S.top_margin) * S.raw_pitch];
          ushort(*dstrow)[4] = (ushort(*)[4]) & imgdata.image[row * S.width];
          for (int col = 0; col < copywidth; col++)
          {
            for (int c = 0; c < 3; c++)
              dstrow[col][c] = srcrow[S.left_margin + col][c];
            dstrow[col][3] = 0;
          }
        }
      }
      else
      {
        // legacy decoder, but no data?
        throw LIBRAW_EXCEPTION_DECODE_RAW;
      }
    }

    // Free PhaseOne separate copy allocated at function start
    if (is_phaseone_compressed())
    {
      phase_one_free_tempbuffer();
    }
    if (load_raw == &LibRaw::canon_600_load_raw && S.width < S.raw_width)
    {
      canon_600_correct();
    }

    if (do_subtract_black)
    {
      C.data_maximum = (int)dmax;
      C.maximum -= C.black;
      //        ZERO(C.cblack);
      C.cblack[0] = C.cblack[1] = C.cblack[2] = C.cblack[3] = 0;
      C.black = 0;
    }

    // hack - clear later flags!
    imgdata.progress_flags =
        LIBRAW_PROGRESS_START | LIBRAW_PROGRESS_OPEN |
        LIBRAW_PROGRESS_RAW2_IMAGE | LIBRAW_PROGRESS_IDENTIFY |
        LIBRAW_PROGRESS_SIZE_ADJUST | LIBRAW_PROGRESS_LOAD_RAW;
    return 0;
  }
  catch (const LibRaw_exceptions& err)
  {
    EXCEPTION_HANDLER(err);
  }
}
