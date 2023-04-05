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

#include "../../internal/libraw_cxx_defs.h"

void LibRaw::phase_one_allocate_tempbuffer()
{
  // Allocate temp raw_image buffer
  imgdata.rawdata.raw_image = (ushort *)malloc(S.raw_pitch * S.raw_height);
}
void LibRaw::phase_one_free_tempbuffer()
{
  free(imgdata.rawdata.raw_image);
  imgdata.rawdata.raw_image = (ushort *)imgdata.rawdata.raw_alloc;
}

int LibRaw::phase_one_subtract_black(ushort *src, ushort *dest)
{

  try
  {
    if (O.user_black < 0 && O.user_cblack[0] <= -1000000 &&
        O.user_cblack[1] <= -1000000 && O.user_cblack[2] <= -1000000 &&
        O.user_cblack[3] <= -1000000)
    {
      if (!imgdata.rawdata.ph1_cblack || !imgdata.rawdata.ph1_rblack)
      {
        int bl = imgdata.color.phase_one_data.t_black;
        for (int row = 0; row < S.raw_height; row++)
        {
          checkCancel();
          for (int col = 0; col < S.raw_width; col++)
          {
            int idx = row * S.raw_width + col;
            int val = int(src[idx]) - bl;
            dest[idx] = val > 0 ? val : 0;
          }
        }
      }
      else
      {
        int bl = imgdata.color.phase_one_data.t_black;
        for (int row = 0; row < S.raw_height; row++)
        {
          checkCancel();
          for (int col = 0; col < S.raw_width; col++)
          {
            int idx = row * S.raw_width + col;
            int val =
                int(src[idx]) - bl +
                imgdata.rawdata
                    .ph1_cblack[row][col >= imgdata.rawdata.color.phase_one_data
                                                .split_col] +
                imgdata.rawdata
                    .ph1_rblack[col][row >= imgdata.rawdata.color.phase_one_data
                                                .split_row];
            dest[idx] = val > 0 ? val : 0;
          }
        }
      }
    }
    else // black set by user interaction
    {
      // Black level in cblack!
      for (int row = 0; row < S.raw_height; row++)
      {
        checkCancel();
        unsigned short cblk[16];
        for (int cc = 0; cc < 16; cc++)
          cblk[cc] = C.cblack[fcol(row, cc)];
        for (int col = 0; col < S.raw_width; col++)
        {
          int idx = row * S.raw_width + col;
          ushort val = src[idx];
          ushort bl = cblk[col & 0xf];
          dest[idx] = val > bl ? val - bl : 0;
        }
      }
    }
    return 0;
  }
  catch (const LibRaw_exceptions& )
  {
    return LIBRAW_CANCELLED_BY_CALLBACK;
  }
}
