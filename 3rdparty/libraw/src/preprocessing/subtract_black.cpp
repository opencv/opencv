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

int LibRaw::subtract_black()
{
  adjust_bl();
  return subtract_black_internal();
}

int LibRaw::subtract_black_internal()
{
  CHECK_ORDER_LOW(LIBRAW_PROGRESS_RAW2_IMAGE);

  try
  {
    if (!is_phaseone_compressed() &&
        (C.cblack[0] || C.cblack[1] || C.cblack[2] || C.cblack[3] ||
         (C.cblack[4] && C.cblack[5])))
    {
      int cblk[4], i;
      for (i = 0; i < 4; i++)
        cblk[i] = C.cblack[i];

      int size = S.iheight * S.iwidth;
      int dmax = 0;
      if (C.cblack[4] && C.cblack[5])
      {
        for (unsigned q = 0; q < (unsigned)size; q++)
        {
          for (unsigned c = 0; c < 4; c++)
          {
            int val = imgdata.image[q][c];
            val -= C.cblack[6 + q / S.iwidth % C.cblack[4] * C.cblack[5] +
                            q % S.iwidth % C.cblack[5]];
            val -= cblk[c];
            imgdata.image[q][c] = CLIP(val);
            if (dmax < val) dmax = val;
          }
        }
      }
      else
      {
        for (unsigned q = 0; q < (unsigned)size; q++)
        {
          for (unsigned c = 0; c < 4; c++)
          {
            int val = imgdata.image[q][c];
            val -= cblk[c];
            imgdata.image[q][c] = CLIP(val);
            if (dmax < val) dmax = val;
          }
        }
      }
      C.data_maximum = dmax & 0xffff;
      C.maximum -= C.black;
      ZERO(C.cblack); // Yeah, we used cblack[6+] values too!
      C.black = 0;
    }
    else
    {
      // Nothing to Do, maximum is already calculated, black level is 0, so no
      // change only calculate channel maximum;
      int idx;
      ushort *p = (ushort *)imgdata.image;
      int dmax = 0;
      for (idx = 0; idx < S.iheight * S.iwidth * 4; idx++)
        if (dmax < p[idx])
          dmax = p[idx];
      C.data_maximum = dmax;
    }
    return 0;
  }
  catch (const LibRaw_exceptions& err)
  {
    EXCEPTION_HANDLER(err);
  }
}
