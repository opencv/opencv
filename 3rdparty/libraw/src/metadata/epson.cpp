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

void LibRaw::parseEpsonMakernote(int base, int uptag, unsigned dng_writer)
{

#define isRIC imgdata.sizes.raw_inset_crops[0]

  unsigned entries, tag, type, len, save;
  short morder, sorder = order;
  ushort c;
  INT64 fsize = ifp->size();

  fseek(ifp, -2, SEEK_CUR);

  entries = get2();
  if (entries > 1000)
    return;
  morder = order;

  while (entries--)
  {
    order = morder;
    tiff_get(base, &tag, &type, &len, &save);
    INT64 pos = ifp->tell();
    if (len > 8 && pos + len > 2 * fsize)
    {
      fseek(ifp, save, SEEK_SET); // Recover tiff-read position!!
      continue;
    }

    tag |= uptag << 16;
    if (len > 100 * 1024 * 1024)
      goto next; // 100Mb tag? No!

    if (tag == 0x020b)
    {
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
        isRIC.cwidth = get4();
      else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT))
        isRIC.cwidth = get2();
    }
    else if (tag == 0x020c)
    {
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
        isRIC.cheight = get4();
      else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT))
        isRIC.cheight = get2();
    }
    else if (tag == 0x0400)
    {                                 // sensor area
      ushort sdims[4] = {0, 0, 0, 0}; // left margin, top margin, width, height
      FORC4 sdims[c] = get2();
      isRIC.cleft = (sdims[2] - sdims[0] - isRIC.cwidth) / 2;
      isRIC.ctop = (sdims[3] - sdims[1] - isRIC.cheight) / 2;
    }

    if (dng_writer == nonDNG)
    {

      if (tag == 0x0280)
      {
        thumb_offset = ftell(ifp);
        thumb_length = len;
      }
      else if (tag == 0x0401)
      {
        FORC4 cblack[RGGB_2_RGBG(c)] = get4();
      }
      else if (tag == 0x0e80)
      {
        fseek(ifp, 48, SEEK_CUR);
        cam_mul[0] = get2() * 567.0 / 0x10000;
        cam_mul[2] = get2() * 431.0 / 0x10000;
      }
    }

  next:
    fseek(ifp, save, SEEK_SET);
  }
  order = sorder;
#undef isRIC
}
