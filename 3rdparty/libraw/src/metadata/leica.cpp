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

void LibRaw::setLeicaBodyFeatures(int LeicaMakernoteSignature)
{
  if (LeicaMakernoteSignature == -3) // M8
  {
    ilm.CameraFormat = LIBRAW_FORMAT_APSH;
    ilm.CameraMount = LIBRAW_MOUNT_Leica_M;
  }
  else if (LeicaMakernoteSignature == -2) // DMR
  {
    ilm.CameraFormat = LIBRAW_FORMAT_Leica_DMR;
    if ((model[0] == 'R') || (model[6] == 'R'))
      ilm.CameraMount = LIBRAW_MOUNT_Leica_R;
  }
  else if (LeicaMakernoteSignature == 0) // "DIGILUX 2"
  {
    ilm.CameraMount = ilm.LensMount = LIBRAW_MOUNT_FixedLens;
    ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
  }
  else if ((LeicaMakernoteSignature == 0x0100) || // X1
           (LeicaMakernoteSignature == 0x0500) || // X2, "X-E (Typ 102)"
           (LeicaMakernoteSignature == 0x0700) || // "X (Typ 113)"
           (LeicaMakernoteSignature == 0x1000))   // "X-U (Typ 113)"
  {
    ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_APSC;
    ilm.CameraMount = ilm.LensMount = LIBRAW_MOUNT_FixedLens;
    ilm.FocalType = LIBRAW_FT_PRIME_LENS;
  }
  else if (LeicaMakernoteSignature == 0x0400) // "X VARIO (Typ 107)"
  {
    ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_APSC;
    ilm.CameraMount = ilm.LensMount = LIBRAW_MOUNT_FixedLens;
    ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
  }
  else if ((LeicaMakernoteSignature ==
            0x0200) || // M10, M10-D, M10-R, "S (Typ 007)", M11
           (LeicaMakernoteSignature ==
            0x02ff) || // "M (Typ 240)", "M (Typ 262)", "M-D (Typ 262)",
                       // "M Monochrom (Typ 246)", "S (Typ 006)", "S-E (Typ 006)", S2, S3
           (LeicaMakernoteSignature ==
            0x0300))   // M9, "M9 Monochrom", "M Monochrom", M-E
  {
    if ((model[0] == 'M') || (model[6] == 'M'))
    {
      ilm.CameraFormat = LIBRAW_FORMAT_FF;
      ilm.CameraMount = LIBRAW_MOUNT_Leica_M;
    }
    else if ((model[0] == 'S') || (model[6] == 'S'))
    {
      ilm.CameraFormat = LIBRAW_FORMAT_LeicaS;
      ilm.CameraMount = LIBRAW_MOUNT_Leica_S;
    }
  }
  else if ((LeicaMakernoteSignature == 0x0600) || // "T (Typ 701)", TL
           (LeicaMakernoteSignature == 0x0900) || // SL2, "SL2-S", "SL (Typ 601)", CL, Q2, "Q2 MONO"
           (LeicaMakernoteSignature == 0x1a00))   // TL2
  {
    if ((model[0] == 'S') || (model[6] == 'S'))
    {
      ilm.CameraFormat = LIBRAW_FORMAT_FF;
      ilm.CameraMount = LIBRAW_MOUNT_LPS_L;
    }
    else if ((model[0] == 'T') || (model[6] == 'T') ||
             (model[0] == 'C') || (model[6] == 'C'))
    {
      ilm.CameraFormat = LIBRAW_FORMAT_APSC;
      ilm.CameraMount = LIBRAW_MOUNT_LPS_L;
    }
    else if (((model[0] == 'Q') || (model[6] == 'Q')) &&
             ((model[1] == '2') || (model[7] == '2')))
    {
      ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_FF;
      ilm.CameraMount = ilm.LensMount = LIBRAW_MOUNT_FixedLens;
      ilm.FocalType = LIBRAW_FT_PRIME_LENS;
    }
  }
  else if (LeicaMakernoteSignature == 0x0800) // "Q (Typ 116)"
  {
    ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_FF;
    ilm.CameraMount = ilm.LensMount = LIBRAW_MOUNT_FixedLens;
    ilm.FocalType = LIBRAW_FT_PRIME_LENS;
  }
}

void LibRaw::parseLeicaLensID()
{
  ilm.LensID = get4();
  if (ilm.LensID)
  {
    ilm.LensID = ((ilm.LensID >> 2) << 8) | (ilm.LensID & 0x3);
    if ((ilm.LensID > 0x00ff) && (ilm.LensID < 0x3b00))
    {
      ilm.LensMount = ilm.CameraMount;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
    }
  }
}

int LibRaw::parseLeicaLensName(unsigned len)
{
#define plln ilm.Lens
  if (!len)
  {
    strcpy(plln, "N/A");
    return 0;
  }
  stmread(plln, len, ifp);
  if ((plln[0] == ' ') || !strncasecmp(plln, "not ", 4) ||
      !strncmp(plln, "---", 3) || !strncmp(plln, "***", 3))
  {
    strcpy(plln, "N/A");
    return 0;
  }
  else
    return 1;
#undef plln
}

int LibRaw::parseLeicaInternalBodySerial(unsigned len)
{
#define plibs imgdata.shootinginfo.InternalBodySerial
  if (!len)
  {
    strcpy(plibs, "N/A");
    return 0;
  }
  stmread(plibs, len, ifp);
  if (!strncmp(plibs, "000000000000", 12))
  {
    plibs[0] = '0';
    plibs[1] = '\0';
    return 1;
  }

  if (strnlen(plibs, len) == 13)
  {
    for (int i = 3; i < 13; i++)
    {
      if (!isdigit(plibs[i]))
        goto non_std;
    }
    memcpy(plibs + 15, plibs + 9, 4);
    memcpy(plibs + 12, plibs + 7, 2);
    memcpy(plibs + 9, plibs + 5, 2);
    memcpy(plibs + 6, plibs + 3, 2);
    plibs[3] = plibs[14] = ' ';
    plibs[8] = plibs[11] = '/';
    if (((short)(plibs[3] - '0') * 10 + (short)(plibs[4] - '0')) < 70)
    {
      memcpy(plibs + 4, "20", 2);
    }
    else
    {
      memcpy(plibs + 4, "19", 2);
    }
    return 2;
  }
non_std:
#undef plibs
  return 1;
}

void LibRaw::parseLeicaMakernote(int base, int uptag, unsigned MakernoteTagType)
{
  int c;
  uchar ci, cj;
  unsigned entries, tag, type, len, save;
  short morder, sorder = order;
  char buf[10];
  int LeicaMakernoteSignature = -1;
  INT64 fsize = ifp->size();

  fread(buf, 1, 10, ifp);
  if (strncmp(buf, "LEICA", 5))
  {
    fseek(ifp, -10, SEEK_CUR);
    if (uptag == 0x3400)
      LeicaMakernoteSignature = 0x3400;
    else
      LeicaMakernoteSignature = -2; // DMR
  }
  else
  {
    fseek(ifp, -2, SEEK_CUR);
    LeicaMakernoteSignature = ((uchar)buf[6] << 8) | (uchar)buf[7];
    // printf ("LeicaMakernoteSignature 0x%04x\n", LeicaMakernoteSignature);
    if (!LeicaMakernoteSignature &&
        (!strncmp(model, "M8", 2) || !strncmp(model + 6, "M8", 2)))
      LeicaMakernoteSignature = -3;
    if ((LeicaMakernoteSignature != 0x0000) &&
        (LeicaMakernoteSignature != 0x0200) &&
        (LeicaMakernoteSignature != 0x0800) &&
        (LeicaMakernoteSignature != 0x0900) &&
        (LeicaMakernoteSignature != 0x02ff))
      base = ftell(ifp) - 8;
  }
  setLeicaBodyFeatures(LeicaMakernoteSignature);

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

    if (LeicaMakernoteSignature == -3) // M8
    {
      if (tag == 0x0310)
      {
        parseLeicaLensID();
      }
      else if ((tag == 0x0313) && (fabs(ilm.CurAp) < 0.17f))
      {
        ilm.CurAp = getreal(type);
        if (ilm.CurAp > 126.3)
        {
          ilm.CurAp = 0.0f;
        } else if (fabs(aperture) < 0.17f)
           aperture = ilm.CurAp;
      }
      else if (tag == 0x0320)
      {
        imCommon.CameraTemperature = getreal(type);
      }
    }
    else if (LeicaMakernoteSignature == -2) // DMR
    {
      if (tag == 0x000d)
      {
        FORC3 cam_mul[c] = get2();
        cam_mul[3] = cam_mul[1];
      }
    }
    else if (LeicaMakernoteSignature == 0) // "DIGILUX 2"
    {
      if (tag == 0x0007)
      {
        imgdata.shootinginfo.FocusMode = get2();
      }
      else if (tag == 0x001a)
      {
        imgdata.shootinginfo.ImageStabilization = get2();
      }
    }
    else if ((LeicaMakernoteSignature == 0x0100) || // X1
             (LeicaMakernoteSignature == 0x0400) || // "X VARIO"
             (LeicaMakernoteSignature == 0x0500) || // X2, "X-E (Typ 102)"
             (LeicaMakernoteSignature == 0x0700) || // "X (Typ 113)"
             (LeicaMakernoteSignature == 0x1000))   // "X-U (Typ 113)"
    {
      if (tag == 0x040d)
      {
        ci = fgetc(ifp);
        cj = fgetc(ifp);
        imgdata.shootinginfo.ExposureMode = ((ushort)ci << 8) | cj;
      }
    }
    else if ((LeicaMakernoteSignature == 0x0600) || // TL, "T (Typ 701)"
             (LeicaMakernoteSignature == 0x1a00))   // TL2
    {
      if (tag == 0x040d)
      {
        ci = fgetc(ifp);
        cj = fgetc(ifp);
        imgdata.shootinginfo.ExposureMode = ((ushort)ci << 8) | cj;
      }
      else if (tag == 0x0303)
      {
        parseLeicaLensName(len);
      }
    }
    else if (LeicaMakernoteSignature == 0x0200) // M10, M10-D, M10-R, "S (Typ 007)", M11
    {
      if ((tag == 0x035a) && (fabs(ilm.CurAp) < 0.17f))
      {
        ilm.CurAp = get4() / 1000.0f;
        if (ilm.CurAp > 126.3)
        {
          ilm.CurAp = 0.0f;
        } else if (fabs(aperture) < 0.17f)
           aperture = ilm.CurAp;
      }
    }
    else if (LeicaMakernoteSignature == 0x02ff)
                     // "M (Typ 240)", "M (Typ 262)", "M-D (Typ 262)"
                     // "M Monochrom (Typ 246)"
                     // "S (Typ 006)", "S-E (Typ 006)", S2, S3
    {
      if (tag == 0x0303)
      {
        if (parseLeicaLensName(len))
        {
          ilm.LensMount = ilm.CameraMount;
          ilm.LensFormat = ilm.CameraFormat;
        }
      }
    }
    else if (LeicaMakernoteSignature == 0x0300) // M9, "M9 Monochrom", "M Monochrom", M-E
    {
      if (tag == 0x3400)
      {
        parseLeicaMakernote(base, 0x3400, MakernoteTagType);
      }
    }
    else if ((LeicaMakernoteSignature == 0x0800) || // "Q (Typ 116)"
             (LeicaMakernoteSignature == 0x0900))   // SL2, "SL2-S", "SL (Typ 601)",
                                                    // CL, Q2, "Q2 MONO"
    {
      if ((tag == 0x0304) && (len == 1) && ((c = fgetc(ifp)) != 0) &&
          (ilm.CameraMount == LIBRAW_MOUNT_LPS_L))
      {
        strcpy(ilm.Adapter, "M-Adapter L");
        ilm.LensMount = LIBRAW_MOUNT_Leica_M;
        ilm.LensFormat = LIBRAW_FORMAT_FF;
        if (c != 0xff) ilm.LensID = c * 256;
      }
      else if (tag == 0x0500)
      {
        parseLeicaInternalBodySerial(len);
      }
    }
    else if (LeicaMakernoteSignature == 0x3400) // tag 0x3400 in M9, "M9 Monochrom", "M Monochrom"
    {
      if (tag == 0x34003402)
      {
        imCommon.CameraTemperature = getreal(type);
      }
      else if (tag == 0x34003405)
      {
        parseLeicaLensID();
      }
      else if ((tag == 0x34003406) && (fabs(ilm.CurAp) < 0.17f))
      {
        ilm.CurAp = getreal(type);
        if (ilm.CurAp > 126.3)
        {
          ilm.CurAp = 0.0f;
        } else if (fabs(aperture) < 0.17f)
           aperture = ilm.CurAp;
      }
    }

  next:
    fseek(ifp, save, SEEK_SET);
  }
  order = sorder;
}
