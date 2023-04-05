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

// void hexDump(char *title, void *addr, int len);

unsigned sget4_order (short _order, uchar *s);
double sget_fixed32u (short _order, uchar *s);
double AngleConversion_a (short _order, uchar *s);
double AngleConversion (short _order, uchar *s);

static const uchar xlat[2][256] = {
    {0xc1, 0xbf, 0x6d, 0x0d, 0x59, 0xc5, 0x13, 0x9d, 0x83, 0x61, 0x6b, 0x4f,
     0xc7, 0x7f, 0x3d, 0x3d, 0x53, 0x59, 0xe3, 0xc7, 0xe9, 0x2f, 0x95, 0xa7,
     0x95, 0x1f, 0xdf, 0x7f, 0x2b, 0x29, 0xc7, 0x0d, 0xdf, 0x07, 0xef, 0x71,
     0x89, 0x3d, 0x13, 0x3d, 0x3b, 0x13, 0xfb, 0x0d, 0x89, 0xc1, 0x65, 0x1f,
     0xb3, 0x0d, 0x6b, 0x29, 0xe3, 0xfb, 0xef, 0xa3, 0x6b, 0x47, 0x7f, 0x95,
     0x35, 0xa7, 0x47, 0x4f, 0xc7, 0xf1, 0x59, 0x95, 0x35, 0x11, 0x29, 0x61,
     0xf1, 0x3d, 0xb3, 0x2b, 0x0d, 0x43, 0x89, 0xc1, 0x9d, 0x9d, 0x89, 0x65,
     0xf1, 0xe9, 0xdf, 0xbf, 0x3d, 0x7f, 0x53, 0x97, 0xe5, 0xe9, 0x95, 0x17,
     0x1d, 0x3d, 0x8b, 0xfb, 0xc7, 0xe3, 0x67, 0xa7, 0x07, 0xf1, 0x71, 0xa7,
     0x53, 0xb5, 0x29, 0x89, 0xe5, 0x2b, 0xa7, 0x17, 0x29, 0xe9, 0x4f, 0xc5,
     0x65, 0x6d, 0x6b, 0xef, 0x0d, 0x89, 0x49, 0x2f, 0xb3, 0x43, 0x53, 0x65,
     0x1d, 0x49, 0xa3, 0x13, 0x89, 0x59, 0xef, 0x6b, 0xef, 0x65, 0x1d, 0x0b,
     0x59, 0x13, 0xe3, 0x4f, 0x9d, 0xb3, 0x29, 0x43, 0x2b, 0x07, 0x1d, 0x95,
     0x59, 0x59, 0x47, 0xfb, 0xe5, 0xe9, 0x61, 0x47, 0x2f, 0x35, 0x7f, 0x17,
     0x7f, 0xef, 0x7f, 0x95, 0x95, 0x71, 0xd3, 0xa3, 0x0b, 0x71, 0xa3, 0xad,
     0x0b, 0x3b, 0xb5, 0xfb, 0xa3, 0xbf, 0x4f, 0x83, 0x1d, 0xad, 0xe9, 0x2f,
     0x71, 0x65, 0xa3, 0xe5, 0x07, 0x35, 0x3d, 0x0d, 0xb5, 0xe9, 0xe5, 0x47,
     0x3b, 0x9d, 0xef, 0x35, 0xa3, 0xbf, 0xb3, 0xdf, 0x53, 0xd3, 0x97, 0x53,
     0x49, 0x71, 0x07, 0x35, 0x61, 0x71, 0x2f, 0x43, 0x2f, 0x11, 0xdf, 0x17,
     0x97, 0xfb, 0x95, 0x3b, 0x7f, 0x6b, 0xd3, 0x25, 0xbf, 0xad, 0xc7, 0xc5,
     0xc5, 0xb5, 0x8b, 0xef, 0x2f, 0xd3, 0x07, 0x6b, 0x25, 0x49, 0x95, 0x25,
     0x49, 0x6d, 0x71, 0xc7},
    {0xa7, 0xbc, 0xc9, 0xad, 0x91, 0xdf, 0x85, 0xe5, 0xd4, 0x78, 0xd5, 0x17,
     0x46, 0x7c, 0x29, 0x4c, 0x4d, 0x03, 0xe9, 0x25, 0x68, 0x11, 0x86, 0xb3,
     0xbd, 0xf7, 0x6f, 0x61, 0x22, 0xa2, 0x26, 0x34, 0x2a, 0xbe, 0x1e, 0x46,
     0x14, 0x68, 0x9d, 0x44, 0x18, 0xc2, 0x40, 0xf4, 0x7e, 0x5f, 0x1b, 0xad,
     0x0b, 0x94, 0xb6, 0x67, 0xb4, 0x0b, 0xe1, 0xea, 0x95, 0x9c, 0x66, 0xdc,
     0xe7, 0x5d, 0x6c, 0x05, 0xda, 0xd5, 0xdf, 0x7a, 0xef, 0xf6, 0xdb, 0x1f,
     0x82, 0x4c, 0xc0, 0x68, 0x47, 0xa1, 0xbd, 0xee, 0x39, 0x50, 0x56, 0x4a,
     0xdd, 0xdf, 0xa5, 0xf8, 0xc6, 0xda, 0xca, 0x90, 0xca, 0x01, 0x42, 0x9d,
     0x8b, 0x0c, 0x73, 0x43, 0x75, 0x05, 0x94, 0xde, 0x24, 0xb3, 0x80, 0x34,
     0xe5, 0x2c, 0xdc, 0x9b, 0x3f, 0xca, 0x33, 0x45, 0xd0, 0xdb, 0x5f, 0xf5,
     0x52, 0xc3, 0x21, 0xda, 0xe2, 0x22, 0x72, 0x6b, 0x3e, 0xd0, 0x5b, 0xa8,
     0x87, 0x8c, 0x06, 0x5d, 0x0f, 0xdd, 0x09, 0x19, 0x93, 0xd0, 0xb9, 0xfc,
     0x8b, 0x0f, 0x84, 0x60, 0x33, 0x1c, 0x9b, 0x45, 0xf1, 0xf0, 0xa3, 0x94,
     0x3a, 0x12, 0x77, 0x33, 0x4d, 0x44, 0x78, 0x28, 0x3c, 0x9e, 0xfd, 0x65,
     0x57, 0x16, 0x94, 0x6b, 0xfb, 0x59, 0xd0, 0xc8, 0x22, 0x36, 0xdb, 0xd2,
     0x63, 0x98, 0x43, 0xa1, 0x04, 0x87, 0x86, 0xf7, 0xa6, 0x26, 0xbb, 0xd6,
     0x59, 0x4d, 0xbf, 0x6a, 0x2e, 0xaa, 0x2b, 0xef, 0xe6, 0x78, 0xb6, 0x4e,
     0xe0, 0x2f, 0xdc, 0x7c, 0xbe, 0x57, 0x19, 0x32, 0x7e, 0x2a, 0xd0, 0xb8,
     0xba, 0x29, 0x00, 0x3c, 0x52, 0x7d, 0xa8, 0x49, 0x3b, 0x2d, 0xeb, 0x25,
     0x49, 0xfa, 0xa3, 0xaa, 0x39, 0xa7, 0xc5, 0xa7, 0x50, 0x11, 0x36, 0xfb,
     0xc6, 0x67, 0x4a, 0xf5, 0xa5, 0x12, 0x65, 0x7e, 0xb0, 0xdf, 0xaf, 0x4e,
     0xb3, 0x61, 0x7f, 0x2f} };

void LibRaw::processNikonLensData(uchar *LensData, unsigned len)
{

  ushort i=0;
  if (imgdata.lens.nikon.LensType & 0x80) {
    strcpy (ilm.LensFeatures_pre, "AF-P");
  } else if (!(imgdata.lens.nikon.LensType & 0x01)) {
    ilm.LensFeatures_pre[0] = 'A';
    ilm.LensFeatures_pre[1] = 'F';
  } else {
    ilm.LensFeatures_pre[0] = 'M';
    ilm.LensFeatures_pre[1] = 'F';
  }

  if (imgdata.lens.nikon.LensType & 0x40) {
    ilm.LensFeatures_suf[0] = 'E';
  } else if (imgdata.lens.nikon.LensType & 0x04) {
    ilm.LensFeatures_suf[0] = 'G';
  } else if (imgdata.lens.nikon.LensType & 0x02) {
    ilm.LensFeatures_suf[0] = 'D';
  }

  if (imgdata.lens.nikon.LensType & 0x08)
  {
    ilm.LensFeatures_suf[1] = ' ';
    ilm.LensFeatures_suf[2] = 'V';
    ilm.LensFeatures_suf[3] = 'R';
  }

  if (imgdata.lens.nikon.LensType & 0x10)
  {
    ilm.LensMount = ilm.CameraMount = LIBRAW_MOUNT_Nikon_CX;
    ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_1INCH;
  }
  else
    ilm.LensMount = ilm.CameraMount = LIBRAW_MOUNT_Nikon_F;

  if (imgdata.lens.nikon.LensType & 0x20)
  {
    strcpy(ilm.Adapter, "FT-1");
    ilm.LensMount = LIBRAW_MOUNT_Nikon_F;
    ilm.CameraMount = LIBRAW_MOUNT_Nikon_CX;
    ilm.CameraFormat = LIBRAW_FORMAT_1INCH;
  }

  imgdata.lens.nikon.LensType = imgdata.lens.nikon.LensType & 0xdf;

  if ((len < 20) || (len == 58) || (len == 108))
  {
    switch (len)
    {
    case 9:
      i = 2;
      break;
    case 15:
      i = 7;
      break;
    case 16:
      i = 8;
      break;
    case  58: // "Z 6", "Z 6 II", "Z 7", "Z 7 II", "Z 50", D780, "Z 5", "Z fc"
    case 108: // "Z 9"
      if (model[6] == 'Z')
        ilm.CameraMount = LIBRAW_MOUNT_Nikon_Z;
      if (imNikon.HighSpeedCropFormat != 12)
        ilm.CameraFormat = LIBRAW_FORMAT_FF;
      i = 1;
      while ((LensData[i] == LensData[0]) && (i < 17))
        i++;
      if (i == 17)
      {
        ilm.LensMount = LIBRAW_MOUNT_Nikon_Z;
        ilm.LensID = sget2(LensData + 0x2c);
        if (
               (ilm.LensID == 11)
            || (ilm.LensID == 12)
            || (ilm.LensID == 26)
           ) ilm.LensFormat = LIBRAW_FORMAT_APSC;
        else ilm.LensFormat = LIBRAW_FORMAT_FF;
        if (ilm.MaxAp4CurFocal < 0.7f)
          ilm.MaxAp4CurFocal = libraw_powf64l(
              2.0f, (float)sget2(LensData + 0x32) / 384.0f - 1.0f);
        if (ilm.CurAp < 0.7f)
          ilm.CurAp = libraw_powf64l(
              2.0f, (float)sget2(LensData + 0x34) / 384.0f - 1.0f);
        if (fabsf(ilm.CurFocal) < 1.1f)
          ilm.CurFocal = sget2(LensData + 0x38);
        return;
      }
      i = 9;
      ilm.LensMount = LIBRAW_MOUNT_Nikon_F;
      if (ilm.CameraMount == LIBRAW_MOUNT_Nikon_Z)
        strcpy(ilm.Adapter, "FTZ");
      break;
    }
    imgdata.lens.nikon.LensIDNumber = LensData[i];
    imgdata.lens.nikon.LensFStops = LensData[i + 1];
    ilm.LensFStops = (float)imgdata.lens.nikon.LensFStops / 12.0f;
    if (fabsf(ilm.MinFocal) < 1.1f)
    {
      if ((imgdata.lens.nikon.LensType ^ (uchar)0x01) || LensData[i + 2])
        ilm.MinFocal =
            5.0f * libraw_powf64l(2.0f, (float)LensData[i + 2] / 24.0f);
      if ((imgdata.lens.nikon.LensType ^ (uchar)0x01) || LensData[i + 3])
        ilm.MaxFocal =
            5.0f * libraw_powf64l(2.0f, (float)LensData[i + 3] / 24.0f);
      if ((imgdata.lens.nikon.LensType ^ (uchar)0x01) || LensData[i + 4])
        ilm.MaxAp4MinFocal =
            libraw_powf64l(2.0f, (float)LensData[i + 4] / 24.0f);
      if ((imgdata.lens.nikon.LensType ^ (uchar)0x01) || LensData[i + 5])
        ilm.MaxAp4MaxFocal =
            libraw_powf64l(2.0f, (float)LensData[i + 5] / 24.0f);
    }
    imgdata.lens.nikon.MCUVersion = LensData[i + 6];
    if (i != 2)
    {
      if ((LensData[i - 1]) && (fabsf(ilm.CurFocal) < 1.1f))
        ilm.CurFocal =
            5.0f * libraw_powf64l(2.0f, (float)LensData[i - 1] / 24.0f);
      if (LensData[i + 7])
        imgdata.lens.nikon.EffectiveMaxAp =
            libraw_powf64l(2.0f, (float)LensData[i + 7] / 24.0f);
    }
    ilm.LensID = (unsigned long long)LensData[i] << 56 |
                 (unsigned long long)LensData[i + 1] << 48 |
                 (unsigned long long)LensData[i + 2] << 40 |
                 (unsigned long long)LensData[i + 3] << 32 |
                 (unsigned long long)LensData[i + 4] << 24 |
                 (unsigned long long)LensData[i + 5] << 16 |
                 (unsigned long long)LensData[i + 6] << 8 |
                 (unsigned long long)imgdata.lens.nikon.LensType;
  }
  else if ((len == 459) || (len == 590))
  {
    memcpy(ilm.Lens, LensData + 390, 64);
  }
  else if (len == 509)
  {
    memcpy(ilm.Lens, LensData + 391, 64);
  }
  else if (len == 879)
  {
    memcpy(ilm.Lens, LensData + 680, 64);
  }

  return;
}

void LibRaw::Nikon_NRW_WBtag(int wb, int skip)
{

  int r, g0, g1, b;
  if (skip)
    get4(); // skip wb "CCT", it is not unique
  r = get4();
  g0 = get4();
  g1 = get4();
  b = get4();
  if (r && g0 && g1 && b)
  {
    icWBC[wb][0] = r << 1;
    icWBC[wb][1] = g0;
    icWBC[wb][2] = b << 1;
    icWBC[wb][3] = g1;
  }
  return;
}

void LibRaw::parseNikonMakernote(int base, int uptag, unsigned /*dng_writer */)
{

  unsigned offset = 0, entries, tag, type, len, save;

  unsigned c, i;
  unsigned LensData_len = 0;
  uchar *LensData_buf=0;
  uchar ColorBalanceData_buf[324];
  int ColorBalanceData_ready = 0;
  uchar ci, cj, ck;
  unsigned serial = 0;
  unsigned custom_serial = 0;

  unsigned ShotInfo_len = 0;
  uchar *ShotInfo_buf=0;

/* for dump:
uchar *cj_block, *ck_block;
*/

  short morder, sorder = order;
  char buf[10];
  INT64 fsize = ifp->size();

  fread(buf, 1, 10, ifp);

  if (!strcmp(buf, "Nikon"))
  {
    if (buf[6] != '\2')
      return;
    base = ftell(ifp);
    order = get2();
    if (get2() != 42)
      goto quit;
    offset = get4();
    fseek(ifp, INT64(offset) - 8LL, SEEK_CUR);
  }
  else
  {
    fseek(ifp, -10, SEEK_CUR);
  }

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

    if (tag == 0x0002)
    {
      if (!iso_speed)
        iso_speed = (get2(), get2());
    }
    else if (tag == 0x000a)
    {
      ilm.LensMount = ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
    }
    else if ((tag == 0x000c) && (len == 4) && tagtypeIs(LIBRAW_EXIFTAG_TYPE_RATIONAL))
    {
      cam_mul[0] = getreal(type);
      cam_mul[2] = getreal(type);
      cam_mul[1] = getreal(type);
      cam_mul[3] = getreal(type);
    }
    else if (tag == 0x0011)
    {
      if (is_raw)
      {
        fseek(ifp, get4() + base, SEEK_SET);
        parse_tiff_ifd(base);
      }
    }
    else if (tag == 0x0012)
    {
      uchar uc1 = fgetc(ifp);
      uchar uc2 = fgetc(ifp);
      uchar uc3 = fgetc(ifp);
      if (uc3)
        imCommon.FlashEC = (float)(uc1 * uc2) / (float)uc3;
    }
    else if (tag == 0x0014)
    {
      if (tagtypeIs(LIBRAW_EXIFTOOLTAGTYPE_binary))
      {
        if (len == 2560)
        { // E5400, E8400, E8700, E8800
          fseek(ifp, 0x4e0L, SEEK_CUR);
          order = 0x4d4d;
          cam_mul[0] = get2() / 256.0;
          cam_mul[2] = get2() / 256.0;
          cam_mul[1] = cam_mul[3] = 1.0;
          icWBC[LIBRAW_WBI_Auto][0] = get2();
          icWBC[LIBRAW_WBI_Auto][2] = get2();
          icWBC[LIBRAW_WBI_Daylight][0] = get2();
          icWBC[LIBRAW_WBI_Daylight][2] = get2();
          fseek(ifp, 0x18L, SEEK_CUR);
          icWBC[LIBRAW_WBI_Tungsten][0] = get2();
          icWBC[LIBRAW_WBI_Tungsten][2] = get2();
          fseek(ifp, 0x18L, SEEK_CUR);
          icWBC[LIBRAW_WBI_FL_W][0] = get2();
          icWBC[LIBRAW_WBI_FL_W][2] = get2();
          icWBC[LIBRAW_WBI_FL_N][0] = get2();
          icWBC[LIBRAW_WBI_FL_N][2] = get2();
          icWBC[LIBRAW_WBI_FL_D][0] = get2();
          icWBC[LIBRAW_WBI_FL_D][2] = get2();
          icWBC[LIBRAW_WBI_Cloudy][0] = get2();
          icWBC[LIBRAW_WBI_Cloudy][2] = get2();
          fseek(ifp, 0x18L, SEEK_CUR);
          icWBC[LIBRAW_WBI_Flash][0] = get2();
          icWBC[LIBRAW_WBI_Flash][2] = get2();

          icWBC[LIBRAW_WBI_Auto][1] = icWBC[LIBRAW_WBI_Auto][3] =
            icWBC[LIBRAW_WBI_Daylight][1] = icWBC[LIBRAW_WBI_Daylight][3] =
            icWBC[LIBRAW_WBI_Tungsten][1] = icWBC[LIBRAW_WBI_Tungsten][3] =
            icWBC[LIBRAW_WBI_FL_W][1] = icWBC[LIBRAW_WBI_FL_W][3] =
            icWBC[LIBRAW_WBI_FL_N][1] = icWBC[LIBRAW_WBI_FL_N][3] =
            icWBC[LIBRAW_WBI_FL_D][1] = icWBC[LIBRAW_WBI_FL_D][3] =
            icWBC[LIBRAW_WBI_Cloudy][1] = icWBC[LIBRAW_WBI_Cloudy][3] =
            icWBC[LIBRAW_WBI_Flash][1] = icWBC[LIBRAW_WBI_Flash][3] = 256;

          if (strncmp(model, "E8700", 5))
          {
            fseek(ifp, 0x18L, SEEK_CUR);
            icWBC[LIBRAW_WBI_Shade][0] = get2();
            icWBC[LIBRAW_WBI_Shade][2] = get2();
            icWBC[LIBRAW_WBI_Shade][1] = icWBC[LIBRAW_WBI_Shade][3] = 256;
          }
        }
        else if (len == 1280)
        { // E5000, E5700
          cam_mul[0] = cam_mul[1] = cam_mul[2] = cam_mul[3] = 1.0;
        }
        else
        {
          fread(buf, 1, 10, ifp);
          if (!strncmp(buf, "NRW ", 4))
          { // P6000, P7000, P7100, B700, P1000
            if (!strcmp(buf + 4, "0100"))
            { // P6000
              fseek(ifp, 0x13deL, SEEK_CUR);
              cam_mul[0] = get4() << 1;
              cam_mul[1] = get4();
              cam_mul[3] = get4();
              cam_mul[2] = get4() << 1;
              Nikon_NRW_WBtag(LIBRAW_WBI_Daylight, 0);
              Nikon_NRW_WBtag(LIBRAW_WBI_Cloudy, 0);
              fseek(ifp, 0x10L, SEEK_CUR);
              Nikon_NRW_WBtag(LIBRAW_WBI_Tungsten, 0);
              Nikon_NRW_WBtag(LIBRAW_WBI_FL_W, 0);
              Nikon_NRW_WBtag(LIBRAW_WBI_Flash, 0);
              fseek(ifp, 0x10L, SEEK_CUR);
              Nikon_NRW_WBtag(LIBRAW_WBI_Custom, 0);
              Nikon_NRW_WBtag(LIBRAW_WBI_Auto, 0);
            }
            else
            { // P7000, P7100, B700, P1000
              fseek(ifp, 0x16L, SEEK_CUR);
              black = get2();
              if (cam_mul[0] < 0.1f)
              {
                fseek(ifp, 0x16L, SEEK_CUR);
                cam_mul[0] = get4() << 1;
                cam_mul[1] = get4();
                cam_mul[3] = get4();
                cam_mul[2] = get4() << 1;
              }
              else
              {
                fseek(ifp, 0x26L, SEEK_CUR);
              }
              if (len != 332)
              { // not A1000
                Nikon_NRW_WBtag(LIBRAW_WBI_Daylight, 1);
                Nikon_NRW_WBtag(LIBRAW_WBI_Cloudy, 1);
                Nikon_NRW_WBtag(LIBRAW_WBI_Shade, 1);
                Nikon_NRW_WBtag(LIBRAW_WBI_Tungsten, 1);
                Nikon_NRW_WBtag(LIBRAW_WBI_FL_W, 1);
                Nikon_NRW_WBtag(LIBRAW_WBI_FL_N, 1);
                Nikon_NRW_WBtag(LIBRAW_WBI_FL_D, 1);
                Nikon_NRW_WBtag(LIBRAW_WBI_HT_Mercury, 1);
                fseek(ifp, 0x14L, SEEK_CUR);
                Nikon_NRW_WBtag(LIBRAW_WBI_Custom, 1);
                Nikon_NRW_WBtag(LIBRAW_WBI_Auto, 1);
              }
              else
              {
                fseek(ifp, 0xc8L, SEEK_CUR);
                Nikon_NRW_WBtag(LIBRAW_WBI_Auto, 1);
              }
            }
          }
        }
      }
    }
    else if (tag == 0x001b)
    {
      imNikon.HighSpeedCropFormat = get2();
      imNikon.SensorHighSpeedCrop.cwidth = get2();
      imNikon.SensorHighSpeedCrop.cheight = get2();
      imNikon.SensorWidth = get2();
      imNikon.SensorHeight = get2();
      imNikon.SensorHighSpeedCrop.cleft = get2();
      imNikon.SensorHighSpeedCrop.ctop = get2();
      switch (imNikon.HighSpeedCropFormat)
      {
      case 0:
      case 1:
      case 2:
      case 4:
        imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_3to2;
        break;
      case 11:
        ilm.CameraFormat = LIBRAW_FORMAT_FF;
        imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_3to2;
        break;
      case 12:
        ilm.CameraFormat = LIBRAW_FORMAT_APSC;
        imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_3to2;
        break;
      case 3:
        imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_5to4;
        break;
      case 6:
        imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_16to9;
        break;
      case 17:
        imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_1to1;
        break;
      default:
        imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_OTHER;
        break;
      }
    }
    else if (tag == 0x001d)
    { // serial number
      if (len > 0)
      {
        int model_len = (int)strbuflen(model);
        while ((c = fgetc(ifp)) && (len-- > 0) && (c != (unsigned)EOF))
        {
          if ((!custom_serial) && (!isdigit(c)))
          {
            if (((model_len == 3) && !strcmp(model, "D50")) ||
                ((model_len >= 4) && !isalnum(model[model_len - 4]) &&
                 !strncmp(&model[model_len - 3], "D50", 3)))
            {
              custom_serial = 34;
            }
            else
            {
              custom_serial = 96;
            }
            break;
          }
          serial = serial * 10 + (isdigit(c) ? c - '0' : c % 10);
        }
        if (!imgdata.shootinginfo.BodySerial[0])
          sprintf(imgdata.shootinginfo.BodySerial, "%d", serial);
      }
    }
    else if (tag == 0x001e) {
      switch (get2()) {
      case 1:
        imCommon.ColorSpace = LIBRAW_COLORSPACE_sRGB;
        break;
      case 2:
        imCommon.ColorSpace = LIBRAW_COLORSPACE_AdobeRGB;
        break;
      default:
        imCommon.ColorSpace = LIBRAW_COLORSPACE_Unknown;
        break;
      }
    } else if (tag == 0x0025)
    {
      imCommon.real_ISO = int(100.0 * libraw_powf64l(2.0, double((uchar)fgetc(ifp)) / 12.0 - 5.0));
      if (!iso_speed || (iso_speed == 65535))
      {
        iso_speed = imCommon.real_ISO;
      }
    }
    else if (tag == 0x0022)
    {
      imNikon.Active_D_Lighting = get2();
    }
    else if (tag == 0x003b)
    { // WB for multi-exposure (ME); all 1s for regular exposures
      imNikon.ME_WB[0] = getreal(type);
      imNikon.ME_WB[2] = getreal(type);
      imNikon.ME_WB[1] = getreal(type);
      imNikon.ME_WB[3] = getreal(type);
    }
    else if (tag == 0x003d)
    { // not corrected for file bitcount, to be patched in open_datastream
      FORC4 cblack[RGGB_2_RGBG(c)] = get2();
      i = cblack[3];
      FORC3 if (i > cblack[c]) i = cblack[c];
      FORC4 cblack[c] -= i;
      black += i;
    }
    else if (tag == 0x0045)
    { /* upper left pixel (x,y), size (width,height) */
      imgdata.sizes.raw_inset_crops[0].cleft = get2();
      imgdata.sizes.raw_inset_crops[0].ctop = get2();
      imgdata.sizes.raw_inset_crops[0].cwidth = get2();
      imgdata.sizes.raw_inset_crops[0].cheight = get2();
    }
    else if (tag == 0x0051)
    {
      fseek(ifp, 10LL, SEEK_CUR);
      imNikon.NEFCompression = get2();
    }
    else if (tag == 0x0082)
    { // lens attachment
      stmread(ilm.Attachment, len, ifp);
    }
    else if (tag == 0x0083)
    { // lens type
      imgdata.lens.nikon.LensType = fgetc(ifp);
    }
    else if (tag == 0x0084)
    { // lens
      ilm.MinFocal = getreal(type);
      ilm.MaxFocal = getreal(type);
      ilm.MaxAp4MinFocal = getreal(type);
      ilm.MaxAp4MaxFocal = getreal(type);
    }
    else if (tag == 0x0088) // AFInfo
    {
      if (!imCommon.afcount)
      {
        imCommon.afdata[imCommon.afcount].AFInfoData_tag = tag;
        imCommon.afdata[imCommon.afcount].AFInfoData_order = order;
        imCommon.afdata[imCommon.afcount].AFInfoData_length = len;
        imCommon.afdata[imCommon.afcount].AFInfoData = (uchar *)malloc(imCommon.afdata[imCommon.afcount].AFInfoData_length);
        fread(imCommon.afdata[imCommon.afcount].AFInfoData, imCommon.afdata[imCommon.afcount].AFInfoData_length, 1, ifp);
        imCommon.afcount = 1;
      }
    }
    else if (tag == 0x008b) // lens f-stops
    {
      uchar uc1 = fgetc(ifp);
      uchar uc2 = fgetc(ifp);
      uchar uc3 = fgetc(ifp);
      if (uc3)
      {
        imgdata.lens.nikon.LensFStops = uc1 * uc2 * (12 / uc3);
        ilm.LensFStops = (float)imgdata.lens.nikon.LensFStops / 12.0f;
      }
    }
    else if ((tag == 0x008c) || (tag == 0x0096))
    {
      meta_offset = ftell(ifp);
    }
    else if ((tag == 0x0091) && (len > 4))
    {
      ShotInfo_len = len;
      ShotInfo_buf = (uchar *)malloc(ShotInfo_len);

/* for dump:
cj_block = (uchar *)malloc(ShotInfo_len);
ck_block = (uchar *)malloc(ShotInfo_len);
*/

      fread(ShotInfo_buf, ShotInfo_len, 1, ifp);
      FORC4 imNikon.ShotInfoVersion =
          imNikon.ShotInfoVersion * 10 + ShotInfo_buf[c] - '0';
    }
    else if (tag == 0x0093)
    {
      imNikon.NEFCompression = i = get2();
      if ((i == 7) || (i == 9))
      {
        ilm.LensMount = LIBRAW_MOUNT_FixedLens;
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      }
    }
    else if (tag == 0x0097)
    { // ver97
      FORC4 imNikon.ColorBalanceVersion =
          imNikon.ColorBalanceVersion * 10 + fgetc(ifp) - '0';
      switch (imNikon.ColorBalanceVersion)
      {
      case 100: // NIKON D100
        fseek(ifp, 0x44L, SEEK_CUR);
        FORC4 cam_mul[RBGG_2_RGBG(c)] = get2();
        break;
      case 102: // NIKON D2H
        fseek(ifp, 0x6L, SEEK_CUR);
        FORC4 cam_mul[RGGB_2_RGBG(c)] = get2();
        break;
      case 103: // NIKON D70, D70s
        fseek(ifp, 0x10L, SEEK_CUR);
        FORC4 cam_mul[c] = get2();
      }
      if (imNikon.ColorBalanceVersion >= 200)
      {
        /*
        204: NIKON D2X, D2Xs
        205: NIKON D50
        206: NIKON D2Hs
        207: NIKON D200
        208: NIKON D40, D40X, D80
        209: NIKON D3, D3X, D300, D700
        210: NIKON D60
        211: NIKON D90, D5000
        212: NIKON D300S
        213: NIKON D3000
        214: NIKON D3S
        215: NIKON D3100
        216: NIKON D5100, D7000
        217: NIKON D4, D600, D800, D800E, D3200
        -= unknown =-
        218: NIKON D5200, D7100
        219: NIKON D5300
        220: NIKON D610, Df
        221: NIKON D3300
        222: NIKON D4S
        223: NIKON D750, D810
        224: NIKON D3400, D3500, D5500, D5600, D7200
        225: NIKON D5, D500
        226: NIKON D7500
        227: NIKON D850
         */
        if (imNikon.ColorBalanceVersion != 205)
        {
          fseek(ifp, 0x118L, SEEK_CUR);
        }
        ColorBalanceData_ready =
            (fread(ColorBalanceData_buf, 324, 1, ifp) == 1);
      }
      if ((imNikon.ColorBalanceVersion >= 400) &&
          (imNikon.ColorBalanceVersion <= 405))
      { // 1 J1, 1 V1, 1 J2, 1 V2, 1 J3, 1 S1, 1 AW1, 1 S2, 1 J4, 1 V3, 1 J5
        ilm.CameraFormat = LIBRAW_FORMAT_1INCH;
        ilm.CameraMount = LIBRAW_MOUNT_Nikon_CX;
      }
      else if ((imNikon.ColorBalanceVersion >= 500) &&
               (imNikon.ColorBalanceVersion <= 502))
      { // P7700, P7800, P330, P340
        ilm.CameraMount = ilm.LensMount = LIBRAW_MOUNT_FixedLens;
        ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
      }
      else if (imNikon.ColorBalanceVersion == 601)
      { // Coolpix A
        ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_APSC;
        ilm.CameraMount = ilm.LensMount = LIBRAW_MOUNT_FixedLens;
        ilm.FocalType = LIBRAW_FT_PRIME_LENS;
      }
    }
    else if (tag == 0x0098) // contains lens data
    {
      FORC4 imNikon.LensDataVersion =
          imNikon.LensDataVersion * 10 + fgetc(ifp) - '0';
      switch (imNikon.LensDataVersion)
      {
      case 100:
        LensData_len = 9;
        break;
      case 101:
      case 201: // encrypted, starting from v.201
      case 202:
      case 203:
        LensData_len = 15;
        break;
      case 204:
        LensData_len = 16;
        break;
      case 400:
        LensData_len = 459;
        break;
      case 401:
        LensData_len = 590;
        break;
      case 402:
        LensData_len = 509;
        break;
      case 403:
        LensData_len = 879;
        break;
      case 800:
      case 801:
        LensData_len = 58;
        break;
      case 802:
        LensData_len = 108;
        break;
      }
      if (LensData_len)
      {
        LensData_buf = (uchar *)malloc(LensData_len);
        fread(LensData_buf, LensData_len, 1, ifp);
      }
    }
    else if (tag == 0x00a0)
    {
      stmread(imgdata.shootinginfo.BodySerial, len, ifp);
    }
    else if (tag == 0x00a7) // shutter count
    {
      imNikon.key = fgetc(ifp) ^ fgetc(ifp) ^ fgetc(ifp) ^ fgetc(ifp);
      if (custom_serial)
      {
        ci = xlat[0][custom_serial];
      }
      else
      {
        ci = xlat[0][serial & 0xff];
      }
      cj = xlat[1][imNikon.key];
      ck = 0x60;
      if (((unsigned)(imNikon.ColorBalanceVersion - 200) < 18) &&
          ColorBalanceData_ready)
      {
        for (i = 0; i < 324; i++)
          ColorBalanceData_buf[i] ^= (cj += ci * ck++);
        i = "66666>666;6A;:;555"[imNikon.ColorBalanceVersion - 200] - '0';
        FORC4 cam_mul[c ^ (c >> 1) ^ (i & 1)] =
            sget2(ColorBalanceData_buf + (i & -2) + c * 2);
      }

      if (LensData_len)
      {
        if (imNikon.LensDataVersion > 200)
        {
          cj = xlat[1][imNikon.key];
          ck = 0x60;
          for (i = 0; i < LensData_len; i++)
          {
            LensData_buf[i] ^= (cj += ci * ck++);
          }
        }
        processNikonLensData(LensData_buf, LensData_len);
        LensData_len = 0;
        free(LensData_buf);
      }
      if (ShotInfo_len && (imNikon.ShotInfoVersion >= 208)) {
        unsigned RotationOffset = 0,
                 OrientationOffset = 0;

        cj = xlat[1][imNikon.key];
        ck = 0x60;
        for (i = 4; i < ShotInfo_len; i++) {
          ShotInfo_buf[i] ^= (cj += ci * ck++);

/* for dump:
cj_block[i-4] = cj;
ck_block[i-4] = ck-1;
*/
        }
/* for dump:
printf ("==>> ci: 0x%02x, cj at start: 0x%02x\n",
ci, xlat[1][imNikon.key]);
hexDump("ck array:", ck_block, ShotInfo_len-4);
hexDump("cj array:", cj_block, ShotInfo_len-4);
free(cj_block);
free(ck_block);
*/

        switch (imNikon.ShotInfoVersion) {
        case 208: // ShotInfoD80, Rotation
          RotationOffset = 590;
          if (RotationOffset<ShotInfo_len) {
            imNikon.MakernotesFlip = *(ShotInfo_buf+RotationOffset) & 0x07;
          }
          break;

        case 231: // ShotInfoD4S, Rotation, Roll/Pitch/Yaw
          OrientationOffset  = 0x350b;
          RotationOffset     = 0x3693;
          if (RotationOffset<ShotInfo_len) {
            imNikon.MakernotesFlip = (*(ShotInfo_buf+RotationOffset)>>4) & 0x03;
          }
          break;

        case 233: // ShotInfoD810, Roll/Pitch/Yaw
          OrientationOffset = sget4_order(morder, ShotInfo_buf+0x84);
          break;

        case 238: // D5,   ShotInfoD500, Rotation, Roll/Pitch/Yaw
        case 239: // D500, ShotInfoD500, Rotation, Roll/Pitch/Yaw
          RotationOffset = sget4_order(morder, ShotInfo_buf+0x10) + 0xca;
          if (RotationOffset > 0xca) {
            RotationOffset -= 0xb0;
          }
          if (RotationOffset<ShotInfo_len) {
            imNikon.MakernotesFlip = *(ShotInfo_buf+RotationOffset) & 0x03;
          }
          OrientationOffset = sget4_order(morder, ShotInfo_buf+0xa0);
          break;

        case 243: // ShotInfoD850, Roll/Pitch/Yaw
          OrientationOffset = sget4_order(morder, ShotInfo_buf+0xa0);
          break;

        case 246: // ShotInfoD6, Roll/Pitch/Yaw
          OrientationOffset = sget4_order(morder, ShotInfo_buf+0x9c);
          break;

        case 800: // Z 6, Z 7,     ShotInfoZ7II, Roll/Pitch/Yaw
        case 801: // Z 50,         ShotInfoZ7II, Roll/Pitch/Yaw
        case 802: // Z 5,          ShotInfoZ7II, Roll/Pitch/Yaw
        case 803: // Z 6_2, Z 7_2, ShotInfoZ7II, Roll/Pitch/Yaw
        case 804: // Z fc          ShotInfoZ7II, Roll/Pitch/Yaw
          OrientationOffset = sget4_order(morder, ShotInfo_buf+0x98);
          break;

        case 805: // Z 9,          ShotInfoZ9, Roll/Pitch/Yaw
          OrientationOffset = sget4_order(morder, ShotInfo_buf+0x84);
          break;
        }

        if (OrientationOffset && ((OrientationOffset+12)<ShotInfo_len) && OrientationOffset < 0xffff) {
          if (imNikon.ShotInfoVersion == 231) // ShotInfoD4S
            imNikon.RollAngle = AngleConversion_a(morder, ShotInfo_buf+OrientationOffset);
          else
            imNikon.RollAngle = AngleConversion(morder, ShotInfo_buf+OrientationOffset);
          imNikon.PitchAngle  = AngleConversion (morder, ShotInfo_buf+OrientationOffset+4);
          imNikon.YawAngle    = AngleConversion (morder, ShotInfo_buf+OrientationOffset+8);
        }
        if ((RotationOffset) && (imNikon.MakernotesFlip < 4) && (imNikon.MakernotesFlip >= 0))
          imNikon.MakernotesFlip = "0863"[imNikon.MakernotesFlip] - '0';
        ShotInfo_len = 0;
        free(ShotInfo_buf);
      }
    }
    else if (tag == 0x00a8)
    { // contains flash data
      FORC4 imNikon.FlashInfoVersion =
          imNikon.FlashInfoVersion * 10 + fgetc(ifp) - '0';
    }
    else if (tag == 0x00b0)
    {
      get4(); // ME (multi-exposure) tag version, 4 symbols
      imNikon.ExposureMode = get4();
      imNikon.nMEshots = get4();
      imNikon.MEgainOn = get4();
    }
    else if (tag == 0x00b7) // AFInfo2
    {
      if (!imCommon.afcount && len > 4)
      {
        imCommon.afdata[imCommon.afcount].AFInfoData_tag = tag;
        imCommon.afdata[imCommon.afcount].AFInfoData_order = order;
        int ver = 0;
        FORC4  ver = ver * 10 + (fgetc(ifp) - '0');
        imCommon.afdata[imCommon.afcount].AFInfoData_version = ver;
        imCommon.afdata[imCommon.afcount].AFInfoData_length = len-4;
        imCommon.afdata[imCommon.afcount].AFInfoData = (uchar *)malloc(imCommon.afdata[imCommon.afcount].AFInfoData_length);
        fread(imCommon.afdata[imCommon.afcount].AFInfoData, imCommon.afdata[imCommon.afcount].AFInfoData_length, 1, ifp);
        imCommon.afcount = 1;
      }
    }
    else if (tag == 0x00b9)
    {
      imNikon.AFFineTune = fgetc(ifp);
      imNikon.AFFineTuneIndex = fgetc(ifp);
      imNikon.AFFineTuneAdj = (int8_t)fgetc(ifp);
    }
    else if ((tag == 0x0100) && tagtypeIs(LIBRAW_EXIFTAG_TYPE_UNDEFINED))
    {
      thumb_offset = ftell(ifp);
      thumb_length = len;
    }
    else if (tag == 0x0e01)
    { /* Nikon Software / in-camera edit Note */
      int loopc = 0;
      int WhiteBalanceAdj_active = 0;
      order = 0x4949;
      fseek(ifp, 22, SEEK_CUR);
      for (offset = 22; offset + 22 < len; offset += 22 + i)
      {
        if (loopc++ > 1024)
          throw LIBRAW_EXCEPTION_IO_CORRUPT;
        tag = get4();
        fseek(ifp, 14, SEEK_CUR);
        i = get4() - 4;

        if (tag == 0x76a43204)
        {
          WhiteBalanceAdj_active = fgetc(ifp);
        }
        else if (tag == 0xbf3c6c20)
        {
          if (WhiteBalanceAdj_active)
          {
            union {
              double dbl;
              unsigned long long lng;
            } un;
            un.dbl = getreal(LIBRAW_EXIFTAG_TYPE_DOUBLE);
            if ((un.lng != 0x3FF0000000000000ULL) &&
                (un.lng != 0x000000000000F03FULL))
            {
              cam_mul[0] = un.dbl;
              cam_mul[2] = getreal(LIBRAW_EXIFTAG_TYPE_DOUBLE);
              cam_mul[1] = cam_mul[3] = 1.0;
              i -= 16;
            }
            else
              i -= 8;
          }
          fseek(ifp, i, SEEK_CUR);
        }
        else if (tag == 0x76a43207)
        {
          flip = get2();
        }
        else
        {
          fseek(ifp, i, SEEK_CUR);
        }
      }
    }
    else if (tag == 0x0e22)
    {
      FORC4 imNikon.NEFBitDepth[c] = get2();
    }
  next:
    fseek(ifp, save, SEEK_SET);
  }
quit:
  order = sorder;
}

unsigned sget4_order (short _order, uchar *s) {
  unsigned v;
  if (_order == 0x4949)
    v= s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24;
  else
    v= s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3];
  return v;
}

double sget_fixed32u (short _order, uchar *s) {
  unsigned v = sget4_order (_order, s);
  return ((double)v / 6.5536 + 0.5) / 10000.0;
}

double AngleConversion_a (short _order, uchar *s) {
  double v = sget_fixed32u(_order, s);
  if (v < 180.0) return -v;
  return 360.0-v;
}

double AngleConversion (short _order, uchar *s) {
  double v = sget_fixed32u(_order, s);
  if (v <= 180.0) return v;
  return v-360.0;
}

/* ========= */
/*
void hexDump(char *title, void *addr, int len)
{
    int i;
    unsigned char buff[17];
    unsigned char *pc = (unsigned char*)addr;

    // Output description if given.
    if (title != NULL)
        printf ("%s:\n", title);

    // Process every byte in the data.
    for (i = 0; i < len; i++) {
        // Multiple of 16 means new line (with line offset).

        if ((i % 16) == 0) {
            // Just don't print ASCII for the zeroth line.
            if (i != 0)
                printf("  %s\n", buff);

            // Output the offset.
            printf("  %04x ", i);
        }

        // Now the hex code for the specific character.
        printf(" %02x", pc[i]);

        // And store a printable ASCII character for later.
        if ((pc[i] < 0x20) || (pc[i] > 0x7e)) {
            buff[i % 16] = '.';
        } else {
            buff[i % 16] = pc[i];
        }

        buff[(i % 16) + 1] = '\0';
    }

    // Pad out last line if not exactly 16 characters.
    while ((i % 16) != 0) {
        printf("   ");
        i++;
    }

    // And print the final ASCII bit.
    printf("  %s\n", buff);
}
*/
