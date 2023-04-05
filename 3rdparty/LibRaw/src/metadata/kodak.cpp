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

void LibRaw::Kodak_KDC_WBtags(int wb, int wbi)
{
  int c;
  FORC3 icWBC[wb][c] = get4();
  icWBC[wb][3] = icWBC[wb][1];
  if (wbi == wb)
    FORC4 cam_mul[c] = icWBC[wb][c];
  return;
}

void LibRaw::Kodak_DCR_WBtags(int wb, unsigned type, int wbi)
{
  float mul[3] = {1.0f, 1.0f, 1.0f}, num, mul2;
  int c;
  FORC3 mul[c] = (num = getreal(type)) <= 0.001f ? 1.0f : num;
  icWBC[wb][1] = icWBC[wb][3] = mul[1];
  mul2 = mul[1] * mul[1];
  icWBC[wb][0] = mul2 / mul[0];
  icWBC[wb][2] = mul2 / mul[2];
  if (wbi == wb)
    FORC4 cam_mul[c] = icWBC[wb][c];
  return;
}

short LibRaw::KodakIllumMatrix(unsigned type, float *romm_camIllum)
{
  int c, j, romm_camTemp[9], romm_camScale[3];
  if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SRATIONAL))
  {
    for (j = 0; j < 9; j++)
      ((float *)romm_camIllum)[j] = getreal(type);
    return 1;
  }
  else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SLONG))
  {
    FORC3
    {
      romm_camScale[c] = 0;
      for (j = 0; j < 3; j++)
      {
        romm_camTemp[c * 3 + j] = get4();
        romm_camScale[c] += romm_camTemp[c * 3 + j];
      }
    }
    if ((romm_camScale[0] > 0x1fff) && (romm_camScale[1] > 0x1fff) &&
        (romm_camScale[2] > 0x1fff))
    {
      FORC3 for (j = 0; j < 3; j++)((float *)romm_camIllum)[c * 3 + j] =
          ((float)romm_camTemp[c * 3 + j]) / ((float)romm_camScale[c]);
      return 1;
    }
  }
  return 0;
}

/* Thanks to Alexey Danilchenko for wb as-shot parsing code */
void LibRaw::parse_kodak_ifd(int base)
{
  unsigned entries, tag, type, len, save;
  int c, wbi = -1;

  static const int wbtag_kdc[] = {
      LIBRAW_WBI_Auto,        // 64037 / 0xfa25
      LIBRAW_WBI_Fluorescent, // 64040 / 0xfa28
      LIBRAW_WBI_Tungsten,    // 64039 / 0xfa27
      LIBRAW_WBI_Daylight,    // 64041 / 0xfa29
      -1,
      -1,
      LIBRAW_WBI_Shade // 64042 / 0xfa2a
  };

  static const int wbtag_dcr[] = {
      LIBRAW_WBI_Daylight,    // 2120 / 0x0848
      LIBRAW_WBI_Tungsten,    // 2121 / 0x0849
      LIBRAW_WBI_Fluorescent, // 2122 / 0x084a
      LIBRAW_WBI_Flash,       // 2123 / 0x084b
      LIBRAW_WBI_Custom,      // 2124 / 0x084c
      LIBRAW_WBI_Auto         // 2125 / 0x084d
  };

  //  int a_blck = 0;

  entries = get2();
  if (entries > 1024)
    return;
  INT64 fsize = ifp->size();
  while (entries--)
  {
    tiff_get(base, &tag, &type, &len, &save);
    INT64 savepos = ftell(ifp);
    if (len > 8 && len + savepos > 2 * fsize)
    {
      fseek(ifp, save, SEEK_SET); // Recover tiff-read position!!
      continue;
    }
    if (callbacks.exif_cb)
    {
      callbacks.exif_cb(callbacks.exifparser_data, tag | 0x20000, type, len,
                        order, ifp, base);
      fseek(ifp, savepos, SEEK_SET);
    }
    if (tag == 0x03eb) // 1003
      imgdata.sizes.raw_inset_crops[0].cleft = get2();
    else if (tag == 0x03ec) // 1004
      imgdata.sizes.raw_inset_crops[0].ctop = get2();
    else if (tag == 0x03ed) // 1005
      imgdata.sizes.raw_inset_crops[0].cwidth = get2();
    else if (tag == 0x03ee) // 1006
      imgdata.sizes.raw_inset_crops[0].cheight = get2();
    else if (tag == 0x03ef) // 1007
    {
      if (!strcmp(model, "EOS D2000C"))
        black = get2();
      else
        imKodak.BlackLevelTop = get2();
    }
    else if (tag == 0x03f0) // 1008
    {
      if (!strcmp(model, "EOS D2000C"))
      {
        if (black) // already set by tag 1007 (0x03ef)
          black = (black + get2()) / 2;
        else
          black = get2();
      }
      else
        imKodak.BlackLevelBottom = get2();
    }

    else if (tag == 0x03f1)
    { // 1009 Kodak TextualInfo
      if (len > 0)
      {
        char kti[1024];
        char *pkti;
        int nsym = MIN(len, 1023);
        fread(kti, 1, nsym, ifp);
        kti[nsym] = 0;
#ifdef LIBRAW_WIN32_CALLS
        pkti = strtok(kti, "\x0a");
#else
        char *last = 0;
        pkti = strtok_r(kti, "\x0a", &last);
#endif
        while (pkti != NULL)
        {
          c = 12;
          if (((int)strlen(pkti) > c) && (!strncasecmp(pkti, "Camera body:", c)))
          {
            while ((pkti[c] == ' ') && (c < (int)strlen(pkti)))
            {
              c++;
            }
            strcpy(ilm.body, pkti + c);
          }
          c = 5;
          if (((int)strlen(pkti) > c) && (!strncasecmp(pkti, "Lens:", c)))
          {
            ilm.CurFocal = atoi(pkti + c);
          }
          c = 9;
          if (((int)strlen(pkti) > c) && (!strncasecmp(pkti, "Aperture:", c)))
          {
            while (((pkti[c] == ' ') || (pkti[c] == 'f')) && (c < (int)strlen(pkti)))
            {
              c++;
            }
            ilm.CurAp = atof(pkti + c);
          }
          c = 10;
          if (((int)strlen(pkti) > c) && (!strncasecmp(pkti, "ISO Speed:", c)))
          {
            iso_speed = atoi(pkti + c);
          }
          c = 13;
          if (((int)strlen(pkti) > c) && (!strncasecmp(pkti, "Focal Length:", c)))
          {
            ilm.CurFocal = atoi(pkti + c);
          }
          c = 13;
          if (((int)strlen(pkti) > c) && (!strncasecmp(pkti, "Max Aperture:", c)))
          {
            while (((pkti[c] == ' ') || (pkti[c] == 'f')) && (c < (int)strlen(pkti)))
            {
              c++;
            }
            ilm.MaxAp4CurFocal = atof(pkti + c);
          }
          c = 13;
          if (((int)strlen(pkti) > c) && (!strncasecmp(pkti, "Min Aperture:", c)))
          {
            while (((pkti[c] == ' ') || (pkti[c] == 'f')) && (c < (int)strlen(pkti)))
            {
              c++;
            }
            ilm.MinAp4CurFocal = atof(pkti + c);
          }
#ifdef LIBRAW_WIN32_CALLS
          pkti = strtok(NULL, "\x0a");
#else
          pkti = strtok_r(NULL, "\x0a", &last);
#endif
        }
      }
    }

    else if (tag == 0x03f3) // 1011
      imCommon.FlashEC = getreal(type);

    else if (tag == 0x03fc) // 1020
    {
      wbi = getint(type);
      if ((wbi >= 0) && (wbi < 6) && (wbi != -2))
        wbi = wbtag_dcr[wbi];
    }
    else if (tag == 0x03fd && len == 72) // 1021
    {                                    /* WB set in software */
      fseek(ifp, 40, SEEK_CUR);
      FORC3 cam_mul[c] = 2048.0 / fMAX(1.0f, get2());
      wbi = -2;
    }

    else if ((tag == 0x0406) && (len == 1)) // 1030
      imCommon.CameraTemperature = getreal(type);
    else if ((tag == 0x0413) && (len == 1)) // 1043
      imCommon.SensorTemperature = getreal(type);
    else if (tag == 0x0848) // 2120
      Kodak_DCR_WBtags(LIBRAW_WBI_Daylight, type, wbi);
    else if (tag == 0x0849) // 2121
      Kodak_DCR_WBtags(LIBRAW_WBI_Tungsten, type, wbi);
    else if (tag == 0x084a) // 2122
      Kodak_DCR_WBtags(LIBRAW_WBI_Fluorescent, type, wbi);
    else if (tag == 0x084b) // 2123
      Kodak_DCR_WBtags(LIBRAW_WBI_Flash, type, wbi);
    else if (tag == 0x084c) // 2124
      Kodak_DCR_WBtags(LIBRAW_WBI_Custom, type, wbi);
    else if (tag == 0x084d) // 2125
    {
      if (wbi == -1)
        wbi = LIBRAW_WBI_Auto;
      Kodak_DCR_WBtags(LIBRAW_WBI_Auto, type, wbi);
    }
    else if (tag == 0x089f) // 2207
      imKodak.ISOCalibrationGain = getreal(type);
    else if (tag == 0x0903) // 2307
      imKodak.AnalogISO = iso_speed = getreal(type);
    else if (tag == 0x090d) // 2317
      linear_table(len);
    else if (tag == 0x09ce) // 2510
      stmread(imgdata.shootinginfo.InternalBodySerial, len, ifp);
    else if (tag == 0x0e92) // 3730
    {
      imKodak.val018percent = get2();
      imgdata.color.linear_max[0] = imgdata.color.linear_max[1] =
          imgdata.color.linear_max[2] = imgdata.color.linear_max[3] =
              (int)(((float)imKodak.val018percent) / 18.0f * 170.0f);
    }
    else if (tag == 0x0e93) // 3731
      imgdata.color.linear_max[0] = imgdata.color.linear_max[1] =
          imgdata.color.linear_max[2] = imgdata.color.linear_max[3] =
              imKodak.val170percent = get2();
    else if (tag == 0x0e94) // 3732
      imKodak.val100percent = get2();
    /*
        else if (tag == 0x1784)    // 6020
          iso_speed = getint(type);
    */
    else if (tag == 0xfa00) // 64000
      stmread(imgdata.shootinginfo.BodySerial, len, ifp);
    else if (tag == 0xfa0d) // 64013
    {
      wbi = fgetc(ifp);
      if ((wbi >= 0) && (wbi < 7))
        wbi = wbtag_kdc[wbi];
    }
    else if (tag == 0xfa13) // 64019
      width = getint(type);
    else if (tag == 0xfa14) // 64020
      height = (getint(type) + 1) & -2;
    /*
          height = getint(type);

        else if (tag == 0xfa16)  // 64022
          raw_width = get2();
        else if (tag == 0xfa17)  // 64023
          raw_height = get2();
    */
    else if (tag == 0xfa18) // 64024
    {
      imKodak.offset_left = getint(LIBRAW_EXIFTAG_TYPE_SSHORT);
      if (type != LIBRAW_EXIFTAG_TYPE_SSHORT)
        imKodak.offset_left += 1;
    }
    else if (tag == 0xfa19) // 64025
    {
      imKodak.offset_top = getint(LIBRAW_EXIFTAG_TYPE_SSHORT);
      if (type != LIBRAW_EXIFTAG_TYPE_SSHORT)
        imKodak.offset_top += 1;
    }

    else if (tag == 0xfa25) // 64037
      Kodak_KDC_WBtags(LIBRAW_WBI_Auto, wbi);
    else if (tag == 0xfa27) // 64039
      Kodak_KDC_WBtags(LIBRAW_WBI_Tungsten, wbi);
    else if (tag == 0xfa28) // 64040
      Kodak_KDC_WBtags(LIBRAW_WBI_Fluorescent, wbi);
    else if (tag == 0xfa29) // 64041
      Kodak_KDC_WBtags(LIBRAW_WBI_Daylight, wbi);
    else if (tag == 0xfa2a) // 64042
      Kodak_KDC_WBtags(LIBRAW_WBI_Shade, wbi);

    else if (tag == 0xfa31) // 64049
      imgdata.sizes.raw_inset_crops[0].cwidth = get2();
    else if (tag == 0xfa32) // 64050
      imgdata.sizes.raw_inset_crops[0].cheight = get2();
    else if (tag == 0xfa3e) // 64062
      imgdata.sizes.raw_inset_crops[0].cleft = get2();
    else if (tag == 0xfa3f) // 64063
      imgdata.sizes.raw_inset_crops[0].ctop = get2();

    else if (((tag == 0x07e4) || (tag == 0xfb01)) &&
             (len == 9)) // 2020 or 64257
    {
      if (KodakIllumMatrix(type, (float *)imKodak.romm_camDaylight))
      {
        romm_coeff(imKodak.romm_camDaylight);
      }
    }
    else if (((tag == 0x07e5) || (tag == 0xfb02)) &&
             (len == 9)) // 2021 or 64258
      KodakIllumMatrix(type, (float *)imKodak.romm_camTungsten);
    else if (((tag == 0x07e6) || (tag == 0xfb03)) &&
             (len == 9)) // 2022 or 64259
      KodakIllumMatrix(type, (float *)imKodak.romm_camFluorescent);
    else if (((tag == 0x07e7) || (tag == 0xfb04)) &&
             (len == 9)) // 2023 or 64260
      KodakIllumMatrix(type, (float *)imKodak.romm_camFlash);
    else if (((tag == 0x07e8) || (tag == 0xfb05)) &&
             (len == 9)) // 2024 or 64261
      KodakIllumMatrix(type, (float *)imKodak.romm_camCustom);
    else if (((tag == 0x07e9) || (tag == 0xfb06)) &&
             (len == 9)) // 2025 or 64262
      KodakIllumMatrix(type, (float *)imKodak.romm_camAuto);

    fseek(ifp, save, SEEK_SET);
  }
}
