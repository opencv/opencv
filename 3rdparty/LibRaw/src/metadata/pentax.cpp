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
#include "../../internal/libraw_cameraids.h"

void LibRaw::setPentaxBodyFeatures(unsigned long long id)
{

  ilm.CamID = id;

  switch (id) {
  case PentaxID_staristD:
  case PentaxID_staristDS:
  case PentaxID_staristDL:
  case PentaxID_staristDS2:
  case PentaxID_GX_1S:
  case PentaxID_staristDL2:
  case PentaxID_GX_1L:
  case PentaxID_K100D:
  case PentaxID_K110D:
  case PentaxID_K100D_Super:
  case PentaxID_K10D:
  case PentaxID_GX10:
  case PentaxID_K20D:
  case PentaxID_GX20:
  case PentaxID_K200D:
  case PentaxID_K2000:
  case PentaxID_K_m:
  case PentaxID_K_7:
  case PentaxID_K_x:
  case PentaxID_K_r:
  case PentaxID_K_5:
  case PentaxID_K_01:
  case PentaxID_K_30:
  case PentaxID_K_5_II:
  case PentaxID_K_5_II_s:
  case PentaxID_K_50:
  case PentaxID_K_3:
  case PentaxID_K_500:
  case PentaxID_K_S1:
  case PentaxID_K_S2:
  case PentaxID_K_3_II:
  case PentaxID_K_3_III:
  case PentaxID_K_70:
  case PentaxID_KP:
    ilm.CameraMount = LIBRAW_MOUNT_Pentax_K;
    ilm.CameraFormat = LIBRAW_FORMAT_APSC;
    break;
  case PentaxID_K_1:
  case PentaxID_K_1_Mark_II:
    ilm.CameraMount = LIBRAW_MOUNT_Pentax_K;
    ilm.CameraFormat = LIBRAW_FORMAT_FF;
    break;
  case PentaxID_645D:
  case PentaxID_645Z:
    ilm.CameraMount = LIBRAW_MOUNT_Pentax_645;
    ilm.CameraFormat = LIBRAW_FORMAT_CROP645;
    break;
  case PentaxID_Q:
  case PentaxID_Q10:
    ilm.CameraMount = LIBRAW_MOUNT_Pentax_Q;
    ilm.CameraFormat = LIBRAW_FORMAT_1div2p3INCH;
    break;
  case PentaxID_Q7:
  case PentaxID_Q_S1:
    ilm.CameraMount = LIBRAW_MOUNT_Pentax_Q;
    ilm.CameraFormat = LIBRAW_FORMAT_1div1p7INCH;
    break;
  case PentaxID_MX_1:
    ilm.LensMount = LIBRAW_MOUNT_FixedLens;
    ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
    ilm.CameraFormat = LIBRAW_FORMAT_1div1p7INCH;
    ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
    break;
  case PentaxID_GR_III:
  case PentaxID_GR_IIIx:
    ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
    ilm.LensMount = LIBRAW_MOUNT_FixedLens;
    ilm.CameraFormat = LIBRAW_FORMAT_APSC;
    ilm.LensFormat = LIBRAW_FORMAT_APSC;
    ilm.FocalType = LIBRAW_FT_PRIME_LENS;
    break;
  default:
    ilm.LensMount = LIBRAW_MOUNT_FixedLens;
    ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
  }
  return;
}

void LibRaw::PentaxISO(ushort c)
{
  int code[] = {3,    4,    5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
                15,   16,   17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
                27,   28,   29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                39,   40,   41,  42,  43,  44,  45,  50,  100, 200, 400, 800,
                1600, 3200, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267,
                268,  269,  270, 271, 272, 273, 274, 275, 276, 277, 278};
  double value[] = {
      50,     64,     80,     100,    125,    160,    200,    250,    320,
      400,    500,    640,    800,    1000,   1250,   1600,   2000,   2500,
      3200,   4000,   5000,   6400,   8000,   10000,  12800,  16000,  20000,
      25600,  32000,  40000,  51200,  64000,  80000,  102400, 128000, 160000,
      204800, 258000, 325000, 409600, 516000, 650000, 819200, 50,     100,
      200,    400,    800,    1600,   3200,   50,     70,     100,    140,
      200,    280,    400,    560,    800,    1100,   1600,   2200,   3200,
      4500,   6400,   9000,   12800,  18000,  25600,  36000,  51200};
#define numel (sizeof(code) / sizeof(code[0]))
  int i;
  for (i = 0; i < (int)numel; i++)
  {
    if (code[i] == c)
    {
      iso_speed = value[i];
      return;
    }
  }
  if (i == numel)
    iso_speed = 65535.0f;
}
#undef numel

void LibRaw::PentaxLensInfo(unsigned long long id, unsigned len) // tag 0x0207
{
  ushort iLensData = 0;
  uchar *table_buf;
  table_buf = (uchar *)malloc(MAX(len, 128));
  fread(table_buf, len, 1, ifp);
  if ((id < PentaxID_K100D) ||
      (((id == PentaxID_K100D) ||
        (id == PentaxID_K110D) ||
        (id == PentaxID_K100D_Super)) &&
       ((!table_buf[20] ||
        (table_buf[20] == 0xff)))))
  {
    iLensData = 3;
    if (ilm.LensID == LIBRAW_LENS_NOT_SET)
      ilm.LensID = (((unsigned)table_buf[0]) << 8) + table_buf[1];
  }
  else
    switch (len)
    {
    case 90: // LensInfo3
      iLensData = 13;
      if (ilm.LensID == LIBRAW_LENS_NOT_SET)
        ilm.LensID = ((unsigned)((table_buf[1] & 0x0f) + table_buf[3]) << 8) +
                     table_buf[4];
      break;
    case 91: // LensInfo4
      iLensData = 12;
      if (ilm.LensID == LIBRAW_LENS_NOT_SET)
        ilm.LensID = ((unsigned)((table_buf[1] & 0x0f) + table_buf[3]) << 8) +
                     table_buf[4];
      break;
    case 80: // LensInfo5
    case 128:
      iLensData = 15;
      if (ilm.LensID == LIBRAW_LENS_NOT_SET)
        ilm.LensID = ((unsigned)((table_buf[1] & 0x0f) + table_buf[4]) << 8) +
                     table_buf[5];
      break;
    case 168: // Ricoh GR III, id 0x1320e
      break;
    default:
      if (id >= 0x12b9cULL) // LensInfo2
      {
        iLensData = 4;
        if (ilm.LensID == LIBRAW_LENS_NOT_SET)
          ilm.LensID = ((unsigned)((table_buf[0] & 0x0f) + table_buf[2]) << 8) +
                       table_buf[3];
      }
    }
  if (iLensData)
  {
    if (table_buf[iLensData + 9] && (fabs(ilm.CurFocal) < 0.1f))
      ilm.CurFocal = 10 * (table_buf[iLensData + 9] >> 2) *
                     libraw_powf64l(4, (table_buf[iLensData + 9] & 0x03) - 2);
    if (table_buf[iLensData + 10] & 0xf0)
      ilm.MaxAp4CurFocal = libraw_powf64l(
          2.0f, (float)((table_buf[iLensData + 10] & 0xf0) >> 4) / 4.0f);
    if (table_buf[iLensData + 10] & 0x0f)
      ilm.MinAp4CurFocal = libraw_powf64l(
          2.0f, (float)((table_buf[iLensData + 10] & 0x0f) + 10) / 4.0f);

    if (iLensData != 12)
    {
      switch (table_buf[iLensData] & 0x06)
      {
      case 0:
        ilm.MinAp4MinFocal = 22.0f;
        break;
      case 2:
        ilm.MinAp4MinFocal = 32.0f;
        break;
      case 4:
        ilm.MinAp4MinFocal = 45.0f;
        break;
      case 6:
        ilm.MinAp4MinFocal = 16.0f;
        break;
      }
      if (table_buf[iLensData] & 0x70)
        ilm.LensFStops =
            ((float)(((table_buf[iLensData] & 0x70) >> 4) ^ 0x07)) / 2.0f +
            5.0f;

      ilm.MinFocusDistance = (float)(table_buf[iLensData + 3] & 0xf8);
      ilm.FocusRangeIndex = (float)(table_buf[iLensData + 3] & 0x07);

      if ((table_buf[iLensData + 14] > 1) && (fabs(ilm.MaxAp4CurFocal) < 0.7f))
        ilm.MaxAp4CurFocal = libraw_powf64l(
            2.0f, (float)((table_buf[iLensData + 14] & 0x7f) - 1) / 32.0f);
    }
    else if ((id != 0x12e76ULL) && // K-5
             (table_buf[iLensData + 15] > 1) &&
             (fabs(ilm.MaxAp4CurFocal) < 0.7f))
    {
      ilm.MaxAp4CurFocal = libraw_powf64l(
          2.0f, (float)((table_buf[iLensData + 15] & 0x7f) - 1) / 32.0f);
    }
  }
  free(table_buf);
  return;
}

void LibRaw::parsePentaxMakernotes(int /*base*/, unsigned tag, unsigned type,
                                   unsigned len, unsigned dng_writer)
{

  int c;
// printf ("==>> =%s= tag:0x%x, type: %d, len:%d\n", model, tag, type, len);

  if (tag == 0x0005)
  {
    unique_id = get4();
    setPentaxBodyFeatures(unique_id);
  }
  else if (tag == 0x0008)
  { /* 4 is raw, 7 is raw w/ pixel shift, 8 is raw w/ dynamic pixel shift */
    imPentax.Quality = get2();
  }
  else if (tag == 0x000d)
  {
    imgdata.shootinginfo.FocusMode = imPentax.FocusMode[0] = get2();
  }
  else if (tag == 0x000e)
  {
    imgdata.shootinginfo.AFPoint = imPentax.AFPointSelected[0] = get2();
    if (len == 2)
      imPentax.AFPointSelected_Area = get2();
  }
  else if (tag == 0x000f)
  {
    if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
    {
      imPentax.AFPointsInFocus = get4();
      if (!imPentax.AFPointsInFocus) imPentax.AFPointsInFocus = 0xffffffff;
      else imPentax.AFPointsInFocus_version = 3;
    }
    else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT))
    {
      imPentax.AFPointsInFocus = (unsigned) get2();
      if (imPentax.AFPointsInFocus == 0x0000ffff)
        imPentax.AFPointsInFocus = 0xffffffff;
      else imPentax.AFPointsInFocus_version = 2;
    }
  }
  else if (tag == 0x0010)
  {
    imPentax.FocusPosition = get2();
  }
  else if (tag == 0x0013)
  {
    ilm.CurAp = (float)get2() / 10.0f;
  }
  else if (tag == 0x0014)
  {
    PentaxISO(get2());
  }
  else if (tag == 0x0017)
  {
    imgdata.shootinginfo.MeteringMode = get2();
  }
  else if (tag == 0x001b) {
    cam_mul[2] = get2() / 256.0;
  }
  else if (tag == 0x001c) {
    cam_mul[0] = get2() / 256.0;
  }
  else if (tag == 0x001d)
  {
    ilm.CurFocal = (float)get4() / 100.0f;
  }
  else if (tag == 0x0034)
  {
    uchar uc;
    FORC4
    {
      fread(&uc, 1, 1, ifp);
      imPentax.DriveMode[c] = uc;
    }
    imgdata.shootinginfo.DriveMode = imPentax.DriveMode[0];
  }
  else if (tag == 0x0037) {
    switch (get2()) {
    case 0:
      imCommon.ColorSpace = LIBRAW_COLORSPACE_sRGB;
      break;
    case 1:
      imCommon.ColorSpace = LIBRAW_COLORSPACE_AdobeRGB;
      break;
    default:
      imCommon.ColorSpace = LIBRAW_COLORSPACE_Unknown;
      break;
    }
  }
  else if (tag == 0x0038)
  {
    imgdata.sizes.raw_inset_crops[0].cleft = get2();
    imgdata.sizes.raw_inset_crops[0].ctop = get2();
  }
  else if (tag == 0x0039)
  {
    imgdata.sizes.raw_inset_crops[0].cwidth = get2();
    imgdata.sizes.raw_inset_crops[0].cheight = get2();
  }
  else if (tag == 0x003c)
  {
    if ((len == 4) && tagtypeIs(LIBRAW_EXIFTAG_TYPE_UNDEFINED)) {
      imPentax.AFPointsInFocus = get4() & 0x7ff;
      if (!imPentax.AFPointsInFocus) {
        imPentax.AFPointsInFocus = 0xffffffff;
      }
      else {
        imPentax.AFPointsInFocus_version = 1;
      }
    }
  }
  else if (tag == 0x003f)
  {
    unsigned a = unsigned(fgetc(ifp)) << 8;
    ilm.LensID = a | fgetc(ifp);
  }
  else if (tag == 0x0047)
  {
    imCommon.CameraTemperature = (float)fgetc(ifp);
  }
  else if (tag == 0x004d)
  {
    if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SLONG))
      imCommon.FlashEC = getreal(type) / 256.0f;
    else
      imCommon.FlashEC = (float)((signed short)fgetc(ifp)) / 6.0f;
  }
  else if (tag == 0x005c)
  {
    fgetc(ifp);
    imgdata.shootinginfo.ImageStabilization = (short)fgetc(ifp);
  }
  else if (tag == 0x0072)
  {
    imPentax.AFAdjustment = get2();
  }
  else if ((tag == 0x007e) && (dng_writer == nonDNG))
  {
    imgdata.color.linear_max[0] = imgdata.color.linear_max[1] =
        imgdata.color.linear_max[2] = imgdata.color.linear_max[3] =
            get4();
  }
  else if (tag == 0x0080)
  {
    short a = (short)fgetc(ifp);
    switch (a)
    {
    case 0:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_4to3;
      break;
    case 1:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_3to2;
      break;
    case 2:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_16to9;
      break;
    case 3:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_1to1;
      break;
    }
  }

  else if ((tag == 0x0200) && (dng_writer == nonDNG)) { // Pentax black level
    FORC4 cblack[RGGB_2_RGBG(c)] = get2();
  }

  else if ((tag == 0x0201) && (dng_writer == nonDNG)) { // Pentax As Shot WB
    FORC4 cam_mul[RGGB_2_RGBG(c)] = get2();
  }

  else if ((tag == 0x0203) && (dng_writer == nonDNG))
  {
    for (int i = 0; i < 3; i++)
      FORC3 cmatrix[i][c] = ((short)get2()) / 8192.0;
  }
  else if (tag == 0x0205)
  {
    if (imCommon.afcount < LIBRAW_AFDATA_MAXCOUNT)
    {
      imCommon.afdata[imCommon.afcount].AFInfoData_tag = tag;
      imCommon.afdata[imCommon.afcount].AFInfoData_order = order;
      imCommon.afdata[imCommon.afcount].AFInfoData_length = len;
      imCommon.afdata[imCommon.afcount].AFInfoData = (uchar *)malloc(imCommon.afdata[imCommon.afcount].AFInfoData_length);
      fread(imCommon.afdata[imCommon.afcount].AFInfoData, imCommon.afdata[imCommon.afcount].AFInfoData_length, 1, ifp);
      if ((len < 25) && (len >= 11))
      {
        imPentax.AFPointMode = (imCommon.afdata[imCommon.afcount].AFInfoData[3] >>4) & 0x0f;
        imPentax.FocusMode[1] = imCommon.afdata[imCommon.afcount].AFInfoData[3] & 0x0f;
        imPentax.AFPointSelected[1] = sget2(imCommon.afdata[imCommon.afcount].AFInfoData+4);
// Pentax K-m has multiexposure set to 8 when no multi-exposure is in effect
        imPentax.MultiExposure = imCommon.afdata[imCommon.afcount].AFInfoData[10] & 0x0f;
      }
      imCommon.afcount++;
    }
  }
  else if (tag == 0x0207)
  {
    if (len < 65535) // Safety belt
      PentaxLensInfo(ilm.CamID, len);
  }
  else if ((tag >= 0x020d) && (tag <= 0x0214))
  {
    FORC4 icWBC[Pentax_wb_list1[tag - 0x020d]][RGGB_2_RGBG(c)] = get2();
  }
  else if ((tag == 0x021d) && (len == 18) &&
           tagtypeIs(LIBRAW_EXIFTAG_TYPE_UNDEFINED) && (dng_writer == nonDNG))
  {
    for (int i = 0; i < 3; i++)
      FORC3 cmatrix[i][c] = ((short)get2()) / 8192.0;
  }
  else if (tag == 0x021f)
  {
    if ((unique_id != PentaxID_K_1)    &&
        (unique_id != PentaxID_K_3)    &&
        (unique_id != PentaxID_K_3_II) &&
        (unique_id != PentaxID_K_1_Mark_II))
    {
      fseek (ifp, 0x0b, SEEK_CUR);
      imPentax.AFPointsInFocus = (unsigned) fgetc(ifp);
      if (!imPentax.AFPointsInFocus) imPentax.AFPointsInFocus = 0xffffffff;
      else imPentax.AFPointsInFocus_version = 4;
    }
  }
  else if ((tag == 0x0220) && (dng_writer == nonDNG)) {
    meta_offset = ftell(ifp);
  }
  else if (tag == 0x0221)
  {
    int nWB = get2();
    if (nWB <= int(sizeof(icWBCCTC) / sizeof(icWBCCTC[0])))
      FORC(nWB)
      {
        icWBCCTC[c][0] = (unsigned)0xcfc6 - get2();
        fseek(ifp, 2, SEEK_CUR);
        icWBCCTC[c][1] = get2();
        icWBCCTC[c][2] = icWBCCTC[c][4] = 0x2000;
        icWBCCTC[c][3] = get2();
      }
  }
  else if (tag == 0x0215)
  {
    fseek(ifp, 16, SEEK_CUR);
    sprintf(imgdata.shootinginfo.InternalBodySerial, "%d", get4());
  }
  else if (tag == 0x0229)
  {
    stmread(imgdata.shootinginfo.BodySerial, len, ifp);
  }
  else if (tag == 0x022d)
  {
    int wb_ind;
    getc(ifp);
    for (int wb_cnt = 0; wb_cnt < (int)Pentax_wb_list2.size(); wb_cnt++)
    {
      wb_ind = getc(ifp);
      if (wb_ind >= 0 && wb_ind < (int)Pentax_wb_list2.size() )
        FORC4 icWBC[Pentax_wb_list2[wb_ind]][RGGB_2_RGBG(c)] = get2();
    }
  }
  else if (tag == 0x0239) // Q-series lens info (LensInfoQ)
  {
    char LensInfo[20];
    fseek(ifp, 12, SEEK_CUR);
    stread(ilm.Lens, 30, ifp);
    strcat(ilm.Lens, " ");
    stread(LensInfo, 20, ifp);
    strcat(ilm.Lens, LensInfo);
  }
  else if (tag == 0x0245)
  {
    if (imCommon.afcount < LIBRAW_AFDATA_MAXCOUNT) {
      imCommon.afdata[imCommon.afcount].AFInfoData_tag = tag;
      imCommon.afdata[imCommon.afcount].AFInfoData_order = order;
      imCommon.afdata[imCommon.afcount].AFInfoData_length = len;
      imCommon.afdata[imCommon.afcount].AFInfoData = (uchar *)malloc(imCommon.afdata[imCommon.afcount].AFInfoData_length);
      fread(imCommon.afdata[imCommon.afcount].AFInfoData, imCommon.afdata[imCommon.afcount].AFInfoData_length, 1, ifp);
      imCommon.afcount++;
    }
  }
}

void LibRaw::parseRicohMakernotes(int /*base*/, unsigned tag, unsigned type,
                                  unsigned /*len*/, unsigned /*dng_writer */)
{
  char buffer[17];
  if (tag == 0x0005)
  {
    int c;
    int count = 0;
    fread(buffer, 16, 1, ifp);
    buffer[16] = 0;
    FORC(16)
    {
      if ((isspace(buffer[c])) || (buffer[c] == 0x2D) || (isalnum(buffer[c])))
        count++;
      else
        break;
    }
    if (count == 16)
    {
      if (strncmp(model, "GXR", 3))
      {
        sprintf(imgdata.shootinginfo.BodySerial, "%8s", buffer + 8);
      }
      buffer[8] = 0;
      sprintf(imgdata.shootinginfo.InternalBodySerial, "%8s", buffer);
    }
    else
    {
      sprintf(imgdata.shootinginfo.BodySerial, "%02x%02x%02x%02x", buffer[4],
              buffer[5], buffer[6], buffer[7]);
      sprintf(imgdata.shootinginfo.InternalBodySerial, "%02x%02x%02x%02x",
              buffer[8], buffer[9], buffer[10], buffer[11]);
    }
  }
  else if ((tag == 0x1001) && tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT))
  {
    ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
    ilm.LensMount = LIBRAW_MOUNT_FixedLens;
    ilm.CameraFormat = LIBRAW_FORMAT_APSC;
    ilm.LensID = LIBRAW_LENS_NOT_SET;
    ilm.FocalType = LIBRAW_FT_PRIME_LENS;
    imgdata.shootinginfo.ExposureProgram = get2();
  }
  else if ((tag == 0x1002) && tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT))
  {
    imgdata.shootinginfo.DriveMode = get2();
  }
  else if (tag == 0x1006)
  {
    imgdata.shootinginfo.FocusMode = get2();
  }
  else if (tag == 0x1007)
  {
    imRicoh.AutoBracketing = get2();
  }
  else if (tag == 0x1009)
  {
    imRicoh.MacroMode = get2();
  }
  else if (tag == 0x100a)
  {
    imRicoh.FlashMode = get2();
  }
  else if (tag == 0x100b)
  {
    imRicoh.FlashExposureComp = getreal(type);
  }
  else if (tag == 0x100c)
  {
    imRicoh.ManualFlashOutput = getreal(type);
  }
  else if ((tag == 0x100b) && tagtypeIs(LIBRAW_EXIFTAG_TYPE_SRATIONAL))
  {
    imCommon.FlashEC = getreal(type);
  }
  else if ((tag == 0x1017) && ((imRicoh.WideAdapter = get2()) == 2))
  {
    strcpy(ilm.Attachment, "Wide-Angle Adapter");
  }
  else if (tag == 0x1018)
  {
    imRicoh.CropMode = get2();
  }
  else if (tag == 0x1019)
  {
    imRicoh.NDFilter = get2();
  }
  else if (tag == 0x1200)
  {
    imRicoh.AFStatus = get2();
  }
  else if (tag == 0x1201)
  {
    imRicoh.AFAreaXPosition[1] = get4();
  }
  else if (tag == 0x1202)
  {
    imRicoh.AFAreaYPosition[1] = get4();
  }
  else if (tag == 0x1203)
  {
    imRicoh.AFAreaXPosition[0] = get4();
  }
  else if (tag == 0x1204)
  {
    imRicoh.AFAreaYPosition[0] = get4();
  }
  else if (tag == 0x1205)
  {
    imRicoh.AFAreaMode = get2();
  }
  else if (tag == 0x1500)
  {
    ilm.CurFocal = getreal(type);
  }
  else if (tag == 0x1601)
  {
    imRicoh.SensorWidth = get4();
  }
  else if (tag == 0x1602)
  {
    imRicoh.SensorHeight = get4();
  }
  else if (tag == 0x1603)
  {
    imRicoh.CroppedImageWidth = get4();
  }
  else if (tag == 0x1604)
  {
    imRicoh.CroppedImageHeight= get4();
  }
  else if ((tag == 0x2001) && !strncmp(model, "GXR", 3))
  {
    short cur_tag;
    fseek(ifp, 20, SEEK_CUR);
    /*ntags =*/ get2();
    cur_tag = get2();
    while (cur_tag != 0x002c)
    {
      fseek(ifp, 10, SEEK_CUR);
      cur_tag = get2();
    }
    fseek(ifp, 6, SEEK_CUR);
    fseek(ifp, get4(), SEEK_SET);
    for (int i=0; i<4; i++) {
      stread(buffer, 16, ifp);
      if ((buffer[0] == 'S') && (buffer[1] == 'I') && (buffer[2] == 'D'))
        memcpy(imgdata.shootinginfo.BodySerial, buffer+4, 12);
      else if ((buffer[0] == 'R') && (buffer[1] == 'L'))
        ilm.LensID = buffer[2] - '0';
      else if ((buffer[0] == 'L') && (buffer[1] == 'I') && (buffer[2] == 'D'))
        memcpy(imgdata.lens.LensSerial, buffer+4, 12);
    }
  }
}
