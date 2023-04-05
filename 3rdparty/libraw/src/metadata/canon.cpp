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

libraw_area_t LibRaw::get_CanonArea() {
  libraw_area_t la = {};
  la.l = get2();
  la.t = get2();
  la.r = get2();
  la.b = get2();
  return la;
}

float LibRaw::_CanonConvertAperture(ushort in)
{
  if ((in == (ushort)0xffe0) || (in == (ushort)0x7fff))
    return 0.0f;
  return LibRaw::libraw_powf64l(2.f, float(in) / 64.f);
}

static float _CanonConvertEV(short in)
{
  short EV, Sign, Frac;
  float Frac_f;
  EV = in;
  if (EV < 0)
  {
    EV = -EV;
    Sign = -1;
  }
  else
  {
    Sign = 1;
  }
  Frac = EV & 0x1f;
  EV -= Frac; // remove fraction

  if (Frac == 0x0c)
  { // convert 1/3 and 2/3 codes
    Frac_f = 32.0f / 3.0f;
  }
  else if (Frac == 0x14)
  {
    Frac_f = 64.0f / 3.0f;
  }
  else
    Frac_f = (float)Frac;

  return ((float)Sign * ((float)EV + Frac_f)) / 32.0f;
}

void LibRaw::setCanonBodyFeatures(unsigned long long id)
{

  ilm.CamID = id;
  if ((id == CanonID_EOS_1D)           ||
      (id == CanonID_EOS_1D_Mark_II)   ||
      (id == CanonID_EOS_1D_Mark_II_N) ||
      (id == CanonID_EOS_1D_Mark_III)  ||
      (id == CanonID_EOS_1D_Mark_IV))
  {
    ilm.CameraFormat = LIBRAW_FORMAT_APSH;
    ilm.CameraMount = LIBRAW_MOUNT_Canon_EF;
  }
  else if ((id == CanonID_EOS_1Ds)           ||
           (id == CanonID_EOS_1Ds_Mark_II)   ||
           (id == CanonID_EOS_1Ds_Mark_III)  ||
           (id == CanonID_EOS_1D_X)          ||
           (id == CanonID_EOS_1D_X_Mark_II)  ||
           (id == CanonID_EOS_1D_X_Mark_III) ||
           (id == CanonID_EOS_1D_C)          ||
           (id == CanonID_EOS_5D)            ||
           (id == CanonID_EOS_5D_Mark_II)    ||
           (id == CanonID_EOS_5D_Mark_III)   ||
           (id == CanonID_EOS_5D_Mark_IV)    ||
           (id == CanonID_EOS_5DS)           ||
           (id == CanonID_EOS_5DS_R)         ||
           (id == CanonID_EOS_6D)            ||
           (id == CanonID_EOS_6D_Mark_II))
  {
    ilm.CameraFormat = LIBRAW_FORMAT_FF;
    ilm.CameraMount = LIBRAW_MOUNT_Canon_EF;
  }
  else if ((id == CanonID_EOS_M)             ||
           (id == CanonID_EOS_M2)            ||
           (id == CanonID_EOS_M3)            ||
           (id == CanonID_EOS_M5)            ||
           (id == CanonID_EOS_M10)           ||
           (id == CanonID_EOS_M50)           ||
           (id == CanonID_EOS_M50_Mark_II)   ||
           (id == CanonID_EOS_M6)            ||
           (id == CanonID_EOS_M6_Mark_II)    ||
           (id == CanonID_EOS_M100))
  {
    ilm.CameraFormat = LIBRAW_FORMAT_APSC;
    ilm.CameraMount = LIBRAW_MOUNT_Canon_EF_M;
  }
  else if ((id == CanonID_EOS_R)  ||
           (id == CanonID_EOS_RP) ||
           (id == CanonID_EOS_R3) ||
           (id == CanonID_EOS_R5) ||
           (id == CanonID_EOS_R6))
  {
    ilm.CameraFormat = LIBRAW_FORMAT_FF;
    ilm.CameraMount = LIBRAW_MOUNT_Canon_RF;
    ilm.LensFormat = LIBRAW_FORMAT_FF;
    ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
  }

  else if ((id == CanonID_EOS_R7)  ||
           (id == CanonID_EOS_R10))
  {
    ilm.CameraFormat = LIBRAW_FORMAT_APSC;
    ilm.CameraMount = LIBRAW_MOUNT_Canon_RF;
    ilm.LensFormat = LIBRAW_FORMAT_APSC;
    ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
  }

  else if ((id == CanonID_EOS_D30) ||
           (id == CanonID_EOS_D60) ||
           (id > 0x80000000ULL))
  {
    ilm.CameraFormat = LIBRAW_FORMAT_APSC;
    ilm.CameraMount = LIBRAW_MOUNT_Canon_EF;
  }
}

int CanonCameraInfo_checkFirmwareRecordLocation (uchar *offset) {
// firmware record location allows
// to determine the subversion of the CameraInfo table
// and to adjust offsets accordingly
	if (
				isdigit(*offset)     &&
				isdigit(*(offset+2)) &&
				isdigit(*(offset+4)) &&
				(*(offset+1) == '.') &&
				(*(offset+3) == '.') &&
				((*(offset+5) == 0) || isspace(*(offset+5)))
			) return 1;
  else return 0; // error
}

void LibRaw::processCanonCameraInfo(unsigned long long id, uchar *CameraInfo,
                                    unsigned maxlen, unsigned type, unsigned dng_writer)
{
  ushort iCanonLensID = 0, iCanonMaxFocal = 0, iCanonMinFocal = 0,
         iCanonLens = 0, iCanonCurFocal = 0, iCanonFocalType = 0,
         iMakernotesFlip = 0,
         iHTP = 0, iALO = 0;
  short SubVersion_offset = 0;
  ushort SubVersion = 0, mgck = 0;

  if (maxlen < 16)
    return; // too short

  mgck = sget2(CameraInfo);
  CameraInfo[0] = 0;
  CameraInfo[1] = 0;
  if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG)) {
    if ((maxlen == 94)  || (maxlen == 138) || (maxlen == 148) ||
        (maxlen == 156) || (maxlen == 162) || (maxlen == 167) ||
        (maxlen == 171) || (maxlen == 264) || (maxlen > 400))
      imCommon.CameraTemperature = float(sget4(CameraInfo + ((maxlen - 3) << 2)));
    else if (maxlen == 72)
      imCommon.CameraTemperature = float(sget4(CameraInfo + ((maxlen - 1) << 2)));
    else if ((maxlen == 85) || (maxlen == 93))
      imCommon.CameraTemperature = float(sget4(CameraInfo + ((maxlen - 2) << 2)));
    else if ((maxlen == 96) || (maxlen == 104))
      imCommon.CameraTemperature = float(sget4(CameraInfo + ((maxlen - 4) << 2)));
  }

  switch (id)
  {
  case CanonID_EOS_1D:
  case CanonID_EOS_1Ds:
    iCanonCurFocal  =  0x0a;
    iCanonLensID    =  0x0d;
    iCanonMinFocal  =  0x0e;
    iCanonMaxFocal  =  0x10;
    if (!ilm.CurFocal)
      ilm.CurFocal = sget2(CameraInfo + iCanonCurFocal);
    if (!ilm.MinFocal)
      ilm.MinFocal = sget2(CameraInfo + iCanonMinFocal);
    if (!ilm.MaxFocal)
      ilm.MaxFocal = sget2(CameraInfo + iCanonMaxFocal);
    imCommon.CameraTemperature = 0.0f;
    break;

  case CanonID_EOS_1D_Mark_II:
  case CanonID_EOS_1Ds_Mark_II:
    iCanonCurFocal  =  0x09;
    iCanonLensID    =  0x0c;
    iCanonMinFocal  =  0x11;
    iCanonMaxFocal  =  0x13;
    iCanonFocalType =  0x2d;
    break;

  case CanonID_EOS_1D_Mark_II_N:
    iCanonCurFocal  =  0x09;
    iCanonLensID    =  0x0c;
    iCanonMinFocal  =  0x11;
    iCanonMaxFocal  =  0x13;
    break;

  case CanonID_EOS_1D_Mark_III:
  case CanonID_EOS_1Ds_Mark_III:
    iCanonCurFocal  =  0x1d;
    iMakernotesFlip =  0x30;
    iCanonLensID    = 0x111;
    iCanonMinFocal  = 0x113;
    iCanonMaxFocal  = 0x115;
    break;

  case CanonID_EOS_1D_Mark_IV:
    if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x1e8))
      SubVersion = 1;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x1ed))
      SubVersion = 2;
// printf ("==>> CanonID_EOS_1D_Mark_IV, SubVersion: %d\n", SubVersion);
    iHTP            =  0x07;
    iCanonCurFocal  =  0x1e;
    iMakernotesFlip =  0x35;

    if (!SubVersion)
      break;
    else if (SubVersion < 2)
      SubVersion_offset += -1;

    iCanonLensID    = 0x14f+SubVersion_offset;
    iCanonMinFocal  = 0x151+SubVersion_offset;
    iCanonMaxFocal  = 0x153+SubVersion_offset;
    break;

  case CanonID_EOS_1D_X:
    if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x271))
      SubVersion = 1;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x279))
      SubVersion = 2;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x280))
      SubVersion = 3;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x285))
      SubVersion = 4;
// printf ("==>> CanonID_EOS_1D_X, SubVersion: %d\n", SubVersion);

    if (SubVersion < 3)
      SubVersion_offset += -3;

    iCanonCurFocal  =  0x23+SubVersion_offset;
    iMakernotesFlip =  0x7d+SubVersion_offset;

    if (SubVersion < 3)
      SubVersion_offset += -4;
    else if (SubVersion == 4)
      SubVersion_offset += 5;

    iCanonLensID    = 0x1a7+SubVersion_offset;
    iCanonMinFocal  = 0x1a9+SubVersion_offset;
    iCanonMaxFocal  = 0x1ab+SubVersion_offset;
    break;

  case CanonID_EOS_5D:
    iMakernotesFlip =  0x27;
    iCanonCurFocal  =  0x28;
    iCanonLensID    =  0x0c;
    if (!sget2Rev(CameraInfo + iCanonLensID))
      iCanonLensID  =  0x97;
    iCanonMinFocal  =  0x93;
    iCanonMaxFocal  =  0x95;
    break;

  case CanonID_EOS_5D_Mark_II:
    iHTP            =  0x07;
    iCanonCurFocal  =  0x1e;
    iMakernotesFlip =  0x31;
    iALO            =  0xbf;
    iCanonLensID    =  0xe6;
    iCanonMinFocal  =  0xe8;
    iCanonMaxFocal  =  0xea;
    break;

  case CanonID_EOS_5D_Mark_III:
    if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x22c))
      SubVersion = 1;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x22d))
      SubVersion = 2;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x23c))
      SubVersion = 3;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x242))
      SubVersion = 4;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x247))
      SubVersion = 5;
// printf ("==>> CanonID_EOS_5D_Mark_III, SubVersion: %d\n", SubVersion);

    if (!SubVersion)
      break;
    else if (SubVersion < 3)
      SubVersion_offset += -1;

    iCanonCurFocal  =  0x23+SubVersion_offset;

    if (SubVersion == 1)
      SubVersion_offset += -3;
    else if (SubVersion == 2)
      SubVersion_offset += -2;
    else if (SubVersion >= 4)
      SubVersion_offset += 6;

    iMakernotesFlip =  0x7d+SubVersion_offset;

    if (SubVersion < 3)
      SubVersion_offset += -4;
    else if (SubVersion > 4)
      SubVersion_offset += 5;

    iCanonLensID    = 0x153+SubVersion_offset;
    iCanonMinFocal  = 0x155+SubVersion_offset;
    iCanonMaxFocal  = 0x157+SubVersion_offset;
    break;

  case CanonID_EOS_6D:
    iCanonCurFocal  =  0x23;
    iMakernotesFlip =  0x83;
    iCanonLensID    = 0x161;
    iCanonMinFocal  = 0x163;
    iCanonMaxFocal  = 0x165;
    break;

  case CanonID_EOS_7D:
    if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x1a8))
      SubVersion = 1;
    else if (CanonCameraInfo_checkFirmwareRecordLocation(CameraInfo + 0x1ac))
      SubVersion = 2;
// printf ("==>> CanonID_EOS_7D, SubVersion: %d\n", SubVersion);
    iHTP            =  0x07;
    iCanonCurFocal  =  0x1e;

    if (!SubVersion)
      break;
    else if (SubVersion < 2)
      SubVersion_offset += -4;

    iMakernotesFlip =  0x35+SubVersion_offset;
    iCanonLensID    = 0x112+SubVersion_offset;
    iCanonMinFocal  = 0x114+SubVersion_offset;
    iCanonMaxFocal  = 0x116+SubVersion_offset;
    break;

  case CanonID_EOS_40D:
    iCanonCurFocal  =  0x1d;
    iMakernotesFlip =  0x30;
    iCanonLensID    =  0xd6;
    iCanonMinFocal  =  0xd8;
    iCanonMaxFocal  =  0xda;
    iCanonLens      = 0x92b;
    break;

  case CanonID_EOS_50D:
    iHTP            =  0x07;
    iCanonCurFocal  =  0x1e;
    iMakernotesFlip =  0x31;
    iALO            =  0xbf;
    iCanonLensID    =  0xea;
    iCanonMinFocal  =  0xec;
    iCanonMaxFocal  =  0xee;
    break;

  case CanonID_EOS_60D:
  case CanonID_EOS_1200D:
    iCanonCurFocal  =  0x1e;
    if (id == CanonID_EOS_60D)
      iMakernotesFlip =  0x36;
    else
      iMakernotesFlip =  0x3a;
    iCanonLensID    =  0xe8;
    iCanonMinFocal  =  0xea;
    iCanonMaxFocal  =  0xec;
    break;

  case CanonID_EOS_70D:
    iCanonCurFocal  =  0x23;
    iMakernotesFlip =  0x84;
    iCanonLensID    = 0x166;
    iCanonMinFocal  = 0x168;
    iCanonMaxFocal  = 0x16a;
    break;

  case CanonID_EOS_80D:
    iCanonCurFocal  =  0x23;
    iMakernotesFlip =  0x96;
    iCanonLensID    = 0x189;
    iCanonMinFocal  = 0x18b;
    iCanonMaxFocal  = 0x18d;
    break;

  case CanonID_EOS_450D:
    iCanonCurFocal  =  0x1d;
    iMakernotesFlip =  0x30;
    iCanonLensID    =  0xde;
    iCanonLens      = 0x933;
    break;

  case CanonID_EOS_500D:
    iHTP            =  0x07;
    iCanonCurFocal  =  0x1e;
    iMakernotesFlip =  0x31;
    iALO            =  0xbe;
    iCanonLensID    =  0xf6;
    iCanonMinFocal  =  0xf8;
    iCanonMaxFocal  =  0xfa;
    break;

  case CanonID_EOS_550D:
    iHTP            =  0x07;
    iCanonCurFocal  =  0x1e;
    iMakernotesFlip =  0x35;
    iCanonLensID    =  0xff;
    iCanonMinFocal  = 0x101;
    iCanonMaxFocal  = 0x103;
    break;

  case CanonID_EOS_600D:
  case CanonID_EOS_1100D:
    iHTP            =  0x07;
    iCanonCurFocal  =  0x1e;
    iMakernotesFlip =  0x38;
    iCanonLensID    =  0xea;
    iCanonMinFocal  =  0xec;
    iCanonMaxFocal  =  0xee;
    break;

  case CanonID_EOS_650D:
  case CanonID_EOS_700D:
    iCanonCurFocal  =  0x23;
    iMakernotesFlip =  0x7d;
    iCanonLensID    = 0x127;
    iCanonMinFocal  = 0x129;
    iCanonMaxFocal  = 0x12b;
    break;

  case CanonID_EOS_750D:
  case CanonID_EOS_760D:
    iCanonCurFocal  =  0x23;
    iMakernotesFlip =  0x96;
    iCanonLensID    = 0x184;
    iCanonMinFocal  = 0x186;
    iCanonMaxFocal  = 0x188;
    break;

  case CanonID_EOS_1000D:
    iCanonCurFocal  =  0x1d;
    iMakernotesFlip =  0x30;
    iCanonLensID    =  0xe2;
    iCanonMinFocal  =  0xe4;
    iCanonMaxFocal  =  0xe6;
    iCanonLens      = 0x937;
    break;
  }

  if (iMakernotesFlip && (CameraInfo[iMakernotesFlip] < 3)) {
    imCanon.MakernotesFlip = "065"[CameraInfo[iMakernotesFlip]] - '0';
// printf ("==>> iMakernotesFlip: 0x%x, flip: %d\n", iMakernotesFlip, imCanon.MakernotesFlip);
  } else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_UNDEFINED) &&
     (mgck == 0xaaaa) && (dng_writer == nonDNG)) { // CameraOrientation
    int c, i;
    for (i = 2; (sget2(CameraInfo+i) != 0xbbbb) && i < (int)maxlen; i++);
    i+=2;
    while (i < int(maxlen - 5))
      if ((sget4(CameraInfo+i) == 257) && ((c = CameraInfo[i+8]) < 3)) {
        imCanon.MakernotesFlip = "065"[c] - '0';
// printf ("==>> MakernotesFlip offset: 0x%x, flip: %d\n", i+8, imCanon.MakernotesFlip);
        break;
      } else i+=4;
  }

  if (iHTP)
  {
    imCanon.HighlightTonePriority = CameraInfo[iHTP];
    if ((imCanon.HighlightTonePriority > 5) ||
        (imCanon.HighlightTonePriority < 0))
      imCanon.HighlightTonePriority = 0;
    if (imCanon.HighlightTonePriority) {
      imCommon.ExposureCalibrationShift -= float(imCanon.HighlightTonePriority);
    }
  }
  if (iALO)
  {
    imCanon.AutoLightingOptimizer = CameraInfo[iALO];
    if ((imCanon.AutoLightingOptimizer > 3) ||
        (imCanon.AutoLightingOptimizer < 0))
      imCanon.AutoLightingOptimizer = 3;
  }
  if (iCanonFocalType)
  {
    if (iCanonFocalType >= maxlen)
      return; // broken;
    ilm.FocalType = CameraInfo[iCanonFocalType];
    if (!ilm.FocalType) // zero means 'prime' here, replacing with standard '1'
      ilm.FocalType = LIBRAW_FT_PRIME_LENS;
  }
  if (!ilm.CurFocal && iCanonCurFocal)
  {
    if (iCanonCurFocal >= maxlen)
      return; // broken;
    ilm.CurFocal = sget2Rev(CameraInfo + iCanonCurFocal);
  }
  if (!ilm.LensID && iCanonLensID)
  {
    if (iCanonLensID >= maxlen)
      return; // broken;
    ilm.LensID = sget2Rev(CameraInfo + iCanonLensID);
  }
  if (!ilm.MinFocal && iCanonMinFocal)
  {
    if (iCanonMinFocal >= maxlen)
      return; // broken;
    ilm.MinFocal = sget2Rev(CameraInfo + iCanonMinFocal);
  }
  if (!ilm.MaxFocal && iCanonMaxFocal)
  {
    if (iCanonMaxFocal >= maxlen)
      return; // broken;
    ilm.MaxFocal = sget2Rev(CameraInfo + iCanonMaxFocal);
  }
  if (!ilm.Lens[0] && iCanonLens)
  {
    if (iCanonLens + 64 >= (int)maxlen) // broken;
      return;

    char *pl = (char *)CameraInfo + iCanonLens;
    if (!strncmp(pl, "EF-S", 4))
    {
      memcpy(ilm.Lens, pl, 4);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, pl, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF_S;
      ilm.LensFormat = LIBRAW_FORMAT_APSC;
      memcpy(ilm.Lens + 5, pl + 4, 60);
    }
    else if (!strncmp(pl, "EF-M", 4))
    {
      memcpy(ilm.Lens, pl, 4);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, pl, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF_M;
      ilm.LensFormat = LIBRAW_FORMAT_APSC;
      memcpy(ilm.Lens + 5, pl + 4, 60);
    }
    else if (!strncmp(pl, "EF", 2))
    {
      memcpy(ilm.Lens, pl, 2);
      ilm.Lens[2] = ' ';
      memcpy(ilm.LensFeatures_pre, pl, 2);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
      memcpy(ilm.Lens + 3, pl + 2, 62);
    }
    else if (!strncmp(ilm.Lens, "CN-E", 4))
    {
      memmove(ilm.Lens + 5, ilm.Lens + 4, 60);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
    }
    else if (!strncmp(pl, "TS-E", 4))
    {
      memcpy(ilm.Lens, pl, 4);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, pl, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
      memcpy(ilm.Lens + 5, pl + 4, 60);
    }
    else if (!strncmp(pl, "MP-E", 4))
    {
      memcpy(ilm.Lens, pl, 4);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, pl, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
      memcpy(ilm.Lens + 5, pl + 4, 60);
    }
    else // non-Canon lens
      memcpy(ilm.Lens, pl, 64);
  }
  return;
}

void LibRaw::Canon_CameraSettings(unsigned len)
{
  fseek(ifp, 6, SEEK_CUR);
  imCanon.Quality = get2();   // 3
  get2();
  imgdata.shootinginfo.DriveMode = get2(); // 5
  get2();
  imgdata.shootinginfo.FocusMode = get2(); // 7
  imCanon.RecordMode = (get2(), get2());   // 9, format
  fseek(ifp, 14, SEEK_CUR);
  imgdata.shootinginfo.MeteringMode = get2(); // 17
  get2();
  imgdata.shootinginfo.AFPoint = get2();      // 19
  imgdata.shootinginfo.ExposureMode = get2(); // 20
  get2();
  ilm.LensID = get2();          // 22
  ilm.MaxFocal = get2();        // 23
  ilm.MinFocal = get2();        // 24
  ilm.FocalUnits = get2();      // 25
  if (ilm.FocalUnits > 1)
  {
    ilm.MaxFocal /= (float)ilm.FocalUnits;
    ilm.MinFocal /= (float)ilm.FocalUnits;
  }
  ilm.MaxAp = _CanonConvertAperture(get2()); // 26
  ilm.MinAp = _CanonConvertAperture(get2()); // 27
  if (len >= 36)
  {
    fseek(ifp, 12, SEEK_CUR);
    imgdata.shootinginfo.ImageStabilization = get2(); // 34
  }
  else
    return;
  if (len >= 48)
  {
    fseek(ifp, 22, SEEK_CUR);
    imCanon.SRAWQuality = get2(); // 46
  }
}

void LibRaw::Canon_WBpresets(int skip1, int skip2)
{
  int c;
  FORC4 imgdata.color.WB_Coeffs[LIBRAW_WBI_Daylight][RGGB_2_RGBG(c)] = get2();

  if (skip1)
    fseek(ifp, skip1, SEEK_CUR);
  FORC4 imgdata.color.WB_Coeffs[LIBRAW_WBI_Shade][RGGB_2_RGBG(c)] = get2();

  if (skip1)
    fseek(ifp, skip1, SEEK_CUR);
  FORC4 imgdata.color.WB_Coeffs[LIBRAW_WBI_Cloudy][RGGB_2_RGBG(c)] = get2();

  if (skip1)
    fseek(ifp, skip1, SEEK_CUR);
  FORC4 imgdata.color.WB_Coeffs[LIBRAW_WBI_Tungsten][RGGB_2_RGBG(c)] = get2();

  if (skip1)
    fseek(ifp, skip1, SEEK_CUR);
  FORC4 imgdata.color.WB_Coeffs[LIBRAW_WBI_FL_W][RGGB_2_RGBG(c)] = get2();

  if (skip2)
    fseek(ifp, skip2, SEEK_CUR);
  FORC4 imgdata.color.WB_Coeffs[LIBRAW_WBI_Flash][RGGB_2_RGBG(c)] = get2();

  return;
}

void LibRaw::Canon_WBCTpresets(short WBCTversion)
{

  int i;
  float norm;

  if (WBCTversion == 0)
  { // tint, as shot R, as shot B, CСT
    for (i = 0; i < 15; i++)
    {
      icWBCCTC[i][2] = icWBCCTC[i][4] = 1.0f;
      fseek(ifp, 2, SEEK_CUR);
      icWBCCTC[i][1] = 1024.0f / fMAX(get2(), 1.f);
      icWBCCTC[i][3] = 1024.0f / fMAX(get2(), 1.f);
      icWBCCTC[i][0] = get2();
    }
  }
  else if (WBCTversion == 1)
  { // as shot R, as shot B, tint, CСT
    for (i = 0; i < 15; i++)
    {
      icWBCCTC[i][2] = icWBCCTC[i][4] = 1.0f;
      icWBCCTC[i][1] = 1024.0f / fMAX(get2(), 1.f);
      icWBCCTC[i][3] = 1024.0f / fMAX(get2(), 1.f);
      fseek(ifp, 2, SEEK_CUR);
      icWBCCTC[i][0] = get2();
    }
  }
  else if (WBCTversion == 2)
  { // tint, offset, as shot R, as shot B, CСT
    if ((unique_id == CanonID_EOS_M3)  ||
        (unique_id == CanonID_EOS_M10) ||
        (imCanon.ColorDataSubVer == 0xfffc))
    {
      for (i = 0; i < 15; i++)
      {
        fseek(ifp, 4, SEEK_CUR);
        icWBCCTC[i][2] = icWBCCTC[i][4] =
            1.0f;
        icWBCCTC[i][1] = 1024.0f / fMAX(1.f, get2());
        icWBCCTC[i][3] = 1024.0f / fMAX(1.f, get2());
        icWBCCTC[i][0] = get2();
      }
    }
    else if (imCanon.ColorDataSubVer == 0xfffd)
    {
      for (i = 0; i < 15; i++)
      {
        fseek(ifp, 2, SEEK_CUR);
        norm = (signed short)get2();
        norm = 512.0f + norm / 8.0f;
        icWBCCTC[i][2] = icWBCCTC[i][4] =
            1.0f;
        icWBCCTC[i][1] = (float)get2();
        if (norm > 0.001f)
          icWBCCTC[i][1] /= norm;
        icWBCCTC[i][3] = (float)get2();
        if (norm > 0.001f)
          icWBCCTC[i][3] /= norm;
        icWBCCTC[i][0] = get2();
      }
    }
  }
  return;
}

void LibRaw::parseCanonMakernotes(unsigned tag, unsigned /*type*/, unsigned len, unsigned dng_writer)
{

#define AsShot_Auto_MeasuredWB(offset)                       \
  imCanon.ColorDataSubVer = get2();                          \
  fseek(ifp, save1 + (offset << 1), SEEK_SET);               \
  FORC4 cam_mul[RGGB_2_RGBG(c)] = (float)get2();             \
  get2();                                                    \
  FORC4 icWBC[LIBRAW_WBI_Auto][RGGB_2_RGBG(c)] = get2();     \
  get2();                                                    \
  FORC4 icWBC[LIBRAW_WBI_Measured][RGGB_2_RGBG(c)] = get2();

#define sRAW_WB(offset)                                      \
  fseek(ifp, save1 + (offset << 1), SEEK_SET);               \
  FORC4 {                                                    \
    sraw_mul[RGGB_2_RGBG(c)] = get2();                       \
    if ((float)sraw_mul[RGGB_2_RGBG(c)] > sraw_mul_max) {    \
      sraw_mul_max = (float)sraw_mul[RGGB_2_RGBG(c)];        \
    }                                                        \
  }                                                          \
  sraw_mul_max /= 1024.f;                                    \
  FORC4 sraw_mul[c] = (ushort)((float)sraw_mul[c] * sraw_mul_max);

#define CR3_ColorData(offset)                                \
  fseek(ifp, save1 + ((offset+0x0041) << 1), SEEK_SET);      \
  Canon_WBpresets(2, 12);                                    \
  fseek(ifp, save1 + ((offset+0x00c3) << 1), SEEK_SET);      \
  Canon_WBCTpresets(0);                                      \
  offsetChannelBlackLevel2 = save1 + ((offset+0x0102) << 1); \
  offsetChannelBlackLevel  = save1 + ((offset+0x02d1) << 1); \
  offsetWhiteLevels        = save1 + ((offset+0x02d5) << 1);

  int c;
  unsigned i;

  if (tag == 0x0001) {
    Canon_CameraSettings(len);

  } else if (tag == 0x0002) { // focal length
    ilm.FocalType = get2();
    ilm.CurFocal = get2();
    if (ilm.FocalUnits > 1) {
      ilm.CurFocal /= (float)ilm.FocalUnits;
    }

  } else if (tag == 0x0004) { // subdir, ShotInfo
    short tempAp;
    if (dng_writer == nonDNG) {
      get2();
      imCanon.ISOgain[0] = get2();
      imCanon.ISOgain[1] = get2();
      if (imCanon.ISOgain[1] != 0x7fff) {
        imCommon.real_ISO = floorf(100.f * libraw_powf64l(2.f, float(imCanon.ISOgain[0]+imCanon.ISOgain[1]) / 32.f - 5.f));
        if (!iso_speed || (iso_speed == 65535))
          iso_speed = imCommon.real_ISO;
      }
      get4();
      if (((i = get2()) != 0xffff) && !shutter) {
        shutter = libraw_powf64l(2.f, float((short)i) / -32.0f);
      }
      imCanon.wbi = (get2(), get2());
      shot_order = (get2(), get2());
      fseek(ifp, 4, SEEK_CUR);
    } else
      fseek(ifp, 24, SEEK_CUR);
    tempAp = get2();
    if (tempAp != 0)
      imCommon.CameraTemperature = (float)(tempAp - 128);
    tempAp = get2();
    if (tempAp != -1)
      imCommon.FlashGN = ((float)tempAp) / 32;
    get2();

    imCommon.FlashEC = _CanonConvertEV((signed short)get2());
    fseek(ifp, 8 - 32, SEEK_CUR);
    if ((tempAp = get2()) != 0x7fff)
      ilm.CurAp = _CanonConvertAperture(tempAp);
    if (ilm.CurAp < 0.7f) {
      fseek(ifp, 32, SEEK_CUR);
      ilm.CurAp = _CanonConvertAperture(get2());
    }
    if (!aperture)
      aperture = ilm.CurAp;

  } else if ((tag == 0x0007) && (dng_writer == nonDNG)) {
    fgets(model2, 64, ifp);

  } else if ((tag == 0x0008) && (dng_writer == nonDNG)) {
    shot_order = get4();

  } else if ((tag == 0x0009)  && (dng_writer == nonDNG)) {
    fread(artist, 64, 1, ifp);

  } else if (tag == 0x000c) {
    unsigned tS = get4();
    sprintf(imgdata.shootinginfo.BodySerial, "%d", tS);

  } else if ((tag == 0x0012) ||
             (tag == 0x0026) ||
             (tag == 0x003c)) {
    if (!imCommon.afcount) {
      imCommon.afdata[imCommon.afcount].AFInfoData_tag = tag;
      imCommon.afdata[imCommon.afcount].AFInfoData_order = order;
      imCommon.afdata[imCommon.afcount].AFInfoData_length = len;
      imCommon.afdata[imCommon.afcount].AFInfoData = (uchar *)malloc(imCommon.afdata[imCommon.afcount].AFInfoData_length);
      fread(imCommon.afdata[imCommon.afcount].AFInfoData, imCommon.afdata[imCommon.afcount].AFInfoData_length, 1, ifp);
      imCommon.afcount = 1;
    }

  } else if ((tag == 0x0029) && (dng_writer == nonDNG)) { // PowerShot G9
    int Got_AsShotWB = 0;
    fseek(ifp, 8, SEEK_CUR);
    for (unsigned linenum = 0; linenum < Canon_G9_linenums_2_StdWBi.size(); linenum++) {
      if (Canon_G9_linenums_2_StdWBi[linenum] != LIBRAW_WBI_Unknown ) {
        FORC4 icWBC[Canon_G9_linenums_2_StdWBi[linenum]][GRBG_2_RGBG(c)] = get4();
        if (Canon_wbi2std[imCanon.wbi] == Canon_G9_linenums_2_StdWBi[linenum]) {
          FORC4 cam_mul[c] = float(icWBC[Canon_G9_linenums_2_StdWBi[linenum]][c]);
          Got_AsShotWB = 1;
        }
      }
      fseek(ifp, 16, SEEK_CUR);
    }
    if (!Got_AsShotWB)
      FORC4 cam_mul[c] = float(icWBC[LIBRAW_WBI_Auto][c]);

  } else if ((tag == 0x0081) && (dng_writer == nonDNG)) { // -1D, -1Ds
    data_offset = get4();
    fseek(ifp, data_offset + 41, SEEK_SET);
    raw_height = get2() * 2;
    raw_width = get2();
    filters = 0x61616161;

  } else if (tag == 0x0093) {
    if (!imCanon.RF_lensID) {
      fseek(ifp, 0x03d<<1, SEEK_CUR);
      imCanon.RF_lensID = get2();
    }

  } else if (tag == 0x0095 && !ilm.Lens[0])
  { // lens model tag
    fread(ilm.Lens, 64, 1, ifp);
    if (!strncmp(ilm.Lens, "EF-S", 4))
    {
      memmove(ilm.Lens + 5, ilm.Lens + 4, 60);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF_S;
      ilm.LensFormat = LIBRAW_FORMAT_APSC;
    }
    else if (!strncmp(ilm.Lens, "EF-M", 4))
    {
      memmove(ilm.Lens + 5, ilm.Lens + 4, 60);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF_M;
      ilm.LensFormat = LIBRAW_FORMAT_APSC;
    }
    else if (!strncmp(ilm.Lens, "EF", 2))
    {
      memmove(ilm.Lens + 3, ilm.Lens + 2, 62);
      ilm.Lens[2] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 2);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
    }
    else if (!strncmp(ilm.Lens, "CN-E", 4))
    {
      memmove(ilm.Lens + 5, ilm.Lens + 4, 60);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
    }
    else if (!strncmp(ilm.Lens, "TS-E", 4))
    {
      memmove(ilm.Lens + 5, ilm.Lens + 4, 60);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
    }
    else if (!strncmp(ilm.Lens, "MP-E", 4))
    {
      memmove(ilm.Lens + 5, ilm.Lens + 4, 60);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_EF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
    }

    else if (!strncmp(ilm.Lens, "RF-S", 4))
    {
      memmove(ilm.Lens + 5, ilm.Lens + 4, 62);
      ilm.Lens[4] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 4);
      ilm.LensMount = LIBRAW_MOUNT_Canon_RF;
      ilm.LensFormat = LIBRAW_FORMAT_APSC;
    }

    else if (!strncmp(ilm.Lens, "RF", 2))
    {
      memmove(ilm.Lens + 3, ilm.Lens + 2, 62);
      ilm.Lens[2] = ' ';
      memcpy(ilm.LensFeatures_pre, ilm.Lens, 2);
      ilm.LensMount = LIBRAW_MOUNT_Canon_RF;
      ilm.LensFormat = LIBRAW_FORMAT_FF;
    }
  }
  else if (tag == 0x009a)
  { // AspectInfo
    i = get4();
    switch (i)
    {
    case 0:
    case 12: /* APS-H crop */
    case 13: /* APS-C crop */
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_3to2;
      break;
    case 1:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_1to1;
      break;
    case 2:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_4to3;
      break;
    case 7:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_16to9;
      break;
    case 8:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_5to4;
      break;
    default:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_OTHER;
      break;
    }
    imgdata.sizes.raw_inset_crops[0].cwidth = get4();
    imgdata.sizes.raw_inset_crops[0].cheight = get4();
    imgdata.sizes.raw_inset_crops[0].cleft = get4();
    imgdata.sizes.raw_inset_crops[0].ctop = get4();

  } else if ((tag == 0x00a4) && (dng_writer == nonDNG)) { // -1D, -1Ds
    fseek(ifp, imCanon.wbi * 48, SEEK_CUR);
    FORC3 cam_mul[c] = get2();

  } else if (tag == 0x00a9) {
    INT64 save1 = ftell(ifp);
    fseek(ifp, (0x1 << 1), SEEK_CUR);
    FORC4 imgdata.color.WB_Coeffs[LIBRAW_WBI_Auto][RGGB_2_RGBG(c)] = get2();
    Canon_WBpresets(0, 0);
    fseek(ifp, save1, SEEK_SET);
  }
  else if (tag == 0x00b4)
  {
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
  }
  else if (tag == 0x00e0) // SensorInfo
  {
    imCanon.SensorWidth  = (get2(), get2());
    imCanon.SensorHeight = get2();
    fseek(ifp, 4, SEEK_CUR);
    imCanon.DefaultCropAbsolute = get_CanonArea();
    imCanon.LeftOpticalBlack    = get_CanonArea();
  }
  else if (tag == 0x4001 && len > 500)
  {
    float sraw_mul_max = 0.f;
    int bls = 0;
    INT64 offsetChannelBlackLevel = 0L;
    INT64 offsetChannelBlackLevel2 = 0L;
    INT64 offsetWhiteLevels = 0L;
    INT64 save1 = ftell(ifp);

    switch (len)
    {

    case 582:
      imCanon.ColorDataVer = 1; // 20D, 350D

      fseek(ifp, save1 + (0x0019 << 1), SEEK_SET);
      FORC4 cam_mul[RGGB_2_RGBG(c)] = (float)get2();
      fseek(ifp, save1 + (0x001e << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Auto][RGGB_2_RGBG(c)] = get2();
      fseek(ifp, save1 + (0x0041 << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom1][RGGB_2_RGBG(c)] = get2();
      fseek(ifp, save1 + (0x0046 << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom2][RGGB_2_RGBG(c)] = get2();

      fseek(ifp, save1 + (0x0023 << 1), SEEK_SET);
      Canon_WBpresets(2, 2);
      fseek(ifp, save1 + (0x004b << 1), SEEK_SET);
      Canon_WBCTpresets(1); // ABCT
      offsetChannelBlackLevel = save1 + (0x00a6 << 1);
      break;

    case 653:
      imCanon.ColorDataVer = 2; // -1D Mark II, -1Ds Mark II

      fseek(ifp, save1 + (0x0018 << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Auto][RGGB_2_RGBG(c)] = get2();
      fseek(ifp, save1 + (0x0022 << 1), SEEK_SET);
      FORC4 cam_mul[RGGB_2_RGBG(c)] = (float)get2();
      fseek(ifp, save1 + (0x0090 << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom1][RGGB_2_RGBG(c)] = get2();
      fseek(ifp, save1 + (0x0095 << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom2][RGGB_2_RGBG(c)] = get2();
      fseek(ifp, save1 + (0x009a << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom3][RGGB_2_RGBG(c)] = get2();

      fseek(ifp, save1 + (0x0027 << 1), SEEK_SET);
      Canon_WBpresets(2, 12);
      fseek(ifp, save1 + (0x00a4 << 1), SEEK_SET);
      Canon_WBCTpresets(1); // ABCT
      offsetChannelBlackLevel = save1 + (0x011e << 1);
      break;

    case 796:
      imCanon.ColorDataVer = 3; // -1D Mark II N, 5D, 30D, 400D; ColorDataSubVer: 1
      AsShot_Auto_MeasuredWB(0x003f);

      fseek(ifp, save1 + (0x0071 << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom1][RGGB_2_RGBG(c)] = get2();
      fseek(ifp, save1 + (0x0076 << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom2][RGGB_2_RGBG(c)] = get2();
      fseek(ifp, save1 + (0x007b << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom3][RGGB_2_RGBG(c)] = get2();
      fseek(ifp, save1 + (0x0080 << 1), SEEK_SET);
      FORC4 icWBC[LIBRAW_WBI_Custom][RGGB_2_RGBG(c)] = get2();

      fseek(ifp, save1 + (0x004e << 1), SEEK_SET);
      Canon_WBpresets(2, 12);
      fseek(ifp, save1 + (0x0085 << 1), SEEK_SET);
      Canon_WBCTpresets(0); // BCAT
      offsetChannelBlackLevel = save1 + (0x00c4 << 1);
      break;

    case 674:  // -1D Mark III; ColorDataSubVer: 2
    case 692:  // 40D; ColorDataSubVer: 3
    case 702:  // -1Ds Mark III; ColorDataSubVer: 4
    case 1227: // 450D, 1000D; ColorDataSubVer: 5
    case 1250: // 5D Mark II, 50D; ColorDataSubVer: 6
    case 1251: // 500D; ColorDataSubVer: 7
    case 1337: // -1D Mark IV, 7D; ColorDataSubVer: 7
    case 1338: // 550D; ColorDataSubVer: 7
    case 1346: // 1100D, 60D; ColorDataSubVer: 9
      imCanon.ColorDataVer = 4;
      AsShot_Auto_MeasuredWB(0x003f);
      sRAW_WB(0x004e);
      fseek(ifp, save1 + (0x0053 << 1), SEEK_SET);
      Canon_WBpresets(2, 12);
      fseek(ifp, save1 + (0x00a8 << 1), SEEK_SET);
      Canon_WBCTpresets(0); // BCAT

      if ((imCanon.ColorDataSubVer == 4) ||
          (imCanon.ColorDataSubVer == 5))
      {
        offsetChannelBlackLevel = save1 + (0x02b4 << 1);
        offsetWhiteLevels = save1 + (0x02b8 << 1);
      }
      else if ((imCanon.ColorDataSubVer == 6) ||
               (imCanon.ColorDataSubVer == 7))
      {
        offsetChannelBlackLevel = save1 + (0x02cb << 1);
        offsetWhiteLevels = save1 + (0x02cf << 1);
      }
      else if (imCanon.ColorDataSubVer == 9)
      {
        offsetChannelBlackLevel = save1 + (0x02cf << 1);
        offsetWhiteLevels = save1 + (0x02d3 << 1);
      }
      else
        offsetChannelBlackLevel = save1 + (0x00e7 << 1);
      break;

    case 5120: // G10, G11, G12, G15, G16
               // G1 X, G1 X Mark II, G1 X Mark III
               // G3 X, G5 X
               // G7 X, G7 X Mark II
               // G9 X, G9 X Mark II
               // S90, S95, S100, S100V, S110, S120
               // SX1 IS, SX50 HS, SX60 HS
               // M3, M5, M6, M10, M100
      imCanon.ColorDataVer = 5;
      imCanon.ColorDataSubVer = get2();

      fseek(ifp, save1 + (0x0047 << 1), SEEK_SET);
      FORC4 cam_mul[RGGB_2_RGBG(c)] = (float)get2();

      if (imCanon.ColorDataSubVer == 0xfffc) // ColorDataSubVer: 65532 (-4)
                                             // G7 X Mark II, G9 X Mark II, G1 X Mark III
                                             // M5, M100, M6
      {
        fseek(ifp, save1 + (0x004f << 1), SEEK_SET);
        FORC4 icWBC[LIBRAW_WBI_Auto][RGGB_2_RGBG(c)] = get2();
        fseek(ifp, 8, SEEK_CUR);
        FORC4 icWBC[LIBRAW_WBI_Measured][RGGB_2_RGBG(c)] =
            get2();
        fseek(ifp, 8, SEEK_CUR);
        FORC4 icWBC[LIBRAW_WBI_Other][RGGB_2_RGBG(c)] = get2();
        fseek(ifp, 8, SEEK_CUR);
        Canon_WBpresets(8, 24);
        fseek(ifp, 168, SEEK_CUR);
        FORC4 icWBC[LIBRAW_WBI_FL_WW][RGGB_2_RGBG(c)] = get2();
        fseek(ifp, 24, SEEK_CUR);
        Canon_WBCTpresets(2); // BCADT
        offsetChannelBlackLevel = save1 + (0x014d << 1);
        offsetWhiteLevels = save1 + (0x0569 << 1);
      }
      else if (imCanon.ColorDataSubVer == 0xfffd) // ColorDataSubVer: 65533 (-3)
                                                  // M10, M3
                                                  // G1 X, G1 X Mark II
                                                  // G3 X, G5 X, G7 X, G9 X
                                                  // G10, G11, G12, G15, G16
                                                  // S90, S95, S100, S100V, S110, S120
                                                  // SX1 IS, SX50 HS, SX60 HS
      {
        fseek(ifp, save1 + (0x004c << 1), SEEK_SET);
        FORC4 icWBC[LIBRAW_WBI_Auto][RGGB_2_RGBG(c)] = get2();
        get2();
        FORC4 icWBC[LIBRAW_WBI_Measured][RGGB_2_RGBG(c)] =
            get2();
        get2();
        FORC4 icWBC[LIBRAW_WBI_Other][RGGB_2_RGBG(c)] = get2();
        get2();
        Canon_WBpresets(2, 12);
        fseek(ifp, save1 + (0x00ba << 1), SEEK_SET);
        Canon_WBCTpresets(2); // BCADT
        offsetChannelBlackLevel = save1 + (0x0108 << 1);
      }
      break;

    case 1273: // 600D; ColorDataSubVer: 10
    case 1275: // 1200D; ColorDataSubVer: 10
      imCanon.ColorDataVer = 6;
      AsShot_Auto_MeasuredWB(0x003f);
      sRAW_WB(0x0062);
      fseek(ifp, save1 + (0x0067 << 1), SEEK_SET);
      Canon_WBpresets(2, 12);
      fseek(ifp, save1 + (0x00bc << 1), SEEK_SET);
      Canon_WBCTpresets(0); // BCAT
      offsetChannelBlackLevel = save1 + (0x01df << 1);
      offsetWhiteLevels = save1 + (0x01e3 << 1);
      break;

    case 1312: // 5D Mark III, 650D, 700D, M; ColorDataSubVer: 10
    case 1313: // 100D, 6D, 70D, EOS M2; ColorDataSubVer: 10
    case 1316: // -1D C, -1D X; ColorDataSubVer: 10
    case 1506: // 750D, 760D, 7D Mark II; ColorDataSubVer: 11
      imCanon.ColorDataVer = 7;
      AsShot_Auto_MeasuredWB(0x003f);
      sRAW_WB(0x007b);
      fseek(ifp, save1 + (0x0080 << 1), SEEK_SET);
      Canon_WBpresets(2, 12);
      fseek(ifp, save1 + (0x00d5 << 1), SEEK_SET);
      Canon_WBCTpresets(0); // BCAT

      if (imCanon.ColorDataSubVer == 10)
      {
        offsetChannelBlackLevel = save1 + (0x01f8 << 1);
        offsetWhiteLevels = save1 + (0x01fc << 1);
      }
      else if (imCanon.ColorDataSubVer == 11)
      {
        offsetChannelBlackLevel = save1 + (0x02d8 << 1);
        offsetWhiteLevels = save1 + (0x02dc << 1);
      }
      break;

    case 1560: // 5DS, 5DS R; ColorDataSubVer: 12
    case 1592: // 5D Mark IV, 80D, -1D X Mark II; ColorDataSubVer: 13
    case 1353: // 1300D, 1500D, 3000D; ColorDataSubVer: 14
    case 1602: // 200D, 6D Mark II, 77D, 800D; ColorDataSubVer: 15
      imCanon.ColorDataVer = 8;
      AsShot_Auto_MeasuredWB(0x003f);
      sRAW_WB(0x0080);
      fseek(ifp, save1 + (0x0085 << 1), SEEK_SET);
      Canon_WBpresets(2, 12);
      fseek(ifp, save1 + (0x0107 << 1), SEEK_SET);
      Canon_WBCTpresets(0); // BCAT

      if (imCanon.ColorDataSubVer == 14) // 1300D, 1500D, 3000D
      {
        offsetChannelBlackLevel = save1 + (0x022c << 1);
        offsetWhiteLevels = save1 + (0x0230 << 1);
      }
      else
      {
        offsetChannelBlackLevel = save1 + (0x030a << 1);
        offsetWhiteLevels = save1 + (0x030e << 1);
      }
      break;

    case 1820: // M50; ColorDataSubVer: 16
    case 1824: // R; ColorDataSubVer: 17
    case 1816: // RP, 250D, SX70 HS; ColorDataSubVer: 18
               // M6 Mark II, M200, 90D, G5 X Mark II, G7 X Mark III, 850D; ColorDataSubVer: 19
      imCanon.ColorDataVer = 9;
      AsShot_Auto_MeasuredWB(0x0047);
      CR3_ColorData(0x0047);
      break;

    case 1770: // R5 CRM
    case 2024: // -1D X Mark III; ColorDataSubVer: 32
    case 3656: // R5, R6; ColorDataSubVer: 33 
      imCanon.ColorDataVer = 10;
      AsShot_Auto_MeasuredWB(0x0055);
      CR3_ColorData(0x0055);
      break;

    case 3973: // R3; ColorDataSubVer: 34
    case 3778: // R7, R10; ColorDataSubVer: 48
      imCanon.ColorDataVer = 11;
      AsShot_Auto_MeasuredWB(0x0069);

      fseek(ifp, save1 + ((0x0069+0x0064) << 1), SEEK_SET);
      Canon_WBpresets(2, 12);
      fseek(ifp, save1 + ((0x0069+0x00c3) << 1), SEEK_SET);
      Canon_WBCTpresets(0);
      offsetChannelBlackLevel2 = save1 + ((0x0069+0x0102) << 1);
      offsetChannelBlackLevel  = save1 + ((0x0069+0x0213) << 1);
      offsetWhiteLevels        = save1 + ((0x0069+0x0217) << 1);
      break;

   default:
      imCanon.ColorDataSubVer = get2();
      break;
    }

    if (offsetChannelBlackLevel)
    {
      fseek(ifp, offsetChannelBlackLevel, SEEK_SET);
      FORC4
        bls += (imCanon.ChannelBlackLevel[RGGB_2_RGBG(c)] = get2());
      imCanon.AverageBlackLevel = bls / 4;
    }
    if (offsetWhiteLevels)
    {
      if ((offsetWhiteLevels - offsetChannelBlackLevel) != 8L)
        fseek(ifp, offsetWhiteLevels, SEEK_SET);
      imCanon.NormalWhiteLevel = get2();
      imCanon.SpecularWhiteLevel = get2();
      FORC4
        imgdata.color.linear_max[c] = imCanon.SpecularWhiteLevel;
    }

    if(!imCanon.AverageBlackLevel && offsetChannelBlackLevel2)
    {
        fseek(ifp, offsetChannelBlackLevel2, SEEK_SET);
        FORC4
            bls += (imCanon.ChannelBlackLevel[RGGB_2_RGBG(c)] = get2());
        imCanon.AverageBlackLevel = bls / 4;
    }
    fseek(ifp, save1, SEEK_SET);

  } else if (tag == 0x4013) {
    get4();
    imCanon.AFMicroAdjMode = get4();
    float a = float(get4());
    float b = float(get4());
    if (fabsf(b) > 0.001f)
      imCanon.AFMicroAdjValue = a / b;

  } else if (tag == 0x4018) {
    fseek(ifp, 8, SEEK_CUR);
    imCanon.AutoLightingOptimizer = get4();
    if ((imCanon.AutoLightingOptimizer > 3) ||
        (imCanon.AutoLightingOptimizer < 0))
      imCanon.AutoLightingOptimizer = 3;
    imCanon.HighlightTonePriority = get4();
    if ((imCanon.HighlightTonePriority > 5) ||
        (imCanon.HighlightTonePriority < 0))
      imCanon.HighlightTonePriority = 0;
    if (imCanon.HighlightTonePriority) {
      imCommon.ExposureCalibrationShift -= float(imCanon.HighlightTonePriority);
    }

  } else if ((tag == 0x4021) && (dng_writer == nonDNG) &&
             (imCanon.multishot[0] = get4()) &&
             (imCanon.multishot[1] = get4())) {
    if (len >= 4) {
      imCanon.multishot[2] = get4();
      imCanon.multishot[3] = get4();
    }
    FORC4 cam_mul[c] = 1024;
  } else if (tag == 0x4026) {
    fseek(ifp, 44, SEEK_CUR);
    imCanon.CanonLog = get4();
  }
#undef CR3_ColorData
#undef sRAW_WB
#undef AsShot_Auto_MeasuredWB
}
