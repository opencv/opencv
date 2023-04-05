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

void LibRaw::setOlympusBodyFeatures(unsigned long long id)
{
  ilm.CamID = id;

  if ((id == OlyID_E_1)   ||
      (id == OlyID_E_300) ||
      ((id & 0x00ffff0000ULL) == 0x0030300000ULL))
  {
    ilm.CameraFormat = LIBRAW_FORMAT_FT;

    if ((id == OlyID_E_1)   ||
        (id == OlyID_E_300) ||
        ((id >= OlyID_E_330) && (id <= OlyID_E_520)) ||
        (id == OlyID_E_620) ||
        (id == OlyID_E_450) ||
        (id == OlyID_E_600) ||
        (id == OlyID_E_5))
    {
      ilm.CameraMount = LIBRAW_MOUNT_FT;
    }
    else
    {
      ilm.CameraMount = LIBRAW_MOUNT_mFT;
    }
  }
  else
  {
    ilm.LensMount = ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
  }
  return;
}

void LibRaw::getOlympus_CameraType2()
{

  if (OlyID != 0x0ULL)
    return;

  int i = 0;
  fread(imOly.CameraType2, 6, 1, ifp);
  imOly.CameraType2[5] = 0;
  while ((i < 6) && imOly.CameraType2[i])
  {
    OlyID = OlyID << 8 | imOly.CameraType2[i];
    if (i < 5 && isspace(imOly.CameraType2[i + 1])) {
      imOly.CameraType2[i + 1] = '\0';
      break;
    }
    i++;
  }
  if (OlyID == OlyID_NORMA) {
    if (strcmp(model, "SP510UZ")) OlyID = OlyID_SP_510UZ;
    else OlyID = 0x0ULL;
  }
  unique_id = OlyID;
  setOlympusBodyFeatures(OlyID);
  return;
}

void LibRaw::getOlympus_SensorTemperature(unsigned len)
{
  if (OlyID != 0x0ULL)
  {
    short temp = get2();
    if ((OlyID == OlyID_E_1)  ||
        (OlyID == OlyID_E_M5) ||
        (len != 1))
      imCommon.SensorTemperature = (float)temp;
    else if ((temp != -32768) && (temp != 0))
    {
      if (temp > 199)
        imCommon.SensorTemperature = 86.474958f - 0.120228f * (float)temp;
      else
        imCommon.SensorTemperature = (float)temp;
    }
  }
  return;
}

void LibRaw::parseOlympus_Equipment(unsigned tag, unsigned /*type */, unsigned len,
                                    unsigned dng_writer)
{
  // uptag 2010

  switch (tag)
  {
  case 0x0100:
    getOlympus_CameraType2();
    break;
  case 0x0101:
    if ((!imgdata.shootinginfo.BodySerial[0]) && (dng_writer == nonDNG))
      stmread(imgdata.shootinginfo.BodySerial, len, ifp);
    break;
  case 0x0102:
    stmread(imgdata.shootinginfo.InternalBodySerial, len, ifp);
    break;
  case 0x0201:
  {
    unsigned char bits[4];
    fread(bits, 1, 4, ifp);
    ilm.LensID = (unsigned long long)bits[0] << 16 |
                 (unsigned long long)bits[2] << 8 | (unsigned long long)bits[3];
    ilm.LensMount = LIBRAW_MOUNT_FT;
    ilm.LensFormat = LIBRAW_FORMAT_FT;
    if (((ilm.LensID < 0x20000) || (ilm.LensID > 0x4ffff)) &&
        (ilm.LensID & 0x10))
      ilm.LensMount = LIBRAW_MOUNT_mFT;
  }
    break;
  case 0x0202:
    if ((!imgdata.lens.LensSerial[0]))
      stmread(imgdata.lens.LensSerial, len, ifp);
    break;
  case 0x0203:
    stmread(ilm.Lens, len, ifp);
    break;
  case 0x0205:
    ilm.MaxAp4MinFocal = libraw_powf64l(sqrt(2.0f), get2() / 256.0f);
    break;
  case 0x0206:
    ilm.MaxAp4MaxFocal = libraw_powf64l(sqrt(2.0f), get2() / 256.0f);
    break;
  case 0x0207:
    ilm.MinFocal = (float)get2();
    break;
  case 0x0208:
    ilm.MaxFocal = (float)get2();
    if (ilm.MaxFocal > 1000.0f)
      ilm.MaxFocal = ilm.MinFocal;
    break;
  case 0x020a:
    ilm.MaxAp4CurFocal = libraw_powf64l(sqrt(2.0f), get2() / 256.0f);
    break;
  case 0x0301:
    ilm.TeleconverterID = fgetc(ifp) << 8;
    fgetc(ifp);
    ilm.TeleconverterID = ilm.TeleconverterID | fgetc(ifp);
    break;
  case 0x0303:
    stmread(ilm.Teleconverter, len, ifp);
    if (!strlen(ilm.Teleconverter) && strchr(ilm.Lens, '+')) {
      if (strstr(ilm.Lens, "MC-20"))
        strcpy(ilm.Teleconverter, "MC-20");
      else if (strstr(ilm.Lens, "MC-14"))
        strcpy(ilm.Teleconverter, "MC-14");
      else if (strstr(ilm.Lens, "EC-20"))
        strcpy(ilm.Teleconverter, "EC-20");
      else if (strstr(ilm.Lens, "EC-14"))
        strcpy(ilm.Teleconverter, "EC-14");    }
    break;
  case 0x0403:
    stmread(ilm.Attachment, len, ifp);
    break;
  }

  return;
}
void LibRaw::parseOlympus_CameraSettings(int base, unsigned tag, unsigned type,
                                         unsigned len, unsigned dng_writer)
{
  // uptag 0x2020

  int c;
  switch (tag)
  {
  case 0x0101:
    if (dng_writer == nonDNG)
    {
      thumb_offset = get4() + base;
    }
    break;
  case 0x0102:
    if (dng_writer == nonDNG)
    {
      thumb_length = get4();
    }
    break;
  case 0x0200:
    imgdata.shootinginfo.ExposureMode = get2();
    break;
  case 0x0202:
    imgdata.shootinginfo.MeteringMode = get2();
    break;
  case 0x0301:
    imgdata.shootinginfo.FocusMode = imOly.FocusMode[0] = get2();
    if (len == 2)
    {
      imOly.FocusMode[1] = get2();
    }
    break;
  case 0x0304:
    for (c = 0; c < 64; c++)
    {
      imOly.AFAreas[c] = get4();
    }
    break;
  case 0x0305:
    for (c = 0; c < 5; c++)
    {
      imOly.AFPointSelected[c] = getreal(type);
    }
    break;
  case 0x0306:
    imOly.AFFineTune = fgetc(ifp);
    break;
  case 0x0307:
    FORC3 imOly.AFFineTuneAdj[c] = get2();
    break;
  case 0x0401:
    imCommon.FlashEC = getreal(type);
    break;
  case 0x0507:
    imOly.ColorSpace = get2();
    switch (imOly.ColorSpace) {
    case 0:
      imCommon.ColorSpace = LIBRAW_COLORSPACE_sRGB;
      break;
    case 1:
      imCommon.ColorSpace = LIBRAW_COLORSPACE_AdobeRGB;
      break;
    case 2:
      imCommon.ColorSpace = LIBRAW_COLORSPACE_ProPhotoRGB;
      break;
    default:
      imCommon.ColorSpace = LIBRAW_COLORSPACE_Unknown;
      break;
    }
    break;
  case 0x0600:
    imgdata.shootinginfo.DriveMode = imOly.DriveMode[0] = get2();
    for (c = 1; c < (int)len && c < 5; c++)
    {
      imOly.DriveMode[c] = get2();
    }
    break;
  case 0x0601:
  	imOly.Panorama_mode = get2();
  	imOly.Panorama_frameNum = get2();
  	break;
  case 0x0604:
    imgdata.shootinginfo.ImageStabilization = get4();
    break;
  case 0x0804:
    imOly.StackedImage[0] = get4();
    imOly.StackedImage[1] = get4();
    if (imOly.StackedImage[0] == 3) {
      imOly.isLiveND = 1;
      imOly.LiveNDfactor = imOly.StackedImage[1];
    } else {
      imOly.isLiveND = 0;
    }
    break;
  }

  return;
}

void LibRaw::parseOlympus_ImageProcessing(unsigned tag, unsigned type,
                                          unsigned len, unsigned dng_writer)
{
  // uptag 0x2040

  int i, c, wb[4], nWB, tWB, wbG;
  ushort CT;
  short sorder;

  if ((tag == 0x0100) && (dng_writer == nonDNG))
  {
    cam_mul[0] = get2() / 256.0;
    cam_mul[2] = get2() / 256.0;
  }
  else if ((tag == 0x0101) && (len == 2) &&
           ((OlyID == OlyID_E_410) || (OlyID == OlyID_E_510)))
  {
    for (i = 0; i < 64; i++)
    {
      icWBCCTC[i][2] = icWBCCTC[i][4] = icWBC[i][1] = icWBC[i][3] = 0x100;
    }
    for (i = 64; i < 256; i++)
    {
      icWBC[i][1] = icWBC[i][3] = 0x100;
    }
  }
  else if ((tag > 0x0101) && (tag <= 0x0111))
  {
    nWB = tag - 0x0101;
    tWB = Oly_wb_list2[nWB << 1];
    CT = Oly_wb_list2[(nWB << 1) | 1];
    wb[0] = get2();
    wb[2] = get2();
    if (tWB != 0x100)
    {
      icWBC[tWB][0] = wb[0];
      icWBC[tWB][2] = wb[2];
    }
    if (CT)
    {
      icWBCCTC[nWB - 1][0] = CT;
      icWBCCTC[nWB - 1][1] = wb[0];
      icWBCCTC[nWB - 1][3] = wb[2];
    }
    if (len == 4)
    {
      wb[1] = get2();
      wb[3] = get2();
      if (tWB != 0x100)
      {
        icWBC[tWB][1] = wb[1];
        icWBC[tWB][3] = wb[3];
      }
      if (CT)
      {
        icWBCCTC[nWB - 1][2] = wb[1];
        icWBCCTC[nWB - 1][4] = wb[3];
      }
    }
  }
  else if ((tag >= 0x0112) && (tag <= 0x011e))
  {
    nWB = tag - 0x0112;
    wbG = get2();
    tWB = Oly_wb_list2[nWB << 1];
    if (nWB)
      icWBCCTC[nWB - 1][2] = icWBCCTC[nWB - 1][4] = wbG;
    if (tWB != 0x100)
      icWBC[tWB][1] = icWBC[tWB][3] = wbG;
  }
  else if (tag == 0x011f)
  {
    wbG = get2();
    if (icWBC[LIBRAW_WBI_Flash][0])
      icWBC[LIBRAW_WBI_Flash][1] =
          icWBC[LIBRAW_WBI_Flash][3] = wbG;
    FORC4 if (icWBC[LIBRAW_WBI_Custom1 + c][0])
        icWBC[LIBRAW_WBI_Custom1 + c][1] =
        icWBC[LIBRAW_WBI_Custom1 + c][3] = wbG;
  }
  else if (tag == 0x0121)
  {
    icWBC[LIBRAW_WBI_Flash][0] = get2();
    icWBC[LIBRAW_WBI_Flash][2] = get2();
    if (len == 4)
    {
      icWBC[LIBRAW_WBI_Flash][1] = get2();
      icWBC[LIBRAW_WBI_Flash][3] = get2();
    }
  }
  else if ((tag == 0x0200) && (dng_writer == nonDNG) &&
           strcmp(software, "v757-71"))
  {
    for (i = 0; i < 3; i++)
    {
      if (!imOly.ColorSpace)
      {
        FORC3 cmatrix[i][c] = ((short)get2()) / 256.0;
      }
      else
      {
        FORC3 imgdata.color.ccm[i][c] = ((short)get2()) / 256.0;
      }
    }
  }
  else if ((tag == 0x0600) && (dng_writer == nonDNG))
  {
    FORC4 cblack[RGGB_2_RGBG(c)] = get2();
  }
  else if ((tag == 0x0611) && (dng_writer == nonDNG))
  {
     imOly.ValidBits = get2();
  }
  else if ((tag == 0x0612) && (dng_writer == nonDNG))
  {
    imgdata.sizes.raw_inset_crops[0].cleft = get2();
  }
  else if ((tag == 0x0613) && (dng_writer == nonDNG))
  {
    imgdata.sizes.raw_inset_crops[0].ctop = get2();
  }
  else if ((tag == 0x0614) && (dng_writer == nonDNG))
  {
    imgdata.sizes.raw_inset_crops[0].cwidth = get2();
  }
  else if ((tag == 0x0615) && (dng_writer == nonDNG))
  {
    imgdata.sizes.raw_inset_crops[0].cheight = get2();
  }
  else if ((tag == 0x0805) && (len == 2))
  {
    imOly.SensorCalibration[0] = getreal(type);
    imOly.SensorCalibration[1] = getreal(type);
    if ((dng_writer == nonDNG) && (OlyID != OlyID_XZ_1))
      FORC4 imgdata.color.linear_max[c] = imOly.SensorCalibration[0];
  }
  else if (tag == 0x1112)
  {
    sorder = order;
    order = 0x4d4d;
    c = get2();
    order = sorder;
    switch (c) {
    case 0x0101:
    case 0x0901:
    case 0x0909:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_4to3;
      break;
    case 0x0104:
    case 0x0401:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_1to1;
      break;
    case 0x0201:
    case 0x0202:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_3to2;
      break;
    case 0x0301:
    case 0x0303:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_16to9;
      break;
    case 0x0404:
//      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_6to6;
        imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_1to1;
      break;
    case 0x0505:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_5to4;
      break;
    case 0x0606:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_7to6;
      break;
    case 0x0707:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_6to5;
      break;
    case 0x0808:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_7to5;
      break;
    default:
      imgdata.sizes.raw_aspect = LIBRAW_IMAGE_ASPECT_OTHER;
      break;
    }
  }
  else if (tag == 0x1113)
  {
    imOly.AspectFrame[0] = get2();
    imOly.AspectFrame[1] = get2();
    imOly.AspectFrame[2] = get2();
    imOly.AspectFrame[3] = get2();
  }
  else if (tag == 0x1306)
  {
    c = get2();
    if ((c != 0) && (c != 100))
    {
      if (c < 61)
        imCommon.CameraTemperature = (float)c;
      else
        imCommon.CameraTemperature = (float)(c - 32) / 1.8f;
      if ((imCommon.exifAmbientTemperature > -273.15f) &&
          ((OlyID == OlyID_TG_5) ||
           (OlyID == OlyID_TG_6))
      )
        imCommon.CameraTemperature += imCommon.exifAmbientTemperature;
    }
  }

  return;
}

void LibRaw::parseOlympus_RawInfo(unsigned tag, unsigned /*type */, unsigned len,
                                  unsigned dng_writer)
{
  // uptag 0x3000

  int wb_ind, c, i;

  if ((tag == 0x0110) && strcmp(software, "v757-71"))
  {
    icWBC[LIBRAW_WBI_Auto][0] = get2();
    icWBC[LIBRAW_WBI_Auto][2] = get2();
    if (len == 2)
    {
      for (i = 0; i < 256; i++)
        icWBC[i][1] = icWBC[i][3] = 0x100;
    }
  }
  else if ((((tag >= 0x0120) && (tag <= 0x0124)) ||
            ((tag >= 0x0130) && (tag <= 0x0133))) &&
           strcmp(software, "v757-71"))
  {
    if (tag <= 0x0124)
      wb_ind = tag - 0x0120;
    else
      wb_ind = tag - 0x0130 + 5;

    icWBC[Oly_wb_list1[wb_ind]][0] = get2();
    icWBC[Oly_wb_list1[wb_ind]][2] = get2();
  }
  else if ((tag == 0x0200) && (dng_writer == nonDNG))
  {
    for (i = 0; i < 3; i++)
    {
      if (!imOly.ColorSpace)
      {
        FORC3 cmatrix[i][c] = ((short)get2()) / 256.0;
      }
      else
      {
        FORC3 imgdata.color.ccm[i][c] = ((short)get2()) / 256.0;
      }
    }
  }
  else if ((tag == 0x0600) && (dng_writer == nonDNG))
  {
    FORC4 cblack[RGGB_2_RGBG(c)] = get2();
  }
  else if ((tag == 0x0612) && (dng_writer == nonDNG))
  {
    imgdata.sizes.raw_inset_crops[0].cleft = get2();
  }
  else if ((tag == 0x0613) && (dng_writer == nonDNG))
  {
    imgdata.sizes.raw_inset_crops[0].ctop = get2();
  }
  else if ((tag == 0x0614) && (dng_writer == nonDNG))
  {
    imgdata.sizes.raw_inset_crops[0].cwidth = get2();
  }
  else if ((tag == 0x0615) && (dng_writer == nonDNG))
  {
    imgdata.sizes.raw_inset_crops[0].cheight = get2();
  }
  return;
}


void LibRaw::parseOlympusMakernotes (int base, unsigned tag, unsigned type, unsigned len, unsigned dng_writer) {

  int c;
  unsigned a;
  if ((tag >= 0x20100000) && (tag <= 0x2010ffff)) {
        parseOlympus_Equipment((tag & 0x0000ffff), type, len, dng_writer);

  } else if ((tag >= 0x20200000) && (tag <= 0x2020ffff)) {
    parseOlympus_CameraSettings(base, (tag & 0x0000ffff), type, len, dng_writer);

  } else if ((tag >= 0x20400000) && (tag <= 0x2040ffff)) {
     parseOlympus_ImageProcessing((tag & 0x0000ffff), type, len, dng_writer);

  } else if ((tag >= 0x30000000) && (tag <= 0x3000ffff)) {
        parseOlympus_RawInfo((tag & 0x0000ffff), type, len, dng_writer);

  } else {
		switch (tag) {
			case 0x0200:
			  FORC3 if ((imOly.SpecialMode[c] = get4()) >= 0xff) imOly.SpecialMode[c] = 0xffffffff;
			  break;
			case 0x0207:
				getOlympus_CameraType2();
				break;
			case 0x0404:
			case 0x101a:
				if (!imgdata.shootinginfo.BodySerial[0] && (dng_writer == nonDNG))
					stmread(imgdata.shootinginfo.BodySerial, len, ifp);
				break;
			case 0x1002:
				ilm.CurAp = libraw_powf64l(2.0f, getreal(type) / 2);
				break;
			case 0x1007:
				imCommon.SensorTemperature = (float)get2();
				break;
			case 0x1008:
				imCommon.LensTemperature = (float)get2();
				break;
			case 0x100b:
				if (imOly.FocusMode[0] == 0xffff) {
					imgdata.shootinginfo.FocusMode = imOly.FocusMode[0] = get2();
					if (imgdata.shootinginfo.FocusMode == 1)
						imgdata.shootinginfo.FocusMode = imOly.FocusMode[0] = 10;
				}
				break;
      case 0x100d:
        if (imOly.ZoomStepCount == 0xffff) imOly.ZoomStepCount = get2();
        break;
      case 0x100e:
        if (imOly.FocusStepCount == 0xffff) imOly.FocusStepCount = get2();
        break;
			case 0x1011:
				if (strcmp(software, "v757-71") && (dng_writer == nonDNG)) {
					for (int i = 0; i < 3; i++) {
						if (!imOly.ColorSpace) {
							FORC3 cmatrix[i][c] = ((short)get2()) / 256.0;
						} else {
							FORC3 imgdata.color.ccm[i][c] = ((short)get2()) / 256.0;
						}
					}
				}
				break;
			case 0x1012:
			  if (dng_writer == nonDNG)
				  FORC4 cblack[RGGB_2_RGBG(c)] = get2();
				break;
			case 0x1017:
				if (dng_writer == nonDNG)
				  cam_mul[0] = get2() / 256.0;
				break;
			case 0x1018:
				if (dng_writer == nonDNG)
				  cam_mul[2] = get2() / 256.0;
				break;
			case 0x102c:
				if (dng_writer == nonDNG)
				  imOly.ValidBits = get2();
				break;
			case 0x1038:
				imOly.AFResult = get2();
				break;
      case 0x103b:
        if (imOly.FocusStepInfinity == 0xffff) imOly.FocusStepInfinity = get2();
        break;
      case 0x103c:
        if (imOly.FocusStepNear == 0xffff) imOly.FocusStepNear = get2();
        break;
			case 0x20300108:
			case 0x20310109:
				if (dng_writer == nonDNG) {
          imOly.ColorSpace = get2();
          switch (imOly.ColorSpace) {
          case 0:
            imCommon.ColorSpace = LIBRAW_COLORSPACE_sRGB;
            break;
          case 1:
            imCommon.ColorSpace = LIBRAW_COLORSPACE_AdobeRGB;
            break;
          case 2:
            imCommon.ColorSpace = LIBRAW_COLORSPACE_ProPhotoRGB;
            break;
          default:
            imCommon.ColorSpace = LIBRAW_COLORSPACE_Unknown;
            break;
          }
				}
			case 0x20500209:
				imOly.AutoFocus = get2();
				break;
			case 0x20500300:
			  imOly.ZoomStepCount = get2();
			  break;
			case 0x20500301:
			  imOly.FocusStepCount = get2();
			  break;
			case 0x20500303:
			  imOly.FocusStepInfinity = get2();
			  break;
			case 0x20500304:
			  imOly.FocusStepNear = get2();
			  break;
			case 0x20500305:
			  a = get4();
			  /*b = */ get4(); // b is not used, so removed
			  if (a >= 0x7f000000) imOly.FocusDistance = -1.0; // infinity
			  else imOly.FocusDistance = (double) a / 1000.0;  // convert to meters
			  break;
			case 0x20500308:
				imOly.AFPoint = get2();
				break;
			case 0x20501500:
				getOlympus_SensorTemperature(len);
				break;
		}
  }
}
