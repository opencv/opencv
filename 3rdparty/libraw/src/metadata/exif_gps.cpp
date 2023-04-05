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

#include "../../internal/dcraw_defs.h"
#include "../../internal/libraw_cameraids.h"

void LibRaw::parse_exif_interop(int base)
{
	unsigned entries, tag, type, len, save;
	char value[4] = { 0,0,0,0 };
	entries = get2();
	INT64 fsize = ifp->size();
	while (entries--)
	{
		tiff_get(base, &tag, &type, &len, &save);

		INT64 savepos = ftell(ifp);
		if (len > 8 && savepos + len > fsize * 2)
		{
			fseek(ifp, save, SEEK_SET); // Recover tiff-read position!!
			continue;
		}
        if (callbacks.exif_cb)
        {
            callbacks.exif_cb(callbacks.exifparser_data, tag | 0x40000, type, len, order, ifp, base);
            fseek(ifp, savepos, SEEK_SET);
        }

		switch (tag)
		{
		case 0x0001: // InteropIndex
			fread(value, 1, MIN(4, len), ifp);
			if (strncmp(value, "R98", 3) == 0 &&
				// Canon bug, when [Canon].ColorSpace = AdobeRGB,
				// but [ExifIFD].ColorSpace = Uncalibrated and
				// [InteropIFD].InteropIndex = "R98"
				imgdata.color.ExifColorSpace == LIBRAW_COLORSPACE_Unknown)
				imgdata.color.ExifColorSpace = LIBRAW_COLORSPACE_sRGB;
			else if (strncmp(value, "R03", 3) == 0)
				imgdata.color.ExifColorSpace = LIBRAW_COLORSPACE_AdobeRGB;
			break;
		}
		fseek(ifp, save, SEEK_SET);
	}
}

void LibRaw::parse_exif(int base)
{
  unsigned entries, tag, type, len, save, c;
  double expo, ape;

  unsigned kodak = !strncmp(make, "EASTMAN", 7) && tiff_nifds < 3;

  if (!libraw_internal_data.unpacker_data.exif_subdir_offset)
  {
    libraw_internal_data.unpacker_data.exif_offset = base;
    libraw_internal_data.unpacker_data.exif_subdir_offset = ftell(ifp);
  }

  entries = get2();
  if (!strncmp(make, "Hasselblad", 10) && (tiff_nifds > 3) && (entries > 512))
    return;
  INT64 fsize = ifp->size();
  while (entries--)
  {
    tiff_get(base, &tag, &type, &len, &save);

    INT64 savepos = ftell(ifp);
    if (len > 8 && savepos + len > fsize * 2)
    {
      fseek(ifp, save, SEEK_SET); // Recover tiff-read position!!
      continue;
    }
    if (callbacks.exif_cb)
    {
      callbacks.exif_cb(callbacks.exifparser_data, tag, type, len, order, ifp,
                        base);
      fseek(ifp, savepos, SEEK_SET);
    }

    switch (tag)
    {
	case 0xA005: // Interoperability IFD
		fseek(ifp, get4() + base, SEEK_SET);
		parse_exif_interop(base);
		break;
	case 0xA001: // ExifIFD.ColorSpace
		c = get2();
		if (c == 1 && imgdata.color.ExifColorSpace == LIBRAW_COLORSPACE_Unknown)
			imgdata.color.ExifColorSpace = LIBRAW_COLORSPACE_sRGB;
		else if (c == 2)
			imgdata.color.ExifColorSpace = LIBRAW_COLORSPACE_AdobeRGB;
		break;
    case 0x9400:
      imCommon.exifAmbientTemperature = getreal(type);
      if ((imCommon.CameraTemperature > -273.15f) &&
          ((OlyID == OlyID_TG_5) ||
           (OlyID == OlyID_TG_6))
      )
        imCommon.CameraTemperature += imCommon.exifAmbientTemperature;
      break;
    case 0x9401:
      imCommon.exifHumidity = getreal(type);
      break;
    case 0x9402:
      imCommon.exifPressure = getreal(type);
      break;
    case 0x9403:
      imCommon.exifWaterDepth = getreal(type);
      break;
    case 0x9404:
      imCommon.exifAcceleration = getreal(type);
      break;
    case 0x9405:
      imCommon.exifCameraElevationAngle = getreal(type);
      break;

    case 0xa405: // FocalLengthIn35mmFormat
      imgdata.lens.FocalLengthIn35mmFormat = get2();
      break;
    case 0xa431: // BodySerialNumber
      stmread(imgdata.shootinginfo.BodySerial, len, ifp);
      break;
    case 0xa432: // LensInfo, 42034dec, Lens Specification per EXIF standard
      imgdata.lens.MinFocal = getreal(type);
      imgdata.lens.MaxFocal = getreal(type);
      imgdata.lens.MaxAp4MinFocal = getreal(type);
      imgdata.lens.MaxAp4MaxFocal = getreal(type);
      break;
    case 0xa435: // LensSerialNumber
      stmread(imgdata.lens.LensSerial, len, ifp);
      if (!strncmp(imgdata.lens.LensSerial, "----", 4))
        imgdata.lens.LensSerial[0] = '\0';
      break;
    case 0xa420: /* 42016, ImageUniqueID */
      stmread(imgdata.color.ImageUniqueID, len, ifp);
      break;
    case 0xc65d: /* 50781, RawDataUniqueID */
      imgdata.color.RawDataUniqueID[16] = 0;
      fread(imgdata.color.RawDataUniqueID, 1, 16, ifp);
      break;
    case 0xc630: // DNG LensInfo, Lens Specification per EXIF standard
      imgdata.lens.dng.MinFocal = getreal(type);
      imgdata.lens.dng.MaxFocal = getreal(type);
      imgdata.lens.dng.MaxAp4MinFocal = getreal(type);
      imgdata.lens.dng.MaxAp4MaxFocal = getreal(type);
      break;
    case 0xc68b: /* 50827, OriginalRawFileName */
      stmread(imgdata.color.OriginalRawFileName, len, ifp);
      break;
    case 0xa433: // LensMake
      stmread(imgdata.lens.LensMake, len, ifp);
      break;
    case 0xa434: // LensModel
      stmread(imgdata.lens.Lens, len, ifp);
      if (!strncmp(imgdata.lens.Lens, "----", 4))
        imgdata.lens.Lens[0] = '\0';
      break;
    case 0x9205:
      imgdata.lens.EXIF_MaxAp = libraw_powf64l(2.0f, (getreal(type) / 2.0f));
      break;
    case 0x829a: // 33434
      shutter = getreal(type);
      if (tiff_nifds > 0 && tiff_nifds <= LIBRAW_IFD_MAXCOUNT)
          tiff_ifd[tiff_nifds - 1].t_shutter = shutter;
      break;
    case 0x829d: // 33437, FNumber
      aperture = getreal(type);
      break;
    case 0x8827: // 34855
      iso_speed = get2();
      break;
    case 0x8831: // 34865
      if (iso_speed == 0xffff && !strncasecmp(make, "FUJI", 4))
        iso_speed = getreal(type);
      break;
    case 0x8832: // 34866
      if (iso_speed == 0xffff &&
          (!strncasecmp(make, "SONY", 4) || !strncasecmp(make, "CANON", 5)))
        iso_speed = getreal(type);
      break;
    case 0x9003: // 36867
    case 0x9004: // 36868
      get_timestamp(0);
      break;
    case 0x9201: // 37377
       if ((expo = -getreal(type)) < 128 && shutter == 0.)
       {
            shutter = libraw_powf64l(2.0, expo);
            if (tiff_nifds > 0 && tiff_nifds <= LIBRAW_IFD_MAXCOUNT)
              tiff_ifd[tiff_nifds - 1].t_shutter = shutter;
       }
      break;
    case 0x9202: // 37378 ApertureValue
      if ((fabs(ape = getreal(type)) < 256.0) && (!aperture))
        aperture = libraw_powf64l(2.0, ape / 2);
      break;
    case 0x9209: // 37385
      flash_used = getreal(type);
      break;
    case 0x920a: // 37386
      focal_len = getreal(type);
      break;
    case 0x927c: // 37500
#ifndef USE_6BY9RPI
      if (((make[0] == '\0') && !strncmp(model, "ov5647", 6)) ||
          (!strncmp(make, "RaspberryPi", 11) &&
           (!strncmp(model, "RP_OV5647", 9) ||
            !strncmp(model, "RP_imx219", 9))))
#else
      if (((make[0] == '\0') && !strncmp(model, "ov5647", 6)) ||
          (!strncmp(make, "RaspberryPi", 11) &&
              (!strncmp(model, "RP_", 3) || !strncmp(model,"imx477",6))))
#endif
      {
        char mn_text[512];
        char *pos;
        char ccms[512];
        ushort l;
        float num;

        fgets(mn_text, MIN(len, 511), ifp);
        mn_text[511] = 0;

        pos = strstr(mn_text, "ev=");
        if (pos)
          imCommon.ExposureCalibrationShift = atof(pos + 3);

        pos = strstr(mn_text, "gain_r=");
        if (pos)
          cam_mul[0] = atof(pos + 7);
        pos = strstr(mn_text, "gain_b=");
        if (pos)
          cam_mul[2] = atof(pos + 7);
        if ((cam_mul[0] > 0.001f) && (cam_mul[2] > 0.001f))
          cam_mul[1] = cam_mul[3] = 1.0f;
        else
          cam_mul[0] = cam_mul[2] = 0.0f;

        pos = strstr(mn_text, "ccm=");
        if (pos)
        {
          pos += 4;
          char *pos2 = strstr(pos, " ");
          if (pos2)
          {
            l = pos2 - pos;
            memcpy(ccms, pos, l);
            ccms[l] = '\0';
#ifdef LIBRAW_WIN32_CALLS
            // Win32 strtok is already thread-safe
            pos = strtok(ccms, ",");
#else
            char *last = 0;
            pos = strtok_r(ccms, ",", &last);
#endif
            if (pos)
            {
              for (l = 0; l < 3; l++) // skip last row
              {
                num = 0.0;
                for (c = 0; c < 3; c++)
                {
                  cmatrix[l][c] = (float)atoi(pos);
                  num += cmatrix[c][l];
#ifdef LIBRAW_WIN32_CALLS
                  pos = strtok(NULL, ",");
#else
                  pos = strtok_r(NULL, ",", &last);
#endif
                  if (!pos)
                    goto end; // broken
                }
                if (num > 0.01)
                    FORC3 cmatrix[l][c] = cmatrix[l][c] / num;
              }
            }
          }
        }
      end:;
      }
      else if (!strncmp(make, "SONY", 4) &&
               (!strncmp(model, "DSC-V3", 6) || !strncmp(model, "DSC-F828", 8)))
      {
        parseSonySRF(len);
        break;
      }
      else if ((len == 1) && !strncmp(make, "NIKON", 5))
      {
        c = get4();
        if (c)
          fseek(ifp, c, SEEK_SET);
        is_NikonTransfer = 1;
      }
      parse_makernote(base, 0);
      break;
    case 0xa002: // 40962
      if (kodak)
        raw_width = get4();
      break;
    case 0xa003: // 40963
      if (kodak)
        raw_height = get4();
      break;
    case 0xa302: // 41730
      if (get4() == 0x20002)
        for (exif_cfa = c = 0; c < 8; c += 2)
          exif_cfa |= fgetc(ifp) * 0x01010101U << c;
    }
    fseek(ifp, save, SEEK_SET);
  }
}

void LibRaw::parse_gps_libraw(int base)
{
  unsigned entries, tag, type, len, save, c;

  entries = get2();
  if (entries > 40)
    return;
  if (entries > 0)
    imgdata.other.parsed_gps.gpsparsed = 1;
  INT64 fsize = ifp->size();
  while (entries--)
  {
    tiff_get(base, &tag, &type, &len, &save);
    if (len > 1024)
    {
      fseek(ifp, save, SEEK_SET); // Recover tiff-read position!!
      continue;                   // no GPS tags are 1k or larger
    }
    INT64 savepos = ftell(ifp);
    if (len > 8 && savepos + len > fsize * 2)
    {
        fseek(ifp, save, SEEK_SET); // Recover tiff-read position!!
        continue;
    }

    if (callbacks.exif_cb)
    {
        callbacks.exif_cb(callbacks.exifparser_data, tag | 0x50000, type, len, order, ifp, base);
        fseek(ifp, savepos, SEEK_SET);
    }

    switch (tag)
    {
    case 0x0001:
      imgdata.other.parsed_gps.latref = getc(ifp);
      break;
    case 0x0003:
      imgdata.other.parsed_gps.longref = getc(ifp);
      break;
    case 0x0005:
      imgdata.other.parsed_gps.altref = getc(ifp);
      break;
    case 0x0002:
      if (len == 3)
        FORC(3) imgdata.other.parsed_gps.latitude[c] = getreal(type);
      break;
    case 0x0004:
      if (len == 3)
        FORC(3) imgdata.other.parsed_gps.longitude[c] = getreal(type);
      break;
    case 0x0007:
      if (len == 3)
        FORC(3) imgdata.other.parsed_gps.gpstimestamp[c] = getreal(type);
      break;
    case 0x0006:
      imgdata.other.parsed_gps.altitude = getreal(type);
      break;
    case 0x0009:
      imgdata.other.parsed_gps.gpsstatus = getc(ifp);
      break;
    }
    fseek(ifp, save, SEEK_SET);
  }
}

void LibRaw::parse_gps(int base)
{
  unsigned entries, tag, type, len, save, c;

  entries = get2();
  if (entries > 40)
    return;
  while (entries--)
  {
    tiff_get(base, &tag, &type, &len, &save);
    if (len > 1024)
    {
      fseek(ifp, save, SEEK_SET); // Recover tiff-read position!!
      continue;                   // no GPS tags are 1k or larger
    }
    switch (tag)
    {
    case 0x0001:
    case 0x0003:
    case 0x0005:
      gpsdata[29 + tag / 2] = getc(ifp);
      break;
    case 0x0002:
    case 0x0004:
    case 0x0007:
      FORC(6) gpsdata[tag / 3 * 6 + c] = get4();
      break;
    case 0x0006:
      FORC(2) gpsdata[18 + c] = get4();
      break;
    case 0x0012: // 18
    case 0x001d: // 29
      fgets((char *)(gpsdata + 14 + tag / 3), MIN(len, 12), ifp);
    }
    fseek(ifp, save, SEEK_SET);
  }
}
