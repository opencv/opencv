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

int LibRaw::parse_tiff_ifd(int base)
{
  unsigned entries, tag, type, len, plen = 16, save, utmp;
  int ifd, use_cm = 0, cfa, i, j, c, ima_len = 0;
  char *cbuf, *cp;
  uchar cfa_pat[16], cfa_pc[] = {0, 1, 2, 3}, tab[256];
  double fm[3][4], cc[4][4], cm[4][3], cam_xyz[4][3], num;
  double ab[] = {1, 1, 1, 1}, asn[] = {0, 0, 0, 0}, xyz[] = {1, 1, 1};
  unsigned sony_curve[] = {0, 0, 0, 0, 0, 4095};
  unsigned *buf, sony_offset = 0, sony_length = 0, sony_key = 0;
  struct jhead jh;

  ushort *rafdata;

  if (tiff_nifds >= sizeof tiff_ifd / sizeof tiff_ifd[0])
    return 1;
  ifd = tiff_nifds++;
  for (j = 0; j < 4; j++)
    for (i = 0; i < 4; i++)
      cc[j][i] = i == j;

  if (libraw_internal_data.unpacker_data.ifd0_offset == -1LL)
    libraw_internal_data.unpacker_data.ifd0_offset = base;

  entries = get2();
  if (entries > 512)
    return 1;

  INT64 fsize = ifp->size();

  while (entries--)
  {
    tiff_get(base, &tag, &type, &len, &save);
    INT64 savepos = ftell(ifp);
    if (len > 8 && savepos + len > 2 * fsize)
    {
      fseek(ifp, save, SEEK_SET); // Recover tiff-read position!!
      continue;
    }
    if (callbacks.exif_cb)
    {
      callbacks.exif_cb(callbacks.exifparser_data,
                        tag | (is_pana_raw ? 0x30000 : ((ifd + 1) << 20)), type,
                        len, order, ifp, base);
      fseek(ifp, savepos, SEEK_SET);
    }

    if (!is_pana_raw)
    { /* processing of EXIF tags that collide w/ PanasonicRaw tags */
      switch (tag)
      {
      case 0x0001:
        if (len == 4)
          is_pana_raw = get4();
        break;
      case 0x000b: /* 11, Std. EXIF Software tag */
        fgets(software, 64, ifp);
        if (!strncmp(software, "Adobe", 5) || !strncmp(software, "dcraw", 5) ||
            !strncmp(software, "UFRaw", 5) || !strncmp(software, "Bibble", 6) ||
            !strcmp(software, "Digital Photo Professional"))
          is_raw = 0;
        break;
      case 0x001c: /*  28, safeguard, probably not needed */
      case 0x001d: /*  29, safeguard, probably not needed */
      case 0x001e: /*  30, safeguard, probably not needed */
        cblack[tag - 0x001c] = get2();
        cblack[3] = cblack[1];
        break;

      case 0x0111: /* 273, StripOffset */
        if (len > 1 && len < 16384)
        {
          off_t sav = ftell(ifp);
          tiff_ifd[ifd].strip_offsets = (int *)calloc(len, sizeof(int));
          tiff_ifd[ifd].strip_offsets_count = len;
          for (int ii = 0; ii < (int)len; ii++)
            tiff_ifd[ifd].strip_offsets[ii] = get4() + base;
          fseek(ifp, sav, SEEK_SET); // restore position
        }
        /* fallback */
      case 0x0201: /* 513, JpegIFOffset */
      case 0xf007: // 61447
        tiff_ifd[ifd].offset = get4() + base;
        if (!tiff_ifd[ifd].bps && tiff_ifd[ifd].offset > 0)
        {
          fseek(ifp, tiff_ifd[ifd].offset, SEEK_SET);
          if (ljpeg_start(&jh, 1))
          {
            if (!dng_version && !strcasecmp(make, "SONY") && tiff_ifd[ifd].phint == 32803 &&
                tiff_ifd[ifd].comp == 7) // Sony/lossless compressed IFD
            {
              tiff_ifd[ifd].comp = 6;
              tiff_ifd[ifd].bps = jh.bits;
              tiff_ifd[ifd].samples = 1;
            }
            else
            {
              tiff_ifd[ifd].comp = 6;
              tiff_ifd[ifd].bps = jh.bits;
              tiff_ifd[ifd].t_width = jh.wide;
              tiff_ifd[ifd].t_height = jh.high;
              tiff_ifd[ifd].samples = jh.clrs;
              if (!(jh.sraw || (jh.clrs & 1)))
                tiff_ifd[ifd].t_width *= jh.clrs;
              if ((tiff_ifd[ifd].t_width > 4 * tiff_ifd[ifd].t_height) & ~jh.clrs)
              {
                tiff_ifd[ifd].t_width /= 2;
                tiff_ifd[ifd].t_height *= 2;
              }
              i = order;
              parse_tiff(tiff_ifd[ifd].offset + 12);
              order = i;
            }
          }
        }
        break;
      }
    }
    else
    { /* processing Panasonic-specific "PanasonicRaw" tags */
      switch (tag)
      {
      case 0x0004: /*   4, SensorTopBorder */
        imgdata.sizes.raw_inset_crops[0].ctop = get2();
        break;
      case 0x000a: /*  10, BitsPerSample */
        pana_bpp = get2();
		pana_bpp = LIM(pana_bpp, 8, 16);
        break;
      case 0x000b: /*  11, Compression */
        imPana.Compression = get2();
        break;
      case 0x000e: /*  14, LinearityLimitRed */
      case 0x000f: /*  15, LinearityLimitGreen */
      case 0x0010: /*  16, LinearityLimitBlue */
        imgdata.color.linear_max[tag - 14] = get2();
        if (imgdata.color.linear_max[tag - 14] == 16383)
            imgdata.color.linear_max[tag - 14] -= 64;
        if (imgdata.color.linear_max[tag - 14] == 4095)
          imgdata.color.linear_max[tag - 14] -= 16;
        if (tag == 0x000f) // 15, LinearityLimitGreen
          imgdata.color.linear_max[3] = imgdata.color.linear_max[1];
        break;
      case 0x0013: /*  19, WBInfo */
        if ((i = get2()) > 0x100)
          break;
        for (c = 0; c < i; c++)
        {
          if ((j = get2()) < 0x100)
          {
            icWBC[j][0] = get2();
            icWBC[j][2] = get2();
            icWBC[j][1] = icWBC[j][3] =
                0x100;
          }
          else // light source out of EXIF numbers range
            get4();
        }
        break;
      case 0x0018: /* 24, HighISOMultiplierRed */
      case 0x0019: /* 25, HighISOMultiplierGreen */
      case 0x001a: /* 26, HighISOMultiplierBlue */
        imPana.HighISOMultiplier[tag - 0x0018] = get2();
        break;
      case 0x001c: /*  28, BlackLevelRed */
      case 0x001d: /*  29, BlackLevelGreen */
      case 0x001e: /*  30, BlackLevelBlue */
        pana_black[tag - 0x001c] = get2();
        break;
      case 0x002d: /*  45, RawFormat */
                   /* pana_encoding: tag 0x002d (45dec)
                        not used - DMC-LX1/FZ30/FZ50/L1/LX1/LX2
                        2 - RAW DMC-FZ8/FZ18
                        3 - RAW DMC-L10
                        4 - RW2 for most other models, including G9 in "pixel shift off"
                      mode and YUNEEC CGO4            (must add 15 to black levels for
                      RawFormat == 4)            5 - RW2 DC-GH5s; G9 in "pixel shift on"
                      mode            6 - RW2            DC-S1, DC-S1R in "pixel shift off"
                      mode            7 -            RW2 DC-S1R (probably            DC-S1 too) in
                      "pixel shift on" mode
                   */
        pana_encoding = get2();
        break;
      case 0x002f: /*  47, CropTop */
        imgdata.sizes.raw_inset_crops[0].ctop = get2();
        break;
      case 0x0030: /*  48, CropLeft */
        imgdata.sizes.raw_inset_crops[0].cleft = get2();
        break;
      case 0x0031: /*  49, CropBottom */
        imgdata.sizes.raw_inset_crops[0].cheight =
            get2() - imgdata.sizes.raw_inset_crops[0].ctop;
        break;
      case 0x0032: /*  50, CropRight */
        imgdata.sizes.raw_inset_crops[0].cwidth =
            get2() - imgdata.sizes.raw_inset_crops[0].cleft;
        break;
      case 0x0037: /*  55, ISO if  ISO in 0x8827 & ISO in 0x0017 == 65535 */
        if (iso_speed == 65535)
          iso_speed = get4();
        break;
      case 0x011c: /* 284, Gamma */
      {
        int n = get2();
        if (n >= 1024)
          imPana.gamma = (float)n / 1024.0f;
        else if (n >= 256)
          imPana.gamma = (float)n / 256.0f;
        else
          imPana.gamma = (float)n / 100.0f;
      }
      break;
      case 0x0120: /* 288, CameraIFD, contains tags 0x1xxx, 0x2xxx, 0x3xxx */
      {
        unsigned sorder = order;
        unsigned long sbase = base;
        base = ftell(ifp);
        order = get2();
        fseek(ifp, 2, SEEK_CUR);
        fseek(ifp, INT64(get4()) - 8LL, SEEK_CUR);
        parse_tiff_ifd(base);
        base = sbase;
        order = sorder;
      }
      break;
      case 0x0121: /* 289, Multishot, 0 is Off, 65536 is Pixel Shift */
        imPana.Multishot = get4();
        break;
      case 0x1001:
      	if (imPana.Multishot == 0) {
      	  imPana.Multishot = get4();
      	  if (imPana.Multishot)
      	    imPana.Multishot += 65535;
      	}
        break;
      case 0x1100:
        imPana.FocusStepNear = get2();
        break;
      case 0x1101:
        imPana.FocusStepCount = get2();
        break;
      case 0x1105:
        imPana.ZoomPosition = get4();
        break;
      case 0x1201:
        if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT)) {
          imPana.LensManufacturer = fgetc(ifp);
        } else if (type == 258) {
          imPana.LensManufacturer = get4();
          if (imPana.LensManufacturer >= 257) {
            ilm.LensMount = LIBRAW_MOUNT_LPS_L;
            ilm.LensFormat = LIBRAW_FORMAT_FF;
          }
        }
        break;
      case 0x1202:
        if (ilm.LensMount == LIBRAW_MOUNT_LPS_L) {
          if ((utmp = get2())) ilm.LensID = utmp;
        } else if ((imPana.LensManufacturer != 0xff) &&
                   (imPana.LensManufacturer != 0xffffffff)) {
          if ((utmp = (fgetc(ifp) << 8) | fgetc(ifp)))
            ilm.LensID = (imPana.LensManufacturer << 16) + utmp;
        }
        break;
      case 0x1203: /* 4611, FocalLengthIn35mmFormat, contained in 0x0120
                      CameraIFD */
        if (imgdata.lens.FocalLengthIn35mmFormat < 0.65f)
          imgdata.lens.FocalLengthIn35mmFormat = get2();
        break;
      case 0x2009: /* 8201, contained in 0x0120 CameraIFD */
        if ((pana_encoding == 4) || (pana_encoding == 5))
        {
          i = MIN(8, len);
          int permut[8] = {3, 2, 1, 0, 3 + 4, 2 + 4, 1 + 4, 0 + 4};
          imPana.BlackLevelDim = len;
          for (j = 0; j < i; j++)
          {
            imPana.BlackLevel[permut[j]] =
                (float)(get2()) / (float)(powf(2.f, 14.f - pana_bpp));
          }
        }
        break;
      case 0x3420: /* 13344, WB_RedLevelAuto, contained in 0x0120 CameraIFD */
        icWBC[LIBRAW_WBI_Auto][0] = get2();
        icWBC[LIBRAW_WBI_Auto][1] = icWBC[LIBRAW_WBI_Auto][3] = 1024.0f;
        break;
      case 0x3421: /* 13345, WB_BlueLevelAuto, contained in 0x0120 CameraIFD */
        icWBC[LIBRAW_WBI_Auto][2] = get2();
        break;
      case 0x0002: /*   2, ImageWidth */
        tiff_ifd[ifd].t_width = getint(type);
        break;
      case 0x0003: /*   3, ImageHeight */
        tiff_ifd[ifd].t_height = getint(type);
        break;
      case 0x0005: /*   5, SensorLeftBorder */
        width = get2();
        imgdata.sizes.raw_inset_crops[0].cleft = width;
        break;
      case 0x0006: /*   6, SensorBottomBorder */
        height = get2();
        imgdata.sizes.raw_inset_crops[0].cheight =
            height - imgdata.sizes.raw_inset_crops[0].ctop;
        break;
      case 0x0007: /*   7, SensorRightBorder */
        i = get2();
        width += i;
        imgdata.sizes.raw_inset_crops[0].cwidth =
            i - imgdata.sizes.raw_inset_crops[0].cleft;
        break;
      case 0x0009: /*   9, CFAPattern */
        if ((i = get2()))
          filters = i;
        break;
      case 0x0011: /*  17, RedBalance */
      case 0x0012: /*  18, BlueBalance */
        if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT) && len == 1)
          cam_mul[(tag - 0x0011) * 2] = get2() / 256.0;
        break;
      case 0x0017: /*  23, ISO */
        if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT))
          iso_speed = get2();
        break;
      case 0x0024: /*  36, WBRedLevel */
      case 0x0025: /*  37, WBGreenLevel */
      case 0x0026: /*  38, WBBlueLevel */
        cam_mul[tag - 0x0024] = get2();
        break;
      case 0x0027: /*  39, WBInfo2 */
        if ((i = get2()) > 0x100)
          break;
        for (c = 0; c < i; c++)
        {
          if ((j = get2()) < 0x100)
          {
            icWBC[j][0] = get2();
            icWBC[j][1] = icWBC[j][3] = get2();
            icWBC[j][2] = get2();
            if (c == 1 && i > 6 && cam_mul[0] <= 0.001f)
                for (int q = 0; q < 4; q++)
                    cam_mul[q] = icWBC[j][q];
          }
          else
            fseek(ifp, 6, SEEK_CUR);
        }
        break;
      case 0x002e: /*  46, JpgFromRaw */
        if ((type != LIBRAW_EXIFTAG_TYPE_UNDEFINED) || (fgetc(ifp) != 0xff) || (fgetc(ifp) != 0xd8))
          break;
        thumb_offset = ftell(ifp) - 2;
        thumb_length = len;
        break;

      case 0x0118: /* 280, Panasonic RW2 offset */
        if (type != LIBRAW_EXIFTAG_TYPE_LONG)
          break;
        load_raw = &LibRaw::panasonic_load_raw;
        load_flags = 0x2008;
      case 0x0111: /* 273, StripOffset */
        if (len > 1 && len < 16384)
        {
          off_t sav = ftell(ifp);
          tiff_ifd[ifd].strip_offsets = (int *)calloc(len, sizeof(int));
          tiff_ifd[ifd].strip_offsets_count = len;
          for (int ii = 0; ii < (int)len; ii++)
            tiff_ifd[ifd].strip_offsets[ii] = get4() + base;
          fseek(ifp, sav, SEEK_SET); // restore position
        }
        /* fallthrough */
        tiff_ifd[ifd].offset = get4() + base;
        if (!tiff_ifd[ifd].bps && tiff_ifd[ifd].offset > 0)
        {
          fseek(ifp, tiff_ifd[ifd].offset, SEEK_SET);
          if (ljpeg_start(&jh, 1))
          {
            tiff_ifd[ifd].comp = 6;
            tiff_ifd[ifd].t_width = jh.wide;
            tiff_ifd[ifd].t_height = jh.high;
            tiff_ifd[ifd].bps = jh.bits;
            tiff_ifd[ifd].samples = jh.clrs;
            if (!(jh.sraw || (jh.clrs & 1)))
              tiff_ifd[ifd].t_width *= jh.clrs;
            if ((tiff_ifd[ifd].t_width > 4 * tiff_ifd[ifd].t_height) & ~jh.clrs)
            {
              tiff_ifd[ifd].t_width /= 2;
              tiff_ifd[ifd].t_height *= 2;
            }
            i = order;
            parse_tiff(tiff_ifd[ifd].offset + 12);
            order = i;
          }
        }
        break;
      }

    } /* processing of Panasonic-specific tags finished */

    switch (tag)
    {            /* processing of general EXIF tags */
    case 0xf000: /* 61440, Fuji HS10 table */
      fseek(ifp, get4() + base, SEEK_SET);
      parse_tiff_ifd(base);
      break;
    case 0x00fe: /* NewSubfileType */
      tiff_ifd[ifd].newsubfiletype = getreal(type);
      break;
    case 0x0100: /* 256, ImageWidth */
    case 0xf001: /* 61441, Fuji RAF RawImageFullWidth */
      tiff_ifd[ifd].t_width = getint(type);
      break;
    case 0x0101: /* 257, ImageHeight */
    case 0xf002: /* 61442, Fuji RAF RawImageFullHeight */
      tiff_ifd[ifd].t_height = getint(type);
      break;
    case 0x0102: /* 258, BitsPerSample */
    case 0xf003: /* 61443, Fuji RAF 0xf003 */
      if(!tiff_ifd[ifd].samples || tag != 0x0102) // ??? already set by tag 0x115
        tiff_ifd[ifd].samples = len & 7;
      tiff_ifd[ifd].bps = getint(type);
      if (tiff_bps < (unsigned)tiff_ifd[ifd].bps)
        tiff_bps = tiff_ifd[ifd].bps;
      break;
    case 0xf006: /* 61446, Fuji RAF 0xf006 */
      raw_height = 0;
      if (tiff_ifd[ifd].bps > 12)
        break;
      load_raw = &LibRaw::packed_load_raw;
      load_flags = get4() ? 24 : 80;
      break;
    case 0x0103: /* 259, Compression */
                 /*
                    262	 = Kodak 262
                  32767  = Sony ARW Compressed
                  32769  = Packed RAW
                  32770  = Samsung SRW Compressed
                  32772  = Samsung SRW Compressed 2
                  32867  = Kodak KDC Compressed
                  34713  = Nikon NEF Compressed
                  65000  = Kodak DCR Compressed
                  65535  = Pentax PEF Compressed
                 */
      tiff_ifd[ifd].comp = getint(type);
      break;
    case 0x0106: /* 262, PhotometricInterpretation */
      tiff_ifd[ifd].phint = get2();
      break;
    case 0x010e: /* 270, ImageDescription */
      fread(desc, 512, 1, ifp);
      break;
    case 0x010f: /* 271, Make */
      fgets(make, 64, ifp);
      break;
    case 0x0110: /* 272, Model */
      if (!strncmp(make, "Hasselblad", 10) && model[0] &&
          (imHassy.format != LIBRAW_HF_Imacon))
        break;
      fgets(model, 64, ifp);
      break;
    case 0x0116: // 278
      tiff_ifd[ifd].rows_per_strip = getint(type);
      break;
    case 0x0112: /* 274, Orientation */
      tiff_ifd[ifd].t_flip = "50132467"[get2() & 7] - '0';
      break;
    case 0x0115: /* 277, SamplesPerPixel */
      tiff_ifd[ifd].samples = getint(type) & 7;
      break;
    case 0x0152: /* Extrasamples */
      tiff_ifd[ifd].extrasamples = (getint(type) & 0xff) + 1024;
      break;
    case 0x0117: /* 279, StripByteCounts */
      if (len > 1 && len < 16384)
      {
        off_t sav = ftell(ifp);
        tiff_ifd[ifd].strip_byte_counts = (int *)calloc(len, sizeof(int));
        tiff_ifd[ifd].strip_byte_counts_count = len;
        for (int ii = 0; ii < (int)len; ii++)
          tiff_ifd[ifd].strip_byte_counts[ii] = get4();
        fseek(ifp, sav, SEEK_SET); // restore position
      }
      /* fallback */
    case 0x0202: // 514
    case 0xf008: // 61448
      tiff_ifd[ifd].bytes = get4();
      break;
    case 0xf00e: // 61454, FujiFilm "As Shot"
      FORC3 cam_mul[GRBG_2_RGBG(c)] = getint(type);
      break;
    case 0x0131: /* 305, Software */
      fgets(software, 64, ifp);
      if (!strncmp(software, "Adobe", 5) || !strncmp(software, "dcraw", 5) ||
          !strncmp(software, "UFRaw", 5) || !strncmp(software, "Bibble", 6) ||
          !strcmp(software, "Digital Photo Professional"))
        is_raw = 0;
      break;
    case 0x0132: /* 306, DateTime */
      get_timestamp(0);
      break;
    case 0x013b: /* 315, Artist */
      fread(artist, 64, 1, ifp);
      break;
    case 0x013d: // 317
      tiff_ifd[ifd].predictor = getint(type);
      break;
    case 0x0142: /* 322, TileWidth */
      tiff_ifd[ifd].t_tile_width = getint(type);
      break;
    case 0x0143: /* 323, TileLength */
      tiff_ifd[ifd].t_tile_length = getint(type);
      break;
    case 0x0144: /* 324, TileOffsets */
      tiff_ifd[ifd].offset = len > 1 ? ftell(ifp) : get4();
      if (len == 1)
        tiff_ifd[ifd].t_tile_width = tiff_ifd[ifd].t_tile_length = 0;
      if (len == 4)
      {
        load_raw = &LibRaw::sinar_4shot_load_raw;
        is_raw = 5;
      }
      break;
    case 0x0145: // 325
      tiff_ifd[ifd].bytes = len > 1 ? ftell(ifp) : get4();
      break;
    case 0x014a: /* 330, SubIFDs */
      if (!strcmp(model, "DSLR-A100") && tiff_ifd[ifd].t_width == 3872)
      {
        load_raw = &LibRaw::sony_arw_load_raw;
        data_offset = get4() + base;
        ifd++;
        if (ifd >= int(sizeof tiff_ifd / sizeof tiff_ifd[0]))
          throw LIBRAW_EXCEPTION_IO_CORRUPT;
        break;
      }
      if (!strncmp(make, "Hasselblad", 10) &&
          libraw_internal_data.unpacker_data.hasselblad_parser_flag)
      {
        fseek(ifp, ftell(ifp) + 4, SEEK_SET);
        fseek(ifp, get4() + base, SEEK_SET);
        parse_tiff_ifd(base);
        break;
      }
      if (len > 1000)
        len = 1000; /* 1000 SubIFDs is enough */
      while (len--)
      {
        i = ftell(ifp);
        fseek(ifp, get4() + base, SEEK_SET);
        if (parse_tiff_ifd(base))
          break;
        fseek(ifp, i + 4, SEEK_SET);
      }
      break;
    case 0x0153: // 339
      tiff_ifd[ifd].sample_format = getint(type);
      break;
    case 0x0190: // 400
      strcpy(make, "Sarnoff");
      maximum = 0xfff;
      break;
    case 0x02bc: // 700
      if ((tagtypeIs(LIBRAW_EXIFTAG_TYPE_BYTE) ||
          tagtypeIs(LIBRAW_EXIFTAG_TYPE_ASCII) ||
          tagtypeIs(LIBRAW_EXIFTAG_TYPE_SBYTE) ||
          tagtypeIs(LIBRAW_EXIFTOOLTAGTYPE_binary)) &&
          (len > 1) && (len < 5100000))
      {
        xmpdata = (char *)malloc(xmplen = len + 1);
        fread(xmpdata, len, 1, ifp);
        xmpdata[len] = 0;
      }
      break;
    case 0x7000:
      imSony.SonyRawFileType = get2();
      break;
    case 0x7010: // 28688
      FORC4 sony_curve[c + 1] = get2() >> 2 & 0xfff;
      for (i = 0; i < 5; i++)
        for (j = sony_curve[i] + 1; j <= (int)sony_curve[i + 1]; j++)
          curve[j] = curve[j - 1] + (1 << i);
      break;
    case 0x7200: // 29184, Sony SR2Private
      sony_offset = get4();
      break;
    case 0x7201: // 29185, Sony SR2Private
      sony_length = get4();
      break;
    case 0x7221: // 29217, Sony SR2Private
      sony_key = get4();
      break;
    case 0x7250: // 29264, Sony SR2Private
      parse_minolta(ftell(ifp));
      raw_width = 0;
      break;
    case 0x7303: // 29443, Sony SR2SubIFD
      FORC4 cam_mul[GRBG_2_RGBG(c)] = get2();
      break;
    case 0x7313: // 29459, Sony SR2SubIFD
      FORC4 cam_mul[RGGB_2_RGBG(c)] = get2();
      break;
    case 0x7310: // 29456, Sony SR2SubIFD
      FORC4 cblack[RGGB_2_RGBG(c)] = get2();
      i = cblack[3];
      FORC3 if (i > (int)cblack[c]) i = cblack[c];
      FORC4 cblack[c] -= i;
      black = i;
      break;
    case 0x827d: /* 33405, Model2 */
                 /*
                  for Kodak ProBack 645 PB645x-yyyy 'x' is:
                  'M' for Mamiya 645
                  'C' for Contax 645
                  'H' for Hasselblad H-series
                 */
      fgets(model2, 64, ifp);
      break;
    case 0x828d: /* 33421, CFARepeatPatternDim */
      if (get2() == 6 && get2() == 6)
        tiff_ifd[ifd].t_filters = filters = 9;
      break;
    case 0x828e: /* 33422, CFAPattern */
      if (filters == 9)
      {
        FORC(36)((char *)xtrans)[c] = fgetc(ifp) & 3;
        break;
      }
    case 0xfd09: /* 64777, Kodak P-series */
      if (len == 36)
      {
        tiff_ifd[ifd].t_filters = filters = 9;
        colors = 3;
        FORC(36)((char *)xtrans)[c] = fgetc(ifp) & 3;
      }
      else if (len > 0)
      {
        if ((plen = len) > 16)
          plen = 16;
        fread(cfa_pat, 1, plen, ifp);
        for (colors = cfa = i = 0; i < (int)plen && colors < 4; i++)
        {
          if (cfa_pat[i] > 31)
            continue; // Skip wrong data
          colors += !(cfa & (1 << cfa_pat[i]));
          cfa |= 1 << cfa_pat[i];
        }
        if (cfa == 070)
          memcpy(cfa_pc, "\003\004\005", 3); /* CMY */
        if (cfa == 072)
          memcpy(cfa_pc, "\005\003\004\001", 4); /* GMCY */
        goto guess_cfa_pc;
      }
      break;
    case 0x8290: // 33424
    case 0xfe00: // 65024
      fseek(ifp, get4() + base, SEEK_SET);
      parse_kodak_ifd(base);
      break;
    case 0x829a: /* 33434, ExposureTime */
      tiff_ifd[ifd].t_shutter = shutter = getreal(type);
      break;
    case 0x829d: /* 33437, FNumber */
      aperture = getreal(type);
      break;
    case 0x9400:
      imCommon.exifAmbientTemperature = getreal(type);
      if ((imCommon.CameraTemperature > -273.15f) &&
          ((OlyID == OlyID_TG_5) || (OlyID == OlyID_TG_6)))
        imCommon.CameraTemperature +=
            imCommon.exifAmbientTemperature;
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
    case 0xc62f:
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
      break;
    case 0xc630: // DNG LensInfo, Lens Specification per EXIF standard
      imgdata.lens.MinFocal = getreal(type);
      imgdata.lens.MaxFocal = getreal(type);
      imgdata.lens.MaxAp4MinFocal = getreal(type);
      imgdata.lens.MaxAp4MaxFocal = getreal(type);
      break;
    case 0xa420: /* 42016, ImageUniqueID */
      stmread(imgdata.color.ImageUniqueID, len, ifp);
      break;
    case 0xc65d: /* 50781, RawDataUniqueID */
      imgdata.color.RawDataUniqueID[16] = 0;
      fread(imgdata.color.RawDataUniqueID, 1, 16, ifp);
      break;
    case 0xa433: // LensMake
      stmread(imgdata.lens.LensMake, len, ifp);
      break;
    case 0xa434: // LensModel
      stmread(imgdata.lens.Lens, len, ifp);
      if (!strncmp(imgdata.lens.Lens, "----", 4))
        imgdata.lens.Lens[0] = 0;
      break;
    case 0x9205:
      imgdata.lens.EXIF_MaxAp = libraw_powf64l(2.0f, (getreal(type) / 2.0f));
      break;
    case 0x8602: /* 34306, Leaf white balance */
      FORC4
      {
        int q = get2();
        if (q)
          cam_mul[GRGB_2_RGBG(c)] = 4096.0 / q;
      }
      break;
    case 0x8603: /* 34307, Leaf CatchLight color matrix */
      fread(software, 1, 7, ifp);
      if (strncmp(software, "MATRIX", 6))
        break;
      colors = 4;
      for (raw_color = i = 0; i < 3; i++)
      {
        FORC4 fscanf(ifp, "%f", &rgb_cam[i][GRGB_2_RGBG(c)]);
        if (!use_camera_wb)
          continue;
        num = 0;
        FORC4 num += rgb_cam[i][c];
        FORC4 rgb_cam[i][c] /= MAX(1, num);
      }
      break;
    case 0x8606: /* 34310, Leaf metadata */
      parse_mos(ftell(ifp));
    case 0x85ff: // 34303
      strcpy(make, "Leaf");
      break;
    case 0x8769: /* 34665, EXIF tag */
      fseek(ifp, get4() + base, SEEK_SET);
      parse_exif(base);
      break;
    case 0x8825: /* 34853, GPSInfo tag */
    {
      unsigned pos;
      fseek(ifp, pos = (get4() + base), SEEK_SET);
      parse_gps(base);
      fseek(ifp, pos, SEEK_SET);
      parse_gps_libraw(base);
    }
    break;
    case 0x8773: /* 34675, InterColorProfile */
    case 0xc68f: /* 50831, AsShotICCProfile */
      profile_offset = ftell(ifp);
      profile_length = len;
      break;
    case 0x9102: /* 37122, CompressedBitsPerPixel */
      kodak_cbpp = get4();
      break;
    case 0x920a: /* 37386, FocalLength */
      focal_len = getreal(type);
      break;
    case 0x9211: /* 37393, ImageNumber */
      shot_order = getint(type);
      break;
    case 0x9215: /* 37397, ExposureIndex */
      imCommon.exifExposureIndex = getreal(type);
      break;
    case 0x9218: /* 37400, old Kodak KDC tag */
      for (raw_color = i = 0; i < 3; i++)
      {
        getreal(type);
        FORC3 rgb_cam[i][c] = getreal(type);
      }
      break;
    case 0xa010: // 40976
      strip_offset = get4();
      switch (tiff_ifd[ifd].comp)
      {
      case 0x8002: // 32770
        load_raw = &LibRaw::samsung_load_raw;
        break;
      case 0x8004: // 32772
        load_raw = &LibRaw::samsung2_load_raw;
        break;
      case 0x8005: // 32773
        load_raw = &LibRaw::samsung3_load_raw;
        break;
      }
      break;
    case 0xb4c3: /* 46275, Imacon tags */
      imHassy.format = LIBRAW_HF_Imacon;
      strcpy(make, "Imacon");
      data_offset = ftell(ifp);
      ima_len = len;
      break;
    case 0xb4c7: // 46279
      if (!ima_len)
        break;
      fseek(ifp, 38, SEEK_CUR);
    case 0xb4c2: // 46274
      fseek(ifp, 40, SEEK_CUR);
      raw_width = get4();
      raw_height = get4();
      left_margin = get4() & 7;
      width = raw_width - left_margin - (get4() & 7);
      top_margin = get4() & 7;
      height = raw_height - top_margin - (get4() & 7);
      if (raw_width == 7262 && ima_len == 234317952)
      {
        height = 5412;
        width = 7216;
        left_margin = 7;
        filters = 0;
      }
      else if (raw_width == 7262)
      {
        height = 5444;
        width = 7244;
        left_margin = 7;
      }
      fseek(ifp, 52, SEEK_CUR);
      FORC3 cam_mul[c] = getreal(LIBRAW_EXIFTAG_TYPE_FLOAT);
      fseek(ifp, 114, SEEK_CUR);
      flip = (get2() >> 7) * 90;
      if (width * (height * 6l) == ima_len)
      {
        if (flip % 180 == 90)
          SWAP(width, height);
        raw_width = width;
        raw_height = height;
        left_margin = top_margin = filters = flip = 0;
      }
      c = unsigned(height) * unsigned(width) / 1000000;
      if (c == 32)
        c--;
      sprintf(model, "Ixpress %d-Mp", c);
      load_raw = &LibRaw::imacon_full_load_raw;
      if (filters)
      {
        if (left_margin & 1)
          filters = 0x61616161;
        load_raw = &LibRaw::unpacked_load_raw;
      }
      maximum = 0xffff;
      break;
    case 0xc516: /* 50454, Sinar tag */
    case 0xc517: // 50455
      if (len < 1 || len > 2560000 || !(cbuf = (char *)malloc(len)))
        break;
      if (fread(cbuf, 1, len, ifp) != (int)len)
        throw LIBRAW_EXCEPTION_IO_CORRUPT; // cbuf to be free'ed in recycle
      cbuf[len - 1] = 0;
      for (cp = cbuf - 1; cp && cp < cbuf + len; cp = strchr(cp, '\n'))
        if (!strncmp(++cp, "Neutral ", 8))
          sscanf(cp + 8, "%f %f %f", cam_mul, cam_mul + 1, cam_mul + 2);
      free(cbuf);
      break;
    case 0xc51a: // 50458
      if (!make[0])
        strcpy(make, "Hasselblad");
      break;
    case 0xc51b: /* 50459, Hasselblad tag */
      if (!libraw_internal_data.unpacker_data.hasselblad_parser_flag)
      {
        libraw_internal_data.unpacker_data.hasselblad_parser_flag = 1;
        i = order;
        j = ftell(ifp);
        c = tiff_nifds;
        order = get2();
        fseek(ifp, j + (get2(), get4()), SEEK_SET);
        parse_tiff_ifd(j);
        maximum = 0xffff;
        tiff_nifds = c;
        order = i;
        break;
      }
    case 0xc612: /* 50706, DNGVersion */
      FORC4 dng_version = (dng_version << 8) + fgetc(ifp);
      if (!make[0])
        strcpy(make, "DNG");
      is_raw = 1;
      break;
    case 0xc614: /* 50708, UniqueCameraModel */
      stmread(imgdata.color.UniqueCameraModel, len, ifp);
      if (model[0])
        break;
      strncpy(make, imgdata.color.UniqueCameraModel,
              MIN(len, sizeof(imgdata.color.UniqueCameraModel)));
      if ((cp = strchr(make, ' ')))
      {
        strcpy(model, cp + 1);
        *cp = 0;
      }
      break;
    case 0xc616: /* 50710, CFAPlaneColor */
      if (filters == 9)
        break;
      if (len > 4)
        len = 4;
      colors = len;
      fread(cfa_pc, 1, colors, ifp);
    guess_cfa_pc:
      FORCC tab[cfa_pc[c]] = c;
      cdesc[c] = 0;
      for (i = 16; i--;)
        filters = filters << 2 | tab[cfa_pat[i % plen]];
      filters -= !filters;
      tiff_ifd[ifd].t_filters = filters;
      break;
    case 0xc617: /* 50711, CFALayout */
      if (get2() == 2)
        tiff_ifd[ifd].t_fuji_width = fuji_width = 1;
      break;
    case 0x0123: // 291
    case 0xc618: /* 50712, LinearizationTable */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_LINTABLE;
      tiff_ifd[ifd].lineartable_offset = ftell(ifp);
      tiff_ifd[ifd].lineartable_len = len;
      linear_table(len);
      break;
    case 0xc619: /* 50713, BlackLevelRepeatDim */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_BLACK;
      tiff_ifd[ifd].dng_levels.dng_fcblack[4] =
          tiff_ifd[ifd].dng_levels.dng_cblack[4] = cblack[4] = get2();
      tiff_ifd[ifd].dng_levels.dng_fcblack[5] =
          tiff_ifd[ifd].dng_levels.dng_cblack[5] = cblack[5] = get2();
      if (cblack[4] * cblack[5] >
          (LIBRAW_CBLACK_SIZE -
           7)) // Use last cblack item as DNG black level count
        tiff_ifd[ifd].dng_levels.dng_fcblack[4] =
            tiff_ifd[ifd].dng_levels.dng_fcblack[5] =
                tiff_ifd[ifd].dng_levels.dng_cblack[4] =
                    tiff_ifd[ifd].dng_levels.dng_cblack[5] = cblack[4] =
                        cblack[5] = 1;
      break;

    case 0xf00c:
      if (imFuji.RAFDataGeneration != 4096)
      {
        unsigned fwb[4];
        FORC4 fwb[c] = get4();
        if (fwb[3] < 0x100)
        {
          FORC3 icWBC[fwb[3]][GRBG_2_RGBG(c)] = fwb[c];
          icWBC[fwb[3]][3] = icWBC[fwb[3]][1];
          if ((fwb[3] == 17) &&                                      // Tungsten WB
              (libraw_internal_data.unpacker_data.lenRAFData > 3) &&
              (libraw_internal_data.unpacker_data.lenRAFData < 10240000))
          {
            INT64 f_save = ftell(ifp);
            rafdata = (ushort *)malloc(
                sizeof(ushort) * libraw_internal_data.unpacker_data.lenRAFData);
            fseek(ifp, libraw_internal_data.unpacker_data.posRAFData, SEEK_SET);
            fread(rafdata, sizeof(ushort),
                  libraw_internal_data.unpacker_data.lenRAFData, ifp);
            fseek(ifp, f_save, SEEK_SET);

            uchar *PrivateMknBuf = (uchar *)rafdata;
            int PrivateMknLength = libraw_internal_data.unpacker_data.lenRAFData
                                   << 1;
            for (int pos = 0; pos < PrivateMknLength - 16; pos++)
            {
              if (!memcmp(PrivateMknBuf + pos, "TSNERDTS", 8)) // STDRENST
              {
                imFuji.isTSNERDTS = 1;
                break;
              }
            }
            int fj; // 31? (fj<<1)-0x3c : 34? (fj<<1)-0x4e : undef
            int is34 = 0;
            if ((imFuji.RAFDataVersion == 0x0260) || // X-Pro3, GFX 100S
                (imFuji.RAFDataVersion == 0x0261) || // X100V, GFX 50S II
                (imFuji.RAFDataVersion == 0x0262) || // X-T4
                (imFuji.RAFDataVersion == 0x0263) || // X-H2S
                (imFuji.RAFDataVersion == 0x0264) || // X-S10
                (imFuji.RAFDataVersion == 0x0265) || // X-E4
                (imFuji.RAFDataVersion == 0x0266) || // X-T30 II
                !strcmp(model, "X-Pro3")     ||
                !strcmp(model, "GFX 100S")   ||
                !strcmp(model, "GFX100S")    ||
                !strcmp(model, "GFX 50S II") ||
                !strcmp(model, "GFX50S II")  ||
                !strcmp(model, "X100V")      ||
                !strcmp(model, "X-T4")       ||
                !strcmp(model, "X-H2S")      ||
                !strcmp(model, "X-E4")       ||
                !strcmp(model, "X-T30 II")   ||
                !strcmp(model, "X-S10"))
// is34 cameras have 34 CCT values instead of 31, manual still claims 2500 to 10000 K
// aligned 3000 K to Incandescent, as it is usual w/ other Fujifilm cameras
              is34 = 1;

            for (int fi = 0;
                 fi < int(libraw_internal_data.unpacker_data.lenRAFData - 3); fi++) // looking for Tungsten WB
            {
              if ((fwb[0] == rafdata[fi]) && (fwb[1] == rafdata[fi + 1]) &&
                  (fwb[2] == rafdata[fi + 2])) // found Tungsten WB
              {
                if (rafdata[fi - 15] !=
                    fwb[0]) // 15 is offset of Tungsten WB from the first
                            // preset, Fine Weather WB
                  continue;
                for (int wb_ind = 0, ofst = fi - 15; wb_ind < (int)Fuji_wb_list1.size();
                     wb_ind++, ofst += 3)
                {
                  icWBC[Fuji_wb_list1[wb_ind]][1] =
                      icWBC[Fuji_wb_list1[wb_ind]][3] = rafdata[ofst];
                  icWBC[Fuji_wb_list1[wb_ind]][0] = rafdata[ofst + 1];
                  icWBC[Fuji_wb_list1[wb_ind]][2] = rafdata[ofst + 2];
                }

                if (is34)
                  fi += 24;
                fi += 96;
                for (fj = fi; fj < (fi + 15); fj += 3) // looking for the end of the WB table
                {
                  if (rafdata[fj] != rafdata[fi])
                  {
                    fj -= 93;
                    if (is34)
                      fj -= 9;
// printf ("wb start in DNG: 0x%04x\n", fj*2-0x4e);
                    for (int iCCT = 0, ofst = fj; iCCT < 31;
                         iCCT++, ofst += 3)
                    {
                      icWBCCTC[iCCT][0] = FujiCCT_K[iCCT];
                      icWBCCTC[iCCT][1] = rafdata[ofst + 1];
                      icWBCCTC[iCCT][2] = icWBCCTC[iCCT][4] = rafdata[ofst];
                      icWBCCTC[iCCT][3] = rafdata[ofst + 2];
                    }
                    break;
                  }
                }
                free(rafdata);
                break;
              }
            }
          }
        }
        FORC4 fwb[c] = get4();
        if (fwb[3] < 0x100) {
          FORC3 icWBC[fwb[3]][GRBG_2_RGBG(c)] = fwb[c];
          icWBC[fwb[3]][3] = icWBC[fwb[3]][1];
        }
      }
      break;
    case 0xf00d:
      if (imFuji.RAFDataGeneration != 4096)
      {
        FORC3 icWBC[LIBRAW_WBI_Auto][GRBG_2_RGBG(c)] = getint(type);
        icWBC[LIBRAW_WBI_Auto][3] = icWBC[LIBRAW_WBI_Auto][1];
      }
      break;
    case 0xc615: /* 50709, LocalizedCameraModel */
      stmread(imgdata.color.LocalizedCameraModel, len, ifp);
      break;
    case 0xf00a: // 61450
      cblack[4] = cblack[5] = MIN(sqrt((double)len), 64);
    case 0xc61a: /* 50714, BlackLevel */
      if (tiff_ifd[ifd].samples > 1 &&
          tiff_ifd[ifd].samples == (int)len) // LinearDNG, per-channel black
      {
        tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_BLACK;
        for (i = 0; i < 4 && i < (int)len; i++)
        {
          tiff_ifd[ifd].dng_levels.dng_fcblack[i] = getreal(type);
          tiff_ifd[ifd].dng_levels.dng_cblack[i] = cblack[i] =
              tiff_ifd[ifd].dng_levels.dng_fcblack[i] + 0.5;
        }
        // Record len in last cblack field
        tiff_ifd[ifd].dng_levels.dng_cblack[LIBRAW_CBLACK_SIZE - 1] = len;

        tiff_ifd[ifd].dng_levels.dng_fblack =
            tiff_ifd[ifd].dng_levels.dng_black = black = 0;
      }
      else if (tiff_ifd[ifd].samples > 1 // Linear DNG w repeat dim
               && (tiff_ifd[ifd].samples * cblack[4] * cblack[5] == len))
      {
        tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_BLACK;
        tiff_ifd[ifd].dng_levels.dng_cblack[LIBRAW_CBLACK_SIZE - 1] =
            cblack[LIBRAW_CBLACK_SIZE - 1] = len;
        for (i = 0; i < (int)len && i < LIBRAW_CBLACK_SIZE - 7; i++)
        {
          tiff_ifd[ifd].dng_levels.dng_fcblack[i + 6] = getreal(type);
          tiff_ifd[ifd].dng_levels.dng_cblack[i + 6] = cblack[i + 6] =
              tiff_ifd[ifd].dng_levels.dng_fcblack[i + 6] + 0.5;
        }
      }
      else if ((cblack[4] * cblack[5] < 2) && len == 1)
      {
        tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_BLACK;
        tiff_ifd[ifd].dng_levels.dng_fblack = getreal(type);
        black = tiff_ifd[ifd].dng_levels.dng_black =
            tiff_ifd[ifd].dng_levels.dng_fblack;
      }
      else if (cblack[4] * cblack[5] <= len)
      {
        FORC(int(cblack[4] * cblack[5]))
        {
          tiff_ifd[ifd].dng_levels.dng_fcblack[6 + c] = getreal(type);
          cblack[6 + c] = tiff_ifd[ifd].dng_levels.dng_fcblack[6 + c];
        }
        black = 0;
        FORC4
        cblack[c] = 0;

        if (tag == 0xc61a)
        {
          tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_BLACK;
          FORC(int(cblack[4] * cblack[5]))
          tiff_ifd[ifd].dng_levels.dng_cblack[6 + c] = cblack[6 + c];
          tiff_ifd[ifd].dng_levels.dng_fblack = 0;
          tiff_ifd[ifd].dng_levels.dng_black = 0;
          FORC4
          tiff_ifd[ifd].dng_levels.dng_fcblack[c] =
              tiff_ifd[ifd].dng_levels.dng_cblack[c] = 0;
        }
      }
      break;
    case 0xc61b: /* 50715, BlackLevelDeltaH */
    case 0xc61c: /* 50716, BlackLevelDeltaV */
      for (num = i = 0; i < (int)len && i < 65536; i++)
        num += getreal(type);
      if (len > 0)
      {
        black += num / len + 0.5;
        tiff_ifd[ifd].dng_levels.dng_fblack += num / float(len);
        tiff_ifd[ifd].dng_levels.dng_black += num / len + 0.5;
        tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_BLACK;
      }
      break;
    case 0xc61d: /* 50717, WhiteLevel */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_WHITE;
      tiff_ifd[ifd].dng_levels.dng_whitelevel[0] = maximum = getint(type);
      if (tiff_ifd[ifd].samples > 1) // Linear DNG case
        for (i = 1; i < 4 && i < (int)len; i++)
          tiff_ifd[ifd].dng_levels.dng_whitelevel[i] = getint(type);
      break;
    case 0xc61e: /* DefaultScale */
    {
      float q1 = getreal(type);
      float q2 = getreal(type);
      if (q1 > 0.00001f && q2 > 0.00001f)
      {
        pixel_aspect = q1 / q2;
        if (pixel_aspect > 0.995 && pixel_aspect < 1.005)
          pixel_aspect = 1.0;
      }
    }
    break;
    case 0xc61f: /* 50719, DefaultCropOrigin */
      if (len == 2)
      {
        tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_CROPORIGIN;
        tiff_ifd[ifd].dng_levels.default_crop[0] = getreal(type);
        tiff_ifd[ifd].dng_levels.default_crop[1] = getreal(type);
        if (!strncasecmp(make, "SONY", 4))
        {
          imgdata.sizes.raw_inset_crops[0].cleft =
              tiff_ifd[ifd].dng_levels.default_crop[0];
          imgdata.sizes.raw_inset_crops[0].ctop =
              tiff_ifd[ifd].dng_levels.default_crop[1];
        }
      }
      break;

    case 0xc620: /* 50720, DefaultCropSize */
      if (len == 2)
      {
        tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_CROPSIZE;
        tiff_ifd[ifd].dng_levels.default_crop[2] = getreal(type);
        tiff_ifd[ifd].dng_levels.default_crop[3] = getreal(type);
        if (!strncasecmp(make, "SONY", 4))
        {
          imgdata.sizes.raw_inset_crops[0].cwidth =
              tiff_ifd[ifd].dng_levels.default_crop[2];
          imgdata.sizes.raw_inset_crops[0].cheight =
              tiff_ifd[ifd].dng_levels.default_crop[3];
        }
      }
      break;

    case 0xc7b5: /* 51125 DefaultUserCrop */
      if (len == 4)
      {
          int cnt = 0;
          FORC4
          {
              float v = getreal(type);
              if (v >= 0.f && v <= 1.f)
              {
                  tiff_ifd[ifd].dng_levels.user_crop[c] = v;
                  cnt++;
              }
          }
          if(cnt == 4 // valid values
              && tiff_ifd[ifd].dng_levels.user_crop[0] < tiff_ifd[ifd].dng_levels.user_crop[2] // top < bottom
              && tiff_ifd[ifd].dng_levels.user_crop[1] < tiff_ifd[ifd].dng_levels.user_crop[3] // left < right
              )
            tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_USERCROP;
      }
      break;
    case 0x74c7:
      if ((len == 2) && !strncasecmp(make, "SONY", 4))
      {
        imgdata.sizes.raw_inset_crops[0].cleft = get4();
        imgdata.sizes.raw_inset_crops[0].ctop = get4();
      }
      break;

    case 0x74c8:
      if ((len == 2) && !strncasecmp(make, "SONY", 4))
      {
        imgdata.sizes.raw_inset_crops[0].cwidth = get4();
        imgdata.sizes.raw_inset_crops[0].cheight = get4();
      }
      break;

    case 0xc65a: // 50778
      tiff_ifd[ifd].dng_color[0].illuminant = get2();
      tiff_ifd[ifd].dng_color[0].parsedfields |= LIBRAW_DNGFM_ILLUMINANT;
      break;
    case 0xc65b: // 50779
      tiff_ifd[ifd].dng_color[1].illuminant = get2();
      tiff_ifd[ifd].dng_color[1].parsedfields |= LIBRAW_DNGFM_ILLUMINANT;
      break;

    case 0xc621: /* 50721, ColorMatrix1 */
    case 0xc622: /* 50722, ColorMatrix2 */
    {
      int chan = (len == 9) ? 3 : (len == 12 ? 4 : 0);
      i = tag == 0xc621 ? 0 : 1;
      if (chan)
      {
        tiff_ifd[ifd].dng_color[i].parsedfields |= LIBRAW_DNGFM_COLORMATRIX;
        imHassy.nIFD_CM[i] = ifd;
      }
      FORC(chan) for (j = 0; j < 3; j++)
      {
        tiff_ifd[ifd].dng_color[i].colormatrix[c][j] = cm[c][j] = getreal(type);
      }
      use_cm = 1;
    }
    break;

    case 0xc714: /* ForwardMatrix1 */
    case 0xc715: /* ForwardMatrix2 */
    {
      int chan = (len == 9) ? 3 : (len == 12 ? 4 : 0);
      i = tag == 0xc714 ? 0 : 1;
      if (chan)
        tiff_ifd[ifd].dng_color[i].parsedfields |= LIBRAW_DNGFM_FORWARDMATRIX;
      for (j = 0; j < 3; j++)
        FORC(chan)
        {
          tiff_ifd[ifd].dng_color[i].forwardmatrix[j][c] = fm[j][c] =
              getreal(type);
        }
    }
    break;

    case 0xc623: /* 50723, CameraCalibration1 */
    case 0xc624: /* 50724, CameraCalibration2 */
    {
      int chan = (len == 9) ? 3 : (len == 16 ? 4 : 0);
      j = tag == 0xc623 ? 0 : 1;
      if (chan)
        tiff_ifd[ifd].dng_color[j].parsedfields |= LIBRAW_DNGFM_CALIBRATION;
      for (i = 0; i < chan; i++)
        FORC(chan)
        {
          tiff_ifd[ifd].dng_color[j].calibration[i][c] = cc[i][c] =
              getreal(type);
        }
    }
    break;
    case 0xc627: /* 50727, AnalogBalance */
      if (len >= 3)
        tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_ANALOGBALANCE;
      for (c = 0; c < (int)len && c < 4; c++)
      {
        tiff_ifd[ifd].dng_levels.analogbalance[c] = ab[c] = getreal(type);
      }
      break;
    case 0xc628: /* 50728, AsShotNeutral */
      if (len >= 3)
        tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_ASSHOTNEUTRAL;
      for (c = 0; c < (int)len && c < 4; c++)
        tiff_ifd[ifd].dng_levels.asshotneutral[c] = asn[c] = getreal(type);
      break;
    case 0xc629: /* 50729, AsShotWhiteXY */
      xyz[0] = getreal(type);
      xyz[1] = getreal(type);
      xyz[2] = 1 - xyz[0] - xyz[1];
      FORC3 xyz[c] /= LibRaw_constants::d65_white[c];
      break;
    case 0xc62a: /* DNG: 50730 BaselineExposure */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_BASELINEEXPOSURE;
      tiff_ifd[ifd].dng_levels.baseline_exposure = getreal(type);
      break;
    case 0xc62e: /* DNG: 50734 LinearResponseLimit */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_LINEARRESPONSELIMIT;
      tiff_ifd[ifd].dng_levels.LinearResponseLimit = getreal(type);
      break;

    case 0xc634: /* 50740 : DNG Adobe, DNG Pentax, Sony SR2, DNG Private */
      {
        char mbuf[64];
        INT64 curr_pos, start_pos = ftell(ifp);
        unsigned MakN_order, m_sorder = order;
        unsigned MakN_length;
        unsigned pos_in_original_raw;
        fread(mbuf, 1, 6, ifp);

        if (!strcmp(mbuf, "Adobe"))
        {
          order = 0x4d4d; // Adobe header is always in "MM" / big endian
          curr_pos = start_pos + 6;
          while (curr_pos + 8 - start_pos <= len)
          {
            fread(mbuf, 1, 4, ifp);
            curr_pos += 8;

            if (!strncmp(mbuf, "Pano", 4))
            { // PanasonicRaw, yes, they use "Pano" as signature
              parseAdobePanoMakernote();
            }

            if (!strncmp(mbuf, "MakN", 4))
            {
              MakN_length = get4();
              MakN_order = get2();
              pos_in_original_raw = get4();
              order = MakN_order;

              INT64 save_pos = ifp->tell();
              parse_makernote_0xc634(curr_pos + 6 - pos_in_original_raw, 0,
                                     AdobeDNG);

              curr_pos = save_pos + MakN_length - 6;
              fseek(ifp, curr_pos, SEEK_SET);

              fread(mbuf, 1, 4, ifp);
              curr_pos += 8;

              if (!strncmp(mbuf, "Pano ", 4))
              {
                parseAdobePanoMakernote();
              }

              if (!strncmp(mbuf, "RAF ", 4))
              { // Fujifilm Raw, AdobeRAF
                parseAdobeRAFMakernote();
              }

              if (!strncmp(mbuf, "SR2 ", 4))
              {
                order = 0x4d4d;
                MakN_length = get4();
                MakN_order = get2();
                pos_in_original_raw = get4();
                order = MakN_order;

                unsigned *buf_SR2;
                unsigned SR2SubIFDOffset = 0;
                unsigned SR2SubIFDLength = 0;
                unsigned SR2SubIFDKey = 0;
                {
                  int _base = curr_pos + 6 - pos_in_original_raw;
                  unsigned _entries, _tag, _type, _len, _save;
                  _entries = get2();
                  while (_entries--)
                  {
                    tiff_get(_base, &_tag, &_type, &_len, &_save);

                    if (_tag == 0x7200)
                    {
                      SR2SubIFDOffset = get4();
                    }
                    else if (_tag == 0x7201)
                    {
                      SR2SubIFDLength = get4();
                    }
                    else if (_tag == 0x7221)
                    {
                      SR2SubIFDKey = get4();
                    }
                    fseek(ifp, _save, SEEK_SET);
                  }
                }

                if (SR2SubIFDLength && (SR2SubIFDLength < 10240000) &&
                    (buf_SR2 = (unsigned *)malloc(SR2SubIFDLength + 1024)))
                { // 1024b for safety
                  fseek(ifp, SR2SubIFDOffset + base, SEEK_SET);
                  fread(buf_SR2, SR2SubIFDLength, 1, ifp);
                  sony_decrypt(buf_SR2, SR2SubIFDLength / 4, 1, SR2SubIFDKey);
                  parseSonySR2((uchar *)buf_SR2, SR2SubIFDOffset,
                               SR2SubIFDLength, AdobeDNG);

                  free(buf_SR2);
                }

              } /* SR2 processed */
              break;
            }
          }
        }
        else
        {
          fread(mbuf + 6, 1, 2, ifp);
          if (!strcmp(mbuf, "RICOH") && ((sget2((uchar *)mbuf + 6) == 0x4949) ||
                                         (sget2((uchar *)mbuf + 6) == 0x4d4d)))
          {
            is_PentaxRicohMakernotes = 1;
          }
          if (!strcmp(mbuf, "PENTAX ") || !strcmp(mbuf, "SAMSUNG") ||
              is_PentaxRicohMakernotes)
          {
            fseek(ifp, start_pos, SEEK_SET);
            parse_makernote_0xc634(base, 0, CameraDNG);
          }
        }
        fseek(ifp, start_pos, SEEK_SET);
        order = m_sorder;
      }
      if (dng_version)
      {
        break;
      }
      parse_minolta(j = get4() + base);
      fseek(ifp, j, SEEK_SET);
      parse_tiff_ifd(base);
      break;
    case 0xc640: // 50752
      read_shorts(cr2_slice, 3);
      break;
    case 0xc68b: /* 50827, OriginalRawFileName */
      stmread(imgdata.color.OriginalRawFileName, len, ifp);
      break;
    case 0xc68d: /* 50829 ActiveArea */
      tiff_ifd[ifd].t_tm = top_margin = getint(type);
      tiff_ifd[ifd].t_lm = left_margin = getint(type);
      tiff_ifd[ifd].t_vheight = height = getint(type) - top_margin;
      tiff_ifd[ifd].t_vwidth = width = getint(type) - left_margin;
      break;
    case 0xc68e: /* 50830 MaskedAreas */
      for (i = 0; i < (int)len && i < 32; i++)
        ((int *)mask)[i] = getint(type);
      black = 0;
      break;
    case 0xc71a: /* 50970, PreviewColorSpace */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_PREVIEWCS;
      tiff_ifd[ifd].dng_levels.preview_colorspace = getint(type);
      break;
    case 0xc740: /* 51008, OpcodeList1 */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_OPCODE1;
      break;
    case 0xc741: /* 51009, OpcodeList2 */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_OPCODE2;
      tiff_ifd[ifd].opcode2_offset = meta_offset = ftell(ifp);
      break;
    case 0xc74e: /* 51022, OpcodeList3 */
      tiff_ifd[ifd].dng_levels.parsedfields |= LIBRAW_DNGFM_OPCODE3;
      break;
    case 0xfd04: /* 64772, Kodak P-series */
      if (len < 13)
        break;
      fseek(ifp, 16, SEEK_CUR);
      data_offset = get4();
      fseek(ifp, 28, SEEK_CUR);
      data_offset += get4();
      load_raw = &LibRaw::packed_load_raw;
      break;
    case 0xfe02: // 65026
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_ASCII))
        fgets(model2, 64, ifp);
    }
    fseek(ifp, save, SEEK_SET);
  }
  if (sony_length && sony_length < 10240000 &&
      (buf = (unsigned *)malloc(sony_length)))
  {
    fseek(ifp, sony_offset, SEEK_SET);
    fread(buf, sony_length, 1, ifp);
    sony_decrypt(buf, sony_length / 4, 1, sony_key);
    parseSonySR2((uchar *)buf, sony_offset, sony_length, nonDNG);
    free(buf);
  }
  for (i = 0; i < colors && i < 4; i++)
    FORCC cc[i][c] *= ab[i];
  if (use_cm)
  {
    FORCC for (i = 0; i < 3; i++) for (cam_xyz[c][i] = j = 0; j < colors; j++)
        cam_xyz[c][i] += cc[c][j] * cm[j][i] * xyz[i];
    cam_xyz_coeff(cmatrix, cam_xyz);
  }
  if (asn[0])
  {
    cam_mul[3] = 0;
    FORCC
    if (fabs(asn[c]) > 0.0001)
      cam_mul[c] = 1 / asn[c];
  }
  if (!use_cm)
    FORCC if (fabs(cc[c][c]) > 0.0001) pre_mul[c] /= cc[c][c];
  return 0;
}

int LibRaw::parse_tiff(int _base)
{
  INT64 base = _base;
  int doff;
  fseek(ifp, base, SEEK_SET);
  order = get2();
  if (order != 0x4949 && order != 0x4d4d)
    return 0;
  get2();
  while ((doff = get4()))
  {
	INT64 doff64 = doff;
	if (doff64 + base > ifp->size()) break;
    fseek(ifp, doff64 + base, SEEK_SET);
    if (parse_tiff_ifd(_base))
      break;
  }
  return 1;
}

struct ifd_size_t
{
  int ifdi;
  INT64 databits;
};

int ifd_size_t_cmp(const void *a, const void *b)
{
  if (!a || !b)
    return 0;
  const ifd_size_t *ai = (ifd_size_t *)a;
  const ifd_size_t *bi = (ifd_size_t *)b;
  return bi->databits > ai->databits ? 1
                                     : (bi->databits < ai->databits ? -1 : 0);
}

static LibRaw_internal_thumbnail_formats tiff2thumbformat(int _comp, int _phint, int _bps, const char *_make);

void LibRaw::apply_tiff()
{
  int max_samp = 0, ties = 0, raw = -1, thm = -1, i;
  unsigned long long ns, os;
  struct jhead jh;

  thumb_misc = 16;
  if (thumb_offset)
  {
    fseek(ifp, thumb_offset, SEEK_SET);
    if (ljpeg_start(&jh, 1))
    {
      if ((unsigned)jh.bits < 17 && (unsigned)jh.wide < 0x10000 &&
          (unsigned)jh.high < 0x10000)
      {
        thumb_misc = jh.bits;
        thumb_width = jh.wide;
        thumb_height = jh.high;
      }
    }
  }
  for (i = tiff_nifds; i--;)
  {
    if (tiff_ifd[i].t_shutter)
      shutter = tiff_ifd[i].t_shutter;
    tiff_ifd[i].t_shutter = shutter;
  }

  if (dng_version)
  {
    int ifdc = 0;
    for (i = 0; i < (int)tiff_nifds; i++)
    {
      if (tiff_ifd[i].t_width < 1 || tiff_ifd[i].t_width > 65535 ||
          tiff_ifd[i].t_height < 1 || tiff_ifd[i].t_height > 65535)
        continue; /* wrong image dimensions */

      int samp = tiff_ifd[i].samples;
      if (samp == 2)
        samp = 1; // Fuji 2-frame
      max_samp = LIM(MAX(max_samp, samp), 1,
                     3); // max_samp is needed for thumbnail selection below

     if ( // Check phint only for RAW subfiletype
         (tiff_ifd[i].newsubfiletype == 16
             || tiff_ifd[i].newsubfiletype == 0
             || (tiff_ifd[i].newsubfiletype & 0xffff) == 1)
         &&
          (tiff_ifd[i].phint != 32803 && tiff_ifd[i].phint != 34892)
         )
              continue;

      if ((tiff_ifd[i].newsubfiletype == 0) // main image
                                            // Enhanced demosaiced:
          || (tiff_ifd[i].newsubfiletype == 16 &&
              (imgdata.rawparams.options & LIBRAW_RAWOPTIONS_DNG_ADD_ENHANCED))
          // Preview: 0x1 or 0x10001
          || ((tiff_ifd[i].newsubfiletype & 0xffff) == 1 &&
              (imgdata.rawparams.options & LIBRAW_RAWOPTIONS_DNG_ADD_PREVIEWS))
        // Transparency mask: 0x4 
          || ((tiff_ifd[i].newsubfiletype & 0xffff) == 4 &&
          (imgdata.rawparams.options & LIBRAW_RAWOPTIONS_DNG_ADD_MASKS)))
      {
        // Add this IFD to dng_frames
        libraw_internal_data.unpacker_data.dng_frames[ifdc] =
            ((tiff_ifd[i].newsubfiletype & 0xffff) << 16) | ((i << 8) & 0xff00);
        ifdc++;
        // Fuji SuperCCD: second frame:
        if ((tiff_ifd[i].newsubfiletype == 0) && tiff_ifd[i].samples == 2)
        {
          libraw_internal_data.unpacker_data.dng_frames[ifdc] =
              ((tiff_ifd[i].newsubfiletype & 0xffff) << 16) |
              ((i << 8) & 0xff00) | 1;
          ifdc++;
        }
      }
    }
    if (ifdc)
    {
      if (ifdc > 1 && (imgdata.rawparams.options & LIBRAW_RAWOPTIONS_DNG_PREFER_LARGEST_IMAGE))
      {
        ifd_size_t arr[LIBRAW_IFD_MAXCOUNT * 2];
        memset(arr, 0, sizeof(arr));
        for (int q = 0; q < ifdc && q < LIBRAW_IFD_MAXCOUNT * 2; q++)
        {
          int ifdidx =
              (libraw_internal_data.unpacker_data.dng_frames[q] >> 8) & 0xff;
          arr[q].ifdi = libraw_internal_data.unpacker_data.dng_frames[q];
          arr[q].databits =
              tiff_ifd[ifdidx].t_width * tiff_ifd[ifdidx].t_height *
                  tiff_ifd[ifdidx].samples * tiff_ifd[ifdidx].bps +
              (0x100 -
               (arr[q].ifdi & 0xff)); // add inverted frame # to ensure same
                                      // sort order for similar sized frames.
          if (tiff_ifd[ifdidx].phint == 4)
              arr[q].databits /= 4; // Force lower bit count for Transp. mask images 
        }
        qsort(arr, MIN(ifdc, LIBRAW_IFD_MAXCOUNT * 2), sizeof(arr[0]),
              ifd_size_t_cmp);
        for (int q = 0; q < ifdc && q < LIBRAW_IFD_MAXCOUNT * 2; q++)
          libraw_internal_data.unpacker_data.dng_frames[q] = arr[q].ifdi;
      }

      int idx = LIM((int)shot_select, 0, ifdc - 1);
      i = (libraw_internal_data.unpacker_data.dng_frames[idx] >> 8) &
          0xff; // extract frame# back

      raw_width = tiff_ifd[i].t_width;
      raw_height = tiff_ifd[i].t_height;
      tiff_bps = tiff_ifd[i].bps;
      tiff_compress = tiff_ifd[i].comp;
      tiff_sampleformat = tiff_ifd[i].sample_format;
      data_offset = tiff_ifd[i].offset;
      data_size = tiff_ifd[i].bytes;
      tiff_flip = tiff_ifd[i].t_flip;
      tiff_samples = tiff_ifd[i].samples;
      tile_width = tiff_ifd[i].t_tile_width;
      tile_length = tiff_ifd[i].t_tile_length;
      fuji_width = tiff_ifd[i].t_fuji_width;
      if (tiff_samples != 2) /* special case: Fuji SuperCCD */
      {
        if (tiff_ifd[i].phint == 34892)
          filters = 0;
        else if (i > 0 && tiff_ifd[i].phint == 32803 &&
                 tiff_ifd[0].phint == 32803 && !tiff_ifd[i].t_filters &&
                 tiff_ifd[0].t_filters)
          filters = tiff_ifd[0].t_filters;
        else
          filters = tiff_ifd[i].t_filters;
        width = tiff_ifd[i].t_vwidth;
        height = tiff_ifd[i].t_vheight;
        top_margin = tiff_ifd[i].t_tm;
        left_margin = tiff_ifd[i].t_lm;
        shutter = tiff_ifd[i].t_shutter;
        if (tiff_ifd[i].dng_levels.dng_whitelevel[0])
          maximum = tiff_ifd[i].dng_levels.dng_whitelevel[0];
        else if (tiff_ifd[i].sample_format <= 2 && tiff_bps > 0 &&
                 tiff_bps < 32) // SampleFormat: 0-default(1), 1 - Uint, 2 - Int
          maximum = (1 << tiff_bps) - 1;
        else if (tiff_ifd[i].sample_format == 3)
          maximum = 1; // Defaults for FP
      }
      raw = i;
      is_raw = ifdc;
    }
    else
      is_raw = 0;
  }
  else
  {
    // Fix for broken Sony bps tag
    if (!strncasecmp(make, "Sony", 4))
    {
        for (i = 0; i < (int)tiff_nifds; i++)
        {
            if (tiff_ifd[i].bps > 33 && tiff_ifd[i].samples == 1)
            {
                int bps = 14; // default
                if (tiff_ifd[i].dng_levels.dng_whitelevel[0] > 0)
                {
                    for(int c = 0,j=1; c < 16; c++, j<<=1)
                        if (j > (int)tiff_ifd[i].dng_levels.dng_whitelevel[0])
                        {
                            bps = c; break;
                        }
                }
                tiff_ifd[i].bps = bps;
            }
        }
    }

    for (i = 0; i < (int)tiff_nifds; i++)
    {
      if (tiff_ifd[i].t_width < 1 || tiff_ifd[i].t_width > 65535 ||
          tiff_ifd[i].t_height < 1 || tiff_ifd[i].t_height > 65535)
        continue; /* wrong image dimensions */
      if (max_samp < tiff_ifd[i].samples)
        max_samp = tiff_ifd[i].samples;
      if (max_samp > 3)
        max_samp = 3;

      os = unsigned(raw_width) * unsigned(raw_height);
      ns = unsigned(tiff_ifd[i].t_width) * unsigned(tiff_ifd[i].t_height);
      if (tiff_bps)
      {
        os *= tiff_bps;
        ns *= tiff_ifd[i].bps;
      }
      /* too complex if below, so separate if to skip RGB+Alpha TIFFs*/
      if (tiff_ifd[i].phint == 2 && tiff_ifd[i].extrasamples > 0 && tiff_ifd[i].samples > 3)
          continue; // SKIP RGB+Alpha IFDs

      if ((tiff_ifd[i].comp != 6 || tiff_ifd[i].samples != 3) &&
            unsigned(tiff_ifd[i].t_width | tiff_ifd[i].t_height) < 0x10000 &&
            (unsigned)tiff_ifd[i].bps < 33 &&
            (unsigned)tiff_ifd[i].samples < 13 && ns &&
            ((ns > os && (ties = 1)) || (ns == os && (int)shot_select == ties++)))
      {
        raw_width = tiff_ifd[i].t_width;
        raw_height = tiff_ifd[i].t_height;
        tiff_bps = tiff_ifd[i].bps;
        tiff_compress = tiff_ifd[i].comp;
        tiff_sampleformat = tiff_ifd[i].sample_format;
        data_offset = tiff_ifd[i].offset;
        data_size = tiff_ifd[i].bytes;
        tiff_flip = tiff_ifd[i].t_flip;
        tiff_samples = tiff_ifd[i].samples;
        tile_width = tiff_ifd[i].t_tile_width;
        tile_length = tiff_ifd[i].t_tile_length;
        shutter = tiff_ifd[i].t_shutter;
        raw = i;
      }
    }
    if (is_raw == 1 && ties)
      is_raw = ties;
  }
  if (is_NikonTransfer && raw >= 0)
  {
    if (tiff_ifd[raw].bps == 16)
    {
      if (tiff_compress == 1)
      {
        if ((raw_width * raw_height * 3) == (tiff_ifd[raw].bytes << 1))
        {
          tiff_bps = tiff_ifd[raw].bps = 12;
        }
        else
        {
          tiff_bps = tiff_ifd[raw].bps = 14;
        }
      }
    }
    else if (tiff_ifd[raw].bps == 8)
    {
      if (tiff_compress == 1)
      {
        is_NikonTransfer = 2; // 8-bit debayered TIFF, like CoolScan NEFs
        imgdata.rawparams.coolscan_nef_gamma = 2.2f;
      }
    }
  }

  if (!tile_width)
    tile_width = INT_MAX;
  if (!tile_length)
    tile_length = INT_MAX;
  for (i = tiff_nifds; i--;)
    if (tiff_ifd[i].t_flip)
      tiff_flip = tiff_ifd[i].t_flip;

#if 0
  if (raw < 0 && is_raw)
      is_raw = 0;
#endif

  if (raw >= 0 && !load_raw)
    switch (tiff_compress)
    {
    case 32767:
      if (!dng_version &&
          INT64(tiff_ifd[raw].bytes) == INT64(raw_width) * INT64(raw_height))
      {
        tiff_bps = 14;
        load_raw = &LibRaw::sony_arw2_load_raw;
        break;
      }
      if (!dng_version && !strncasecmp(make, "Sony", 4) &&
          INT64(tiff_ifd[raw].bytes) ==
              INT64(raw_width) * INT64(raw_height) * 2LL)
      {
        tiff_bps = 14;
        load_raw = &LibRaw::unpacked_load_raw;
        break;
      }
      if (INT64(tiff_ifd[raw].bytes) * 8LL !=
          INT64(raw_width) * INT64(raw_height) * INT64(tiff_bps))
      {
        raw_height += 8;
        load_raw = &LibRaw::sony_arw_load_raw;
        break;
      }
      load_flags = 79;
    case 32769:
      load_flags++;
    case 32770:
    case 32773:
      goto slr;
    case 0:
    case 1:
        if (dng_version && tiff_sampleformat == 3 &&
          (tiff_bps > 8 && (tiff_bps % 8 == 0) && (tiff_bps <= 32))) // only 16,24, and 32 are allowed
        {
            load_raw = &LibRaw::uncompressed_fp_dng_load_raw;
            break;
        }
      // Sony 14-bit uncompressed
      if (!dng_version && !strncasecmp(make, "Sony", 4) &&
          INT64(tiff_ifd[raw].bytes) ==
              INT64(raw_width) * INT64(raw_height) * 2LL)
      {
        tiff_bps = 14;
        load_raw = &LibRaw::unpacked_load_raw;
        break;
      }
      if (!dng_version && !strncasecmp(make, "Sony", 4) &&
          tiff_ifd[raw].samples == 4 &&
          INT64(tiff_ifd[raw].bytes) ==
              INT64(raw_width) * INT64(raw_height) * 8LL) // Sony ARQ
      {
        // maybe to detect ARQ with the following:
        // if (tiff_ifd[raw].phint == 32892)
        tiff_bps = 14;
        tiff_samples = 4;
        load_raw = &LibRaw::sony_arq_load_raw;
        filters = 0;
        strcpy(cdesc, "RGBG");
        break;
      }
      if (!strncasecmp(make, "Nikon", 5) &&
          (!strncmp(software, "Nikon Scan", 10) || (is_NikonTransfer == 2) ||
           strcasestr(model, "COOLSCAN")))
      {
        load_raw = &LibRaw::nikon_coolscan_load_raw;
        raw_color = 1;
        colors = (tiff_samples == 3) ? 3 : 1;
        filters = 0;
        break;
      }
      if ((!strncmp(make, "OLYMPUS", 7) || !strncmp(make, "OM Digi", 7) ||
           (!strncasecmp(make, "CLAUSS", 6) &&
            !strncasecmp(model, "piX 5oo", 7))) && // 0x5330303539 works here
          (INT64(tiff_ifd[raw].bytes) * 2ULL ==
           INT64(raw_width) * INT64(raw_height) * 3ULL))
        load_flags = 24;
      if (!dng_version && INT64(tiff_ifd[raw].bytes) * 5ULL ==
                              INT64(raw_width) * INT64(raw_height) * 8ULL)
      {
        load_flags = 81;
        tiff_bps = 12;
      }
    slr:
      switch (tiff_bps)
      {
      case 8:
        load_raw = &LibRaw::eight_bit_load_raw;
        break;
      case 12:
        if (tiff_ifd[raw].phint == 2)
          load_flags = 6;
        if (!strncasecmp(make, "NIKON", 5) &&
            !strncasecmp(model, "COOLPIX A1000", 13) &&
            data_size == raw_width * raw_height * 2u)
          load_raw = &LibRaw::unpacked_load_raw;
        else
          load_raw = &LibRaw::packed_load_raw;
        break;
      case 14:
        load_flags = 0;
      case 16:
        load_raw = &LibRaw::unpacked_load_raw;
        if ((!strncmp(make, "OLYMPUS", 7) || !strncmp(make, "OM Digi", 7) ||
             (!strncasecmp(make, "CLAUSS", 6) &&
              !strncasecmp(model, "piX 5oo", 7))) && // 0x5330303539 works here
            (INT64(tiff_ifd[raw].bytes) * 7LL >
             INT64(raw_width) * INT64(raw_height)))
          load_raw = &LibRaw::olympus_load_raw;
      }
      break;
    case 6:
    case 7:
    case 99:
      if (!dng_version && tiff_compress == 6 && !strcasecmp(make, "SONY"))
        load_raw = &LibRaw::sony_ljpeg_load_raw;
      else
        load_raw = &LibRaw::lossless_jpeg_load_raw;
      break;
    case 262:
      load_raw = &LibRaw::kodak_262_load_raw;
      break;
    case 34713:
      if ((INT64(raw_width) + 9LL) / 10LL * 16LL * INT64(raw_height) ==
          INT64(tiff_ifd[raw].bytes))
      {
        load_raw = &LibRaw::packed_load_raw;
        load_flags = 1;
      }
      else if (INT64(raw_width) * INT64(raw_height) * 3LL ==
               INT64(tiff_ifd[raw].bytes) * 2LL)
      {
        load_raw = &LibRaw::packed_load_raw;
        if (model[0] == 'N')
          load_flags = 80;
      }
      else if (INT64(raw_width) * INT64(raw_height) * 3LL ==
               INT64(tiff_ifd[raw].bytes))
      {
        load_raw = &LibRaw::nikon_yuv_load_raw;
        gamma_curve(1 / 2.4, 12.92, 1, 4095);
        memset(cblack, 0, sizeof cblack);
        filters = 0;
      }
      else if (INT64(raw_width) * INT64(raw_height) * 2LL ==
               INT64(tiff_ifd[raw].bytes))
      {
        load_raw = &LibRaw::unpacked_load_raw;
        load_flags = 4;
        order = 0x4d4d;
      }
#if 0 /* Never used because of same condition above, but need to recheck */
      else if (INT64(raw_width) * INT64(raw_height) * 3LL ==
               INT64(tiff_ifd[raw].bytes) * 2LL)
      {
        load_raw = &LibRaw::packed_load_raw;
        load_flags = 80;
      }
#endif
      else if (tiff_ifd[raw].rows_per_strip &&
               tiff_ifd[raw].strip_offsets_count &&
               tiff_ifd[raw].strip_offsets_count ==
                   tiff_ifd[raw].strip_byte_counts_count)
      {
        int fit = 1;
        for (int q = 0; q < tiff_ifd[raw].strip_byte_counts_count - 1;
             q++) // all but last
          if (INT64(tiff_ifd[raw].strip_byte_counts[q]) * 2LL !=
              INT64(tiff_ifd[raw].rows_per_strip) * INT64(raw_width) * 3LL)
          {
            fit = 0;
            break;
          }
        if (fit)
          load_raw = &LibRaw::nikon_load_striped_packed_raw;
        else
          load_raw = &LibRaw::nikon_load_raw; // fallback
      }
      else if ((((INT64(raw_width) * 3LL / 2LL) + 15LL) / 16LL) * 16LL *
                   INT64(raw_height) ==
               INT64(tiff_ifd[raw].bytes))
      {
        load_raw = &LibRaw::nikon_load_padded_packed_raw;
        load_flags = (((INT64(raw_width) * 3ULL / 2ULL) + 15ULL) / 16ULL) *
                     16ULL; // bytes per row
      }
      else if (!strncmp(model, "NIKON Z 9", 9) && tiff_ifd[raw].offset)
      {
          INT64 pos = ftell(ifp);
          unsigned char cmp[] = "CONTACT_INTOPIX"; // 15
          unsigned char buf[16];
          fseek(ifp, INT64(tiff_ifd[raw].offset) + 6LL, SEEK_SET);
          fread(buf, 1, 16, ifp);
          fseek(ifp, pos, SEEK_SET);
          if(!memcmp(buf,cmp,15))
            load_raw = &LibRaw::nikon_he_load_raw_placeholder;
          else
            load_raw = &LibRaw::nikon_load_raw;
      }
      else
        load_raw = &LibRaw::nikon_load_raw;
      break;
    case 65535:
      load_raw = &LibRaw::pentax_load_raw;
      break;
    case 65000:
      switch (tiff_ifd[raw].phint)
      {
      case 2:
        load_raw = &LibRaw::kodak_rgb_load_raw;
        filters = 0;
        break;
      case 6:
        load_raw = &LibRaw::kodak_ycbcr_load_raw;
        filters = 0;
        break;
      case 32803:
        load_raw = &LibRaw::kodak_65000_load_raw;
      }
    case 32867:
    case 34892:
      break;
    case 8:
      break;
#ifdef USE_GPRSDK
    case 9:
      if (dng_version)
        break; /* Compression=9 supported for dng if we compiled with GPR SDK */
               /* Else: fallthrough */
#endif
    default:
      is_raw = 0;
    }
  if (!dng_version)
  {
      if (((tiff_samples == 3 && tiff_ifd[raw].bytes &&
          !(tiff_bps == 16 &&
              !strncmp(make, "Leaf", 4)) && // Allow Leaf/16bit/3color files
          tiff_bps != 14 &&
          (tiff_compress & -16) != 32768) ||
          (tiff_bps == 8 && strncmp(make, "Phase", 5) &&
              strncmp(make, "Leaf", 4) && !strcasestr(make, "Kodak") &&
              !strstr(model2, "DEBUG RAW"))) &&
          !strcasestr(model, "COOLSCAN") && strncmp(software, "Nikon Scan", 10) &&
          is_NikonTransfer != 2)
          is_raw = 0;

      if (is_raw && raw >= 0 && tiff_ifd[raw].phint == 2 && tiff_ifd[raw].extrasamples > 0 && tiff_ifd[raw].samples > 3)
          is_raw = 0; // SKIP RGB+Alpha IFDs
  }

  INT64 fsizecheck = 0ULL;

  if (imgdata.rawparams.options & LIBRAW_RAWOPTIONS_CHECK_THUMBNAILS_ALL_VENDORS)
      fsizecheck = ifp->size();
  else if ((imgdata.rawparams.options & LIBRAW_RAWOPTIONS_CHECK_THUMBNAILS_KNOWN_VENDORS)
      && !strncasecmp(make,"Ricoh",5))
      fsizecheck = ifp->size();

  for (i = 0; i < (int)tiff_nifds; i++)
    if (i != raw &&
        (tiff_ifd[i].samples == max_samp ||
         (tiff_ifd[i].comp == 7 &&
          tiff_ifd[i].samples == 1)) /* Allow 1-bps JPEGs */
        && tiff_ifd[i].bps > 0 && tiff_ifd[i].bps < 33 &&
        tiff_ifd[i].phint != 32803 && tiff_ifd[i].phint != 34892 &&
        unsigned(tiff_ifd[i].t_width | tiff_ifd[i].t_height) < 0x10000 &&
        tiff_ifd[i].comp != 34892)
    {
        if (fsizecheck > 0LL)
        {
            bool ok = true;
            if (tiff_ifd[i].strip_byte_counts_count && tiff_ifd[i].strip_offsets_count)
                for (int s = 0; s < MIN(tiff_ifd[i].strip_byte_counts_count, tiff_ifd[i].strip_offsets_count); s++)
                {
                    if (tiff_ifd[i].strip_offsets[s] + tiff_ifd[i].strip_byte_counts[s] > fsizecheck)
                    {
                        ok = false;
                        break;
                    }
                }
            else if (tiff_ifd[i].bytes > 0)
                if (tiff_ifd[i].offset + tiff_ifd[i].bytes > fsizecheck)
                    ok = false;

            if(!ok)
                continue;
        }
		if ( (INT64(tiff_ifd[i].t_width) * INT64(tiff_ifd[i].t_height) / INT64(SQR(tiff_ifd[i].bps) + 1)) >
			 (INT64(thumb_width) * INT64(thumb_height) / INT64(SQR(thumb_misc) + 1)) ) 
		{

			thumb_width = tiff_ifd[i].t_width;
			thumb_height = tiff_ifd[i].t_height;
			thumb_offset = tiff_ifd[i].offset;
			thumb_length = tiff_ifd[i].bytes;
			thumb_misc = tiff_ifd[i].bps;
			thm = i;
		}
		if (imgdata.thumbs_list.thumbcount < LIBRAW_THUMBNAIL_MAXCOUNT && tiff_ifd[i].bytes > 0)
		{
			bool already = false;
			for(int idx = 0; idx < imgdata.thumbs_list.thumbcount ; idx++)
				if (imgdata.thumbs_list.thumblist[idx].toffset == tiff_ifd[i].offset)
				{
					already = true;
					break;
				}
			if (!already)
			{
				int idx = imgdata.thumbs_list.thumbcount;
				imgdata.thumbs_list.thumblist[idx].tformat = tiff2thumbformat(tiff_ifd[i].comp, tiff_ifd[i].phint,
					tiff_ifd[i].bps, make);
				imgdata.thumbs_list.thumblist[idx].twidth = tiff_ifd[i].t_width;
				imgdata.thumbs_list.thumblist[idx].theight = tiff_ifd[i].t_height;
				imgdata.thumbs_list.thumblist[idx].tflip = tiff_ifd[i].t_flip;
				imgdata.thumbs_list.thumblist[idx].tlength = tiff_ifd[i].bytes;
				imgdata.thumbs_list.thumblist[idx].tmisc = tiff_ifd[i].bps | (tiff_ifd[i].samples << 5);
				imgdata.thumbs_list.thumblist[idx].toffset = tiff_ifd[i].offset;
				imgdata.thumbs_list.thumbcount++;
			}
		}
    }
  if (thm >= 0)
  {
    thumb_misc |= tiff_ifd[thm].samples << 5;
	thumb_format = tiff2thumbformat(tiff_ifd[thm].comp, tiff_ifd[thm].phint, tiff_ifd[thm].bps, make);
  }
}

static LibRaw_internal_thumbnail_formats tiff2thumbformat(int _comp, int _phint, int _bps, const char *_make)
{
  switch (_comp)
  {
  case 0:
    return LIBRAW_INTERNAL_THUMBNAIL_LAYER;
  case 1:
    if (_bps <= 8)
      return LIBRAW_INTERNAL_THUMBNAIL_PPM;
    else if (!strncmp(_make, "Imacon", 6))
      return LIBRAW_INTERNAL_THUMBNAIL_PPM16;
    else
      return LIBRAW_INTERNAL_THUMBNAIL_KODAK_THUMB;
  case 65000:
    return _phint == 6 ? LIBRAW_INTERNAL_THUMBNAIL_KODAK_YCBCR : LIBRAW_INTERNAL_THUMBNAIL_KODAK_RGB;
  }
  return LIBRAW_INTERNAL_THUMBNAIL_JPEG; // default
}
