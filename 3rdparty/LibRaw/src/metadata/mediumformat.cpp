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

void LibRaw::parse_phase_one(int base)
{
  unsigned entries, tag, type, len, data, i, c;
  INT64 save;
  float romm_cam[3][3];
  char *cp;

  memset(&ph1, 0, sizeof ph1);
  fseek(ifp, base, SEEK_SET);
  order = get4() & 0xffff;
  if (get4() >> 8 != 0x526177)
    return; /* "Raw" */
  unsigned offset = get4();
  if (offset == 0xbad0bad)
    return;
  fseek(ifp, offset + base, SEEK_SET);
  entries = get4();
  if (entries > 8192)
    return; // too much??
  get4();
  while (entries--)
  {
    tag = get4();
    type = get4();
    len = get4();
	if (feof(ifp))
		break;
    data = get4();
    save = ftell(ifp);
	bool do_seek = (tag < 0x0108 || tag > 0x0110); // to make it single rule, not copy-paste
	if(do_seek)
		fseek(ifp, base + data, SEEK_SET);
    switch (tag)
    {

    case 0x0100:
      flip = "0653"[data & 3] - '0';
      break;
    case 0x0102:
      stmread(imgdata.shootinginfo.BodySerial, len, ifp);
      if ((imgdata.shootinginfo.BodySerial[0] == 0x4c) && (imgdata.shootinginfo.BodySerial[1] == 0x49))
      {
        unique_id =
            (((imgdata.shootinginfo.BodySerial[0] & 0x3f) << 5) | (imgdata.shootinginfo.BodySerial[2] & 0x3f)) - 0x41;
      }
      else
      {
        unique_id =
            (((imgdata.shootinginfo.BodySerial[0] & 0x3f) << 5) | (imgdata.shootinginfo.BodySerial[1] & 0x3f)) - 0x41;
      }
      setPhaseOneFeatures(unique_id);
      break;
    case 0x0106:
      for (i = 0; i < 9; i++)
        imgdata.color.P1_color[0].romm_cam[i] = ((float *)romm_cam)[i] =
            (float)getreal(LIBRAW_EXIFTAG_TYPE_FLOAT);
      romm_coeff(romm_cam);
      break;
    case 0x0107:
      FORC3 cam_mul[c] = (float)getreal(LIBRAW_EXIFTAG_TYPE_FLOAT);
      break;
    case 0x0108:
      raw_width = data;
      break;
    case 0x0109:
      raw_height = data;
      break;
    case 0x010a:
      left_margin = data;
      break;
    case 0x010b:
      top_margin = data;
      break;
    case 0x010c:
      width = data;
      break;
    case 0x010d:
      height = data;
      break;
    case 0x010e:
      ph1.format = data;
      break;
    case 0x010f:
      data_offset = data + base;
	  data_size = len;
      break;
    case 0x0110:
      meta_offset = data + base;
      meta_length = len;
      break;
    case 0x0112:
      ph1.key_off = int(save - 4);
      break;
    case 0x0203:
      stmread(imPhaseOne.Software, len, ifp);
    case 0x0204:
      stmread(imPhaseOne.SystemType, len, ifp);
    case 0x0210:
      ph1.tag_210 = int_to_float(data);
      imCommon.SensorTemperature = ph1.tag_210;
      break;
    case 0x0211:
      imCommon.SensorTemperature2 = int_to_float(data);
      break;
    case 0x021a:
      ph1.tag_21a = data;
      break;
    case 0x021c:
      strip_offset = data + base;
      break;
    case 0x021d:
      ph1.t_black = data;
      break;
    case 0x0222:
      ph1.split_col = data;
      break;
    case 0x0223:
      ph1.black_col = data + base;
      break;
    case 0x0224:
      ph1.split_row = data;
      break;
    case 0x0225:
      ph1.black_row = data + base;
      break;
    case 0x0226:
      for (i = 0; i < 9; i++)
        imgdata.color.P1_color[1].romm_cam[i] = (float)getreal(LIBRAW_EXIFTAG_TYPE_FLOAT);
      break;
    case 0x0301:
      model[63] = 0;
      fread(imPhaseOne.FirmwareString, 1, 255, ifp);
      imPhaseOne.FirmwareString[255] = 0;
      memcpy(model, imPhaseOne.FirmwareString, 63);
	  model[63] = 0;
      if ((cp = strstr(model, " camera")))
        *cp = 0;
      else if ((cp = strchr(model, ',')))
        *cp = 0;
      /* minus and the letter after it are not always present
        if present, last letter means:
          C : Contax 645AF
          H : Hasselblad H1 / H2
          M : Mamiya
          V : Hasselblad 555ELD / 553ELX / 503CW / 501CM; not included below
        because of adapter conflicts (Mamiya RZ body) if not present, Phase One
        645 AF, Mamiya 645AFD Series, or anything
       */
      strcpy(imPhaseOne.SystemModel, model);
      if ((cp = strchr(model, '-')))
      {
        if (cp[1] == 'C')
        {
          strcpy(ilm.body, "Contax 645AF");
          ilm.CameraMount = LIBRAW_MOUNT_Contax645;
          ilm.CameraFormat = LIBRAW_FORMAT_645;
        }
        else if (cp[1] == 'M')
        {
          strcpy(ilm.body, "Mamiya 645");
          ilm.CameraMount = LIBRAW_MOUNT_Mamiya645;
          ilm.CameraFormat = LIBRAW_FORMAT_645;
        }
        else if (cp[1] == 'H')
        {
          strcpy(ilm.body, "Hasselblad H1/H2");
          ilm.CameraMount = LIBRAW_MOUNT_Hasselblad_H;
          ilm.CameraFormat = LIBRAW_FORMAT_645;
        }
        *cp = 0;
      }
    case 0x0401:
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
        ilm.CurAp = libraw_powf64l(2.0f, (int_to_float(data) / 2.0f));
      else
        ilm.CurAp = libraw_powf64l(2.0f, float(getreal(type) / 2.0f));
      break;
    case 0x0403:
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
        ilm.CurFocal = int_to_float(data);
      else
        ilm.CurFocal = (float)getreal(type);
      break;
    case 0x0410:
      stmread(ilm.body, len, ifp);
      if (((unsigned char)ilm.body[0]) == 0xff)
        ilm.body[0] = 0;
      break;
    case 0x0412:
      stmread(ilm.Lens, len, ifp);
      if (((unsigned char)ilm.Lens[0]) == 0xff)
        ilm.Lens[0] = 0;
      break;
    case 0x0414:
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
      {
        ilm.MaxAp4CurFocal = libraw_powf64l(2.0f, (int_to_float(data) / 2.0f));
      }
      else
      {
        ilm.MaxAp4CurFocal = libraw_powf64l(2.0f, float(getreal(type) / 2.0f));
      }
      break;
    case 0x0415:
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
      {
        ilm.MinAp4CurFocal = libraw_powf64l(2.0f, (int_to_float(data) / 2.0f));
      }
      else
      {
        ilm.MinAp4CurFocal = libraw_powf64l(2.0f, float(getreal(type) / 2.0f));
      }
      break;
    case 0x0416:
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
      {
        ilm.MinFocal = int_to_float(data);
      }
      else
      {
        ilm.MinFocal = (float)getreal(type);
      }
      if (ilm.MinFocal > 1000.0f)
      {
        ilm.MinFocal = 0.0f;
      }
      break;
    case 0x0417:
      if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_LONG))
      {
        ilm.MaxFocal = int_to_float(data);
      }
      else
      {
        ilm.MaxFocal = (float)getreal(type);
      }
      break;
    }
    if (do_seek)
      fseek(ifp, save, SEEK_SET);
  }

  if (!ilm.body[0] && !imgdata.shootinginfo.BodySerial[0])
  {
    fseek(ifp, meta_offset, SEEK_SET);
    order = get2();
    fseek(ifp, 6, SEEK_CUR);
    fseek(ifp, meta_offset + get4(), SEEK_SET);
    entries = get4();
    if (entries > 8192)
      return; // too much??
    get4();
    while (entries--)
    {
      tag = get4();
      len = get4();
	  if (feof(ifp))
		  break;
      data = get4();
      save = ftell(ifp);
      fseek(ifp, meta_offset + data, SEEK_SET);
      if (tag == 0x0407)
      {
        stmread(imgdata.shootinginfo.BodySerial, len, ifp);
        if ((imgdata.shootinginfo.BodySerial[0] == 0x4c) &&
            (imgdata.shootinginfo.BodySerial[1] == 0x49))
        {
          unique_id = (((imgdata.shootinginfo.BodySerial[0] & 0x3f) << 5) |
                       (imgdata.shootinginfo.BodySerial[2] & 0x3f)) -
                      0x41;
        }
        else
        {
          unique_id = (((imgdata.shootinginfo.BodySerial[0] & 0x3f) << 5) |
                       (imgdata.shootinginfo.BodySerial[1] & 0x3f)) -
                      0x41;
        }
        setPhaseOneFeatures(unique_id);
      }
      fseek(ifp, save, SEEK_SET);
    }
  }

  if ((ilm.MaxAp4CurFocal > 0.7f) &&
      (ilm.MinAp4CurFocal > 0.7f)) {
    float MinAp4CurFocal = MAX(ilm.MaxAp4CurFocal,ilm.MinAp4CurFocal);
    ilm.MaxAp4CurFocal   = MIN(ilm.MaxAp4CurFocal,ilm.MinAp4CurFocal);
    ilm.MinAp4CurFocal = MinAp4CurFocal;
  }

  if (ph1.format == 6)
	  load_raw = &LibRaw::phase_one_load_raw_s;
  else
    load_raw = ph1.format < 3 ? &LibRaw::phase_one_load_raw : &LibRaw::phase_one_load_raw_c;
  maximum = 0xffff; // Always scaled to 16bit?
  strcpy(make, "Phase One");
  if (model[0])
    return;
  switch (raw_height)
  {
  case 2060:
    strcpy(model, "LightPhase");
    break;
  case 2682:
    strcpy(model, "H 10");
    break;
  case 4128:
    strcpy(model, "H 20");
    break;
  case 5488:
    strcpy(model, "H 25");
    break;
  }
}

void LibRaw::parse_mos(INT64 offset)
{
  char data[40];
  int i, c, neut[4], planes = 0, frot = 0;
  INT64 from;
  unsigned skip;
  static const char *mod[] = {
      /* DM22, DM28, DM40, DM56 are somewhere here too */
      "",             //  0
      "DCB2",         //  1
      "Volare",       //  2
      "Cantare",      //  3
      "CMost",        //  4
      "Valeo 6",      //  5
      "Valeo 11",     //  6
      "Valeo 22",     //  7
      "Valeo 11p",    //  8
      "Valeo 17",     //  9
      "",             // 10
      "Aptus 17",     // 11
      "Aptus 22",     // 12
      "Aptus 75",     // 13
      "Aptus 65",     // 14
      "Aptus 54S",    // 15
      "Aptus 65S",    // 16
      "Aptus 75S",    // 17
      "AFi 5",        // 18
      "AFi 6",        // 19
      "AFi 7",        // 20
      "AFi-II 7",     // 21
      "Aptus-II 7",   // 22 (same CMs as Mamiya DM33)
      "",             // 23
      "Aptus-II 6",   // 24 (same CMs as Mamiya DM28)
      "AFi-II 10",    // 25
      "",             // 26
      "Aptus-II 10",  // 27 (same CMs as Mamiya DM56)
      "Aptus-II 5",   // 28 (same CMs as Mamiya DM22)
      "",             // 29
      "DM33",         // 30, make is Mamiya
      "",             // 31
      "",             // 32
      "Aptus-II 10R", // 33
      "Aptus-II 8",   // 34 (same CMs as Mamiya DM40)
      "",             // 35
      "Aptus-II 12",  // 36
      "",             // 37
      "AFi-II 12"     // 38
  };
  float romm_cam[3][3];

  fseek(ifp, offset, SEEK_SET);
  while (!feof(ifp))
  {
    if (get4() != 0x504b5453)
      break;
    get4();
    fread(data, 1, 40, ifp);
    skip = get4();
    from = ftell(ifp);

    if (!strcmp(data, "CameraObj_camera_type"))
    {
      stmread(ilm.body, (unsigned)skip, ifp);
      if (ilm.body[0])
      {
        if (!strncmp(ilm.body, "Mamiya R", 8))
        {
          ilm.CameraMount = LIBRAW_MOUNT_Mamiya67;
          ilm.CameraFormat = LIBRAW_FORMAT_67;
        }
        else if (!strncmp(ilm.body, "Hasselblad 5", 12))
        {
          ilm.CameraFormat = LIBRAW_FORMAT_66;
          ilm.CameraMount = LIBRAW_MOUNT_Hasselblad_V;
        }
        else if (!strncmp(ilm.body, "Hasselblad H", 12))
        {
          ilm.CameraMount = LIBRAW_MOUNT_Hasselblad_H;
          ilm.CameraFormat = LIBRAW_FORMAT_645;
        }
        else if (!strncmp(ilm.body, "Mamiya 6", 8) ||
                 !strncmp(ilm.body, "Phase One 6", 11))
        {
          ilm.CameraMount = LIBRAW_MOUNT_Mamiya645;
          ilm.CameraFormat = LIBRAW_FORMAT_645;
        }
        else if (!strncmp(ilm.body, "Large F", 7))
        {
          ilm.CameraMount = LIBRAW_MOUNT_LF;
          ilm.CameraFormat = LIBRAW_FORMAT_LF;
        }
        else if (!strncmp(model, "Leaf AFi", 8))
        {
          ilm.CameraMount = LIBRAW_MOUNT_Rollei_bayonet;
          ilm.CameraFormat = LIBRAW_FORMAT_66;
        }
      }
    }
    if (!strcmp(data, "back_serial_number"))
    {
      char buffer[sizeof(imgdata.shootinginfo.BodySerial)];
      char *words[4] = {0, 0, 0, 0};
      stmread(buffer, (unsigned)skip, ifp);
      /*nwords = */
          getwords(buffer, words, 4, sizeof(imgdata.shootinginfo.BodySerial));
	  if(words[0])
		strcpy(imgdata.shootinginfo.BodySerial, words[0]);
    }
    if (!strcmp(data, "CaptProf_serial_number"))
    {
      char buffer[sizeof(imgdata.shootinginfo.InternalBodySerial)];
      char *words[4] = {0, 0, 0, 0};
      stmread(buffer, (unsigned)skip, ifp);
      getwords(buffer, words, 4, sizeof(imgdata.shootinginfo.InternalBodySerial));
	  if(words[0])
		strcpy(imgdata.shootinginfo.InternalBodySerial, words[0]);
    }

    if (!strcmp(data, "JPEG_preview_data"))
    {
      thumb_offset = from;
      thumb_length = skip;
    }
    if (!strcmp(data, "icc_camera_profile"))
    {
      profile_offset = from;
      profile_length = skip;
    }
    if (!strcmp(data, "ShootObj_back_type"))
    {
      fscanf(ifp, "%d", &i);
      if ((unsigned)i < sizeof mod / sizeof(*mod))
      {
        strcpy(model, mod[i]);
        if (!strncmp(model, "AFi", 3))
        {
          ilm.CameraMount = LIBRAW_MOUNT_Rollei_bayonet;
          ilm.CameraFormat = LIBRAW_FORMAT_66;
        }
        ilm.CamID = i;
      }
    }
    if (!strcmp(data, "icc_camera_to_tone_matrix"))
    {
      for (i = 0; i < 9; i++)
        ((float *)romm_cam)[i] = int_to_float(get4());
      romm_coeff(romm_cam);
    }
    if (!strcmp(data, "CaptProf_color_matrix"))
    {
      for (i = 0; i < 9; i++)
        fscanf(ifp, "%f", (float *)romm_cam + i);
      romm_coeff(romm_cam);
    }
    if (!strcmp(data, "CaptProf_number_of_planes"))
      fscanf(ifp, "%d", &planes);
    if (!strcmp(data, "CaptProf_raw_data_rotation"))
      fscanf(ifp, "%d", &flip);
    if (!strcmp(data, "CaptProf_mosaic_pattern"))
      FORC4
      {
        fscanf(ifp, "%d", &i);
        if (i == 1)
          frot = c ^ (c >> 1); // 0123 -> 0132
      }
    if (!strcmp(data, "ImgProf_rotation_angle"))
    {
      fscanf(ifp, "%d", &i);
      flip = i - flip;
    }
    if (!strcmp(data, "NeutObj_neutrals") && !cam_mul[0])
    {
      FORC4 fscanf(ifp, "%d", neut + c);
      FORC3
      if (neut[c + 1])
        cam_mul[c] = (float)neut[0] / neut[c + 1];
    }
    if (!strcmp(data, "Rows_data"))
      load_flags = get4();
    parse_mos(from);
    fseek(ifp, skip + from, SEEK_SET);
  }
  if (planes)
    filters = (planes == 1) * 0x01010101U *
              (uchar) "\x94\x61\x16\x49"[(flip / 90 + frot) & 3];
}
