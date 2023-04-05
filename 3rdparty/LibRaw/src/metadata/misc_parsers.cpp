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

/*
   Returns 1 for a Coolpix 2100, 0 for anything else.
 */
int LibRaw::nikon_e2100()
{
  uchar t[12];
  int i;

  fseek(ifp, 0, SEEK_SET);
  for (i = 0; i < 1024; i++)
  {
    fread(t, 1, 12, ifp);
    if (((t[2] & t[4] & t[7] & t[9]) >> 4 & t[1] & t[6] & t[8] & t[11] & 3) !=
        3)
      return 0;
  }
  return 1;
}

void LibRaw::nikon_3700()
{
  int bits, i;
  uchar dp[24];
  static const struct
  {
    int bits;
    char t_make[12], t_model[15];
    int t_maker_idx;
  } table[] = {{0x00, "Pentax", "Optio 33WR", LIBRAW_CAMERAMAKER_Pentax},
               {0x03, "Nikon", "E3200", LIBRAW_CAMERAMAKER_Nikon},
               {0x32, "Nikon", "E3700", LIBRAW_CAMERAMAKER_Nikon},
               {0x33, "Olympus", "C-740UZ", LIBRAW_CAMERAMAKER_Olympus}};

  fseek(ifp, 3072, SEEK_SET);
  fread(dp, 1, 24, ifp);
  bits = (dp[8] & 3) << 4 | (dp[20] & 3);
  for (i = 0; i < int(sizeof table / sizeof *table); i++)
    if (bits == table[i].bits)
    {
      strcpy(make, table[i].t_make);
      maker_index = table[i].t_maker_idx;
      strcpy(model, table[i].t_model);
    }
}

/*
   Separates a Minolta DiMAGE Z2 from a Nikon E4300.
 */
int LibRaw::minolta_z2()
{
  int i, nz;
  char tail[424];

  fseek(ifp, -int(sizeof tail), SEEK_END);
  fread(tail, 1, sizeof tail, ifp);
  for (nz = i = 0; i < int(sizeof tail); i++)
    if (tail[i])
      nz++;
  return nz > 20;
}

int LibRaw::canon_s2is()
{
  unsigned row;

  for (row = 0; row < 100; row++)
  {
    fseek(ifp, row * 3340 + 3284, SEEK_SET);
    if (getc(ifp) > 15)
      return 1;
  }
  return 0;
}

#ifdef LIBRAW_OLD_VIDEO_SUPPORT
void LibRaw::parse_redcine()
{
  unsigned i, len, rdvo;

  order = 0x4d4d;
  is_raw = 0;
  fseek(ifp, 52, SEEK_SET);
  width = get4();
  height = get4();
  fseek(ifp, 0, SEEK_END);
  fseek(ifp, -(i = ftello(ifp) & 511), SEEK_CUR);
  if (get4() != i || get4() != 0x52454f42)
  {
    fseek(ifp, 0, SEEK_SET);
    while ((len = get4()) != (unsigned)EOF)
    {
      if (get4() == 0x52454456)
        if (is_raw++ == shot_select)
          data_offset = ftello(ifp) - 8;
      fseek(ifp, len - 8, SEEK_CUR);
    }
  }
  else
  {
    rdvo = get4();
    fseek(ifp, 12, SEEK_CUR);
    is_raw = get4();
    fseeko(ifp, rdvo + 8 + shot_select * 4, SEEK_SET);
    data_offset = get4();
  }
}
#endif

void LibRaw::parse_cine()
{
  unsigned off_head, off_setup, off_image, i, temp;

  order = 0x4949;
  fseek(ifp, 4, SEEK_SET);
  is_raw = get2() == 2;
  fseek(ifp, 14, SEEK_CUR);
  is_raw *= get4();
  off_head = get4();
  off_setup = get4();
  off_image = get4();
  timestamp = get4();
  if ((i = get4()))
    timestamp = i;
  fseek(ifp, off_head + 4, SEEK_SET);
  raw_width = get4();
  raw_height = get4();
  switch (get2(), get2())
  {
  case 8:
    load_raw = &LibRaw::eight_bit_load_raw;
    break;
  case 16:
    load_raw = &LibRaw::unpacked_load_raw;
  }
  fseek(ifp, off_setup + 792, SEEK_SET);
  strcpy(make, "CINE");
  sprintf(model, "%d", get4());
  fseek(ifp, 12, SEEK_CUR);
  switch ((i = get4()) & 0xffffff)
  {
  case 3:
    filters = 0x94949494;
    break;
  case 4:
    filters = 0x49494949;
    break;
  default:
    is_raw = 0;
  }
  fseek(ifp, 72, SEEK_CUR);
  switch ((get4() + 3600) % 360)
  {
  case 270:
    flip = 4;
    break;
  case 180:
    flip = 1;
    break;
  case 90:
    flip = 7;
    break;
  case 0:
    flip = 2;
  }
  cam_mul[0] = getreal(LIBRAW_EXIFTAG_TYPE_FLOAT);
  cam_mul[2] = getreal(LIBRAW_EXIFTAG_TYPE_FLOAT);
  temp = get4();
  maximum = ~((~0u) << LIM(temp, 1, 31));
  fseek(ifp, 668, SEEK_CUR);
  shutter = get4() / 1000000000.0;
  fseek(ifp, off_image, SEEK_SET);
  if (shot_select < is_raw)
    fseek(ifp, shot_select * 8, SEEK_CUR);
  data_offset = (INT64)get4() + 8;
  data_offset += (INT64)get4() << 32;
}

void LibRaw::parse_qt(int end)
{
  unsigned save, size;
  char tag[4];

  order = 0x4d4d;
  while (ftell(ifp) + 7 < end)
  {
    save = ftell(ifp);
    if ((size = get4()) < 8)
      return;
    if ((int)size < 0)
      return; // 2+GB is too much
    if (save + size < save)
      return; // 32bit overflow
    fread(tag, 4, 1, ifp);
    if (!memcmp(tag, "moov", 4) || !memcmp(tag, "udta", 4) ||
        !memcmp(tag, "CNTH", 4))
      parse_qt(save + size);
    if (!memcmp(tag, "CNDA", 4))
      parse_jpeg(ftell(ifp));
    fseek(ifp, save + size, SEEK_SET);
  }
}

void LibRaw::parse_smal(int offset, int fsize)
{
  int ver;

  fseek(ifp, offset + 2, SEEK_SET);
  order = 0x4949;
  ver = fgetc(ifp);
  if (ver == 6)
    fseek(ifp, 5, SEEK_CUR);
  if (get4() != (unsigned)fsize)
    return;
  if (ver > 6)
    data_offset = get4();
  raw_height = height = get2();
  raw_width = width = get2();
  strcpy(make, "SMaL");
  sprintf(model, "v%d %dx%d", ver, width, height);
  if (ver == 6)
    load_raw = &LibRaw::smal_v6_load_raw;
  if (ver == 9)
    load_raw = &LibRaw::smal_v9_load_raw;
}

void LibRaw::parse_riff(int maxdepth)
{
  unsigned i, size, end;
  char tag[4], date[64], month[64];
  static const char mon[12][4] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  struct tm t;
  if (maxdepth < 1)
	  throw LIBRAW_EXCEPTION_IO_CORRUPT;

  order = 0x4949;
  fread(tag, 4, 1, ifp);
  size = get4();
  end = ftell(ifp) + size;
  if (!memcmp(tag, "RIFF", 4) || !memcmp(tag, "LIST", 4))
  {
    int maxloop = 1000;
    get4();
    while (ftell(ifp) + 7 < end && !feof(ifp) && maxloop--)
      parse_riff(maxdepth-1);
  }
  else if (!memcmp(tag, "nctg", 4))
  {
    while (ftell(ifp) + 7 < end)
    {
		if (feof(ifp))
			break;
      i = get2();
      size = get2();
      if ((i + 1) >> 1 == 10 && size == 20)
        get_timestamp(0);
      else
        fseek(ifp, size, SEEK_CUR);
    }
  }
  else if (!memcmp(tag, "IDIT", 4) && size < 64)
  {
    fread(date, 64, 1, ifp);
    date[size] = 0;
    memset(&t, 0, sizeof t);
    if (sscanf(date, "%*s %s %d %d:%d:%d %d", month, &t.tm_mday, &t.tm_hour,
               &t.tm_min, &t.tm_sec, &t.tm_year) == 6)
    {
      for (i = 0; i < 12 && strcasecmp(mon[i], month); i++)
        ;
      t.tm_mon = i;
      t.tm_year -= 1900;
      if (mktime(&t) > 0)
        timestamp = mktime(&t);
    }
  }
  else
    fseek(ifp, size, SEEK_CUR);
}

void LibRaw::parse_rollei()
{
  char line[128], *val;
  struct tm t;

  fseek(ifp, 0, SEEK_SET);
  memset(&t, 0, sizeof t);
  do
  {
    line[0] = 0;
    if (!fgets(line, 128, ifp))
      break;
    line[127] = 0;
    if(!line[0]) break; // zero-length
    if ((val = strchr(line, '=')))
      *val++ = 0;
    else
      val = line + strbuflen(line);
    if (!strcmp(line, "DAT"))
      sscanf(val, "%d.%d.%d", &t.tm_mday, &t.tm_mon, &t.tm_year);
    if (!strcmp(line, "TIM"))
      sscanf(val, "%d:%d:%d", &t.tm_hour, &t.tm_min, &t.tm_sec);
    if (!strcmp(line, "HDR"))
      thumb_offset = atoi(val);
    if (!strcmp(line, "X  "))
      raw_width = atoi(val);
    if (!strcmp(line, "Y  "))
      raw_height = atoi(val);
    if (!strcmp(line, "TX "))
      thumb_width = atoi(val);
    if (!strcmp(line, "TY "))
      thumb_height = atoi(val);
    if (!strcmp(line, "APT"))
      aperture = atof(val);
    if (!strcmp(line, "SPE"))
      shutter = atof(val);
    if (!strcmp(line, "FOCLEN"))
      focal_len = atof(val);
    if (!strcmp(line, "BLKOFS"))
      black = atoi(val) +1;
    if (!strcmp(line, "ORI"))
      switch (atoi(val)) {
      case 1:
        flip = 6;
        break;
      case 2:
        flip = 3;
        break;
      case 3:
        flip = 5;
        break;
      }
    if (!strcmp(line, "CUTRECT")) {
      sscanf(val, "%hu %hu %hu %hu",
             &imgdata.sizes.raw_inset_crops[0].cleft,
             &imgdata.sizes.raw_inset_crops[0].ctop,
             &imgdata.sizes.raw_inset_crops[0].cwidth,
             &imgdata.sizes.raw_inset_crops[0].cheight);
    }
  } while (strncmp(line, "EOHD", 4));
  data_offset = thumb_offset + thumb_width * thumb_height * 2;
  t.tm_year -= 1900;
  t.tm_mon -= 1;
  if (mktime(&t) > 0)
    timestamp = mktime(&t);
  strcpy(make, "Rollei");
  strcpy(model, "d530flex");
  thumb_format = LIBRAW_INTERNAL_THUMBNAIL_ROLLEI;
}

void LibRaw::parse_sinar_ia()
{
  int entries, off;
  char str[8], *cp;

  order = 0x4949;
  fseek(ifp, 4, SEEK_SET);
  entries = get4();
  if (entries < 1 || entries > 8192)
    return;
  fseek(ifp, get4(), SEEK_SET);
  while (entries--)
  {
    off = get4();
    get4();
    fread(str, 8, 1, ifp);
    str[7] = 0; // Ensure end of string
    if (!strcmp(str, "META"))
      meta_offset = off;
    if (!strcmp(str, "THUMB"))
      thumb_offset = off;
    if (!strcmp(str, "RAW0"))
      data_offset = off;
  }
  fseek(ifp, meta_offset + 20, SEEK_SET);
  fread(make, 64, 1, ifp);
  make[63] = 0;
  if ((cp = strchr(make, ' ')))
  {
    strcpy(model, cp + 1);
    *cp = 0;
  }
  raw_width = get2();
  raw_height = get2();
  load_raw = &LibRaw::unpacked_load_raw;
  thumb_width = (get4(), get2());
  thumb_height = get2();
  thumb_format = LIBRAW_INTERNAL_THUMBNAIL_PPM;
  maximum = 0x3fff;
}

void LibRaw::parse_kyocera()
{

  int c;
  static const ushort table[13] = {25,  32,  40,  50,  64,  80, 100,
                                   125, 160, 200, 250, 320, 400};

  fseek(ifp, 33, SEEK_SET);
  get_timestamp(1);
  fseek(ifp, 52, SEEK_SET);
  c = get4();
  if ((c > 6) && (c < 20))
    iso_speed = table[c - 7];
  shutter = libraw_powf64l(2.0f, (((float)get4()) / 8.0f)) / 16000.0f;
  FORC4 cam_mul[RGGB_2_RGBG(c)] = get4();
  fseek(ifp, 88, SEEK_SET);
  aperture = libraw_powf64l(2.0f, ((float)get4()) / 16.0f);
  fseek(ifp, 112, SEEK_SET);
  focal_len = get4();

  fseek(ifp, 104, SEEK_SET);
  ilm.MaxAp4CurFocal = libraw_powf64l(2.0f, ((float)get4()) / 16.0f);
  fseek(ifp, 124, SEEK_SET);
  stmread(ilm.Lens, 32, ifp);
  ilm.CameraMount = LIBRAW_MOUNT_Contax_N;
  ilm.CameraFormat = LIBRAW_FORMAT_FF;
  if (ilm.Lens[0])
  {
    ilm.LensMount = LIBRAW_MOUNT_Contax_N;
    ilm.LensFormat = LIBRAW_FORMAT_FF;
  }
}

int LibRaw::parse_jpeg(int offset)
{
  int len, save, hlen, mark;
  fseek(ifp, offset, SEEK_SET);
  if (fgetc(ifp) != 0xff || fgetc(ifp) != 0xd8)
    return 0;

  while (fgetc(ifp) == 0xff && (mark = fgetc(ifp)) != 0xda)
  {
    order = 0x4d4d;
    len = get2() - 2;
    save = ftell(ifp);
    if (mark == 0xc0 || mark == 0xc3 || mark == 0xc9)
    {
      fgetc(ifp);
      raw_height = get2();
      raw_width = get2();
    }
    order = get2();
    hlen = get4();
    if (get4() == 0x48454150 && (save + hlen) >= 0 &&
        (save + hlen) <= ifp->size()) /* "HEAP" */
    {
      parse_ciff(save + hlen, len - hlen, 0);
    }
    if (parse_tiff(save + 6))
      apply_tiff();
    fseek(ifp, save + len, SEEK_SET);
  }
  return 1;
}

void LibRaw::parse_thumb_note(int base, unsigned toff, unsigned tlen)
{
  unsigned entries, tag, type, len, save;

  entries = get2();
  while (entries--)
  {
    tiff_get(base, &tag, &type, &len, &save);
    if (tag == toff)
      thumb_offset = get4() + base;
    if (tag == tlen)
      thumb_length = get4();
    fseek(ifp, save, SEEK_SET);
  }
}

void LibRaw::parse_broadcom()
{

  /* This structure is at offset 0xb0 from the 'BRCM' ident. */
  struct
  {
    uint8_t umode[32];
    uint16_t uwidth;
    uint16_t uheight;
    uint16_t padding_right;
    uint16_t padding_down;
    uint32_t unknown_block[6];
    uint16_t transform;
    uint16_t format;
    uint8_t bayer_order;
    uint8_t bayer_format;
  } header;

  header.bayer_order = 0;
  fseek(ifp, 0xb0 - 0x20, SEEK_CUR);
  fread(&header, 1, sizeof(header), ifp);
  raw_stride =
      ((((((header.uwidth + header.padding_right) * 5) + 3) >> 2) + 0x1f) &
       (~0x1f));
  raw_width = width = header.uwidth;
  raw_height = height = header.uheight;
  filters = 0x16161616; /* default Bayer order is 2, BGGR */

  switch (header.bayer_order)
  {
  case 0: /* RGGB */
    filters = 0x94949494;
    break;
  case 1: /* GBRG */
    filters = 0x49494949;
    break;
  case 3: /* GRBG */
    filters = 0x61616161;
    break;
  }
}

/*
   Returns 1 for a Coolpix 995, 0 for anything else.
 */
int LibRaw::nikon_e995()
{
  int i, histo[256];
  const uchar often[] = {0x00, 0x55, 0xaa, 0xff};

  memset(histo, 0, sizeof histo);
  fseek(ifp, -2000, SEEK_END);
  for (i = 0; i < 2000; i++)
    histo[fgetc(ifp)]++;
  for (i = 0; i < 4; i++)
    if (histo[often[i]] < 200)
      return 0;
  return 1;
}

/*
   Since the TIFF DateTime string has no timezone information,
   assume that the camera's clock was set to Universal Time.
 */
void LibRaw::get_timestamp(int reversed)
{
  struct tm t;
  char str[20];
  int i;

  str[19] = 0;
  if (reversed)
    for (i = 19; i--;)
      str[i] = fgetc(ifp);
  else
    fread(str, 19, 1, ifp);
  memset(&t, 0, sizeof t);
  if (sscanf(str, "%d:%d:%d %d:%d:%d", &t.tm_year, &t.tm_mon, &t.tm_mday,
             &t.tm_hour, &t.tm_min, &t.tm_sec) != 6)
    return;
  t.tm_year -= 1900;
  t.tm_mon -= 1;
  t.tm_isdst = -1;
  if (mktime(&t) > 0)
    timestamp = mktime(&t);
}

#ifdef USE_6BY9RPI
void LibRaw::parse_raspberrypi()
{
	//This structure is at offset 0xB0 from the 'BRCM' ident.
	struct brcm_raw_header {
		uint8_t name[32];
		uint16_t h_width;
		uint16_t h_height;
		uint16_t padding_right;
		uint16_t padding_down;
		uint32_t dummy[6];
		uint16_t transform;
		uint16_t format;
		uint8_t bayer_order;
		uint8_t bayer_format;
	};
	//Values taken from https://github.com/raspberrypi/userland/blob/master/interface/vctypes/vc_image_types.h
#define BRCM_FORMAT_BAYER  33
#define BRCM_BAYER_RAW8    2
#define BRCM_BAYER_RAW10   3
#define BRCM_BAYER_RAW12   4
#define BRCM_BAYER_RAW14   5
#define BRCM_BAYER_RAW16   6

	struct brcm_raw_header header;
	uint8_t brcm_tag[4];

    if (ftell(ifp) > 22LL) // 22 bytes is minimum jpeg size
    {
        thumb_length = ftell(ifp);
        thumb_offset = 0;
        thumb_width = thumb_height = 0;
        load_flags |= 0x4000; // flag: we have JPEG from beginning to meta_offset
    }

	// Sanity check that the caller has found a BRCM header
	if (!fread(brcm_tag, 1, sizeof(brcm_tag), ifp) ||
		memcmp(brcm_tag, "BRCM", sizeof(brcm_tag)))
		return;

	width = raw_width;
	data_offset = ftell(ifp) + 0x8000 - sizeof(brcm_tag);

	if (!fseek(ifp, 0xB0 - int(sizeof(brcm_tag)), SEEK_CUR) &&
		fread(&header, 1, sizeof(header), ifp)) {
		switch (header.bayer_order) {
		case 0: //RGGB
			filters = 0x94949494;
			break;
		case 1: //GBRG
			filters = 0x49494949;
			break;
		default:
		case 2: //BGGR
			filters = 0x16161616;
			break;
		case 3: //GRBG
			filters = 0x61616161;
			break;
		}

		if (header.format == BRCM_FORMAT_BAYER) {
			switch (header.bayer_format) {
			case BRCM_BAYER_RAW8:
				load_raw = &LibRaw::rpi_load_raw8;
				//1 pixel per byte
				raw_stride = ((header.h_width + header.padding_right) + 31)&(~31);
				width = header.h_width;
				raw_height = height = header.h_height;
				is_raw = 1;
				order = 0x4d4d;
				break;
			case BRCM_BAYER_RAW10:
				load_raw = &LibRaw::nokia_load_raw;
				//4 pixels per 5 bytes
				raw_stride = (((((header.h_width + header.padding_right) * 5) + 3) >> 2) + 31)&(~31);
				width = header.h_width;
				raw_height = height = header.h_height;
				is_raw = 1;
				order = 0x4d4d;
				break;
			case BRCM_BAYER_RAW12:
				load_raw = &LibRaw::rpi_load_raw12;
				//2 pixels per 3 bytes
				raw_stride = (((((header.h_width + header.padding_right) * 3) + 1) >> 1) + 31)&(~31);
				width = header.h_width;
				raw_height = height = header.h_height;
				is_raw = 1;
				order = 0x4d4d;
				break;
			case BRCM_BAYER_RAW14:
				load_raw = &LibRaw::rpi_load_raw14;
				//4 pixels per 7 bytes
				raw_stride = (((((header.h_width + header.padding_right) * 7) + 3) >> 2) + 31)&(~31);
				width = header.h_width;
				raw_height = height = header.h_height;
				is_raw = 1;
				order = 0x4d4d;
				break;
			case BRCM_BAYER_RAW16:
				load_raw = &LibRaw::rpi_load_raw16;
				//1 pixel per 2 bytes
				raw_stride = (((header.h_width + header.padding_right) << 1) + 31)&(~31);
				width = header.h_width;
				raw_height = height = header.h_height;
				is_raw = 1;
				order = 0x4d4d;
				break;
			default:
				break;
			}
		}
	}
}
#endif
