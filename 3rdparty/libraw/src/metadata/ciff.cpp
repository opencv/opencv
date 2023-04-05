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
#ifdef _MSC_VER
#if _MSC_VER < 1800 /* below MSVC 2013 */
float roundf(float f)
{
 return floorf(f + 0.5);
}

#endif
#endif

/*
   CIFF block 0x1030 contains an 8x8 white sample.
   Load this into white[][] for use in scale_colors().
 */
void LibRaw::ciff_block_1030()
{
  static const ushort key[] = {0x410, 0x45f3};
  int i, bpp, row, col, vbits = 0;
  unsigned long bitbuf = 0;

  if ((get2(), get4()) != 0x80008 || !get4())
    return;
  bpp = get2();
  if (bpp != 10 && bpp != 12)
    return;
  for (i = row = 0; row < 8; row++)
    for (col = 0; col < 8; col++)
    {
      if (vbits < bpp)
      {
        bitbuf = bitbuf << 16 | (get2() ^ key[i++ & 1]);
        vbits += 16;
      }
      white[row][col] = bitbuf >> (vbits -= bpp) & ~(-1 << bpp);
    }
}

/*
   Parse a CIFF file, better known as Canon CRW format.
 */
void LibRaw::parse_ciff(int offset, int length, int depth)
{
  int nrecs, c, type, len, wbi = -1;
  INT64 save, tboff;
  ushort key[] = {0x410, 0x45f3};
  ushort CanonColorInfo1_key;
  ushort Appendix_A = 0;
  INT64 WB_table_offset = 0;
  int UseWBfromTable_as_AsShot = 1;
  int Got_AsShotWB = 0;
  INT64 fsize = ifp->size();
  if (metadata_blocks++ > LIBRAW_MAX_METADATA_BLOCKS)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  fseek(ifp, offset + length - 4, SEEK_SET);
  tboff = INT64(get4()) + offset;
  fseek(ifp, tboff, SEEK_SET);
  nrecs = get2();
  if (nrecs < 1)
    return;
  if ((nrecs | depth) > 127)
    return;

  if (nrecs * 10 + offset > fsize)
    return;

  while (nrecs--)
  {
    type = get2();
    len = get4();
    INT64 see = offset + get4();
    save = ftell(ifp);

    /* the following tags are not sub-tables
     * they contain the value in the "len" field
     * for such tags skip the check against filesize
     */
    if ((type != 0x2007) && (type != 0x580b) && (type != 0x501c) &&
        (type != 0x5029) && (type != 0x5813) && (type != 0x5814) &&
        (type != 0x5817) && (type != 0x5834) && (type != 0x580e))
    {

      if (see >= fsize)
      { // At least one byte
        fseek(ifp, save, SEEK_SET);
        continue;
      }
      fseek(ifp, see, SEEK_SET);
      if ((((type >> 8) + 8) | 8) == 0x38)
      {
        parse_ciff(ftell(ifp), len, depth + 1); /* Parse a sub-table */
      }
    }

    if (type == 0x3004)
    {
      parse_ciff(ftell(ifp), len, depth + 1);
    }
    else if (type == 0x0810)
    {
      fread(artist, 64, 1, ifp);
    }
    else if (type == 0x080a)
    {
      fread(make, 64, 1, ifp);
      fseek(ifp, strbuflen(make) - 63, SEEK_CUR);
      fread(model, 64, 1, ifp);

    } else if (type == 0x080b) {
      stmread(imCommon.firmware, (unsigned)len, ifp);
      if (!strncasecmp(imCommon.firmware, "Firmware Version", 16))
        memmove(imCommon.firmware, imCommon.firmware + 16, strlen(imCommon.firmware) - 15);
      trimSpaces(imCommon.firmware);

    } else if (type == 0x1810)
    {
      width = get4();
      height = get4();
      pixel_aspect = int_to_float(get4());
      flip = get4();
    }
    else if (type == 0x1835)
    { /* Get the decoder table */
      tiff_compress = get4();
    }
    else if (type == 0x2007)
    {
      thumb_offset = see;
      thumb_length = len;
    }
    else if (type == 0x1818)
    {
      shutter = libraw_powf64l(2.0f, -int_to_float((get4(), get4())));
      ilm.CurAp = aperture = libraw_powf64l(2.0f, int_to_float(get4()) / 2);
    }
    else if (type == 0x102a) // CanonShotInfo
    {
      //      iso_speed = pow (2.0, (get4(),get2())/32.0 - 4) * 50;
      get2(); // skip one
      iso_speed =
          libraw_powf64l(2.0f, (get2() + get2()) / 32.0f - 5.0f) * 100.0f;
      ilm.CurAp = aperture = _CanonConvertAperture((get2(), get2()));
      shutter = libraw_powf64l(2.0f, -float((short)get2()) / 32.f);
      imCanon.wbi = wbi = (get2(), get2());
      if (wbi >= (int)Canon_wbi2std.size())
        wbi = 0;
      fseek(ifp, 32, SEEK_CUR);
      if (shutter > 1e6)
        shutter = float(get2()) / 10.f;
    }
    else if (type == 0x102c) // CanonColorInfo2 / Appendix A: Pro90IS, G1, G2, S30, S40
    {
      int CanonColorInfo2_type = get2(); // G1 1028, G2 272, Pro90 IS 769, S30 274, S40 273, EOS D30 276
      if (CanonColorInfo2_type > 512) { /* Pro90 IS, G1 */
        fseek(ifp, 118, SEEK_CUR);
        FORC4 cam_mul[BG2RG1_2_RGBG(c)] = get2();
      }
      else if (CanonColorInfo2_type != 276) { /* G2, S30, S40 */
        Appendix_A = 1;
        WB_table_offset = -14;
        fseek(ifp, 98, SEEK_CUR);
        FORC4 cam_mul[GRBG_2_RGBG(c)] = get2();
        if (cam_mul[0] > 0.001f) Got_AsShotWB = 1;
      }
    }
    else if (type == 0x10a9) // ColorBalance: Canon D60, 10D, 300D, and clones
    {
      int bls = 0;
/*
      int table[] = {
          LIBRAW_WBI_Auto,     // 0
          LIBRAW_WBI_Daylight, // 1
          LIBRAW_WBI_Cloudy,   // 2
          LIBRAW_WBI_Tungsten, // 3
          LIBRAW_WBI_FL_W,     // 4
          LIBRAW_WBI_Flash,    // 5
          LIBRAW_WBI_Custom,   // 6, absent in Canon D60
          LIBRAW_WBI_Auto,     // 7, use this if camera is set to b/w JPEG
          LIBRAW_WBI_Shade,    // 8
          LIBRAW_WBI_Kelvin    // 9, absent in Canon D60
      };
*/
      int nWB =
          ((get2() - 2) / 8) -
          1; // 2 bytes this, N recs 4*2bytes each, last rec is black level
      if (nWB)
        FORC4 icWBC[LIBRAW_WBI_Auto][RGGB_2_RGBG(c)] = get2();
      if (nWB >= 7)
        Canon_WBpresets(0, 0);
      else
        FORC4 cam_mul[c] = float(icWBC[LIBRAW_WBI_Auto][c]);
      if (nWB == 7) // mostly Canon EOS D60 + some fw#s for 300D;
                    // check for 0x1668000 is unreliable
      {
        if ((wbi >= 0) && (wbi < 9) && (wbi != 6))
        {
          FORC4 cam_mul[c] = float(icWBC[Canon_wbi2std[wbi]][c]);
        }
        else
        {
          FORC4 cam_mul[c] = float(icWBC[LIBRAW_WBI_Auto][c]);
        }
      }
      else if (nWB == 9) // Canon 10D, 300D
      {
        FORC4 icWBC[LIBRAW_WBI_Custom][RGGB_2_RGBG(c)] = get2();
        FORC4 icWBC[LIBRAW_WBI_Kelvin][RGGB_2_RGBG(c)] = get2();
        if ((wbi >= 0) && (wbi < 10))
        {
          FORC4 cam_mul[c] = float(icWBC[Canon_wbi2std[wbi]][c]);
        }
        else
        {
          FORC4 cam_mul[c] = float(icWBC[LIBRAW_WBI_Auto][c]);
        }
      }
      FORC4
      bls += (imCanon.ChannelBlackLevel[RGGB_2_RGBG(c)] = get2());
      imCanon.AverageBlackLevel = bls / 4;
    }
    else if (type == 0x102d)
    {
      Canon_CameraSettings(len >> 1);
    }

    else if (type == 0x10b4) {
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

    } else if (type == 0x580b)
    {
      if (strcmp(model, "Canon EOS D30"))
        sprintf(imgdata.shootinginfo.BodySerial, "%d", len);
      else
        sprintf(imgdata.shootinginfo.BodySerial, "%0x-%05d", len >> 16,
                len & 0xffff);
    }
    else if (type == 0x0032) // CanonColorInfo1
    {
      if (len == 768) { // EOS D30

        ushort q;
        fseek(ifp, 4, SEEK_CUR);
        for (unsigned linenum = 0; linenum < Canon_D30_linenums_2_StdWBi.size(); linenum++) {
          if (Canon_D30_linenums_2_StdWBi[linenum] != LIBRAW_WBI_Unknown) {
            FORC4 {
              q = get2();
              icWBC[Canon_D30_linenums_2_StdWBi[linenum]][RGGB_2_RGBG(c)] =
                (int)(roundf(1024000.0f / (float)MAX(1, q)));
            }
//         if (Canon_wbi2std[imCanon.wbi] == *(Canon_D30_linenums_2_StdWBi + linenum)) {
//           FORC4 cam_mul[c] = icWBC[*(Canon_D30_linenums_2_StdWBi + linenum)][c];
//           Got_AsShotWB = 1;
//           }
          }
        }
        fseek (ifp, 68-int(Canon_D30_linenums_2_StdWBi.size())*8, SEEK_CUR);

        FORC4 {
          q = get2();
          cam_mul[RGGB_2_RGBG(c)] = 1024.f / float(MAX(1, q));
        }
        if (!wbi)
          cam_mul[0] = -1; // use my auto white balance

      }
      else if ((cam_mul[0] <= 0.001f) || // Pro1, G3, G5, G6, S45, S50, S60, S70
               Appendix_A)               // G2, S30, S40
      {
        libraw_static_table_t linenums_2_StdWBi;
        unsigned AsShotWB_linenum = Canon_wbi2std.size();

        CanonColorInfo1_key = get2();
        if ((CanonColorInfo1_key == key[0]) && (len == 2048)) { // Pro1
          linenums_2_StdWBi = Canon_KeyIs0x0410_Len2048_linenums_2_StdWBi;
          WB_table_offset = 8;

        } else if ((CanonColorInfo1_key == key[0]) && (len == 3072)) { // S60, S70, G6
          linenums_2_StdWBi = Canon_KeyIs0x0410_Len3072_linenums_2_StdWBi;
          WB_table_offset = 16;

        } else if (!CanonColorInfo1_key && (len == 2048)) { // G2, S30, S40; S45, S50, G3, G5
          key[0] = key[1] = 0;
          linenums_2_StdWBi = Canon_KeyIsZero_Len2048_linenums_2_StdWBi;
          if (atof(imCommon.firmware) < 1.02f)
            UseWBfromTable_as_AsShot = 0;

        } else goto next_tag;

        if ((Canon_wbi2std[wbi] == LIBRAW_WBI_Auto)    ||
            (Canon_wbi2std[wbi] == LIBRAW_WBI_Unknown) ||
            Got_AsShotWB)
          UseWBfromTable_as_AsShot = 0;

        if (UseWBfromTable_as_AsShot) {
          int temp_wbi;
          if (Canon_wbi2std[wbi] == LIBRAW_WBI_Custom) temp_wbi = LIBRAW_WBI_Daylight;
          else temp_wbi = wbi;
          for (AsShotWB_linenum = 0; AsShotWB_linenum < linenums_2_StdWBi.size(); AsShotWB_linenum++) {
            if (Canon_wbi2std[temp_wbi] == linenums_2_StdWBi[AsShotWB_linenum]) {
              break;
            }
          }
        }

        fseek (ifp, 78LL+WB_table_offset, SEEK_CUR);
        for (unsigned linenum = 0; linenum < linenums_2_StdWBi.size(); linenum++) {
          if (linenums_2_StdWBi[linenum] != LIBRAW_WBI_Unknown) {
            FORC4 icWBC[linenums_2_StdWBi[linenum]][GRBG_2_RGBG(c)] = get2() ^ key[c & 1];
            if (UseWBfromTable_as_AsShot && (AsShotWB_linenum == linenum)) {
              FORC4 cam_mul[c] = float(icWBC[linenums_2_StdWBi[linenum]][c]);
              Got_AsShotWB = 1;
            }
          } else {
            fseek(ifp, 8, SEEK_CUR);
          }
        }
        if (!Got_AsShotWB)
          cam_mul[0] = -1;
      }
    }
    else if (type == 0x1030 && wbi >= 0 && (0x18040 >> wbi & 1))
    {
      ciff_block_1030(); // all that don't have 0x10a9
    }
    else if (type == 0x1031)
    {
			raw_width  = imCanon.SensorWidth = (get2(), get2());
			raw_height = imCanon.SensorHeight = get2();
			fseek(ifp, 4, SEEK_CUR);
			imCanon.DefaultCropAbsolute = get_CanonArea();
			imCanon.LeftOpticalBlack    = get_CanonArea();
    }
    else if (type == 0x501c)
    {
      iso_speed = float(len & 0xffff);
    }
    else if (type == 0x5029)
    {
      ilm.CurFocal = float( len >> 16);
      ilm.FocalType = len & 0xffff;
      if (ilm.FocalType == LIBRAW_FT_ZOOM_LENS)
      {
        ilm.FocalUnits = 32;
        if (ilm.FocalUnits > 1)
          ilm.CurFocal /= (float)ilm.FocalUnits;
      }
      focal_len = ilm.CurFocal;
    }
    else if (type == 0x5813)
    {
      flash_used = int_to_float(len);
    }
    else if (type == 0x5814)
    {
      canon_ev = int_to_float(len);
    }
    else if (type == 0x5817)
    {
      shot_order = len;
    }
    else if (type == 0x5834)
    {
      unique_id = ((unsigned long long)len << 32) >> 32;
      setCanonBodyFeatures(unique_id);
    }
    else if (type == 0x580e)
    {
      timestamp = len;
    }
    else if (type == 0x180e)
    {
      timestamp = get4();
    }

next_tag:;
#ifdef LOCALTIME
    if ((type | 0x4000) == 0x580e)
      timestamp = mktime(gmtime(&timestamp));
#endif
    fseek(ifp, save, SEEK_SET);
  }
}
