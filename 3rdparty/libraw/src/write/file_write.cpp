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
#include <vector>

int LibRaw::flip_index(int row, int col)
{
  if (flip & 4)
    SWAP(row, col);
  if (flip & 2)
    row = iheight - 1 - row;
  if (flip & 1)
    col = iwidth - 1 - col;
  return row * iwidth + col;
}

void LibRaw::tiff_set(struct tiff_hdr *th, ushort *ntag, ushort tag,
                      ushort type, int count, int val)
{
  struct libraw_tiff_tag *tt;
  int c;

  tt = (struct libraw_tiff_tag *)(ntag + 1) + (*ntag)++;
  tt->val.i = val;
  if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_BYTE) && count <= 4)
    FORC(4) tt->val.c[c] = val >> (c << 3);
  else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_ASCII))
  {
    count = int(strnlen((char *)th + val, count - 1)) + 1;
    if (count <= 4)
      FORC(4) tt->val.c[c] = ((char *)th)[val + c];
  }
  else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT) && count <= 2)
    FORC(2) tt->val.s[c] = val >> (c << 4);
  tt->count = count;
  tt->type = type;
  tt->tag = tag;
}

#define TOFF(ptr) ((char *)(&(ptr)) - (char *)th)

void LibRaw::tiff_head(struct tiff_hdr *th, int full)
{
  int c, psize = 0;
  struct tm *t;

  memset(th, 0, sizeof *th);
  th->t_order = htonl(0x4d4d4949) >> 16;
  th->magic = 42;
  th->ifd = 10;
  th->rat[0] = th->rat[2] = 300;
  th->rat[1] = th->rat[3] = 1;
  FORC(6) th->rat[4 + c] = 1000000;
  th->rat[4] *= shutter;
  th->rat[6] *= aperture;
  th->rat[8] *= focal_len;
  strncpy(th->t_desc, desc, 512);
  strncpy(th->t_make, make, 64);
  strncpy(th->t_model, model, 64);
  strcpy(th->soft, "dcraw v" DCRAW_VERSION);
  t = localtime(&timestamp);
  sprintf(th->date, "%04d:%02d:%02d %02d:%02d:%02d", t->tm_year + 1900,
          t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
  strncpy(th->t_artist, artist, 64);
  if (full)
  {
    tiff_set(th, &th->ntag, 254, 4, 1, 0);
    tiff_set(th, &th->ntag, 256, 4, 1, width);
    tiff_set(th, &th->ntag, 257, 4, 1, height);
    tiff_set(th, &th->ntag, 258, 3, colors, output_bps);
    if (colors > 2)
      th->tag[th->ntag - 1].val.i = TOFF(th->bps);
    FORC4 th->bps[c] = output_bps;
    tiff_set(th, &th->ntag, 259, 3, 1, 1);
    tiff_set(th, &th->ntag, 262, 3, 1, 1 + (colors > 1));
  }
  tiff_set(th, &th->ntag, 270, 2, 512, TOFF(th->t_desc));
  tiff_set(th, &th->ntag, 271, 2, 64, TOFF(th->t_make));
  tiff_set(th, &th->ntag, 272, 2, 64, TOFF(th->t_model));
  if (full)
  {
    if (oprof)
      psize = ntohl(oprof[0]);
    tiff_set(th, &th->ntag, 273, 4, 1, sizeof *th + psize);
    tiff_set(th, &th->ntag, 277, 3, 1, colors);
    tiff_set(th, &th->ntag, 278, 4, 1, height);
    tiff_set(th, &th->ntag, 279, 4, 1,
             height * width * colors * output_bps / 8);
  }
  else
    tiff_set(th, &th->ntag, 274, 3, 1, "12435867"[flip] - '0');
  tiff_set(th, &th->ntag, 282, 5, 1, TOFF(th->rat[0]));
  tiff_set(th, &th->ntag, 283, 5, 1, TOFF(th->rat[2]));
  tiff_set(th, &th->ntag, 284, 3, 1, 1);
  tiff_set(th, &th->ntag, 296, 3, 1, 2);
  tiff_set(th, &th->ntag, 305, 2, 32, TOFF(th->soft));
  tiff_set(th, &th->ntag, 306, 2, 20, TOFF(th->date));
  tiff_set(th, &th->ntag, 315, 2, 64, TOFF(th->t_artist));
  tiff_set(th, &th->ntag, 34665, 4, 1, TOFF(th->nexif));
  if (psize)
    tiff_set(th, &th->ntag, 34675, 7, psize, sizeof *th);
  tiff_set(th, &th->nexif, 33434, 5, 1, TOFF(th->rat[4]));
  tiff_set(th, &th->nexif, 33437, 5, 1, TOFF(th->rat[6]));
  tiff_set(th, &th->nexif, 34855, 3, 1, iso_speed);
  tiff_set(th, &th->nexif, 37386, 5, 1, TOFF(th->rat[8]));
  if (gpsdata[1])
  {
    uchar latref[4] = { (uchar)(gpsdata[29]),0,0,0 },
          lonref[4] = { (uchar)(gpsdata[30]),0,0,0 };
    tiff_set(th, &th->ntag, 34853, 4, 1, TOFF(th->ngps));
    tiff_set(th, &th->ngps, 0, 1, 4, 0x202);
    tiff_set(th, &th->ngps, 1, 2, 2, TOFF(latref));
    tiff_set(th, &th->ngps, 2, 5, 3, TOFF(th->gps[0]));
    tiff_set(th, &th->ngps, 3, 2, 2, TOFF(lonref));
    tiff_set(th, &th->ngps, 4, 5, 3, TOFF(th->gps[6]));
    tiff_set(th, &th->ngps, 5, 1, 1, gpsdata[31]);
    tiff_set(th, &th->ngps, 6, 5, 1, TOFF(th->gps[18]));
    tiff_set(th, &th->ngps, 7, 5, 3, TOFF(th->gps[12]));
    tiff_set(th, &th->ngps, 18, 2, 12, TOFF(th->gps[20]));
    tiff_set(th, &th->ngps, 29, 2, 12, TOFF(th->gps[23]));
    memcpy(th->gps, gpsdata, sizeof th->gps);
  }
}

void LibRaw::jpeg_thumb_writer(FILE *tfp, char *t_humb, int t_humb_length)
{
  ushort exif[5];
  struct tiff_hdr th;
  fputc(0xff, tfp);
  fputc(0xd8, tfp);
  if (strcmp(t_humb + 6, "Exif"))
  {
    memcpy(exif, "\xff\xe1  Exif\0\0", 10);
    exif[1] = htons(8 + sizeof th);
    fwrite(exif, 1, sizeof exif, tfp);
    tiff_head(&th, 0);
    fwrite(&th, 1, sizeof th, tfp);
  }
  fwrite(t_humb + 2, 1, t_humb_length - 2, tfp);
}
void LibRaw::write_ppm_tiff()
{
    try
    {
        struct tiff_hdr th;
        ushort *ppm2;
        int c, row, col, soff, rstep, cstep;
        int perc, val, total, t_white = 0x2000;

        perc = width * height * auto_bright_thr;

        if (fuji_width)
            perc /= 2;
        if (!((highlight & ~2) || no_auto_bright))
            for (t_white = c = 0; c < colors; c++)
            {
                for (val = 0x2000, total = 0; --val > 32;)
                    if ((total += histogram[c][val]) > perc)
                        break;
                if (t_white < val)
                    t_white = val;
            }
        gamma_curve(gamm[0], gamm[1], 2, (t_white << 3) / bright);
        iheight = height;
        iwidth = width;
        if (flip & 4)
            SWAP(height, width);

        std::vector<uchar> ppm(width * colors * output_bps / 8);
        ppm2 = (ushort *)ppm.data();
        if (output_tiff)
        {
            tiff_head(&th, 1);
            fwrite(&th, sizeof th, 1, ofp);
            if (oprof)
                fwrite(oprof, ntohl(oprof[0]), 1, ofp);
        }
        else if (colors > 3)
	{
	    if(imgdata.params.output_flags & LIBRAW_OUTPUT_FLAGS_PPMMETA)
	      fprintf(ofp,
              "P7\n# EXPTIME=%0.5f\n# TIMESTAMP=%d\n# ISOSPEED=%d\n"
              "# APERTURE=%0.1f\n# FOCALLEN=%0.1f\n# MAKE=%s\n# MODEL=%s\n"
              "WIDTH %d\nHEIGHT %d\nDEPTH %d\nMAXVAL %d\nTUPLTYPE %s\nENDHDR\n",
              shutter, (int)timestamp, (int)iso_speed,aperture, 
	      focal_len, make, model,
	      width, height, colors, (1 << output_bps) - 1, cdesc);
	    else
            fprintf(
                ofp,
                "P7\nWIDTH %d\nHEIGHT %d\nDEPTH %d\nMAXVAL %d\nTUPLTYPE %s\nENDHDR\n",
                width, height, colors, (1 << output_bps) - 1, cdesc);
	}
        else
	{
	    if(imgdata.params.output_flags & LIBRAW_OUTPUT_FLAGS_PPMMETA)
	    	fprintf(ofp, "P%d\n# EXPTIME=%0.5f\n# TIMESTAMP=%d\n"
		"# ISOSPEED=%d\n# APERTURE=%0.1f\n# FOCALLEN=%0.1f\n"
		"# MAKE=%s\n# MODEL=%s\n%d %d\n%d\n",
                colors/2+5,
		shutter, (int)timestamp, (int)iso_speed,aperture,focal_len,
		make,model,
		width, height, (1 << output_bps)-1);
	    else
             fprintf(ofp, "P%d\n%d %d\n%d\n", colors / 2 + 5, width, height,
            (1 << output_bps) - 1);
        }
        soff = flip_index(0, 0);
        cstep = flip_index(0, 1) - soff;
        rstep = flip_index(1, 0) - flip_index(0, width);
        for (row = 0; row < height; row++, soff += rstep)
        {
            for (col = 0; col < width; col++, soff += cstep)
                if (output_bps == 8)
                    FORCC ppm[col * colors + c] = curve[image[soff][c]] >> 8;
                else
                    FORCC ppm2[col * colors + c] = curve[image[soff][c]];
            if (output_bps == 16 && !output_tiff && htons(0x55aa) != 0x55aa)
                libraw_swab(ppm2, width * colors * 2);
            fwrite(ppm.data(), colors * output_bps / 8, width, ofp);
        }
    }
    catch (...)
    {
      throw LIBRAW_EXCEPTION_ALLOC; // rethrow
    }
}
#if 0
void LibRaw::ppm_thumb()
{
    try
    {
        thumb_length = thumb_width * thumb_height * 3;
        std::vector<char> thumb(thumb_length);
        fprintf(ofp, "P6\n%d %d\n255\n", thumb_width, thumb_height);
        fread(thumb.data(), 1, thumb_length, ifp);
        fwrite(thumb.data(), 1, thumb_length, ofp);
    }
    catch (...)
    {
      throw LIBRAW_EXCEPTION_ALLOC; // rethrow
    }
}

void LibRaw::ppm16_thumb()
{
    try
    {
        unsigned i;
        thumb_length = thumb_width * thumb_height * 3;
        std::vector<char> thumb(thumb_length * 2, 0);
        read_shorts((ushort *)thumb.data(), thumb_length);
        for (i = 0; i < thumb_length; i++)
            thumb[i] = ((ushort *)thumb.data())[i] >> 8;
        fprintf(ofp, "P6\n%d %d\n255\n", thumb_width, thumb_height);
        fwrite(thumb.data(), 1, thumb_length, ofp);
    }
    catch (...)
    {
      throw LIBRAW_EXCEPTION_ALLOC; // rethrow
    }
}

void LibRaw::layer_thumb()
{
    try
    {
        unsigned int i;
        int c;
        char map[][4] = { "012", "102" };

        colors = thumb_misc >> 5 & 7;
        thumb_length = thumb_width * thumb_height;
        std::vector<char> thumb(colors * thumb_length, 0);
        fprintf(ofp, "P%d\n%d %d\n255\n", 5 + (colors >> 1), thumb_width,
            thumb_height);
        fread(thumb.data(), thumb_length, colors, ifp);
        for (i = 0; i < thumb_length; i++)
            FORCC putc(thumb[i + thumb_length * (map[thumb_misc >> 8][c] - '0')], ofp);
    }
    catch (...)
    {
      throw LIBRAW_EXCEPTION_ALLOC; // rethrow
    }
}

void LibRaw::rollei_thumb()
{
    try
    {
        unsigned i;
        thumb_length = thumb_width * thumb_height;
        std::vector<ushort> thumb(thumb_length, 0);
        fprintf(ofp, "P6\n%d %d\n255\n", thumb_width, thumb_height);
        read_shorts(thumb.data(), thumb_length);
        for (i = 0; i < thumb_length; i++)
        {
            putc(thumb[i] << 3, ofp);
            putc(thumb[i] >> 5 << 2, ofp);
            putc(thumb[i] >> 11 << 3, ofp);
        }
    }
    catch (...)
    {
      throw LIBRAW_EXCEPTION_ALLOC; // rethrow
    }
}

void LibRaw::jpeg_thumb()
{
    try
    {
        std::vector<char> thumb(thumb_length);
        fread(thumb.data(), 1, thumb_length, ifp);
        jpeg_thumb_writer(ofp, thumb.data(), thumb_length);
    }
    catch (...)
    {
      throw LIBRAW_EXCEPTION_ALLOC; // rethrow
    }
}
#endif
