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


static libraw_area_t sget_CanonArea(uchar *s) {
  libraw_area_t la = {};
  la.l = s[0] << 8 | s[1];
  la.t = s[2] << 8 | s[3];
  la.r = s[4] << 8 | s[5];
  la.b = s[6] << 8 | s[7];
  return la;
}

int LibRaw::selectCRXFrame(short trackNum, unsigned frameIndex)
{
  uint32_t sample_size;
  uint32_t stsc_index = 0;
  uint32_t current_sample = 0;
  crx_data_header_t *hdr = &libraw_internal_data.unpacker_data.crx_header[trackNum];

  if (frameIndex >= hdr->sample_count)
    return -1;

  for (int i = 0; i < hdr->chunk_count; i++)
  {
    int64_t current_offset = hdr->chunk_offsets[i];

    while((stsc_index < hdr->stsc_count) && (i+1 == hdr->stsc_data[stsc_index+1].first))
      stsc_index++;

    for (int j = 0; j < hdr->stsc_data[stsc_index].count; j++)
    {
      if (current_sample > hdr->sample_count)
        return -1;

      sample_size = hdr->sample_size > 0 ? hdr->sample_size : hdr->sample_sizes[current_sample];
      if(current_sample == frameIndex)
      {
        hdr->MediaOffset = current_offset;
        hdr->MediaSize = sample_size;
        return 0;
      }
      current_offset += sample_size;
      current_sample++;
    }
  }
  return -1;
}

void LibRaw::selectCRXTrack()
{
  short maxTrack = libraw_internal_data.unpacker_data.crx_track_count;
  if (maxTrack < 0)
    return;

  INT64 bitcounts[LIBRAW_CRXTRACKS_MAXCOUNT], maxbitcount = 0;
  int framecounts[LIBRAW_CRXTRACKS_MAXCOUNT], maxframecount = 0;
  uint32_t maxjpegbytes = 0;
  int framecnt = 0;
  int media_tracks = 0;
  int track_select = 0;
  int frame_select = 0;
  int err;
  memset(bitcounts, 0, sizeof(bitcounts));
  memset(framecounts, 0, sizeof(framecounts));

  // Calc max frame bitcount for max-sized RAW track(s) selection
  for (int i = 0; i <= maxTrack && i < LIBRAW_CRXTRACKS_MAXCOUNT; i++)
  {
    crx_data_header_t *d = &libraw_internal_data.unpacker_data.crx_header[i];
    if (d->MediaType == 1) // RAW
    {
      bitcounts[i] = INT64(d->nBits) * INT64(d->f_width) * INT64(d->f_height);
      maxbitcount = MAX(bitcounts[i], maxbitcount);
	  if (d->sample_count > 1)
		  framecounts[i] = d->sample_count;
    }
  }

  if (maxbitcount < 8) // no raw tracks
	  return;

  // Calc  RAW tracks and frames
  for (int i = 0; i <= maxTrack && i < LIBRAW_CRXTRACKS_MAXCOUNT; i++)
  {
	  if (bitcounts[i] == maxbitcount)
	  {
		  media_tracks++;
		  if (framecounts[i] > 1)
			  framecnt = MAX(framecnt, framecounts[i]);
	  }
  }
  
  // If the file has only 1 media track shot_select represents frames select.
  // If the file has multiple media tracks shot_select represents track select.
  // If the file has multiple media tracks and multiple frames it is currently unsupported.

  if (framecnt && media_tracks > 1)
    return;
  else if (framecnt)
    frame_select = shot_select;
  else
    track_select = shot_select;

  int tracki = -1;
  for (int i = 0, trackcnt = 0; i <= maxTrack && i < LIBRAW_CRXTRACKS_MAXCOUNT; i++)
  {
    if (bitcounts[i] == maxbitcount)
    {
      if (trackcnt <= (int)track_select)
        tracki = i;
	  trackcnt++;
    }
  }

  if (tracki >= 0 && tracki < LIBRAW_CRXTRACKS_MAXCOUNT /* && frame_select > 0 */)
  {
	  framecnt = framecounts[tracki]; // Update to selected track
	  frame_select = LIM(frame_select, 0, framecnt);
	  if(frame_select > 0)
		if (selectCRXFrame(tracki, frame_select))
			  return;
  }
  else
	  return; // No RAW track index

  // Frame selected: parse CTMD metadata
  for (int i = 0, trackcnt = 0; i <= maxTrack && i < LIBRAW_CRXTRACKS_MAXCOUNT; i++)
  {
	  crx_data_header_t *d = &libraw_internal_data.unpacker_data.crx_header[i];
	  int fsel = LIM(frame_select, 0, d->sample_count);
	  if (d->MediaType == 3) // CTMD metadata
	  {
		  /* ignore errors !*/
		  if (fsel)
			  selectCRXFrame(i, fsel);
		  parseCR3_CTMD(i);
	  }
	  else if (d->MediaType == 2) // JPEG
	  {
		  if (fsel)
			  selectCRXFrame(i, fsel);
		  if (d->MediaSize > maxjpegbytes)
		  {
			  maxjpegbytes = d->MediaSize;
			  thumb_offset = d->MediaOffset;
			  thumb_length = d->MediaSize;
              if (imgdata.thumbs_list.thumbcount < LIBRAW_THUMBNAIL_MAXCOUNT)
              {
                bool do_add = true;
                for (int idx = 0; idx < imgdata.thumbs_list.thumbcount; idx++)
                  if (imgdata.thumbs_list.thumblist[idx].toffset == thumb_offset)
                  {
                    do_add = false;
                    break;
                  }
                if (do_add)
                {
                  int idx = imgdata.thumbs_list.thumbcount;
                  imgdata.thumbs_list.thumblist[idx].tformat = LIBRAW_INTERNAL_THUMBNAIL_JPEG;
                  imgdata.thumbs_list.thumblist[idx].toffset = thumb_offset;
                  imgdata.thumbs_list.thumblist[idx].tlength = thumb_length;
                  imgdata.thumbs_list.thumblist[idx].tflip = 0xffff;
                  imgdata.thumbs_list.thumblist[idx].tmisc = (3 << 5) | 8; // 3 samples/8 bps
                  imgdata.thumbs_list.thumblist[idx].twidth = 0;
                  imgdata.thumbs_list.thumblist[idx].theight = 0;
                  imgdata.thumbs_list.thumbcount++;
                }
              }
		  }
	  }
  }

  if (framecnt)
    is_raw = framecnt;
  else
    is_raw = media_tracks;

  if (tracki >= 0 && tracki < LIBRAW_CRXTRACKS_MAXCOUNT)
  {
    crx_data_header_t *d =
        &libraw_internal_data.unpacker_data.crx_header[tracki];
    data_offset = d->MediaOffset;
    data_size = d->MediaSize;
    raw_width = d->f_width;
    raw_height = d->f_height;
    load_raw = &LibRaw::crxLoadRaw;
    tiff_bps = d->encType == 3? d->medianBits : d->nBits;
    switch (d->cfaLayout)
    {
    case 0:
      filters = 0x94949494;
      break;
    case 1:
      filters = 0x61616161;
      break;
    case 2:
      filters = 0x49494949;
      break;
    case 3:
      filters = 0x16161616;
      break;
    }

    libraw_internal_data.unpacker_data.crx_track_selected = tracki;

    int tiff_idx = -1;
    INT64 tpixels = 0;
    for (unsigned i = 0; i < tiff_nifds && i < LIBRAW_IFD_MAXCOUNT; i++)
      if (INT64(tiff_ifd[i].t_height) * INT64(tiff_ifd[i].t_height) > tpixels)
      {
        tpixels = INT64(tiff_ifd[i].t_height) * INT64(tiff_ifd[i].t_height);
        tiff_idx = i;
      }
    if (tiff_idx >= 0)
      flip = tiff_ifd[tiff_idx].t_flip;
  }
}

#define bad_hdr()                                                              \
  (((order != 0x4d4d) && (order != 0x4949)) || (get2() != 0x002a) ||           \
   (get4() != 0x00000008))

int LibRaw::parseCR3_CTMD(short trackNum)
{
  int err = 0;
  short s_order = order;
  order = 0x4949;
  uint32_t relpos_inDir = 0;
  uint32_t relpos_inBox = 0;
  unsigned szItem, Tag, lTag;
  ushort tItem;

#define track libraw_internal_data.unpacker_data.crx_header[trackNum]

  if (track.MediaType != 3)
  {
    err = -10;
    goto ctmd_fin;
  }

  while (relpos_inDir + 6 < track.MediaSize)
  {
    if (track.MediaOffset + relpos_inDir > ifp->size() - 6) // need at least 6 bytes
    {
        err = -11;
        goto ctmd_fin;
    }
    fseek(ifp, track.MediaOffset + relpos_inDir, SEEK_SET);
    szItem = get4();
    tItem = get2();
    if (szItem < 1 || (  (relpos_inDir + szItem) > track.MediaSize))
    {
      err = -11;
      goto ctmd_fin;
    }
    if ((tItem == 7) || (tItem == 8) || (tItem == 9))
    {
      relpos_inBox = relpos_inDir + 12L;
      while (relpos_inBox + 8 < relpos_inDir + szItem)
      {
        if (track.MediaOffset + relpos_inBox > ifp->size() - 8) // need at least 8 bytes
        {
            err = -11;
            goto ctmd_fin;
        }
        fseek(ifp, track.MediaOffset + relpos_inBox, SEEK_SET);
        lTag = get4();
        Tag = get4();
        if (lTag < 8)
        {
          err = -12;
          goto ctmd_fin;
        }
        else if ((relpos_inBox + lTag) > (relpos_inDir + szItem))
        {
          err = -11;
          goto ctmd_fin;
        }
        if ((Tag == 0x927c) && ((tItem == 7) || (tItem == 8)))
        {
          fseek(ifp, track.MediaOffset + relpos_inBox + 8L,
                SEEK_SET);
          short q_order = order;
          order = get2();
          if (bad_hdr())
          {
            err = -13;
            goto ctmd_fin;
          }
          fseek(ifp, -8L, SEEK_CUR);
          libraw_internal_data.unpacker_data.CR3_CTMDtag = 1;
          parse_makernote(track.MediaOffset + relpos_inBox + 8,
                          0);
          libraw_internal_data.unpacker_data.CR3_CTMDtag = 0;
          order = q_order;
        }
        relpos_inBox += lTag;
      }
    }
    relpos_inDir += szItem;
  }

ctmd_fin:
  order = s_order;
  return err;
}
#undef track

int LibRaw::parseCR3(INT64 oAtomList,
                     INT64 szAtomList, short &nesting,
                     char *AtomNameStack, short &nTrack, short &TrackType)
{
  /*
  Atom starts with 4 bytes for Atom size and 4 bytes containing Atom name
  Atom size includes the length of the header and the size of all "contained"
  Atoms if Atom size == 1, Atom has the extended size stored in 8 bytes located
  after the Atom name if Atom size == 0, it is the last top-level Atom extending
  to the end of the file Atom name is often a 4 symbol mnemonic, but can be a
  4-byte integer
  */
  const char UIID_Canon[17] =
      "\x85\xc0\xb6\x87\x82\x0f\x11\xe0\x81\x11\xf4\xce\x46\x2b\x6a\x48";
  const unsigned char UIID_CanonPreview[17] = "\xea\xf4\x2b\x5e\x1c\x98\x4b\x88\xb9\xfb\xb7\xdc\x40\x6e\x4d\x16";
  const unsigned char UUID_XMP[17] = "\xbe\x7a\xcf\xcb\x97\xa9\x42\xe8\x9c\x71\x99\x94\x91\xe3\xaf\xac";
  
  /*
  AtomType = 0 - unknown: "unk."
  AtomType = 1 - container atom: "cont"
  AtomType = 2 - leaf atom: "leaf"
  AtomType = 3 - can be container, can be leaf: "both"
  */
  short AtomType;
  static const struct
  {
    char AtomName[5];
    short AtomType;
  } AtomNamesList[] = {
      {"dinf", 1},
      {"edts", 1},
      {"fiin", 1},
      {"ipro", 1},
      {"iprp", 1},
      {"mdia", 1},
      {"meco", 1},
      {"mere", 1},
      {"mfra", 1},
      {"minf", 1},
      {"moof", 1},
      {"moov", 1},
      {"mvex", 1},
      {"paen", 1},
      {"schi", 1},
      {"sinf", 1},
      {"skip", 1},
      {"stbl", 1},
      {"stsd", 1},
      {"strk", 1},
      {"tapt", 1},
      {"traf", 1},
      {"trak", 1},

      {"cdsc", 2},
      {"colr", 2},
      {"dimg", 2},
      // {"dref", 2},
      {"free", 2},
      {"frma", 2},
      {"ftyp", 2},
      {"hdlr", 2},
      {"hvcC", 2},
      {"iinf", 2},
      {"iloc", 2},
      {"infe", 2},
      {"ipco", 2},
      {"ipma", 2},
      {"iref", 2},
      {"irot", 2},
      {"ispe", 2},
      {"meta", 2},
      {"mvhd", 2},
      {"pitm", 2},
      {"pixi", 2},
      {"schm", 2},
      {"thmb", 2},
      {"tkhd", 2},
      {"url ", 2},
      {"urn ", 2},

      {"CCTP", 1},
      {"CRAW", 1},

      {"JPEG", 2},
      {"CDI1", 2},
      {"CMP1", 2},

      {"CNCV", 2},
      {"CCDT", 2},
      {"CTBO", 2},
      {"CMT1", 2},
      {"CMT2", 2},
      {"CMT3", 2},
      {"CMT4", 2},
      {"CNOP", 2},
      {"THMB", 2},
      {"co64", 2},
      {"mdat", 2},
      {"mdhd", 2},
      {"nmhd", 2},
      {"stsc", 2},
      {"stsz", 2},
      {"stts", 2},
      {"vmhd", 2},

      {"dref", 3},
      {"uuid", 3},
  };

  const char sHandlerType[5][5] = {"unk.", "soun", "vide", "hint", "meta"};

  int c, err=0;

  ushort tL;                        // Atom length represented in 4 or 8 bytes
  char nmAtom[5];                   // Atom name
  INT64 oAtom, szAtom; // Atom offset and Atom size
  INT64 oAtomContent,
      szAtomContent; // offset and size of Atom content
  INT64 lHdr;

  char UIID[16];
  uchar CMP1[85];
  uchar CDI1[60];
  char HandlerType[5], MediaFormatID[5];
  uint32_t relpos_inDir, relpos_inBox;
  unsigned szItem, Tag, lTag;
  ushort tItem;

  nmAtom[0] = MediaFormatID[0] = nmAtom[4] = MediaFormatID[4] = '\0';
  strcpy(HandlerType, sHandlerType[0]);
  oAtom = oAtomList;
  nesting++;
  if (nesting > 31)
    return -14; // too deep nesting
  short s_order = order;

  while ((oAtom + 8LL) <= (oAtomList + szAtomList))
  {
    lHdr = 0ULL;
    err = 0;
    order = 0x4d4d;
    fseek(ifp, oAtom, SEEK_SET);
    szAtom = get4();
    FORC4 nmAtom[c] = AtomNameStack[nesting * 4 + c] = fgetc(ifp);
    AtomNameStack[(nesting + 1) * 4] = '\0';
    tL = 4;
    AtomType = 0;

    for (c = 0; c < int(sizeof AtomNamesList / sizeof *AtomNamesList); c++)
      if (!strcmp(nmAtom, AtomNamesList[c].AtomName))
      {
        AtomType = AtomNamesList[c].AtomType;
        break;
      }

    if (!AtomType)
    {
      err = 1;
    }

    if (szAtom == 0ULL)
    {
      if (nesting != 0)
      {
        err = -2;
        goto fin;
      }
      szAtom = szAtomList - oAtom;
      oAtomContent = oAtom + 8ULL;
      szAtomContent = szAtom - 8ULL;
    }
    else if (szAtom == 1LL)
    {
      if ((oAtom + 16LL) > (oAtomList + szAtomList))
      {
        err = -3;
        goto fin;
      }
      tL = 8;
      szAtom = (((unsigned long long)get4()) << 32) | get4();
      oAtomContent = oAtom + 16ULL;
      szAtomContent = szAtom - 16ULL;
    }
    else
    {
      oAtomContent = oAtom + 8ULL;
      szAtomContent = szAtom - 8ULL;
    }

	if (!strcmp(AtomNameStack, "uuid")) // Top level uuid
	{
		INT64 tt = ftell(ifp);
		lHdr = 16ULL;
		fread(UIID, 1, lHdr, ifp);
		if (!memcmp(UIID, UUID_XMP, 16) && szAtom > 24LL && szAtom < 1024000LL)
		{
			xmpdata = (char *)malloc(xmplen = unsigned(szAtom - 23));
			fread(xmpdata, szAtom - 24, 1, ifp);
			xmpdata[szAtom - 24] = 0;
		}
		else if (!memcmp(UIID, UIID_CanonPreview, 16) && szAtom > 48LL && szAtom < 100LL * 1024000LL)
		{
			// read next 48 bytes, check for 'PRVW'
			unsigned char xdata[32];
			fread(xdata, 32, 1, ifp);	
			if (!memcmp(xdata + 12, "PRVW", 4))
			{
				thumb_length = unsigned(szAtom - 56);
				thumb_offset = ftell(ifp);
				if (imgdata.thumbs_list.thumbcount < LIBRAW_THUMBNAIL_MAXCOUNT)
				{
					bool do_add = true;
					for(int idx = 0; idx < imgdata.thumbs_list.thumbcount; idx++)
						if (imgdata.thumbs_list.thumblist[idx].toffset == thumb_offset)
						{
							do_add = false;
							break;
						}
					if (do_add)
					{
						int idx = imgdata.thumbs_list.thumbcount;
						imgdata.thumbs_list.thumblist[idx].tformat = LIBRAW_INTERNAL_THUMBNAIL_JPEG;
						imgdata.thumbs_list.thumblist[idx].toffset = thumb_offset;
						imgdata.thumbs_list.thumblist[idx].tlength = thumb_length;
						imgdata.thumbs_list.thumblist[idx].tflip = 0xffff;
						imgdata.thumbs_list.thumblist[idx].tmisc = (3 << 5) | 8; // 3 samples/8 bps
						imgdata.thumbs_list.thumblist[idx].twidth = (xdata[22] << 8) + xdata[23];
                        imgdata.thumbs_list.thumblist[idx].theight = (xdata[24] << 8) + xdata[25];
						imgdata.thumbs_list.thumbcount++;
					}
				}

			}
		}
		fseek(ifp, tt, SEEK_SET);
	}

    if (!strcmp(nmAtom, "trak"))
    {
      nTrack++;
      TrackType = 0;
      if (nTrack >= LIBRAW_CRXTRACKS_MAXCOUNT)
        break;
    }
    if (!strcmp(AtomNameStack, "moovuuid"))
    {
      lHdr = 16ULL;
      fread(UIID, 1, lHdr, ifp);
      if (!strncmp(UIID, UIID_Canon, lHdr))
      {
        AtomType = 1;
      }
      else
        fseek(ifp, -lHdr, SEEK_CUR);
    }
    else if (!strcmp(AtomNameStack, "moovuuidCCTP"))
    {
      lHdr = 12ULL;
    }
    else if (!strcmp(AtomNameStack, "moovuuidCMT1"))
    {
      short q_order = order;
      order = get2();
      if ((tL != 4) || bad_hdr())
      {
        err = -4;
        goto fin;
      }
      if (!libraw_internal_data.unpacker_data.cr3_ifd0_length)
        libraw_internal_data.unpacker_data.cr3_ifd0_length = unsigned(szAtomContent);
      parse_tiff_ifd(oAtomContent);
      order = q_order;
    }
	else if (!strcmp(AtomNameStack, "moovuuidTHMB") && szAtom > 24)
	{
		unsigned char xdata[16];
		fread(xdata, 16, 1, ifp);
		INT64 xoffset = ftell(ifp);
		if (imgdata.thumbs_list.thumbcount < LIBRAW_THUMBNAIL_MAXCOUNT)
		{
			bool do_add = true;
			for (int idx = 0; idx < imgdata.thumbs_list.thumbcount; idx++)
				if (imgdata.thumbs_list.thumblist[idx].toffset == xoffset)
				{
					do_add = false;
					break;
				}
            if (do_add)
            {
              int idx = imgdata.thumbs_list.thumbcount;
              imgdata.thumbs_list.thumblist[idx].tformat = LIBRAW_INTERNAL_THUMBNAIL_JPEG;
			  imgdata.thumbs_list.thumblist[idx].toffset = xoffset;
              imgdata.thumbs_list.thumblist[idx].tlength = szAtom-24;
			  imgdata.thumbs_list.thumblist[idx].tflip = 0xffff;
              imgdata.thumbs_list.thumblist[idx].tmisc = (3 << 5) | 8; // 3 samples/8 bps
              imgdata.thumbs_list.thumblist[idx].twidth = (xdata[4] << 8) + xdata[5];
              imgdata.thumbs_list.thumblist[idx].theight = (xdata[6] << 8) + xdata[7];
              imgdata.thumbs_list.thumbcount++;
            }
		}
	}
	else if (!strcmp(AtomNameStack, "moovuuidCMT2"))
	{
		short q_order = order;
		order = get2();
		if ((tL != 4) || bad_hdr())
		{
			err = -5;
			goto fin;
		}
		if (!libraw_internal_data.unpacker_data.cr3_exif_length)
			libraw_internal_data.unpacker_data.cr3_exif_length = unsigned(szAtomContent); 
      parse_exif(oAtomContent);
      order = q_order;
    }
    else if (!strcmp(AtomNameStack, "moovuuidCMT3"))
    {
      short q_order = order;
      order = get2();
      if ((tL != 4) || bad_hdr())
      {
        err = -6;
        goto fin;
      }
      fseek(ifp, -12L, SEEK_CUR);
      parse_makernote(oAtomContent, 0);
      order = q_order;
    }
    else if (!strcmp(AtomNameStack, "moovuuidCMT4"))
    {
      short q_order = order;
      order = get2();
      if ((tL != 4) || bad_hdr())
      {
        err = -6;
        goto fin;
      }
      INT64 off = ftell(ifp);
      parse_gps(oAtomContent);
      fseek(ifp, off, SEEK_SET);
      parse_gps_libraw(oAtomContent);
      order = q_order;
    }
    else if (!strcmp(AtomNameStack, "moovtrakmdiahdlr"))
    {
      fseek(ifp, 8L, SEEK_CUR);
      FORC4 HandlerType[c] = fgetc(ifp);
      for (c = 1; c < int(sizeof sHandlerType / sizeof *sHandlerType); c++)
        if (!strcmp(HandlerType, sHandlerType[c]))
        {
          TrackType = c;
          break;
        }
    }
    else if (!strcmp(AtomNameStack, "moovtrakmdiaminfstblstsd"))
    {
      if (szAtomContent >= 16)
      {
        fseek(ifp, 12L, SEEK_CUR);
        lHdr = 8;
      }
      else
      {
        err = -7;
        goto fin;
      }
      FORC4 MediaFormatID[c] = fgetc(ifp);
      if ((TrackType == 2) && (!strcmp(MediaFormatID, "CRAW")))
      {
        if (szAtomContent >= 44)
          fseek(ifp, 24L, SEEK_CUR);
        else
        {
          err = -8;
          goto fin;
        }
      }
      else
      {
        AtomType = 2; // only continue for CRAW
        lHdr = 0;
      }
#define current_track libraw_internal_data.unpacker_data.crx_header[nTrack]

      /*ImageWidth =*/ get2();
      /*ImageHeight =*/ get2();
    }
    else if (!strcmp(AtomNameStack, "moovtrakmdiaminfstblstsdCRAW"))
    {
      lHdr = 82;
    }
    else if (!strcmp(AtomNameStack, "moovtrakmdiaminfstblstsdCRAWCMP1"))
    {
      int read_size = szAtomContent > 85 ? 85 : szAtomContent;
      if (szAtomContent >= 40)
        fread(CMP1, 1, read_size, ifp);
      else
      {
        err = -7;
        goto fin;
      }
      if (!crxParseImageHeader(CMP1, nTrack, read_size))
        current_track.MediaType = 1;
    }

    else if (!strcmp(AtomNameStack, "moovtrakmdiaminfstblstsdCRAWCDI1")) {
      if (szAtomContent >= 60) {
        fread(CDI1, 1, 60, ifp);
        if (!strncmp((char *)CDI1+8, "IAD1", 4) && (sgetn(8, CDI1) == 0x38)) {
          // sensor area at CDI1+12, 4 16-bit values
          // Bayer pattern? - next 4 16-bit values
          imCanon.RecommendedImageArea = sget_CanonArea(CDI1+12 + 2*4*2);
          imCanon.LeftOpticalBlack     = sget_CanonArea(CDI1+12 + 3*4*2);
          imCanon.UpperOpticalBlack    = sget_CanonArea(CDI1+12 + 4*4*2);
          imCanon.ActiveArea           = sget_CanonArea(CDI1+12 + 5*4*2);
        }
      }
    }

    else if (!strcmp(AtomNameStack, "moovtrakmdiaminfstblstsdCRAWJPEG"))
    {
      current_track.MediaType = 2;
    }
    else if (!strcmp(AtomNameStack, "moovtrakmdiaminfstblstsc"))
    {
      if (szAtomContent >= 12) {
        fseek(ifp, 4L, SEEK_CUR);
        int entries = get4();
        if (entries < 1 || entries > 1000000)
        {
          err =  -9;
          goto fin;
        }

        current_track.stsc_data = (crx_sample_to_chunk_t*) malloc(entries * sizeof(crx_sample_to_chunk_t));
        if(!current_track.stsc_data)
        {
          err =  -9;
          goto fin;
        }
        current_track.stsc_count = entries;
        for(int i = 0; i < entries; i++)
        {
          current_track.stsc_data[i].first = get4();
          current_track.stsc_data[i].count = get4();
          current_track.stsc_data[i].id = get4();
        }
      }
    }
    else if (!strcmp(AtomNameStack, "moovtrakmdiaminfstblstsz"))
    {
      if (szAtomContent >= 12)
      {
        fseek(ifp, 4L, SEEK_CUR);
        int sample_size = get4();
        int entries = get4();
        current_track.sample_count = entries;

        // if sample size is zero sample size is fixed
        if (sample_size)
        {
           current_track.MediaSize = sample_size;
           current_track.sample_size = sample_size;
        }
        else
        {
          current_track.sample_size = 0;
          if (entries < 1 || entries > 1000000) {
            err = -10;
            goto fin;
          }
          current_track.sample_sizes = (int32_t*)malloc(entries * sizeof(int32_t));
          if (!current_track.sample_sizes)
          {
            err = -10;
            goto fin;
          }
          for (int i = 0; i < entries; i++)
            current_track.sample_sizes[i] = get4();

          current_track.MediaSize = current_track.sample_sizes[0];
        }
      }
    }
    else if (!strcmp(AtomNameStack, "moovtrakmdiaminfstblco64"))
    {
      if (szAtomContent >= 16) {
        fseek(ifp, 4L, SEEK_CUR);
        uint32_t entries = get4();
        int i;
        if (entries < 1 || entries > 1000000)
        {
          err = -11;
          goto fin;
        }
        current_track.chunk_offsets = (INT64*)malloc(entries * sizeof(int64_t));
        if(!current_track.chunk_offsets)
        {
          err = -11;
          goto fin;
        }

        current_track.chunk_count = entries;
        for (i = 0; i < entries; i++)
          current_track.chunk_offsets[i] = (((int64_t)get4()) << 32) | get4();

        current_track.chunk_count = i;
        current_track.MediaOffset =  current_track.chunk_offsets[0];
      }
    }

    if (nTrack >= 0 && nTrack < LIBRAW_CRXTRACKS_MAXCOUNT &&
        current_track.MediaSize && current_track.MediaOffset &&
        ((oAtom + szAtom) >= (oAtomList + szAtomList)) &&
        !strncmp(AtomNameStack, "moovtrakmdiaminfstbl", 20))
    {
      if ((TrackType == 4) && (!strcmp(MediaFormatID, "CTMD")))
      {
        current_track.MediaType = 3;
      }
    }
#undef current_track
    if (AtomType == 1)
    {
      err = parseCR3(oAtomContent + lHdr, szAtomContent - lHdr, nesting,
                     AtomNameStack, nTrack, TrackType);
      if (err)
        goto fin;
    }
    oAtom += szAtom;
  }

fin:
  nesting--;
  if (nesting >= 0)
    AtomNameStack[nesting * 4] = '\0';
  order = s_order;
  return err;
}
#undef bad_hdr

void LibRaw::parseCR3_Free()
{
  short maxTrack = libraw_internal_data.unpacker_data.crx_track_count;
  if (maxTrack < 0)
    return;

  for (int i = 0; i <= maxTrack && i < LIBRAW_CRXTRACKS_MAXCOUNT; i++)
  {
    crx_data_header_t *d = &libraw_internal_data.unpacker_data.crx_header[i];
    if (d->stsc_data)
    {
      free(d->stsc_data);
      d->stsc_data = NULL;
    }
    if (d->chunk_offsets)
    {
      free(d->chunk_offsets);
      d->chunk_offsets = NULL;
    }

    if (d->sample_sizes)
    {
      free(d->sample_sizes);
      d->sample_sizes = NULL;
    }
    d->stsc_count   = 0;
    d->sample_count = 0;
    d->sample_size  = 0;
    d->chunk_count  = 0;
  }
  libraw_internal_data.unpacker_data.crx_track_count = -1;
}
