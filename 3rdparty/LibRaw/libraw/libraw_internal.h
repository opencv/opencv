/* -*- C++ -*-
 * File: libraw_internal.h
 * Copyright 2008-2021 LibRaw LLC (info@libraw.org)
 * Created: Sat Mar  8 , 2008
 *
 * LibRaw internal data structures (not visible outside)

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#ifndef _LIBRAW_INTERNAL_TYPES_H
#define _LIBRAW_INTERNAL_TYPES_H

#include <stdio.h>

#ifdef __cplusplus

#include "libraw_datastream.h"
#include "libraw_types.h"

class LibRaw_TLS
{
public:
  struct
  {
    unsigned bitbuf;
    int vbits, reset;
  } getbits;
  struct
  {
    UINT64 bitbuf;
    int vbits;

  } ph1_bits;
  struct
  {
    unsigned pad[128], p;
  } sony_decrypt;
  struct
  {
    uchar buf[0x4002];
    int vpos, padding;
  } pana_data;
  uchar jpeg_buffer[4096];
  struct
  {
    float cbrt[0x10000], xyz_cam[3][4];
  } ahd_data;
  void init()
  {
    getbits.bitbuf = 0;
    getbits.vbits = getbits.reset = 0;
    ph1_bits.bitbuf = 0;
    ph1_bits.vbits = 0;
    pana_data.vpos = 0;
    ahd_data.cbrt[0] = -2.0f;
  }
};

class LibRaw_constants
{
public:
  static const float d65_white[3];
  static const double xyz_rgb[3][3];
  static const double xyzd50_srgb[3][3];
  static const double rgb_rgb[3][3];
  static const double adobe_rgb[3][3];
  static const double wide_rgb[3][3];
  static const double prophoto_rgb[3][3];
  static const double aces_rgb[3][3];
  static const double dcip3d65_rgb[3][3];
  static const double rec2020_rgb[3][3];
};
#endif /* __cplusplus */

typedef struct
{
#ifndef __cplusplus
  struct
#endif
      LibRaw_abstract_datastream *input;
  FILE *output;
  int input_internal;
  char *meta_data;
  INT64 profile_offset;
  INT64 toffset;
  unsigned pana_black[4];

} internal_data_t;

#define LIBRAW_HISTOGRAM_SIZE 0x2000
typedef struct
{
  int (*histogram)[LIBRAW_HISTOGRAM_SIZE];
  unsigned *oprof;
} output_data_t;

typedef struct
{
  unsigned olympus_exif_cfa;
  unsigned long long unique_id;
  unsigned long long OlyID;
  unsigned tiff_nifds;
  int tiff_flip;
  int metadata_blocks;
} identify_data_t;

typedef struct
{
  uint32_t first;
  uint32_t count;
  uint32_t id;
} crx_sample_to_chunk_t;

// contents of tag CMP1 for relevant track in CR3 file
typedef struct
{
  int32_t version;
  int32_t f_width;
  int32_t f_height;
  int32_t tileWidth;
  int32_t tileHeight;
  int32_t nBits;
  int32_t nPlanes;
  int32_t cfaLayout;
  int32_t encType;
  int32_t imageLevels;
  int32_t hasTileCols;
  int32_t hasTileRows;
  int32_t mdatHdrSize;
  int32_t medianBits;
  // Not from header, but from datastream
  uint32_t MediaSize;
  INT64 MediaOffset;
  uint32_t MediaType; /* 1 -> /C/RAW, 2-> JPEG, 3-> CTMD metadata*/
  crx_sample_to_chunk_t * stsc_data; /* samples to chunk */
  uint32_t stsc_count;
  uint32_t sample_count;
  uint32_t sample_size; /* zero if not fixed sample size */
  int32_t *sample_sizes;
  uint32_t chunk_count;
  INT64  *chunk_offsets;
} crx_data_header_t;

typedef struct
{
  short order;
  ushort sraw_mul[4], cr2_slice[3];
  unsigned kodak_cbpp;
  INT64 strip_offset, data_offset;
  INT64 meta_offset;
  INT64 exif_offset, exif_subdir_offset, ifd0_offset;
  unsigned data_size;
  unsigned meta_length;
  unsigned cr3_exif_length, cr3_ifd0_length;
  unsigned thumb_misc;
  enum LibRaw_internal_thumbnail_formats thumb_format;
  unsigned fuji_layout;
  unsigned tiff_samples;
  unsigned tiff_bps;
  unsigned tiff_compress;
  unsigned tiff_sampleformat;
  unsigned zero_after_ff;
  unsigned tile_width, tile_length, load_flags;
  unsigned data_error;
  int hasselblad_parser_flag;
  long long posRAFData;
  unsigned lenRAFData;
  int fuji_total_lines, fuji_total_blocks, fuji_block_width, fuji_bits,
      fuji_raw_type, fuji_lossless;
  int pana_encoding, pana_bpp;
  crx_data_header_t crx_header[LIBRAW_CRXTRACKS_MAXCOUNT];
  int crx_track_selected;
  int crx_track_count;
  short CR3_CTMDtag;
  short CR3_Version;
  int CM_found;
  unsigned is_NikonTransfer;
  unsigned is_Olympus;
  int OlympusDNG_SubDirOffsetValid;
  unsigned is_Sony;
  unsigned is_pana_raw;
  unsigned is_PentaxRicohMakernotes; /* =1 for Ricoh software by Pentax, Camera DNG */

  unsigned dng_frames[LIBRAW_IFD_MAXCOUNT*2]; /* bits: 0-7: shot_select, 8-15: IFD#, 16-31: low 16 bit of newsubfile type */
  unsigned short raw_stride;
} unpacker_data_t;

typedef struct
{
  internal_data_t internal_data;
  libraw_internal_output_params_t internal_output_params;
  output_data_t output_data;
  identify_data_t identify_data;
  unpacker_data_t unpacker_data;
} libraw_internal_data_t;

struct decode
{
  struct decode *branch[2];
  int leaf;
};

struct tiff_ifd_t
{
  int t_width, t_height, bps, comp, phint, offset, t_flip, samples, bytes, extrasamples;
  int t_tile_width, t_tile_length, sample_format, predictor;
  int rows_per_strip;
  int *strip_offsets, strip_offsets_count;
  int *strip_byte_counts, strip_byte_counts_count;
  unsigned t_filters;
  int t_vwidth, t_vheight, t_lm,t_tm;
  int t_fuji_width;
  float t_shutter;
  /* Per-IFD DNG fields */
  INT64 opcode2_offset;
  INT64 lineartable_offset;
  int lineartable_len;
  libraw_dng_color_t dng_color[2];
  libraw_dng_levels_t dng_levels;
  int newsubfiletype;
};

struct jhead
{
  int algo, bits, high, wide, clrs, sraw, psv, restart, vpred[6];
  ushort quant[64], idct[64], *huff[20], *free[20], *row;
};

struct libraw_tiff_tag
{
  ushort tag, type;
  int count;
  union {
    char c[4];
    short s[2];
    int i;
  } val;
};

struct tiff_hdr
{
  ushort t_order, magic;
  int ifd;
  ushort pad, ntag;
  struct libraw_tiff_tag tag[23];
  int nextifd;
  ushort pad2, nexif;
  struct libraw_tiff_tag exif[4];
  ushort pad3, ngps;
  struct libraw_tiff_tag gpst[10];
  short bps[4];
  int rat[10];
  unsigned gps[26];
  char t_desc[512], t_make[64], t_model[64], soft[32], date[20], t_artist[64];
};

#ifdef DEBUG_STAGE_CHECKS
#define CHECK_ORDER_HIGH(expected_stage)                                       \
  do                                                                           \
  {                                                                            \
    if ((imgdata.progress_flags & LIBRAW_PROGRESS_THUMB_MASK) >=               \
        expected_stage)                                                        \
    {                                                                          \
      fprintf(stderr, "CHECK_HIGH: check %d >=  %d\n",                         \
              imgdata.progress_flags &LIBRAW_PROGRESS_THUMB_MASK,              \
              expected_stage);                                                 \
      return LIBRAW_OUT_OF_ORDER_CALL;                                         \
    }                                                                          \
  } while (0)

#define CHECK_ORDER_LOW(expected_stage)                                        \
  do                                                                           \
  {                                                                            \
    printf("Checking LOW %d/%d : %d\n", imgdata.progress_flags,                \
           expected_stage, imgdata.progress_flags < expected_stage);           \
    if ((imgdata.progress_flags & LIBRAW_PROGRESS_THUMB_MASK) <                \
        expected_stage)                                                        \
    {                                                                          \
      printf("failed!\n");                                                     \
      return LIBRAW_OUT_OF_ORDER_CALL;                                         \
    }                                                                          \
  } while (0)
#define CHECK_ORDER_BIT(expected_stage)                                        \
  do                                                                           \
  {                                                                            \
    if (imgdata.progress_flags & expected_stage)                               \
      return LIBRAW_OUT_OF_ORDER_CALL;                                         \
  } while (0)

#define SET_PROC_FLAG(stage)                                                   \
  do                                                                           \
  {                                                                            \
    imgdata.progress_flags |= stage;                                           \
    fprintf(stderr, "SET_FLAG: %d\n", stage);                                  \
  } while (0)

#else

#define CHECK_ORDER_HIGH(expected_stage)                                       \
  do                                                                           \
  {                                                                            \
    if ((imgdata.progress_flags & LIBRAW_PROGRESS_THUMB_MASK) >=               \
        expected_stage)                                                        \
    {                                                                          \
      return LIBRAW_OUT_OF_ORDER_CALL;                                         \
    }                                                                          \
  } while (0)

#define CHECK_ORDER_LOW(expected_stage)                                        \
  do                                                                           \
  {                                                                            \
    if ((imgdata.progress_flags & LIBRAW_PROGRESS_THUMB_MASK) <                \
        expected_stage)                                                        \
      return LIBRAW_OUT_OF_ORDER_CALL;                                         \
  } while (0)

#define CHECK_ORDER_BIT(expected_stage)                                        \
  do                                                                           \
  {                                                                            \
    if (imgdata.progress_flags & expected_stage)                               \
      return LIBRAW_OUT_OF_ORDER_CALL;                                         \
  } while (0)

#define SET_PROC_FLAG(stage)                                                   \
  do                                                                           \
  {                                                                            \
    imgdata.progress_flags |= stage;                                           \
  } while (0)

#endif

#endif
