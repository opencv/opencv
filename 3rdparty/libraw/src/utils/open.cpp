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

#include "../../internal/libraw_cxx_defs.h"
#include "../../internal/libraw_cameraids.h"

#ifndef LIBRAW_NO_IOSTREAMS_DATASTREAM
int LibRaw::open_file(const char *fname, INT64 max_buf_size)
{
	int big = 0;
	if (max_buf_size == LIBRAW_OPEN_BIGFILE)
		big = 1;
	else if (max_buf_size == LIBRAW_OPEN_FILE)
		big = 0;
	else
	{
#ifndef LIBRAW_WIN32_CALLS
		struct stat st;
		if (stat(fname, &st))
			return LIBRAW_IO_ERROR;
		big = (st.st_size > max_buf_size) ? 1 : 0;
#else
		struct _stati64 st;
		if (_stati64(fname, &st))
			return LIBRAW_IO_ERROR;
		big = (st.st_size > max_buf_size) ? 1 : 0;
#endif
	}

  LibRaw_abstract_datastream *stream;
  try
  {
    if (big)
      stream = new LibRaw_bigfile_datastream(fname);
    else
      stream = new LibRaw_file_datastream(fname);
  }

  catch (const std::bad_alloc& )
  {
    recycle();
    return LIBRAW_UNSUFFICIENT_MEMORY;
  }
  if (!stream->valid())
  {
    delete stream;
    return LIBRAW_IO_ERROR;
  }
  ID.input_internal = 0; // preserve from deletion on error
  int ret = open_datastream(stream);
  if (ret == LIBRAW_SUCCESS)
  {
    ID.input_internal = 1; // flag to delete datastream on recycle
  }
  else
  {
    delete stream;
    ID.input_internal = 0;
  }
  return ret;
}

#if defined(WIN32) || defined(_WIN32)
#ifndef LIBRAW_WIN32_UNICODEPATHS
int LibRaw::open_file(const wchar_t *, INT64)
{
  return LIBRAW_NOT_IMPLEMENTED;
}
#else
int LibRaw::open_file(const wchar_t *fname, INT64 max_buf_size)
{
	int big = 0;
	if (max_buf_size == LIBRAW_OPEN_BIGFILE)
		big = 1;
	else if (max_buf_size == LIBRAW_OPEN_FILE)
		big = 0;
	else
	{
		struct _stati64 st;
		if (_wstati64(fname, &st))
			return LIBRAW_IO_ERROR;
		big = (st.st_size > max_buf_size) ? 1 : 0;
	}

  LibRaw_abstract_datastream *stream;
  try
  {
    if (big)
      stream = new LibRaw_bigfile_datastream(fname);
    else
      stream = new LibRaw_file_datastream(fname);
  }

  catch (const std::bad_alloc&)
  {
    recycle();
    return LIBRAW_UNSUFFICIENT_MEMORY;
  }
  if (!stream->valid())
  {
    delete stream;
    return LIBRAW_IO_ERROR;
  }
  ID.input_internal = 0; // preserve from deletion on error
  int ret = open_datastream(stream);
  if (ret == LIBRAW_SUCCESS)
  {
    ID.input_internal = 1; // flag to delete datastream on recycle
  }
  else
  {
    delete stream;
    ID.input_internal = 0;
  }
  return ret;
}
#endif
#endif

#else /* LIBRAW_NO_IOSTREAMS_DATASTREAM*/

int LibRaw::libraw_openfile_tail(LibRaw_abstract_datastream *stream)
{
    if (!stream->valid())
    {
        delete stream;
        return LIBRAW_IO_ERROR;
    }
    ID.input_internal = 0; // preserve from deletion on error
    int ret = open_datastream(stream);
    if (ret == LIBRAW_SUCCESS)
    {
        ID.input_internal = 1; // flag to delete datastream on recycle
    }
    else
    {
        delete stream;
        ID.input_internal = 0;
    }
    return ret;
}

int LibRaw::open_file(const char *fname)
{
    LibRaw_abstract_datastream *stream;
    try
    {
#ifdef LIBRAW_WIN32_CALLS
        stream = new LibRaw_bigfile_buffered_datastream(fname);
#else
        stream = new LibRaw_bigfile_datastream(fname);
#endif
    }
    catch (const std::bad_alloc&)
    {
        recycle();
        return LIBRAW_UNSUFFICIENT_MEMORY;
    }
    if ((stream->size() > (INT64)LIBRAW_MAX_NONDNG_RAW_FILE_SIZE) && (stream->size() > (INT64)LIBRAW_MAX_DNG_RAW_FILE_SIZE))
    {
      delete stream;
      return LIBRAW_TOO_BIG;
    }
    return libraw_openfile_tail(stream);
}

#if defined(WIN32) || defined(_WIN32)
#ifndef LIBRAW_WIN32_UNICODEPATHS
int LibRaw::open_file(const wchar_t *)
{
    return LIBRAW_NOT_IMPLEMENTED;
}
#else
int LibRaw::open_file(const wchar_t *fname)
{
    LibRaw_abstract_datastream *stream;
    try
    {
#ifdef LIBRAW_WIN32_CALLS
        stream = new LibRaw_bigfile_buffered_datastream(fname);
#else
        stream = new LibRaw_bigfile_datastream(fname);
#endif
    }
    catch (const std::bad_alloc&)
    {
        recycle();
        return LIBRAW_UNSUFFICIENT_MEMORY;
    }
    if ((stream->size() > (INT64)LIBRAW_MAX_DNG_RAW_FILE_SIZE) && (stream->size() > (INT64)LIBRAW_MAX_NONDNG_RAW_FILE_SIZE))
    {
      delete stream;
      return LIBRAW_TOO_BIG;
    }

    return libraw_openfile_tail(stream);
}
#endif
#endif

#endif

int LibRaw::open_buffer(const void *buffer, size_t size)
{
  // this stream will close on recycle()
  if (!buffer || buffer == (const void *)-1)
    return LIBRAW_IO_ERROR;

  if ((size > (INT64)LIBRAW_MAX_DNG_RAW_FILE_SIZE) && (size > (INT64)LIBRAW_MAX_NONDNG_RAW_FILE_SIZE))   
      return LIBRAW_TOO_BIG;

  LibRaw_buffer_datastream *stream;
  try
  {
    stream = new LibRaw_buffer_datastream(buffer, size);
  }
  catch (const std::bad_alloc& )
  {
    recycle();
    return LIBRAW_UNSUFFICIENT_MEMORY;
  }
  if (!stream->valid())
  {
    delete stream;
    return LIBRAW_IO_ERROR;
  }
  ID.input_internal = 0; // preserve from deletion on error
  int ret = open_datastream(stream);
  if (ret == LIBRAW_SUCCESS)
  {
    ID.input_internal = 1; // flag to delete datastream on recycle
  }
  else
  {
    delete stream;
    ID.input_internal = 0;
  }
  return ret;
}

int LibRaw::open_bayer(const unsigned char *buffer, unsigned datalen,
                       ushort _raw_width, ushort _raw_height,
                       ushort _left_margin, ushort _top_margin,
                       ushort _right_margin, ushort _bottom_margin,
                       unsigned char procflags, unsigned char bayer_pattern,
                       unsigned unused_bits, unsigned otherflags,
                       unsigned black_level)
{
  // this stream will close on recycle()
  if (!buffer || buffer == (const void *)-1)
    return LIBRAW_IO_ERROR;

  LibRaw_buffer_datastream *stream;
  try
  {
    stream = new LibRaw_buffer_datastream(buffer, datalen);
  }
  catch (const std::bad_alloc& )
  {
    recycle();
    return LIBRAW_UNSUFFICIENT_MEMORY;
  }
  if (!stream->valid())
  {
    delete stream;
    return LIBRAW_IO_ERROR;
  }
  ID.input = stream;
  SET_PROC_FLAG(LIBRAW_PROGRESS_OPEN);
  // From identify
  initdata();
  strcpy(imgdata.idata.make, "BayerDump");
  snprintf(imgdata.idata.model, sizeof(imgdata.idata.model) - 1,
           "%u x %u pixels", _raw_width, _raw_height);
  S.flip = procflags >> 2;
  libraw_internal_data.internal_output_params.zero_is_bad = procflags & 2;
  libraw_internal_data.unpacker_data.data_offset = 0;
  S.raw_width = _raw_width;
  S.raw_height = _raw_height;
  S.left_margin = _left_margin;
  S.top_margin = _top_margin;
  S.width = S.raw_width - S.left_margin - _right_margin;
  S.height = S.raw_height - S.top_margin - _bottom_margin;

  imgdata.idata.filters = 0x1010101 * bayer_pattern;
  imgdata.idata.colors =
      4 - !((imgdata.idata.filters & imgdata.idata.filters >> 1) & 0x5555);
  libraw_internal_data.unpacker_data.load_flags = otherflags;
  switch (libraw_internal_data.unpacker_data.tiff_bps =
              (datalen)*8 / (S.raw_width * S.raw_height))
  {
  case 8:
    load_raw = &LibRaw::eight_bit_load_raw;
    break;
  case 10:
    if ((datalen) / S.raw_height * 3u >= S.raw_width * 4u)
    {
      load_raw = &LibRaw::android_loose_load_raw;
      break;
    }
    else if (libraw_internal_data.unpacker_data.load_flags & 1)
    {
      load_raw = &LibRaw::android_tight_load_raw;
      break;
    }
  case 12:
    libraw_internal_data.unpacker_data.load_flags |= 128;
    load_raw = &LibRaw::packed_load_raw;
    break;
  case 16:
    libraw_internal_data.unpacker_data.order =
        0x4949 | 0x404 * (libraw_internal_data.unpacker_data.load_flags & 1);
    libraw_internal_data.unpacker_data.tiff_bps -=
        libraw_internal_data.unpacker_data.load_flags >> 4;
    libraw_internal_data.unpacker_data.tiff_bps -=
        libraw_internal_data.unpacker_data.load_flags =
            libraw_internal_data.unpacker_data.load_flags >> 1 & 7;
    load_raw = &LibRaw::unpacked_load_raw;
  }
  C.maximum =
      (1 << libraw_internal_data.unpacker_data.tiff_bps) - (1 << unused_bits);
  C.black = black_level;
  S.iwidth = S.width;
  S.iheight = S.height;
  imgdata.idata.colors = 3;
  imgdata.idata.filters |= ((imgdata.idata.filters >> 2 & 0x22222222) |
                            (imgdata.idata.filters << 2 & 0x88888888)) &
                           imgdata.idata.filters << 1;

  imgdata.idata.raw_count = 1;
  for (int i = 0; i < 4; i++)
    imgdata.color.pre_mul[i] = 1.0;

  strcpy(imgdata.idata.cdesc, "RGBG");

  ID.input_internal = 1;
  SET_PROC_FLAG(LIBRAW_PROGRESS_IDENTIFY);
  return LIBRAW_SUCCESS;
}

struct foveon_data_t
{
  const char *make;
  const char *model;
  const int raw_width, raw_height;
  const int white;
  const int left_margin, top_margin;
  const int width, height;
} foveon_data[] = {
    {"Sigma", "SD9", 2304, 1531, 12000, 20, 8, 2266, 1510},
    {"Sigma", "SD9", 1152, 763, 12000, 10, 2, 1132, 755},
    {"Sigma", "SD10", 2304, 1531, 12000, 20, 8, 2266, 1510},
    {"Sigma", "SD10", 1152, 763, 12000, 10, 2, 1132, 755},
    {"Sigma", "SD14", 2688, 1792, 14000, 18, 12, 2651, 1767},
    {"Sigma", "SD14", 2688, 896, 14000, 18, 6, 2651, 883}, // 2/3
    {"Sigma", "SD14", 1344, 896, 14000, 9, 6, 1326, 883},  // 1/2
    {"Sigma", "SD15", 2688, 1792, 2900, 18, 12, 2651, 1767},
    {"Sigma", "SD15", 2688, 896, 2900, 18, 6, 2651, 883}, // 2/3 ?
    {"Sigma", "SD15", 1344, 896, 2900, 9, 6, 1326, 883},  // 1/2 ?
    {"Sigma", "DP1", 2688, 1792, 2100, 18, 12, 2651, 1767},
    {"Sigma", "DP1", 2688, 896, 2100, 18, 6, 2651, 883}, // 2/3 ?
    {"Sigma", "DP1", 1344, 896, 2100, 9, 6, 1326, 883},  // 1/2 ?
    {"Sigma", "DP1S", 2688, 1792, 2200, 18, 12, 2651, 1767},
    {"Sigma", "DP1S", 2688, 896, 2200, 18, 6, 2651, 883}, // 2/3
    {"Sigma", "DP1S", 1344, 896, 2200, 9, 6, 1326, 883},  // 1/2
    {"Sigma", "DP1X", 2688, 1792, 3560, 18, 12, 2651, 1767},
    {"Sigma", "DP1X", 2688, 896, 3560, 18, 6, 2651, 883}, // 2/3
    {"Sigma", "DP1X", 1344, 896, 3560, 9, 6, 1326, 883},  // 1/2
    {"Sigma", "DP2", 2688, 1792, 2326, 13, 16, 2651, 1767},
    {"Sigma", "DP2", 2688, 896, 2326, 13, 8, 2651, 883}, // 2/3 ??
    {"Sigma", "DP2", 1344, 896, 2326, 7, 8, 1325, 883},  // 1/2 ??
    {"Sigma", "DP2S", 2688, 1792, 2300, 18, 12, 2651, 1767},
    {"Sigma", "DP2S", 2688, 896, 2300, 18, 6, 2651, 883}, // 2/3
    {"Sigma", "DP2S", 1344, 896, 2300, 9, 6, 1326, 883},  // 1/2
    {"Sigma", "DP2X", 2688, 1792, 2300, 18, 12, 2651, 1767},
    {"Sigma", "DP2X", 2688, 896, 2300, 18, 6, 2651, 883},           // 2/3
    {"Sigma", "DP2X", 1344, 896, 2300, 9, 6, 1325, 883},            // 1/2
    {"Sigma", "SD1", 4928, 3264, 3900, 12, 52, 4807, 3205},         // Full size
    {"Sigma", "SD1", 4928, 1632, 3900, 12, 26, 4807, 1603},         // 2/3 size
    {"Sigma", "SD1", 2464, 1632, 3900, 6, 26, 2403, 1603},          // 1/2 size
    {"Sigma", "SD1 Merrill", 4928, 3264, 3900, 12, 52, 4807, 3205}, // Full size
    {"Sigma", "SD1 Merrill", 4928, 1632, 3900, 12, 26, 4807, 1603}, // 2/3 size
    {"Sigma", "SD1 Merrill", 2464, 1632, 3900, 6, 26, 2403, 1603},  // 1/2 size
    {"Sigma", "DP1 Merrill", 4928, 3264, 3900, 12, 0, 4807, 3205},
    {"Sigma", "DP1 Merrill", 2464, 1632, 3900, 12, 0, 2403, 1603}, // 1/2 size
    {"Sigma", "DP1 Merrill", 4928, 1632, 3900, 12, 0, 4807, 1603}, // 2/3 size
    {"Sigma", "DP2 Merrill", 4928, 3264, 3900, 12, 0, 4807, 3205},
    {"Sigma", "DP2 Merrill", 2464, 1632, 3900, 12, 0, 2403, 1603}, // 1/2 size
    {"Sigma", "DP2 Merrill", 4928, 1632, 3900, 12, 0, 4807, 1603}, // 2/3 size
    {"Sigma", "DP3 Merrill", 4928, 3264, 3900, 12, 0, 4807, 3205},
    {"Sigma", "DP3 Merrill", 2464, 1632, 3900, 12, 0, 2403, 1603}, // 1/2 size
    {"Sigma", "DP3 Merrill", 4928, 1632, 3900, 12, 0, 4807, 1603}, // 2/3 size
    {"Polaroid", "x530", 1440, 1088, 2700, 10, 13, 1419, 1059},
    // dp2 Q
    {"Sigma", "dp3 Quattro", 5888, 3776, 16383, 204, 76, 5446,
     3624}, // full size, new fw ??
    {"Sigma", "dp3 Quattro", 5888, 3672, 16383, 204, 24, 5446,
     3624}, // full size
    {"Sigma", "dp3 Quattro", 2944, 1836, 16383, 102, 12, 2723,
     1812}, // half size
    {"Sigma", "dp3 Quattro", 2944, 1888, 16383, 102, 38, 2723,
     1812}, // half size, new fw??

    {"Sigma", "dp2 Quattro", 5888, 3776, 16383, 204, 76, 5446,
     3624}, // full size, new fw
    {"Sigma", "dp2 Quattro", 5888, 3672, 16383, 204, 24, 5446,
     3624}, // full size
    {"Sigma", "dp2 Quattro", 2944, 1836, 16383, 102, 12, 2723,
     1812}, // half size
    {"Sigma", "dp2 Quattro", 2944, 1888, 16383, 102, 38, 2723,
     1812}, // half size, new fw

    {"Sigma", "dp1 Quattro", 5888, 3776, 16383, 204, 76, 5446,
     3624}, // full size, new fw??
    {"Sigma", "dp1 Quattro", 5888, 3672, 16383, 204, 24, 5446,
     3624}, // full size
    {"Sigma", "dp1 Quattro", 2944, 1836, 16383, 102, 12, 2723,
     1812}, // half size
    {"Sigma", "dp1 Quattro", 2944, 1888, 16383, 102, 38, 2723,
     1812}, // half size, new fw

    {"Sigma", "dp0 Quattro", 5888, 3776, 16383, 204, 76, 5446,
     3624}, // full size, new fw??
    {"Sigma", "dp0 Quattro", 5888, 3672, 16383, 204, 24, 5446,
     3624}, // full size
    {"Sigma", "dp0 Quattro", 2944, 1836, 16383, 102, 12, 2723,
     1812}, // half size
    {"Sigma", "dp0 Quattro", 2944, 1888, 16383, 102, 38, 2723,
     1812}, // half size, new fw
    // Sigma sd Quattro
    {"Sigma", "sd Quattro", 5888, 3776, 16383, 204, 76, 5446,
     3624}, // full size
    {"Sigma", "sd Quattro", 2944, 1888, 16383, 102, 38, 2723,
     1812}, // half size
    // Sd Quattro H
    {"Sigma", "sd Quattro H", 6656, 4480, 4000, 224, 160, 6208,
     4160}, // full size
    {"Sigma", "sd Quattro H", 3328, 2240, 4000, 112, 80, 3104,
     2080},                                                        // half size
    {"Sigma", "sd Quattro H", 5504, 3680, 4000, 0, 4, 5496, 3668}, // full size
    {"Sigma", "sd Quattro H", 2752, 1840, 4000, 0, 2, 2748, 1834}, // half size
};
const int foveon_count = sizeof(foveon_data) / sizeof(foveon_data[0]);

int LibRaw::open_datastream(LibRaw_abstract_datastream *stream)
{

  if (!stream)
    return ENOENT;
  if (!stream->valid())
    return LIBRAW_IO_ERROR;
  if ((stream->size() > (INT64)LIBRAW_MAX_DNG_RAW_FILE_SIZE) && (stream->size() > (INT64)LIBRAW_MAX_NONDNG_RAW_FILE_SIZE))
      return LIBRAW_TOO_BIG;

  recycle();
  if (callbacks.pre_identify_cb)
  {
    int r = (callbacks.pre_identify_cb)(this);
    if (r == 1)
      goto final;
  }

  try
  {
	  ID.input = stream;
	  SET_PROC_FLAG(LIBRAW_PROGRESS_OPEN);

	  identify();

	  // Fuji layout files: either DNG or unpacked_load_raw should be used
	  if (libraw_internal_data.internal_output_params.fuji_width || libraw_internal_data.unpacker_data.fuji_layout)
	  {
        if (!imgdata.idata.dng_version && load_raw != &LibRaw::unpacked_load_raw
			&& load_raw != &LibRaw::unpacked_load_raw_FujiDBP 
			&& load_raw != &LibRaw::unpacked_load_raw_fuji_f700s20
			)
          return LIBRAW_FILE_UNSUPPORTED;
	  }

	  // promote the old single thumbnail to the thumbs_list if not present already
	  if (imgdata.thumbs_list.thumbcount < LIBRAW_THUMBNAIL_MAXCOUNT)
	  {
		  bool already = false;
		  if(imgdata.thumbnail.tlength || libraw_internal_data.internal_data.toffset)
			  for(int i = 0; i < imgdata.thumbs_list.thumbcount; i++)
				  if (imgdata.thumbs_list.thumblist[i].toffset == libraw_internal_data.internal_data.toffset
					  && imgdata.thumbs_list.thumblist[i].tlength == imgdata.thumbnail.tlength)
				  {
					  already = true;
					  break;
				  }
		  if (!already)
		  {
			  int idx = imgdata.thumbs_list.thumbcount;
			  imgdata.thumbs_list.thumblist[idx].toffset = libraw_internal_data.internal_data.toffset;
			  imgdata.thumbs_list.thumblist[idx].tlength = imgdata.thumbnail.tlength;
			  imgdata.thumbs_list.thumblist[idx].tflip = 0xffff;
			  imgdata.thumbs_list.thumblist[idx].tformat = libraw_internal_data.unpacker_data.thumb_format;
              imgdata.thumbs_list.thumblist[idx].tmisc = libraw_internal_data.unpacker_data.thumb_misc;
			  // promote if set
			  imgdata.thumbs_list.thumblist[idx].twidth = imgdata.thumbnail.twidth;
              imgdata.thumbs_list.thumblist[idx].theight = imgdata.thumbnail.theight;
			  imgdata.thumbs_list.thumbcount++;
		  }
	  }


	  imgdata.lens.Lens[sizeof(imgdata.lens.Lens) - 1] = 0; // make sure lens is 0-terminated

	  if (callbacks.post_identify_cb)
		  (callbacks.post_identify_cb)(this);

#define isRIC imgdata.sizes.raw_inset_crops[0]

	  if (!imgdata.idata.dng_version && makeIs(LIBRAW_CAMERAMAKER_Fujifilm)
		  && (!strcmp(imgdata.idata.normalized_model, "S3Pro")
			  || !strcmp(imgdata.idata.normalized_model, "S5Pro")
			  || !strcmp(imgdata.idata.normalized_model, "S2Pro")))
	  {
		  isRIC.cleft = isRIC.ctop = 0xffff;
		  isRIC.cwidth = isRIC.cheight = 0;
	  }
      // Wipe out canon  incorrect in-camera crop
      if (!imgdata.idata.dng_version && makeIs(LIBRAW_CAMERAMAKER_Canon)
          && isRIC.cleft == 0 && isRIC.ctop == 0 // non symmetric!
          && isRIC.cwidth < (imgdata.sizes.raw_width * 4 / 5))  // less than 80% of sensor width
      {
        isRIC.cleft = isRIC.ctop = 0xffff;
        isRIC.cwidth = isRIC.cheight = 0;
      }

      // Wipe out non-standard WB
      if (!imgdata.idata.dng_version &&
          (makeIs(LIBRAW_CAMERAMAKER_Sony) && !strcmp(imgdata.idata.normalized_model, "DSC-F828"))
          && !(imgdata.rawparams.options & LIBRAW_RAWOPTIONS_PROVIDE_NONSTANDARD_WB))
      {
          for (int i = 0; i < 4; i++) imgdata.color.cam_mul[i] = (i == 1);
          memset(imgdata.color.WB_Coeffs, 0, sizeof(imgdata.color.WB_Coeffs));
          memset(imgdata.color.WBCT_Coeffs, 0, sizeof(imgdata.color.WBCT_Coeffs));
      }

	  if (load_raw == &LibRaw::nikon_load_raw)
		  nikon_read_curve();

	  if (load_raw == &LibRaw::lossless_jpeg_load_raw &&
		  MN.canon.RecordMode && makeIs(LIBRAW_CAMERAMAKER_Kodak) &&
		  /* Not normalized models here, it is intentional */
		  (!strncasecmp(imgdata.idata.model, "EOS D2000", 9) || // if we want something different for B&W cameras,
			  !strncasecmp(imgdata.idata.model, "EOS D6000", 9)))  // it's better to compare with CamIDs
	  {
		  imgdata.color.black = 0;
		  imgdata.color.maximum = 4501;
		  memset(imgdata.color.cblack, 0, sizeof(imgdata.color.cblack));
		  memset(imgdata.sizes.mask, 0, sizeof(imgdata.sizes.mask));
		  imgdata.sizes.mask[0][3] = 1; // to skip mask re-calc
		  libraw_internal_data.unpacker_data.load_flags |= 512;
	  }

	  if (load_raw == &LibRaw::panasonic_load_raw)
	  {
		  if (libraw_internal_data.unpacker_data.pana_encoding == 6 ||
			  libraw_internal_data.unpacker_data.pana_encoding == 7)
		  {
			  for (int i = 0; i < 3; i++)
				  imgdata.color.cblack[i] =
				  libraw_internal_data.internal_data.pana_black[i];
			  imgdata.color.cblack[3] = imgdata.color.cblack[1];
			  imgdata.color.cblack[4] = imgdata.color.cblack[5] = 0;
			  imgdata.color.black = 0;
			  imgdata.color.maximum =
				  MAX(imgdata.color.linear_max[0],
					  MAX(imgdata.color.linear_max[1], imgdata.color.linear_max[2]));
		  }

		  if (libraw_internal_data.unpacker_data.pana_encoding == 6)
		  {
			  int rowbytes11 = imgdata.sizes.raw_width / 11 * 16;
              int rowbytes14 = imgdata.sizes.raw_width / 14 * 16;
              INT64 ds = INT64(libraw_internal_data.unpacker_data.data_size);
              if (!ds)
                  ds = libraw_internal_data.internal_data.input->size() - libraw_internal_data.unpacker_data.data_offset;
              if ((imgdata.sizes.raw_width % 11) == 0 &&
				  (INT64(imgdata.sizes.raw_height) * rowbytes11 == ds))
				  load_raw = &LibRaw::panasonicC6_load_raw;
              else if ((imgdata.sizes.raw_width % 14) == 0 &&
                (INT64(imgdata.sizes.raw_height) * rowbytes14 == ds))
                  load_raw = &LibRaw::panasonicC6_load_raw;
              else
				  imgdata.idata.raw_count = 0; // incorrect size
		  }
		  else if (libraw_internal_data.unpacker_data.pana_encoding == 7)
		  {
			  int pixperblock =
				  libraw_internal_data.unpacker_data.pana_bpp == 14 ? 9 : 10;
			  int rowbytes = imgdata.sizes.raw_width / pixperblock * 16;
			  if ((imgdata.sizes.raw_width % pixperblock) == 0 &&
				  (INT64(imgdata.sizes.raw_height) * rowbytes ==
					  INT64(libraw_internal_data.unpacker_data.data_size)))
				  load_raw = &LibRaw::panasonicC7_load_raw;
			  else
				  imgdata.idata.raw_count = 0; // incorrect size
		  }
	  }

#define NIKON_14BIT_SIZE(rw, rh)                                               \
  (((unsigned)(ceilf((float)(rw * 7 / 4) / 16.0)) * 16) * rh)

	  // Ugly hack, replace with proper data/line size for different
	  // cameras/format when available
	  if (makeIs(LIBRAW_CAMERAMAKER_Nikon)
		  && (!strncasecmp(imgdata.idata.model, "Z", 1) || !strcasecmp(imgdata.idata.model,"D6"))
		  &&  NIKON_14BIT_SIZE(imgdata.sizes.raw_width, imgdata.sizes.raw_height) ==
		  libraw_internal_data.unpacker_data.data_size)
	  {
		  load_raw = &LibRaw::nikon_14bit_load_raw;
	  }
#undef NIKON_14BIT_SIZE

	  // Linear max from 14-bit camera, but on 12-bit data?
	  if (makeIs(LIBRAW_CAMERAMAKER_Sony) &&
		  imgdata.color.maximum > 0 &&
		  imgdata.color.linear_max[0] > (long)imgdata.color.maximum &&
		  imgdata.color.linear_max[0] <= (long)imgdata.color.maximum * 4)
		  for (int c = 0; c < 4; c++)
			  imgdata.color.linear_max[c] /= 4;

	  if (makeIs(LIBRAW_CAMERAMAKER_Canon))
	  {
		  if (MN.canon.DefaultCropAbsolute.l != -1)  // tag 0x00e0 SensorInfo was parsed
		  {
			  if (imgdata.sizes.raw_aspect != LIBRAW_IMAGE_ASPECT_UNKNOWN)
			  { // tag 0x009a AspectInfo was parsed
				  isRIC.cleft += MN.canon.DefaultCropAbsolute.l;
				  isRIC.ctop  += MN.canon.DefaultCropAbsolute.t;
			  }
			  else
			  {
				  isRIC.cleft   = MN.canon.DefaultCropAbsolute.l;
				  isRIC.ctop    = MN.canon.DefaultCropAbsolute.t;
				  isRIC.cwidth  = MN.canon.DefaultCropAbsolute.r - MN.canon.DefaultCropAbsolute.l + 1;
				  isRIC.cheight = MN.canon.DefaultCropAbsolute.b - MN.canon.DefaultCropAbsolute.t + 1;
			  }
		  }
		  else
		  {
			  if (imgdata.sizes.raw_aspect != LIBRAW_IMAGE_ASPECT_UNKNOWN)
			  {
			  }
			  else
			  { // Canon PowerShot S2 IS
			  }
		  }
#undef isRIC
          if (imgdata.color.raw_bps < 14 && !imgdata.idata.dng_version && load_raw != &LibRaw::canon_sraw_load_raw)
          {
              int xmax = (1 << imgdata.color.raw_bps) - 1;
              if (MN.canon.SpecularWhiteLevel > xmax) // Adjust 14-bit metadata to real bps
              {
                int div = 1 << (14 - imgdata.color.raw_bps);
                for (int c = 0; c < 4; c++) imgdata.color.linear_max[c] /= div;
                for (int c = 0; c < 4; c++)  MN.canon.ChannelBlackLevel[c] /= div;
                MN.canon.AverageBlackLevel /= div;
                MN.canon.SpecularWhiteLevel /= div;
                MN.canon.NormalWhiteLevel /= div;
              }
          }
	  }

	  if (makeIs(LIBRAW_CAMERAMAKER_Canon) &&
		  (load_raw == &LibRaw::canon_sraw_load_raw) &&
		  imgdata.sizes.raw_width > 0)
	  {
		  float ratio =
			  float(imgdata.sizes.raw_height) / float(imgdata.sizes.raw_width);
		  if ((ratio < 0.57 || ratio > 0.75) &&
			  MN.canon.SensorHeight > 1 &&
			  MN.canon.SensorWidth > 1)
		  {
			  imgdata.sizes.raw_width = MN.canon.SensorWidth;
			  imgdata.sizes.left_margin = MN.canon.DefaultCropAbsolute.l;
			  imgdata.sizes.iwidth = imgdata.sizes.width =
				  MN.canon.DefaultCropAbsolute.r - MN.canon.DefaultCropAbsolute.l + 1;
			  imgdata.sizes.raw_height = MN.canon.SensorHeight;
			  imgdata.sizes.top_margin = MN.canon.DefaultCropAbsolute.t;
			  imgdata.sizes.iheight = imgdata.sizes.height =
				  MN.canon.DefaultCropAbsolute.b - MN.canon.DefaultCropAbsolute.t + 1;
			  libraw_internal_data.unpacker_data.load_flags |=
				  256; // reset width/height in canon_sraw_load_raw()
			  imgdata.sizes.raw_pitch = 8 * imgdata.sizes.raw_width;
		  }
		  else if (imgdata.sizes.raw_width == 4032 &&
			  imgdata.sizes.raw_height == 3402 &&
			  !strcasecmp(imgdata.idata.model, "EOS 80D")) // 80D hardcoded
		  {
			  imgdata.sizes.raw_width = 4536;
			  imgdata.sizes.left_margin = 28;
			  imgdata.sizes.iwidth = imgdata.sizes.width =
				  imgdata.sizes.raw_width - imgdata.sizes.left_margin;
			  imgdata.sizes.raw_height = 3024;
			  imgdata.sizes.top_margin = 8;
			  imgdata.sizes.iheight = imgdata.sizes.height =
				  imgdata.sizes.raw_height - imgdata.sizes.top_margin;
			  libraw_internal_data.unpacker_data.load_flags |= 256;
			  imgdata.sizes.raw_pitch = 8 * imgdata.sizes.raw_width;
		  }
	  }

#ifdef USE_DNGSDK
	  if (imgdata.idata.dng_version
		  &&libraw_internal_data.unpacker_data.tiff_compress == 34892
		  && libraw_internal_data.unpacker_data.tiff_bps == 8
		  && libraw_internal_data.unpacker_data.tiff_samples == 3
		  && load_raw == &LibRaw::lossy_dng_load_raw)
	  {
		  // Data should be linearized by DNG SDK
		  C.black = 0;
		  memset(C.cblack, 0, sizeof(C.cblack));
	  }
#endif

	  // XTrans Compressed?
	  if (!imgdata.idata.dng_version &&
		  makeIs(LIBRAW_CAMERAMAKER_Fujifilm) &&
		  (load_raw == &LibRaw::unpacked_load_raw))
	  {
		  if (imgdata.sizes.raw_width * (imgdata.sizes.raw_height * 2ul) !=
			  libraw_internal_data.unpacker_data.data_size)
		  {
			  if ((imgdata.sizes.raw_width * (imgdata.sizes.raw_height * 7ul)) / 4 ==
				  libraw_internal_data.unpacker_data.data_size)
				  load_raw = &LibRaw::fuji_14bit_load_raw;
			  else
				  parse_fuji_compressed_header();
		  }
		  else if (!strcmp(imgdata.idata.normalized_model, "X-H2S") 
			  && libraw_internal_data.internal_data.input->size() 
			  < (libraw_internal_data.unpacker_data.data_size + libraw_internal_data.unpacker_data.data_offset))
		  {
            parse_fuji_compressed_header(); // try to use compressed header: X-H2S may record wrong data size
		  }
	  }
      // set raw_inset_crops[1] via raw_aspect
      if (imgdata.sizes.raw_aspect >= LIBRAW_IMAGE_ASPECT_MINIMAL_REAL_ASPECT_VALUE
          && imgdata.sizes.raw_aspect <= LIBRAW_IMAGE_ASPECT_MAXIMAL_REAL_ASPECT_VALUE
          /* crops[0] is valid*/
          && (imgdata.sizes.raw_inset_crops[0].cleft < 0xffff)
          && (imgdata.sizes.raw_inset_crops[0].cleft + imgdata.sizes.raw_inset_crops[0].cwidth <= imgdata.sizes.raw_width)
          && (imgdata.sizes.raw_inset_crops[0].ctop < 0xffff)
          && (imgdata.sizes.raw_inset_crops[0].ctop + imgdata.sizes.raw_inset_crops[0].cheight <= imgdata.sizes.raw_height)
          && imgdata.sizes.raw_inset_crops[0].cwidth > 0 && imgdata.sizes.raw_inset_crops[0].cheight >0
          /* crops[1] is not set*/
          && (imgdata.sizes.raw_inset_crops[1].cleft == 0xffff)
          && (imgdata.sizes.raw_inset_crops[1].ctop == 0xffff)
          )
      {
          float c0_ratio = float(imgdata.sizes.raw_inset_crops[0].cwidth) / float(imgdata.sizes.raw_inset_crops[0].cheight);
          float c1_ratio = float(imgdata.sizes.raw_aspect) / 1000.f;
          if (c0_ratio / c1_ratio < 0.98 || c0_ratio / c1_ratio > 1.02) // set crops[1]
          {
              if (c1_ratio > c0_ratio) // requested image is wider, cut from top/bottom
              {
                  int newheight =  int(imgdata.sizes.raw_inset_crops[0].cwidth / c1_ratio);
                  int dtop = (imgdata.sizes.raw_inset_crops[0].cheight - newheight) / 2;
                  imgdata.sizes.raw_inset_crops[1].ctop = imgdata.sizes.raw_inset_crops[0].ctop + dtop;
                  imgdata.sizes.raw_inset_crops[1].cheight = newheight;
                  imgdata.sizes.raw_inset_crops[1].cleft = imgdata.sizes.raw_inset_crops[0].cleft;
                  imgdata.sizes.raw_inset_crops[1].cwidth = imgdata.sizes.raw_inset_crops[0].cwidth;
              }
              else
              {
                  int newwidth = int(imgdata.sizes.raw_inset_crops[0].cheight * c1_ratio);
                  int dleft = (imgdata.sizes.raw_inset_crops[0].cwidth - newwidth) / 2;
                  imgdata.sizes.raw_inset_crops[1].cleft = imgdata.sizes.raw_inset_crops[0].cleft + dleft;
                  imgdata.sizes.raw_inset_crops[1].cwidth = newwidth;
                  imgdata.sizes.raw_inset_crops[1].ctop = imgdata.sizes.raw_inset_crops[0].ctop;
                  imgdata.sizes.raw_inset_crops[1].cheight = imgdata.sizes.raw_inset_crops[0].cheight;
              }
          }
      }

      int adjust_margins = 0;
      if (makeIs(LIBRAW_CAMERAMAKER_Fujifilm) && (imgdata.idata.filters == 9))
      {
          // Adjust top/left margins for X-Trans
          int newtm = imgdata.sizes.top_margin % 6
              ? (imgdata.sizes.top_margin / 6 + 1) * 6
              : imgdata.sizes.top_margin;
          int newlm = imgdata.sizes.left_margin % 6
              ? (imgdata.sizes.left_margin / 6 + 1) * 6
              : imgdata.sizes.left_margin;
          if (newtm != imgdata.sizes.top_margin ||
              newlm != imgdata.sizes.left_margin)
          {
              imgdata.sizes.height -= (newtm - imgdata.sizes.top_margin);
              imgdata.sizes.top_margin = newtm;
              imgdata.sizes.width -= (newlm - imgdata.sizes.left_margin);
              imgdata.sizes.left_margin = newlm;
              for (int c1 = 0; c1 < 6; c1++)
                  for (int c2 = 0; c2 < 6; c2++)
                      imgdata.idata.xtrans[c1][c2] = imgdata.idata.xtrans_abs[c1][c2];
          }
          adjust_margins = 6;
      }
      else if (!libraw_internal_data.internal_output_params.fuji_width
          && imgdata.idata.filters >= 1000)
	  {
          if ((imgdata.sizes.top_margin % 2) || (imgdata.sizes.left_margin % 2))
          {
              int crop[2] = { 0,0 };
              unsigned filt;
              int c;
              if (imgdata.sizes.top_margin % 2)
              {
                  imgdata.sizes.top_margin += 1;
                  imgdata.sizes.height -= 1;
                  crop[1] = 1;
              }
              if (imgdata.sizes.left_margin % 2)
              {
                  imgdata.sizes.left_margin += 1;
                  imgdata.sizes.width -= 1;
                  crop[0] = 1;
              }
              for (filt = c = 0; c < 16; c++)
                  filt |= FC((c >> 1) + (crop[1]), (c & 1) + (crop[0])) << c * 2;
              imgdata.idata.filters = filt;
          }
          adjust_margins = 2;
	  }

      if(adjust_margins) // adjust crop_inset margins
          for (int i = 0; i < 2; i++)
          {
              if (imgdata.sizes.raw_inset_crops[i].cleft && imgdata.sizes.raw_inset_crops[i].cleft < 0xffff
                  && imgdata.sizes.raw_inset_crops[i].cwidth && imgdata.sizes.raw_inset_crops[i].cwidth < 0xffff
                  && (imgdata.sizes.raw_inset_crops[i].cleft%adjust_margins)
                  && (imgdata.sizes.raw_inset_crops[i].cwidth > adjust_margins))
              {
                  int newleft = ((imgdata.sizes.raw_inset_crops[i].cleft / adjust_margins) + 1) * adjust_margins;
                  int diff = newleft - imgdata.sizes.raw_inset_crops[i].cleft;
                  if (diff > 0)
                  {
                      imgdata.sizes.raw_inset_crops[i].cleft += diff;
                      imgdata.sizes.raw_inset_crops[i].cwidth -= diff;
                  }
              }
              if (imgdata.sizes.raw_inset_crops[i].ctop && imgdata.sizes.raw_inset_crops[i].ctop < 0xffff
                  && imgdata.sizes.raw_inset_crops[i].cheight && imgdata.sizes.raw_inset_crops[i].cheight < 0xffff
                  && (imgdata.sizes.raw_inset_crops[i].ctop%adjust_margins)
                  && (imgdata.sizes.raw_inset_crops[i].cheight > adjust_margins))
              {
                  int newtop = ((imgdata.sizes.raw_inset_crops[i].ctop / adjust_margins) + 1) * adjust_margins;
                  int diff = newtop - imgdata.sizes.raw_inset_crops[i].ctop;
                  if (diff > 0)
                  {
                      imgdata.sizes.raw_inset_crops[i].ctop += diff;
                      imgdata.sizes.raw_inset_crops[i].cheight -= diff;
                  }
              }
          }


#ifdef USE_DNGSDK
	  if (
		  imgdata.rawparams.use_dngsdk &&
		  !(imgdata.rawparams.options & (LIBRAW_RAWOPTIONS_DNG_STAGE2 | LIBRAW_RAWOPTIONS_DNG_STAGE3 | LIBRAW_RAWOPTIONS_DNG_DISABLEWBADJUST)))
#endif
	  {
		  // Fix DNG white balance if needed: observed only for Kalpanika X3F tools produced DNGs
		  if (imgdata.idata.dng_version && (imgdata.idata.filters == 0) &&
			  imgdata.idata.colors > 1 && imgdata.idata.colors < 5)
		  {
			  float delta[4] = { 0.f, 0.f, 0.f, 0.f };
			  int black[4];
			  for (int c = 0; c < 4; c++)
				  black[c] = imgdata.color.dng_levels.dng_black +
				  imgdata.color.dng_levels.dng_cblack[c];
			  for (int c = 0; c < imgdata.idata.colors; c++)
				  delta[c] = imgdata.color.dng_levels.dng_whitelevel[c] - black[c];
			  float mindelta = delta[0], maxdelta = delta[0];
			  for (int c = 1; c < imgdata.idata.colors; c++)
			  {
				  if (mindelta > delta[c])
					  mindelta = delta[c];
				  if (maxdelta < delta[c])
					  maxdelta = delta[c];
			  }
			  if (mindelta > 1 && maxdelta < (mindelta * 20)) // safety
			  {
				  for (int c = 0; c < imgdata.idata.colors; c++)
				  {
					  imgdata.color.cam_mul[c] /= (delta[c] / maxdelta);
					  imgdata.color.pre_mul[c] /= (delta[c] / maxdelta);
				  }
				  imgdata.color.maximum = imgdata.color.cblack[0] + maxdelta;
			  }
		  }
	  }

    if (imgdata.idata.dng_version &&
		makeIs(LIBRAW_CAMERAMAKER_Panasonic)
          && !strcasecmp(imgdata.idata.normalized_model, "DMC-LX100"))
      imgdata.sizes.width = 4288;

    if (imgdata.idata.dng_version
    	&& makeIs(LIBRAW_CAMERAMAKER_Leica)
        && !strcasecmp(imgdata.idata.normalized_model, "SL2"))
        	imgdata.sizes.height -= 16;

	if (makeIs(LIBRAW_CAMERAMAKER_Sony) &&
        imgdata.idata.dng_version)
    {
      if (S.raw_width == 3984)
        S.width = 3925;
      else if (S.raw_width == 4288)
        S.width = S.raw_width - 32;
      else if (S.raw_width == 4928 && S.height < 3280)
        S.width = S.raw_width - 8;
      else if (S.raw_width == 5504)
        S.width = S.raw_width - (S.height > 3664 ? 8 : 32);
    }

	if (makeIs(LIBRAW_CAMERAMAKER_Sony) &&
        !imgdata.idata.dng_version)
    {
        if(load_raw ==&LibRaw::sony_arq_load_raw)
        {
            if(S.raw_width > 12000) // A7RM4 16x, both APS-C and APS
                S.width = S.raw_width - 64;
            else // A7RM3/M4 4x merge
                S.width = S.raw_width - 32;
        }

      if (((!strncasecmp(imgdata.idata.model, "ILCE-7RM", 8) ||
            !strcasecmp(imgdata.idata.model, "ILCA-99M2")) &&
           (S.raw_width == 5216 || S.raw_width == 6304)) // A7RM2/M3/A99M2 in APS mode; A7RM4 in APS-C
          ||
          (!strcasecmp(imgdata.idata.model, "ILCE-7R") && S.raw_width >= 4580 &&
           S.raw_width < 5020) // A7R in crop mode, no samples, so size est.
          || (!strcasecmp(imgdata.idata.model, "ILCE-7") &&
              S.raw_width == 3968) // A7 in crop mode
          ||
          ((!strncasecmp(imgdata.idata.model, "ILCE-7M", 7) ||
            !strcasecmp(imgdata.idata.model, "ILCE-9") ||
#if 0
            !strcasecmp(imgdata.idata.model,
                        "SLT-A99V")) // Does SLT-A99 also have APS-C mode??
#endif
           (mnCamID == SonyID_SLT_A99)) // 2 reasons: some cameras are SLT-A99, no 'V'; some are Hasselblad HV
           && S.raw_width > 3750 &&
           S.raw_width < 4120) // A7M2, A7M3, AA9, most likely APS-C raw_width
                               // is 3968 (same w/ A7), but no samples, so guess
          || (!strncasecmp(imgdata.idata.model, "ILCE-7S", 7) &&
              S.raw_width == 2816) // A7S2=> exact, hope it works for A7S-I too
      )
        S.width = S.raw_width - 32;
    }


    // FIXME: it is possible that DNG contains 4x main frames + some previews; in this case imgdata.idata.raw_count will be greater than 4
	if (makeIs(LIBRAW_CAMERAMAKER_Pentax) &&
        /*!strcasecmp(imgdata.idata.model,"K-3 II")  &&*/
            imgdata.idata.raw_count == 4 &&
        (imgdata.rawparams.options & LIBRAW_RAWOPTIONS_PENTAX_PS_ALLFRAMES))
    {
      imgdata.idata.raw_count = 1;
      imgdata.idata.filters = 0;
      imgdata.idata.colors = 4;
      imgdata.sizes.top_margin+=2;
      imgdata.sizes.left_margin+=2;
      imgdata.sizes.width-=4;
      imgdata.sizes.height-=4;
      IO.mix_green = 1;
      pentax_component_load_raw = load_raw;
      load_raw = &LibRaw::pentax_4shot_load_raw;
    }

	if (!imgdata.idata.dng_version && makeIs(LIBRAW_CAMERAMAKER_Leaf) &&
        !strcmp(imgdata.idata.model, "Credo 50"))
    {
      imgdata.color.pre_mul[0] = 1.f / 0.3984f;
      imgdata.color.pre_mul[2] = 1.f / 0.7666f;
      imgdata.color.pre_mul[1] = imgdata.color.pre_mul[3] = 1.0;
    }

	if (!imgdata.idata.dng_version && makeIs(LIBRAW_CAMERAMAKER_Fujifilm) &&
        (!strncmp(imgdata.idata.model, "S20Pro", 6) ||
         !strncmp(imgdata.idata.model, "F700", 4)))
    {
      imgdata.sizes.raw_width /= 2;
      load_raw = &LibRaw::unpacked_load_raw_fuji_f700s20;
    }

    if (load_raw == &LibRaw::packed_load_raw &&
		makeIs(LIBRAW_CAMERAMAKER_Nikon) &&
        !libraw_internal_data.unpacker_data.load_flags &&
        (!strncasecmp(imgdata.idata.model, "D810", 4) ||
         !strcasecmp(imgdata.idata.model, "D4S")) &&
        libraw_internal_data.unpacker_data.data_size * 2u ==
            imgdata.sizes.raw_height * imgdata.sizes.raw_width * 3u)
    {
      libraw_internal_data.unpacker_data.load_flags = 80;
    }
    // Adjust BL for Sony A900/A850
    if (load_raw == &LibRaw::packed_load_raw &&
		makeIs(LIBRAW_CAMERAMAKER_Sony)) // 12 bit sony, but metadata may be for 14-bit range
    {
      if (C.maximum > 4095)
        C.maximum = 4095;
      if (C.black > 256 || C.cblack[0] > 256)
      {
        C.black /= 4;
        for (int c = 0; c < 4; c++)
          C.cblack[c] /= 4;
        for (unsigned c = 0; c < C.cblack[4] * C.cblack[5]; c++)
          C.cblack[6 + c] /= 4;
      }
    }

	if (load_raw == &LibRaw::nikon_yuv_load_raw) // Is it Nikon sRAW?
    {
      load_raw = &LibRaw::nikon_load_sraw;
      C.black = 0;
      memset(C.cblack, 0, sizeof(C.cblack));
      imgdata.idata.filters = 0;
      libraw_internal_data.unpacker_data.tiff_samples = 3;
      imgdata.idata.colors = 3;
      double beta_1 = -5.79342238397656E-02;
      double beta_2 = 3.28163551282665;
      double beta_3 = -8.43136004842678;
      double beta_4 = 1.03533181861023E+01;
      for (int i = 0; i <= 3072; i++)
      {
        double x = (double)i / 3072.;
        double y = (1. - exp(-beta_1 * x - beta_2 * x * x - beta_3 * x * x * x -
                             beta_4 * x * x * x * x));
        if (y < 0.)
          y = 0.;
        imgdata.color.curve[i] = (y * 16383.);
      }
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
          imgdata.color.rgb_cam[i][j] = float(i == j);
    }
    // Adjust BL for Nikon 12bit
    if ((load_raw == &LibRaw::nikon_load_raw ||
         load_raw == &LibRaw::packed_load_raw ||
         load_raw == &LibRaw::nikon_load_padded_packed_raw) &&
		 makeIs(LIBRAW_CAMERAMAKER_Nikon) &&
        strncmp(imgdata.idata.model, "COOLPIX", 7) &&
        libraw_internal_data.unpacker_data.tiff_bps == 12)
    {
      C.maximum = 4095;
      C.black /= 4;
      for (int c = 0; c < 4; c++)
        C.cblack[c] /= 4;
      for (unsigned c = 0; c < C.cblack[4] * C.cblack[5]; c++)
        C.cblack[6 + c] /= 4;
    }

    // Adjust wb_already_applied
    if (load_raw == &LibRaw::nikon_load_sraw)
      imgdata.color.as_shot_wb_applied =
          LIBRAW_ASWB_APPLIED | LIBRAW_ASWB_NIKON_SRAW;
    else if (makeIs(LIBRAW_CAMERAMAKER_Canon) &&
             MN.canon.multishot[0] >= 8 &&
             MN.canon.multishot[1] > 0)
      imgdata.color.as_shot_wb_applied =
          LIBRAW_ASWB_APPLIED | LIBRAW_ASWB_CANON;
    else if (makeIs(LIBRAW_CAMERAMAKER_Nikon) &&
             MN.nikon.ExposureMode == 1)
      imgdata.color.as_shot_wb_applied =
          LIBRAW_ASWB_APPLIED | LIBRAW_ASWB_NIKON;
	else if (makeIs(LIBRAW_CAMERAMAKER_Pentax) &&
             ((MN.pentax.MultiExposure & 0x01) == 1))
      imgdata.color.as_shot_wb_applied =
          LIBRAW_ASWB_APPLIED | LIBRAW_ASWB_PENTAX;
    else
      imgdata.color.as_shot_wb_applied = 0;

    // Adjust Highlight Linearity limit
    if (C.linear_max[0] < 0)
    {
      if (imgdata.idata.dng_version)
      {
        for (int c = 0; c < 4; c++)
          C.linear_max[c] = -1 * C.linear_max[c] + imgdata.color.cblack[c + 6];
      }
      else
      {
        for (int c = 0; c < 4; c++)
          C.linear_max[c] = -1 * C.linear_max[c] + imgdata.color.cblack[c];
      }
    }

	if (makeIs(LIBRAW_CAMERAMAKER_Nikon) &&
		(!C.linear_max[0]) && (C.maximum > 1024) && (load_raw != &LibRaw::nikon_load_sraw))
    {
      C.linear_max[0] = C.linear_max[1] = C.linear_max[2] = C.linear_max[3] =
          (long)((float)(C.maximum) / 1.07f);
    }

    // Correct WB for Samsung GX20
	if (
#if 0
        /* GX20 should be corrected, but K20 is not */
        makeIs(LIBRAW_CAMERAMAKER_Pentax) &&
		!strcasecmp(imgdata.idata.normalized_model, "K20D")
#endif
#if 0
		!strcasecmp(imgdata.idata.make, "Samsung") &&
        !strcasecmp(imgdata.idata.model, "GX20")
#endif
    makeIs(LIBRAW_CAMERAMAKER_Pentax) &&
    (mnCamID == PentaxID_GX20) // Samsung rebranding
		)
    {
      for (int cnt = LIBRAW_WBI_Unknown; cnt <= LIBRAW_WBI_StudioTungsten; cnt++) {
        if (C.WB_Coeffs[cnt][1]) {
          C.WB_Coeffs[cnt][0] = (int)((float)(C.WB_Coeffs[cnt][0]) * 1.0503f);
          C.WB_Coeffs[cnt][2] = (int)((float)(C.WB_Coeffs[cnt][2]) * 2.2867f);
        }
      }
      for (int cnt = 0; cnt < 64; cnt++) {
        if (C.WBCT_Coeffs[cnt][0] > 0.0f) {
          C.WBCT_Coeffs[cnt][1] *= 1.0503f;
          C.WBCT_Coeffs[cnt][3] *= 2.2867f;
        }
      }
      for(int cnt = 0; cnt < 4; cnt++)
        imgdata.color.pre_mul[cnt] =
          C.WB_Coeffs[LIBRAW_WBI_Daylight][cnt];
    }

    // Adjust BL for Panasonic
    if (load_raw == &LibRaw::panasonic_load_raw &&
		makeIs(LIBRAW_CAMERAMAKER_Panasonic) &&
        ID.pana_black[0] && ID.pana_black[1] && ID.pana_black[2])
    {
      if (libraw_internal_data.unpacker_data.pana_encoding == 5)
        libraw_internal_data.internal_output_params.zero_is_bad = 0;
      C.black = 0;
      int add = libraw_internal_data.unpacker_data.pana_encoding == 4 ? 15 : 0;
      C.cblack[0] = ID.pana_black[0] + add;
      C.cblack[1] = C.cblack[3] = ID.pana_black[1] + add;
      C.cblack[2] = ID.pana_black[2] + add;
      unsigned i = C.cblack[3];
      for (int c = 0; c < 3; c++)
        if (i > C.cblack[c])
          i = C.cblack[c];
      for (int c = 0; c < 4; c++)
        C.cblack[c] -= i;
      C.black = i;
    }

    // Adjust sizes for X3F processing
#ifdef USE_X3FTOOLS
	if (load_raw == &LibRaw::x3f_load_raw)
    {
      for (int i = 0; i < foveon_count; i++)
        if (!strcasecmp(imgdata.idata.make, foveon_data[i].make) &&
            !strcasecmp(imgdata.idata.model, foveon_data[i].model) &&
            imgdata.sizes.raw_width == foveon_data[i].raw_width &&
            imgdata.sizes.raw_height == foveon_data[i].raw_height)
        {
          imgdata.sizes.top_margin = foveon_data[i].top_margin;
          imgdata.sizes.left_margin = foveon_data[i].left_margin;
          imgdata.sizes.width = imgdata.sizes.iwidth = foveon_data[i].width;
          imgdata.sizes.height = imgdata.sizes.iheight = foveon_data[i].height;
          C.maximum = foveon_data[i].white;
          break;
        }
    }
#endif
#if 0
    size_t bytes = ID.input->size()-libraw_internal_data.unpacker_data.data_offset;
    float bpp = float(bytes)/float(S.raw_width)/float(S.raw_height);
    float bpp2 = float(bytes)/float(S.width)/float(S.height);
    if(!strcasecmp(imgdata.idata.make,"Hasselblad") && bpp == 6.0f)
      {
        load_raw = &LibRaw::hasselblad_full_load_raw;
        S.width = S.raw_width;
        S.height = S.raw_height;
        P1.filters = 0;
        P1.colors=3;
        P1.raw_count=1;
        C.maximum=0xffff;
      }
#endif
    if (C.profile_length)
    {
      if (C.profile)
        free(C.profile);
      C.profile = malloc(C.profile_length);
      ID.input->seek(ID.profile_offset, SEEK_SET);
      ID.input->read(C.profile, C.profile_length, 1);
    }

    SET_PROC_FLAG(LIBRAW_PROGRESS_IDENTIFY);
  }
  catch (const std::bad_alloc&)
  {
      EXCEPTION_HANDLER(LIBRAW_EXCEPTION_ALLOC);
  }
  catch (const LibRaw_exceptions& err)
  {
    EXCEPTION_HANDLER(err);
  }
  catch (const std::exception& )
  {
    EXCEPTION_HANDLER(LIBRAW_EXCEPTION_IO_CORRUPT);
  }

final:;

  if (P1.raw_count < 1)
    return LIBRAW_FILE_UNSUPPORTED;

  write_fun = &LibRaw::write_ppm_tiff;

  if (load_raw == &LibRaw::kodak_ycbcr_load_raw)
  {
    S.height += S.height & 1;
    S.width += S.width & 1;
  }

  IO.shrink =
      P1.filters &&
      (O.half_size || ((O.threshold || O.aber[0] != 1 || O.aber[2] != 1)));
  if (IO.shrink && P1.filters >= 1000)
  {
    S.width &= 65534;
    S.height &= 65534;
  }

  S.iheight = (S.height + IO.shrink) >> IO.shrink;
  S.iwidth = (S.width + IO.shrink) >> IO.shrink;

  // Save color,sizes and internal data into raw_image fields
  memmove(&imgdata.rawdata.color, &imgdata.color, sizeof(imgdata.color));
  memmove(&imgdata.rawdata.sizes, &imgdata.sizes, sizeof(imgdata.sizes));
  memmove(&imgdata.rawdata.iparams, &imgdata.idata, sizeof(imgdata.idata));
  memmove(&imgdata.rawdata.ioparams,
          &libraw_internal_data.internal_output_params,
          sizeof(libraw_internal_data.internal_output_params));

  SET_PROC_FLAG(LIBRAW_PROGRESS_SIZE_ADJUST);

  return LIBRAW_SUCCESS;
}
