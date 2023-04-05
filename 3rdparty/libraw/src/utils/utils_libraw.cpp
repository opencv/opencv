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

#ifdef __cplusplus
extern "C"
{
#endif

  void default_data_callback(void *, const char *file, const int offset)
  {
    if (offset < 0)
      fprintf(stderr, "%s: Unexpected end of file\n",
              file ? file : "unknown file");
    else
      fprintf(stderr, "%s: data corrupted at %d\n",
              file ? file : "unknown file", offset);
  }
  const char *libraw_strerror(int e)
  {
    enum LibRaw_errors errorcode = (LibRaw_errors)e;
    switch (errorcode)
    {
    case LIBRAW_SUCCESS:
      return "No error";
    case LIBRAW_UNSPECIFIED_ERROR:
      return "Unspecified error";
    case LIBRAW_FILE_UNSUPPORTED:
      return "Unsupported file format or not RAW file";
    case LIBRAW_REQUEST_FOR_NONEXISTENT_IMAGE:
      return "Request for nonexisting image number";
    case LIBRAW_OUT_OF_ORDER_CALL:
      return "Out of order call of libraw function";
    case LIBRAW_NO_THUMBNAIL:
      return "No thumbnail in file";
    case LIBRAW_UNSUPPORTED_THUMBNAIL:
      return "Unsupported thumbnail format";
    case LIBRAW_INPUT_CLOSED:
      return "No input stream, or input stream closed";
    case LIBRAW_NOT_IMPLEMENTED:
      return "Decoder not implemented for this data format";
    case LIBRAW_REQUEST_FOR_NONEXISTENT_THUMBNAIL:
      return "Request for nonexisting thumbnail number";
    case LIBRAW_MEMPOOL_OVERFLOW:
      return "Libraw internal mempool overflowed";
    case LIBRAW_UNSUFFICIENT_MEMORY:
      return "Unsufficient memory";
    case LIBRAW_DATA_ERROR:
      return "Corrupted data or unexpected EOF";
    case LIBRAW_IO_ERROR:
      return "Input/output error";
    case LIBRAW_CANCELLED_BY_CALLBACK:
      return "Cancelled by user callback";
    case LIBRAW_BAD_CROP:
      return "Bad crop box";
    case LIBRAW_TOO_BIG:
      return "Image too big for processing";
    default:
      return "Unknown error code";
    }
  }

#ifdef __cplusplus
}
#endif

unsigned LibRaw::parse_custom_cameras(unsigned limit,
                                      libraw_custom_camera_t table[],
                                      char **list)
{
  if (!list)
    return 0;
  unsigned index = 0;
  for (unsigned i = 0; i < limit; i++)
  {
    if (!list[i])
      break;
    if (strlen(list[i]) < 10)
      continue;
    char *string = (char *)malloc(strlen(list[i]) + 1);
    strcpy(string, list[i]);
    char *start = string;
    memset(&table[index], 0, sizeof(table[0]));
    for (int j = 0; start && j < 14; j++)
    {
      char *end = strchr(start, ',');
      if (end)
      {
        *end = 0;
        end++;
      } // move to next char
      while (isspace(*start) && *start)
        start++; // skip leading spaces?
      unsigned val = strtol(start, 0, 10);
      switch (j)
      {
      case 0:
        table[index].fsize = val;
        break;
      case 1:
        table[index].rw = val;
        break;
      case 2:
        table[index].rh = val;
        break;
      case 3:
        table[index].lm = val;
        break;
      case 4:
        table[index].tm = val;
        break;
      case 5:
        table[index].rm = val;
        break;
      case 6:
        table[index].bm = val;
        break;
      case 7:
        table[index].lf = val;
        break;
      case 8:
        table[index].cf = val;
        break;
      case 9:
        table[index].max = val;
        break;
      case 10:
        table[index].flags = val;
        break;
      case 11:
        strncpy(table[index].t_make, start, sizeof(table[index].t_make) - 1);
        break;
      case 12:
        strncpy(table[index].t_model, start, sizeof(table[index].t_model) - 1);
        break;
      case 13:
        table[index].offset = val;
        break;
      default:
        break;
      }
      start = end;
    }
    free(string);
    if (table[index].t_make[0])
      index++;
  }
  return index;
}

void LibRaw::derror()
{
  if (!libraw_internal_data.unpacker_data.data_error &&
      libraw_internal_data.internal_data.input)
  {
    if (libraw_internal_data.internal_data.input->eof())
    {
      if (callbacks.data_cb)
        (*callbacks.data_cb)(callbacks.datacb_data,
                             libraw_internal_data.internal_data.input->fname(),
                             -1);
      throw LIBRAW_EXCEPTION_IO_EOF;
    }
    else
    {
      if (callbacks.data_cb)
        (*callbacks.data_cb)(callbacks.datacb_data,
                             libraw_internal_data.internal_data.input->fname(),
                             libraw_internal_data.internal_data.input->tell());
      // throw LIBRAW_EXCEPTION_IO_CORRUPT;
    }
  }
  libraw_internal_data.unpacker_data.data_error++;
}

const char *LibRaw::version() { return LIBRAW_VERSION_STR; }
int LibRaw::versionNumber() { return LIBRAW_VERSION; }
const char *LibRaw::strerror(int p) { return libraw_strerror(p); }

unsigned LibRaw::capabilities()
{
  unsigned ret = 0;
#ifdef USE_RAWSPEED
  ret |= LIBRAW_CAPS_RAWSPEED;
#endif
#ifdef USE_RAWSPEED3
  ret |= LIBRAW_CAPS_RAWSPEED3;
#endif
#ifdef USE_RAWSPEED_BITS
  ret |= LIBRAW_CAPS_RAWSPEED_BITS;
#endif
#ifdef USE_DNGSDK
  ret |= LIBRAW_CAPS_DNGSDK;
#ifdef USE_GPRSDK
  ret |= LIBRAW_CAPS_GPRSDK;
#endif
#ifdef LIBRAW_WIN32_UNICODEPATHS
  ret |= LIBRAW_CAPS_UNICODEPATHS;
#endif
#endif
#ifdef USE_X3FTOOLS
  ret |= LIBRAW_CAPS_X3FTOOLS;
#endif
#ifdef USE_6BY9RPI
  ret |= LIBRAW_CAPS_RPI6BY9;
#endif
#ifdef USE_ZLIB
  ret |= LIBRAW_CAPS_ZLIB;
#endif
#ifdef USE_JPEG
  ret |= LIBRAW_CAPS_JPEG;
#endif
  return ret;
}

int LibRaw::is_sraw()
{
  return load_raw == &LibRaw::canon_sraw_load_raw ||
         load_raw == &LibRaw::nikon_load_sraw;
}
int LibRaw::is_coolscan_nef()
{
  return load_raw == &LibRaw::nikon_coolscan_load_raw;
}
int LibRaw::is_jpeg_thumb()
{
  return libraw_internal_data.unpacker_data.thumb_format == LIBRAW_INTERNAL_THUMBNAIL_JPEG;
}

int LibRaw::is_nikon_sraw() { return load_raw == &LibRaw::nikon_load_sraw; }
int LibRaw::sraw_midpoint()
{
  if (load_raw == &LibRaw::canon_sraw_load_raw)
    return 8192;
  else if (load_raw == &LibRaw::nikon_load_sraw)
    return 2048;
  else
    return 0;
}

void *LibRaw::malloc(size_t t)
{
  void *p = memmgr.malloc(t);
  if (!p)
    throw LIBRAW_EXCEPTION_ALLOC;
  return p;
}
void *LibRaw::realloc(void *q, size_t t)
{
  void *p = memmgr.realloc(q, t);
  if (!p)
    throw LIBRAW_EXCEPTION_ALLOC;
  return p;
}

void *LibRaw::calloc(size_t n, size_t t)
{
  void *p = memmgr.calloc(n, t);
  if (!p)
    throw LIBRAW_EXCEPTION_ALLOC;
  return p;
}
void LibRaw::free(void *p) { memmgr.free(p); }

void LibRaw::recycle_datastream()
{
  if (libraw_internal_data.internal_data.input &&
      libraw_internal_data.internal_data.input_internal)
  {
    delete libraw_internal_data.internal_data.input;
    libraw_internal_data.internal_data.input = NULL;
  }
  libraw_internal_data.internal_data.input_internal = 0;
}

void LibRaw::clearCancelFlag()
{
#ifdef _MSC_VER
  InterlockedExchange(&_exitflag, 0);
#else
  __sync_fetch_and_and(&_exitflag, 0);
#endif
#ifdef RAWSPEED_FASTEXIT
  if (_rawspeed_decoder)
  {
    RawSpeed::RawDecoder *d =
        static_cast<RawSpeed::RawDecoder *>(_rawspeed_decoder);
    d->resumeProcessing();
  }
#endif
}

void LibRaw::setCancelFlag()
{
#ifdef _MSC_VER
  InterlockedExchange(&_exitflag, 1);
#else
  __sync_fetch_and_add(&_exitflag, 1);
#endif
#ifdef RAWSPEED_FASTEXIT
  if (_rawspeed_decoder)
  {
    RawSpeed::RawDecoder *d =
        static_cast<RawSpeed::RawDecoder *>(_rawspeed_decoder);
    d->cancelProcessing();
  }
#endif
}

void LibRaw::checkCancel()
{
#ifdef _MSC_VER
  if (InterlockedExchange(&_exitflag, 0))
    throw LIBRAW_EXCEPTION_CANCELLED_BY_CALLBACK;
#else
  if (__sync_fetch_and_and(&_exitflag, 0))
    throw LIBRAW_EXCEPTION_CANCELLED_BY_CALLBACK;
#endif
}

int LibRaw::is_curve_linear()
{
  for (int i = 0; i < 0x10000; i++)
    if (imgdata.color.curve[i] != i)
      return 0;
  return 1;
}

void LibRaw::free_image(void)
{
  if (imgdata.image)
  {
    free(imgdata.image);
    imgdata.image = 0;
    imgdata.progress_flags = LIBRAW_PROGRESS_START | LIBRAW_PROGRESS_OPEN |
                             LIBRAW_PROGRESS_IDENTIFY |
                             LIBRAW_PROGRESS_SIZE_ADJUST |
                             LIBRAW_PROGRESS_LOAD_RAW;
  }
}

int LibRaw::is_phaseone_compressed()
{
  return (load_raw == &LibRaw::phase_one_load_raw_c ||
		  load_raw == &LibRaw::phase_one_load_raw_s ||
          load_raw == &LibRaw::phase_one_load_raw);
}

int LibRaw::is_canon_600() { return load_raw == &LibRaw::canon_600_load_raw; }
const char *LibRaw::strprogress(enum LibRaw_progress p)
{
  switch (p)
  {
  case LIBRAW_PROGRESS_START:
    return "Starting";
  case LIBRAW_PROGRESS_OPEN:
    return "Opening file";
  case LIBRAW_PROGRESS_IDENTIFY:
    return "Reading metadata";
  case LIBRAW_PROGRESS_SIZE_ADJUST:
    return "Adjusting size";
  case LIBRAW_PROGRESS_LOAD_RAW:
    return "Reading RAW data";
  case LIBRAW_PROGRESS_REMOVE_ZEROES:
    return "Clearing zero values";
  case LIBRAW_PROGRESS_BAD_PIXELS:
    return "Removing dead pixels";
  case LIBRAW_PROGRESS_DARK_FRAME:
    return "Subtracting dark frame data";
  case LIBRAW_PROGRESS_FOVEON_INTERPOLATE:
    return "Interpolating Foveon sensor data";
  case LIBRAW_PROGRESS_SCALE_COLORS:
    return "Scaling colors";
  case LIBRAW_PROGRESS_PRE_INTERPOLATE:
    return "Pre-interpolating";
  case LIBRAW_PROGRESS_INTERPOLATE:
    return "Interpolating";
  case LIBRAW_PROGRESS_MIX_GREEN:
    return "Mixing green channels";
  case LIBRAW_PROGRESS_MEDIAN_FILTER:
    return "Median filter";
  case LIBRAW_PROGRESS_HIGHLIGHTS:
    return "Highlight recovery";
  case LIBRAW_PROGRESS_FUJI_ROTATE:
    return "Rotating Fuji diagonal data";
  case LIBRAW_PROGRESS_FLIP:
    return "Flipping image";
  case LIBRAW_PROGRESS_APPLY_PROFILE:
    return "ICC conversion";
  case LIBRAW_PROGRESS_CONVERT_RGB:
    return "Converting to RGB";
  case LIBRAW_PROGRESS_STRETCH:
    return "Stretching image";
  case LIBRAW_PROGRESS_THUMB_LOAD:
    return "Loading thumbnail";
  default:
    return "Some strange things";
  }
}
int LibRaw::adjust_sizes_info_only(void)
{
  CHECK_ORDER_LOW(LIBRAW_PROGRESS_IDENTIFY);

  raw2image_start();
  if (O.use_fuji_rotate)
  {
    if (IO.fuji_width)
    {
      IO.fuji_width = (IO.fuji_width - 1 + IO.shrink) >> IO.shrink;
      S.iwidth = (ushort)(IO.fuji_width / sqrt(0.5));
      S.iheight = (ushort)((S.iheight - IO.fuji_width) / sqrt(0.5));
    }
    else
    {
      if (S.pixel_aspect < 0.995)
        S.iheight = (ushort)(S.iheight / S.pixel_aspect + 0.5);
      if (S.pixel_aspect > 1.005)
        S.iwidth = (ushort)(S.iwidth * S.pixel_aspect + 0.5);
    }
  }
  SET_PROC_FLAG(LIBRAW_PROGRESS_FUJI_ROTATE);
  if (S.flip & 4)
  {
    unsigned short t = S.iheight;
    S.iheight = S.iwidth;
    S.iwidth = t;
    SET_PROC_FLAG(LIBRAW_PROGRESS_FLIP);
  }
  return 0;
}
int LibRaw::adjust_maximum()
{
  ushort real_max;
  float auto_threshold;

  if (O.adjust_maximum_thr < 0.00001)
    return LIBRAW_SUCCESS;
  else if (O.adjust_maximum_thr > 0.99999)
    auto_threshold = LIBRAW_DEFAULT_ADJUST_MAXIMUM_THRESHOLD;
  else
    auto_threshold = O.adjust_maximum_thr;

  real_max = C.data_maximum;
  if (real_max > 0 && real_max < C.maximum &&
      real_max > C.maximum * auto_threshold)
  {
    C.maximum = real_max;
  }
  return LIBRAW_SUCCESS;
}
void LibRaw::adjust_bl()
{
  int clear_repeat = 0;
  if (O.user_black >= 0)
  {
    C.black = O.user_black;
    clear_repeat = 1;
  }
  for (int i = 0; i < 4; i++)
    if (O.user_cblack[i] > -1000000)
    {
      C.cblack[i] = O.user_cblack[i];
      clear_repeat = 1;
    }

  if (clear_repeat)
    C.cblack[4] = C.cblack[5] = 0;

  // Add common part to cblack[] early
  if (imgdata.idata.filters > 1000 && (C.cblack[4] + 1) / 2 == 1 &&
      (C.cblack[5] + 1) / 2 == 1)
  {
    int clrs[4];
    int lastg = -1, gcnt = 0;
    for (int c = 0; c < 4; c++)
    {
      clrs[c] = FC(c / 2, c % 2);
      if (clrs[c] == 1)
      {
        gcnt++;
        lastg = c;
      }
    }
    if (gcnt > 1 && lastg >= 0)
      clrs[lastg] = 3;
    for (int c = 0; c < 4; c++)
      C.cblack[clrs[c]] +=
          C.cblack[6 + c / 2 % C.cblack[4] * C.cblack[5] + c % 2 % C.cblack[5]];
    C.cblack[4] = C.cblack[5] = 0;
    // imgdata.idata.filters = sfilters;
  }
  else if (imgdata.idata.filters <= 1000 && C.cblack[4] == 1 &&
           C.cblack[5] == 1) // Fuji RAF dng
  {
    for (int c = 0; c < 4; c++)
      C.cblack[c] += C.cblack[6];
    C.cblack[4] = C.cblack[5] = 0;
  }
  // remove common part from C.cblack[]
  int i = C.cblack[3];
  int c;
  for (c = 0; c < 3; c++)
    if (i > (int)C.cblack[c])
      i = C.cblack[c];

  for (c = 0; c < 4; c++)
    C.cblack[c] -= i; // remove common part
  C.black += i;

  // Now calculate common part for cblack[6+] part and move it to C.black

  if (C.cblack[4] && C.cblack[5])
  {
    i = C.cblack[6];
    for (c = 1; c < int(C.cblack[4] * C.cblack[5]); c++)
      if (i > int(C.cblack[6 + c]))
        i = C.cblack[6 + c];
    // Remove i from cblack[6+]
    int nonz = 0;
    for (c = 0; c < int(C.cblack[4] * C.cblack[5]); c++)
    {
      C.cblack[6 + c] -= i;
      if (C.cblack[6 + c])
        nonz++;
    }
    C.black += i;
    if (!nonz)
      C.cblack[4] = C.cblack[5] = 0;
  }
  for (c = 0; c < 4; c++)
    C.cblack[c] += C.black;
}
int LibRaw::getwords(char *line, char *words[], int maxwords, int maxlen)
{
  line[maxlen - 1] = 0;
  unsigned char *p = (unsigned char*)line;
  int nwords = 0;

  while (1)
  {
    while (isspace(*p))
      p++;
    if (*p == '\0')
      return nwords;
    words[nwords++] = (char*)p;
    while (!isspace(*p) && *p != '\0')
      p++;
    if (*p == '\0')
      return nwords;
    *p++ = '\0';
    if (nwords >= maxwords)
      return nwords;
  }
}
int LibRaw::stread(char *buf, size_t len, LibRaw_abstract_datastream *fp)
{
  if (len > 0)
  {
    int r = fp->read(buf, len, 1);
    buf[len - 1] = 0;
    return r;
  }
  else
    return 0;
}

int LibRaw::find_ifd_by_offset(int o)
{
    for(unsigned i = 0; i < libraw_internal_data.identify_data.tiff_nifds && i < LIBRAW_IFD_MAXCOUNT; i++)
        if(tiff_ifd[i].offset == o)
            return i;
    return -1;
}

short LibRaw::tiff_sget (unsigned save, uchar *buf, unsigned buf_len, INT64 *tag_offset,
                         unsigned *tag_id, unsigned *tag_type, INT64 *tag_dataoffset,
                         unsigned *tag_datalen, int *tag_dataunitlen) {
  uchar *pos = buf + *tag_offset;
  if ((((*tag_offset) + 12) > buf_len) || (*tag_offset < 0)) { // abnormal, tag buffer overrun
    return -1;
  }
  *tag_id      = sget2(pos); pos += 2;
  *tag_type    = sget2(pos); pos += 2;
  *tag_datalen = sget4(pos); pos += 4;
  *tag_dataunitlen = tagtype_dataunit_bytes[(*tag_type <= LIBRAW_EXIFTAG_TYPE_IFD8) ? *tag_type : 0];
  if ((*tag_datalen * (*tag_dataunitlen)) > 4) {
    *tag_dataoffset = sget4(pos) - save;
    if ((*tag_dataoffset + *tag_datalen) > buf_len) { // abnormal, tag data buffer overrun
      return -2;
    }
  } else *tag_dataoffset = *tag_offset + 8;
  *tag_offset += 12;
  return 0;
}

#define rICC  imgdata.sizes.raw_inset_crops
#define S imgdata.sizes
#define RS imgdata.rawdata.sizes
int LibRaw::adjust_to_raw_inset_crop(unsigned mask, float maxcrop)

{
    int adjindex = -1;
	int limwidth = S.width * maxcrop;
	int limheight = S.height * maxcrop;

    for(int i = 1; i >= 0; i--)
        if (mask & (1<<i))
            if (rICC[i].ctop < 0xffff && rICC[i].cleft < 0xffff
                && rICC[i].cleft + rICC[i].cwidth <= S.raw_width
                && rICC[i].ctop + rICC[i].cheight <= S.raw_height
				&& rICC[i].cwidth >= limwidth && rICC[i].cheight >= limheight)
            {
                adjindex = i;
                break;
            }

    if (adjindex >= 0)
    {
        RS.left_margin = S.left_margin = rICC[adjindex].cleft;
        RS.top_margin = S.top_margin = rICC[adjindex].ctop;
        RS.width = S.width = MIN(rICC[adjindex].cwidth, int(S.raw_width) - int(S.left_margin));
        RS.height = S.height = MIN(rICC[adjindex].cheight, int(S.raw_height) - int(S.top_margin));
    }
    return adjindex + 1;
}

char** LibRaw::malloc_omp_buffers(int buffer_count, size_t buffer_size)
{
    char** buffers = (char**)calloc(sizeof(char*), buffer_count);

    for (int i = 0; i < buffer_count; i++)
    {
        buffers[i] = (char*)malloc(buffer_size);
    }
    return buffers;
}

void LibRaw::free_omp_buffers(char** buffers, int buffer_count)
{
    for (int i = 0; i < buffer_count; i++)
        if(buffers[i])
            free(buffers[i]);
    free(buffers);
}

void 	LibRaw::libraw_swab(void *arr, size_t len)
{
#ifdef LIBRAW_OWN_SWAB
	uint16_t *array = (uint16_t*)arr;
	size_t bytes = len/2;
	for(; bytes; --bytes)
	{
		*array = ((*array << 8) & 0xff00) | ((*array >> 8) & 0xff);
		array++;
	}
#else
	swab((char*)arr,(char*)arr,len);
#endif

}
