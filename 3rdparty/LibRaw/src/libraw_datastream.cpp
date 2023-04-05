/* -*- C++ -*-
 * File: libraw_datastream.cpp
 * Copyright 2008-2021 LibRaw LLC (info@libraw.org)
 *
 * LibRaw C++ interface (implementation)

 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

*/

#ifdef _WIN32
#ifdef __MINGW32__
#define _WIN32_WINNT 0x0500
#include <stdexcept>
#endif
#endif

#define LIBRAW_LIBRARY_BUILD
#include "libraw/libraw.h"
#include "libraw/libraw_types.h"
#include "libraw/libraw_datastream.h"
#include <sys/stat.h>
#ifdef USE_JASPER
#include <jasper/jasper.h> /* Decode RED camera movies */
#else
#define NO_JASPER
#endif
#ifdef USE_JPEG
#include <jpeglib.h>
#include <jerror.h>
#else
#define NO_JPEG
#endif

#ifdef USE_JPEG

typedef struct
{
    struct jpeg_source_mgr pub; /* public fields */
    LibRaw_abstract_datastream *instream;            /* source stream */
    JOCTET *buffer;             /* start of buffer */
    boolean start_of_file;      /* have we gotten any data yet? */
} lr_jpg_source_mgr;

typedef lr_jpg_source_mgr *lr_jpg_src_ptr;

#define LR_JPEG_INPUT_BUF_SIZE 16384 

static void f_init_source(j_decompress_ptr cinfo)
{
    lr_jpg_src_ptr src = (lr_jpg_src_ptr)cinfo->src;
    src->start_of_file = TRUE;
}

#ifdef ERREXIT
#undef ERREXIT
#endif

#define ERREXIT(cinfo, code)                                                   \
  ((cinfo)->err->msg_code = (code),                                            \
   (*(cinfo)->err->error_exit)((j_common_ptr)(cinfo)))

static boolean lr_fill_input_buffer(j_decompress_ptr cinfo)
{
    lr_jpg_src_ptr src = (lr_jpg_src_ptr)cinfo->src;
    size_t nbytes;

    nbytes = src->instream->read((void*)src->buffer, 1, LR_JPEG_INPUT_BUF_SIZE);

    if (nbytes <= 0)
    {
        if (src->start_of_file) /* Treat empty input file as fatal error */
            ERREXIT(cinfo, JERR_INPUT_EMPTY);
        WARNMS(cinfo, JWRN_JPEG_EOF);
        /* Insert a fake EOI marker */
        src->buffer[0] = (JOCTET)0xFF;
        src->buffer[1] = (JOCTET)JPEG_EOI;
        nbytes = 2;
    }

    src->pub.next_input_byte = src->buffer;
    src->pub.bytes_in_buffer = nbytes;
    src->start_of_file = FALSE;
    return TRUE;
}

static void lr_skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
    struct jpeg_source_mgr *src = cinfo->src;
    if (num_bytes > 0)
    {
        while (num_bytes > (long)src->bytes_in_buffer)
        {
            num_bytes -= (long)src->bytes_in_buffer;
            (void)(*src->fill_input_buffer)(cinfo);
            /* note we assume that fill_input_buffer will never return FALSE,
             * so suspension need not be handled.
             */
        }
        src->next_input_byte += (size_t)num_bytes;
        src->bytes_in_buffer -= (size_t)num_bytes;
    }
}

static void lr_term_source(j_decompress_ptr /*cinfo*/) {}

static void lr_jpeg_src(j_decompress_ptr cinfo, LibRaw_abstract_datastream *inf)
{
    lr_jpg_src_ptr src;
    if (cinfo->src == NULL)
    { /* first time for this JPEG object? */
        cinfo->src = (struct jpeg_source_mgr *)(*cinfo->mem->alloc_small)(
            (j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(lr_jpg_source_mgr));
        src = (lr_jpg_src_ptr)cinfo->src;
        src->buffer = (JOCTET *)(*cinfo->mem->alloc_small)(
            (j_common_ptr)cinfo, JPOOL_PERMANENT,
            LR_JPEG_INPUT_BUF_SIZE * sizeof(JOCTET));
    }
    else if (cinfo->src->init_source != f_init_source)
    {
        ERREXIT(cinfo, JERR_BUFFER_SIZE);
    }

    src = (lr_jpg_src_ptr)cinfo->src;
    src->pub.init_source = f_init_source;
    src->pub.fill_input_buffer = lr_fill_input_buffer;
    src->pub.skip_input_data = lr_skip_input_data;
    src->pub.resync_to_restart = jpeg_resync_to_restart; /* use default method */
    src->pub.term_source = lr_term_source;
    src->instream = inf;
    src->pub.bytes_in_buffer = 0;    /* forces fill_input_buffer on first read */
    src->pub.next_input_byte = NULL; /* until buffer loaded */
}
#endif

int LibRaw_abstract_datastream::jpeg_src(void *jpegdata)
{
#ifdef NO_JPEG
    return -1;
#else
    j_decompress_ptr cinfo = (j_decompress_ptr)jpegdata;
    buffering_off();
    lr_jpeg_src(cinfo, this);
    return 0; // OK
#endif
}


#ifndef LIBRAW_NO_IOSTREAMS_DATASTREAM
// == LibRaw_file_datastream ==

LibRaw_file_datastream::~LibRaw_file_datastream()
{
  if (jas_file)
    fclose(jas_file);
}

LibRaw_file_datastream::LibRaw_file_datastream(const char *fname)
    : filename(fname), _fsize(0)
#ifdef LIBRAW_WIN32_UNICODEPATHS
      ,
      wfilename()
#endif
      ,
      jas_file(NULL)
{
  if (filename.size() > 0)
  {
#ifndef LIBRAW_WIN32_CALLS
    struct stat st;
    if (!stat(filename.c_str(), &st))
      _fsize = st.st_size;
#else
    struct _stati64 st;
    if (!_stati64(filename.c_str(), &st))
      _fsize = st.st_size;
#endif
#ifdef LIBRAW_USE_AUTOPTR
    std::auto_ptr<std::filebuf> buf(new std::filebuf());
#else
    std::unique_ptr<std::filebuf> buf(new std::filebuf());
#endif
    buf->open(filename.c_str(), std::ios_base::in | std::ios_base::binary);
    if (buf->is_open())
    {
#ifdef LIBRAW_USE_AUTOPTR
      f = buf;
#else
      f = std::move(buf);
#endif
    }
  }
}
#ifdef LIBRAW_WIN32_UNICODEPATHS
LibRaw_file_datastream::LibRaw_file_datastream(const wchar_t *fname)
    : filename(), wfilename(fname), jas_file(NULL), _fsize(0)
{
  if (wfilename.size() > 0)
  {
    struct _stati64 st;
    if (!_wstati64(wfilename.c_str(), &st))
      _fsize = st.st_size;
#ifdef LIBRAW_USE_AUTOPTR
    std::auto_ptr<std::filebuf> buf(new std::filebuf());
#else
    std::unique_ptr<std::filebuf> buf(new std::filebuf());
#endif
    buf->open(wfilename.c_str(), std::ios_base::in | std::ios_base::binary);
    if (buf->is_open())
    {
#ifdef LIBRAW_USE_AUTOPTR
      f = buf;
#else
      f = std::move(buf);
#endif
	}
  }
}
const wchar_t *LibRaw_file_datastream::wfname()
{
  return wfilename.size() > 0 ? wfilename.c_str() : NULL;
}
#endif

int LibRaw_file_datastream::valid() { return f.get() ? 1 : 0; }

#define LR_STREAM_CHK()                                                        \
  do                                                                           \
  {                                                                            \
    if (!f.get())                                                              \
      throw LIBRAW_EXCEPTION_IO_EOF;                                           \
  } while (0)

int LibRaw_file_datastream::read(void *ptr, size_t size, size_t nmemb)
{
/* Visual Studio 2008 marks sgetn as insecure, but VS2010 does not. */
#if defined(WIN32SECURECALLS) && (_MSC_VER < 1600)
  LR_STREAM_CHK();
  return int(f->_Sgetn_s(static_cast<char *>(ptr), nmemb * size, nmemb * size) /
             (size > 0 ? size : 1));
#else
  LR_STREAM_CHK();
  return int(f->sgetn(static_cast<char *>(ptr), std::streamsize(nmemb * size)) /
             (size > 0 ? size : 1));
#endif
}

int LibRaw_file_datastream::eof()
{
  LR_STREAM_CHK();
  return f->sgetc() == EOF;
}

int LibRaw_file_datastream::seek(INT64 o, int whence)
{
  LR_STREAM_CHK();
  std::ios_base::seekdir dir;
  switch (whence)
  {
  case SEEK_SET:
    dir = std::ios_base::beg;
    break;
  case SEEK_CUR:
    dir = std::ios_base::cur;
    break;
  case SEEK_END:
    dir = std::ios_base::end;
    break;
  default:
    dir = std::ios_base::beg;
  }
  return f->pubseekoff((long)o, dir) < 0;
}

INT64 LibRaw_file_datastream::tell()
{
  LR_STREAM_CHK();
  return f->pubseekoff(0, std::ios_base::cur);
}

char *LibRaw_file_datastream::gets(char *str, int sz)
{
  if(sz<1) return NULL;
  LR_STREAM_CHK();
  std::istream is(f.get());
  is.getline(str, sz);
  if (is.fail())
    return 0;
  return str;
}

int LibRaw_file_datastream::scanf_one(const char *fmt, void *val)
{
  LR_STREAM_CHK();

  std::istream is(f.get());

  /* HUGE ASSUMPTION: *fmt is either "%d" or "%f" */
  if (strcmp(fmt, "%d") == 0)
  {
    int d;
    is >> d;
    if (is.fail())
      return EOF;
    *(static_cast<int *>(val)) = d;
  }
  else
  {
    float f;
    is >> f;
    if (is.fail())
      return EOF;
    *(static_cast<float *>(val)) = f;
  }

  return 1;
}

const char *LibRaw_file_datastream::fname()
{
  return filename.size() > 0 ? filename.c_str() : NULL;
}

#undef LR_STREAM_CHK

#ifdef LIBRAW_OLD_VIDEO_SUPPORT
void *LibRaw_file_datastream::make_jas_stream()
{
#ifdef NO_JASPER
  return NULL;
#else
#ifdef LIBRAW_WIN32_UNICODEPATHS
  if (wfname())
  {
    jas_file = _wfopen(wfname(), L"rb");
    return jas_stream_fdopen(fileno(jas_file), "rb");
  }
  else
#endif
  {
    return jas_stream_fopen(fname(), "rb");
  }
#endif
}
#endif
#endif

// == LibRaw_buffer_datastream
LibRaw_buffer_datastream::LibRaw_buffer_datastream(const void *buffer, size_t bsize)
{
  buf = (unsigned char *)buffer;
  streampos = 0;
  streamsize = bsize;
}

LibRaw_buffer_datastream::~LibRaw_buffer_datastream() {}

int LibRaw_buffer_datastream::read(void *ptr, size_t sz, size_t nmemb)
{
  size_t to_read = sz * nmemb;
  if (to_read > streamsize - streampos)
    to_read = streamsize - streampos;
  if (to_read < 1)
    return 0;
  memmove(ptr, buf + streampos, to_read);
  streampos += to_read;
  return int((to_read + sz - 1) / (sz > 0 ? sz : 1));
}

int LibRaw_buffer_datastream::seek(INT64 o, int whence)
{
  switch (whence)
  {
  case SEEK_SET:
    if (o < 0)
      streampos = 0;
    else if (size_t(o) > streamsize)
      streampos = streamsize;
    else
      streampos = size_t(o);
    return 0;
  case SEEK_CUR:
    if (o < 0)
    {
      if (size_t(-o) >= streampos)
        streampos = 0;
      else
        streampos += (size_t)o;
    }
    else if (o > 0)
    {
      if (o + streampos > streamsize)
        streampos = streamsize;
      else
        streampos += (size_t)o;
    }
    return 0;
  case SEEK_END:
    if (o > 0)
      streampos = streamsize;
    else if (size_t(-o) > streamsize)
      streampos = 0;
    else
      streampos = streamsize + (size_t)o;
    return 0;
  default:
    return 0;
  }
}

INT64 LibRaw_buffer_datastream::tell()
{
  return INT64(streampos);
}

char *LibRaw_buffer_datastream::gets(char *s, int sz)
{
  if(sz<1) return NULL;
  unsigned char *psrc, *pdest, *str;
  str = (unsigned char *)s;
  psrc = buf + streampos;
  pdest = str;
  if(streampos >= streamsize) return NULL;
  while ((size_t(psrc - buf) < streamsize) && ((pdest - str) < (sz-1)))
  {
    *pdest = *psrc;
    if (*psrc == '\n')
      break;
    psrc++;
    pdest++;
  }
  if (size_t(psrc - buf) < streamsize)
    psrc++;
  if ((pdest - str) < sz-1)
    *(++pdest) = 0;
  else
    s[sz - 1] = 0; // ensure trailing zero

  streampos = psrc - buf;
  return s;
}

int LibRaw_buffer_datastream::scanf_one(const char *fmt, void *val)
{
  int scanf_res;
  if (streampos > streamsize)
    return 0;
#ifndef WIN32SECURECALLS
  scanf_res = sscanf((char *)(buf + streampos), fmt, val);
#else
  scanf_res = sscanf_s((char *)(buf + streampos), fmt, val);
#endif
  if (scanf_res > 0)
  {
    int xcnt = 0;
    while (streampos < streamsize-1)
    {
      streampos++;
      xcnt++;
      if (buf[streampos] == 0 || buf[streampos] == ' ' ||
          buf[streampos] == '\t' || buf[streampos] == '\n' || xcnt > 24)
        break;
    }
  }
  return scanf_res;
}

int LibRaw_buffer_datastream::eof()
{
  return streampos >= streamsize;
}
int LibRaw_buffer_datastream::valid() { return buf ? 1 : 0; }

#ifdef LIBRAW_OLD_VIDEO_SUPPORT
void *LibRaw_buffer_datastream::make_jas_stream()
{
#ifdef NO_JASPER
  return NULL;
#else
  return jas_stream_memopen((char *)buf + streampos, streamsize - streampos);
#endif
}
#endif

int LibRaw_buffer_datastream::jpeg_src(void *jpegdata)
{
#if defined(NO_JPEG) || !defined(USE_JPEG)
  return -1;
#else
  j_decompress_ptr cinfo = (j_decompress_ptr)jpegdata;
  jpeg_mem_src(cinfo, (unsigned char *)buf + streampos,(unsigned long)(streamsize - streampos));
  return 0;
#endif
}

// int LibRaw_buffer_datastream

// == LibRaw_bigfile_datastream
LibRaw_bigfile_datastream::LibRaw_bigfile_datastream(const char *fname)
    : filename(fname)
#ifdef LIBRAW_WIN32_UNICODEPATHS
      ,
      wfilename()
#endif
{
  if (filename.size() > 0)
  {
#ifndef LIBRAW_WIN32_CALLS
    struct stat st;
    if (!stat(filename.c_str(), &st))
      _fsize = st.st_size;
#else
    struct _stati64 st;
    if (!_stati64(filename.c_str(), &st))
      _fsize = st.st_size;
#endif

#ifndef WIN32SECURECALLS
    f = fopen(fname, "rb");
#else
    if (fopen_s(&f, fname, "rb"))
      f = 0;
#endif
  }
  else
  {
    filename = std::string();
    f = 0;
  }
}

#ifdef LIBRAW_WIN32_UNICODEPATHS
LibRaw_bigfile_datastream::LibRaw_bigfile_datastream(const wchar_t *fname)
    : filename(), wfilename(fname)
{
  if (wfilename.size() > 0)
  {
    struct _stati64 st;
    if (!_wstati64(wfilename.c_str(), &st))
      _fsize = st.st_size;
#ifndef WIN32SECURECALLS
    f = _wfopen(wfilename.c_str(), L"rb");
#else
    if (_wfopen_s(&f, fname, L"rb"))
      f = 0;
#endif
  }
  else
  {
    wfilename = std::wstring();
    f = 0;
  }
}
const wchar_t *LibRaw_bigfile_datastream::wfname()
{
  return wfilename.size() > 0 ? wfilename.c_str() : NULL;
}
#endif

LibRaw_bigfile_datastream::~LibRaw_bigfile_datastream()
{
  if (f)
    fclose(f);
}
int LibRaw_bigfile_datastream::valid() { return f ? 1 : 0; }

#define LR_BF_CHK()                                                            \
  do                                                                           \
  {                                                                            \
    if (!f)                                                                    \
      throw LIBRAW_EXCEPTION_IO_EOF;                                           \
  } while (0)

int LibRaw_bigfile_datastream::read(void *ptr, size_t size, size_t nmemb)
{
  LR_BF_CHK();
  return int(fread(ptr, size, nmemb, f));
}

int LibRaw_bigfile_datastream::eof()
{
  LR_BF_CHK();
  return feof(f);
}

int LibRaw_bigfile_datastream::seek(INT64 o, int whence)
{
  LR_BF_CHK();
#if defined(_WIN32)
#ifdef WIN32SECURECALLS
  return _fseeki64(f, o, whence);
#else
  return fseek(f, (long)o, whence);
#endif
#else
  return fseeko(f, o, whence);
#endif
}

INT64 LibRaw_bigfile_datastream::tell()
{
  LR_BF_CHK();
#if defined(_WIN32)
#ifdef WIN32SECURECALLS
  return _ftelli64(f);
#else
  return ftell(f);
#endif
#else
  return ftello(f);
#endif
}

char *LibRaw_bigfile_datastream::gets(char *str, int sz)
{
  if(sz<1) return NULL;
  LR_BF_CHK();
  return fgets(str, sz, f);
}

int LibRaw_bigfile_datastream::scanf_one(const char *fmt, void *val)
{
  LR_BF_CHK();
  return 
#ifndef WIN32SECURECALLS
                   fscanf(f, fmt, val)
#else
                   fscanf_s(f, fmt, val)
#endif
      ;
}

const char *LibRaw_bigfile_datastream::fname()
{
  return filename.size() > 0 ? filename.c_str() : NULL;
}

#ifdef LIBRAW_OLD_VIDEO_SUPPORT
void *LibRaw_bigfile_datastream::make_jas_stream()
{
#ifdef NO_JASPER
  return NULL;
#else
  return jas_stream_fdopen(fileno(f), "rb");
#endif
}
#endif

// == LibRaw_windows_datastream
#ifdef LIBRAW_WIN32_CALLS

LibRaw_windows_datastream::LibRaw_windows_datastream(const TCHAR *sFile)
    : LibRaw_buffer_datastream(NULL, 0), hMap_(0), pView_(NULL)
{
#if defined(WINAPI_FAMILY) && defined(WINAPI_FAMILY_APP) && (WINAPI_FAMILY == WINAPI_FAMILY_APP)
    HANDLE hFile = CreateFile2(sFile, GENERIC_READ, 0, OPEN_EXISTING, 0);
#else
  HANDLE hFile = CreateFile(sFile, GENERIC_READ, 0, 0, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, 0);
#endif
  if (hFile == INVALID_HANDLE_VALUE)
    throw std::runtime_error("failed to open the file");

  try
  {
    Open(hFile);
  }
  catch (...)
  {
    CloseHandle(hFile);
    throw;
  }

  CloseHandle(hFile); // windows will defer the actual closing of this handle
                      // until the hMap_ is closed
  reconstruct_base();
}

// ctor: construct with a file handle - caller is responsible for closing the
// file handle
LibRaw_windows_datastream::LibRaw_windows_datastream(HANDLE hFile)
    : LibRaw_buffer_datastream(NULL, 0), hMap_(0), pView_(NULL)
{
  Open(hFile);
  reconstruct_base();
}

// dtor: unmap and close the mapping handle
LibRaw_windows_datastream::~LibRaw_windows_datastream()
{
  if (pView_ != NULL)
    ::UnmapViewOfFile(pView_);

  if (hMap_ != 0)
    ::CloseHandle(hMap_);
}

void LibRaw_windows_datastream::Open(HANDLE hFile)
{
  // create a file mapping handle on the file handle
  hMap_ = ::CreateFileMapping(hFile, 0, PAGE_READONLY, 0, 0, 0);
  if (hMap_ == NULL)
    throw std::runtime_error("failed to create file mapping");

  // now map the whole file base view
  if (!::GetFileSizeEx(hFile, (PLARGE_INTEGER)&cbView_))
    throw std::runtime_error("failed to get the file size");

  pView_ = ::MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, (size_t)cbView_);
  if (pView_ == NULL)
    throw std::runtime_error("failed to map the file");
}

#endif

#if defined (LIBRAW_NO_IOSTREAMS_DATASTREAM)  && defined (LIBRAW_WIN32_CALLS)

/* LibRaw_bigfile_buffered_datastream: copypasted from LibRaw_bigfile_datastream + extra cache on read */

#undef LR_BF_CHK
#define LR_BF_CHK()                                                    \
  do                                                                    \
  {                                                                     \
     if (fhandle ==0 || fhandle == INVALID_HANDLE_VALUE)                \
         throw LIBRAW_EXCEPTION_IO_EOF;                                 \
  } while (0)

#define LIBRAW_BUFFER_ALIGN 4096

int LibRaw_bufio_params::bufsize = 16384;

void LibRaw_bufio_params::set_bufsize(int bs)
{
    if (bs > 0)
        bufsize = bs;
}


LibRaw_bigfile_buffered_datastream::LibRaw_bigfile_buffered_datastream(const char *fname)
    : filename(fname), _fsize(0), _fpos(0)
#ifdef LIBRAW_WIN32_UNICODEPATHS
    , wfilename()
#endif
    , iobuffers(), buffered(1)
{
    if (filename.size() > 0)
    {
        std::string fn(fname);
        std::wstring fpath(fn.begin(), fn.end());
#if defined(WINAPI_FAMILY) && defined(WINAPI_FAMILY_APP) && (WINAPI_FAMILY == WINAPI_FAMILY_APP)
        if ((fhandle = CreateFile2(fpath.c_str(), GENERIC_READ, 0, OPEN_EXISTING, 0)) != INVALID_HANDLE_VALUE)
#else
        if ((fhandle = CreateFileW(fpath.c_str(), GENERIC_READ, FILE_SHARE_READ, 0,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0)) != INVALID_HANDLE_VALUE)
#endif
        {
            LARGE_INTEGER fs;
            if (GetFileSizeEx(fhandle, &fs))
                _fsize = fs.QuadPart;
        }
    }
    else
    {
        filename = std::string();
        fhandle = INVALID_HANDLE_VALUE;
    }
}

#ifdef LIBRAW_WIN32_UNICODEPATHS
LibRaw_bigfile_buffered_datastream::LibRaw_bigfile_buffered_datastream(const wchar_t *fname)
    : filename(), _fsize(0), _fpos(0),
    wfilename(fname), iobuffers(), buffered(1)
{
    if (wfilename.size() > 0)
    {
#if defined(WINAPI_FAMILY) && defined(WINAPI_FAMILY_APP) && (WINAPI_FAMILY == WINAPI_FAMILY_APP)
        if ((fhandle = CreateFile2(wfilename.c_str(), GENERIC_READ, 0, OPEN_EXISTING, 0)) != INVALID_HANDLE_VALUE)
#else
        if ((fhandle = CreateFileW(wfilename.c_str(), GENERIC_READ, FILE_SHARE_READ, 0,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0)) != INVALID_HANDLE_VALUE)
#endif
        {
            LARGE_INTEGER fs;
            if (GetFileSizeEx(fhandle, &fs))
                _fsize = fs.QuadPart;
        }

    }
    else
    {
        wfilename = std::wstring();
        fhandle = INVALID_HANDLE_VALUE;
    }
}

const wchar_t *LibRaw_bigfile_buffered_datastream::wfname()
{
    return wfilename.size() > 0 ? wfilename.c_str() : NULL;
}
#endif

LibRaw_bigfile_buffered_datastream::~LibRaw_bigfile_buffered_datastream()
{
    if (valid())
        CloseHandle(fhandle);
}
int LibRaw_bigfile_buffered_datastream::valid() {
    return (fhandle != NULL) && (fhandle != INVALID_HANDLE_VALUE);
}

const char *LibRaw_bigfile_buffered_datastream::fname()
{
    return filename.size() > 0 ? filename.c_str() : NULL;
}

#ifdef LIBRAW_OLD_VIDEO_SUPPORT
void *LibRaw_bigfile_buffered_datastream::make_jas_stream()
{
#ifdef NO_JASPER
    return NULL;
#else
    return NULL;
#endif
}
#endif

INT64 LibRaw_bigfile_buffered_datastream::readAt(void *ptr, size_t size, INT64 off)
{
    LR_BF_CHK();
    DWORD NumberOfBytesRead;
    DWORD nNumberOfBytesToRead = (DWORD)size;
    struct _OVERLAPPED olap;
    memset(&olap, 0, sizeof(olap));
    olap.Offset = off & 0xffffffff;
    olap.OffsetHigh = off >> 32;
    if (ReadFile(fhandle, ptr, nNumberOfBytesToRead, &NumberOfBytesRead, &olap))
        return NumberOfBytesRead;
    else
        return 0;
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#ifdef _MSC_VER
#pragma intrinsic(memcpy)
#endif

int LibRaw_bigfile_buffered_datastream::read(void *data, size_t size, size_t nmemb)
{
    if (size < 1 || nmemb < 1)
        return 0;
    LR_BF_CHK();
    INT64 count = size * nmemb;
    INT64 partbytes = 0;
    if (!buffered)
    {
        INT64 r = readAt(data, count, _fpos);
        _fpos += r;
        return int(r / size);
    }

    unsigned char *fBuffer = (unsigned char*)iobuffers[0].data();
    while (count)
    {
        INT64 inbuffer = 0;
        // See if the request is totally inside buffer.
        if (iobuffers[0].contains(_fpos, inbuffer))
        {
            if (inbuffer >= count)
            {
                memcpy(data, fBuffer + (unsigned)(_fpos - iobuffers[0]._bstart), count);
                _fpos += count;
                return int((count + partbytes) / size);
            }
            memcpy(data, fBuffer + (_fpos - iobuffers[0]._bstart), inbuffer);
            partbytes += inbuffer;
            count -= inbuffer;
            data = (void *)(((char *)data) + inbuffer);
            _fpos += inbuffer;
        }
        if (count > (INT64) iobuffers[0].size())
        {
        fallback:
            if (_fpos + count > _fsize)
                count = MAX(0, _fsize - _fpos);
            if (count > 0)
            {
                INT64 r = readAt(data, count, _fpos);
                _fpos += r;
                return int((r + partbytes) / size);
            }
            else
                return 0;
        }

        if (!fillBufferAt(0, _fpos))
            goto fallback;
    }
    return 0;
}

bool LibRaw_bigfile_buffered_datastream::fillBufferAt(int bi, INT64 off)
{
    if (off < 0LL) return false;
    iobuffers[bi]._bstart = off;
    if (iobuffers[bi].size() >= LIBRAW_BUFFER_ALIGN * 2)// Align to a file block.
        iobuffers[bi]._bstart &= (INT64)~((INT64)(LIBRAW_BUFFER_ALIGN - 1));

    iobuffers[bi]._bend = MIN(iobuffers[bi]._bstart + (INT64)iobuffers[bi].size(), _fsize);
    if (iobuffers[bi]._bend <= off) // Buffer alignment problem, fallback
        return false;
    INT64 rr = readAt(iobuffers[bi].data(), (uint32_t)(iobuffers[bi]._bend - iobuffers[bi]._bstart), iobuffers[bi]._bstart);
    if (rr > 0)
    {
        iobuffers[bi]._bend = iobuffers[bi]._bstart + rr;
        return true;
    }
    return false;
}


int LibRaw_bigfile_buffered_datastream::eof()
{
    LR_BF_CHK();
    return _fpos >= _fsize;
}

int LibRaw_bigfile_buffered_datastream::seek(INT64 o, int whence)
{
    LR_BF_CHK();
    if (whence == SEEK_SET) _fpos = o;
    else if (whence == SEEK_END) _fpos = o > 0 ? _fsize : _fsize + o;
    else if (whence == SEEK_CUR) _fpos += o;
    return 0;
}

INT64 LibRaw_bigfile_buffered_datastream::tell()
{
    LR_BF_CHK();
    return _fpos;
}

char *LibRaw_bigfile_buffered_datastream::gets(char *s, int sz)
{
    if (sz < 1)
        return NULL;
    else if (sz < 2)
    {
        s[0] = 0;
        return s;
    }

    LR_BF_CHK();
    INT64 contains;
    int bufindex = selectStringBuffer(sz, contains);
    if (bufindex < 0) return NULL;
    if (contains >= sz)
    {
        unsigned char *buf = iobuffers[bufindex].data() + (_fpos - iobuffers[bufindex]._bstart);
        int streampos = 0;
        int streamsize = contains;
        unsigned char *str = (unsigned char *)s;
        unsigned char *psrc, *pdest;
        psrc = buf + streampos;
        pdest = str;

        while ((size_t(psrc - buf) < streamsize) && ((pdest - str) < sz-1)) // sz-1: to append \0
        {
            *pdest = *psrc;
            if (*psrc == '\n')
                break;
            psrc++;
            pdest++;
        }
        if (size_t(psrc - buf) < streamsize)
            psrc++;
        if ((pdest - str) < sz - 1)
            *(++pdest) = 0;
        else
            s[sz - 1] = 0; // ensure trailing zero
        streampos = psrc - buf;
        _fpos += streampos;
        return s;
    }
    return NULL;
}

int LibRaw_bigfile_buffered_datastream::selectStringBuffer(INT64 len, INT64& contains)
{
    if (iobuffers[0].contains(_fpos, contains) && contains >= len)
        return 0;

    if (iobuffers[1].contains(_fpos, contains) && contains >= len)
        return 1;

    fillBufferAt(1, _fpos);
    if (iobuffers[1].contains(_fpos, contains) && contains >= len)
        return 1;
    return -1;
}

int LibRaw_bigfile_buffered_datastream::scanf_one(const char *fmt, void *val)
{
    LR_BF_CHK();
    INT64 contains = 0;
    int bufindex = selectStringBuffer(24, contains);
    if (bufindex < 0) return -1;
    if (contains >= 24)
    {
        unsigned char *bstart = iobuffers[bufindex].data() + (_fpos - iobuffers[bufindex]._bstart);
        int streampos = 0;
        int streamsize = contains;
        int
#ifndef WIN32SECURECALLS
            scanf_res = sscanf((char *)(bstart), fmt, val);
#else
            scanf_res = sscanf_s((char *)(bstart), fmt, val);
#endif
        if (scanf_res > 0)
        {
            int xcnt = 0;
            while (streampos < streamsize)
            {
                streampos++;
                xcnt++;
                if (bstart[streampos] == 0 || bstart[streampos] == ' ' ||
                    bstart[streampos] == '\t' || bstart[streampos] == '\n' || xcnt > 24)
                    break;
            }
            _fpos += streampos;
            return scanf_res;
        }
    }
    return -1;
}

#endif

