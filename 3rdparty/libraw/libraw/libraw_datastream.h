/* -*- C -*-
 * File: libraw_datastream.h
 * Copyright 2008-2021 LibRaw LLC (info@libraw.org)
 * Created: Sun Jan 18 13:07:35 2009
 *
 * LibRaw Data stream interface

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#ifndef __LIBRAW_DATASTREAM_H
#define __LIBRAW_DATASTREAM_H

#include <stdio.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>

#ifndef __cplusplus

#else /* __cplusplus */
#if defined _WIN32
#ifndef LIBRAW_NO_WINSOCK2
#include <winsock2.h>
#endif
#endif
/* No unique_ptr on Apple ?? */
#if __cplusplus >= 201103L || (defined(_CPPLIB_VER) && _CPPLIB_VER >= 520) ||  \
    (defined(_MSC_VER) && _MSVC_LANG >= 201103L)
/* OK - use unique_ptr unless LIBRAW_USE_AUTOPTR defined externally*/
#else
/* Force to use auto_ptr */
#ifndef LIBRAW_USE_AUTOPTR
#define LIBRAW_USE_AUTOPTR
#endif
#endif

#include "libraw_const.h"
#include "libraw_types.h"
#include <fstream>
#include <memory>
#include <vector>

#if defined(_WIN32) && (_MSC_VER) >= 1500
#define WIN32SECURECALLS
#endif

#ifdef USE_DNGSDK

#if defined LIBRAW_WIN32_CALLS
#define qWinOS 1
#define qMacOS 0
#elif defined(__APPLE__)
#define qWinOS 0
#define qMacOS 1
#else
/* define OS types for DNG here */
#endif
#define qDNGXMPDocOps 0
#define qDNGUseLibJPEG 1
#define qDNGXMPFiles 0
#define qDNGExperimental 1
#define qDNGThreadSafe 1
#include "dng_stream.h"
#endif /* DNGSDK */

#define IOERROR()                                                              \
  do                                                                           \
  {                                                                            \
    throw LIBRAW_EXCEPTION_IO_EOF;                                             \
  } while (0)

class LibRaw_buffer_datastream;
class LibRaw_bit_buffer;

class DllDef LibRaw_abstract_datastream
{
public:
  LibRaw_abstract_datastream() { };
  virtual ~LibRaw_abstract_datastream(void) { }
  virtual int valid() = 0;
  virtual int read(void *, size_t, size_t) = 0;
  virtual int seek(INT64, int) = 0;
  virtual INT64 tell() = 0;
  virtual INT64 size() = 0;
  virtual int get_char() = 0;
  virtual char *gets(char *, int) = 0;
  virtual int scanf_one(const char *, void *) = 0;
  virtual int eof() = 0;
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
  virtual void *make_jas_stream() = 0;
#endif
  virtual int jpeg_src(void *);
  virtual void buffering_off() {}
  /* reimplement in subclass to use parallel access in xtrans_load_raw() if
   * OpenMP is not used */
  virtual int lock() { return 1; } /* success */
  virtual void unlock() {}
  virtual const char *fname() { return NULL; };
#ifdef LIBRAW_WIN32_UNICODEPATHS
  virtual const wchar_t *wfname() { return NULL; };
#endif
};

#ifndef LIBRAW_NO_IOSTREAMS_DATASTREAM

#ifdef LIBRAW_WIN32_DLLDEFS
#ifdef LIBRAW_USE_AUTOPTR
template class DllDef std::auto_ptr<std::streambuf>;
#else
template class DllDef std::unique_ptr<std::streambuf>;
#endif
#endif

class DllDef LibRaw_file_datastream : public LibRaw_abstract_datastream
{
protected:
#ifdef LIBRAW_USE_AUTOPTR
  std::auto_ptr<std::streambuf> f; /* will close() automatically through dtor */
#else
  std::unique_ptr<std::streambuf> f;
#endif
  std::string filename;
  INT64 _fsize;
#ifdef LIBRAW_WIN32_UNICODEPATHS
  std::wstring wfilename;
#endif
  FILE *jas_file;

public:
  virtual ~LibRaw_file_datastream();
  LibRaw_file_datastream(const char *fname);
#ifdef LIBRAW_WIN32_UNICODEPATHS
  LibRaw_file_datastream(const wchar_t *fname);
#endif
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
  virtual void *make_jas_stream();
#endif
  virtual int valid();
  virtual int read(void *ptr, size_t size, size_t nmemb);
  virtual int eof();
  virtual int seek(INT64 o, int whence);
  virtual INT64 tell();
  virtual INT64 size() { return _fsize; }
  virtual int get_char() {return f->sbumpc();}
  virtual char *gets(char *str, int sz);
  virtual int scanf_one(const char *fmt, void *val);
  virtual const char *fname();
#ifdef LIBRAW_WIN32_UNICODEPATHS
  virtual const wchar_t *wfname();
#endif
};
#endif

#if defined (LIBRAW_NO_IOSTREAMS_DATASTREAM)  && defined (LIBRAW_WIN32_CALLS)

struct DllDef LibRaw_bufio_params
{
    static int bufsize;
    static void set_bufsize(int bs);
};

class buffer_t : public std::vector<unsigned char>
{
public:
    INT64 _bstart, _bend;
    buffer_t() : std::vector<unsigned char>(LibRaw_bufio_params::bufsize), _bstart(0), _bend(0) {}
    int charOReof(INT64 _fpos)
    {
        if (_bstart < 0LL || _bend < 0LL || _bend < _bstart || _fpos < 0LL)  
            return -1;
        if ((_bend - _bstart) > (INT64)size()) 
            return -1;
        if (_fpos >= _bstart && _fpos < _bend)
            return data()[_fpos - _bstart];
        return -1;
    }
    bool contains(INT64 _fpos, INT64& contains)
    {
        if (_bstart < 0LL || _bend < 0LL || _bend < _bstart || _fpos < 0LL)
        {
            contains = 0;
            return false;
        }
        if ((_bend - _bstart) > (INT64)size())
        {
          contains = 0;
          return false;
        }       
        if (_fpos >= _bstart && _fpos < _bend)
        {
            contains = _bend - _fpos;
            return true;
        }
        contains = 0;
        return false;
    }
};


class DllDef LibRaw_bigfile_buffered_datastream : public LibRaw_abstract_datastream
{
public:
    LibRaw_bigfile_buffered_datastream(const char *fname);
#ifdef LIBRAW_WIN32_UNICODEPATHS
    LibRaw_bigfile_buffered_datastream(const wchar_t *fname);
#endif
    virtual ~LibRaw_bigfile_buffered_datastream();
    virtual int valid();
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
    virtual void *make_jas_stream();
#endif
    virtual void buffering_off() { buffered = 0; }
    virtual int read(void *ptr, size_t size, size_t nmemb);
    virtual int eof();
    virtual int seek(INT64 o, int whence);
    virtual INT64 tell();
    virtual INT64 size() { return _fsize; }
    virtual char *gets(char *str, int sz);
    virtual int scanf_one(const char *fmt, void *val);
    virtual const char *fname();
#ifdef LIBRAW_WIN32_UNICODEPATHS
    virtual const wchar_t *wfname();
#endif
    virtual int get_char()
    {
        int r = iobuffers[0].charOReof(_fpos);
        if (r >= 0)
        {
            _fpos++;
            return r;
        }
        unsigned char c;
        r = read(&c, 1, 1);
        return r > 0 ? c : r;
    }

protected:
    INT64   readAt(void *ptr, size_t size, INT64 off);
    bool	fillBufferAt(int buf, INT64 off);
    int		selectStringBuffer(INT64 len, INT64& contains);
    HANDLE fhandle;
    INT64 _fsize;
    INT64 _fpos; /* current file position; current buffer start position */
#ifdef LIBRAW_WIN32_UNICODEPATHS
    std::wstring wfilename;
#endif
    std::string filename;
    buffer_t iobuffers[2];
    int buffered;
};

#endif

class DllDef LibRaw_buffer_datastream : public LibRaw_abstract_datastream
{
public:
  LibRaw_buffer_datastream(const void *buffer, size_t bsize);
  virtual ~LibRaw_buffer_datastream();
  virtual int valid();
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
  virtual void *make_jas_stream();
#endif
  virtual int jpeg_src(void *jpegdata);
  virtual int read(void *ptr, size_t sz, size_t nmemb);
  virtual int eof();
  virtual int seek(INT64 o, int whence);
  virtual INT64 tell();
  virtual INT64 size() { return streamsize; }
  virtual char *gets(char *s, int sz);
  virtual int scanf_one(const char *fmt, void *val);
  virtual int get_char()
  {
    if (streampos >= streamsize)   return -1;
    return buf[streampos++];
  }

private:
  unsigned char *buf;
  size_t streampos, streamsize;
};

class DllDef LibRaw_bigfile_datastream : public LibRaw_abstract_datastream
{
public:
  LibRaw_bigfile_datastream(const char *fname);
#ifdef LIBRAW_WIN32_UNICODEPATHS
  LibRaw_bigfile_datastream(const wchar_t *fname);
#endif
  virtual ~LibRaw_bigfile_datastream();
  virtual int valid();
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
  virtual void *make_jas_stream();
#endif

  virtual int read(void *ptr, size_t size, size_t nmemb);
  virtual int eof();
  virtual int seek(INT64 o, int whence);
  virtual INT64 tell();
  virtual INT64 size() { return _fsize; }
  virtual char *gets(char *str, int sz);
  virtual int scanf_one(const char *fmt, void *val);
  virtual const char *fname();
#ifdef LIBRAW_WIN32_UNICODEPATHS
  virtual const wchar_t *wfname();
#endif
  virtual int get_char()
  {
#ifndef LIBRAW_WIN32_CALLS
    return getc_unlocked(f);
#else
    return fgetc(f);
#endif
  }

protected:
  FILE *f;
  std::string filename;
  INT64 _fsize;
#ifdef LIBRAW_WIN32_UNICODEPATHS
  std::wstring wfilename;
#endif
};

#ifdef LIBRAW_WIN32_CALLS
class DllDef LibRaw_windows_datastream : public LibRaw_buffer_datastream
{
public:
  /* ctor: high level constructor opens a file by name */
  LibRaw_windows_datastream(const TCHAR *sFile);
  /* ctor: construct with a file handle - caller is responsible for closing the
   * file handle */
  LibRaw_windows_datastream(HANDLE hFile);
  /* dtor: unmap and close the mapping handle */
  virtual ~LibRaw_windows_datastream();
  virtual INT64 size() { return cbView_; }

protected:
  void Open(HANDLE hFile);
  inline void reconstruct_base()
  {
    /* this subterfuge is to overcome the private-ness of
     * LibRaw_buffer_datastream */
    (LibRaw_buffer_datastream &)*this =
        LibRaw_buffer_datastream(pView_, (size_t)cbView_);
  }

  HANDLE hMap_;    /* handle of the file mapping */
  void *pView_;    /* pointer to the mapped memory */
  __int64 cbView_; /* size of the mapping in bytes */
};

#endif

#ifdef USE_DNGSDK

class libraw_dng_stream : public dng_stream
{
public:
  libraw_dng_stream(LibRaw_abstract_datastream *p)
      : dng_stream((dng_abort_sniffer *)NULL, kBigBufferSize, 0),
        parent_stream(p)
  {
    if (parent_stream)
    {
        parent_stream->buffering_off();
      off = parent_stream->tell();
      parent_stream->seek(0UL, SEEK_SET); /* seek to start */
    }
  }
  ~libraw_dng_stream()
  {
    if (parent_stream)
      parent_stream->seek(off, SEEK_SET);
  }
  virtual uint64 DoGetLength()
  {
    if (parent_stream)
      return parent_stream->size();
    return 0;
  }
  virtual void DoRead(void *data, uint32 count, uint64 offset)
  {
    if (parent_stream)
    {
      parent_stream->seek(offset, SEEK_SET);
      parent_stream->read(data, 1, count);
    }
  }

private:
  libraw_dng_stream(const libraw_dng_stream &stream);
  libraw_dng_stream &operator=(const libraw_dng_stream &stream);
  LibRaw_abstract_datastream *parent_stream;
  INT64 off;
};

#endif

#endif /* cplusplus */

#endif
