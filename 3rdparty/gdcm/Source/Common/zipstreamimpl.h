/*
zipstream Library License:
--------------------------

The zlib/libpng License Copyright (c) 2003 Jonathan de Halleux.

This software is provided 'as-is', without any express or implied warranty. In
no event will the authors be held liable for any damages arising from the use
of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim
   that you wrote the original software. If you use this software in a
   product, an acknowledgment in the product documentation would be
   appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution

Author: Jonathan de Halleux, dehalleux@pelikhan.com, 2003

Altered by: Andreas Zieringer 2003 for OpenSG project
            made it platform independent, gzip conform, fixed gzip footer

Altered by: Geoffrey Hutchison 2005 for Open Babel project
            minor namespace modifications, VC++ compatibility

Altered by: Mathieu Malaterre 2008, for GDCM project
            when reading deflate bit stream in DICOM special handling of \0 is needed
            also when writing deflate back to disk, the add_footer must be called
*/

/*
Code is from:

http://www.codeproject.com/KB/stl/zipstream.aspx

 TODO:

zran.c
    index a zlib or gzip stream and randomly access it
    - illustrates the use of Z_BLOCK, inflatePrime(), and
      inflateSetDictionary() to provide random access

  So I might after all be able to implement seeking :)
*/
#ifndef _ZIPSTREAM_H_
#define _ZIPSTREAM_H_

#include <vector>
#include <string>
#include <streambuf>
#include <sstream>
#include <iostream>
#include <algorithm>

#include <string.h> // memcpy
#include <stdio.h> // EOF

#include <gdcm_zlib.h>

#ifdef WIN32 /* Window 95 & Windows NT */
#  define OS_CODE  0x0b
#endif
#if defined(MACOS) || defined(TARGET_OS_MAC)
#  define OS_CODE  0x07
#endif
#ifndef OS_CODE
#  define OS_CODE  0x03  /* assume Unix */
#endif

namespace zlib_stream {

namespace detail
{
    const int gz_magic[2] = {0x1f, 0x8b}; /* gzip magic header */

    /* gzip flag byte */
    const int gz_ascii_flag =  0x01; /* bit 0 set: file probably ascii text */
    const int gz_head_crc    = 0x02; /* bit 1 set: header CRC present */
    const int gz_extra_field = 0x04; /* bit 2 set: extra field present */
    const int gz_orig_name  =  0x08; /* bit 3 set: original file name present */
    const int gz_comment    =  0x10; /* bit 4 set: file comment present */
    const int gz_reserved   =  0xE0; /* bits 5..7: reserved */
}

/// default gzip buffer size,
/// change this to suite your needs
const size_t zstream_default_buffer_size = 4096;

/// Compression strategy, see zlib doc.
enum EStrategy
{
    StrategyFiltered = 1,
    StrategyHuffmanOnly = 2,
    DefaultStrategy = 0
};

//*****************************************************************************
//  template class basic_zip_streambuf
//*****************************************************************************

/** \brief A stream decorator that takes raw input and zips it to a ostream.

The class wraps up the inflate method of the zlib library 1.1.4 http://www.gzip.org/zlib/
*/
template <class charT,
          class traits = std::char_traits<charT> >
class basic_zip_streambuf : public std::basic_streambuf<charT, traits>
{
public:
    typedef std::basic_ostream<charT, traits>& ostream_reference;
    typedef unsigned char byte_type;
    typedef char          char_type;
    typedef byte_type* byte_buffer_type;
    typedef std::vector<byte_type> byte_vector_type;
    typedef std::vector<char_type> char_vector_type;
    typedef int int_type;

    basic_zip_streambuf(ostream_reference ostream,
                            int level,
                            EStrategy strategy,
                            int window_size,
                            int memory_level,
                            size_t buffer_size);

    ~basic_zip_streambuf(void);

    int               sync        (void);
    int_type          overflow    (int_type c);
    std::streamsize   flush       (void);
    inline
    ostream_reference get_ostream (void) const;
    inline
    int               get_zerr    (void) const;
    inline
    unsigned long     get_crc     (void) const;
    inline
    unsigned long     get_in_size (void) const;
    inline
    long              get_out_size(void) const;

private:

    bool              zip_to_stream(char_type *buffer,
                                    std::streamsize buffer_size);

    ostream_reference   _ostream;
    z_stream            _zip_stream;
    int                 _err;
    byte_vector_type    _output_buffer;
    char_vector_type    _buffer;
    unsigned long       _crc;
};


//*****************************************************************************
//  template class basic_unzip_streambuf
//*****************************************************************************

/** \brief A stream decorator that takes compressed input and unzips it to a istream.

The class wraps up the deflate method of the zlib library 1.1.4 http://www.gzip.org/zlib/
*/
template <class charT,
          class traits = std::char_traits<charT> >
class basic_unzip_streambuf :
    public std::basic_streambuf<charT, traits>
{
public:
    typedef std::basic_istream<charT,traits>& istream_reference;
    typedef unsigned char byte_type;
    typedef charT          char_type;
    typedef byte_type* byte_buffer_type;
    typedef std::vector<byte_type> byte_vector_type;
    typedef std::vector<char_type> char_vector_type;
    typedef int int_type;

    /** Construct a unzip stream
     * More info on the following parameters can be found in the zlib documentation.
     */
    basic_unzip_streambuf(istream_reference istream,
                          int window_size,
                          size_t read_buffer_size,
                          size_t input_buffer_size);

    ~basic_unzip_streambuf(void);

    int_type underflow(void);

    /// returns the compressed input istream
    inline
    istream_reference get_istream              (void);
    inline
    z_stream&         get_zip_stream           (void);
    inline
    int               get_zerr                 (void) const;
    inline
    unsigned long     get_crc                  (void) const;
    inline
    long              get_out_size             (void) const;
    inline
    long              get_in_size              (void) const;


private:

    void              put_back_from_zip_stream (void);

    std::streamsize   unzip_from_stream        (char_type* buffer,
                                                std::streamsize buffer_size);

    size_t            fill_input_buffer        (void);

    istream_reference   _istream;
    z_stream            _zip_stream;
    int                 _err;
    byte_vector_type    _input_buffer;
    char_vector_type    _buffer;
    unsigned long       _crc;
};

//*****************************************************************************
//  template class basic_zip_ostream
//*****************************************************************************

template <class charT,
          class traits = std::char_traits<charT> >
class basic_zip_ostream :
    public basic_zip_streambuf<charT, traits>,
    public std::basic_ostream<charT, traits>
{
public:

    typedef char char_type;
    typedef std::basic_ostream<charT, traits>& ostream_reference;
    typedef std::basic_ostream<charT, traits> ostream_type;

    inline
    explicit basic_zip_ostream(ostream_reference ostream,
                               bool is_gzip = false,
                               int level = Z_DEFAULT_COMPRESSION,
                               EStrategy strategy = DefaultStrategy,
                               int window_size = -15 /*windowBits is passed < 0 to suppress zlib header */,
                               int memory_level = 8,
                               size_t buffer_size = zstream_default_buffer_size);

    ~basic_zip_ostream(void);

    inline
    bool                              is_gzip   (void) const;
    inline
    basic_zip_ostream<charT, traits>& zflush    (void);
    void                              finished  (void);

//    void                              make_gzip()
//       { add_header(); _is_gzip = true; }

private:

    basic_zip_ostream<charT,traits>&  add_header(void);
    basic_zip_ostream<charT,traits>&  add_footer(void);

    bool _is_gzip;
    bool _added_footer;
};

//*****************************************************************************
//  template class basic_zip_istream
//*****************************************************************************

template <class charT,
          class traits = std::char_traits<charT> >
class basic_zip_istream :
    public basic_unzip_streambuf<charT, traits>,
    public std::basic_istream<charT, traits>
{
public:
    typedef std::basic_istream<charT, traits>& istream_reference;
    typedef std::basic_istream<charT, traits> istream_type;

    explicit basic_zip_istream(istream_reference istream,
                               int window_size = -15 /*windowBits is passed < 0 to suppress zlib header */,
                               size_t read_buffer_size = zstream_default_buffer_size,
                               size_t input_buffer_size = zstream_default_buffer_size);

    inline
    bool     is_gzip           (void) const;
    inline
    bool     check_crc         (void);
    inline
    bool     check_data_size   (void) const;
    inline
    long     get_gzip_crc      (void) const;
    inline
    long     get_gzip_data_size(void) const;

protected:

    int      check_header      (void);
    void     read_footer       (void);

    bool _is_gzip;
    long _gzip_crc;
    long _gzip_data_size;
};

/// A typedef for basic_zip_ostream<char>
typedef basic_zip_ostream<char> zip_ostream;
/// A typedef for basic_zip_istream<char>
typedef basic_zip_istream<char> zip_istream;

/// A typedef for basic_zip_ostream<wchar_t>
//typedef basic_zip_ostream<wchar_t> zip_wostream;
/// A typedef for basic_zip_istream<wchart>
//typedef basic_zip_istream<wchar_t> zip_wistream;

//! Helper function to check whether stream is compressed or not.
inline bool isGZip(std::istream &is)
{
    const int gz_magic[2] = {0x1f, 0x8b};

    int c1 = is.get();
    if(c1 != gz_magic[0])
    {
        is.putback((char)c1);
        return false;
    }

    int c2 = is.get();
    if(c2 != gz_magic[1])
    {
        is.putback((char)c2);
        is.putback((char)c1);
        return false;
    }

    is.putback((char)c2);
    is.putback((char)c1);
    return true;
}

#include "zipstreamimpl.hpp"

}

#endif // _ZIPSTREAM_H_
