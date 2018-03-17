/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#ifdef HAVE_PNG

/****************************************************************************************\
    This part of the file implements PNG codec on base of libpng library,
    in particular, this code is based on example.c from libpng
    (see otherlibs/_graphics/readme.txt for copyright notice)
    and png2bmp sample from libpng distribution (Copyright (C) 1999-2001 MIYASAKA Masaru)
\****************************************************************************************/

#ifndef _LFS64_LARGEFILE
#  define _LFS64_LARGEFILE 0
#endif
#ifndef _FILE_OFFSET_BITS
#  define _FILE_OFFSET_BITS 0
#endif

#ifdef HAVE_LIBPNG_PNG_H
#include <libpng/png.h>
#else
#include <png.h>
#endif
#include <zlib.h>

#include "grfmt_png.hpp"

#if defined _MSC_VER && _MSC_VER >= 1200
    // interaction between '_setjmp' and C++ object destruction is non-portable
    #pragma warning( disable: 4611 )
#endif

// the following defines are a hack to avoid multiple problems with frame ponter handling and setjmp
// see http://gcc.gnu.org/ml/gcc/2011-10/msg00324.html for some details
#define mingw_getsp(...) 0
#define __builtin_frame_address(...) 0

namespace cv
{

/////////////////////// PngDecoder ///////////////////

PngDecoder::PngDecoder()
{
    m_signature = "\x89\x50\x4e\x47\xd\xa\x1a\xa";
    m_color_type = 0;
    m_png_ptr = 0;
    m_info_ptr = m_end_info = 0;
    m_f = 0;
    m_buf_supported = true;
    m_buf_pos = 0;
    m_bit_depth = 0;
}


PngDecoder::~PngDecoder()
{
    close();
}

ImageDecoder PngDecoder::newDecoder() const
{
    return makePtr<PngDecoder>();
}

void  PngDecoder::close()
{
    if( m_f )
    {
        fclose( m_f );
        m_f = 0;
    }

    if( m_png_ptr )
    {
        png_structp png_ptr = (png_structp)m_png_ptr;
        png_infop info_ptr = (png_infop)m_info_ptr;
        png_infop end_info = (png_infop)m_end_info;
        png_destroy_read_struct( &png_ptr, &info_ptr, &end_info );
        m_png_ptr = m_info_ptr = m_end_info = 0;
    }
}


void  PngDecoder::readDataFromBuf( void* _png_ptr, uchar* dst, size_t size )
{
    png_structp png_ptr = (png_structp)_png_ptr;
    PngDecoder* decoder = (PngDecoder*)(png_get_io_ptr(png_ptr));
    CV_Assert( decoder );
    const Mat& buf = decoder->m_buf;
    if( decoder->m_buf_pos + size > buf.cols*buf.rows*buf.elemSize() )
    {
        png_error(png_ptr, "PNG input buffer is incomplete");
        return;
    }
    memcpy( dst, decoder->m_buf.ptr() + decoder->m_buf_pos, size );
    decoder->m_buf_pos += size;
}

bool  PngDecoder::readHeader()
{
    volatile bool result = false;
    close();

    png_structp png_ptr = png_create_read_struct( PNG_LIBPNG_VER_STRING, 0, 0, 0 );

    if( png_ptr )
    {
        png_infop info_ptr = png_create_info_struct( png_ptr );
        png_infop end_info = png_create_info_struct( png_ptr );

        m_png_ptr = png_ptr;
        m_info_ptr = info_ptr;
        m_end_info = end_info;
        m_buf_pos = 0;

        if( info_ptr && end_info )
        {
            if( setjmp( png_jmpbuf( png_ptr ) ) == 0 )
            {
                if( !m_buf.empty() )
                    png_set_read_fn(png_ptr, this, (png_rw_ptr)readDataFromBuf );
                else
                {
                    m_f = fopen( m_filename.c_str(), "rb" );
                    if( m_f )
                        png_init_io( png_ptr, m_f );
                }

                if( !m_buf.empty() || m_f )
                {
                    png_uint_32 wdth, hght;
                    int bit_depth, color_type, num_trans=0;
                    png_bytep trans;
                    png_color_16p trans_values;

                    png_read_info( png_ptr, info_ptr );

                    png_get_IHDR( png_ptr, info_ptr, &wdth, &hght,
                                  &bit_depth, &color_type, 0, 0, 0 );

                    m_width = (int)wdth;
                    m_height = (int)hght;
                    m_color_type = color_type;
                    m_bit_depth = bit_depth;

                    if( bit_depth <= 8 || bit_depth == 16 )
                    {
                        switch(color_type)
                        {
                            case PNG_COLOR_TYPE_RGB:
                            case PNG_COLOR_TYPE_PALETTE:
                                png_get_tRNS(png_ptr, info_ptr, &trans, &num_trans, &trans_values);
                                if( num_trans > 0 )
                                    m_type = CV_8UC4;
                                else
                                    m_type = CV_8UC3;
                                break;
                            case PNG_COLOR_TYPE_GRAY_ALPHA:
                            case PNG_COLOR_TYPE_RGB_ALPHA:
                                m_type = CV_8UC4;
                                break;
                            default:
                                m_type = CV_8UC1;
                        }
                        if( bit_depth == 16 )
                            m_type = CV_MAKETYPE(CV_16U, CV_MAT_CN(m_type));
                        result = true;
                    }
                }
            }
        }
    }

    if( !result )
        close();

    return result;
}


bool  PngDecoder::readData( Mat& img )
{
    volatile bool result = false;
    AutoBuffer<uchar*> _buffer(m_height);
    uchar** buffer = _buffer;
    int color = img.channels() > 1;

    png_structp png_ptr = (png_structp)m_png_ptr;
    png_infop info_ptr = (png_infop)m_info_ptr;
    png_infop end_info = (png_infop)m_end_info;

    if( m_png_ptr && m_info_ptr && m_end_info && m_width && m_height )
    {
        if( setjmp( png_jmpbuf ( png_ptr ) ) == 0 )
        {
            int y;

            if( img.depth() == CV_8U && m_bit_depth == 16 )
                png_set_strip_16( png_ptr );
            else if( !isBigEndian() )
                png_set_swap( png_ptr );

            if(img.channels() < 4)
            {
                /* observation: png_read_image() writes 400 bytes beyond
                 * end of data when reading a 400x118 color png
                 * "mpplus_sand.png".  OpenCV crashes even with demo
                 * programs.  Looking at the loaded image I'd say we get 4
                 * bytes per pixel instead of 3 bytes per pixel.  Test
                 * indicate that it is a good idea to always ask for
                 * stripping alpha..  18.11.2004 Axel Walthelm
                 */
                 png_set_strip_alpha( png_ptr );
            } else
                png_set_tRNS_to_alpha( png_ptr );

            if( m_color_type == PNG_COLOR_TYPE_PALETTE )
                png_set_palette_to_rgb( png_ptr );

            if( (m_color_type & PNG_COLOR_MASK_COLOR) == 0 && m_bit_depth < 8 )
#if (PNG_LIBPNG_VER_MAJOR*10000 + PNG_LIBPNG_VER_MINOR*100 + PNG_LIBPNG_VER_RELEASE >= 10209) || \
    (PNG_LIBPNG_VER_MAJOR == 1 && PNG_LIBPNG_VER_MINOR == 0 && PNG_LIBPNG_VER_RELEASE >= 18)
                png_set_expand_gray_1_2_4_to_8( png_ptr );
#else
                png_set_gray_1_2_4_to_8( png_ptr );
#endif

            if( (m_color_type & PNG_COLOR_MASK_COLOR) && color )
                png_set_bgr( png_ptr ); // convert RGB to BGR
            else if( color )
                png_set_gray_to_rgb( png_ptr ); // Gray->RGB
            else
                png_set_rgb_to_gray( png_ptr, 1, 0.299, 0.587 ); // RGB->Gray

            png_set_interlace_handling( png_ptr );
            png_read_update_info( png_ptr, info_ptr );

            for( y = 0; y < m_height; y++ )
                buffer[y] = img.data + y*img.step;

            png_read_image( png_ptr, buffer );
            png_read_end( png_ptr, end_info );

            result = true;
        }
    }

    close();
    return result;
}


/////////////////////// PngEncoder ///////////////////


PngEncoder::PngEncoder()
{
    m_description = "Portable Network Graphics files (*.png)";
    m_buf_supported = true;
}


PngEncoder::~PngEncoder()
{
}


bool  PngEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_16U;
}

ImageEncoder PngEncoder::newEncoder() const
{
    return makePtr<PngEncoder>();
}


void PngEncoder::writeDataToBuf(void* _png_ptr, uchar* src, size_t size)
{
    if( size == 0 )
        return;
    png_structp png_ptr = (png_structp)_png_ptr;
    PngEncoder* encoder = (PngEncoder*)(png_get_io_ptr(png_ptr));
    CV_Assert( encoder && encoder->m_buf );
    size_t cursz = encoder->m_buf->size();
    encoder->m_buf->resize(cursz + size);
    memcpy( &(*encoder->m_buf)[cursz], src, size );
}


void PngEncoder::flushBuf(void*)
{
}

bool  PngEncoder::write( const Mat& img, const std::vector<int>& params )
{
    png_structp png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, 0, 0, 0 );
    png_infop info_ptr = 0;
    FILE * volatile f = 0;
    int y, width = img.cols, height = img.rows;
    int depth = img.depth(), channels = img.channels();
    volatile bool result = false;
    AutoBuffer<uchar*> buffer;

    if( depth != CV_8U && depth != CV_16U )
        return false;

    if( png_ptr )
    {
        info_ptr = png_create_info_struct( png_ptr );

        if( info_ptr )
        {
            if( setjmp( png_jmpbuf ( png_ptr ) ) == 0 )
            {
                if( m_buf )
                {
                    png_set_write_fn(png_ptr, this,
                        (png_rw_ptr)writeDataToBuf, (png_flush_ptr)flushBuf);
                }
                else
                {
                    f = fopen( m_filename.c_str(), "wb" );
                    if( f )
                        png_init_io( png_ptr, (png_FILE_p)f );
                }

                int compression_level = -1; // Invalid value to allow setting 0-9 as valid
                int compression_strategy = IMWRITE_PNG_STRATEGY_RLE; // Default strategy
                bool isBilevel = false;

                for( size_t i = 0; i < params.size(); i += 2 )
                {
                    if( params[i] == IMWRITE_PNG_COMPRESSION )
                    {
                        compression_strategy = IMWRITE_PNG_STRATEGY_DEFAULT; // Default strategy
                        compression_level = params[i+1];
                        compression_level = MIN(MAX(compression_level, 0), Z_BEST_COMPRESSION);
                    }
                    if( params[i] == IMWRITE_PNG_STRATEGY )
                    {
                        compression_strategy = params[i+1];
                        compression_strategy = MIN(MAX(compression_strategy, 0), Z_FIXED);
                    }
                    if( params[i] == IMWRITE_PNG_BILEVEL )
                    {
                        isBilevel = params[i+1] != 0;
                    }
                }

                if( m_buf || f )
                {
                    if( compression_level >= 0 )
                    {
                        png_set_compression_level( png_ptr, compression_level );
                    }
                    else
                    {
                        // tune parameters for speed
                        // (see http://wiki.linuxquestions.org/wiki/Libpng)
                        png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
                        png_set_compression_level(png_ptr, Z_BEST_SPEED);
                    }
                    png_set_compression_strategy(png_ptr, compression_strategy);

                    png_set_IHDR( png_ptr, info_ptr, width, height, depth == CV_8U ? isBilevel?1:8 : 16,
                        channels == 1 ? PNG_COLOR_TYPE_GRAY :
                        channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA,
                        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                        PNG_FILTER_TYPE_DEFAULT );

                    png_write_info( png_ptr, info_ptr );

                    if (isBilevel)
                        png_set_packing(png_ptr);

                    png_set_bgr( png_ptr );
                    if( !isBigEndian() )
                        png_set_swap( png_ptr );

                    buffer.allocate(height);
                    for( y = 0; y < height; y++ )
                        buffer[y] = img.data + y*img.step;

                    png_write_image( png_ptr, buffer );
                    png_write_end( png_ptr, info_ptr );

                    result = true;
                }
            }
        }
    }

    png_destroy_write_struct( &png_ptr, &info_ptr );
    if(f) fclose( (FILE*)f );

    return result;
}

}

#endif

/* End of file. */
