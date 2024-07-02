// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#ifdef HAVE_PNG

/****************************************************************************************\
    This part of the file implements PNG codec on base of libpng library,
    in particular, this code is based on example.c from libpng
    (see otherlibs/_graphics/readme.txt for copyright notice)
    and png2bmp sample from libpng distribution (Copyright (C) 1999-2001 MIYASAKA Masaru)
\****************************************************************************************/

/****************************************************************************\
 *
 *  this file includes some modified part of apngasm and APNG Optimizer 1.4
 *  both have zlib license.
 *
 ****************************************************************************/


 /*  apngasm
 *
 *  The next generation of apngasm, the APNG Assembler.
 *  The apngasm CLI tool and library can assemble and disassemble APNG image files.
 *
 *  https://github.com/apngasm/apngasm


 /* APNG Optimizer 1.4
 *
 * Makes APNG files smaller.
 *
 * http://sourceforge.net/projects/apng/files
 *
 * Copyright (c) 2011-2015 Max Stepin
 * maxst at users.sourceforge.net
 *
 * zlib license
 * ------------
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#ifndef _LFS64_LARGEFILE
#  define _LFS64_LARGEFILE 0
#endif
#ifndef _FILE_OFFSET_BITS
#  define _FILE_OFFSET_BITS 0
#endif

#include <png.h>
#include <zlib.h>

#include "grfmt_png.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <array>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <stdexcept>
#include <iomanip>

#if defined _MSC_VER && _MSC_VER >= 1200
    // interaction between '_setjmp' and C++ object destruction is non-portable
    #pragma warning( disable: 4611 )
    #pragma warning( disable: 4244 )
#endif

// the following defines are a hack to avoid multiple problems with frame pointer handling and setjmp
// see http://gcc.gnu.org/ml/gcc/2011-10/msg00324.html for some details
#define mingw_getsp(...) 0
#define __builtin_frame_address(...) 0


#define id_IHDR 0x52444849
#define id_acTL 0x4C546361
#define id_fcTL 0x4C546366
#define id_IDAT 0x54414449
#define id_fdAT 0x54416466
#define id_IEND 0x444E4549

namespace cv
{
    void info_fn(png_structp png_ptr, png_infop info_ptr);
    void row_fn(png_structp png_ptr, png_bytep new_row, png_uint_32 row_num, int pass);

    void info_fn(png_structp png_ptr, png_infop info_ptr)
    {
        png_set_expand(png_ptr);
        png_set_strip_16(png_ptr);
        png_set_gray_to_rgb(png_ptr);
        png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);
        (void)png_set_interlace_handling(png_ptr);
        png_read_update_info(png_ptr, info_ptr);
    }

    void row_fn(png_structp png_ptr, png_bytep new_row, png_uint_32 row_num, int pass)
    {
        CV_UNUSED(pass);
        APNGFrame* frame = (APNGFrame*)png_get_progressive_ptr(png_ptr);
        png_progressive_combine_row(png_ptr, frame->rows()[row_num], new_row);
    }


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
    m_is_animated = false;
    m_loops = 0;
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

                    if (m_f)
                    {
                        uchar sig[8];
                        uint id;
                        CHUNK chunkIHDR, chunkacTL, chunkfcTL;

                        if (fread(sig, 1, 8, m_f))
                        {
                            id = read_chunk(m_f, &chunkIHDR);
                            if (id == id_IHDR && chunkIHDR.size == 25)
                            {
                                id = read_chunk(m_f, &chunkacTL);

                                if (id == id_acTL && chunkacTL.size == 20)
                                {
                                    m_is_animated = true;
                                    m_loops = png_get_uint_32(chunkacTL.p + 12);

                                    id = read_chunk(m_f, &chunkfcTL);
                                    if (id == id_fcTL && chunkfcTL.size == 38)
                                    {
                                        uint w0 = png_get_uint_32(chunkfcTL.p + 12);
                                        uint h0 = png_get_uint_32(chunkfcTL.p + 16);
                                        uint x0 = png_get_uint_32(chunkfcTL.p + 20);
                                        uint y0 = png_get_uint_32(chunkfcTL.p + 24);
                                        int delay_num = png_get_uint_16(chunkfcTL.p + 28);
                                        int delay_den = png_get_uint_16(chunkfcTL.p + 30);
                                        char dop = chunkfcTL.p[32];
                                        char bop = chunkfcTL.p[33];
                                    }
                                }
                            }
                        }
                        fseek(m_f, 0, SEEK_SET);
                    }

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

    return result;
}


bool  PngDecoder::readData( Mat& img )
{
    volatile bool result = false;
    AutoBuffer<uchar*> _buffer(m_height);
    uchar** buffer = _buffer.data();
    bool color = img.channels() > 1;

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

            if (m_is_animated)
            {

                return true;
            }
            png_read_end( png_ptr, end_info );

#ifdef PNG_eXIf_SUPPORTED
            png_uint_32 num_exif = 0;
            png_bytep exif = 0;

            // Exif info could be in info_ptr (intro_info) or end_info per specification
            if( png_get_valid(png_ptr, info_ptr, PNG_INFO_eXIf) )
                png_get_eXIf_1(png_ptr, info_ptr, &num_exif, &exif);
            else if( png_get_valid(png_ptr, end_info, PNG_INFO_eXIf) )
                png_get_eXIf_1(png_ptr, end_info, &num_exif, &exif);

            if( exif && num_exif > 0 )
            {
                m_exif.parseExif(exif, num_exif);
            }
#endif

            result = true;
        }
    }

    return result;
}

bool PngDecoder::nextPage() {
    if (m_f)
    {
        uint id;
        CHUNK chunkfcTL;

        id = read_chunk(m_f, &chunkfcTL);
        if (id == id_fcTL && chunkfcTL.size == 38)
        {
            // At this point the old frame is done. Let's start a new one.
            uint w0 = png_get_uint_32(chunkfcTL.p + 12);
            uint h0 = png_get_uint_32(chunkfcTL.p + 16);
            uint x0 = png_get_uint_32(chunkfcTL.p + 20);
            uint y0 = png_get_uint_32(chunkfcTL.p + 24);
            int delay_num = png_get_uint_16(chunkfcTL.p + 28);
            int delay_den = png_get_uint_16(chunkfcTL.p + 30);
            char dop = chunkfcTL.p[32];
            char bop = chunkfcTL.p[33];

            uchar sig[8];
            if (fread(sig, 1, 8, m_f))
                return true;
        }
    }
    return false;
}

void PngDecoder::compose_frame(uchar** rows_dst, uchar** rows_src, uchar bop, uint x, uint y, uint w, uint h)
{
    uint  i, j;
    int u, v, al;

    for (j = 0; j < h; j++)
    {
        uchar* sp = rows_src[j];
        uchar* dp = rows_dst[j + y] + x * 4;

        if (bop == 0)
            memcpy(dp, sp, w * 4);
        else
            for (i = 0; i < w; i++, sp += 4, dp += 4)
            {
                if (sp[3] == 255)
                    memcpy(dp, sp, 4);
                else
                    if (sp[3] != 0)
                    {
                        if (dp[3] != 0)
                        {
                            u = sp[3] * 255;
                            v = (255 - sp[3]) * dp[3];
                            al = u + v;
                            dp[0] = (sp[0] * u + dp[0] * v) / al;
                            dp[1] = (sp[1] * u + dp[1] * v) / al;
                            dp[2] = (sp[2] * u + dp[2] * v) / al;
                            dp[3] = al / 255;
                        }
                        else
                            memcpy(dp, sp, 4);
                    }
            }
    }
}

uint PngDecoder::read_chunk(FILE* f, CHUNK* pChunk)
{
    uchar len[4];
    pChunk->size = 0;
    pChunk->p = 0;
    if (fread(&len, 4, 1, f) == 1)
    {
        pChunk->size = png_get_uint_32(len);
        if (pChunk->size > PNG_USER_CHUNK_MALLOC_MAX)
            return 0;
        pChunk->size += 12;
        pChunk->p = new uchar[pChunk->size];
        memcpy(pChunk->p, len, 4);
        if (fread(pChunk->p + 4, pChunk->size - 4, 1, f) == 1)
            return *(uint*)(pChunk->p + 4);
    }
    return 0;
}

int PngDecoder::processing_start(png_structp& png_ptr, png_infop& info_ptr, void* frame_ptr, bool hasInfo, CHUNK& chunkIHDR, std::vector<CHUNK>& chunksInfo)
{
    uchar header[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);
    if (!png_ptr || !info_ptr)
        return 1;

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_read_struct(&png_ptr, &info_ptr, 0);
        return 1;
    }

    png_set_crc_action(png_ptr, PNG_CRC_QUIET_USE, PNG_CRC_QUIET_USE);
    png_set_progressive_read_fn(png_ptr, frame_ptr, info_fn, row_fn, NULL);
    png_set_bgr(png_ptr);

    png_process_data(png_ptr, info_ptr, header, 8);
    png_process_data(png_ptr, info_ptr, chunkIHDR.p, chunkIHDR.size);

    if (hasInfo)
        for (uint i = 0; i < chunksInfo.size(); i++)
            png_process_data(png_ptr, info_ptr, chunksInfo[i].p, chunksInfo[i].size);

    return 0;
}

int PngDecoder::processing_data(png_structp png_ptr, png_infop info_ptr, uchar* p, uint size)
{
    if (!png_ptr || !info_ptr)
        return 1;

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_read_struct(&png_ptr, &info_ptr, 0);
        return 1;
    }

    png_process_data(png_ptr, info_ptr, p, size);
    return 0;
}

int PngDecoder::processing_finish(png_structp png_ptr, png_infop info_ptr)
{
    uchar footer[12] = { 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130 };

    if (!png_ptr || !info_ptr)
        return 1;

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_read_struct(&png_ptr, &info_ptr, 0);
        return 1;
    }

    png_process_data(png_ptr, info_ptr, footer, 12);
    png_destroy_read_struct(&png_ptr, &info_ptr, 0);

    return 0;
}

int PngDecoder::load_apng(std::string inputFileName, std::vector<APNGFrame>& frames, uint& first, uint& loops)
{
    FILE* f;
    uint id, i, j, w, h, w0, h0, x0, y0;
    uint delay_num, delay_den, dop, bop, rowbytes, imagesize;
    uchar sig[8];
    png_structp png_ptr;
    png_infop info_ptr;
    CHUNK chunk;
    CHUNK chunkIHDR;
    std::vector<CHUNK> chunksInfo;
    bool isAnimated = false;
    bool hasInfo = false;
    int result = -1;
    first = 0;
    const unsigned long cMaxPNGSize = 1000000UL;

    if ((f = fopen(inputFileName.c_str(), "rb")) != 0)
    {
        if (fread(sig, 1, 8, f) == 8 && png_sig_cmp(sig, 0, 8) == 0)
        {
            id = read_chunk(f, &chunkIHDR);

            if (id == id_IHDR && chunkIHDR.size == 25)
            {
                w0 = w = png_get_uint_32(chunkIHDR.p + 8);
                h0 = h = png_get_uint_32(chunkIHDR.p + 12);

                if (w > cMaxPNGSize || h > cMaxPNGSize)
                {
                    fclose(f);
                    return result;
                }

                x0 = 0;
                y0 = 0;
                delay_num = 1;
                delay_den = 10;
                dop = 0;
                bop = 0;
                rowbytes = w * 4;
                imagesize = h * rowbytes;

                rgba* frameRaw_pixels = new rgba[w * h];
                rgba* frameCur_pixels = new rgba[w * h];
                rgba* frameNext_pixels = new rgba[w * h];

                APNGFrame frameRaw(frameRaw_pixels, w, h);
                APNGFrame frameCur(frameCur_pixels, w, h);
                APNGFrame frameNext(frameNext_pixels, w, h);

                if (!processing_start(png_ptr, info_ptr, (void*)&frameRaw, hasInfo, chunkIHDR, chunksInfo))
                {
                    while (!feof(f))
                    {
                        id = read_chunk(f, &chunk);
                        if (!id)
                            break;

                        if (id == id_acTL && !hasInfo && !isAnimated)
                        {
                            isAnimated = true;
                            first = 1;
                            loops = png_get_uint_32(chunk.p + 12);
                        }
                        else if (id == id_fcTL && (!hasInfo || isAnimated))
                        {
                            if (hasInfo)
                            {
                                if (!processing_finish(png_ptr, info_ptr))
                                {
                                    if (dop == 2)
                                        memcpy(frameNext.pixels(), frameCur.pixels(), imagesize);

                                    compose_frame(frameCur.rows(), frameRaw.rows(), bop, x0, y0, w0, h0);
                                    frameCur.delayNum(delay_num);
                                    frameCur.delayDen(delay_den);
                                    frames.push_back(frameCur);

                                    if (dop != 2)
                                    {
                                        memcpy(frameNext.pixels(), frameCur.pixels(), imagesize);
                                        if (dop == 1)
                                            for (j = 0; j < h0; j++)
                                                memset(frameNext.rows()[y0 + j] + x0 * 4, 0, w0 * 4);
                                    }
                                    frameCur.pixels(frameNext.pixels());
                                    frameCur.rows(frameNext.rows());
                                }
                                else
                                {
                                    delete[] frameCur.rows();
                                    delete[] frameCur.pixels();
                                    delete[] chunk.p;
                                    break;
                                }
                            }

                            // At this point the old frame is done. Let's start a new one.
                            w0 = png_get_uint_32(chunk.p + 12);
                            h0 = png_get_uint_32(chunk.p + 16);
                            x0 = png_get_uint_32(chunk.p + 20);
                            y0 = png_get_uint_32(chunk.p + 24);
                            delay_num = png_get_uint_16(chunk.p + 28);
                            delay_den = png_get_uint_16(chunk.p + 30);
                            dop = chunk.p[32];
                            bop = chunk.p[33];

                            if (w0 > cMaxPNGSize || h0 > cMaxPNGSize || x0 > cMaxPNGSize || y0 > cMaxPNGSize || x0 + w0 > w || y0 + h0 > h || dop > 2 || bop > 1)
                            {
                                delete[] frameCur.rows();
                                delete[] frameCur.pixels();
                                delete[] chunk.p;
                                break;
                            }

                            if (hasInfo)
                            {
                                memcpy(chunkIHDR.p + 8, chunk.p + 12, 8);
                                if (processing_start(png_ptr, info_ptr, (void*)&frameRaw, hasInfo, chunkIHDR, chunksInfo))
                                {
                                    delete[] frameCur.rows();
                                    delete[] frameCur.pixels();
                                    delete[] chunk.p;
                                    break;
                                }
                            }
                            else
                                first = 0;

                            if (frames.size() == first)
                            {
                                bop = 0;
                                if (dop == 2)
                                    dop = 1;
                            }
                        }
                        else if (id == id_IDAT)
                        {
                            hasInfo = true;
                            if (processing_data(png_ptr, info_ptr, chunk.p, chunk.size))
                            {
                                delete[] frameCur.rows();
                                delete[] frameCur.pixels();
                                delete[] chunk.p;
                                break;
                            }
                        }
                        else if (id == id_fdAT && isAnimated)
                        {
                            png_save_uint_32(chunk.p + 4, chunk.size - 16);
                            memcpy(chunk.p + 8, "IDAT", 4);
                            if (processing_data(png_ptr, info_ptr, chunk.p + 4, chunk.size - 4))
                            {
                                delete[] frameCur.rows();
                                delete[] frameCur.pixels();
                                delete[] chunk.p;
                                break;
                            }
                        }
                        else if (id == id_IEND)
                        {
                            if (hasInfo && !processing_finish(png_ptr, info_ptr))
                            {
                                compose_frame(frameCur.rows(), frameRaw.rows(), bop, x0, y0, w0, h0);
                                frameCur.delayNum(delay_num);
                                frameCur.delayDen(delay_den);
                                frames.push_back(frameCur);
                            }
                            else
                            {
                                delete[] frameCur.rows();
                                delete[] frameCur.pixels();
                            }
                            delete[] chunk.p;
                            break;
                        }
                        else if (!isalpha(chunk.p[4]) || !isalpha(chunk.p[5]) || !isalpha(chunk.p[6]) || !isalpha(chunk.p[7]))
                        {
                            delete[] chunk.p;
                            break;
                        }
                        else if (!hasInfo)
                        {
                            if (processing_data(png_ptr, info_ptr, chunk.p, chunk.size))
                            {
                                delete[] frameCur.rows();
                                delete[] frameCur.pixels();
                                delete[] chunk.p;
                                break;
                            }
                            chunksInfo.push_back(chunk);
                            continue;
                        }
                        delete[] chunk.p;
                    }
                }
                delete[] frameRaw.rows();
                delete[] frameRaw.pixels();

                if (!frames.empty())
                {
                    printf("***********************************\nfile name : %s .. frames in the file : %zd\n", inputFileName.c_str(), frames.size());
                    result = 0;
                }
            }

            for (i = 0; i < chunksInfo.size(); i++)
                delete[] chunksInfo[i].p;

            chunksInfo.clear();
            delete[] chunkIHDR.p;
        }
        fclose(f);
    }

    return result;
}

/////////////////////// PngEncoder ///////////////////


PngEncoder::PngEncoder()
{
    m_description = "Portable Network Graphics files (*.png)";
    m_buf_supported = true;
    op_zbuf1 = NULL;
    op_zbuf2 = NULL;
    op_zstream1.zalloc = NULL;
    op_zstream2.zalloc = NULL;
    row_buf = NULL;
    sub_row = NULL;
    up_row = NULL;
    avg_row = NULL;
    paeth_row = NULL;
    next_seq_num = 0;
    trnssize = 0;
    palsize = 0;
    memset(palette, 0, sizeof(palette));
    memset(trns, 0, sizeof(trns));
    memset(op, 0, sizeof(op));
    process_callback = { 0 };
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

                    png_write_image( png_ptr, buffer.data() );
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

void PngEncoder::optim_dirty(std::vector<APNGFrame>& frames)
{
    uint i, j;
    uchar* sp;
    uint size = frames[0].width() * frames[0].height();

    for (i = 0; i < frames.size(); i++)
    {
        sp = frames[i].pixels();
        for (j = 0; j < size; j++, sp += 4)
            if (sp[3] == 0)
                sp[0] = sp[1] = sp[2] = 0;
        process_callback(0.1 + i / float(frames.size()) * 0.1);
    }
}

void PngEncoder::optim_duplicates(std::vector<APNGFrame>& frames, uint first)
{
    uint imagesize = frames[0].width() * frames[0].height() * 4;
    uint i = first;

    while (++i < frames.size())
    {
        if (memcmp(frames[i - 1].pixels(), frames[i].pixels(), imagesize) != 0)
            continue;

        i--;
        delete[] frames[i].pixels();
        delete[] frames[i].rows();
        uint num = frames[i].delayNum();
        uint den = frames[i].delayDen();
        frames.erase(frames.begin() + i);

        if (frames[i].delayDen() == den)
            frames[i].delayNum(frames[i].delayNum()+num);
        else
        {
            frames[i].delayNum(num * frames[i].delayDen() + den * frames[i].delayNum());
            frames[i].delayDen(den * frames[i].delayDen());
            while (num && den)
            {
                if (num > den)
                    num = num % den;
                else
                    den = den % num;
            }
            num += den;
            frames[i].delayNum(frames[i].delayNum() / num);
            frames[i].delayDen(frames[i].delayDen() / num);
        }
    }
}

void PngEncoder::write_chunk(FILE* f, const char* name, uchar* data, uint length)
{
    uchar buf[4];
    uint crc = crc32(0, Z_NULL, 0);

    png_save_uint_32(buf, length);
    fwrite(buf, 1, 4, f);
    fwrite(name, 1, 4, f);
    crc = crc32(crc, (const Bytef*)name, 4);

    if (memcmp(name, "fdAT", 4) == 0)
    {
        png_save_uint_32(buf, next_seq_num++);
        fwrite(buf, 1, 4, f);
        crc = crc32(crc, buf, 4);
        length -= 4;
    }

    if (data != NULL && length > 0)
    {
        fwrite(data, 1, length, f);
        crc = crc32(crc, data, length);
    }

    png_save_uint_32(buf, crc);
    fwrite(buf, 1, 4, f);
}

void PngEncoder::write_IDATs(FILE* f, int frame, uchar* data, uint length, uint idat_size)
{
    uint z_cmf = data[0];
    if ((z_cmf & 0x0f) == 8 && (z_cmf & 0xf0) <= 0x70)
    {
        if (length >= 2)
        {
            uint z_cinfo = z_cmf >> 4;
            uint half_z_window_size = 1 << (z_cinfo + 7);
            while (idat_size <= half_z_window_size && half_z_window_size >= 256)
            {
                z_cinfo--;
                half_z_window_size >>= 1;
            }
            z_cmf = (z_cmf & 0x0f) | (z_cinfo << 4);
            if (data[0] != (uchar)z_cmf)
            {
                data[0] = (uchar)z_cmf;
                data[1] &= 0xe0;
                data[1] += (uchar)(0x1f - ((z_cmf << 8) + data[1]) % 0x1f);
            }
        }
    }

    while (length > 0)
    {
        uint ds = length;
        if (ds > 32768)
            ds = 32768;

        if (frame == 0)
            write_chunk(f, "IDAT", data, ds);
        else
            write_chunk(f, "fdAT", data, ds + 4);

        data += ds;
        length -= ds;
    }
}

void PngEncoder::process_rect(uchar* row, int rowbytes, int bpp, int stride, int h, uchar* rows)
{
    int i, j, v;
    int a, b, c, pa, pb, pc, p;
    uchar* prev = NULL;
    uchar* dp = rows;
    uchar* out;

    for (j = 0; j < h; j++)
    {
        uint sum = 0;
        uchar* best_row = row_buf;
        uint mins = ((uint)(-1)) >> 1;

        out = row_buf + 1;
        for (i = 0; i < rowbytes; i++)
        {
            v = out[i] = row[i];
            sum += (v < 128) ? v : 256 - v;
        }
        mins = sum;

        sum = 0;
        out = sub_row + 1;
        for (i = 0; i < bpp; i++)
        {
            v = out[i] = row[i];
            sum += (v < 128) ? v : 256 - v;
        }
        for (i = bpp; i < rowbytes; i++)
        {
            v = out[i] = row[i] - row[i - bpp];
            sum += (v < 128) ? v : 256 - v;
            if (sum > mins)
                break;
        }
        if (sum < mins)
        {
            mins = sum;
            best_row = sub_row;
        }

        if (prev)
        {
            sum = 0;
            out = up_row + 1;
            for (i = 0; i < rowbytes; i++)
            {
                v = out[i] = row[i] - prev[i];
                sum += (v < 128) ? v : 256 - v;
                if (sum > mins)
                    break;
            }
            if (sum < mins)
            {
                mins = sum;
                best_row = up_row;
            }

            sum = 0;
            out = avg_row + 1;
            for (i = 0; i < bpp; i++)
            {
                v = out[i] = row[i] - prev[i] / 2;
                sum += (v < 128) ? v : 256 - v;
            }
            for (i = bpp; i < rowbytes; i++)
            {
                v = out[i] = row[i] - (prev[i] + row[i - bpp]) / 2;
                sum += (v < 128) ? v : 256 - v;
                if (sum > mins)
                    break;
            }
            if (sum < mins)
            {
                mins = sum;
                best_row = avg_row;
            }

            sum = 0;
            out = paeth_row + 1;
            for (i = 0; i < bpp; i++)
            {
                v = out[i] = row[i] - prev[i];
                sum += (v < 128) ? v : 256 - v;
            }
            for (i = bpp; i < rowbytes; i++)
            {
                a = row[i - bpp];
                b = prev[i];
                c = prev[i - bpp];
                p = b - c;
                pc = a - c;
                pa = abs(p);
                pb = abs(pc);
                pc = abs(p + pc);
                p = (pa <= pb && pa <= pc) ? a : (pb <= pc) ? b
                    : c;
                v = out[i] = row[i] - p;
                sum += (v < 128) ? v : 256 - v;
                if (sum > mins)
                    break;
            }
            if (sum < mins)
            {
                best_row = paeth_row;
            }
        }

        if (rows == NULL)
        {
            // deflate_rect_op()
            op_zstream1.next_in = row_buf;
            op_zstream1.avail_in = rowbytes + 1;
            deflate(&op_zstream1, Z_NO_FLUSH);

            op_zstream2.next_in = best_row;
            op_zstream2.avail_in = rowbytes + 1;
            deflate(&op_zstream2, Z_NO_FLUSH);
        }
        else
        {
            // deflate_rect_fin()
            memcpy(dp, best_row, rowbytes + 1);
            dp += rowbytes + 1;
        }

        prev = row;
        row += stride;
    }
}

void PngEncoder::deflate_rect_op(uchar* pdata, int x, int y, int w, int h, int bpp, int stride, int zbuf_size, int n)
{
    uchar* row = pdata + y * stride + x * bpp;
    int rowbytes = w * bpp;

    op_zstream1.data_type = Z_BINARY;
    op_zstream1.next_out = op_zbuf1;
    op_zstream1.avail_out = zbuf_size;

    op_zstream2.data_type = Z_BINARY;
    op_zstream2.next_out = op_zbuf2;
    op_zstream2.avail_out = zbuf_size;

    process_rect(row, rowbytes, bpp, stride, h, NULL);

    deflate(&op_zstream1, Z_FINISH);
    deflate(&op_zstream2, Z_FINISH);
    op[n].p = pdata;

    if (op_zstream1.total_out < op_zstream2.total_out)
    {
        op[n].size = op_zstream1.total_out;
        op[n].filters = 0;
    }
    else
    {
        op[n].size = op_zstream2.total_out;
        op[n].filters = 1;
    }
    op[n].x = x;
    op[n].y = y;
    op[n].w = w;
    op[n].h = h;
    op[n].valid = 1;
    deflateReset(&op_zstream1);
    deflateReset(&op_zstream2);
}

void PngEncoder::get_rect(uint w, uint h, uchar* pimage1, uchar* pimage2, uchar* ptemp, uint bpp, uint stride, int zbuf_size, uint has_tcolor, uint tcolor, int n)
{
    uint i, j, x0, y0, w0, h0;
    uint x_min = w - 1;
    uint y_min = h - 1;
    uint x_max = 0;
    uint y_max = 0;
    uint diffnum = 0;
    uint over_is_possible = 1;

    if (!has_tcolor)
        over_is_possible = 0;

    if (bpp == 1)
    {
        uchar* pa = pimage1;
        uchar* pb = pimage2;
        uchar* pc = ptemp;

        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
            {
                uchar c = *pb++;
                if (*pa++ != c)
                {
                    diffnum++;
                    if (has_tcolor && c == tcolor)
                        over_is_possible = 0;
                    if (i < x_min)
                        x_min = i;
                    if (i > x_max)
                        x_max = i;
                    if (j < y_min)
                        y_min = j;
                    if (j > y_max)
                        y_max = j;
                }
                else
                    c = tcolor;

                *pc++ = c;
            }
    }
    else if (bpp == 2)
    {
        unsigned short* pa = (unsigned short*)pimage1;
        unsigned short* pb = (unsigned short*)pimage2;
        unsigned short* pc = (unsigned short*)ptemp;

        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
            {
                uint c1 = *pa++;
                uint c2 = *pb++;
                if ((c1 != c2) && ((c1 >> 8) || (c2 >> 8)))
                {
                    diffnum++;
                    if ((c2 >> 8) != 0xFF)
                        over_is_possible = 0;
                    if (i < x_min)
                        x_min = i;
                    if (i > x_max)
                        x_max = i;
                    if (j < y_min)
                        y_min = j;
                    if (j > y_max)
                        y_max = j;
                }
                else
                    c2 = 0;

                *pc++ = c2;
            }
    }
    else if (bpp == 3)
    {
        uchar* pa = pimage1;
        uchar* pb = pimage2;
        uchar* pc = ptemp;

        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
            {
                uint c1 = (pa[2] << 16) + (pa[1] << 8) + pa[0];
                uint c2 = (pb[2] << 16) + (pb[1] << 8) + pb[0];
                if (c1 != c2)
                {
                    diffnum++;
                    if (has_tcolor && c2 == tcolor)
                        over_is_possible = 0;
                    if (i < x_min)
                        x_min = i;
                    if (i > x_max)
                        x_max = i;
                    if (j < y_min)
                        y_min = j;
                    if (j > y_max)
                        y_max = j;
                }
                else
                    c2 = tcolor;

                memcpy(pc, &c2, 3);
                pa += 3;
                pb += 3;
                pc += 3;
            }
    }
    else if (bpp == 4)
    {
        uint* pa = (uint*)pimage1;
        uint* pb = (uint*)pimage2;
        uint* pc = (uint*)ptemp;

        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
            {
                uint c1 = *pa++;
                uint c2 = *pb++;
                if ((c1 != c2) && ((c1 >> 24) || (c2 >> 24)))
                {
                    diffnum++;
                    if ((c2 >> 24) != 0xFF)
                        over_is_possible = 0;
                    if (i < x_min)
                        x_min = i;
                    if (i > x_max)
                        x_max = i;
                    if (j < y_min)
                        y_min = j;
                    if (j > y_max)
                        y_max = j;
                }
                else
                    c2 = 0;

                *pc++ = c2;
            }
    }

    if (diffnum == 0)
    {
        x0 = y0 = 0;
        w0 = h0 = 1;
    }
    else
    {
        x0 = x_min;
        y0 = y_min;
        w0 = x_max - x_min + 1;
        h0 = y_max - y_min + 1;
    }

    deflate_rect_op(pimage2, x0, y0, w0, h0, bpp, stride, zbuf_size, n * 2);

    if (over_is_possible)
        deflate_rect_op(ptemp, x0, y0, w0, h0, bpp, stride, zbuf_size, n * 2 + 1);
}

void PngEncoder::deflate_rect_fin(int deflate_method, int iter, uchar* zbuf, uint* zsize, int bpp, int stride, uchar* rows, int zbuf_size, int n)
{
    uchar* row = op[n].p + op[n].y * stride + op[n].x * bpp;
    int rowbytes = op[n].w * bpp;

    if (op[n].filters == 0)
    {
        uchar* dp = rows;
        for (int j = 0; j < op[n].h; j++)
        {
            *dp++ = 0;
            memcpy(dp, row, rowbytes);
            dp += rowbytes;
            row += stride;
        }
    }
    else
        process_rect(row, rowbytes, bpp, stride, op[n].h, rows);

    if (deflate_method == 2)
    {
        CV_UNUSED(iter);
#if 0  // needs include "zopfli.h" 
        ZopfliOptions opt_zopfli;
        uchar* data = 0;
        size_t size = 0;
        ZopfliInitOptions(&opt_zopfli);
        opt_zopfli.numiterations = iter;
        ZopfliCompress(&opt_zopfli, ZOPFLI_FORMAT_ZLIB, rows, op[n].h * (rowbytes + 1), &data, &size);
        if (size < (size_t)zbuf_size)
        {
            memcpy(zbuf, data, size);
            *zsize = size;
        }
        free(data);
#endif
    }
    else if (deflate_method == 1)
    {
#if 0  // needs include "7z.h"
        unsigned size = zbuf_size;
        compress_rfc1950_7z(rows, op[n].h * (rowbytes + 1), zbuf, size, iter < 100 ? iter : 100, 255);
        *zsize = size;
#endif
    }
    else
    {
        z_stream fin_zstream;

        fin_zstream.data_type = Z_BINARY;
        fin_zstream.zalloc = Z_NULL;
        fin_zstream.zfree = Z_NULL;
        fin_zstream.opaque = Z_NULL;
        deflateInit2(&fin_zstream, Z_BEST_COMPRESSION, 8, 15, 8, op[n].filters ? Z_FILTERED : Z_DEFAULT_STRATEGY);

        fin_zstream.next_out = zbuf;
        fin_zstream.avail_out = zbuf_size;
        fin_zstream.next_in = rows;
        fin_zstream.avail_in = op[n].h * (rowbytes + 1);
        deflate(&fin_zstream, Z_FINISH);
        *zsize = fin_zstream.total_out;
        deflateEnd(&fin_zstream);
    }
}

bool PngEncoder::writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params)
{

    int compression_level = 6;
    int compression_strategy = IMWRITE_PNG_STRATEGY_RLE; // Default strategy
    bool isBilevel = false;

    for (size_t i = 0; i < params.size(); i += 2)
    {
        if (params[i] == IMWRITE_PNG_COMPRESSION)
        {
            compression_strategy = IMWRITE_PNG_STRATEGY_DEFAULT; // Default strategy
            compression_level = params[i + 1];
            compression_level = MIN(MAX(compression_level, 0), Z_BEST_COMPRESSION);
        }
        if (params[i] == IMWRITE_PNG_STRATEGY)
        {
            compression_strategy = params[i + 1];
            compression_strategy = MIN(MAX(compression_strategy, 0), Z_FIXED);
        }
        if (params[i] == IMWRITE_PNG_BILEVEL)
        {
            isBilevel = params[i + 1] != 0;
        }
    }

    std::vector<APNGFrame> frames;

    for (size_t i = 0; i < img_vec.size(); i++)
    {
        Mat frame = img_vec[i];
        if (frame.type() == CV_8UC3)
        {
            cvtColor(frame, frame, COLOR_BGR2RGBA);
        }
        else
        {
            cvtColor(frame, frame, COLOR_BGRA2RGBA);
        }

        rgba* pixels = (rgba*)frame.data;
        frames.push_back(APNGFrame(pixels, frame.cols, frame.rows));
    }

    CV_UNUSED(isBilevel);
    uint first =0;
    uint loops=10;
    uint coltype=6;
    int deflate_method=0;
    int iter=0;

    FILE* f;
    uint i, j, k;
    uint x0, y0, w0, h0, dop, bop;
    uint idat_size, zbuf_size, zsize;
    uchar* zbuf;
    uchar header[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    uint num_frames = (int)frames.size();
    uint width = frames[0].width();
    uint height = frames[0].height();
    uint bpp = (coltype == 6) ? 4 : (coltype == 2) ? 3
        : (coltype == 4) ? 2
        : 1;
    uint has_tcolor = (coltype >= 4 || (coltype <= 2 && trnssize)) ? 1 : 0;
    uint tcolor = 0;
    uint rowbytes = width * bpp;
    uint imagesize = rowbytes * height;

    uchar* temp = new uchar[imagesize];
    uchar* over1 = new uchar[imagesize];
    uchar* over2 = new uchar[imagesize];
    uchar* over3 = new uchar[imagesize];
    uchar* rest = new uchar[imagesize];
    uchar* rows = new uchar[(rowbytes + 1) * height];

    if (trnssize)
    {
        if (coltype == 0)
            tcolor = trns[1];
        else if (coltype == 2)
            tcolor = (((trns[5] << 8) + trns[3]) << 8) + trns[1];
        else if (coltype == 3)
        {
            for (i = 0; i < trnssize; i++)
                if (trns[i] == 0)
                {
                    has_tcolor = 1;
                    tcolor = i;
                    break;
                }
        }
    }

    if ((f = fopen(m_filename.c_str(), "wb")) != 0)
    {
        uchar buf_IHDR[13];
        uchar buf_acTL[8];
        uchar buf_fcTL[26];

        png_save_uint_32(buf_IHDR, width);
        png_save_uint_32(buf_IHDR + 4, height);
        buf_IHDR[8] = 8;
        buf_IHDR[9] = coltype;
        buf_IHDR[10] = 0;
        buf_IHDR[11] = 0;
        buf_IHDR[12] = 0;

        png_save_uint_32(buf_acTL, num_frames - first);
        png_save_uint_32(buf_acTL + 4, loops);

        fwrite(header, 1, 8, f);

        write_chunk(f, "IHDR", buf_IHDR, 13);

        if (num_frames > 1)
            write_chunk(f, "acTL", buf_acTL, 8);
        else
            first = 0;

        if (palsize > 0)
            write_chunk(f, "PLTE", (uchar*)(&palette), palsize * 3);

        if (trnssize > 0)
            write_chunk(f, "tRNS", trns, trnssize);

        op_zstream1.data_type = Z_BINARY;
        op_zstream1.zalloc = Z_NULL;
        op_zstream1.zfree = Z_NULL;
        op_zstream1.opaque = Z_NULL;
        deflateInit2(&op_zstream1, compression_level, 8, 15, 8, compression_strategy);

        op_zstream2.data_type = Z_BINARY;
        op_zstream2.zalloc = Z_NULL;
        op_zstream2.zfree = Z_NULL;
        op_zstream2.opaque = Z_NULL;
        deflateInit2(&op_zstream2, compression_level, 8, 15, 8, Z_FILTERED);

        idat_size = (rowbytes + 1) * height;
        zbuf_size = idat_size + ((idat_size + 7) >> 3) + ((idat_size + 63) >> 6) + 11;

        zbuf = new uchar[zbuf_size];
        op_zbuf1 = new uchar[zbuf_size];
        op_zbuf2 = new uchar[zbuf_size];
        row_buf = new uchar[rowbytes + 1];
        sub_row = new uchar[rowbytes + 1];
        up_row = new uchar[rowbytes + 1];
        avg_row = new uchar[rowbytes + 1];
        paeth_row = new uchar[rowbytes + 1];

        row_buf[0] = 0;
        sub_row[0] = 1;
        up_row[0] = 2;
        avg_row[0] = 3;
        paeth_row[0] = 4;

        x0 = 0;
        y0 = 0;
        w0 = width;
        h0 = height;
        bop = 0;
        next_seq_num = 0;

        for (j = 0; j < 6; j++)
            op[j].valid = 0;
        deflate_rect_op(frames[0].pixels(), x0, y0, w0, h0, bpp, rowbytes, zbuf_size, 0);
        deflate_rect_fin(deflate_method, iter, zbuf, &zsize, bpp, rowbytes, rows, zbuf_size, 0);

        if (first)
        {
            write_IDATs(f, 0, zbuf, zsize, idat_size);
            for (j = 0; j < 6; j++)
                op[j].valid = 0;
            deflate_rect_op(frames[1].pixels(), x0, y0, w0, h0, bpp, rowbytes, zbuf_size, 0);
            deflate_rect_fin(deflate_method, iter, zbuf, &zsize, bpp, rowbytes, rows, zbuf_size, 0);
        }

        for (i = first; i < num_frames - 1; i++)
        {
            uint op_min;
            int op_best;

            for (j = 0; j < 6; j++)
                op[j].valid = 0;

            /* dispose = none */
            get_rect(width, height, frames[i].pixels(), frames[i + 1].pixels(), over1, bpp, rowbytes, zbuf_size, has_tcolor, tcolor, 0);

            /* dispose = background */
            if (has_tcolor)
            {
                memcpy(temp, frames[i].pixels(), imagesize);
                if (coltype == 2)
                    for (j = 0; j < h0; j++)
                        for (k = 0; k < w0; k++)
                            memcpy(temp + ((j + y0) * width + (k + x0)) * 3, &tcolor, 3);
                else
                    for (j = 0; j < h0; j++)
                        memset(temp + ((j + y0) * width + x0) * bpp, tcolor, w0 * bpp);

                get_rect(width, height, temp, frames[i + 1].pixels(), over2, bpp, rowbytes, zbuf_size, has_tcolor, tcolor, 1);
            }

            /* dispose = previous */
            if (i > first)
                get_rect(width, height, rest, frames[i + 1].pixels(), over3, bpp, rowbytes, zbuf_size, has_tcolor, tcolor, 2);

            op_min = op[0].size;
            op_best = 0;
            for (j = 1; j < 6; j++)
                if (op[j].valid)
                {
                    if (op[j].size < op_min)
                    {
                        op_min = op[j].size;
                        op_best = j;
                    }
                }

            dop = op_best >> 1;

            png_save_uint_32(buf_fcTL, next_seq_num++);
            png_save_uint_32(buf_fcTL + 4, w0);
            png_save_uint_32(buf_fcTL + 8, h0);
            png_save_uint_32(buf_fcTL + 12, x0);
            png_save_uint_32(buf_fcTL + 16, y0);
            png_save_uint_16(buf_fcTL + 20, frames[i].delayNum());
            png_save_uint_16(buf_fcTL + 22, frames[i].delayDen());
            buf_fcTL[24] = dop;
            buf_fcTL[25] = bop;
            write_chunk(f, "fcTL", buf_fcTL, 26);

            write_IDATs(f, i, zbuf, zsize, idat_size);

            /* process apng dispose - begin */
            if (dop != 2)
                memcpy(rest, frames[i].pixels(), imagesize);

            if (dop == 1)
            {
                if (coltype == 2)
                    for (j = 0; j < h0; j++)
                        for (k = 0; k < w0; k++)
                            memcpy(rest + ((j + y0) * width + (k + x0)) * 3, &tcolor, 3);
                else
                    for (j = 0; j < h0; j++)
                        memset(rest + ((j + y0) * width + x0) * bpp, tcolor, w0 * bpp);
            }
            /* process apng dispose - end */

            x0 = op[op_best].x;
            y0 = op[op_best].y;
            w0 = op[op_best].w;
            h0 = op[op_best].h;
            bop = op_best & 1;

            deflate_rect_fin(deflate_method, iter, zbuf, &zsize, bpp, rowbytes, rows, zbuf_size, op_best);
        }

        if (num_frames > 1)
        {
            png_save_uint_32(buf_fcTL, next_seq_num++);
            png_save_uint_32(buf_fcTL + 4, w0);
            png_save_uint_32(buf_fcTL + 8, h0);
            png_save_uint_32(buf_fcTL + 12, x0);
            png_save_uint_32(buf_fcTL + 16, y0);
            png_save_uint_16(buf_fcTL + 20, frames[num_frames - 1].delayNum());
            png_save_uint_16(buf_fcTL + 22, frames[num_frames - 1].delayDen());
            buf_fcTL[24] = 0;
            buf_fcTL[25] = bop;
            write_chunk(f, "fcTL", buf_fcTL, 26);
        }

        write_IDATs(f, num_frames - 1, zbuf, zsize, idat_size);

        write_chunk(f, "IEND", 0, 0);

        fclose(f);

        delete[] zbuf;
        delete[] op_zbuf1;
        delete[] op_zbuf2;
        delete[] row_buf;
        delete[] sub_row;
        delete[] up_row;
        delete[] avg_row;
        delete[] paeth_row;

        deflateEnd(&op_zstream1);
        deflateEnd(&op_zstream2);
    }

    delete[] temp;
    delete[] over1;
    delete[] over2;
    delete[] over3;
    delete[] rest;
    delete[] rows;

    return true;
}

}

#endif

/* End of file. */
