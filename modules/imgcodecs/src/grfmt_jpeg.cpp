/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "grfmt_jpeg.hpp"
#include "jpeg_exif.hpp"

#ifdef HAVE_JPEG

#ifdef _MSC_VER
//interaction between '_setjmp' and C++ object destruction is non-portable
#pragma warning(disable: 4611)
#endif

#include <stdio.h>
#include <setjmp.h>

// the following defines are a hack to avoid multiple problems with frame ponter handling and setjmp
// see http://gcc.gnu.org/ml/gcc/2011-10/msg00324.html for some details
#define mingw_getsp(...) 0
#define __builtin_frame_address(...) 0

#ifdef WIN32

#define XMD_H // prevent redefinition of INT32
#undef FAR  // prevent FAR redefinition

#endif

#if defined WIN32 && defined __GNUC__
typedef unsigned char boolean;
#endif

#undef FALSE
#undef TRUE

extern "C" {
#include "jpeglib.h"
}

namespace cv
{

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4324) //structure was padded due to __declspec(align())
#endif
struct JpegErrorMgr
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};
#ifdef _MSC_VER
# pragma warning(pop)
#endif

struct JpegSource
{
    struct jpeg_source_mgr pub;
    int skip;
};

struct JpegState
{
    jpeg_decompress_struct cinfo; // IJG JPEG codec structure
    JpegErrorMgr jerr; // error processing manager state
    JpegSource source; // memory buffer source
};

/////////////////////// Error processing /////////////////////

METHODDEF(void)
stub(j_decompress_ptr)
{
}

METHODDEF(boolean)
fill_input_buffer(j_decompress_ptr)
{
    return FALSE;
}

// emulating memory input stream

METHODDEF(void)
skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
    JpegSource* source = (JpegSource*) cinfo->src;

    if( num_bytes > (long)source->pub.bytes_in_buffer )
    {
        // We need to skip more data than we have in the buffer.
        // This will force the JPEG library to suspend decoding.
        source->skip = (int)(num_bytes - source->pub.bytes_in_buffer);
        source->pub.next_input_byte += source->pub.bytes_in_buffer;
        source->pub.bytes_in_buffer = 0;
    }
    else
    {
        // Skip portion of the buffer
        source->pub.bytes_in_buffer -= num_bytes;
        source->pub.next_input_byte += num_bytes;
        source->skip = 0;
    }
}


static void jpeg_buffer_src(j_decompress_ptr cinfo, JpegSource* source)
{
    cinfo->src = &source->pub;

    // Prepare for suspending reader
    source->pub.init_source = stub;
    source->pub.fill_input_buffer = fill_input_buffer;
    source->pub.skip_input_data = skip_input_data;
    source->pub.resync_to_restart = jpeg_resync_to_restart;
    source->pub.term_source = stub;
    source->pub.bytes_in_buffer = 0; // forces fill_input_buffer on first read

    source->skip = 0;
}


METHODDEF(void)
error_exit( j_common_ptr cinfo )
{
    JpegErrorMgr* err_mgr = (JpegErrorMgr*)(cinfo->err);

    /* Return control to the setjmp point */
    longjmp( err_mgr->setjmp_buffer, 1 );
}


/////////////////////// JpegDecoder ///////////////////


JpegDecoder::JpegDecoder()
{
    m_signature = "\xFF\xD8\xFF";
    m_state = 0;
    m_f = 0;
    m_buf_supported = true;
    m_orientation = JPEG_ORIENTATION_TL;
}


JpegDecoder::~JpegDecoder()
{
    close();
}


void  JpegDecoder::close()
{
    if( m_state )
    {
        JpegState* state = (JpegState*)m_state;
        jpeg_destroy_decompress( &state->cinfo );
        delete state;
        m_state = 0;
    }

    if( m_f )
    {
        fclose( m_f );
        m_f = 0;
    }

    m_width = m_height = 0;
    m_type = -1;
}

ImageDecoder JpegDecoder::newDecoder() const
{
    return makePtr<JpegDecoder>();
}

bool  JpegDecoder::readHeader()
{
    volatile bool result = false;
    close();

    JpegState* state = new JpegState;
    m_state = state;
    state->cinfo.err = jpeg_std_error(&state->jerr.pub);
    state->jerr.pub.error_exit = error_exit;

    if( setjmp( state->jerr.setjmp_buffer ) == 0 )
    {
        jpeg_create_decompress( &state->cinfo );

        if( !m_buf.empty() )
        {
            jpeg_buffer_src(&state->cinfo, &state->source);
            state->source.pub.next_input_byte = m_buf.ptr();
            state->source.pub.bytes_in_buffer = m_buf.cols*m_buf.rows*m_buf.elemSize();
        }
        else
        {
            m_f = fopen( m_filename.c_str(), "rb" );
            if( m_f )
                jpeg_stdio_src( &state->cinfo, m_f );
        }

        if (state->cinfo.src != 0)
        {
            jpeg_read_header( &state->cinfo, TRUE );

            state->cinfo.scale_num=1;
            state->cinfo.scale_denom = m_scale_denom;
            m_scale_denom=1; // trick! to know which decoder used scale_denom see imread_
            jpeg_calc_output_dimensions(&state->cinfo);
            m_width = state->cinfo.output_width;
            m_height = state->cinfo.output_height;
            m_type = state->cinfo.num_components > 1 ? CV_8UC3 : CV_8UC1;
            result = true;
        }
    }

    m_orientation = getOrientation();

    if( !result )
        close();

    return result;
}

int JpegDecoder::getOrientation()
{
    int orientation = JPEG_ORIENTATION_TL;

    if (m_filename.size() > 0)
    {
        ExifReader reader( m_filename );
        if( reader.parse() )
        {
            ExifEntry_t entry = reader.getTag( ORIENTATION );
            if (entry.tag != INVALID_TAG)
            {
                orientation = entry.field_u16; //orientation is unsigned short, so check field_u16
            }
        }
    }

    return orientation;
}

void JpegDecoder::setOrientation(Mat& img)
{
    switch( m_orientation )
    {
        case    JPEG_ORIENTATION_TL: //0th row == visual top, 0th column == visual left-hand side
            //do nothing, the image already has proper orientation
            break;
        case    JPEG_ORIENTATION_TR: //0th row == visual top, 0th column == visual right-hand side
            flip(img, img, 1); //flip horizontally
            break;
        case    JPEG_ORIENTATION_BR: //0th row == visual bottom, 0th column == visual right-hand side
            flip(img, img, -1);//flip both horizontally and vertically
            break;
        case    JPEG_ORIENTATION_BL: //0th row == visual bottom, 0th column == visual left-hand side
            flip(img, img, 0); //flip vertically
            break;
        case    JPEG_ORIENTATION_LT: //0th row == visual left-hand side, 0th column == visual top
            transpose(img, img);
            break;
        case    JPEG_ORIENTATION_RT: //0th row == visual right-hand side, 0th column == visual top
            transpose(img, img);
            flip(img, img, 1); //flip horizontally
            break;
        case    JPEG_ORIENTATION_RB: //0th row == visual right-hand side, 0th column == visual bottom
            transpose(img, img);
            flip(img, img, -1); //flip both horizontally and vertically
            break;
        case    JPEG_ORIENTATION_LB: //0th row == visual left-hand side, 0th column == visual bottom
            transpose(img, img);
            flip(img, img, 0); //flip vertically
            break;
        default:
            //by default the image read has normal (JPEG_ORIENTATION_TL) orientation
            break;
    }
}

/***************************************************************************
 * following code is for supporting MJPEG image files
 * based on a message of Laurent Pinchart on the video4linux mailing list
 ***************************************************************************/

/* JPEG DHT Segment for YCrCb omitted from MJPEG data */
static
unsigned char my_jpeg_odml_dht[0x1a4] = {
    0xff, 0xc4, 0x01, 0xa2,

    0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,

    0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,

    0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04,
    0x04, 0x00, 0x00, 0x01, 0x7d,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
    0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1,
    0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a,
    0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
    0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85,
    0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
    0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
    0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
    0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4,
    0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,

    0x11, 0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04,
    0x04, 0x00, 0x01, 0x02, 0x77,
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41,
    0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09,
    0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17,
    0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83,
    0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
    0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8,
    0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6,
    0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4,
    0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

/*
 * Parse the DHT table.
 * This code comes from jpeg6b (jdmarker.c).
 */
static
int my_jpeg_load_dht (struct jpeg_decompress_struct *info, unsigned char *dht,
              JHUFF_TBL *ac_tables[], JHUFF_TBL *dc_tables[])
{
    unsigned int length = (dht[2] << 8) + dht[3] - 2;
    unsigned int pos = 4;
    unsigned int count, i;
    int index;

    JHUFF_TBL **hufftbl;
    unsigned char bits[17];
    unsigned char huffval[256];

    while (length > 16)
    {
       bits[0] = 0;
       index = dht[pos++];
       count = 0;
       for (i = 1; i <= 16; ++i)
       {
           bits[i] = dht[pos++];
           count += bits[i];
       }
       length -= 17;

       if (count > 256 || count > length)
           return -1;

       for (i = 0; i < count; ++i)
           huffval[i] = dht[pos++];
       length -= count;

       if (index & 0x10)
       {
           index -= 0x10;
           hufftbl = &ac_tables[index];
       }
       else
           hufftbl = &dc_tables[index];

       if (index < 0 || index >= NUM_HUFF_TBLS)
           return -1;

       if (*hufftbl == NULL)
           *hufftbl = jpeg_alloc_huff_table ((j_common_ptr)info);
       if (*hufftbl == NULL)
           return -1;

       memcpy ((*hufftbl)->bits, bits, sizeof (*hufftbl)->bits);
       memcpy ((*hufftbl)->huffval, huffval, sizeof (*hufftbl)->huffval);
    }

    if (length != 0)
       return -1;

    return 0;
}

/***************************************************************************
 * end of code for supportting MJPEG image files
 * based on a message of Laurent Pinchart on the video4linux mailing list
 ***************************************************************************/

bool  JpegDecoder::readData( Mat& img )
{
    volatile bool result = false;
    int step = (int)img.step;
    bool color = img.channels() > 1;

    if( m_state && m_width && m_height )
    {
        jpeg_decompress_struct* cinfo = &((JpegState*)m_state)->cinfo;
        JpegErrorMgr* jerr = &((JpegState*)m_state)->jerr;
        JSAMPARRAY buffer = 0;

        if( setjmp( jerr->setjmp_buffer ) == 0 )
        {
            /* check if this is a mjpeg image format */
            if ( cinfo->ac_huff_tbl_ptrs[0] == NULL &&
                cinfo->ac_huff_tbl_ptrs[1] == NULL &&
                cinfo->dc_huff_tbl_ptrs[0] == NULL &&
                cinfo->dc_huff_tbl_ptrs[1] == NULL )
            {
                /* yes, this is a mjpeg image format, so load the correct
                huffman table */
                my_jpeg_load_dht( cinfo,
                    my_jpeg_odml_dht,
                    cinfo->ac_huff_tbl_ptrs,
                    cinfo->dc_huff_tbl_ptrs );
            }

            if( color )
            {
                if( cinfo->num_components != 4 )
                {
                    cinfo->out_color_space = JCS_RGB;
                    cinfo->out_color_components = 3;
                }
                else
                {
                    cinfo->out_color_space = JCS_CMYK;
                    cinfo->out_color_components = 4;
                }
            }
            else
            {
                if( cinfo->num_components != 4 )
                {
                    cinfo->out_color_space = JCS_GRAYSCALE;
                    cinfo->out_color_components = 1;
                }
                else
                {
                    cinfo->out_color_space = JCS_CMYK;
                    cinfo->out_color_components = 4;
                }
            }

            jpeg_start_decompress( cinfo );

            buffer = (*cinfo->mem->alloc_sarray)((j_common_ptr)cinfo,
                                              JPOOL_IMAGE, m_width*4, 1 );

            uchar* data = img.ptr();
            for( ; m_height--; data += step )
            {
                jpeg_read_scanlines( cinfo, buffer, 1 );
                if( color )
                {
                    if( cinfo->out_color_components == 3 )
                        icvCvt_RGB2BGR_8u_C3R( buffer[0], 0, data, 0, cvSize(m_width,1) );
                    else
                        icvCvt_CMYK2BGR_8u_C4C3R( buffer[0], 0, data, 0, cvSize(m_width,1) );
                }
                else
                {
                    if( cinfo->out_color_components == 1 )
                        memcpy( data, buffer[0], m_width );
                    else
                        icvCvt_CMYK2Gray_8u_C4C1R( buffer[0], 0, data, 0, cvSize(m_width,1) );
                }
            }

            result = true;
            jpeg_finish_decompress( cinfo );
            setOrientation( img );
        }
    }

    close();
    return result;
}


/////////////////////// JpegEncoder ///////////////////

struct JpegDestination
{
    struct jpeg_destination_mgr pub;
    std::vector<uchar> *buf, *dst;
};

METHODDEF(void)
stub(j_compress_ptr)
{
}

METHODDEF(void)
term_destination (j_compress_ptr cinfo)
{
    JpegDestination* dest = (JpegDestination*)cinfo->dest;
    size_t sz = dest->dst->size(), bufsz = dest->buf->size() - dest->pub.free_in_buffer;
    if( bufsz > 0 )
    {
        dest->dst->resize(sz + bufsz);
        memcpy( &(*dest->dst)[0] + sz, &(*dest->buf)[0], bufsz);
    }
}

METHODDEF(boolean)
empty_output_buffer (j_compress_ptr cinfo)
{
    JpegDestination* dest = (JpegDestination*)cinfo->dest;
    size_t sz = dest->dst->size(), bufsz = dest->buf->size();
    dest->dst->resize(sz + bufsz);
    memcpy( &(*dest->dst)[0] + sz, &(*dest->buf)[0], bufsz);

    dest->pub.next_output_byte = &(*dest->buf)[0];
    dest->pub.free_in_buffer = bufsz;
    return TRUE;
}

static void jpeg_buffer_dest(j_compress_ptr cinfo, JpegDestination* destination)
{
    cinfo->dest = &destination->pub;

    destination->pub.init_destination = stub;
    destination->pub.empty_output_buffer = empty_output_buffer;
    destination->pub.term_destination = term_destination;
}


JpegEncoder::JpegEncoder()
{
    m_description = "JPEG files (*.jpeg;*.jpg;*.jpe)";
    m_buf_supported = true;
}


JpegEncoder::~JpegEncoder()
{
}

ImageEncoder JpegEncoder::newEncoder() const
{
    return makePtr<JpegEncoder>();
}

bool JpegEncoder::write( const Mat& img, const std::vector<int>& params )
{
    m_last_error.clear();

    struct fileWrapper
    {
        FILE* f;

        fileWrapper() : f(0) {}
        ~fileWrapper() { if(f) fclose(f); }
    };
    volatile bool result = false;
    fileWrapper fw;
    int width = img.cols, height = img.rows;

    std::vector<uchar> out_buf(1 << 12);
    AutoBuffer<uchar> _buffer;
    uchar* buffer;

    struct jpeg_compress_struct cinfo;
    JpegErrorMgr jerr;
    JpegDestination dest;

    jpeg_create_compress(&cinfo);
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = error_exit;

    if( !m_buf )
    {
        fw.f = fopen( m_filename.c_str(), "wb" );
        if( !fw.f )
            goto _exit_;
        jpeg_stdio_dest( &cinfo, fw.f );
    }
    else
    {
        dest.dst = m_buf;
        dest.buf = &out_buf;

        jpeg_buffer_dest( &cinfo, &dest );

        dest.pub.next_output_byte = &out_buf[0];
        dest.pub.free_in_buffer = out_buf.size();
    }

    if( setjmp( jerr.setjmp_buffer ) == 0 )
    {
        cinfo.image_width = width;
        cinfo.image_height = height;

        int _channels = img.channels();
        int channels = _channels > 1 ? 3 : 1;
        cinfo.input_components = channels;
        cinfo.in_color_space = channels > 1 ? JCS_RGB : JCS_GRAYSCALE;

        int quality = 95;
        int progressive = 0;
        int optimize = 0;
        int rst_interval = 0;
        int luma_quality = -1;
        int chroma_quality = -1;

        for( size_t i = 0; i < params.size(); i += 2 )
        {
            if( params[i] == CV_IMWRITE_JPEG_QUALITY )
            {
                quality = params[i+1];
                quality = MIN(MAX(quality, 0), 100);
            }

            if( params[i] == CV_IMWRITE_JPEG_PROGRESSIVE )
            {
                progressive = params[i+1];
            }

            if( params[i] == CV_IMWRITE_JPEG_OPTIMIZE )
            {
                optimize = params[i+1];
            }

            if( params[i] == CV_IMWRITE_JPEG_LUMA_QUALITY )
            {
                if (params[i+1] >= 0)
                {
                    luma_quality = MIN(MAX(params[i+1], 0), 100);

                    quality = luma_quality;

                    if (chroma_quality < 0)
                    {
                        chroma_quality = luma_quality;
                    }
                }
            }

            if( params[i] == CV_IMWRITE_JPEG_CHROMA_QUALITY )
            {
                if (params[i+1] >= 0)
                {
                    chroma_quality = MIN(MAX(params[i+1], 0), 100);
                }
            }

            if( params[i] == CV_IMWRITE_JPEG_RST_INTERVAL )
            {
                rst_interval = params[i+1];
                rst_interval = MIN(MAX(rst_interval, 0), 65535L);
            }
        }

        jpeg_set_defaults( &cinfo );
        cinfo.restart_interval = rst_interval;

        jpeg_set_quality( &cinfo, quality,
                          TRUE /* limit to baseline-JPEG values */ );
        if( progressive )
            jpeg_simple_progression( &cinfo );
        if( optimize )
            cinfo.optimize_coding = TRUE;

#if JPEG_LIB_VERSION >= 70
        if (luma_quality >= 0 && chroma_quality >= 0)
        {
            cinfo.q_scale_factor[0] = jpeg_quality_scaling(luma_quality);
            cinfo.q_scale_factor[1] = jpeg_quality_scaling(chroma_quality);
            if ( luma_quality != chroma_quality )
            {
                /* disable subsampling - ref. Libjpeg.txt */
                cinfo.comp_info[0].v_samp_factor = 1;
                cinfo.comp_info[0].h_samp_factor = 1;
                cinfo.comp_info[1].v_samp_factor = 1;
                cinfo.comp_info[1].h_samp_factor = 1;
            }
            jpeg_default_qtables( &cinfo, TRUE );
        }
#endif // #if JPEG_LIB_VERSION >= 70

        jpeg_start_compress( &cinfo, TRUE );

        if( channels > 1 )
            _buffer.allocate(width*channels);
        buffer = _buffer;

        for( int y = 0; y < height; y++ )
        {
            uchar *data = img.data + img.step*y, *ptr = data;

            if( _channels == 3 )
            {
                icvCvt_BGR2RGB_8u_C3R( data, 0, buffer, 0, cvSize(width,1) );
                ptr = buffer;
            }
            else if( _channels == 4 )
            {
                icvCvt_BGRA2BGR_8u_C4C3R( data, 0, buffer, 0, cvSize(width,1), 2 );
                ptr = buffer;
            }

            jpeg_write_scanlines( &cinfo, &ptr, 1 );
        }

        jpeg_finish_compress( &cinfo );
        result = true;
    }

_exit_:

    if(!result)
    {
        char jmsg_buf[JMSG_LENGTH_MAX];
        jerr.pub.format_message((j_common_ptr)&cinfo, jmsg_buf);
        m_last_error = jmsg_buf;
    }

    jpeg_destroy_compress( &cinfo );

    return result;
}

}

#endif

/* End of file. */
