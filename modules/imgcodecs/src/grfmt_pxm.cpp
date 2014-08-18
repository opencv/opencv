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
#include "utils.hpp"
#include "grfmt_pxm.hpp"

namespace cv
{

///////////////////////// P?M reader //////////////////////////////

static int ReadNumber( RLByteStream& strm, int maxdigits )
{
    int code;
    int val = 0;
    int digits = 0;

    code = strm.getByte();

    if( !isdigit(code))
    {
        do
        {
            if( code == '#' )
            {
                do
                {
                    code = strm.getByte();
                }
                while( code != '\n' && code != '\r' );
            }

            code = strm.getByte();

            while( isspace(code))
                code = strm.getByte();
        }
        while( !isdigit( code ));
    }

    do
    {
        val = val*10 + code - '0';
        if( ++digits >= maxdigits ) break;
        code = strm.getByte();
    }
    while( isdigit(code));

    return val;
}


PxMDecoder::PxMDecoder()
{
    m_offset = -1;
    m_buf_supported = true;
}


PxMDecoder::~PxMDecoder()
{
    close();
}

size_t PxMDecoder::signatureLength() const
{
    return 3;
}

bool PxMDecoder::checkSignature( const String& signature ) const
{
    return signature.size() >= 3 && signature[0] == 'P' &&
           '1' <= signature[1] && signature[1] <= '6' &&
           isspace(signature[2]);
}

ImageDecoder PxMDecoder::newDecoder() const
{
    return makePtr<PxMDecoder>();
}

void  PxMDecoder::close()
{
    m_strm.close();
}


bool  PxMDecoder::readHeader()
{
    bool result = false;

    if( !m_buf.empty() )
    {
        if( !m_strm.open(m_buf) )
            return false;
    }
    else if( !m_strm.open( m_filename ))
        return false;

    try
    {
        int code = m_strm.getByte();
        if( code != 'P' )
            throw RBS_BAD_HEADER;

        code = m_strm.getByte();
        switch( code )
        {
        case '1': case '4': m_bpp = 1; break;
        case '2': case '5': m_bpp = 8; break;
        case '3': case '6': m_bpp = 24; break;
        default: throw RBS_BAD_HEADER;
        }

        m_binary = code >= '4';
        m_type = m_bpp > 8 ? CV_8UC3 : CV_8UC1;

        m_width = ReadNumber( m_strm, INT_MAX );
        m_height = ReadNumber( m_strm, INT_MAX );

        m_maxval = m_bpp == 1 ? 1 : ReadNumber( m_strm, INT_MAX );
        if( m_maxval > 65535 )
            throw RBS_BAD_HEADER;

        //if( m_maxval > 255 ) m_binary = false; nonsense
        if( m_maxval > 255 )
            m_type = CV_MAKETYPE(CV_16U, CV_MAT_CN(m_type));

        if( m_width > 0 && m_height > 0 && m_maxval > 0 && m_maxval < (1 << 16))
        {
            m_offset = m_strm.getPos();
            result = true;
        }
    }
    catch(...)
    {
    }

    if( !result )
    {
        m_offset = -1;
        m_width = m_height = -1;
        m_strm.close();
    }
    return result;
}


bool  PxMDecoder::readData( Mat& img )
{
    int color = img.channels() > 1;
    uchar* data = img.ptr();
    int step = (int)img.step;
    PaletteEntry palette[256];
    bool   result = false;
    int  bit_depth = CV_ELEM_SIZE1(m_type)*8;
    int  src_pitch = (m_width*m_bpp*bit_depth/8 + 7)/8;
    int  nch = CV_MAT_CN(m_type);
    int  width3 = m_width*nch;
    int  i, x, y;

    if( m_offset < 0 || !m_strm.isOpened())
        return false;

    AutoBuffer<uchar> _src(src_pitch + 32);
    uchar* src = _src;
    AutoBuffer<uchar> _gray_palette;
    uchar* gray_palette = _gray_palette;

    // create LUT for converting colors
    if( bit_depth == 8 )
    {
        _gray_palette.allocate(m_maxval + 1);
        gray_palette = _gray_palette;

        for( i = 0; i <= m_maxval; i++ )
            gray_palette[i] = (uchar)((i*255/m_maxval)^(m_bpp == 1 ? 255 : 0));

        FillGrayPalette( palette, m_bpp==1 ? 1 : 8 , m_bpp == 1 );
    }

    try
    {
        m_strm.setPos( m_offset );

        switch( m_bpp )
        {
        ////////////////////////// 1 BPP /////////////////////////
        case 1:
            if( !m_binary )
            {
                for( y = 0; y < m_height; y++, data += step )
                {
                    for( x = 0; x < m_width; x++ )
                        src[x] = ReadNumber( m_strm, 1 ) != 0;

                    if( color )
                        FillColorRow8( data, src, m_width, palette );
                    else
                        FillGrayRow8( data, src, m_width, gray_palette );
                }
            }
            else
            {
                for( y = 0; y < m_height; y++, data += step )
                {
                    m_strm.getBytes( src, src_pitch );

                    if( color )
                        FillColorRow1( data, src, m_width, palette );
                    else
                        FillGrayRow1( data, src, m_width, gray_palette );
                }
            }
            result = true;
            break;

        ////////////////////////// 8 BPP /////////////////////////
        case 8:
        case 24:
            for( y = 0; y < m_height; y++, data += step )
            {
                if( !m_binary )
                {
                    for( x = 0; x < width3; x++ )
                    {
                        int code = ReadNumber( m_strm, INT_MAX );
                        if( (unsigned)code > (unsigned)m_maxval ) code = m_maxval;
                        if( bit_depth == 8 )
                            src[x] = gray_palette[code];
                        else
                            ((ushort *)src)[x] = (ushort)code;
                    }
                }
                else
                {
                    m_strm.getBytes( src, src_pitch );
                    if( bit_depth == 16 && !isBigEndian() )
                    {
                        for( x = 0; x < width3; x++ )
                        {
                            uchar v = src[x * 2];
                            src[x * 2] = src[x * 2 + 1];
                            src[x * 2 + 1] = v;
                        }
                    }
                }

                if( img.depth() == CV_8U && bit_depth == 16 )
                {
                    for( x = 0; x < width3; x++ )
                    {
                        int v = ((ushort *)src)[x];
                        src[x] = (uchar)(v >> 8);
                    }
                }

                if( m_bpp == 8 ) // image has one channel
                {
                    if( color )
                    {
                        if( img.depth() == CV_8U ) {
                            uchar *d = data, *s = src, *end = src + m_width;
                            for( ; s < end; d += 3, s++)
                                d[0] = d[1] = d[2] = *s;
                        } else {
                            ushort *d = (ushort *)data, *s = (ushort *)src, *end = ((ushort *)src) + m_width;
                            for( ; s < end; s++, d += 3)
                                d[0] = d[1] = d[2] = *s;
                        }
                    }
                    else
                        memcpy( data, src, m_width*(bit_depth/8) );
                }
                else
                {
                    if( color )
                    {
                        if( img.depth() == CV_8U )
                            icvCvt_RGB2BGR_8u_C3R( src, 0, data, 0, cvSize(m_width,1) );
                        else
                            icvCvt_RGB2BGR_16u_C3R( (ushort *)src, 0, (ushort *)data, 0, cvSize(m_width,1) );
                    }
                    else if( img.depth() == CV_8U )
                        icvCvt_BGR2Gray_8u_C3C1R( src, 0, data, 0, cvSize(m_width,1), 2 );
                    else
                        icvCvt_BGRA2Gray_16u_CnC1R( (ushort *)src, 0, (ushort *)data, 0, cvSize(m_width,1), 3, 2 );
                }
            }
            result = true;
            break;
        default:
            assert(0);
        }
    }
    catch(...)
    {
    }

    return result;
}


//////////////////////////////////////////////////////////////////////////////////////////

PxMEncoder::PxMEncoder()
{
    m_description = "Portable image format (*.pbm;*.pgm;*.ppm;*.pxm;*.pnm)";
    m_buf_supported = true;
}


PxMEncoder::~PxMEncoder()
{
}


ImageEncoder  PxMEncoder::newEncoder() const
{
    return makePtr<PxMEncoder>();
}


bool  PxMEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_16U;
}


bool  PxMEncoder::write( const Mat& img, const std::vector<int>& params )
{
    bool isBinary = true;

    int  width = img.cols, height = img.rows;
    int  _channels = img.channels(), depth = (int)img.elemSize1()*8;
    int  channels = _channels > 1 ? 3 : 1;
    int  fileStep = width*(int)img.elemSize();
    int  x, y;

    for( size_t i = 0; i < params.size(); i += 2 )
        if( params[i] == CV_IMWRITE_PXM_BINARY )
            isBinary = params[i+1] != 0;

    WLByteStream strm;

    if( m_buf )
    {
        if( !strm.open(*m_buf) )
            return false;
        int t = CV_MAKETYPE(img.depth(), channels);
        m_buf->reserve( alignSize(256 + (isBinary ? fileStep*height :
            ((t == CV_8UC1 ? 4 : t == CV_8UC3 ? 4*3+2 :
            t == CV_16UC1 ? 6 : 6*3+2)*width+1)*height), 256));
    }
    else if( !strm.open(m_filename) )
        return false;

    int  lineLength;
    int  bufferSize = 128; // buffer that should fit a header

    if( isBinary )
        lineLength = width * (int)img.elemSize();
    else
        lineLength = (6 * channels + (channels > 1 ? 2 : 0)) * width + 32;

    if( bufferSize < lineLength )
        bufferSize = lineLength;

    AutoBuffer<char> _buffer(bufferSize);
    char* buffer = _buffer;

    // write header;
    sprintf( buffer, "P%c\n%d %d\n%d\n",
             '2' + (channels > 1 ? 1 : 0) + (isBinary ? 3 : 0),
             width, height, (1 << depth) - 1 );

    strm.putBytes( buffer, (int)strlen(buffer) );

    for( y = 0; y < height; y++ )
    {
        const uchar* const data = img.ptr(y);
        if( isBinary )
        {
            if( _channels == 3 )
            {
                if( depth == 8 )
                    icvCvt_BGR2RGB_8u_C3R( (const uchar*)data, 0,
                        (uchar*)buffer, 0, cvSize(width,1) );
                else
                    icvCvt_BGR2RGB_16u_C3R( (const ushort*)data, 0,
                        (ushort*)buffer, 0, cvSize(width,1) );
            }

            // swap endianness if necessary
            if( depth == 16 && !isBigEndian() )
            {
                if( _channels == 1 )
                    memcpy( buffer, data, fileStep );
                for( x = 0; x < width*channels*2; x += 2 )
                {
                    uchar v = buffer[x];
                    buffer[x] = buffer[x + 1];
                    buffer[x + 1] = v;
                }
            }
            strm.putBytes( (channels > 1 || depth > 8) ? buffer : (const char*)data, fileStep );
        }
        else
        {
            char* ptr = buffer;

            if( channels > 1 )
            {
                if( depth == 8 )
                {
                    for( x = 0; x < width*channels; x += channels )
                    {
                        sprintf( ptr, "% 4d", data[x + 2] );
                        ptr += 4;
                        sprintf( ptr, "% 4d", data[x + 1] );
                        ptr += 4;
                        sprintf( ptr, "% 4d", data[x] );
                        ptr += 4;
                        *ptr++ = ' ';
                        *ptr++ = ' ';
                    }
                }
                else
                {
                    for( x = 0; x < width*channels; x += channels )
                    {
                        sprintf( ptr, "% 6d", ((const ushort *)data)[x + 2] );
                        ptr += 6;
                        sprintf( ptr, "% 6d", ((const ushort *)data)[x + 1] );
                        ptr += 6;
                        sprintf( ptr, "% 6d", ((const ushort *)data)[x] );
                        ptr += 6;
                        *ptr++ = ' ';
                        *ptr++ = ' ';
                    }
                }
            }
            else
            {
                if( depth == 8 )
                {
                    for( x = 0; x < width; x++ )
                    {
                        sprintf( ptr, "% 4d", data[x] );
                        ptr += 4;
                    }
                }
                else
                {
                    for( x = 0; x < width; x++ )
                    {
                        sprintf( ptr, "% 6d", ((const ushort *)data)[x] );
                        ptr += 6;
                    }
                }
            }

            *ptr++ = '\n';

            strm.putBytes( buffer, (int)(ptr - buffer) );
        }
    }

    strm.close();
    return true;
}

}
