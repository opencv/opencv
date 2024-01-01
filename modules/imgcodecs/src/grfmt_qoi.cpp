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
#include "bitstrm.hpp"
#include "grfmt_qoi.hpp"
#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_QOI

#define QOI_IMPLEMENTATION
#include "qoi.h"

namespace cv
{

QoiDecoder::QoiDecoder()
{
    m_signature = "qoif";
    m_buf_supported = true;
}

QoiDecoder::~QoiDecoder()
{
}

static void swap_endianess(int32_t& val)
{
    static const uint32_t A(0x000000ffU);
    static const uint32_t B(0x0000ff00U);
    static const uint32_t C(0x00ff0000U);
    static const uint32_t D(0xff000000U);

    union {
        uint32_t asU32;
        int32_t  asS32;
    } tmp;

    tmp.asS32 = val;

    tmp.asU32 = ( (tmp.asU32 & A) << 24 )
              | ( (tmp.asU32 & B) <<  8 )
              | ( (tmp.asU32 & C) >>  8 )
              | ( (tmp.asU32 & D) >> 24 );

    val = tmp.asS32;
}

bool QoiDecoder::readHeader()
{
    RLByteStream strm;

    if( !m_buf.empty() )
    {
        CV_Assert( m_buf.type() == CV_8UC1 );
        CV_Assert( m_buf.rows == 1 );

        size_t mbuf_len = m_buf.cols * m_buf.rows * m_buf.elemSize();
        if( mbuf_len < QOI_HEADER_SIZE + sizeof(qoi_padding) )
        {
            CV_LOG_WARNING( NULL, "QOI: readHeader(): Too short data." );
            return false;
        }
        uint8_t* padding_pos = m_buf.data + mbuf_len - sizeof(qoi_padding);
        if( memcmp( padding_pos, qoi_padding, sizeof(qoi_padding) ) != 0 )
        {
            CV_LOG_WARNING( NULL, "QOI: readHeader(): missing padding data." );
            return false;
        }
        if( !strm.open(m_buf) )
        {
            return false;
        }
    }
    else if( !strm.open( m_filename ) )
    {
        return false;
    }

    strm.skip(4); // Signature
    m_width    = strm.getDWord(); // As Big Endian
    m_height   = strm.getDWord(); // As Bit Endian
    int ncn    = strm.getByte();

    if( !isBigEndian() )
    {
        swap_endianess( m_width );  // As Little Endian
        swap_endianess( m_height ); // As Little Endian
    }

    CV_Assert( ncn == 3 || ncn == 4 );
    m_type = CV_MAKETYPE( CV_8U, ncn );

    strm.close();
    return true;
}


bool QoiDecoder::readData( Mat& img )
{
    bool result = true;

    const int img_height = img.rows;
    const int img_width = img.cols;
    const bool is_img_gray = ( img.channels() == 1 );
    const int img_ncn = ( is_img_gray )? 3 /* Force to RGB */ : img.channels();

    CV_Assert( img_ncn == 3 || img_ncn == 4 );

    // Decode
    uint8_t* rawImg = nullptr;
    qoi_desc desc;
    if( !m_buf.empty() )
    {
        size_t m_buf_len = m_buf.cols * m_buf.rows * m_buf.elemSize();
        rawImg = static_cast<uint8_t*>( qoi_decode( m_buf.ptr(), m_buf_len, &desc, img_ncn ) );
    }
    else
    {
        rawImg = static_cast<uint8_t*>( qoi_read( m_filename.c_str(), &desc, img_ncn ) );
    }

    if( rawImg == nullptr )
    {
        CV_Error(Error::StsParseError, "Cannot decode from QOI");
    }
    CV_Assert( rawImg != nullptr );

    // Convert to cv::Mat()
    if( img_ncn == 3 )
    {
        const int code = (is_img_gray) ? cv::COLOR_RGB2GRAY : cv::COLOR_RGB2BGR;
        cv::Mat src( img_height, img_width, CV_8UC3, rawImg );
        cv::cvtColor( src, img, code );
        src.release();
    }
    else // if( img_ncn == 4 )
    {
        cv::Mat src( img_height, img_width, CV_8UC4, rawImg );
        cv::cvtColor( src, img, cv::COLOR_RGBA2BGRA );
        src.release();
    }

    QOI_FREE(rawImg); // for qoi_decode() or qoi_read()

    return result;
}


//////////////////////////////////////////////////////////////////////////////////////////

QoiEncoder::QoiEncoder()
{
    m_description = "Quite OK Image format (*.qoi)";
    m_buf_supported = true;
}

QoiEncoder::~QoiEncoder()
{
}

bool QoiEncoder::write(const Mat& _img, const std::vector<int>& params)
{
    // Qoi has no options.
    CV_UNUSED( params );

    // QOI supports only RGB/RGBA, not GRAY. In OpenCV, GRAY will save as RGB.
    const int _img_ncn = _img.channels();
    CV_Assert( _img_ncn == 1 || _img_ncn == 3 || _img_ncn == 4 );

    Mat img; // BGR or BGRA Mat

#ifdef QOI_EXT_ENCODE_ORDER_BGRA
    if( _img_ncn == 1 )
    {
        cv::cvtColor( _img, img, cv::COLOR_GRAY2RGB );
    }
    else
    {
        img = _img; // use it directly.
    }
#else
    if( _img_ncn == 1 )
    {
        cv::cvtColor( _img, img, cv::COLOR_GRAY2RGB );
    }
    else if( _img_ncn == 3 )
    {
        cv::cvtColor( _img, img, cv::COLOR_BGR2RGB );
    }
    else if( _img_ncn == 4 )
    {
        cv::cvtColor( _img, img, cv::COLOR_BGRA2RGBA );
    }
#endif

    const int img_height = img.rows;
    const int img_width = img.cols;
    const int img_ncn = img.channels();
    CV_Assert( img_ncn == 3 || img_ncn == 4 );

    uint8_t* rawImg = nullptr;
    std::vector<uint8_t> rawBuf;

    // QOI requests 1 plane RGB/RGBA image, it should not be splitted.
    if ( img.isContinuous() )
    {
        rawImg = img.ptr(0);
    }
    else
    {
        rawBuf.resize( img_height * img_width * img_ncn );
        uint8_t* dst = rawBuf.data();
        for( int iy = 0; iy < img_height; iy++ )
        {
            memcpy( img.ptr(iy), dst, img_width * img_ncn );
            dst += img_width * img_ncn;
        }
        img.release();
        rawImg = rawBuf.data();
    }

    qoi_desc desc
    {
        static_cast<unsigned int>(img_width),
        static_cast<unsigned int>(img_height),
        static_cast<unsigned char>(img_ncn),
        QOI_SRGB
    };

    int encode_len = 0;
    if( m_buf )
    {
        void *encode_ptr = nullptr;
        encode_ptr = qoi_encode( rawImg, &desc, &encode_len );
        if( encode_ptr != nullptr )
        {
            m_buf->resize( encode_len );
            memcpy( m_buf->data(), encode_ptr, encode_len );
            QOI_FREE( encode_ptr ); // for qoi_encode()
        } else {
            encode_len = 0; // Failed.
        }
    }
    else
    {
        encode_len = qoi_write( m_filename.c_str(), rawImg, &desc);
    }

    rawBuf.clear();
    img.release();
    return ( encode_len > 0 );
}

} // namespace cv

#endif // HAVE_QOI
