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

#ifdef HAVE_WEBP

#include "precomp.hpp"

#include <webp/decode.h>
#include <webp/encode.h>
#include <stdio.h>
#include <limits.h>

#include "grfmt_webp.hpp"

#include "opencv2/imgproc.hpp"

namespace cv
{

WebPDecoder::WebPDecoder()
{
    m_signature = "RIFF....WEBPVP8 ";
    m_buf_supported = true;
}

WebPDecoder::~WebPDecoder()
{
}

ImageDecoder WebPDecoder::newDecoder() const
{
    return new WebPDecoder;
}

bool WebPDecoder::checkSignature( const std::string& signature ) const
{
    size_t len = signatureLength();
    bool ret = false;

    if(signature.size() >= len)
    {
        ret = ( (memcmp(signature.c_str(), m_signature.c_str(), 4) == 0) &&
            (memcmp(signature.c_str() + 8, m_signature.c_str() + 8, 4) == 0) );
    }

    return ret;
}

bool WebPDecoder::readHeader()
{
    if (m_buf.empty())
    {
        FILE * wfile = NULL;

        wfile = fopen(m_filename.c_str(), "rb");

        if(wfile == NULL)
        {
            return false;
        }

        fseek(wfile, 0, SEEK_END);
        size_t wfile_size = ftell(wfile);
        fseek(wfile, 0, SEEK_SET);

        if(wfile_size > (size_t)INT_MAX)
        {
            fclose(wfile);
            return false;
        }

        data.create(1, (int)wfile_size, CV_8U);

        size_t data_size = fread(data.data, 1, wfile_size, wfile);

        if(wfile)
        {
            fclose(wfile);
        }

        if( data_size < wfile_size )
        {
            return false;
        }
    }
    else
    {
        data = m_buf;
    }

    if(WebPGetInfo(data.data, data.total(), &m_width, &m_height) == 1)
    {
        m_type = CV_8UC3;
        return true;
    }

    return false;
}

bool WebPDecoder::readData(Mat &img)
{
    if( m_width > 0 && m_height > 0 )
    {
        uchar* out_data = img.data;
        unsigned int out_data_size = m_width * m_height * 3 * sizeof(uchar);

        uchar *res_ptr = WebPDecodeBGRInto(data.data, data.total(), out_data, out_data_size, m_width * 3);

        if(res_ptr == out_data)
        {
            return true;
        }
    }

    return false;
}

WebPEncoder::WebPEncoder()
{
    m_description = "WebP files (*.webp)";
    m_buf_supported = true;
}

WebPEncoder::~WebPEncoder()
{
}

ImageEncoder WebPEncoder::newEncoder() const
{
    return new WebPEncoder();
}

bool WebPEncoder::write(const Mat& img, const std::vector<int>& params)
{
    int channels = img.channels(), depth = img.depth();
    int width = img.cols, height = img.rows;

    const Mat *image = &img;
    Mat temp;
    int size = 0;

    bool comp_lossless = true;
    int quality = 100;

    if (params.size() > 1)
    {
        if (params[0] == CV_IMWRITE_WEBP_QUALITY)
        {
            comp_lossless = false;
            quality = params[1];
            if (quality < 1)
            {
                quality = 1;
            }
            if (quality > 100)
            {
                comp_lossless = true;
            }
        }
    }

    uint8_t *out = NULL;

    if(depth != CV_8U)
    {
        return false;
    }

    if(channels == 1)
    {
        cvtColor(*image, temp, CV_GRAY2BGR);
        image = &temp;
        channels = 3;
    }

    if (comp_lossless)
    {
        size = WebPEncodeLosslessBGR(image->data, width, height, ((width * 3 + 3) & ~3), &out);
    }
    else
    {
        size = WebPEncodeBGR(image->data, width, height, ((width * 3 + 3) & ~3), (float)quality, &out);
    }

    if(size > 0)
    {
        if(m_buf)
        {
            m_buf->resize(size);
            memcpy(&(*m_buf)[0], out, size);
        }
        else
        {
            FILE *fd = fopen(m_filename.c_str(), "wb");
            if(fd != NULL)
            {
                fwrite(out, size, sizeof(uint8_t), fd);
                fclose(fd); fd = NULL;
            }
        }
    }

    if (out != NULL)
    {
        free(out);
        out = NULL;
    }

    return size > 0;
}

}

#endif
