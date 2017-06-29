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

const size_t WEBP_HEADER_SIZE = 32;

namespace cv
{

WebPDecoder::WebPDecoder()
{
    m_buf_supported = true;
    channels = 0;
}

WebPDecoder::~WebPDecoder() {}

size_t WebPDecoder::signatureLength() const
{
    return WEBP_HEADER_SIZE;
}

bool WebPDecoder::checkSignature(const String & signature) const
{
    bool ret = false;

    if(signature.size() >= WEBP_HEADER_SIZE)
    {
        WebPBitstreamFeatures features;
        if(VP8_STATUS_OK == WebPGetFeatures((uint8_t *)signature.c_str(),
                                            WEBP_HEADER_SIZE, &features))
        {
            ret = true;
        }
    }

    return ret;
}

ImageDecoder WebPDecoder::newDecoder() const
{
    return makePtr<WebPDecoder>();
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
        long int wfile_size = ftell(wfile);
        fseek(wfile, 0, SEEK_SET);

        if(wfile_size > static_cast<long int>(INT_MAX))
        {
            fclose(wfile);
            return false;
        }

        data.create(1, (int)wfile_size, CV_8U);

        size_t data_size = fread(data.ptr(), 1, wfile_size, wfile);

        if(wfile)
        {
            fclose(wfile);
        }

        if(static_cast<long int>(data_size) != wfile_size)
        {
            return false;
        }
    }
    else
    {
        data = m_buf;
    }

    WebPBitstreamFeatures features;
    if(VP8_STATUS_OK == WebPGetFeatures(data.ptr(), WEBP_HEADER_SIZE, &features))
    {
        m_width  = features.width;
        m_height = features.height;

        if (features.has_alpha)
        {
            m_type = CV_8UC4;
            channels = 4;
        }
        else
        {
            m_type = CV_8UC3;
            channels = 3;
        }

        return true;
    }

    return false;
}

bool WebPDecoder::readData(Mat &img)
{
    if( m_width > 0 && m_height > 0 )
    {
        bool convert_grayscale = (img.type() == CV_8UC1); // IMREAD_GRAYSCALE requested

        if (img.cols != m_width || img.rows != m_height || img.type() != m_type)
        {
            img.create(m_height, m_width, m_type);
        }

        uchar* out_data = img.ptr();
        size_t out_data_size = img.cols * img.rows * img.elemSize();

        uchar *res_ptr = 0;
        if (channels == 3)
        {
            res_ptr = WebPDecodeBGRInto(data.ptr(), data.total(), out_data,
                                        (int)out_data_size, (int)img.step);
        }
        else if (channels == 4)
        {
            res_ptr = WebPDecodeBGRAInto(data.ptr(), data.total(), out_data,
                                         (int)out_data_size, (int)img.step);
        }

        if(res_ptr == out_data)
        {
            if (convert_grayscale)
            {
                cvtColor(img, img, COLOR_BGR2GRAY);
            }
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

WebPEncoder::~WebPEncoder() { }

ImageEncoder WebPEncoder::newEncoder() const
{
    return makePtr<WebPEncoder>();
}

bool WebPEncoder::write(const Mat& img, const std::vector<int>& params)
{
    int channels = img.channels(), depth = img.depth();
    int width = img.cols, height = img.rows;

    const Mat *image = &img;
    Mat temp;
    size_t size = 0;

    bool comp_lossless = true;
    float quality = 100.0f;

    if (params.size() > 1)
    {
        if (params[0] == CV_IMWRITE_WEBP_QUALITY)
        {
            comp_lossless = false;
            quality = static_cast<float>(params[1]);
            if (quality < 1.0f)
            {
                quality = 1.0f;
            }
            if (quality > 100.0f)
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
    else if (channels == 2)
    {
        return false;
    }

    if (comp_lossless)
    {
        if(channels == 3)
        {
            size = WebPEncodeLosslessBGR(image->ptr(), width, height, (int)image->step, &out);
        }
        else if(channels == 4)
        {
            size = WebPEncodeLosslessBGRA(image->ptr(), width, height, (int)image->step, &out);
        }
    }
    else
    {
        if(channels == 3)
        {
            size = WebPEncodeBGR(image->ptr(), width, height, (int)image->step, quality, &out);
        }
        else if(channels == 4)
        {
            size = WebPEncodeBGRA(image->ptr(), width, height, (int)image->step, quality, &out);
        }
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
