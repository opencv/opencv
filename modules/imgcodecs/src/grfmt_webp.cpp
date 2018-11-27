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

#include <opencv2/core/utils/configuration.private.hpp>

namespace cv
{

// 64Mb limit to avoid memory DDOS
static size_t param_maxFileSize = utils::getConfigurationParameterSizeT("OPENCV_IMGCODECS_WEBP_MAX_FILE_SIZE", 64*1024*1024);

static const size_t WEBP_HEADER_SIZE = 32;

WebPDecoder::WebPDecoder()
{
    m_buf_supported = true;
    channels = 0;
    fs_size = 0;
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
    uint8_t header[WEBP_HEADER_SIZE] = { 0 };
    if (m_buf.empty())
    {
        fs.open(m_filename.c_str(), std::ios::binary);
        fs.seekg(0, std::ios::end);
        fs_size = safeCastToSizeT(fs.tellg(), "File is too large");
        fs.seekg(0, std::ios::beg);
        CV_Assert(fs && "File stream error");
        CV_CheckGE(fs_size, WEBP_HEADER_SIZE, "File is too small");
        CV_CheckLE(fs_size, param_maxFileSize, "File is too large. Increase OPENCV_IMGCODECS_WEBP_MAX_FILE_SIZE parameter if you want to process large files");

        fs.read((char*)header, sizeof(header));
        CV_Assert(fs && "Can't read WEBP_HEADER_SIZE bytes");
    }
    else
    {
        CV_CheckGE(m_buf.total(), WEBP_HEADER_SIZE, "Buffer is too small");
        memcpy(header, m_buf.ptr(), sizeof(header));
        data = m_buf;
    }

    WebPBitstreamFeatures features;
    if (VP8_STATUS_OK == WebPGetFeatures(header, sizeof(header), &features))
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
    CV_CheckGE(m_width, 0, ""); CV_CheckGE(m_height, 0, "");

    CV_CheckEQ(img.cols, m_width, "");
    CV_CheckEQ(img.rows, m_height, "");

    if (m_buf.empty())
    {
        fs.seekg(0, std::ios::beg); CV_Assert(fs && "File stream error");
        data.create(1, validateToInt(fs_size), CV_8UC1);
        fs.read((char*)data.ptr(), fs_size);
        CV_Assert(fs && "Can't read file data");
        fs.close();
    }
    CV_Assert(data.type() == CV_8UC1); CV_Assert(data.rows == 1);

    {
        Mat read_img;
        CV_CheckType(img.type(), img.type() == CV_8UC1 || img.type() == CV_8UC3 || img.type() == CV_8UC4, "");
        if (img.type() != m_type)
        {
            read_img.create(m_height, m_width, m_type);
        }
        else
        {
            read_img = img;  // copy header
        }

        uchar* out_data = read_img.ptr();
        size_t out_data_size = read_img.dataend - out_data;

        uchar *res_ptr = NULL;
        if (channels == 3)
        {
            CV_CheckTypeEQ(read_img.type(), CV_8UC3, "");
            res_ptr = WebPDecodeBGRInto(data.ptr(), data.total(), out_data,
                                        (int)out_data_size, (int)read_img.step);
        }
        else if (channels == 4)
        {
            CV_CheckTypeEQ(read_img.type(), CV_8UC4, "");
            res_ptr = WebPDecodeBGRAInto(data.ptr(), data.total(), out_data,
                                         (int)out_data_size, (int)read_img.step);
        }

        if (res_ptr != out_data)
            return false;

        if (read_img.data == img.data && img.type() == m_type)
        {
            // nothing
        }
        else if (img.type() == CV_8UC1)
        {
            cvtColor(read_img, img, COLOR_BGR2GRAY);
        }
        else if (img.type() == CV_8UC3 && m_type == CV_8UC4)
        {
            cvtColor(read_img, img, COLOR_BGRA2BGR);
        }
        else if (img.type() == CV_8UC4 && m_type == CV_8UC3)
        {
            cvtColor(read_img, img, COLOR_BGR2BGRA);
        }
        else
        {
            CV_Error(Error::StsInternal, "");
        }
    }
    return true;
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
    CV_CheckDepthEQ(img.depth(), CV_8U, "WebP codec supports 8U images only");

    const int width = img.cols, height = img.rows;

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

    int channels = img.channels();
    CV_Check(channels, channels == 1 || channels == 3 || channels == 4, "");

    const Mat *image = &img;
    Mat temp;

    if (channels == 1)
    {
        cvtColor(*image, temp, COLOR_GRAY2BGR);
        image = &temp;
        channels = 3;
    }

    uint8_t *out = NULL;
    size_t size = 0;
    if (comp_lossless)
    {
        if (channels == 3)
        {
            size = WebPEncodeLosslessBGR(image->ptr(), width, height, (int)image->step, &out);
        }
        else if (channels == 4)
        {
            size = WebPEncodeLosslessBGRA(image->ptr(), width, height, (int)image->step, &out);
        }
    }
    else
    {
        if (channels == 3)
        {
            size = WebPEncodeBGR(image->ptr(), width, height, (int)image->step, quality, &out);
        }
        else if (channels == 4)
        {
            size = WebPEncodeBGRA(image->ptr(), width, height, (int)image->step, quality, &out);
        }
    }
#if WEBP_DECODER_ABI_VERSION >= 0x0206
    Ptr<uint8_t> out_cleaner(out, WebPFree);
#else
    Ptr<uint8_t> out_cleaner(out, free);
#endif

    CV_Assert(size > 0);

    if (m_buf)
    {
        m_buf->resize(size);
        memcpy(&(*m_buf)[0], out, size);
    }
    else
    {
        FILE *fd = fopen(m_filename.c_str(), "wb");
        if (fd != NULL)
        {
            fwrite(out, size, sizeof(uint8_t), fd);
            fclose(fd); fd = NULL;
        }
    }

    return size > 0;
}

}

#endif
