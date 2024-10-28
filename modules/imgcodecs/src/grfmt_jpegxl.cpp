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
#include "grfmt_jpegxl.hpp"

#ifdef HAVE_JPEGXL

#include <opencv2/core/utils/logger.hpp>

namespace cv
{

/////////////////////// JpegXLDecoder ///////////////////

JpegXLDecoder::JpegXLDecoder() : m_f(nullptr, fclose)
{
    m_signature = "\xFF\x0A";
    m_decoder = nullptr;
    m_buf_supported = true;
    m_type = -1;
}

JpegXLDecoder::~JpegXLDecoder()
{
    close();
}

void JpegXLDecoder::close()
{
    if (m_decoder)
    {
        m_decoder.release();
    }

    if (m_f)
    {
        m_f.release();
    }

    m_width = m_height = 0;
    m_type = -1;
}

ImageDecoder JpegXLDecoder::newDecoder() const
{
    return makePtr<JpegXLDecoder>();
}

bool JpegXLDecoder::read(Mat* pimg)
{
    // Open file
    if (!m_f) {
        m_f.reset(fopen(m_filename.c_str(), "rb"));
        if (!m_f)
            return false;
    }

    // Initialize decoder
    if (!m_decoder) {
        m_decoder = JxlDecoderMake(nullptr);
        if (!m_decoder)
            return false;
    }

    // Set up parallel m_parallel_runner
    if (!m_parallel_runner) {
        m_parallel_runner = JxlThreadParallelRunnerMake(nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
        if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(m_decoder.get(),
                                                            JxlThreadParallelRunner,
                                                            m_parallel_runner.get())) {
            return false;
        }
    }

    // Create buffer for reading
    const size_t read_buffer_size = 16384;  // 16KB chunks
    if (m_read_buffer.capacity() < read_buffer_size) {
        m_read_buffer.resize(read_buffer_size);
    }
    
    // Start decoding loop
    JxlDecoderStatus status = JXL_DEC_SUCCESS;
    do {
        if (m_type != -1 && pimg) {
            pimg->create(m_height, m_width, m_type);
            // Set desired output format
            if (!pimg->isContinuous())
                return false;
            if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(m_decoder.get(), 
                                                            &m_format,
                                                            pimg->ptr<uint8_t>(),
                                                            pimg->total() * pimg->elemSize())) {
                return false;
            }
        }

        // Check if we need more input
        if (status == JXL_DEC_NEED_MORE_INPUT) {
            size_t remaining = JxlDecoderReleaseInput(m_decoder.get());
            // Move any remaining bytes to the beginning
            if (remaining > 0) {
                memmove(m_read_buffer.data(), m_read_buffer.data() + m_read_buffer.size() - remaining, remaining);
            }
            
            // Read more data from file
            size_t bytes_read = fread(m_read_buffer.data() + remaining, 
                                    1, m_read_buffer.size() - remaining, m_f.get());
            if (bytes_read == 0) {
                if (ferror(m_f.get())) {
                    throw std::runtime_error("Error reading input file");
                }
                // If we reached EOF but decoder needs more input, file is truncated
                if (status == JXL_DEC_NEED_MORE_INPUT) {
                    throw std::runtime_error("Truncated JXL file");
                }
            }
            
            // Set input buffer
            if (JXL_DEC_SUCCESS != JxlDecoderSetInput(m_decoder.get(), 
                                                        m_read_buffer.data(),
                                                        bytes_read + remaining)) {
                throw std::runtime_error("JxlDecoderSetInput failed");
            }
        }

        // Get the next decoder status
        status = JxlDecoderProcessInput(m_decoder.get());

        // Handle different decoder states
        switch (status) {
            case JXL_DEC_BASIC_INFO: {
                if (m_type != -1)
                    return false; // duplicate info
                JxlBasicInfo info;
                if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(m_decoder.get(), &info)) {
                    throw std::runtime_error("JxlDecoderGetBasicInfo failed");
                }
                m_width = info.xsize;
                m_height = info.ysize;
                m_format = {info.num_color_channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
                if (info.exponent_bits_per_sample > 0) {
                    m_format.data_type = JXL_TYPE_FLOAT;
                    m_type = info.num_color_channels == 3 ? CV_32FC3 : CV_32FC1;
                } else {
                    switch (info.bits_per_sample) {
                        case 8:
                            m_type = info.num_color_channels == 3 ? CV_8UC3 : CV_8UC1;
                            break;
                        case 16:
                            m_type = info.num_color_channels == 3 ? CV_16UC3 : CV_16UC1;
                            break;
                        default:
                            return false;
                    }
                }
                if (!pimg)
                    return true;
                break;
            }
            case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
                // Buffer already set in BASIC_INFO
                break;
            }
            case JXL_DEC_FULL_IMAGE: {
                // Image is ready
                break;
            }
            case JXL_DEC_SUCCESS: {
                // Decoding finished
                break;
            }
            case JXL_DEC_ERROR: {
                throw std::runtime_error("Decoder error");
            }
            default: {
                // Other events we don't handle for this example
                break;
            }
        }
    } while (status != JXL_DEC_SUCCESS);

    return true;
}

bool JpegXLDecoder::readHeader()
{
    close();

    return read(nullptr);
}

bool JpegXLDecoder::readData(Mat& img)
{
    if (!m_decoder || m_width == 0 || m_height == 0)
        return false;

    return read(&img);
}

/////////////////////// JpegXLEncoder ///////////////////

JpegXLEncoder::JpegXLEncoder()
{
    m_description = "JPEG XL files (*.jxl)";
    m_buf_supported = true;
}

JpegXLEncoder::~JpegXLEncoder()
{
}

ImageEncoder JpegXLEncoder::newEncoder() const
{
    return makePtr<JpegXLEncoder>();
}

bool JpegXLEncoder::write(const Mat& img, const std::vector<int>& params)
{
    m_last_error.clear();

    JxlEncoder* encoder = JxlEncoderCreate(nullptr);
    if (!encoder)
        return false;

    JxlEncoderFrameSettings* frame_settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
    if (!frame_settings)
    {
        JxlEncoderDestroy(encoder);
        return false;
    }

    JxlBasicInfo info;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = img.cols;
    info.ysize = img.rows;
    info.num_color_channels = img.channels();
    info.bits_per_sample = 8;

    if (JxlEncoderSetBasicInfo(encoder, &info) != JXL_ENC_SUCCESS)
    {
        JxlEncoderDestroy(encoder);
        return false;
    }

    JxlPixelFormat format = {img.channels(), JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    if (JxlEncoderAddImageFrame(frame_settings, &format, img.data, img.total() * img.elemSize()) != JXL_ENC_SUCCESS)
    {
        JxlEncoderDestroy(encoder);
        return false;
    }

    JxlEncoderCloseInput(encoder);

    std::vector<uint8_t> compressed;
    compressed.resize(1 << 20);

    uint8_t* next_out = compressed.data();
    size_t avail_out = compressed.size();

    JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
    while (status == JXL_ENC_NEED_MORE_OUTPUT)
    {
        status = JxlEncoderProcessOutput(encoder, &next_out, &avail_out);
        if (status == JXL_ENC_NEED_MORE_OUTPUT)
        {
            size_t offset = next_out - compressed.data();
            compressed.resize(compressed.size() * 2);
            next_out = compressed.data() + offset;
            avail_out = compressed.size() - offset;
        }
    }

    if (status != JXL_ENC_SUCCESS)
    {
        JxlEncoderDestroy(encoder);
        return false;
    }

    compressed.resize(next_out - compressed.data());

    if (!m_buf)
    {
        FILE* f = fopen(m_filename.c_str(), "wb");
        if (!f)
        {
            JxlEncoderDestroy(encoder);
            return false;
        }

        fwrite(compressed.data(), 1, compressed.size(), f);
        fclose(f);
    }
    // else
    // {
    //     m_buf.create(1, compressed.size(), CV_8UC1);
    //     memcpy(m_buf.data, compressed.data(), compressed.size());
    // }

    JxlEncoderDestroy(encoder);
    return true;
}

}

#endif

/* End of file. */
