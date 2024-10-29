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
        m_decoder.release();
    if (m_f)
        m_f.release();
    m_read_buffer = {};
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
        // Subscribe to the basic info event
        JxlDecoderStatus status = JxlDecoderSubscribeEvents(m_decoder.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE);
        if (status != JXL_DEC_SUCCESS)
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
    if (m_read_buffer.capacity() < read_buffer_size)
        m_read_buffer.resize(read_buffer_size);

    // Create image if needed
    if (m_type != -1 && pimg) {
        pimg->create(m_height, m_width, m_type);
        if (!pimg->isContinuous())
            return false;
        if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(m_decoder.get(), 
                                                        &m_format,
                                                        pimg->ptr<uint8_t>(),
                                                        pimg->total() * pimg->elemSize())) {
            return false;
        }
    }

    // Start decoding loop
    JxlDecoderStatus status = JXL_DEC_NEED_MORE_INPUT;
    do {
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
                return false;
            }
        }

        // Get the next decoder status
        status = JxlDecoderProcessInput(m_decoder.get());

        // Handle different decoder states
        switch (status) {
            case JXL_DEC_BASIC_INFO: {
                if (m_type != -1)
                    return false;
                JxlBasicInfo info;
                if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(m_decoder.get(), &info))
                    return false;
                m_width = info.xsize;
                m_height = info.ysize;
                m_format = {
                    info.num_color_channels, // num channels (BGR/RGB + Alpha)
                    JXL_TYPE_UINT8, // 8 bits per channel
                    JXL_LITTLE_ENDIAN, // endianness
                    0 // align stride to bytes
                };
                switch (info.num_color_channels) {
                case 3:
                    m_convert = cv::COLOR_RGB2BGR;
                    break;
                case 4:
                    m_convert = cv::COLOR_RGBA2BGRA;
                    break;
                default:
                    m_convert = -1;
                }
                if (info.exponent_bits_per_sample > 0) {
                    m_format.data_type = JXL_TYPE_FLOAT;
                    m_type = info.num_color_channels == 3 ? CV_32FC3 : (info.num_color_channels == 4 ? CV_32FC4 : CV_32FC1);
                } else {
                    switch (info.bits_per_sample) {
                        case 8:
                            m_type = info.num_color_channels == 3 ? CV_8UC3 : (info.num_color_channels == 4 ? CV_8UC4 : CV_8UC1);
                            break;
                        case 16:
                            m_format.data_type = JXL_TYPE_UINT16;
                            m_type = info.num_color_channels == 3 ? CV_16UC3 : (info.num_color_channels == 4 ? CV_16UC4 : CV_16UC1);
                            break;
                        default:
                            return false;
                    }
                }
                if (!pimg)
                    return true;
                break;
            }
            case JXL_DEC_FULL_IMAGE: {
                // Image is ready
                if (m_convert != -1)
                    cv::cvtColor(*pimg, *pimg, m_convert);
                break;
            }
            case JXL_DEC_ERROR: {
                close();
                return false;
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

    JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
    if (!encoder)
        return false;

    JxlThreadParallelRunnerPtr runner = JxlThreadParallelRunnerMake(
        /*memory_manager=*/nullptr,
        JxlThreadParallelRunnerDefaultNumWorkerThreads());
    if (JXL_ENC_SUCCESS != JxlEncoderSetParallelRunner(encoder.get(),
        JxlThreadParallelRunner,
        runner.get()))
        return false;

    JxlBasicInfo info;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = img.cols;
    info.ysize = img.rows;
    info.num_color_channels = img.channels();
    info.bits_per_sample = 8 * int(img.elemSize() / img.channels());
    info.exponent_bits_per_sample = img.depth() == CV_32F ? 8 : 0;
    info.uses_original_profile = JXL_FALSE;
    if (JxlEncoderSetBasicInfo(encoder.get(), &info) != JXL_ENC_SUCCESS)
        return false;

    JxlDataType type = JXL_TYPE_UINT8;
    if (img.depth() == CV_32F)
        type = JXL_TYPE_FLOAT;
    else if (img.depth() == CV_16U)
        type = JXL_TYPE_UINT16;
    JxlPixelFormat format = {(uint32_t)img.channels(), type, JXL_NATIVE_ENDIAN, 0};
    JxlColorEncoding color_encoding = {};
    JXL_BOOL is_gray = TO_JXL_BOOL(format.num_channels < 3);
    JxlColorEncodingSetToSRGB(&color_encoding, is_gray);
    if (JXL_ENC_SUCCESS != JxlEncoderSetColorEncoding(encoder.get(), &color_encoding))
        return false;

    Mat image;
    switch (info.num_color_channels) {
    case 3:
        cv::cvtColor(img, image, cv::COLOR_BGR2RGB);
        break;
    case 4:
        cv::cvtColor(img, image, cv::COLOR_BGRA2RGBA);
        break;
    default:
        image = img;
    }
    if (!image.isContinuous())
        return false;

    JxlEncoderFrameSettings* frame_settings = JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
    // set frame settings from params if available
    for( size_t i = 0; i < params.size(); i += 2 )
    {
        if( params[i] == IMWRITE_JPEG_QUALITY )
        {
            int quality = params[i+1];
            quality = MIN(MAX(quality, 0), 100);
            const float distance = JxlEncoderDistanceFromQuality(quality);
            JxlEncoderSetFrameDistance(frame_settings, distance);
            if (distance == 0)
                JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE);
        }
        if( params[i] == IMWRITE_JPEGXL_DISTANCE )
        {
            int distance = params[i+1];
            distance = MIN(MAX(distance, 0), 25);
            JxlEncoderSetFrameDistance(frame_settings, distance);
            if (distance == 0)
                JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE);
        }
        if( params[i] == IMWRITE_JPEGXL_EFFORT )
        {
            int effort = params[i+1];
            effort = MIN(MAX(effort, 1), 10);
            JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, effort);
        }
        if( params[i] == IMWRITE_JPEGXL_DECODING_SPEED )
        {
            int speed = params[i+1];
            speed = MIN(MAX(speed, 0), 4);
            JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_DECODING_SPEED, speed);
        }
    }
    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddImageFrame(frame_settings, &format,
            static_cast<const void*>(image.ptr<uint8_t>()),
            image.total() * image.elemSize())) {
        return false;
    }
    JxlEncoderCloseInput(encoder.get());

    const size_t buffer_size = 16384;  // 16KB chunks
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(m_filename.c_str(), "wb"), fclose);
    std::vector<uint8_t> compressed(buffer_size);
    JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
    while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
        uint8_t* next_out = compressed.data();
        size_t avail_out = buffer_size;
        process_result = JxlEncoderProcessOutput(encoder.get(), &next_out, &avail_out);
        if (JXL_ENC_ERROR == process_result)
            return false;
        const size_t offset = next_out - compressed.data();
        if (offset != fwrite(compressed.data(), 1, offset, f.get()))
            return false;
    }
    return true;
}

}

#endif

/* End of file. */
