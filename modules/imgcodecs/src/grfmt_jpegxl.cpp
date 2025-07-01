// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "grfmt_jpegxl.hpp"

#ifdef HAVE_JPEGXL

#include <jxl/encode_cxx.h>
#include <jxl/version.h>
#include <opencv2/core/utils/logger.hpp>

namespace cv
{
// Callback functions for JpegXLDecoder
static void cbRGBtoBGR_8U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBAtoBGRA_8U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBtoBGR_16U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBAtoBGRA_16U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBtoBGR_32F(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBAtoBGRA_32F(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBtoGRAY_8U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBAtoGRAY_8U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBtoGRAY_16U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBAtoGRAY_16U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBtoGRAY_32F(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);
static void cbRGBAtoGRAY_32F(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels);

/////////////////////// JpegXLDecoder ///////////////////

JpegXLDecoder::JpegXLDecoder() : m_f(nullptr, &fclose),
                                 m_read_buffer(16384,0) // 16KB chunks
{
    m_signature = "\xFF\x0A";
    m_decoder = nullptr;
    m_buf_supported = true;
    m_type = -1;
    m_status = JXL_DEC_NEED_MORE_INPUT;
    m_is_mbuf_set = false;
}

JpegXLDecoder::~JpegXLDecoder()
{
    close();
}

void JpegXLDecoder::close()
{
    if (m_decoder)
        m_decoder.reset();
    if (m_f)
        m_f.reset();
    m_read_buffer = {};
    m_width = m_height = 0;
    m_type = -1;
    m_status = JXL_DEC_NEED_MORE_INPUT;
    m_is_mbuf_set = false;
}

// see https://github.com/libjxl/libjxl/blob/v0.10.0/doc/format_overview.md
size_t JpegXLDecoder::signatureLength() const
{
    return 12; // For an ISOBMFF-based container
}

bool JpegXLDecoder::checkSignature( const String& signature ) const
{
    // A "naked" codestream.
    if (
        ( signature.size() >= 2 ) &&
        ( memcmp( signature.c_str(), "\xFF\x0A", 2 ) == 0 )
    )
    {
        return true;
    }

    // An ISOBMFF-based container.
    // 0x0000_000C_4A58_4C20_0D0A_870A.
    if (
        ( signature.size() >= 12 ) &&
        ( memcmp( signature.c_str(), "\x00\x00\x00\x0C\x4A\x58\x4C\x20\x0D\x0A\x87\x0A", 12 ) == 0 )
    )
    {
        return true;
    }

    return false;
}

ImageDecoder JpegXLDecoder::newDecoder() const
{
    return makePtr<JpegXLDecoder>();
}

bool JpegXLDecoder::readHeader()
{
    if (m_buf.empty()) {
        // Open file
        if (!m_f) {
            m_f.reset(fopen(m_filename.c_str(), "rb"));
            if (!m_f) {
                return false;
            }
        }
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
        m_parallel_runner = JxlThreadParallelRunnerMake(nullptr, cv::getNumThreads());
        if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(m_decoder.get(),
                                                            JxlThreadParallelRunner,
                                                            m_parallel_runner.get())) {
            return false;
        }
    }

    // Reset to read header data stream
    m_is_mbuf_set = false;

    return read();
}

bool JpegXLDecoder::readData(Mat& img)
{
    if (!m_decoder || m_width == 0 || m_height == 0 || m_type == -1)
        return false;

    // Prepare to decode image
    const uint32_t scn = CV_MAT_CN(m_type);        // from image
    const uint32_t dcn = (uint32_t)img.channels(); // to OpenCV
    const int depth = CV_MAT_DEPTH(img.type());
    JxlImageOutCallback cbFunc = nullptr;

    CV_CheckChannels(scn, (scn == 1 || scn == 3 || scn == 4), "Unsupported src channels");
    CV_CheckChannels(dcn, (dcn == 1 || dcn == 3 || dcn == 4), "Unsupported dst channels");
    CV_CheckDepth(depth, (depth == CV_8U || depth == CV_16U || depth == CV_32F), "Unsupported depth");

    m_format = {
        dcn,
        JXL_TYPE_UINT8, // (temporary)
        JXL_NATIVE_ENDIAN, // endianness
        0 // align stride to bytes
    };
    switch (depth) {
        case CV_8U:  m_format.data_type = JXL_TYPE_UINT8; break;
        case CV_16U: m_format.data_type = JXL_TYPE_UINT16; break;
        case CV_32F: m_format.data_type = JXL_TYPE_FLOAT; break;
        default: break;
    }
    // libjxl cannot read to BGR pixel order directly.
    // So we have to use callback function to convert from RGB(A) to BGR(A).
    if (!m_use_rgb) {
        switch (dcn) {
            case 1:  break;
            case 3:  cbFunc = (depth == CV_32F)? cbRGBtoBGR_32F:   (depth == CV_16U)? cbRGBtoBGR_16U:   cbRGBtoBGR_8U; break;
            case 4:  cbFunc = (depth == CV_32F)? cbRGBAtoBGRA_32F: (depth == CV_16U)? cbRGBAtoBGRA_16U: cbRGBAtoBGRA_8U; break;
            default: break;
        }
    }
    // libjxl cannot convert from color image to gray image directly.
    // So we have to use callback function to convert from RGB(A) to GRAY.
    if( (scn >= 3) && (dcn == 1) )
    {
        m_format.num_channels = scn;
        switch (scn) {
            case 3:  cbFunc = (depth == CV_32F)? cbRGBtoGRAY_32F:  (depth == CV_16U)? cbRGBtoGRAY_16U:  cbRGBtoGRAY_8U; break;
            case 4:  cbFunc = (depth == CV_32F)? cbRGBAtoGRAY_32F: (depth == CV_16U)? cbRGBAtoGRAY_16U: cbRGBAtoGRAY_8U; break;
            default: break;
        }
    }
    if(cbFunc != nullptr)
    {
        if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutCallback(m_decoder.get(),
                                                             &m_format,
                                                             cbFunc,
                                                             static_cast<void*>(&img)))
        {
            return false;
        }
    }else{
        if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(m_decoder.get(),
                                                           &m_format,
                                                           img.ptr<uint8_t>(),
                                                           img.total() * img.elemSize()))
        {
            return false;
        }
    }

    return read();
}

// Common reading routine for readHeader() and readBody()
bool JpegXLDecoder::read()
{
    // Start decoding loop
    do {
        // Check if we need more input
        if (m_status == JXL_DEC_NEED_MORE_INPUT) {
            uint8_t* data_ptr = nullptr;
            size_t   data_len = 0;

            if( !m_buf.empty() ) {
                // When data source in on memory
                if (m_is_mbuf_set) {
                    // We expect m_buf contains whole JpegXL data stream.
                    // If it had been truncated, m_status will be JXL_DEC_NEED_MORE_INPUT again.
                    CV_LOG_WARNING(NULL, "Truncated JXL data in memory");
                    return false;
                }
                data_ptr = m_buf.ptr();
                data_len = m_buf.total();
                m_is_mbuf_set = true;
            }
            else {
                // When data source is on file
                // Release input buffer if it had been set already. If not, there are no errors.
                size_t remaining = JxlDecoderReleaseInput(m_decoder.get());
                // Move any remaining bytes to the beginning
                if (remaining > 0)
                    memmove(m_read_buffer.data(), m_read_buffer.data() + m_read_buffer.size() - remaining, remaining);
                // Read more data from file
                size_t bytes_read = fread(m_read_buffer.data() + remaining,
                                          1, m_read_buffer.size() - remaining, m_f.get());
                if (bytes_read == 0) {
                    if (ferror(m_f.get())) {
                        CV_LOG_WARNING(NULL, "Error reading input file");
                        return false;
                    }
                    // If we reached EOF but decoder needs more input, file is truncated
                    if (m_status == JXL_DEC_NEED_MORE_INPUT) {
                        CV_LOG_WARNING(NULL, "Truncated JXL file");
                        return false;
                    }
                }
                data_ptr = m_read_buffer.data();
                data_len = bytes_read + remaining;
            }

            // Set input buffer
            // It must be kept until calling JxlDecoderReleaseInput() or m_decoder.reset().
            if (JXL_DEC_SUCCESS != JxlDecoderSetInput(m_decoder.get(), data_ptr, data_len)) {
                return false;
            }
        }

        // Get the next decoder status
        m_status = JxlDecoderProcessInput(m_decoder.get());

        // Handle different decoder states
        switch (m_status) {
            case JXL_DEC_BASIC_INFO: {
                if (m_type != -1)
                    return false;

                JxlBasicInfo info;
                if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(m_decoder.get(), &info))
                    return false;

                // total channels (Color + Alpha)
                const uint32_t ncn = info.num_color_channels + info.num_extra_channels;

                m_width = info.xsize;
                m_height = info.ysize;
                int depth = (info.exponent_bits_per_sample > 0)?CV_32F:
                            (info.bits_per_sample == 16)?CV_16U:
                            (info.bits_per_sample == 8)?CV_8U: -1;
                if(depth == -1)
                {
                    return false; // Return to readHeader()
                }
                m_type = CV_MAKETYPE( depth, ncn );
                return true;
            }
            case JXL_DEC_FULL_IMAGE: {
                // Image is ready
                break;
            }
            case JXL_DEC_ERROR: {
                close();
                return false;
            }
            default:
                break;
        }
    } while (m_status != JXL_DEC_SUCCESS);

    return true;
}

// Callback functopms
static void cbRGBtoBGR_8U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    const uint8_t* src = static_cast<const uint8_t*>(pixels);

    constexpr int dstStep = 3;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint8_t* dstBase = const_cast<uint8_t*>(pDst->ptr(y));
    uint8_t* dst = dstBase + x * dstStep;

    icvCvt_RGB2BGR_8u_C3R( src, 0, dst, 0, Size(num_pixels , 1) );
}
static void cbRGBAtoBGRA_8U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    const uint8_t* src = static_cast<const uint8_t*>(pixels);

    constexpr int dstStep = 4;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint8_t* dstBase = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(pDst->ptr(y)));
    uint8_t* dst = dstBase + x * dstStep;

    icvCvt_RGBA2BGRA_8u_C4R( src, 0, dst, 0, Size(num_pixels, 1) );
}
static void cbRGBtoBGR_16U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    const uint16_t* src = static_cast<const uint16_t*>(pixels);

    constexpr int dstStep = 3;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint16_t* dstBase = const_cast<uint16_t*>(reinterpret_cast<const uint16_t*>(pDst->ptr(y)));
    uint16_t* dst = dstBase + x * dstStep;

    icvCvt_BGR2RGB_16u_C3R( src, 0, dst, 0, Size(num_pixels, 1));
}
static void cbRGBAtoBGRA_16U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    const uint16_t* src = static_cast<const uint16_t*>(pixels);

    constexpr int dstStep = 4;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint16_t* dstBase = const_cast<uint16_t*>(reinterpret_cast<const uint16_t*>(pDst->ptr(y)));
    uint16_t* dst = dstBase + x * dstStep;

    icvCvt_BGRA2RGBA_16u_C4R( src, 0, dst, 0, Size(num_pixels, 1));
}
static void cbRGBtoBGR_32F(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    constexpr int srcStep = 3;
    const uint32_t* src = static_cast<const uint32_t*>(pixels);

    constexpr int dstStep = 3;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint32_t* dstBase = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(pDst->ptr(y)));
    uint32_t* dst = dstBase + x * dstStep;

    for(size_t i = 0 ; i < num_pixels; i++)
    {
        dst[ i * dstStep + 0 ] = src[ i * srcStep + 2];
        dst[ i * dstStep + 1 ] = src[ i * srcStep + 1];
        dst[ i * dstStep + 2 ] = src[ i * srcStep + 0];
    }
}
static void cbRGBAtoBGRA_32F(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    constexpr int srcStep = 4;
    const uint32_t* src = static_cast<const uint32_t*>(pixels);

    constexpr int dstStep = 4;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint32_t* dstBase = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(pDst->ptr(y)));
    uint32_t* dst = dstBase + x * dstStep;

    for(size_t i = 0 ; i < num_pixels; i++)
    {
        dst[ i * dstStep + 0 ] = src[ i * srcStep + 2];
        dst[ i * dstStep + 1 ] = src[ i * srcStep + 1];
        dst[ i * dstStep + 2 ] = src[ i * srcStep + 0];
        dst[ i * dstStep + 3 ] = src[ i * srcStep + 3];
    }
}

static void cbRGBtoGRAY_8U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    const uint8_t* src = static_cast<const uint8_t*>(pixels);

    constexpr int dstStep = 1;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint8_t* dstBase = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(pDst->ptr(y)));
    uint8_t* dst = dstBase + x * dstStep;

    icvCvt_BGR2Gray_8u_C3C1R(src, 0, dst, 0, Size(num_pixels, 1) );
}
static void cbRGBAtoGRAY_8U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    const uint8_t* src = static_cast<const uint8_t*>(pixels);

    constexpr int dstStep = 1;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint8_t* dstBase = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(pDst->ptr(y)));
    uint8_t* dst = dstBase + x * dstStep;

    icvCvt_BGRA2Gray_8u_C4C1R(src, 0, dst, 0, Size(num_pixels, 1) );
}
static void cbRGBtoGRAY_16U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    const uint16_t* src = static_cast<const uint16_t*>(pixels);

    constexpr int dstStep = 1;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint16_t* dstBase = const_cast<uint16_t*>(reinterpret_cast<const uint16_t*>(pDst->ptr(y)));
    uint16_t* dst = dstBase + x * dstStep;

    icvCvt_BGRA2Gray_16u_CnC1R(src, 0, dst, 0, Size(num_pixels, 1), /* ncn= */ 3 );
}
static void cbRGBAtoGRAY_16U(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    const uint16_t* src = static_cast<const uint16_t*>(pixels);

    constexpr int dstStep = 1;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    uint16_t* dstBase = const_cast<uint16_t*>(reinterpret_cast<const uint16_t*>(pDst->ptr(y)));
    uint16_t* dst = dstBase + x * dstStep;

    icvCvt_BGRA2Gray_16u_CnC1R(src, 0, dst, 0, Size(num_pixels, 1), /* ncn= */ 4 );
}
static void cbRGBtoGRAY_32F(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    constexpr float cR = 0.299f;
    constexpr float cG = 0.587f;
    constexpr float cB = 1.000f - cR - cG;

    constexpr int srcStep = 3;
    const float* src = static_cast<const float*>(pixels);

    constexpr int dstStep = 1;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    float* dstBase = const_cast<float*>(reinterpret_cast<const float*>(pDst->ptr(y)));
    float* dst = dstBase + x * dstStep;

    for(size_t i = 0 ; i < num_pixels; i++)
    {
        dst[ i * dstStep ] = src[ i * srcStep + 0] * cR +
                             src[ i * srcStep + 1] * cG +
                             src[ i * srcStep + 2] * cB;
    }
}
static void cbRGBAtoGRAY_32F(void *opaque, size_t x, size_t y, size_t num_pixels, const void *pixels)
{
    constexpr float cR = 0.299f;
    constexpr float cG = 0.587f;
    constexpr float cB = 1.000f - cR - cG;

    constexpr int srcStep = 4;
    const float* src = static_cast<const float*>(pixels);

    constexpr int dstStep = 1;
    const cv::Mat *pDst = static_cast<cv::Mat*>(opaque);
    float* dstBase = const_cast<float*>(reinterpret_cast<const float*>(pDst->ptr(y)));
    float* dst = dstBase + x * dstStep;

    for(size_t i = 0 ; i < num_pixels; i++)
    {
        dst[ i * dstStep ] = src[ i * srcStep + 0] * cR +
                             src[ i * srcStep + 1] * cG +
                             src[ i * srcStep + 2] * cB;
    }
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

bool JpegXLEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_16U || depth == CV_32F;
}

bool JpegXLEncoder::write(const Mat& img, const std::vector<int>& params)
{
    m_last_error.clear();

    JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
    if (!encoder)
        return false;

    JxlThreadParallelRunnerPtr runner = JxlThreadParallelRunnerMake(
        /*memory_manager=*/nullptr, cv::getNumThreads());
    if (JXL_ENC_SUCCESS != JxlEncoderSetParallelRunner(encoder.get(),  JxlThreadParallelRunner, runner.get()))
        return false;

    CV_CheckDepth(img.depth(),
             ( img.depth() == CV_8U || img.depth() == CV_16U || img.depth() == CV_32F ),
             "JPEG XL encoder only supports CV_8U, CV_16U, CV_32F");
    CV_CheckChannels(img.channels(),
             ( img.channels() == 1 || img.channels() == 3 || img.channels() == 4) ,
             "JPEG XL encoder only supports 1, 3, 4 channels");

    WLByteStream strm;
    if( m_buf ) {
        if( !strm.open( *m_buf ) )
            return false;
    }
    else if( !strm.open( m_filename )) {
        return false;
    }

    // get distance param for JxlBasicInfo.
    float distance = -1.0; // Negative means not set
    for( size_t i = 0; i < params.size(); i += 2 )
    {
        if( params[i] == IMWRITE_JPEGXL_QUALITY )
        {
#if JPEGXL_MAJOR_VERSION > 0 || JPEGXL_MINOR_VERSION >= 10
            int quality = params[i+1];
            quality = MIN(MAX(quality, 0), 100);
            distance = JxlEncoderDistanceFromQuality(static_cast<float>(quality));
#else
            CV_LOG_ONCE_WARNING(NULL, "Quality parameter is supported with libjxl v0.10.0 or later");
#endif
        }
        if( params[i] == IMWRITE_JPEGXL_DISTANCE )
        {
            int distanceInt = params[i+1];
            distanceInt = MIN(MAX(distanceInt, 0), 25);
            distance = static_cast<float>(distanceInt);
        }
    }

    JxlBasicInfo info;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = img.cols;
    info.ysize = img.rows;
    // Lossless encoding requires uses_original_profile = true.
    info.uses_original_profile = (distance == 0.0) ? JXL_TRUE : JXL_FALSE;

    if( img.channels() == 4 )
    {
        info.num_color_channels = 3;
        info.num_extra_channels = 1;

        info.bits_per_sample =
        info.alpha_bits      = 8 * static_cast<int>(img.elemSize1());

        info.exponent_bits_per_sample =
        info.alpha_exponent_bits      = img.depth() == CV_32F ? 8 : 0;
    }else{
        info.num_color_channels = img.channels();
        info.bits_per_sample = 8 * static_cast<int>(img.elemSize1());
        info.exponent_bits_per_sample = img.depth() == CV_32F ? 8 : 0;
    }

    if (JxlEncoderSetBasicInfo(encoder.get(), &info) != JXL_ENC_SUCCESS)
        return false;

    JxlDataType type = JXL_TYPE_UINT8;
    if (img.depth() == CV_32F)
        type = JXL_TYPE_FLOAT;
    else if (img.depth() == CV_16U)
        type = JXL_TYPE_UINT16;
    JxlPixelFormat format = {(uint32_t)img.channels(), type, JXL_NATIVE_ENDIAN, 0};
    JxlColorEncoding color_encoding = {};
    JXL_BOOL is_gray(format.num_channels < 3 ? JXL_TRUE : JXL_FALSE);
    JxlColorEncodingSetToSRGB(&color_encoding, is_gray);
    if (JXL_ENC_SUCCESS != JxlEncoderSetColorEncoding(encoder.get(), &color_encoding))
        return false;

    Mat image;
    switch ( img.channels() ) {
    case 3:
        cv::cvtColor(img, image, cv::COLOR_BGR2RGB);
        break;
    case 4:
        cv::cvtColor(img, image, cv::COLOR_BGRA2RGBA);
        break;
    case 1:
    default:
        if(img.isContinuous()) {
            image = img;
        } else {
            image = img.clone(); // reconstruction as continuous image.
        }
        break;
    }
    if (!image.isContinuous())
        return false;

    JxlEncoderFrameSettings* frame_settings = JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);

    // set frame settings with distance params
    if(distance == 0.0) // lossless
    {
        if( JXL_ENC_SUCCESS != JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE) )
        {
            CV_LOG_WARNING(NULL, "Failed to call JxlEncoderSetFrameLossless()");
        }
    }
    else if(distance > 0.0) // lossy
    {
        if( JXL_ENC_SUCCESS != JxlEncoderSetFrameDistance(frame_settings, distance) )
        {
            CV_LOG_WARNING(NULL, "Failed to call JxlEncoderSetFrameDistance()");
        }
    }

    // set frame settings from params if available
    for( size_t i = 0; i < params.size(); i += 2 )
    {
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

    std::vector<uint8_t> compressed(buffer_size);
    JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
    while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
        uint8_t* next_out = compressed.data();
        size_t avail_out = buffer_size;
        process_result = JxlEncoderProcessOutput(encoder.get(), &next_out, &avail_out);
        if (JXL_ENC_ERROR == process_result)
            return false;
        const size_t write_size = buffer_size - avail_out;
        if ( strm.putBytes(compressed.data(), write_size) == false )
            return false;
    }
    return true;
}

}

#endif

/* End of file. */
