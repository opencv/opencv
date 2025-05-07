// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_SPNG

/****************************************************************************************\
    This part of the file implements PNG codec on base of libspng library,
    in particular, this code is based on example.c from libspng
    (see 3rdparty/libspng/LICENSE for copyright notice)
\****************************************************************************************/

#ifndef _LFS64_LARGEFILE
#define _LFS64_LARGEFILE 0
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 0
#endif

#include <spng.h>
#include <zlib.h>

#include "grfmt_spng.hpp"

/*
 * libspng does not support RGB -> Gray conversion. In order to decode colorful images as grayscale
 * we need conversion functions. In the previous png implementation(grfmt_png), the author was set
 * to particular values for rgb coefficients. OpenCV icvCvt_BGR2Gray function values does not match
 * with these values. (png_set_rgb_to_gray( png_ptr, 1, 0.299, 0.587 );) For this codec implementation,
 * slightly modified versions are implemented in the below of this page.
*/
void spngCvt_BGRA2Gray_8u_CnC1R(const uchar *bgr, int bgr_step,
                                uchar *gray, int gray_step,
                                cv::Size size, int ncn, int _swap_rb);

void spngCvt_BGRA2Gray_16u_CnC1R(const ushort *bgr, int bgr_step,
                                 ushort *gray, int gray_step,
                                 cv::Size size, int ncn, int _swap_rb);

void spngCvt_BGRA2Gray_16u28u_CnC1R(const ushort *bgr, int bgr_step,
                                    uchar *gray, int gray_step,
                                    cv::Size size, int ncn, int _swap_rb);

namespace cv
{

/////////////////////// SPngDecoder ///////////////////

SPngDecoder::SPngDecoder()
{
    m_signature = "\x89\x50\x4e\x47\xd\xa\x1a\xa";
    m_color_type = 0;
    m_ctx = 0;
    m_f = 0;
    m_buf_supported = true;
    m_buf_pos = 0;
    m_bit_depth = 0;
}

SPngDecoder::~SPngDecoder()
{
    close();
}

ImageDecoder SPngDecoder::newDecoder() const
{
    return makePtr<SPngDecoder>();
}

void SPngDecoder::close()
{
    if (m_f)
    {
        fclose(m_f);
        m_f = 0;
    }

    if (m_ctx)
    {
        struct spng_ctx *ctx = (struct spng_ctx *)m_ctx;
        spng_ctx_free(ctx);
        m_ctx = 0;
    }
}

int SPngDecoder::readDataFromBuf(void *sp_ctx, void *user, void *dst, size_t size)
{
    /*
     * typedef int spng_read_fn(spng_ctx *ctx, void *user, void *dest, size_t length)
     *   Type definition for callback passed to spng_set_png_stream() for decoders.
     *   A read callback function should copy length bytes to dest and return 0 or SPNG_IO_EOF/SPNG_IO_ERROR on error.
     */
    CV_UNUSED(sp_ctx);
    SPngDecoder *decoder = (SPngDecoder *)(user);
    CV_Assert(decoder);

    const Mat &buf = decoder->m_buf;
    if (decoder->m_buf_pos + size > buf.cols * buf.rows * buf.elemSize())
    {
        return SPNG_IO_ERROR;
    }
    memcpy(dst, decoder->m_buf.ptr() + decoder->m_buf_pos, size);
    decoder->m_buf_pos += size;

    return 0;
}

bool SPngDecoder::readHeader()
{
    bool result = false;
    close();

    spng_ctx *ctx = spng_ctx_new(SPNG_CTX_IGNORE_ADLER32);

    if (!ctx)
    {
        spng_ctx_free(ctx);
        return false;
    }

    m_ctx = ctx;

    if (!m_buf.empty())
        spng_set_png_stream((struct spng_ctx *)m_ctx, (spng_rw_fn *)readDataFromBuf, this);
    else
    {
        m_f = fopen(m_filename.c_str(), "rb");
        if (m_f)
        {
            spng_set_png_file(ctx, m_f);
        }
    }

    if (!m_buf.empty() || m_f)
    {
        struct spng_ihdr ihdr;

        if (spng_get_ihdr(ctx, &ihdr) == SPNG_OK)
        {
            m_width = static_cast<int>(ihdr.width);
            m_height = static_cast<int>(ihdr.height);
            m_color_type = ihdr.color_type;
            m_bit_depth = ihdr.bit_depth;

            int num_trans;
            switch (ihdr.color_type)
            {
            case SPNG_COLOR_TYPE_TRUECOLOR:
            case SPNG_COLOR_TYPE_INDEXED:
                struct spng_trns trns;
                num_trans = !spng_get_trns(ctx, &trns);
                if (num_trans > 0)
                    m_type = CV_8UC4;
                else
                    m_type = CV_8UC3;
                break;
            case SPNG_COLOR_TYPE_GRAYSCALE_ALPHA:
            case SPNG_COLOR_TYPE_TRUECOLOR_ALPHA:
                m_type = CV_8UC4;
                break;
            default:
                m_type = CV_8UC1;
            }
            if (ihdr.bit_depth == 16)
                m_type = CV_MAKETYPE(CV_16U, CV_MAT_CN(m_type));
            result = true;
        }
    }

    return result;
}

bool SPngDecoder::readData(Mat &img)
{
    bool result = false;

    if (m_ctx && m_width && m_height)
    {
        struct spng_ctx* png_ptr = (struct spng_ctx*)m_ctx;
        bool color = img.channels() > 1;
        int fmt = img.channels() == 4 ? m_bit_depth == 16 ? SPNG_FMT_RGBA16 : SPNG_FMT_RGBA8 : SPNG_FMT_PNG;
        int decode_flags = img.channels() == 4 ? SPNG_DECODE_TRNS : 0;

        if (img.type() == CV_8UC3)
        {
            fmt = SPNG_FMT_RGB8;
        }
        else if (img.channels() == 1)
        {
            if (m_color_type == SPNG_COLOR_TYPE_GRAYSCALE && m_bit_depth <= 8)
                fmt = SPNG_FMT_G8;
            else
                fmt = img.depth() == CV_16U ? SPNG_FMT_RGBA16 : SPNG_FMT_RGB8;
        }

        if (fmt == SPNG_FMT_PNG && m_bit_depth == 16 && m_color_type >= SPNG_COLOR_TYPE_GRAYSCALE_ALPHA)
        {
            Mat tmp(m_height, m_width, CV_16UC4);
            if (SPNG_OK != spng_decode_image(png_ptr, tmp.data, tmp.total() * tmp.elemSize(), SPNG_FMT_RGBA16, 0))
                return false;
            cvtColor(tmp, img, m_use_rgb ? COLOR_RGBA2RGB : COLOR_RGBA2BGR);
            return true;
        }

        struct spng_ihdr ihdr;
        spng_get_ihdr(png_ptr, &ihdr);

        size_t image_width, image_size = 0;
        int ret = spng_decoded_image_size(png_ptr, fmt, &image_size);

        if (ret == SPNG_OK)
        {
            image_width = image_size / m_height;

            if (!color && fmt == SPNG_FMT_RGB8 && m_bit_depth == 16 && (m_color_type == SPNG_COLOR_TYPE_TRUECOLOR || m_color_type == SPNG_COLOR_TYPE_TRUECOLOR_ALPHA))
            {
                Mat tmp(m_height, m_width, CV_16UC4);
                if (SPNG_OK != spng_decode_image(png_ptr, tmp.data, tmp.total() * tmp.elemSize(), SPNG_FMT_RGBA16, 0))
                    return false;
                spngCvt_BGRA2Gray_16u28u_CnC1R(reinterpret_cast<const ushort*>(tmp.data), (int)tmp.step1(),
                    img.data, (int)img.step1(), Size(m_width, m_height), 4, 2);
                return true;
            }

            if (!color && ihdr.interlace_method && (fmt == SPNG_FMT_RGB8 || fmt == SPNG_FMT_RGBA16))
            {
                if (fmt == SPNG_FMT_RGBA16)
                {
                    Mat tmp(m_height, m_width, CV_16UC4);
                    if (SPNG_OK != spng_decode_image(png_ptr, tmp.data, tmp.total() * tmp.elemSize(), fmt, 0))
                        return false;
                    spngCvt_BGRA2Gray_16u_CnC1R(reinterpret_cast<const ushort*>(tmp.data), (int)tmp.step1(),
                        reinterpret_cast<ushort*>(img.data), (int)img.step1(), Size(m_width, m_height), 4, 2);
                    return true;
                }
                else
                {
                    Mat tmp(m_height, m_width, CV_8UC3);
                    if (SPNG_OK != spng_decode_image(png_ptr, tmp.data, image_size, fmt, 0))
                        return false;
                    spngCvt_BGRA2Gray_8u_CnC1R(tmp.data, (int)tmp.step1(), img.data, (int)img.step1(), Size(m_width, m_height), 3, 2);
                    return true;
                }
            }

            if (fmt == SPNG_FMT_PNG && img.elemSize() * m_width / 3 == image_width)
            {
                Mat tmp(m_height, m_width, CV_16U);
                if (SPNG_OK != spng_decode_image(png_ptr, tmp.data, image_size, SPNG_FMT_PNG, 0))
                    return false;
                cvtColor(tmp, img, COLOR_GRAY2BGR);
                return true;
            }

            ret = spng_decode_image(png_ptr, nullptr, 0, fmt, SPNG_DECODE_PROGRESSIVE | decode_flags);
            if (ret == SPNG_OK)
            {
                struct spng_row_info row_info{};

                // If user wants to read image as grayscale(IMREAD_GRAYSCALE), but image format is not
                // decode image then convert to grayscale
                if (!color && (fmt == SPNG_FMT_RGB8 || fmt == SPNG_FMT_RGBA8 || fmt == SPNG_FMT_RGBA16))
                {
                    AutoBuffer<unsigned char> buffer;
                    buffer.allocate(image_width);
                    if (fmt == SPNG_FMT_RGB8)
                    {
                        do
                        {
                            ret = spng_get_row_info(png_ptr, &row_info);
                            if (ret)
                                break;

                            ret = spng_decode_row(png_ptr, buffer.data(), image_width);
                            spngCvt_BGRA2Gray_8u_CnC1R(buffer.data(), 0, img.data + row_info.row_num * img.step, 0, Size(m_width, 1), 3, 2);
                        } while (ret == SPNG_OK);
                    }
                    else if (fmt == SPNG_FMT_RGBA8)
                    {
                        do
                        {
                            ret = spng_get_row_info(png_ptr, &row_info);
                            if (ret)
                                break;

                            ret = spng_decode_row(png_ptr, buffer.data(), image_width);
                            spngCvt_BGRA2Gray_8u_CnC1R(buffer.data(), 0, img.data + row_info.row_num * img.step, 0, Size(m_width, 1), 4, 2);
                        } while (ret == SPNG_OK);
                    }
                    else if (fmt == SPNG_FMT_RGBA16)
                    {
                        do
                        {
                            ret = spng_get_row_info(png_ptr, &row_info);
                            if (ret)
                                break;

                            ret = spng_decode_row(png_ptr, buffer.data(), image_width);
                            spngCvt_BGRA2Gray_16u_CnC1R(
                                reinterpret_cast<const ushort*>(buffer.data()), 0,
                                reinterpret_cast<ushort*>(img.data + row_info.row_num * img.step),
                                0, Size(m_width, 1), 4, 2);
                        } while (ret == SPNG_OK);
                    }
                }
                else if (color)
                { // RGB -> BGR, convert row by row if png is non-interlaced, otherwise convert image as one
                    int step = m_width * img.channels();
                    AutoBuffer<uchar *> _buffer(m_height);
                    uchar **buffer = _buffer.data();
                    for (int y = 0; y < m_height; y++)
                    {
                        buffer[y] = img.data + y * img.step;
                    }
                    if (img.channels() == 4 && m_bit_depth == 16)
                    {
                        do
                        {
                            ret = spng_get_row_info(png_ptr, &row_info);
                            if (ret)
                                break;

                            ret = spng_decode_row(png_ptr, buffer[row_info.row_num], image_width);
                            if (ihdr.interlace_method == 0 && !m_use_rgb)
                            {
                                icvCvt_RGBA2BGRA_16u_C4R(reinterpret_cast<const ushort*>(buffer[row_info.row_num]), 0,
                                    reinterpret_cast<ushort*>(buffer[row_info.row_num]), 0, Size(m_width, 1));
                            }
                        } while (ret == SPNG_OK);
                        if (ihdr.interlace_method && !m_use_rgb)
                        {
                            icvCvt_RGBA2BGRA_16u_C4R(reinterpret_cast<const ushort *>(img.data), step * 2, reinterpret_cast<ushort *>(img.data), step * 2, Size(m_width, m_height));
                        }
                    }
                    else if (img.channels() == 4)
                    {
                        do
                        {
                            ret = spng_get_row_info(png_ptr, &row_info);
                            if (ret)
                                break;

                            ret = spng_decode_row(png_ptr, buffer[row_info.row_num], image_width);
                            if (ihdr.interlace_method == 0 && !m_use_rgb)
                            {
                                icvCvt_RGBA2BGRA_8u_C4R(buffer[row_info.row_num], 0, buffer[row_info.row_num], 0, Size(m_width, 1));
                            }
                        } while (ret == SPNG_OK);
                        if (ihdr.interlace_method && !m_use_rgb)
                        {
                            icvCvt_RGBA2BGRA_8u_C4R(img.data, step, img.data, step, Size(m_width, m_height));
                        }
                    }
                    else if (fmt == SPNG_FMT_PNG)
                    {
                        AutoBuffer<unsigned char> bufcn4;
                        bufcn4.allocate(image_width);
                        do
                        {
                            ret = spng_get_row_info(png_ptr, &row_info);
                            if (ret)
                                break;

                            ret = spng_decode_row(png_ptr, buffer[row_info.row_num], image_width);

                            if (ihdr.interlace_method == 0 && !m_use_rgb)
                            {
                                icvCvt_RGB2BGR_16u_C3R(reinterpret_cast<const ushort*>(buffer[row_info.row_num]), 0,
                                    reinterpret_cast<ushort*>(buffer[row_info.row_num]), 0, Size(m_width, 1));
                            }
                        } while (ret == SPNG_OK);
                        if (ihdr.interlace_method && !m_use_rgb)
                        {
                            icvCvt_RGB2BGR_16u_C3R(reinterpret_cast<const ushort*>(img.data), step,
                                reinterpret_cast<ushort*>(img.data), step, Size(m_width, m_height));
                        }
                    }
                    else
                    {
                        do
                        {
                            ret = spng_get_row_info(png_ptr, &row_info);
                            if (ret)
                                break;

                            ret = spng_decode_row(png_ptr, buffer[row_info.row_num], image_width);
                            if (ihdr.interlace_method == 0 && !m_use_rgb)
                            {
                                icvCvt_RGB2BGR_8u_C3R(buffer[row_info.row_num], 0, buffer[row_info.row_num], 0, Size(m_width, 1));
                            }
                        } while (ret == SPNG_OK);
                        if (ihdr.interlace_method && !m_use_rgb)
                        {
                            icvCvt_RGB2BGR_8u_C3R(img.data, step, img.data, step, Size(m_width, m_height));
                        }
                    }
                }
                else
                    do
                    {
                        ret = spng_get_row_info(png_ptr, &row_info);
                        if (ret)
                            break;

                        ret = spng_decode_row(png_ptr, img.data + row_info.row_num * image_width, image_width);

                    } while (ret == SPNG_OK);
            }

            if (ret == SPNG_EOI)
            {
                ret = spng_decode_chunks(png_ptr);
                if(ret == SPNG_OK) result = true;
                struct spng_exif exif_s{};
                ret = spng_get_exif(png_ptr, &exif_s);
                if (ret == SPNG_OK)
                {
                    if (exif_s.data && exif_s.length > 0)
                    {
                        result = m_exif.parseExif((unsigned char *)exif_s.data, exif_s.length);
                    }
                }
            }
        }
    }

    return result;
}

/////////////////////// SPngEncoder ///////////////////

SPngEncoder::SPngEncoder()
{
    m_description = "Portable Network Graphics files (*.png)";
    m_buf_supported = true;
}

SPngEncoder::~SPngEncoder()
{
}

bool SPngEncoder::isFormatSupported(int depth) const
{
    return depth == CV_8U || depth == CV_16U;
}

ImageEncoder SPngEncoder::newEncoder() const
{
    return makePtr<SPngEncoder>();
}

int SPngEncoder::writeDataToBuf(void *ctx, void *user, void *dst_src, size_t length)
{
    CV_UNUSED(ctx);
    SPngEncoder *encoder = (SPngEncoder *)(user);
    CV_Assert(encoder && encoder->m_buf);
    size_t cursz = encoder->m_buf->size();
    encoder->m_buf->resize(cursz + length);
    memcpy(&(*encoder->m_buf)[cursz], dst_src, length);
    return 0;
}

bool SPngEncoder::write(const Mat &img, const std::vector<int> &params)
{
    spng_ctx *ctx = spng_ctx_new(SPNG_CTX_ENCODER);
    FILE *volatile f = 0;
    int width = img.cols, height = img.rows;
    int depth = img.depth(), channels = img.channels();
    volatile bool result = false;

    if (depth != CV_8U && depth != CV_16U)
        return false;

    if (ctx)
    {
        struct spng_ihdr ihdr = {};
        ihdr.height = height;
        ihdr.width = width;
        int compression_level = Z_BEST_SPEED;
        int compression_strategy = IMWRITE_PNG_STRATEGY_RLE; // Default strategy
        int filter = IMWRITE_PNG_FILTER_SUB; // Default filter
        bool isBilevel = false;
        bool set_compression_level = false;
        bool set_filter = false;

        for (size_t i = 0; i < params.size(); i += 2)
        {
            if (params[i] == IMWRITE_PNG_COMPRESSION)
            {
                compression_strategy = IMWRITE_PNG_STRATEGY_DEFAULT; // Default strategy
                compression_level = params[i + 1];
                compression_level = MIN(MAX(compression_level, 0), Z_BEST_COMPRESSION);
                set_compression_level = true;
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
            if( params[i] == IMWRITE_PNG_FILTER )
            {
                filter = params[i+1];
                set_filter = true;
            }
        }

        ihdr.bit_depth = depth == CV_8U ? isBilevel ? 1 : 8 : 16;
        ihdr.color_type = (uint8_t)(channels == 1 ? SPNG_COLOR_TYPE_GRAYSCALE : channels == 3 ? SPNG_COLOR_TYPE_TRUECOLOR
                                                                                              : SPNG_COLOR_TYPE_TRUECOLOR_ALPHA);
        ihdr.interlace_method = SPNG_INTERLACE_NONE;
        ihdr.filter_method = SPNG_FILTER_NONE;
        ihdr.compression_method = 0;
        spng_set_ihdr(ctx, &ihdr);

        if (m_buf)
        {
            spng_set_png_stream(ctx, (spng_rw_fn *)writeDataToBuf, this);
        }
        else
        {
            f = fopen(m_filename.c_str(), "wb");
            if (f)
                spng_set_png_file(ctx, f);
        }

        if (m_buf || f)
        {
            if (!set_compression_level || set_filter)
                spng_set_option(ctx, SPNG_FILTER_CHOICE, filter);
            spng_set_option(ctx, SPNG_IMG_COMPRESSION_LEVEL, compression_level);
            spng_set_option(ctx, SPNG_IMG_COMPRESSION_STRATEGY, compression_strategy);

            int ret;
            spng_encode_chunks(ctx);
            ret = spng_encode_image(ctx, nullptr, 0, SPNG_FMT_PNG, SPNG_ENCODE_PROGRESSIVE);
            if (channels > 1)
            {
                int error = SPNG_OK;
                if (ret == SPNG_OK)
                {
                    if (depth == CV_16U)
                    {
                        AutoBuffer<ushort *> buff2;
                        buff2.allocate(height);
                        for (int y = 0; y < height; y++)
                            buff2[y] = reinterpret_cast<unsigned short *>(img.data + y * img.step);

                        AutoBuffer<ushort> _buffer;
                        _buffer.allocate(width * channels * 2);
                        for (int y = 0; y < height; y++)
                        {
                            if (channels == 3)
                            {
                                icvCvt_BGR2RGB_16u_C3R(buff2[y], 0,
                                                       _buffer.data(), 0, Size(width, 1));
                            }
                            else if (channels == 4)
                            {
                                icvCvt_BGRA2RGBA_16u_C4R(buff2[y], 0,
                                                         _buffer.data(), 0, Size(width, 1));
                            }
                            error = spng_encode_row(ctx, _buffer.data(), width * channels * 2);
                            if (error)
                                break;
                        }
                    }
                    else
                    {
                        AutoBuffer<uchar *> buff;
                        buff.allocate(height);
                        for (int y = 0; y < height; y++)
                            buff[y] = img.data + y * img.step;

                        AutoBuffer<uchar> _buffer;
                        _buffer.allocate(width * channels);
                        for (int y = 0; y < height; y++)
                        {
                            if (channels == 3)
                            {
                                icvCvt_BGR2RGB_8u_C3R(buff[y], 0, _buffer.data(), 0, Size(width, 1));
                            }
                            else if (channels == 4)
                            {
                                icvCvt_BGRA2RGBA_8u_C4R(buff[y], 0, _buffer.data(), 0, Size(width, 1));
                            }
                            error = spng_encode_row(ctx, _buffer.data(), width * channels);
                            if (error)
                                break;
                        }
                    }
                    if (error == SPNG_EOI)
                    { // success
                        spng_encode_chunks(ctx);
                        ret = SPNG_OK;
                    }
                }
            }
            else
            {
                int error = SPNG_OK;
                for (int y = 0; y < height; y++)
                {
                    error = spng_encode_row(ctx, img.data + y * img.step, width * channels * (depth == CV_16U ? 2 : 1));
                    if (error)
                        break;
                }
                if (error == SPNG_EOI)
                { // success
                    spng_encode_chunks(ctx);
                    ret = SPNG_OK;
                }
            }
            if (ret == SPNG_OK)
                result = true;
        }
    }

    spng_ctx_free(ctx);
    if (f)
        fclose((FILE *)f);

    return result;
}

}

void spngCvt_BGRA2Gray_8u_CnC1R(const uchar *bgr, int bgr_step,
                                uchar *gray, int gray_step,
                                cv::Size size, int ncn, int _swap_rb)
{
    int i;
    for (; size.height--; gray += gray_step)
    {
        int cBGR0 = 3737;
        int cBGR1 = 19234;
        int cBGR2 = 9797;

        if (_swap_rb)
            std::swap(cBGR0, cBGR2);
        for (i = 0; i < size.width; i++, bgr += ncn)
        {
            if (bgr[0] != bgr[1] || bgr[0] != bgr[2])
            {
                gray[i] = (uchar)((cBGR0 * bgr[0] + cBGR1 * bgr[1] + cBGR2 * bgr[2]) >> 15);
            }
            else
            {
                gray[i] = bgr[0];
            }
        }

        bgr += bgr_step - size.width * ncn;
    }
}

void spngCvt_BGRA2Gray_16u_CnC1R(const ushort *bgr, int bgr_step,
                                 ushort *gray, int gray_step,
                                 cv::Size size, int ncn, int _swap_rb)
{
    for (; size.height--; gray += gray_step)
    {
        int cBGR0 = 3737;
        int cBGR1 = 19234;
        int cBGR2 = 9797;

        if (_swap_rb)
            std::swap(cBGR0, cBGR2);
        for (int i = 0; i < size.width; i++, bgr += ncn)
        {
            if (bgr[0] != bgr[1] || bgr[0] != bgr[2])
            {
                gray[i] = (ushort)((cBGR0 * bgr[0] + cBGR1 * bgr[1] + cBGR2 * bgr[2] + 16384) >> 15);
            }
            else
            {
                gray[i] = bgr[0];
            }
        }

        bgr += bgr_step - size.width * ncn;
    }
}

void spngCvt_BGRA2Gray_16u28u_CnC1R(const ushort *bgr, int bgr_step,
                                    uchar *gray, int gray_step,
                                    cv::Size size, int ncn, int _swap_rb)
{
    int cBGR0 = 3737;
    int cBGR1 = 19234;
    int cBGR2 = 9797;
    if (_swap_rb)
        std::swap(cBGR0, cBGR2);

    for (; size.height--; gray += gray_step)
    {
        for (int i = 0; i < size.width; i++, bgr += ncn)
        {
            gray[i] = static_cast<uchar>(((cBGR0 * bgr[0] + cBGR1 * bgr[1] + cBGR2 * bgr[2] + 16384) >> 15) >> 8);
        }

        bgr += bgr_step - size.width * ncn;
    }
}

#endif

/* End of file. */
