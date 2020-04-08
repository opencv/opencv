// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Stefan Brüns <stefan.bruens@rwth-aachen.de>

#ifndef _GRFMT_OPENJPEG_H_
#define _GRFMT_OPENJPEG_H_

#ifdef HAVE_OPENJPEG

#include "grfmt_base.hpp"
#include <openjpeg.h>

namespace cv {
namespace detail {
struct OpjStreamDeleter
{
    void operator()(opj_stream_t* stream) const
    {
        opj_stream_destroy(stream);
    }
};

struct OpjCodecDeleter
{
    void operator()(opj_codec_t* codec) const
    {
        opj_destroy_codec(codec);
    }
};

struct OpjImageDeleter
{
    void operator()(opj_image_t* image) const
    {
        opj_image_destroy(image);
    }
};

struct OpjMemoryBuffer {
    OPJ_BYTE* pos{nullptr};
    OPJ_BYTE* begin{nullptr};
    OPJ_SIZE_T length{0};

    OpjMemoryBuffer() = default;

    explicit OpjMemoryBuffer(cv::Mat& mat)
        : pos{ mat.ptr() }, begin{ mat.ptr() }, length{ mat.rows * mat.cols * mat.elemSize() }
    {
    }

    OPJ_SIZE_T availableBytes() const CV_NOEXCEPT {
        return begin + length - pos;
    }
};

using StreamPtr = std::unique_ptr<opj_stream_t, detail::OpjStreamDeleter>;
using CodecPtr = std::unique_ptr<opj_codec_t, detail::OpjCodecDeleter>;
using ImagePtr = std::unique_ptr<opj_image_t, detail::OpjImageDeleter>;

} // namespace detail

class Jpeg2KOpjDecoder CV_FINAL : public BaseImageDecoder
{
public:
    Jpeg2KOpjDecoder();
     ~Jpeg2KOpjDecoder() CV_OVERRIDE = default;

    ImageDecoder newDecoder() const CV_OVERRIDE;
    bool readData( Mat& img ) CV_OVERRIDE;
    bool readHeader() CV_OVERRIDE;

private:
    detail::StreamPtr stream_{nullptr};
    detail::CodecPtr codec_{nullptr};
    detail::ImagePtr image_{nullptr};

    detail::OpjMemoryBuffer opjBuf_;

    OPJ_UINT32 m_maxPrec = 0;
};

class Jpeg2KOpjEncoder CV_FINAL : public BaseImageEncoder
{
public:
    Jpeg2KOpjEncoder();
    ~Jpeg2KOpjEncoder() CV_OVERRIDE = default;

    bool isFormatSupported( int depth ) const CV_OVERRIDE;
    bool write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    ImageEncoder newEncoder() const CV_OVERRIDE;
};

} //namespace cv

#endif

#endif/*_GRFMT_OPENJPEG_H_*/
