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
} // namespace detail

class Jpeg2KOpjDecoder CV_FINAL : public BaseImageDecoder
{
    using StreamPtr = std::unique_ptr<opj_stream_t, detail::OpjStreamDeleter>;
    using CodecPtr = std::unique_ptr<opj_codec_t, detail::OpjCodecDeleter>;
    using ImagePtr = std::unique_ptr<opj_image_t, detail::OpjImageDeleter>;

public:
    Jpeg2KOpjDecoder();
    virtual ~Jpeg2KOpjDecoder();

    ImageDecoder newDecoder() const CV_OVERRIDE;
    bool readData( Mat& img ) CV_OVERRIDE;
    bool readHeader() CV_OVERRIDE;

private:
    void setMessageHandlers();

    StreamPtr stream_{nullptr};
    CodecPtr codec_{nullptr};
    ImagePtr image_{nullptr};

    String m_errorMessage;
    OPJ_UINT32 m_maxPrec = 0;
};

} // namespace cv

#endif

#endif/*_GRFMT_OPENJPEG_H_*/
