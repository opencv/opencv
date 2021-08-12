// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_PRIV_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_PRIV_HPP

#include <stdio.h>

#include <memory>
#include <string>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/oneVPL/onevpl_source.hpp>

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif // MFX_VERSION

#include <vpl/mfx.h>

#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

struct OneVPLSource::Priv
{
    explicit Priv(const std::string& file_path);
    ~Priv();

    bool pull(cv::gapi::wip::Data& data);
    GMetaArg descr_of() const;
private:
    Priv();
    mfxLoader mfx_handle;
    bool description_is_valid;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#else // HAVE_ONEVPL
struct OneVPLSource::Priv final
{
    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;
};
#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_PRIV_HPP
