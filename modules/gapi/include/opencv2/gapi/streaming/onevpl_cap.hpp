// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_CAP_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_CAP_HPP

#ifdef HAVE_ONEVPL

#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif

#include <vpl/mfx.h>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/meta.hpp>

namespace cv {
namespace gapi {
namespace wip {

class GAPI_EXPORTS OneVPLCapture : public IStreamSource
{
public:

    struct IPriv;
 
    explicit OneVPLCapture(const std::string& filepath);
    ~OneVPLCapture() override;

    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;

private:
    std::unique_ptr<IPriv> m_priv;
};

template<class... Args>
GAPI_EXPORTS_W cv::Ptr<IStreamSource> inline make_vpl_src(Args&&... args)
{
    return make_src<OneVPLCapture>(std::forward<Args>(args)...);
}

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_ONEVPL_CAP_HPP
