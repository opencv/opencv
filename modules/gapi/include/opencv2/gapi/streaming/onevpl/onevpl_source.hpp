// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/source.hpp>

namespace cv {
namespace gapi {
namespace wip {

class GAPI_EXPORTS OneVPLSource : public IStreamSource
{
public:
    struct Priv;

    explicit OneVPLSource(const std::string& filePath);
    ~OneVPLSource() override;

    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;

private:
    explicit OneVPLSource(std::unique_ptr<Priv>&& impl);
    std::unique_ptr<Priv> m_priv;
};

template<class... Args>
GAPI_EXPORTS_W cv::Ptr<IStreamSource> inline make_vpl_src(const std::string& filePath, Args&&... args)
{
    return make_src<OneVPLSource>(filePath, std::forward<Args>(args)...);
}

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP
