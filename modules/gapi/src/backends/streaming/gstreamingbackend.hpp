// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_GSTREAMINGBACKEND_HPP
#define OPENCV_GAPI_GSTREAMINGBACKEND_HPP

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/streaming/format.hpp>
#include "gstreamingkernel.hpp"

namespace cv {
namespace gimpl {

struct RMatMediaBGRAdapter final: public cv::RMat::Adapter
{
    RMatMediaBGRAdapter(cv::MediaFrame frame) : m_frame(frame) { };

    virtual cv::RMat::View access(cv::RMat::Access a) override
    {
        auto view = m_frame.access(a == cv::RMat::Access::W ? cv::MediaFrame::Access::W
                                                            : cv::MediaFrame::Access::R);
        auto ptr = reinterpret_cast<uchar*>(view.ptr[0]);
        auto stride = view.stride[0];

        std::shared_ptr<cv::MediaFrame::View> view_ptr =
            std::make_shared<cv::MediaFrame::View>(std::move(view));
        auto callback = [view_ptr]() mutable { view_ptr.reset(); };

        return cv::RMat::View(desc(), ptr, stride, callback);
    }

    virtual cv::GMatDesc desc() const override
    {
        const auto& desc = m_frame.desc();
        GAPI_Assert(desc.fmt == cv::MediaFormat::BGR);
        return cv::GMatDesc{CV_8U, 3, desc.size};
    }

    cv::MediaFrame m_frame;
};

struct Copy: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GCopy;

    static gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public cv::gapi::streaming::IActor
    {
        public:
            explicit Actor() {}
            virtual void run(cv::gimpl::GIslandExecutable::IInput  &in,
                             cv::gimpl::GIslandExecutable::IOutput &out) override;
    };

    static cv::gapi::streaming::IActor::Ptr create()
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor());
    }

    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; };
};

struct BGR: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GBGR;
    static gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public cv::gapi::streaming::IActor {
        public:
            explicit Actor() {}
            virtual void run(cv::gimpl::GIslandExecutable::IInput &in,
                             cv::gimpl::GIslandExecutable::IOutput&out) override;
    };

    static cv::gapi::streaming::IActor::Ptr create()
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor());
    }
    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; };
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMINGBACKEND_HPP
