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
namespace gapi {
namespace streaming {

cv::gapi::GBackend backend();

}} // namespace gapi::streaming

namespace gimpl {
namespace streaming {

cv::gapi::GKernelPackage kernels();

struct GCopy final : public cv::detail::NoTag
{
    static constexpr const char* id() { return "org.opencv.streaming.copy"; }

    static GMetaArgs getOutMeta(const GMetaArgs &in_meta, const GArgs&) {
        GAPI_Assert(in_meta.size() == 1u);
        return in_meta;
    }

    template<typename T> static T on(const T& arg) {
        return cv::GKernelType<GCopy, std::function<T(T)>>::on(arg);
    }
};

} // namespace streaming

struct Copy: public cv::detail::KernelTag
{
    using API = streaming::GCopy;

    static gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public cv::gapi::streaming::IActor
    {
        public:
            explicit Actor(const cv::GCompileArgs&) {}
            virtual void run(cv::gimpl::GIslandExecutable::IInput  &in,
                             cv::gimpl::GIslandExecutable::IOutput &out) override;
    };

    static cv::gapi::streaming::IActor::Ptr create(const cv::GCompileArgs& args)
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor(args));
    }

    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; };
};

struct BGR: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GBGR;
    static gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public cv::gapi::streaming::IActor {
        public:
            explicit Actor(const cv::GCompileArgs&) {}
            virtual void run(cv::gimpl::GIslandExecutable::IInput &in,
                             cv::gimpl::GIslandExecutable::IOutput&out) override;
    };

    static cv::gapi::streaming::IActor::Ptr create(const cv::GCompileArgs& args)
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor(args));
    }
    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; };
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMINGBACKEND_HPP
