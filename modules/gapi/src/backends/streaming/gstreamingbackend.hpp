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

struct Copy: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GCopy;

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
