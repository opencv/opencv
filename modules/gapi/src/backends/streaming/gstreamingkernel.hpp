// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_GSTREAMINGKERNEL_HPP
#define OPENCV_GAPI_GSTREAMINGKERNEL_HPP

#include "compiler/gislandmodel.hpp"

namespace cv {
namespace gapi {
namespace streaming {

GAPI_EXPORTS cv::gapi::GBackend backend();

class IActor {
public:
    using Ptr = std::shared_ptr<IActor>;

    virtual void run(cv::gimpl::GIslandExecutable::IInput  &in,
                     cv::gimpl::GIslandExecutable::IOutput &out) = 0;

    virtual ~IActor() = default;
};

using CreateActorFunction = std::function<IActor::Ptr(const cv::GCompileArgs&)>;
struct GStreamingKernel
{
    CreateActorFunction createActorFunction;
};

} // namespace streaming
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMINGKERNEL_HPP
