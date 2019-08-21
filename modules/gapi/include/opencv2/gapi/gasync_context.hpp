// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GASYNC_CONTEXT_HPP
#define OPENCV_GAPI_GASYNC_CONTEXT_HPP

#if !defined(GAPI_STANDALONE)
#  include <opencv2/core/cvdef.h>
#else   // Without OpenCV
#  include <opencv2/gapi/own/cvdefs.hpp>
#endif // !defined(GAPI_STANDALONE)

#include <opencv2/gapi/own/exports.hpp>

namespace cv {
namespace gapi{
namespace wip {

class GAPI_EXPORTS GAsyncContext{
    std::atomic<bool> cancelation_requested = {false};
public:
    //returns true if it was a first request to cancel the context
    bool cancel();
    bool isCanceled() const;
};

class GAPI_EXPORTS GAsyncCanceled : public std::exception {
public:
    virtual const char* what() const noexcept CV_OVERRIDE;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif //OPENCV_GAPI_GASYNC_CONTEXT_HPP
