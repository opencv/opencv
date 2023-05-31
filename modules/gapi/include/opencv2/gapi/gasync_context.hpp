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

/**
 * @brief This namespace contains experimental G-API functionality,
 * functions or structures in this namespace are subjects to change or
 * removal in the future releases. This namespace also contains
 * functions which API is not stabilized yet.
 */
namespace wip {

/**
 * @brief A class to group async requests to cancel them in a single shot.
 *
 * GAsyncContext is passed as an argument to async() and async_apply() functions
 */

class GAPI_EXPORTS GAsyncContext{
    std::atomic<bool> cancelation_requested = {false};
public:
    /**
     * @brief Start cancellation process for an associated request.
     *
     * User still has to wait for each individual request (either via callback or according std::future object) to make sure it actually canceled.
     *
     * @return true if it was a first request to cancel the context
     */
    bool cancel();

    /**
    * @brief Returns true if cancellation was requested for this context.
    *
    * @return true if cancellation was requested for this context
    */
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
