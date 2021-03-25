// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPILED_ASYNC_HPP
#define OPENCV_GAPI_GCOMPILED_ASYNC_HPP

#include <future>           //for std::future
#include <exception>        //for std::exception_ptr
#include <functional>       //for std::function
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/own/exports.hpp>

namespace cv {
    //fwd declaration
    class GCompiled;

namespace gapi{
namespace wip {
    class GAsyncContext;
    /**
    These functions asynchronously (i.e. probably on a separate thread of execution) call GCompiled::operator() member function of their first argument with copies of rest of arguments (except callback) passed in.
    The difference between the function is the way to get the completion notification (via callback or a waiting on std::future object)
    If exception is occurred during execution of apply it is transferred to the callback (via function parameter) or passed to future (and will be thrown on call to std::future::get)

    N.B. :
    Input arguments are copied on call to async function (actually on call to cv::gin) and thus do not have to outlive the actual completion of asynchronous activity.
    While output arguments are "captured" by reference(pointer) and therefore _must_ outlive the asynchronous activity
    (i.e. live at least until callback is called or future is unblocked)

    @param gcmpld       Compiled computation (graph) to start asynchronously
    @param callback     Callback to be called when execution of gcmpld is done
    @param ins          Input parameters for gcmpld
    @param outs         Output parameters for gcmpld
    */
    GAPI_EXPORTS void                async(GCompiled& gcmpld, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs);

    /** @overload
    @param gcmpld       Compiled computation (graph) to run asynchronously
    @param callback     Callback to be called when execution of gcmpld is done
    @param ins          Input parameters for gcmpld
    @param outs         Output parameters for gcmpld
    @param ctx          Context this request belongs to
    @see   async GAsyncContext
    */
    GAPI_EXPORTS void                async(GCompiled& gcmpld, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GAsyncContext& ctx);

    /** @overload
    @param gcmpld       Compiled computation (graph) to run asynchronously
    @param ins          Input parameters for gcmpld
    @param outs         Output parameters for gcmpld
    @return             std::future<void> object to wait for completion of async operation
    @see async
    */
    GAPI_EXPORTS std::future<void>   async(GCompiled& gcmpld, GRunArgs &&ins, GRunArgsP &&outs);

    /**
    @param gcmpld       Compiled computation (graph) to run asynchronously
    @param ins          Input parameters for gcmpld
    @param outs         Output parameters for gcmpld
    @param ctx          Context this request belongs to
    @return             std::future<void> object to wait for completion of async operation
    @see   async GAsyncContext
    */
    GAPI_EXPORTS std::future<void>   async(GCompiled& gcmpld, GRunArgs &&ins, GRunArgsP &&outs, GAsyncContext& ctx);
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GCOMPILED_ASYNC_HPP
