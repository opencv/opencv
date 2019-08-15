// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GCOMPUTATION_ASYNC_HPP
#define OPENCV_GAPI_GCOMPUTATION_ASYNC_HPP


#include <future>                           //for std::future
#include <exception>                        //for std::exception_ptr
#include <functional>                       //for std::function
#include <opencv2/gapi/garg.hpp>            //for GRunArgs, GRunArgsP
#include <opencv2/gapi/gcommon.hpp>         //for GCompileArgs
#include <opencv2/gapi/own/exports.hpp>


namespace cv {
    //fwd declaration
    class GComputation;
namespace gapi {
namespace wip  {
    class GAsyncContext;
    //These functions asynchronously (i.e. probably on a separate thread of execution) call apply member function of their first argument with copies of rest of arguments (except callback) passed in.
    //The difference between the function is the way to get the completion notification (via callback or a waiting on std::future object)
    //If exception is occurred during execution of apply it is transferred to the callback (via function parameter) or passed to future (and will be thrown on call to std::future::get)

    //N.B. :
    //Input arguments are copied on call to async function (actually on call to cv::gin) and thus do not have to outlive the actual completion of asynchronous activity.
    //While Output arguments are "captured" by reference(pointer) and therefore _must_ outlive the asynchronous activity
    //(i.e. live at least until callback is called or future is unblocked)
    GAPI_EXPORTS void                async_apply(GComputation& gcomp, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args = {});
    GAPI_EXPORTS void                async_apply(GComputation& gcomp, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args, GAsyncContext& ctx);

    GAPI_EXPORTS std::future<void>   async_apply(GComputation& gcomp, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args = {});
    GAPI_EXPORTS std::future<void>   async_apply(GComputation& gcomp, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args,  GAsyncContext& ctx);
} // namespace wip
} // namespace gapi
} // namespace cv


#endif //OPENCV_GAPI_GCOMPUTATION_ASYNC_HPP
