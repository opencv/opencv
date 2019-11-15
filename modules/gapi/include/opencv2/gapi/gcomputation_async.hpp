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
    /** In contrast to async() functions, these do call GComputation::apply() member function of the GComputation passed in.

    @param gcomp        Computation (graph) to run asynchronously
    @param callback     Callback to be called when execution of gcomp is done
    @param ins          Input parameters for gcomp
    @param outs         Output parameters for gcomp
    @param args         Compile arguments to pass to GComputation::apply()
    @see                async
    */
    GAPI_EXPORTS void                async_apply(GComputation& gcomp, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args = {});
    /** @overload
    @param gcomp        Computation (graph) to run asynchronously
    @param callback     Callback to be called when execution of gcomp is done
    @param ins          Input parameters for gcomp
    @param outs         Output parameters for gcomp
    @param args         Compile arguments to pass to GComputation::apply()
    @param ctx          Context this request belongs to
    @see                async_apply async GAsyncContext
    */
    GAPI_EXPORTS void                async_apply(GComputation& gcomp, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args, GAsyncContext& ctx);
    /** @overload
    @param gcomp        Computation (graph) to run asynchronously
    @param ins          Input parameters for gcomp
    @param outs         Output parameters for gcomp
    @param args         Compile arguments to pass to GComputation::apply()
    @return             std::future<void> object to wait for completion of async operation
    @see                async_apply async
    */
    GAPI_EXPORTS std::future<void>   async_apply(GComputation& gcomp, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args = {});
    /** @overload
    @param gcomp        Computation (graph) to run asynchronously
    @param ins          Input parameters for gcomp
    @param outs         Output parameters for gcomp
    @param args         Compile arguments to pass to GComputation::apply()
    @param ctx          Context this request belongs to
    @return             std::future<void> object to wait for completion of async operation
    @see                async_apply async GAsyncContext
    */
    GAPI_EXPORTS std::future<void>   async_apply(GComputation& gcomp, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args,  GAsyncContext& ctx);
} // namespace wip
} // namespace gapi
} // namespace cv


#endif //OPENCV_GAPI_GCOMPUTATION_ASYNC_HPP
