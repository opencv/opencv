// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPILED_ASYNC_HPP
#define OPENCV_GAPI_GCOMPILED_ASYNC_HPP

#include <future>
#include <exception>        //for std::exception_ptr
#include <functional>       //for std::function
#include "opencv2/gapi/garg.hpp"

namespace cv {
    //fwd declaration
    class GCompiled;

namespace gapi{
namespace wip {
    GAPI_EXPORTS void                async(GCompiled& gcmpld, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs);
    GAPI_EXPORTS std::future<void>   async(GCompiled& gcmpld, GRunArgs &&ins, GRunArgsP &&outs);
} // namespace gapi
} // namespace wip
} // namespace cv

#endif // OPENCV_GAPI_GCOMPILED_ASYNC_HPP
