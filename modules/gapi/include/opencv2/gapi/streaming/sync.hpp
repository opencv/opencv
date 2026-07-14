// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_SYNC_HPP
#define OPENCV_GAPI_STREAMING_SYNC_HPP

namespace cv {
namespace gapi {
namespace streaming {

enum class sync_policy {
    dont_sync,
    drop
};

} // namespace streaming
} // namespace gapi

namespace detail {
    template<> struct CompileArgTag<gapi::streaming::sync_policy> {
        static const char* tag() { return "gapi.streaming.sync_policy"; }
    };

} // namespace detail
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_SYNC_HPP
