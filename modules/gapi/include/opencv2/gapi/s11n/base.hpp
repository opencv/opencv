// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_S11N_BASE_HPP
#define OPENCV_GAPI_S11N_BASE_HPP

#include <opencv2/gapi/own/assert.hpp>

namespace cv {
namespace gapi {
namespace s11n {
struct IOStream;
struct IIStream;

namespace detail {
// Will be used along with default types if possible in specific cases (compile args, etc)
// Note: actual implementation is defined by user
template<typename T>
struct S11N {
    static void serialize(IOStream &, const T &) {
        GAPI_Assert(false && "No serialization routine is provided!");
    }
    static T deserialize(IIStream &) {
        GAPI_Assert(false && "No deserialization routine is provided!");
    }
};

} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_BASE_HPP
