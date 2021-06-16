// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_S11N_BASE_HPP
#define OPENCV_GAPI_S11N_BASE_HPP

#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/own/exports.hpp>

namespace cv {
namespace gapi {

/**
 * @brief This namespace contains G-API serialization and
 * deserialization functions and data structures.
 */
namespace s11n {
struct IOStream;
struct IIStream;

namespace detail {

struct NotImplemented {
};

// The default S11N for custom types is NotImplemented
// Don't! sublass from NotImplemented if you actually implement S11N.
template<typename T>
struct S11N: public NotImplemented {
    static void serialize(IOStream &, const T &) {
        GAPI_Assert(false && "No serialization routine is provided!");
    }
    static T deserialize(IIStream &) {
        GAPI_Assert(false && "No deserialization routine is provided!");
    }
};

template<typename T> struct has_S11N_spec {
    static constexpr bool value = !std::is_base_of<NotImplemented,
                                        S11N<typename std::decay<T>::type>>::value;
};

} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_BASE_HPP
