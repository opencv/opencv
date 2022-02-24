// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

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

//! @addtogroup gapi_serialization
//! @{

struct NotImplemented {
};

/** @brief This structure allows to implement serialization routines for custom types.
 *
 * The default S11N for custom types is not implemented.
 *
 * @note When providing an overloaded implementation for S11N with your type
 * don't inherit it from NotImplemented structure.
 *
 * @note There are lots of overloaded >> and << operators for basic and OpenCV/G-API types
 * which can be utilized when serializing a custom type.
 *
 * Example of usage:
 * @snippet samples/cpp/tutorial_code/gapi/doc_snippets/api_ref_snippets.cpp S11N usage
 *
 */
template<typename T>
struct S11N: public NotImplemented {
    /**
     * @brief This function allows user to serialize their custom type.
     *
     * @note The default overload throws an exception if called. User need to
     * properly overload the function to use it.
     */
    static void serialize(IOStream &, const T &) {
        GAPI_Assert(false && "No serialization routine is provided!");
    }
    /**
     * @brief This function allows user to deserialize their custom type.
     *
     * @note The default overload throws an exception if called. User need to
     * properly overload the function to use it.
     */
    static T deserialize(IIStream &) {
        GAPI_Assert(false && "No deserialization routine is provided!");
    }
};

/// @private -- Exclude this struct from OpenCV documentation
template<typename T> struct has_S11N_spec {
    static constexpr bool value = !std::is_base_of<NotImplemented,
                                        S11N<typename std::decay<T>::type>>::value;
};
//! @} gapi_serialization

} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_BASE_HPP
