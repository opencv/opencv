// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_UTIL_COPY_THROUGH_MOVE_HPP
#define OPENCV_GAPI_UTIL_COPY_THROUGH_MOVE_HPP

#include <opencv2/gapi/util/type_traits.hpp> //decay_t

namespace cv
{
namespace util
{
    //This is a tool to move initialize captures of a lambda in C++11
    template<typename T>
    struct copy_through_move_t{
       T value;
       const T& get() const {return value;}
       T&       get()       {return value;}
       copy_through_move_t(T&& g) : value(std::move(g)) {}
       copy_through_move_t(copy_through_move_t&&) = default;
       copy_through_move_t(copy_through_move_t const& lhs) : copy_through_move_t(std::move(const_cast<copy_through_move_t&>(lhs))) {}
    };

    template<typename T>
    copy_through_move_t<util::decay_t<T>> copy_through_move(T&& t){
        return std::forward<T>(t);
    }
} // namespace util
} // namespace cv

#endif /* OPENCV_GAPI_UTIL_COPY_THROUGH_MOVE_HPP */
