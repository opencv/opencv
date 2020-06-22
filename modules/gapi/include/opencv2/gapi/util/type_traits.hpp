// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_UTIL_TYPE_TRAITS_HPP
#define OPENCV_GAPI_UTIL_TYPE_TRAITS_HPP

#include <type_traits>

namespace cv
{
namespace util
{
    //these are C++14 parts of type_traits :
    template< bool B, class T = void >
    using enable_if_t = typename std::enable_if<B,T>::type;

    template<typename T>
    using decay_t = typename std::decay<T>::type;

    //this is not part of C++14 but still, of pretty common usage
    template<class T, class U, class V = void>
    using are_different_t = enable_if_t< !std::is_same<decay_t<T>, decay_t<U>>::value, V>;

} // namespace cv
} // namespace util

#endif // OPENCV_GAPI_UTIL_TYPE_TRAITS_HPP
