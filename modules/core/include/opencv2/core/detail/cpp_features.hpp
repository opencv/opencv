// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_DETAIL_CPP_FEATURES_HPP
#define OPENCV_CORE_DETAIL_CPP_FEATURES_HPP

//Feature testing
//GCC: https://godbolt.org/z/hTobeP866
//clang: https://godbolt.org/z/G7KfT98xx
//MSVC: https://godbolt.org/z/jx97v5W9d

#include <numeric>
#include <tuple> //This needs to be here to acquire the right definitions. Replace with <version> by C++20

//Define all features needed for the experimental forward stl
#if defined __cpp_decltype_auto && __cpp_decltype_auto  >= 201304L

#define _stl_forward_cpp_features_present

#endif //find features

#endif
