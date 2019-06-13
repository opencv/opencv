// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_TESTS_HELPERS_HPP
#define OPENCV_GAPI_TESTS_HELPERS_HPP

#include <tuple>
#include <limits>

namespace opencv_test
{

// out_type == in_type in matrices initialization if out_type is marked as SAME_TYPE
enum {
    // TODO: why is it different from -1?
    SAME_TYPE = std::numeric_limits<int>::max()
};

// Ensure correct __VA_ARGS__ expansion on Windows
#define FIX_VAARGS(x) x

// implementation of recursive in-class declaration and initialization of member variables
#define __DEFINE_PARAMS_IMPL1(params_type, params, index, param_name) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params);

#define __DEFINE_PARAMS_IMPL2(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL1(params_type, params, index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL3(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL2(params_type, params, index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL4(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL3(params_type, params, index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL5(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL4(params_type, params, index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL6(params_type, params, index, param_name, ...) \
    std::tuple_element<index, params_type>::type param_name = std::get<index>(params); \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL5(params_type, params, index+1, __VA_ARGS__))

// user interface to define member variables of specified names
#define DEFINE_SPECIFIC_PARAMS_1(...) \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL1(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_2(...) \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL2(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_3(...) \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL3(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_4(...) \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL4(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_5(...) \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL5(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_6(...) \
    FIX_VAARGS(__DEFINE_PARAMS_IMPL6(specific_params_t, GetParam().specificParams(), 0, __VA_ARGS__))
} // namespace opencv_test

#endif //OPENCV_GAPI_TESTS_HELPERS_HPP
