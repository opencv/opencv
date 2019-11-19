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

// Ensure correct __VA_ARGS__ expansion on Windows
#define __WRAP_VAARGS(x) x

#define __TUPLE_PARAM_TYPE(i) std::tuple_element<i, AllParams::specific_params_t>::type

// implementation of recursive in-class declaration and initialization of member variables
#define __DEFINE_PARAMS_IMPL1(index, param_name) \
    __TUPLE_PARAM_TYPE(index) param_name = getSpecificParam<index>();

#define __DEFINE_PARAMS_IMPL2(index, param_name, ...) \
    __TUPLE_PARAM_TYPE(index) param_name = getSpecificParam<index>(); \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL1(index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL3(index, param_name, ...) \
    __TUPLE_PARAM_TYPE(index) param_name = getSpecificParam<index>(); \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL2(index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL4(index, param_name, ...) \
    __TUPLE_PARAM_TYPE(index) param_name = getSpecificParam<index>(); \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL3(index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL5(index, param_name, ...) \
    __TUPLE_PARAM_TYPE(index) param_name = getSpecificParam<index>(); \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL4(index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL6(index, param_name, ...) \
    __TUPLE_PARAM_TYPE(index) param_name = getSpecificParam<index>(); \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL5(index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL7(index, param_name, ...) \
    __TUPLE_PARAM_TYPE(index) param_name = getSpecificParam<index>(); \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL6(index+1, __VA_ARGS__))

#define __DEFINE_PARAMS_IMPL8(index, param_name, ...) \
    __TUPLE_PARAM_TYPE(index) param_name = getSpecificParam<index>(); \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL7(index+1, __VA_ARGS__))

// user interface to define member variables of specified names
#define DEFINE_SPECIFIC_PARAMS_0()

#define DEFINE_SPECIFIC_PARAMS_1(...) \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL1(0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_2(...) \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL2(0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_3(...) \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL3(0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_4(...) \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL4(0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_5(...) \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL5(0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_6(...) \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL6(0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_7(...) \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL7(0, __VA_ARGS__))

#define DEFINE_SPECIFIC_PARAMS_8(...) \
    __WRAP_VAARGS(__DEFINE_PARAMS_IMPL8(0, __VA_ARGS__))
} // namespace opencv_test

#endif //OPENCV_GAPI_TESTS_HELPERS_HPP
