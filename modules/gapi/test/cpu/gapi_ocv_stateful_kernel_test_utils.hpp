// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_OCV_STATEFUL_KERNEL_TESTS_UTILS_HPP
#define OPENCV_GAPI_OCV_STATEFUL_KERNEL_TESTS_UTILS_HPP

#include "../test_precomp.hpp"

// TODO: Reuse Anatoliy's logic for support of types with commas in macro.
//       Retrieve the common part from Anatoliy's logic to the separate place.
#define DEFINE_INITIALIZER(Name, StateType, ...) \
struct Name                                      \
{                                                \
    static StateType value()                     \
    {                                            \
       return __VA_ARGS__;                       \
    }                                            \
}                                                \

namespace opencv_test
{
namespace
{
struct UserStruct
{
    UserStruct() = default;
    UserStruct(short myShortVal, float myFloatVal):
    _myShortVal(myShortVal),
    _myFloatVal(myFloatVal) { }

    bool operator==(const UserStruct& rhs) const
    {
        return ((_myShortVal == rhs._myShortVal) &&
                (_myFloatVal == rhs._myFloatVal));
    }

private:
    short _myShortVal;
    float _myFloatVal;
};
} // anonymous namespace
} // opencv_test

#endif // OPENCV_GAPI_OCV_STATEFUL_KERNEL_TESTS_UTILS_HPP
