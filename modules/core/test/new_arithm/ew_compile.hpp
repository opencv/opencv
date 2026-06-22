// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 2: the graph compiler. The expression graph and its compiler used to live in their own
// types (EwGraph / compile()); they are now folded into cv::ew::TExpr (declared in ew_op.hpp):
// the graph builders input()/constant()/unary()/binary()/... and the lowering pass TExpr::compile()
// are all members of TExpr. This header is kept only so existing `#include "ew_compile.hpp"` users
// keep compiling; the implementation of the lowering pass still lives in ew_compile.cpp.

#ifndef OPENCV_EW_COMPILE_HPP
#define OPENCV_EW_COMPILE_HPP

#include "ew_op.hpp"

#endif // OPENCV_EW_COMPILE_HPP
