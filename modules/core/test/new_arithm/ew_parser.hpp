// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 4: the cv::expression() frontend. Parses a std::format-like string into an EwGraph,
// compiles it and runs it over the given inputs, producing one or more output tensors.
//
// Grammar (informal):
//   program   := (assign ';')* result
//   assign    := IDENT '=' expr
//   result    := expr | '(' expr (',' expr)+ ')'        // a tuple => several outputs
//   expr      := precedence-climbing over the binary operators below
//   unary     := ('-' | '!') unary | primary
//   primary   := NUMBER | '{' INT '}' | IDENT | IDENT '(' args ')' | '(' expr ')'
// Inputs are referenced positionally as {0}, {1}, ...  Function names are either element-wise
// ops (max, min, abs, sqrt, exp, log, sin, cos, tanh, erf, relu, pow, clamp, select) or
// readable type casts (float, double, uint8, int8, uint16, int16, int32, ...).
// Binary operator precedence (high->low): * /  >  + -  >  < <= > >=  >  == !=  >  &  >  ^  >  |.
// Constants are scalars for now (per-channel vector constants will come later).

#ifndef OPENCV_EW_PARSER_HPP
#define OPENCV_EW_PARSER_HPP

#include "opencv2/core.hpp"
#include <string_view>

namespace cv { namespace ew {

// Evaluate an element-wise expression over `inputs`, writing the result tensor(s) to `outputs`.
void expression(std::string_view expr, InputArrayOfArrays inputs, OutputArrayOfArrays outputs);

}} // namespace cv::ew

#endif // OPENCV_EW_PARSER_HPP
