// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef OPENCV_OBJDETECT_MCC_HPP
#define OPENCV_OBJDETECT_MCC_HPP


#include "objdetect/checker_detector.hpp"
#include "objdetect/checker_model.hpp"
#include "objdetect/ccm.hpp"

/** @defgroup mcc Macbeth Chart module
@{
    @defgroup color_correction Color Correction Model
@}


@addtogroup mcc

Introduction
------------

ColorCharts are a tool for calibrating the color profile of camera, which not
only depends on the intrinsic and extrinsic parameters of camera but also on the
lighting conditions. This is done by taking the image of a chart, such that the
value of its colors present in it known, in the image the color values changes
depeding on many variables, this gives us the colors initially present and the
colors that are present in the image, based on this information we can apply any
suitable algorithm to find the actual color of all the objects present in the
image.

*/

#endif
