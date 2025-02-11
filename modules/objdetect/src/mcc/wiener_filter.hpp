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

/**
  *  @file wiener_filter
  *  @brief filter wiener for denoise
  *  @author: Pedro D. Marrero Fernandez
  *  @data: 17/05/2016
  */

#ifndef _WIENER_FILTER_HPP
#define _WIENER_FILTER_HPP

#include "precomp.hpp"
namespace cv
{
namespace mcc
{

/// CWienerFilter
/**  @brief wiener class filter for denoise
      *  @author: Pedro D. Marrero Fernandez
      *  @data: 17/05/2016
      */
class CWienerFilter
{
public:
    CWienerFilter();
    ~CWienerFilter();

    /** cvWiener2
          * @brief A Wiener 2D Filter implementation for OpenCV
          * @author: Ray Juang / rayver{ _at_ } hkn{ / _dot_ / } berkeley(_dot_) edu
          * @date : 12.1.2006
          */
    void wiener2(InputArray _src, OutputArray _dst, int szWindowX, int szWindowY);
};

} // namespace mcc

} // namespace cv

#endif //_WIENER_FILTER_HPP
