/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#pragma once

#ifndef __OPENCV_CUDEV_UTIL_TUPLE_HPP__
#define __OPENCV_CUDEV_UTIL_TUPLE_HPP__

#include "../common.hpp"
#include "detail/tuple.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

using tuple_detail::tuple;
using tuple_detail::tuple_size;
using tuple_detail::get;
using tuple_detail::tuple_element;
using tuple_detail::make_tuple;
using tuple_detail::tie;

template <typename T> struct TupleTraits
{
    enum { is_tuple = 0 };
    enum { size = 1 };
};
template <typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9>
struct TupleTraits< tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> >
{
    enum { is_tuple = 1 };
    enum { size = tuple_size< tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> >::value };
};

template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple
{
    typedef typename tuple_detail::ConvertTuple<Tuple, tuple_size<Tuple>::value, CvtOp>::type type;
};

//! @}

}}

#endif
