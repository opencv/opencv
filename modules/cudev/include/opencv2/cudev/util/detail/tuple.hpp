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

#ifndef __OPENCV_CUDEV_UTIL_TUPLE_DETAIL_HPP__
#define __OPENCV_CUDEV_UTIL_TUPLE_DETAIL_HPP__

#include <thrust/tuple.h>

namespace cv { namespace cudev {

namespace tuple_detail
{
    using thrust::tuple;
    using thrust::tuple_size;
    using thrust::get;
    using thrust::tuple_element;
    using thrust::make_tuple;
    using thrust::tie;

    template <class Tuple, int SIZE, template <typename T> class CvtOp> struct ConvertTuple;

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 2, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type
        > type;
    };

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 3, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<2, Tuple>::type>::type
        > type;
    };

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 4, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<2, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<3, Tuple>::type>::type
        > type;
    };

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 5, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<2, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<3, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<4, Tuple>::type>::type
        > type;
    };

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 6, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<2, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<3, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<4, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<5, Tuple>::type>::type
        > type;
    };

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 7, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<2, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<3, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<4, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<5, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<6, Tuple>::type>::type
        > type;
    };

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 8, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<2, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<3, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<4, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<5, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<6, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<7, Tuple>::type>::type
        > type;
    };

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 9, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<2, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<3, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<4, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<5, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<6, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<7, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<8, Tuple>::type>::type
        > type;
    };

    template <class Tuple, template <typename T> class CvtOp> struct ConvertTuple<Tuple, 10, CvtOp>
    {
        typedef tuple<
            typename CvtOp<typename tuple_element<0, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<1, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<2, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<3, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<4, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<5, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<6, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<7, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<8, Tuple>::type>::type,
            typename CvtOp<typename tuple_element<9, Tuple>::type>::type
        > type;
    };
}

}}

#endif
