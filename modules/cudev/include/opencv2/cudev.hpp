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

#ifndef __OPENCV_CUDEV_HPP__
#define __OPENCV_CUDEV_HPP__

#include "cudev/common.hpp"

#include "cudev/util/atomic.hpp"
#include "cudev/util/limits.hpp"
#include "cudev/util/saturate_cast.hpp"
#include "cudev/util/simd_functions.hpp"
#include "cudev/util/tuple.hpp"
#include "cudev/util/type_traits.hpp"
#include "cudev/util/vec_math.hpp"
#include "cudev/util/vec_traits.hpp"

#include "cudev/functional/color_cvt.hpp"
#include "cudev/functional/functional.hpp"
#include "cudev/functional/tuple_adapter.hpp"

#include "cudev/warp/reduce.hpp"
#include "cudev/warp/scan.hpp"
#include "cudev/warp/shuffle.hpp"
#include "cudev/warp/warp.hpp"

#include "cudev/block/block.hpp"
#include "cudev/block/dynamic_smem.hpp"
#include "cudev/block/reduce.hpp"
#include "cudev/block/scan.hpp"
#include "cudev/block/vec_distance.hpp"

#include "cudev/grid/copy.hpp"
#include "cudev/grid/reduce.hpp"
#include "cudev/grid/histogram.hpp"
#include "cudev/grid/integral.hpp"
#include "cudev/grid/pyramids.hpp"
#include "cudev/grid/reduce_to_vec.hpp"
#include "cudev/grid/split_merge.hpp"
#include "cudev/grid/transform.hpp"
#include "cudev/grid/transpose.hpp"

#include "cudev/ptr2d/constant.hpp"
#include "cudev/ptr2d/deriv.hpp"
#include "cudev/ptr2d/extrapolation.hpp"
#include "cudev/ptr2d/glob.hpp"
#include "cudev/ptr2d/gpumat.hpp"
#include "cudev/ptr2d/interpolation.hpp"
#include "cudev/ptr2d/lut.hpp"
#include "cudev/ptr2d/mask.hpp"
#include "cudev/ptr2d/remap.hpp"
#include "cudev/ptr2d/resize.hpp"
#include "cudev/ptr2d/texture.hpp"
#include "cudev/ptr2d/traits.hpp"
#include "cudev/ptr2d/transform.hpp"
#include "cudev/ptr2d/warping.hpp"
#include "cudev/ptr2d/zip.hpp"

#include "cudev/expr/binary_func.hpp"
#include "cudev/expr/binary_op.hpp"
#include "cudev/expr/color.hpp"
#include "cudev/expr/deriv.hpp"
#include "cudev/expr/expr.hpp"
#include "cudev/expr/per_element_func.hpp"
#include "cudev/expr/reduction.hpp"
#include "cudev/expr/unary_func.hpp"
#include "cudev/expr/unary_op.hpp"
#include "cudev/expr/warping.hpp"

#endif
