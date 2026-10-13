// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_INT8LAYERS_INT8_ELTWISE_KERNELS_HPP__
#define __OPENCV_DNN_INT8LAYERS_INT8_ELTWISE_KERNELS_HPP__

#include <opencv2/core.hpp>

namespace cv {
namespace dnn {

// Buffers are uint8 if isU8, else int8. A 256-entry lut is indexed by byte value for uint8 and by
// value + 128 for int8; a null lut skips that step.

//! out = lut(clamp(round(c0*a + c1*b + offset), lo, hi))
void addInt8(const void* a, const void* b, void* out, int len, bool isU8,
             float c0, float c1, float offset, int lo, int hi, const uchar* lut);

//! out = lut(clamp(round(scale*(a - zp0)*(b - zp1) + offset), lo, hi))
void mulInt8(const void* a, const void* b, void* out, int len, bool isU8,
             float zp0, float zp1, float scale, float offset, int lo, int hi, const uchar* lut);

//! dst = lut(src); src and dst may be the same buffer.
void lutInt8(const void* src, void* dst, int len, bool isU8, const uchar* lut);

}} // namespace cv::dnn

#endif
