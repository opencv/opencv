// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TRILU_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TRILU_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

/* ONNX Trilu over the trailing [height, width] of each of `loops` matrices.
 *
 * Mirrors trilu_layer.cpp, including one quirk worth stating: the CPU only rewrites
 * rows below min(height, width) and leaves any row past that as a straight copy. This
 * reproduces that rather than zeroing the full triangle.
 */
template <class T>
void trilu(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    int loops, int height, int width, int k, bool upper);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TRILU_HPP */
