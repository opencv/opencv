// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TOPK_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TOPK_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>
#include <cstdint>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

/* ONNX TopK along one axis. The input is treated as [outer, dim_axis, inner] and the
 * outputs as [outer, k, inner], matching TopK2's CPU layout.
 *
 * Selection is by strict lexicographic order -- value first, then smaller index on a
 * tie -- which is what ComparatorGreater/ComparatorLess in topk2_layer.cpp impose. The
 * results come out in rank order, so `sorted=1` is satisfied and `sorted=0` (which
 * leaves order unspecified) is satisfied too.
 *
 * One block owns one slice and makes `k` passes over it, so the cost is O(k * dim_axis).
 * That suits the small k detection heads use; it is not a general large-k sort.
 */
template <class T>
void topk(const csl::Stream& stream,
    csl::Span<T> values, csl::Span<std::int64_t> indices, csl::View<T> input,
    int outer, int dim_axis, int inner, int k, bool largest);

template <class T>
std::size_t topk_sort_workspace(const csl::Stream& stream, int outer, int dim_axis, bool largest);

template <class T>
void topk_sort(const csl::Stream& stream,
    csl::Span<T> values, csl::Span<std::int64_t> indices, csl::View<T> input,
    int outer, int dim_axis, int k, bool largest,
    void* workspace, std::size_t workspace_bytes);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TOPK_HPP */
