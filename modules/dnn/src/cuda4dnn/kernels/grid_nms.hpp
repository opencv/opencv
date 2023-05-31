// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GRID_NMS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GRID_NMS_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"
#include "../csl/tensor.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

std::size_t getGridNMSWorkspaceSizePerBatchItem(std::size_t num_classes, std::size_t classwise_topK);

template <class T>
void grid_nms(const csl::Stream& stream, csl::Span<unsigned int> workspace, csl::TensorSpan<int> indices, csl::TensorSpan<int> count, csl::TensorView<T> bboxes, int background_class_id, bool normalized_bbox, float nms_threshold);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GRID_NMS_HPP */
