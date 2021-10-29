// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_DETECTION_OUTPUT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_DETECTION_OUTPUT_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"
#include "../csl/tensor.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void decode_bboxes(const csl::Stream& stream, csl::Span<T> output, csl::View<T> locations, csl::View<T> priors,
        std::size_t num_loc_classes, bool share_location, std::size_t background_label_id,
        bool transpose_location, bool variance_encoded_in_target,
        bool corner_true_or_center_false, bool normalized_bbox,
        bool clip_box, float clip_width, float clip_height);

    template <class T>
    void findTopK(const csl::Stream& stream, csl::TensorSpan<int> indices, csl::TensorSpan<int> count, csl::TensorView<T> scores, std::size_t background_label_id, float threshold);

    template <class T>
    void box_collect(const csl::Stream& stream, csl::TensorSpan<T> collected_bboxes, csl::TensorView<T> decoded_bboxes, csl::TensorView<int> indices, csl::TensorView<int> count, bool share_location, std::size_t background_label_id);

    template <class T>
    void blockwise_class_nms(const csl::Stream& stream, csl::TensorSpan<int> indices, csl::TensorSpan<int> count, csl::TensorView<T> collected_bboxes,
            bool normalized_bbox, std::size_t background_label_id, float nms_threshold);

    template <class T>
    void nms_collect(const csl::Stream& stream, csl::TensorSpan<int> kept_indices, csl::TensorSpan<int> kept_count,
        csl::TensorView<int> indices, csl::TensorView<int> count, csl::TensorView<T> scores, float, std::size_t background_label_id);

    template <class T>
    void consolidate_detections(const csl::Stream& stream, csl::TensorSpan<T> output,
        csl::TensorView<int> kept_indices, csl::TensorView<int> kept_count,
        csl::TensorView<T> decoded_bboxes, csl::TensorView<T> scores, bool share_location, csl::DevicePtr<int> num_detections);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_DETECTION_OUTPUT_HPP */
