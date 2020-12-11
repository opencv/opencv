// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DETECTION_OUTPUT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DETECTION_OUTPUT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/fill_copy.hpp"
#include "../kernels/permute.hpp"
#include "../kernels/detection_output.hpp"
#include "../kernels/grid_nms.hpp"

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct DetectionOutputConfiguration {
        std::size_t batch_size;

        enum class CodeType {
            CORNER,
            CENTER_SIZE
        };
        CodeType code_type;

        bool share_location;
        std::size_t num_priors;
        std::size_t num_classes;
        std::size_t background_class_id;

        bool transpose_location;
        bool variance_encoded_in_target;
        bool normalized_bbox;
        bool clip_box;

        std::size_t classwise_topK;
        float confidence_threshold;
        float nms_threshold;

        int keepTopK;
    };

    template <class T>
    class DetectionOutputOp final : public CUDABackendNode {
    private:
        /* We have block level NMS kernel where each block handles one class of one batch item.
         * If the number of classes and batch size together is very low, the blockwise NMS kernel
         * won't able to fully saturate the GPU with work.
         *
         * We also have a grid level NMS kernel where multiple blocks handle each class of every batch item.
         * This performs better in the worst case and utilizes resources better when block level kernel isn't
         * able to saturate the GPU with enough work. However, this is not efficient in the average case where
         * the block level kernel is able to saturate the GPU. It does better when the blockwise NMS barely
         * saturates the GPU.
         *
         * `GRID_NMS_CUTOFF` is the cutoff for `num_classes * batch_size` above which we will switch from grid
         * level NMS to block level NMS.
         */
        static constexpr int GRID_NMS_CUTOFF = 32;

    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        DetectionOutputOp(csl::Stream stream_, const DetectionOutputConfiguration& config)
            : stream(std::move(stream_))
        {
            corner_true_or_center_false = (config.code_type == DetectionOutputConfiguration::CodeType::CORNER);

            share_location = config.share_location;
            num_priors = config.num_priors;
            num_classes = config.num_classes;
            background_class_id = config.background_class_id;

            transpose_location = config.transpose_location;
            variance_encoded_in_target = config.variance_encoded_in_target;
            normalized_bbox = config.normalized_bbox;
            clip_box = config.clip_box;

            classwise_topK = config.classwise_topK;
            confidence_threshold = config.confidence_threshold;
            nms_threshold = config.nms_threshold;

            keepTopK = config.keepTopK;
            CV_Assert(keepTopK > 0);

            if (classwise_topK == -1)
            {
                classwise_topK = num_priors;
                if (keepTopK > 0 && keepTopK < num_priors)
                    classwise_topK = keepTopK;
            }

            auto batch_size = config.batch_size;
            auto num_loc_classes = (share_location ? 1 : num_classes);

            csl::WorkspaceBuilder builder;
            builder.require<T>(batch_size * num_priors * num_loc_classes * 4); /* decoded boxes */
            builder.require<T>(batch_size * num_classes * num_priors); /* transposed scores */
            builder.require<int>(batch_size * num_classes * classwise_topK); /* indices */
            builder.require<int>(batch_size * num_classes); /* classwise topK count */
            builder.require<T>(batch_size * num_classes * classwise_topK * 4); /* topK decoded boxes */

            if (batch_size * num_classes <= GRID_NMS_CUTOFF)
            {
                auto workspace_per_batch_item = kernels::getGridNMSWorkspaceSizePerBatchItem(num_classes, classwise_topK);
                builder.require(batch_size * workspace_per_batch_item);
            }

            builder.require<int>(batch_size * keepTopK); /* final kept indices */
            builder.require<int>(batch_size); /* kept indices count */
            builder.require<int>(1); /* total number of detections */

            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            /* locations, scores and priors make the first three inputs in order */
            /* the 4th input is used to obtain the shape for clipping */
            CV_Assert((inputs.size() == 3 || inputs.size() == 4) && outputs.size() == 1);

            // locations: [batch_size, num_priors, num_loc_classes, 4]
            auto locations_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto locations = locations_wrapper->getView();

            // scores: [batch_size, num_priors, num_classes]
            auto scores_wrapper = inputs[1].dynamicCast<wrapper_type>();
            auto scores = scores_wrapper->getView();
            scores.unsqueeze();
            scores.reshape(-1, num_priors, num_classes);

            // priors: [1, 2, num_priors, 4]
            auto priors_wrapper = inputs[2].dynamicCast<wrapper_type>();
            auto priors = priors_wrapper->getView();

            // output: [1, 1, batch_size * keepTopK, 7]
            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            auto batch_size = locations.get_axis_size(0);
            auto num_loc_classes = (share_location ? 1 : num_classes);
            while(locations.rank() < 4)
                locations.unsqueeze();
            locations.reshape(batch_size, num_priors, num_loc_classes, 4);

            float clip_width = 0.0, clip_height = 0.0;
            if (clip_box)
            {
                if (normalized_bbox)
                {
                    clip_width = clip_height = 1.0f;
                }
                else
                {
                    auto image_wrapper = inputs[3].dynamicCast<wrapper_type>();
                    auto image_shape = image_wrapper->getShape();

                    CV_Assert(image_shape.size() == 4);
                    clip_width = image_shape[3] - 1;
                    clip_height = image_shape[2] - 1;
                }
            }

            csl::WorkspaceAllocator allocator(workspace);

            // decoded_boxes: [batch_size, num_priors, num_loc_classes, 4]
            csl::TensorSpan<T> decoded_boxes;
            {
                auto shape = std::vector<std::size_t>{batch_size, num_priors, num_loc_classes, 4};
                decoded_boxes = allocator.get_tensor_span<T>(std::begin(shape), std::end(shape));
                CV_Assert(is_shape_same(decoded_boxes, locations));
            }

            kernels::decode_bboxes<T>(stream, decoded_boxes, locations, priors,
                num_loc_classes, share_location, background_class_id,
                transpose_location, variance_encoded_in_target,
                corner_true_or_center_false, normalized_bbox,
                clip_box, clip_width, clip_height);

            // scores_permuted: [batch_size, num_classes, num_priors]
            csl::TensorSpan<T> scores_permuted;
            {
                auto shape = std::vector<std::size_t>{batch_size, num_classes, num_priors};
                scores_permuted = allocator.get_tensor_span<T>(std::begin(shape), std::end(shape));
            }

            kernels::permute<T>(stream, scores_permuted, scores, {0, 2, 1});

            // indices: [batch_size, num_classes, classwise_topK]
            csl::TensorSpan<int> indices;
            {
                auto shape = std::vector<std::size_t>{batch_size, num_classes, classwise_topK};
                indices = allocator.get_tensor_span<int>(std::begin(shape), std::end(shape));
            }

            // count: [batch_size, num_classes]
            csl::TensorSpan<int> count;
            {
                auto shape = std::vector<std::size_t>{batch_size, num_classes};
                count = allocator.get_tensor_span<int>(std::begin(shape), std::end(shape));
            }

            kernels::findTopK<T>(stream, indices, count, scores_permuted, background_class_id, confidence_threshold);

            // collected_bboxes: [batch_size, num_classes, classwise_topK, 4]
            csl::TensorSpan<T> collected_bboxes;
            {
                auto shape = std::vector<std::size_t>{batch_size, num_classes, classwise_topK, 4};
                collected_bboxes = allocator.get_tensor_span<T>(std::begin(shape), std::end(shape));
            }

            kernels::box_collect<T>(stream, collected_bboxes, decoded_boxes, indices, count, share_location, background_class_id);

            if (batch_size * num_classes <= GRID_NMS_CUTOFF)
            {
                auto workspace_per_batch_item = kernels::getGridNMSWorkspaceSizePerBatchItem(num_classes, classwise_topK);
                auto workspace = allocator.get_span<unsigned int>(batch_size * workspace_per_batch_item / sizeof(unsigned int));
                kernels::grid_nms<T>(stream, workspace, indices, count, collected_bboxes, background_class_id, normalized_bbox, nms_threshold);
            }
            else
            {
                kernels::blockwise_class_nms<T>(stream, indices, count, collected_bboxes, normalized_bbox, background_class_id, nms_threshold);
            }

            // kept_indices: [batch_size, keepTopK]
            csl::TensorSpan<int> kept_indices;
            {
                auto shape = std::vector<std::size_t>{batch_size, static_cast<std::size_t>(keepTopK)};
                kept_indices = allocator.get_tensor_span<int>(std::begin(shape), std::end(shape));
            }

            // kept_count: [batch_size]
            csl::TensorSpan<int> kept_count;
            {
                auto shape = std::vector<std::size_t>{batch_size};
                kept_count = allocator.get_tensor_span<int>(std::begin(shape), std::end(shape));
            }

            kernels::nms_collect<T>(stream, kept_indices, kept_count, indices, count, scores_permuted, confidence_threshold, background_class_id);

            auto num_detections = allocator.get_span<int>(1);
            kernels::fill<int>(stream, num_detections, 0);
            kernels::fill<T>(stream, output, 0.0);
            kernels::consolidate_detections<T>(stream, output, kept_indices, kept_count, decoded_boxes, scores_permuted, share_location, num_detections.data());
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;
        std::size_t scratch_mem_in_bytes;

        bool share_location;
        std::size_t num_priors;
        std::size_t num_classes;
        std::size_t background_class_id;

        bool transpose_location;
        bool variance_encoded_in_target;
        bool corner_true_or_center_false;
        bool normalized_bbox;
        bool clip_box;

        std::size_t classwise_topK;
        float confidence_threshold;
        float nms_threshold;

        int keepTopK;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DETECTION_OUTPUT_HPP */
