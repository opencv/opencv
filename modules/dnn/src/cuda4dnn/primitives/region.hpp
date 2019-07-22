// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_PRIMITIVES_REGION_HPP
#define OPENCV_DNN_CUDA4DNN_PRIMITIVES_REGION_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/cudnn.hpp"
#include "../csl/tensor_ops.hpp"
#include "../csl/kernels.hpp"

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class squash_method {
        softmax,
        sigmoid
    };

    template <class T>
    struct RegionConfiguration {
        /* The image is divided into (H, W) cells.
         *
         * Each cell is interested in exactly one object and predicts `boxes_per_cell` bounding boxes
         * for that object.
         *
         * Each bounding box contains:
         * - 4 box coordinates
         * - objectness confidence score
         * - `classes` number of class scores
         *
         * The object score is reduced to a probability using sigmoid and the class scores are reduced to
         * probabilities by either applying sigmoid or softmax (which is a configuration option).
         *
         * object_prob = sigmoid(object_score)
         * conditional_class_prob = sigmoid, softmax across all classes
         *
         * actual class probability = conditional_class_prob * object_prob
         */

        /* method for reducing "class" scores to probabilities */
        squash_method squash;

        std::size_t classes, boxes_per_cell;

        std::size_t width_norm, height_norm;

        /* prob cutoffs below which the prediction is nulled */
        T object_prob_cutoff;
        T class_prob_cutoff;

        T nms_iou_threshold;
    };

    template <class T>
    class RegionOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        template <class V>
        RegionOp(csl::Stream stream_, const cv::Mat& bias, const RegionConfiguration<V>& config)
            : stream(std::move(stream_))
        {
            biasTensor = csl::makeTensorHeader<T>(bias);
            csl::copyMatToTensor<T>(biasTensor, bias, stream);

            classes = config.classes;
            boxes_per_cell = config.boxes_per_cell;

            width_norm = config.width_norm;
            height_norm = config.height_norm;

            squash_type = config.squash;

            object_prob_cutoff = config.object_prob_cutoff;
            class_prob_cutoff = config.class_prob_cutoff;

            nms_iou_threshold = config.nms_iou_threshold;
        }

        void forward(
            std::vector<cv::Ptr<BackendWrapper>>& inputs,
            std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            csl::memcpy<T>(output.get(), input.get(), output.size(), stream);

            auto rows = input.get_axis_size(1);
            auto cols = input.get_axis_size(2);

            auto cell_box_size = classes + 4 + 1;

            /* we squash class scores into probabilities using softmax or sigmoid */
            if (squash_type == squash_method::softmax)
                csl::kernels::softmax_strided<T>(stream, output, input, classes, cell_box_size, 5);
            else if (squash_type == squash_method::sigmoid)
                csl::kernels::sigmoid_strided<T>(stream, output, input, classes, cell_box_size, 5);

            csl::kernels::region_finalize<T>(stream, output, input, biasTensor, object_prob_cutoff, class_prob_cutoff,
                height_norm, width_norm, rows, cols, boxes_per_cell, cell_box_size, classes);

            if (nms_iou_threshold > 0) {
                /* TODO nms on gpu */
            }
        }

    private:
        csl::Stream stream;

        csl::Tensor<T> biasTensor;
        std::size_t classes, boxes_per_cell;
        std::size_t width_norm, height_norm;
        squash_method squash_type;

        T object_prob_cutoff, class_prob_cutoff;
        T nms_iou_threshold;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_CUDA4DNN_PRIMITIVES_REGION_HPP */
