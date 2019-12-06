// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PRIOR_BOX_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PRIOR_BOX_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/span.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/prior_box.hpp"

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct PriorBoxConfiguration {
        std::size_t feature_map_width, feature_map_height;
        std::size_t image_width, image_height;

        /* parameters for prior boxes for each feature point */
        std::vector<float> box_widths, box_heights;
        std::vector<float> offsets_x, offsets_y;
        float stepX, stepY;

        std::vector<float> variance;

        /* number of priors per feature point */
        std::size_t num_priors;

        /* clamps the box coordinates to [0, 1] range */
        bool clip;

        /* normalizes the box coordinates using the image dimensions */
        bool normalize;
    };

    template <class T>
    class PriorBoxOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        PriorBoxOp(csl::Stream stream_, const PriorBoxConfiguration& config)
            : stream(std::move(stream_))
        {
            feature_map_width = config.feature_map_width;
            feature_map_height = config.feature_map_height;

            image_width = config.image_width;
            image_height = config.image_height;

            const auto& box_widths = config.box_widths;
            const auto& box_heights = config.box_heights;
            CV_Assert(box_widths.size() == box_heights.size());

            box_size = box_widths.size();

            const auto& offsets_x = config.offsets_x;
            const auto& offsets_y = config.offsets_y;
            CV_Assert(offsets_x.size() == offsets_y.size());

            offset_size = offsets_x.size();

            /* for better memory utilization and preassumably better cache performance, we merge
             * the four vectors and put them in a single tensor
             */
            auto total = box_widths.size() * 2 + offsets_x.size() * 2;
            std::vector<float> merged_params;
            merged_params.insert(std::end(merged_params), std::begin(box_widths), std::end(box_widths));
            merged_params.insert(std::end(merged_params), std::begin(box_heights), std::end(box_heights));
            merged_params.insert(std::end(merged_params), std::begin(offsets_x), std::end(offsets_x));
            merged_params.insert(std::end(merged_params), std::begin(offsets_y), std::end(offsets_y));
            CV_Assert(merged_params.size() == total);

            paramsTensor.resize(total);
            csl::memcpy(paramsTensor.get(), merged_params.data(), total, stream); /* synchronous copy */

            const auto& variance_ = config.variance;
            variance.assign(std::begin(variance_), std::end(variance_));

            num_priors = config.num_priors;
            stepX = config.stepX;
            stepY = config.stepY;
            clip = config.clip;
            normalize = config.normalize;
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 2); /* we don't need the inputs but we are given */
            CV_Assert(outputs.size() == 1);

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            /* we had stored all the parameters in a single tensor; now we create appropriate views
             * for each of the parameter arrays from the single tensor
             */
            auto boxWidths  = csl::View<float>(paramsTensor.get(), box_size);
            auto boxHeights = csl::View<float>(paramsTensor.get() + box_size, box_size);
            auto offsetsX   = csl::View<float>(paramsTensor.get() + 2 * box_size, offset_size);
            auto offsetsY   = csl::View<float>(paramsTensor.get() + 2 * box_size + offset_size, offset_size);

            kernels::generate_prior_boxes<T>(stream, output,
                boxWidths, boxHeights, offsetsX, offsetsY, stepX, stepY,
                variance, num_priors, feature_map_width, feature_map_height, image_width, image_height, normalize, clip);
        }

    private:
        csl::Stream stream;
        csl::Tensor<float> paramsTensor; /* widths, heights, offsetsX, offsetsY */

        std::size_t feature_map_width, feature_map_height;
        std::size_t image_width, image_height;

        std::size_t box_size, offset_size;
        float stepX, stepY;

        std::vector<float> variance;

        std::size_t num_priors;
        bool clip, normalize;
    };


}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PRIOR_BOX_HPP */
