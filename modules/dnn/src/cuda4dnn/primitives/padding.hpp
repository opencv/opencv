// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PADDING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PADDING_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/fill.hpp"
#include "../kernels/concat.hpp"
#include "../kernels/padding.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <algorithm>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class PaddingType {
        CONSTANT,
        REFLECTION101
    };

    template <class T>
    class PaddingOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        /* `ranges` is indexed by axis and contains the range in the output where the input is copied to */
        PaddingOp(csl::Stream stream_, PaddingType type_, T value_, std::vector<cv::Range> ranges)
            : stream(std::move(stream_)),  type{ type_ }, value{ value_ }, dstRanges(std::move(ranges))
        {
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            auto effective_rank = get_effective_rank(input);
            CV_Assert(get_effective_rank(input) == get_effective_rank(output));

            /* suppose we require padding for the first spatial axis (H in NCHW or D in NCDHW)
             *
             * there could be a case where the batch axis, channel axis, and the first spatial axis are all one
             * this would result in effective rank being less than the number of axes requiring padding
             */
            effective_rank = std::max(effective_rank, dstRanges.size());

            for (int i = effective_rank - dstRanges.size(); i < effective_rank; i++)
            {
                if (dstRanges[i] == Range::all())
                    CV_Assert(input.get_axis_size(i) == output.get_axis_size(i));
                else
                    CV_Assert(input.get_axis_size(i) == dstRanges[i].size());
            }

            if (type == PaddingType::CONSTANT)
            {
                kernels::fill<T>(stream, output, value);

                std::vector<std::size_t> offsets(effective_rank, 0);
                for (int i = 0; i < dstRanges.size(); i++)
                {
                    const auto delta = effective_rank - dstRanges.size();
                    if (dstRanges[i] != Range::all())
                        offsets[delta + i] = dstRanges[i].start;
                }

                kernels::concat_with_offsets<T>(stream, output, input, offsets);
            }
            else if (type == PaddingType::REFLECTION101)
            {
                std::vector<std::pair<std::size_t, std::size_t>> ranges(effective_rank);
                for (int i = 0; i < effective_rank; i++)
                {
                    const auto delta = effective_rank - dstRanges.size();
                    if (i < delta || dstRanges[i - delta] == Range::all())
                        ranges[i] = { 0, input.get_axis_size(i) };
                    else
                        ranges[i] = { dstRanges[i].start, dstRanges[i].end };
                }

                kernels::copy_with_reflection101<T>(stream, output, input, ranges);
            }
        }

    private:
        csl::Stream stream;
        PaddingType type;
        T value;

        std::vector<cv::Range> dstRanges;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PADDING_HPP */
