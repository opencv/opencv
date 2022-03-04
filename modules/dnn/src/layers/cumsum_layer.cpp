// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class CumSumLayerImpl CV_FINAL : public CumSumLayer
{
public:
    CumSumLayerImpl(const LayerParams &params)
    {
        axis_raw = params.get<int>("axis", 0);
        exclusive_raw = params.get<int>("exclusive", 0);
        reverse_raw = params.get<int>("reverse", 0);
        setParamsFrom(params);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        // Get x tensor.
        const auto &src_mat = inputs[0];
        const auto *src_ptr = src_mat.ptr<float>();

        // Get axis.
        const int axis = normalize_axis(axis_raw, src_mat.dims);

        // Get y tensor.
        auto &dst_mat = outputs[0];
        src_mat.copyTo(dst_mat);
        auto *dst_ptr = dst_mat.ptr<float>();

        // Get flags.
        const auto exclusive = exclusive_raw == 1;
        const auto reverse = reverse_raw == 1;

        // Get parameters to iterate outer dimension.
        const size_t outer_size = src_mat.total(0, axis);
        const size_t outer_step_length = src_mat.total(axis);

        // Get parameters to iterate inner dimension.
        const size_t inner_size = src_mat.size[axis];

        if (!inner_size)
            return;

        const size_t inner_step_length = src_mat.total(axis + 1);
        const int inner_step = (reverse ? -1 : 1) * inner_step_length;
        const int inner_start = reverse ? inner_size - 1 : 0;
        const int inner_stop = reverse ? -1 : inner_size;
        const int inner_delta = reverse ? -1 : 1;

        // Get parameters to populate channels.
        const size_t num_channels = src_mat.total(axis + 1);

        for (size_t outer_dim = 0; outer_dim < outer_size; outer_dim++)
        {
            const size_t outer_offset = outer_dim * outer_step_length;
            size_t src_offset = outer_offset + inner_start * inner_step_length;

            // Populate first element of inner dimension.
            for (size_t channel = 0; channel < num_channels; channel++)
            {
                if (exclusive)
                {
                    dst_ptr[src_offset + channel] = 0.0f;
                }
                else
                {
                    dst_ptr[src_offset + channel] = src_ptr[src_offset + channel];
                    src_offset += inner_step;
                }
            }

            // Populate remaining elements of inner dimension.
            for (int inner_dim = inner_start + inner_delta; inner_dim != inner_stop; inner_dim += inner_delta)
            {
                const size_t dst_offset = outer_offset + inner_dim * inner_step_length;

                for (size_t channel = 0; channel < num_channels; channel++)
                {
                    const size_t previous_dst_offset = dst_offset - inner_step;
                    dst_ptr[dst_offset + channel] = dst_ptr[previous_dst_offset + channel] +
                            src_ptr[src_offset + channel];
                    src_offset += inner_step;
                }
            }
        }
    }

    int axis_raw;
    int exclusive_raw;
    int reverse_raw;
};

Ptr<CumSumLayer> CumSumLayer::create(const LayerParams& params)
{
    return Ptr<CumSumLayer>(new CumSumLayerImpl(params));
}

}
}
