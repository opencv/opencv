// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2021, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <float.h>
#include <algorithm>
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
        exclusive = params.get<int>("exclusive", 0);
        reverse = params.get<int>("reverse", 0);
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
        const auto axis = normalize_axis(axis_raw, src_mat.dims);

        // Get y tensor.
        auto &dst_mat = outputs[0];
        src_mat.copyTo(dst_mat);
        auto *dst_ptr = dst_mat.ptr<float>();

        const auto outer_size = src_mat.total(0, axis);
        const int inner_size = src_mat.size[axis];
        const auto num_channels = src_mat.total(axis + 1);
        const auto outer_step = src_mat.total(axis);
        const auto inner_step = src_mat.total(axis + 1);

        for (size_t outer_dim = 0; outer_dim < outer_size; outer_dim++)
        {
            auto outer_offset = outer_dim * outer_step;

            // Compute cumsum along axis.
            if (reverse)
            {
                auto src_offset = outer_offset + (inner_size - 1) * inner_step;

                for (int inner_dim = inner_size - 1; inner_dim >= 0; inner_dim--)
                {
                    const auto dst_offset = outer_offset + inner_dim * inner_step;

                    // Duplicate logic across channels.
                    for (size_t channel = 0; channel < num_channels; channel++)
                    {
                        // Cumsum begins by copying the last element from input to output.
                        if ((inner_size - 1) - inner_dim == exclusive)
                        {
                            dst_ptr[dst_offset + channel] = src_ptr[src_offset + channel];
                            src_offset -= inner_step;
                        }
                        else if ((inner_size - 1) - inner_dim > exclusive)
                        {
                            const auto previous_dst_offset = dst_offset + inner_step;
                            dst_ptr[dst_offset + channel] = dst_ptr[previous_dst_offset + channel] +
                                                            src_ptr[src_offset + channel];
                            src_offset -= inner_step;
                        }
                        else
                        {
                            dst_ptr[dst_offset + channel] = 0.0f;
                        }
                    }
                }
            }
            else
            {
                auto src_offset = outer_offset;

                for (int inner_dim = 0; inner_dim < inner_size; inner_dim++)
                {
                    const auto dst_offset = outer_offset + inner_dim * inner_step;

                    // Duplicate logic across channels.
                    for (size_t channel = 0; channel < num_channels; channel++)
                    {
                        // Cumsum begins by copying the first element from input to output.
                        if (inner_dim == exclusive)
                        {
                            dst_ptr[dst_offset + channel] = src_ptr[src_offset + channel];
                            src_offset += inner_step;
                        }
                        else if (inner_dim > exclusive)
                        {
                            const auto previous_dst_offset = dst_offset - inner_step;
                            dst_ptr[dst_offset + channel] = dst_ptr[previous_dst_offset + channel] +
                                                            src_ptr[src_offset + channel];
                            src_offset += inner_step;
                        }
                        else
                        {
                            dst_ptr[dst_offset + channel] = 0.0f;
                        }
                    }
                }
            }
        }
    }

    int axis_raw;
    int exclusive;
    int reverse;
};

Ptr<CumSumLayer> CumSumLayer::create(const LayerParams& params)
{
    return Ptr<CumSumLayer>(new CumSumLayerImpl(params));
}

}
}
