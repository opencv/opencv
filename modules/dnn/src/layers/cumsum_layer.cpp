// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
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
        return exclusive_raw == 0;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        // Get input tensor.
        const auto& src_mat = inputs[0];
        const auto* src_ptr = src_mat.ptr<float>();

        // Get target axis.
        int axis = inputs.size() > 1 ? parseAxis(inputs[1]) : axis_raw;
        axis = normalize_axis(axis, src_mat.dims);


        // Get output tensor.
        auto& dst_mat = outputs[0];
        auto* dst_ptr = dst_mat.ptr<float>();

        // Get flags.
        const auto exclusive = exclusive_raw == 1;
        const auto reverse = reverse_raw == 1;

        // Data with [dim_1, .. , dim_k-1, target_dim, dim_k+1, .. , dim_n]
        // dimensions is represented here as [outer_dim, target_dim, inner_dim]
        const size_t outer_size = src_mat.total(0, axis);
        const size_t target_size = src_mat.size[axis];
        const size_t inner_size = src_mat.total(axis + 1);
        const size_t outer_step_length = target_size * inner_size;

        // Calculating steps in target dimensions
        const int target_start = reverse ? target_size - 1 : 0;
        const int target_stop = reverse ? -1 : target_size;
        const int target_delta = reverse ? -1 : 1;
        const int target_step = target_delta * inner_size;

        // If exclusive, the j-th output element would be the sum of the first (j-1) elements.
        // Otherwise, it would be the sum of the first j elements.
        const int exclusive_delta = exclusive ? target_step : 0;

        for (size_t outer_idx = 0; outer_idx < outer_size; outer_idx++)
        {
            const size_t target_offset = outer_idx * outer_step_length;

            // Handle first element of target dimension.
            size_t first_inner_offset = target_offset + target_start * inner_size;
            if (exclusive)
                for (size_t inner_idx = 0; inner_idx < inner_size; inner_idx++)
                    dst_ptr[first_inner_offset + inner_idx] = 0.0f;
            else
                for (size_t inner_idx = 0; inner_idx < inner_size; inner_idx++)
                    dst_ptr[first_inner_offset + inner_idx] = src_ptr[first_inner_offset + inner_idx];

            // Handle remaining elements of target dimension.
            for (int target_idx = target_start + target_delta; target_idx != target_stop; target_idx += target_delta)
            {
                const size_t inner_offset = target_offset + target_idx * inner_size;

                for (size_t inner_idx = 0; inner_idx < inner_size; inner_idx++)
                {
                    dst_ptr[inner_offset + inner_idx] = dst_ptr[inner_offset - target_step + inner_idx] +
                        src_ptr[inner_offset - exclusive_delta + inner_idx];
                }
            }
        }
    }

    int parseAxis(const Mat& axis_mat) {
        CV_CheckEQ(axis_mat.total(), 1u, "Axis tensor should contain single value");
        if (axis_mat.type() == CV_32SC1)
            return axis_mat.at<int32_t>(0);
        else
        {
            Mat axis_mat_int;
            axis_mat.convertTo(axis_mat_int, CV_32SC1);
            return axis_mat_int.at<int32_t>(0);
        }
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        std::shared_ptr<ov::op::v0::CumSum> cumsum;
        if (nodes.size() == 2)
        {
            int32_t axis_shape = 1;
            auto axis_scalar = std::make_shared<ov::op::v1::Reshape>(
                nodes[1].dynamicCast<InfEngineNgraphNode>()->node,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, &axis_shape),
                false);
            cumsum = std::make_shared<ov::op::v0::CumSum>(
                nodes[0].dynamicCast<InfEngineNgraphNode>()->node,
                std::make_shared<ov::op::v0::Convert>(axis_scalar, ov::element::i32),
                exclusive_raw,
                reverse_raw);
        }
        else
        {
            cumsum = std::make_shared<ov::op::v0::CumSum>(
                nodes[0].dynamicCast<InfEngineNgraphNode>()->node,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, &axis_raw),
                exclusive_raw,
                reverse_raw);
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(cumsum));
    }
#endif  // HAVE_DNN_NGRAPH

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
