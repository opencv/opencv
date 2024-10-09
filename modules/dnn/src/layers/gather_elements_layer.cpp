// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

static inline int calculateOffset(int outer_dim, const MatShape &shape_indices, int axis_skip, const MatStep &step_data) {
    int offset = 0;
    for (int axis = static_cast<int>(shape_indices.size()) - 2; axis >= 0; axis--) {
        int dim = shape_indices[axis];
        if (axis != axis_skip) {
            offset += (outer_dim % dim) * step_data[axis];
        }
        outer_dim /= dim;
    }
    return offset;
}

class GatherElementsLayerImpl CV_FINAL : public GatherElementsLayer
{
public:
    GatherElementsLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), 2ull, "GatherElements: requires two inputs");

        const auto &data = inputs[0];
        const auto &indices = inputs[1];
        CV_CheckEQ(data.size(), indices.size(), "GatherElements: data and indices should have the same dimension");

        int normalized_axis = normalize_axis(axis, static_cast<int>(data.size()));
        CV_CheckGE(normalized_axis, 0, "GatherElements: axis out of range");
        CV_CheckLT(normalized_axis, static_cast<int>(data.size()), "GatherElements: axis out of range");
        for (size_t i = 0; i < data.size(); i++) {
            if (i != normalized_axis) {
                CV_CheckEQ(data[i], indices[i], "GatherElements: shape mismatched");
            }
        }

        outputs.assign(1, inputs[1]); // shape of output is same as indices
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        const auto &data = inputs[0];
        axis = normalize_axis(axis, data.dims);
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

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& data = inputs[0];
        const Mat& indices = inputs[1];
        Mat& out = outputs[0];

        typeDispatch(outputs[0].type(), data, indices, out);
    }

    template <typename T>
    void forward_impl(const Mat& data_, const Mat& indices_,  Mat& out_)
    {
        const auto *ptr_data = data_.ptr<const T>();
        const auto *ptr_indices = indices_.ptr<const T>();
        auto *ptr_out = out_.ptr<T>();

        const auto shape_data = shape(data_);
        const auto &step_data = data_.step;
        const auto shape_indices = shape(indices_);

        int inner_most_dim = shape_indices.back();
        int axis_dim = shape_data[axis];
        size_t axis_step = static_cast<size_t>(step_data[axis] / sizeof(T));

        bool innermost_axis = axis == static_cast<int>(shape_data.size() - 1);

        auto fn = [&](const Range &r) {
            for (int i = r.start; i < r.end; i++) {
                auto *data = ptr_data + static_cast<size_t>(calculateOffset(i, shape_indices, axis, step_data) / sizeof(T));
                auto *indices = ptr_indices + i * inner_most_dim;
                auto *out = ptr_out + i * inner_most_dim;

                if (innermost_axis) {
                    for (int j = 0; j < inner_most_dim; j++) {
                        int index = static_cast<int>((indices[j] + axis_dim)) % axis_dim; // TODO: Check out-of-range index
                        out[j] = data[index];
                    }
                } else {
                    for (int j = 0; j < inner_most_dim; j++) {
                        int index = static_cast<int>(indices[j] + axis_dim) % axis_dim; // TODO: Check out-of-range index
                        out[j] = data[index * axis_step + j];
                    }
                }
            }
        };

        int outer_dims = total(shape_indices, 0, shape_indices.size() - 1);
        double nstripes = static_cast<size_t>(outer_dims * inner_most_dim * (1 / 1024.0));
        parallel_for_(Range(0, outer_dims), fn, nstripes);
    }

    template<typename... Args>
    inline void typeDispatch(const int type, Args&&... args)
    {
        switch (type)
        {
            case CV_8U:
                forward_impl<uint8_t>(std::forward<Args>(args)...);
                break;
            case CV_32S:
                forward_impl<int32_t>(std::forward<Args>(args)...);
                break;
            case CV_32F:
                forward_impl<float>(std::forward<Args>(args)...);
                break;
            default:
                CV_Error(cv::Error::BadDepth, "DNN/GatherElements: Unsupported type.");
        };
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        int32_t indicesBoundValue = nodes[0].dynamicCast<InfEngineNgraphNode>()->node.get_shape()[axis];
        auto indicesBound = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, &indicesBoundValue);
        auto indices = std::make_shared<ov::op::v0::Convert>(nodes[1].dynamicCast<InfEngineNgraphNode>()->node, ov::element::i32);
        auto indicesNonNegative = std::make_shared<ov::op::v1::Mod>(
            std::make_shared<ov::op::v1::Add>(indices, indicesBound),
            indicesBound);

        auto gatherElements = std::make_shared<ov::op::v6::GatherElements>(
            nodes[0].dynamicCast<InfEngineNgraphNode>()->node,
            indicesNonNegative,
            axis);
        return Ptr<BackendNode>(new InfEngineNgraphNode(gatherElements));
    }
#endif  // HAVE_DNN_NGRAPH

private:
    int axis;
};

Ptr<GatherElementsLayer> GatherElementsLayer::create(const LayerParams& params)
{
    return makePtr<GatherElementsLayerImpl>(params);
}

}} // namespace cv::dnn
