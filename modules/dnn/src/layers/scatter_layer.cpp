// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "layers_common.hpp"

#include <algorithm> // for std::max & std::min

namespace cv { namespace dnn {

class ScatterLayerImpl CV_FINAL : public ScatterLayer
{
public:
    enum class REDUCTION
    {
        NONE = 1,
        ADD,
        MUL,
        MAX,
        MIN
    } reduction;

    ScatterLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", 0);
        String reduction_name = toLowerCase(params.get<String>("reduction", "none"));
        if (reduction_name == "none")
            reduction = REDUCTION::NONE;
        else if (reduction_name == "add")
            reduction = REDUCTION::ADD;
        else if (reduction_name == "mul")
            reduction = REDUCTION::MUL;
        else if (reduction_name == "max")
            reduction = REDUCTION::MAX;
        else if (reduction_name == "min")
            reduction = REDUCTION::MIN;
        else
            CV_Error(cv::Error::StsBadArg, "Unkown reduction \"" + reduction_name + "\"");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && reduction == REDUCTION::NONE);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), 3ull, "Scatter: require three inputs.");
        CV_CheckEQ(inputs[0].size(), inputs[1].size(), "Scatter: input data should have the same ndim with indices.");
        CV_CheckEQ(inputs[0].size(), inputs[2].size(), "Scatter: input data should have the same ndim with updates.");
        for (size_t i = 0; i < inputs[0].size(); i++)
        {
            CV_CheckGE(inputs[0][i], inputs[1][i], "Scatter: each dim of input data should be greater than (or equal to) indices'.");
            CV_CheckEQ(inputs[1][i], inputs[2][i], "Scatter: each dim of indices should be equal to updates'.");
        }
        outputs.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& data = inputs[0];
        const Mat& indices = inputs[1];
        const Mat& updates = inputs[2];
        Mat& out = outputs[0];

        typeDispatch(outputs[0].type(), data, indices, updates, out);
    }

    template<typename T, typename Functor>
    void forward_impl(const Functor &reduce_operation, const Mat &input_mat, const Mat &indices_mat, const Mat &updates_mat, Mat &output_mat) {
        input_mat.copyTo(output_mat);

        const int ndims = input_mat.dims;

        const auto &input_mat_shape = shape(input_mat);
        std::vector<size_t> input_mat_step(ndims);

        const auto &indices_mat_shape = shape(indices_mat);
        std::vector<size_t> indices_mat_step(ndims);

        for (int i = 0; i < ndims; i++) {
            input_mat_step[i] = static_cast<size_t>(input_mat.step.p[i] / sizeof(T));
            indices_mat_step[i] = static_cast<size_t>(indices_mat.step.p[i] / sizeof(T));
        }

        auto fn = [&](const Range &r) {
            size_t input_offset = 0, indices_offset = 0;

            int indices_index, index;
            size_t axis_offset, tmp_index, j_index;
            for (int i = r.start; i < r.end; i++) {
                const T* indices = indices_mat.ptr<const T>();
                const T* updates = updates_mat.ptr<const T>();
                T* output = output_mat.ptr<T>();

                input_offset = 0;
                indices_offset = 0;
                indices_index = i;
                axis_offset = 0;
                for (int j = ndims - 1; j >= 0; j--) {
                    tmp_index = indices_index / indices_mat_shape[j];
                    j_index = (size_t)(indices_index - tmp_index * indices_mat_shape[j]);
                    input_offset += j_index * input_mat_step[j];
                    indices_offset += j_index * indices_mat_step[j];
                    indices_index = tmp_index;
                    if (j == axis) {
                        axis_offset = j_index * input_mat_step[j];
                    }
                }

                // get index and overwrite current indices
                index = static_cast<int>(*(indices + indices_offset));
                index = (index + input_mat_shape[axis]) % input_mat_shape[axis];
                CV_Assert(index < input_mat_shape[axis] && index >= 0);
                input_offset = input_offset - axis_offset + index * input_mat_step[axis];

                updates += indices_offset;
                output += input_offset;
                *output = reduce_operation(*output, *updates);
            }
        };

        size_t total = indices_mat.total();
        double nstripes = (size_t)total * ndims * (1 / 1024.0);
        parallel_for_(Range(0, total), fn, nstripes);
    }

    template<typename... Args>
    inline void typeDispatch(const int type, Args&&... args)
    {
        switch (type)
        {
            case CV_8U:
                reductionDispatch<uint8_t>(std::forward<Args>(args)...);
                break;
            case CV_32S:
                reductionDispatch<int32_t>(std::forward<Args>(args)...);
                break;
            case CV_32F:
                reductionDispatch<float>(std::forward<Args>(args)...);
                break;
            default:
                CV_Error(cv::Error::BadDepth, "Unsupported type.");
        };
    }

    template<typename T, typename... Args>
    inline void reductionDispatch(Args&&... args)
    {
        switch (reduction)
        {
            case REDUCTION::NONE:
            {
                auto rd = [](const T& a, const T& b) { return b; }; // a from input data, b from updates
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            case REDUCTION::ADD:
            {
                auto rd = [](const T& a, const T& b) { return a + b; };
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            case REDUCTION::MUL:
            {
                auto rd = [](const T& a, const T& b) { return a * b; };
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            case REDUCTION::MAX:
            {
                auto rd = [](const T& a, const T& b) { return std::max(a, b); };
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            case REDUCTION::MIN:
            {
                auto rd = [](const T& a, const T& b) { return std::min(a, b); };
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            default:
                CV_Error(Error::StsBadArg, "Unsupported reduction.");
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

        auto axis_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, &axis);
        auto scatterElements = std::make_shared<ov::op::v3::ScatterElementsUpdate>(
            nodes[0].dynamicCast<InfEngineNgraphNode>()->node,
            indicesNonNegative,
            nodes[2].dynamicCast<InfEngineNgraphNode>()->node,
            axis_node);
        return Ptr<BackendNode>(new InfEngineNgraphNode(scatterElements));
    }
#endif  // HAVE_DNN_NGRAPH

private:
    // Attributes
    int axis;
};

Ptr<ScatterLayer> ScatterLayer::create(const LayerParams& params)
{
    return makePtr<ScatterLayerImpl>(params);
}

}} // namespace cv::dnn
