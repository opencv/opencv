// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "layers_common.hpp"


namespace cv { namespace dnn {

class GatherLayerImpl CV_FINAL : public GatherLayer
{
public:
    GatherLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        m_axis = params.get<int>("axis", 0);
        m_real_ndims = params.get<int>("real_ndims", -1);
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
        CV_CheckEQ(inputs.size(), 2ull, "");
        MatShape inpShape = inputs[0];
        if (inpShape.size() == 0 ){
            outputs.assign(1, inpShape);
            return false;
        }

        const int axis = normalize_axis(m_axis, inpShape);
        if (!inpShape.empty())
            inpShape.erase(inpShape.begin() + axis);
        auto end = m_real_ndims == -1 ? inputs[1].end() : inputs[1].begin() + m_real_ndims;
        inpShape.insert(inpShape.begin() + axis, inputs[1].begin(), end);
        outputs.assign(1, inpShape);
        return false;
    }

    virtual  void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), (size_t)2, "");
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_32S || inputs[0] == CV_64S || inputs[0] == CV_16F || inputs[0] == CV_8U || inputs[0] == CV_8S || inputs[0] == CV_Bool, "");
        CV_CheckType(inputs[1], inputs[1] == CV_64S || inputs[1] == CV_32S, "");
        outputs.assign(1, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        // FP16 fallback is not needed as we handle FP16 below

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_CheckEQ(inputs.size(), (size_t)2, "");
        CV_CheckEQ(outputs.size(), (size_t)1, "");
        CV_CheckTypeEQ(inputs[0].type(), outputs[0].type(), "");

        if (inputs[1].type() == CV_32SC1)
            forward_impl<int32_t>(inputs[0], inputs[1], outputs[0]);
        else if (inputs[1].type() == CV_64SC1)
            forward_impl<int64_t>(inputs[0], inputs[1], outputs[0]);
        else
            CV_CheckType(inputs[1].type(), inputs[1].type() == CV_32SC1 || inputs[1].type() == CV_64SC1, "");
    }

    template<typename T_INDEX>
    void forward_impl(const Mat& inp, const Mat& indices, Mat& out)
    {

        const size_t indices_total = indices.total();
        const int axis = normalize_axis(m_axis, shape(inp));

        // FIXIT: why should we work with non-normalized input? it should be handled in importer or layers's output generator
        const int axis_size = (int)inp.size[axis];

        const size_t outer_size = axis == 0 ? inp.total() : inp.step1(axis - 1);
        const size_t outer_dims = inp.total() / outer_size;
        const size_t inner_size = inp.step1(axis);

        const T_INDEX* idx = indices.ptr<T_INDEX>();
        const char* src = inp.ptr<const char>();
        char* dst = out.ptr<char>();
        CV_CheckEQ(out.total(), outer_dims * indices_total * inner_size, "");

        const size_t es = inp.elemSize1();
        // TODO: optimize through switch (inner_size * es)
        const size_t inner_bytes = inner_size * es;
        for (size_t i = 0; i < outer_dims; ++i)
        {
            const size_t src_offset = i * outer_size;
            for (size_t j = 0 ; j < indices_total; ++j)
            {
                const int index = normalize_axis(idx[j], axis_size);
                CV_DbgCheck(index, index >= 0 && index < axis_size, "");
                const size_t new_offset = src_offset + index * inner_size;
                std::memcpy(dst, src + new_offset * es, inner_bytes);
                dst += inner_bytes;
            }
        }
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto axisNode = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, &m_axis);
        auto gather = std::make_shared<ov::op::v8::Gather>(
            nodes[0].dynamicCast<InfEngineNgraphNode>()->node,
            nodes[1].dynamicCast<InfEngineNgraphNode>()->node,
            axisNode);
        return Ptr<BackendNode>(new InfEngineNgraphNode(gather));
    }
#endif  // HAVE_DNN_NGRAPH

private:
    // The axis to gather along
    int m_axis;
    int m_real_ndims;
};

Ptr<GatherLayer> GatherLayer::create(const LayerParams& params)
{
    return makePtr<GatherLayerImpl>(params);
}

}}  // namespace cv::dnn
