// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"


namespace cv { namespace dnn {

class NotLayerImpl CV_FINAL : public NotLayer
{
public:
    NotLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
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
        CV_CheckEQ(inputs.size(), (size_t)1, "");
        outputs.assign(1, inputs[0]);
        return true;
    }

    virtual  void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        int t = inputs[0];
        bool isInt = (t == CV_8S || t == CV_8U || t == CV_16S || t == CV_16U || t == CV_32S || t == CV_32U || t == CV_64S || t == CV_64U);
        CV_CheckType(inputs[0], t == CV_Bool || isInt, "Not/BitwiseNot expects bool or integer tensor");
        outputs.assign(1, t);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_Assert(inputs[0].isContinuous());
        CV_Assert(outputs[0].isContinuous());

        int t = inputs[0].type();
        size_t size = inputs[0].total();
        if (t == CV_Bool)
        {
            bool* input = inputs[0].ptr<bool>();
            bool* output = outputs[0].ptr<bool>();
            for (size_t i = 0; i < size; ++i)
                output[i] = !input[i];
            return;
        }

        switch (t)
        {
            case CV_8U: {
                const uint8_t* in = inputs[0].ptr<uint8_t>();
                uint8_t* out = outputs[0].ptr<uint8_t>();
                for (size_t i = 0; i < size; ++i) out[i] = static_cast<uint8_t>(~in[i]);
                break; }
            case CV_8S: {
                const int8_t* in = inputs[0].ptr<int8_t>();
                int8_t* out = outputs[0].ptr<int8_t>();
                for (size_t i = 0; i < size; ++i) out[i] = static_cast<int8_t>(~in[i]);
                break; }
            case CV_16U: {
                const uint16_t* in = inputs[0].ptr<uint16_t>();
                uint16_t* out = outputs[0].ptr<uint16_t>();
                for (size_t i = 0; i < size; ++i) out[i] = static_cast<uint16_t>(~in[i]);
                break; }
            case CV_16S: {
                const int16_t* in = inputs[0].ptr<int16_t>();
                int16_t* out = outputs[0].ptr<int16_t>();
                for (size_t i = 0; i < size; ++i) out[i] = static_cast<int16_t>(~in[i]);
                break; }
            case CV_32U: {
                const uint32_t* in = inputs[0].ptr<uint32_t>();
                uint32_t* out = outputs[0].ptr<uint32_t>();
                for (size_t i = 0; i < size; ++i) out[i] = static_cast<uint32_t>(~in[i]);
                break; }
            case CV_32S: {
                const int32_t* in = inputs[0].ptr<int32_t>();
                int32_t* out = outputs[0].ptr<int32_t>();
                for (size_t i = 0; i < size; ++i) out[i] = static_cast<int32_t>(~in[i]);
                break; }
            case CV_64U: {
                const uint64_t* in = inputs[0].ptr<uint64_t>();
                uint64_t* out = outputs[0].ptr<uint64_t>();
                for (size_t i = 0; i < size; ++i) out[i] = static_cast<uint64_t>(~in[i]);
                break; }
            case CV_64S: {
                const int64_t* in = inputs[0].ptr<int64_t>();
                int64_t* out = outputs[0].ptr<int64_t>();
                for (size_t i = 0; i < size; ++i) out[i] = static_cast<int64_t>(~in[i]);
                break; }
            default:
                CV_Error(Error::BadDepth, "Not/BitwiseNot expects bool or integer tensor");
        }
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto node = std::make_shared<ov::op::v1::LogicalNot>(nodes[0].dynamicCast<InfEngineNgraphNode>()->node);
        return Ptr<BackendNode>(new InfEngineNgraphNode(node));
    }
#endif  // HAVE_DNN_NGRAPH

};

Ptr<NotLayer> NotLayer::create(const LayerParams& params)
{
    return makePtr<NotLayerImpl>(params);
}

}}  // namespace cv::dnn
