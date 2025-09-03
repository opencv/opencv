// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "layers_common.hpp"

namespace cv { namespace dnn {

namespace
{
    inline void convertToBF16(const Mat& src, Mat& dst)
    {
        const int ddepth = dst.depth();
        if (!(ddepth == CV_16BF || ddepth == CV_16U))
        {
            CV_Error(Error::StsNotImplemented, "Unsupported destination depth for BF16 cast");
        }

        Mat src32;
        if (src.depth() != CV_32F)
            src.convertTo(src32, CV_32F);
        const Mat& s = src32.empty() ? src : src32;

        uint16_t* outRaw = NULL;
        Mat dst_bf16;
        if (ddepth == CV_16BF)
        {
            outRaw = reinterpret_cast<uint16_t*>(dst.ptr<bfloat>());
        }
        else
        {
            dst_bf16 = Mat(dst.size(), CV_MAKETYPE(CV_16BF, dst.channels()), dst.data, dst.step);
            outRaw = dst_bf16.ptr<uint16_t>();
        }

        const float* in = s.ptr<float>();
        size_t numElems = (size_t)s.total() * (size_t)s.channels();
        for (size_t i = 0; i < numElems; ++i)
        {
            uint32_t bits;
            memcpy(&bits, &in[i], sizeof(bits));
            outRaw[i] = (uint16_t)(bits >> 16);
        }
    }
}

class CastLayerImpl CV_FINAL : public CastLayer
{
public:
    CastLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        outputType = params.get<int>("outputType");
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
        return false;
    }

    virtual  void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        if (outputType == CV_16F)
            outputs.assign(1, CV_32F);
        else if (outputType == CV_16BF)
            outputs.assign(1, CV_16U);
        else
            outputs.assign(1, outputType);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs, outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        CV_CheckEQ(inputs.size(), (size_t)1, "");
        CV_CheckEQ(outputs.size(), (size_t)1, "");

        if (inputs[0].depth() == outputs[0].depth())
            inputs[0].copyTo(outputs[0]);
        else
            inputs[0].convertTo(outputs[0], outputs[0].depth());
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
            forward_ocl(inputs_arr, outputs_arr, internals_arr));

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_CheckEQ(inputs.size(), (size_t)1, "");
        CV_CheckEQ(outputs.size(), (size_t)1, "");

        const Mat& src0 = inputs[0];
        Mat& dst0 = outputs[0];

        if (src0.type() == dst0.type())
        {
            if (!(outputType == CV_16F && dst0.depth() == CV_32F))
            {
                src0.copyTo(dst0);
                return;
            }
        }

        Mat src = src0.isContinuous() ? src0 : src0.clone();
        Mat dst = dst0.isContinuous() ? dst0 : dst0.clone();

        const int sdepth = src.depth();
        const int ddepth = dst.depth();

        if (outputType == CV_16BF && (ddepth == CV_16BF || ddepth == CV_16U))
        {
            convertToBF16(src, dst);
        }
        else if (sdepth == CV_16BF)
        {
            src.convertTo(dst, ddepth);
        }
        else
        {
            src.convertTo(dst, ddepth);
        }

        if (outputType == CV_16F && ddepth == CV_32F)
        {
            Mat tmp16;
            src.convertTo(tmp16, CV_16F);
            tmp16.convertTo(dst, CV_32F);
        }

        if (dst.data != dst0.data)
            dst.copyTo(dst0);
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto cast = std::make_shared<ov::op::v0::Convert>(nodes[0].dynamicCast<InfEngineNgraphNode>()->node, cvTypeToOvType(outputType));
        return Ptr<BackendNode>(new InfEngineNgraphNode(cast));
    }
#endif  // HAVE_DNN_NGRAPH

private:
    int outputType;
};

Ptr<CastLayer> CastLayer::create(const LayerParams& params)
{
    return makePtr<CastLayerImpl>(params);
}

}}  // namespace cv::dnn
