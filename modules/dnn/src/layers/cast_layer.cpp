// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

#include "opencv-onnx.pb.h"

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
            Cv32suf u;
            u.f = in[i];
            outRaw[i] = (uint16_t)(u.u >> 16);
        }
    }
}

class CastLayerImpl CV_FINAL : public CastLayer
{
public:
    CastLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasToParam = false;
        toCvDepth_ = -1;
        if (params.has("to"))
        {
            hasToParam = true;
            toCvDepth_ = mapToCvDepth(params.get<int>("to"));
        }
        else if (params.has("outputType"))
        {
            const int v = params.get<int>("outputType");
            if (v == CV_Bool || v == CV_8U || v == CV_8S || v == CV_16U || v == CV_16S ||
                v == CV_32S || v == CV_64S || v == CV_32F || v == CV_64F || v == CV_16F || v == CV_16BF)
            {
                hasToParam = true;
                toCvDepth_ = v;
            }
            else
            {
                CV_Error(Error::StsNotImplemented, "Cast: unsupported 'outputType' value");
            }
        }
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
        CV_Check(inputs.size(), inputs.size() == 1 || inputs.size() == 2, "Cast expects 1 (Cast) or 2 (CastLike) inputs");
        outputs.assign(1, inputs[0]);
        return false;
    }

    virtual  void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Check(inputs.size(), !inputs.empty(), "Cast expects at least 1 input");

        int targetDepth = -1;
        if (hasToParam)
        {
            targetDepth = toCvDepth_;
        }
        else
        {
            Net::Impl* netimpl_ = getNetImpl(const_cast<CastLayerImpl*>(this));
            if (netimpl_ && this->inputs.size() >= 2)
            {
                const Arg& in1_arg = this->inputs[1];
                if (in1_arg.idx >= 0)
                {
                    const ArgData& ad = netimpl_->argData(in1_arg);
                    if (ad.type >= 0)
                        targetDepth = CV_MAT_DEPTH(ad.type);
                }
            }
        }

        if (targetDepth < 0)
        {
            targetDepth = CV_32F;
        }

        const int in0Type = inputs[0];
        const int in0CN   = in0Type >= 0 ? CV_MAT_CN(in0Type) : 1;
        int planDepth = targetDepth;
        if (planDepth == CV_16F)   planDepth = CV_32F;
        if (planDepth == CV_16BF)  planDepth = CV_16U;
        const int outType = CV_MAKETYPE(planDepth, in0CN);
        outputs.assign(1, outType);
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

        CV_Check(inputs.size(), inputs.size() == 1 || inputs.size() == 2, "Cast expects 1 (Cast) or 2 (CastLike) inputs");
        CV_CheckEQ(outputs.size(), (size_t)1, "");

        const Mat& src0 = inputs[0];
        Mat& dst0 = outputs[0];

        int runtimeTargetDepth = -1;
        if (hasToParam)
        {
            runtimeTargetDepth = toCvDepth_;
        }
        else
        {
            Net::Impl* netimpl_ = getNetImpl(this);
            if (netimpl_ && this->inputs.size() >= 2)
            {
                const Arg& in1_arg = this->inputs[1];
                const ArgData& ad = netimpl_->argData(in1_arg);
                if (ad.type >= 0)
                    runtimeTargetDepth = CV_MAT_DEPTH(ad.type);
            }
            if (runtimeTargetDepth < 0 && inputs.size() >= 2 && !inputs[1].empty())
                runtimeTargetDepth = inputs[1].depth();
            if (runtimeTargetDepth < 0)
                runtimeTargetDepth = src0.depth();
        }
        CV_CheckGE(runtimeTargetDepth, 0, "Cast: failed to resolve target data type at runtime");

        int plannedDDepth = (runtimeTargetDepth == CV_16F) ? CV_32F :
                            (runtimeTargetDepth == CV_16BF ? CV_16U : runtimeTargetDepth);
        if (dst0.depth() != plannedDDepth)
            dst0.create(dst0.size(), CV_MAKETYPE(plannedDDepth, src0.channels()));

        Mat src = src0.isContinuous() ? src0 : src0.clone();
        Mat dst = dst0.isContinuous() ? dst0 : dst0.clone();

        const int sdepth = src.depth();
        const int ddepth = dst.depth();

        if (sdepth == runtimeTargetDepth && !(runtimeTargetDepth == CV_16F && ddepth == CV_32F))
        {
            src0.copyTo(dst0);
            return;
        }

        if (runtimeTargetDepth == CV_16BF && (ddepth == CV_16BF || ddepth == CV_16U))
        {
            convertToBF16(src, dst);
        }
        else if (sdepth == CV_16BF)
        {
            src.convertTo(dst, ddepth);
        }
        else if (runtimeTargetDepth == CV_16F && ddepth == CV_32F)
        {
            Mat tmp16;
            src.convertTo(tmp16, CV_16F);
            tmp16.convertTo(dst, CV_32F);
        }
        else if (runtimeTargetDepth == CV_64F && ddepth != CV_64F)
        {
            Mat tmp64;
            src.convertTo(tmp64, CV_64F);
            if (ddepth == CV_32F)
                tmp64.convertTo(dst, CV_32F);
            else if (ddepth == CV_16U || ddepth == CV_16BF)
            {
                Mat tmp32;
                tmp64.convertTo(tmp32, CV_32F);
                convertToBF16(tmp32, dst);
            }
            else
                tmp64.convertTo(dst, ddepth);
        }
        else
        {
            src.convertTo(dst, ddepth);
        }

        if (dst.data != dst0.data)
            dst.copyTo(dst0);
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        ov::element::Type dstType;
        if (hasToParam)
        {
            dstType = cvTypeToOvType(CV_MAKETYPE(toCvDepth_, 1));
        }
        else if (nodes.size() >= 2)
        {
            dstType = nodes[1].dynamicCast<InfEngineNgraphNode>()->node.get_element_type();
        }
        else
        {
            dstType = nodes[0].dynamicCast<InfEngineNgraphNode>()->node.get_element_type();
        }
        auto cast = std::make_shared<ov::op::v0::Convert>(nodes[0].dynamicCast<InfEngineNgraphNode>()->node, dstType);
        return Ptr<BackendNode>(new InfEngineNgraphNode(cast));
    }
#endif  // HAVE_DNN_NGRAPH

private:
    bool hasToParam = false;
    int  toCvDepth_ = -1;

    static int mapToCvDepth(int v)
    {
        switch (v)
        {
            case opencv_onnx::TensorProto_DataType_FLOAT:    return CV_32F;
            case opencv_onnx::TensorProto_DataType_UINT8:    return CV_8U;
            case opencv_onnx::TensorProto_DataType_INT8:     return CV_8S;
            case opencv_onnx::TensorProto_DataType_UINT16:   return CV_16U;
            case opencv_onnx::TensorProto_DataType_INT16:    return CV_16S;
            case opencv_onnx::TensorProto_DataType_INT32:    return CV_32S;
            case opencv_onnx::TensorProto_DataType_INT64:    return CV_64S;
            case opencv_onnx::TensorProto_DataType_BOOL:     return CV_Bool;
            case opencv_onnx::TensorProto_DataType_FLOAT16:  return CV_16F;
            case opencv_onnx::TensorProto_DataType_DOUBLE:   return CV_64F;
            case opencv_onnx::TensorProto_DataType_BFLOAT16: return CV_16BF;
            default: break;
        }

        CV_Error(Error::StsNotImplemented, "Cast: unsupported 'to' / dtype value");
    }

    int resolveTargetDepthAtTypeTime(const std::vector<MatType>& inputs) const
    {
        if (hasToParam)
            return toCvDepth_;
        if (inputs.size() == 2)
        {
            int likeType = inputs[1];
            if (likeType >= 0)
                return CV_MAT_DEPTH(likeType);
            return -1;
        }
        return CV_MAT_DEPTH(inputs[0]);
    }
};

Ptr<CastLayer> CastLayer::create(const LayerParams& params)
{
    return makePtr<CastLayerImpl>(params);
}

}}  // namespace cv::dnn
