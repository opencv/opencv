// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

#include "opencv-onnx.pb.h"
#include "../onnx/onnx_dtype_convert.hpp"

namespace cv { namespace dnn {

// ONNX Cast operator
// Spec: https://onnx.ai/onnx/operators/onnx__Cast.html
// Supported opsets: 1-24
// ONNX CastLike operator
// Spec: https://onnx.ai/onnx/operators/onnx__CastLike.html
// Supported opsets: 15-24

namespace
{
    template<typename DT>
    inline void truncateToIntImpl(const Mat& src, Mat& dst)
    {
        const int n = (int)src.total() * src.channels();
        DT* d = dst.ptr<DT>();
        if (src.depth() == CV_16F)
        {
            const hfloat* s = src.ptr<hfloat>();
            for (int i = 0; i < n; ++i)
                d[i] = saturate_cast<DT>(std::trunc((float)s[i]));
        }
        else if (src.depth() == CV_32F)
        {
            const float* s = src.ptr<float>();
            for (int i = 0; i < n; ++i)
                d[i] = saturate_cast<DT>(std::trunc(s[i]));
        }
        else
        {
            const double* s = src.ptr<double>();
            for (int i = 0; i < n; ++i)
                d[i] = saturate_cast<DT>(std::trunc(s[i]));
        }
    }

    inline void truncateFloatToInt(const Mat& src, Mat& dst)
    {
        switch (dst.depth())
        {
            case CV_8U:   truncateToIntImpl<uchar>(src, dst);   break;
            case CV_8S:   truncateToIntImpl<schar>(src, dst);   break;
            case CV_16U:  truncateToIntImpl<ushort>(src, dst);  break;
            case CV_16S:  truncateToIntImpl<short>(src, dst);   break;
            case CV_32S:  truncateToIntImpl<int>(src, dst);     break;
            case CV_64S:  truncateToIntImpl<int64_t>(src, dst); break;
            default:      src.convertTo(dst, dst.depth());      break;
        }
    }

}

class Cast2LayerImpl CV_FINAL : public Cast2Layer
{
public:
    Cast2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasToParam = false;
        toCvDepth_ = -1;
        toOnnxType_ = -1;
        saturate_ = params.get<int>("saturate", 1) != 0;
        if (params.has("to"))
        {
            hasToParam = true;
            toOnnxType_ = params.get<int>("to");
            toCvDepth_ = mapToCvDepth(toOnnxType_);
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
        // Exotic dtypes (FP8/FP4/INT4/UINT4) are handled on the CPU path only.
        if (onnx_dtype::isExotic(toOnnxType_))
            return backendId == DNN_BACKEND_OPENCV;
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

    // Half targets are stored as FP32 unless native FP16 is enabled; forward() still rounds.
    int resolveStorageDepth(int targetDepth, bool exotic) const
    {
        if (!exotic && (targetDepth == CV_16F || targetDepth == CV_16BF))
        {
            Net::Impl* ni = getNetImpl(const_cast<Cast2LayerImpl*>(this));
            if (ni && !ni->enableFP16)
                return CV_32F;
        }
        return targetDepth;
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
            Net::Impl* netimpl_ = getNetImpl(const_cast<Cast2LayerImpl*>(this));
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
        const bool exotic = hasToParam && onnx_dtype::isExotic(toOnnxType_);
        const int outType = CV_MAKETYPE(resolveStorageDepth(targetDepth, exotic), in0CN);
        outputs.assign(1, outType);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs, outputs;

        if (hasToParam && onnx_dtype::isExotic(toOnnxType_))
            return false; // exotic conversions run on the CPU path

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        CV_CheckEQ(inputs.size(), (size_t)1, "");
        CV_CheckEQ(outputs.size(), (size_t)1, "");

        int runtimeTargetDepth = -1;
        if (hasToParam)
        {
            runtimeTargetDepth = toCvDepth_;
        }
        else
        {
            if (inputs.size() >= 2 && !inputs[1].empty())
                runtimeTargetDepth = inputs[1].depth();
            else
                runtimeTargetDepth = inputs[0].depth();
        }

        if (runtimeTargetDepth == CV_16F && outputs[0].depth() == CV_32F)
        {
            return false;
        }

        // bf16 needs the float->bfloat16 bit reduction; a plain OpenCL convertTo
        // would numerically round instead. Fall back to the CPU path.
        if (runtimeTargetDepth == CV_16BF)
        {
            return false;
        }

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

        const bool exotic = hasToParam && onnx_dtype::isExotic(toOnnxType_);
        const int storeDepth = resolveStorageDepth(runtimeTargetDepth, exotic);
        if (dst0.depth() != storeDepth)
            dst0.create(dst0.size(), CV_MAKETYPE(storeDepth, src0.channels()));

        Mat src = src0;
        Mat dst = dst0;

        if (exotic)
        {
            castExotic(src, dst, toOnnxType_, saturate_);
            return;
        }

        // Cast to half yields half-representable values even when FP32 carries them.
        if (storeDepth != runtimeTargetDepth &&
            (runtimeTargetDepth == CV_16F || runtimeTargetDepth == CV_16BF))
        {
            Mat half;
            src.convertTo(half, runtimeTargetDepth);
            half.convertTo(dst, storeDepth);
            return;
        }

        const int sdepth = src.depth();
        const int ddepth = dst.depth();

        if (sdepth == ddepth && sdepth == runtimeTargetDepth)
        {
            src0.copyTo(dst0);
            return;
        }

        if ((sdepth == CV_16F || sdepth == CV_32F || sdepth == CV_64F) && CV_IS_INT_TYPE(ddepth))
        {
            truncateFloatToInt(src, dst);       // ONNX float->int truncates toward zero
        }
        else
        {
            src.convertTo(dst, ddepth);
        }
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

    void castExotic(const Mat& src, Mat& dst, int onnxType, bool saturate)
    {
        const int sdepth = src.depth();
        const float*  sf = (sdepth == CV_32F) ? src.ptr<float>()  : nullptr;
        const hfloat* sh = (sdepth == CV_16F) ? src.ptr<hfloat>() : nullptr;
        Mat src32;
        if (!sf && !sh) { src.convertTo(src32, CV_32F); sf = src32.ptr<float>(); }
        const size_t total = src.total() * src.channels();
        #define CV_DNN_SRC_F(i) (sf ? sf[i] : (float)sh[i])

        if (onnx_dtype::isFp8(onnxType))
        {
            const onnx_dtype::Fp8Fmt fmt = onnx_dtype::fp8FmtFor(onnxType);
            const int ddepth = dst.depth();
            if (ddepth == CV_8F_E4M3FN || ddepth == CV_8F_E4M3FNUZ)
            {
                // Store the ONNX-encoded byte: core's E4M3 encode rounds differently.
                uchar* d = dst.ptr<uchar>();
                for (size_t i = 0; i < total; i++)
                    d[i] = onnx_dtype::f32ToFp8(CV_DNN_SRC_F(i), fmt, saturate);
            }
            else
            {
                // E5M2/E5M2FNUZ have no native depth: round onto the FP8 grid, keep CV_16F.
                hfloat* d = dst.ptr<hfloat>();
                for (size_t i = 0; i < total; i++)
                    d[i] = hfloat(onnx_dtype::fp8ToF32(onnx_dtype::f32ToFp8(CV_DNN_SRC_F(i), fmt, saturate), fmt));
            }
        }
        else if (onnxType == onnx_dtype::ONNX_FLOAT8E8M0)
        {
            float* d = dst.ptr<float>();   // E8M0 range exceeds FP16, stays CV_32F
            for (size_t i = 0; i < total; i++)
                d[i] = onnx_dtype::e8m0ToF32(onnx_dtype::f32ToE8M0(CV_DNN_SRC_F(i)));
        }
        else if (onnxType == opencv_onnx::TensorProto_DataType_FLOAT4E2M1)
        {
            hfloat* d = dst.ptr<hfloat>();
            for (size_t i = 0; i < total; i++)
                d[i] = hfloat(onnx_dtype::fp4ToF32(onnx_dtype::f32ToFp4(CV_DNN_SRC_F(i))));
        }
        else if (onnx_dtype::isInt4(onnxType))
        {
            schar* d = dst.ptr<schar>();
            for (size_t i = 0; i < total; i++)
                d[i] = onnx_dtype::f32ToInt4(CV_DNN_SRC_F(i));
        }
        else // UINT4
        {
            uchar* d = dst.ptr<uchar>();
            for (size_t i = 0; i < total; i++)
                d[i] = onnx_dtype::f32ToUint4(CV_DNN_SRC_F(i));
        }
        #undef CV_DNN_SRC_F
    }

private:
    bool hasToParam = false;
    int  toCvDepth_ = -1;
    int  toOnnxType_ = -1;
    bool saturate_ = true;

    // ONNX TensorProto::DataType values (see opencv-onnx.proto); the 'to'
    // attribute stores the raw ONNX value. Fixed by the ONNX specification,
    // duplicated here to keep this layer independent of protobuf-generated headers.
    enum OnnxDataType
    {
        ONNX_DT_FLOAT    = 1,
        ONNX_DT_UINT8    = 2,
        ONNX_DT_INT8     = 3,
        ONNX_DT_UINT16   = 4,
        ONNX_DT_INT16    = 5,
        ONNX_DT_INT32    = 6,
        ONNX_DT_INT64    = 7,
        ONNX_DT_BOOL     = 9,
        ONNX_DT_FLOAT16  = 10,
        ONNX_DT_DOUBLE   = 11,
        ONNX_DT_BFLOAT16 = 16
    };

    static int mapToCvDepth(int v)
    {
        if (v == onnx_dtype::ONNX_FLOAT8E8M0) return CV_32F;                  // range exceeds FP16
        if (onnx_dtype::isFp8Native(v)) return onnx_dtype::fp8NativeDepth(v); // E4M3FN/E4M3FNUZ
        if (onnx_dtype::isExoticFloat(v)) return CV_16F;                     // E5M2/FP4
        if (onnx_dtype::isInt4(v))        return CV_8S;
        if (onnx_dtype::isUint4(v))       return CV_8U;
        switch (v)
        {
            case ONNX_DT_FLOAT:    return CV_32F;
            case ONNX_DT_UINT8:    return CV_8U;
            case ONNX_DT_INT8:     return CV_8S;
            case ONNX_DT_UINT16:   return CV_16U;
            case ONNX_DT_INT16:    return CV_16S;
            case ONNX_DT_INT32:    return CV_32S;
            case ONNX_DT_INT64:    return CV_64S;
            case ONNX_DT_BOOL:     return CV_Bool;
            case ONNX_DT_FLOAT16:  return CV_16F;
            case ONNX_DT_DOUBLE:   return CV_64F;
            case ONNX_DT_BFLOAT16: return CV_16BF;
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

Ptr<Cast2Layer> Cast2Layer::create(const LayerParams& params)
{
    return makePtr<Cast2LayerImpl>(params);
}

}}  // namespace cv::dnn
