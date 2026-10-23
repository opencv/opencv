// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/constant_of_shape.hpp"
#endif

namespace cv
{
namespace dnn
{

/*
    ConstantOfShape layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html

    Opset's 9 to 23 are covered.
*/

// out must be pre-allocated
static void constantOfShape(const Mat& value, Mat& out)
{
    CV_Assert(value.total() == 1);
    CV_Assert(out.isContinuous());
    CV_CheckEQ(value.type(), out.type(), "input and output tensor types must be the same");

    size_t esz = value.elemSize();
    size_t total = out.total();
    const uchar* inpdata_ = value.data;
    uchar* outdata_ = out.data;

    #undef IMPL_CONST_OF_SHAPE
    #define IMPL_CONST_OF_SHAPE(T) \
        T val = *(const T*)inpdata_; \
        T* outdata = (T*)outdata_; \
        for (size_t i = 0; i < total; i++) \
            outdata[i] = val

    if (esz == 1) {
        IMPL_CONST_OF_SHAPE(uint8_t);
    } else if (esz == 2) {
        IMPL_CONST_OF_SHAPE(uint16_t);
    } else if (esz == 4) {
        IMPL_CONST_OF_SHAPE(uint32_t);
    } else if (esz == 8) {
        IMPL_CONST_OF_SHAPE(uint64_t);
    } else {
        CV_Error_(Error::StsNotImplemented, ("invalid/unsupported tensor type: %s", typeToString(value.type()).c_str()));
    }
}

class ConstantOfShapeLayerImpl CV_FINAL : public ConstantOfShapeLayer
{
public:
    ConstantOfShapeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA) {
            if (blobs.size() != 1 || blobs[0].total() != 1)
                return false;
            // make_cuda_node_with_type() hard-errors outside these types; bool goes through make_cuda_node_bool()
            return isCudaNodeType(blobs[0].type());
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void* context_,
                              InputArrayOfArrays,
                              InputArrayOfArrays) CV_OVERRIDE
    {
        auto context = reinterpret_cast<cuda4dnn::csl::CSLContext*>(context_);
        CV_Assert(blobs.size() == 1);
        const Mat& value = blobs[0];

        if (value.type() == CV_Bool)
            return make_cuda_node_bool<cuda4dnn::ConstantOfShapeOp>(std::move(context->stream), value);
        // the FP16 target stores float tensors as half, and the op reads the value's raw bytes as T
        if (value.type() == CV_32F && preferableTarget == DNN_TARGET_CUDA_FP16) {
            Mat halfValue;
            value.convertTo(halfValue, CV_16F);
            return Ptr<BackendNode>(new cuda4dnn::ConstantOfShapeOp<half>(std::move(context->stream), halfValue));
        }
        return make_cuda_node_with_type<cuda4dnn::ConstantOfShapeOp>(preferableTarget, value.type(),
                                                                     std::move(context->stream), value);
    }
#endif

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        CV_Assert(netimpl_);
        CV_Assert(this->inputs.size() == 1);
        return !netimpl_->isConstArg(this->inputs[0]);
    }

    bool getMemoryShapes(const std::vector<MatShape>&,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(this->inputs.size() == (size_t)1);
        Net::Impl* netimpl_ = getNetImpl(this);
        Mat shapeTensor = netimpl_->argTensor(this->inputs[0]);
        // a dynamic shape is readable once its values are on the host (CUDA downloads them first)
        CV_Assert(!dynamicOutputShapes() || !shapeTensor.empty());
        MatShape shape = tensorToShape(shapeTensor);
        outputs.assign(1, shape);
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(blobs.size() == 1);
        size_t ninputs = inputs.size();
        CV_Assert(ninputs == (size_t)1);
        outputs.assign(requiredOutputs, blobs[0].type());
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_Assert(blobs.size() == 1);
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        CV_Assert(ninputs == 1);

        const Mat& value = blobs[0];
        Mat shapeTensor = inputs_arr.getMat(0);
        MatShape shape = tensorToShape(shapeTensor);

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(shape, value.type());
            constantOfShape(value, outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(shape, value.type());
            Mat temp(shape, value.type());
            constantOfShape(value, temp);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }
};

Ptr<ConstantOfShapeLayer> ConstantOfShapeLayer::create(const LayerParams& params)
{
    return Ptr<ConstantOfShapeLayer>(new ConstantOfShapeLayerImpl(params));
}

}
}
