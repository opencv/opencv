// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

namespace cv
{
namespace dnn
{

/*
    Range layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Range.html

    Opset's 11 to 11 are covered.
*/

static int rangeSize(double start, double limit, double delta)
{
    return std::max((int)ceil((limit - start)/delta), 0);
}

// out must be pre-allocated
static void makeRange(double start, double limit, double delta, Mat& out)
{
    int i, nout = rangeSize(start, limit, delta);
    CV_Assert(out.dims == 1);
    CV_Assert(out.total() == (size_t)nout);
    uchar* outdata_ = out.data;

    int type = out.type();

    #undef IMPL_RANGE
    #define IMPL_RANGE(T) \
        T* outdata = (T*)outdata_; \
        for (int i = 0; i < nout; i++) \
            outdata[i] = T(start + i*delta)

    if (type == CV_32F) {
        IMPL_RANGE(float);
    } else if (type == CV_64F) {
        IMPL_RANGE(double);
    } else if (type == CV_32S) {
        IMPL_RANGE(int32_t);
    } else if (type == CV_64S) {
        IMPL_RANGE(int64_t);
    } else {
        CV_Error_(Error::StsNotImplemented, ("invalid/unsupported tensor type: %s", typeToString(out.type()).c_str()));
    }
}

class RangeLayerImpl CV_FINAL : public RangeLayer
{
public:
    RangeLayerImpl(const LayerParams& params)
    {
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        CV_Assert(netimpl_);
        CV_Assert(this->inputs.size() == 3);
        return  !netimpl_->isConstArg(this->inputs[0]) ||
                !netimpl_->isConstArg(this->inputs[1]) ||
                !netimpl_->isConstArg(this->inputs[2]);
    }

    bool getMemoryShapes(const std::vector<MatShape>&,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!dynamicOutputShapes());

        CV_Assert(this->inputs.size() == (size_t)3);
        Net::Impl* netimpl_ = getNetImpl(this);
        double start = tensorToScalar<double>(netimpl_->argTensor(this->inputs[0]));
        double limit = tensorToScalar<double>(netimpl_->argTensor(this->inputs[1]));
        double delta = tensorToScalar<double>(netimpl_->argTensor(this->inputs[2]));
        MatShape shape(1);
        shape[0] = rangeSize(start, limit, delta);
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
        size_t ninputs = inputs.size();
        CV_Assert(ninputs == (size_t)3);
        CV_Assert(inputs[0] == inputs[1]);
        CV_Assert(inputs[0] == inputs[2]);
        outputs.assign(requiredOutputs, inputs[0]);
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
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        CV_Assert(ninputs == 3);

        Mat startTensor = inputs_arr.getMat(0);
        Mat limitTensor = inputs_arr.getMat(1);
        Mat deltaTensor = inputs_arr.getMat(2);

        CV_Assert(startTensor.total() == (size_t)1);
        CV_Assert(limitTensor.total() == (size_t)1);
        CV_Assert(deltaTensor.total() == (size_t)1);

        int type = startTensor.type();
        CV_Assert(type == limitTensor.type());
        CV_Assert(type == deltaTensor.type());

        double start = tensorToScalar<double>(startTensor);
        double limit = tensorToScalar<double>(limitTensor);
        double delta = tensorToScalar<double>(deltaTensor);

        MatShape shape(1);
        shape[0] = rangeSize(start, limit, delta);

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(shape, type);
            makeRange(start, limit, delta, outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(shape, type);
            Mat temp(shape, type);
            makeRange(start, limit, delta, temp);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }
};

Ptr<RangeLayer> RangeLayer::create(const LayerParams& params)
{
    return Ptr<RangeLayer>(new RangeLayerImpl(params));
}

}
}

