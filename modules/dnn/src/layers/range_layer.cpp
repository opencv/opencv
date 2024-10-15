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

static int rangeSize(int64_t start, int64_t limit, int64_t delta)
{
    return delta > 0 ?
        std::max((int)((limit - start + delta - 1)/delta), 0) :
        std::max((int)((start - limit - delta - 1)/-delta), 0);
}

// out must be pre-allocated
template <typename _Tp>
static void makeRange(_Tp start, _Tp limit, _Tp delta, Mat& out)
{
    int nout = rangeSize(start, limit, delta);
    CV_Assert(out.dims == 1);
    CV_Assert(out.total() == (size_t)nout);
    uchar* outdata_ = out.data;

    int type = out.type();

    #undef IMPL_RANGE
    #define IMPL_RANGE(T) \
        T* outdata = (T*)outdata_; \
        for (int i = 0; i < nout; i++) \
            outdata[i] = saturate_cast<T>(start + i*delta)

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

    int getRangeParams(const Mat& startTensor, const Mat& limitTensor, const Mat& deltaTensor,
                       double& fstart, double& flimit, double& fdelta,
                       int64_t& istart, int64_t& ilimit, int64_t& idelta, bool& isflt) const
    {
        CV_Assert(startTensor.total() == (size_t)1);
        CV_Assert(limitTensor.total() == (size_t)1);
        CV_Assert(deltaTensor.total() == (size_t)1);

        int rtype = startTensor.type();
        CV_Assert(rtype == limitTensor.type());
        CV_Assert(rtype == deltaTensor.type());

        fstart = flimit = fdelta = 0.;
        istart = ilimit = idelta = 0;

        isflt = rtype == CV_32F || rtype == CV_64F || rtype == CV_16F || rtype == CV_16BF;

        if (isflt) {
            fstart = tensorToScalar<double>(startTensor);
            flimit = tensorToScalar<double>(limitTensor);
            fdelta = tensorToScalar<double>(deltaTensor);

            return rangeSize(fstart, flimit, fdelta);
        } else {
            istart = tensorToScalar<int64_t>(startTensor);
            ilimit = tensorToScalar<int64_t>(limitTensor);
            idelta = tensorToScalar<int64_t>(deltaTensor);

            return rangeSize(istart, ilimit, idelta);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!dynamicOutputShapes());

        CV_Assert(inputs.size() == (size_t)3);
        CV_Assert(inputs.size() == this->inputs.size());
        Net::Impl* netimpl_ = getNetImpl(this);

        Mat startTensor = netimpl_->argTensor(this->inputs[0]);
        Mat limitTensor = netimpl_->argTensor(this->inputs[1]);
        Mat deltaTensor = netimpl_->argTensor(this->inputs[2]);

        double fstart, flimit, fdelta;
        int64_t istart, ilimit, idelta;
        bool isflt;

        int nout = getRangeParams(startTensor, limitTensor, deltaTensor,
                                  fstart, flimit, fdelta, istart, ilimit, idelta, isflt);
        MatShape shape(1);
        shape[0] = nout;
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

        double fstart, flimit, fdelta;
        int64_t istart, ilimit, idelta;
        bool isflt;

        int nout = getRangeParams(startTensor, limitTensor, deltaTensor,
                                  fstart, flimit, fdelta, istart, ilimit, idelta, isflt);
        MatShape shape(1);
        shape[0] = nout;

        int rtype = startTensor.type();

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(shape, rtype);
            if (isflt) {
                makeRange(fstart, flimit, fdelta, outs[0]);
            } else {
                makeRange(istart, ilimit, idelta, outs[0]);
            }
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(shape, rtype);
            Mat temp(shape, rtype);
            if (isflt) {
                makeRange(fstart, flimit, fdelta, temp);
            } else {
                makeRange(istart, ilimit, idelta, temp);
            }
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
