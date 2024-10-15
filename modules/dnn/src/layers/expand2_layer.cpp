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
    Expand layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Expand.html

    Opset's 8 to 13 are covered.
*/

class Expand2LayerImpl CV_FINAL : public Expand2Layer
{
public:
    Expand2LayerImpl(const LayerParams& params)
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
        size_t ninputs = this->inputs.size();
        CV_Assert(ninputs == 2);
        return !netimpl_->isConstArg(this->inputs[1]);
    }

    MatShape getOutShape(const MatShape& inpshape, const Mat& shapeTensor) const
    {
        MatShape shape0 = tensorToShape(shapeTensor);
        MatShape shape = inpshape.expand(shape0);
        // according to ONNX specification, the specified shape can be smaller than the input!
        // so we comment off the check
        // CV_Assert(shape == shape0); // check that input can be expanded to the specified shape
        return shape;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!dynamicOutputShapes());

        size_t ninputs = inputs.size();
        CV_Assert(ninputs == (size_t)2);
        Net::Impl* netimpl_ = getNetImpl(this);

        Mat shapeTensor = netimpl_->argTensor(this->inputs[1]);

        outputs.assign(1, getOutShape(inputs[0], shapeTensor));
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
        CV_Assert(ninputs == (size_t)2);
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
        CV_Assert(ninputs == 2);

        Mat inp = inputs_arr.getMat(0);
        int inptype = inp.type();
        Mat shapeTensor = inputs_arr.getMat(1);

        MatShape outshape = getOutShape(inp.shape(), shapeTensor);

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            broadcast(inp, outshape, outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            Mat temp(outshape, inptype);
            broadcast(inp, outshape, temp);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }
};

Ptr<Expand2Layer> Expand2Layer::create(const LayerParams& params)
{
    return Ptr<Expand2Layer>(new Expand2LayerImpl(params));
}

}
}
