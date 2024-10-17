// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
//#include "../op_cuda.hpp"
//#include "../op_inf_engine.hpp"
//#include "../ie_ngraph.hpp"
//#include "../op_webnn.hpp"
//#include "../op_timvx.hpp"
//#include "../op_cann.hpp"

//#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

/*
    Unsqueeze layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Unsqueeze.html

    Opset's 1 to 23 are covered.

    See description in reshape2_layer.cpp
    for more some common implementation details.
*/
class UnsqueezeLayerImpl CV_FINAL : public UnsqueezeLayer
{
public:
    UnsqueezeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axes = params.getVector<int>("axes");
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        return inputs.size() == 2 && !netimpl_->isConstArg(inputs[1]);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const MatShape& inpShape, const std::vector<int>& axes_) const
    {
        bool unsqueezeMask[MatShape::MAX_DIMS];

        int outDims = inpShape.dims + (int)axes_.size();
        CV_Assert(0 <= outDims && outDims <= MatShape::MAX_DIMS);

        for (int i = 0; i < outDims; i++)
            unsqueezeMask[i] = false;
        for (int a: axes_) {
            int a_ = normalize_axis(a, outDims);
            if (unsqueezeMask[a_]) {
                CV_Error_(Error::StsBadArg, ("duplicate unsqueezed axis #%d", a));
            }
            unsqueezeMask[a_] = true;
        }

        MatShape outShape(outDims);
        int j = 0;
        for (int i = 0; i < outDims; i++) {
            if (unsqueezeMask[i])
                outShape[i] = 1;
            else {
                CV_Assert(j < inpShape.dims);
                outShape[i] = inpShape[j++];
            }
        }
        return outShape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert((inputs.size() == 1 && !axes.empty()) ||
                  (inputs.size() == 2 && axes.empty()));
        MatShape outShape;
        std::vector<int> tempAxes;
        const std::vector<int>* axes_ = &axes;

        if (inputs.size() == 2)
        {
            Net::Impl* netimpl_ = getNetImpl(this);
            Mat axesTensor = netimpl_->argTensor(this->inputs[1]);
            tensorToIntVec(axesTensor, tempAxes);
            axes_ = &tempAxes;
        }
        outputs.assign(1, getOutShape(inputs[0], *axes_));
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
        CV_Assert(ninputs == 1 || ninputs == 2);
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
        CV_Assert((ninputs == 1 && !axes.empty()) ||
                  (ninputs == 2 && axes.empty()));

        MatShape inpShape = inputs_arr.shape(0);
        std::vector<int> tempAxes;
        const std::vector<int>* axes_ = &axes;

        if (ninputs == 2)
        {
            CV_Assert(axes.empty()); // if we have a dedicated 'axes' input,
                                     // we should not have 'axes' attribute at the same time
            Mat axesTensor = inputs_arr.getMat(1);
            tensorToIntVec(axesTensor, tempAxes);
            axes_ = &tempAxes;
        }
        MatShape outShape = getOutShape(inpShape, *axes_);
        reshapeAndCopyFirst(inputs_arr, outputs_arr, outShape);
    }
};

Ptr<UnsqueezeLayer> UnsqueezeLayer::create(const LayerParams& params)
{
    return Ptr<UnsqueezeLayer>(new UnsqueezeLayerImpl(params));
}

}
}
