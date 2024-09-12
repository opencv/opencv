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
    Squeeze layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Squeeze.html

    Opset's 1 to 13 are covered.
    
    See description in reshape2_layer.cpp
    for more some common implementation details.
*/
class SqueezeLayerImpl CV_FINAL : public SqueezeLayer
{
public:
    SqueezeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axes = params.getVector<int>("axes");
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = reinterpret_cast<Net::Impl*>(netimpl);
        return inputs.size() == 2 && !netimpl_->isConstArg(inputs[1]);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const MatShape& inpShape, const std::vector<int>& axes_) const
    {
        bool squeezeMask[MatShape::MAX_DIMS];

        if (axes_.empty()) {
            // remove all 1's
            for (int i = 0; i < inpShape.dims; i++)
                squeezeMask[i] = inpShape[i] == 1;
        } else {
            for (int i = 0; i < inpShape.dims; i++)
                squeezeMask[i] = false;
            for (int a: axes_) {
                int a_ = normalize_axis(a, inpShape.dims);
                if (squeezeMask[a_]) {
                    CV_Error_(Error::StsBadArg, ("duplicate squeezed axis #%d", a));
                }
                if (inpShape[a_] != 1) {
                    CV_Error_(Error::StsBadArg, ("squeezed axis #%d (== %d) != 1", a, inpShape[a_]));
                }
                squeezeMask[a_] = true;
            }
        }

        MatShape outShape(inpShape.dims);
        int j = 0;
        for (int i = 0; i < inpShape.dims; i++) {
            if (!squeezeMask[i])
                outShape[j++] = inpShape[i];
        }
        outShape.dims = j;
        return outShape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1 || inputs.size() == 2);
        MatShape outShape;
        std::vector<int> tempAxes;
        const std::vector<int>* axes_ = &axes;

        if (inputs.size() == 2)
        {
            CV_Assert(axes.empty()); // if we have a dedicated 'axes' input,
                                     // we should not have 'axes' attribute at the same time
            Net::Impl* netimpl_ = reinterpret_cast<Net::Impl*>(netimpl);
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
        CV_Assert(ninputs == 1 || ninputs == 2);

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

Ptr<SqueezeLayer> SqueezeLayer::create(const LayerParams& params)
{
    return Ptr<SqueezeLayer>(new SqueezeLayerImpl(params));
}

}
}
