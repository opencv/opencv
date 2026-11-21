// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
#include "../op_cuda.hpp"

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/dropout_mask.hpp"
#endif

namespace cv {
namespace dnn {

class DropoutMaskLayerImpl CV_FINAL : public DropoutMaskLayer
{
public:
    DropoutMaskLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA) {
            Net::Impl* netimpl_ = getNetImpl(this);
            if (!netimpl_ || inputs.empty())
                return false;
            // an intermediate arg has no type until the first forward; refuse only a known unsupported one
            const int t = netimpl_->argData(inputs[0]).type;
            return t < 0 || isCudaNodeType(t);
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void* context_,
                              InputArrayOfArrays inputs_arr,
                              InputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        CV_UNUSED(outputs_arr);
        auto context = reinterpret_cast<cuda4dnn::csl::CSLContext*>(context_);
        const int depth = inputs_arr.depth(0);
        if (depth == CV_Bool)
            return make_cuda_node_bool<cuda4dnn::DropoutMaskOp>(std::move(context->stream));
        return make_cuda_node_with_type<cuda4dnn::DropoutMaskOp>(preferableTarget, depth, std::move(context->stream));
    }
#endif

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty() && requiredOutputs >= 2);
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty() && requiredOutputs >= 2);
        outputs.assign(requiredOutputs, MatType(CV_Bool));
        outputs[0] = inputs[0];
        internals.clear();
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(!inputs.empty() && outputs.size() >= 2);
        const Mat& x = inputs[0];
        if (outputs[0].data != x.data)
            x.copyTo(outputs[0]);

        for (size_t i = 1; i < outputs.size(); ++i)
        {
            outputs[i].setTo(true);
        }
    }
};

Ptr<DropoutMaskLayer> DropoutMaskLayer::create(const LayerParams& params)
{
    return Ptr<DropoutMaskLayer>(new DropoutMaskLayerImpl(params));
}

}}
