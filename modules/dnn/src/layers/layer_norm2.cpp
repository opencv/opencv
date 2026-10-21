// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
#include "../op_cuda.hpp"
#include "cpu_kernels/fast_norm.hpp"

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/layer_norm.hpp"
#endif

namespace cv {
namespace dnn {

// ONNX LayerNormalization operator
// Spec: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
// Supported opsets: 17

class LayerNorm2LayerImpl CV_FINAL : public LayerNorm2Layer
{
public:
    int axis0;

    LayerNorm2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = axis0 = params.get<int>("axis", -1);
        epsilon = params.get<float>("epsilon", 1e-5f);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA) {
            // LayerNormOp writes Y only; the optional Mean/InvStdDev outputs stay on CPU
            Net::Impl* netimpl_ = getNetImpl(this);
            if (!netimpl_ || inputs.empty() || outputs.size() != 1)
                return false;
            // weights are either all constant blobs or all runtime inputs
            const bool constWeights = inputs.size() == 1 && (blobs.size() == 1 || blobs.size() == 2);
            const bool inputWeights = blobs.empty() && (inputs.size() == 2 || inputs.size() == 3);
            if (!constWeights && !inputWeights)
                return false;
            // an intermediate arg has no type until the first forward; refuse only a known non-float one
            for (const Arg& a : inputs) {
                const int t = netimpl_->argData(a).type;
                if (t >= 0 && t != CV_32F)
                    return false;
            }
            for (const Mat& b : blobs)
                if (b.type() != CV_32F)
                    return false;
            return true;
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void* context_,
                              InputArrayOfArrays inputs_arr,
                              InputArrayOfArrays) CV_OVERRIDE
    {
        auto context = reinterpret_cast<cuda4dnn::csl::CSLContext*>(context_);
        const MatShape inpShape = inputs_arr.shape(0);
        const int axis_ = normalize_axis(axis0, inpShape.dims);
        const size_t loops = static_cast<size_t>(total(inpShape, 0, axis_));

        // a single blob is the scale only; do not reuse it as the bias
        const Mat scale = blobs.empty() ? Mat() : blobs.front();
        const Mat bias = blobs.size() >= 2 ? blobs.back() : Mat();

        return make_cuda_node<cuda4dnn::LayerNormOp>(preferableTarget, std::move(context->stream),
                                                     scale, bias, axis_, epsilon, loops);
    }
#endif

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        int noutputs = std::max(requiredOutputs > 0 ? requiredOutputs : (int)this->outputs.size(), 1);
        CV_Assert(noutputs >= 1 && noutputs <= 3);

        int num_inputs = inputs.size() + blobs.size();
        CV_Check(num_inputs, num_inputs >= 2 && num_inputs <= 3, "LayerNorm2: require two (x, weight) or three (x, weight, bias) inputs");

        auto x_shape = inputs[0];
        int x_ndims = static_cast<int>(x_shape.size());
        int axis_ = normalize_axis(axis0, x_shape.dims);

        auto w_shape = blobs.empty() ? inputs[1] : shape(blobs.front());
        int w_ndims = static_cast<int>(w_shape.size());
        w_ndims = (axis_ == x_ndims - 1 && w_ndims == 2) ? (w_ndims - 1) : w_ndims;
        CV_CheckEQ(x_ndims - axis_, w_ndims, "LayerNorm2: weight rank mismatch");
        for (int i = 0; i < w_ndims; ++i)
            CV_CheckEQ(x_shape[axis_ + i], w_shape[i], "LayerNorm2: weight dims mismatch");
        if (num_inputs >= 3)
        {
            auto b_shape = blobs.empty() ? inputs[2] : shape(blobs.back());
            CV_CheckEQ(w_shape.size(), b_shape.size(), "LayerNorm2: bias rank mismatch");
            for (size_t i = 0; i < w_shape.size(); ++i)
                CV_CheckEQ(w_shape[i], b_shape[i], "LayerNorm2: bias dims mismatch");
        }

        outputs.resize(noutputs, inputs[0]);
        for (int i = 1; i < noutputs; i++) {
            for (int j = axis_; j < x_ndims; j++)
                outputs[i][j] = 1;
        }
        internals.clear();
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& input = inputs[0];
        const Mat& scale = blobs.empty() ? inputs[1] : blobs.front();
        Mat& output = outputs[0];

        int axis_ = normalize_axis(axis0, input.dims);

        if (outputs.size() >= 3)
        {
            Mat& mean = outputs[1];
            Mat& invStdDev = outputs[2];
            fastNormMeanInvStdDev(input, mean, invStdDev, epsilon, (size_t)axis_);
        }

        if ((int)inputs.size() + (int)blobs.size() >= 3)
        {
            const Mat& bias = blobs.empty() ? inputs[2] : blobs.back();
            fastNorm(input, scale, bias, output, epsilon, (size_t)axis_);
        }
        else
        {
            fastNorm(input, scale, output, epsilon, (size_t)axis_);
        }
    }
};

Ptr<LayerNorm2Layer> LayerNorm2Layer::create(const LayerParams& params)
{
    return makePtr<LayerNorm2LayerImpl>(params);
}

}
}
