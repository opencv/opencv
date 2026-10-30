// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_LAYERS_FUSION_CUDA_HPP
#define OPENCV_DNN_SRC_LAYERS_FUSION_CUDA_HPP

#ifdef HAVE_CUDA
#include "../op_cuda.hpp"
#include "cpu_kernels/fusion_apply.hpp"
#include "../cuda4dnn/primitives/activation.hpp"
#include "../cuda4dnn/primitives/scale_shift.hpp"

namespace cv { namespace dnn {

// runs the main node, then the fused pointwise math in place on its output
class FusedPostOpNode final : public CUDABackendNode {
public:
    FusedPostOpNode(Ptr<CUDABackendNode> main_, Ptr<CUDABackendNode> post_)
        : main(std::move(main_)), post(std::move(post_)) { }

    void forward(const std::vector<UMat>& inputs, const std::vector<UMat>& outputs,
                 cuda4dnn::csl::Workspace& workspace) override
    {
        main->forward(inputs, outputs, workspace);
        post->forward(outputs, outputs, workspace);
    }

    std::size_t get_workspace_memory_in_bytes() const noexcept override
    {
        return std::max(main->get_workspace_memory_in_bytes(), post->get_workspace_memory_in_bytes());
    }

private:
    Ptr<CUDABackendNode> main, post;
};

// the fused math has a CUDA form: one activation with a CUDA op, or a last-axis scale/shift;
// channels is the output's last-axis length, -1 while it is unknown (checked again at build)
inline bool cudaSupportsFusion(const PreparedFusion& fusion, int channels)
{
    const ActivationFunc fn = fusion.activationFn;
    if (fn)
        return fn == getActivationFunc(ACTIV_GELU) || fn == getActivationFunc(ACTIV_GELU_APPROX) ||
               fn == getActivationFunc(ACTIV_RELU) || fn == getActivationFunc(ACTIV_SWISH) ||
               fn == getActivationFunc(ACTIV_SIGMOID) || fn == getActivationFunc(ACTIV_TANH) ||
               fn == getActivationFunc(ACTIV_MISH);
    if (channels < 0)
        return true;
    return (fusion.scaleBuf.empty() || (int)fusion.scaleBuf.total() == channels) &&
           (fusion.shiftBuf.empty() || (int)fusion.shiftBuf.total() == channels);
}

// the main node followed, on the same stream, by the CUDA form of the fused math
inline Ptr<BackendNode> makeFusedCudaNode(const PreparedFusion& fusion, const Ptr<BackendNode>& mainNode,
                                          int target, const cuda4dnn::csl::Stream& stream, const MatShape& outShape)
{
    if (!fusion.expr)
        return mainNode;
    CV_Assert(outShape.dims >= 1);
    const int channels = outShape[outShape.dims - 1];
    CV_Assert(cudaSupportsFusion(fusion, channels));

    const ActivationFunc fn = fusion.activationFn;
    Ptr<BackendNode> postNode;
    if (!fn) {
        // a last-axis scale/shift, applied to the output as the CPU path does
        Mat scale = fusion.scaleBuf, shift = fusion.shiftBuf;
        if (scale.empty() && fusion.scale != 1.f)
            scale = Mat(1, channels, CV_32F, Scalar(fusion.scale));
        if (shift.empty() && fusion.shift != 0.f)
            shift = Mat(1, channels, CV_32F, Scalar(fusion.shift));
        if (scale.empty() && shift.empty())
            return mainNode;
        cuda4dnn::ScaleShiftConfiguration config;
        config.scaleMode = scale.empty() ? cuda4dnn::ScaleShiftConfiguration::OpMode::NONE
                                         : cuda4dnn::ScaleShiftConfiguration::OpMode::TRAINABLE;
        config.shiftMode = shift.empty() ? cuda4dnn::ScaleShiftConfiguration::OpMode::NONE
                                         : cuda4dnn::ScaleShiftConfiguration::OpMode::TRAINABLE;
        config.axis = (std::size_t)(outShape.dims - 1);
        postNode = make_cuda_node<cuda4dnn::ScaleShiftOp>(target, stream, config, scale, shift);
    } else if (fn == getActivationFunc(ACTIV_GELU)) {
        postNode = make_cuda_node<cuda4dnn::GeluOp>(target, stream);
    } else if (fn == getActivationFunc(ACTIV_GELU_APPROX)) {
        postNode = make_cuda_node<cuda4dnn::GeluApproximationOp>(target, stream);
    } else if (fn == getActivationFunc(ACTIV_SWISH)) {
        postNode = make_cuda_node<cuda4dnn::SwishOp>(target, stream);
    } else if (fn == getActivationFunc(ACTIV_SIGMOID)) {
        postNode = make_cuda_node<cuda4dnn::SigmoidOp>(target, stream);
    } else if (fn == getActivationFunc(ACTIV_TANH)) {
        postNode = make_cuda_node<cuda4dnn::TanHOp>(target, stream);
    } else if (fn == getActivationFunc(ACTIV_MISH)) {
        postNode = make_cuda_node<cuda4dnn::MishOp>(target, stream);
    } else {
        const float slope = fusion.activationParams.empty() ? 0.f : fusion.activationParams[0];
        postNode = make_cuda_node<cuda4dnn::ReLUOp>(target, stream, slope);
    }
    return makePtr<FusedPostOpNode>(mainNode.dynamicCast<CUDABackendNode>(),
                                    postNode.dynamicCast<CUDABackendNode>());
}

}} // namespace cv::dnn
#endif // HAVE_CUDA

#endif // OPENCV_DNN_SRC_LAYERS_FUSION_CUDA_HPP
