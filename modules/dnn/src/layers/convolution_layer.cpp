/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "../op_vkcom.hpp"

#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <iostream>
#include <numeric>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
using namespace cv::dnn::ocl4dnn;
#endif
#ifdef HAVE_TENGINE
#include "../tengine4dnn/include/tengine_graph_convolution.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/convolution.hpp"
#include "../cuda4dnn/primitives/transpose_convolution.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class BaseConvolutionLayerImpl : public ConvolutionLayer
{
public:
    bool fusedWeights, fusedBias;
    std::vector<double> weightsMultipliers;
    BaseConvolutionLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        getConvolutionKernelParams(params, kernel_size, pads_begin, pads_end, strides, dilations, padMode, adjust_pads);

        numOutput = params.get<int>("num_output");
        int ngroups = params.get<int>("group", 1);
        CV_Assert(numOutput % ngroups == 0);

        if (kernel_size.size() == 2) {
            kernel = Size(kernel_size[1], kernel_size[0]);
            stride = Size(strides[1], strides[0]);
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in convolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
            dilation = Size(dilations[1], dilations[0]);

            adjustPad.height = adjust_pads[0];
            adjustPad.width = adjust_pads[1];
        }

        for (int i = 0; i < adjust_pads.size(); i++) {
            CV_Assert(adjust_pads[i] < strides[i]);
        }

        fusedWeights = false;
        fusedBias = false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert((inputs.size() > outputs.size() && blobs.empty()) ||
                  (!inputs.empty() && (blobs.size() == 1 || blobs.size() == 2)));
        MatSize weightShape = blobs.empty() ? inputs[1].size : blobs[0].size;

        CV_Assert(inputs[0].dims == outputs[0].dims);
        if (weightShape.dims() == 3)
        {
            kernel_size.assign(1, kernel_size[0]);
            strides.assign(1, strides[0]);
            dilations.assign(1, dilations[0]);
            pads_begin.assign(1, pads_begin[0]);
            pads_end.assign(1, pads_end[0]);
        }
        CV_Assert(weightShape.dims() == kernel_size.size() + 2);
        for (int i = 0; i < kernel_size.size(); i++) {
            CV_Assert(weightShape[i + 2] == kernel_size[i]);
        }

        const Mat &input = inputs[0];
        CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || input.dims == 4 || input.dims == 5) && (input.type() == CV_32F || input.type() == CV_16S));
        for (size_t i = 0; i < outputs.size(); i++)
        {
            CV_Assert(inputs[i].type() == input.type());
            CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || inputs[i].dims == 4 || inputs[i].dims == 5) && inputs[i].size[1] == input.size[1]);
            for (int j = 0; j < inputs[i].dims; j++) {
                CV_Assert(inputs[i].size[j] == input.size[j]);
            }
        }

        std::vector<int> inpShape;
        std::vector<int> outShape;
        for (int i = 2; i < inputs[0].dims; i++) {
            inpShape.push_back(inputs[0].size[i]);
            outShape.push_back(outputs[0].size[i]);
        }
        getConvPoolPaddings(inpShape, kernel_size, strides, padMode, pads_begin, pads_end);
        if (pads_begin.size() == 2) {
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in convolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
        }
        fusedWeights = false;
        fusedBias = false;
    }

    bool hasBias() const
    {
        return blobs.size() >= 2;
    }

    virtual MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const = 0;
    bool is1x1() const
    {
        return (kernel.height == 1 && kernel.width == 1) &&
               (stride.height == 1 && stride.width == 1) &&
               (dilation.height == 1 && dilation.width == 1);
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Ptr<BlankLayer> blank_layer = top.dynamicCast<BlankLayer>();
        if (blank_layer)
            return true;

        Mat w, b;
        top->getScaleShift(w, b);
        if (!w.empty() || !b.empty())
        {
            fuseWeights(w, b);
            fusedWeights = fusedWeights || !w.empty();
            fusedBias = fusedBias || (hasBias() && !w.empty()) || !b.empty();
            return true;
        }
        return false;
    }

    virtual void fuseWeights(const Mat& w_, const Mat& b_) = 0;

    virtual void applyHalideScheduler(Ptr<BackendNode>& node,
                                      const std::vector<Mat*> &inputs,
                                      const std::vector<Mat> &outputs,
                                      int targetId) const CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        if (targetId != DNN_TARGET_CPU)
        {
            Layer::applyHalideScheduler(node, inputs, outputs, targetId);
            return;
        }
        Halide::Var x("x"), y("y"), c("c"), n("n"), tile("tile"), yi("yi"), yo("yo"), co("co"), ci("ci");
        Halide::Func& top = node.dynamicCast<HalideBackendNode>()->funcs[1];
        Halide::Func& padded_input = node.dynamicCast<HalideBackendNode>()->funcs[0];

        int outW, outH, outC, outN;
        getCanonicalSize(outputs[0].size, &outW, &outH, &outC, &outN);

        if (outW == 1 || outH <= 2)
            return;

        if (is1x1() || outC <= 16)
            top.reorder(x, c, y)
               .split(y, yo, yi, 2)
               .fuse(yo, n, tile)
               .parallel(tile)
               .unroll(yi)
               .vectorize(x, outW >= 16 ? 16 : outW);
        else
            top.reorder(x, c, y)
               .split(y, yo, yi, 2)
               .split(c, co, ci, 16)
               .fuse(yo, co, tile).fuse(n, tile, tile)
               .parallel(tile)
               .unroll(yi)
               .vectorize(x, outW >= 16 ? 16 : outW);
        padded_input.compute_at(top, yi);
#endif  // HAVE_HALIDE
    }
};


#define IS_POWER_LAYER(layer) \
            (!layer.empty() && !layer->type.compare("Power"))
//TODO: simultaneously convolution and bias addition for cache optimization
class ConvolutionLayerImpl CV_FINAL : public BaseConvolutionLayerImpl
{
public:
    enum { VEC_ALIGN = 8, DFT_TYPE = CV_32F };
    Mat weightsMat;
    std::vector<float> biasvec;
    std::vector<float> reluslope;
    Ptr<ActivationLayer> activ;

#ifdef HAVE_OPENCL
    Ptr<OCL4DNNConvSpatial<float> > convolutionOp;
    std::vector<UMat> umat_blobs;
    bool newActiv;
    ocl4dnnFusedActiv_t activType;
    float power;
#endif

#ifdef HAVE_TENGINE
    teng_graph_t tengine_graph;
#endif

#ifdef HAVE_CUDA
    cuda4dnn::ConvolutionConfiguration::FusionMode cudaFusionMode;
    cuda4dnn::ConvolutionConfiguration::ActivationType cudaActType;
    float cuda_relu_slope, cuda_crelu_floor, cuda_crelu_ceil;
    float cuda_power_exp, cuda_power_scale, cuda_power_shift;
#endif

    ConvolutionLayerImpl(const LayerParams &params) : BaseConvolutionLayerImpl(params)
    {
#ifdef HAVE_OPENCL
        newActiv = false;
        activType = OCL4DNN_CONV_FUSED_ACTIV_NONE;
        power = 0.f;
#endif

#ifdef HAVE_CUDA
        cudaFusionMode = cuda4dnn::ConvolutionConfiguration::FusionMode::NONE;
        cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::IDENTITY;
#endif
#ifdef HAVE_TENGINE
        tengine_graph=NULL;
#endif
    }
#ifdef HAVE_TENGINE
    ~ConvolutionLayerImpl()
    {
        if(NULL != tengine_graph )
        {
            tengine_release(tengine_graph);
        }
    }
#endif

    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        int dims = inpShape.size();
        int inpD = dims == 5 ? inpShape[2] : 1;
        int inpH = inpShape[dims - 2];
        int inpW = inpShape.back();
        int inpGroupCn = blobs[0].size[1];
        int ksize = inpGroupCn * std::accumulate(kernel_size.begin(), kernel_size.end(),
                                                 1, std::multiplies<size_t>());
        return shape(inpD * inpH * inpW, ksize);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        size_t ksize = kernel_size.size();
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA)
        {
            /* only 1d, 2d and 3d convolutions supported */
            if (ksize > 0 && ksize <= 3)
                return true;

            return false;
        }
#endif
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            bool isArmTarget = preferableTarget == DNN_TARGET_CPU && isArmComputePlugin();
            if (isArmTarget && blobs.empty())
                return false;
            if (ksize == 1)
                return isArmTarget;
            if (ksize == 3)
                return preferableTarget != DNN_TARGET_MYRIAD && !isArmTarget;
            bool isMyriad = preferableTarget == DNN_TARGET_MYRIAD || preferableTarget == DNN_TARGET_HDDL;
            if ((backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || !isMyriad) && blobs.empty())
                return false;
            return (!isMyriad || dilation.width == dilation.height);
        }
#endif
        if (backendId == DNN_BACKEND_OPENCV)
            return ksize >= 1 && ksize <= 3;
#ifdef HAVE_HALIDE
        if (backendId == DNN_BACKEND_HALIDE)
            return ksize == 2 && !blobs.empty();
#endif
#ifdef HAVE_VULKAN
        if (backendId == DNN_BACKEND_VKCOM)
            return ksize == 2;
#endif
        return false;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!blobs.empty() || inputs.size() > 1);
        const int* weightShape = blobs.empty() ? &inputs[1][0] : blobs[0].size.p;
        CV_Assert(!hasBias() || blobs[1].total() == (size_t)weightShape[0]);

        internals.clear();

        CV_Assert(inputs.size() != 0);
        std::vector<int> inpShape(inputs[0].begin() + 2, inputs[0].end());

        int outCn = weightShape[0];
        std::vector<int> outShape;
        outShape.push_back(inputs[0][0]);
        outShape.push_back(outCn);

        int inpCn = inputs[0][1];
        if (padMode.empty())
        {
            for (int i = 0; i < inpShape.size(); i++)
                outShape.push_back((inpShape[i] + pads_begin[i] + pads_end[i] - dilations[i] * (kernel_size[i] - 1) - 1) / strides[i] + 1);
        }
        else
        {
            getConvPoolOutParams(inpShape, kernel_size, strides, padMode, dilations, outShape);
        }

        int ngroups = inpCn / weightShape[1];
        if (ngroups == 0 || ngroups * weightShape[1] != inpCn)
            CV_Error(Error::StsError, format("Number of input channels should "
                     "be multiple of %d but got %d", weightShape[1], inpCn));
        CV_Assert(ngroups > 0 && inpCn % ngroups == 0 && outCn % ngroups == 0);

        outputs.resize(1, outShape);

        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        BaseConvolutionLayerImpl::finalize(inputs_arr, outputs_arr);

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        // prepare weightsMat where each row is aligned and has enough zero padding on the right to
        // use vectorized (i.e. with intrinsics) loops without tail processing
        if (!blobs.empty())
        {
            Mat wm = blobs[0].reshape(1, numOutput);
            if( wm.step1() % VEC_ALIGN != 0 )
            {
                int newcols = (int)alignSize(wm.step1(), VEC_ALIGN);
                Mat wm_buffer = Mat(numOutput, newcols, wm.type());
                Mat wm_padding = wm_buffer.colRange(wm.cols, newcols);
                wm_padding.setTo(Scalar::all(0.));
                Mat wm_aligned = wm_buffer.colRange(0, wm.cols);
                wm.copyTo(wm_aligned);
                wm = wm_aligned;
            }
            weightsMat = wm;
        }
        else
        {
            // initialized in .forward()
            weightsMat.release();
        }

        weightsMultipliers.assign(numOutput, 1.0);

        Mat biasMat = hasBias() ? blobs[1].reshape(1, numOutput) : Mat();
        biasvec.resize(numOutput+2);
        if( biasMat.empty() )
        {
            for(int i = 0; i < numOutput; i++ )
                biasvec[i] = 0.f;
        }
        else
        {
            for(int i = 0; i < numOutput; i++ )
                biasvec[i] = biasMat.at<float>(i);
        }
#ifdef HAVE_TENGINE
        if(NULL != tengine_graph )
        {
            tengine_release(tengine_graph);
            tengine_graph = NULL ;
        }
#endif
#ifdef HAVE_OPENCL
        convolutionOp.release();
#endif
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        if ((!activ.empty() && !layer.empty()) || blobs.empty())
            return false;

        activ = layer;
        if (activ.empty())
            reluslope.clear();
#ifdef HAVE_OPENCL
        newActiv = true;
        activType = OCL4DNN_CONV_FUSED_ACTIV_NONE;

        if (IS_DNN_OPENCL_TARGET(preferableTarget))
        {
            Ptr<PowerLayer> activ_power = activ.dynamicCast<PowerLayer>();
            if (!activ_power.empty())
            {
                if (activ_power->scale != 1.0f)  // not supported well by implementation, #17964
                {
                    // FIXIT no way to check number of blobs (like, eltwise input)
                    CV_LOG_DEBUG(NULL, "DNN/OpenCL: can't configure Power activation (scale != 1.0f)");
                    activ.release();
                    newActiv = false;
                    return false;
                }
                if (activ_power->scale != 1.f || activ_power->shift != 0.f)
                {
                    const int outCh = blobs[0].size[0];
                    fuseWeights(Mat(1, outCh, CV_32F, Scalar(activ_power->scale)),
                                Mat(1, outCh, CV_32F, Scalar(activ_power->shift)));
                }

                power = activ_power->power;
                activType = OCL4DNN_CONV_FUSED_ACTIV_POWER;
            }
            Ptr<TanHLayer> activ_tanh = activ.dynamicCast<TanHLayer>();
            if (!activ_tanh.empty())
            {
                activType = OCL4DNN_CONV_FUSED_ACTIV_TANH;
            }
        }
#endif

#ifdef HAVE_CUDA
        if (activ.empty())
        {
            /* setActivation was called with empty argument => reset all fusions */
            cudaFusionMode = cuda4dnn::ConvolutionConfiguration::FusionMode::NONE;
            cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::IDENTITY;
        }

        if(IS_DNN_CUDA_TARGET(preferableTarget))
        {
            CV_Assert(cudaFusionMode == ConvolutionConfiguration::FusionMode::NONE ||
                      cudaFusionMode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM);

            Ptr<ReLULayer> activ_relu = activ.dynamicCast<ReLULayer>();
            if(!activ_relu.empty())
            {
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::RELU;
                cuda_relu_slope = activ_relu->negativeSlope;
            }

            Ptr<ReLU6Layer> activ_relu6 = activ.dynamicCast<ReLU6Layer>();
            if(!activ_relu6.empty())
            {
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::CLIPPED_RELU;
                cuda_crelu_floor = activ_relu6->minValue;
                cuda_crelu_ceil = activ_relu6->maxValue;
            }

            Ptr<PowerLayer> activ_power = activ.dynamicCast<PowerLayer>();
            if (!activ_power.empty())
            {
                cuda_power_scale = activ_power->scale;
                cuda_power_shift = activ_power->shift;
                cuda_power_exp = activ_power->power;
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::POWER;
            }

            Ptr<TanHLayer> activ_tanh = activ.dynamicCast<TanHLayer>();
            if(!activ_tanh.empty())
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::TANH;

            Ptr<SigmoidLayer> activ_sigmoid = activ.dynamicCast<SigmoidLayer>();
            if(!activ_sigmoid.empty())
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::SIGMOID;

            Ptr<SwishLayer> activ_swish = activ.dynamicCast<SwishLayer>();
            if(!activ_swish.empty())
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::SWISH;

            Ptr<MishLayer> activ_mish = activ.dynamicCast<MishLayer>();
            if(!activ_mish.empty())
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::MISH;

            if (cudaActType == cuda4dnn::ConvolutionConfiguration::ActivationType::IDENTITY)
            {
                /* no activation fused */
                activ.reset();
            }
            else
            {
                /* activation was fused */
                if (cudaFusionMode == ConvolutionConfiguration::FusionMode::NONE) /* no previous fusion */
                    cudaFusionMode = ConvolutionConfiguration::FusionMode::ACTIVATION; /* now activation */
                else if (cudaFusionMode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM) /* previously eltwise was fused */
                    cudaFusionMode = ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION; /* now activation on eltwise output */
            }
        }
#endif
        return !activ.empty();
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
#ifdef HAVE_CUDA
        if(IS_DNN_CUDA_TARGET(preferableTarget))
        {
            Ptr<EltwiseLayer> eltwise = top.dynamicCast<EltwiseLayer>();
            if (!eltwise.empty()) // && eltwise->op == EltwiseLayer::SUM && eltwise->coeffs.empty())
            {
                /* we also need to check that the eltwise input does not require shortcut mechanism
                 * it's difficult to verify it here but we hope that `fuseLayers` has done the check already
                 */
                if (cudaFusionMode == ConvolutionConfiguration::FusionMode::NONE)
                {
                    /* no previous fusion */
                    cudaFusionMode = ConvolutionConfiguration::FusionMode::ELTWISE_SUM; /* now eltwise */
                    return true;
                }
                else if(cudaFusionMode == ConvolutionConfiguration::FusionMode::ACTIVATION)
                {
                    /* previously an activation was fused */
                    cudaFusionMode = ConvolutionConfiguration::FusionMode::ACTIVATION_THEN_ELTWISE_SUM;
                    return true;
                }
                return false;
            }
        }
#endif
        return BaseConvolutionLayerImpl::tryFuse(top);
    }

    void fuseWeights(const Mat& w_, const Mat& b_) CV_OVERRIDE
    {
        // Convolution weights have OIHW data layout. Parameters fusion in case of
        // (conv(I) + b1 ) * w + b2
        // means to replace convolution's weights to [w*conv(I)] and bias to [b1 * w + b2]
        const int outCn = weightsMat.size[0];
        Mat w = w_.total() == 1 ? Mat(1, outCn, CV_32F, Scalar(w_.at<float>(0))) : w_;
        Mat b = b_.total() == 1 ? Mat(1, outCn, CV_32F, Scalar(b_.at<float>(0))) : b_;
        CV_Assert_N(!weightsMat.empty(), biasvec.size() == outCn + 2,
                    w.empty() || outCn == w.total(), b.empty() || outCn == b.total());

        if (!w.empty())
        {
            // Keep origin weights unchanged.
            if (weightsMat.data == blobs[0].data)
                weightsMat = weightsMat.clone();

            Mat originWeights = blobs[0].reshape(1, outCn);
            for (int i = 0; i < outCn; ++i)
            {
                double wi = w.at<float>(i);
                weightsMultipliers[i] *= wi;
                cv::multiply(originWeights.row(i), weightsMultipliers[i], weightsMat.row(i));
                biasvec[i] *= wi;
            }
        }

        if (!b.empty())
        {
            for (int i = 0; i < outCn; ++i)
                biasvec[i] += b.at<float>(i);
        }
        biasvec[outCn] = biasvec[outCn+1] = biasvec[outCn-1];
    }

    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
#ifdef HAVE_VULKAN
        int out_channel = blobs[0].size[0];
        bool has_bias = hasBias() || fusedBias;
        int filter_size[2] = {kernel.height, kernel.width};
        int pad_size[2] = {pad.height, pad.width};
        int stride_size[2] = {stride.height, stride.width};
        int dilation_size[2] = {dilation.height, dilation.width};
        int activation = 0;
        vkcom::Tensor input_tensor = VkComTensor(inputs[0]);
        int in_channel = input_tensor.dimSize(1);
        int group = in_channel / blobs[0].size[1];

        // TODO: support group > 1
        if (group != 1)
            return Ptr<BackendNode>();

        int padding_mode;
        if (padMode.empty())
        {
            padding_mode = vkcom::kPaddingModeCaffe;
        }
        else if (padMode == "VALID")
        {
            padding_mode = vkcom::kPaddingModeValid;
        }
        else if (padMode == "SAME")
        {
            padding_mode = vkcom::kPaddingModeSame;
        }
        else
            CV_Error(Error::StsError, "Unsupported padding mode " + padMode);

        std::shared_ptr<vkcom::OpBase> op(new vkcom::OpConv(out_channel, has_bias,
                    filter_size, pad_size,
                    stride_size, dilation_size,
                    activation, group,
                    padding_mode));

        std::vector<Ptr<BackendWrapper> > blobsWrapper;

        if (fusedWeights)
        {
            Mat wm;
            weightsMat.copyTo(wm); // to handle the case of isContinuous() == false
            wm = wm.reshape(1, blobs[0].dims, blobs[0].size);
            blobsWrapper.push_back(Ptr<BackendWrapper>(new VkComBackendWrapper(wm)));
        }
        else
        {
            blobsWrapper.push_back(Ptr<BackendWrapper>(new VkComBackendWrapper(blobs[0])));
        }

        if (has_bias)
        {
            Mat biasesMat({out_channel}, CV_32F, &biasvec[0]);
            blobsWrapper.push_back(Ptr<BackendWrapper>(new VkComBackendWrapper(biasesMat)));
        }

        return Ptr<BackendNode>(new VkComBackendNode(inputs, op, blobsWrapper));
#endif  // HAVE_VULKAN
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);

        const int inpCn = inputBuffer.channels();
        const int outCn = blobs[0].size[0];
        const int inpGroupCn = blobs[0].size[1];
        const int group = inpCn / inpGroupCn;
        const int outGroupCn = outCn / group;

        Halide::Buffer<float> weights = wrapToHalideBuffer(blobs[0]);

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Func padded_input(name + "_constant_exterior");
        if (pad.width || pad.height)
        {
            Halide::Func bounded =
                Halide::BoundaryConditions::constant_exterior(inputBuffer, 0);
            padded_input(x, y, c, n) = bounded(x, y, c, n);
        }
        else
        {
            padded_input(x, y, c, n) = inputBuffer(x, y, c, n);
        }

        Halide::RDom r(0, kernel.width, 0, kernel.height, 0, inpGroupCn);
        Halide::Expr kx = x * stride.width - pad.width + r.x * dilation.width;
        Halide::Expr ky = y * stride.height - pad.height + r.y * dilation.height;
        Halide::Expr kc = r.z;
        for (int i = 1; i < group; ++i)
        {
            kc = select(c < outGroupCn * i, kc, inpGroupCn * i + r.z);
        }
        Halide::Expr topExpr = sum(padded_input(kx, ky, kc, n) *
                                   weights(r.x, r.y, r.z, c));
        if (hasBias())
        {
            Halide::Buffer<float> bias = wrapToHalideBuffer(blobs[1], {outCn});
            topExpr += bias(c);
        }
        top(x, y, c, n) = topExpr;
        return Ptr<BackendNode>(new HalideBackendNode({ padded_input, top }));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
        InferenceEngine::DataPtr input = infEngineDataNode(inputs[0]);
        std::vector<size_t> dims = input->getDims();
        CV_Assert(dims.size() == 4 || dims.size() == 5);
        const int inpCn = dims[1];
        const int outCn = blobs[0].size[0];
        const int inpGroupCn = blobs[0].size[1];
        const int group = inpCn / inpGroupCn;
        InferenceEngine::Layout layout = (dims.size() == 4) ? InferenceEngine::Layout::OIHW :
                                                              InferenceEngine::Layout::NCDHW;

        auto ieWeights = wrapToInfEngineBlob(blobs[0], layout);
        if (fusedWeights)
        {
            if (weightsMat.isContinuous())
            {
                Mat cvWeights = weightsMat.reshape(1, blobs[0].dims, blobs[0].size);
                ieWeights = wrapToInfEngineBlob(cvWeights, layout);
            }
            else
            {
                ieWeights = InferenceEngine::make_shared_blob<float>({
                                InferenceEngine::Precision::FP32,
                                ieWeights->getTensorDesc().getDims(), layout
                            });
                ieWeights->allocate();

                Mat newWeights = infEngineBlobToMat(ieWeights).reshape(1, outCn);
                Mat cvWeights = weightsMat.colRange(0, newWeights.cols);
                cvWeights.copyTo(newWeights);
            }
        }
        InferenceEngine::Blob::Ptr ieBiases;
        if (hasBias() || fusedBias)
        {
            Mat biasesMat({outCn}, CV_32F, &biasvec[0]);
            ieBiases = wrapToInfEngineBlob(biasesMat, {(size_t)outCn}, InferenceEngine::Layout::C);
        }

        InferenceEngine::Builder::ConvolutionLayer ieLayer(name);

        ieLayer.setKernel(kernel_size);
        ieLayer.setStrides(strides);
        ieLayer.setDilation(dilations);
        ieLayer.setPaddingsBegin(pads_begin);
        ieLayer.setPaddingsEnd(pads_end);
        ieLayer.setGroup((size_t)group);
        ieLayer.setOutDepth((size_t)outCn);

        InferenceEngine::Builder::Layer l = ieLayer;
        addConstantData("weights", ieWeights, l);
        if (ieBiases)
            addConstantData("biases", ieBiases, l);

        if (!padMode.empty())
            l.getParameters()["auto_pad"] = padMode == "VALID" ? std::string("valid") : std::string("same_upper");

        return Ptr<BackendNode>(new InfEngineBackendNode(l));
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert_N(inputs.size() >= 1, nodes.size() >= 1);
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::vector<size_t> dims = ieInpNode->get_shape();
        CV_Check(dims.size(), dims.size() >= 3 && dims.size() <= 5, "");
        std::shared_ptr<ngraph::Node> ieWeights = nodes.size() > 1 ? nodes[1].dynamicCast<InfEngineNgraphNode>()->node : nullptr;
        if (nodes.size() > 1)
            CV_Assert(ieWeights);  // dynamic_cast should not fail
        const int inpCn = dims[1];
        const int inpGroupCn = nodes.size() > 1 ? ieWeights->get_shape()[1] : blobs[0].size[1];
        const int group = inpCn / inpGroupCn;

        std::vector<size_t> kernel_shape;
        if (group != 1)
        {
            kernel_shape.push_back(group);
        }
        kernel_shape.push_back(numOutput / group);
        kernel_shape.push_back(inpCn / group);
        std::copy(kernel_size.begin(), kernel_size.end(), back_inserter(kernel_shape));

        if (nodes.size() == 1)
        {
            ieWeights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, kernel_shape, blobs[0].data);
            if (fusedWeights)
            {
                if (weightsMat.isContinuous())
                {
                    ieWeights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, kernel_shape, weightsMat.data);
                }
                else
                {
                    Mat newWeights;
                    Mat cvWeights = weightsMat.colRange(0, blobs[0].total() / numOutput);
                    cvWeights.copyTo(newWeights);
                    ieWeights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, kernel_shape, newWeights.data);
                }
            }
        }
        else
        {
            auto shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                             ngraph::Shape{kernel_shape.size()}, std::vector<int64_t>(kernel_shape.begin(), kernel_shape.end()));
            ieWeights  = std::make_shared<ngraph::op::v1::Reshape>(ieWeights, shape, true);
        }

        ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
        if (!padMode.empty())
            pad_type = padMode == "VALID" ? ngraph::op::PadType::VALID : ngraph::op::PadType::SAME_UPPER;

        std::shared_ptr<ngraph::Node> conv_node;
        if (group != 1) {
            conv_node = std::make_shared<ngraph::op::v1::GroupConvolution>(
                                ieInpNode, ieWeights,
                                ngraph::Strides(strides),
                                ngraph::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                                ngraph::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_end.begin(),   pads_end.end())),
                                ngraph::Strides(dilations),
                                pad_type);
        } else {
            conv_node = std::make_shared<ngraph::op::v1::Convolution>(
                                ieInpNode, ieWeights,
                                ngraph::Strides(strides),
                                ngraph::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                                ngraph::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_end.begin(), pads_end.end())),
                                ngraph::Strides(dilations),
                                pad_type);
        }

        if (hasBias() || fusedBias || nodes.size() == 3)
        {
            std::vector<size_t> shape(conv_node->get_shape().size(), 1);
            shape[1] = conv_node->get_shape()[1];
            std::shared_ptr<ngraph::Node> bias;
            if (nodes.size() == 3)
            {
                auto bias_shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                    ngraph::Shape{shape.size()}, std::vector<int64_t>(shape.begin(), shape.end()));
                bias = std::make_shared<ngraph::op::v1::Reshape>(nodes[2].dynamicCast<InfEngineNgraphNode>()->node, bias_shape, true);
            }
            else
            {
                bias = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape(shape), biasvec.data());
            }
            auto conv_bias = std::make_shared<ngraph::op::v1::Add>(conv_node, bias, ngraph::op::AutoBroadcastType::NUMPY);
            return Ptr<BackendNode>(new InfEngineNgraphNode(conv_bias));
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(conv_node));
    }
#endif  // HAVE_DNN_NGRAPH

    class ParallelConv : public cv::ParallelLoopBody
    {
    public:
        enum { BLK_SIZE = 32, BLK_SIZE_CN = 64 };

        const Mat* input_;
        const Mat* weights_;
        Mat* output_;
        int outShape[4]; // used only for conv2d
        std::vector<size_t> kernel_size, pads_begin, pads_end, strides, dilations;
        int ngroups_, nstripes_;
        std::vector<int> ofstab_;
        const std::vector<float>* biasvec_;
        const std::vector<float>* reluslope_;
        const ActivationLayer* activ_;
        bool is1x1_;
        bool useAVX;
        bool useAVX2;
        bool useAVX512;
        int blk_size_cn;

        ParallelConv()
            : input_(0), weights_(0), output_(0), ngroups_(0), nstripes_(0),
              biasvec_(0), reluslope_(0), activ_(0), is1x1_(false), useAVX(false), useAVX2(false), useAVX512(false)
            , blk_size_cn(0)
        {}

        static void run( const Mat& input, Mat& output, const Mat& weights,
                         const std::vector<float>& biasvec,
                         const std::vector<float>& reluslope,
                         const std::vector<size_t>& kernel_size, const std::vector<size_t>& strides,
                         const std::vector<size_t>& pads_begin, const std::vector<size_t>& pads_end,
                         const std::vector<size_t>& dilations,
                         const ActivationLayer* activ, int ngroups, int nstripes )
        {
            size_t karea = std::accumulate(kernel_size.begin(), kernel_size.end(),
                                           1, std::multiplies<size_t>());
            bool isConv1D = input.dims == 3;
            bool isConv2D = input.dims == 4;
            bool isConv3D = input.dims == 5;
            CV_CheckEQ(static_cast<int>(kernel_size.size()), input.dims - 2, "");
            CV_Assert_N(input.dims == output.dims,
                       input.size[0] == output.size[0],
                       weights.rows == output.size[1],
                       weights.cols == (input.size[1]/ngroups)*karea,
                       input.type() == output.type(),
                       input.type() == weights.type(),
                       input.type() == CV_32FC1,
                       input.isContinuous(),
                       output.isContinuous(),
                       biasvec.size() == (size_t)output.size[1]+2);
            CV_Check(weights.step1(), weights.step1() % VEC_ALIGN == 0, "");
            CV_CheckType(weights.type(), CV_32FC1, "");
            ParallelConv p;

            p.input_ = &input;
            p.weights_ = &weights;
            p.output_ = &output;
            int max_ind = isConv1D? 3: 4;
            for( int i = 0; i < max_ind; i++ ) p.outShape[i] = output.size[i];
            p.outShape[1] /= ngroups;

            p.kernel_size = kernel_size; p.strides = strides; p.dilations = dilations;
            p.pads_begin = pads_begin; p.pads_end = pads_end;

            p.ngroups_ = ngroups;
            p.nstripes_ = nstripes;

            int inpCnAll = input.size[1];
            int depth = (input.dims == 5) ? input.size[2] : 1;
            int width = input.size[input.dims - 1];
            int height = isConv1D? 1 : input.size[input.dims - 2];
            int inpCn = inpCnAll / ngroups;

            p.is1x1_ = (isConv2D && kernel_size[0] == 1 && kernel_size[1] == 1 &&
                       pads_begin[0] == 0  && pads_begin[1] == 0) ||
                       (isConv1D && pads_begin[0] == 0 && kernel_size[0] == 1);

            p.useAVX    = checkHardwareSupport(CPU_AVX)  && isConv2D;
            p.useAVX2   = checkHardwareSupport(CPU_AVX2) && isConv2D;
            p.useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX  && isConv2D;

            int kernel_d = isConv3D? kernel_size[0] : 1;
            int kernel_h = isConv1D? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();

            int blk_size_cn0 = cvCeil(800./(kernel_w*kernel_h));
            int ncn = 16;
            while (ncn*2 < blk_size_cn0 && ncn < inpCn)
                ncn *= 2;
            ncn = std::min(ncn, inpCn);
            p.blk_size_cn = ncn;

            int dil_d = isConv3D? dilations[0] : 1;
            int dil_h = isConv1D? 1 : dilations[dilations.size() - 2];
            int dil_w = dilations.back();

            p.ofstab_.resize(karea * ncn);
            int* ofstab = &p.ofstab_[0];

            if (isConv1D)
            {
                for( int k = 0; k < ncn; k++ )
                    for( int k_c = 0; k_c < kernel_w; k_c++ )
                        ofstab[k*kernel_w + k_c] = k*width + k_c*dil_w;
            }
            else if (isConv2D)
            {
                for( int k = 0; k < ncn; k++ )
                    for( int k_r = 0; k_r < kernel_h; k_r++ )
                        for( int k_c = 0; k_c < kernel_w; k_c++ )
                            ofstab[(k*kernel_h + k_r)*kernel_w + k_c] =
                                   (k*height + k_r*dil_h)*width + k_c*dil_w;
            }
            else
            {
                for( int k = 0; k < ncn; k++ )
                    for (int k_d = 0; k_d < kernel_d; k_d++)
                        for( int k_r = 0; k_r < kernel_h; k_r++ )
                            for( int k_c = 0; k_c < kernel_w; k_c++ )
                                ofstab[(k*kernel_d*kernel_h + k_d*kernel_h + k_r)*kernel_w + k_c] =
                                       (k*depth*height + k_d*dil_d*height + k_r*dil_h)*width + k_c*dil_w;
            }

            p.biasvec_ = &biasvec;
            p.reluslope_ = &reluslope;
            p.activ_ = p.reluslope_->empty() ? activ : 0;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        virtual void operator ()(const Range &r0) const CV_OVERRIDE
        {
            const int valign = ConvolutionLayerImpl::VEC_ALIGN;
            int ngroups = ngroups_, batchSize = input_->size[0]*ngroups;
            bool isConv1D = input_->dims == 3;
            bool isConv2D = input_->dims == 4;
            bool isConv3D = input_->dims == 5;

            int outW = output_->size[output_->dims - 1];
            int outH = isConv1D? 1 : output_->size[output_->dims - 2];
            int outCn = output_->size[1]/ngroups;

            int depth = isConv3D? input_->size[2] : 1;
            int height = isConv1D? 1 : input_->size[input_->dims - 2];
            int width = input_->size[input_->dims - 1];
            int inpCn = input_->size[1]/ngroups;

            const int nstripes = nstripes_;

            int kernel_d = isConv3D? kernel_size[0] : 1;
            int kernel_h = isConv1D? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();
            int karea = kernel_w*kernel_h*kernel_d;

            int pad_d = isConv3D? pads_begin[0] : 0;
            int pad_t = isConv1D? 0 : pads_begin[pads_begin.size() - 2];
            int pad_l = pads_begin.back();

            int stride_d = isConv3D? strides[0] : 0;
            int stride_h = isConv1D? 0 : strides[strides.size() - 2];
            int stride_w = strides.back();

            int dilation_d = isConv3D? dilations[0] : 1;
            int dilation_h = isConv1D? 1 : dilations[dilations.size() - 2];
            int dilation_w = dilations.back();

            int i, j, k, d;
            int inpPlaneSize = (int)input_->total(2);
            int outPlaneSize = (int)output_->total(2);
            bool is1x1 = is1x1_;

            int stripesPerSample;
            int stripeSize;
            Range r = r0;
            bool depthWiseConvolution = !is1x1 && isConv2D && ngroups > 1 && inpCn == 1 &&
                outCn == 1 && kernel_d == 1 && dilation_d == 1 && stride_d == 0 && pad_d == 0 &&
                width >= 16 + dilation_w*(kernel_w - 1);
            // for now only 3x3 depth-wise convolutions are supported
            depthWiseConvolution = depthWiseConvolution && kernel_w == 3 && kernel_h == 3 &&
                // computing at most 1 pixel from each side can involve padding
                max(stride_w, dilation_w) >= pad_l && max(stride_h, dilation_h) >= pad_t &&
                pad_l <= 1 && pad_t <= 1;

            if( !depthWiseConvolution && nstripes >= batchSize*2 )
            {
                stripesPerSample = nstripes/batchSize;
                stripeSize = (int)alignSize((outPlaneSize + stripesPerSample - 1)/stripesPerSample, valign);
                stripeSize = std::min(stripeSize, outPlaneSize);
            }
            else
            {
                stripesPerSample = 1;
                int samplesPerStripe = std::max((batchSize + nstripes - 1)/nstripes, 1);
                r.start *= samplesPerStripe;
                r.end *= samplesPerStripe;
                stripeSize = outPlaneSize;
            }

            const float* data_inp0_ = input_->ptr<float>();
            const int* ofstab = &ofstab_[0];
            const float* wptr_orig_ = weights_->ptr<float>();
            size_t wstep = weights_->step1();
            const float* biasptr_ = &biasvec_->at(0);
            const float* reluptr_ = reluslope_->empty() ? 0 : &reluslope_->at(0);
            float* data_out0_ = output_->ptr<float>();
            AutoBuffer<float> rowbuf0_;
            float* rowbuf0 = 0;
            bool use_rowbuf = !depthWiseConvolution;
            int blk_size = depthWiseConvolution ? outPlaneSize : min((int)BLK_SIZE, stripeSize);

            // im2row buffer is not used for depth-wise convolution
            if(use_rowbuf)
            {
                size_t rowbufsz = alignSize(karea*blk_size_cn, valign)*min((int)BLK_SIZE, blk_size);
                //printf("karea=%d, blk_size_cn=%d, rowbufsz=%d, stripeSize=%d\n", karea, blk_size_cn, (int)rowbufsz, stripeSize);
                rowbuf0_.allocate(rowbufsz + valign);
                rowbuf0 = alignPtr(rowbuf0_.data(), (int)(valign*sizeof(float)));
                // we clear the buffer once; ultimately, it lets us to avoid
                // tail processing after running the unrolled/vectorized loop.
                // the main idea is to make sure that the tail (a.k.a. padding) of each row
                // (i.e. the elements with indices between vsz=karea*ncn and vsz_a)
                // does not contain NaNs or Infs. Because the padding in the weights
                // matrix is explicitly initialized with 0's, we handle all other
                // cases nicely, i.e. we can skip expliciting re-initialization
                // of the padding - we just retain elements from the previous iteration
                // of the loop over channels (cn0).
                memset(rowbuf0, 0, rowbufsz*sizeof(rowbuf0[0]) );
            }

            for( int stripe = r.start; stripe < r.end; stripe++ )
            {
                int subsampleIdx = stripe/stripesPerSample;
                if( subsampleIdx >= batchSize )
                    break;
                int stripeStart = (int)((stripe - subsampleIdx*stripesPerSample)*stripeSize);
                int stripeEnd = (int)std::min(stripeStart + stripeSize, outPlaneSize);
                const float* data_inp0 = data_inp0_ + subsampleIdx*inpPlaneSize*inpCn;
                float* data_out0 = data_out0_ + subsampleIdx*outPlaneSize*outCn;
                int startOutCn = (subsampleIdx % ngroups)*outCn;
                const float* wptr_orig = wptr_orig_ + wstep*startOutCn;
                const float* biasptr = biasptr_ + startOutCn;

                for( int cn0 = 0; cn0 < inpCn; cn0 += blk_size_cn )
                {
                    int cn1 = std::min(cn0 + blk_size_cn, inpCn);
                    int ncn = cn1 - cn0, vsz = karea*ncn;
                    int vsz_a = (int)alignSize(vsz, valign);
                    const float* wptr = wptr_orig + cn0*karea;
                    // we apply [Channels][P]ReLU (if any) during the final pass only.
                    const float* relu = cn1 == inpCn && reluptr_ ? reluptr_ + startOutCn : 0;

                    for( int ofs0 = stripeStart; ofs0 < stripeEnd; ofs0 += blk_size )
                    {
                        int ofs, ofs1 = std::min(ofs0 + blk_size, stripeEnd);
                        int bsz = ofs1 - ofs0;

                        int out_d = ofs0 / (outH * outW);
                        int out_i = (ofs0 - out_d * outH * outW) / outW;
                        int out_j = ofs0 % outW;

                        if (depthWiseConvolution)
                        {
                            CV_Assert(out_i == 0 && out_j == 0);
                            int in_d = out_d * stride_d - pad_d;
                            const float* inptr_ = data_inp0 + (cn0*depth*height + in_d*height)*width;
                            float* outptr_ = data_out0 + ofs0;

                        #if CV_TRY_AVX2
                            if(useAVX2)
                                opt_AVX2::fastDepthwiseConv(wptr, kernel_h, kernel_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_t, pad_l,
                                    biasptr, relu, inptr_, height, width, outptr_, out_d, outH, outW);
                            else
                        #endif
                        #if CV_TRY_AVX
                            if(useAVX)
                                opt_AVX::fastDepthwiseConv(wptr, kernel_h, kernel_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_t, pad_l,
                                    biasptr, relu, inptr_, height, width, outptr_, out_d, outH, outW);
                            else
                        #endif
                            {
                                const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                                            w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                                            w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
                                int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
                                float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

                                for (int out_i = 0; out_i < outH; out_i++)
                                {
                                    int in_i = out_i * stride_h - pad_t, out_j = 0;
                                    const float* imgptr0 = inptr_ + in_i*width;
                                    const float* imgptr1 = imgptr0 + dilation_h*width;
                                    const float* imgptr2 = imgptr0 + (dilation_h*2)*width;
                                    float out, w00 = w00_, w01 = w01_, w02 = w02_;
                                    float w20 = w20_, w21 = w21_, w22 = w22_;
                                    if (in_i < 0)
                                    {
                                        w00 = w01 = w02 = 0.f;
                                        imgptr0 = imgptr1;
                                    }
                                    else if (in_i + dilation_h*(kernel_h-1) >= height)
                                    {
                                        w20 = w21 = w22 = 0.f;
                                        imgptr2 = imgptr1;
                                    }
                                    float* outptr = outptr_ + out_i*outW;
                                    if (pad_l > 0)
                                    {
                                        out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                                              imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                                              imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + bias;
                                        if (relu)
                                            out = out > 0.f ? out : out*relu_coeff;
                                        outptr[0] = out;
                                        out_j = 1;
                                    }

                                #if CV_SIMD
                                    // maybe with AVX or AVX512 strided depthwise convolution
                                    // can be accelerated with vector code, but with 4xfloat vectors
                                    // it's hardly the case
                                    if( stride_w == 1 )
                                    {
                                        const int VECSZ = v_float32::nlanes;
                                        const int out_delta = VECSZ/stride_w;
                                        v_float32 vw00 = vx_setall_f32(w00), vw01 = vx_setall_f32(w01), vw02 = vx_setall_f32(w02),
                                                  vw10 = vx_setall_f32(w10), vw11 = vx_setall_f32(w11), vw12 = vx_setall_f32(w12),
                                                  vw20 = vx_setall_f32(w20), vw21 = vx_setall_f32(w21), vw22 = vx_setall_f32(w22);
                                        v_float32 z = vx_setzero_f32(), vbias = vx_setall_f32(bias), vrc = vx_setall_f32(relu_coeff);
                                        for( ; out_j < outW1; out_j += out_delta )
                                        {
                                            if (out_j + out_delta > outW1)
                                            {
                                                if (out_j <= pad_l)
                                                    break;
                                                out_j = outW1 - out_delta;
                                            }
                                            int in_j = out_j * stride_w - pad_l;
                                            v_float32 v00 = vx_load(imgptr0 + in_j),
                                                      v01 = vx_load(imgptr0 + in_j + dilation_w),
                                                      v02 = vx_load(imgptr0 + in_j + dilation_w*2),
                                                      v10 = vx_load(imgptr1 + in_j),
                                                      v11 = vx_load(imgptr1 + in_j + dilation_w),
                                                      v12 = vx_load(imgptr1 + in_j + dilation_w*2),
                                                      v20 = vx_load(imgptr2 + in_j),
                                                      v21 = vx_load(imgptr2 + in_j + dilation_w),
                                                      v22 = vx_load(imgptr2 + in_j + dilation_w*2);

                                            v_float32 vout = v00*vw00 + v01*vw01 + v02*vw02 +
                                                             v10*vw10 + v11*vw11 + v12*vw12 +
                                                             v20*vw20 + v21*vw21 + v22*vw22 + vbias;
                                            if (relu)
                                                vout = v_select(vout > z, vout, vout*vrc);
                                            vx_store(outptr + out_j, vout);
                                        }
                                    }
                                #endif
                                    for (; out_j < outW1; out_j++)
                                    {
                                        int in_j = out_j * stride_w - pad_l;
                                        out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                                              imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                                              imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + bias;
                                        if (relu)
                                            out = out > 0.f ? out : out*relu_coeff;
                                        outptr[out_j] = out;
                                    }

                                    for (; out_j < outW; out_j++ )
                                    {
                                        int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
                                        float s0 = 1.f, s1 = 1.f, s2 = 1.f;
                                        if (in_j0 >= width)
                                        {
                                            in_j0 = 0;
                                            s0 = 0.f;
                                        }
                                        if (in_j1 >= width)
                                        {
                                            in_j1 = 0;
                                            s1 = 0.f;
                                        }
                                        if (in_j2 >= width)
                                        {
                                            in_j2 = 0;
                                            s2 = 0.f;
                                        }
                                        out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                                              imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                                              imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + bias;
                                        if (relu)
                                            out = out > 0.f ? out : out*relu_coeff;
                                        outptr[out_j] = out;
                                    }
                                }
                            }
                            continue;
                        }

                        // do im2row for a part of input tensor
                        float* rowbuf = rowbuf0;

                        if (isConv1D)
                        {
                            for( ofs = ofs0; ofs < ofs1; out_j = 0, ++out_i )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_j = out_j * stride_w - pad_l;
                                const float* imgptr = data_inp0 + cn0*width + in_j;
                                ofs += delta;

                                // do im2row for a part of input tensor
                                if( is1x1 )
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w )
                                    {
                                        for( k = 0; k < vsz; k++ )
                                            rowbuf[k] = imgptr[k*inpPlaneSize];
                                    }
                                }
                                else
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                    {
                                        // this condition should be true for most of the tensor elements, i.e.
                                        // most of the time the kernel aperture is inside the tensor X-Y plane.
                                        if( out_j + 2 <= out_j1 && 0 <= in_j && in_j + stride_w*2 <= width - (kernel_w-1)*dilation_w )
                                        {
                                            for( k = 0; k < vsz; k++ )
                                            {
                                                int k1 = ofstab[k];
                                                float v0 = imgptr[k1];
                                                float v1 = imgptr[k1 + stride_w];
                                                rowbuf[k] = v0;
                                                rowbuf[k+vsz_a] = v1;
                                            }
                                            out_j++;
                                            rowbuf += vsz_a;
                                            imgptr += stride_w;
                                            in_j += stride_w;
                                        }
                                        else
                                        {
                                            int i0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                            int i1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                            // here some non-continuous sub-row of the row will not be
                                            // filled from the tensor; we need to make sure that the uncovered
                                            // elements are explicitly set to 0's. the easiest way is to
                                            // set all the elements to 0's before the loop.
                                            memset(rowbuf, 0, vsz*sizeof(rowbuf[0]));
                                            for( k = 0; k < ncn; k++ )
                                            {
                                                for( i = i0; i < i1; i++ )
                                                {
                                                    int imgofs = k*width + i*dilation_w;
                                                    rowbuf[k*kernel_w + i] = imgptr[imgofs];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else if (isConv2D)
                        {
                            if( is1x1 && stride_w == 1 && stride_h == 1 )
                            {
                                const float* imgptr = data_inp0 + (cn0*height + out_i)*width + out_j;
                                for( int j = 0; j < bsz; j++, rowbuf += vsz_a )
                                {
                                    if( j + 4 <= bsz )
                                    {
                                        k = 0;
                                    #if CV_SIMD128
                                        for( ; k <= vsz - 4; k += 4 )
                                        {
                                            const float* inp = imgptr + j + k*inpPlaneSize;
                                            v_float32x4 p0 = v_load(inp), p1 = v_load(inp + inpPlaneSize);
                                            v_float32x4 p2 = v_load(inp + inpPlaneSize*2), p3 = v_load(inp + inpPlaneSize*3);
                                            v_float32x4 r0, r1, r2, r3;
                                            v_transpose4x4(p0, p1, p2, p3, r0, r1, r2, r3);
                                            v_store(rowbuf + k, r0);
                                            v_store(rowbuf + k + vsz_a, r1);
                                            v_store(rowbuf + k + vsz_a*2, r2);
                                            v_store(rowbuf + k + vsz_a*3, r3);
                                        }
                                    #endif
                                        for( ; k < vsz; k++ )
                                        {
                                            const float* inp = imgptr + j + k*inpPlaneSize;
                                            float v0 = inp[0], v1 = inp[1], v2 = inp[2], v3 = inp[3];
                                            rowbuf[k] = v0;
                                            rowbuf[k + vsz_a] = v1;
                                            rowbuf[k + vsz_a*2] = v2;
                                            rowbuf[k + vsz_a*3] = v3;
                                        }
                                        j += 3;
                                        rowbuf += vsz_a*3;
                                    }
                                    else
                                    {
                                        for( k = 0; k < vsz; k++ )
                                        {
                                            rowbuf[k] = imgptr[j + k*inpPlaneSize];
                                        }
                                    }
                                }
                            }
                            else
                            for( ofs = ofs0; ofs < ofs1; out_j = 0, ++out_i )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_i = out_i * stride_h - pad_t;
                                int in_j = out_j * stride_w - pad_l;
                                const float* imgptr = data_inp0 + (cn0*height + in_i)*width + in_j;
                                ofs += delta;

                                // do im2row for a part of input tensor
                                if( is1x1 )
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w )
                                    {
                                        for( k = 0; k < vsz; k++ )
                                            rowbuf[k] = imgptr[k*inpPlaneSize];
                                    }
                                }
                                else
                                {
                                    bool ok_i = 0 <= in_i && in_i < height - (kernel_h-1)*dilation_h;
                                    int i0 = std::max(0, (-in_i + dilation_h-1)/dilation_h);
                                    int i1 = std::min(kernel_h, (height - in_i + dilation_h-1)/dilation_h);

                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                    {
                                        // this condition should be true for most of the tensor elements, i.e.
                                        // most of the time the kernel aperture is inside the tensor X-Y plane.
                                        if( ok_i && out_j + 2 <= out_j1 && 0 <= in_j && in_j + stride_w*2 <= width - (kernel_w-1)*dilation_w )
                                        {
                                            for( k = 0; k < vsz; k++ )
                                            {
                                                int k1 = ofstab[k];
                                                float v0 = imgptr[k1];
                                                float v1 = imgptr[k1 + stride_w];
                                                rowbuf[k] = v0;
                                                rowbuf[k+vsz_a] = v1;
                                            }
                                            out_j++;
                                            rowbuf += vsz_a;
                                            imgptr += stride_w;
                                            in_j += stride_w;
                                        }
                                        else
                                        {
                                            int j0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                            int j1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                            // here some non-continuous sub-row of the row will not be
                                            // filled from the tensor; we need to make sure that the uncovered
                                            // elements are explicitly set to 0's. the easiest way is to
                                            // set all the elements to 0's before the loop.
                                            memset(rowbuf, 0, vsz*sizeof(rowbuf[0]));
                                            for( k = 0; k < ncn; k++ )
                                            {
                                                for( i = i0; i < i1; i++ )
                                                {
                                                    for( j = j0; j < j1; j++ )
                                                    {
                                                        int imgofs = k*(width*height) + i*(dilation_h*width) + j*dilation_w;
                                                        rowbuf[(k*kernel_h + i)*kernel_w + j] = imgptr[imgofs];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            for( ofs = ofs0; ofs < ofs1; out_d += (out_i + 1) / outH, out_i = (out_i + 1) % outH, out_j = 0 )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_d = out_d * stride_d - pad_d;
                                int in_i = out_i * stride_h - pad_t;
                                int in_j = out_j * stride_w - pad_l;
                                const float* imgptr = data_inp0 + (cn0*depth*height + in_d*height + in_i)*width + in_j;
                                ofs += delta;

                                int d0 = std::max(0, (-in_d + dilation_d - 1) / dilation_d);
                                int d1 = std::min(kernel_d, (depth - in_d + dilation_d - 1) / dilation_d);

                                int i0 = std::max(0, (-in_i + dilation_h-1)/dilation_h);
                                int i1 = std::min(kernel_h, (height - in_i + dilation_h-1)/dilation_h);

                                for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                {
                                    int j0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                    int j1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                    // here some non-continuous sub-row of the row will not be
                                    // filled from the tensor; we need to make sure that the uncovered
                                    // elements are explicitly set to 0's. the easiest way is to
                                    // set all the elements to 0's before the loop.
                                    memset(rowbuf, 0, vsz*sizeof(rowbuf[0]));
                                    for( k = 0; k < ncn; k++ )
                                    {
                                        for ( d = d0; d < d1; d++)
                                        {
                                            for( i = i0; i < i1; i++ )
                                            {
                                                for( j = j0; j < j1; j++ )
                                                {
                                                    int imgofs = k*(depth*width*height) + d*dilation_d*width*height + i*(dilation_h*width) + j*dilation_w;
                                                    rowbuf[(k*kernel_d*kernel_h + d*kernel_h + i)*kernel_w + j] = imgptr[imgofs];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // now compute dot product of the weights
                        // and im2row-transformed part of the tensor
                    #if CV_TRY_AVX512_SKX
                        /* AVX512 convolution requires an alignment of 16, and ROI is only there for larger vector sizes */
                        if(useAVX512)
                            opt_AVX512_SKX::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, relu, cn0 == 0);
                        else
                    #endif
                    #if CV_TRY_AVX2
                        if(useAVX2)
                            opt_AVX2::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, relu, cn0 == 0);
                        else
                    #endif
                    #if CV_TRY_AVX
                        if(useAVX)
                            opt_AVX::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                         outShape, bsz, vsz, vsz_a, relu, cn0 == 0);
                        else
                    #endif
                        for( int i = 0; i < outCn; i += 2 )
                        {
                            const float* wptr0 = wptr + i*wstep;
                            const float* wptr1 = wptr0 + wstep;
                            float* outptr0 = data_out0 + ofs0 + i*outPlaneSize;
                            float* outptr1 = outptr0 + outPlaneSize;
                            float bias0 = biasptr[i], bias1 = biasptr[i+1];
                            float r0 = 1.f, r1 = 1.f;

                            if( i+1 >= outCn )
                            {
                                wptr1 = wptr0;
                                outptr1 = outptr0;
                                bias1 = bias0;
                            }

                            if( relu )
                            {
                                r0 = relu[i]; r1 = relu[i+1];
                                if( i+1 >= outCn )
                                    r1 = r0;
                            }

                            int j = 0;
                        #if CV_SIMD128
                            v_float32x4 vr0 = v_setall_f32(r0), vr1 = v_setall_f32(r1), z = v_setzero_f32();

                            for( ; j <= bsz - 4; j += 4 )
                            {
                                const float* rptr = rowbuf0 + j*vsz_a;
                                v_float32x4 s0, s1;

                                if( cn0 == 0 )
                                {
                                    s0 = v_setall_f32(bias0);
                                    s1 = v_setall_f32(bias1);
                                }
                                else
                                {
                                    s0 = v_load(outptr0 + j);
                                    s1 = v_load(outptr1 + j);
                                }

                                v_float32x4 vs00 = v_setzero_f32(), vs01 = v_setzero_f32(),
                                            vs02 = v_setzero_f32(), vs03 = v_setzero_f32(),
                                            vs10 = v_setzero_f32(), vs11 = v_setzero_f32(),
                                            vs12 = v_setzero_f32(), vs13 = v_setzero_f32();
                                for( k = 0; k < vsz; k += 4, rptr += 4 )
                                {
                                    v_float32x4 w0 = v_load_aligned(wptr0 + k);
                                    v_float32x4 w1 = v_load_aligned(wptr1 + k);
                                    v_float32x4 r0 = v_load_aligned(rptr);
                                    v_float32x4 r1 = v_load_aligned(rptr + vsz_a);
                                    v_float32x4 r2 = v_load_aligned(rptr + vsz_a*2);
                                    v_float32x4 r3 = v_load_aligned(rptr + vsz_a*3);

                                    vs00 = v_fma(w0, r0, vs00);
                                    vs01 = v_fma(w0, r1, vs01);
                                    vs02 = v_fma(w0, r2, vs02);
                                    vs03 = v_fma(w0, r3, vs03);

                                    vs10 = v_fma(w1, r0, vs10);
                                    vs11 = v_fma(w1, r1, vs11);
                                    vs12 = v_fma(w1, r2, vs12);
                                    vs13 = v_fma(w1, r3, vs13);
                                }
                                s0 += v_reduce_sum4(vs00, vs01, vs02, vs03);
                                s1 += v_reduce_sum4(vs10, vs11, vs12, vs13);
                                if( relu )
                                {
                                    s0 = v_select(s0 > z, s0, s0*vr0);
                                    s1 = v_select(s1 > z, s1, s1*vr1);
                                }

                                v_store(outptr0 + j, s0);
                                v_store(outptr1 + j, s1);
                            }
                        #endif
                            for( ; j < bsz; j++ )
                            {
                                const float* rptr = rowbuf0 + j*vsz_a;
                                float s00, s10;

                                if( cn0 == 0 )
                                {
                                    s00 = bias0;
                                    s10 = bias1;
                                }
                                else
                                {
                                    s00 = outptr0[j];
                                    s10 = outptr1[j];
                                }

                                for( k = 0; k < vsz; k++ )
                                {
                                    float r0 = rptr[k];
                                    s00 += wptr0[k]*r0;
                                    s10 += wptr1[k]*r0;
                                }
                                if( relu )
                                {
                                    s00 = s00 > 0.f ? s00 : s00*r0;
                                    s10 = s10 > 0.f ? s10 : s10*r1;
                                }

                                outptr0[j] = s00;
                                outptr1[j] = s10;
                            }
                        }
                    }
                }

                if( activ_ )
                    activ_->forwardSlice(data_out0 + stripeStart, data_out0 + stripeStart,
                                         (int)(stripeEnd - stripeStart),
                                         outPlaneSize, startOutCn, startOutCn + outCn);
            }
        }
    };

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        if (kernel_size.size() != 2)
        {
            // no OpenCL optimizations, see .supportedBacked()
            return false;
        }

        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inps.depth() == CV_16S);
        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        CV_Assert(outputs.size() == 1);
        for (int i = 0; i < inputs.size(); ++i)
            CV_Assert(inputs[i].u != outputs[0].u);

        if (blobs.empty())
        {
            size_t n = inputs.size() - 1;
            umat_blobs.resize(n);
            for (size_t i = 0; i < n; i++)
            {
                inputs[i + 1].copyTo(umat_blobs[i]);
            }
            inputs.resize(1);
        }

        if (umat_blobs.empty())
        {
            size_t n = blobs.size();
            umat_blobs.resize(n);
            for (size_t i = 0; i < n; i++)
            {
                if (use_half)
                    convertFp16(blobs[i], umat_blobs[i]);
                else
                    blobs[i].copyTo(umat_blobs[i]);
            }
        }

        if (convolutionOp.empty() || blobs.empty())
        {
            OCL4DNNConvConfig config;
            config.in_shape = shape(inputs[0]);
            config.out_shape = shape(outputs[0]);
            config.kernel = kernel;
            config.pad = pad;
            config.stride = stride;
            config.dilation = dilation;
            config.group = inputs[0].size[1] / umat_blobs[0].size[1];
            config.bias_term = umat_blobs.size() == 2;
            config.use_half = use_half;

            convolutionOp = Ptr<OCL4DNNConvSpatial<float> >(new OCL4DNNConvSpatial<float>(config));
        }

        int outCn = umat_blobs[0].size[0];

        reluslope.clear();
        if( activ )
        {
            Ptr<ReLULayer> activ_relu = activ.dynamicCast<ReLULayer>();
            if( !activ_relu.empty() )
            {
                reluslope.assign(outCn+2, activ_relu->negativeSlope);
                activType = OCL4DNN_CONV_FUSED_ACTIV_RELU;
            }

            Ptr<ReLU6Layer> activ_relu6 = activ.dynamicCast<ReLU6Layer>();
            if( !activ_relu6.empty() )
            {
                reluslope.resize(2);
                reluslope[0] = activ_relu6->minValue;
                reluslope[1] = activ_relu6->maxValue;
                activType = OCL4DNN_CONV_FUSED_ACTIV_RELU6;
            }

            Ptr<ChannelsPReLULayer> activ_chprelu = activ.dynamicCast<ChannelsPReLULayer>();
            if( !activ_chprelu.empty() )
            {
                const Mat& m = activ_chprelu->blobs[0];
                CV_Assert(m.isContinuous() && m.type() == CV_32F && (int)m.total() == outCn);
                const float* mdata = m.ptr<float>();
                reluslope.resize(outCn+2);
                std::copy(mdata, mdata + outCn, reluslope.begin());
                reluslope[outCn] = reluslope[outCn+1] = reluslope[outCn-1];
                activType = OCL4DNN_CONV_FUSED_ACTIV_PRELU;
            }
        }

        if (fusedWeights)
        {
            if (use_half)
                convertFp16(weightsMat, umat_blobs[0]);
            else
                weightsMat.copyTo(umat_blobs[0]);
            fusedWeights = false;
        }
        if (fusedBias)
        {
            if ( umat_blobs.size() < 2 )
                umat_blobs.resize(2);
            if (use_half)
                convertFp16(Mat(biasvec, true), umat_blobs[1]);
            else
                Mat(biasvec, true).copyTo(umat_blobs[1]);
            convolutionOp->setBias(true);
            fusedBias = false;
        }

        if ( newActiv )
        {
            if ( activType == OCL4DNN_CONV_FUSED_ACTIV_RELU )
            {
                CV_Assert(!reluslope.empty());
                convolutionOp->setActivReLU(true, reluslope[0]);
            }
            else if ( activType == OCL4DNN_CONV_FUSED_ACTIV_PRELU)
            {
                CV_Assert(!reluslope.empty());
                convolutionOp->setActivPReLU(true, reluslope);
            }
            else if ( activType == OCL4DNN_CONV_FUSED_ACTIV_POWER)
            {
                convolutionOp->setActivPower(true, power);
            }
            else if ( activType == OCL4DNN_CONV_FUSED_ACTIV_TANH)
            {
                convolutionOp->setActivTanh(true);
            }
            else if ( activType == OCL4DNN_CONV_FUSED_ACTIV_RELU6)
            {
                convolutionOp->setActivReLU6(true, reluslope[0], reluslope[1]);
            }
            else
            {
                convolutionOp->setActivReLU(false, 0);
                convolutionOp->setActivPReLU(false, reluslope);
                convolutionOp->setActivPower(false, 1.f);
                convolutionOp->setActivTanh(false);
                convolutionOp->setActivReLU6(false, 0, 0);
            }
            newActiv = false;
        }

        UMat& inpMat = inputs[0];
        UMat& outMat = outputs[0];
        int batch_size = inpMat.size[0];

        return convolutionOp->Forward(inpMat,
                                      inputs.size() == 2 ? inputs[1] : UMat(),
                                      umat_blobs[0],
                                      umat_blobs.size() > 1 ? umat_blobs[1] : UMat(),
                                      outMat,
                                      batch_size);
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

#if CV_SSE3
        uint32_t ftzMode = _MM_GET_FLUSH_ZERO_MODE();
        uint32_t dazMode = _MM_GET_DENORMALS_ZERO_MODE();
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        int outCn = blobs.empty() ? inputs[1].size[0] : blobs[0].size[0];
        // Need to align non-const blobs
        if (blobs.empty())
        {
            Mat wm = inputs[1].reshape(1, outCn);
            if (wm.data != weightsMat.data)
            {
                int newcols = (int)alignSize(wm.step1(), VEC_ALIGN);
                Mat wm_buffer = Mat(numOutput, newcols, wm.type());
                Mat wm_padding = wm_buffer.colRange(wm.cols, newcols);
                wm_padding.setTo(Scalar::all(0.));
                weightsMat = wm_buffer.colRange(0, wm.cols);

                wm.copyTo((const Mat&)weightsMat);
                if (inputs.size() > 2)
                {
                    Mat biasMat = inputs[2].reshape(1, outCn);
                    biasMat.col(0).copyTo(biasvec);
                }
                biasvec.resize(outCn + 2, 0);
            }
        }
        /*if (inputs[0].dims > 3) {
            printf("conv %s: input (%d x %d x %d x %d), kernel (%d x %d), pad (%d x %d), stride (%d x %d), dilation (%d x %d)\n",
                   name.c_str(), inputs[0].size[0], inputs[0].size[1], inputs[0].size[2], inputs[0].size[3],
                   kernel.width, kernel.height, pad.width, pad.height,
                   stride.width, stride.height, dilation.width, dilation.height);
        }
        else {
            printf("conv %s: input (%d x %d x %d), kernel (%d x %d), pad (%d x %d), stride (%d x %d), dilation (%d x %d)\n",
                   name.c_str(), inputs[0].size[0], inputs[0].size[1], inputs[0].size[2],
                   kernel.width, kernel.height, pad.width, pad.height,
                   stride.width, stride.height, dilation.width, dilation.height);
        }*/
        int inpGroupCn = blobs.empty() ? inputs[1].size[1] : blobs[0].size[1];
        CV_Assert_N(inputs.size() >= (size_t)1, inputs[0].size[1] % inpGroupCn == 0,
                    outputs.size() == 1, inputs[0].data != outputs[0].data);

        int ngroups = inputs[0].size[1] / inpGroupCn;
        CV_Assert(outputs[0].size[1] % ngroups == 0);

        reluslope.clear();
        if( activ )
        {
            Ptr<ReLULayer> activ_relu = activ.dynamicCast<ReLULayer>();
            if( !activ_relu.empty() )
            {
                reluslope.assign(outCn+2, activ_relu->negativeSlope);
            }

            Ptr<ChannelsPReLULayer> activ_chprelu = activ.dynamicCast<ChannelsPReLULayer>();
            if( !activ_chprelu.empty() )
            {
                const Mat& m = activ_chprelu->blobs[0];
                CV_Assert(m.isContinuous() && m.type() == CV_32F && (int)m.total() == outCn);
                const float* mdata = m.ptr<float>();
                reluslope.resize(outCn+2);
                std::copy(mdata, mdata + outCn, reluslope.begin());
                reluslope[outCn] = reluslope[outCn+1] = reluslope[outCn-1];
            }
        }

#ifdef HAVE_TENGINE
        bool tengine_ret = false; ;

        std::vector<Mat> teng_in, teng_out;
        inputs_arr.getMatVector(teng_in);
        outputs_arr.getMatVector(teng_out);

        int inch = teng_in[0].size[1];    // inch
        int in_h = teng_in[0].size[2];    // in_h
        int in_w = teng_in[0].size[3];    // in_w

        int out_b = teng_out[0].size[0];  // out batch size
        int outch = teng_out[0].size[1];  // outch
        int out_h = teng_out[0].size[2];  // out_h
        int out_w = teng_out[0].size[3];  // out_w

        float *input_  = teng_in[0].ptr<float>();
        float *output_ = teng_out[0].ptr<float>();
        float *kernel_ = weightsMat.ptr<float>();
        float *teg_bias = &biasvec[0];

        int nstripes = std::max(getNumThreads(), 1);

        /* tengine_init will run when first time. */
        if(NULL == tengine_graph)
        {
            tengine_graph = tengine_init(name.c_str(), input_, inch, ngroups, in_h, in_w,
                                         output_, out_b, outch, out_h, out_w,
                                         kernel_, kernel_size.size(), kernel.height, kernel.width,
                                         teg_bias, stride.height, stride.width,
                                         pad.height,  pad.width, dilation.height, dilation.width,
                                         weightsMat.step1(), padMode, tengine_graph, nstripes);
            /*printf("Init(%s):  input=%p(%d %d %d %d ),output=%p(%d %d %d %d ),kernel=%p(%ld %d %d ), bias=%p ,"
                   "stride(%d %d), pad(%d %d), dilation(%d %d) ,weightsMat=%ld, padMode=%s ,tengine_graph = %p \n",
                   name.c_str(),input_, inch, ngroups, in_h, in_w,
                   output_, out_b, outch, out_h, out_w,
                   kernel_, kernel_size.size(), kernel.height, kernel.width,
                   teg_bias, stride.height, stride.width,
                   pad.height,  pad.width, dilation.height, dilation.width,
                   weightsMat.step1(), padMode.c_str() ,tengine_graph);*/
        }
        if(NULL != tengine_graph)
        {
            tengine_ret = tengine_forward(tengine_graph);
        }
        /* activation */
        if((true == tengine_ret) && activ )
        {
            int out_cstep = out_h * out_w;	    // out_cstep

            ActivationLayer* activ_ = activ.get();
            activ_->forwardSlice(output_, output_, out_cstep, out_cstep, 0, outch);
        }
        if(false == tengine_ret)
#endif
        {
            int nstripes = std::max(getNumThreads(), 1);

            ParallelConv::run(inputs[0], outputs[0], weightsMat, biasvec, reluslope,
                            kernel_size, strides, pads_begin, pads_end, dilations, activ.get(), ngroups, nstripes);
        }
#if CV_SSE3
        _MM_SET_FLUSH_ZERO_MODE(ftzMode);
        _MM_SET_DENORMALS_ZERO_MODE(dazMode);
#endif
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        CV_Assert(inputs.size() == 1 || inputs.size() == 2);
        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();

        CV_Assert(outputs.size() == 1);
        auto output_wrapper = outputs[0].dynamicCast<CUDABackendWrapper>();
        auto output_shape = output_wrapper->getShape();

        const auto output_feature_maps = blobs[0].size[0];
        const auto input_feature_maps = input_shape[1];
        const auto input_feature_maps_per_group = blobs[0].size[1];
        const auto groups = input_feature_maps / input_feature_maps_per_group;

        ConvolutionConfiguration config;

        if (input_shape.size() == 3)
        {
            // Conv1D
            // We add an extra dim for input and output tensors, because CuDNN doesn't support convolution with 3D tensors
            input_shape.insert(std::end(input_shape) - 1, 1);
            output_shape.insert(std::end(output_shape) - 1, 1);

            // Do the similar thing for the other parameters
            pads_begin.insert(std::begin(pads_begin), 0);
            pads_end.insert(std::begin(pads_end), 0);
            strides.insert(std::begin(strides), 1);
            dilations.insert(std::begin(dilations), 1);
            kernel_size.insert(std::begin(kernel_size), 1);
        }
        config.kernel_size.assign(std::begin(kernel_size), std::end(kernel_size));
        config.dilations.assign(std::begin(dilations), std::end(dilations));
        config.strides.assign(std::begin(strides), std::end(strides));

        if (padMode.empty())
        {
            config.padMode = ConvolutionConfiguration::PaddingMode::MANUAL;
            config.pads_begin.assign(std::begin(pads_begin), std::end(pads_begin));
            config.pads_end.assign(std::begin(pads_end), std::end(pads_end));
        }
        else if (padMode == "VALID")
        {
            config.padMode = ConvolutionConfiguration::PaddingMode::VALID;
        }
        else if (padMode == "SAME")
        {
            config.padMode = ConvolutionConfiguration::PaddingMode::SAME;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, padMode + " padding mode not supported by ConvolutionLayer");
        }

        config.input_shape.assign(std::begin(input_shape), std::end(input_shape));
        config.output_shape.assign(std::begin(output_shape), std::end(output_shape));
        config.groups = groups;

        config.fusion_mode = cudaFusionMode;
        config.activation_type = cudaActType;
        config.relu_negative_slope = cuda_relu_slope;
        config.crelu_floor = cuda_crelu_floor;
        config.crelu_ceil = cuda_crelu_ceil;
        config.power_exp = cuda_power_exp;
        config.power_scale = cuda_power_scale;
        config.power_shift = cuda_power_shift;

        Mat filtersMat = fusedWeights ? weightsMat : blobs[0];
        Mat biasMat = (hasBias() || fusedBias) ? Mat(output_feature_maps, 1, CV_32F, biasvec.data()) : Mat();
        if (countNonZero(biasMat) == 0)
            biasMat = Mat();

        return make_cuda_node<cuda4dnn::ConvolutionOp>(
            preferableTarget, std::move(context->stream), std::move(context->cudnn_handle), config, filtersMat, biasMat);
    }
#endif

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == outputs.size() || inputs.size() == outputs.size() + blobs.size());

        int64 flops = 0;
        int karea = std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<size_t>());
        for (int i = 0; i < outputs.size(); i++)
        {
            flops += total(outputs[i])*(CV_BIG_INT(2)*karea*inputs[i][1] + 1);
        }

        return flops;
    }
};

class DeConvolutionLayerImpl CV_FINAL : public BaseConvolutionLayerImpl
{
public:
    Mat weightsMat, biasesMat;
    UMat umat_weights;
    UMat umat_biases;

    DeConvolutionLayerImpl(const LayerParams& params) : BaseConvolutionLayerImpl(params) {}

    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const CV_OVERRIDE
    {
        int dims = inpShape.size();
        int inpCn = inpShape[1];
        int inpD = dims == 5 ? inpShape[2] : 1;
        int inpH = inpShape[dims - 2];
        int inpW = inpShape.back();
        int outCn = outShape[1];
        int ngroups = inpCn / blobs[0].size[0];
        int outGroupCn = outCn / ngroups;
        int ksize = outGroupCn * std::accumulate(kernel_size.begin(), kernel_size.end(),
                                                 1, std::multiplies<size_t>());
        return shape(ksize, inpD * inpH * inpW);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_CUDA)
        {
            /* only deconvolution 2d and 3d supported */
            if (kernel_size.size() == 2 || kernel_size.size() == 3)
                return true;

            return false;
        }

#ifdef HAVE_INF_ENGINE
        const int outGroupCn = blobs[0].size[1];  // Weights are in IOHW or IODHW layout
        const int group = numOutput / outGroupCn;

        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
            return group == 1;
        }

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        {
            if (kernel_size.size() == 3 && preferableTarget != DNN_TARGET_CPU) {
                return false;
            }

            if (std::accumulate(adjust_pads.begin(), adjust_pads.end(), 0, std::plus<size_t>()) > 0)
            {
                if (padMode.empty())
                {
                    if (preferableTarget != DNN_TARGET_CPU && group != 1)
                    {
                        for (int i = 0; i < adjust_pads.size(); i++) {
                            if (adjust_pads[i] && pads_begin[i])
                                return false;
                        }
                    }
                    for (int i = 0; i < adjust_pads.size(); i++) {
                        if (pads_end[i] < adjust_pads[i])
                            return false;
                    }
                    return true;
                }
                else if (padMode == "SAME")
                {
                    for (int i = 0; i < adjust_pads.size(); i++) {
                        if (kernel_size[i] < pads_begin[i] + 1 + adjust_pads[i])
                            return false;
                    }
                    return true;
                }
                else if (padMode == "VALID")
                    return false;
            }

            if (group != 1)
            {
                return preferableTarget == DNN_TARGET_CPU;
            }
            if (preferableTarget == DNN_TARGET_OPENCL || preferableTarget == DNN_TARGET_OPENCL_FP16)
                return std::accumulate(dilations.begin(), dilations.end(), 1, std::multiplies<size_t>()) == 1;
            return true;
        }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019
#endif  // HAVE_INF_ENGINE
        {
            return backendId == DNN_BACKEND_CUDA ||
            (kernel_size.size() == 2 && (backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_HALIDE));
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!hasBias() || blobs[1].total() == (size_t)numOutput);
        CV_Assert(inputs.size() != 0);

        int outCn = numOutput;
        std::vector<int> outShape;
        outShape.push_back(inputs[0][0]);  // batch
        outShape.push_back(outCn);
        if (padMode.empty())
        {
            for (int i = 0; i < kernel_size.size(); i++)
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + kernel_size[i] - pads_begin[i] - pads_end[i] + adjust_pads[i]);
        }
        else if (padMode == "VALID")
        {
            for (int i = 0; i < kernel_size.size(); i++)
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + kernel_size[i] + adjust_pads[i]);
        }
        else if (padMode == "SAME")
        {
            for (int i = 0; i < kernel_size.size(); i++)
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + 1 + adjust_pads[i]);
        }
        else
            CV_Error(Error::StsError, "Unsupported padding mode " + padMode);

        CV_Assert(outCn % blobs[0].size[1] == 0);
        int ngroups = outCn / blobs[0].size[1];

        int inpCn = inputs[0][1];
        CV_Assert(inpCn % ngroups == 0 && outCn % ngroups == 0);
        CV_Assert(blobs[0].size[0] == inpCn);

        outputs.resize(1, outShape);

        if (!is1x1())
            internals.push_back(computeColRowShape(inputs[0], outputs[0]));

        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        BaseConvolutionLayerImpl::finalize(inputs_arr, outputs_arr);

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        std::vector<int> inpShape;
        std::vector<int> outShape;
        for (int i = 2; i < inputs[0].dims; i++) {
            inpShape.push_back(inputs[0].size[i]);
            outShape.push_back(outputs[0].size[i]);
        }
        getConvPoolPaddings(outShape, kernel_size, strides, padMode, pads_begin, pads_end);
        if (pads_begin.size() == 2) {
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in deconvolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
        }

        weightsMultipliers.assign(numOutput, 1.0);
        if (weightsMat.empty())
        {
            transpose(blobs[0].reshape(1, blobs[0].size[0]), weightsMat);
            biasesMat = hasBias() ? blobs[1].reshape(1, numOutput)
                                  : Mat::zeros(numOutput, 1, CV_32F);
        }
    }

    void fuseWeights(const Mat& w_, const Mat& b_) CV_OVERRIDE
    {
        Mat w = w_.total() == 1 ? Mat(1, numOutput, CV_32F, Scalar(w_.at<float>(0))) : w_;
        Mat b = b_.total() == 1 ? Mat(1, numOutput, CV_32F, Scalar(b_.at<float>(0))) : b_;

        CV_Assert_N(!weightsMat.empty(),
                     w.empty() || numOutput == w.total(),
                     b.empty() || numOutput == b.total());

        if (!w.empty())
        {
            transpose(blobs[0].reshape(1, blobs[0].size[0]), weightsMat);
            weightsMat = weightsMat.reshape(1, numOutput);
            for (int i = 0; i < numOutput; ++i)
            {
                double wi = w.at<float>(i);
                weightsMultipliers[i] *= wi;
                cv::multiply(weightsMat.row(i), weightsMultipliers[i], weightsMat.row(i));
                biasesMat.at<float>(i) *= wi;
            }
            weightsMat = weightsMat.reshape(1, weightsMat.total() / blobs[0].size[0]);
        }

        if (!b.empty())
        {
            cv::add(biasesMat, b.reshape(1, numOutput), biasesMat);
        }
    }

    class MatMulInvoker : public ParallelLoopBody
    {
    public:
        MatMulInvoker(const Mat& a, const Mat& b, Mat& c, int nstripes)
        {
            a_ = &a;
            b_ = &b;
            c_ = &c;
            nstripes_ = nstripes;
            useAVX = checkHardwareSupport(CPU_AVX);
            useAVX2 = checkHardwareSupport(CPU_AVX2);
            useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX;
        }

        void operator()(const Range& range_) const CV_OVERRIDE
        {
            int stripeSize = (int)alignSize((b_->cols + nstripes_ - 1)/nstripes_, 16);
            Range range(range_.start*stripeSize, std::min(range_.end*stripeSize, b_->cols));
            int mmax = a_->rows;
            int nmax = range.end - range.start;
            int kmax = a_->cols;
            int m, n, k;
            const float* aptr = a_->ptr<float>();
            const float* bptr = b_->ptr<float>() + range.start;
            float* cptr = c_->ptr<float>() + range.start;
            size_t astep = a_->step1();
            size_t bstep = b_->step1();
            size_t cstep = c_->step1();

        #if CV_TRY_AVX512_SKX
            if( useAVX512 )
                opt_AVX512_SKX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
        #if CV_TRY_AVX2
            if( useAVX2 )
                opt_AVX2::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
        #if CV_TRY_AVX
            if( useAVX )
                opt_AVX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
            for( m = 0; m < mmax; m += 2 )
            {
                float* dst0 = cptr + cstep*m;
                float* dst1 = cptr + cstep*std::min(m+1, mmax-1);
                const float* aptr0 = aptr + astep*m;
                const float* aptr1 = aptr + astep*std::min(m+1, mmax-1);

                for( n = 0; n < nmax; n++ )
                {
                    dst0[n] = 0.f;
                    dst1[n] = 0.f;
                }

                for( k = 0; k < kmax; k += 4 )
                {
                    float alpha00 = aptr0[k];
                    float alpha01 = aptr1[k];
                    float alpha10 = 0.f, alpha11 = 0.f;
                    float alpha20 = 0.f, alpha21 = 0.f;
                    float alpha30 = 0.f, alpha31 = 0.f;
                    const float* bptr0 = bptr + k*bstep;
                    const float* bptr1 = bptr0;
                    const float* bptr2 = bptr0;
                    const float* bptr3 = bptr0;

                    if( k+1 < kmax )
                    {
                        alpha10 = aptr0[k+1];
                        alpha11 = aptr1[k+1];
                        bptr1 = bptr0 + bstep;
                        if( k+2 < kmax )
                        {
                            alpha20 = aptr0[k+2];
                            alpha21 = aptr1[k+2];
                            bptr2 = bptr1 + bstep;
                            if( k+3 < kmax )
                            {
                                alpha30 = aptr0[k+3];
                                alpha31 = aptr1[k+3];
                                bptr3 = bptr2 + bstep;
                            }
                        }
                    }
                    n = 0;

                #if CV_SIMD128
                    v_float32x4 a00 = v_setall_f32(alpha00);
                    v_float32x4 a01 = v_setall_f32(alpha01);
                    v_float32x4 a10 = v_setall_f32(alpha10);
                    v_float32x4 a11 = v_setall_f32(alpha11);
                    v_float32x4 a20 = v_setall_f32(alpha20);
                    v_float32x4 a21 = v_setall_f32(alpha21);
                    v_float32x4 a30 = v_setall_f32(alpha30);
                    v_float32x4 a31 = v_setall_f32(alpha31);

                    for( ; n <= nmax - 4; n += 4 )
                    {
                        v_float32x4 d0 = v_load(dst0 + n);
                        v_float32x4 d1 = v_load(dst1 + n);
                        v_float32x4 b0 = v_load(bptr0 + n);
                        v_float32x4 b1 = v_load(bptr1 + n);
                        v_float32x4 b2 = v_load(bptr2 + n);
                        v_float32x4 b3 = v_load(bptr3 + n);
                        // TODO try to improve pipeline width
                        d0 = v_fma(b0, a00, d0);
                        d1 = v_fma(b0, a01, d1);
                        d0 = v_fma(b1, a10, d0);
                        d1 = v_fma(b1, a11, d1);
                        d0 = v_fma(b2, a20, d0);
                        d1 = v_fma(b2, a21, d1);
                        d0 = v_fma(b3, a30, d0);
                        d1 = v_fma(b3, a31, d1);
                        v_store(dst0 + n, d0);
                        v_store(dst1 + n, d1);
                    }
                #endif

                    for( ; n < nmax; n++ )
                    {
                        float b0 = bptr0[n];
                        float b1 = bptr1[n];
                        float b2 = bptr2[n];
                        float b3 = bptr3[n];
                        float d0 = dst0[n] + alpha00*b0 + alpha10*b1 + alpha20*b2 + alpha30*b3;
                        float d1 = dst1[n] + alpha01*b0 + alpha11*b1 + alpha21*b2 + alpha31*b3;
                        dst0[n] = d0;
                        dst1[n] = d1;
                    }
                }
            }
        }

        const Mat *a_, *b_;
        Mat* c_;
        int nstripes_;
        bool useAVX;
        bool useAVX2;
        bool useAVX512;
    };

    class Col2ImInvoker : public cv::ParallelLoopBody
    {
    public:
        const float* data_col;
        const float* biasvec;
        int channels, height, width;
        int kernel_h, kernel_w;
        int pad_h, pad_w;
        int stride_h, stride_w;
        float* data_im;
        int height_col, width_col;
        int nstripes;
        bool is1x1;

        Col2ImInvoker()
            : data_col(0), biasvec(0), channels(0), height(0), width(0),
              kernel_h(0), kernel_w(0), pad_h(0), pad_w(0), stride_h(0), stride_w(0), data_im(0),
              height_col(0), width_col(0), nstripes(0), is1x1(0)
        {}

        static void run(const float* data_col,
                        int channels, int height, int width,
                        int kernel_h, int kernel_w,
                        int pad_h, int pad_w,
                        int stride_h, int stride_w,
                        int height_col, int width_col,
                        float* data_im,
                        const float* biasvec,
                        bool is1x1)
        {
            const int nstripes = getNumThreads();

            Col2ImInvoker t;
            t.data_col = data_col;
            t.data_im = data_im;
            t.channels = channels; t.height = height; t.width = width;
            t.kernel_h = kernel_h; t.kernel_w = kernel_w;
            t.pad_h = pad_h; t.pad_w = pad_w;
            t.stride_h = stride_h; t.stride_w = stride_w;
            t.height_col = height_col;
            t.width_col = width_col;
            t.nstripes = nstripes;
            t.is1x1 = is1x1;
            t.biasvec = biasvec;

            parallel_for_(Range(0, nstripes), t, nstripes);
        }

        virtual void operator ()(const Range &r) const CV_OVERRIDE
        {
            const float* data_col_ = data_col;
            float* data_im_ = data_im;
            int coeff_h = (1 - stride_h * kernel_w * height_col) * width_col;
            int coeff_w = (1 - stride_w * height_col * width_col);
            size_t total = (size_t)channels * height * width;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t startIndex = r.start*stripeSize;
            size_t endIndex = std::min(r.end*stripeSize, total);
            int w = (int)(startIndex % width + pad_w);
            int h = (int)((startIndex / width) % height + pad_h);
            int c = (int)(startIndex / (width * height));
            int h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
            int h_col_end = std::min(h / stride_h + 1, height_col);
            int plane_size_col = height_col * width_col;
            int offset = (c * kernel_h * kernel_w + h * kernel_w + w) * plane_size_col;
            bool is1x1_ = is1x1;
            const float* biasvec_ = biasvec;

            for (size_t index = startIndex; index < endIndex; index++)
            {
                // compute the start and end of the output
                int w_col_start = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
                int w_col_end = std::min(w / stride_w + 1, width_col);
                float val;

                if( is1x1_ )
                    val = data_im_[index];
                else
                {
                    val = 0.f;
                    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
                        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                            val += data_col_[offset + h_col * coeff_h + w_col * coeff_w];
                        }
                    }
                }
                data_im_[index] = val + biasvec_[c];

                offset += plane_size_col;
                if( ++w >= width + pad_w )
                {
                    w = (int)((index + 1)% width + pad_w);
                    h = (int)(((index + 1) / width) % height + pad_h);
                    c = (int)((index + 1) / (width * height));
                    h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
                    h_col_end = std::min(h / stride_h + 1, height_col);
                    offset = (c * kernel_h * kernel_w + h * kernel_w + w) * plane_size_col;
                }
            }
        }
    };

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        std::vector<UMat> internals;

        if (inputs_.depth() == CV_16S)
            return false;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        internals_.getUMatVector(internals);

        int outCn = numOutput;
        int inpCn = inputs[0].size[1];

        if (is1x1())
            return false;

        if (umat_weights.empty())
        {
            if (fusedWeights)
                weightsMat.copyTo(umat_weights);
            else
                transpose(blobs[0].reshape(1, inpCn), umat_weights);

            if (fusedBias)
                biasesMat.copyTo(umat_biases);
            else
            {
                if (hasBias())
                    blobs[1].reshape(1, outCn).copyTo(umat_biases);
                else
                    umat_biases = UMat::zeros(outCn, 1, CV_32F);
            }
        }

        String buildopt = format("-DT=%s ", ocl::typeToStr(inputs[0].type()));
        buildopt += format("-DPAD_H=%d -DPAD_W=%d -DKERNEL_H=%d -DKERNEL_W=%d -DSTRIDE_H=%d -DSTRIDE_W=%d ",
                           pad.height, pad.width, kernel.height, kernel.width, stride.height, stride.width);

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int ngroups = outCn / blobs[0].size[1];
            int inpGroupCn = inpCn / ngroups;
            int outGroupCn = blobs[0].size[1];
            const UMat& inp = inputs[ii];
            UMat& out = outputs[ii];
            int numImg = inp.size[0];
            int inpH = inp.size[2], inpW = inp.size[3];
            int outH = out.size[2], outW = out.size[3];

            MatShape inpshape = shape(numImg*inpCn, inpH*inpW);
            MatShape outshape = shape(numImg*outCn, outH*outW);
            UMat convBlob = inputs[ii].reshape(1, inpshape.size(), &inpshape[0]);
            UMat decnBlob = out.reshape(1, outshape.size(), &outshape[0]);
            int rows = internals[0].rows / ngroups;

            for (int n = 0; n < numImg; n++)
            {
                for (int g = 0; g < ngroups; g++)
                {
                    UMat colMat = internals[0].rowRange(_Range(g * rows, rows));
                    UMat convMat = convBlob.rowRange(_Range((g + n * ngroups) * inpGroupCn, inpGroupCn));
                    UMat wghtMat = umat_weights.colRange(_Range(g * inpGroupCn, inpGroupCn));
                    gemm(wghtMat, convMat, 1, noArray(), 0, colMat, 0);
                }

                for (int g = 0; g < ngroups; g++)
                {
                    int total = outGroupCn * decnBlob.cols;
                    int index = 0;
                    int height_col = inpH;
                    int width_col = inpW;
                    int coeff_h = (1 - stride.height * kernel.width * height_col) * width_col;
                    int coeff_w = (1 - stride.width * height_col * width_col);

                    ocl::Kernel k("col2im", ocl::dnn::col2im_oclsrc, buildopt);
                    k.set(index++, total);
                    k.set(index++, ocl::KernelArg::PtrReadOnly(internals[0]));
                    k.set(index++, (int)(g * rows * internals[0].cols));
                    k.set(index++, outGroupCn);
                    k.set(index++, outH);
                    k.set(index++, outW);
                    k.set(index++, height_col);
                    k.set(index++, width_col);
                    k.set(index++, coeff_h);
                    k.set(index++, coeff_w);
                    k.set(index++, ocl::KernelArg::PtrReadOnly(umat_biases));
                    k.set(index++, (int)(g * outGroupCn * umat_biases.cols));
                    k.set(index++, ocl::KernelArg::PtrWriteOnly(decnBlob));
                    k.set(index++, (int)((g + n * ngroups) * outGroupCn * decnBlob.cols));

                    size_t global[] = { (size_t)total };
                    bool ret = k.run(1, global, NULL, false);
                    if (!ret)
                        return false;
                }
            }
        }

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr));

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        int outCn = numOutput;
        int inpCn = inputs[0].size[1];
        bool is1x1flag = is1x1();
        int nstripes = getNumThreads();

        if( weightsMat.empty() )
        {
            transpose(blobs[0].reshape(1, inpCn), weightsMat);
            biasesMat = hasBias() ? blobs[1].reshape(1, outCn) : Mat::zeros(outCn, 1, CV_32F);
        }

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int ngroups = outCn / blobs[0].size[1];
            int inpGroupCn = inpCn / ngroups;
            int outGroupCn = blobs[0].size[1];
            const Mat& inp = inputs[ii];
            Mat& out = outputs[ii];
            int numImg = inp.size[0];
            int inpH = inp.size[2], inpW = inp.size[3];
            int outH = out.size[2], outW = out.size[3];

            Mat convBlob = inputs[ii].reshape(1, numImg*inpCn);
            Mat decnBlob = out.reshape(1, numImg*outCn);

            for (int n = 0; n < numImg; n++)
            {
                for (int g = 0; g < ngroups; g++)
                {
                    Mat dstMat = decnBlob.rowRange(_Range((g + n * ngroups) * outGroupCn, outGroupCn));
                    Mat &colMat = is1x1flag ? dstMat : internals[0];

                    Mat convMat = convBlob.rowRange(_Range((g + n * ngroups) * inpGroupCn, inpGroupCn));
                    Mat wghtMat = weightsMat.colRange(_Range(g * inpGroupCn, inpGroupCn));
                    Mat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));

                    //gemm(wghtMat, convMat, 1, colMat, 0, colMat, 0);
                    MatMulInvoker mminvoker(wghtMat, convMat, colMat, nstripes);
                    parallel_for_(Range(0, nstripes), mminvoker, nstripes);

                    Col2ImInvoker::run(colMat.ptr<float>(), outGroupCn, outH, outW,
                                       kernel.height, kernel.width, pad.height, pad.width,
                                       stride.height, stride.width, inpH, inpW, dstMat.ptr<float>(),
                                       curBiasMat.ptr<float>(), is1x1flag);
                }
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        CV_Assert(inputs.size() == 1);
        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();

        CV_Assert(outputs.size() == 1);
        auto output_wrapper = outputs[0].dynamicCast<CUDABackendWrapper>();
        auto output_shape = output_wrapper->getShape();

        const auto output_feature_maps = numOutput;
        const auto output_feature_maps_per_group = blobs[0].size[1];
        const auto groups = output_feature_maps / output_feature_maps_per_group;

        TransposeConvolutionConfiguration config;
        config.kernel_size.assign(std::begin(kernel_size), std::end(kernel_size));
        config.dilations.assign(std::begin(dilations), std::end(dilations));
        config.strides.assign(std::begin(strides), std::end(strides));

        if (padMode.empty())
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::MANUAL;
            config.pads_begin.assign(std::begin(pads_begin), std::end(pads_begin));
            config.pads_end.assign(std::begin(pads_end), std::end(pads_end));
        }
        else if (padMode == "VALID")
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::VALID;
        }
        else if (padMode == "SAME")
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::SAME;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, padMode + " padding mode not supported by DeconvolutionLayer");
        }

        config.input_shape.assign(std::begin(input_shape), std::end(input_shape));
        config.output_shape.assign(std::begin(output_shape), std::end(output_shape));
        config.groups = groups;

        CV_Assert(blobs.size() >= 1);
        Mat filtersMat = fusedWeights ? weightsMat.t() : blobs[0];

        Mat biasMat = (hasBias() || fusedBias) ? biasesMat : Mat();
        if (countNonZero(biasMat) == 0)
            biasMat = Mat();

        return make_cuda_node<cuda4dnn::TransposeConvolutionOp>(
            preferableTarget, std::move(context->stream), std::move(context->cudnn_handle), config, filtersMat, biasMat);
    }
#endif

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);

        int inW, inH, inC, inN;
        getCanonicalSize(inputBuffer, &inW, &inH, &inC, &inN);
        const int outGroupCn = blobs[0].size[1];
        const int group = numOutput / outGroupCn;
        const int inpGroupCn = blobs[0].size[0] / group;

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Func padded_input(name + "_constant_exterior");
        auto weights = wrapToHalideBuffer(blobs[0]);

        Halide::Func dilated_input("dilated_input");
        dilated_input(x, y, c, n) = 0.0f;
        Halide::RDom r1(0, inW, 0, inH);
        dilated_input(r1.x * stride.width, r1.y * stride.height, c, n) =
              inputBuffer(r1.x, r1.y, c, n);
        dilated_input.compute_root();

        Halide::Func bounded =
            Halide::BoundaryConditions::constant_exterior(dilated_input, 0,
                                                          0, (inW - 1) * stride.width + 1,
                                                          0, (inH - 1) * stride.height + 1,
                                                          0, inC, 0, inN);
        padded_input(x, y, c, n) = bounded(x, y, c, n);

        Halide::RDom r(0, kernel.width, 0, kernel.height, 0, inpGroupCn);
        Halide::Expr kx = x + pad.width - r.x;
        Halide::Expr ky = y + pad.height - r.y;
        Halide::Expr kInC = r.z;
        Halide::Expr kOutC = c;
        for (int i = 1; i < group; ++i)
        {
            kInC = select(c < outGroupCn * i, kInC, inpGroupCn * i + r.z);
            kOutC = select(c < outGroupCn * i, kOutC, c - outGroupCn * i);
        }
        Halide::Expr topExpr = sum(padded_input(kx, ky, kInC, n) *
                                   weights(r.x, r.y, kOutC, kInC));
        if (hasBias())
        {
            auto bias = wrapToHalideBuffer(blobs[1], {numOutput});
            topExpr += bias(c);
        }
        top(x, y, c, n) = topExpr;
        return Ptr<BackendNode>(new HalideBackendNode({ padded_input, top }));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> > &) CV_OVERRIDE
    {
        InferenceEngine::Layout layout = blobs[0].dims == 5? InferenceEngine::Layout::NCDHW :
                                                             InferenceEngine::Layout::OIHW;

        auto ieWeights = wrapToInfEngineBlob(blobs[0], layout);
        if (fusedWeights)
        {
            ieWeights = InferenceEngine::make_shared_blob<float>({
                            InferenceEngine::Precision::FP32,
                            ieWeights->getTensorDesc().getDims(), layout
                        });
            ieWeights->allocate();

            int inpCn = blobs[0].size[0];
            Mat newWeights = infEngineBlobToMat(ieWeights).reshape(1, inpCn);
            transpose(weightsMat, newWeights);
        }

        const int outGroupCn = blobs[0].size[1];  // Weights are in IOHW or OIDHW layout
        const int group = numOutput / outGroupCn;

        InferenceEngine::Builder::DeconvolutionLayer ieLayer(name);

        ieLayer.setKernel(kernel_size);
        ieLayer.setStrides(strides);
        ieLayer.setDilation(dilations);
        ieLayer.setPaddingsBegin(pads_begin);

        if (padMode.empty())
        {
            std::vector<size_t> paddings_end;
            for (int i = 0; i < pads_end.size(); i++) {
                paddings_end.push_back(pads_end[i] - adjust_pads[i]);
            }
            ieLayer.setPaddingsEnd(paddings_end);
        }
        else if (padMode == "SAME")
        {
            std::vector<size_t> paddings_end;
            for (int i = 0; i < pads_begin.size(); i++) {
                paddings_end.push_back(kernel_size[i] - pads_begin[i] - 1 - adjust_pads[i]);
            }
            ieLayer.setPaddingsEnd(paddings_end);
        }
        ieLayer.setGroup((size_t)group);
        ieLayer.setOutDepth((size_t)numOutput);

        InferenceEngine::Builder::Layer l = ieLayer;
        addConstantData("weights", ieWeights, l);
        if (hasBias())
            addConstantData("biases", wrapToInfEngineBlob(biasesMat, {(size_t)numOutput}, InferenceEngine::Layout::C), l);
        return Ptr<BackendNode>(new InfEngineBackendNode(l));
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
       const int outGroupCn = blobs[0].size[1];
       const int group = numOutput / outGroupCn;
       CV_Assert(group == 1);

       auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
       std::vector<size_t> kernel_shape = getShape<size_t>(blobs[0]);
       auto ieWeights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, kernel_shape, blobs[0].data);

        if (fusedWeights)
        {
            Mat newWeights;
            transpose(weightsMat, newWeights);
            ieWeights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, kernel_shape, newWeights.data);
        }
        std::vector<size_t> paddings_end;
        if (padMode == "SAME")
        {
            for (int i = 0; i < pads_begin.size(); i++) {
                paddings_end.push_back(kernel_size[i] - pads_begin[i] - 1 - adjust_pads[i]);
            }
            adjust_pads = std::vector<size_t>(pads_begin.size(), 0);
        } else {
            paddings_end = pads_end;
        }
        ngraph::op::PadType pad_type = padMode == "VALID" ? ngraph::op::PadType::VALID : ngraph::op::PadType::EXPLICIT;

        auto deconv = std::make_shared<ngraph::op::v1::ConvolutionBackpropData>(
                          ieInpNode,
                          ieWeights,
                          ngraph::Strides(strides),
                          ngraph::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                          ngraph::CoordinateDiff(std::vector<std::ptrdiff_t>(paddings_end.begin(), paddings_end.end())),
                          ngraph::Strides(dilations),
                          pad_type,
                          ngraph::CoordinateDiff(std::vector<std::ptrdiff_t>(adjust_pads.begin(), adjust_pads.end())));

        if (hasBias() || fusedBias)
        {
            std::vector<size_t> shape(deconv->get_shape().size(), 1);
            shape[1] = numOutput;
            auto bias = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape(shape), blobs[1].data);
            auto deconv_bias = std::make_shared<ngraph::op::v1::Add>(deconv, bias, ngraph::op::AutoBroadcastType::NUMPY);
            return Ptr<BackendNode>(new InfEngineNgraphNode(deconv_bias));
        }


        return Ptr<BackendNode>(new InfEngineNgraphNode(deconv));
    }
#endif  // HAVE_DNN_NGRAPH

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == outputs.size());

        float flops = 0;
        int outChannels = blobs[0].size[0];
        size_t karea = std::accumulate(kernel_size.begin(), kernel_size.end(),
                                       1, std::multiplies<size_t>());

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += CV_BIG_INT(2)*outChannels*karea*total(inputs[i]);
        }

        return flops;
    }
};

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(const LayerParams &params)
{
    Ptr<ConvolutionLayerImpl> l(new ConvolutionLayerImpl(params));
    return l;
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(const LayerParams &params)
{
    return Ptr<BaseConvolutionLayer>(new DeConvolutionLayerImpl(params));
}

}
}
