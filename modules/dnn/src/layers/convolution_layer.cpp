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
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "../op_vkcom.hpp"
#include "../op_webnn.hpp"
#include "../op_cann.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <iostream>
#include <numeric>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
using namespace cv::dnn::ocl4dnn;
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/convolution.hpp"
#include "../cuda4dnn/primitives/transpose_convolution.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

#include "cpu_kernels/convolution.hpp"

namespace cv
{
namespace dnn
{

class BaseConvolutionLayerImpl : public ConvolutionLayer
{
public:
    bool fusedWeights, fusedBias;
    std::vector<double> weightsMultipliers;
    int groups;
    BaseConvolutionLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        getConvolutionKernelParams(params, kernel_size, pads_begin, pads_end, strides, dilations,
                                   padMode, adjust_pads, useWinograd);

        numOutput = -1;
        groups = params.get<int>("group", 1);

        if (kernel_size.size() == 2) {
            kernel = Size(kernel_size[1], kernel_size[0]);
            stride = Size(strides[1], strides[0]);
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
        MatShape weightShape = blobs.empty() ? inputs[1].shape() : blobs[0].shape();
        numOutput = weightShape[0];

        CV_Assert(inputs[0].dims == outputs[0].dims);
        if (weightShape.dims == 3)
        {
            kernel_size.resize(1, kernel_size[0]);
            strides.resize(1, strides[0]);
            dilations.resize(1, dilations[0]);
            pads_begin.resize(1, pads_begin[0]);
            pads_end.resize(1, pads_end[0]);
        }
        CV_Assert(weightShape.dims == kernel_size.size() + 2);
        for (int i = 0; i < kernel_size.size(); i++) {
            CV_Assert(weightShape[i + 2] == kernel_size[i]);
        }

        const Mat &input = inputs[0];
        CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || input.dims == 4 || input.dims == 5) && (input.type() == CV_32F || input.type() == CV_16F));
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
        if (fusedAdd)   // If the Conv layer has fused Add layer, it cannot fuse other layers.
            return false;

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
};


//TODO: simultaneously convolution and bias addition for cache optimization
class ConvolutionLayerImpl CV_FINAL : public BaseConvolutionLayerImpl
{
public:
    enum { VEC_ALIGN = 8, DFT_TYPE = CV_32F };
    Mat weightsMat;  // Used to store weight params. It will be used for layer fusion and memory alignment.
    std::vector<float> biasvec;
    std::vector<float> reluslope;
    Ptr<ActivationLayer> activ;

    Ptr<FastConv> fastConvImpl;

#ifdef HAVE_OPENCL
    Ptr<OCL4DNNConvSpatial<float> > convolutionOp;
    std::vector<UMat> umat_blobs;
    bool newActiv;
    ocl4dnnFusedActiv_t activType;
    float power;
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
    }

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
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            bool isArmTarget = preferableTarget == DNN_TARGET_CPU && isArmComputePlugin();
            if (isArmTarget && blobs.empty())
                return false;
            if (ksize == 1)
                return isArmTarget;
            if (ksize == 3)
                return preferableTarget != DNN_TARGET_MYRIAD && !isArmTarget;
            bool isMyriad = preferableTarget == DNN_TARGET_MYRIAD || preferableTarget == DNN_TARGET_HDDL;
            if (!isMyriad && blobs.empty())
                return false;
            return (!isMyriad || dilation.width == dilation.height);
        }
#endif
        if (backendId == DNN_BACKEND_OPENCV)
            return ksize >= 1 && ksize <= 3;
#ifdef HAVE_VULKAN
        if (backendId == DNN_BACKEND_VKCOM)
            return ksize == 2;
#endif
#ifdef HAVE_WEBNN
        if (backendId == DNN_BACKEND_WEBNN)
        {
            if (ksize != 2)
            {
                CV_LOG_WARNING(NULL, "WebNN only supports Conv2d.");
                return false;
            }
            return true;
        }
#endif
#ifdef HAVE_CANN
        if (backendId == DNN_BACKEND_CANN)
        {
            if (ksize != 2)
            {
                CV_LOG_WARNING(NULL, "CANN supports Conv2D for now");
                return false;
            }
            return true;
        }
#endif // HAVE_CANN
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

        CV_Assert(!inputs.empty());
        CV_Assert(inputs[0].size() > 2);
        std::vector<int> inpShape(inputs[0].begin() + 2, inputs[0].end());

        int outCn = weightShape[0];
        std::vector<int> outShape;
        outShape.push_back(inputs[0][0]);
        outShape.push_back(outCn);

        int inpCn = inputs[0][1];
        if (padMode.empty())
        {
            for (int i = 0; i < inpShape.size(); i++)
                outShape.push_back((inpShape[i] + pads_begin[i] + pads_end[i] -
                                    dilations[i] * (kernel_size[i] - 1) - 1) / strides[i] + 1);
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

        outputs.resize(1, MatShape(outShape));

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
            if ((wm.step1() % VEC_ALIGN != 0) ||
                !isAligned<VEC_ALIGN * sizeof(float)>(wm.data)
            )
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
        fusedActivation = !activ.empty();
        return fusedActivation;
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        if (fusedAdd)   // If the Conv layer has fused Add layer, it cannot fuse other layers.
            return false;

#ifdef HAVE_CUDA
        if(IS_DNN_CUDA_TARGET(preferableTarget))
        {
            Ptr<EltwiseLayer> eltwise = top.dynamicCast<EltwiseLayer>();
            Ptr<NaryEltwiseLayer> naryEltwise = top.dynamicCast<NaryEltwiseLayer>();
            if (!eltwise.empty() || !naryEltwise.empty())
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

    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs, std::vector<Ptr<BackendWrapper> > &outputs) CV_OVERRIDE
    {
#ifdef HAVE_VULKAN
        int activationType = transFusedActivType(activ);

        CV_Assert(inputs.size() == 1 && outputs.size() == 1);
        Ptr<VkComBackendWrapper> inputWrap = inputs[0].dynamicCast<VkComBackendWrapper>();
        Ptr<VkComBackendWrapper> outputWrap = outputs[0].dynamicCast<VkComBackendWrapper>();
        CV_Assert(inputWrap && outputWrap);

        MatShape inpShape = shape(*inputWrap->getMat());
        MatShape outShape = shape(*outputWrap->getMat());

        CV_Assert(inpShape.size() == 4 && inpShape.size() == outShape.size());

        if (activationType == -1)
        {
            CV_LOG_WARNING(NULL, "Unsupported fused Active type in Conv layer!!!");
            return Ptr<BackendNode>();
        }

        const int inpGroupCn = blobs[0].size[1];
        int ngroups = inpShape[1] / inpGroupCn;
        CV_Assert(outShape[1] % ngroups == 0);
        if (ngroups != 1)
            return Ptr<BackendNode>();

        Mat weightVK;
        if (fusedWeights)
        {
            weightsMat.copyTo(weightVK); // to handle the case of isContinuous() == false
            weightVK = weightVK.reshape(1, blobs[0].dims, blobs[0].size);
        }
        else
            weightVK = blobs[0];

        CV_Assert(weightVK.isContinuous());
        CV_Assert(pads_begin.size() == 2);
        CV_Assert(fusedAdd == false && "Vulkan Backend can not support the Conv_Add optimization.");
        Ptr<vkcom::OpBase> op(new vkcom::OpConv(weightVK, biasvec, activationType, ngroups, outShape[1], inpShape[1],
                                                            kernel.height, kernel.width, stride.height, stride.width,
                                                            dilation.height, dilation.width, pads_begin[1], pads_begin[0]));

        return Ptr<BackendNode>(new VkComBackendNode(inputs, op, outputs));
#endif  // HAVE_VULKAN
        return Ptr<BackendNode>();
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        CV_Assert(inputs.size() == 1);
        CV_Assert(nodes.size() == 1);

        bool has_bias = hasBias() || fusedBias;

        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        const auto shape_x = x->host->size; // [b, c, h, w]
        const int filter_out_channel = blobs[0].size[1];
        const int groups = shape_x[1] / filter_out_channel;

        // create operator
        auto op = std::make_shared<ge::op::Conv2D>(name);

        // set attributes
        op->set_attr_strides(ge::Operator::OpListInt(
            {1, 1, (int64_t)strides[0], (int64_t)strides[1]}
        ));
        // recalculate pads in case of "SAME" padMode with odd pads
        // since in 'getConvPoolPaddings' pads are divided equally
        // leading to the loss of one pad
        if (padMode == "SAME")
        {
            for (int i = 0; i < pads_begin.size(); i++) {
                if (strides[i] <= kernel_size[i])
                {
                    int pads_at_i = kernel_size[i] - 1 - (shape_x[i+2] - 1 + strides[i]) % strides[i];
                    pads_begin[i] = pads_at_i / 2;
                    // if odd, add extra padding to the end for SAME_UPPER
                    // or to the beginning for SAME_LOWER. Since here we cannot
                    // identity SAME_UPPER and SAME_LOWER, extra padding is always
                    // added to the end.
                    pads_end[i] = pads_at_i - pads_begin[i];
                }
            }
        }
        op->set_attr_pads(ge::Operator::OpListInt(
            {(int64_t)pads_begin[1], (int64_t)pads_end[1], (int64_t)pads_begin[0], (int64_t)pads_end[0]}
        ));
        op->set_attr_dilations(ge::Operator::OpListInt(
            {1, 1, (int64_t)dilations[0], (int64_t)dilations[1]}
        ));
        op->set_attr_groups(groups);
        op->set_attr_data_format("NCHW");

        // set inputs
        // set inputs : x
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);
        // set inputs : weight
        const Mat& w_mat = blobs[0];
        auto op_const_weight = std::make_shared<CannConstOp>(w_mat.data, w_mat.type(), shape(w_mat), cv::format("%s_w", name.c_str()));
        op->set_input_filter(*(op_const_weight->getOp()));
        op->update_input_desc_filter(*(op_const_weight->getTensorDesc()));
        // set inputs : bias
        if (has_bias)
        {
            int out_channel = blobs[0].size[0];
            Mat b_mat({out_channel}, CV_32F, &biasvec[0]);

            std::vector<int> bias_shape{out_channel};
            auto op_const_bias = std::make_shared<CannConstOp>(b_mat.data, b_mat.type(), bias_shape, cv::format("%s_b", name.c_str()));
            op->set_input_bias(*(op_const_bias->getOp()));
            op->update_input_desc_bias(*(op_const_bias->getTensorDesc()));
        }

        // set outputs
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        CV_Assert_N(inputs.size() >= 1, nodes.size() >= 1);
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::vector<size_t> dims = ieInpNode.get_shape();
        CV_Check(dims.size(), dims.size() >= 3 && dims.size() <= 5, "");
        ov::Output<ov::Node> ieWeights;
        if (nodes.size() > 1)
            ieWeights = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
        const int inpCn = dims[1];
        const int inpGroupCn = nodes.size() > 1 ? ieWeights.get_shape()[1] : blobs[0].size[1];
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
            ieWeights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, kernel_shape, blobs[0].data);
            if (fusedWeights)
            {
                if (weightsMat.isContinuous())
                {
                    ieWeights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, kernel_shape, weightsMat.data);
                }
                else
                {
                    Mat newWeights;
                    Mat cvWeights = weightsMat.colRange(0, blobs[0].total() / numOutput);
                    cvWeights.copyTo(newWeights);
                    ieWeights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, kernel_shape, newWeights.data);
                }
            }
        }
        else
        {
            auto shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                             ov::Shape{kernel_shape.size()}, std::vector<int64_t>(kernel_shape.begin(), kernel_shape.end()));
            ieWeights  = std::make_shared<ov::op::v1::Reshape>(ieWeights, shape, true);
        }

        ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
        if (!padMode.empty())
            pad_type = padMode == "VALID" ? ov::op::PadType::VALID : ov::op::PadType::SAME_UPPER;

        std::shared_ptr<ov::Node> conv_node;
        if (group != 1) {
            conv_node = std::make_shared<ov::op::v1::GroupConvolution>(
                                ieInpNode, ieWeights,
                                ov::Strides(strides),
                                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_end.begin(),   pads_end.end())),
                                ov::Strides(dilations),
                                pad_type);
        } else {
            conv_node = std::make_shared<ov::op::v1::Convolution>(
                                ieInpNode, ieWeights,
                                ov::Strides(strides),
                                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_end.begin(), pads_end.end())),
                                ov::Strides(dilations),
                                pad_type);
        }

        if (hasBias() || fusedBias || nodes.size() == 3)
        {
            std::vector<size_t> shape(conv_node->get_shape().size(), 1);
            shape[1] = conv_node->get_shape()[1];
            std::shared_ptr<ov::Node> bias;
            if (nodes.size() == 3)
            {
                auto bias_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                    ov::Shape{shape.size()}, std::vector<int64_t>(shape.begin(), shape.end()));
                bias = std::make_shared<ov::op::v1::Reshape>(nodes[2].dynamicCast<InfEngineNgraphNode>()->node, bias_shape, true);
            }
            else
            {
                bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape(shape), biasvec.data());
            }
            auto conv_bias = std::make_shared<ov::op::v1::Add>(conv_node, bias, ov::op::AutoBroadcastType::NUMPY);
            return Ptr<BackendNode>(new InfEngineNgraphNode(conv_bias));
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(conv_node));
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        CV_Assert_N(inputs.size() >= 1, nodes.size() >= 1);
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        ml::Operand webnnWeights = nodes.size() > 1 ? nodes[1].dynamicCast<WebnnBackendNode>()->operand : nullptr;
        if (nodes.size() > 1)
            CV_Assert(webnnWeights);
        const int inpCn = weightsMat.total()/(kernel_size[0]*kernel_size[1]*numOutput);
        const int group = groups;
        const int inpGroupCn = inpCn / group;
        std::vector<int32_t> kernel_shape;
        if (group != 1)
        {
            kernel_shape.push_back(group);
        }
        kernel_shape.push_back(numOutput / group);
        kernel_shape.push_back(inpGroupCn);
        std::copy(kernel_size.begin(), kernel_size.end(), back_inserter(kernel_shape));

        if (nodes.size() == 1)
        {
            webnnWeights = webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(blobs[0]), blobs[0].data, blobs[0].total()*blobs[0].elemSize(), ml::OperandType::Float32);
            if (fusedWeights)
            {
                if (weightsMat.isContinuous())
                {
                    webnnWeights = webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(weightsMat), weightsMat.data, weightsMat.total()*weightsMat.elemSize(), ml::OperandType::Float32);
                }
                else
                {
                    Mat newWeights;
                    Mat cvWeights = weightsMat.colRange(0, blobs[0].total() / numOutput);
                    cvWeights.copyTo(newWeights);
                    webnnWeights = webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(newWeights), newWeights.data, newWeights.total()*newWeights.elemSize(), ml::OperandType::Float32);
                }
            }
        }
        else
        {
            webnnWeights  = webnnGraphBuilder.Reshape(webnnWeights, kernel_shape.data(), kernel_shape.size());
        }

        ml::AutoPad pad_type = ml::AutoPad::Explicit;
        if (!padMode.empty())
            pad_type = padMode == "VALID" ? ml::AutoPad::Explicit : ml::AutoPad::SameUpper;

        ml::Conv2dOptions options = {};
        options.groups = group;
        options.autoPad = pad_type;
        std::vector<int32_t> Strides(strides.begin(), strides.end());
        if (!Strides.empty())
        {
            options.stridesCount = Strides.size();
            options.strides = Strides.data();
        }
        std::vector<int32_t> Padding;
        if (padMode.empty())
        {
            Padding = {static_cast<int32_t>(pads_begin[0]),
                       static_cast<int32_t>(pads_end[0]),
                       static_cast<int32_t>(pads_begin[1]),
                       static_cast<int32_t>(pads_end[1])};
        }
        else if (padMode == "VALID")
        {
            Padding = {0, 0, 0, 0};
        }
        if (!Padding.empty())
        {
            options.paddingCount = Padding.size();
            options.padding = Padding.data();
        }
        std::vector<int32_t> Dilations(dilations.begin(), dilations.end());
        if (!Dilations.empty())
        {
            options.dilationsCount = Dilations.size();
            options.dilations = Dilations.data();
        }
        ml::Operand operand = webnnGraphBuilder.Conv2d(webnnInpOperand, webnnWeights, &options);

        // ml::Operand result = operand;
        if (hasBias() || fusedBias || nodes.size() == 3)
        {
            ml::Operand webnnBias = nullptr;
            if (nodes.size() == 3)
            {
                std::vector<int32_t> bias_shape = {1, numOutput, 1, 1};
                webnnBias = webnnGraphBuilder.Reshape(nodes[2].dynamicCast<WebnnBackendNode>()->operand, bias_shape.data(), bias_shape.size());
            }
            else
            {
                webnnBias = webnn::BuildConstant(webnnGraphBuilder, {1, numOutput, 1, 1}, biasvec.data(), (numOutput) * sizeof(float), ml::OperandType::Float32);
            }
            operand = webnnGraphBuilder.Add(operand, webnnBias);
        }
        return Ptr<BackendNode>(new WebnnBackendNode(operand));
    }
#endif // HAVE_WEBNN

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

        bool use_half = (inps.depth() == CV_16F);
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
                CV_Assert(!use_half);  // TODO: not implemented
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
                    blobs[i].convertTo(umat_blobs[i], CV_16F);
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
            // pads_begin: 0 - pad_top, 1 - pad_left
            // pads_end: 0 - pad_bottom, 1 - pad_right
            std::vector<int> pads = {int(pads_begin[0]), int(pads_end[0]), int(pads_begin[1]), int(pads_end[1])};
            config.pads = pads;
            config.stride = stride;
            config.dilation = dilation;
            if (inputs[0].dims != 4 && inputs[0].dims != (blobs.empty() ? umat_blobs[0].dims : blobs[0].dims))
            {
                static bool bypassCheck = utils::getConfigurationParameterBool("OPENCV_OCL4DNN_CONVOLUTION_IGNORE_INPUT_DIMS_4_CHECK", false);
                if (!bypassCheck)
                {
                    CV_LOG_ERROR(NULL, "DNN/OpenCL: Unsupported configuration: inputs[0].dims=" << inputs[0].dims << "  umat_blobs[0].dims=" << umat_blobs[0].dims
                        << ". Consider reporting complete reproducer to https://github.com/opencv/opencv/issues/20833."
                        << " You can skip this check temporary through OPENCV_OCL4DNN_CONVOLUTION_IGNORE_INPUT_DIMS_4_CHECK=1"
                    );
                    return false;
                }
            }
            config.group = inputs[0].size[1] / (blobs.empty() ? umat_blobs[0].size[1] : blobs[0].size[1]);
            if (config.group < 1)  // config.group == 0 causes div by zero in ocl4dnn code
            {
                CV_LOG_WARNING(NULL, "DNN/OpenCL: Unsupported config.group=" << config.group
                    << ". Consider reporting complete reproducer to https://github.com/opencv/opencv/issues/20833"
                );
                return false;
            }
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
                weightsMat.convertTo(umat_blobs[0], CV_16F);
            else
                weightsMat.copyTo(umat_blobs[0]);
            fusedWeights = false;
        }
        if (fusedBias)
        {
            if ( umat_blobs.size() < 2 )
                umat_blobs.resize(2);
            if (use_half)
                Mat(biasvec, true).convertTo(umat_blobs[1], CV_16F);
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

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        int outCn = blobs.empty() ? inputs[1].size[0] : blobs[0].size[0];
        // Need to align non-const blobs
        bool variableWeight = false;
        if (blobs.empty())
        {
            variableWeight = true;
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

        {
            int nstripes = std::max(getNumThreads(), 1);
            int conv_dim = CONV_2D;
            if (inputs[0].dims == 3)
                conv_dim = CONV_1D;
            if (inputs[0].dims == 5)
                conv_dim = CONV_3D;

            // Initialization of FastCovn2d, pack weight.
            if (!fastConvImpl || variableWeight)
            {
                int K = outputs[0].size[1];
                int C = inputs[0].size[1];

                // Winograd only works when input h and w >= 12.
                bool canUseWinograd = useWinograd && conv_dim == CONV_2D && inputs[0].size[2] >= 12 && inputs[0].size[3] >= 12;

                CV_Assert(outputs[0].size[1] % ngroups == 0);
                fastConvImpl = initFastConv(weightsMat, &biasvec[0], ngroups, K, C, kernel_size, strides,
                                            dilations, pads_begin, pads_end, conv_dim,
                                            preferableTarget == DNN_TARGET_CPU_FP16, canUseWinograd);
                // This is legal to release weightsMat here as this is not used anymore for
                // OpenCV inference. If network needs to be reinitialized (new shape, new backend)
                // a new version of weightsMat is created at .finalize() from original weights
                weightsMat.release();
            }

            runFastConv(inputs[0], outputs[0], fastConvImpl, nstripes, activ, reluslope, fusedAdd);
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

        // TODO: extract bias from inputs and pass it
        CV_Assert(inputs.size() == 1 || inputs.size() == 2);
        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();

        CV_Assert(outputs.size() == 1);
        auto output_wrapper = outputs[0].dynamicCast<CUDABackendWrapper>();
        auto output_shape = output_wrapper->getShape();

        CV_Assert(!blobs.empty());
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

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(const LayerParams &params)
{
    Ptr<ConvolutionLayerImpl> l(new ConvolutionLayerImpl(params));
    return l;
}

}
}
