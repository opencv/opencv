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
#endif  // HAVE_INF_ENGINE
        {
            return backendId == DNN_BACKEND_CUDA ||
            (kernel_size.size() == 2 && backendId == DNN_BACKEND_OPENCV) ||
            (kernel_size.size() == 2 && backendId == DNN_BACKEND_CANN);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        //CV_Assert(!hasBias() || blobs[1].total() == (size_t)numOutput);
        CV_Assert(inputs.size() != 0);

        int outCn = numOutput;
        if (outCn < 0) {
            CV_Assert(inputs.size() > 1 || !blobs.empty());
            MatShape weightShape = blobs.empty() ? inputs[1] : blobs[0].shape();
            outCn = weightShape[1];
        }
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

        outputs.resize(1, MatShape(outShape));

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

        CV_Assert(inputs.size() > 1 || !blobs.empty());

        MatShape weightShape = blobs.empty() ? inputs[1].shape() : blobs[0].shape();
        numOutput = weightShape[1];

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
            bool constBias = blobs.size() >= 2;
            biasesMat = constBias ? blobs[1].reshape(1, numOutput)
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
            useRVV = checkHardwareSupport(CPU_RVV);
            useLASX = checkHardwareSupport(CPU_LASX);
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
        #if CV_TRY_RVV && CV_RVV
            if( useRVV ) {
                opt_RVV::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            }
            else
        #endif
        #if CV_TRY_LASX
            if( useLASX )
                opt_LASX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
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
        bool useRVV;
        bool useLASX;
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

        if (inputs_.depth() == CV_16F)
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

        if (inputs_arr.depth() == CV_16F)
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

        if( weightsMat.empty() ) {
            Mat inpWeights = !blobs.empty() ? blobs[0] : inputs[1];
            transpose(inpWeights.reshape(1, inpCn), weightsMat);
        }

        if (biasesMat.empty()) {
            Mat inpBias = blobs.size() >= 2 ? blobs[1] : inputs.size() >= 3 ? inputs[2] : Mat();
            biasesMat = !inpBias.empty() ? inpBias.reshape(1, outCn) : Mat::zeros(outCn, 1, CV_32F);
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
        CV_Assert(!blobs.empty());
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
        auto y = outputs[0].dynamicCast<CannBackendWrapper>();
        const auto shape_x = x->host->size; // [N, C, H, W]
        const auto shape_y = y->host->size; // [N, C, H, W]
        const int filter_out_channel = blobs[0].size[0];
        const int groups = shape_x[1] / filter_out_channel;

        // create operator
        auto op = std::make_shared<ge::op::Conv2DTransposeD>(name);

        // set attributes
        op->set_attr_input_size(
            ge::Operator::OpListInt({(int64_t)shape_y[0],
                                     (int64_t)shape_y[1],
                                     (int64_t)shape_y[2],
                                     (int64_t)shape_y[3],})
        );
        op->set_attr_strides(
            ge::Operator::OpListInt({1, 1, (int64_t)strides[0], (int64_t)strides[1]})
        );
        op->set_attr_pads(ge::Operator::OpListInt(
            {(int64_t)pads_begin[1], (int64_t)pads_end[1], (int64_t)pads_begin[0], (int64_t)pads_end[0]}
        ));
        op->set_attr_dilations(ge::Operator::OpListInt(
            {1, 1, (int64_t)dilations[0], (int64_t)dilations[1]}
        ));
        op->set_attr_groups(groups);
        op->set_attr_data_format("NCHW");
        op->set_attr_output_padding(
            ge::Operator::OpListInt({0, 0, (int64_t)adjust_pads[0], (int64_t)adjust_pads[1]}) // adjust_pads: [height, width]
        );

        // set inputs
        // set inputs : x
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto desc_x = x->getTensorDesc();
        op->update_input_desc_x(*desc_x);
        // set inputs : weight
        const Mat& mat_w = blobs[0];
        auto op_const_w = std::make_shared<CannConstOp>(mat_w.data, mat_w.type(), shape(mat_w), cv::format("%s_w", name.c_str()));
        op->set_input_filter(*(op_const_w->getOp()));
        op->update_input_desc_filter(*(op_const_w->getTensorDesc()));
        // set inputs : bias
        if (has_bias)
        {
            int out_channel = blobs[0].size[0];
            const Mat& mat_b = blobs[1];

            std::vector<int> shape_b{out_channel};
            auto op_const_b = std::make_shared<CannConstOp>(mat_b.data, mat_b.type(), shape_b, cv::format("%s_b", name.c_str()));
            op->set_input_bias(*(op_const_b->getOp()));
            op->update_input_desc_bias(*(op_const_b->getTensorDesc()));
        }

        // set outputs
        auto desc_output = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*desc_output);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
       CV_Assert(!blobs.empty());
       const int outGroupCn = blobs[0].size[1];
       const int group = numOutput / outGroupCn;
       CV_Assert(group == 1);

       auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
       std::vector<size_t> kernel_shape = getShape<size_t>(blobs[0]);
       auto ieWeights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, kernel_shape, blobs[0].data);

        if (fusedWeights)
        {
            Mat newWeights;
            transpose(weightsMat, newWeights);
            ieWeights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, kernel_shape, newWeights.data);
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
        ov::op::PadType pad_type = padMode == "VALID" ? ov::op::PadType::VALID : ov::op::PadType::EXPLICIT;

        auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
                          ieInpNode,
                          ieWeights,
                          ov::Strides(strides),
                          ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                          ov::CoordinateDiff(std::vector<std::ptrdiff_t>(paddings_end.begin(), paddings_end.end())),
                          ov::Strides(dilations),
                          pad_type,
                          ov::CoordinateDiff(std::vector<std::ptrdiff_t>(adjust_pads.begin(), adjust_pads.end())));

        if (hasBias() || fusedBias)
        {
            std::vector<size_t> shape(deconv->get_shape().size(), 1);
            shape[1] = numOutput;
            auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape(shape), blobs[1].data);
            auto deconv_bias = std::make_shared<ov::op::v1::Add>(deconv, bias, ov::op::AutoBroadcastType::NUMPY);
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
