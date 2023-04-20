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
#include "opencv2/core/hal/intrin.hpp"
#include "../op_cuda.hpp"
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include "../op_webnn.hpp"
#include "../op_cann.hpp"

#ifdef HAVE_DNN_NGRAPH
#include "../ie_ngraph.hpp"
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
#include <ngraph/op/roi_pooling.hpp>
#include <ngraph/op/psroi_pooling.hpp>
#else
#include <ngraph/op/experimental/layers/roi_pooling.hpp>
#include <ngraph/op/experimental/layers/psroi_pooling.hpp>
#endif
#endif

#include "../op_vkcom.hpp"

#include <float.h>
#include <algorithm>
#include <numeric>
using std::max;
using std::min;

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
using namespace cv::dnn::ocl4dnn;
#endif

#ifdef HAVE_HALIDE
#if 0  // size_t is not well supported in Halide operations
typedef size_t HALIDE_DIFF_T;
#else
typedef int HALIDE_DIFF_T;
#endif
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/pooling.hpp"
#include "../cuda4dnn/primitives/roi_pooling.hpp"
#include "../cuda4dnn/primitives/max_unpooling.hpp"
using namespace cv::dnn::cuda4dnn;
#endif
#include <opencv2/core/utils/logger.hpp>


namespace cv
{
namespace dnn
{
static inline int roundRoiSize(float v)
{
    return (int)(v + (v >= 0.f ? 0.5f : -0.5f));
}

class PoolingLayerImpl CV_FINAL : public PoolingLayer
{
public:
    PoolingLayerImpl(const LayerParams& params)
    {
        computeMaxIdx = true;
        globalPooling = false;
        isGlobalPooling = std::vector<bool>(3, false);

        hasDynamicShapes = params.get<bool>("has_dynamic_shapes", false);
        shapesInitialized = !hasDynamicShapes;

        if (params.has("pool") || params.has("kernel_size") ||
            params.has("kernel_w") || params.has("kernel_h"))
        {
            String pool = toLowerCase(params.get<String>("pool", "max"));
            if (pool == "max")
                type = MAX;
            else if (pool == "ave")
                type = AVE;
            else if (pool == "stochastic")
                type = STOCHASTIC;
            else if (pool == "sum")
                type = SUM;
            else
                CV_Error(Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");

            getPoolingKernelParams(params, kernel_size, isGlobalPooling, pads_begin, pads_end, strides, padMode);
            globalPooling = isGlobalPooling[0] || isGlobalPooling[1] || isGlobalPooling[2];
        }
        else if (params.has("pooled_w") || params.has("pooled_h"))
        {
            type = ROI;
            pooledSize.width = params.get<uint32_t>("pooled_w", 1);
            pooledSize.height = params.get<uint32_t>("pooled_h", 1);
        }
        else if (params.has("output_dim") && params.has("group_size"))
        {
            type = PSROI;
            pooledSize.width = params.get<int>("group_size");
            pooledSize.height = pooledSize.width;
            psRoiOutChannels = params.get<int>("output_dim");
        }
        else
            CV_Error(Error::StsBadArg, "Cannot determine pooling type");
        setParamsFrom(params);
        ceilMode = params.get<bool>("ceil_mode", true);
        spatialScale = params.get<float>("spatial_scale", 1);
        avePoolPaddedArea = params.get<bool>("ave_pool_padded_area", true);
    }

#ifdef HAVE_OPENCL
    Ptr<OCL4DNNPool<float> > poolOp;
#endif

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(!inputs.empty());

        std::vector<int> inp;
        std::vector<int> out;
        for (int i = 2; i < inputs[0].dims; i++) {
            inp.push_back(inputs[0].size[i]);
            out.push_back(outputs[0].size[i]);
        }
        if (globalPooling) {
            std::vector<size_t> finalKernel;
            for (int i = 0; i < inp.size(); i++) {
                int idx = isGlobalPooling.size() - inp.size() + i;
                finalKernel.push_back(isGlobalPooling[idx] ? inp[i] : kernel_size[idx]);
             }
             kernel_size = finalKernel;
         }

        getConvPoolPaddings(inp, kernel_size, strides, padMode, pads_begin, pads_end);

        if (inputs[0].dims == 3)
        {
            // Pool1D
            kernel_size.assign(1, kernel_size[0]);
            strides.assign(1, strides[0]);
            pads_begin.assign(1, pads_begin[0]);
            pads_end.assign(1, pads_end[0]);
        }

#ifdef HAVE_OPENCL
        poolOp.release();
#endif
        computeMaxIdx = type == MAX && outputs.size() == 2;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_CUDA)
        {
            return type == MAX || type == AVE || type == ROI;
        }
#ifdef HAVE_CANN
        if (backendId == DNN_BACKEND_CANN)
        {
            return type == MAX || type == AVE;
        }
#endif
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            return !computeMaxIdx && type != STOCHASTIC && kernel_size.size() > 1 && (kernel_size.size() != 3 || !isArmComputePlugin());
        }
#endif
        if (backendId == DNN_BACKEND_OPENCV)
        {
            if (kernel_size.size() == 3)
                return preferableTarget == DNN_TARGET_CPU;
            if (kernel_size.size() <= 2)
                return true;
            else
                return false;
        }
        else if (backendId == DNN_BACKEND_HALIDE)
        {
            if (kernel_size.empty() || kernel_size.size() == 2)
                return haveHalide() &&
                       (type == MAX || (type == AVE && !pads_begin[0] && !pads_begin[1] && !pads_end[0] && !pads_end[1]));
        }
        else if (backendId == DNN_BACKEND_VKCOM)
        {
            if (kernel_size.empty() || kernel_size.size() == 2)
                return haveVulkan() &&
                           (type == MAX || type == AVE);
            return false;
        }
        else if (backendId == DNN_BACKEND_WEBNN)
        {
            if (kernel_size.empty() || kernel_size.size() == 2)
            {
                if (!haveWebnn())
                {
                    return false;
                }
                else
                {
                    if (!ceilMode)
                    {
                        CV_LOG_WARNING(NULL, "ceilMode is not supported by WebNN backend.");
                        return false;
                    }
                    if (computeMaxIdx)
                    {
                        CV_LOG_WARNING(NULL, "Mask is not supported by WebNN backend.");
                        return false;
                    }
                    if (type != MAX && type != AVE)
                    {
                        if (type == STOCHASTIC)
                        {
                            CV_LOG_WARNING(NULL, "Stochastic Pooling is not supported by WebNN backend.");
                        }
                        if (type == SUM)
                        {
                            CV_LOG_WARNING(NULL, "Sum Pooling is not supported by WebNN backend.");
                        }
                        if (type == ROI)
                        {
                            CV_LOG_WARNING(NULL, "ROI Pooling is not supported by WebNN backend.");
                        }
                        if (type == PSROI)
                        {
                            CV_LOG_WARNING(NULL, "Position-sensitive ROI Pooling is not supported by WebNN backend.");
                        }
                        CV_LOG_WARNING(NULL, "WebNN backend only supports MaxPooling and AveragePooling currently.");
                        return false;
                    }
                }
                return true;
            }
        }
        else if (backendId == DNN_BACKEND_TIMVX)
        {
#ifdef HAVE_TIMVX
            if (kernel_size.size() == 3)
            {
                // fallback to CPU implementation.
                preferableTarget = DNN_TARGET_CPU;
            }
#endif
            return false;
        }
        return false;
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, InputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inps.depth() == CV_16S);
        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        if (poolOp.empty())
        {
            OCL4DNNPoolConfig config;

            config.in_shape = shape(inputs[0]);
            config.out_shape = shape(outputs[0]);
            if (inputs[0].dims == 3)
            {
                //Pool1D
                config.kernel = Size(kernel_size[0], 1);
                config.stride = Size(strides[0], 1);
                config.pad_l = pads_begin[0];
                config.pad_t = 0;
                config.pad_r = pads_end[0];
                config.pad_b = 0;
            }
            else
            {
                config.kernel = Size(kernel_size[1], kernel_size[0]);
                config.stride = Size(strides[1], strides[0]);
                config.pad_l = pads_begin[1];
                config.pad_t = pads_begin[0];
                config.pad_r = pads_end[1];
                config.pad_b = pads_end[0];
            }
            config.channels = inputs[0].size[1];
            config.pool_method = type == MAX ? LIBDNN_POOLING_METHOD_MAX :
                                (type == AVE ? LIBDNN_POOLING_METHOD_AVE :
                                               LIBDNN_POOLING_METHOD_STO);
            config.avePoolPaddedArea = avePoolPaddedArea;
            config.computeMaxIdx = computeMaxIdx;
            config.use_half = use_half;
            poolOp = Ptr<OCL4DNNPool<float> >(new OCL4DNNPool<float>(config));
        }

        CV_Assert_N(inputs.size() == 1, !outputs.empty(), !computeMaxIdx || outputs.size() == 2);
        UMat& inpMat = inputs[0];
        UMat& outMat = outputs[0];
        UMat maskMat = computeMaxIdx ? outputs[1] : UMat();

        CV_Assert(inpMat.offset == 0 && outMat.offset == 0);

        return poolOp->Forward(inpMat, outMat, maskMat);
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (type == MAX || type == AVE || type == STOCHASTIC)
        {
            CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                       forward_ocl(inputs_arr, outputs_arr, internals_arr))
        }
        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        switch (type)
        {
            case MAX:
            {
                CV_Assert_N(inputs.size() == 1, !computeMaxIdx || outputs.size() == 2);
                Mat mask = computeMaxIdx ? outputs[1] : Mat();
                maxPooling(inputs[0], outputs[0], mask);
                break;
            }
            case AVE: case SUM:
                CV_Assert_N(inputs.size() == 1, outputs.size() == 1);
                avePooling(inputs[0], outputs[0]);
                break;
            case ROI: case PSROI:
                CV_Assert_N(inputs.size() == 2, outputs.size() == 1);
                roiPooling(inputs[0], inputs[1], outputs[0]);
                break;
            default:
                CV_Error(Error::StsNotImplemented, "Not implemented");
                break;
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
        if (type == ROI)
            return make_cuda_node<cuda4dnn::ROIPoolingOp>(preferableTarget, std::move(context->stream), spatialScale);

        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();

        /* storing max indices is a special case and we deal with it separately */
        if (computeMaxIdx) {
            CV_Assert(type == MAX);

            cuda4dnn::MaxPoolingConfiguration config;
            config.window_size.assign(std::begin(kernel_size), std::end(kernel_size));
            config.strides.assign(std::begin(strides), std::end(strides));

            if (padMode.empty())
            {
                config.padMode = MaxPoolingConfiguration::PaddingMode::MANUAL;
                config.pads_begin.assign(std::begin(pads_begin), std::end(pads_begin));
            }
            else if (padMode == "VALID")
            {
                config.padMode = MaxPoolingConfiguration::PaddingMode::VALID;
            }
            else if (padMode == "SAME")
            {
                config.padMode = MaxPoolingConfiguration::PaddingMode::SAME;
            }
            else
            {
                CV_Error(Error::StsNotImplemented, padMode + " padding mode not supported by PoolingLayer");
            }

            config.input_shape.assign(std::begin(input_shape), std::end(input_shape));

            return make_cuda_node<cuda4dnn::MaxPoolingOp>(preferableTarget, std::move(context->stream), config);
        }

        if (input_shape.size() == 3)
        {
            // Pool1D
            // We add an extra dim for input tensor, because CuDNN support pooling only with 2 and 3 spatial dimensions
            input_shape.insert(std::end(input_shape) - 1, 1);

            // Do the similar thing for the other parameters
            pads_begin.insert(std::begin(pads_begin), 0);
            pads_end.insert(std::begin(pads_end), 0);
            strides.insert(std::begin(strides), 1);
            kernel_size.insert(std::begin(kernel_size), 1);
        }

        PoolingConfiguration config;
        if (type == MAX)
        {
            config.poolMode = PoolingConfiguration::PoolingMode::MAX;
        }
        else if (type == AVE && !avePoolPaddedArea)
        {
            config.poolMode = PoolingConfiguration::PoolingMode::AVERAGE_EXCLUDE_PADDING;
        }
        else if (type == AVE && avePoolPaddedArea)
        {
            config.poolMode = PoolingConfiguration::PoolingMode::AVERAGE_INCLUDE_PADDING;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "Unsupported pooling mode");
        }

        config.window_size.assign(std::begin(kernel_size), std::end(kernel_size));
        config.strides.assign(std::begin(strides), std::end(strides));

        if (padMode.empty())
        {
            config.padMode = PoolingConfiguration::PaddingMode::MANUAL;
            config.pads_begin.assign(std::begin(pads_begin), std::end(pads_begin));
            config.pads_end.assign(std::begin(pads_end), std::end(pads_end));
        }
        else if (padMode == "VALID")
        {
            config.padMode = PoolingConfiguration::PaddingMode::VALID;
        }
        else if (padMode == "SAME")
        {
            config.padMode = PoolingConfiguration::PaddingMode::SAME;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, padMode + " padding mode not supported by PoolingLayer");
        }

        if (ceilMode)
            config.roundMode = PoolingConfiguration::RoundingMode::CEIL;
        else
            config.roundMode = PoolingConfiguration::RoundingMode::FLOOR;

        config.input_shape.assign(std::begin(input_shape), std::end(input_shape));

        return make_cuda_node<cuda4dnn::PoolingOp>(preferableTarget, std::move(context->cudnn_handle), config);
    }
#endif


#ifdef HAVE_VULKAN
    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
        int padding_mode;
        vkcom::PoolType pool_type;
        int filter_size[2] = {static_cast<int>(kernel_size[0]), static_cast<int>(kernel_size[1])};
        int pad_size[2] = {static_cast<int>(pads_begin[0]), static_cast<int>(pads_begin[1])};
        int stride_size[2] = {static_cast<int>(strides[0]), static_cast<int>(strides[1])};
        pool_type = type == MAX ? vkcom::kPoolTypeMax:
                   (type == AVE ? vkcom::kPoolTypeAvg:
                            vkcom::kPoolTypeNum);

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

        std::shared_ptr<vkcom::OpBase> op(new vkcom::OpPool(filter_size, pad_size,
                                                            stride_size, padding_mode,
                                                            pool_type, avePoolPaddedArea));
        return Ptr<BackendNode>(new VkComBackendNode(inputs, op));
    }
#endif


    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
        if (type == MAX)
            return initMaxPoolingHalide(inputs);
        else if (type == AVE)
            return initAvePoolingHalide(inputs);
        else
            return Ptr<BackendNode>();
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto x_desc = x->getTensorDesc();
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        if (type == MAX)
        {
            auto op = std::make_shared<ge::op::MaxPoolV3>(name);

            // set attributes
            op->set_attr_ksize(ge::Operator::OpListInt(
                {1, 1, (int64_t)kernel_size[0], (int64_t)kernel_size[1]}
            ));
            op->set_attr_strides(ge::Operator::OpListInt(
                {1, 1, (int64_t)strides[0], (int64_t)strides[1]}
            ));
            std::string cann_pad_mode{"CALCULATED"};
            if (padMode == "SAME" || padMode == "VALID")
                cann_pad_mode = padMode;
            op->set_attr_padding_mode(cann_pad_mode.c_str());
            op->set_attr_pads(ge::Operator::OpListInt(
                {(int64_t)pads_begin[0], (int64_t)pads_end[0], (int64_t)pads_begin[1], (int64_t)pads_end[1]}
            ));
            op->set_attr_data_format("NCHW");
            op->set_attr_global_pooling(globalPooling);
            op->set_attr_ceil_mode(ceilMode);

            // set inputs
            op->set_input_x_by_name(*op_x, x->name.c_str());
            op->update_input_desc_x(*x_desc);
            // set outputs
            op->update_output_desc_y(*output_desc);

            return Ptr<BackendNode>(new CannBackendNode(op));
        }
        else if (type == AVE)
        {
            auto op = std::make_shared<ge::op::AvgPoolV2>(name);

            // set attributes
            op->set_attr_ksize(ge::Operator::OpListInt(
                {1, 1, (int64_t)kernel_size[0], (int64_t)kernel_size[1]}
            ));
            op->set_attr_strides(ge::Operator::OpListInt(
                {1, 1, (int64_t)strides[0], (int64_t)strides[1]}
            ));
            std::string cann_pad_mode{"CALCULATED"};
            if (padMode == "SAME" || padMode == "VALID")
                cann_pad_mode = padMode;
            op->set_attr_padding_mode(cann_pad_mode.c_str());
            op->set_attr_pads(ge::Operator::OpListInt(
                {(int64_t)pads_begin[0], (int64_t)pads_end[0], (int64_t)pads_begin[1], (int64_t)pads_end[1]}
            ));
            op->set_attr_global_pooling(globalPooling);
            op->set_attr_ceil_mode(ceilMode);
            auto cann_exclusive = !avePoolPaddedArea;
            op->set_attr_exclusive(cann_exclusive);

            // set inputs
            op->set_input_x_by_name(*op_x, x->name.c_str());
            op->update_input_desc_x(*x_desc);
            // set outputs
            op->update_output_desc_y(*output_desc);

            return Ptr<BackendNode>(new CannBackendNode(op));
        }
        else
            CV_Error(Error::StsNotImplemented, "Unsupported pooling type");
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert_N((inputs.size() == 1 && (type == MAX || type == AVE || type == SUM)) || inputs.size() == 2, nodes.size() == inputs.size());
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;

        ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
        if (!padMode.empty())
            pad_type = padMode == "VALID" ? ngraph::op::PadType::VALID : ngraph::op::PadType::SAME_UPPER;

        auto rounding_type = ceilMode ? ngraph::op::RoundingType::CEIL : ngraph::op::RoundingType::FLOOR;
        if (type == AVE) {
            auto exclude_pad = !avePoolPaddedArea;
            auto ave_pool = std::make_shared<ngraph::op::v1::AvgPool>(ieInpNode, ngraph::Strides(strides),
                            ngraph::Shape(pads_begin), ngraph::Shape(pads_end), ngraph::Shape(kernel_size),
                            exclude_pad, rounding_type, pad_type);
            return Ptr<BackendNode>(new InfEngineNgraphNode(ave_pool));
        }
        else if (type == SUM) {
            ngraph::Shape inpShape = ieInpNode->get_shape();
            CV_Assert(inpShape.size() == 2 + kernel_size.size());
            std::vector<int64_t> axes;
            for (size_t i = 0; i < kernel_size.size(); i++)
            {
                if (inpShape[2 + i] == kernel_size[i])
                    axes.push_back(2 + i);
            }
            auto reduction_axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{axes.size()}, axes);
            auto reduce_sum = std::make_shared<ngraph::op::v1::ReduceSum>(ieInpNode, reduction_axes, true);
            return Ptr<BackendNode>(new InfEngineNgraphNode(reduce_sum));
        }
        else if (type == MAX) {
            auto max_pool = std::make_shared<ngraph::op::v1::MaxPool>(ieInpNode, ngraph::Strides(strides),
                            ngraph::Shape(pads_begin), ngraph::Shape(pads_end), ngraph::Shape(kernel_size),
                            rounding_type, pad_type);
            return Ptr<BackendNode>(new InfEngineNgraphNode(max_pool));
        }
        else if (type == ROI) {
            auto& coords = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
            auto roi = std::make_shared<ngraph::op::ROIPooling>(ieInpNode, coords,
                       ngraph::Shape{(size_t)pooledSize.height, (size_t)pooledSize.width}, spatialScale, "max");
            return Ptr<BackendNode>(new InfEngineNgraphNode(roi));
        }
        else if (type == PSROI) {
            auto& coords = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
            auto psroi = std::make_shared<ngraph::op::PSROIPooling>(ieInpNode, coords,
                         (size_t)psRoiOutChannels, (size_t)pooledSize.width, spatialScale, 1, 1, "average");
            return Ptr<BackendNode>(new InfEngineNgraphNode(psroi));
        }
        else
            CV_Error(Error::StsNotImplemented, "Unsupported pooling type");
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        // std::cout << "Use WebNN Pooling Layer's Implementation." << std::endl;
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        webnn::Pool2dOptions options;
        std::vector<int32_t> kernelSize(kernel_size.begin(), kernel_size.end());
        std::vector<int32_t> Strides(strides.begin(), strides.end());
        std::vector<int32_t> Padding;
        if (padMode.empty()) {
            Padding = {static_cast<int32_t>(pads_begin[0]),
                      static_cast<int32_t>(pads_end[0]),
                      static_cast<int32_t>(pads_begin[1]),
                      static_cast<int32_t>(pads_end[1])};
        } else if (padMode == "VALID") {
            Padding = {0, 0, 0, 0};
        } else if (padMode == "SAME") {
            options.autoPad = ml::AutoPad::SameUpper;
        }
        // std::cout << "padMode: " << padMode << std::endl;
        options.windowDimensions = kernelSize;
        options.strides = Strides;
        options.padding = Padding;
        if (type == MAX)
        {
            auto operand = webnnGraphBuilder.MaxPool2d(webnnInpOperand, options.AsPtr());
            return Ptr<BackendNode>(new WebnnBackendNode(operand));
        }
        else if (type == AVE)
        {
            auto operand = webnnGraphBuilder.AveragePool2d(webnnInpOperand, options.AsPtr());
            return Ptr<BackendNode>(new WebnnBackendNode(operand));
        } else {
            CV_Error(Error::StsNotImplemented, "Unsupported pooling type");
        }
    }
#endif // HAVE_WEBNN

    class PoolingInvoker : public ParallelLoopBody
    {
    public:
        const Mat* src, *rois;
        Mat *dst, *mask;
        int pad_l, pad_t, pad_r, pad_b;
        bool avePoolPaddedArea;
        int nstripes;
        bool computeMaxIdx;
        std::vector<int> ofsbuf;
        int poolingType;
        float spatialScale;

        std::vector<size_t> pads_begin, pads_end;
        std::vector<size_t> kernel_size;
        std::vector<size_t> strides;

        PoolingInvoker() : src(0), rois(0), dst(0), mask(0), pad_l(0), pad_t(0), pad_r(0), pad_b(0),
                           avePoolPaddedArea(false), nstripes(0),
                           computeMaxIdx(0), poolingType(MAX), spatialScale(0) {}

        static void run(const Mat& src, const Mat& rois, Mat& dst, Mat& mask,
                        std::vector<size_t> kernel_size, std::vector<size_t> strides,
                        std::vector<size_t> pads_begin, std::vector<size_t> pads_end,
                        bool avePoolPaddedArea, int poolingType, float spatialScale,
                        bool computeMaxIdx, int nstripes)
        {
            CV_Assert_N(
                      src.isContinuous(), dst.isContinuous(),
                      src.type() == CV_32F, src.type() == dst.type(),
                      src.dims == 3 || src.dims == 4 || src.dims == 5, dst.dims == 3 || dst.dims == 4 || dst.dims == 5,
                      (((poolingType == ROI || poolingType == PSROI) &&
                      dst.size[0] == rois.size[0]) || src.size[0] == dst.size[0]),
                      poolingType == PSROI || src.size[1] == dst.size[1],
                      (mask.empty() || (mask.type() == src.type() && mask.size == dst.size)));

            PoolingInvoker p;

            bool isPool1D = src.dims == 3;
            bool isPool3D = src.dims == 5;

            p.src = &src;
            p.rois = &rois;
            p.dst = &dst;

            p.kernel_size = kernel_size;
            p.strides = strides;
            p.pads_begin = pads_begin;
            p.pads_end = pads_end;

            p.mask = &mask;
            p.pad_l = pads_begin.back();
            p.pad_t = isPool1D ? 0 : pads_begin[pads_begin.size() - 2];
            p.pad_r = pads_end.back();
            p.pad_b = isPool1D ? 0 : pads_end[pads_end.size() - 2];

            p.avePoolPaddedArea = avePoolPaddedArea;
            p.nstripes = nstripes;
            p.computeMaxIdx = computeMaxIdx;
            p.poolingType = poolingType;
            p.spatialScale = spatialScale;

            if( !computeMaxIdx )
            {
                int height = isPool1D ? 1 : src.size[src.dims - 2];
                int width = src.size[src.dims - 1];

                int kernel_d = isPool3D ? kernel_size[0] : 1;
                int kernel_h = isPool1D ? 1 : kernel_size[kernel_size.size() - 2];
                int kernel_w = kernel_size.back();

                p.ofsbuf.resize(kernel_d * kernel_h * kernel_w);
                for (int i = 0; i < kernel_d; ++i) {
                    for (int j = 0; j < kernel_h; ++j) {
                        for (int k = 0; k < kernel_w; ++k) {
                            p.ofsbuf[i * kernel_h * kernel_w + j * kernel_w + k] = width * height * i + width * j + k;
                        }
                    }
                }
            }

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            int channels = dst->size[1];

            bool isPool3D = src->dims == 5;
            bool isPool2D = src->dims == 4;
            bool isPool1D = src->dims == 3;
            int depth = isPool3D? dst->size[2] : 1;
            int height = isPool1D? 1 : dst->size[dst->dims - 2];
            int width = dst->size[dst->dims - 1];

            int inp_depth = isPool3D? src->size[2] : 1;
            int inp_height = isPool1D? 1 : src->size[src->dims - 2];
            int inp_width = src->size[src->dims - 1];

            size_t total = dst->total();
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);

            int kernel_d = isPool3D? kernel_size[0] : 1;
            int kernel_h = isPool1D? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();

            int stride_d = isPool3D? strides[0] : 0;
            int stride_h = isPool1D? 1 :strides[strides.size() - 2];
            int stride_w = strides.back();
            bool compMaxIdx = computeMaxIdx;

#if CV_SIMD128
            const int* ofsptr = ofsbuf.empty() ? 0 : (const int*)&ofsbuf[0];
            if (poolingType == MAX && !compMaxIdx && !ofsptr)
                CV_Error(Error::StsBadArg, "ofsbuf should be initialized in this mode");
            v_float32x4 idx00(0.f, (float)stride_w, (float)(stride_w*2), (float)(stride_w*3));
            v_float32x4 ones = v_setall_f32(1.f);
            v_float32x4 idx_delta = v_setall_f32((float)(inp_width - kernel_w));
#endif

            for( size_t ofs0 = stripeStart; ofs0 < stripeEnd; )
            {
                size_t ofs = ofs0;
                int x0 = (int)(ofs % width);
                ofs /= width;
                int y0 = (int)(ofs % height);
                ofs /= height;

                int d0 = (int)(ofs % depth);
                ofs /= depth;

                int c = (int)(ofs % channels);
                int n = (int)(ofs / channels);
                int ystart, yend;
                int dstart = 0, dend = 1;

                const float *srcData = 0;
                if (poolingType == ROI)
                {
                    const float *roisData = rois->ptr<float>(n);
                    int ystartROI = roundRoiSize(roisData[2] * spatialScale);
                    int yendROI = roundRoiSize(roisData[4] * spatialScale);
                    int roiHeight = std::max(yendROI - ystartROI + 1, 1);
                    float roiRatio = (float)roiHeight / height;

                    ystart = ystartROI + y0 * roiRatio;
                    yend = ystartROI + std::ceil((y0 + 1) * roiRatio);

                    CV_Assert(roisData[0] < src->size[0]);
                    srcData = src->ptr<float>(roisData[0], c);
                }
                else if (poolingType == PSROI)
                {
                    const float *roisData = rois->ptr<float>(n);
                    float ystartROI = roundRoiSize(roisData[2]) * spatialScale;
                    float yendROI = roundRoiSize(roisData[4] + 1) * spatialScale;
                    float roiHeight = std::max(yendROI - ystartROI, 0.1f);
                    float roiRatio = roiHeight / height;

                    ystart = (int)std::floor(ystartROI + y0 * roiRatio);
                    yend = (int)std::ceil(ystartROI + (y0 + 1) * roiRatio);
                }
                else
                {
                    int pad_d_begin = (pads_begin.size() == 3) ? pads_begin[0] : 0;
                    dstart = d0 * stride_d - pad_d_begin;
                    dend = min(dstart + kernel_d, (int)(inp_depth + pads_end[0]));

                    ystart = y0 * stride_h - pad_t;
                    yend = min(ystart + kernel_h, inp_height + pad_b);
                    srcData = src->ptr<float>(n, c);
                }
                int ddelta = dend - dstart;
                dstart = max(dstart, 0);
                dend = min(dend, inp_depth);
                int ydelta = yend - ystart;
                ystart = max(ystart, 0);
                yend = min(yend, inp_height);
                float *dstData = &dst->ptr<float>(n, c, d0)[y0 * width];
                float *dstMaskData = mask->data ? &mask->ptr<float>(n, c, d0)[y0 * width] : 0;

                int delta = std::min((int)(stripeEnd - ofs0), width - x0);
                ofs0 += delta;
                int x1 = x0 + delta;

                if( poolingType == MAX)
                    for( ; x0 < x1; x0++ )
                    {
                        int xstart = x0 * stride_w - pad_l;
                        int xend = min(xstart + kernel_w, inp_width);
                        xstart = max(xstart, 0);
                        if (xstart >= xend || ystart >= yend)
                        {
                            dstData[x0] = 0;
                            if (compMaxIdx && dstMaskData)
                                dstMaskData[x0] = -1;
                            continue;
                        }
#if CV_SIMD128
                        if( isPool2D && xstart > 0 && x0 + 7 < x1 && (x0 + 7) * stride_w - pad_l + kernel_w < inp_width )
                        {
                            if( compMaxIdx )
                            {
                                v_float32x4 max_val0 = v_setall_f32(-FLT_MAX);
                                v_float32x4 max_val1 = max_val0;
                                v_float32x4 max_idx0 = v_setall_f32(-1.f);
                                v_float32x4 max_idx1 = max_idx0;
                                int index0 = ystart * inp_width + xstart;
                                v_float32x4 idx0 = idx00 + v_setall_f32((float)index0);
                                v_float32x4 idx1 = idx0 + v_setall_f32((float)(stride_w*4));

                                for (int y = ystart; y < yend; ++y)
                                {
                                    for (int x = xstart; x < xend; ++x, idx0 += ones, idx1 += ones)
                                    {
                                        const int index = y * inp_width + x;
                                        v_float32x4 v0(srcData[index], srcData[index + stride_w],
                                                       srcData[index + stride_w*2], srcData[index + stride_w*3]);
                                        v_float32x4 v1(srcData[index + stride_w*4], srcData[index + stride_w*5],
                                                       srcData[index + stride_w*6], srcData[index + stride_w*7]);
                                        max_idx0 = v_select(v0 > max_val0, idx0, max_idx0);
                                        max_idx1 = v_select(v1 > max_val1, idx1, max_idx1);
                                        max_val0 = v_max(max_val0, v0);
                                        max_val1 = v_max(max_val1, v1);
                                    }
                                    idx0 += idx_delta;
                                    idx1 += idx_delta;
                                }
                                v_store(dstData + x0, max_val0);
                                v_store(dstData + x0 + 4, max_val1);
                                if (dstMaskData)
                                {
                                    v_store(dstMaskData + x0, max_idx0);
                                    v_store(dstMaskData + x0 + 4, max_idx1);
                                }
                                x0 += 7;
                            }
                            else
                            {
                                v_float32x4 max_val0 = v_setall_f32(-FLT_MAX);
                                v_float32x4 max_val1 = max_val0;

                                if( yend - ystart == kernel_h )
                                {
                                    const float* srcData1 = srcData + ystart*inp_width + xstart;
                                    if( stride_w == 1 )
                                        for (int k = 0; k < kernel_w*kernel_h; k++)
                                        {
                                            int index = ofsptr[k];
                                            v_float32x4 v0 = v_load(srcData1 + index);
                                            v_float32x4 v1 = v_load(srcData1 + index + 4);
                                            max_val0 = v_max(max_val0, v0);
                                            max_val1 = v_max(max_val1, v1);
                                        }
                                    else if( stride_w == 2 )
                                        for (int k = 0; k < kernel_w*kernel_h; k++)
                                        {
                                            int index = ofsptr[k];
                                            v_float32x4 v0, v1, dummy;
                                            v_load_deinterleave(srcData1 + index, v0, dummy);     // f0  f2  f4  f6  ,f1  f3  f5  f7
                                            v_load_deinterleave(srcData1 + index + 8, v1, dummy); // f8  f10 f12 f14 ,f9  f11 f13 f15
                                            max_val0 = v_max(max_val0, v0);
                                            max_val1 = v_max(max_val1, v1);
                                        }
                                    else
                                        for (int k = 0; k < kernel_w*kernel_h; k++)
                                        {
                                            int index = ofsptr[k];
                                            v_float32x4 v0(srcData1[index], srcData1[index + stride_w],
                                                           srcData1[index + stride_w*2], srcData1[index + stride_w*3]);
                                            v_float32x4 v1(srcData1[index + stride_w*4], srcData1[index + stride_w*5],
                                                           srcData1[index + stride_w*6], srcData1[index + stride_w*7]);
                                            max_val0 = v_max(max_val0, v0);
                                            max_val1 = v_max(max_val1, v1);
                                        }
                                }
                                else
                                {
                                    for (int y = ystart; y < yend; ++y)
                                    {
                                        for (int x = xstart; x < xend; ++x)
                                        {
                                            const int index = y * inp_width + x;
                                            v_float32x4 v0(srcData[index], srcData[index + stride_w],
                                                           srcData[index + stride_w*2], srcData[index + stride_w*3]);
                                            v_float32x4 v1(srcData[index + stride_w*4], srcData[index + stride_w*5],
                                                           srcData[index + stride_w*6], srcData[index + stride_w*7]);
                                            max_val0 = v_max(max_val0, v0);
                                            max_val1 = v_max(max_val1, v1);
                                        }
                                    }
                                }
                                v_store(dstData + x0, max_val0);
                                v_store(dstData + x0 + 4, max_val1);
                                x0 += 7;
                            }
                        }
                        else
#else
                        CV_UNUSED(isPool2D);
#endif
                        if( isPool1D )
                        {
                            const float* first = srcData + xstart;
                            const float* last = srcData + xend;
                            const float* max_elem = std::max_element(first, last);
                            if (max_elem!=last)
                            {
                                dstData[x0] = *max_elem;
                                if( compMaxIdx && dstMaskData )
                                {
                                    dstMaskData[x0] = std::distance(first, max_elem);
                                }
                            }
                        }
                        else
                        {
                            float max_val = -FLT_MAX;
                            if( compMaxIdx )
                            {
                                int max_index = -1;
                                for (int d = dstart; d < dend; ++d)
                                    for (int y = ystart; y < yend; ++y)
                                        for (int x = xstart; x < xend; ++x)
                                        {
                                            const int index = d * inp_width * inp_height + y * inp_width + x;
                                            float val = srcData[index];
                                            if (val > max_val)
                                            {
                                                max_val = val;
                                                max_index = index;
                                            }
                                        }
                                dstData[x0] = max_val;
                                if (dstMaskData)
                                    dstMaskData[x0] = max_index;
                            }
                            else
                            {
                                for (int d = dstart; d < dend; ++d) {
                                    for (int y = ystart; y < yend; ++y) {
                                        for (int x = xstart; x < xend; ++x) {
                                            const int index = d * inp_width * inp_height + y * inp_width + x;
                                            float val = srcData[index];
                                            max_val = std::max(max_val, val);
                                        }
                                    }
                                }
                                dstData[x0] = max_val;
                            }
                        }
                    }
                else if (poolingType == AVE || poolingType == SUM)
                {
                    for( ; x0 < x1; ++x0)
                    {
                        int xstart = x0 * stride_w - pad_l;
                        int xend = min(xstart + kernel_w, inp_width + pad_r);
                        int xdelta = xend - xstart;
                        xstart = max(xstart, 0);
                        xend = min(xend, inp_width);
                        float inv_kernel_area = avePoolPaddedArea ? xdelta * ydelta * ddelta :
                                                ((dend - dstart) * (yend - ystart) * (xend - xstart));
                        inv_kernel_area = poolingType == AVE ? 1.0 / inv_kernel_area : 1.0;
#if CV_SIMD128
                        if( isPool2D && xstart > 0 && x0 + 7 < x1 && (x0 + 7) * stride_w - pad_l + kernel_w < inp_width )
                        {
                            v_float32x4 sum_val0 = v_setzero_f32(), sum_val1 = v_setzero_f32();
                            v_float32x4 ikarea = v_setall_f32(inv_kernel_area);

                            for (int y = ystart; y < yend; ++y)
                            {
                                for (int x = xstart; x < xend; ++x)
                                {
                                    const int index = y * inp_width + x;
                                    v_float32x4 v0(srcData[index], srcData[index + stride_w],
                                                   srcData[index + stride_w*2], srcData[index + stride_w*3]);
                                    v_float32x4 v1(srcData[index + stride_w*4], srcData[index + stride_w*5],
                                                   srcData[index + stride_w*6], srcData[index + stride_w*7]);
                                    sum_val0 += v0;
                                    sum_val1 += v1;
                                }
                            }
                            v_store(dstData + x0, sum_val0*ikarea);
                            v_store(dstData + x0 + 4, sum_val1*ikarea);
                            x0 += 7;
                        }
                        else
#endif
                        if( isPool1D )
                        {
                            const float* first = srcData + xstart;
                            const float* last = srcData + xend;
                            float sum_val = std::accumulate(first, last, 0.f);
                            dstData[x0] = sum_val*inv_kernel_area;
                        }
                        else
                        {
                            float sum_val = 0.f;
                            for (int d = dstart; d < dend; ++d) {
                                for (int y = ystart; y < yend; ++y) {
                                    for (int x = xstart; x < xend; ++x) {
                                        const int index = d * inp_width * inp_height + y * inp_width + x;
                                        float val = srcData[index];
                                        sum_val += val;
                                    }
                                }
                            }
                            dstData[x0] = sum_val*inv_kernel_area;
                        }
                    }
                }
                else if (poolingType == ROI)
                {
                    const float *roisData = rois->ptr<float>(n);
                    int xstartROI = roundRoiSize(roisData[1] * spatialScale);
                    int xendROI = roundRoiSize(roisData[3] * spatialScale);
                    int roiWidth = std::max(xendROI - xstartROI + 1, 1);
                    float roiRatio = (float)roiWidth / width;
                    for( ; x0 < x1; x0++ )
                    {
                        int xstart = xstartROI + x0 * roiRatio;
                        int xend = xstartROI + std::ceil((x0 + 1) * roiRatio);
                        xstart = max(xstart, 0);
                        xend = min(xend, inp_width);
                        if (xstart >= xend || ystart >= yend)
                        {
                            dstData[x0] = 0;
                            if (compMaxIdx && dstMaskData)
                                dstMaskData[x0] = -1;
                            continue;
                        }
                        float max_val = -FLT_MAX;
                        for (int y = ystart; y < yend; ++y)
                            for (int x = xstart; x < xend; ++x)
                            {
                                const int index = y * inp_width + x;
                                float val = srcData[index];
                                max_val = std::max(max_val, val);
                            }
                        dstData[x0] = max_val;
                    }
                }
                else  // PSROI
                {
                    const float *roisData = rois->ptr<float>(n);
                    CV_Assert(roisData[0] < src->size[0]);
                    float xstartROI = roundRoiSize(roisData[1]) * spatialScale;
                    float xendROI = roundRoiSize(roisData[3] + 1) * spatialScale;
                    float roiWidth = std::max(xendROI - xstartROI, 0.1f);
                    float roiRatio = roiWidth / width;
                    for( ; x0 < x1; x0++ )
                    {
                        int xstart = (int)std::floor(xstartROI + x0 * roiRatio);
                        int xend = (int)std::ceil(xstartROI + (x0 + 1) * roiRatio);
                        xstart = max(xstart, 0);
                        xend = min(xend, inp_width);
                        if (xstart >= xend || ystart >= yend)
                        {
                            dstData[x0] = 0;
                            continue;
                        }

                        srcData = src->ptr<float>(roisData[0], (c * height + y0) * width + x0);
                        float sum_val = 0.f;
                        for (int y = ystart; y < yend; ++y)
                            for (int x = xstart; x < xend; ++x)
                            {
                                const int index = y * inp_width + x;
                                float val = srcData[index];
                                sum_val += val;
                            }
                        dstData[x0] = sum_val / ((yend - ystart) * (xend - xstart));
                    }
                }
            }
        }
    };

    void maxPooling(Mat &src, Mat &dst, Mat &mask)
    {
        const int nstripes = getNumThreads();
        Mat rois;
        PoolingInvoker::run(src, rois, dst, mask, kernel_size, strides, pads_begin, pads_end, avePoolPaddedArea, type, spatialScale, computeMaxIdx, nstripes);
    }

    void avePooling(Mat &src, Mat &dst)
    {
        const int nstripes = getNumThreads();
        Mat rois, mask;
        PoolingInvoker::run(src, rois, dst, mask, kernel_size, strides, pads_begin, pads_end, avePoolPaddedArea, type, spatialScale, computeMaxIdx, nstripes);
    }

    void roiPooling(const Mat &src, const Mat &rois, Mat &dst)
    {
        const int nstripes = getNumThreads();
        Mat mask;
        kernel_size.resize(2);
        strides.resize(2);
        pads_begin.resize(2);
        pads_end.resize(2);
        PoolingInvoker::run(src, rois, dst, mask, kernel_size, strides, pads_begin, pads_end, avePoolPaddedArea, type, spatialScale, computeMaxIdx, nstripes);
    }

    virtual Ptr<BackendNode> initMaxPoolingHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);
        const int inWidth = inputBuffer.width();
        const int inHeight = inputBuffer.height();
        const HALIDE_DIFF_T kernelHeight = (HALIDE_DIFF_T)kernel_size[0];
        const HALIDE_DIFF_T kernelWidth = (HALIDE_DIFF_T)kernel_size[1];
        const HALIDE_DIFF_T strideHeight = (HALIDE_DIFF_T)strides[0];
        const HALIDE_DIFF_T strideWidth = (HALIDE_DIFF_T)strides[1];
        const HALIDE_DIFF_T paddingTop = (HALIDE_DIFF_T)pads_begin[0];
        const HALIDE_DIFF_T paddingLeft = (HALIDE_DIFF_T)pads_begin[1];

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::RDom r(0, kernelWidth, 0, kernelHeight);
        Halide::Expr kx, ky;
        if(paddingLeft || paddingTop)
        {
            kx = clamp(x * strideWidth + r.x - paddingLeft, 0, inWidth - 1);
            ky = clamp(y * strideHeight + r.y - paddingTop, 0, inHeight - 1);
        }
        else
        {
            kx = min(x * strideWidth + r.x, inWidth - 1);
            ky = min(y * strideHeight + r.y, inHeight - 1);
        }

        // Halide::argmax returns tuple (r.x, r.y, max).
        Halide::Tuple res = argmax(inputBuffer(kx, ky, c, n));

        if (!computeMaxIdx)
        {
            top(x, y, c, n) = res[2];
            return Ptr<BackendNode>(new HalideBackendNode(top));
        }

        // Compute offset from argmax in range [0, kernel_size).
        Halide::Expr max_index;
        if(paddingLeft || paddingTop)
        {
            max_index = clamp(y * strideHeight + res[1] - paddingTop,
                              0, inHeight - 1) * inWidth +
                        clamp(x * strideWidth + res[0] - paddingLeft,
                              0, inWidth - 1);
        }
        else
        {
            max_index = min(y * strideHeight + res[1], inHeight - 1) * inWidth +
                        min(x * strideWidth + res[0], inWidth - 1);
        }
        top(x, y, c, n) = { res[2], Halide::cast<float>(max_index) };
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initAvePoolingHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);

        const int inW = inputBuffer.width(), inH = inputBuffer.height();
        const HALIDE_DIFF_T kernelHeight = (HALIDE_DIFF_T)kernel_size[0];
        const HALIDE_DIFF_T kernelWidth = (HALIDE_DIFF_T)kernel_size[1];
        const HALIDE_DIFF_T strideHeight = (HALIDE_DIFF_T)strides[0];
        const HALIDE_DIFF_T strideWidth = (HALIDE_DIFF_T)strides[1];
        if ((inW - kernelWidth) % strideWidth || (inH - kernelHeight) % strideHeight)
        {
            CV_Error(cv::Error::StsNotImplemented,
                     "Halide backend for average pooling with partial "
                     "kernels is not implemented");
        }

        const float norm = 1.0f / (kernelWidth * kernelHeight);

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::RDom r(0, kernelWidth, 0, kernelHeight);
        top(x, y, c, n) = sum(
            inputBuffer(x * strideWidth + r.x,
                        y * strideHeight + r.y, c, n)) * norm;
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

    virtual void applyHalideScheduler(Ptr<BackendNode>& node,
                                      const std::vector<Mat*> &inputs,
                                      const std::vector<Mat> &outputs,
                                      int targetId) const CV_OVERRIDE
    {
#ifdef  HAVE_HALIDE
        if (targetId != DNN_TARGET_CPU)
        {
            Layer::applyHalideScheduler(node, inputs, outputs, targetId);
            return;
        }
        Halide::Var x("x"), y("y"), c("c"), n("n"), tile("tile"),
                    xi("xi"), yi("yi"), ci("ci"), xo("xo"), yo("yo"), co("co");
        Halide::Func& top = node.dynamicCast<HalideBackendNode>()->funcs.back();

        int outW, outH, outC, outN;
        getCanonicalSize(outputs[0].size, &outW, &outH, &outC, &outN);

        if (outW < 8 || outH < 8)
        {
            if (outC > 8)
                top.split(c, co, ci, 8)
                   .fuse(x, y, tile).fuse(co, tile, tile).fuse(n, tile, tile)
                   .parallel(tile)
                   .vectorize(ci);
            else
            {
                top.fuse(y, c, tile).fuse(n, tile, tile)
                   .parallel(tile);
                if (outW > 1)
                    top.vectorize(x);
            }
        }
        else
        {
            if (outC > 8)
                top.split(x, xo, xi, 8).split(y, yo, yi, 8).split(c, co, ci, 8)
                   .fuse(xo, yo, tile).fuse(co, tile, tile).fuse(n, tile, tile)
                   .parallel(tile)
                   .vectorize(xi);
            else
                top.split(x, xo, xi, 8).split(y, yo, yi, 8)
                   .fuse(xo, yo, tile).fuse(c, tile, tile).fuse(n, tile, tile)
                   .parallel(tile)
                   .vectorize(xi);
        }
#endif  // HAVE_HALIDE
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() != 0);

        bool isPool1D = inputs[0].size() == 3;
        std::vector<int> inpShape(inputs[0].begin() + 2, inputs[0].end());
        std::vector<int> outShape(inputs[0].begin(), inputs[0].begin() + 2);

        std::vector<size_t> local_kernel;
        if (globalPooling) {
            for (int i = 0; i < inpShape.size(); i++) {
                int idx = isGlobalPooling.size() - inpShape.size() + i;
                local_kernel.push_back(isGlobalPooling[idx] ? inpShape[i] : kernel_size[idx]);
            }
        } else {
            local_kernel = kernel_size;
        }

        if (type == ROI || type == PSROI)
        {
            outShape.push_back(pooledSize.height);
            outShape.push_back(pooledSize.width);
        }
        else
        {
            if (hasDynamicShapes && !shapesInitialized)
            {
                //Just copy input shapes for width and height to prevent errors on loading stage
                for (int i = 0; i < inpShape.size(); i++)
                    outShape.push_back(inpShape[i]);
            }
            else if (padMode.empty())
            {
                size_t addedDims = isPool1D? inpShape.size() : local_kernel.size();
                CV_CheckLE(addedDims, inpShape.size(), "");
                CV_CheckLE(addedDims, pads_begin.size(), "");
                CV_CheckLE(addedDims, pads_end.size(), "");
                CV_CheckLE(addedDims, local_kernel.size(), "");
                CV_CheckLE(addedDims, strides.size(), "");
                for (int i = 0; i < addedDims; i++)
                {
                    float dst = (float) (inpShape[i] + pads_begin[i] + pads_end[i] - local_kernel[i]) / strides[i];
                    CV_CheckGE(dst, 0.0f, "");
                    outShape.push_back(1 + (ceilMode ? ceil(dst) : floor(dst)));
                }

                // If we have padding, ensure that the last pooling starts strictly
                // inside the image (instead of at the padding); otherwise clip the last.
                for (int i = 0; i < addedDims; i++) {
                    if (pads_end[i] && (outShape[2 + i] - 1) * strides[i] >= inpShape[i] + pads_end[i]) {
                        --outShape[2 + i];
                        CV_Assert((outShape[2 + i] - 1) * strides[i] < inpShape[i] + pads_end[i]);
                    }
                }
            } else {
                getConvPoolOutParams(inpShape, local_kernel, strides, padMode,
                                     std::vector<size_t>(local_kernel.size(), 1), outShape);
            }
        }
        if (type == ROI)
        {
            CV_Assert(inputs.size() == 2);
            outShape[0] = inputs[1][0];  // Number of proposals;
        }
        else if (type == PSROI)
        {
            CV_Assert(inputs.size() == 2);
            CV_Assert(psRoiOutChannels * pooledSize.width * pooledSize.height == inputs[0][1]);
            outShape[0] = inputs[1][0];  // Number of proposals;
            outShape[1] = psRoiOutChannels;
        }
        int numOutputs = requiredOutputs ? requiredOutputs : (type == MAX ? 2 : 1);
        CV_Assert(numOutputs == 1 || (numOutputs == 2 && type == MAX));

        outputs.assign(numOutputs, outShape);

        return false;
    }

    bool updateMemoryShapes(const std::vector<MatShape> &inputs) CV_OVERRIDE
    {
        int dims = inputs[0].size();
        CV_Assert(inputs[0][dims - 1] > 0 && inputs[0][dims - 2] > 0);
        shapesInitialized = true;
        return true;
    }

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        if (type == MAX && !computeMaxIdx)
        {
            return true;
        }
        else if (type == AVE || type == SUM)
        {
            float multiplier = scales[0][0] / scales[1][0];
            params.set("multiplier", multiplier);
            params.set("input_zeropoint", zeropoints[0][0]);
            return true;
        }
        return false;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(inputs); // suppress unused variable warning
        long flops = 0;
        bool isPool1D = inputs[0].size() == 3;
        size_t karea = std::accumulate(kernel_size.begin(), isPool1D? kernel_size.begin() + 1 : kernel_size.end(),
                                    1, std::multiplies<size_t>());
        for(int i = 0; i < outputs.size(); i++)
        {
            if (type == MAX)
            {
                if (i%2 == 0)
                    flops += total(outputs[i])*karea;
            }
            else
            {
                flops += total(outputs[i])*(karea + 1);
            }
        }
        return flops;
    }
private:
    enum Type
    {
        MAX,
        AVE,
        STOCHASTIC,
        SUM,
        ROI,   // RoI pooling, https://arxiv.org/pdf/1504.08083.pdf
        PSROI  // Position-sensitive RoI pooling, https://arxiv.org/pdf/1605.06409.pdf
    };
    bool hasDynamicShapes;
    bool shapesInitialized;
};

Ptr<PoolingLayer> PoolingLayer::create(const LayerParams& params)
{
    return Ptr<PoolingLayer>(new PoolingLayerImpl(params));
}

}
}
