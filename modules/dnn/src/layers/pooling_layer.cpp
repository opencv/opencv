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
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include "../op_vkcom.hpp"
#include <float.h>
#include <algorithm>
using std::max;
using std::min;

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
using namespace cv::dnn::ocl4dnn;
#endif

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
        stride = Size(1, 1);

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
            else
                CV_Error(Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");

            getPoolingKernelParams(params, kernel.height, kernel.width, globalPooling,
                                   pad_t, pad_l, pad_b, pad_r, stride.height, stride.width, padMode);

            pad.width = pad_l;
            pad.height = pad_t;
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

        cv::Size inp(inputs[0].size[3], inputs[0].size[2]),
                out(outputs[0].size[3], outputs[0].size[2]);

        if(globalPooling)
        {
            kernel = inp;
        }

        getConvPoolPaddings(inp, out, kernel, stride, padMode, Size(1, 1), pad_t, pad_l, pad_b, pad_r);
        pad.width = pad_l;
        pad.height = pad_t;

#ifdef HAVE_OPENCL
        poolOp.release();
#endif
        computeMaxIdx = type == MAX;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE)
        {
            if (preferableTarget == DNN_TARGET_MYRIAD)
                return type == MAX || type == AVE;
            else
                return type != STOCHASTIC;
        }
        else
            return backendId == DNN_BACKEND_OPENCV ||
                   backendId == DNN_BACKEND_HALIDE && haveHalide() &&
                   (type == MAX || type == AVE && !pad_t && !pad_l && !pad_b && !pad_r) ||
                   backendId == DNN_BACKEND_VKCOM && haveVulkan() &&
                   (type == MAX || type == AVE);
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
            config.kernel = kernel;
            config.pad_l = pad_l;
            config.pad_t = pad_t;
            config.pad_r = pad_r;
            config.pad_b = pad_b;
            config.stride = stride;
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
            case AVE:
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

    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
#ifdef HAVE_VULKAN
        int padding_mode;
        vkcom::PoolType pool_type;
        int filter_size[2] = {kernel.height, kernel.width};
        int pad_size[2] = {pad.height, pad.width};
        int stride_size[2] = {stride.height, stride.width};
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
#endif
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
        if (type == MAX)
            return initMaxPoolingHalide(inputs);
        else if (type == AVE)
            return initAvePoolingHalide(inputs);
        else
            return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.precision = InferenceEngine::Precision::FP32;

        std::shared_ptr<InferenceEngine::CNNLayer> ieLayer;
        if (type == MAX || type == AVE)
        {
            lp.type = "Pooling";
            InferenceEngine::PoolingLayer* poolLayer = new InferenceEngine::PoolingLayer(lp);
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2018R3)
            poolLayer->_kernel.insert(InferenceEngine::X_AXIS, kernel.width);
            poolLayer->_kernel.insert(InferenceEngine::Y_AXIS, kernel.height);
            poolLayer->_stride.insert(InferenceEngine::X_AXIS, stride.width);
            poolLayer->_stride.insert(InferenceEngine::Y_AXIS, stride.height);
            poolLayer->_padding.insert(InferenceEngine::X_AXIS, pad_l);
            poolLayer->_padding.insert(InferenceEngine::Y_AXIS, pad_t);
            poolLayer->_pads_end.insert(InferenceEngine::X_AXIS, pad_r);
            poolLayer->_pads_end.insert(InferenceEngine::Y_AXIS, pad_b);
#else
            poolLayer->_kernel_x = kernel.width;
            poolLayer->_kernel_y = kernel.height;
            poolLayer->_stride_x = stride.width;
            poolLayer->_stride_y = stride.height;
            poolLayer->_padding_x = pad_l;
            poolLayer->_padding_y = pad_t;
            poolLayer->params["pad-r"] = format("%d", pad_r);
            poolLayer->params["pad-b"] = format("%d", pad_b);
#endif
            poolLayer->_exclude_pad = type == AVE && padMode == "SAME";
            poolLayer->params["rounding-type"] = ceilMode ? "ceil" : "floor";
            poolLayer->_type = type == MAX ? InferenceEngine::PoolingLayer::PoolType::MAX :
                                             InferenceEngine::PoolingLayer::PoolType::AVG;
            ieLayer = std::shared_ptr<InferenceEngine::CNNLayer>(poolLayer);
        }
        else if (type == ROI)
        {
            lp.type = "ROIPooling";
            ieLayer = std::shared_ptr<InferenceEngine::CNNLayer>(new InferenceEngine::CNNLayer(lp));
            ieLayer->params["pooled_w"] = format("%d", pooledSize.width);
            ieLayer->params["pooled_h"] = format("%d", pooledSize.height);
            ieLayer->params["spatial_scale"] = format("%f", spatialScale);
        }
        else if (type == PSROI)
        {
            lp.type = "PSROIPooling";
            ieLayer = std::shared_ptr<InferenceEngine::CNNLayer>(new InferenceEngine::CNNLayer(lp));
            ieLayer->params["output_dim"] = format("%d", psRoiOutChannels);
            ieLayer->params["group_size"] = format("%d", pooledSize.width);
            ieLayer->params["spatial_scale"] = format("%f", spatialScale);
        }
        else
            CV_Error(Error::StsNotImplemented, "Unsupported pooling type");

        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }


    class PoolingInvoker : public ParallelLoopBody
    {
    public:
        const Mat* src, *rois;
        Mat *dst, *mask;
        Size kernel, stride;
        int pad_l, pad_t, pad_r, pad_b;
        bool avePoolPaddedArea;
        int nstripes;
        bool computeMaxIdx;
        std::vector<int> ofsbuf;
        int poolingType;
        float spatialScale;

        PoolingInvoker() : src(0), rois(0), dst(0), mask(0), pad_l(0), pad_t(0), pad_r(0), pad_b(0),
                           avePoolPaddedArea(false), nstripes(0),
                           computeMaxIdx(0), poolingType(MAX), spatialScale(0) {}

        static void run(const Mat& src, const Mat& rois, Mat& dst, Mat& mask, Size kernel,
                        Size stride, int pad_l, int pad_t, int pad_r, int pad_b, bool avePoolPaddedArea, int poolingType, float spatialScale,
                        bool computeMaxIdx, int nstripes)
        {
            CV_Assert_N(
                      src.isContinuous(), dst.isContinuous(),
                      src.type() == CV_32F, src.type() == dst.type(),
                      src.dims == 4, dst.dims == 4,
                      ((poolingType == ROI || poolingType == PSROI) && dst.size[0] ==rois.size[0] || src.size[0] == dst.size[0]),
                       poolingType == PSROI || src.size[1] == dst.size[1],
                      (mask.empty() || (mask.type() == src.type() && mask.size == dst.size)));

            PoolingInvoker p;

            p.src = &src;
            p.rois = &rois;
            p.dst = &dst;
            p.mask = &mask;
            p.kernel = kernel;
            p.stride = stride;
            p.pad_l = pad_l;
            p.pad_t = pad_t;
            p.pad_r = pad_r;
            p.pad_b = pad_b;
            p.avePoolPaddedArea = avePoolPaddedArea;
            p.nstripes = nstripes;
            p.computeMaxIdx = computeMaxIdx;
            p.poolingType = poolingType;
            p.spatialScale = spatialScale;

            if( !computeMaxIdx )
            {
                p.ofsbuf.resize(kernel.width*kernel.height);
                for( int i = 0; i < kernel.height; i++ )
                    for( int j = 0; j < kernel.width; j++ )
                        p.ofsbuf[i*kernel.width + j] = src.size[3]*i + j;
            }

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            int channels = dst->size[1], width = dst->size[3], height = dst->size[2];
            int inp_width = src->size[3], inp_height = src->size[2];
            size_t total = dst->total();
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);
            int kernel_w = kernel.width, kernel_h = kernel.height;
            int stride_w = stride.width, stride_h = stride.height;
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
                int c = (int)(ofs % channels);
                int n = (int)(ofs / channels);
                int ystart, yend;

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
                    ystart = y0 * stride_h - pad_t;
                    yend = min(ystart + kernel_h, inp_height + pad_b);
                    srcData = src->ptr<float>(n, c);
                }
                int ydelta = yend - ystart;
                ystart = max(ystart, 0);
                yend = min(yend, inp_height);
                float *dstData = dst->ptr<float>(n, c, y0);
                float *dstMaskData = mask->data ? mask->ptr<float>(n, c, y0) : 0;

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
                        if( xstart > 0 && x0 + 7 < x1 && (x0 + 7) * stride_w - pad_l + kernel_w < inp_width )
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
#endif
                        {
                            float max_val = -FLT_MAX;
                            if( compMaxIdx )
                            {
                                int max_index = -1;
                                for (int y = ystart; y < yend; ++y)
                                    for (int x = xstart; x < xend; ++x)
                                    {
                                        const int index = y * inp_width + x;
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
                    }
                else if (poolingType == AVE)
                {
                    for( ; x0 < x1; x0++ )
                    {
                        int xstart = x0 * stride_w - pad_l;
                        int xend = min(xstart + kernel_w, inp_width + pad_r);
                        int xdelta = xend - xstart;
                        xstart = max(xstart, 0);
                        xend = min(xend, inp_width);
                        float inv_kernel_area = avePoolPaddedArea ? xdelta * ydelta : ((yend - ystart) * (xend - xstart));
                        inv_kernel_area = 1.0 / inv_kernel_area;
#if CV_SIMD128
                        if( xstart > 0 && x0 + 7 < x1 && (x0 + 7) * stride_w - pad_l + kernel_w < inp_width )
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
                        {
                            float sum_val = 0.f;
                            for (int y = ystart; y < yend; ++y)
                                for (int x = xstart; x < xend; ++x)
                                {
                                    const int index = y * inp_width + x;
                                    float val = srcData[index];
                                    sum_val += val;
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
        PoolingInvoker::run(src, rois, dst, mask, kernel, stride, pad_l, pad_t, pad_r, pad_b,  avePoolPaddedArea, type, spatialScale, computeMaxIdx, nstripes);
    }

    void avePooling(Mat &src, Mat &dst)
    {
        const int nstripes = getNumThreads();
        Mat rois, mask;
        PoolingInvoker::run(src, rois, dst, mask, kernel, stride, pad_l, pad_t, pad_r, pad_b, avePoolPaddedArea, type, spatialScale, computeMaxIdx, nstripes);
    }

    void roiPooling(const Mat &src, const Mat &rois, Mat &dst)
    {
        const int nstripes = getNumThreads();
        Mat mask;
        PoolingInvoker::run(src, rois, dst, mask, kernel, stride, pad_l, pad_t, pad_r, pad_b, avePoolPaddedArea, type, spatialScale, computeMaxIdx, nstripes);
    }

    virtual Ptr<BackendNode> initMaxPoolingHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);
        const int inWidth = inputBuffer.width();
        const int inHeight = inputBuffer.height();

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::RDom r(0, kernel.width, 0, kernel.height);
        Halide::Expr kx, ky;
        if(pad_l || pad_t)
        {
            kx = clamp(x * stride.width + r.x - pad_l, 0, inWidth - 1);
            ky = clamp(y * stride.height + r.y - pad_t, 0, inHeight - 1);
        }
        else
        {
            kx = min(x * stride.width + r.x, inWidth - 1);
            ky = min(y * stride.height + r.y, inHeight - 1);
        }

        // Halide::argmax returns tuple (r.x, r.y, max).
        Halide::Tuple res = argmax(inputBuffer(kx, ky, c, n));

        // Compute offset from argmax in range [0, kernel_size).
        Halide::Expr max_index;
        if(pad_l || pad_t)
        {
            max_index = clamp(y * stride.height + res[1] - pad_t,
                              0, inHeight - 1) * inWidth +
                        clamp(x * stride.width + res[0] - pad_l,
                              0, inWidth - 1);
        }
        else
        {
            max_index = min(y * stride.height + res[1], inHeight - 1) * inWidth +
                        min(x * stride.width + res[0], inWidth - 1);
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
        if ((inW - kernel.width) % stride.width || (inH - kernel.height) % stride.height)
        {
            CV_Error(cv::Error::StsNotImplemented,
                     "Halide backend for average pooling with partial "
                     "kernels is not implemented");
        }

        const float norm = 1.0f / (kernel.width * kernel.height);

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::RDom r(0, kernel.width, 0, kernel.height);
        top(x, y, c, n) = sum(
            inputBuffer(x * stride.width + r.x,
                        y * stride.height + r.y, c, n)) * norm;
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
        Size in(inputs[0][3], inputs[0][2]), out;

        if (globalPooling)
        {
            out.height = 1;
            out.width = 1;
        }
        else if (type == ROI || type == PSROI)
        {
            out.height = pooledSize.height;
            out.width = pooledSize.width;
        }
        else if (padMode.empty())
        {
            float height = (float)(in.height + pad_t + pad_b - kernel.height) / stride.height;
            float width = (float)(in.width + pad_l + pad_r - kernel.width) / stride.width;
            out.height = 1 + (ceilMode ? ceil(height) : floor(height));
            out.width = 1 + (ceilMode ? ceil(width) : floor(width));

            if (pad_r || pad_b)
            {
                // If we have padding, ensure that the last pooling starts strictly
                // inside the image (instead of at the padding); otherwise clip the last.
                if ((out.height - 1) * stride.height >= in.height + pad_b)
                    --out.height;
                if ((out.width - 1) * stride.width >= in.width + pad_r)
                    --out.width;
                CV_Assert((out.height - 1) * stride.height < in.height + pad_b);
                CV_Assert((out.width - 1) * stride.width < in.width + pad_r);
            }
        }
        else
        {
            getConvPoolOutParams(in, kernel, stride, padMode, Size(1, 1), out);
        }

        int dims[] = {inputs[0][0], inputs[0][1], out.height, out.width};
        if (type == ROI)
        {
            CV_Assert(inputs.size() == 2);
            dims[0] = inputs[1][0];  // Number of proposals;
        }
        else if (type == PSROI)
        {
            CV_Assert(inputs.size() == 2);
            CV_Assert(psRoiOutChannels * pooledSize.width * pooledSize.height == inputs[0][1]);
            dims[0] = inputs[1][0];  // Number of proposals;
            dims[1] = psRoiOutChannels;
        }

        int numOutputs = requiredOutputs ? requiredOutputs : (type == MAX ? 2 : 1);
        CV_Assert(numOutputs == 1 || (numOutputs == 2 && type == MAX));
        outputs.assign(numOutputs, shape(dims, 4));

        return false;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(inputs); // suppress unused variable warning
        long flops = 0;

        for(int i = 0; i < outputs.size(); i++)
        {
            if (type == MAX)
            {
                if (i%2 == 0)
                    flops += total(outputs[i])*kernel.area();
            }
            else
            {
                flops += total(outputs[i])*(kernel.area() + 1);
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
        ROI,   // RoI pooling, https://arxiv.org/pdf/1504.08083.pdf
        PSROI  // Position-sensitive RoI pooling, https://arxiv.org/pdf/1605.06409.pdf
    };
};

Ptr<PoolingLayer> PoolingLayer::create(const LayerParams& params)
{
    return Ptr<PoolingLayer>(new PoolingLayerImpl(params));
}

}
}
