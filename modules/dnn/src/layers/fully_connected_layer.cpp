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
#include "../op_webnn.hpp"
#include "../op_cann.hpp"
#include "../op_vkcom.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
using namespace cv::dnn::ocl4dnn;
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/matmul.hpp"
#include "../cuda4dnn/primitives/inner_product.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class FullyConnectedLayerImpl CV_FINAL : public InnerProductLayer
{
public:
    enum { VEC_ALIGN = 8 };

#ifdef HAVE_OPENCL
    Ptr<OCL4DNNInnerProduct<float> > innerProductOp;
    std::vector<UMat> umat_blobs;
    std::vector<UMat> half_blobs;
#endif

    FullyConnectedLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        transA = params.get<bool>("transA", false);
        transB = params.get<bool>("transB", false);

        bias = params.get<bool>("bias_term", true);
        axis = params.get<int>("axis", 1);
        isMatMul = params.get<bool>("is_matmul", false);
        if (!blobs.empty())
        {
            CV_Assert(1 <= blobs.size() && blobs.size() <= 2);
            int numOutput = params.get<int>("num_output");
            int innerSize = (int)blobs[0].total() / numOutput;

            CV_Assert(blobs[0].dims >= 2 && (size_t)(innerSize * numOutput) == blobs[0].total());
            CV_Assert(!bias || (blobs.size() == 2 && (size_t)numOutput == blobs[1].total()));

            blobs[0].copyTo(oriMat);
            weightsMat = blobs[0] = blobs[0].reshape(1, numOutput);
            int vecsize = weightsMat.cols;
            if (vecsize % VEC_ALIGN != 0)
            {
                int vecsize_aligned = (int)alignSize(vecsize, VEC_ALIGN);
                Mat weightsBuf(weightsMat.rows, vecsize_aligned, weightsMat.type());
                Mat wpadding = weightsBuf.colRange(vecsize, vecsize_aligned);
                wpadding.setTo(Scalar::all(0.));
                weightsMat = weightsBuf.colRange(0, vecsize);
                blobs[0].copyTo(weightsMat);
            }

            if (bias)
                biasMat = blobs[1] = blobs[1].reshape(1, 1);
            else if(isMatMul)
                biasMat = Mat::zeros(1, oriMat.size[oriMat.dims - 2], weightsMat.type());
            else
                biasMat = Mat::zeros(1, numOutput, weightsMat.type());

            transB = !transB;
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &) const CV_OVERRIDE
    {
        int numOutput, cAxis;

        std::vector<MatShape> inputsTmp;
        inputsTmp.assign(inputs.begin(), inputs.end());

        if (blobs.empty())
        {
            CV_CheckEQ(inputsTmp.size(), (size_t)2, "");

            if (transA)
            {
                CV_CheckEQ(inputsTmp[0].size(), (size_t)2, "");
                std::swap(inputsTmp[0][0], inputsTmp[0][1]);
            }

            if (transB)
            {
                CV_CheckEQ(inputsTmp[1].size(), (size_t)2, "");
                std::swap(inputsTmp[1][0], inputsTmp[1][1]);
            }

            numOutput = inputsTmp[1].back();
            cAxis = inputsTmp[0].size() - 1;
            int dims = inputsTmp[0].size();
            CV_CheckEQ(inputsTmp[1].size(), (size_t)dims, "");
            CV_CheckGE(dims, 2, "");
            for (int i = 0; i < dims - 2; i++)
                CV_CheckEQ(inputsTmp[0][i], inputsTmp[1][i], "");
            CV_CheckEQ(inputsTmp[0].back(), inputsTmp[1][dims - 2], "");
        }
        else
        {
            CV_CheckEQ(inputsTmp.size(), (size_t)1, "");
            CV_CheckEQ(blobs[0].dims, 2, "");
            if(isMatMul)
                numOutput = oriMat.size[oriMat.dims - 2];
            else
                numOutput = blobs[0].size[0];
            CV_Assert(!bias || (size_t)numOutput == blobs[1].total());
            cAxis = normalize_axis(axis, inputsTmp[0]);
        }

        MatShape outShape(cAxis + 1);
        for (int i = 0; i < cAxis; ++i)
            outShape[i] = inputsTmp[0][i];
        outShape.back() = numOutput;

        outputs.resize(1, outShape);
        return false;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        bool tranAorB = transA || transB;
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return axis == 1 && !tranAorB;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               (backendId == DNN_BACKEND_HALIDE && haveHalide() && axis == 1 && !tranAorB) ||
               (backendId == DNN_BACKEND_WEBNN && axis == 1 && !tranAorB) ||
               backendId == DNN_BACKEND_CANN ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan() && !tranAorB);
    }

    virtual bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        if (activ.empty() || layer.empty())
        {
            activ = layer;
            return !activ.empty();
        }
        else
            return false;
    }

    class FullyConnected : public ParallelLoopBody
    {
    public:
        FullyConnected() : srcMat(0), weights(0), biasMat(0), activ(0), dstMat(0), nstripes(0), useAVX(false), useAVX2(false), useAVX512(false), useRVV(false), useLASX(false) {}

        static void run(const Mat& srcMat, const Mat& weights, const Mat& biasMat,
                        Mat& dstMat, const ActivationLayer* activ, int nstripes)
        {
            CV_Assert( srcMat.dims == 2 && srcMat.cols == weights.cols &&
                       dstMat.rows == srcMat.rows && dstMat.cols == weights.rows &&
                       srcMat.type() == weights.type() && weights.type() == dstMat.type() &&
                       srcMat.type() == CV_32F &&
                       (biasMat.empty() || (biasMat.type() == srcMat.type() &&
                                           biasMat.isContinuous() && (int)biasMat.total() == dstMat.cols)) );

            FullyConnected p;

            p.srcMat = &srcMat;
            p.weights = &weights;
            p.biasMat = &biasMat;
            p.dstMat = &dstMat;
            p.nstripes = nstripes;
            p.activ = activ;
            p.useAVX = checkHardwareSupport(CPU_AVX);
            p.useAVX2 = checkHardwareSupport(CPU_AVX2);
            p.useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX;
            p.useRVV = checkHardwareSupport(CPU_RVV);
            p.useLASX = checkHardwareSupport(CPU_LASX);

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            int valign = FullyConnectedLayerImpl::VEC_ALIGN;
            int nsamples = srcMat->rows;
            int nw0 = weights->rows;
            int k, vecsize = srcMat->cols;
            int vecsize_aligned = (int)alignSize(vecsize, VEC_ALIGN);
            size_t total = (size_t)nsamples*nw0;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = r.end == nstripes ? total : std::min(r.end*stripeSize, total);
            size_t wstep = weights->step1();
            AutoBuffer<float> srcbuf(vecsize_aligned + valign);
            float* sptr = alignPtr(srcbuf.data(), (int)(valign*sizeof(float)));

            for( k = vecsize; k < vecsize_aligned; k++ )
                sptr[k] = 0.f;

            for( size_t ofs = stripeStart; ofs < stripeEnd; )
            {
                int sampleIdx = (int)(ofs / nw0);
                int delta = (int)(ofs - (size_t)sampleIdx*nw0);
                const float* sptr_ = srcMat->ptr<float>(sampleIdx);
                const float* wptr = weights->ptr<float>(delta);
                float* dptr = dstMat->ptr<float>(sampleIdx) + delta;
                const float* biasptr = biasMat->ptr<float>() + delta;
                int nw = std::min(nw0 - delta, (int)(stripeEnd - ofs));

                memcpy(sptr, sptr_, vecsize*sizeof(sptr[0]));

            #if CV_TRY_AVX512_SKX
                if( useAVX512 )
                    opt_AVX512_SKX::fastGEMM1T( sptr, wptr, wstep, biasptr, dptr, nw, vecsize_aligned);
                else
            #endif
            #if CV_TRY_AVX2
                if( useAVX2 )
                    opt_AVX2::fastGEMM1T( sptr, wptr, wstep, biasptr, dptr, nw, vecsize_aligned);
                else
            #endif
            #if CV_TRY_AVX
                if( useAVX )
                    opt_AVX::fastGEMM1T( sptr, wptr, wstep, biasptr, dptr, nw, vecsize_aligned);
                else
            #endif
            #if CV_TRY_RVV
                if( useRVV )
                    opt_RVV::fastGEMM1T( sptr, wptr, wstep, biasptr, dptr, nw, vecsize);
                else
            #endif
            #if CV_TRY_LASX
                if( useLASX )
                    opt_LASX::fastGEMM1T( sptr, wptr, wstep, biasptr, dptr, nw, vecsize);
                else
            #endif
                {
                    int i = 0;

            #if CV_SIMD128
                    for( ; i <= nw - 4; i += 4, wptr += 4*wstep )
                    {
                        v_float32x4 vs0 = v_setall_f32(0.f);
                        v_float32x4 vs1 = v_setall_f32(0.f);
                        v_float32x4 vs2 = v_setall_f32(0.f);
                        v_float32x4 vs3 = v_setall_f32(0.f);

                        for( k = 0; k < vecsize; k += 4 )
                        {
                            v_float32x4 v = v_load_aligned(sptr + k);
                            vs0 = v_fma(v, v_load_aligned(wptr + k), vs0);
                            vs1 = v_fma(v, v_load_aligned(wptr + wstep + k), vs1);
                            vs2 = v_fma(v, v_load_aligned(wptr + wstep*2 + k), vs2);
                            vs3 = v_fma(v, v_load_aligned(wptr + wstep*3 + k), vs3);
                        }

                        v_float32x4 s = v_reduce_sum4(vs0, vs1, vs2, vs3);
                        s += v_load(biasptr + i);
                        v_store(dptr + i, s);
                    }
            #endif

                    for( ; i < nw; i++, wptr += wstep )
                    {
                        float s0=biasptr[i];

                        for( k = 0; k < vecsize; k++ )
                        {
                            float v = sptr[k];
                            s0 += v*wptr[k];
                        }
                        dptr[i] = s0;
                    }
                }

                if(activ)
                    activ->forwardSlice(dptr, dptr, 1, 1, delta, delta + nw);

                ofs += nw;
            }
        }

        const Mat *srcMat, *weights, *biasMat;
        const ActivationLayer* activ;
        Mat* dstMat;
        int nstripes;
        bool useAVX;
        bool useAVX2;
        bool useAVX512;
        bool useRVV;
        bool useLASX;
    };

#ifdef HAVE_OPENCL
    virtual void finalize(InputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE
    {
        innerProductOp.release();
        umat_blobs.clear();
        half_blobs.clear();
    }

    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, InputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inps.depth() == CV_16S);
        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        if (inputs.size() == 2)
        {
            int dims = outputs[0].dims;
            int m = inputs[0].size[dims - 2];
            int n = inputs[0].size[dims - 1];
            int k = inputs[1].size[dims - 1];
            int rows = inputs[0].total() / (m * n);

            MatShape sh_A = shape(rows, m * n);
            MatShape sh_B = shape(rows, n * k);
            MatShape sh_C = shape(rows, m * k);
            UMat inp = inputs[0].reshape(1, sh_A.size(), &sh_A[0]);
            UMat weight = inputs[1].reshape(1, sh_B.size(), &sh_B[0]);
            UMat out = outputs[0].reshape(1, sh_C.size(), &sh_C[0]);

            UMat A, B, C, A_fp32, B_fp32, C_fp32;
            for (int i = 0; i < rows; ++i)
            {
                A = inp.row(i).reshape(1, m);
                B = weight.row(i).reshape(1, n);
                C = out.row(i).reshape(1, m);

                if (use_half)
                {
                    convertFp16(A, A_fp32);
                    convertFp16(B, B_fp32);
                    convertFp16(C, C_fp32);
                }
                else
                {
                    A_fp32 = A;
                    B_fp32 = B;
                    C_fp32 = C;
                }
                cv::gemm(A_fp32, B_fp32, 1, noArray(), 0, C_fp32);
                if (use_half)
                {
                    convertFp16(A_fp32, A);
                    convertFp16(B_fp32, B);
                    convertFp16(C_fp32, C);
                }
            }
            return true;
        }

        int axisCan = normalize_axis(axis, inputs[0].dims);
        int numOutput = blobs[0].size[0];
        int innerSize = blobs[0].size[1];
        int outerSize = total(shape(inputs[0]), 0, axisCan);
        bool ret = true;

        if (innerProductOp.empty())
        {
            size_t n = blobs.size();
            umat_blobs.resize(n);
            for (int i = 0; i < n; i++) blobs[i].copyTo(umat_blobs[i]);

            OCL4DNNInnerProductConfig config;
            config.num_output = numOutput;
            config.bias_term = bias;
            config.M = outerSize;
            config.K = innerSize;
            config.use_half = use_half;

            if (use_half)
            {
                half_blobs.resize(umat_blobs.size());
                for (int i = 0; i < umat_blobs.size(); i++)
                {
                    if (!umat_blobs[i].empty())
                        convertFp16(umat_blobs[i], half_blobs[i]);
                }
            }

            innerProductOp = Ptr<OCL4DNNInnerProduct<float> >(new OCL4DNNInnerProduct<float>(config));
        }

        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape inshape, outshape;
            inshape = shape(outerSize, innerSize);
            outshape = shape(outerSize, numOutput);

            UMat srcMat, dstMat;
            srcMat = inputs[i].reshape(1, inshape.size(), &inshape[0]);
            dstMat = outputs[i].reshape(1, outshape.size(), &outshape[0]);

            if (!innerProductOp->Forward(srcMat, (use_half) ? half_blobs[0] : umat_blobs[0],
                                         (bias) ? (use_half ? half_blobs[1] : umat_blobs[1]) : UMat(),
                                         dstMat))
            {
                ret = false;
                break;
            }

            if (!use_half && bias && (outerSize > 1))
            {
                UMat biasOnesMat = UMat::ones(outerSize, 1, umat_blobs[0].type());
                UMat& biases = umat_blobs[1];
                cv::gemm(biasOnesMat, biases, 1, dstMat, 1, dstMat, 0);
            }
        }

        if (ret) return true;

        UMat& weights = umat_blobs[0];
        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape inshape, outshape;
            inshape = shape(outerSize, innerSize);
            outshape = shape(outerSize, numOutput);

            UMat srcMat, dstMat, srcMat_fp32, dstMat_fp32;
            srcMat = inputs[i].reshape(1, inshape.size(), &inshape[0]);
            dstMat = outputs[i].reshape(1, outshape.size(), &outshape[0]);

            if (use_half)
            {
                convertFp16(srcMat, srcMat_fp32);
                convertFp16(dstMat, dstMat_fp32);
            }
            else
            {
                srcMat_fp32 = srcMat;
                dstMat_fp32 = dstMat;
            }

            cv::gemm(srcMat_fp32, weights, 1, noArray(), 0, dstMat_fp32, GEMM_2_T);

            if (bias)
            {
                UMat biasOnesMat = UMat::ones(outerSize, 1, umat_blobs[0].type());
                UMat& biases = umat_blobs[1];
                cv::gemm(biasOnesMat, biases, 1, dstMat_fp32, 1, dstMat_fp32, 0);
            }
            if (use_half)
            {
                convertFp16(srcMat_fp32, srcMat);
                convertFp16(dstMat_fp32, dstMat);
            }
        }

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) && !isMatMul,
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> input, output;
        inputs_arr.getMatVector(input);
        outputs_arr.getMatVector(output);

        if (!blobs.empty())
        {
            int inp1Dim = input[0].dims;
            if (isMatMul)
            {
                int matNum = input[0].total(0, inp1Dim - 2);
                int rowMatMul = oriMat.size[oriMat.dims - 2];
                Mat srcMatTmp = input[0].reshape(1, matNum);
                Mat dstMatTmp = output[0].reshape(1, matNum);

                int outerSize = input[0].size[inp1Dim - 2];
                int rowStart = -rowMatMul;
                for (int n = 0; n < matNum; ++n)
                {
                    Mat srcMat = srcMatTmp.row(n).reshape(1, outerSize);
                    Mat dstMat = dstMatTmp.row(n).reshape(1, outerSize);
                    rowStart = (rowStart + rowMatMul) % weightsMat.rows;
                    Mat weiMat = weightsMat.rowRange(rowStart, rowStart + rowMatMul);

                    const int nstripes = getNumThreads();
                    FullyConnected::run(srcMat, weiMat, biasMat, dstMat, activ.get(), nstripes);
                }
            }
            else
            {
                int axisCan = normalize_axis(axis, inp1Dim);
                int outerSize = input[0].total(0, axisCan);

                for (size_t i = 0; i < input.size(); i++)
                {
                    Mat srcMat = input[i].reshape(1, outerSize);
                    Mat dstMat = output[i].reshape(1, outerSize);

                    const int nstripes = getNumThreads();
                    FullyConnected::run(srcMat, weightsMat, biasMat, dstMat, activ.get(), nstripes);
                }
            }
        }
        else
        {
            Mat input0 = input[0];
            Mat input1 = input[1];

            if (transA)
            {
                CV_Assert(input0.dims == 2);
                input0 = input0.t();
            }

            if (transB)
            {
                CV_Assert(input1.dims == 2);
                input1 = input1.t();
            }

            float* inpData = input0.ptr<float>();
            float* weightData = input1.ptr<float>();
            float* outData = output[0].ptr<float>();

            int dims = output[0].dims;
            int numSlice = output[0].total() / output[0].total(dims - 2);
            int m = input0.size[dims - 2];
            int n = input0.size[dims - 1];
            int k = input1.size[dims - 1];
            for (int i = 0; i < numSlice; i++)
            {
                Mat inpSlice(m, n, CV_32F, inpData);
                Mat weightSlice(n, k, CV_32F, weightData);
                Mat outSlice(m, k, CV_32F, outData);

                outSlice = inpSlice * weightSlice;
                inpData += inpSlice.total();
                weightData += weightSlice.total();
                outData += outSlice.total();
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
        auto biasMat_ = bias ? biasMat : Mat();
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();

        if (weightsMat.empty() || isMatMul)
        {
            int inp2Dim;
            // broadcast is not supported with CUDA
            if(weightsMat.empty())
            {
                auto input_wrapper2 = inputs[1].dynamicCast<CUDABackendWrapper>();
                inp2Dim = input_wrapper2->getRank();
            }else
                inp2Dim = oriMat.dims;

            if(input_wrapper->getRank() == inp2Dim)
                return make_cuda_node<cuda4dnn::MatMulOp>(preferableTarget, std::move(context->stream), std::move(context->cublas_handle), oriMat, biasMat_, transA, transB);
            else
                return Ptr<BackendNode>();
        }

        auto flatten_start_axis = normalize_axis(axis, input_wrapper->getRank());
        return make_cuda_node<cuda4dnn::InnerProductOp>(preferableTarget, std::move(context->stream), std::move(context->cublas_handle), flatten_start_axis, weightsMat, biasMat_);
    }
#endif

#ifdef HAVE_VULKAN
    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs,
                                       std::vector<Ptr<BackendWrapper> > &outputs) CV_OVERRIDE
    {
        auto biasMat_ = bias ? biasMat : Mat();
        auto input_wrapper = inputs[0].dynamicCast<VkComBackendWrapper>();

        CV_Assert((inputs.size() == 2 || inputs.size() == 1) && outputs.size() == 1);
        std::vector<Mat> vkBlobs;
        Ptr<vkcom::OpBase> op;

        if (!biasMat_.empty() || !activ.empty())
        {
            return Ptr<BackendNode>();
        }

        Ptr<VkComBackendWrapper> outputWrap = outputs[0].dynamicCast<VkComBackendWrapper>();
        CV_Assert(outputWrap);
        // TODO: Currently, we only support the 2D MatMul. Need support the FC layer and bias case in the future.

        if (inputs.size() == 2)
        {
            Ptr<VkComBackendWrapper> inputWrap0 = inputs[0].dynamicCast<VkComBackendWrapper>();
            Ptr<VkComBackendWrapper> inputWrap1 = inputs[1].dynamicCast<VkComBackendWrapper>();
            CV_Assert(inputWrap0 && inputWrap1);

            MatShape inpShape0 = shape(*inputWrap0->getMat());
            MatShape inpShape1 = shape(*inputWrap1->getMat());
            MatShape outShape = shape(*outputWrap->getMat());

            // TODO Currently, vulkan only support 2D matmul. Try to support 3D and 4D matmul.
            if (inpShape0.size() != 2 || inpShape1.size() != 2)
                return Ptr<BackendNode>();

            op = (new vkcom::OpMatMul(vkBlobs, inpShape0[0], inpShape0[1], outShape[1]));
        }
        else
        {
            CV_Assert(!weightsMat.empty());
            Mat wm;
            weightsMat.copyTo(wm); // to handle the case of isContinuous() == false
            wm = wm.reshape(1, blobs[0].dims, blobs[0].size);
            vkBlobs.push_back(wm.t());

            Ptr<VkComBackendWrapper> inputWrap = inputs[0].dynamicCast<VkComBackendWrapper>();
            CV_Assert(inputWrap);

            MatShape inpShape = shape(*inputWrap->getMat());
            MatShape outShape = shape(*outputWrap->getMat());
            MatShape wShape = shape(weightsMat);

            // TODO Currently, vulkan only support 2D matmul. Try to support 3D and 4D matmul.
            if (inpShape.size() != 2 || wShape.size() != 2)
                return Ptr<BackendNode>();

            // TODO: Currently, only focus on 2D MatMul.
            CV_Assert(inpShape.size() == 2 && outShape.size() == 2 && wShape.size() == 2);
            CV_Assert(inpShape[1] == outShape[0]);
            op = (new vkcom::OpMatMul(vkBlobs, inpShape[0], inpShape[1], outShape[1]));
        }

        return Ptr<BackendNode>(new VkComBackendNode(inputs, op, outputs));
    }
#endif


    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        int inW, inH, inC, inN, outC = blobs[0].size[0];
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);
        getCanonicalSize(inputBuffer, &inW, &inH, &inC, &inN);
        auto weights = wrapToHalideBuffer(blobs[0], {inW, inH, inC, outC});

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::RDom r(0, inW, 0, inH, 0, inC);
        Halide::Expr topExpr = sum(inputBuffer(r.x, r.y, r.z, n) *
                                   weights(r.x, r.y, r.z, c));
        if (bias)
        {
            Halide::Buffer<float> bias = wrapToHalideBuffer(blobs[1], {outC});
            topExpr += bias(c);
        }
        top(x, y, c, n) = topExpr;
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto x1 = inputs[0].dynamicCast<CannBackendWrapper>();
        auto x1_desc = x1->getTensorDesc();
        auto op_x1 = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        auto op = std::make_shared<ge::op::MatMulV2>(name);

        if (!blobs.empty()) // if B is const
        {
            // set attributes
            op->set_attr_transpose_x1(false);
            // weightMat always needs to be transposed, since CPU backend
            // implementation is input * weight.im2row
            op->set_attr_transpose_x2(true);

            // set inputs
            // set inputs : x2 (weight)
            auto op_const_weight = std::make_shared<CannConstOp>(weightsMat.data, weightsMat.type(), shape(weightsMat), cv::format("%s_w", name.c_str()));
            op->set_input_x2_by_name(*(op_const_weight->getOp()), "y");
            op->update_input_desc_x2(*(op_const_weight->getTensorDesc()));
        }
        else
        {
            // A and B are variable inputs; non-const bias is not considered
            CV_Assert(inputs.size() == 2);
            CV_Assert(nodes.size() == 2);

            // set attributes
            op->set_attr_transpose_x1(transA);
            op->set_attr_transpose_x2(transB);

            // set inputs : x2 (weight)
            auto op_x2 = nodes[1].dynamicCast<CannBackendNode>()->getOp();
            auto x2_desc = inputs[1].dynamicCast<CannBackendWrapper>()->getTensorDesc();
            op->set_input_x2_by_name(*op_x2, "y");
            op->update_input_desc_x2(*x2_desc);
        }

        // set inputs
        // set inputs : x1 (input)
        op->set_input_x1_by_name(*op_x1, x1->name.c_str());
        op->update_input_desc_x1(*x1_desc);
        // set inputs : bias (bias)
        auto bias_mat = bias ? biasMat : Mat::zeros(1, weightsMat.size[0], weightsMat.type());
        std::vector<int> bias_shape{weightsMat.size[0]};
        auto op_const_bias = std::make_shared<CannConstOp>(bias_mat.data, bias_mat.type(), bias_shape, cv::format("%s_b", name.c_str()));
        op->set_input_bias(*(op_const_bias->getOp()));
        op->update_input_desc_bias(*(op_const_bias->getTensorDesc()));

        // set outputs
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::shared_ptr<ngraph::Node> matmul;

        if (nodes.size() == 2)
        {
            auto& inp2 = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
            matmul = std::make_shared<ngraph::op::MatMul>(ieInpNode, inp2, false, false);
        }
        else
        {
            std::vector<int64_t> data = {(int64_t)ieInpNode->get_shape()[0], (int64_t)blobs[0].size[1]};
            auto new_shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2}, data.data());
            auto inp = std::make_shared<ngraph::op::v1::Reshape>(ieInpNode, new_shape, true);

            std::vector<size_t> weight_shape{(size_t)blobs[0].size[0], (size_t)blobs[0].size[1]};
            auto ieWeights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, weight_shape, blobs[0].data);
            matmul = std::make_shared<ngraph::op::MatMul>(inp, ieWeights, false, true);
        }

        if (bias) {
            auto bias_node = std::make_shared<ngraph::op::Constant>(ngraph::element::f32,
                                              ngraph::Shape{(size_t)blobs[1].size[1]}, blobs[1].data);
            matmul = std::make_shared<ngraph::op::v1::Add>(matmul, bias_node, ngraph::op::AutoBroadcastType::NUMPY);
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(matmul));
    }
#endif  // HAVE_DNN_NGRAPH

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        if (blobs.empty())
            return false;

        int numOutput = blobs[0].size[0];
        float inputScale = scales[0][0], outputScale = scales[1][0];
        int inputZp = zeropoints[0][0];

        Mat weightsQuantized(weightsMat.rows, weightsMat.cols, CV_8S);
        Mat biasQuantized(1, numOutput, CV_32S);
        Mat outputMultiplier(1, numOutput, CV_32F);
        bool perChannel = params.get<bool>("per_channel", true);

        if (perChannel) // per-Channel quantization.
        {
            for (int i = 0; i < numOutput; i++)
            {
                double weightsScale = getWeightScale(weightsMat.row(i));

                weightsMat.row(i).convertTo(weightsQuantized.row(i), CV_8S, 1.f/weightsScale);
                float biasScale = inputScale * weightsScale;
                biasQuantized.at<int>(i) = cvRound(biasMat.at<float>(i)/biasScale) - inputZp*(cv::sum(weightsQuantized.row(i))[0]);
                outputMultiplier.at<float>(i) = biasScale / outputScale;
            }
        }
        else // per-Tensor quantization.
        {
            double weightsScale = getWeightScale(weightsMat);

            weightsMat.convertTo(weightsQuantized, CV_8S, 1.f/weightsScale);
            float biasScale = inputScale * weightsScale;

            for (int i = 0; i < numOutput; i++)
            {
                biasQuantized.at<int>(i) = cvRound(biasMat.at<float>(i)/biasScale) - inputZp*(cv::sum(weightsQuantized.row(i))[0]);
                outputMultiplier.at<float>(i) = biasScale / outputScale;
            }
        }

        params.blobs.clear();
        params.set("per_channel", perChannel);
        params.blobs.push_back(weightsQuantized.reshape(1, shape(blobs[0])));
        params.blobs.push_back(biasQuantized);
        params.blobs.push_back(outputMultiplier);
        params.set("input_scale", inputScale);
        params.set("input_zeropoint", inputZp);
        return true;
    }

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        ml::GemmOptions gemmOptions = {};
        if (bias)
        {
            std::vector<int32_t> biasDims = {(int32_t)blobs[1].size[1]};
            ml::Operand bias = webnn::BuildConstant(webnnGraphBuilder, biasDims, blobs[1].data, blobs[1].total()*blobs[1].elemSize(), ml::OperandType::Float32);
            gemmOptions.c = bias;
        }
        ml::Operand result = nullptr;
        if (nodes.size() == 2)
        {
            auto& inp2 = nodes[1].dynamicCast<WebnnBackendNode>()->operand;
            result = webnnGraphBuilder.Gemm(webnnInpOperand, inp2, &gemmOptions);
        }
        else
        {
            std::vector<int32_t> input_shape(2, -1);
            input_shape[1] = blobs[0].size[1];
            ml::Operand webnnInpOperand_reshaped = webnnGraphBuilder.Reshape(webnnInpOperand, input_shape.data(), input_shape.size());
            std::vector<int32_t> weight_shape = {(int32_t)blobs[0].size[0], (int32_t)blobs[0].size[1]};
            // std::cout<<"weight size: "<<weight_shape[1]<<" "<<weight_shape[0]<<std::endl;
            ml::Operand inp2 = webnn::BuildConstant(webnnGraphBuilder, weight_shape, blobs[0].data, blobs[0].total()*blobs[0].elemSize(), ml::OperandType::Float32);
            gemmOptions.bTranspose = true;
            result = webnnGraphBuilder.Gemm(webnnInpOperand_reshaped, inp2, &gemmOptions);
        }
        return Ptr<BackendNode>(new WebnnBackendNode(result));
    }
#endif // HAVE_WEBNN

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(inputs); // suppress unused variable warning
        long flops = 0;
        int innerSize = 0;

        if (!blobs.empty())
        {
            innerSize = blobs[0].size[1];
        }
        else
        {
            CV_Assert(inputs.size() == 2);
            if (transB)
                innerSize = inputs[1][1];
            else
                innerSize = inputs[1][0];
        }

        for(int i = 0; i < outputs.size(); i++)
        {
            flops += CV_BIG_INT(3)*innerSize*total(outputs[i]);
        }

        return flops;
    }

    bool bias;
    Mat weightsMat, biasMat, oriMat;
    bool transA, transB;
    bool isMatMul = false;
    Ptr<ActivationLayer> activ;
};

Ptr<InnerProductLayer> InnerProductLayer::create(const LayerParams& params)
{
    return Ptr<InnerProductLayer>(new FullyConnectedLayerImpl(params));
}

}
}
