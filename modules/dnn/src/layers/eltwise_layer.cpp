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
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

namespace cv
{
namespace dnn
{

class EltwiseLayerImpl CV_FINAL : public EltwiseLayer
{
public:
    enum EltwiseOp
    {
        PROD = 0,
        SUM = 1,
        MAX = 2,
    } op;
    std::vector<float> coeffs;

    EltwiseLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        op = SUM;
        if (params.has("operation"))
        {
            String operation = toLowerCase(params.get<String>("operation"));
            if (operation == "prod")
                op = PROD;
            else if (operation == "sum")
                op = SUM;
            else if (operation == "max")
                op = MAX;
            else
                CV_Error(cv::Error::StsBadArg, "Unknown operation type \"" + operation + "\"");
        }

        if (params.has("coeff"))
        {
            DictValue paramCoeff = params.get("coeff");
            int i, n = paramCoeff.size();
            coeffs.resize(n);
            for (i = 0; i < n; i++)
            {
                coeffs[i] = paramCoeff.get<float>(i);
            }
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_HALIDE ||
               (backendId == DNN_BACKEND_INFERENCE_ENGINE &&
                (preferableTarget != DNN_TARGET_OPENCL || coeffs.empty()));
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        CV_Assert(coeffs.size() == 0 || coeffs.size() == inputs.size());
        CV_Assert(op == SUM || coeffs.size() == 0);

        for (int i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[0] == inputs[i]);
        }

        outputs.assign(1, inputs[0]);

        return false;
    }

    class EltwiseInvoker : public ParallelLoopBody
    {
    public:
        const Mat* srcs;
        int nsrcs;
        Mat* dst;
        const std::vector<float>* coeffs;
        EltwiseOp op;
        int nstripes;
        const ActivationLayer* activ;
        int channels;
        size_t planeSize;

        EltwiseInvoker() : srcs(0), nsrcs(0), dst(0), coeffs(0), op(PROD), nstripes(0), activ(0), channels(0), planeSize(0)  {}

        static void run(const Mat* srcs, int nsrcs, Mat& dst,
                        const std::vector<float>& coeffs, EltwiseOp op,
                        const ActivationLayer* activ, int nstripes)
        {
            CV_Check(dst.dims, 1 < dst.dims && dst.dims <= 4, ""); CV_CheckTypeEQ(dst.type(), CV_32FC1, ""); CV_Assert(dst.isContinuous());
            CV_Assert(coeffs.empty() || coeffs.size() == (size_t)nsrcs);

            for( int i = 0; i > nsrcs; i++ )
            {
                CV_Assert(srcs[i].size == dst.size &&
                          srcs[i].type() == dst.type() &&
                          srcs[i].isContinuous());
            }

            EltwiseInvoker p;
            p.srcs = srcs;
            p.nsrcs = nsrcs;
            p.dst = &dst;
            p.op = op;
            p.nstripes = nstripes;
            p.channels = (dst.dims == 4 ? dst.size[1] : 1);
            p.planeSize = (dst.dims >= 3 ? dst.size[dst.dims - 1] * dst.size[dst.dims - 2] :
                                           dst.size[dst.dims - 1]);
            CV_Assert(dst.total() == dst.size[0] * p.channels * p.planeSize);

            bool simpleCoeffs = true;
            if( op == SUM && !coeffs.empty() )
            {
                CV_Assert( coeffs.size() == (size_t)nsrcs );

                for( size_t i = 0; i < coeffs.size(); i++ )
                    if( coeffs[i] != 1 )
                    {
                        simpleCoeffs = false;
                        break;
                    }
            }
            p.coeffs = simpleCoeffs ? 0 : &coeffs;
            p.activ = activ;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            size_t total = dst->size[0]*planeSize;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);
            int c, j, k, n = nsrcs;
            const float* coeffsptr = coeffs && !coeffs->empty() ? &coeffs->at(0) : 0;
            float* dstptr0 = dst->ptr<float>();
            int blockSize0 = 1 << 12, blockSize;

            for( size_t ofs = stripeStart; ofs < stripeEnd; ofs += blockSize )
            {
                int sampleIdx = (int)(ofs / planeSize);
                int delta = (int)ofs - sampleIdx * planeSize;
                blockSize = std::min(blockSize0, std::min((int)(stripeEnd - ofs), (int)planeSize - delta));
                if( blockSize <= 0 )
                    break;

                for( c = 0; c < channels; c++ )
                {
                    size_t globalDelta = delta + (sampleIdx*channels + c)*planeSize;
                    const float* srcptr0 = srcs[0].ptr<float>() + globalDelta;
                    float* dstptr = dstptr0 + globalDelta;

                    if( op == PROD )
                    {
                        for( k = 1; k < n; k++ )
                        {
                            const float* srcptr1 = srcs[k].ptr<float>() + globalDelta;
                            for( j = 0; j < blockSize; j++ )
                            {
                                dstptr[j] = srcptr0[j]*srcptr1[j];
                            }
                            srcptr0 = (const float*)dstptr;
                        }
                    }
                    else if( op == MAX )
                    {
                        for( k = 1; k < n; k++ )
                        {
                            const float* srcptr1 = srcs[k].ptr<float>() + globalDelta;
                            for( j = 0; j < blockSize; j++ )
                            {
                                dstptr[j] = std::max(srcptr0[j], srcptr1[j]);
                            }
                            srcptr0 = (const float*)dstptr;
                        }
                    }
                    else if( !coeffsptr )
                    {
                        for( k = 1; k < n; k++ )
                        {
                            const float* srcptr1 = srcs[k].ptr<float>() + globalDelta;
                            for( j = 0; j < blockSize; j++ )
                            {
                                dstptr[j] = srcptr0[j] + srcptr1[j];
                            }
                            srcptr0 = (const float*)dstptr;
                        }
                    }
                    else
                    {
                        float c0 = coeffsptr[0];
                        for( k = 1; k < n; k++ )
                        {
                            const float* srcptr1 = srcs[k].ptr<float>() + globalDelta;
                            float c1 = coeffsptr[k];
                            for( j = 0; j < blockSize; j++ )
                            {
                                dstptr[j] = c0*srcptr0[j] + c1*srcptr1[j];
                            }
                            srcptr0 = (const float*)dstptr;
                            c0 = 1;
                        }
                    }
                }

                if( activ )
                {
                    float* ptr = dstptr0 + delta + sampleIdx*channels*planeSize;
                    activ->forwardSlice(ptr, ptr, blockSize, planeSize, 0, channels);
                }
            }
        }
    };

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        if (inputs_.depth() == CV_16S && op != SUM)
            return false;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        switch (op)
        {
            case SUM:
                {
                    int channels = total(shape(outputs[0]), 0, 2);
                    int plane_size = total(shape(outputs[0]), 2);
                    if (channels % 4 == 0 && plane_size % 4 == 0)
                    {
                        size_t localsize[] = { 128 };
                        size_t globalsize[] = { (size_t)channels / 4 * localsize[0] };
                        String opts;
                        if (inputs_.depth() == CV_16S)
                            opts = " -DDtype=half -DDtype4=half4 -DDtype8=half8";
                        else
                            opts = " -DDtype=float -DDtype4=float4 -DDtype8=float8";

                        for (int i = 0; i < (inputs.size() - 1); ++i)
                        {
                            String buildopt = format("-DLOOP=%d", i) + opts;
                            ocl::Kernel kernel("op_sum4", ocl::dnn::eltwise_oclsrc, buildopt);
                            int idx = 0;
                            UMat inpMat = (i == 0) ? inputs[0] : UMat();
                            float coeff1 = (coeffs.empty() || i > 0) ? 1.0f : coeffs[i];
                            float coeff2 = coeffs.empty() ? 1.0f : coeffs[i + 1];
                            kernel.set(idx++, ocl::KernelArg::PtrReadOnly(inputs[0]));
                            kernel.set(idx++, ocl::KernelArg::PtrReadOnly(inputs[1]));
                            kernel.set(idx++, (int)plane_size);
                            kernel.set(idx++, (float)coeff1);
                            kernel.set(idx++, (float)coeff2);
                            kernel.set(idx++, ocl::KernelArg::PtrReadWrite(outputs[0]));
                            bool ret = kernel.run(1, globalsize, localsize, false);
                            if (!ret)
                                return false;
                        }
                    }
                    else
                    {
                        if (inputs_.depth() == CV_16S)
                            return false;

                        float coeff1 = coeffs.empty() ? 1.f : coeffs[0];
                        float coeff2 = coeffs.empty() ? 1.f : coeffs[1];
                        UMat mul0, mul1;
                        multiply(coeff1, inputs[0], mul0);
                        multiply(coeff2, inputs[1], mul1);
                        add(mul0, mul1, outputs[0]);
                        for (int i = 2; i < inputs.size(); ++i)
                        {
                            float coeff = coeffs.empty() ? 1.f : coeffs[i];
                            multiply(coeff, inputs[i], mul0);
                            add(mul0, outputs[0], outputs[0]);
                        }
                    }
                }
                break;
            case PROD:
                multiply(inputs[0], inputs[1], outputs[0]);
                for (int i = 2; i < inputs.size(); ++i)
                    multiply(inputs[i], outputs[0], outputs[0]);
                break;
            case MAX:
                max(inputs[0], inputs[1], outputs[0]);
                for (int i = 2; i < inputs.size(); ++i)
                    max(inputs[i], outputs[0], outputs[0]);
                break;
            default:
                return false;
        }
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

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

        CV_Assert(outputs.size() == 1);
        const int nstripes = getNumThreads();
        EltwiseInvoker::run(&inputs[0], (int)inputs.size(), outputs[0],
                            coeffs, op, activ.get(), nstripes);
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &input) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Expr topExpr;
        std::vector<Halide::Buffer<> > inputBuffers = halideBuffers(input);
        switch (op)
        {
            case SUM:
                if (coeffs.empty())
                {
                    topExpr = inputBuffers[0](x, y, c, n) +
                              inputBuffers[1](x, y, c, n);
                    for (int i = 2; i < inputBuffers.size(); ++i)
                        topExpr += inputBuffers[i](x, y, c, n);
                }
                else
                {
                  topExpr = coeffs[0] * inputBuffers[0](x, y, c, n) +
                            coeffs[1] * inputBuffers[1](x, y, c, n);
                  for (int i = 2; i < inputBuffers.size(); ++i)
                      topExpr += coeffs[i] * inputBuffers[i](x, y, c, n);
                }
                break;
            case PROD:
                topExpr = inputBuffers[0](x, y, c, n) *
                          inputBuffers[1](x, y, c, n);
                for (int i = 2; i < inputBuffers.size(); ++i)
                    topExpr *= inputBuffers[i](x, y, c, n);
                break;
            case MAX:
                topExpr = max(inputBuffers[0](x, y, c, n),
                              inputBuffers[1](x, y, c, n));
                for (int i = 2; i < inputBuffers.size(); ++i)
                    topExpr = max(topExpr, inputBuffers[i](x, y, c, n));
                break;
            default:
                return Ptr<BackendNode>();
        }
        top(x, y, c, n) = topExpr;
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
        InferenceEngine::Builder::EltwiseLayer ieLayer(name);

        ieLayer.setInputPorts(std::vector<InferenceEngine::Port>(inputs.size()));

        if (op == SUM)
            ieLayer.setEltwiseType(InferenceEngine::Builder::EltwiseLayer::EltwiseType::SUM);
        else if (op == PROD)
            ieLayer.setEltwiseType(InferenceEngine::Builder::EltwiseLayer::EltwiseType::MUL);
        else if (op == MAX)
            ieLayer.setEltwiseType(InferenceEngine::Builder::EltwiseLayer::EltwiseType::MAX);
        else
            CV_Error(Error::StsNotImplemented, "Unsupported eltwise operation");

        InferenceEngine::Builder::Layer l = ieLayer;
        if (!coeffs.empty())
            l.getParameters()["coeff"] = coeffs;

        return Ptr<BackendNode>(new InfEngineBackendNode(l));
#else
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.type = "Eltwise";
        lp.precision = InferenceEngine::Precision::FP32;
        std::shared_ptr<InferenceEngine::EltwiseLayer> ieLayer(new InferenceEngine::EltwiseLayer(lp));
        ieLayer->coeff = coeffs;
        if (op == SUM)
            ieLayer->_operation = InferenceEngine::EltwiseLayer::Sum;
        else if (op == PROD)
            ieLayer->_operation = InferenceEngine::EltwiseLayer::Prod;
        else if (op == MAX)
            ieLayer->_operation = InferenceEngine::EltwiseLayer::Max;
        else
            CV_Error(Error::StsNotImplemented, "Unsupported eltwise operation");
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        CV_Assert(inputs.size());

        long flops = inputs.size() * total(inputs[0]);

        return flops;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        if (activ.empty() || layer.empty())
        {
            activ = layer;
            return !activ.empty();
        }
        else
            return false;
    }

    Ptr<ActivationLayer> activ;
};

Ptr<EltwiseLayer> EltwiseLayer::create(const LayerParams& params)
{
    return Ptr<EltwiseLayer>(new EltwiseLayerImpl(params));
}

}
}
