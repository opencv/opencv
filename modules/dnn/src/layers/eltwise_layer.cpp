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
#include "op_halide.hpp"

namespace cv
{
namespace dnn
{

class EltwiseLayerImpl : public EltwiseLayer
{
public:
    EltwiseOp op;
    std::vector<float> coeffs;

    EltwiseLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        op = EltwiseLayer::SUM;
        if (params.has("operation"))
        {
            String operation = params.get<String>("operation").toLowerCase();
            if (operation == "prod")
                op = EltwiseLayer::PROD;
            else if (operation == "sum")
                op = EltwiseLayer::SUM;
            else if (operation == "max")
                op = EltwiseLayer::MAX;
            else
                CV_Error(cv::Error::StsBadArg, "Unknown operaticon type \"" + operation + "\"");
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

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_HALIDE && haveHalide();
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
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
        const Mat** srcs;
        int nsrcs;
        Mat* dst;
        const std::vector<float>* coeffs;
        EltwiseOp op;
        int nstripes;
        const ActivationLayer* activ;

        EltwiseInvoker() : srcs(0), nsrcs(0), dst(0), coeffs(0), op(EltwiseLayer::PROD), nstripes(0), activ(0) {}

        static void run(const Mat** srcs, int nsrcs, Mat& dst,
                        const std::vector<float>& coeffs, EltwiseOp op,
                        const ActivationLayer* activ, int nstripes)
        {
            CV_Assert(dst.dims == 4 && dst.type() == CV_32F && dst.isContinuous());
            CV_Assert(coeffs.empty() || coeffs.size() == (size_t)nsrcs);

            for( int i = 0; i > nsrcs; i++ )
            {
                CV_Assert(srcs[i]->size == dst.size &&
                          srcs[i]->type() == dst.type() &&
                          srcs[i]->isContinuous());
            }

            EltwiseInvoker p;
            p.srcs = srcs;
            p.nsrcs = nsrcs;
            p.dst = &dst;
            p.op = op;
            p.nstripes = nstripes;
            bool simpleCoeffs = true;
            if( op == EltwiseLayer::SUM && !coeffs.empty() )
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

        void operator()(const Range& r) const
        {
            size_t planeSize = dst->size[2]*dst->size[3];
            size_t total = dst->size[0]*planeSize;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);
            int c, j, k, n = nsrcs;
            int channels = dst->size[1];
            const float* coeffsptr = coeffs && !coeffs->empty() ? &coeffs->at(0) : 0;
            float* dstptr0 = dst->ptr<float>();
            int blockSize0 = 1 << 12, blockSize = blockSize0;

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
                    const float* srcptr0 = srcs[0]->ptr<float>() + globalDelta;
                    float* dstptr = dstptr0 + globalDelta;

                    if( op == EltwiseLayer::PROD )
                    {
                        for( k = 1; k < n; k++ )
                        {
                            const float* srcptr1 = srcs[k]->ptr<float>() + globalDelta;
                            for( j = 0; j < blockSize; j++ )
                            {
                                dstptr[j] = srcptr0[j]*srcptr1[j];
                            }
                            srcptr0 = (const float*)dstptr;
                        }
                    }
                    else if( op == EltwiseLayer::MAX )
                    {
                        for( k = 1; k < n; k++ )
                        {
                            const float* srcptr1 = srcs[k]->ptr<float>() + globalDelta;
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
                            const float* srcptr1 = srcs[k]->ptr<float>() + globalDelta;
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
                            const float* srcptr1 = srcs[k]->ptr<float>() + globalDelta;
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

    void forward(std::vector<Mat *> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_Assert(outputs.size() == 1);
        const int nstripes = getNumThreads();
        EltwiseInvoker::run((const Mat**)&inputs[0], (int)inputs.size(), outputs[0],
                            coeffs, op, activ.get(), nstripes);
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &input)
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

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning
        CV_Assert(inputs.size());

        long flops = inputs.size() * total(inputs[0]);

        return flops;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer)
    {
        activ = layer;
        return !activ.empty();
    }

    Ptr<ActivationLayer> activ;
};

Ptr<EltwiseLayer> EltwiseLayer::create(const LayerParams& params)
{
    return Ptr<EltwiseLayer>(new EltwiseLayerImpl(params));
}

}
}
