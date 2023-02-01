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
#include "../op_cann.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/eltwise.hpp"
#include "../cuda4dnn/primitives/shortcut.hpp"
using namespace cv::dnn::cuda4dnn;
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
        DIV = 3,
        MIN = 4,
    } op;
    std::vector<float> coeffs;

    enum OutputChannelsMode
    {
        ELTWISE_CHANNNELS_SAME = 0,              //!< number of channels from inputs must be the same and equal to output's number of channels
        ELTWISE_CHANNNELS_INPUT_0,               //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to number of channels of first input
                                                 //!< number of channels of other inputs should not be greater than number of channels of first input
        ELTWISE_CHANNNELS_INPUT_0_TRUNCATE,      //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to number of channels of first input
                                                 //!< there is restriction on number of channels of other inputs
                                                 //!< extra channels of other inputs is ignored
        ELTWISE_CHANNNELS_USE_MAX,               //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to maximal number of input channels
                                                 //!< @note supported operation: `SUM`
    } channelsModeInput;


    mutable OutputChannelsMode channelsMode;     //!< "optimized" channels mode (switch to ELTWISE_CHANNNELS_SAME if number of input channels are equal)
    mutable /*size_t*/int outputChannels;

    EltwiseLayerImpl(const LayerParams& params)
        : outputChannels(0)
    {
        setParamsFrom(params);
        hasVecInput = false;
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
            else if (operation == "min")
                op = MIN;
            else if (operation == "div")
                op = DIV;
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

        channelsModeInput = ELTWISE_CHANNNELS_SAME;
        if (params.has("output_channels_mode"))
        {
            String v = toLowerCase(params.get<String>("output_channels_mode"));
            if (v == "same")
            {
                channelsModeInput = ELTWISE_CHANNNELS_SAME;
            }
            else if (v == "input_0")
            {
                channelsModeInput = ELTWISE_CHANNNELS_INPUT_0;
            }
            else if (v == "input_0_truncate")
            {
                channelsModeInput = ELTWISE_CHANNNELS_INPUT_0_TRUNCATE;
            }
            else if (v == "max_input_channels")
            {
                channelsModeInput = ELTWISE_CHANNNELS_USE_MAX;
                if (op != SUM)
                    CV_Error(cv::Error::StsBadArg, "[" + type + "]:(" + name + ") 'max' channels mode is limited to SUM operation only");
            }
            else
                CV_Error(cv::Error::StsBadArg, "[" + type + "]:(" + name + ") unknown channels mode: \"" + v + "\"");
        }
        channelsMode = channelsModeInput;

        // TODO Must have checks for other unknown options
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (hasVecInput && ELTWISE_CHANNNELS_SAME)
            return backendId == DNN_BACKEND_OPENCV;

#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return channelsMode == ELTWISE_CHANNNELS_SAME;
#endif

#ifdef HAVE_CANN
        if (backendId == DNN_BACKEND_CANN)
            return channelsMode == ELTWISE_CHANNNELS_SAME && coeffs.empty();
#endif

        if (backendId == DNN_BACKEND_CUDA)
        {
            if(channelsModeInput == ELTWISE_CHANNNELS_INPUT_0 || channelsModeInput == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE)
                return op == SUM && coeffs.empty();
            return channelsModeInput == ELTWISE_CHANNNELS_SAME;
        }

        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_HALIDE && op != DIV)  // TODO: not implemented, see PR #15811
               ;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        CV_Assert(inputs[0].size() >= 2);
        CV_Assert(coeffs.size() == 0 || coeffs.size() == inputs.size());
        CV_Assert(op == SUM || coeffs.size() == 0);

        int dims = inputs[0].size();
        // Number of channels in output shape is determined by the first input tensor.
        bool variableChannels = false;
        int numChannels = inputs[0][1];
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[0][0] == inputs[i][0]);  // batch sizes are equal

            int input_channels = inputs[i][1];
            if (numChannels != input_channels)
                variableChannels = true;

            if (channelsModeInput == ELTWISE_CHANNNELS_SAME)
            {
                CV_Assert(numChannels == input_channels);
            }
            else if (channelsModeInput == ELTWISE_CHANNNELS_INPUT_0)
            {
                CV_Assert(numChannels >= input_channels);
            }
            else if (channelsModeInput == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE)
            {
                // nothing to check
            }
            else if (channelsModeInput == ELTWISE_CHANNNELS_USE_MAX)
            {
                numChannels = std::max(numChannels, input_channels);
            }
            else
            {
                CV_Assert(0 && "Internal error");
            }
        }

        channelsMode = variableChannels ? channelsModeInput : ELTWISE_CHANNNELS_SAME;
        outputChannels = numChannels;

        outputs.assign(1, inputs[0]);
        outputs[0][1] = numChannels;

        if (dims > 2)
        {
            size_t vecIdx = 0;
            bool isVecFound = false;
            for (size_t i = 0; i < inputs.size(); i++)
            {
                bool allOnes = isAllOnes(inputs[i], 2, dims);
                if (!allOnes && !isVecFound)
                {
                    vecIdx = i;
                    isVecFound = true;
                }

                if (!allOnes && i != vecIdx)
                {
                    for (size_t j = 2; j < dims; j++)
                    {
                         CV_Assert(inputs[vecIdx][j] == inputs[i][j]);
                    }
                }
            }

            if (channelsModeInput == ELTWISE_CHANNNELS_SAME && isVecFound)
            {
                for (size_t j = 2; j < dims; j++)
                {
                    outputs[0][j] = inputs[vecIdx][j];
                }
            }
        }

        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape inpShape = shape(inputs[i].size);
            if (isAllOnes(inpShape, 2, inputs[i].dims))
            {
                hasVecInput = true;
                return;
            }
        }
    }

    class EltwiseInvoker : public ParallelLoopBody
    {
        EltwiseLayerImpl& self;
        std::vector<const Mat*> srcs;
        std::vector<int> srcNumChannels;
        int nsrcs;
        Mat* dst;
        std::vector<float> coeffs;
        int nstripes;
        const ActivationLayer* activ;
        int channels;
        size_t planeSize;

        EltwiseInvoker(EltwiseLayerImpl& self_)
            : self(self_)
            , nsrcs(0), dst(0), nstripes(0), activ(0), channels(0)
            , planeSize(0)
        {}

    public:
        static void run(EltwiseLayerImpl& self,
                        const Mat* srcs, int nsrcs, Mat& dst,
                        int nstripes)
        {
            const EltwiseOp op = self.op;
            CV_Check(dst.dims, 1 < dst.dims && dst.dims <= 5, ""); CV_CheckTypeEQ(dst.type(), CV_32FC1, ""); CV_Assert(dst.isContinuous());
            CV_Assert(self.coeffs.empty() || self.coeffs.size() == (size_t)nsrcs);
            CV_CheckGE(nsrcs, 2, "");

            CV_Assert(self.outputChannels == dst.size[1]);

            EltwiseInvoker p(self);
            p.srcs.resize(nsrcs);
            p.srcNumChannels.resize(nsrcs);
            p.coeffs = self.coeffs;  // can be sorted

            bool sortInputs = false;
            for( int i = 0; i < nsrcs; i++ )
            {
                p.srcs[i] = &srcs[i];
                CV_CheckEQ(srcs[i].dims, dst.dims, "");
                CV_Assert(srcs[i].isContinuous());
                CV_Assert(srcs[i].type() == dst.type());
                p.srcNumChannels[i] = (srcs[i].dims >= 4) ? srcs[i].size[1] : 1;

                if (self.channelsMode == ELTWISE_CHANNNELS_SAME)
                {
                    CV_Assert(srcs[i].size == dst.size);
                }
                else if (self.channelsMode == ELTWISE_CHANNNELS_INPUT_0)
                {
                    if (i == 0)
                        CV_Assert(srcs[0].size == dst.size);
                    CV_Assert(self.outputChannels >= p.srcNumChannels[i]);
                    sortInputs = true;
                }
                else if (self.channelsMode == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE)
                {
                    if (i == 0)
                        CV_Assert(srcs[0].size == dst.size);
                    sortInputs = true;
                }
                else if (self.channelsMode == ELTWISE_CHANNNELS_USE_MAX)
                {
                    CV_Assert(op == SUM);
                    CV_Assert(self.outputChannels >= p.srcNumChannels[i]);
                    sortInputs = true;
                }
                else
                {
                    CV_Assert(0 && "Internal error");
                }

                if (sortInputs)
                {
                    // Sort srcs and coefficients in the desc order by number of channels
                    for (int j = i; j >= 1; j--)
                    {
                        if (std::min(self.outputChannels, p.srcs[j - 1]->size[1]) < std::min(self.outputChannels, p.srcs[j]->size[1]))
                        {
                            std::swap(p.srcs[j - 1], p.srcs[j]);
                            std::swap(p.srcNumChannels[j - 1], p.srcNumChannels[j]);
                            if (!p.coeffs.empty())
                                std::swap(p.coeffs[j - 1], p.coeffs[j]);
                        }
                        else
                            break;
                    }
                }
            }

            p.nsrcs = nsrcs;
            p.dst = &dst;
            p.nstripes = nstripes;
            p.channels = (dst.dims >= 4 ? dst.size[1] : 1);

            p.planeSize = dst.total(dst.dims >= 4 ? 2 : 1);
            CV_CheckEQ(dst.total(), dst.size[0] * p.channels * p.planeSize, "");

            bool simpleCoeffs = true;
            if (op == SUM && !p.coeffs.empty())
            {
                CV_CheckEQ(p.coeffs.size(), (size_t)nsrcs, "");

                for (size_t i = 0; i < p.coeffs.size(); i++)
                {
                    if (p.coeffs[i] != 1)
                    {
                        simpleCoeffs = false;
                        break;
                    }
                }
            }
            if (simpleCoeffs)
                p.coeffs.clear();
            p.activ = self.activ.get();

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            const EltwiseOp op = self.op;
            size_t total = dst->size[0]*planeSize;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);
            const float* coeffsptr = !coeffs.empty() ? &coeffs[0] : 0;
            float* dstptr0 = dst->ptr<float>();
            int blockSize0 = 1 << 12;

            for (size_t ofs = stripeStart; ofs < stripeEnd; )
            {
                int sampleIdx = (int)(ofs / planeSize);
                int delta = (int)ofs - sampleIdx * planeSize;
                int blockSize = std::min(blockSize0, std::min((int)(stripeEnd - ofs), (int)planeSize - delta));
                if( blockSize <= 0 )
                    break;
                ofs += blockSize;

                for (int c = 0; c < channels; c++)
                {
                    size_t dstIdx = delta + (sampleIdx*channels + c)*planeSize;
                    float* dstptr = dstptr0 + dstIdx;

                    // process first two inputs
                    {
                        const float* srcptr0 = srcs[0]->ptr<float>() + dstIdx;

                        const int inputIdx = 1;
                        int src1_channels = srcNumChannels[inputIdx];
                        if (c >= src1_channels)
                        {
                            // no data from second input
                            if (!coeffsptr || coeffsptr[0] == 1.0f)
                            {
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] = srcptr0[j];
                                }
                            }
                            else
                            {
                                float c0 = coeffsptr[0];
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] = c0*srcptr0[j];
                                }
                            }
                        }
                        else
                        {
                            size_t srcIdx = delta + (sampleIdx * src1_channels + c) * planeSize;
                            const float* srcptrI = srcs[inputIdx]->ptr<float>() + srcIdx;

                            if (op == PROD)
                            {
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] = srcptr0[j] * srcptrI[j];
                                }
                            }
                            else if (op == DIV)
                            {
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] = srcptr0[j] / srcptrI[j];
                                }
                            }
                            else if (op == MAX)
                            {
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] = std::max(srcptr0[j], srcptrI[j]);
                                }
                            }
                            else if (op == MIN)
                            {
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] = std::min(srcptr0[j], srcptrI[j]);
                                }
                            }
                            else if (op == SUM)
                            {
                                if (!coeffsptr || (coeffsptr[0] == 1.0f && coeffsptr[1] == 1.0f))
                                {
                                    for (int j = 0; j < blockSize; j++)
                                    {
                                        dstptr[j] = srcptr0[j] + srcptrI[j];
                                    }
                                }
                                else
                                {
                                    float c0 = coeffsptr[0];
                                    float c1 = coeffsptr[1];
                                    for (int j = 0; j < blockSize; j++)
                                    {
                                        dstptr[j] = c0*srcptr0[j] + c1*srcptrI[j];
                                    }
                                }
                            }
                            else
                                CV_Error(Error::StsInternal, "");
                        }
                    }

                    // aggregate other inputs (3+)
                    for (size_t inputIdx = 2; inputIdx < nsrcs; inputIdx++)
                    {
                        int srcI_channels = srcNumChannels[inputIdx];
                        if (c >= srcI_channels)
                            continue;  // no data from second input
                        size_t srcIdx = delta + (sampleIdx * srcI_channels + c) * planeSize;
                        const float* srcptrI = srcs[inputIdx]->ptr<float>() + srcIdx;

                        if (op == PROD)
                        {
                            for (int j = 0; j < blockSize; j++)
                            {
                                dstptr[j] *= srcptrI[j];
                            }
                        }
                        else if (op == DIV)
                        {
                            for (int j = 0; j < blockSize; j++)
                            {
                                dstptr[j] /= srcptrI[j];
                            }
                        }
                        else if (op == MAX)
                        {
                            for (int j = 0; j < blockSize; j++)
                            {
                                dstptr[j] = std::max(dstptr[j], srcptrI[j]);
                            }
                        }
                        else if (op == MIN)
                        {
                            for (int j = 0; j < blockSize; j++)
                            {
                                dstptr[j] = std::min(dstptr[j], srcptrI[j]);
                            }
                        }
                        else if (op == SUM)
                        {
                            if (!coeffsptr || coeffsptr[inputIdx] == 1.0f)
                            {
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] += srcptrI[j];
                                }
                            }
                            else
                            {
                                float cI = coeffsptr[inputIdx];
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] += cI * srcptrI[j];
                                }
                            }
                        }
                        else
                            CV_Error(Error::StsInternal, "");
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

        if ((inputs_.depth() == CV_16S && op != SUM) || (channelsMode != ELTWISE_CHANNNELS_SAME))
            return false;

        if (hasVecInput)
            return false; // TODO not implemented yet: https://github.com/opencv/opencv/pull/19477

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
            case DIV:
                divide(inputs[0], inputs[1], outputs[0]);
                for (int i = 2; i < inputs.size(); ++i)
                    divide(outputs[0], inputs[i], outputs[0]);
                break;
            case MAX:
                max(inputs[0], inputs[1], outputs[0]);
                for (int i = 2; i < inputs.size(); ++i)
                    max(inputs[i], outputs[0], outputs[0]);
                break;
            case MIN:
                min(inputs[0], inputs[1], outputs[0]);
                for (int i = 2; i < inputs.size(); ++i)
                    min(inputs[i], outputs[0], outputs[0]);
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

        if (channelsModeInput == ELTWISE_CHANNNELS_SAME && inputs[0].dims > 2)
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                MatShape inpShape = shape(inputs[i].size);
                bool allOnes = isAllOnes(inpShape, 2, inputs[i].dims);

                if (allOnes)
                {
                    Mat tmpInput = inputs[i];
                    MatShape outShape = shape(outputs[0].size);
                    size_t xSize = outShape[2];
                    for (size_t j = 3; j < outShape.size(); j++)
                        xSize *= outShape[j];

                    int dimVec[3] = {outShape[0], outShape[1], (int) xSize};
                    std::vector<int> matSizesVec(&dimVec[0], &dimVec[0] + 3);
                    inputs[i] = Mat(matSizesVec, tmpInput.type());

                    std::vector<int> idx(outShape.size(), 0);
                    std::vector<int> outIdx(inpShape.size(), 0);

                    for (size_t j = 0; j < outShape[0]; j++)
                    {
                        outIdx[0] = idx[0] = j;
                        for(size_t k = 0; k < outShape[1]; k++)
                        {
                            outIdx[1] = idx[1] = k;
                            for (size_t x = 0; x < xSize; x++)
                            {
                                outIdx[2] = x;
                                inputs[i].at<float>(outIdx.data()) = tmpInput.at<float>(idx.data());
                            }
                        }
                    }
                    inputs[i] = inputs[i].reshape(0, outShape);
                }
            }
        }

        EltwiseInvoker::run(*this,
                            &inputs[0], (int)inputs.size(), outputs[0],
                            nstripes);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        CV_Assert(channelsModeInput == ELTWISE_CHANNNELS_INPUT_0 ||
                  channelsModeInput == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE ||
                  channelsModeInput == ELTWISE_CHANNNELS_SAME);

        if(channelsModeInput == ELTWISE_CHANNNELS_INPUT_0 || channelsModeInput == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE)
        {
            auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
            for (int i = 1; i < inputs.size(); i++)
            {
                auto from_wrapper = inputs[i].dynamicCast<CUDABackendWrapper>();
                if (input_wrapper->getShape()[1] != from_wrapper->getShape()[1])
                {
                    CV_Assert(op == SUM);
                    CV_Assert(coeffs.empty());
                    return make_cuda_node<cuda4dnn::ShortcutOp>(preferableTarget, std::move(context->stream));
                }
            }
        }

        auto op_ = [this] {
            switch (op) {
            case MAX: return cuda4dnn::EltwiseOpType::MAX;
            case MIN: return cuda4dnn::EltwiseOpType::MIN;
            case SUM: return cuda4dnn::EltwiseOpType::SUM;
            case PROD: return cuda4dnn::EltwiseOpType::PRODUCT;
            case DIV: return cuda4dnn::EltwiseOpType::DIV;
            }
            return cuda4dnn::EltwiseOpType::SUM;
        }();

        return make_cuda_node<cuda4dnn::EltwiseOp>(preferableTarget, std::move(context->stream), op_, coeffs);
    }
#endif

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
            case DIV:
                topExpr = inputBuffers[0](x, y, c, n) /
                          inputBuffers[1](x, y, c, n);
                for (int i = 2; i < inputBuffers.size(); ++i)
                    topExpr /= inputBuffers[i](x, y, c, n);
                break;
            case MAX:
                topExpr = max(inputBuffers[0](x, y, c, n),
                              inputBuffers[1](x, y, c, n));
                for (int i = 2; i < inputBuffers.size(); ++i)
                    topExpr = max(topExpr, inputBuffers[i](x, y, c, n));
                break;
            case MIN:
                topExpr = min(inputBuffers[0](x, y, c, n),
                              inputBuffers[1](x, y, c, n));
                for (int i = 2; i < inputBuffers.size(); ++i)
                    topExpr = min(topExpr, inputBuffers[i](x, y, c, n));
                break;
            default:
                return Ptr<BackendNode>();
        }
        top(x, y, c, n) = topExpr;
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputsWrapper, const int index, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(inputsWrapper.size() == 2);
        CV_Assert(nodes.size() == 2);

        auto op_x1 = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto x1 = inputsWrapper[0].dynamicCast<CannBackendWrapper>();
        auto x1_desc = x1->getTensorDesc();
        auto op_x2 = nodes[1].dynamicCast<CannBackendNode>()->getOp();
        auto x2 = inputsWrapper[1].dynamicCast<CannBackendWrapper>();
        auto x2_desc = x2->getTensorDesc();
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        std::shared_ptr<ge::Operator> eltwise_operator = nullptr;
        // add, mul, div, max, min
        switch (op)
        {
#define BUILD_CANN_ELTWISE_OP(op_type, class_name, op_name)      \
            case op_type: {                                      \
                auto eltwise_op =                                \
                  std::make_shared<ge::op::class_name>(op_name); \
                eltwise_op->set_input_x1_by_name(*op_x1, "y");   \
                eltwise_op->set_input_x2_by_name(*op_x2, "y");   \
                eltwise_op->update_input_desc_x1(*x1_desc);      \
                eltwise_op->update_input_desc_x2(*x2_desc);      \
                eltwise_op->update_output_desc_y(*output_desc);  \
                eltwise_operator = eltwise_op;                   \
            } break;
            BUILD_CANN_ELTWISE_OP(SUM, Add, cv::format("add_%d", index));
            BUILD_CANN_ELTWISE_OP(PROD, Mul, cv::format("mul_%d", index));
            BUILD_CANN_ELTWISE_OP(DIV, Xdivy, cv::format("div_%d", index));
            BUILD_CANN_ELTWISE_OP(MAX, Maximum, cv::format("max_%d", index));
            BUILD_CANN_ELTWISE_OP(MIN, Minimum, cv::format("min_%d", index));
#undef BUILD_CANN_ELTWISE_OP
            default: CV_Error(Error::StsNotImplemented, "Unsupported eltwise operation");
        }

        return Ptr<BackendNode>(new CannBackendNode(eltwise_operator));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto curr_node = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        if (!coeffs.empty()) {
            auto coeff = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{1}, &coeffs[0]);
            curr_node = std::make_shared<ngraph::op::v1::Multiply>(curr_node, coeff, ngraph::op::AutoBroadcastType::NUMPY);
        }

        for (size_t i = 1; i < nodes.size(); i++)
        {
            auto next_node = nodes[i].dynamicCast<InfEngineNgraphNode>()->node;
            if (!coeffs.empty()) {
                auto coeff = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{1}, &coeffs[i]);
                next_node = std::make_shared<ngraph::op::v1::Multiply>(next_node, coeff, ngraph::op::AutoBroadcastType::NUMPY);
            }
            switch (op) {
                case SUM:  curr_node = std::make_shared<ngraph::op::v1::Add>(curr_node, next_node); break;
                case PROD: curr_node = std::make_shared<ngraph::op::v1::Multiply>(curr_node, next_node); break;
                case DIV:  curr_node = std::make_shared<ngraph::op::v1::Divide>(curr_node, next_node); break;
                case MAX:  curr_node = std::make_shared<ngraph::op::v1::Maximum>(curr_node, next_node); break;
                case MIN:  curr_node = std::make_shared<ngraph::op::v1::Minimum>(curr_node, next_node); break;
                default: CV_Error(Error::StsNotImplemented, "Unsupported eltwise operation");
            }
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(curr_node));
    }
#endif  // HAVE_DNN_NGRAPH

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        params.set("input_scales", DictValue::arrayReal(scales[0].data(), scales[0].size()));
        params.set("input_zeropoints", DictValue::arrayInt(zeropoints[0].data(), zeropoints[0].size()));
        if (op == SUM)
        {
            std::vector<float> newCoeffs;
            float offset = zeropoints[1][0];
            float out_sc = scales[1][0];
            for (int i = 0; i < scales[0].size(); i++)
            {
                float coeff = coeffs.empty() ? 1.f : coeffs[i];
                float newcoeff = (scales[0][i] * coeff) / out_sc;
                newCoeffs.push_back(newcoeff);
                offset -= (newcoeff * zeropoints[0][i]);
            }
            params.set("coeff", DictValue::arrayReal(newCoeffs.data(), newCoeffs.size()));
            params.set("offset", offset);
            return true;
        }
        else if (op == PROD)
        {
            std::vector<float> newCoeffs = scales[0];
            newCoeffs[0] /= scales[1][0];
            params.set("coeff", DictValue::arrayReal(newCoeffs.data(), newCoeffs.size()));
            params.set("offset", zeropoints[1][0]);
            return true;
        }
        return op == MAX;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        CV_Assert(inputs.size());

        // FIXIT: handle inputs with different number of channels
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

private:
    bool hasVecInput;
};

Ptr<EltwiseLayer> EltwiseLayer::create(const LayerParams& params)
{
    return Ptr<EltwiseLayer>(new EltwiseLayerImpl(params));
}

}
}
