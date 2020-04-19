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

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/concat.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class ConcatLayerImpl CV_FINAL : public ConcatLayer
{
public:
    ConcatLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 1);
        padding = params.get<bool>("padding", false);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() > 0);
        outputs.resize(1, inputs[0]);
        int cAxis = clamp(axis, inputs[0]);

        int axisSum = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape curShape = inputs[i];

            if (padding)
            {
                for (int curAxis = 0; curAxis < outputs[0].size(); curAxis++)
                {
                    outputs[0][curAxis] = std::max(outputs[0][curAxis], curShape[curAxis]);
                }
            }
            else
            {
                CV_Assert(curShape.size() == outputs[0].size());
                for (int curAxis = 0; curAxis < outputs[0].size(); curAxis++)
                {
                    if (curAxis != cAxis && outputs[0][curAxis] != curShape[curAxis])
                        CV_Error(Error::StsBadSize, "Inconsistent shape for ConcatLayer");
                }
            }

            axisSum += curShape[cAxis];
        }
        outputs[0][cAxis] = axisSum;
        return false;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               (backendId == DNN_BACKEND_HALIDE && haveHalide() && axis == 1 && !padding) ||  // By channels
               (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && haveInfEngine() && !padding) ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan() && !padding);
    }

    class ChannelConcatInvoker : public ParallelLoopBody
    {
    public:
        std::vector<Mat>* inputs;
        Mat* output;
        int nstripes;
        std::vector<const float*> chptrs;

        static void run(std::vector<Mat>& inputs, Mat& output, int nstripes)
        {
            ChannelConcatInvoker cc;
            cc.inputs = &inputs;
            cc.output = &output;
            cc.nstripes = nstripes;

            size_t i, ninputs = inputs.size();
            int nchannels = 0, batchsz = output.size[0];
            for( i = 0; i < ninputs; i++ )
            {
                Mat& inp = inputs[i];
                CV_Assert( inp.isContinuous() && (inp.type() == CV_32F || inp.type() == CV_16S) &&
                           inp.dims == 4 && inp.size[0] == output.size[0] &&
                           inp.size[2] == output.size[2] &&
                           inp.size[3] == output.size[3] );
                nchannels += inp.size[1];
            }
            CV_Assert( nchannels == output.size[1] );
            CV_Assert( output.isContinuous() && (output.type() == CV_32F || output.type() == CV_16S) );

            cc.chptrs.resize(nchannels*batchsz);

            int ofs = 0;
            for( i = 0; i < ninputs; i++)
            {
                Mat& inp = inputs[i];
                for( int j = 0; j < batchsz; j++ )
                    for( int k = 0; k < inp.size[1]; k++ )
                    {
                        const float* ptr = inp.ptr<float>(j, k);
                        cc.chptrs[ofs + j*nchannels + k] = ptr;
                    }
                ofs += inp.size[1];
            }

            parallel_for_(Range(0, nstripes), cc, nstripes);
        }

        ChannelConcatInvoker()  : inputs(0), output(0), nstripes(0) {}

        void operator()(const Range& r) const CV_OVERRIDE
        {
            size_t planeSize = (size_t)output->size[2]*output->size[3];
            size_t nch = chptrs.size();
            size_t total = nch*planeSize;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(total, r.end*stripeSize);
            const float** ptrs = (const float**)&chptrs[0];
            float* outptr = output->ptr<float>();
            size_t blockSize0 = 1 << 16;

            for( size_t ofs0 = stripeStart; ofs0 < stripeEnd; )
            {
                size_t ch = ofs0/planeSize;
                size_t ofs = ofs0 - ch*planeSize;
                size_t blockSize = std::min(blockSize0, planeSize - ofs);
                memcpy(outptr + ofs0, ptrs[ch] + ofs, blockSize*sizeof(outptr[0]));
                ofs0 += blockSize;
            }
        }
    };

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inps.depth() == CV_16S);
        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        int cAxis = clamp(axis, inputs[0].dims);
        if (padding)
            return false;

        int bottom_concat_axis;
        int concat_size = total(shape(inputs[0]), cAxis + 1);
        int top_concat_axis = outputs[0].size[cAxis];
        int num_concats = total(shape(inputs[0]), 0, cAxis);
        int offset_concat_axis = 0;
        UMat& outMat = outputs[0];
        String buildopt = format(" -DDtype=%s", (use_half) ? "half" : "float");
        String kname = format("concat_%s", use_half ? "half" : "float");

        for (size_t i = 0; i < inputs.size(); i++)
        {
            ocl::Kernel kernel(kname.c_str(), ocl::dnn::concat_oclsrc, buildopt);
            if (kernel.empty())
                return false;

            UMat& inpMat = inputs[i];
            bottom_concat_axis = inputs[i].size[cAxis];
            size_t nthreads = inputs[i].total();

            kernel.set(0, (int)nthreads);
            kernel.set(1, ocl::KernelArg::PtrReadOnly(inpMat));
            kernel.set(2, (int)num_concats);
            kernel.set(3, (int)concat_size);
            kernel.set(4, (int)top_concat_axis);
            kernel.set(5, (int)bottom_concat_axis);
            kernel.set(6, (int)offset_concat_axis);
            kernel.set(7, ocl::KernelArg::PtrWriteOnly(outMat));

            if (!kernel.run(1, &nthreads, NULL, false))
                return false;

            offset_concat_axis += bottom_concat_axis;
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

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        int cAxis = clamp(axis, inputs[0].dims);
        Mat& outMat = outputs[0];

        if (padding)
            outMat.setTo(0);

        if( cAxis == 1 && outMat.dims == 4 && !padding)
        {
            int nstripes = getNumThreads();
            ChannelConcatInvoker::run(inputs, outMat, nstripes);
        }
        else
        {
            std::vector<Range> ranges(outputs[0].dims, Range::all());

            ranges[cAxis].start = 0;
            for (size_t i = 0; i < inputs.size(); i++)
            {
                ranges[cAxis].end = ranges[cAxis].start + inputs[i].size[cAxis];
                for (int j = 0; j < outMat.dims; ++j)
                {
                    if (j == cAxis) continue;
                    ranges[j].start = (outMat.size[j] - inputs[i].size[j]) / 2;
                    ranges[j].end = ranges[j].start + inputs[i].size[j];
                }
                inputs[i].copyTo(outMat(&ranges[0]));
                ranges[cAxis].start = ranges[cAxis].end;
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

        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto concat_axis = clamp(axis, input_wrapper->getRank());
        return make_cuda_node<cuda4dnn::ConcatOp>(preferableTarget, std::move(context->stream), concat_axis, padding);
    }
#endif

    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &input) CV_OVERRIDE
    {
#ifdef HAVE_VULKAN
        vkcom::Tensor in = VkComTensor(input[0]);
        int cAxis = clamp(axis, in.dimNum());
        std::shared_ptr<vkcom::OpBase> op(new vkcom::OpConcat(cAxis));
        return Ptr<BackendNode>(new VkComBackendNode(input, op));
#endif // HAVE_VULKAN
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &input) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        std::vector<Halide::Buffer<> > inputBuffers = halideBuffers(input);

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        int offset = inputBuffers[0].channels();
        Halide::Expr topExpr = select(c < offset,
                                      inputBuffers[0](x, y, c, n),
                                      inputBuffers[1](x, y, c - offset, n));
        for (int i = 2; i < input.size(); ++i)
        {
            offset += inputBuffers[i - 1].channels();
            topExpr = select(c < offset, topExpr,
                             inputBuffers[i](x, y, c - offset, n));
        }
        top(x, y, c, n) = topExpr;
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
    {
        InferenceEngine::DataPtr input = infEngineDataNode(inputs[0]);

        InferenceEngine::Builder::ConcatLayer ieLayer(name);
        ieLayer.setAxis(clamp(axis, input->getDims().size()));
        ieLayer.setInputPorts(std::vector<InferenceEngine::Port>(inputs.size()));
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        InferenceEngine::DataPtr data = ngraphDataNode(inputs[0]);
        const int numDims = data->getDims().size();
        const int cAxis = clamp(axis, numDims);
        std::vector<size_t> maxDims(numDims, 0);

        CV_Assert(inputs.size() == nodes.size());
        ngraph::OutputVector inp_nodes;
        for (int i = 0; i < nodes.size(); ++i)
        {
            inp_nodes.push_back(nodes[i].dynamicCast<InfEngineNgraphNode>()->node);

            std::vector<size_t> inpShape = ngraphDataNode(inputs[i])->getDims();
            for (int i = 0; i < numDims; ++i)
                maxDims[i] = std::max(maxDims[i], inpShape[i]);
        }
        for (int i = 0; i < inp_nodes.size(); ++i)
        {
            bool needPadding = false;
            std::vector<size_t> inpShape = ngraphDataNode(inputs[i])->getDims();
            std::vector<int64_t> begins(inpShape.size(), 0), ends(inpShape.size(), 0);
            for (int j = 0; j < inpShape.size(); ++j)
            {
                if (j != cAxis && inpShape[j] != maxDims[j])
                {
                    needPadding = true;
                    begins[j] = static_cast<int64_t>((maxDims[j] - inpShape[j]) / 2);
                    ends[j] = static_cast<int64_t>(maxDims[j] - inpShape[j] - begins[j]);
                }
            }
            if (needPadding)
            {
                inp_nodes[i] = std::make_shared<ngraph::op::v1::Pad>(
                    inp_nodes[i],
                    std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{begins.size()}, begins.data()),
                    std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ends.size()}, ends.data()),
                    ngraph::op::PadMode::CONSTANT);
            }
        }
        auto concat = std::make_shared<ngraph::op::Concat>(inp_nodes, cAxis);
        return Ptr<BackendNode>(new InfEngineNgraphNode(concat));
    }
#endif  // HAVE_DNN_NGRAPH
};

Ptr<ConcatLayer> ConcatLayer::create(const LayerParams& params)
{
    return Ptr<ConcatLayer>(new ConcatLayerImpl(params));
}

}
}
