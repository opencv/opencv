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
#include "../op_timvx.hpp"
#include "../op_cann.hpp"

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
        paddingValue = params.get<int>("padding_value", 0);

        zeropoint = params.get<int>("zeropoints", 0);
        scale = params.get<float>("scales", 1.0f);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() > 0);
        outputs.resize(1, inputs[0]);
        int cAxis = normalize_axis(axis, inputs[0]);

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

            axisSum += curShape.dims >= cAxis ? curShape[cAxis] : 1;
        }
        outputs[0].dims = std::max(outputs[0].dims, 1);
        outputs[0][cAxis] = axisSum;
        return false;
    }

    virtual  void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size());
        for (int i = 1; i < inputs.size(); i++)
            CV_CheckTypeEQ(inputs[i], inputs[0], "All input types should be equal");
        outputs.assign(1, inputs[0]);
    }


    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        if (backendId == DNN_BACKEND_TIMVX && haveTimVX() && !padding)
        {
            if (axis == -1)
                return false;
            int len = this->type.length();
            if (len <= 4)
                return false;
            if (this->type.substr(len - 4) == "Int8")
                return true;
            else
                return false;
        }
#endif

#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               (backendId == DNN_BACKEND_WEBNN && !padding) ||
               (backendId == DNN_BACKEND_CANN && !padding);
    }

    template <class T>
    class ChannelConcatInvoker : public ParallelLoopBody
    {
    public:
        std::vector<Mat>* inputs;
        Mat* output;
        int nstripes;
        std::vector<const T*> chptrs;

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
                CV_Assert( inp.isContinuous() && (inp.type() == CV_32F || inp.type() == CV_16F || inp.type() == CV_8S) &&
                           inp.dims == 4 && inp.size[0] == output.size[0] &&
                           inp.size[2] == output.size[2] &&
                           inp.size[3] == output.size[3] );
                nchannels += inp.size[1];
            }
            CV_Assert( nchannels == output.size[1] );
            CV_Assert( output.isContinuous() && (output.type() == CV_32F || output.type() == CV_16F || output.type() == CV_8S) );

            cc.chptrs.resize(nchannels*batchsz);

            int ofs = 0;
            for( i = 0; i < ninputs; i++)
            {
                Mat& inp = inputs[i];
                for( int j = 0; j < batchsz; j++ )
                    for( int k = 0; k < inp.size[1]; k++ )
                    {
                        const T* ptr = inp.ptr<T>(j, k);
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
            const T** ptrs = (const T**)&chptrs[0];
            T* outptr = output->ptr<T>();
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

        bool use_half = (inps.depth() == CV_16F);
        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        int cAxis = normalize_axis(axis, inputs[0].dims);
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
        std::cout << "\n==>ConcatLayerImpl::forward" << std::endl;
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) &&
                   (inputs_arr.depth() == CV_32F || inputs_arr.depth() == CV_16F),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        for (auto &inp : inputs) {
            std::cout << "input shape: " << inp.size << std::endl;
            std::cout << "input sum: " << cv::sum(inp) << std::endl;
            std::cout << "--------------------------------" << std::endl;
        }

        int cAxis = normalize_axis(axis, inputs[0].dims);
        Mat& outMat = outputs[0];

        if (padding)
            outMat.setTo(paddingValue);

        if(cAxis == 1 && outMat.dims == 4 && !padding && (inputs[0].depth() == CV_32F || inputs[0].depth() == CV_8S))
        {
            int nstripes = getNumThreads();
            if (outMat.type() == CV_8S)
                ChannelConcatInvoker<int8_t>::run(inputs, outMat, nstripes);
            else
                ChannelConcatInvoker<float>::run(inputs, outMat, nstripes);
        }
        else
        {
            std::vector<Range> ranges(outputs[0].dims, Range::all());

            ranges[cAxis].start = 0;
            for (size_t i = 0; i < inputs.size(); i++)
            {
                if (inputs[i].empty())
                    continue;
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
        auto concat_axis = normalize_axis(axis, input_wrapper->getRank());
        if (inputs[0]->getHostMatDepth() == CV_Bool)
            return make_cuda_node_bool<cuda4dnn::ConcatOp>(std::move(context->stream), concat_axis, padding);
        else
            return make_cuda_node_with_type<cuda4dnn::ConcatOp>(preferableTarget, inputs[0]->getHostMatDepth(), std::move(context->stream), concat_axis, padding);
    }
#endif

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(inputs.size() == nodes.size());

        // create operator
        auto op = std::make_shared<ge::op::ConcatD>(name);

        // set attributes
        int N = inputs.size();
        op->set_attr_concat_dim(axis);
        op->set_attr_N(N);

        // set inputs : x (dynamic)
        op->create_dynamic_input_x(N);
        for (int i = 0; i < N; i++)
        {
            auto x_i = inputs[i].dynamicCast<CannBackendWrapper>();
            auto x_i_desc = x_i->getTensorDesc();
            auto op_x_i = nodes[i].dynamicCast<CannBackendNode>()->getOp();
            op->set_dynamic_input_x(i, *op_x_i, x_i->name.c_str());
            op->update_dynamic_input_desc_x(i, *x_i_desc);
        }

        // set outputs
        auto output_y_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_y_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        const int numDims = nodes[0].dynamicCast<InfEngineNgraphNode>()->node.get_shape().size();
        const int cAxis = normalize_axis(axis, numDims);
        std::vector<size_t> maxDims(numDims, 0);

        CV_Assert(inputs.size() == nodes.size());
        ov::OutputVector inp_nodes;
        for (int i = 0; i < nodes.size(); ++i)
        {
            auto inp = nodes[i].dynamicCast<InfEngineNgraphNode>()->node;
            inp_nodes.push_back(inp);

            std::vector<size_t> inpShape = inp.get_shape();
            for (int i = 0; i < numDims; ++i)
                maxDims[i] = std::max(maxDims[i], inpShape[i]);
        }
        for (int i = 0; i < inp_nodes.size(); ++i)
        {
            bool needPadding = false;
            std::vector<size_t> inpShape = inp_nodes[i].get_shape();
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
                inp_nodes[i] = std::make_shared<ov::op::v1::Pad>(
                    inp_nodes[i],
                    std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{begins.size()}, begins.data()),
                    std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ends.size()}, ends.data()),
                    ov::op::PadMode::CONSTANT);
            }
        }
        auto concat = std::make_shared<ov::op::v0::Concat>(inp_nodes, cAxis);
        return Ptr<BackendNode>(new InfEngineNgraphNode(concat));
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_TIMVX
    virtual Ptr<BackendNode> initTimVX(void* timVXInfo_,
                                       const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                       const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                       bool isLast) CV_OVERRIDE
    {
        // tvGraph Initialization.
        auto timVxInfo = reinterpret_cast<TimVXInfo *>(timVXInfo_);
        CV_Assert(timVxInfo);
        Ptr<TimVXGraph> tvGraph = timVxInfo->getGraph();
        CV_Assert(tvGraph);
        Ptr<tim::vx::Graph> graph = tvGraph->graph;

        Ptr<TimVXBackendWrapper> inputWrapper = inputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        // convert axis from OpenCV NCHW toTimVX WHCN.
        Mat blob0 = inputWrapper->getMat();

        // TODO! support TimVX 5 dim in future.
        if(blob0.dims >4)
            return Ptr<TimVXBackendNode>();

        int cAxis = normalize_axis(axis, blob0.dims);
        int tvAxis = blob0.dims - 1 - cAxis;
        CV_Assert(tvAxis>= 0);
        std::vector<int> inputsIndex, outputsIndex;
        int input_index = -1, output_index = -1;

        // Input
        Ptr<tim::vx::Quantization> tvQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, scale, zeropoint));

        for (int i = 0; i<inputsWrapper.size(); i++)
        {
            inputWrapper = inputsWrapper[i].dynamicCast<TimVXBackendWrapper>();
            if (inputWrapper->isTensor())
            {
                input_index = tvGraph->getTensorIndex(inputWrapper->getTensor());
                if (input_index == -1)
                {
                    // Copy To New inputWrapper
                    Mat tmp = inputWrapper->getMat();
                    inputWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tmp));
                }
            }

            if (!inputWrapper->isTensor())
            {
                inputWrapper->createTensor(graph,tim::vx::TensorAttribute::INPUT, tvQuant);
                input_index = tvGraph->addWrapper(inputWrapper);
            }
            inputsIndex.push_back(input_index);
        }

        //Output
        CV_Assert(outputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> outputWrapper = outputsWrapper[0].dynamicCast<TimVXBackendWrapper>();

        if (isLast)
        {
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, tvQuant);
        }
        else
        {
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, tvQuant);
        }
        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

        std::shared_ptr<tim::vx::Operation> tvConcate = graph->CreateOperation<tim::vx::ops::Concat>(tvAxis, inputsWrapper.size());

        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvConcate, inputsIndex, outputsIndex);

        return tvBackendNode;
    }
#endif // HAVE_TIMVX

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnGraphBuilder = node->net->builder;
        std::vector<ml::Operand> inputsOperand;
        for (int i = 0; i < nodes.size(); i++)
        {
            inputsOperand.push_back(nodes[i].dynamicCast<WebnnBackendNode>()->operand);
        }
        auto operand = webnnGraphBuilder.Concat(inputsOperand.size(), inputsOperand.data(), axis);
        return Ptr<BackendNode>(new WebnnBackendNode(operand));
    }
#endif

    int zeropoint;
    float scale;
};

Ptr<ConcatLayer> ConcatLayer::create(const LayerParams& params)
{
    return Ptr<ConcatLayer>(new ConcatLayerImpl(params));
}

}
}
