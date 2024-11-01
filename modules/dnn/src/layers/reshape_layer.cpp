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
#include "../op_webnn.hpp"
#include "../op_timvx.hpp"
#include "../op_cann.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/reshape.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

static void computeShapeByReshapeMask(const MatShape &srcShape,
                                      const MatShape &maskShape,
                                      Range srcRange /*= Range::all()*/,
                                      MatShape& dstShape)
{
    int srcShapeSize = (int)srcShape.size();
    int maskShapeSize = (int)maskShape.size();

    srcRange = normalize_axis_range(srcRange, srcShapeSize);

    bool explicitMask = !maskShape.empty();  // All mask values are positive.
    for (int i = 0, n = maskShape.size(); i < n && explicitMask; ++i)
    {
        explicitMask = maskShape[i] > 0;
    }
    // Working range of source shape is a range where area(src) == area(mask).
    if (explicitMask)
    {
        int maskTotal = total(maskShape);
        // Go from the end of mask until we collect required total.
        bool matched = false;
        for (int i = srcRange.end - 1; i >= srcRange.start; --i)
        {
            if (matched)
            {
                if (total(srcShape, i, srcRange.end) != maskTotal)
                {
                    srcRange.start = i + 1;
                    break;
                }
                else if (i == 0)
                {
                    srcRange.start = 0;
                    break;
                }
            }
            else
            {
                matched = total(srcShape, i, srcRange.end) == maskTotal;
            }
        }
        while (total(srcShape, srcRange.start, srcRange.end) != maskTotal && srcRange.start > 0)
        {
            srcRange.start -= 1;
        }
        CV_Assert(total(srcShape, srcRange.start, srcRange.end) == maskTotal);
    }

    CV_Assert(0 <= srcRange.start && srcRange.start <= srcRange.end && srcRange.end <= srcShapeSize);
    int dstShapeSize = srcShapeSize - srcRange.size() + maskShapeSize;
    dstShape.resize(dstShapeSize);

    std::copy(srcShape.begin(), srcShape.begin() + srcRange.start, dstShape.begin());
    std::copy(srcShape.begin() + srcRange.end, srcShape.begin() + srcShapeSize, dstShape.begin() + srcRange.start + maskShapeSize);

    int inferDim = -1;
    for (int i = 0; i < maskShapeSize; i++)
    {
        if (maskShape[i] > 0)
        {
            dstShape[srcRange.start + i] = maskShape[i];
        }
        else if (maskShape[i] == 0)
        {
            if (srcRange.start + i >= srcShapeSize)
                CV_Error(Error::StsBadArg, format("Copy dim[%d] (which has zero size) is out of the source shape bounds", srcRange.start + i));
            dstShape[srcRange.start + i] = srcShape[srcRange.start + i];
        }
        else if (maskShape[i] == -1)
        {
            if (inferDim != -1)
                CV_Error(Error::StsAssert, "Duplicate of inferred dim (which is denoted by -1)");
            inferDim = srcRange.start + i;
            dstShape[inferDim] = 1;
        }
        else
            CV_Error(Error::StsBadArg, "maskShape[i] >= -1");
    }

    size_t srcTotal = total(srcShape);
    size_t dstTotal = total(dstShape);
    CV_Assert(dstTotal != 0);

    if (inferDim != -1)
    {
        if (srcTotal % dstTotal != 0)
            CV_Error(Error::StsBackTrace, "Can't infer a dim denoted by -1");

        dstShape[inferDim] = (int)(srcTotal / dstTotal);
    }
    else
    {
        CV_Assert(srcTotal == dstTotal);
    }
}


class ReshapeLayerImpl CV_FINAL : public ReshapeLayer
{
public:
    ReshapeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 0);
        numAxes = params.get<int>("num_axes", -1);
        hasDynamicShapes = params.get<bool>("has_dynamic_shapes", false);
        shapesInitialized = !hasDynamicShapes;

        zeropoint = params.get<int>("zeropoints", 0);
        scale = params.get<float>("scales", 1.0f);

        CV_Assert(numAxes >= -1);
        newShapeRange = (numAxes == -1) ? Range(axis, INT_MAX) : Range(axis, axis + numAxes);

        newShapeDesc.clear();
        if (params.has("dim"))
        {
            const DictValue &paramShape = params.get("dim");
            int i, dims = paramShape.size();
            newShapeDesc.resize(dims);
            for (i = 0; i < dims; i++)
                newShapeDesc[i] = paramShape.get<int>(i);
        }
        if (hasDynamicShapes)
        {
            dynamicShapes.clear();
            inputIndices.clear();
            if (params.has("dynamic_axes")) {
                CV_Assert(params.has("input_indices"));
                const DictValue &dynamicAxes = params.get("dynamic_axes");
                const DictValue &dynamicInputShapes = params.get("input_indices");
                int i, dims = dynamicAxes.size();
                CV_Assert(dims == dynamicInputShapes.size());
                CV_Assert(dims > 0);
                dynamicShapes.resize(dims);
                inputIndices.resize(dims);
                for (i = 0; i < dims; i++) {
                    dynamicShapes[i] = dynamicAxes.get<int>(i);
                    inputIndices[i] = dynamicInputShapes.get<int>(i);
                }
            }
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_TIMVX && haveTimVX())
        {
            int len = this->type.length();
            if (len <= 4)
                return false;

            if (this->type.substr(len - 4) == "Int8")
                return true;
            else
                return false;
        }

#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_WEBNN ||
               backendId == DNN_BACKEND_CANN;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        if (inputs.size() == 1 || inputs.size() == requiredOutputs)
        {
            outputs.clear();
            for (size_t i = 0; i < inputs.size(); i++)
            {
                if (hasDynamicShapes && !shapesInitialized)
                {
                    outputs.push_back(newShapeDesc);
                }
                else
                {
                    outputs.push_back(MatShape());
                    computeShapeByReshapeMask(inputs[i], newShapeDesc, newShapeRange, outputs.back());
                }
            }
        }
        else
        {
            std::cout << "total(inputs[0]): " << total(inputs[0]) << std::endl;
            std::cout << "total(inputs[1]): " << total(inputs[1]) << std::endl;

            for (size_t i = 0; i < inputs.size(); i++)
            {
                for(size_t j = 0; j < inputs[i].size(); j++)
                {
                    std::cout << inputs[i][j] << ", ";
                }
                std::cout << std::endl;
            }


            CV_Assert_N(inputs.size() == 2, total(inputs[0]) == total(inputs[1]));
            outputs.assign(1, inputs[1]);
        }

        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size());
        for (auto input : inputs)
        {
            if (preferableTarget == DNN_TARGET_OPENCL_FP16)
                CV_CheckType(input, input == CV_16F || input == CV_8S || input == CV_8U || input == CV_32S || input == CV_64S || input == CV_Bool, "");
            else
                CV_CheckType(input, input == CV_32F || input == CV_8S || input == CV_8U || input == CV_32S || input == CV_64S || input == CV_Bool, "");
        }

        outputs.assign(requiredOutputs, inputs[0]);
    }


    bool updateMemoryShapes(const std::vector<MatShape> &inputs) CV_OVERRIDE
    {
        if (hasDynamicShapes)
        {
            for (int i = 0; i < dynamicShapes.size(); ++i)
            {
                newShapeDesc[dynamicShapes[i]] = inputs[0][inputIndices[i]];
            }
        }
        shapesInitialized = true;
        return true;
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);

        CV_Assert(!outputs.empty());
        outShapes.resize(outputs.size());
        for (int i = 0; i < outputs.size(); ++i)
            outShapes[i] = shape(outputs[i]);
    }

    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        for (size_t i = 0; i < outputs.size(); i++)
        {
            UMat srcBlob = inputs[i];
            void *src_handle = inputs[i].handle(ACCESS_READ);
            void *dst_handle = outputs[i].handle(ACCESS_WRITE);
            if (src_handle != dst_handle)
            {
                UMat umat = srcBlob.reshape(1, (int)outShapes[i].size(), &outShapes[i][0]);
                umat.copyTo(outputs[i]);
            }
        }
        outs.assign(outputs);

        return true;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        for (size_t i = 0; i < outputs.size(); i++)
        {
            Mat srcBlob = inputs[i];
            if (outputs[i].data != srcBlob.data)
                srcBlob.reshape(1, shape(outputs[i])).copyTo(outputs[i]);
        }
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        // create operator
        auto op = std::make_shared<ge::op::Reshape>(name);

        // set attributes
        op->set_attr_axis(axis);
        op->set_attr_num_axes(numAxes);

        // set inputs
        // set inputs : x
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);
        // set inputs : shape
        std::vector<int> shape_of_shape{(int)newShapeDesc.size()};
        Mat shape_mat(shape_of_shape, CV_32S, newShapeDesc.data());
        auto op_const_shape = std::make_shared<CannConstOp>(shape_mat.data, shape_mat.type(), shape_of_shape, cv::format("%s_shape", name.c_str()));
        op->set_input_shape(*(op_const_shape->getOp()));
        op->update_input_desc_shape(*(op_const_shape->getTensorDesc()));

        // set outputs
        auto output_y_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_y_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(outShapes.size() == 1);
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;

        std::vector<int64_t> out(outShapes[0].begin(), outShapes[0].end());
        auto shape   = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                       ov::Shape{out.size()}, out.data());
        auto reshape = std::make_shared<ov::op::v1::Reshape>(ieInpNode, shape, true);
        return Ptr<BackendNode>(new InfEngineNgraphNode(reshape));
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        const std::vector<int32_t> out(outShapes[0].begin(), outShapes[0].end());
        auto operand = webnnGraphBuilder.Reshape(webnnInpOperand, out.data(), out.size());
        return Ptr<BackendNode>(new WebnnBackendNode(operand));
    }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        if (inputs[0]->getHostMatDepth() == CV_Bool)
            return make_cuda_node_bool<cuda4dnn::ReshapeOp>(std::move(context->stream));
        else
            return make_cuda_node_with_type<cuda4dnn::ReshapeOp>(preferableTarget, inputs[0]->getHostMatDepth(), std::move(context->stream));
    }
#endif

    virtual Ptr<BackendNode> initTimVX(void* timVXInfo_,
                                       const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                       const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                       bool isLast) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        // tvGraph Initialization.
        auto timVxInfo = reinterpret_cast<TimVXInfo *>(timVXInfo_);
        CV_Assert(timVxInfo);
        Ptr<TimVXGraph> tvGraph = timVxInfo->getGraph();
        CV_Assert(tvGraph);
        Ptr<tim::vx::Graph> graph = tvGraph->graph;

        std::vector<int> inputsIndex, outputsIndex;
        int input_index = -1, output_index = -1;

        int reshapeNum = 0;
        Ptr<TimVXBackendWrapper> tmpWrapper, inputWrapper, outputWrapper;
        for (size_t i = 0; i < outputsWrapper.size(); i++)
        {
            tmpWrapper = inputsWrapper[i].dynamicCast<TimVXBackendWrapper>();
            Mat srcBlob = tmpWrapper->getMat();

            tmpWrapper = outputsWrapper[i].dynamicCast<TimVXBackendWrapper>();
            Mat dstBlob = tmpWrapper->getMat();
            if (dstBlob.data != srcBlob.data)
            {
                reshapeNum++;
                inputWrapper = inputsWrapper[i].dynamicCast<TimVXBackendWrapper>();
                outputWrapper = outputsWrapper[i].dynamicCast<TimVXBackendWrapper>();
            }
        }

        // Only work for single reshape Mat
        if (reshapeNum != 1)
        {
          return Ptr<BackendNode>();
        }

        // Input
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

        if (!inputWrapper->isTensor() || input_index == -1)
        {
            Ptr<tim::vx::Quantization> tvInputQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, scale, zeropoint));
          inputWrapper->createTensor(graph,tim::vx::TensorAttribute::INPUT,tvInputQuant);
          input_index = tvGraph->addWrapper(inputWrapper);
        }
        inputsIndex.push_back(input_index);

        //Output
        // Output Tensor has the same quantized attrib as Input Tesor.
        Ptr<tim::vx::Quantization> outputQuant = inputWrapper->getTensorQuantization();
        if (isLast)
        {
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, outputQuant);
        }
        else
        {
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, outputQuant);
        }
        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

        // generate output shape.
        MatShape outputShape = shape(outputWrapper->getMat());
        // reverse shape, from NCHW to WHCN
        std::reverse(outputShape.begin(), outputShape.end());
        std::vector<uint32_t> tvShape(outputShape.begin(), outputShape.end());

        std::shared_ptr<tim::vx::Operation> tvReshape = graph->CreateOperation<tim::vx::ops::Reshape>(tvShape);

        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvReshape, inputsIndex, outputsIndex);

        return tvBackendNode;
#endif  // HAVE_TIMVX
        return Ptr<BackendNode>();
    }

private:
    int axis;
    int numAxes;
    std::vector<MatShape> outShapes;
    std::vector<int> dynamicShapes; // Which axes shapes are dynamic and require reinitialization with new input
    std::vector<int> inputIndices; // Which axes from input are needed to compute correct output shape
    bool hasDynamicShapes;
    bool shapesInitialized;
    float scale;
    int zeropoint;
};

Ptr<ReshapeLayer> ReshapeLayer::create(const LayerParams& params)
{
    return Ptr<ReshapeLayer>(new ReshapeLayerImpl(params));
}


}
}
