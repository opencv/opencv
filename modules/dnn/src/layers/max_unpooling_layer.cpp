// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Batch Normalization layer.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../ie_ngraph.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/max_unpooling.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class MaxUnpoolLayerImpl CV_FINAL : public MaxUnpoolLayer
{
public:
    MaxUnpoolLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        poolKernel = Size(params.get<int>("pool_k_w"), params.get<int>("pool_k_h"));
        poolPad = Size(params.get<int>("pool_pad_w"), params.get<int>("pool_pad_h"));
        poolStride = Size(params.get<int>("pool_stride_w"), params.get<int>("pool_stride_h"));
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2 || inputs.size() == 3);
        CV_Assert(total(inputs[0]) == total(inputs[1]));

        MatShape outShape;
        if (inputs.size() == 2)
        {
            outShape = inputs[0];
            outShape[2] = (outShape[2] - 1) * poolStride.height + poolKernel.height - 2 * poolPad.height;
            outShape[3] = (outShape[3] - 1) * poolStride.width + poolKernel.width - 2 * poolPad.width;
        }
        else
            outShape = inputs[2];

        outputs.clear();
        outputs.push_back(outShape);

        return false;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckGE(inputs.size(), (size_t)2, "");
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_16F || inputs[0] == CV_32S || inputs[0] == CV_64S || inputs[0] == CV_8S || inputs[0] == CV_8U, "");
        CV_CheckType(inputs[1], inputs[1] == CV_64S || inputs[1] == CV_32S, "");
        outputs.assign(1, inputs[0]);
    }


    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() == 2 || inputs.size() == 3);
        Mat& input = inputs[0];
        Mat& indices = inputs[1];

        if (indices.depth() == CV_32S)
            typeDispatch<int32_t>(input.type(), input, indices, outputs);
        else if (indices.depth() == CV_64S)
            typeDispatch<int64_t>(input.type(), input, indices, outputs);
        else
            CV_Error(cv::Error::BadDepth, "Unsupported type.");
    }

    template<typename T_INDEX, typename... Args>
    inline void typeDispatch(const int type, Args&&... args)
    {
        switch (type)
        {
            case CV_8S:
                run<int8_t, T_INDEX>(std::forward<Args>(args)...);
                break;
            case CV_8U:
                run<uint8_t, T_INDEX>(std::forward<Args>(args)...);
                break;
            case CV_32S:
                run<int32_t, T_INDEX>(std::forward<Args>(args)...);
                break;
            case CV_64S:
                run<int64_t, T_INDEX>(std::forward<Args>(args)...);
                break;
            case CV_32F:
                run<float, T_INDEX>(std::forward<Args>(args)...);
                break;
            case CV_16F:
                run<int16_t, T_INDEX>(std::forward<Args>(args)...);
                break;
            default:
                CV_Error(cv::Error::BadDepth, "Unsupported type.");
        };
    }

    template<typename T, typename INDEX_TYPE>
    void run(cv::Mat& input, cv::Mat& indices, std::vector<cv::Mat>& outputs)
    {
        CV_Assert(input.total() == indices.total());
        CV_Assert(input.size[0] == 1);
        CV_Assert(input.isContinuous());

        for(int i_n = 0; i_n < outputs.size(); i_n++)
        {
            Mat& outBlob = outputs[i_n];
            outBlob.setTo(0);
            CV_Assert(input.size[1] == outBlob.size[1]);
            int outPlaneTotal = outBlob.size[2]*outBlob.size[3];

            for (int i_c = 0; i_c < input.size[1]; i_c++)
            {
                Mat outPlane = getPlane(outBlob, 0, i_c);
                int wh_area = input.size[2]*input.size[3];
                const T* inptr = input.ptr<T>(0, i_c);
                const INDEX_TYPE* idxptr = indices.ptr<INDEX_TYPE>(0, i_c);
                T* outptr = outPlane.ptr<T>();

                for(int i_wh = 0; i_wh < wh_area; i_wh++)
                {
                    int index = idxptr[i_wh];
                    if (!(0 <= index && index < outPlaneTotal))
                    {
                        CV_LOG_ERROR(NULL, cv::format(
                            "i_n=%d\ni_c=%d\ni_wh=%d\nindex=%d\noutPlaneTotal=%d\n",
                            i_n, i_c, i_wh, index, outPlaneTotal));
                        CV_LOG_ERROR(NULL, "input.size=" << input.size);
                        CV_LOG_ERROR(NULL, "indices.size=" << indices.size);
                        CV_LOG_ERROR(NULL, "outBlob=" << outBlob.size);
                        CV_Assert(0 <= index && index < outPlaneTotal);
                    }
                    outptr[index] = inptr[i_wh];
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
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        cuda4dnn::MaxUnpoolingConfiguration config;
        auto& window_size = config.window_size;
        window_size.resize(2);
        window_size[0] = poolKernel.height;
        window_size[1] = poolKernel.width;

        auto& strides = config.strides;
        strides.resize(2);
        strides[0] = poolStride.height;
        strides[1] = poolStride.width;

        auto& pads_begin = config.pads_begin;
        pads_begin.resize(2);
        pads_begin[0] = poolPad.height;
        pads_begin[1] = poolPad.width;

        int indicesType = inputs[1]->getHostMatDepth();
        CV_CheckType(indicesType, indicesType == CV_32S || indicesType == CV_64S, "Unsupported indices type");

        if (indicesType == CV_32S)
            return make_cuda_node_with_indices<cuda4dnn::MaxUnpoolingOp, int32_t>(preferableTarget, inputs[0]->getHostMatDepth(), std::move(context->stream), config);
        else if (indicesType == CV_64S)
            return make_cuda_node_with_indices<cuda4dnn::MaxUnpoolingOp, int64_t>(preferableTarget, inputs[0]->getHostMatDepth(), std::move(context->stream), config);

        CV_Error(Error::BadDepth, "Unsupported indices type");
        return Ptr<BackendNode>();
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto features = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto indices = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;

        std::vector<MatShape> inpShapes(nodes.size());
        std::vector<MatShape> outShapes, internals;
        for (int i = 0; i < nodes.size(); ++i) {
            std::vector<size_t> shape = nodes[i].dynamicCast<InfEngineNgraphNode>()->node.get_shape();
            inpShapes[i] = MatShape(shape.begin(), shape.end());
        }
        getMemoryShapes(inpShapes, 1, outShapes, internals);

        Mat zeros = Mat::zeros(1, total(outShapes[0]), CV_32F);
        auto zeroInp = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{zeros.total()}, zeros.data);

        int newShape = -1;
        features = std::make_shared<ov::op::v1::Reshape>(
            features,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, &newShape),
            true
        );
        indices = std::make_shared<ov::op::v1::Reshape>(
            indices,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, &newShape),
            true
        );
        if (indices.get_element_type() != ov::element::i32 && indices.get_element_type() != ov::element::i64) {
            indices = std::make_shared<ov::op::v0::Convert>(indices, ov::element::i64);
        }

        int axis = 0;
        std::shared_ptr<ov::Node> unpool = std::make_shared<ov::op::v3::ScatterElementsUpdate>(zeroInp, indices, features,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, &axis));

        auto shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{outShapes[0].size()}, outShapes[0].data());
        unpool = std::make_shared<ov::op::v1::Reshape>(unpool, shape, true);

        return Ptr<BackendNode>(new InfEngineNgraphNode(unpool));
    }
#endif  // HAVE_DNN_NGRAPH
};

Ptr<MaxUnpoolLayer> MaxUnpoolLayer::create(const LayerParams& params)
{
    return Ptr<MaxUnpoolLayer>(new MaxUnpoolLayerImpl(params));
}

}
}
