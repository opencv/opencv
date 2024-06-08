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
#include "../op_halide.hpp"
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
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               (backendId == DNN_BACKEND_HALIDE && haveHalide() && !poolPad.width && !poolPad.height);
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

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() == 2 || inputs.size() == 3);
        Mat& input = inputs[0];
        Mat& indices = inputs[1];

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
                const float* inptr = input.ptr<float>(0, i_c);
                const float* idxptr = indices.ptr<float>(0, i_c);
                float* outptr = outPlane.ptr<float>();

                for(int i_wh = 0; i_wh < wh_area; i_wh++)
                {
                    int index = idxptr[i_wh];
                    if (!(0 <= index && index < outPlaneTotal))
                    {
                        CV_LOG_ERROR(NULL, cv::format(
                            "i_n=%d\ni_c=%d\ni_wh=%d\nindex=%d\nmaxval=%lf\noutPlaneTotal=%d\n",
                            i_n, i_c, i_wh, index, inptr[i_wh], outPlaneTotal));
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

        return make_cuda_node<cuda4dnn::MaxUnpoolingOp>(preferableTarget, std::move(context->stream), config);
    }
#endif

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &input) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        // Meaningless operation if false because if kernel > stride
        // it is not deterministic and if kernel < stride we just
        // skip a part of input data (you'd better change your model).
        if (poolKernel.width != poolStride.width ||
            poolKernel.height != poolStride.height)
            CV_Error(cv::Error::StsNotImplemented,
                     "Halide backend for maximum unpooling "
                     "is not support cases when kernel != stride");

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Buffer<float> inputBuffer = halideBuffer(input[0]);
        Halide::Buffer<float> indices = halideBuffer(input[1]);

        Halide::Expr pooledX = x / poolKernel.width;
        Halide::Expr pooledY = y / poolKernel.height;

        const int outW = inputBuffer.width() * poolKernel.width;
        top(x, y, c, n) = select(y * outW + x == indices(pooledX, pooledY, c, n),
                                 inputBuffer(pooledX, pooledY, c, n), 0.0f);
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

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
            inpShapes[i] = std::vector<int>(shape.begin(), shape.end());
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
