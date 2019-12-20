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
#include <opencv2/dnn/shape_utils.hpp>

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

        if (inputs_arr.depth() == CV_16S)
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
                        std::cerr
                            << "i_n=" << i_n << std::endl
                            << "i_c=" << i_c << std::endl
                            << "i_wh=" << i_wh << std::endl
                            << "index=" << index << std::endl
                            << "maxval=" << inptr[i_wh] << std::endl
                            << "outPlaneTotal=" << outPlaneTotal << std::endl
                            << "input.size=" << input.size << std::endl
                            << "indices.size=" << indices.size << std::endl
                            << "outBlob=" << outBlob.size << std::endl
                            ;
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
};

Ptr<MaxUnpoolLayer> MaxUnpoolLayer::create(const LayerParams& params)
{
    return Ptr<MaxUnpoolLayer>(new MaxUnpoolLayerImpl(params));
}

}
}
