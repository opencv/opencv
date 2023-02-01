// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "../precomp.hpp"
#include "../op_cuda.hpp"

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/shuffle_channel.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv { namespace dnn {

class ShuffleChannelLayerImpl CV_FINAL : public ShuffleChannelLayer
{
public:
    ShuffleChannelLayerImpl(const LayerParams& params)
    {
        group = params.get<int>("group", 1);
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1 && inputs[0].size() == 4);
        CV_Assert(inputs[0][1] % group == 0);
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return group == 1;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        if (group != 1)
        {
            std::vector<Mat> inputs, outputs;
            inputs_arr.getMatVector(inputs);
            outputs_arr.getMatVector(outputs);

            LayerParams lp;
            float order[] = {0, 2, 1, 3};
            lp.set("order", DictValue::arrayInt(&order[0], 4));
            permute = PermuteLayer::create(lp);

            const Mat& inp = inputs[0];
            const Mat& out = outputs[0];

            permuteInpShape.resize(4);
            permuteInpShape[0] = inp.size[0];
            permuteInpShape[1] = group;
            permuteInpShape[2] = inp.size[1] / group;
            permuteInpShape[3] = inp.size[2]*inp.size[3];

            permuteOutShape.resize(4);
            permuteOutShape[0] = permuteInpShape[0];
            permuteOutShape[1] = permuteInpShape[2];
            permuteOutShape[2] = permuteInpShape[1];
            permuteOutShape[3] = permuteInpShape[3];

            std::vector<Mat> permuteInputs(1, inp.reshape(1, permuteInpShape));
            std::vector<Mat> permuteOutputs(1, out.reshape(1, permuteOutShape));
            permute->finalize(permuteInputs, permuteOutputs);
        }
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        if (inputs[0].u != outputs[0].u)
        {
            if (!permute.empty())
            {
                inputs[0] = inputs[0].reshape(1, permuteInpShape.size(), &permuteInpShape[0]);
                outputs[0] = outputs[0].reshape(1, permuteOutShape.size(), &permuteOutShape[0]);
                permute->preferableTarget = preferableTarget;
                permute->forward(inputs, outputs, internals);
            }
            else
                inputs[0].copyTo(outputs[0]);
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

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        Mat inp = inputs[0];
        Mat out = outputs[0];
        if (inp.data != out.data)
        {
            if (!permute.empty())
            {
                inp = inp.reshape(1, permuteInpShape);
                out = out.reshape(1, permuteOutShape);
                std::vector<Mat> permuteInputs(1, inp);
                std::vector<Mat> permuteOutputs(1, out);
                permute->forward(permuteInputs, permuteOutputs, internals);
            }
            else
                inp.copyTo(out);
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
        return make_cuda_node<cuda4dnn::ShuffleChannelOp>(preferableTarget, std::move(context->stream), group);
    }
#endif

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        return true;
    }

private:
    Ptr<PermuteLayer> permute;
    std::vector<int> permuteInpShape, permuteOutShape;
};

Ptr<Layer> ShuffleChannelLayer::create(const LayerParams& params)
{
    return Ptr<Layer>(new ShuffleChannelLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
