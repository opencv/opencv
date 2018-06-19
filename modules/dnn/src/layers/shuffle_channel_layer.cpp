// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "../precomp.hpp"

namespace cv { namespace dnn {

class ShuffleChannelLayerImpl CV_FINAL : public ShuffleChannelLayer
{
public:
    ShuffleChannelLayerImpl(const LayerParams& params)
    {
        group = params.get<int>("group", 1);
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

    virtual void finalize(const std::vector<Mat*>& inputs, std::vector<Mat> &outputs) CV_OVERRIDE
    {
        if (group != 1)
        {
            LayerParams lp;
            float order[] = {0, 2, 1, 3};
            lp.set("order", DictValue::arrayInt(&order[0], 4));
            permute = PermuteLayer::create(lp);

            Mat inp = *inputs[0];
            Mat out = outputs[0];

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

            inp = inp.reshape(1, permuteInpShape);
            out = out.reshape(1, permuteOutShape);

            std::vector<Mat*> permuteInputs(1, &inp);
            std::vector<Mat> permuteOutputs(1, out);
            permute->finalize(permuteInputs, permuteOutputs);
        }
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Mat inp = *inputs[0];
        Mat out = outputs[0];
        if (inp.data != out.data)
        {
            if (!permute.empty())
            {
                inp = inp.reshape(1, permuteInpShape);
                out = out.reshape(1, permuteOutShape);
                std::vector<Mat*> permuteInputs(1, &inp);
                std::vector<Mat> permuteOutputs(1, out);
                permute->forward(permuteInputs, permuteOutputs, internals);
            }
            else
                inp.copyTo(out);
        }
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
