// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"


namespace cv { namespace dnn {

class AccumLayerImpl CV_FINAL : public AccumLayer
{
public:
    AccumLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        top_height = params.get<int>("top_height", 0);
        top_width = params.get<int>("top_width", 0);
        divisor = params.get<int>("size_divisible_by", 0);
        have_reference = params.get<String>("have_reference", "false") == "true";
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        std::vector<int> outShape;
        int batch = inputs[0][0];
        outShape.push_back(batch);

        if (have_reference)
        {
            CV_Assert(inputs.size() >= 2);
            int totalchannels = 0;
            for (int i = 0; i < inputs.size() - 1; i++) {
                CV_Assert(inputs[i][0] == batch);
                totalchannels += inputs[i][1];
            }
            outShape.push_back(totalchannels);

            int height = inputs.back()[2];
            int width = inputs.back()[3];

            outShape.push_back(height);
            outShape.push_back(width);
        }
        else
        {
            int maxwidth = -1;
            int maxheight = -1;
            int totalchannels = 0;

            // Find largest blob size and count total channels
            for (int i = 0; i < inputs.size(); ++i)
            {
                totalchannels += inputs[i][1];
                maxheight = std::max(maxheight, inputs[i][2]);
                maxwidth = std::max(maxwidth, inputs[i][3]);
                CV_Assert(inputs[i][0] == batch);
            }
            outShape.push_back(totalchannels);

            int out_h = divisor ? static_cast<int>(ceil(maxheight / divisor) * divisor) : top_height;
            int out_w = divisor ? static_cast<int>(ceil(maxwidth / divisor) * divisor) : top_width;

            // Layer can specify custom top size which is larger than default
            if (out_h <= maxheight || out_w <= maxwidth)
            {
                out_h = maxheight;
                out_w = maxwidth;
            }

            outShape.push_back(out_h);
            outShape.push_back(out_w);
        }

        outputs.assign(1, outShape);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        LayerParams resizeParams;
        resizeParams.set("interpolation", "bilinear");
        resizeParams.set("align_corners", true);
        resize = ResizeLayer::create(resizeParams);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const int out_h = outputs[0].size[2];
        const int out_w = outputs[0].size[3];
        float* out_data = outputs[0].ptr<float>();
        std::vector<int> sizes(&outputs[0].size[0], &outputs[0].size[0] + outputs[0].size.dims());
        for (int i = 0; i < inputs.size() - have_reference; i++)
        {
            sizes[1] = inputs[i].size[1];
            Mat outSlice(sizes, CV_32F, out_data);

            if (out_h == inputs[i].size[2] && out_w == inputs[i].size[3])
            {
                inputs[i].copyTo(outSlice);
            }
            else
            {
                std::vector<Mat> inp_slices, out_slices;
                inp_slices.push_back(inputs[i]);
                out_slices.push_back(outSlice);

                resize->finalize(inp_slices, out_slices);
                resize->forward(inp_slices, out_slices, internals_arr);
            }
            out_data += outSlice.total(1);
        }
    }

private:
    int top_height;
    int top_width;
    int divisor;
    bool have_reference;
    Ptr<ResizeLayer> resize;
};

Ptr<AccumLayer> AccumLayer::create(const LayerParams& params)
{
    return Ptr<AccumLayer>(new AccumLayerImpl(params));
}

}}  // namespace cv::dnn
