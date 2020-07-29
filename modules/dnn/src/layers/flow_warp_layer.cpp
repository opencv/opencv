// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"


namespace cv { namespace dnn {

class FlowWarpLayerImpl CV_FINAL : public FlowWarpLayer
{
public:
    FlowWarpLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        String fill_string = params.get<String>("FillParameter", "ZERO").toLowerCase();
        if (fill_string != "zero")
            CV_Error(Error::StsNotImplemented, "Only zero filling supported.");
        fill_value = 0;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        CV_Assert_N(inputs[0][0] == inputs[1][0], inputs[1][1] == 2,
                    inputs[0][2] == inputs[1][2], inputs[0][3] == inputs[1][3]);

        outputs.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const int out_n = outputs[0].size[0];
        const int out_c = outputs[0].size[1];
        const int out_h = outputs[0].size[2];
        const int out_w = outputs[0].size[3];

        const int area = out_w * out_h;
        const int total = area * out_c;

        const float* image_data = inputs[0].ptr<float>();
        const float* flow_data  = inputs[1].ptr<float>();
        float* out_data = outputs[0].ptr<float>();

        for (int n = 0; n < out_n; n++)
        {
            int off = total * n;
            for (int x = 0; x < out_w; x++)
            {
                for (int y = 0; y < out_h; y++)
                {
                    int idx = 2 * area * n + y * out_w + x;
                    float fx = flow_data[idx];
                    float fy = flow_data[idx + area];

                    float x2 = x + fx;
                    float y2 = y + fy;

                    if (x2 >= 0 && y2 >= 0 && x2 < out_w && y2 < out_h)
                    {
                        int ix2_L = x2;
                        float alpha = x2 - ix2_L;

                        int iy2_T = y2;
                        float beta = y2 - iy2_T;

                        int ix2_R = std::min(ix2_L + 1, out_w - 1);
                        int iy2_B = std::min(iy2_T + 1, out_h - 1);

                        for (int c = 0; c < out_c; c++)
                        {
                            float TL = image_data[off + c * area + iy2_T * out_w + ix2_L];
                            float TR = image_data[off + c * area + iy2_T * out_w + ix2_R];
                            float BL = image_data[off + c * area + iy2_B * out_w + ix2_L];
                            float BR = image_data[off + c * area + iy2_B * out_w + ix2_R];

                            out_data[off + c * area + y * out_w + x] = (1 - alpha) * (1 - beta) * TL +
                                                                       (1 - alpha) * beta       * BL +
                                                                        alpha      * (1 - beta) * TR +
                                                                        alpha      * beta       * BR;
                        }
                    }
                    else
                    {
                        for (int c = 0; c < out_c; c++)
                            out_data[off + c * area + y * out_w + x] = fill_value;
                    }
                }
            }
        }
    }

private:
    float fill_value;
};

Ptr<FlowWarpLayer> FlowWarpLayer::create(const LayerParams& params)
{
    return Ptr<FlowWarpLayer>(new FlowWarpLayerImpl(params));
}

}}  // namespace cv::dnn
