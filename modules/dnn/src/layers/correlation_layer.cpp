// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"


namespace cv { namespace dnn {

class CorrelationLayerImpl CV_FINAL : public CorrelationLayer
{
public:
    CorrelationLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        pad = params.get<int>("pad", 0);
        CV_Assert_N(params.has("kernel_size"), params.has("max_displacement"));
        max_displacement = params.get<int>("max_displacement");
        kernel = params.get<int>("kernel_size");
        if (kernel % 2 == 0)
            CV_Error(Error::StsNotImplemented, "Odd kernel size required.");

        stride_1 = params.get<int>("stride_1", 1);
        stride_2 = params.get<int>("stride_2", 1);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert_N(inputs.size() == 2, inputs[0].size() == 4, inputs[1].size() == 4);

        int padded_height = inputs[0][2] + 2 * pad;
        int padded_width  = inputs[0][3] + 2 * pad;

        int kernel_radius = (kernel - 1) / 2;
        int border_size = max_displacement + kernel_radius;

        int neighborhood_grid_radius = max_displacement / stride_2;
        int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

        std::vector<int> outShape;

        int num = inputs[0][0];
        outShape.push_back(num);

        int out_c = neighborhood_grid_width * neighborhood_grid_width;
        outShape.push_back(out_c);

        int out_h = ceil(static_cast<float>(padded_height - border_size * 2) / stride_1);
        int out_w = ceil(static_cast<float>(padded_width - border_size * 2)  / stride_1);
        CV_Assert_N(out_h >= 1, out_w >= 1);

        outShape.push_back(out_h);
        outShape.push_back(out_w);
        outputs.assign(1, outShape);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        int padded_height = inputs[0].size[2] + 2 * pad;
        int padded_width  = inputs[0].size[3] + 2 * pad;

        int size[] = {inputs[0].size[0], padded_height, padded_width, inputs[0].size[1]};
        rbot0 = Mat(4, &size[0], CV_32F, float(0));
        rbot1 = Mat(4, &size[0], CV_32F, float(0));
    }

    void blobRearrangeKernel2(const Mat& input, Mat& output)
    {
        const int num      = input.size[0];
        const int channels = input.size[1];
        const int height   = input.size[2];
        const int width    = input.size[3];
        const int area     = height * width;
        const int pad_area = (width + 2 * pad) * (height + 2 * pad);

        const float* in = input.ptr<float>();
        float* out = output.ptr<float>();
        for (int n = 0; n < num; n++)
        {
            for (int ch = 0; ch < channels; ch++)
            {
                for (int xy = 0; xy < area; xy++)
                {
                    float value = in[(n * channels + ch) * area + xy];
                    int xpad  = (xy % width + pad);
                    int ypad  = (xy / width + pad);
                    int xypad = ypad * (width + 2 * pad) + xpad;
                    out[(n * pad_area + xypad) * channels + ch] = value;
                }
            }
        }
    }

    void correlationKernelSubtraction(const Mat& input0, const Mat& input1, Mat& output, int item)
    {
        const int inp_h = input0.size[1];
        const int inp_w = input0.size[2];
        const int inp_c = input0.size[3];

        const int out_c = output.size[1];
        const int out_h = output.size[2];
        const int out_w = output.size[3];

        int topcount = output.total(1);
        int neighborhood_grid_radius = max_displacement / stride_2;
        int neighborhood_grid_width  = neighborhood_grid_radius * 2 + 1;

        const float* inp0_data = input0.ptr<float>();
        const float* inp1_data = input1.ptr<float>();
        float* out_data  = output.ptr<float>();
        int sumelems = kernel * kernel * inp_c;
        std::vector<float> patch_data(sumelems, 0);
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                int x1 = x * stride_1 + max_displacement;
                int y1 = y * stride_1 + max_displacement;

                for (int j = 0; j < kernel; j++)
                {
                    for (int i = 0; i < kernel; i++)
                    {
                        int ji_off = ((j * kernel) + i) * inp_c;
                        for (int ch = 0; ch < inp_c; ch++)
                        {
                            int idx1 = ((item * inp_h + y1 + j) * inp_w + x1 + i) * inp_c + ch;
                            int idxPatchData = ji_off + ch;
                            patch_data[idxPatchData] = inp0_data[idx1];
                        }
                    }
                }

                for (int out_ch = 0; out_ch < out_c; out_ch++)
                {
                    float sum = 0;
                    int s2o = (out_ch % neighborhood_grid_width - neighborhood_grid_radius) * stride_2;
                    int s2p = (out_ch / neighborhood_grid_width - neighborhood_grid_radius) * stride_2;

                    int x2 = x1 + s2o;
                    int y2 = y1 + s2p;
                    for (int j = 0; j < kernel; j++)
                    {
                        for (int i = 0; i < kernel; i++)
                        {
                            int ji_off = ((j * kernel) + i) * inp_c;
                            for (int ch = 0; ch < inp_c; ch++)
                            {
                                int idxPatchData = ji_off + ch;
                                int idx2 = ((item * inp_h + y2 + j) * inp_w + x2 + i) * inp_c + ch;
                                sum += patch_data[idxPatchData] * inp1_data[idx2];
                            }
                        }
                    }
                    int index = ((out_ch * out_h + y) * out_w) + x;
                    out_data[index + item * topcount] = static_cast<float>(sum) / sumelems;
                }
            }
        }
    }


    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        blobRearrangeKernel2(inputs[0], rbot0);
        blobRearrangeKernel2(inputs[1], rbot1);
        for (int i = 0; i < inputs[0].size[0]; i++)
        {
            correlationKernelSubtraction(rbot0, rbot1, outputs[0], i);
        }
    }

private:
    int pad;
    int kernel;
    int max_displacement;
    int stride_1;
    int stride_2;
    Mat rbot0;
    Mat rbot1;
};

Ptr<CorrelationLayer> CorrelationLayer::create(const LayerParams& params)
{
    return Ptr<CorrelationLayer>(new CorrelationLayerImpl(params));
}

}}  // namespace cv::dnn
