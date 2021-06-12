// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Quantization & Dequantization Layers.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

////////////////////////////// Quantize Layer /////////////////////////////

class QLinearAddLayerImpl CV_FINAL : public QLinearAddLayer
{
public:
    QLinearAddLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        
        a_scale = params.get<float>("a_scale", 1.f);
        b_scale = params.get<float>("b_scale", 1.f);
        c_scale = params.get<float>("c_scale", 1.f);
        
        a_zeropoint = params.get<int>("a_zeropoint", 0);
        b_zeropoint = params.get<int>("b_zeropoint", 0);
        c_zeropoint = params.get<int>("c_zeropoint", 0);
        
        params.blobs[0].convertTo(bias, CV_32F, b_scale, -b_zeropoint*b_scale);
        outDepth = CV_32F; // should be changed to CV_8S (or CV_8U) eventually
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        outputs.assign(1, inputs[0]);
        return true;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(outputs.size() == 1);

        Mat &inpBlob = inputs[0];
        Mat &outBlob = outputs[0];
        int bdepth = bias.depth();
        int idepth = inpBlob.depth();
        int odepth = outBlob.depth();
        
        bool isContinuous = inpBlob.isContinuous() && outBlob.isContinuous() && bias.isContinuous();
        
        CV_Assert(idepth == CV_32F || idepth == CV_8U || idepth == CV_8S);
        CV_Assert(odepth == CV_32F);
        CV_Assert(bdepth == CV_32F);
        
        int ndims = inpBlob.dims;
        
        CV_Assert(ndims >= 2 && ndims <= 4 && ndims == bias.dims || ndims == bias.dims+1);
        int W = inpBlob.size.p[ndims-1];
        int H = inpBlob.size.p[ndims-2];
        int C = ndims > 2 ? inpBlob.size.p[ndims-3] : 1;
        int N = ndims > 3 ? inpBlob.size.p[ndims-4] : 1;
        
        float a_sc = a_scale, ic_scale = 1.f/c_scale;
        float a_zp = (float)a_zeropoint, c_zp = (float)c_zeropoint;
        
        if (isContinuous) {
            W *= H;
            H = 1;
        }
        
        for(int n = 0; n < N; n++)
        {
            for(int c = 0; c < C; c++)
            {
                // [TODO] change the output to int8_t/uint8_t
                float* outptr =
                    ndims == 2 ? outBlob.ptr<float>() :
                    ndims == 3 ? outBlob.ptr<float>(c) :
                    outBlob.ptr<float>(n, c);
                const float* biasptr = bias.ptr<float>();
                size_t ostep = outBlob.step.p[0]/sizeof(outptr[0]);
                size_t bstep = bias.step.p[1]/sizeof(biasptr[0]);
                
                if(idepth == CV_8U) {
                    const uint8_t* inptr =
                        ndims == 2 ? inpBlob.ptr<uint8_t>() :
                        ndims == 3 ? inpBlob.ptr<uint8_t>(c) :
                        inpBlob.ptr<uint8_t>(n, c);
                    size_t istep = inpBlob.step.p[ndims-2]/sizeof(inptr[0]);

                    for(int y = 0; y < H; y++, inptr += istep, outptr += ostep, biasptr += bstep)
                    {
                        for(int x = 0; x < W; x++)
                            outptr[x] = ((inptr[x] - a_zp)*a_sc + biasptr[x])*ic_scale + c_zp;
                    }
                } else if(idepth == CV_8S) {
                    const int8_t* inptr =
                        ndims == 2 ? inpBlob.ptr<int8_t>() :
                        ndims == 3 ? inpBlob.ptr<int8_t>(c) :
                        inpBlob.ptr<int8_t>(n, c);
                    size_t istep = inpBlob.step.p[ndims-2]/sizeof(inptr[0]);

                    for(int y = 0; y < H; y++, inptr += istep, outptr += ostep, biasptr += bstep)
                    {
                        for(int x = 0; x < W; x++)
                            outptr[x] = ((inptr[x] - a_zp)*a_sc + biasptr[x])*ic_scale + c_zp;
                    }
                } else {
                    CV_Assert(idepth == CV_32F);
                    const float* inptr =
                        ndims == 2 ? inpBlob.ptr<float>() :
                        ndims == 3 ? inpBlob.ptr<float>(c) :
                        inpBlob.ptr<float>(n, c);
                    size_t istep = inpBlob.step.p[ndims-2]/sizeof(inptr[0]);

                    for(int y = 0; y < H; y++,
                        inptr += istep, outptr += ostep, biasptr += bstep)
                    {
                        for(int x = 0; x < W; x++)
                            outptr[x] = ((inptr[x] - a_zp)*a_sc + biasptr[x])*ic_scale + c_zp;
                    }
                }
            }
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        long flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 2*total(inputs[i]);
        }
        return flops;
    }

private:
    int outDepth;
    Mat bias;
};


Ptr<QLinearAddLayer> QLinearAddLayer::create(const LayerParams& params)
{
    return Ptr<QLinearAddLayer>(new QLinearAddLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
